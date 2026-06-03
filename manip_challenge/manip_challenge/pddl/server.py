#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import shutil
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

try:
    import cv2
except ImportError:
    cv2 = None


if __package__ in {None, ""}:
    _PDDL_DIR = Path(__file__).resolve().parent
    _PACKAGE_ROOT = _PDDL_DIR.parents[1]
    if str(_PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(_PACKAGE_ROOT))
    from manip_challenge.pddl.utils import ensure_project_paths, ensure_ros_python
else:
    from .utils import ensure_project_paths, ensure_ros_python


ensure_ros_python()
ensure_project_paths()

import numpy as np
if not hasattr(np, "mat"):
    np.mat = np.asmatrix

import rclpy
import tf2_geometry_msgs  # noqa: F401
from geometry_msgs.msg import Pose, PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from riro_srvs.srv import StringString
from std_msgs.msg import String
from tf2_ros import Buffer, ConnectivityException, ExtrapolationException, LookupException, TransformListener

from assignment_2.move_joint import ArmClient
from manip_challenge.pddl.actions import (
    ActionContext,
    DYNAMIC_BUFFER_DEFAULT_OBJECT_RADIUS_M,
    DYNAMIC_BUFFER_MIN_CLEARANCE_M,
    DYNAMIC_BUFFER_WORKSPACE_X,
    DYNAMIC_BUFFER_WORKSPACE_Y,
    execute_action,
)
from manip_challenge.pddl.nlp import parse_goals
from manip_challenge.pddl.planner import plan
from manip_challenge.pddl.predicate_builder import (
    _annotate_relations,
    _bind_goals_to_instances,
    _build_predicates,
    build_predicate_state,
)
from manip_challenge.pddl.problem_generator import ensure_domain, write_problem
from manip_challenge.pddl.ros_helpers import (
    HOME_JOINTS,
    adjust_target_depth_from_rgbd,
    choose_grasp_target_from_points,
    load_grasp_points_from_detection,
    load_top_view_perception_module,
    patch_move_gripper,
    save_grasp_selection_visualization,
    stop_child_processes,
)
from manip_challenge.pddl.pddl_types import (
    ActionLedgerEntry,
    DYNAMIC_BUFFER_LOCATION,
    Goal,
    ObjectState,
    PlanAction,
    PredicateState,
    KNOWN_OBJECTS,
    BUFFER_LOCATIONS,
)
from manip_challenge.pddl.utils import PDDL_DIR, load_dotenv
from manip_challenge.custom.perception.icp.rgbd_seg_crop_server import RgbdSegCropServiceNode
from manip_challenge.custom.grasping.grasping_item import (
    _load_grasp_database,
    _lookup_object_grasp_configs,
    _select_state_config,
    _measured_object_area_from_features,
    _extract_grasp_transform,
)
from manip_challenge.custom.grasping.pose_math import apply_grasp_transform
from manip_challenge.custom.grasping.perception_features import extract_perception_features


patch_move_gripper()


OBSERVE_JOINTS = list(HOME_JOINTS)
OBSERVE_X_OFFSET_M = 0.15
OBSERVE_RETREAT_DURATION_S = 2.0


def move_arm_to_observe_pose(
    arm,
    observe_joints,
    x_offset_m: float,
    duration_s: float,
    logger=None,
    reason: str = "observe",
) -> None:
    if logger is not None:
        logger.info(f"Moving to observe base joints for {reason}: {observe_joints}")
    arm.move_joint(observe_joints)
    x_offset_m = float(x_offset_m)
    if abs(x_offset_m) <= 1e-6:
        return
    pose = copy.deepcopy(arm.fk_request(arm.js_joint_position, attach_tool=True))
    pose.position.x -= x_offset_m
    if logger is not None:
        logger.info(
            f"Retreating gripper in base -X for {reason}: "
            f"offset={x_offset_m:.3f}m target_x={pose.position.x:.3f}"
        )
    arm.move_position(pose, duration=float(duration_s))


class TfNode(Node):
    def __init__(self):
        super().__init__("pddl_tamp_tf_node")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)


def pose_from_xyz(xyz) -> Pose:
    pose = Pose()
    pose.position.x = float(xyz[0])
    pose.position.y = float(xyz[1])
    pose.position.z = float(xyz[2])
    pose.orientation.w = 1.0
    return pose


def pose_to_dict(pose: Pose) -> dict:
    return {
        "position": {
            "x": float(pose.position.x),
            "y": float(pose.position.y),
            "z": float(pose.position.z),
        },
        "orientation": {
            "x": float(pose.orientation.x),
            "y": float(pose.orientation.y),
            "z": float(pose.orientation.z),
            "w": float(pose.orientation.w),
        },
    }


def object_summary(obj) -> dict:
    files = (obj.detection or {}).get("files") or {}
    return {
        "target": obj.is_target,
        "class_name": obj.class_name,
        "instance_index": obj.instance_index,
        "location": obj.location,
        "detected": obj.detected,
        "visible": obj.visible,
        "pose_known": obj.pose_known,
        "graspable": obj.graspable,
        "clear": obj.clear,
        "safe": obj.safe,
        "confidence": obj.confidence,
        "mask_pixels": obj.mask_pixels,
        "foreground_points": obj.foreground_points,
        "bbox_xyxy": obj.bbox_xyxy,
        "centroid_xyz": obj.centroid_xyz,
        "grasp_xyz": obj.grasp_xyz,
        "base_link_xy": obj.base_link_xy,
        "scene_region": obj.scene_region,
        "classification_reason": obj.classification_reason,
        "relation_candidate": obj.relation_candidate,
        "depth_median": obj.depth_median,
        "blocks": sorted(obj.blocks),
        "blocked_by": sorted(obj.blocked_by),
        "near": sorted(obj.near),
        "error": obj.error,
        "debug_files": {
            key: files[key]
            for key in (
                "annotated",
                "rgb",
                "rgb_masked",
                "mask_overlay",
                "mask",
                "mask_full",
                "mask_raw",
                "depth_png",
                "depth_visualization",
            )
            if key in files
        },
    }


def object_command_payload(obj) -> dict | None:
    if obj is None:
        return None
    return {
        "name": obj.name,
        "class_name": obj.class_name,
        "instance_index": obj.instance_index,
        "is_target": obj.is_target,
        "location": obj.location,
        "detected": obj.detected,
        "visible": obj.visible,
        "pose_known": obj.pose_known,
        "graspable": obj.graspable,
        "clear": obj.clear,
        "safe": obj.safe,
        "confidence": obj.confidence,
        "mask_pixels": obj.mask_pixels,
        "foreground_points": obj.foreground_points,
        "bbox_xyxy": list(obj.bbox_xyxy) if obj.bbox_xyxy is not None else None,
        "centroid_xyz": list(obj.centroid_xyz) if obj.centroid_xyz is not None else None,
        "grasp_xyz": list(obj.grasp_xyz) if obj.grasp_xyz is not None else None,
        "base_link_xy": list(obj.base_link_xy) if obj.base_link_xy is not None else None,
        "scene_region": obj.scene_region,
        "classification_reason": obj.classification_reason,
        "relation_candidate": obj.relation_candidate,
        "depth_median": obj.depth_median,
        "blocks": sorted(obj.blocks),
        "blocked_by": sorted(obj.blocked_by),
        "near": sorted(obj.near),
        "detection": obj.detection,
        "error": obj.error,
    }


def object_state_from_command(payload: dict | None, fallback_name: str) -> ObjectState:
    payload = payload or {}
    bbox = payload.get("bbox_xyxy")
    centroid = payload.get("centroid_xyz")
    grasp = payload.get("grasp_xyz")
    base_xy = payload.get("base_link_xy")
    return ObjectState(
        name=str(payload.get("name") or fallback_name),
        class_name=str(payload.get("class_name") or fallback_name),
        instance_index=payload.get("instance_index"),
        is_target=bool(payload.get("is_target", True)),
        location=str(payload.get("location") or "table"),
        detected=bool(payload.get("detected", False)),
        visible=bool(payload.get("visible", False)),
        pose_known=bool(payload.get("pose_known", False)),
        graspable=bool(payload.get("graspable", False)),
        clear=bool(payload.get("clear", False)),
        safe=bool(payload.get("safe", True)),
        confidence=float(payload.get("confidence") or 0.0),
        mask_pixels=int(payload.get("mask_pixels") or 0),
        foreground_points=int(payload.get("foreground_points") or 0),
        bbox_xyxy=tuple(int(v) for v in bbox) if bbox and len(bbox) == 4 else None,
        centroid_xyz=tuple(float(v) for v in centroid) if centroid and len(centroid) >= 3 else None,
        grasp_xyz=tuple(float(v) for v in grasp) if grasp and len(grasp) >= 3 else None,
        base_link_xy=tuple(float(v) for v in base_xy) if base_xy and len(base_xy) >= 2 else None,
        scene_region=str(payload.get("scene_region") or "active_table"),
        classification_reason=payload.get("classification_reason"),
        relation_candidate=bool(payload.get("relation_candidate", True)),
        depth_median=float(payload["depth_median"]) if payload.get("depth_median") is not None else None,
        blocks=set(payload.get("blocks") or []),
        blocked_by=set(payload.get("blocked_by") or []),
        near=set(payload.get("near") or []),
        detection=payload.get("detection"),
        error=payload.get("error"),
    )


VERTICAL_DOWN_GRASP_QUATERNION = np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
GRASP_AXIS_ENDPOINT_OFFSET_M = 0.05
WRIST_VIEW_DEFAULTS = {
    "model_path": str(Path(__file__).resolve().parents[1] / "custom" / "best.pt"),
    "image_topic": "/wrist_camera/wrist_camera/color/image_raw",
    "depth_topic": "/wrist_camera/wrist_camera/depth/color/image_raw",
    "points_topic": "/wrist_camera/wrist_camera/depth/color/points",
    "camera_frame": "wrist_camera_color_optical_frame",
    "service_name": "detect_object_rgbd_seg_crop",
    "display": False,
    "confidence": 0.35,
    "input_crop_ratio": 1.0,
    "save_dir": str(Path(__file__).resolve().parents[1] / "custom" / "perception" / "icp" / "results" / "wrist_seg"),
    "annotated_image_topic": "/wrist_view/seg/detection_image",
    "roi_mask_topic": "/wrist_view/seg/mask",
    "roi_info_topic": "/wrist_view/seg/info",
    "roi_points_topic": "/wrist_view/seg/points",
    "max_depth_m": 1.5,
    "depth_margin_m": 0.04,
}


def grasp_pose_xyz_from_selection(selection):
    target = np.asarray(selection["target_xyz_m"], dtype=np.float64)
    if target.shape[0] < 3:
        raise ValueError("target_xyz_m must contain at least 3 values")
    raw = selection.get("raw_centroid_xyz_m")
    if raw is not None:
        raw = np.asarray(raw, dtype=np.float64)
        if raw.shape[0] >= 3 and np.isfinite(raw[:3]).all():
            return (
                np.asarray([raw[0], raw[1], raw[2]], dtype=np.float64),
                "raw_centroid_xyz",
            )
    return target[:3].copy(), "target_xyz_m"


def store_grasp_pose_reference(selection):
    grasp_xyz, source = grasp_pose_xyz_from_selection(selection)
    selection["grasp_pose_xyz_m"] = [float(v) for v in grasp_xyz]
    selection["grasp_pose_source"] = source
    return grasp_xyz, source


def normalize_vector(vector, min_norm=1e-9):
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm < min_norm or not np.isfinite(norm):
        return None
    return vector / norm


def quaternion_to_matrix(quaternion):
    x, y, z, w = (float(quaternion[i]) for i in range(4))
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0.0:
        return np.eye(3, dtype=np.float64)
    x /= norm; y /= norm; z /= norm; w /= norm
    return np.asarray([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def matrix_to_quaternion(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / scale
        x = 0.25 * scale
        y = (matrix[0, 1] + matrix[1, 0]) / scale
        z = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / scale
        x = (matrix[0, 1] + matrix[1, 0]) / scale
        y = 0.25 * scale
        z = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        w = (matrix[1, 0] - matrix[0, 1]) / scale
        x = (matrix[0, 2] + matrix[2, 0]) / scale
        y = (matrix[1, 2] + matrix[2, 1]) / scale
        z = 0.25 * scale
    q = normalize_vector([x, y, z, w])
    return q if q is not None else np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)


def set_pose_orientation_from_quaternion(pose, quaternion):
    pose.orientation.x = float(quaternion[0])
    pose.orientation.y = float(quaternion[1])
    pose.orientation.z = float(quaternion[2])
    pose.orientation.w = float(quaternion[3])


def build_grasp_orientation_perpendicular_to_axis(reference_quaternion, gripper_axis_base_xy):
    reference_matrix = quaternion_to_matrix(reference_quaternion)
    tool_z_axis = normalize_vector(reference_matrix[:, 2])
    desired_tool_x = normalize_vector([gripper_axis_base_xy[0], gripper_axis_base_xy[1], 0.0])
    if tool_z_axis is None or desired_tool_x is None:
        return None, None
    desired_tool_x = desired_tool_x - np.dot(desired_tool_x, tool_z_axis) * tool_z_axis
    desired_tool_x = normalize_vector(desired_tool_x)
    if desired_tool_x is None:
        return None, None
    reference_tool_x = reference_matrix[:, 0]
    if float(np.dot(desired_tool_x, reference_tool_x)) < 0.0:
        desired_tool_x = -desired_tool_x
    desired_tool_y = normalize_vector(np.cross(tool_z_axis, desired_tool_x))
    if desired_tool_y is None:
        return None, None
    desired_tool_x = normalize_vector(np.cross(desired_tool_y, tool_z_axis))
    if desired_tool_x is None:
        return None, None
    orientation_matrix = np.column_stack((desired_tool_x, desired_tool_y, tool_z_axis))
    return matrix_to_quaternion(orientation_matrix), desired_tool_x


class PddlTampServer(Node):
    def __init__(self, args, tf_buffer, arm):
        super().__init__("pddl_tamp_server")
        self.args = args
        self.tf_buffer = tf_buffer
        self.arm = arm
        self.callback_group = ReentrantCallbackGroup()
        self.command_lock = threading.Lock()

        # Attributes consumed by existing gripper/motion wrappers.
        self.gripper_force = args.gripper_force
        self.gripper_close_pos = args.gripper_close_pos
        self.gripper_settle_time = args.gripper_settle_time
        self.gripper_result_timeout_margin = args.gripper_result_timeout_margin
        self.strict_gripper_result = args.strict_gripper_result

        self.perception_client = self.create_client(
            StringString,
            args.perception_service,
            callback_group=self.callback_group,
        )
        self.wrist_perception_client = self.create_client(
            StringString,
            args.wrist_perception_service,
            callback_group=self.callback_group,
        )
        self.command_service = self.create_service(
            StringString,
            args.service_name,
            self.handle_command,
            callback_group=self.callback_group,
        )
        self.command_sub = self.create_subscription(
            String,
            args.command_topic,
            self.handle_command_topic,
            10,
            callback_group=self.callback_group,
        )
        self.status_pub = self.create_publisher(String, "/planner_status", 10)
        self.pred_pub = self.create_publisher(String, "/world_predicates", 10)
        self.action_pub = self.create_publisher(String, "/planned_action", 10)
        self.result_pub = self.create_publisher(String, "/action_result", 10)
        self.pddl_log_pub = self.create_publisher(String, args.log_topic, 10)
        self.grasp_pose_publisher = self.create_publisher(PoseStamped, '/grasp_target_pose', 10)
        self.corrected_grasp_pose_publisher = self.create_publisher(PoseStamped, '/corrected_grasp_pose', 10)
        self.pca_debug_publisher = self.create_publisher(MarkerArray, '/pca_debug', 10)
        from sensor_msgs.msg import PointCloud2
        self.pc2_publisher = self.create_publisher(PointCloud2, '/pca_debug_cloud', 10)
        self.executor_command_pub = self.create_publisher(String, args.executor_command_topic, 10)
        self.executor_result_sub = self.create_subscription(
            String,
            args.executor_result_topic,
            self.handle_executor_result,
            10,
            callback_group=self.callback_group,
        )
        self.executor_results: dict[str, dict] = {}
        self.executor_condition = threading.Condition()
        self.prefetch_lock = threading.Lock()
        self.prefetch_condition = threading.Condition(self.prefetch_lock)
        self.prefetch_states: dict[tuple, PredicateState] = {}
        self.prefetch_pending: set[tuple] = set()
        self.warm_scene_condition = threading.Condition()
        self.warm_scene_state: PredicateState | None = None
        self.warm_scene_pending = False
        self.warm_scene_scheduled = False
        self.warm_scene_consumed = False

        self.get_logger().info(f"Waiting for top-view perception service '{args.perception_service}'...")
        while rclpy.ok() and not self.perception_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f"Still waiting for '{args.perception_service}'...")
        self.get_logger().info(f"Waiting for wrist perception service '{args.wrist_perception_service}'...")
        while rclpy.ok() and not self.wrist_perception_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f"Still waiting for '{args.wrist_perception_service}'...")
        self.get_logger().info(
            f"PDDL TAMP server ready. Publish natural-language commands to '{args.command_topic}' "
            f"or call service '{args.service_name}'."
        )
        self.reset_debug_artifacts()
        self.warm_start_timer = None
        if not self.args.disable_warm_start:
            self.warm_scene_scheduled = True
            self.warm_start_timer = self.create_timer(
                0.5,
                self.kickoff_warm_scene_scan,
                callback_group=self.callback_group,
            )

    def handle_executor_result(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f"[Executor] Ignoring malformed executor result: {exc}")
            return
        command_id = payload.get("command_id")
        if not command_id:
            self.get_logger().warn("[Executor] Ignoring executor result without command_id.")
            return
        with self.executor_condition:
            self.executor_results[str(command_id)] = payload
            self.executor_condition.notify_all()

    @staticmethod
    def _xy_tuple(values) -> tuple[float, float] | None:
        if values is None:
            return None
        try:
            if len(values) < 2:
                return None
            xy = (float(values[0]), float(values[1]))
        except (TypeError, ValueError, IndexError):
            return None
        return xy if all(math.isfinite(v) for v in xy) else None

    @staticmethod
    def _ledger_from_payload(payload: list[dict] | None) -> list[ActionLedgerEntry]:
        return [ActionLedgerEntry.from_dict(item) for item in (payload or []) if isinstance(item, dict)]

    @staticmethod
    def _ledger_key(goals: list[Goal], action_ledger: list[ActionLedgerEntry], scan_all_targets: bool) -> tuple:
        goal_key = tuple((goal.object_name, goal.location) for goal in goals)
        return (
            goal_key,
            tuple(entry.planning_key() for entry in action_ledger),
            bool(scan_all_targets),
        )

    @staticmethod
    def _matches_ledger_object(obj: ObjectState, entry: ActionLedgerEntry) -> bool:
        if entry.object_name and obj.name == entry.object_name:
            return True
        return bool(entry.class_name and obj.class_name == entry.class_name)

    @staticmethod
    def _active_workspace_object(obj: ObjectState) -> bool:
        return (
            obj.detected
            and obj.visible
            and obj.location == "table"
            and obj.scene_region in {"active_table", "active_workspace", "unknown"}
        )

    @staticmethod
    def _dynamic_buffer_place_xy_from_result(result: dict | None) -> tuple[float, float] | None:
        result = result or {}
        dynamic = result.get("dynamic_buffer") or {}
        config = result.get("buffer_config") or {}
        return (
            PddlTampServer._xy_tuple(dynamic.get("place_xy"))
            or PddlTampServer._xy_tuple(config.get("place_xy"))
        )

    def object_base_link_xy(self, obj: ObjectState) -> tuple[float, float] | None:
        if obj.base_link_xy is not None:
            return obj.base_link_xy
        detection = obj.detection or {}
        source_frame = detection.get("frame_id") or self.args.camera_frame
        for xyz in (obj.centroid_xyz, obj.grasp_xyz):
            if xyz is None:
                continue
            try:
                pose = self.transform_pose(pose_from_xyz(xyz), source_frame, "base_link")
            except Exception as exc:
                obj.classification_reason = f"TF unavailable for scene classification: {exc}"
                continue
            obj.base_link_xy = (float(pose.position.x), float(pose.position.y))
            return obj.base_link_xy
        return None

    def action_ledger_entry(
        self,
        action: PlanAction,
        state: PredicateState,
        step_idx: int,
        result: dict | None = None,
        command_id: str | None = None,
        anticipated: bool = False,
    ) -> ActionLedgerEntry:
        object_name = action.args[0] if action.args else ""
        selected = state.objects.get(object_name)
        class_name = selected.class_name if selected is not None and selected.class_name else object_name
        source_xy = self.object_base_link_xy(selected) if selected is not None else None
        place_xy = self._dynamic_buffer_place_xy_from_result(result)
        ok = result.get("ok") if isinstance(result, dict) else None
        return ActionLedgerEntry(
            action_name=action.name,
            object_name=object_name,
            class_name=class_name,
            source_location=action.args[1] if len(action.args) >= 2 else "",
            destination_location=action.args[2] if len(action.args) >= 3 else "",
            dynamic_buffer_place_xy=place_xy,
            dynamic_buffer_radius_m=DYNAMIC_BUFFER_DEFAULT_OBJECT_RADIUS_M + DYNAMIC_BUFFER_MIN_CLEARANCE_M,
            source_base_link_xy=source_xy,
            result_ok=bool(ok) if ok is not None else None,
            result_status="anticipated_ok" if anticipated else ("ok" if ok else "failure"),
            step=int(step_idx),
            stamp_sec=time.time(),
            command_id=command_id or (result or {}).get("command_id"),
            anticipated=anticipated,
            result=result,
        )

    @staticmethod
    def dynamic_buffer_contains(xy: tuple[float, float] | None, entry: ActionLedgerEntry) -> bool:
        if xy is None:
            return False
        if entry.dynamic_buffer_place_xy is not None:
            radius = float(entry.dynamic_buffer_radius_m or (DYNAMIC_BUFFER_DEFAULT_OBJECT_RADIUS_M + DYNAMIC_BUFFER_MIN_CLEARANCE_M))
            return math.hypot(xy[0] - entry.dynamic_buffer_place_xy[0], xy[1] - entry.dynamic_buffer_place_xy[1]) <= radius
        return (
            DYNAMIC_BUFFER_WORKSPACE_X[0] <= xy[0] <= DYNAMIC_BUFFER_WORKSPACE_X[1]
            and DYNAMIC_BUFFER_WORKSPACE_Y[0] <= xy[1] <= DYNAMIC_BUFFER_WORKSPACE_Y[1]
        )

    def classify_observed_object(self, obj: ObjectState, action_ledger: list[ActionLedgerEntry]) -> None:
        if not obj.detected or not obj.visible:
            obj.scene_region = "unknown"
            obj.location = "table"
            obj.classification_reason = obj.classification_reason or "not visible in fresh observation"
            return
        xy = self.object_base_link_xy(obj)
        obj.scene_region = "active_table"
        obj.location = "table"
        obj.classification_reason = "fresh top-view observation in active scene"
        obstacle_buffer_entries = [
            entry for entry in action_ledger
            if entry.action_name == "move-obstacle-to-buffer"
            and entry.result_ok
            and self._matches_ledger_object(obj, entry)
        ]
        if not obstacle_buffer_entries:
            return
        latest = obstacle_buffer_entries[-1]
        if xy is not None and self.dynamic_buffer_contains(xy, latest):
            obj.location = DYNAMIC_BUFFER_LOCATION
            obj.scene_region = "dynamic_buffer"
            obj.classification_reason = "matched successful buffer ledger and observed inside dynamic buffer area"
        elif xy is None:
            obj.scene_region = "unknown"
            obj.classification_reason = "matched buffer ledger but base_link XY was unavailable; kept active for retry safety"
        else:
            obj.classification_reason = "matched buffer ledger but observed outside dynamic buffer area; treating as active obstacle"

    def reconcile_state_with_ledger(
        self,
        state: PredicateState,
        goals: list[Goal],
        action_ledger: list[ActionLedgerEntry],
    ) -> PredicateState:
        state.action_ledger = list(action_ledger)
        state.reconciliation = []
        state.ignored_objects = {}

        for obj in state.objects.values():
            self.classify_observed_object(obj, action_ledger)
        state.raw_observed_objects = {name: object_summary(obj) for name, obj in state.objects.items()}

        completed: set[str] = set()
        occupied_buffers: set[str] = set()
        buffered_obstacles: set[str] = set()
        for index, entry in enumerate(action_ledger):
            record = {
                "ledger_index": index,
                "action": entry.to_dict(),
                "status": "ignored",
                "reason": "",
                "observed_objects": [],
            }
            matches = [
                obj for obj in state.objects.values()
                if self._matches_ledger_object(obj, entry)
            ]
            record["observed_objects"] = [obj.name for obj in matches]
            if not entry.result_ok:
                record["status"] = "failed"
                record["reason"] = "executor result was not ok; no symbolic completion inferred"
            elif entry.action_name == "move-target-to-goal":
                active = [obj for obj in matches if self._active_workspace_object(obj)]
                if active:
                    record["status"] = "retry"
                    record["reason"] = "target still appears in active workspace after move-target-to-goal"
                else:
                    completed.add(entry.class_name or entry.object_name)
                    record["status"] = "completed"
                    record["reason"] = "successful target move plus disappearance from active workspace"
            elif entry.action_name == "move-obstacle-to-buffer":
                dynamic = [obj for obj in matches if obj.location == DYNAMIC_BUFFER_LOCATION]
                active = [obj for obj in matches if self._active_workspace_object(obj)]
                if dynamic:
                    record["status"] = "buffered"
                    record["reason"] = "obstacle observed in dynamic buffer; excluded from relations/PDDL"
                    for obj in dynamic:
                        buffered_obstacles.add(f"name:{obj.name}")
                        if obj.class_name:
                            buffered_obstacles.add(f"class:{obj.class_name}")
                elif active:
                    record["status"] = "retry"
                    record["reason"] = "obstacle still appears in active workspace; keeping it as an obstacle"
                else:
                    record["status"] = "ignored"
                    record["reason"] = "successful obstacle move and no active observation; treating as removed from active scene"
                    if entry.object_name:
                        buffered_obstacles.add(f"name:{entry.object_name}")
                    if entry.class_name:
                        buffered_obstacles.add(f"class:{entry.class_name}")
                if entry.destination_location and entry.destination_location != DYNAMIC_BUFFER_LOCATION:
                    occupied_buffers.add(entry.destination_location)
            else:
                record["reason"] = "ledger action has no reconciliation rule"
            state.reconciliation.append(record)

        kept: dict[str, ObjectState] = {}
        for name, obj in state.objects.items():
            if obj.is_target and obj.class_name in completed:
                state.ignored_objects[name] = "target completed by ledger/observation reconciliation"
                continue
            kept[name] = obj
        state.objects = kept
        state.completed = completed
        state.occupied_buffers = occupied_buffers
        state.buffered_obstacles = buffered_obstacles
        _annotate_relations(state.objects)
        state.goals = _bind_goals_to_instances(goals, state.objects, completed)
        state.relation_input_objects = sorted(obj.name for obj in state.objects.values() if obj.relation_candidate)
        state.predicates = _build_predicates(state)
        if state.reconciliation:
            state.notes.append("state reconciled from fresh observation and action ledger")
        return state

    def dispatch_action_to_executor(
        self,
        action: PlanAction,
        state: PredicateState,
        step_idx: int,
        action_ledger: list[ActionLedgerEntry],
    ) -> dict:
        command_id = f"step{step_idx}-{uuid.uuid4().hex[:8]}"
        selected_object = state.objects.get(action.args[0]) if action.args else None
        anticipated_entry = self.action_ledger_entry(
            action,
            state,
            step_idx,
            result={"ok": True, "anticipated": True},
            command_id=command_id,
            anticipated=True,
        )
        anticipated_ledger = [*action_ledger, anticipated_entry]
        payload = {
            "event": "execute_action",
            "command_id": command_id,
            "step": int(step_idx),
            "stamp_sec": time.time(),
            "action": action.to_dict(),
            "selected_object": object_command_payload(selected_object),
            "prefetch": {
                "goals": [goal.__dict__ for goal in state.goals],
                "action_ledger": [entry.to_dict() for entry in anticipated_ledger],
                "scan_all_targets": not self.args.scan_goals_only,
            },
        }
        msg = String()
        msg.data = json.dumps(payload, sort_keys=True)
        self.get_logger().info(
            f"[Executor] Dispatching committed action {action.pddl()} command_id={command_id}"
        )
        self.executor_command_pub.publish(msg)

        deadline = time.monotonic() + float(self.args.action_timeout)
        with self.executor_condition:
            while command_id not in self.executor_results:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(
                        f"Timed out waiting for manipulator executor result for command_id={command_id}."
                    )
                self.executor_condition.wait(timeout=min(0.2, remaining))
            payload = self.executor_results.pop(command_id)

        result = payload.get("result")
        if not isinstance(result, dict):
            return {
                "ok": False,
                "action": action.to_dict(),
                "error": f"Malformed executor result for command_id={command_id}: {payload}",
                "command_id": command_id,
            }
        result.setdefault("command_id", command_id)
        return result

    def start_warm_scene_scan(self) -> None:
        with self.warm_scene_condition:
            if self.warm_scene_pending or self.warm_scene_state is not None:
                return
            self.warm_scene_pending = True
        self.get_logger().info("[WarmStart] Starting initial top-view scene scan from observe pose.")
        worker = threading.Thread(target=self._warm_scene_worker, daemon=True)
        worker.start()

    def kickoff_warm_scene_scan(self) -> None:
        if self.warm_start_timer is not None:
            self.warm_start_timer.cancel()
            self.warm_start_timer = None
        with self.warm_scene_condition:
            self.warm_scene_scheduled = False
        self.start_warm_scene_scan()

    def _warm_scene_worker(self) -> None:
        try:
            state = build_predicate_state(
                [],
                self.detect_object_with_grasp,
                completed=set(),
                occupied_buffers=set(),
                scan_all_targets=True,
            )
        except Exception as exc:
            self.get_logger().warn(f"[WarmStart] Initial scene scan failed: {exc}")
            state = None
        with self.warm_scene_condition:
            self.warm_scene_state = state
            self.warm_scene_pending = False
            self.warm_scene_condition.notify_all()
        if state is not None:
            self.get_logger().info(
                f"[WarmStart] Initial top-view scene ready: objects={len(state.objects)}"
            )
            self.save_warm_start_artifacts(state)

    def consume_warm_scene_state(
        self,
        goals: list[Goal],
        action_ledger: list[ActionLedgerEntry],
        scan_all_targets: bool,
    ) -> PredicateState | None:
        if action_ledger or self.args.disable_warm_start:
            return None
        deadline = time.monotonic() + float(self.args.warm_start_wait_timeout)
        with self.warm_scene_condition:
            if self.warm_scene_consumed:
                return None
            while self.warm_scene_scheduled or self.warm_scene_pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                self.warm_scene_condition.wait(timeout=min(0.1, remaining))
            if self.warm_scene_state is None:
                return None
            base = copy.deepcopy(self.warm_scene_state)
            self.warm_scene_consumed = True
        state = self.bind_goals_to_warm_scene(
            base,
            goals,
            action_ledger,
            scan_all_targets,
        )
        self.get_logger().info("[WarmStart] Using initial top-view scene; running planner without a new scan.")
        return state

    def bind_goals_to_warm_scene(
        self,
        state: PredicateState,
        goals: list[Goal],
        action_ledger: list[ActionLedgerEntry],
        scan_all_targets: bool,
    ) -> PredicateState:
        goal_names = {goal.object_name for goal in goals}
        for obj in state.objects.values():
            obj.is_target = obj.class_name in goal_names
        self.reconcile_state_with_ledger(state, goals, action_ledger)
        state.notes = [
            *state.notes,
            "used warm-start top-view scene captured from observe pose before command",
        ]
        return state

    def filter_buffered_obstacles(self, state: PredicateState, buffered_obstacles: set[str]) -> PredicateState:
        state.buffered_obstacles = set(buffered_obstacles or set())
        if not state.buffered_obstacles:
            return state

        removed = []
        kept = {}
        for name, obj in state.objects.items():
            if obj.is_target:
                kept[name] = obj
                continue
            keys = {f"name:{name}"}
            if obj.class_name:
                keys.add(f"class:{obj.class_name}")
            if keys & state.buffered_obstacles:
                removed.append(f"{name}/{obj.class_name}")
                continue
            kept[name] = obj

        if not removed:
            return state
        state.objects = kept
        _annotate_relations(state.objects)
        state.goals = _bind_goals_to_instances(state.goals, state.objects, state.completed)
        state.predicates = _build_predicates(state)
        state.notes.append(
            "ignored already-buffered obstacles during this command: " + ", ".join(sorted(removed))
        )
        return state

    def move_to_observe_pose(self, reason: str = "observe") -> None:
        if self.args.dry_run or self.args.no_home:
            return
        move_arm_to_observe_pose(
            self.arm,
            self.args.observe_joints,
            self.args.observe_x_offset,
            self.args.observe_retreat_duration,
            logger=self.get_logger(),
            reason=reason,
        )

    @staticmethod
    def prefetch_key(
        goals: list[Goal],
        action_ledger: list[ActionLedgerEntry],
        scan_all_targets: bool,
    ) -> tuple:
        return PddlTampServer._ledger_key(goals, action_ledger, scan_all_targets)

    @staticmethod
    def goals_from_payload(payload: list[dict]) -> list[Goal]:
        return [
            Goal(
                object_name=str(item["object_name"]),
                location=str(item["location"]),
                bound_object_name=item.get("bound_object_name"),
            )
            for item in payload
        ]

    @staticmethod
    def buffered_obstacle_keys_for_action(action: PlanAction, state: PredicateState) -> set[str]:
        if action.name != "move-obstacle-to-buffer" or not action.args:
            return set()
        object_name = action.args[0]
        moved = state.objects.get(object_name)
        keys = {f"name:{object_name}"}
        if moved is None or not moved.class_name:
            return keys
        same_class_obstacles = [
            obj
            for obj in state.objects.values()
            if not obj.is_target and obj.class_name == moved.class_name
        ]
        if len(same_class_obstacles) == 1:
            keys.add(f"class:{moved.class_name}")
        return keys

    def anticipated_progress_after_action(
        self,
        action: PlanAction,
        state: PredicateState,
    ) -> tuple[set[str], set[str], set[str]]:
        completed = set(state.completed)
        occupied_buffers = set(state.occupied_buffers)
        buffered_obstacles = set(state.buffered_obstacles)
        if action.name == "move-target-to-goal" and action.args:
            moved = state.objects.get(action.args[0])
            completed.add(moved.class_name if moved is not None and moved.class_name else action.args[0])
        elif action.name == "move-obstacle-to-buffer" and len(action.args) >= 3:
            buffered_obstacles.update(self.buffered_obstacle_keys_for_action(action, state))
            if action.args[2] != DYNAMIC_BUFFER_LOCATION:
                occupied_buffers.add(action.args[2])
        return completed, occupied_buffers, buffered_obstacles

    def start_scene_prefetch(
        self,
        prefetch_payload: dict | None,
        command_id: str,
        trigger: str = "observe-ready",
        use_wrist: bool = False,
    ) -> None:
        if not prefetch_payload:
            return
        try:
            goals = self.goals_from_payload(prefetch_payload.get("goals") or [])
            action_ledger = self._ledger_from_payload(prefetch_payload.get("action_ledger") or [])
            scan_all_targets = bool(prefetch_payload.get("scan_all_targets", True))
        except Exception as exc:
            self.get_logger().warn(f"[Prefetch] Ignoring malformed prefetch payload: {exc}")
            return
        key = self.prefetch_key(goals, action_ledger, scan_all_targets)
        with self.prefetch_condition:
            if key in self.prefetch_states or key in self.prefetch_pending:
                return
            self.prefetch_pending.add(key)
        self.get_logger().info(
            f"[Prefetch] Starting {trigger} {'wrist-camera' if use_wrist else 'top-view'} scene scan "
            f"for command_id={command_id}; "
            f"ledger_entries={len(action_ledger)}"
        )
        worker = threading.Thread(
            target=self._prefetch_scene_worker,
            args=(
                key,
                goals,
                action_ledger,
                scan_all_targets,
                command_id,
                use_wrist,
            ),
            daemon=True,
        )
        worker.start()

    def _prefetch_scene_worker(
        self,
        key: tuple,
        goals: list[Goal],
        action_ledger: list[ActionLedgerEntry],
        scan_all_targets: bool,
        command_id: str,
        use_wrist: bool,
    ) -> None:
        try:
            state = build_predicate_state(
                goals,
                self.detect_object_with_grasp_wrist if use_wrist else self.detect_object_with_grasp,
                completed=set(),
                occupied_buffers=set(),
                scan_all_targets=scan_all_targets,
            )
            self.reconcile_state_with_ledger(state, goals, action_ledger)
        except Exception as exc:
            self.get_logger().warn(f"[Prefetch] Scene scan failed for command_id={command_id}: {exc}")
            state = None
        with self.prefetch_condition:
            self.prefetch_pending.discard(key)
            if state is not None:
                self.prefetch_states[key] = state
            self.prefetch_condition.notify_all()
        if state is not None:
            self.get_logger().info(
                f"[Prefetch] Scene scan ready for command_id={command_id}: objects={len(state.objects)}"
            )

    def consume_prefetched_state(
        self,
        goals: list[Goal],
        action_ledger: list[ActionLedgerEntry],
        scan_all_targets: bool,
    ) -> PredicateState | None:
        key = self.prefetch_key(goals, action_ledger, scan_all_targets)
        deadline = time.monotonic() + float(self.args.prefetch_wait_timeout)
        with self.prefetch_condition:
            while key in self.prefetch_pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                self.prefetch_condition.wait(timeout=min(0.1, remaining))
            return self.prefetch_states.pop(key, None)

    def publish_json(self, publisher, payload: dict) -> None:
        msg = String()
        msg.data = json.dumps(payload, sort_keys=True)
        publisher.publish(msg)

    def publish_pddl_log(self, event: str, lines: list[str], **_payload) -> None:
        msg = String()
        header = f"[PDDL_LOG] event={event} stamp_sec={time.time():.3f}"
        msg.data = "\n".join([header, *lines])
        self.pddl_log_pub.publish(msg)

    def handle_command(self, request, response):
        try:
            with self.command_lock:
                result = self.execute_command(request.data)
        except Exception as exc:
            self.get_logger().error(f"Command failed: {exc}\n{traceback.format_exc()}")
            result = {"ok": False, "error": str(exc), "command": request.data}
            self.publish_pddl_log(
                "command_error",
                ["[PDDL] command failed", f"error: {exc}", f"command: {request.data}"],
                error=str(exc),
                command=request.data,
            )
        response.data = json.dumps(result, sort_keys=True)
        return response

    def handle_command_topic(self, msg: String) -> None:
        command_text = str(msg.data or "").strip()
        if not command_text:
            self.get_logger().warn("[PDDL] Ignoring empty topic command.")
            return
        self.get_logger().info(f"[PDDL] Topic command received on '{self.args.command_topic}': {command_text}")
        worker = threading.Thread(
            target=self.execute_topic_command,
            args=(command_text,),
            daemon=True,
        )
        worker.start()

    def execute_topic_command(self, command_text: str) -> None:
        try:
            with self.command_lock:
                result = self.execute_command(command_text)
        except Exception as exc:
            self.get_logger().error(f"Topic command failed: {exc}\n{traceback.format_exc()}")
            result = {"ok": False, "error": str(exc), "command": command_text}
            self.publish_pddl_log(
                "topic_command_error",
                ["[PDDL] topic command failed", f"error: {exc}", f"command: {command_text}"],
                error=str(exc),
                command=command_text,
            )
        self.publish_json(
            self.result_pub,
            {
                "event": "topic_command_result",
                "command": command_text,
                "result": result,
            },
        )

    def execute_command(self, command_text: str) -> dict:
        lowered = str(command_text or "").strip().lower()
        if lowered in {"home", "idle", "reset"}:
            if not self.args.dry_run:
                self.arm.move_joint(HOME_JOINTS)
            self.publish_pddl_log(
                "home",
                [f"[PDDL] {lowered}: robot moved to home joints"],
                action=lowered,
                dry_run=bool(self.args.dry_run),
            )
            return {"ok": True, "action": lowered, "message": "Robot moved to home joints."}

        goals = parse_goals(command_text, use_gemini=not self.args.no_gemini)
        if not goals:
            raise ValueError("Could not parse any target goals from command.")

        _valid_goal_locations = {"left_storage", "right_storage", "bookshelf"}
        invalid_goals = [g for g in goals if g.location not in _valid_goal_locations]
        for g in invalid_goals:
            self.get_logger().warn(f"[PDDL] Ignoring goal with unknown destination: {g.object_name} -> {g.location}")
        goals = [g for g in goals if g.location in _valid_goal_locations]
        if not goals:
            raise ValueError("No valid goals after filtering unknown destinations.")

        self.get_logger().info("[PDDL] Parsed goals:")
        for goal in goals:
            self.get_logger().info(f"[PDDL]   {goal.object_name} -> {goal.location}")
        self.publish_json(self.status_pub, {"event": "parsed_goals", "goals": [goal.__dict__ for goal in goals]})
        self.publish_pddl_log(
            "parsed_goals",
            [
                "[PDDL] parsed goals",
                *[f"  {goal.object_name} -> {goal.location}" for goal in goals],
            ],
            goals=[goal.__dict__ for goal in goals],
        )

        completed: set[str] = set()
        occupied_buffers: set[str] = set()
        buffered_obstacles: set[str] = set()
        action_ledger: list[ActionLedgerEntry] = []
        steps = []
        scenes = []
        domain_path = ensure_domain(PDDL_DIR / "domain.pddl")
        problem_path = PDDL_DIR / "problem.pddl"

        for step_idx in range(1, self.args.max_steps + 1):
            self.get_logger().info(f"[PDDL] ===== planning step {step_idx}/{self.args.max_steps} =====")
            scan_all_targets = not self.args.scan_goals_only
            state = self.consume_prefetched_state(
                goals,
                action_ledger,
                scan_all_targets,
            )
            if state is not None:
                self.get_logger().info("[Prefetch] Using cached scene state.")
            else:
                state = self.consume_warm_scene_state(
                    goals,
                    action_ledger,
                    scan_all_targets,
                )
                if state is None:
                    state = build_predicate_state(
                        goals,
                        self.detect_object_with_grasp,
                        completed=set(),
                        occupied_buffers=set(),
                        scan_all_targets=scan_all_targets,
                    )
                    self.reconcile_state_with_ledger(state, goals, action_ledger)
            completed = set(state.completed)
            occupied_buffers = set(state.occupied_buffers)
            buffered_obstacles = set(state.buffered_obstacles)
            scene_summary = {
                "completed": sorted(completed),
                "occupied_buffers": sorted(occupied_buffers),
                "buffered_obstacles": sorted(buffered_obstacles),
                "action_ledger": [entry.to_dict() for entry in action_ledger],
                "reconciliation": state.reconciliation,
                "raw_observed_objects": state.raw_observed_objects,
                "ignored_objects": state.ignored_objects,
                "relation_input_objects": state.relation_input_objects,
                "objects": {name: object_summary(obj) for name, obj in state.objects.items()},
                "predicates": sorted(state.predicates),
                "notes": state.notes,
            }
            self.save_step_artifacts(scene_summary, state, step_idx, command_text)
            scenes.append(scene_summary)
            self.log_state(scene_summary)
            self.publish_json(self.pred_pub, scene_summary)
            self.publish_pddl_log(
                "predicate_judgement",
                self.judgement_log_lines(scene_summary, step_idx),
                step=step_idx,
                scene=scene_summary,
            )
            if self.args.debug:
                self.show_debug_view(scene_summary, step_idx)

            if not state.unfinished_goals():
                self.get_logger().info("[PDDL] All goals completed.")
                self.publish_pddl_log(
                    "all_goals_completed",
                    ["[PDDL] all goals completed", f"completed: {', '.join(sorted(completed))}"],
                    step=step_idx,
                    completed=sorted(completed),
                )
                break

            generated_problem = write_problem(state, problem_path)
            self.get_logger().info(f"[PDDL] Generated problem: {generated_problem}")
            actions, raw_output = plan(domain_path, generated_problem, state)
            self.get_logger().info("[PDDL] Planner raw output:\n" + raw_output)
            plan_payload = [action.to_dict() for action in actions]
            self.save_step_json(
                step_idx,
                "planner.json",
                {
                    "event": "planner_output",
                    "step": int(step_idx),
                    "stamp_sec": time.time(),
                    "problem": str(generated_problem),
                    "raw_output": raw_output,
                    "plan": plan_payload,
                },
                latest_name="latest_planner.json",
            )
            self.publish_json(
                self.status_pub,
                {
                    "event": "planner_output",
                    "problem": str(generated_problem),
                    "raw_output": raw_output,
                    "plan": plan_payload,
                },
            )
            self.publish_pddl_log(
                "planner_output",
                self.planner_log_lines(generated_problem, actions, raw_output),
                step=step_idx,
                problem=str(generated_problem),
                raw_output=raw_output,
                plan=plan_payload,
            )
            if not actions:
                self.save_step_json(
                    step_idx,
                    "planning_failed.json",
                    {
                        "event": "planning_failed",
                        "step": int(step_idx),
                        "stamp_sec": time.time(),
                        "completed": sorted(completed),
                        "action_ledger": [entry.to_dict() for entry in action_ledger],
                        "message": "No executable PDDL action found.",
                    },
                    latest_name="latest_planning_failed.json",
                )
                self.publish_pddl_log(
                    "planning_failed",
                    ["[PDDL] no executable action found", f"completed: {', '.join(sorted(completed)) or '<none>'}"],
                    step=step_idx,
                    completed=sorted(completed),
                )
                return {
                    "ok": False,
                    "error": "No executable PDDL action found.",
                    "goals": [goal.__dict__ for goal in goals],
                    "completed": sorted(completed),
                    "occupied_buffers": sorted(occupied_buffers),
                    "buffered_obstacles": sorted(buffered_obstacles),
                    "action_ledger": [entry.to_dict() for entry in action_ledger],
                    "steps": steps,
                    "scenes": scenes,
                }

            action = self.select_executable_action(actions)
            self.get_logger().info(f"[PDDL] Selected next action: {action.pddl()} from {action.source}")
            self.save_step_json(
                step_idx,
                "selected_action.json",
                {
                    "event": "selected_action",
                    "step": int(step_idx),
                    "stamp_sec": time.time(),
                    "action": action.to_dict(),
                },
                latest_name="latest_selected_action.json",
            )
            self.publish_json(self.action_pub, {"event": "selected_action", "action": action.to_dict()})
            self.publish_pddl_log(
                "selected_action",
                self.selected_action_log_lines(action, state, step_idx),
                step=step_idx,
                action=action.to_dict(),
            )

            self.publish_pddl_log(
                "executing_action",
                self.executing_action_log_lines(action, step_idx),
                step=step_idx,
                action=action.to_dict(),
            )
            self.save_step_json(
                step_idx,
                "executing_action.json",
                {
                    "event": "executing_action",
                    "step": int(step_idx),
                    "stamp_sec": time.time(),
                    "action": action.to_dict(),
                },
                latest_name="latest_executing_action.json",
            )

            result = self.dispatch_action_to_executor(action, state, step_idx, action_ledger)
            steps.append(result)
            self.publish_json(self.result_pub, {"event": "action_result", "result": result})
            self.get_logger().info("[PDDL] Action execution result: " + json.dumps(result, sort_keys=True))
            self.save_step_json(
                step_idx,
                "action_result.json",
                {
                    "event": "action_result",
                    "step": int(step_idx),
                    "stamp_sec": time.time(),
                    "action": action.to_dict(),
                    "result": result,
                },
                latest_name="latest_action_result.json",
            )
            self.publish_pddl_log(
                "action_result",
                self.action_result_log_lines(action, result, step_idx),
                step=step_idx,
                action=action.to_dict(),
                result=result,
            )

            action_ledger.append(
                self.action_ledger_entry(
                    action,
                    state,
                    step_idx,
                    result=result,
                    command_id=result.get("command_id"),
                    anticipated=False,
                )
            )

            self.get_logger().info("[PDDL] Replanning after action execution.")
            self.publish_pddl_log(
                "replan",
                [
                    "[PDDL] replanning after action execution",
                    "completion/buffer status will be reconciled from the next fresh observation",
                    f"action_ledger_entries: {len(action_ledger)}",
                ],
                step=step_idx,
                action_ledger=[entry.to_dict() for entry in action_ledger],
            )
            if self.args.one_step:
                break

        ok = len(completed) == len({goal.object_name for goal in goals})
        self.publish_pddl_log(
            "command_finished",
            [
                f"[PDDL] command finished ok={ok}",
                f"completed: {', '.join(sorted(completed)) or '<none>'}",
                f"occupied_buffers: {', '.join(sorted(occupied_buffers)) or '<none>'}",
                f"buffered_obstacles: {', '.join(sorted(buffered_obstacles)) or '<none>'}",
            ],
            ok=ok,
            completed=sorted(completed),
            occupied_buffers=sorted(occupied_buffers),
            buffered_obstacles=sorted(buffered_obstacles),
            action_ledger=[entry.to_dict() for entry in action_ledger],
        )
        return {
            "ok": ok,
            "action": "pddl_tamp",
            "goals": [goal.__dict__ for goal in goals],
            "completed": sorted(completed),
            "occupied_buffers": sorted(occupied_buffers),
            "buffered_obstacles": sorted(buffered_obstacles),
            "action_ledger": [entry.to_dict() for entry in action_ledger],
            "steps": steps,
            "scenes": scenes,
            "message": "PDDL TAMP completed." if ok else "PDDL TAMP stopped before all goals completed.",
            "dry_run": bool(self.args.dry_run),
        }

    def select_executable_action(self, actions: list[PlanAction]) -> PlanAction:
        for action in actions:
            if action.name in {"move-target-to-goal", "move-obstacle-to-buffer"}:
                return action
        raise RuntimeError(f"Planner returned no physical action: {[action.pddl() for action in actions]}")

    def log_state(self, summary: dict) -> None:
        self.get_logger().info("[PDDL] Detected objects and predicates:")
        self.get_logger().info(
            "[PDDL]   buffered_obstacles: "
            + (", ".join(summary.get("buffered_obstacles", [])) if summary.get("buffered_obstacles") else "<none>")
        )
        self.get_logger().info(
            "[PDDL]   relation_input_objects: "
            + (", ".join(summary.get("relation_input_objects", [])) if summary.get("relation_input_objects") else "<none>")
        )
        for name, obj in sorted(summary["objects"].items()):
            self.get_logger().info(
                "[PDDL]   "
                f"{name}: class={obj['class_name']}, target={obj['target']}, detected={obj['detected']}, "
                f"region={obj.get('scene_region')}, base_xy={obj.get('base_link_xy')}, "
                f"visible={obj['visible']}, pose_known={obj['pose_known']}, "
                f"graspable={obj['graspable']}, clear={obj['clear']}, safe={obj['safe']}, "
                f"conf={obj['confidence']:.3f}, bbox={obj['bbox_xyxy']}, "
                f"centroid={obj['centroid_xyz']}, blocks={obj['blocks']}, "
                f"blocked_by={obj['blocked_by']}, near={obj['near']}, error={obj['error']}"
            )
        self.get_logger().info("[PDDL]   predicates: " + ", ".join(summary["predicates"]))
        for note in summary["notes"]:
            self.get_logger().warn("[PDDL]   note: " + note)

    def judgement_log_lines(self, summary: dict, step_idx: int) -> list[str]:
        lines = [
            f"[PDDL] judgement step {step_idx}",
            "completed: " + (", ".join(summary["completed"]) if summary["completed"] else "<none>"),
            "occupied_buffers: "
            + (", ".join(summary.get("occupied_buffers", [])) if summary.get("occupied_buffers") else "<none>"),
            "buffered_obstacles: "
            + (", ".join(summary.get("buffered_obstacles", [])) if summary.get("buffered_obstacles") else "<none>"),
            "relation_input_objects: "
            + (", ".join(summary.get("relation_input_objects", [])) if summary.get("relation_input_objects") else "<none>"),
            "objects:",
        ]
        for name, obj in sorted(summary["objects"].items()):
            lines.append(
                f"  {name}: target={obj['target']} detected={obj['detected']} visible={obj['visible']} "
                f"class={obj['class_name']} region={obj.get('scene_region')} base_xy={obj.get('base_link_xy')} "
                f"pose_known={obj['pose_known']} graspable={obj['graspable']} clear={obj['clear']} "
                f"safe={obj['safe']} conf={obj['confidence']:.3f}"
            )
            lines.append(
                f"    bbox={obj['bbox_xyxy']} centroid={obj['centroid_xyz']} depth={obj['depth_median']} "
                f"blocks={obj['blocks']} blocked_by={obj['blocked_by']} near={obj['near']}"
            )
            if obj["error"]:
                lines.append(f"    error={obj['error']}")
            if obj.get("classification_reason"):
                lines.append(f"    classification={obj['classification_reason']}")
        if summary.get("reconciliation"):
            lines.append("reconciliation:")
            for record in summary["reconciliation"]:
                action = record.get("action") or {}
                lines.append(
                    f"  {action.get('action_name')} {action.get('object_name')}: "
                    f"{record.get('status')} - {record.get('reason')}"
                )
        lines.append("predicates:")
        lines.extend(f"  {predicate}" for predicate in summary["predicates"])
        if summary["notes"]:
            lines.append("notes:")
            lines.extend(f"  {note}" for note in summary["notes"])
        return lines

    @staticmethod
    def selected_action_log_lines(action: PlanAction, state, step_idx: int) -> list[str]:
        lines = [
            f"[PDDL] selected action at step {step_idx}",
            f"  action: {action.pddl()}",
            f"  source: {action.source}",
            "  unfinished goal objects:",
        ]
        unfinished = {goal.target_name for goal in state.unfinished_goals()}
        for name in sorted(unfinished):
            obj = state.objects.get(name)
            if obj is None:
                lines.append(f"    {name}: not in current perception state")
                continue
            ready = obj.graspable and obj.clear and obj.safe
            lines.append(
                f"    {name}: class={obj.class_name} ready={ready} detected={obj.detected} visible={obj.visible} "
                f"pose_known={obj.pose_known} graspable={obj.graspable} clear={obj.clear} "
                f"safe={obj.safe} conf={obj.confidence:.3f}"
            )
            lines.append(
                f"      bbox={obj.bbox_xyxy} centroid={obj.centroid_xyz} depth={obj.depth_median} "
                f"blocked_by={sorted(obj.blocked_by)} near={sorted(obj.near)}"
            )
        if action.args:
            selected = state.objects.get(action.args[0])
            if selected is not None:
                lines.append(
                    f"  selected object judgement: {selected.name} clear={selected.clear} "
                    f"graspable={selected.graspable} safe={selected.safe} centroid={selected.centroid_xyz}"
                )
        return lines

    @staticmethod
    def executing_action_log_lines(action: PlanAction, step_idx: int) -> list[str]:
        lines = [
            f"[PDDL] executing physical action at step {step_idx}",
            f"  action: {action.pddl()}",
        ]
        if action.name == "move-target-to-goal" and len(action.args) == 3:
            object_name, source, destination = action.args
            lines.append(f"  moving target: {object_name} {source} -> {destination}")
        elif action.name == "move-obstacle-to-buffer" and len(action.args) == 3:
            object_name, source, buffer_name = action.args
            lines.append(f"  moving obstacle: {object_name} {source} -> {buffer_name}")
        lines.append("  motion note: low-level pick-place retreats and returns to idle before replanning")
        return lines

    @staticmethod
    def action_result_log_lines(action: PlanAction, result: dict, step_idx: int) -> list[str]:
        lines = [
            f"[PDDL] action result at step {step_idx}",
            f"  action: {action.pddl()}",
            f"  ok: {result.get('ok')}",
            f"  destination: {result.get('destination', '<none>')}",
        ]
        selection = result.get("grasp_selection") or {}
        if selection:
            lines.append(
                "  grasp_selection: "
                f"method={selection.get('method')} target_xyz_m={selection.get('target_xyz_m')} "
                f"reason={selection.get('reason', '<none>')}"
            )
        grasp_pose = (result.get("grasp_pose_in_base") or {}).get("position") or {}
        if grasp_pose:
            lines.append(
                "  grasp_pose_in_base: "
                f"x={grasp_pose.get('x'):.4f} y={grasp_pose.get('y'):.4f} z={grasp_pose.get('z'):.4f}"
            )
        return lines

    @staticmethod
    def planner_log_lines(problem_path: Path, actions: list[PlanAction], raw_output: str) -> list[str]:
        lines = [
            "[PDDL] planner output",
            f"problem: {problem_path}",
            "plan:",
        ]
        if actions:
            lines.extend(f"  {action.pddl()} [{action.source}]" for action in actions)
        else:
            lines.append("  <empty>")
        lines.append("raw_output:")
        raw_lines = str(raw_output or "").splitlines()
        if raw_lines:
            lines.extend(f"  {line}" for line in raw_lines)
        else:
            lines.append("  <empty>")
        return lines

    def show_debug_view(self, summary: dict, step_idx: int) -> None:
        if cv2 is None:
            self.get_logger().warn("[PDDL] --debug requested, but cv2 is not available.")
            return

        image = self.load_debug_image(summary)
        if image is None:
            image = np.full((720, 900, 3), 245, dtype=np.uint8)
            cv2.putText(
                image,
                "No perception debug image available",
                (28, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (30, 30, 30),
                2,
                cv2.LINE_AA,
            )

        max_h = 720
        scale = min(1.0, max_h / max(1, image.shape[0]))
        if scale < 1.0:
            image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))

        panel_w = 760
        panel_h = max(image.shape[0], 720)
        panel = np.full((panel_h, panel_w, 3), 32, dtype=np.uint8)
        lines = self.debug_lines(summary, step_idx)
        y = 34
        for line, color in lines:
            for chunk in self.wrap_debug_line(line, 78):
                cv2.putText(panel, chunk, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
                y += 22
                if y > panel_h - 20:
                    break
            if y > panel_h - 20:
                cv2.putText(panel, "...", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
                break

        if image.shape[0] < panel_h:
            pad = np.full((panel_h - image.shape[0], image.shape[1], 3), 245, dtype=np.uint8)
            image = np.vstack([image, pad])
        canvas = np.hstack([image, panel])

        debug_dir = PDDL_DIR / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        latest = debug_dir / "latest_debug.png"
        step_path = debug_dir / f"step_{step_idx:02d}_debug.png"
        cv2.imwrite(str(latest), canvas)
        cv2.imwrite(str(step_path), canvas)
        self.get_logger().info(f"[PDDL] Debug view saved: {latest}")

        if self.args.debug_window and self.debug_display_available():
            try:
                cv2.imshow("PDDL TAMP Debug", canvas)
                key = cv2.waitKey(0 if self.args.debug_wait else 1)
                if key in (27, ord("q")):
                    self.args.debug = False
                    self.get_logger().info("[PDDL] Debug window disabled by key press.")
            except Exception as exc:
                self.get_logger().warn(f"[PDDL] Could not open debug window: {exc}")
        elif self.args.debug_window:
            self.get_logger().info("[PDDL] No usable GUI display found; debug image was saved only.")

    def save_step_artifacts(self, summary: dict, state, step_idx: int, command_text: str) -> None:
        debug_dir = PDDL_DIR / "debug"
        step_dir = debug_dir / f"step_{step_idx:02d}"
        if step_dir.exists():
            shutil.rmtree(step_dir)
        step_dir.mkdir(parents=True, exist_ok=True)

        images = []
        for object_name, obj in sorted(summary["objects"].items()):
            for key, source_text in sorted((obj.get("debug_files") or {}).items()):
                source = Path(source_text)
                if not source.is_file():
                    continue
                suffix = source.suffix or ".png"
                filename = f"{self.safe_debug_filename(object_name)}_{self.safe_debug_filename(key)}{suffix}"
                destination = step_dir / filename
                try:
                    shutil.copyfile(source, destination)
                except OSError as exc:
                    self.get_logger().warn(f"[PDDL] Could not save debug image '{source}': {exc}")
                    continue
                images.append({
                    "object": object_name,
                    "class_name": obj.get("class_name"),
                    "instance_index": obj.get("instance_index"),
                    "kind": key,
                    "source": str(source),
                    "path": str(destination),
                })

        payload = {
            "event": "planning_step_start",
            "step": int(step_idx),
            "stamp_sec": time.time(),
            "command": command_text,
            "goals": [
                {
                    "object_name": goal.object_name,
                    "location": goal.location,
                    "bound_object_name": goal.bound_object_name,
                    "target_name": goal.target_name,
                }
                for goal in state.goals
            ],
            "completed": summary["completed"],
            "occupied_buffers": summary.get("occupied_buffers", []),
            "buffered_obstacles": summary.get("buffered_obstacles", []),
            "action_ledger": summary.get("action_ledger", []),
            "reconciliation": summary.get("reconciliation", []),
            "raw_observed_objects": summary.get("raw_observed_objects", {}),
            "ignored_objects": summary.get("ignored_objects", {}),
            "relation_input_objects": summary.get("relation_input_objects", []),
            "objects": summary["objects"],
            "predicates": summary["predicates"],
            "notes": summary["notes"],
            "images": images,
        }
        predicates_path = step_dir / "predicates.json"
        latest_path = debug_dir / "latest_predicates.json"
        text = json.dumps(payload, indent=2, sort_keys=True)
        predicates_path.write_text(text + "\n", encoding="utf-8")
        latest_path.write_text(text + "\n", encoding="utf-8")
        self.get_logger().info(f"[PDDL] Step artifacts saved: {predicates_path}")

    def save_warm_start_artifacts(self, state: PredicateState) -> None:
        summary = {
            "completed": sorted(state.completed),
            "occupied_buffers": sorted(state.occupied_buffers),
            "buffered_obstacles": sorted(state.buffered_obstacles),
            "action_ledger": [entry.to_dict() for entry in state.action_ledger],
            "reconciliation": state.reconciliation,
            "raw_observed_objects": state.raw_observed_objects,
            "ignored_objects": state.ignored_objects,
            "relation_input_objects": state.relation_input_objects,
            "objects": {name: object_summary(obj) for name, obj in state.objects.items()},
            "predicates": sorted(state.predicates),
            "notes": state.notes,
        }
        debug_dir = PDDL_DIR / "debug"
        warm_dir = debug_dir / "warm_start"
        if warm_dir.exists():
            shutil.rmtree(warm_dir)
        warm_dir.mkdir(parents=True, exist_ok=True)

        images = []
        for object_name, obj in sorted(summary["objects"].items()):
            for key, source_text in sorted((obj.get("debug_files") or {}).items()):
                source = Path(source_text)
                if not source.is_file():
                    continue
                suffix = source.suffix or ".png"
                filename = f"{self.safe_debug_filename(object_name)}_{self.safe_debug_filename(key)}{suffix}"
                destination = warm_dir / filename
                try:
                    shutil.copyfile(source, destination)
                except OSError as exc:
                    self.get_logger().warn(f"[WarmStart] Could not save debug image '{source}': {exc}")
                    continue
                images.append({
                    "object": object_name,
                    "class_name": obj.get("class_name"),
                    "instance_index": obj.get("instance_index"),
                    "kind": key,
                    "source": str(source),
                    "path": str(destination),
                })

        payload = {
            "event": "warm_start_scene",
            "stamp_sec": time.time(),
            "completed": summary["completed"],
            "occupied_buffers": summary["occupied_buffers"],
            "buffered_obstacles": summary["buffered_obstacles"],
            "action_ledger": summary["action_ledger"],
            "reconciliation": summary["reconciliation"],
            "raw_observed_objects": summary["raw_observed_objects"],
            "ignored_objects": summary["ignored_objects"],
            "relation_input_objects": summary["relation_input_objects"],
            "objects": summary["objects"],
            "predicates": summary["predicates"],
            "notes": summary["notes"],
            "images": images,
        }
        text = json.dumps(payload, indent=2, sort_keys=True)
        predicates_path = warm_dir / "predicates.json"
        predicates_path.write_text(text + "\n", encoding="utf-8")
        (debug_dir / "latest_warm_start_predicates.json").write_text(text + "\n", encoding="utf-8")
        (debug_dir / "latest_predicates.json").write_text(text + "\n", encoding="utf-8")
        self.get_logger().info(f"[WarmStart] Debug artifacts saved: {predicates_path}")

    def reset_debug_artifacts(self) -> None:
        debug_dir = PDDL_DIR / "debug"
        if debug_dir.exists():
            shutil.rmtree(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        self.get_logger().info(f"[PDDL] Cleared previous debug artifacts: {debug_dir}")

    def save_step_json(self, step_idx: int, filename: str, payload: dict, latest_name: str | None = None) -> None:
        debug_dir = PDDL_DIR / "debug"
        step_dir = debug_dir / f"step_{step_idx:02d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        text = json.dumps(payload, indent=2, sort_keys=True)
        path = step_dir / filename
        path.write_text(text + "\n", encoding="utf-8")
        if latest_name:
            (debug_dir / latest_name).write_text(text + "\n", encoding="utf-8")
        self.get_logger().info(f"[PDDL] Step JSON saved: {path}")

    def save_grasp_pose_debug(
        self,
        obj_name: str,
        detection: dict,
        selection: dict,
        source_frame: str,
        base_pose: Pose,
        target_pose_base: Pose | None,
        grasp_pose: Pose,
        corrected_grasp_pose: Pose | None,
        correction: dict | None,
    ) -> None:
        visualization = selection.setdefault("visualization", {})
        summary_path_text = visualization.get("summary_json")
        if summary_path_text:
            summary_path = Path(summary_path_text).expanduser()
        else:
            debug_dir = PDDL_DIR / "debug" / "grasp_debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            summary_path = debug_dir / f"{self.safe_debug_filename(obj_name)}_latest_grasp.json"

        payload = {}
        if summary_path.is_file():
            try:
                payload = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception as exc:
                payload = {"previous_summary_error": str(exc)}

        payload.update({
            "object_name": str(obj_name),
            "source_frame": source_frame,
            "base_frame": "base_link",
            "detection_save_dir": detection.get("save_dir"),
            "points_path": selection.get("points_path"),
            "selection": selection,
            "base_link_debug": {
                "target_pose_base": pose_to_dict(target_pose_base) if target_pose_base is not None else None,
                "grasp_reference_pose_base": pose_to_dict(base_pose),
                "final_grasp_pose_base": pose_to_dict(grasp_pose),
                "corrected_grasp_pose_base": pose_to_dict(corrected_grasp_pose) if corrected_grasp_pose is not None else None,
                "grasp_database_correction": correction,
            },
            "notes": {
                "target_pose_base": "Pose transformed from target_xyz_m, the point selected by PCA band search.",
                "grasp_reference_pose_base": "Pose transformed from grasp_pose_xyz_m, the point actually used before orientation and clearance offsets.",
                "final_grasp_pose_base": "Pose sent into execute_pick_place_sequence before grasp_database correction.",
                "corrected_grasp_pose_base": "Preview of the pose after grasp_database transform; grasping_item applies the same correction at close time.",
            },
        })

        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        visualization["summary_json"] = str(summary_path)
        self.get_logger().info(f"[GraspDebug] saved grasp pose debug: {summary_path}")

    @staticmethod
    def safe_debug_filename(text: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(text or "unknown"))
        return safe.strip("_") or "unknown"

    def load_debug_image(self, summary: dict):
        for obj in summary["objects"].values():
            files = obj.get("debug_files") or {}
            for key in ("annotated", "mask_overlay", "rgb"):
                path = files.get(key)
                if path and Path(path).is_file():
                    image = cv2.imread(str(path))
                    if image is not None:
                        return image
        return None

    def debug_lines(self, summary: dict, step_idx: int) -> list[tuple[str, tuple[int, int, int]]]:
        lines = [
            (f"PDDL TAMP DEBUG - planning step {step_idx}", (120, 230, 255)),
            ("completed: " + (", ".join(summary["completed"]) if summary["completed"] else "<none>"), (220, 220, 220)),
            ("", (220, 220, 220)),
            ("OBJECT JUDGEMENT", (120, 230, 255)),
        ]
        for name, obj in sorted(summary["objects"].items()):
            status_color = (120, 230, 120) if obj["graspable"] and obj["clear"] and obj["safe"] else (80, 190, 255)
            if obj["error"] or not obj["detected"]:
                status_color = (80, 120, 255)
            lines.append((
                f"{name}: target={obj['target']} det={obj['detected']} vis={obj['visible']} "
                f"class={obj['class_name']} region={obj.get('scene_region')} pose={obj['pose_known']} "
                f"grasp={obj['graspable']} clear={obj['clear']} safe={obj['safe']} "
                f"conf={obj['confidence']:.2f}",
                status_color,
            ))
            lines.append((
                f"  base_xy={obj.get('base_link_xy')} bbox={obj['bbox_xyxy']} centroid={obj['centroid_xyz']} depth={obj['depth_median']} "
                f"blocks={obj['blocks']} blocked_by={obj['blocked_by']} near={obj['near']}",
                (205, 205, 205),
            ))
            if obj.get("classification_reason"):
                lines.append((f"  classification={obj['classification_reason']}", (170, 210, 255)))
            if obj["error"]:
                lines.append((f"  error={obj['error']}", (80, 120, 255)))
        if summary.get("reconciliation"):
            lines.extend([("", (220, 220, 220)), ("RECONCILIATION", (120, 230, 255))])
            for record in summary["reconciliation"]:
                action = record.get("action") or {}
                lines.append((
                    f"{action.get('action_name')} {action.get('object_name')}: {record.get('status')} - {record.get('reason')}",
                    (190, 190, 190),
                ))
        if summary.get("relation_input_objects"):
            lines.append((f"relation inputs: {summary['relation_input_objects']}", (170, 210, 255)))
        lines.extend([
            ("", (220, 220, 220)),
            ("TRUE PREDICATES", (120, 230, 255)),
        ])
        for predicate in summary["predicates"]:
            lines.append((predicate, (190, 190, 190)))
        if summary["notes"]:
            lines.extend([("", (220, 220, 220)), ("NOTES", (120, 230, 255))])
            for note in summary["notes"]:
                lines.append((note, (80, 120, 255)))
        return lines

    @staticmethod
    def wrap_debug_line(line: str, limit: int) -> list[str]:
        if not line:
            return [""]
        chunks = []
        text = str(line)
        while len(text) > limit:
            split_at = text.rfind(" ", 0, limit)
            if split_at <= 0:
                split_at = limit
            chunks.append(text[:split_at])
            text = text[split_at:].lstrip()
        chunks.append(text)
        return chunks

    @staticmethod
    def debug_display_available() -> bool:
        wayland = os.environ.get("WAYLAND_DISPLAY")
        if wayland:
            runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
            if runtime_dir and (Path(runtime_dir) / wayland).exists():
                return True
        display = os.environ.get("DISPLAY")
        if not display:
            return False
        if display.startswith(":"):
            display_num = display[1:].split(".")[0]
            return (Path("/tmp/.X11-unix") / f"X{display_num}").exists()
        return True

    def detect_object_from_client(self, obj_name: str, client, service_name: str) -> dict:
        req = StringString.Request()
        req.data = obj_name
        future = client.call_async(req)
        deadline = time.monotonic() + self.args.perception_timeout
        while rclpy.ok() and not future.done():
            if time.monotonic() > deadline:
                raise TimeoutError(f"Timed out waiting for perception result from '{service_name}' for '{obj_name}'.")
            time.sleep(0.02)
        result = future.result()
        if result is None:
            raise RuntimeError(f"Perception service '{service_name}' returned no result for '{obj_name}'.")
        return json.loads(result.data)

    def detect_object(self, obj_name: str) -> dict:
        return self.detect_object_from_client(obj_name, self.perception_client, self.args.perception_service)

    def detect_object_wrist(self, obj_name: str) -> dict:
        return self.detect_object_from_client(obj_name, self.wrist_perception_client, self.args.wrist_perception_service)

    def detect_object_with_grasp(self, obj_name: str) -> dict:
        detection = self.detect_object(obj_name)
        if detection and detection.get("ok"):
            try:
                selection = self.select_grasp_target(detection, obj_name)
                detection["grasp_selection"] = selection
            except Exception as exc:
                self.get_logger().debug(f"[GraspPredict] select_grasp_target failed for {obj_name}: {exc}")
        return detection

    def detect_object_with_grasp_wrist(self, obj_name: str) -> dict:
        detection = self.detect_object_wrist(obj_name)
        if detection and detection.get("ok"):
            try:
                selection = self.select_grasp_target(detection, obj_name)
                detection["grasp_selection"] = selection
            except Exception as exc:
                self.get_logger().debug(f"[GraspPredict] wrist select_grasp_target failed for {obj_name}: {exc}")
        return detection

    def select_grasp_target(self, detection: dict, obj_name: str) -> dict:
        points, points_path = load_grasp_points_from_detection(detection)
        if points is None or len(points) == 0:
            fallback = detection["location_xyz_m"]
            return {
                "method": "perception_centroid_fallback",
                "target_xyz_m": [float(v) for v in fallback],
                "raw_centroid_xyz_m": [float(v) for v in fallback],
                "points_path": points_path,
                "reason": "foreground_points_unavailable",
            }
        label = detection.get("target") or detection.get("detected_label") or obj_name
        selection = choose_grasp_target_from_points(points, label=label)
        selection["points_path"] = points_path
        # Removed adjust_target_depth_from_rgbd per user request; use exact centroid height.
        store_grasp_pose_reference(selection)
        output_dir = detection.get("save_dir")
        if not output_dir and points_path:
            output_dir = str(Path(points_path).expanduser().parent)
        if output_dir:
            try:
                visualization = save_grasp_selection_visualization(points, selection, output_dir, label)
            except Exception as exc:
                visualization = None
                selection["visualization_error"] = str(exc)
            if visualization:
                selection["visualization"] = visualization
        raw = np.asarray(selection.get("raw_centroid_xyz_m", detection["location_xyz_m"]), dtype=float)
        target = np.asarray(selection["target_xyz_m"], dtype=float)
        selection["delta_from_raw_centroid_m"] = (target - raw).astype(float).tolist()
        return selection

    def transform_pose(self, pose: Pose, source_frame: str, target_frame: str) -> Pose:
        deadline = time.monotonic() + self.args.tf_timeout
        while rclpy.ok() and time.monotonic() < deadline:
            if self.tf_buffer.can_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5),
            ):
                break
            time.sleep(0.05)
        else:
            raise TimeoutError(f"TF unavailable: {source_frame} -> {target_frame}")
        stamped = PoseStamped()
        stamped.header.frame_id = source_frame
        stamped.header.stamp = rclpy.time.Time().to_msg()
        stamped.pose = pose
        try:
            return self.tf_buffer.transform(
                stamped,
                target_frame,
                timeout=rclpy.duration.Duration(seconds=1.0),
            ).pose
        except (LookupException, ConnectivityException, ExtrapolationException) as exc:
            raise RuntimeError(f"TF transform failed: {exc}") from exc

    def make_grasp_pose(self, base_pose: Pose, gripper_axis_base_xy=None) -> Pose:
        grasp_pose = copy.deepcopy(base_pose)
        orientation_quat = VERTICAL_DOWN_GRASP_QUATERNION.copy()
        applied_gripper_axis = None
        if gripper_axis_base_xy is not None:
            pca_aligned_quat, applied_gripper_axis = build_grasp_orientation_perpendicular_to_axis(
                VERTICAL_DOWN_GRASP_QUATERNION,
                gripper_axis_base_xy,
            )
            if pca_aligned_quat is not None:
                orientation_quat = pca_aligned_quat
        set_pose_orientation_from_quaternion(grasp_pose, orientation_quat)
        if applied_gripper_axis is not None:
            axis_yaw = math.atan2(applied_gripper_axis[1], applied_gripper_axis[0])
            self.get_logger().info(
                f"Using PCA-aligned grasp orientation: gripper_axis_yaw={math.degrees(axis_yaw):.1f} deg"
            )
        grasp_pose.position.y += -0.015
        return grasp_pose

    def _compute_grasp_pose(self, detection: dict, obj_name: str):
        if not detection.get("ok"):
            raise RuntimeError(f"Detection failed for '{obj_name}': {detection.get('error')}")
        source_frame = detection.get("frame_id") or self.args.camera_frame
        selection = self.select_grasp_target(detection, obj_name)
        grasp_xyz, _ = store_grasp_pose_reference(selection)
        detected_pose = pose_from_xyz(grasp_xyz)
        base_pose = self.transform_pose(detected_pose, source_frame, "base_link")
        selection["centroid_xyz_base"] = [float(base_pose.position.x), float(base_pose.position.y), float(base_pose.position.z)]

        target_pose_base = None
        target_xyz = selection.get("target_xyz_m")
        if target_xyz is not None:
            try:
                target_pose_base = self.transform_pose(pose_from_xyz(target_xyz), source_frame, "base_link")
                selection["target_xyz_base"] = [
                    float(target_pose_base.position.x),
                    float(target_pose_base.position.y),
                    float(target_pose_base.position.z),
                ]
            except Exception as exc:
                selection["target_xyz_base_error"] = str(exc)
        
        xyz_major_axis = selection.get("xyz_major_axis")
        major_axis_base_3d = None
        if xyz_major_axis is not None:
            major_axis_base_3d = self.transform_vector_to_base(xyz_major_axis, source_frame)
            
        if major_axis_base_3d is not None and np.linalg.norm(major_axis_base_3d) > 1e-6:
            selection["xyz_major_axis_base"] = major_axis_base_3d.tolist()
            
            z_vertical = np.array([0.0, 0.0, -1.0])
            z_approach = z_vertical - np.dot(z_vertical, major_axis_base_3d) * major_axis_base_3d
            if np.linalg.norm(z_approach) < 1e-6:
                z_approach = np.array([1.0, 0.0, 0.0])
            z_approach = normalize_vector(z_approach)
            
            x_closing = normalize_vector(np.cross(major_axis_base_3d, z_approach))
            y_dir = normalize_vector(np.cross(z_approach, x_closing))
            
            matrix = np.column_stack((x_closing, y_dir, z_approach))
            quat = matrix_to_quaternion(matrix)
            
            grasp_pose = copy.deepcopy(base_pose)
            grasp_pose.orientation.x = float(quat[0])
            grasp_pose.orientation.y = float(quat[1])
            grasp_pose.orientation.z = float(quat[2])
            grasp_pose.orientation.w = float(quat[3])
            
            # Apply the surface clearance offset along the 3D approach vector
            # grasp_surface_clearance is negative (e.g. -0.012), meaning push INTO the object.
            # z_approach points into the object. So we move in direction of z_approach by abs(clearance).
            clearance = self.args.grasp_surface_clearance
            grasp_pose.position.x += -clearance * float(z_approach[0])
            grasp_pose.position.y += -clearance * float(z_approach[1])
            grasp_pose.position.z += -clearance * float(z_approach[2])
            
            # The Y offset is likely a camera/base calibration offset, so apply it in global base_link Y.
            grasp_pose.position.y += self.args.grasp_y_offset
            
            selection["gripper_axis_rule"] = "3d_perpendicular_to_pca_major_axis"
        else:
            pca_major_axis_base_xy = self.transform_selection_axis_to_base_xy(
                selection, source_frame, base_pose, grasp_xyz,
            )
            gripper_axis_base_xy = None
            if pca_major_axis_base_xy is not None:
                gripper_axis_base_xy = np.asarray(
                    [-pca_major_axis_base_xy[1], pca_major_axis_base_xy[0]],
                    dtype=np.float64,
                )
                selection["pca_major_axis_base_xy"] = pca_major_axis_base_xy.astype(float).tolist()
                selection["pca_major_axis_base_yaw_rad"] = float(
                    math.atan2(pca_major_axis_base_xy[1], pca_major_axis_base_xy[0])
                )
                selection["gripper_axis_base_xy"] = gripper_axis_base_xy.astype(float).tolist()
                selection["gripper_axis_base_yaw_rad"] = float(
                    math.atan2(gripper_axis_base_xy[1], gripper_axis_base_xy[0])
                )
                selection["gripper_axis_rule"] = "perpendicular_to_pca_major_axis"
                
            grasp_pose = self.make_grasp_pose(base_pose, gripper_axis_base_xy)
        
        # Publish the final grasp pose for RViz
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = "base_link"
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.pose = grasp_pose
        self.grasp_pose_publisher.publish(pose_msg)
        
        # PREVIEW CORRECTED POSE (from grasp_database.json)
        corrected_grasp_pose = None
        grasp_correction = None
        try:
            perception_features = extract_perception_features(detection)
            measured_area, measured_area_source = _measured_object_area_from_features(perception_features)
                
            database = _load_grasp_database()
            matched_name, object_configs = _lookup_object_grasp_configs(database, obj_name)
            item_state, grasp_config, _ = _select_state_config(object_configs, measured_area)
            grasp_transform = _extract_grasp_transform(grasp_config)
            
            corrected_grasp_pose = apply_grasp_transform(grasp_pose, grasp_transform)
            grasp_value = grasp_config.get("grasp_value")
            grasp_correction = {
                "matched_name": matched_name,
                "selected_state": item_state,
                "measured_area_m2": measured_area,
                "measured_area_source": measured_area_source,
                "transform": grasp_transform,
                "grasp_value_deg": float(grasp_value) if grasp_value is not None else None,
            }
            selection["grasp_database_correction"] = grasp_correction
            selection["corrected_grasp_pose_base"] = pose_to_dict(corrected_grasp_pose)
            
            corrected_msg = PoseStamped()
            corrected_msg.header = pose_msg.header
            corrected_msg.pose = corrected_grasp_pose
            self.corrected_grasp_pose_publisher.publish(corrected_msg)
            
            self.get_logger().info(
                f"Published preview of corrected grasp pose for '{obj_name}' "
                f"(state={item_state}, area={measured_area}, source={measured_area_source})"
            )
        except Exception as e:
            selection["grasp_database_correction_error"] = str(e)
            self.get_logger().warn(f"Failed to publish preview of corrected grasp pose: {e}")

        self.save_grasp_pose_debug(
            obj_name,
            detection,
            selection,
            source_frame,
            base_pose,
            target_pose_base,
            grasp_pose,
            corrected_grasp_pose,
            grasp_correction,
        )
        
        # Visualize actual PCA and centroid
        marker_array = MarkerArray()
        
        # Centroid Sphere (Yellow)
        centroid_marker = Marker()
        centroid_marker.header.frame_id = "base_link"
        centroid_marker.header.stamp.sec = 0
        centroid_marker.header.stamp.nanosec = 0
        centroid_marker.ns = "pca_debug"
        centroid_marker.id = 0
        centroid_marker.type = Marker.SPHERE
        centroid_marker.action = Marker.ADD
        centroid_marker.pose.position = base_pose.position
        centroid_marker.scale.x = 0.02
        centroid_marker.scale.y = 0.02
        centroid_marker.scale.z = 0.02
        centroid_marker.color.r = 1.0
        centroid_marker.color.g = 1.0
        centroid_marker.color.b = 0.0
        centroid_marker.color.a = 0.8
        marker_array.markers.append(centroid_marker)

        if target_pose_base is not None:
            target_marker = Marker()
            target_marker.header.frame_id = "base_link"
            target_marker.header.stamp.sec = 0
            target_marker.header.stamp.nanosec = 0
            target_marker.ns = "pca_debug"
            target_marker.id = 2
            target_marker.type = Marker.SPHERE
            target_marker.action = Marker.ADD
            target_marker.pose.position = target_pose_base.position
            target_marker.scale.x = 0.025
            target_marker.scale.y = 0.025
            target_marker.scale.z = 0.025
            target_marker.color.r = 1.0
            target_marker.color.g = 0.0
            target_marker.color.b = 0.0
            target_marker.color.a = 0.9
            marker_array.markers.append(target_marker)
        
        if major_axis_base_3d is not None:
            # 3D PCA Arrow (Cyan)
            pca_arrow = Marker()
            pca_arrow.header.frame_id = "base_link"
            pca_arrow.header.stamp.sec = 0
            pca_arrow.header.stamp.nanosec = 0
            pca_arrow.ns = "pca_debug"
            pca_arrow.id = 1
            pca_arrow.type = Marker.ARROW
            pca_arrow.action = Marker.ADD
            p1 = base_pose.position
            p2 = Point()
            p2.x = p1.x + major_axis_base_3d[0] * 0.15
            p2.y = p1.y + major_axis_base_3d[1] * 0.15
            p2.z = p1.z + major_axis_base_3d[2] * 0.15
            pca_arrow.points = [p1, p2]
            pca_arrow.scale.x = 0.005
            pca_arrow.scale.y = 0.01
            pca_arrow.scale.z = 0.02
            pca_arrow.color.r = 0.0
            pca_arrow.color.g = 1.0
            pca_arrow.color.b = 1.0
            pca_arrow.color.a = 1.0
            marker_array.markers.append(pca_arrow)
            
        points_path = selection.get("points_path")
        if points_path and os.path.isfile(points_path):
            try:
                points = np.load(points_path)
                
                # Filter out NaN values
                points = points.reshape(-1, points.shape[-1])[:, :3]
                points = points[np.isfinite(points).all(axis=1)]
                
                # Create a proper PointCloud2 message
                from sensor_msgs_py.point_cloud2 import create_cloud_xyz32
                from std_msgs.msg import Header
                
                header = Header()
                header.frame_id = source_frame
                # timestamp 0 bypasses strict TF timing checks
                header.stamp.sec = 0
                header.stamp.nanosec = 0
                
                pc2 = create_cloud_xyz32(header, points)
                self.pc2_publisher.publish(pc2)
                self.get_logger().info(f"Published {len(points)} valid points to /pca_debug_cloud")
            except Exception as e:
                self.get_logger().error(f"Failed to load point cloud for visualization: {e}")
            
        self.pca_debug_publisher.publish(marker_array)
        
        return grasp_pose, selection

    def transform_vector_to_base(self, vector_source, source_frame):
        p1 = pose_from_xyz([0.0, 0.0, 0.0])
        p2 = pose_from_xyz(vector_source)
        try:
            p1_base = self.transform_pose(p1, source_frame, "base_link")
            p2_base = self.transform_pose(p2, source_frame, "base_link")
        except Exception as exc:
            self.get_logger().error(f"Failed to transform vector: {exc}")
            return None
        v_base = np.array([
            p2_base.position.x - p1_base.position.x,
            p2_base.position.y - p1_base.position.y,
            p2_base.position.z - p1_base.position.z,
        ], dtype=np.float64)
        return normalize_vector(v_base)

    def transform_selection_axis_to_base_xy(self, grasp_selection, source_frame, base_pose, axis_origin_xyz_m):
        axis = grasp_selection.get("xy_major_axis")
        if axis is None:
            return None
        try:
            axis_source = np.asarray([float(axis[0]), float(axis[1]), 0.0], dtype=np.float64)
            origin = np.asarray(axis_origin_xyz_m, dtype=np.float64)
        except (TypeError, ValueError, IndexError):
            return None
        if origin.shape[0] < 3:
            return None
        axis_source = normalize_vector(axis_source)
        if axis_source is None:
            return None
        endpoint = origin[:3].copy()
        endpoint[:3] += axis_source * GRASP_AXIS_ENDPOINT_OFFSET_M
        endpoint_base = self.transform_pose(pose_from_xyz(endpoint), source_frame, "base_link")
        axis_base = np.asarray([
            endpoint_base.position.x - base_pose.position.x,
            endpoint_base.position.y - base_pose.position.y,
            0.0,
        ], dtype=np.float64)
        axis_base = normalize_vector(axis_base)
        if axis_base is None:
            return None
        return axis_base[:2]


class ManipulatorExecutor(Node):
    """Topic-based executor for one committed PDDL action at a time."""

    def __init__(self, args, coordinator: PddlTampServer):
        super().__init__("pddl_manipulator_executor")
        self.args = args
        self.coordinator = coordinator
        self.command_lock = threading.Lock()
        self.result_pub = self.create_publisher(String, args.executor_result_topic, 10)
        self.command_sub = self.create_subscription(
            String,
            args.executor_command_topic,
            self.handle_command,
            10,
        )
        self.get_logger().info(
            f"Manipulator executor ready. Listening on '{args.executor_command_topic}', "
            f"publishing results on '{args.executor_result_topic}'."
        )

    def publish_result(self, payload: dict) -> None:
        msg = String()
        msg.data = json.dumps(payload, sort_keys=True)
        self.result_pub.publish(msg)

    def handle_command(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f"[Executor] Ignoring malformed command: {exc}")
            return
        command_id = str(payload.get("command_id") or "")
        action_payload = payload.get("action") or {}
        if not command_id or not action_payload:
            self.get_logger().warn("[Executor] Ignoring command without command_id/action.")
            return

        with self.command_lock:
            try:
                action = PlanAction(
                    str(action_payload["name"]),
                    tuple(str(arg) for arg in action_payload.get("args", [])),
                    source=str(action_payload.get("source") or "executor-command"),
                )
                if not action.args:
                    raise ValueError("Action has no arguments.")
                obj = object_state_from_command(payload.get("selected_object"), action.args[0])
                state = PredicateState(goals=[], objects={obj.name: obj})
                prefetch_payload = payload.get("prefetch")
                use_storage_place_prefetch = (
                    action.name == "move-target-to-goal"
                    and len(action.args) >= 3
                    and action.args[2] in {"left_storage", "right_storage"}
                )

                def on_storage_place(_prefetch_payload=prefetch_payload, _command_id=command_id):
                    self.coordinator.start_scene_prefetch(
                        _prefetch_payload,
                        _command_id,
                        trigger="storage-place",
                    )

                def on_observe_ready(_prefetch_payload=prefetch_payload, _command_id=command_id):
                    self.coordinator.start_scene_prefetch(
                        _prefetch_payload,
                        _command_id,
                        trigger="observe-ready",
                    )

                context = ActionContext(
                    self.coordinator,
                    on_before_idle=on_storage_place if use_storage_place_prefetch else None,
                    on_observe_ready=None if use_storage_place_prefetch else on_observe_ready,
                    skip_observe_after_place=use_storage_place_prefetch,
                )
                self.get_logger().info(
                    f"[Executor] Executing committed action {action.pddl()} command_id={command_id}"
                )
                result = execute_action(context, action, state)
            except Exception as exc:
                self.get_logger().error(
                    f"[Executor] Command failed command_id={command_id}: {exc}\n{traceback.format_exc()}"
                )
                result = {
                    "ok": False,
                    "action": action_payload,
                    "error": str(exc),
                    "dry_run": bool(self.args.dry_run),
                }

        self.publish_result(
            {
                "event": "executor_result",
                "command_id": command_id,
                "stamp_sec": time.time(),
                "result": result,
            }
        )


def parse_joint_list(text: str) -> list[float]:
    values = [float(part.strip()) for part in str(text).split(",") if part.strip()]
    if len(values) != 6:
        raise ValueError(f"--observe-joints must contain exactly 6 comma-separated values, got {len(values)}")
    return values


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="PDDL-based TAMP command server.")
    parser.add_argument("--service-name", default="pddl_tamp_command")
    parser.add_argument("--command-topic", default="/task_commands")
    parser.add_argument("--log-topic", default="/pddl_tamp_log")
    parser.add_argument("--perception-service", default="detect_object_top_rgbd_seg_crop")
    parser.add_argument("--wrist-perception-service", default="detect_object_rgbd_seg_crop")
    parser.add_argument("--camera-frame", default="camera_color_optical_frame")
    parser.add_argument("--perception-timeout", type=float, default=15.0)
    parser.add_argument("--tf-timeout", type=float, default=5.0)
    parser.add_argument("--action-timeout", type=float, default=300.0)
    parser.add_argument("--prefetch-wait-timeout", type=float, default=5.0)
    parser.add_argument("--warm-start-wait-timeout", type=float, default=30.0)
    parser.add_argument("--disable-warm-start", action="store_true", help="Do not scan the initial scene before the first command.")
    parser.add_argument("--executor-command-topic", default="/pddl_action_commands")
    parser.add_argument("--executor-result-topic", default="/pddl_action_results")
    parser.add_argument(
        "--observe-joints",
        default=",".join(str(value) for value in OBSERVE_JOINTS),
        help="Comma-separated six-joint base pose used before the Cartesian -X observation retreat.",
    )
    parser.add_argument(
        "--observe-x-offset",
        type=float,
        default=OBSERVE_X_OFFSET_M,
        help="Base-frame -X distance in meters used to retreat the gripper for top-view observation.",
    )
    parser.add_argument(
        "--observe-y-offset",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--observe-retreat-duration",
        type=float,
        default=OBSERVE_RETREAT_DURATION_S,
        help="Duration in seconds for the Cartesian -X observation retreat.",
    )
    parser.add_argument("--max-steps", type=int, default=int(os.environ.get("PDDL_TAMP_MAX_STEPS", "8")))
    parser.add_argument("--one-step", action="store_true", help="Execute only the first selected physical action.")
    parser.add_argument("--scan-goals-only", action="store_true", help="Only scan requested targets instead of all five fixed targets.")
    parser.add_argument("--no-gemini", action="store_true", help="Disable Gemini parsing and use the rule-based parser.")
    parser.add_argument("--debug", action="store_true", help="Show and save a perception+predicate debug view each planning step.")
    parser.add_argument("--debug-window", action="store_true", help="With --debug, also open an OpenCV window when a GUI display is usable.")
    parser.add_argument("--debug-wait", action="store_true", help="With --debug-window, wait for a key press at each planning step.")
    parser.add_argument("--external-perception", action="store_true", help="Use an already running top-view perception service.")
    parser.add_argument("--dry-run", action="store_true", help="Plan and bind actions without moving the arm.")
    parser.add_argument("--no-home", action="store_true", help="Do not return to home before executing a command.")
    parser.add_argument("--approach-height", type=float, default=0.15)
    parser.add_argument("--final-z-offset", type=float, default=0.0)
    parser.add_argument("--grasp-surface-clearance", type=float, default=-0.012)
    parser.add_argument("--grasp-depth-local-radius", type=float, default=0.0005)
    parser.add_argument("--grasp-depth-percentile", type=float, default=10.0)
    parser.add_argument("--grasp-y-offset", type=float, default=-0.015)
    parser.add_argument("--gripper-force", type=float, default=0.5)
    parser.add_argument("--gripper-close-pos", type=float, default=0.5)
    parser.add_argument("--gripper-settle-time", type=float, default=0.4)
    parser.add_argument("--gripper-result-timeout-margin", type=float, default=8.0)
    parser.add_argument("--strict-gripper-result", action="store_true")
    parsed = parser.parse_args(rclpy.utilities.remove_ros_args(args=argv or sys.argv)[1:])
    parsed.observe_joints = parse_joint_list(parsed.observe_joints)
    return parsed


def main(argv=None):
    load_dotenv()
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s %(message)s")
    top_view_module = None if args.external_perception else load_top_view_perception_module()
    ros_argv = argv
    rclpy.init(args=ros_argv)
    arm = ArmClient()
    if not args.dry_run and not args.no_home:
        move_arm_to_observe_pose(
            arm,
            args.observe_joints,
            args.observe_x_offset,
            args.observe_retreat_duration,
            logger=arm.get_logger(),
            reason="startup",
        )
    perception_node = None
    wrist_perception_node = None
    tf_node = TfNode()
    if top_view_module is not None:
        top_defaults = dict(top_view_module.DEFAULTS)
        top_defaults["service_name"] = args.perception_service
        perception_node = top_view_module.RgbdSegCropServiceNode(
            node_name="pddl_top_view_seg_crop_service_node",
            default_params=top_defaults,
        )
        perception_node.get_logger().info("Embedded top-view perception is running inside PDDL TAMP server.")
        wrist_defaults = dict(WRIST_VIEW_DEFAULTS)
        wrist_defaults["service_name"] = args.wrist_perception_service
        wrist_perception_node = RgbdSegCropServiceNode(
            node_name="pddl_wrist_seg_crop_service_node",
            default_params=wrist_defaults,
        )
        wrist_perception_node.get_logger().info("Embedded wrist-camera perception is running inside PDDL TAMP server.")
    server = PddlTampServer(args, tf_node.tf_buffer, arm)
    manipulator_executor = ManipulatorExecutor(args, server)
    child_processes = []
    try:
        executor = MultiThreadedExecutor(num_threads=4)
        if perception_node is not None:
            executor.add_node(perception_node)
        if wrist_perception_node is not None:
            executor.add_node(wrist_perception_node)
        executor.add_node(tf_node)
        executor.add_node(server)
        executor.add_node(manipulator_executor)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if "executor" in locals():
            executor.shutdown()
        server.destroy_node()
        manipulator_executor.destroy_node()
        tf_node.destroy_node()
        if perception_node is not None:
            perception_node.destroy_node()
        if wrist_perception_node is not None:
            wrist_perception_node.destroy_node()
        arm.destroy_node()
        rclpy.shutdown()
        stop_child_processes(child_processes)


if __name__ == "__main__":
    main()
