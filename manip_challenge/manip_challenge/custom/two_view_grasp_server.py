#!/usr/bin/env python3
"""Two-view perception command server for pick-and-place motion.

Run this server continuously, then send commands with two_view_grasp_client.py:

    python3 two_view_grasp_client.py banana left

The server embeds the top-view perception node in this same process, chooses a
grasp point from the segmented point cloud, transforms it into base_link, and
then uses the custom motion/grasping pipeline to move the object to the
requested destination.
"""

import argparse
import copy
import importlib.util
import json
import math
import os
import random
import re
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

try:
    import cv2
except ImportError:
    cv2 = None


def ensure_ros_python():
    ros_python = "/usr/bin/python3"
    if sys.version_info[:2] != (3, 10) and os.path.exists(ros_python):
        os.execv(ros_python, [ros_python, *sys.argv])


ensure_ros_python()

import numpy as np
import rclpy
import tf2_geometry_msgs  # noqa: F401  Needed for PoseStamped TF transforms.
from geometry_msgs.msg import Pose, PoseStamped
from std_msgs.msg import String
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from riro_srvs.srv import StringString
from tf2_ros import Buffer, ConnectivityException, ExtrapolationException, LookupException, TransformListener
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


if not hasattr(np, "mat"):
    np.mat = np.asmatrix

CUSTOM_DIR = Path(__file__).resolve().parent
PACKAGE_SRC = CUSTOM_DIR.parents[1]
WORKSPACE_ROOT = PACKAGE_SRC.parent
ASSIGNMENT2_SRC = WORKSPACE_ROOT / "assignment_2"
TWO_VIEW_SERVER = CUSTOM_DIR / "perception" / "two-view" / "top_view_seg_server.py"

for path in (str(PACKAGE_SRC), str(ASSIGNMENT2_SRC), str(WORKSPACE_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from assignment_2.move_joint import ArmClient  # noqa: E402
from manip_challenge import move_gripper  # noqa: E402
import manip_challenge.misc as misc  # noqa: E402
from manip_challenge.custom.grasping.gripper_control import JOINT_NAME as CUSTOM_GRIPPER_JOINT_NAME  # noqa: E402


def executor_safe_gripper_goto(node, pos, force=1.0, timeout=3.0, **kwargs):
    """Send a gripper action without recursively spinning an executor-owned node."""
    timeout = float(timeout)
    result_timeout_margin = float(getattr(node, "gripper_result_timeout_margin", 8.0))
    strict_result = bool(getattr(node, "strict_gripper_result", False))

    if not hasattr(node, "_two_view_gripper_client"):
        node._two_view_gripper_client = ActionClient(
            node,
            FollowJointTrajectory,
            "/gripper_controller/follow_joint_trajectory",
            callback_group=getattr(node, "callback_group", None),
        )

    client = node._two_view_gripper_client
    if not client.wait_for_server(timeout_sec=5.0):
        raise RuntimeError("Timed out waiting for gripper action server.")

    goal = FollowJointTrajectory.Goal()
    goal.trajectory = JointTrajectory()
    goal.trajectory.joint_names = [CUSTOM_GRIPPER_JOINT_NAME]
    goal.trajectory.points = [
        JointTrajectoryPoint(
            positions=[float(pos)],
            velocities=[0.0],
            time_from_start=Duration(sec=int(timeout), nanosec=int((timeout - int(timeout)) * 1e9)),
        )
    ]

    goal_future = client.send_goal_async(goal)
    deadline = time.monotonic() + timeout + result_timeout_margin
    while rclpy.ok() and not goal_future.done():
        if time.monotonic() > deadline:
            raise TimeoutError("Timed out sending gripper goal.")
        time.sleep(0.01)

    goal_handle = goal_future.result()
    if goal_handle is None or not goal_handle.accepted:
        raise RuntimeError("Gripper goal was rejected.")

    result_future = goal_handle.get_result_async()
    deadline = time.monotonic() + timeout + result_timeout_margin
    while rclpy.ok() and not result_future.done():
        if time.monotonic() > deadline:
            message = (
                "Timed out waiting for gripper action result. Continuing because "
                "a grasp can stop before the requested close position when it contacts an object."
            )
            if strict_result:
                raise TimeoutError("Timed out waiting for gripper motion.")
            node.get_logger().warn(message)
            try:
                cancel_future = goal_handle.cancel_goal_async()
                cancel_deadline = time.monotonic() + 0.75
                while rclpy.ok() and not cancel_future.done() and time.monotonic() < cancel_deadline:
                    time.sleep(0.01)
                if cancel_future.done():
                    node.get_logger().info("Cancelled timed-out gripper goal; controller should hold current position.")
            except Exception as exc:
                node.get_logger().warn(f"Could not cancel timed-out gripper goal: {exc}")
            settle_time = float(getattr(node, "gripper_settle_time", 0.0))
            if settle_time > 0.0:
                time.sleep(settle_time)
            return {
                "ok": True,
                "timed_out": True,
                "position": float(pos),
                "timeout": timeout,
            }
        time.sleep(0.01)

    result = result_future.result()
    if result is None:
        message = f"Gripper action returned no result: {result_future.exception()!r}"
        if strict_result:
            raise RuntimeError(message)
        node.get_logger().warn(message)
        return {"ok": True, "missing_result": True, "position": float(pos)}
    return result


def executor_safe_gripper_open(node, force=1.0, timeout=1.0, gripper_open_pos=0.0, **kwargs):
    node.get_logger().info("Opening gripper.")
    return executor_safe_gripper_goto(node, gripper_open_pos, force=force, timeout=timeout, **kwargs)


def executor_safe_gripper_close(node, force=1.0, timeout=3.0, gripper_close_pos=None, **kwargs):
    force = float(getattr(node, "gripper_force", force))
    if gripper_close_pos is None:
        gripper_close_pos = float(getattr(node, "gripper_close_pos", 0.8))
    else:
        gripper_close_pos = float(gripper_close_pos)
    node.get_logger().info("Closing gripper.")
    return executor_safe_gripper_goto(node, gripper_close_pos, force=force, timeout=timeout, **kwargs)


move_gripper.gripperGotoPos = executor_safe_gripper_goto
move_gripper.gripper_open = executor_safe_gripper_open
move_gripper.gripper_close = executor_safe_gripper_close

from manip_challenge.custom.motion.motion import PLACE_CONFIGS, execute_pick_place_sequence  # noqa: E402
from manip_challenge.custom.tamp.executor import execute_tamp_command  # noqa: E402
from manip_challenge.custom.tamp.nlp import parse_task_goals  # noqa: E402


# wrist_1=-1.0 keeps shoulder_lift + elbow + wrist_1 == -pi/2 for the vertical home posture.
HOME_JOINTS = [0.0, -np.pi / 2.0, 1.0, -1.0, -np.pi / 2.0, 0.0]
# Keep the commanded grasp approach exactly vertical.
VERTICAL_DOWN_GRASP_QUATERNION = np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
GRASP_AXIS_ENDPOINT_OFFSET_M = 0.05

DESTINATION_ALIASES = {
    "left": "left storage",
    "left_storage": "left storage",
    "left storage": "left storage",
    "right": "right storage",
    "right_storage": "right storage",
    "right storage": "right storage",
    "book": "bookshelf",
    "shelf": "bookshelf",
    "bookshelf": "bookshelf",
}


class CommandParseError(ValueError):
    pass


def home_joints_for_pan(pan):
    joints = list(HOME_JOINTS)
    joints[0] = float(pan)
    return joints


def normalize_object_name(name):
    text = str(name or "").strip().lower()
    text = re.sub(r"^(a|an|the)\s+", "", text)
    return text.replace("-", "_").replace(" ", "_")


def normalize_destination(name):
    text = str(name or "").strip().lower().replace("-", " ").replace("_", " ")
    text = re.sub(r"^(the|a|an)\s+", "", text)
    return DESTINATION_ALIASES.get(text, DESTINATION_ALIASES.get(text.replace(" ", "_"), text))


def parse_command(text):
    """Return (action, object_name, destination) from a single short command."""
    raw = str(text or "").strip()
    if not raw:
        raise CommandParseError("Empty command. Try: banana left")

    lowered = raw.lower().strip()
    if lowered in {"home", "idle", "reset"}:
        return lowered, None, None

    move_match = re.search(
        r"^(?:move|pick|grasp)\s+(?:a\s+|an\s+|the\s+)?(.+?)\s+"
        r"(?:to|into|onto)\s+(?:the\s+)?(.+)$",
        lowered,
    )
    if move_match:
        return "approach", normalize_object_name(move_match.group(1)), normalize_destination(move_match.group(2))

    tokens = lowered.split()
    for suffix_len in (2, 1):
        if len(tokens) > suffix_len:
            suffix = " ".join(tokens[-suffix_len:])
            destination = normalize_destination(suffix)
            if destination in set(DESTINATION_ALIASES.values()):
                obj = " ".join(tokens[:-suffix_len])
                return "approach", normalize_object_name(obj), destination

    return "approach", normalize_object_name(lowered), None


def parse_commands(text):
    """Parse one or more sentences into a list of (action, object_name, destination).

    Handles:
      - Simple commands: "banana left", "home"
      - Natural language: "Move a banana to left storage."
      - Multiple objects: "Move a banana, a coke can, and a hammer to left storage."
      - Multiple sentences: "Move a banana to left. Move a coke can to right."
    """
    raw = str(text or "").strip()
    if not raw:
        raise CommandParseError("Empty command.")

    lowered = raw.lower().strip()
    if lowered in {"home", "idle", "reset"}:
        return [(lowered, None, None)]

    results = []

    # 문장 단위로 분리
    sentences = re.split(r"[.!?\n]+", raw)
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        lowered_s = sentence.lower()

        # "Move X, Y, and Z to destination" — 여러 물체 동시 처리 (parsing.py 로직)
        multi_match = re.search(
            r"(?:move|pick|grasp)\s+(.*?)\s+(?:to|into|onto)\s+(?:the\s+)?([\w\s]+?)$",
            lowered_s,
            re.IGNORECASE,
        )
        if multi_match:
            objects_str = multi_match.group(1)
            destination = normalize_destination(multi_match.group(2).strip())
            objects_str = re.sub(r"\band\b", ",", objects_str, flags=re.IGNORECASE)
            for raw_obj in objects_str.split(","):
                obj = re.sub(r"^\s*(a|an|the)\s+", "", raw_obj.strip(), flags=re.IGNORECASE).strip()
                obj = normalize_object_name(obj)
                if obj:
                    results.append(("approach", obj, destination))
            continue

        # 단순 단일 명령 fallback
        try:
            results.append(parse_command(sentence))
        except CommandParseError:
            pass

    if not results:
        raise CommandParseError(f"Could not parse: '{text}'")

    return results


def pose_from_xyz(xyz):
    pose = Pose()
    pose.position.x = float(xyz[0])
    pose.position.y = float(xyz[1])
    pose.position.z = float(xyz[2])
    pose.orientation.w = 1.0
    return pose


def pose_to_dict(pose):
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


def grasp_pose_xyz_from_selection(selection):
    target = np.asarray(selection["target_xyz_m"], dtype=np.float64)
    if target.shape[0] < 3:
        raise ValueError("target_xyz_m must contain at least 3 values")

    raw = selection.get("raw_centroid_xyz_m")
    if raw is not None:
        raw = np.asarray(raw, dtype=np.float64)
        if raw.shape[0] >= 2 and np.isfinite(raw[:2]).all():
            return (
                np.asarray([raw[0], raw[1], target[2]], dtype=np.float64),
                "raw_centroid_xy_with_adjusted_target_z",
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
    x = float(quaternion[0])
    y = float(quaternion[1])
    z = float(quaternion[2])
    w = float(quaternion[3])
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0.0:
        return np.eye(3, dtype=np.float64)
    x /= norm
    y /= norm
    z /= norm
    w /= norm

    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def orientation_to_matrix(orientation):
    return quaternion_to_matrix([orientation.x, orientation.y, orientation.z, orientation.w])


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

    quaternion = normalize_vector([x, y, z, w])
    if quaternion is None:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return quaternion


def set_pose_orientation_from_quaternion(pose, quaternion):
    pose.orientation.x = float(quaternion[0])
    pose.orientation.y = float(quaternion[1])
    pose.orientation.z = float(quaternion[2])
    pose.orientation.w = float(quaternion[3])


def build_grasp_orientation_perpendicular_to_axis(reference_quaternion, gripper_axis_base_xy):
    reference_matrix = quaternion_to_matrix(reference_quaternion)
    tool_z_axis = normalize_vector(reference_matrix[:, 2])
    # In the ArmClient tool frame, local X is the gripper closing/opening axis.
    # Keep the reference approach axis, and rotate only the in-plane gripper axis.
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


def calc_rot_time(start_angle, target_angle, sec_per_rad=1.2, min_time=0.8):
    return max(min_time, abs(float(target_angle) - float(start_angle)) * sec_per_rad)


def normalize_to_uint8(values, valid=None):
    values = np.asarray(values, dtype=np.float64)
    if valid is None:
        valid = np.isfinite(values)
    valid = valid & np.isfinite(values)
    out = np.zeros(values.shape, dtype=np.uint8)
    if not np.any(valid):
        return out
    low, high = np.percentile(values[valid], [2.0, 98.0])
    if high <= low:
        high = low + 1e-6
    clipped = np.clip(values, low, high)
    out[valid] = np.round((clipped[valid] - low) * 255.0 / (high - low)).astype(np.uint8)
    return out


def finite_xyz_points(points):
    points = np.asarray(points, dtype=np.float64).reshape(-1, np.asarray(points).shape[-1])[:, :3]
    return points[np.isfinite(points).all(axis=1) & (points[:, 2] > 0.0)]


def raw_centroid_xyz(points):
    points = finite_xyz_points(points)
    if len(points) == 0:
        return np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
    return np.mean(points[:, :3], axis=0)


def pca_axes_xy(points):
    points = finite_xyz_points(points)
    xy = points[:, :2].astype(np.float64)
    center = raw_centroid_xyz(points)[:2]
    centered = xy - center
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    major = vt[0]
    minor = vt[1]
    if major[0] < 0:
        major = -major
        minor = -minor
    return center, major, minor


def pca_bbox_xy(points, center, major, minor, percentile_low=2.0, percentile_high=98.0):
    centered = points[:, :2].astype(np.float64) - center
    along_major = centered @ major
    along_minor = centered @ minor
    major_min, major_max = np.percentile(along_major, [percentile_low, percentile_high])
    minor_min, minor_max = np.percentile(along_minor, [percentile_low, percentile_high])
    length_m = float(max(0.0, major_max - major_min))
    width_m = float(max(0.0, minor_max - minor_min))
    return {
        "center_xy_m": center.astype(float).tolist(),
        "length_m": length_m,
        "width_m": width_m,
        "area_m2": length_m * width_m,
        "major_min_m": float(major_min),
        "major_max_m": float(major_max),
        "minor_min_m": float(minor_min),
        "minor_max_m": float(minor_max),
        "percentile_low": float(percentile_low),
        "percentile_high": float(percentile_high),
    }


def choose_grasp_target_from_points(points, label="", min_bin_points=20):
    """Choose a grasp point from visible object points instead of raw centroid.

    For elongated/curved objects, the global centroid can fall in an awkward
    location. This chooses a locally thick, central band along the top-down
    major axis and returns the median real surface point in that band.
    """
    points = finite_xyz_points(points)
    if len(points) < max(8, min_bin_points):
        centroid = raw_centroid_xyz(points)
        return {
            "method": "raw_centroid_fallback",
            "target_xyz_m": centroid.astype(float).tolist(),
            "raw_centroid_xyz_m": centroid.astype(float).tolist(),
            "reason": f"too_few_points:{len(points)}",
        }

    xy_center, major, minor = pca_axes_xy(points)
    centered = points[:, :2] - xy_center
    along = centered @ major
    across = centered @ minor
    raw_centroid = np.median(points[:, :3], axis=0)
    pca_bbox = pca_bbox_xy(points, xy_center, major, minor)

    low, high = np.percentile(along, [12.0, 88.0])
    if high <= low:
        target = raw_centroid
        return {
            "method": "raw_centroid_fallback",
            "target_xyz_m": target.astype(float).tolist(),
            "raw_centroid_xyz_m": raw_centroid.astype(float).tolist(),
            "reason": "degenerate_major_axis",
            "xy_major_axis": major.astype(float).tolist(),
            "xy_minor_axis": minor.astype(float).tolist(),
            "pca_bbox": pca_bbox,
        }

    bin_count = 18
    edges = np.linspace(low, high, bin_count + 1)
    min_points = max(min_bin_points, int(round(len(points) * 0.025)))
    best = None
    span = max(high - low, 1e-6)
    label_norm = normalize_object_name(label)

    for index in range(bin_count):
        mask = (along >= edges[index]) & (along < edges[index + 1])
        count = int(np.count_nonzero(mask))
        if count < min_points:
            continue
        local_across = across[mask]
        local_z = points[mask, 2]
        width = float(np.percentile(local_across, 90.0) - np.percentile(local_across, 10.0))
        z_spread = float(np.percentile(local_z, 90.0) - np.percentile(local_z, 10.0))
        center_s = float((edges[index] + edges[index + 1]) * 0.5)
        center_penalty = abs(center_s - np.median(along)) / span

        # Bananas benefit from avoiding the very ends and preferring a thick
        # central section. The same score is still useful for other objects.
        score = width + 0.0015 * math.sqrt(count) - 0.04 * center_penalty - 0.02 * z_spread
        if label_norm == "banana":
            score += 0.02 * (1.0 - min(center_penalty, 1.0))

        if best is None or score > best["score"]:
            best = {
                "score": float(score),
                "index": index,
                "mask": mask,
                "count": count,
                "width_m": width,
                "z_spread_m": z_spread,
                "along_min_m": float(edges[index]),
                "along_max_m": float(edges[index + 1]),
            }

    if best is None:
        central = (along >= np.percentile(along, 35.0)) & (along <= np.percentile(along, 65.0))
        if int(np.count_nonzero(central)) >= min_bin_points:
            target = np.median(points[central, :3], axis=0)
            method = "central_band_median"
        else:
            target = raw_centroid
            method = "raw_centroid_fallback"
        return {
            "method": method,
            "target_xyz_m": target.astype(float).tolist(),
            "raw_centroid_xyz_m": raw_centroid.astype(float).tolist(),
            "xy_major_axis": major.astype(float).tolist(),
            "xy_minor_axis": minor.astype(float).tolist(),
            "pca_bbox": pca_bbox,
            "reason": "no_dense_axis_bin",
        }

    selected_points = points[best["mask"], :3]
    target = np.median(selected_points, axis=0)
    return {
        "method": "local_thick_axis_band",
        "target_xyz_m": target.astype(float).tolist(),
        "raw_centroid_xyz_m": raw_centroid.astype(float).tolist(),
        "xy_major_axis": major.astype(float).tolist(),
        "xy_minor_axis": minor.astype(float).tolist(),
        "pca_bbox": pca_bbox,
        "selected_band": {
            "along_min_m": best["along_min_m"],
            "along_max_m": best["along_max_m"],
            "point_count": best["count"],
            "width_m": best["width_m"],
            "z_spread_m": best["z_spread_m"],
            "score": best["score"],
        },
    }


def adjust_target_depth_from_rgbd(points, selection, clearance_m=0.008, local_radius_m=0.035, percentile=10.0):
    """Use local RGB-D point depth to place the grasp target just above the object surface."""
    points = finite_xyz_points(points)
    if len(points) == 0 or "target_xyz_m" not in selection:
        return selection

    target = np.asarray(selection["target_xyz_m"], dtype=np.float64)
    xy_distance = np.linalg.norm(points[:, :2] - target[:2], axis=1)
    local = points[xy_distance <= float(local_radius_m)]

    if len(local) < 12:
        nearest_count = min(max(12, len(points) // 20), len(points))
        nearest_indices = np.argsort(xy_distance)[:nearest_count]
        local = points[nearest_indices]
        mode = "nearest_points"
    else:
        mode = "local_radius"

    local_depths = local[:, 2]
    local_depths = local_depths[np.isfinite(local_depths) & (local_depths > 0.0)]
    if len(local_depths) == 0:
        return selection

    surface_depth = float(np.percentile(local_depths, float(percentile)))
    previous_depth = float(target[2])
    target[2] = surface_depth - float(clearance_m)
    selection["target_xyz_m"] = target.astype(float).tolist()
    selection["rgbd_depth_adjustment"] = {
        "method": mode,
        "local_points": int(len(local_depths)),
        "local_radius_m": float(local_radius_m),
        "surface_depth_percentile": float(percentile),
        "measured_surface_depth_m": surface_depth,
        "clearance_above_surface_m": float(clearance_m),
        "previous_target_depth_m": previous_depth,
        "adjusted_target_depth_m": float(target[2]),
        "delta_depth_m": float(target[2] - previous_depth),
    }
    return selection


def load_grasp_points_from_detection(detection):
    files = detection.get("files") or {}
    points_path = files.get("foreground_points_npy")
    if not points_path:
        return None, None
    path = Path(points_path).expanduser()
    if not path.is_file():
        return None, str(path)
    return finite_xyz_points(np.load(path)), str(path)


def save_grasp_selection_visualization(points, selection, output_dir, label):
    if cv2 is None or points is None or len(points) == 0:
        return None

    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "grasp_candidate_topdown.png"

    points = finite_xyz_points(points)
    xy = points[:, :2]
    valid = np.isfinite(xy).all(axis=1)
    xy = xy[valid]
    if len(xy) == 0:
        return None

    width = 900
    height = 900
    margin = 80
    canvas = np.full((height, width, 3), 248, dtype=np.uint8)
    mins = np.percentile(xy, 1.0, axis=0)
    maxs = np.percentile(xy, 99.0, axis=0)
    center = (mins + maxs) * 0.5
    span = np.maximum(maxs - mins, 1e-4)
    span[:] = np.max(span)
    mins = center - span * 0.5
    maxs = center + span * 0.5

    def to_px(point_xy):
        px = margin + (point_xy[0] - mins[0]) * (width - 2 * margin) / span[0]
        py = height - margin - (point_xy[1] - mins[1]) * (height - 2 * margin) / span[1]
        return int(round(px)), int(round(py))

    z_values = points[valid, 2]
    colors = cv2.applyColorMap(normalize_to_uint8(z_values), cv2.COLORMAP_TURBO)
    order = np.argsort(z_values)
    for idx in order:
        x, y = to_px(xy[idx])
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(canvas, (x, y), 2, tuple(int(v) for v in colors[idx, 0]), -1, cv2.LINE_AA)

    raw_centroid = np.asarray(selection.get("raw_centroid_xyz_m", selection["target_xyz_m"]), dtype=np.float64)
    target = np.asarray(selection["target_xyz_m"], dtype=np.float64)
    grasp_xyz, _ = store_grasp_pose_reference(selection)
    raw_px = to_px(raw_centroid[:2])
    target_px = to_px(target[:2])
    grasp_px = to_px(grasp_xyz[:2])

    major = np.asarray(selection.get("xy_major_axis", [1.0, 0.0]), dtype=np.float64)
    minor = np.asarray(selection.get("xy_minor_axis", [0.0, 1.0]), dtype=np.float64)
    if np.linalg.norm(major) > 0:
        major = major / np.linalg.norm(major)
    if np.linalg.norm(minor) > 0:
        minor = minor / np.linalg.norm(minor)

    axis_len = float(np.max(span)) * 0.42
    axis_origin_xy = grasp_xyz[:2]
    cv2.line(canvas, to_px(axis_origin_xy - major * axis_len), to_px(axis_origin_xy + major * axis_len), (0, 180, 255), 3, cv2.LINE_AA)
    cv2.line(canvas, to_px(axis_origin_xy - minor * axis_len * 0.35), to_px(axis_origin_xy + minor * axis_len * 0.35), (255, 180, 0), 3, cv2.LINE_AA)

    band = selection.get("selected_band") or {}
    if "along_min_m" in band and "along_max_m" in band:
        xy_center, _, _ = pca_axes_xy(points)
        for along_value in (band["along_min_m"], band["along_max_m"]):
            a = xy_center + major * along_value - minor * axis_len * 0.45
            b = xy_center + major * along_value + minor * axis_len * 0.45
            cv2.line(canvas, to_px(a), to_px(b), (70, 70, 70), 2, cv2.LINE_AA)

    cv2.drawMarker(canvas, raw_px, (30, 30, 30), cv2.MARKER_CROSS, 30, 3, cv2.LINE_AA)
    cv2.circle(canvas, target_px, 13, (40, 220, 40), -1, cv2.LINE_AA)
    cv2.circle(canvas, target_px, 13, (10, 80, 10), 3, cv2.LINE_AA)
    cv2.circle(canvas, grasp_px, 8, (40, 40, 240), -1, cv2.LINE_AA)
    cv2.circle(canvas, grasp_px, 8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "raw centroid", (raw_px[0] + 14, raw_px[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(canvas, "selected grasp", (target_px[0] + 16, target_px[1] + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (10, 80, 10), 2, cv2.LINE_AA)
    cv2.putText(canvas, "grasp pose", (grasp_px[0] + 14, grasp_px[1] + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 240), 2, cv2.LINE_AA)

    cv2.rectangle(canvas, (0, 0), (width, 46), (25, 25, 25), -1)
    title = f"{label} top-down grasp candidate: {selection.get('method', 'unknown')}"
    cv2.putText(canvas, title[:82], (16, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "XY projection, color = depth Z", (18, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (35, 35, 35), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), canvas)
    return str(out_path)


def start_child_processes(args):
    processes = []
    if args.launch_world:
        processes.append(
            subprocess.Popen(
                ["ros2", "launch", "manip_challenge", "ur5_setup_random_picking.launch.py"],
                preexec_fn=os.setsid,
            )
        )
        time.sleep(args.launch_wait)

    if args.external_perception and args.start_perception:
        processes.append(
            subprocess.Popen(
                [sys.executable, str(TWO_VIEW_SERVER)],
                preexec_fn=os.setsid,
            )
        )
        time.sleep(args.perception_wait)

    return processes


def stop_child_processes(processes):
    for proc in processes:
        if proc.poll() is not None:
            continue
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        except ProcessLookupError:
            continue
    for proc in processes:
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.terminate()


def load_top_view_perception_module():
    spec = importlib.util.spec_from_file_location("custom_top_view_seg_server", TWO_VIEW_SERVER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load top-view perception module: {TWO_VIEW_SERVER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ros_args_with_embedded_perception_defaults(argv, args, top_view_module):
    ros_argv = top_view_module.argv_with_default_params(list(argv or sys.argv))
    default_service = top_view_module.DEFAULTS.get("service_name")
    if args.perception_service != default_service:
        ros_argv.extend(["--ros-args", "-p", f"service_name:={args.perception_service}"])
    return ros_argv


class TfNode(Node):
    def __init__(self):
        super().__init__("two_view_grasp_tf_node")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)


class TwoViewGraspServer(Node):
    def __init__(self, args, tf_buffer, arm):
        super().__init__("two_view_grasp_command_server")
        self.args = args
        self.tf_buffer = tf_buffer
        self.arm = arm
        self.command_lock = threading.Lock()
        self.gripper_force = args.gripper_force
        self.gripper_close_pos = args.gripper_close_pos
        self.gripper_settle_time = args.gripper_settle_time
        self.gripper_result_timeout_margin = args.gripper_result_timeout_margin
        self.strict_gripper_result = args.strict_gripper_result
        self.grasp_descend_duration = args.descend_duration
        self.callback_group = ReentrantCallbackGroup()
        self.perception_client = self.create_client(
            StringString,
            args.perception_service,
            callback_group=self.callback_group,
        )
        self.task_queue = []
        self.task_queue_lock = threading.Lock()
        self.command_subscription = self.create_subscription(
            String,
            args.command_topic,
            self._command_callback,
            10,
            callback_group=self.callback_group,
        )
        self.result_publisher = self.create_publisher(String, args.result_topic, 10)
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

        self.get_logger().info(f"Waiting for perception service '{args.perception_service}'...")
        while rclpy.ok() and not self.perception_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f"Still waiting for '{args.perception_service}'...")

        self.get_logger().info(
            f"Ready. Listening on topic '{args.command_topic}', e.g. 'banana left'. "
            "Each command detects the object and executes pick-and-place in this single server."
        )

    def _publish_result(self, result):
        msg = String()
        msg.data = json.dumps(result, sort_keys=True)
        self.result_publisher.publish(msg)

    def _command_callback(self, msg):
        try:
            parsed_list = parse_commands(msg.data)
        except CommandParseError as exc:
            self.get_logger().error(f"Parse error: {exc}")
            return
        with self.task_queue_lock:
            self.task_queue.extend(parsed_list)
        self.get_logger().info(f"Queued {len(parsed_list)} command(s) from: '{msg.data}'")

    def _worker_loop(self):
        # Pipeline state (mirrors main.py pattern)
        pending_future = None   # async detection future for the next queued object
        pending_obj = None      # object name the future is for
        next_pick_done = False  # True when the next item's pick was already done inline
        next_grasp_pose = None  # grasp pose used in that inline pick
        next_perception_info = None

        while rclpy.ok():
            item = None
            with self.task_queue_lock:
                if self.task_queue:
                    item = self.task_queue.pop(0)
            if item is None:
                time.sleep(0.01)
                continue

            action, obj_name, destination = item
            result = None
            try:
                with self.command_lock:
                    # --- home / idle ---
                    if action in {"home", "idle", "reset"}:
                        self.arm.move_joint(HOME_JOINTS)
                        result = {"ok": True, "action": action, "message": "Robot moved to home joints."}
                        pending_future = None
                        pending_obj = None
                        next_pick_done = False
                        next_grasp_pose = None
                        next_perception_info = None
                        self._publish_result(result)
                        continue

                    # --- validation ---
                    if not destination and not self.args.approach_only:
                        raise CommandParseError(
                            "Destination is required. Try: 'banana left' or 'move banana to bookshelf'."
                        )
                    if destination and destination not in PLACE_CONFIGS:
                        raise CommandParseError(
                            f"Unknown destination '{destination}'. Choose one of: {sorted(PLACE_CONFIGS)}"
                        )

                    # --- get grasp pose (use pre-fetched result if available) ---
                    grasp_selection = None
                    perception_info = None
                    if next_pick_done and pending_obj == obj_name:
                        # Pick phase already executed inline during previous place
                        self.get_logger().info(f"[Pipeline] Pick already done for '{obj_name}', skipping to place.")
                        grasp_pose = next_grasp_pose
                        perception_info = next_perception_info
                        if isinstance(perception_info, dict):
                            grasp_selection = perception_info.get("grasp_selection")
                        pick_already_done = True
                        next_pick_done = False
                        next_grasp_pose = None
                        next_perception_info = None
                        pending_obj = None
                    else:
                        if not self.args.dry_run and not self.args.no_command_home:
                            self.get_logger().info("Returning to home before new command.")
                            self.arm.move_joint(HOME_JOINTS)

                        # Use pre-fetched future if it's for the right object
                        if pending_future is not None and pending_obj == obj_name:
                            self.get_logger().info(f"[Pipeline] Using pre-fetched detection for '{obj_name}'.")
                            detection = self._collect_detection(pending_future, obj_name)
                            pending_future = None
                            pending_obj = None
                        else:
                            detection = self.detect_object(obj_name)

                        grasp_pose, grasp_selection = self._compute_grasp_pose(detection, obj_name)
                        perception_info = {
                            "detection_result": detection,
                            "grasp_selection": grasp_selection,
                        }
                        pick_already_done = False

                    # --- build pipeline callbacks (mirrors main.py make_callbacks) ---
                    def on_before_idle():
                        nonlocal pending_future, pending_obj
                        with self.task_queue_lock:
                            if not self.task_queue:
                                return
                            _, next_obj, _ = self.task_queue[0]
                        self.get_logger().info(f"[Pipeline] Starting detection for next: '{next_obj}'")
                        pending_future = self._detect_async(next_obj)
                        pending_obj = next_obj

                    def get_next_pick_data():
                        nonlocal pending_future, pending_obj, next_pick_done, next_grasp_pose, next_perception_info
                        if pending_future is None:
                            return None
                        with self.task_queue_lock:
                            if not self.task_queue:
                                pending_future = None
                                pending_obj = None
                                return None
                            _, next_obj, _ = self.task_queue[0]

                        if next_obj != pending_obj:
                            self.get_logger().warn(
                                f"[Pipeline] Queue head changed: expected '{pending_obj}', got '{next_obj}'"
                            )
                            pending_future = None
                            pending_obj = None
                            return None

                        try:
                            det = self._collect_detection(pending_future, next_obj)
                            pending_future = None
                        except Exception as exc:
                            self.get_logger().warn(f"[Pipeline] Detection failed for '{next_obj}': {exc}")
                            pending_future = None
                            pending_obj = None
                            return None

                        try:
                            ng_pose, ng_selection = self._compute_grasp_pose(det, next_obj)
                        except Exception as exc:
                            self.get_logger().warn(f"[Pipeline] Grasp compute failed for '{next_obj}': {exc}")
                            pending_obj = None
                            return None

                        approach = copy.deepcopy(ng_pose)
                        approach.position.z += self.args.approach_height
                        pan = math.atan2(ng_pose.position.y, ng_pose.position.x)

                        next_pick_done = True
                        next_grasp_pose = ng_pose
                        next_perception_info = {
                            "detection_result": det,
                            "grasp_selection": ng_selection,
                        }
                        self.get_logger().info(f"[Pipeline] Next pick ready for '{next_obj}', pan={pan:.3f}")
                        return {
                            "obj_name": next_obj,
                            "pick_joint": home_joints_for_pan(pan),
                            "approach_pose": approach,
                            "grasp_pose": ng_pose,
                            "perception_info": next_perception_info,
                        }

                    # --- execute ---
                    if self.args.dry_run:
                        self.get_logger().info("Dry run: skipping motion.")
                    elif self.args.approach_only:
                        approach_pose = copy.deepcopy(grasp_pose)
                        approach_pose.position.z += self.args.approach_height
                        move_gripper.gripper_open(self)
                        self.move_to_grasp_approach(grasp_pose, approach_pose)
                    else:
                        execute_pick_place_sequence(
                            self, self.arm, grasp_pose, destination, obj_name,
                            on_before_idle=on_before_idle,
                            get_next_pick_data=get_next_pick_data,
                            pick_already_done=pick_already_done,
                            perception_info=perception_info,
                        )

                    result = {
                        "ok": True,
                        "action": "grasp_ready" if self.args.approach_only else "pick_place",
                        "object": obj_name,
                        "destination": destination,
                        "grasp_selection": grasp_selection,
                        "message": "Pick-and-place sequence completed.",
                    }

            except Exception as exc:
                self.get_logger().error(f"Command failed: {exc}\n{traceback.format_exc()}")
                result = {"ok": False, "error": str(exc), "object": obj_name, "destination": destination}
                # Reset pipeline state on error
                pending_future = None
                pending_obj = None
                next_pick_done = False
                next_grasp_pose = None
                next_perception_info = None

            if result is not None:
                self._publish_result(result)

    def execute_command(self, action, obj_name, destination):
        if action in {"home", "idle", "reset"}:
            self.arm.move_joint(HOME_JOINTS)
            return {"ok": True, "action": action, "message": "Robot moved to home joints."}

        self.get_logger().info(f"Command parsed: object={obj_name}, destination={destination}")
        if not destination and not self.args.approach_only:
            raise CommandParseError(
                "Destination is required for pick-and-place. Try: 'banana left' or 'move banana to bookshelf'."
            )
        if destination and destination not in PLACE_CONFIGS:
            raise CommandParseError(
                f"Unknown destination '{destination}'. Choose one of: {sorted(PLACE_CONFIGS)}"
            )
        if not self.args.dry_run and not self.args.no_command_home:
            self.get_logger().info("Returning to home before starting the new command.")
            self.arm.move_joint(HOME_JOINTS)

        detection = self.detect_object(obj_name)
        grasp_pose, grasp_selection = self._compute_grasp_pose(detection, obj_name)
        perception_info = {
            "detection_result": detection,
            "grasp_selection": grasp_selection,
        }
        source_frame = detection.get("frame_id") or self.args.camera_frame

        detected_pose = pose_from_xyz(grasp_selection["target_xyz_m"])
        approach_pose = copy.deepcopy(grasp_pose)
        approach_pose.position.z += self.args.approach_height

        action_name = "grasp_ready" if self.args.approach_only else "pick_place"
        message = "Moved to grasp-ready pose." if self.args.approach_only else "Pick-and-place sequence completed."

        if self.args.dry_run:
            self.get_logger().info("Dry run: skipping arm and gripper motion.")
        elif self.args.approach_only:
            move_gripper.gripper_open(self)
            self.move_to_grasp_approach(grasp_pose, approach_pose)
        else:
            execute_pick_place_sequence(
                self,
                self.arm,
                grasp_pose,
                destination,
                obj_name,
                perception_info=perception_info,
            )

        return {
            "ok": True,
            "action": action_name,
            "object": obj_name,
            "destination": destination,
            "perception_target": detection.get("target"),
            "detected_label": detection.get("detected_label"),
            "source_frame": source_frame,
            "grasp_selection": grasp_selection,
            "detected_pose_in_source": pose_to_dict(detected_pose),
            "grasp_pose_in_base": pose_to_dict(grasp_pose),
            "approach_pose_in_base": pose_to_dict(approach_pose),
            "dry_run": bool(self.args.dry_run),
            "closed_gripper": bool(not self.args.dry_run and (not self.args.approach_only or self.args.close_gripper)),
            "message": message,
        }

    def _detect_async(self, obj_name):
        """Start detection asynchronously and return the future."""
        req = StringString.Request()
        req.data = obj_name
        return self.perception_client.call_async(req)

    def _collect_detection(self, future, obj_name):
        """Wait for a detection future and return the result dict."""
        deadline = time.monotonic() + self.args.perception_timeout
        while rclpy.ok() and not future.done():
            if time.monotonic() > deadline:
                raise TimeoutError(f"Timed out waiting for perception result for '{obj_name}'.")
            time.sleep(0.02)
        result = future.result()
        if result is None:
            raise RuntimeError(f"Perception service returned no result for '{obj_name}'.")
        return json.loads(result.data)

    def detect_object(self, obj_name):
        return self._collect_detection(self._detect_async(obj_name), obj_name)

    def _compute_grasp_pose(self, detection, obj_name):
        """Detection dict → (grasp_pose, grasp_selection). Raises on failure."""
        if not detection.get("ok"):
            raise RuntimeError(detection.get("error", "Perception failed."))
        source_frame = detection.get("frame_id") or self.args.camera_frame
        grasp_selection = self.select_grasp_target(detection, obj_name)

        # 기존: 잡기 좋은 grasping point 를 계산하여 grasp_pose 로 설정
        # detected_pose = pose_from_xyz(grasp_selection["target_xyz_m"])

        # 변경: raw centroid XY + adjusted target Z
        grasp_xyz, _ = store_grasp_pose_reference(grasp_selection)
        detected_pose = pose_from_xyz(grasp_xyz)

        base_pose = self.transform_pose(detected_pose, source_frame, "base_link")
        pca_major_axis_base_xy = self.transform_selection_axis_to_base_xy(
            grasp_selection,
            source_frame,
            base_pose,
            grasp_xyz,
        )
        gripper_axis_base_xy = None
        if pca_major_axis_base_xy is not None:
            gripper_axis_base_xy = np.asarray(
                [-pca_major_axis_base_xy[1], pca_major_axis_base_xy[0]],
                dtype=np.float64,
            )
            grasp_selection["pca_major_axis_base_xy"] = pca_major_axis_base_xy.astype(float).tolist()
            grasp_selection["pca_major_axis_base_yaw_rad"] = float(
                math.atan2(pca_major_axis_base_xy[1], pca_major_axis_base_xy[0])
            )
            grasp_selection["gripper_axis_base_xy"] = gripper_axis_base_xy.astype(float).tolist()
            grasp_selection["gripper_axis_base_yaw_rad"] = float(
                math.atan2(gripper_axis_base_xy[1], gripper_axis_base_xy[0])
            )
            grasp_selection["gripper_axis_rule"] = "perpendicular_to_pca_major_axis"

        grasp_pose = self.make_grasp_pose(base_pose, gripper_axis_base_xy)
        return grasp_pose, grasp_selection

    def select_grasp_target(self, detection, obj_name):
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
        selection = adjust_target_depth_from_rgbd(
            points,
            selection,
            clearance_m=self.args.grasp_surface_clearance,
            local_radius_m=self.args.grasp_depth_local_radius,
            percentile=self.args.grasp_depth_percentile,
        )
        store_grasp_pose_reference(selection)

        output_dir = detection.get("save_dir")
        if not output_dir and points_path:
            output_dir = str(Path(points_path).expanduser().parent)
        try:
            visualization = save_grasp_selection_visualization(points, selection, output_dir, label) if output_dir else None
        except Exception as exc:
            visualization = None
            selection["visualization_error"] = str(exc)
        if visualization:
            selection["visualization"] = visualization

        target = selection["target_xyz_m"]
        raw = selection.get("raw_centroid_xyz_m", detection["location_xyz_m"])
        delta = np.asarray(target, dtype=float) - np.asarray(raw, dtype=float)
        selection["delta_from_raw_centroid_m"] = delta.astype(float).tolist()
        self.get_logger().info(
            "Selected grasp target "
            f"method={selection.get('method')} "
            f"xyz=({target[0]:.4f}, {target[1]:.4f}, {target[2]:.4f}) "
            f"delta_from_centroid=({delta[0]:+.4f}, {delta[1]:+.4f}, {delta[2]:+.4f})"
        )
        return selection

    def transform_pose(self, pose, source_frame, target_frame):
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
            transformed = self.tf_buffer.transform(
                stamped,
                target_frame,
                timeout=rclpy.duration.Duration(seconds=1.0),
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as exc:
            raise RuntimeError(f"TF transform failed: {exc}") from exc
        return transformed.pose

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
        axis_base = np.asarray(
            [
                endpoint_base.position.x - base_pose.position.x,
                endpoint_base.position.y - base_pose.position.y,
                0.0,
            ],
            dtype=np.float64,
        )
        axis_base = normalize_vector(axis_base)
        if axis_base is None:
            return None
        return axis_base[:2]

    def make_grasp_pose(self, base_pose, gripper_axis_base_xy=None):
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
                "Using PCA-aligned grasp orientation: "
                f"gripper_axis_yaw={math.degrees(axis_yaw):.1f} deg"
            )
        # grasp_pose.position.y += self.args.grasp_y_offset
        # grasp_pose.position.z += self.args.final_z_offset

        # 왜인지는 모르겠는데 필요함 -> 더 정확함
        grasp_pose.position.y += -0.015  # +: 로봇 기준 왼쪽, -: 오른쪽

        return grasp_pose

    def move_to_grasp_approach(self, grasp_pose, approach_pose):
        current_pan = float(self.arm.js_joint_position[0])
        target_pan = math.atan2(grasp_pose.position.y, grasp_pose.position.x)
        pick_joint = home_joints_for_pan(target_pan)
        rot_time = calc_rot_time(current_pan, target_pan)
        self.get_logger().info(
            "Moving to grasp approach: "
            f"x={grasp_pose.position.x:.3f}, y={grasp_pose.position.y:.3f}, z={grasp_pose.position.z:.3f}"
        )
        self.arm.execute_trajectory(
            [pick_joint, approach_pose, grasp_pose],
            durations=[rot_time, self.args.approach_duration, self.args.descend_duration],
        )
        if self.args.close_gripper:
            self.get_logger().info("Closing gripper at grasp-ready pose.")
            move_gripper.gripper_close(
                self,
                force=self.args.gripper_force,
                gripper_close_pos=self.args.gripper_close_pos,
            )
        if self.args.retreat_after_command:
            self.get_logger().info("Retreating to approach pose so the next command can run cleanly.")
            self.arm.execute_trajectory(
                [approach_pose],
                durations=[self.args.retreat_duration],
            )


def random_spawn_pose(index):
    x = random.uniform(0.45, 0.65)
    y = random.uniform(-0.30, 0.30)
    z = 0.63
    yaw = random.uniform(-math.pi, math.pi)
    return misc.list2Pose([x, y, z + 0.02 * index, 0.0, 0.0, yaw])


def spawn_objects(node, object_names):
    if not object_names:
        return

    from ament_index_python.packages import get_package_share_directory
    from gazebo_msgs.srv import SpawnEntity

    client = node.create_client(SpawnEntity, "/spawn_entity")
    node.get_logger().info("Waiting for Gazebo '/spawn_entity' service...")
    client.wait_for_service()

    model_root = Path(get_package_share_directory("manip_challenge")) / "data" / "models"
    for index, object_name in enumerate(object_names):
        model_name = normalize_object_name(object_name)
        sdf_path = model_root / model_name / "model.sdf"
        if not sdf_path.is_file():
            raise FileNotFoundError(f"Model SDF not found: {sdf_path}")

        req = SpawnEntity.Request()
        req.name = f"{model_name}_{int(time.time() * 1000)}_{index}"
        req.xml = sdf_path.read_text(encoding="utf-8")
        req.robot_namespace = model_name
        req.initial_pose = random_spawn_pose(index)

        future = client.call_async(req)
        rclpy.spin_until_future_complete(node, future)
        if future.result() is None:
            raise RuntimeError(f"Failed to spawn {model_name}: {future.exception()}")
        node.get_logger().info(f"Spawned {model_name}: {future.result().status_message}")
        time.sleep(0.5)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Two-view pick-and-place command server.")
    parser.add_argument("--command-topic", default="/task_commands")
    parser.add_argument("--result-topic", default="/task_results")
    parser.add_argument("--perception-service", default="detect_object_top_rgbd_seg_crop")
    parser.add_argument("--camera-frame", default="camera_color_optical_frame")
    parser.add_argument("--perception-timeout", type=float, default=15.0)
    parser.add_argument("--tf-timeout", type=float, default=5.0)
    parser.add_argument("--approach-height", type=float, default=0.15)
    parser.add_argument(
        "--final-z-offset",
        type=float,
        default=0.0,
        help="Extra base-link Z offset after RGB-D depth targeting. Keep near 0 unless calibration needs a bias.",
    )
    parser.add_argument(
        "--grasp-surface-clearance",
        type=float,
        default=-0.012,
        help="Meters to stay above the locally measured RGB-D object surface. Negative values descend into the object.",
    )
    parser.add_argument(
        "--grasp-depth-local-radius",
        type=float,
        default=0.035,
        help="Meters around the selected grasp XY used to estimate local object depth.",
    )
    parser.add_argument(
        "--grasp-depth-percentile",
        type=float,
        default=10.0,
        help="Local depth percentile used as the top surface estimate; lower is closer to the top camera.",
    )
    parser.add_argument("--grasp-y-offset", type=float, default=-0.01)
    parser.add_argument("--approach-duration", type=float, default=1.5)
    parser.add_argument("--descend-duration", type=float, default=1.0)
    parser.add_argument("--retreat-duration", type=float, default=1.0)
    parser.add_argument("--approach-only", action="store_true", help="Only move to the grasp-ready pose instead of placing the object.")
    parser.add_argument("--retreat-after-command", action="store_true", help="Retreat to the approach pose after reaching grasp-ready pose.")
    parser.add_argument("--no-command-home", action="store_true", help="Do not return home before each client command.")
    parser.add_argument("--close-gripper", action="store_true", help="In --approach-only mode, close the gripper after reaching grasp-ready pose.")
    parser.add_argument("--gripper-force", type=float, default=0.5)
    parser.add_argument("--gripper-close-pos", type=float, default=0.5)
    parser.add_argument("--gripper-settle-time", type=float, default=0.4)
    parser.add_argument(
        "--gripper-result-timeout-margin",
        type=float,
        default=8.0,
        help="Extra seconds to wait for a gripper action result after its trajectory duration.",
    )
    parser.add_argument(
        "--strict-gripper-result",
        action="store_true",
        help="Treat gripper action result timeouts as fatal errors. By default they are warnings so contact grasps can continue.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Detect and transform, but do not move the arm.")
    parser.add_argument("--disable-tamp", action="store_true", help="Disable natural-language TAMP planning and use the original one-object command path.")
    parser.add_argument(
        "--tamp-max-steps",
        type=int,
        default=int(os.environ.get("TAMP_MAX_STEPS", "8")),
        help="Maximum pick/place actions the TAMP planner may execute for one natural-language command.",
    )
    parser.add_argument("--no-home", action="store_true", help="Do not move the arm to the home joint pose at startup.")
    parser.add_argument("--launch-world", action="store_true", help="Start ur5_setup_random_picking.launch.py as a child process.")
    parser.add_argument("--launch-wait", type=float, default=12.0)
    parser.add_argument("--external-perception", action="store_true", help="Use an already running perception service instead of embedding top-view perception.")
    parser.add_argument("--start-perception", action="store_true", help="With --external-perception, start perception/two-view/top_view_seg_server.py as a child process.")
    parser.add_argument("--perception-wait", type=float, default=3.0)
    parser.add_argument(
        "--spawn-object",
        action="append",
        default=[],
        help="Spawn an object model at startup. Can be repeated, e.g. --spawn-object banana --spawn-object coke_can.",
    )
    return parser.parse_args(rclpy.utilities.remove_ros_args(args=argv or sys.argv)[1:])


def main(argv=None):
    args = parse_args(argv)
    top_view_module = None
    if not args.external_perception:
        top_view_module = load_top_view_perception_module()
    child_processes = start_child_processes(args)

    ros_argv = (
        ros_args_with_embedded_perception_defaults(argv or sys.argv, args, top_view_module)
        if top_view_module is not None
        else argv
    )
    rclpy.init(args=ros_argv)
    perception_node = None
    tf_node = TfNode()
    if top_view_module is not None:
        perception_node = top_view_module.RgbdSegCropServiceNode()
        perception_node.get_logger().info("Embedded top-view perception is running inside two_view_grasp_server.")
    arm = ArmClient()
    server = TwoViewGraspServer(args, tf_node.tf_buffer, arm)

    try:
        if not args.no_home:
            arm.move_joint(HOME_JOINTS)
        spawn_objects(server, args.spawn_object)

        executor = MultiThreadedExecutor(num_threads=4)
        if perception_node is not None:
            executor.add_node(perception_node)
        executor.add_node(tf_node)
        executor.add_node(server)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if "executor" in locals():
            executor.shutdown()
        server.destroy_node()
        tf_node.destroy_node()
        if perception_node is not None:
            perception_node.destroy_node()
        arm.destroy_node()
        rclpy.shutdown()
        stop_child_processes(child_processes)


if __name__ == "__main__":
    main()
