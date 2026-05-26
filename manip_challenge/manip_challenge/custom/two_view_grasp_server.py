#!/usr/bin/env python3
"""Two-view perception command server for grasp-approach motion.

Run this server continuously, then send commands with two_view_grasp_client.py:

    python3 two_view_grasp_client.py banana left

This first version detects the requested object with the top-view two-view
segmentation service, chooses the ROI point-cloud centroid as the grasp target,
transforms it into base_link, and moves the gripper through an approach pose
down to a grasp-ready pose. It does not place the object yet.
"""

import argparse
import copy
import json
import math
import os
import random
import re
import signal
import subprocess
import sys
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
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from riro_srvs.srv import StringString
from tf2_ros import Buffer, ConnectivityException, ExtrapolationException, LookupException, TransformListener


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


HOME_JOINTS = [0.0, -np.pi / 2.0, 1.0, -np.pi / 3.0, -np.pi / 2.0, 0.0]

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


def normalize_object_name(name):
    text = str(name or "").strip().lower()
    text = re.sub(r"^(a|an|the)\s+", "", text)
    return text.replace("-", "_").replace(" ", "_")


def normalize_destination(name):
    text = str(name or "").strip().lower().replace("-", " ").replace("_", " ")
    text = re.sub(r"^(the|a|an)\s+", "", text)
    return DESTINATION_ALIASES.get(text, DESTINATION_ALIASES.get(text.replace(" ", "_"), text))


def parse_command(text):
    """Return (action, object_name, destination) from a short client command."""
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


def pca_axes_xy(points):
    xy = points[:, :2].astype(np.float64)
    center = np.median(xy, axis=0)
    centered = xy - center
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    major = vt[0]
    minor = vt[1]
    if major[0] < 0:
        major = -major
        minor = -minor
    return center, major, minor


def choose_grasp_target_from_points(points, label="", min_bin_points=20):
    """Choose a grasp point from visible object points instead of raw centroid.

    For elongated/curved objects, the global centroid can fall in an awkward
    location. This chooses a locally thick, central band along the top-down
    major axis and returns the median real surface point in that band.
    """
    points = finite_xyz_points(points)
    if len(points) < max(8, min_bin_points):
        centroid = np.median(points, axis=0) if len(points) else np.asarray([0.0, 0.0, 0.0])
        return {
            "method": "median_fallback",
            "target_xyz_m": centroid.astype(float).tolist(),
            "raw_centroid_xyz_m": centroid.astype(float).tolist(),
            "reason": f"too_few_points:{len(points)}",
        }

    xy_center, major, minor = pca_axes_xy(points)
    centered = points[:, :2] - xy_center
    along = centered @ major
    across = centered @ minor
    raw_centroid = np.median(points[:, :3], axis=0)

    low, high = np.percentile(along, [12.0, 88.0])
    if high <= low:
        target = raw_centroid
        return {
            "method": "median_fallback",
            "target_xyz_m": target.astype(float).tolist(),
            "raw_centroid_xyz_m": raw_centroid.astype(float).tolist(),
            "reason": "degenerate_major_axis",
            "xy_major_axis": major.astype(float).tolist(),
            "xy_minor_axis": minor.astype(float).tolist(),
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
            method = "median_fallback"
        return {
            "method": method,
            "target_xyz_m": target.astype(float).tolist(),
            "raw_centroid_xyz_m": raw_centroid.astype(float).tolist(),
            "xy_major_axis": major.astype(float).tolist(),
            "xy_minor_axis": minor.astype(float).tolist(),
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
        "selected_band": {
            "along_min_m": best["along_min_m"],
            "along_max_m": best["along_max_m"],
            "point_count": best["count"],
            "width_m": best["width_m"],
            "z_spread_m": best["z_spread_m"],
            "score": best["score"],
        },
    }


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
    raw_px = to_px(raw_centroid[:2])
    target_px = to_px(target[:2])

    major = np.asarray(selection.get("xy_major_axis", [1.0, 0.0]), dtype=np.float64)
    minor = np.asarray(selection.get("xy_minor_axis", [0.0, 1.0]), dtype=np.float64)
    if np.linalg.norm(major) > 0:
        major = major / np.linalg.norm(major)
    if np.linalg.norm(minor) > 0:
        minor = minor / np.linalg.norm(minor)

    axis_len = float(np.max(span)) * 0.42
    cv2.line(canvas, to_px(target[:2] - major * axis_len), to_px(target[:2] + major * axis_len), (0, 180, 255), 3, cv2.LINE_AA)
    cv2.line(canvas, to_px(target[:2] - minor * axis_len * 0.35), to_px(target[:2] + minor * axis_len * 0.35), (255, 180, 0), 3, cv2.LINE_AA)

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
    cv2.putText(canvas, "raw centroid", (raw_px[0] + 14, raw_px[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(canvas, "selected grasp", (target_px[0] + 16, target_px[1] + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (10, 80, 10), 2, cv2.LINE_AA)

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

    if args.start_perception:
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
        self.callback_group = ReentrantCallbackGroup()
        self.perception_client = self.create_client(
            StringString,
            args.perception_service,
            callback_group=self.callback_group,
        )
        self.command_service = self.create_service(
            StringString,
            args.service_name,
            self.handle_command,
            callback_group=self.callback_group,
        )

        self.get_logger().info(f"Waiting for perception service '{args.perception_service}'...")
        while rclpy.ok() and not self.perception_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f"Still waiting for '{args.perception_service}'...")

        self.get_logger().info(
            f"Ready. Send commands to '{args.service_name}', e.g. 'banana left'. "
            "Each command returns home, detects the object, then moves to a grasp-ready pose."
        )

    def handle_command(self, request, response):
        try:
            result = self.execute_command(request.data)
        except Exception as exc:
            self.get_logger().error(f"Command failed: {exc}\n{traceback.format_exc()}")
            result = {"ok": False, "error": str(exc), "command": request.data}
        response.data = json.dumps(result, sort_keys=True)
        return response

    def execute_command(self, command_text):
        action, obj_name, destination = parse_command(command_text)
        if action in {"home", "idle", "reset"}:
            self.arm.move_joint(HOME_JOINTS)
            return {"ok": True, "action": action, "message": "Robot moved to home joints."}

        self.get_logger().info(f"Command parsed: object={obj_name}, destination={destination}")
        if not self.args.dry_run and not self.args.no_command_home:
            self.get_logger().info("Returning to home before starting the new command.")
            self.arm.move_joint(HOME_JOINTS)

        detection = self.detect_object(obj_name)
        if not detection.get("ok"):
            raise RuntimeError(detection.get("error", "Perception failed."))

        source_frame = detection.get("frame_id") or self.args.camera_frame
        grasp_selection = self.select_grasp_target(detection, obj_name)
        detected_pose = pose_from_xyz(grasp_selection["target_xyz_m"])
        base_pose = self.transform_pose(detected_pose, source_frame, "base_link")
        grasp_pose = self.make_grasp_pose(base_pose)
        approach_pose = copy.deepcopy(grasp_pose)
        approach_pose.position.z += self.args.approach_height

        move_gripper.gripper_open(self)

        if not self.args.dry_run:
            self.move_to_grasp_approach(grasp_pose, approach_pose)

        return {
            "ok": True,
            "action": "grasp_ready",
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
            "closed_gripper": bool(self.args.close_gripper),
            "message": "Moved to grasp-ready pose.",
        }

    def detect_object(self, obj_name):
        req = StringString.Request()
        req.data = obj_name
        future = self.perception_client.call_async(req)
        deadline = time.monotonic() + self.args.perception_timeout
        while rclpy.ok() and not future.done():
            if time.monotonic() > deadline:
                raise TimeoutError(f"Timed out waiting for perception result for '{obj_name}'.")
            time.sleep(0.02)

        result = future.result()
        if result is None:
            raise RuntimeError(f"Perception service returned no result for '{obj_name}'.")
        return json.loads(result.data)

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

    def make_grasp_pose(self, base_pose):
        home_pose = self.arm.fk_request(HOME_JOINTS)
        grasp_pose = copy.deepcopy(base_pose)
        grasp_pose.orientation = home_pose.orientation
        grasp_pose.position.y += self.args.grasp_y_offset
        grasp_pose.position.z += self.args.final_z_offset
        return grasp_pose

    def move_to_grasp_approach(self, grasp_pose, approach_pose):
        current_pan = float(self.arm.js_joint_position[0])
        target_pan = math.atan2(grasp_pose.position.y, grasp_pose.position.x)
        pick_joint = [target_pan, -np.pi / 2.0, 1.0, -np.pi / 3.0, -np.pi / 2.0, 0.0]
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
    parser = argparse.ArgumentParser(description="Two-view grasp-approach command server.")
    parser.add_argument("--service-name", default="two_view_grasp_command")
    parser.add_argument("--perception-service", default="detect_object_top_rgbd_seg_crop")
    parser.add_argument("--camera-frame", default="camera_color_optical_frame")
    parser.add_argument("--perception-timeout", type=float, default=15.0)
    parser.add_argument("--tf-timeout", type=float, default=5.0)
    parser.add_argument("--approach-height", type=float, default=0.15)
    parser.add_argument("--final-z-offset", type=float, default=0.025)
    parser.add_argument("--grasp-y-offset", type=float, default=-0.01)
    parser.add_argument("--approach-duration", type=float, default=1.5)
    parser.add_argument("--descend-duration", type=float, default=1.0)
    parser.add_argument("--retreat-duration", type=float, default=1.0)
    parser.add_argument("--retreat-after-command", action="store_true", help="Retreat to the approach pose after reaching grasp-ready pose.")
    parser.add_argument("--no-command-home", action="store_true", help="Do not return home before each client command.")
    parser.add_argument("--close-gripper", action="store_true", help="Close the gripper after reaching grasp-ready pose.")
    parser.add_argument("--gripper-force", type=float, default=0.5)
    parser.add_argument("--gripper-close-pos", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true", help="Detect and transform, but do not move the arm.")
    parser.add_argument("--no-home", action="store_true", help="Do not move the arm to the home joint pose at startup.")
    parser.add_argument("--launch-world", action="store_true", help="Start ur5_setup_random_picking.launch.py as a child process.")
    parser.add_argument("--launch-wait", type=float, default=12.0)
    parser.add_argument("--start-perception", action="store_true", help="Start perception/two-view/top_view_seg_server.py as a child process.")
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
    child_processes = start_child_processes(args)

    rclpy.init(args=argv)
    tf_node = TfNode()
    arm = ArmClient()
    server = TwoViewGraspServer(args, tf_node.tf_buffer, arm)

    try:
        if not args.no_home:
            arm.move_joint(HOME_JOINTS)
        spawn_objects(server, args.spawn_object)

        executor = MultiThreadedExecutor(num_threads=4)
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
        arm.destroy_node()
        rclpy.shutdown()
        stop_child_processes(child_processes)


if __name__ == "__main__":
    main()
