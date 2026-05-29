#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import math
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
if not hasattr(np, "mat"):
    np.mat = np.asmatrix

import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from manip_challenge import move_gripper
from manip_challenge.custom.grasping.gripper_control import JOINT_NAME as CUSTOM_GRIPPER_JOINT_NAME


HOME_JOINTS = [0.0, -np.pi / 2.0, 1.0, -1.0, -np.pi / 2.0, 0.0]
PACKAGE_DIR = Path(__file__).resolve().parents[1]
CUSTOM_DIR = PACKAGE_DIR / "custom"
TWO_VIEW_SERVER = CUSTOM_DIR / "perception" / "two-view" / "top_view_seg_server.py"


def patch_move_gripper() -> None:
    move_gripper.gripperGotoPos = executor_safe_gripper_goto
    move_gripper.gripper_open = executor_safe_gripper_open
    move_gripper.gripper_close = executor_safe_gripper_close


def executor_safe_gripper_goto(node, pos, force=1.0, timeout=3.0, **kwargs):
    timeout = float(timeout)
    margin = float(getattr(node, "gripper_result_timeout_margin", 8.0))
    strict = bool(getattr(node, "strict_gripper_result", False))
    if not hasattr(node, "_pddl_gripper_client"):
        node._pddl_gripper_client = ActionClient(
            node,
            FollowJointTrajectory,
            "/gripper_controller/follow_joint_trajectory",
            callback_group=getattr(node, "callback_group", None),
        )
    client = node._pddl_gripper_client
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
    deadline = time.monotonic() + timeout + margin
    while rclpy.ok() and not goal_future.done():
        if time.monotonic() > deadline:
            raise TimeoutError("Timed out sending gripper goal.")
        time.sleep(0.01)
    goal_handle = goal_future.result()
    if goal_handle is None or not goal_handle.accepted:
        raise RuntimeError("Gripper goal was rejected.")

    result_future = goal_handle.get_result_async()
    deadline = time.monotonic() + timeout + margin
    while rclpy.ok() and not result_future.done():
        if time.monotonic() > deadline:
            if strict:
                raise TimeoutError("Timed out waiting for gripper motion.")
            node.get_logger().warn(
                "Timed out waiting for gripper result; continuing because object contact can stop closing early."
            )
            try:
                goal_handle.cancel_goal_async()
            except Exception as exc:
                node.get_logger().warn(f"Could not cancel timed-out gripper goal: {exc}")
            settle = float(getattr(node, "gripper_settle_time", 0.0))
            if settle > 0.0:
                time.sleep(settle)
            return {"ok": True, "timed_out": True, "position": float(pos)}
        time.sleep(0.01)
    result = result_future.result()
    if result is None:
        message = f"Gripper action returned no result: {result_future.exception()!r}"
        if strict:
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


def pca_axes_3d(points):
    xyz = points[:, :3].astype(np.float64)
    if len(xyz) == 0:
        return np.zeros(3), np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])
    center = np.median(xyz, axis=0)
    centered = xyz - center
    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        major = vt[0]
        minor = vt[1]
        if major[0] < 0:
            major = -major
            minor = -minor
    except np.linalg.LinAlgError:
        major = np.array([0.0, 1.0, 0.0])
        minor = np.array([1.0, 0.0, 0.0])
    return center, major, minor


def choose_grasp_target_from_points(points, label="", min_bin_points=20):
    points = finite_xyz_points(points)
    if len(points) < max(8, min_bin_points):
        centroid = np.median(points, axis=0) if len(points) else np.asarray([0.0, 0.0, 0.0])
        return {
            "method": "median_fallback",
            "target_xyz_m": centroid.astype(float).tolist(),
            "raw_centroid_xyz_m": centroid.astype(float).tolist(),
            "reason": f"too_few_points:{len(points)}",
            "xyz_major_axis": [0.0, 1.0, 0.0],
        }
    xy_center, major, minor = pca_axes_xy(points)
    _, major_3d, _ = pca_axes_3d(points)
    centered = points[:, :2] - xy_center
    along = centered @ major
    across = centered @ minor
    raw_centroid = np.median(points[:, :3], axis=0)
    low, high = np.percentile(along, [12.0, 88.0])
    if high <= low:
        return {
            "method": "median_fallback",
            "target_xyz_m": raw_centroid.astype(float).tolist(),
            "raw_centroid_xyz_m": raw_centroid.astype(float).tolist(),
            "reason": "degenerate_major_axis",
            "xy_major_axis": major.astype(float).tolist(),
            "xy_minor_axis": minor.astype(float).tolist(),
            "xyz_major_axis": major_3d.astype(float).tolist(),
        }
    edges = np.linspace(low, high, 19)
    min_points = max(min_bin_points, int(round(len(points) * 0.025)))
    best = None
    span = max(high - low, 1e-6)
    for index in range(18):
        mask = (along >= edges[index]) & (along < edges[index + 1])
        count = int(np.count_nonzero(mask))
        if count < min_points:
            continue
        width = float(np.percentile(across[mask], 90.0) - np.percentile(across[mask], 10.0))
        z_spread = float(np.percentile(points[mask, 2], 90.0) - np.percentile(points[mask, 2], 10.0))
        center_s = float((edges[index] + edges[index + 1]) * 0.5)
        center_penalty = abs(center_s - np.median(along)) / span
        score = width + 0.0015 * math.sqrt(count) - 0.04 * center_penalty - 0.02 * z_spread
        if str(label).lower().replace(" ", "_") == "banana":
            score += 0.02 * (1.0 - min(center_penalty, 1.0))
        if best is None or score > best["score"]:
            best = {
                "score": float(score),
                "mask": mask,
                "count": count,
                "width_m": width,
                "z_spread_m": z_spread,
                "along_min_m": float(edges[index]),
                "along_max_m": float(edges[index + 1]),
            }
    if best is None:
        target = raw_centroid
        return {
            "method": "median_fallback",
            "target_xyz_m": target.astype(float).tolist(),
            "raw_centroid_xyz_m": raw_centroid.astype(float).tolist(),
            "xy_major_axis": major.astype(float).tolist(),
            "xy_minor_axis": minor.astype(float).tolist(),
            "xyz_major_axis": major_3d.astype(float).tolist(),
            "reason": "no_dense_axis_bin",
        }
    selected = points[best["mask"], :3]
    target = np.median(selected, axis=0)
    
    # Ensure consistent axis direction: make major axis point from the heavy head to the handle.
    # raw_centroid is near the heavy head, target is on the thin handle.
    head_to_handle = target - raw_centroid
    if np.linalg.norm(head_to_handle) > 1e-3:
        if np.dot(major_3d, head_to_handle) < 0:
            major_3d = -major_3d
        if np.dot(major, head_to_handle[:2]) < 0:
            major = -major
            minor = -minor
            
    return {
        "method": "local_thick_axis_band",
        "target_xyz_m": target.astype(float).tolist(),
        "raw_centroid_xyz_m": raw_centroid.astype(float).tolist(),
        "xy_major_axis": major.astype(float).tolist(),
        "xy_minor_axis": minor.astype(float).tolist(),
        "xyz_major_axis": major_3d.astype(float).tolist(),
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
    points = finite_xyz_points(points)
    if len(points) == 0 or "target_xyz_m" not in selection:
        return selection
    target = np.asarray(selection["target_xyz_m"], dtype=np.float64)
    xy_distance = np.linalg.norm(points[:, :2] - target[:2], axis=1)
    local = points[xy_distance <= float(local_radius_m)]
    if len(local) < 12:
        local = points[np.argsort(xy_distance)[: min(max(12, len(points) // 20), len(points))]]
        mode = "nearest_points"
    else:
        mode = "local_radius"
    depths = local[:, 2]
    depths = depths[np.isfinite(depths) & (depths > 0.0)]
    if len(depths) == 0:
        return selection
    surface_depth = float(np.percentile(depths, float(percentile)))
    previous_depth = float(target[2])
    target[2] = surface_depth - float(clearance_m)
    selection["target_xyz_m"] = target.astype(float).tolist()
    selection["rgbd_depth_adjustment"] = {
        "method": mode,
        "local_points": int(len(depths)),
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
    path_text = ((detection.get("files") or {}).get("foreground_points_npy"))
    if not path_text:
        return None, None
    path = Path(path_text).expanduser()
    if not path.is_file():
        return None, str(path)
    return finite_xyz_points(np.load(path)), str(path)


def save_grasp_selection_visualization(points, selection, output_dir, label):
    # Kept as a no-op dependency boundary for the pddl server. The older server
    # has a richer OpenCV visualization; planning does not require it.
    return None


def load_top_view_perception_module():
    spec = importlib.util.spec_from_file_location("pddl_top_view_seg_server", TWO_VIEW_SERVER)
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


def stop_child_processes(processes):
    for proc in processes:
        if proc.poll() is not None:
            continue
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        except ProcessLookupError:
            continue
