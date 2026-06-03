#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
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
    tolerance = float(kwargs.get("tolerance", 0.004)) #0.007 is failed
    max_retries = int(kwargs.get("max_retries", 10))

    if not hasattr(node, "_pddl_gripper_client"):
        node._pddl_gripper_client = ActionClient(
            node,
            FollowJointTrajectory,
            "/gripper_controller/follow_joint_trajectory",
            callback_group=getattr(node, "callback_group", None),
        )
        
    if not hasattr(node, "_gripper_pos"):
        node._gripper_pos = None
        def js_callback(msg):
            if 'robotiq_85_left_knuckle_joint' in msg.name:
                idx = msg.name.index('robotiq_85_left_knuckle_joint')
                node._gripper_pos = msg.position[idx]
        from sensor_msgs.msg import JointState
        node._js_sub = node.create_subscription(
            JointState, 
            '/joint_states', 
            js_callback, 
            10, 
            callback_group=getattr(node, "callback_group", None)
        )

    client = node._pddl_gripper_client
    if not client.wait_for_server(timeout_sec=5.0):
        raise RuntimeError("Timed out waiting for gripper action server.")

    for attempt in range(max_retries):
        node.get_logger().info(f'Attempt {attempt + 1}: Executing trajectory to pos {pos}')
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
        timed_out = False
        while rclpy.ok() and not result_future.done():
            if time.monotonic() > deadline:
                timed_out = True
                if strict:
                    raise TimeoutError("Timed out waiting for gripper motion.")
                node.get_logger().warn(
                    "Timed out waiting for gripper result; continuing because object contact can stop closing early."
                )
                try:
                    goal_handle.cancel_goal_async()
                except Exception as exc:
                    node.get_logger().warn(f"Could not cancel timed-out gripper goal: {exc}")
                break
            time.sleep(0.01)

        settle = float(getattr(node, "gripper_settle_time", 0.0))
        if settle > 0.0:
            time.sleep(settle)

        # Closed loop check
        if timed_out:
            # If it timed out, it might have grasped an object, so we accept the result.
            return {"ok": True, "timed_out": True, "position": float(pos)}
            
        if node._gripper_pos is not None:
            error = abs(node._gripper_pos - pos)
            if error <= tolerance:
                node.get_logger().info(f"Gripper reached target {pos} within tolerance (Error: {error:.4f}).")
                return {"ok": True, "timed_out": False, "position": float(pos)}
            else:
                node.get_logger().warn(f"Tolerance not met. Target: {pos}, Actual: {node._gripper_pos:.4f}, Error: {error:.4f} > {tolerance}")
                # If we are closing the gripper (pos > 0.1) and we hit something, the controller succeeds but tolerance isn't met.
                if not strict and pos > 0.1:
                    node.get_logger().info("Accepting position anyway because gripper is closing and might have contacted an object.")
                    return {"ok": True, "timed_out": False, "position": float(pos)}
        else:
            node.get_logger().warn("Current position unknown (no joint states received).")

        time.sleep(0.5)

    node.get_logger().error(f"Failed to reach target {pos} within tolerance after {max_retries} attempts.")
    return {"ok": False, "timed_out": False, "position": float(pos)}


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
            "point_count": int(len(points)),
            "target_xyz_m": centroid.astype(float).tolist(),
            "raw_centroid_xyz_m": centroid.astype(float).tolist(),
            "pca_xy_center_m": centroid[:2].astype(float).tolist(),
            "pca_3d_center_m": centroid.astype(float).tolist(),
            "reason": f"too_few_points:{len(points)}",
            "xyz_major_axis": [0.0, 1.0, 0.0],
        }
    xy_center, major, minor = pca_axes_xy(points)
    pca_3d_center, major_3d, _ = pca_axes_3d(points)
    centered = points[:, :2] - xy_center
    along = centered @ major
    across = centered @ minor
    raw_centroid = np.median(points[:, :3], axis=0)
    low, high = np.percentile(along, [12.0, 88.0])
    if high <= low:
        return {
            "method": "median_fallback",
            "point_count": int(len(points)),
            "target_xyz_m": raw_centroid.astype(float).tolist(),
            "raw_centroid_xyz_m": raw_centroid.astype(float).tolist(),
            "reason": "degenerate_major_axis",
            "pca_xy_center_m": xy_center.astype(float).tolist(),
            "pca_3d_center_m": pca_3d_center.astype(float).tolist(),
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
            "point_count": int(len(points)),
            "target_xyz_m": target.astype(float).tolist(),
            "raw_centroid_xyz_m": raw_centroid.astype(float).tolist(),
            "pca_xy_center_m": xy_center.astype(float).tolist(),
            "pca_3d_center_m": pca_3d_center.astype(float).tolist(),
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
        "point_count": int(len(points)),
        "target_xyz_m": target.astype(float).tolist(),
        "raw_centroid_xyz_m": raw_centroid.astype(float).tolist(),
        "pca_xy_center_m": xy_center.astype(float).tolist(),
        "pca_3d_center_m": pca_3d_center.astype(float).tolist(),
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


def _safe_debug_stem(text):
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(text or "unknown"))
    return safe.strip("_") or "unknown"


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _point_bounds(points):
    if len(points) == 0:
        return None
    return {
        "min_xyz_m": np.min(points[:, :3], axis=0).astype(float).tolist(),
        "max_xyz_m": np.max(points[:, :3], axis=0).astype(float).tolist(),
    }


def _selection_band_mask(points, selection):
    band = selection.get("selected_band") or {}
    major = selection.get("xy_major_axis")
    center = selection.get("pca_xy_center_m") or selection.get("raw_centroid_xyz_m")
    if not band or major is None or center is None:
        return np.zeros(len(points), dtype=bool)
    major = np.asarray(major, dtype=np.float64)[:2]
    center = np.asarray(center, dtype=np.float64)[:2]
    if major.shape[0] < 2 or center.shape[0] < 2 or np.linalg.norm(major) < 1e-9:
        return np.zeros(len(points), dtype=bool)
    along = (points[:, :2] - center) @ (major / np.linalg.norm(major))
    return (along >= float(band["along_min_m"])) & (along < float(band["along_max_m"]))


def _sample_points(points, max_points=6000):
    if len(points) <= max_points:
        return points
    step = max(1, int(math.ceil(len(points) / float(max_points))))
    return points[::step]


def _plot_grasp_selection(points, selection, png_path, label):
    import cv2

    height, width = 520, 1560
    margin = 36
    panel_w = width // 3
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    sampled = _sample_points(points)
    selected_mask = _selection_band_mask(points, selection)
    sampled_selected = _sample_points(points[selected_mask]) if np.any(selected_mask) else np.empty((0, 3))

    raw = np.asarray(selection.get("raw_centroid_xyz_m", [np.nan, np.nan, np.nan]), dtype=np.float64)
    target = np.asarray(selection.get("target_xyz_m", [np.nan, np.nan, np.nan]), dtype=np.float64)
    grasp = np.asarray(selection.get("grasp_pose_xyz_m", target), dtype=np.float64)
    center_xy = np.asarray(selection.get("pca_xy_center_m", raw[:2]), dtype=np.float64)[:2]
    major = selection.get("xy_major_axis")
    minor = selection.get("xy_minor_axis")
    bounds = _point_bounds(points) or {}
    min_xyz = np.asarray(bounds.get("min_xyz_m", np.min(points[:, :3], axis=0)), dtype=np.float64)
    max_xyz = np.asarray(bounds.get("max_xyz_m", np.max(points[:, :3], axis=0)), dtype=np.float64)
    axis_len = max(float(np.linalg.norm(max_xyz[:2] - min_xyz[:2])) * 0.22, 0.03)

    def panel_bounds(points2):
        lo = np.nanmin(points2, axis=0)
        hi = np.nanmax(points2, axis=0)
        span = np.maximum(hi - lo, 1e-4)
        return lo - 0.08 * span, hi + 0.08 * span

    def project(point2, panel_idx, lo, hi):
        span = np.maximum(hi - lo, 1e-6)
        x0 = panel_idx * panel_w
        x = x0 + margin + (float(point2[0]) - lo[0]) / span[0] * (panel_w - 2 * margin)
        y = margin + (hi[1] - float(point2[1])) / span[1] * (height - 2 * margin)
        return int(round(x)), int(round(y))

    z = sampled[:, 2]
    z_norm = (z - np.nanmin(z)) / max(float(np.nanmax(z) - np.nanmin(z)), 1e-6)
    colors = cv2.applyColorMap((z_norm * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS).reshape(-1, 3)

    def draw_points(points2, panel_idx, lo, hi, point_colors, radius=1):
        for point, color in zip(points2, point_colors):
            cv2.circle(image, project(point, panel_idx, lo, hi), radius, tuple(int(c) for c in color), -1, cv2.LINE_AA)

    def draw_marker(point2, panel_idx, lo, hi, color, marker):
        px, py = project(point2, panel_idx, lo, hi)
        if marker == "circle":
            cv2.circle(image, (px, py), 7, color, -1, cv2.LINE_AA)
            cv2.circle(image, (px, py), 8, (0, 0, 0), 1, cv2.LINE_AA)
        elif marker == "x":
            cv2.line(image, (px - 8, py - 8), (px + 8, py + 8), color, 2, cv2.LINE_AA)
            cv2.line(image, (px - 8, py + 8), (px + 8, py - 8), color, 2, cv2.LINE_AA)
        else:
            cv2.drawMarker(image, (px, py), color, markerType=cv2.MARKER_STAR, markerSize=16, thickness=2)

    def draw_axis_cv(panel_idx, lo, hi, origin, axis, length, color):
        axis = np.asarray(axis, dtype=np.float64)[:2]
        norm = np.linalg.norm(axis)
        if norm < 1e-9:
            return
        start = project(origin, panel_idx, lo, hi)
        end = project(np.asarray(origin) + axis / norm * length, panel_idx, lo, hi)
        cv2.arrowedLine(image, start, end, color, 2, cv2.LINE_AA, tipLength=0.25)

    xy_points = sampled[:, :2]
    xy_extra = np.vstack([xy_points, raw[:2], target[:2], grasp[:2]])
    xy_lo, xy_hi = panel_bounds(xy_extra)
    cv2.putText(image, "XY top view (source frame)", (18, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 2, cv2.LINE_AA)
    draw_points(xy_points, 0, xy_lo, xy_hi, colors, radius=1)
    if len(sampled_selected):
        orange = np.repeat(np.asarray([[0, 140, 255]], dtype=np.uint8), len(sampled_selected), axis=0)
        draw_points(sampled_selected[:, :2], 0, xy_lo, xy_hi, orange, radius=2)
    if major is not None:
        draw_axis_cv(0, xy_lo, xy_hi, center_xy, major, axis_len, (214, 166, 0))
    if minor is not None:
        draw_axis_cv(0, xy_lo, xy_hi, center_xy, minor, axis_len * 0.7, (255, 60, 122))
    draw_marker(raw[:2], 0, xy_lo, xy_hi, (0, 212, 255), "circle")
    draw_marker(target[:2], 0, xy_lo, xy_hi, (0, 0, 255), "x")
    draw_marker(grasp[:2], 0, xy_lo, xy_hi, (210, 77, 255), "star")

    cv2.putText(image, "yellow=raw centroid  red=target  magenta=actual grasp ref", (18, height - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)

    cv2.line(image, (panel_w, 0), (panel_w, height), (220, 220, 220), 1)
    cv2.line(image, (2 * panel_w, 0), (2 * panel_w, height), (220, 220, 220), 1)

    pca_points = None
    if major is not None and minor is not None:
        major_arr = np.asarray(major, dtype=np.float64)[:2]
        minor_arr = np.asarray(minor, dtype=np.float64)[:2]
        along = (sampled[:, :2] - center_xy) @ major_arr
        across = (sampled[:, :2] - center_xy) @ minor_arr
        pca_points = np.column_stack([along, across])
        target_pca = np.asarray([
            float((target[:2] - center_xy) @ major_arr),
            float((target[:2] - center_xy) @ minor_arr),
        ])
        grasp_pca = np.asarray([
            float((grasp[:2] - center_xy) @ major_arr),
            float((grasp[:2] - center_xy) @ minor_arr),
        ])
        raw_pca = np.asarray([
            float((raw[:2] - center_xy) @ major_arr),
            float((raw[:2] - center_xy) @ minor_arr),
        ])
        pca_lo, pca_hi = panel_bounds(np.vstack([pca_points, target_pca, grasp_pca, raw_pca]))
        cv2.putText(image, "PCA coordinates", (panel_w + 18, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 2, cv2.LINE_AA)
        draw_points(pca_points, 1, pca_lo, pca_hi, colors, radius=1)
        if len(sampled_selected):
            selected_along = (sampled_selected[:, :2] - center_xy) @ major_arr
            selected_across = (sampled_selected[:, :2] - center_xy) @ minor_arr
            orange = np.repeat(np.asarray([[0, 140, 255]], dtype=np.uint8), len(sampled_selected), axis=0)
            draw_points(np.column_stack([selected_along, selected_across]), 1, pca_lo, pca_hi, orange, radius=2)
        band = selection.get("selected_band") or {}
        if "along_min_m" in band and "along_max_m" in band:
            x1, _ = project([float(band["along_min_m"]), pca_lo[1]], 1, pca_lo, pca_hi)
            x2, _ = project([float(band["along_max_m"]), pca_hi[1]], 1, pca_lo, pca_hi)
            cv2.rectangle(image, (x1, margin), (x2, height - margin), (230, 245, 255), -1)
            draw_points(pca_points, 1, pca_lo, pca_hi, colors, radius=1)
            if len(sampled_selected):
                orange = np.repeat(np.asarray([[0, 140, 255]], dtype=np.uint8), len(sampled_selected), axis=0)
                draw_points(np.column_stack([selected_along, selected_across]), 1, pca_lo, pca_hi, orange, radius=2)
        draw_marker(raw_pca, 1, pca_lo, pca_hi, (0, 212, 255), "circle")
        draw_marker(target_pca, 1, pca_lo, pca_hi, (0, 0, 255), "x")
        draw_marker(grasp_pca, 1, pca_lo, pca_hi, (210, 77, 255), "star")

    xz_points = sampled[:, [0, 2]]
    xz_extra = np.vstack([xz_points, raw[[0, 2]], target[[0, 2]], grasp[[0, 2]]])
    xz_lo, xz_hi = panel_bounds(xz_extra)
    cv2.putText(image, "XZ side view (source frame)", (2 * panel_w + 18, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 2, cv2.LINE_AA)
    gray_colors = np.repeat(np.asarray([[128, 128, 128]], dtype=np.uint8), len(xz_points), axis=0)
    draw_points(xz_points, 2, xz_lo, xz_hi, gray_colors, radius=1)
    if len(sampled_selected):
        orange = np.repeat(np.asarray([[0, 140, 255]], dtype=np.uint8), len(sampled_selected), axis=0)
        draw_points(sampled_selected[:, [0, 2]], 2, xz_lo, xz_hi, orange, radius=2)
    draw_marker(raw[[0, 2]], 2, xz_lo, xz_hi, (0, 212, 255), "circle")
    draw_marker(target[[0, 2]], 2, xz_lo, xz_hi, (0, 0, 255), "x")
    draw_marker(grasp[[0, 2]], 2, xz_lo, xz_hi, (210, 77, 255), "star")

    cv2.putText(
        image,
        f"{label} | method={selection.get('method')} | points={len(points)}",
        (18, height - 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (40, 40, 40),
        1,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(png_path), image)


def save_grasp_selection_visualization(points, selection, output_dir, label):
    points = finite_xyz_points(points)
    debug_dir = Path(output_dir).expanduser() / "grasp_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    stem = f"{timestamp}_{_safe_debug_stem(label)}_{_safe_debug_stem(selection.get('method'))}"
    summary_path = debug_dir / f"{stem}.json"
    png_path = debug_dir / f"{stem}.png"

    selected_mask = _selection_band_mask(points, selection)
    summary = {
        "label": str(label),
        "method": selection.get("method"),
        "point_count": int(len(points)),
        "selected_point_count": int(np.count_nonzero(selected_mask)),
        "point_bounds": _point_bounds(points),
        "selection": selection,
        "legend": {
            "raw_centroid_xyz_m": "yellow circle; median foreground point in source RGB-D frame",
            "target_xyz_m": "red x; selected grasp target from PCA band search",
            "grasp_pose_xyz_m": "magenta star; actual point used to create the robot grasp pose",
            "xy_major_axis": "cyan arrow; PCA major axis in source RGB-D XY",
            "xy_minor_axis": "purple arrow; PCA minor axis in source RGB-D XY",
            "selected_band": "orange points/shaded band",
        },
    }
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    try:
        _plot_grasp_selection(points, selection, png_path, label)
    except Exception as exc:
        selection["visualization_png_error"] = str(exc)
        png_path = None

    return {
        "summary_json": str(summary_path),
        "selection_png": str(png_path) if png_path else None,
    }


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
