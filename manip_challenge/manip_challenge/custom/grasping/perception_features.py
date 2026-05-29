#!/usr/bin/env python3
"""Perception feature extraction helpers for grasping."""

import math
from pathlib import Path

import numpy as np


def _as_float_list(values, expected_len=None):
    if values is None:
        return None
    try:
        output = [float(value) for value in values]
    except (TypeError, ValueError):
        return None
    if expected_len is not None and len(output) != expected_len:
        return None
    return output


def _bbox_metrics(bbox_xyxy):
    bbox = _as_float_list(bbox_xyxy, expected_len=4)
    if bbox is None:
        return {
            "bbox_xyxy": None,
            "bbox_width_px": None,
            "bbox_length_px": None,
            "bbox_area_px2": None,
        }

    left, top, right, bottom = bbox
    width = max(0.0, right - left)
    length = max(0.0, bottom - top)
    return {
        "bbox_xyxy": bbox,
        "bbox_width_px": width,
        "bbox_length_px": length,
        "bbox_area_px2": width * length,
    }


def _pca_bbox_from_points(points_path, major_axis, minor_axis, percentile_low=2.0, percentile_high=98.0):
    if not points_path or major_axis is None or minor_axis is None:
        return {}

    path = Path(points_path).expanduser()
    if not path.is_file():
        return {}

    try:
        points = np.load(path)
    except Exception:
        return {}

    points = np.asarray(points, dtype=np.float64).reshape(-1, np.asarray(points).shape[-1])[:, :3]
    points = points[np.isfinite(points).all(axis=1) & (points[:, 2] > 0.0)]
    if len(points) < 3:
        return {}

    major = np.asarray(major_axis, dtype=np.float64)
    minor = np.asarray(minor_axis, dtype=np.float64)
    if np.linalg.norm(major) <= 0.0 or np.linalg.norm(minor) <= 0.0:
        return {}

    major = major / np.linalg.norm(major)
    minor = minor / np.linalg.norm(minor)
    center_xy = np.median(points[:, :2], axis=0)
    centered_xy = points[:, :2] - center_xy
    along_major = centered_xy @ major
    along_minor = centered_xy @ minor

    major_min, major_max = np.percentile(along_major, [percentile_low, percentile_high])
    minor_min, minor_max = np.percentile(along_minor, [percentile_low, percentile_high])
    length_m = float(max(0.0, major_max - major_min))
    width_m = float(max(0.0, minor_max - minor_min))

    return {
        "pca_bbox_center_xy_m": center_xy.astype(float).tolist(),
        "pca_bbox_length_m": length_m,
        "pca_bbox_width_m": width_m,
        "pca_bbox_area_m2": length_m * width_m,
        "pca_bbox_major_min_m": float(major_min),
        "pca_bbox_major_max_m": float(major_max),
        "pca_bbox_minor_min_m": float(minor_min),
        "pca_bbox_minor_max_m": float(minor_max),
    }


def extract_perception_features(perception_info):
    """Extract target pose/PCA/bbox values from two-view perception output."""
    perception_info = perception_info or {}
    if not isinstance(perception_info, dict):
        perception_info = {}

    detection_result = (
        perception_info.get("detection_result")
        or perception_info.get("perception_result")
        or {}
    )
    if not detection_result and (
        "location_xyz_m" in perception_info
        or "roi" in perception_info
        or "files" in perception_info
    ):
        detection_result = perception_info
    if not isinstance(detection_result, dict):
        detection_result = {}

    grasp_selection = perception_info.get("grasp_selection") or {}
    if not isinstance(grasp_selection, dict):
        grasp_selection = {}

    roi = detection_result.get("roi") or {}
    selected_detection = detection_result.get("detection") or {}
    files = detection_result.get("files") or {}

    target_position_xyz_m = (
        _as_float_list(detection_result.get("location_xyz_m"), expected_len=3)
        or _as_float_list(roi.get("centroid_xyz"), expected_len=3)
        or _as_float_list(grasp_selection.get("raw_centroid_xyz_m"), expected_len=3)
    )
    selected_grasp_xyz_m = _as_float_list(grasp_selection.get("target_xyz_m"), expected_len=3)

    pca_major_axis_xy = _as_float_list(grasp_selection.get("xy_major_axis"), expected_len=2)
    pca_minor_axis_xy = _as_float_list(grasp_selection.get("xy_minor_axis"), expected_len=2)

    target_orientation_yaw_rad = None
    target_orientation_yaw_deg = None
    if pca_major_axis_xy is not None:
        target_orientation_yaw_rad = math.atan2(pca_major_axis_xy[1], pca_major_axis_xy[0])
        target_orientation_yaw_deg = math.degrees(target_orientation_yaw_rad)

    bbox = roi.get("bbox_xyxy") or selected_detection.get("bbox_xyxy")
    bbox_info = _bbox_metrics(bbox)

    pca_bbox = grasp_selection.get("pca_bbox") or {}
    if not isinstance(pca_bbox, dict):
        pca_bbox = {}
    pca_bbox_info = {
        "pca_bbox_length_m": pca_bbox.get("length_m"),
        "pca_bbox_width_m": pca_bbox.get("width_m"),
        "pca_bbox_area_m2": pca_bbox.get("area_m2"),
        "pca_bbox_center_xy_m": pca_bbox.get("center_xy_m"),
    }
    if pca_bbox_info["pca_bbox_length_m"] is None or pca_bbox_info["pca_bbox_width_m"] is None:
        pca_bbox_info.update(
            _pca_bbox_from_points(
                grasp_selection.get("points_path") or files.get("foreground_points_npy"),
                pca_major_axis_xy,
                pca_minor_axis_xy,
            )
        )

    return {
        "target_position_xyz_m": target_position_xyz_m,
        "selected_grasp_xyz_m": selected_grasp_xyz_m,
        "target_frame_id": detection_result.get("frame_id"),
        "target_orientation_yaw_rad": target_orientation_yaw_rad,
        "target_orientation_yaw_deg": target_orientation_yaw_deg,
        "pca_major_axis_xy": pca_major_axis_xy,
        "pca_minor_axis_xy": pca_minor_axis_xy,
        **bbox_info,
        **pca_bbox_info,
    }
