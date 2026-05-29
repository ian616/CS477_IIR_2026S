#!/usr/bin/env python3
"""PCA-oriented bbox computation and drawing helpers for RGB-D perception."""

import math

import cv2
import numpy as np


def normalize_object_name(name):
    return str(name or "").strip().lower().replace("-", "_").replace(" ", "_")


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
        return {
            "method": "median_fallback",
            "target_xyz_m": raw_centroid.astype(float).tolist(),
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
        score = width + 0.0015 * math.sqrt(count) - 0.04 * center_penalty - 0.02 * z_spread
        if label_norm == "banana":
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

    target = np.median(points[best["mask"], :3], axis=0)
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


def grasp_xyz_from_selection(selection):
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


def pca_bbox_xy(points, percentile_low=2.0, percentile_high=98.0):
    center, major, minor = pca_axes_xy(points)
    centered = points[:, :2].astype(np.float64) - center
    along_major = centered @ major
    along_minor = centered @ minor
    major_min, major_max = np.percentile(along_major, [percentile_low, percentile_high])
    minor_min, minor_max = np.percentile(along_minor, [percentile_low, percentile_high])
    length_m = float(max(0.0, major_max - major_min))
    width_m = float(max(0.0, minor_max - minor_min))

    corners_xy = []
    for major_value, minor_value in (
        (major_min, minor_min),
        (major_max, minor_min),
        (major_max, minor_max),
        (major_min, minor_max),
    ):
        corner = center + major * major_value + minor * minor_value
        corners_xy.append(corner.astype(float).tolist())

    return {
        "center_xy_m": center.astype(float).tolist(),
        "major_axis_xy": major.astype(float).tolist(),
        "minor_axis_xy": minor.astype(float).tolist(),
        "length_m": length_m,
        "width_m": width_m,
        "area_m2": length_m * width_m,
        "major_min_m": float(major_min),
        "major_max_m": float(major_max),
        "minor_min_m": float(minor_min),
        "minor_max_m": float(minor_max),
        "corners_xy_m": corners_xy,
    }


def affine_xy_to_pixels(points_xy, pixels_uv, corners_xy):
    if len(points_xy) < 3:
        return None
    matrix = np.c_[points_xy, np.ones(len(points_xy), dtype=np.float64)]
    try:
        coeff_u, *_ = np.linalg.lstsq(matrix, pixels_uv[:, 0], rcond=None)
        coeff_v, *_ = np.linalg.lstsq(matrix, pixels_uv[:, 1], rcond=None)
    except np.linalg.LinAlgError:
        return None
    corners_matrix = np.c_[corners_xy, np.ones(len(corners_xy), dtype=np.float64)]
    corners_u = corners_matrix @ coeff_u
    corners_v = corners_matrix @ coeff_v
    return np.c_[corners_u, corners_v]


def pca_bbox_pixels_from_mask(pixels_uv, percentile_low=2.0, percentile_high=98.0):
    if len(pixels_uv) < 3:
        return None
    center = np.median(pixels_uv, axis=0)
    centered = pixels_uv - center
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    major = vt[0]
    minor = vt[1]
    if major[0] < 0:
        major = -major
        minor = -minor
    along_major = centered @ major
    along_minor = centered @ minor
    major_min, major_max = np.percentile(along_major, [percentile_low, percentile_high])
    minor_min, minor_max = np.percentile(along_minor, [percentile_low, percentile_high])
    corners = []
    for major_value, minor_value in (
        (major_min, minor_min),
        (major_max, minor_min),
        (major_max, minor_max),
        (major_min, minor_max),
    ):
        corners.append(center + major * major_value + minor * minor_value)
    return np.asarray(corners, dtype=np.float64)


def compute_detection_pca_bbox(
    rgb_image,
    depth_image,
    cloud,
    detection,
    bbox_padding_ratio,
    max_depth_m,
    depth_filter,
    depth_margin_m,
    min_points,
    extract_roi,
):
    try:
        roi = extract_roi(
            rgb_image=rgb_image,
            depth_image=depth_image,
            cloud=cloud,
            detection=detection,
            bbox_padding_ratio=bbox_padding_ratio,
            max_depth_m=max_depth_m,
            depth_filter=depth_filter,
            depth_margin_m=depth_margin_m,
            min_points=min_points,
        )
    except Exception as exc:
        return {
            "label": detection.label,
            "confidence": float(detection.confidence),
            "ok": False,
            "error": str(exc),
        }

    valid_mask = (roi.mask_cloud_crop > 0) & np.isfinite(roi.cloud_crop).all(axis=2) & (roi.cloud_crop[:, :, 2] > 0.0)
    points = roi.cloud_crop[valid_mask][:, :3].astype(np.float64)
    if len(points) < max(3, min_points):
        return {
            "label": detection.label,
            "confidence": float(detection.confidence),
            "ok": False,
            "error": f"Only {len(points)} valid PCA points found.",
        }

    cloud_h, cloud_w = cloud.shape[:2]
    image_h, image_w = rgb_image.shape[:2]
    ys, xs = np.nonzero(valid_mask)
    cx1, cy1, _, _ = roi.cloud_bbox_xyxy
    cloud_u = cx1 + xs.astype(np.float64) + 0.5
    cloud_v = cy1 + ys.astype(np.float64) + 0.5
    pixels_uv = np.c_[
        cloud_u * image_w / float(cloud_w),
        cloud_v * image_h / float(cloud_h),
    ]

    pca_bbox = pca_bbox_xy(points)
    corners_xy = np.asarray(pca_bbox["corners_xy_m"], dtype=np.float64)
    corners_px = affine_xy_to_pixels(points[:, :2], pixels_uv, corners_xy)
    if corners_px is None or not np.isfinite(corners_px).all():
        corners_px = pca_bbox_pixels_from_mask(pixels_uv)
    if corners_px is None or not np.isfinite(corners_px).all():
        return {
            "label": detection.label,
            "confidence": float(detection.confidence),
            "ok": False,
            "error": "Could not project PCA bbox corners to image pixels.",
        }

    selection = choose_grasp_target_from_points(points, label=detection.label)
    target = np.asarray(selection["target_xyz_m"], dtype=np.float64)
    grasp_xyz, grasp_source = grasp_xyz_from_selection(selection)
    point_px = affine_xy_to_pixels(points[:, :2], pixels_uv, np.asarray([grasp_xyz[:2]], dtype=np.float64))
    if point_px is None or not np.isfinite(point_px).all():
        nearest_index = int(np.argmin(np.linalg.norm(points[:, :2] - grasp_xyz[:2], axis=1)))
        point_px = pixels_uv[[nearest_index]]

    pca_bbox["corners_px"] = corners_px.astype(float).tolist()
    pca_bbox["point_count"] = int(len(points))
    pca_bbox["label"] = detection.label
    pca_bbox["confidence"] = float(detection.confidence)
    pca_bbox["position_point"] = {
        "ok": True,
        "method": selection.get("method"),
        "target_xyz_m": target.astype(float).tolist(),
        "raw_centroid_xyz_m": selection.get("raw_centroid_xyz_m"),
        "grasp_xyz_m": grasp_xyz.astype(float).tolist(),
        "grasp_xyz_source": grasp_source,
        "point_px": point_px[0].astype(float).tolist(),
    }
    pca_bbox["ok"] = True
    return pca_bbox


def compute_all_detection_pca_bboxes(
    rgb_image,
    depth_image,
    cloud,
    detections,
    bbox_padding_ratio,
    max_depth_m,
    depth_filter,
    depth_margin_m,
    min_points,
    extract_roi,
):
    return [
        compute_detection_pca_bbox(
            rgb_image,
            depth_image,
            cloud,
            detection,
            bbox_padding_ratio,
            max_depth_m,
            depth_filter,
            depth_margin_m,
            min_points,
            extract_roi,
        )
        for detection in detections
    ]


def draw_pca_bbox_debug(image, pca_bboxes):
    debug = image.copy()
    colors = [
        (0, 255, 255),
        (0, 180, 0),
        (255, 128, 0),
        (255, 80, 180),
        (180, 80, 255),
        (80, 220, 255),
    ]

    drawn = 0
    for index, item in enumerate(pca_bboxes or []):
        if not item.get("ok"):
            continue
        corners = np.asarray(item.get("corners_px"), dtype=np.float64)
        if corners.shape != (4, 2) or not np.isfinite(corners).all():
            continue
        pts = np.round(corners).astype(np.int32)
        color = colors[index % len(colors)]
        cv2.polylines(debug, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
        x, y = pts[np.argmin(pts[:, 1])]
        label = f"{item.get('label', 'object')} PCA"
        cv2.putText(debug, label, (int(x), max(18, int(y) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)
        drawn += 1

    if drawn == 0:
        cv2.putText(debug, "no valid PCA bbox", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 255), 2, cv2.LINE_AA)

    for item in pca_bboxes or []:
        marker = item.get("position_point") or {}
        if not marker.get("ok"):
            continue
        point = np.asarray(marker.get("point_px"), dtype=np.float64)
        if point.shape != (2,) or not np.isfinite(point).all():
            continue
        center = tuple(np.round(point).astype(np.int32))
        cv2.circle(debug, center, 2, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(debug, center, 1, (0, 0, 255), -1, cv2.LINE_AA)
    return debug
