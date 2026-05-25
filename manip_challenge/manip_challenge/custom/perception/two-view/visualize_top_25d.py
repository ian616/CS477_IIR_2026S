#!/usr/bin/env python3
"""Visualize saved top-view YOLO RGB-D segmentation crops as 2.5D images."""

import argparse
import json
from pathlib import Path
import re

import cv2
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = THIS_DIR / "results" / "top_seg"
RESULT_DIR_RE = re.compile(r"^(\d{8})_(\d{6})_(\d{6})_(.+)$")


def load_json(path):
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def latest_crop_dir(results_dir, target=None):
    candidates = [path for path in Path(results_dir).expanduser().iterdir() if path.is_dir()]
    candidates = [path for path in candidates if (path / "foreground_points.npy").is_file()]
    if target:
        normalized_target = normalize_label(target)
        candidates = [path for path in candidates if normalize_label(label_from_result_dir(path)) == normalized_target]
    if not candidates:
        suffix = f" for target '{target}'" if target else ""
        raise FileNotFoundError(f"No crop result folders with foreground_points.npy{suffix} under {results_dir}")
    return max(candidates, key=result_sort_key)


def result_sort_key(path):
    match = RESULT_DIR_RE.match(path.name)
    if match:
        date, time, micros, _ = match.groups()
        return date, time, micros
    return "", "", f"{(path / 'foreground_points.npy').stat().st_mtime:020.6f}"


def normalize_label(label):
    return str(label or "").lower().replace("-", "_").replace(" ", "_").strip()


def label_from_result_dir(path):
    match = RESULT_DIR_RE.match(path.name)
    if match:
        return match.group(4)
    metadata = load_json(path / "metadata.json")
    return metadata.get("label") or metadata.get("target") or ""


def require_file(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def read_gray(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    return image


def read_color(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    return image


def normalize_to_uint8(values, valid=None):
    values = np.asarray(values, dtype=np.float32)
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


def colorize_depth(depth, mask=None):
    depth = np.asarray(depth)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    depth = depth.astype(np.float32, copy=False)
    valid = np.isfinite(depth) & (depth > 0.0)
    if mask is not None and mask.shape[:2] == depth.shape[:2]:
        valid &= mask > 0
    gray = normalize_to_uint8(depth, valid)
    color = cv2.applyColorMap(255 - gray, cv2.COLORMAP_TURBO)
    color[~valid] = (35, 35, 35)
    return color


def resize_keep_aspect(image, size):
    target_w, target_h = size
    h, w = image.shape[:2]
    scale = min(target_w / max(w, 1), target_h / max(h, 1))
    resized = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_h, target_w, 3), 245, dtype=np.uint8)
    y = (target_h - resized.shape[0]) // 2
    x = (target_w - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def add_title(image, title):
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (20, 20, 20), -1)
    cv2.putText(out, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def overlay_mask(rgb, mask):
    if mask.shape[:2] != rgb.shape[:2]:
        mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    out = rgb.copy()
    red = np.zeros_like(out)
    red[:, :, 2] = 255
    alpha = (mask > 0)[:, :, None]
    out = np.where(alpha, cv2.addWeighted(out, 0.55, red, 0.45, 0.0), out)
    return out


def project_points(points, axis_x, axis_y, width=900, height=900, margin=70, equal_scale=True):
    xy = points[:, [axis_x, axis_y]].astype(np.float64)
    finite = np.isfinite(xy).all(axis=1)
    xy = xy[finite]
    if len(xy) == 0:
        raise RuntimeError("No finite points to project.")

    mins = np.percentile(xy, 1.0, axis=0)
    maxs = np.percentile(xy, 99.0, axis=0)
    span = np.maximum(maxs - mins, 1e-4)
    center = (mins + maxs) * 0.5
    if equal_scale:
        max_span = float(np.max(span))
        span[:] = max_span
        mins = center - span * 0.5
        maxs = center + span * 0.5

    usable_w = width - 2 * margin
    usable_h = height - 2 * margin
    scale_x = usable_w / span[0]
    scale_y = usable_h / span[1]
    pixels_x = margin + (xy[:, 0] - mins[0]) * scale_x
    pixels_y = height - margin - (xy[:, 1] - mins[1]) * scale_y
    pixels = np.c_[pixels_x, pixels_y].round().astype(np.int32)
    return pixels, finite, mins, maxs


def draw_grasp_axes(canvas, points, valid_mask, axis_x, axis_y, mins, maxs, centroid):
    xy = points[valid_mask][:, [axis_x, axis_y]].astype(np.float64)
    if len(xy) < 8:
        return None

    xy_center = np.median(xy, axis=0)
    centered = xy - xy_center
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    major = vt[0]
    minor = vt[1]

    width = canvas.shape[1]
    height = canvas.shape[0]
    margin = 70
    span = np.maximum(maxs - mins, 1e-4)

    def to_px(point):
        x = margin + (point[0] - mins[0]) * (width - 2 * margin) / span[0]
        y = height - margin - (point[1] - mins[1]) * (height - 2 * margin) / span[1]
        return int(round(x)), int(round(y))

    center_px = to_px(xy_center)
    length = float(np.max(np.ptp(xy, axis=0))) * 0.32
    jaw_half = min(max(float(np.min(np.ptp(xy, axis=0))) * 0.30, 0.015), 0.055)
    finger_half_len = max(length * 0.42, 0.025)

    major_a = to_px(xy_center - major * length)
    major_b = to_px(xy_center + major * length)
    minor_a = to_px(xy_center - minor * length * 0.65)
    minor_b = to_px(xy_center + minor * length * 0.65)

    cv2.line(canvas, major_a, major_b, (0, 220, 255), 3, cv2.LINE_AA)
    cv2.line(canvas, minor_a, minor_b, (255, 220, 0), 3, cv2.LINE_AA)
    cv2.circle(canvas, center_px, 7, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(canvas, center_px, 7, (20, 20, 20), 2, cv2.LINE_AA)

    for sign in (-1.0, 1.0):
        jaw_center = xy_center + minor * jaw_half * sign
        p1 = to_px(jaw_center - major * finger_half_len)
        p2 = to_px(jaw_center + major * finger_half_len)
        cv2.line(canvas, p1, p2, (255, 255, 255), 7, cv2.LINE_AA)
        cv2.line(canvas, p1, p2, (25, 25, 25), 2, cv2.LINE_AA)

    grasp_xyz = centroid.astype(float).tolist()
    return {
        "center_xyz_m": grasp_xyz,
        "xy_major_axis": major.astype(float).tolist(),
        "xy_minor_axis": minor.astype(float).tolist(),
        "jaw_half_width_m": jaw_half,
    }


def render_projection(points, axis_x, axis_y, color_axis, title, centroid=None, draw_candidate=False):
    canvas = np.full((900, 900, 3), 250, dtype=np.uint8)
    pixels, valid_mask, mins, maxs = project_points(points, axis_x, axis_y)
    color_values = points[valid_mask, color_axis].astype(np.float32)
    colors = cv2.applyColorMap(normalize_to_uint8(color_values), cv2.COLORMAP_TURBO)
    order = np.argsort(color_values)
    for idx in order:
        x, y = pixels[idx]
        if 0 <= x < canvas.shape[1] and 0 <= y < canvas.shape[0]:
            color = tuple(int(v) for v in colors[idx, 0])
            cv2.circle(canvas, (x, y), 2, color, -1, cv2.LINE_AA)

    candidate = None
    if centroid is not None:
        candidate = draw_grasp_axes(canvas, points, valid_mask, axis_x, axis_y, mins, maxs, centroid) if draw_candidate else None
        c = centroid[[axis_x, axis_y]].astype(np.float64)
        span = np.maximum(maxs - mins, 1e-4)
        cx = int(round(70 + (c[0] - mins[0]) * (900 - 140) / span[0]))
        cy = int(round(900 - 70 - (c[1] - mins[1]) * (900 - 140) / span[1]))
        cv2.drawMarker(canvas, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 28, 4, cv2.LINE_AA)
        cv2.drawMarker(canvas, (cx, cy), (0, 0, 0), cv2.MARKER_CROSS, 28, 1, cv2.LINE_AA)

    axis_names = ["x", "y", "z"]
    cv2.putText(
        canvas,
        f"{axis_names[axis_x]} vs {axis_names[axis_y]} colored by {axis_names[color_axis]}",
        (18, 880),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    return add_title(canvas, title), candidate


def make_summary_panel(metadata, points, candidate):
    panel = np.full((420, 900, 3), 245, dtype=np.uint8)
    label = metadata.get("label") or metadata.get("target") or "unknown"
    frame_id = metadata.get("frame_id", "unknown_frame")
    centroid = np.median(points[:, :3], axis=0)
    mins = np.min(points[:, :3], axis=0)
    maxs = np.max(points[:, :3], axis=0)
    lines = [
        f"target: {label}",
        f"frame: {frame_id}",
        f"foreground points: {len(points)}",
        f"centroid xyz: {centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f} m",
        f"bounds x: {mins[0]:.4f} .. {maxs[0]:.4f} m",
        f"bounds y: {mins[1]:.4f} .. {maxs[1]:.4f} m",
        f"bounds z: {mins[2]:.4f} .. {maxs[2]:.4f} m",
    ]
    if candidate:
        lines.append(f"candidate jaw half-width: {candidate['jaw_half_width_m']:.4f} m")

    cv2.rectangle(panel, (0, 0), (panel.shape[1], 40), (20, 20, 20), -1)
    cv2.putText(panel, "2.5D Top-View Summary", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    y = 78
    for line in lines:
        cv2.putText(panel, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (30, 30, 30), 2, cv2.LINE_AA)
        y += 42
    return panel


def build_report(crop_dir, output_dir=None):
    crop_dir = Path(crop_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve() if output_dir else crop_dir / "visual_25d"
    output_dir.mkdir(parents=True, exist_ok=True)

    rgb = read_color(require_file(crop_dir / "rgb.png"))
    mask_path = crop_dir / "mask.png"
    mask = read_gray(mask_path) if mask_path.is_file() else np.full(rgb.shape[:2], 255, dtype=np.uint8)
    depth = np.load(require_file(crop_dir / "depth.npy"))
    points = np.load(require_file(crop_dir / "foreground_points.npy"))
    points = points.reshape(-1, points.shape[-1])[:, :3].astype(np.float64)
    points = points[np.isfinite(points).all(axis=1) & (points[:, 2] > 0.0)]
    if len(points) == 0:
        raise RuntimeError("foreground_points.npy has no valid XYZ points.")

    metadata = load_json(crop_dir / "metadata.json")
    centroid = np.median(points[:, :3], axis=0)

    rgb_overlay = overlay_mask(rgb, mask)
    depth_vis = colorize_depth(depth, mask if mask.shape[:2] == depth.shape[:2] else None)
    annotated = read_color(crop_dir / "annotated.png") if (crop_dir / "annotated.png").is_file() else rgb_overlay

    topdown, candidate = render_projection(points, 0, 1, 2, "Top-down XY 2.5D + grasp candidate", centroid, True)
    side_xz, _ = render_projection(points, 0, 2, 1, "Side view XZ", centroid, False)
    side_yz, _ = render_projection(points, 1, 2, 0, "Side view YZ", centroid, False)
    summary = make_summary_panel(metadata, points, candidate)

    cv2.imwrite(str(output_dir / "rgb_mask_overlay.png"), rgb_overlay)
    cv2.imwrite(str(output_dir / "depth_colormap.png"), depth_vis)
    cv2.imwrite(str(output_dir / "topdown_xy_grasp_candidate.png"), topdown)
    cv2.imwrite(str(output_dir / "side_xz.png"), side_xz)
    cv2.imwrite(str(output_dir / "side_yz.png"), side_yz)
    cv2.imwrite(str(output_dir / "summary.png"), summary)

    tile_size = (640, 420)
    tiles = [
        add_title(resize_keep_aspect(annotated, tile_size), "full RGB detection"),
        add_title(resize_keep_aspect(rgb_overlay, tile_size), "crop mask overlay"),
        add_title(resize_keep_aspect(depth_vis, tile_size), "masked depth"),
        resize_keep_aspect(topdown, tile_size),
        resize_keep_aspect(side_xz, tile_size),
        resize_keep_aspect(side_yz, tile_size),
    ]
    row1 = np.hstack(tiles[:3])
    row2 = np.hstack(tiles[3:])
    composite = np.vstack([row1, row2])
    cv2.imwrite(str(output_dir / "report.png"), composite)

    report = {
        "crop_dir": str(crop_dir),
        "output_dir": str(output_dir),
        "num_points": int(len(points)),
        "centroid_xyz_m": centroid.astype(float).tolist(),
        "bounds_min_xyz_m": np.min(points, axis=0).astype(float).tolist(),
        "bounds_max_xyz_m": np.max(points, axis=0).astype(float).tolist(),
        "candidate": candidate,
        "files": {
            "report": str(output_dir / "report.png"),
            "topdown_xy_grasp_candidate": str(output_dir / "topdown_xy_grasp_candidate.png"),
            "side_xz": str(output_dir / "side_xz.png"),
            "side_yz": str(output_dir / "side_yz.png"),
            "summary": str(output_dir / "summary.png"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description="Create visual 2.5D reports from a saved top-view segmentation crop.")
    parser.add_argument("--crop-dir", help="One saved result folder. Defaults to latest folder under --results-dir.")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR), help="Directory containing top-view result folders.")
    parser.add_argument("--target", help="Use the latest result for this target label, e.g. banana or meat_can.")
    parser.add_argument("--output-dir", help="Output directory. Defaults to <crop-dir>/visual_25d.")
    args = parser.parse_args()

    crop_dir = Path(args.crop_dir).expanduser() if args.crop_dir else latest_crop_dir(args.results_dir, args.target)
    report = build_report(crop_dir, args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
