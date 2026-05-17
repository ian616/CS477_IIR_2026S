#!/usr/bin/env python3
"""Draw projected 6D pose axes on a saved RGB image."""

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np


def read_pose_matrix(path, pose_index):
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"pose_matrix:\s*!!opencv-matrix\s*\n\s*rows:\s*4\s*\n\s*cols:\s*4\s*\n\s*dt:\s*[df]\s*\n\s*data:\s*\[([^\]]+)\]", text, flags=re.S)
    if not matches:
        raise RuntimeError(f"No OpenCV pose_matrix found in {path}")
    if pose_index >= len(matches):
        raise RuntimeError(f"pose_index={pose_index} out of range. available={len(matches)}")
    values = np.fromstring(matches[pose_index].replace(",", " "), sep=" ", dtype=np.float64)
    if values.size != 16:
        raise RuntimeError(f"Expected 16 pose values, got {values.size}")
    return values.reshape(4, 4)


def orthonormalize_rotation(rotation):
    u, _, vt = np.linalg.svd(rotation)
    result = u @ vt
    if np.linalg.det(result) < 0:
        u[:, -1] *= -1
        result = u @ vt
    return result


def load_bbox_offset(metadata_path, use_crop_offset):
    if not metadata_path or not use_crop_offset:
        return 0.0, 0.0
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    bbox = metadata.get("roi", {}).get("bbox_xyxy") or metadata.get("padded_detection", {}).get("bbox_xyxy")
    if not bbox:
        return 0.0, 0.0
    return float(bbox[0]), float(bbox[1])


def estimate_intrinsics_from_crop(metadata_path):
    if not metadata_path:
        raise RuntimeError("--estimate-intrinsics requires --metadata")
    metadata_path = Path(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    bbox = metadata.get("roi", {}).get("cloud_bbox_xyxy") or metadata.get("roi", {}).get("bbox_xyxy")
    if not bbox:
        raise RuntimeError("metadata.json does not contain roi bbox.")

    cloud_path = metadata_path.parent / "cloud.npy"
    if not cloud_path.is_file():
        raise RuntimeError(f"cloud.npy not found next to metadata: {cloud_path}")

    cloud = np.load(cloud_path)
    if cloud.ndim != 3 or cloud.shape[2] < 3:
        raise RuntimeError(f"Expected cloud shape HxWx3, got {cloud.shape}")

    left, top, _, _ = [float(v) for v in bbox]
    h, w = cloud.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    u = (xx.astype(np.float64) + left).reshape(-1)
    v = (yy.astype(np.float64) + top).reshape(-1)
    xyz = cloud.reshape(-1, cloud.shape[2])[:, :3].astype(np.float64)
    valid = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 1e-6)
    xyz = xyz[valid]
    u = u[valid]
    v = v[valid]
    if len(xyz) < 20:
        raise RuntimeError("Not enough valid cloud points to estimate intrinsics.")

    x_over_z = xyz[:, 0] / xyz[:, 2]
    y_over_z = xyz[:, 1] / xyz[:, 2]
    fx, cx = np.linalg.lstsq(np.c_[x_over_z, np.ones_like(x_over_z)], u, rcond=None)[0]
    fy, cy = np.linalg.lstsq(np.c_[y_over_z, np.ones_like(y_over_z)], v, rcond=None)[0]
    return float(fx), float(fy), float(cx), float(cy)


def project(points, fx, fy, cx, cy, crop_left=0.0, crop_top=0.0):
    projected = []
    for p in points:
        x, y, z = p
        if z <= 1e-9:
            projected.append(None)
            continue
        u = fx * x / z + cx - crop_left
        v = fy * y / z + cy - crop_top
        projected.append((int(round(u)), int(round(v))))
    return projected


def draw_arrow(image, start, end, color, label):
    if start is None or end is None:
        return
    cv2.arrowedLine(image, start, end, color, 3, cv2.LINE_AA, tipLength=0.18)
    cv2.circle(image, start, 4, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.putText(image, label, end, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Project PPF pose axes onto an RGB image.")
    parser.add_argument("--image", required=True, help="Input RGB/crop image path.")
    parser.add_argument("--result", required=True, help="PPF result .yml path.")
    parser.add_argument("--output", required=True, help="Output image path.")
    parser.add_argument("--metadata", help="metadata.json path. Used for crop offset if --crop-image is set.")
    parser.add_argument("--crop-image", action="store_true", help="Input image is rgb.png crop, so subtract bbox offset.")
    parser.add_argument("--pose-index", type=int, default=0)
    parser.add_argument("--axis-length", type=float, default=0.08, help="Axis length in meters.")
    parser.add_argument("--fx", type=float, default=1044.87)
    parser.add_argument("--fy", type=float, default=1045.69141)
    parser.add_argument("--cx", type=float, default=320.0)
    parser.add_argument("--cy", type=float, default=240.0)
    parser.add_argument("--estimate-intrinsics", action="store_true", help="Fit fx/fy/cx/cy from cloud.npy and metadata bbox.")
    args = parser.parse_args()

    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {args.image}")

    pose = read_pose_matrix(args.result, args.pose_index)
    origin = pose[:3, 3]
    rotation = orthonormalize_rotation(pose[:3, :3])
    crop_left, crop_top = load_bbox_offset(args.metadata, args.crop_image)
    fx, fy, cx, cy = args.fx, args.fy, args.cx, args.cy
    if args.estimate_intrinsics:
        fx, fy, cx, cy = estimate_intrinsics_from_crop(args.metadata)
        print(f"estimated intrinsics: fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}")

    points = [
        origin,
        origin + rotation[:, 0] * args.axis_length,
        origin + rotation[:, 1] * args.axis_length,
        origin + rotation[:, 2] * args.axis_length,
    ]
    projected = project(points, fx, fy, cx, cy, crop_left, crop_top)

    start = projected[0]
    draw_arrow(image, start, projected[1], (0, 0, 255), "X")
    draw_arrow(image, start, projected[2], (0, 200, 0), "Y")
    draw_arrow(image, start, projected[3], (255, 0, 0), "Z")

    text = f"t=({origin[0]:.3f}, {origin[1]:.3f}, {origin[2]:.3f}) m"
    cv2.putText(image, text, (10, max(25, image.shape[0] - 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(args.output, image)
    print(f"wrote {args.output}")
    print(f"projected origin/endpoints: {projected}")
    print(f"crop offset used: left={crop_left}, top={crop_top}")


if __name__ == "__main__":
    main()
