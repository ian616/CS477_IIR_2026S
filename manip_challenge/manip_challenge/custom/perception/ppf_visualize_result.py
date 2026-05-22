#!/usr/bin/env python3
"""Create PLY overlays for inspecting an OpenCV PPF pose result."""

import argparse
import re
from pathlib import Path

import numpy as np


def read_ply_xyz(path):
    path = Path(path)
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    vertex_count = None
    header_end = None
    for i, line in enumerate(lines):
        if line.startswith("element vertex"):
            vertex_count = int(line.split()[2])
        if line.strip() == "end_header":
            header_end = i + 1
            break

    if vertex_count is None or header_end is None:
        raise RuntimeError(f"Invalid PLY header: {path}")

    points = []
    for line in lines[header_end:header_end + vertex_count]:
        parts = line.split()
        if len(parts) < 3:
            continue
        points.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return np.asarray(points, dtype=np.float64)


def parse_float_array(text, start_index):
    open_end = text.find(">", start_index)
    close = text.find("</float_array>", open_end)
    if open_end < 0 or close < 0:
        return None, None, None
    tag = text[start_index:open_end + 1]
    body = text[open_end + 1:close]
    numbers = np.fromstring(body, sep=" ", dtype=np.float64)
    return tag, numbers, close + len("</float_array>")


def read_dae_positions(path, model_scale):
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    points = []
    pos = 0
    while True:
        idx = text.find("<float_array", pos)
        if idx < 0:
            break
        tag, values, pos = parse_float_array(text, idx)
        if tag is None:
            break
        tag_upper = tag.upper()
        if "POSITION" not in tag_upper and "POSITIONS" not in tag_upper:
            continue
        if values.size < 3:
            continue
        usable = values[: values.size - (values.size % 3)].reshape(-1, 3)
        points.append(usable)

    if not points:
        raise RuntimeError(f"No DAE position arrays found: {path}")
    return np.vstack(points) * float(model_scale)


def read_obj_positions(path, model_scale):
    points = []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) >= 4:
                points.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not points:
        raise RuntimeError(f"No OBJ vertices found: {path}")
    return np.asarray(points, dtype=np.float64) * float(model_scale)


def read_model_points(path, model_scale):
    suffix = Path(path).suffix.lower()
    if suffix == ".ply":
        return read_ply_xyz(path) * float(model_scale)
    if suffix == ".dae":
        return read_dae_positions(path, model_scale)
    if suffix == ".obj":
        return read_obj_positions(path, model_scale)
    raise RuntimeError(f"Unsupported model type for visualization: {path}")


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


def downsample(points, max_points):
    if max_points <= 0 or len(points) <= max_points:
        return points
    indices = np.linspace(0, len(points) - 1, max_points).astype(np.int64)
    return points[indices]


def write_colored_ply(path, point_groups):
    total = sum(len(points) for points, _ in point_groups)
    with Path(path).open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {total}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for points, color in point_groups:
            r, g, b = color
            for x, y, z in points:
                f.write(f"{x:.8f} {y:.8f} {z:.8f} {r} {g} {b}\n")


def rotation_report(matrix):
    r = matrix[:3, :3]
    det = np.linalg.det(r)
    col_norms = [np.linalg.norm(r[:, i]) for i in range(3)]
    u, _, vt = np.linalg.svd(r)
    r_ortho = u @ vt
    if np.linalg.det(r_ortho) < 0:
        u[:, -1] *= -1
        r_ortho = u @ vt
    return det, col_norms, r_ortho


def main():
    parser = argparse.ArgumentParser(description="Create a colored PLY overlay for PPF result inspection.")
    parser.add_argument("--model", required=True, help="Model mesh path: .dae, .obj, or .ply")
    parser.add_argument("--scene", required=True, help="Scene foreground PLY path")
    parser.add_argument("--result", required=True, help="PPF result .yml path")
    parser.add_argument("--output", required=True, help="Output colored overlay .ply path")
    parser.add_argument("--pose-index", type=int, default=0, help="Pose index in result file")
    parser.add_argument("--model-scale", type=float, default=1.0, help="Scale model points before applying pose")
    parser.add_argument("--max-model-points", type=int, default=12000)
    parser.add_argument("--max-scene-points", type=int, default=12000)
    args = parser.parse_args()

    scene = downsample(read_ply_xyz(args.scene), args.max_scene_points)
    model = downsample(read_model_points(args.model, args.model_scale), args.max_model_points)
    pose = read_pose_matrix(args.result, args.pose_index)

    homog = np.c_[model, np.ones(len(model))]
    transformed = (pose @ homog.T).T[:, :3]

    det, col_norms, r_ortho = rotation_report(pose)
    print(f"scene points: {len(scene)}")
    print(f"model points: {len(model)}")
    print(f"pose translation: {pose[:3, 3].tolist()}")
    print(f"rotation det: {det:.6f}")
    print(f"rotation column norms: {[round(v, 6) for v in col_norms]}")
    print("orthonormalized rotation:")
    print(r_ortho)

    write_colored_ply(
        args.output,
        [
            (scene, (40, 180, 255)),
            (transformed, (255, 80, 40)),
        ],
    )
    print(f"wrote overlay: {args.output}")
    print("Open it with MeshLab, CloudCompare, or any PLY viewer.")
    print("Color meaning: blue=scene foreground, orange=transformed model.")


if __name__ == "__main__":
    main()
