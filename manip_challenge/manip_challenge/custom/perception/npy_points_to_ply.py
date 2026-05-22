#!/usr/bin/env python3
"""Convert a saved foreground_points.npy file to a simple XYZ PLY point cloud."""

import argparse
from pathlib import Path

import numpy as np


def voxel_downsample(points, voxel_size):
    if voxel_size <= 0:
        return points
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, unique_indices = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(unique_indices)]


def write_ply_xyz(path, points):
    with Path(path).open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for x, y, z in points:
            f.write(f"{x:.8f} {y:.8f} {z:.8f}\n")


def main():
    parser = argparse.ArgumentParser(description="Convert foreground_points.npy to XYZ PLY.")
    parser.add_argument("input_npy", help="Path to foreground_points.npy")
    parser.add_argument("output_ply", help="Output .ply path")
    parser.add_argument("--voxel-size", type=float, default=0.003, help="Voxel downsample size in meters.")
    parser.add_argument("--max-points", type=int, default=5000, help="Maximum points to keep after downsampling.")
    args = parser.parse_args()

    points = np.load(args.input_npy)
    points = np.asarray(points, dtype=np.float32).reshape(-1, points.shape[-1])[:, :3]
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    points = points[points[:, 2] > 0.0]
    points = voxel_downsample(points, args.voxel_size)

    if args.max_points > 0 and len(points) > args.max_points:
        rng = np.random.default_rng(7)
        indices = rng.choice(len(points), size=args.max_points, replace=False)
        points = points[np.sort(indices)]

    if len(points) == 0:
        raise RuntimeError("No valid points to write.")

    write_ply_xyz(args.output_ply, points)
    print(f"wrote {len(points)} points to {args.output_ply}")


if __name__ == "__main__":
    main()
