#!/usr/bin/env python3
"""Run ICP 6D pose estimation on one saved segmentation crop folder."""

import argparse
import json
from pathlib import Path

try:
    from .icp_pose_pipeline import DEFAULT_MODEL_DIR, estimate_pose_for_crop
except ImportError:
    from icp_pose_pipeline import DEFAULT_MODEL_DIR, estimate_pose_for_crop


def main():
    parser = argparse.ArgumentParser(description="Estimate 6D pose from a saved rgbd seg crop folder.")
    parser.add_argument("--crop-dir", required=True, help="Path to perception/icp/results/<timestamp>_<object>.")
    parser.add_argument("--object", dest="object_name", help="Object name. Defaults to metadata.json label.")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--scene-voxel-size", type=float, default=0.005)
    parser.add_argument("--model-voxel-size", type=float, default=0.003)
    parser.add_argument("--normal-radius", type=float, default=0.015)
    parser.add_argument("--normal-max-nn", type=int, default=30)
    parser.add_argument("--candidate-mode", choices=("pca", "cube", "identity"), default="pca")
    parser.add_argument("--axis-length", type=float, default=0.08)
    args = parser.parse_args()

    result = estimate_pose_for_crop(
        crop_dir=Path(args.crop_dir),
        object_name=args.object_name,
        model_dir=Path(args.model_dir),
        scene_voxel_size=args.scene_voxel_size,
        model_voxel_size=args.model_voxel_size,
        normal_radius=args.normal_radius,
        normal_max_nn=args.normal_max_nn,
        candidate_mode=args.candidate_mode,
        axis_length=args.axis_length,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
