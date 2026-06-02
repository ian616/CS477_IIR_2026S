#!/usr/bin/env python3
"""Print mask-based object area for a saved RGB-D segmentation crop."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from manip_challenge.custom.grasping.perception_features import extract_perception_features


def existing_file(path: Path) -> str | None:
    return str(path) if path.is_file() else None


def measured_area(features: dict) -> tuple[float | None, str | None]:
    for key in (
        "mask_projected_area_m2",
        "mask_surface_area_m2",
        "pca_bbox_area_m2",
    ):
        value = features.get(key)
        if value is not None:
            return float(value), key
    return None, None


def load_crop_payload(crop_dir: Path) -> dict:
    metadata_path = crop_dir / "metadata.json"
    if metadata_path.is_file():
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        payload = {}

    payload["files"] = {
        key: value
        for key, value in {
            "cloud_npy": existing_file(crop_dir / "cloud.npy"),
            "mask_cloud_size": existing_file(crop_dir / "mask_cloud_size.png"),
            "mask": existing_file(crop_dir / "mask.png"),
            "foreground_points_npy": existing_file(crop_dir / "foreground_points.npy"),
        }.items()
        if value is not None
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure actual object area from mask_cloud_size.png and cloud.npy."
    )
    parser.add_argument("crop_dir", help="Directory containing metadata.json, cloud.npy, and mask_cloud_size.png.")
    args = parser.parse_args()

    crop_dir = Path(args.crop_dir).expanduser().resolve()
    if not crop_dir.is_dir():
        raise SystemExit(f"Crop directory does not exist: {crop_dir}")

    features = extract_perception_features(load_crop_payload(crop_dir))
    area, source = measured_area(features)
    output = {
        "crop_dir": str(crop_dir),
        "measured_area_m2": area,
        "measured_area_source": source,
        "mask_projected_area_m2": features.get("mask_projected_area_m2"),
        "mask_surface_area_m2": features.get("mask_surface_area_m2"),
        "mask_area_valid_points": features.get("mask_area_valid_points"),
        "pca_bbox_area_m2": features.get("pca_bbox_area_m2"),
        "bbox_area_px2": features.get("bbox_area_px2"),
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
