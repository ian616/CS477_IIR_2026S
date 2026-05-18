#!/usr/bin/env python3
"""Draw a saved PPF pose on the full-size RGB frame and write it to results/."""

import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Draw pose axes on full-size RGB image from one rgbd_crops folder.")
    parser.add_argument("crop_dir", help="Path to rgbd_crops/<timestamp>_<object>")
    parser.add_argument("--results-dir", default="/home/lhs/CS477_IIR_2026S/manip_challenge/manip_challenge/custom/perception/results")
    parser.add_argument("--axis-length", default="0.08")
    parser.add_argument("--fx", default="1044.87")
    parser.add_argument("--fy", default="1045.69141")
    parser.add_argument("--cx", default="320.0")
    parser.add_argument("--cy", default="240.0")
    parser.add_argument("--estimate-intrinsics", action="store_true", default=True)
    args = parser.parse_args()

    crop_dir = Path(args.crop_dir).expanduser().resolve()
    results_dir = Path(args.results_dir).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    image = crop_dir / "annotated.png"
    if not image.is_file():
        image = crop_dir / "rgb.png"

    result = crop_dir / "ppf_pose_result.yml"
    metadata = crop_dir / "metadata.json"
    if not result.is_file():
        raise FileNotFoundError(f"PPF result not found: {result}")

    output = results_dir / f"{crop_dir.name}_full_rgb_pose_axes.png"
    script = Path(__file__).resolve().parent / "draw_pose_axes_on_rgb.py"
    cmd = [
        str(script),
        "--image", str(image),
        "--result", str(result),
        "--metadata", str(metadata),
        "--output", str(output),
        "--axis-length", args.axis_length,
        "--fx", args.fx,
        "--fy", args.fy,
        "--cx", args.cx,
        "--cy", args.cy,
    ]
    if args.estimate_intrinsics:
        cmd.append("--estimate-intrinsics")
    subprocess.run(cmd, check=True)
    print(f"saved full RGB pose image: {output}")


if __name__ == "__main__":
    main()
