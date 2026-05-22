#!/usr/bin/env python3
"""Run the PPF pose test pipeline for all available saved object crops."""

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from draw_pose_axes_on_rgb import (
    estimate_intrinsics_from_crop,
    orthonormalize_rotation,
    project,
    read_pose_matrix,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CROP_ROOT = SCRIPT_DIR / "rgbd_crops"
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"

MODEL_PATHS = {
    "coke_can": SCRIPT_DIR / "gazebo_object" / "coke_can" / "meshes" / "coke_can.dae",
    "hammer": SCRIPT_DIR / "gazebo_object" / "hammer" / "meshes" / "hammer.dae",
    "banana": SCRIPT_DIR / "gazebo_object" / "banana" / "meshes" / "Banana.dae",
    "meat_can": SCRIPT_DIR / "gazebo_object" / "meat_can" / "textured.dae",
    "strawberry": SCRIPT_DIR / "gazebo_object" / "strawberry" / "textured.dae",
}

AXIS_COLORS = {
    "x": (0, 0, 255),
    "y": (0, 200, 0),
    "z": (255, 0, 0),
}

OBJECT_COLORS = {
    "coke_can": (255, 255, 255),
    "hammer": (0, 255, 255),
    "banana": (0, 220, 255),
    "meat_can": (255, 170, 80),
    "strawberry": (180, 120, 255),
}


def detect_object_from_crop_name(crop_dir):
    name = crop_dir.name
    for object_name in MODEL_PATHS:
        if name.endswith("_" + object_name) or object_name in name:
            return object_name
    return None


def list_latest_crops(crop_root, object_names):
    selected = {}
    for crop_dir in sorted(crop_root.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True):
        if not crop_dir.is_dir():
            continue
        object_name = detect_object_from_crop_name(crop_dir)
        if object_name not in object_names:
            continue
        if object_name not in selected:
            selected[object_name] = crop_dir
    return selected


def run(cmd):
    print("+ " + " ".join(str(part) for part in cmd))
    subprocess.run([str(part) for part in cmd], check=True)


def ensure_scene_ply(crop_dir, voxel_size):
    output_ply = crop_dir / "scene_foreground.ply"
    run([
        SCRIPT_DIR / "npy_points_to_ply.py",
        crop_dir / "foreground_points.npy",
        output_ply,
        "--voxel-size",
        voxel_size,
    ])
    return output_ply


def run_ppf(object_name, crop_dir, scene_ply, args):
    output_yml = crop_dir / "ppf_pose_result.yml"
    run([
        SCRIPT_DIR / "ppf_pose_test",
        "--model",
        MODEL_PATHS[object_name],
        "--scene",
        scene_ply,
        "--output",
        output_yml,
        "--model-scale",
        args.model_scale,
        "--relative-sampling-step",
        args.relative_sampling_step,
        "--relative-distance-step",
        args.relative_distance_step,
        "--scene-sample-step",
        args.scene_sample_step,
        "--scene-distance",
        args.scene_distance,
        "--num-angles",
        args.num_angles,
    ])
    return output_yml


def draw_single_result(crop_dir, result_yml, results_dir, args):
    output_png = results_dir / f"{crop_dir.name}_full_rgb_pose_axes.png"
    run([
        SCRIPT_DIR / "draw_pose_axes_on_rgb.py",
        "--image",
        crop_dir / "annotated.png",
        "--result",
        result_yml,
        "--metadata",
        crop_dir / "metadata.json",
        "--output",
        output_png,
        "--axis-length",
        args.axis_length,
        "--estimate-intrinsics",
    ])
    return output_png


def draw_arrow(image, start, end, color, label):
    if start is None or end is None:
        return
    cv2.arrowedLine(image, start, end, color, 3, cv2.LINE_AA, tipLength=0.18)
    cv2.circle(image, start, 4, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.putText(image, label, end, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def draw_combined(processed, output_path, axis_length):
    if not processed:
        return None

    base_image = None
    for item in processed:
        candidate = item["crop_dir"] / "annotated.png"
        base_image = cv2.imread(str(candidate), cv2.IMREAD_COLOR)
        if base_image is not None:
            break
    if base_image is None:
        print("combined image skipped: could not read any annotated.png")
        return None

    for item in processed:
        object_name = item["object_name"]
        crop_dir = item["crop_dir"]
        result_yml = item["result_yml"]
        metadata = crop_dir / "metadata.json"
        pose = read_pose_matrix(result_yml, 0)
        origin = pose[:3, 3]
        rotation = orthonormalize_rotation(pose[:3, :3])
        fx, fy, cx, cy = estimate_intrinsics_from_crop(metadata)
        points = [
            origin,
            origin + rotation[:, 0] * axis_length,
            origin + rotation[:, 1] * axis_length,
            origin + rotation[:, 2] * axis_length,
        ]
        projected = project(points, fx, fy, cx, cy)
        draw_arrow(base_image, projected[0], projected[1], AXIS_COLORS["x"], f"{object_name} X")
        draw_arrow(base_image, projected[0], projected[2], AXIS_COLORS["y"], "Y")
        draw_arrow(base_image, projected[0], projected[3], AXIS_COLORS["z"], "Z")
        if projected[0] is not None:
            cv2.putText(
                base_image,
                object_name,
                (projected[0][0] + 6, projected[0][1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                OBJECT_COLORS.get(object_name, (255, 255, 255)),
                2,
                cv2.LINE_AA,
            )

    cv2.imwrite(str(output_path), base_image)
    print(f"saved combined image: {output_path}")
    return output_path


def summarize(processed, results_dir):
    summary = []
    for item in processed:
        pose = read_pose_matrix(item["result_yml"], 0)
        summary.append({
            "object": item["object_name"],
            "crop_dir": str(item["crop_dir"]),
            "result_yml": str(item["result_yml"]),
            "single_overlay": str(item["single_overlay"]),
            "translation_xyz_m": pose[:3, 3].astype(float).tolist(),
        })
    path = results_dir / "ppf_batch_summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"saved summary: {path}")


def print_completion(item):
    pose = read_pose_matrix(item["result_yml"], 0)
    t = pose[:3, 3]
    print(
        "\n"
        f"[DONE] {item['object_name']}\n"
        f"  crop: {item['crop_dir']}\n"
        f"  pose xyz: [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}] m\n"
        f"  result: {item['result_yml']}\n"
        f"  image: {item['single_overlay']}\n",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Run PPF pose estimation for latest saved crops of all objects.")
    parser.add_argument("--crop-root", default=str(DEFAULT_CROP_ROOT))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--objects", nargs="+", default=list(MODEL_PATHS.keys()))
    parser.add_argument("--all-crops", action="store_true", help="Process every crop folder instead of latest per object.")
    parser.add_argument("--voxel-size", default="0.003")
    parser.add_argument("--axis-length", type=float, default=0.08)
    parser.add_argument("--model-scale", default="1.0")
    parser.add_argument("--relative-sampling-step", default="0.04")
    parser.add_argument("--relative-distance-step", default="0.04")
    parser.add_argument("--scene-sample-step", default="0.2")
    parser.add_argument("--scene-distance", default="0.03")
    parser.add_argument("--num-angles", default="30")
    args = parser.parse_args()

    crop_root = Path(args.crop_root).expanduser().resolve()
    results_dir = Path(args.results_dir).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    object_names = set(args.objects)

    if args.all_crops:
        crop_map = {}
        for crop_dir in sorted(crop_root.glob("*")):
            object_name = detect_object_from_crop_name(crop_dir)
            if object_name in object_names:
                crop_map[f"{object_name}:{crop_dir.name}"] = crop_dir
    else:
        crop_map = list_latest_crops(crop_root, object_names)

    if not crop_map:
        raise RuntimeError(f"No matching crop folders found under {crop_root}")

    processed = []
    for key, crop_dir in crop_map.items():
        object_name = key.split(":", 1)[0] if ":" in key else key
        print(f"\n=== {object_name}: {crop_dir} ===")
        try:
            if not MODEL_PATHS[object_name].is_file():
                print(f"skip {object_name}: model missing: {MODEL_PATHS[object_name]}")
                continue
            if not (crop_dir / "foreground_points.npy").is_file():
                print(f"skip {object_name}: foreground_points.npy missing")
                continue
            if not (crop_dir / "metadata.json").is_file():
                print(f"skip {object_name}: metadata.json missing")
                continue
            scene_ply = ensure_scene_ply(crop_dir, args.voxel_size)
            result_yml = run_ppf(object_name, crop_dir, scene_ply, args)
            single_overlay = draw_single_result(crop_dir, result_yml, results_dir, args)
            item = {
                "object_name": object_name,
                "crop_dir": crop_dir,
                "scene_ply": scene_ply,
                "result_yml": result_yml,
                "single_overlay": single_overlay,
            }
            processed.append(item)
            print_completion(item)
        except subprocess.CalledProcessError as exc:
            print(f"\n[FAILED] {object_name}: command exited with {exc.returncode}\n", flush=True)
        except Exception as exc:
            print(f"\n[FAILED] {object_name}: {exc}\n", flush=True)

    combined = results_dir / "ppf_all_objects_full_rgb_pose_axes.png"
    draw_combined(processed, combined, args.axis_length)
    summarize(processed, results_dir)


if __name__ == "__main__":
    main()
