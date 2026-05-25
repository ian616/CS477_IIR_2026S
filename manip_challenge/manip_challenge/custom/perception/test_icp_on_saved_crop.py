#!/usr/bin/env python3
"""Run model-to-scene ICP on a saved rgbd_crops folder."""

import argparse
import json
from pathlib import Path

try:
    from . import icp_pose_estimator as icp
except ImportError:
    import icp_pose_estimator as icp


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = SCRIPT_DIR / "icp_models"
DEFAULT_CROP_ROOT = SCRIPT_DIR / "rgbd_crops"
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"


def parse_args():
    parser = argparse.ArgumentParser(description="Test ICP pose estimation on saved foreground_points.npy data.")
    parser.add_argument("--crop-dir", required=True, help="Path to rgbd_crops/<timestamp>_<object>.")
    parser.add_argument("--object", dest="object_name", help="Object name. Defaults to metadata.json label.")
    parser.add_argument("--scene-points", default="foreground_points.npy", help="Scene .npy file name/path.")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--scene-voxel-size", type=float, default=0.005)
    parser.add_argument("--model-voxel-size", type=float, default=0.003)
    parser.add_argument("--normal-radius", type=float, default=0.015)
    parser.add_argument("--normal-max-nn", type=int, default=30)
    parser.add_argument("--candidate-mode", choices=["identity", "yaw", "cube"], default="cube")
    parser.add_argument("--coarse-threshold", type=float, default=0.08)
    parser.add_argument("--refine-threshold", type=float, default=0.03)
    parser.add_argument("--coarse-iterations", type=int, default=15)
    parser.add_argument("--refine-iterations", type=int, default=40)
    parser.add_argument("--coarse-method", choices=["point_to_point", "point_to_plane"], default="point_to_point")
    parser.add_argument("--refine-method", choices=["point_to_point", "point_to_plane"], default="point_to_plane")
    parser.add_argument("--write-aligned-clouds", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Print model/scene bounds before ICP.")
    parser.add_argument("--distance-thresholds", type=float, nargs="+", default=[0.01, 0.02, 0.04])
    return parser.parse_args()


def metadata_label(crop_dir):
    metadata_path = crop_dir / "metadata.json"
    if not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return metadata.get("label")


def main():
    args = parse_args()
    icp.require_deps()

    crop_dir = Path(args.crop_dir).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    object_name = args.object_name or metadata_label(crop_dir)
    if not object_name:
        raise RuntimeError("--object is required when metadata.json has no label.")

    scene_path = Path(args.scene_points).expanduser()
    if not scene_path.is_absolute():
        scene_path = crop_dir / scene_path
    model_path = model_dir / f"{object_name}.ply"
    if not scene_path.is_file():
        raise FileNotFoundError(scene_path)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    np = icp.np
    o3d = icp.o3d

    print(f"object: {object_name}")
    print(f"crop:   {crop_dir}")
    print(f"model:  {model_path}")
    print(f"scene:  {scene_path}")

    scene_points = np.load(scene_path)
    scene_cloud = icp.cloud_from_points(scene_points)
    scene_cloud = icp.preprocess_cloud(
        scene_cloud,
        voxel_size=args.scene_voxel_size,
        normal_radius=args.normal_radius,
        normal_max_nn=args.normal_max_nn,
        remove_outliers=True,
    )

    model_cloud = icp.load_model_cloud(model_path)
    model_cloud = icp.preprocess_cloud(
        model_cloud,
        voxel_size=args.model_voxel_size,
        normal_radius=args.normal_radius,
        normal_max_nn=args.normal_max_nn,
        remove_outliers=False,
    )

    if args.debug:
        scene_min, scene_max, scene_extent = icp.bounds(scene_cloud)
        model_min, model_max, model_extent = icp.bounds(model_cloud)
        print("\nDebug bounds")
        print(f"  raw scene points: {scene_points.reshape(-1, scene_points.shape[-1]).shape[0]}")
        print(f"  scene centroid:   {[round(v, 5) for v in icp.centroid(scene_cloud)]}")
        print(f"  scene min:        {[round(v, 5) for v in scene_min]}")
        print(f"  scene max:        {[round(v, 5) for v in scene_max]}")
        print(f"  scene extent:     {[round(v, 5) for v in scene_extent]}")
        print(f"  model centroid:   {[round(v, 5) for v in icp.centroid(model_cloud)]}")
        print(f"  model min:        {[round(v, 5) for v in model_min]}")
        print(f"  model max:        {[round(v, 5) for v in model_max]}")
        print(f"  model extent:     {[round(v, 5) for v in model_extent]}")
        print(f"  coarse threshold: {args.coarse_threshold}")
        print(f"  refine threshold: {args.refine_threshold}")

    result = icp.estimate_pose(
        object_name=object_name,
        model_cloud=model_cloud,
        scene_cloud=scene_cloud,
        candidate_mode=args.candidate_mode,
        coarse_threshold=args.coarse_threshold,
        refine_threshold=args.refine_threshold,
        coarse_iterations=args.coarse_iterations,
        refine_iterations=args.refine_iterations,
        coarse_method=args.coarse_method,
        refine_method=args.refine_method,
    )

    result_path = output_dir / f"{crop_dir.name}_icp_pose_result.json"
    aligned_model = icp.transform_cloud(model_cloud, result.transformation)
    model_to_scene = icp.cloud_distance_stats(aligned_model, scene_cloud, args.distance_thresholds)
    scene_to_model = icp.cloud_distance_stats(scene_cloud, aligned_model, args.distance_thresholds)
    result_data = result.to_dict()
    result_data["model_to_scene_distance"] = model_to_scene
    result_data["scene_to_model_distance"] = scene_to_model
    result_path.write_text(json.dumps(result_data, indent=2, sort_keys=True), encoding="utf-8")

    print("\nICP result")
    print(f"  fitness:      {result.fitness:.4f}")
    print(f"  inlier_rmse:  {result.inlier_rmse:.5f} m")
    print(f"  scene points: {result.num_scene_points}")
    print(f"  model points: {result.num_model_points}")
    print(f"  candidate:    {result.candidate_index}")
    print(f"  stage:        {result.selected_stage}")
    print(f"  result:       {result_path}")
    print("  model->scene:")
    print(f"    median={model_to_scene['median_m']:.5f}m p95={model_to_scene['p95_m']:.5f}m")
    print("  scene->model:")
    print(f"    median={scene_to_model['median_m']:.5f}m p95={scene_to_model['p95_m']:.5f}m")
    for threshold in args.distance_thresholds:
        key = f"within_{int(round(threshold * 1000.0))}mm"
        print(
            f"    within {threshold * 1000.0:.0f}mm: "
            f"model->scene={model_to_scene[key]:.3f}, scene->model={scene_to_model[key]:.3f}"
        )
    print("  pose_matrix:")
    for row in icp.np.asarray(result.transformation):
        print("    " + " ".join(f"{value: .6f}" for value in row))

    if args.write_aligned_clouds:
        aligned_path = output_dir / f"{crop_dir.name}_icp_aligned_model.ply"
        scene_out_path = output_dir / f"{crop_dir.name}_icp_scene.ply"
        o3d.io.write_point_cloud(str(aligned_path), aligned_model)
        o3d.io.write_point_cloud(str(scene_out_path), scene_cloud)
        print(f"  aligned model: {aligned_path}")
        print(f"  scene cloud:   {scene_out_path}")


if __name__ == "__main__":
    main()
