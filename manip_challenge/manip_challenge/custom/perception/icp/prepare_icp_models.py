#!/usr/bin/env python3
"""Prepare Open3D point-cloud assets for ICP pose estimation."""

import argparse
import json
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PERCEPTION_DIR = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "icp_models"
DEFAULT_TARGET_POINTS = 1000

np = None
o3d = None
trimesh = None

MODEL_PATHS = {
    "coke_can": PERCEPTION_DIR / "gazebo_object" / "coke_can" / "meshes" / "coke_can.dae",
    "hammer": PERCEPTION_DIR / "gazebo_object" / "hammer" / "meshes" / "hammer.dae",
    "banana": PERCEPTION_DIR / "gazebo_object" / "banana" / "meshes" / "Banana.dae",
    "meat_can": PERCEPTION_DIR / "gazebo_object" / "meat_can" / "textured.dae",
    "strawberry": PERCEPTION_DIR / "gazebo_object" / "strawberry" / "textured.dae",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Gazebo meshes to ICP-ready Open3D point clouds.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for generated .ply/.json assets.")
    parser.add_argument("--objects", nargs="+", default=sorted(MODEL_PATHS), choices=sorted(MODEL_PATHS))
    parser.add_argument("--sample-points", type=int, default=5000, help="Surface points sampled before voxel downsampling.")
    parser.add_argument("--target-points", type=int, default=DEFAULT_TARGET_POINTS, help="Soft cap after downsampling/random thinning.")
    parser.add_argument("--min-points", type=int, default=500, help="Verification lower bound for each prepared model cloud.")
    parser.add_argument("--max-points", type=int, default=1000, help="Verification upper bound for each prepared model cloud.")
    parser.add_argument("--voxel-size", type=float, default=0.0025, help="Model voxel size in meters.")
    parser.add_argument("--normal-radius", type=float, default=0.01, help="Normal estimation search radius in meters.")
    parser.add_argument("--normal-max-nn", type=int, default=30)
    parser.add_argument(
        "--normal-orientation",
        choices=("outward", "consistent", "camera"),
        default="outward",
        help="How to orient point normals after estimation.",
    )
    parser.add_argument("--model-scale", type=float, default=1.0, help="Scale applied to mesh vertices before sampling.")
    parser.add_argument("--report-file", default=None, help="Path for a JSON verification report.")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing .ply/.json assets.")
    parser.add_argument("--no-verify", action="store_true", help="Skip verification after writing assets.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files.")
    return parser.parse_args()


def require_deps():
    global np, o3d, trimesh
    if np is not None and o3d is not None and trimesh is not None:
        return
    try:
        import numpy as numpy_module
        import open3d as open3d_module
        import trimesh as trimesh_module
    except ImportError as exc:
        raise ImportError(
            "numpy, open3d, trimesh, and pycollada are required. Install them with: "
            "python3 -m pip install --user numpy open3d trimesh pycollada"
        ) from exc
    np = numpy_module
    o3d = open3d_module
    trimesh = trimesh_module


def load_mesh_with_open3d(mesh_path):
    return o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=True)


def as_trimesh(loaded):
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    if isinstance(loaded, trimesh.Scene):
        geometries = []
        for node_name in loaded.graph.nodes_geometry:
            transform, geometry_name = loaded.graph[node_name]
            geometry = loaded.geometry[geometry_name].copy()
            geometry.apply_transform(transform)
            geometries.append(geometry)
        if not geometries:
            raise RuntimeError("trimesh scene has no geometry")
        return trimesh.util.concatenate(geometries)
    raise RuntimeError(f"Unsupported trimesh load result: {type(loaded).__name__}")


def dae_unit_scale(mesh_path):
    if mesh_path.suffix.lower() != ".dae":
        return 1.0
    try:
        root = ET.parse(mesh_path).getroot()
    except ET.ParseError:
        return 1.0

    for element in root.iter():
        if element.tag.endswith("unit") and "meter" in element.attrib:
            try:
                return float(element.attrib["meter"])
            except ValueError:
                return 1.0
    return 1.0


def load_mesh_with_trimesh(mesh_path):
    loaded = trimesh.load(str(mesh_path), force="scene", process=True)
    tm = as_trimesh(loaded)
    if tm.vertices is None or tm.faces is None or len(tm.vertices) == 0 or len(tm.faces) == 0:
        raise RuntimeError(f"trimesh could not load triangles from: {mesh_path}")

    unit_scale = dae_unit_scale(mesh_path)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(tm.vertices, dtype=np.float64) * unit_scale)
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(tm.faces, dtype=np.int32))
    return mesh, unit_scale


def load_mesh(mesh_path, scale):
    if mesh_path.suffix.lower() == ".dae":
        mesh, unit_scale = load_mesh_with_trimesh(mesh_path)
        loader = "trimesh"
    else:
        mesh = load_mesh_with_open3d(mesh_path)
        loader = "open3d"
        unit_scale = 1.0
        if mesh.is_empty():
            mesh, unit_scale = load_mesh_with_trimesh(mesh_path)
            loader = "trimesh"
    if not mesh.has_triangles():
        raise RuntimeError(f"Mesh has no triangles: {mesh_path}")

    if scale != 1.0:
        mesh.scale(scale, center=(0.0, 0.0, 0.0))

    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    return mesh, loader, unit_scale


def orient_normals_outward_from_center(cloud):
    points = np.asarray(cloud.points)
    normals = np.asarray(cloud.normals)
    if len(points) == 0 or len(normals) != len(points):
        return 0.0

    center = points.mean(axis=0)
    rays = points - center
    ray_norms = np.linalg.norm(rays, axis=1)
    valid = ray_norms > 1e-12
    if not np.any(valid):
        return 0.0

    dots = np.einsum("ij,ij->i", normals[valid], rays[valid])
    valid_indices = np.flatnonzero(valid)
    normals[valid_indices[dots < 0.0]] *= -1.0
    cloud.normals = o3d.utility.Vector3dVector(normals)
    cloud.normalize_normals()
    return normal_outward_ratio(cloud)


def orient_cloud_normals(cloud, orientation):
    if orientation == "consistent":
        cloud.orient_normals_consistent_tangent_plane(30)
        return normal_outward_ratio(cloud)
    if orientation == "camera":
        cloud.orient_normals_towards_camera_location(np.zeros(3, dtype=np.float64))
        return normal_outward_ratio(cloud)
    if orientation == "outward":
        cloud.orient_normals_consistent_tangent_plane(30)
        return orient_normals_outward_from_center(cloud)
    raise ValueError(f"Unknown normal orientation: {orientation}")


def normal_outward_ratio(cloud):
    points = np.asarray(cloud.points)
    normals = np.asarray(cloud.normals)
    if len(points) == 0 or len(normals) != len(points):
        return 0.0
    center = points.mean(axis=0)
    rays = points - center
    ray_norms = np.linalg.norm(rays, axis=1)
    normal_norms = np.linalg.norm(normals, axis=1)
    valid = (ray_norms > 1e-12) & (normal_norms > 1e-12)
    if not np.any(valid):
        return 0.0
    dots = np.einsum("ij,ij->i", normals[valid], rays[valid])
    return float(np.mean(dots > 0.0))


def sample_model_cloud(mesh, sample_points, voxel_size, target_points, normal_radius, normal_max_nn, normal_orientation, seed):
    if sample_points <= 0:
        raise ValueError("--sample-points must be positive")

    # Poisson disk sampling gives a more even surface cloud than raw vertices,
    # which makes point-to-plane ICP less sensitive to mesh tessellation.
    try:
        cloud = mesh.sample_points_poisson_disk(number_of_points=sample_points, init_factor=5)
    except RuntimeError:
        cloud = mesh.sample_points_uniformly(number_of_points=sample_points)

    if voxel_size > 0.0:
        cloud = cloud.voxel_down_sample(voxel_size)

    points = np.asarray(cloud.points)
    if len(points) == 0:
        raise RuntimeError("Sampled model cloud has no points.")

    if target_points > 0 and len(points) > target_points:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(points), size=target_points, replace=False))
        cloud = cloud.select_by_index(keep.tolist())

    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=max(float(normal_radius), voxel_size * 4.0, 0.005),
            max_nn=int(normal_max_nn),
        )
    )
    orient_cloud_normals(cloud, normal_orientation)
    return cloud


def bounds_dict(geometry):
    min_bound = np.asarray(geometry.get_min_bound(), dtype=float)
    max_bound = np.asarray(geometry.get_max_bound(), dtype=float)
    extent = max_bound - min_bound
    center = (min_bound + max_bound) * 0.5
    return {
        "min_xyz": min_bound.tolist(),
        "max_xyz": max_bound.tolist(),
        "extent_xyz": extent.tolist(),
        "center_xyz": center.tolist(),
    }


def write_model_assets(object_name, mesh_path, output_dir, args):
    ply_path = output_dir / f"{object_name}.ply"
    metadata_path = output_dir / f"{object_name}.json"
    if not args.force and (ply_path.exists() or metadata_path.exists()):
        raise FileExistsError(f"Output exists for {object_name}. Re-run with --force: {ply_path}")

    mesh, loader, unit_scale = load_mesh(mesh_path, args.model_scale)
    cloud = sample_model_cloud(
        mesh=mesh,
        sample_points=args.sample_points,
        voxel_size=args.voxel_size,
        target_points=args.target_points,
        normal_radius=args.normal_radius,
        normal_max_nn=args.normal_max_nn,
        normal_orientation=args.normal_orientation,
        seed=args.seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    ok = o3d.io.write_point_cloud(str(ply_path), cloud, write_ascii=False, compressed=False)
    if not ok:
        raise RuntimeError(f"Failed to write point cloud: {ply_path}")

    points = np.asarray(cloud.points)
    metadata = {
        "object": object_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mesh_path": str(mesh_path),
        "mesh_loader": loader,
        "mesh_unit_scale": unit_scale,
        "point_cloud_path": str(ply_path),
        "model_scale": args.model_scale,
        "sample_points_requested": args.sample_points,
        "target_points": args.target_points,
        "voxel_size_m": args.voxel_size,
        "normal_radius_m": args.normal_radius,
        "normal_max_nn": args.normal_max_nn,
        "normal_orientation": args.normal_orientation,
        "normal_outward_ratio": normal_outward_ratio(cloud),
        "num_points": int(len(points)),
        "mesh_bounds": bounds_dict(mesh),
        "cloud_bounds": bounds_dict(cloud),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return ply_path, metadata_path, metadata


def verify_model_assets(object_name, output_dir, args):
    ply_path = output_dir / f"{object_name}.ply"
    metadata_path = output_dir / f"{object_name}.json"
    result = {
        "object": object_name,
        "point_cloud_path": str(ply_path),
        "metadata_path": str(metadata_path),
        "status": "ok",
        "errors": [],
        "warnings": [],
    }

    if not ply_path.is_file():
        result["errors"].append(f"missing point cloud: {ply_path}")
    if not metadata_path.is_file():
        result["errors"].append(f"missing metadata: {metadata_path}")
    if result["errors"]:
        result["status"] = "fail"
        return result

    cloud = o3d.io.read_point_cloud(str(ply_path))
    if cloud.is_empty():
        result["errors"].append("point cloud is empty")
        result["status"] = "fail"
        return result

    points = np.asarray(cloud.points)
    normals = np.asarray(cloud.normals)
    finite_points = np.isfinite(points).all(axis=1)
    normal_norms = np.linalg.norm(normals, axis=1) if len(normals) else np.asarray([])
    finite_normals = np.isfinite(normals).all(axis=1) if len(normals) else np.asarray([])

    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)

    num_points = int(len(points))
    extent = (points.max(axis=0) - points.min(axis=0)).astype(float)
    result.update(
        {
            "num_points": num_points,
            "has_normals": bool(cloud.has_normals()),
            "finite_point_ratio": float(np.mean(finite_points)),
            "finite_normal_ratio": float(np.mean(finite_normals)) if len(finite_normals) else 0.0,
            "normal_norm_mean": float(normal_norms.mean()) if len(normal_norms) else 0.0,
            "normal_norm_min": float(normal_norms.min()) if len(normal_norms) else 0.0,
            "normal_norm_max": float(normal_norms.max()) if len(normal_norms) else 0.0,
            "normal_outward_ratio": normal_outward_ratio(cloud),
            "extent_xyz_m": extent.tolist(),
            "metadata_num_points": int(metadata.get("num_points", -1)),
            "metadata_normal_orientation": metadata.get("normal_orientation"),
        }
    )

    if num_points < args.min_points or num_points > args.max_points:
        result["errors"].append(f"point count {num_points} is outside [{args.min_points}, {args.max_points}]")
    if not cloud.has_normals():
        result["errors"].append("point cloud has no normals")
    if result["finite_point_ratio"] < 1.0:
        result["errors"].append("point cloud contains non-finite XYZ values")
    if result["finite_normal_ratio"] < 1.0:
        result["errors"].append("point cloud contains non-finite normal values")
    if result["normal_norm_min"] < 0.95 or result["normal_norm_max"] > 1.05:
        result["errors"].append("normal vectors are not unit length")
    if result["normal_outward_ratio"] < 0.90:
        result["warnings"].append("less than 90% of normals point away from the cloud centroid")
    if int(metadata.get("num_points", -1)) != num_points:
        result["warnings"].append("metadata num_points does not match the .ply file")
    if np.any(extent <= 0.0):
        result["errors"].append("point cloud has a zero-size bounding box axis")

    if result["errors"]:
        result["status"] = "fail"
    elif result["warnings"]:
        result["status"] = "warn"
    return result


def write_report(report_path, report):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def verify_all(output_dir, args):
    results = [verify_model_assets(object_name, output_dir, args) for object_name in args.objects]
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
        "min_points": args.min_points,
        "max_points": args.max_points,
        "objects": results,
        "status": "ok",
    }
    if any(item["status"] == "fail" for item in results):
        report["status"] = "fail"
    elif any(item["status"] == "warn" for item in results):
        report["status"] = "warn"

    report_path = Path(args.report_file).expanduser().resolve() if args.report_file else output_dir / "prepare_report.json"
    write_report(report_path, report)

    print("\nVerification:")
    for item in results:
        count = item.get("num_points", 0)
        outward = item.get("normal_outward_ratio", 0.0)
        extent = [round(v, 5) for v in item.get("extent_xyz_m", [])]
        print(f"  {item['object']}: {item['status']} points={count} outward={outward:.3f} extent_xyz_m={extent}")
        for message in item["errors"]:
            print(f"    ERROR: {message}")
        for message in item["warnings"]:
            print(f"    WARN: {message}")
    print(f"report: {report_path}")
    return report


def main():
    args = parse_args()
    require_deps()
    output_dir = Path(args.output_dir).expanduser().resolve()

    failures = []
    if not args.verify_only:
        for object_name in args.objects:
            mesh_path = MODEL_PATHS[object_name]
            print(f"\n=== {object_name} ===")
            print(f"mesh: {mesh_path}")
            try:
                if not mesh_path.is_file():
                    raise FileNotFoundError(mesh_path)
                ply_path, metadata_path, metadata = write_model_assets(object_name, mesh_path, output_dir, args)
                extent = metadata["cloud_bounds"]["extent_xyz"]
                print(f"wrote: {ply_path}")
                print(f"meta:  {metadata_path}")
                print(
                    "points="
                    f"{metadata['num_points']} "
                    f"normal_outward={metadata['normal_outward_ratio']:.3f} "
                    f"extent_xyz_m={[round(v, 5) for v in extent]}"
                )
            except Exception as exc:
                failures.append((object_name, str(exc)))
                print(f"FAILED: {exc}")

    if failures:
        print("\nFailures:")
        for object_name, error in failures:
            print(f"  {object_name}: {error}")
        raise SystemExit(1)

    if not args.no_verify:
        report = verify_all(output_dir, args)
        if report["status"] == "fail":
            raise SystemExit(1)

    print(f"\nDone. ICP model assets are in: {output_dir}")


if __name__ == "__main__":
    main()
