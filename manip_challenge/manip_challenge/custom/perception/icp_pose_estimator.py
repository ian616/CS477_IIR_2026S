#!/usr/bin/env python3
"""Open3D ICP utilities for model-to-scene pose estimation."""

import itertools
import math
from dataclasses import dataclass
from pathlib import Path


np = None
o3d = None


def require_deps():
    global np, o3d
    if np is not None and o3d is not None:
        return
    try:
        import numpy as numpy_module
        import open3d as open3d_module
    except ImportError as exc:
        raise ImportError(
            "numpy and open3d are required. Install them with: "
            "python3 -m pip install --user 'numpy<2' open3d"
        ) from exc
    np = numpy_module
    o3d = open3d_module


@dataclass
class IcpResult:
    object_name: str
    transformation: object
    fitness: float
    inlier_rmse: float
    num_scene_points: int
    num_model_points: int
    candidate_index: int
    selected_stage: str = "refined"

    def to_dict(self):
        require_deps()
        return {
            "object": self.object_name,
            "pose_matrix": np.asarray(self.transformation, dtype=float).tolist(),
            "fitness": float(self.fitness),
            "inlier_rmse": float(self.inlier_rmse),
            "num_scene_points": int(self.num_scene_points),
            "num_model_points": int(self.num_model_points),
            "candidate_index": int(self.candidate_index),
            "selected_stage": self.selected_stage,
        }


def load_model_cloud(path):
    require_deps()
    cloud = o3d.io.read_point_cloud(str(path))
    if cloud.is_empty():
        raise RuntimeError(f"Could not read model point cloud: {path}")
    if not cloud.has_normals():
        estimate_normals(cloud, radius=0.01, max_nn=30, orient_to_camera=False)
    return cloud


def cloud_from_points(points):
    require_deps()
    points = np.asarray(points, dtype=np.float64)
    points = points.reshape(-1, points.shape[-1])[:, :3]
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0.0)
    points = points[valid]
    if len(points) == 0:
        raise RuntimeError("No valid XYZ points.")

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    return cloud


def preprocess_cloud(cloud, voxel_size, normal_radius, normal_max_nn, remove_outliers=True):
    require_deps()
    cloud = o3d.geometry.PointCloud(cloud)
    cloud.remove_non_finite_points()

    if voxel_size > 0.0:
        cloud = cloud.voxel_down_sample(float(voxel_size))

    if remove_outliers and len(cloud.points) >= 80:
        cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    if len(cloud.points) == 0:
        raise RuntimeError("Point cloud is empty after preprocessing.")

    estimate_normals(cloud, normal_radius, normal_max_nn, orient_to_camera=True)
    return cloud


def estimate_normals(cloud, radius, max_nn, orient_to_camera):
    require_deps()
    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=float(radius), max_nn=int(max_nn))
    )
    if orient_to_camera:
        cloud.orient_normals_towards_camera_location(np.zeros(3, dtype=np.float64))


def centroid(cloud):
    require_deps()
    points = np.asarray(cloud.points)
    if len(points) == 0:
        raise RuntimeError("Cannot compute centroid of an empty cloud.")
    return points.mean(axis=0)


def bounds(cloud):
    require_deps()
    points = np.asarray(cloud.points)
    if len(points) == 0:
        raise RuntimeError("Cannot compute bounds of an empty cloud.")
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    return min_bound, max_bound, max_bound - min_bound


def cube_rotation_candidates():
    require_deps()
    rotations = []
    eye = np.eye(3)
    for permutation in itertools.permutations(range(3)):
        base = eye[:, permutation]
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            rotation = base * np.asarray(signs)
            if np.linalg.det(rotation) > 0.5:
                rotations.append(rotation)
    return rotations


def axis_angle(axis, angle):
    require_deps()
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )


def rotation_candidates(mode="cube"):
    require_deps()
    if mode == "identity":
        return [np.eye(3)]
    if mode == "yaw":
        return [axis_angle((0, 0, 1), angle) for angle in (0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0)]
    if mode == "cube":
        return cube_rotation_candidates()
    raise ValueError(f"Unknown candidate mode: {mode}")


def initial_transform(rotation, model_centroid, scene_centroid):
    require_deps()
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = scene_centroid - rotation @ model_centroid
    return transform


def icp_estimation(method):
    require_deps()
    if method == "point_to_point":
        return o3d.pipelines.registration.TransformationEstimationPointToPoint()
    if method == "point_to_plane":
        return o3d.pipelines.registration.TransformationEstimationPointToPlane()
    raise ValueError(f"Unknown ICP estimation method: {method}")


def evaluate_icp(model_cloud, scene_cloud, init, threshold, iterations, method):
    require_deps()
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(iterations))
    return o3d.pipelines.registration.registration_icp(
        model_cloud,
        scene_cloud,
        float(threshold),
        init,
        icp_estimation(method),
        criteria,
    )


def choose_better(current_best, candidate):
    if current_best is None:
        return candidate
    _, current_result = current_best
    _, candidate_result = candidate
    if is_better_result(candidate_result, current_result):
        return candidate
    return current_best


def is_better_result(candidate_result, current_result):
    if candidate_result.fitness > current_result.fitness + 1e-9:
        return True
    if abs(candidate_result.fitness - current_result.fitness) <= 1e-9:
        if candidate_result.inlier_rmse < current_result.inlier_rmse:
            return True
    return False


def estimate_pose(
    object_name,
    model_cloud,
    scene_cloud,
    candidate_mode="cube",
    coarse_threshold=0.08,
    refine_threshold=0.03,
    coarse_iterations=15,
    refine_iterations=40,
    coarse_method="point_to_point",
    refine_method="point_to_plane",
):
    require_deps()
    model_centroid = centroid(model_cloud)
    scene_centroid = centroid(scene_cloud)

    best = None
    for index, rotation in enumerate(rotation_candidates(candidate_mode)):
        init = initial_transform(rotation, model_centroid, scene_centroid)
        result = evaluate_icp(model_cloud, scene_cloud, init, coarse_threshold, coarse_iterations, coarse_method)
        best = choose_better(best, (index, result))

    best_index, coarse = best
    refined = evaluate_icp(
        model_cloud,
        scene_cloud,
        coarse.transformation,
        refine_threshold,
        refine_iterations,
        refine_method,
    )
    selected = refined
    selected_stage = "refined"
    if is_better_result(coarse, refined):
        selected = coarse
        selected_stage = "coarse"

    return IcpResult(
        object_name=object_name,
        transformation=selected.transformation,
        fitness=selected.fitness,
        inlier_rmse=selected.inlier_rmse,
        num_scene_points=len(scene_cloud.points),
        num_model_points=len(model_cloud.points),
        candidate_index=best_index,
        selected_stage=selected_stage,
    )


def transform_cloud(cloud, transformation):
    require_deps()
    transformed = o3d.geometry.PointCloud(cloud)
    transformed.transform(transformation)
    return transformed


def cloud_distance_stats(source_cloud, target_cloud, thresholds):
    require_deps()
    distances = np.asarray(source_cloud.compute_point_cloud_distance(target_cloud), dtype=np.float64)
    if len(distances) == 0:
        raise RuntimeError("Cannot compute distances for an empty cloud.")
    stats = {
        "mean_m": float(distances.mean()),
        "median_m": float(np.median(distances)),
        "p90_m": float(np.percentile(distances, 90.0)),
        "p95_m": float(np.percentile(distances, 95.0)),
        "max_m": float(distances.max()),
    }
    for threshold in thresholds:
        key = f"within_{int(round(threshold * 1000.0))}mm"
        stats[key] = float(np.mean(distances <= threshold))
    return stats
