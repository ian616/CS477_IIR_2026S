#!/usr/bin/env python3
"""Estimate 6D pose from a saved segmentation RGB-D crop and draw pose axes."""

import json
import math
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = Path(
    "/home/lhs/CS477_IIR_2026S/manip_challenge/manip_challenge/custom/perception/icp/icp_models"
)

np = None
o3d = None
cv2 = None


def require_deps():
    global np, o3d, cv2
    if np is not None and o3d is not None and cv2 is not None:
        return
    try:
        import cv2 as cv2_module
        import numpy as numpy_module
        import open3d as open3d_module
    except ImportError as exc:
        raise ImportError("numpy, opencv-python, and open3d are required for ICP pose estimation.") from exc
    np = numpy_module
    o3d = open3d_module
    cv2 = cv2_module


@dataclass
class PoseEstimate:
    object_name: str
    transformation: object
    fitness: float
    inlier_rmse: float
    selected_candidate: int
    selected_stage: str
    num_scene_points: int
    num_model_points: int
    intrinsics: tuple
    files: dict

    def to_dict(self):
        require_deps()
        rotation = np.asarray(self.transformation, dtype=float)[:3, :3]
        translation = np.asarray(self.transformation, dtype=float)[:3, 3]
        return {
            "object": self.object_name,
            "pose_matrix": np.asarray(self.transformation, dtype=float).tolist(),
            "translation_xyz_m": translation.tolist(),
            "quaternion_xyzw": rotation_matrix_to_quaternion(rotation).tolist(),
            "euler_rpy_rad": rotation_matrix_to_euler_rpy(rotation).tolist(),
            "euler_rpy_deg": np.degrees(rotation_matrix_to_euler_rpy(rotation)).tolist(),
            "fitness": float(self.fitness),
            "inlier_rmse": float(self.inlier_rmse),
            "selected_candidate": int(self.selected_candidate),
            "selected_stage": self.selected_stage,
            "num_scene_points": int(self.num_scene_points),
            "num_model_points": int(self.num_model_points),
            "intrinsics": {
                "fx": float(self.intrinsics[0]),
                "fy": float(self.intrinsics[1]),
                "cx": float(self.intrinsics[2]),
                "cy": float(self.intrinsics[3]),
            },
            "files": self.files,
        }


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def object_name_from_metadata(crop_dir):
    metadata_path = crop_dir / "metadata.json"
    if metadata_path.is_file():
        metadata = load_json(metadata_path)
        if metadata.get("label"):
            return metadata["label"]
        if metadata.get("target"):
            return metadata["target"]
    name = crop_dir.name
    return name.split("_")[-1] if "_" in name else name


def cloud_from_points(points):
    require_deps()
    points = np.asarray(points, dtype=np.float64).reshape(-1, points.shape[-1])[:, :3]
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0.0)
    points = points[valid]
    if len(points) == 0:
        raise RuntimeError("No valid scene XYZ points.")
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    return cloud


def preprocess_cloud(cloud, voxel_size, normal_radius, normal_max_nn, orient_to_camera, remove_outliers):
    require_deps()
    cloud = o3d.geometry.PointCloud(cloud)
    cloud.remove_non_finite_points()
    if voxel_size > 0.0:
        cloud = cloud.voxel_down_sample(float(voxel_size))
    if remove_outliers and len(cloud.points) >= 80:
        cloud, _ = cloud.remove_radius_outlier(nb_points=8, radius=max(float(voxel_size) * 3.0, 0.01))
    if len(cloud.points) == 0:
        raise RuntimeError("Point cloud became empty during preprocessing.")
    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=float(normal_radius), max_nn=int(normal_max_nn))
    )
    if orient_to_camera:
        cloud.orient_normals_towards_camera_location(np.zeros(3, dtype=np.float64))
    return cloud


def load_model_cloud(path):
    require_deps()
    cloud = o3d.io.read_point_cloud(str(path))
    if cloud.is_empty():
        raise RuntimeError(f"Could not read model point cloud: {path}")
    return cloud


def centroid(cloud):
    require_deps()
    points = np.asarray(cloud.points)
    if len(points) == 0:
        raise RuntimeError("Cannot compute centroid of empty cloud.")
    return points.mean(axis=0)


def pca_axes(cloud):
    require_deps()
    points = np.asarray(cloud.points, dtype=np.float64)
    points = points - points.mean(axis=0)
    covariance = np.cov(points.T)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    axes = vectors[:, order]
    if np.linalg.det(axes) < 0.0:
        axes[:, -1] *= -1.0
    return axes


def pca_rotation_candidates(model_cloud, scene_cloud):
    require_deps()
    model_axes = pca_axes(model_cloud)
    scene_axes = pca_axes(scene_cloud)
    candidates = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                signs = np.diag([sx, sy, sz])
                if np.linalg.det(signs) < 0.0:
                    continue
                rotation = scene_axes @ signs @ model_axes.T
                if np.linalg.det(rotation) > 0.5:
                    candidates.append(rotation)
    return candidates


def cube_rotation_candidates():
    require_deps()
    rotations = []
    eye = np.eye(3)
    for permutation in __import__("itertools").permutations(range(3)):
        base = eye[:, permutation]
        for signs in __import__("itertools").product((-1.0, 1.0), repeat=3):
            rotation = base * np.asarray(signs)
            if np.linalg.det(rotation) > 0.5:
                rotations.append(rotation)
    return rotations


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
    raise ValueError(f"Unknown ICP method: {method}")


def registration_icp(source, target, init, threshold, iterations, method):
    require_deps()
    return o3d.pipelines.registration.registration_icp(
        source,
        target,
        float(threshold),
        init,
        icp_estimation(method),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(iterations)),
    )


def is_better(candidate, current):
    if current is None:
        return True
    if candidate.fitness > current.fitness + 1e-9:
        return True
    if abs(candidate.fitness - current.fitness) <= 1e-9 and candidate.inlier_rmse < current.inlier_rmse:
        return True
    return False


def estimate_icp_pose(
    object_name,
    model_cloud,
    scene_cloud,
    candidate_mode="pca",
    coarse_threshold=0.08,
    refine_threshold=0.03,
    coarse_iterations=25,
    refine_iterations=60,
):
    require_deps()
    model_ctr = centroid(model_cloud)
    scene_ctr = centroid(scene_cloud)

    if candidate_mode == "pca":
        rotations = pca_rotation_candidates(model_cloud, scene_cloud)
    elif candidate_mode == "cube":
        rotations = cube_rotation_candidates()
    else:
        rotations = [np.eye(3)]

    best_index = 0
    best_coarse = None
    for index, rotation in enumerate(rotations):
        init = initial_transform(rotation, model_ctr, scene_ctr)
        result = registration_icp(model_cloud, scene_cloud, init, coarse_threshold, coarse_iterations, "point_to_point")
        if is_better(result, best_coarse):
            best_index = index
            best_coarse = result

    refined = registration_icp(
        model_cloud,
        scene_cloud,
        best_coarse.transformation,
        refine_threshold,
        refine_iterations,
        "point_to_plane",
    )
    if is_better(refined, best_coarse):
        return refined, best_index, "refined"
    return best_coarse, best_index, "coarse"


def estimate_intrinsics_from_crop(crop_dir, metadata):
    require_deps()
    cloud_path = crop_dir / "cloud.npy"
    if not cloud_path.is_file():
        raise FileNotFoundError(cloud_path)
    cloud = np.load(cloud_path)
    bbox = metadata.get("roi", {}).get("cloud_bbox_xyxy") or metadata.get("roi", {}).get("bbox_xyxy")
    if not bbox:
        raise RuntimeError("metadata.json has no roi cloud bbox.")
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
        raise RuntimeError("Not enough cloud points to estimate camera intrinsics.")
    fx, cx = np.linalg.lstsq(np.c_[xyz[:, 0] / xyz[:, 2], np.ones(len(xyz))], u, rcond=None)[0]
    fy, cy = np.linalg.lstsq(np.c_[xyz[:, 1] / xyz[:, 2], np.ones(len(xyz))], v, rcond=None)[0]
    return float(fx), float(fy), float(cx), float(cy)


def rotation_matrix_to_quaternion(rotation):
    require_deps()
    m = np.asarray(rotation, dtype=np.float64)
    trace = np.trace(m)
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    else:
        index = int(np.argmax(np.diag(m)))
        if index == 0:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            qw = (m[2, 1] - m[1, 2]) / s
            qx = 0.25 * s
            qy = (m[0, 1] + m[1, 0]) / s
            qz = (m[0, 2] + m[2, 0]) / s
        elif index == 1:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            qw = (m[0, 2] - m[2, 0]) / s
            qx = (m[0, 1] + m[1, 0]) / s
            qy = 0.25 * s
            qz = (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            qw = (m[1, 0] - m[0, 1]) / s
            qx = (m[0, 2] + m[2, 0]) / s
            qy = (m[1, 2] + m[2, 1]) / s
            qz = 0.25 * s
    quaternion = np.array([qx, qy, qz, qw], dtype=np.float64)
    return quaternion / np.linalg.norm(quaternion)


def rotation_matrix_to_euler_rpy(rotation):
    require_deps()
    r = np.asarray(rotation, dtype=np.float64)
    sy = math.sqrt(r[0, 0] * r[0, 0] + r[1, 0] * r[1, 0])
    singular = sy < 1e-6
    if not singular:
        roll = math.atan2(r[2, 1], r[2, 2])
        pitch = math.atan2(-r[2, 0], sy)
        yaw = math.atan2(r[1, 0], r[0, 0])
    else:
        roll = math.atan2(-r[1, 2], r[1, 1])
        pitch = math.atan2(-r[2, 0], sy)
        yaw = 0.0
    return np.array([roll, pitch, yaw], dtype=np.float64)


def project_points(points, intrinsics, crop_offset=(0.0, 0.0)):
    require_deps()
    fx, fy, cx, cy = intrinsics
    off_x, off_y = crop_offset
    projected = []
    for point in np.asarray(points, dtype=np.float64):
        x, y, z = point
        if z <= 1e-9:
            projected.append(None)
            continue
        projected.append((int(round(fx * x / z + cx - off_x)), int(round(fy * y / z + cy - off_y))))
    return projected


def draw_pose_axes(image_path, output_path, transformation, intrinsics, crop_offset=(0.0, 0.0), axis_length=0.08, text_lines=None):
    require_deps()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    transform = np.asarray(transformation, dtype=np.float64)
    origin = transform[:3, 3]
    rotation = transform[:3, :3]
    points = [
        origin,
        origin + rotation[:, 0] * axis_length,
        origin + rotation[:, 1] * axis_length,
        origin + rotation[:, 2] * axis_length,
    ]
    projected = project_points(points, intrinsics, crop_offset)
    start = projected[0]
    arrows = [((0, 0, 255), "X", projected[1]), ((0, 200, 0), "Y", projected[2]), ((255, 0, 0), "Z", projected[3])]
    for color, label, end in arrows:
        if start is not None and end is not None:
            cv2.arrowedLine(image, start, end, color, 3, cv2.LINE_AA, tipLength=0.18)
            cv2.circle(image, start, 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(image, label, end, cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    if text_lines:
        y = 24
        for line in text_lines:
            cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
            y += 22
    cv2.imwrite(str(output_path), image)
    return projected


def distance_stats(source_cloud, target_cloud, thresholds=(0.01, 0.02, 0.04)):
    require_deps()
    distances = np.asarray(source_cloud.compute_point_cloud_distance(target_cloud), dtype=np.float64)
    stats = {
        "mean_m": float(distances.mean()),
        "median_m": float(np.median(distances)),
        "p95_m": float(np.percentile(distances, 95.0)),
    }
    for threshold in thresholds:
        stats[f"within_{int(round(threshold * 1000.0))}mm"] = float(np.mean(distances <= threshold))
    return stats


def estimate_pose_for_crop(
    crop_dir,
    object_name=None,
    model_dir=DEFAULT_MODEL_DIR,
    scene_voxel_size=0.005,
    model_voxel_size=0.003,
    normal_radius=0.015,
    normal_max_nn=30,
    candidate_mode="pca",
    axis_length=0.08,
    write_aligned_clouds=True,
):
    require_deps()
    crop_dir = Path(crop_dir).expanduser().resolve()
    model_dir = Path(model_dir).expanduser().resolve()
    metadata_path = crop_dir / "metadata.json"
    metadata = load_json(metadata_path) if metadata_path.is_file() else {}
    object_name = object_name or object_name_from_metadata(crop_dir)

    scene_points_path = crop_dir / "foreground_points.npy"
    model_path = model_dir / f"{object_name}.ply"
    if not scene_points_path.is_file():
        raise FileNotFoundError(scene_points_path)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    scene_points = np.load(scene_points_path)
    raw_scene_points = int(scene_points.reshape(-1, scene_points.shape[-1]).shape[0])
    scene_cloud = preprocess_cloud(
        cloud_from_points(scene_points),
        voxel_size=scene_voxel_size,
        normal_radius=normal_radius,
        normal_max_nn=normal_max_nn,
        orient_to_camera=True,
        remove_outliers=True,
    )
    model_cloud = preprocess_cloud(
        load_model_cloud(model_path),
        voxel_size=model_voxel_size,
        normal_radius=normal_radius,
        normal_max_nn=normal_max_nn,
        orient_to_camera=False,
        remove_outliers=False,
    )

    result, candidate_index, selected_stage = estimate_icp_pose(
        object_name=object_name,
        model_cloud=model_cloud,
        scene_cloud=scene_cloud,
        candidate_mode=candidate_mode,
    )
    aligned_model = o3d.geometry.PointCloud(model_cloud)
    aligned_model.transform(result.transformation)
    intrinsics = estimate_intrinsics_from_crop(crop_dir, metadata)

    transform = np.asarray(result.transformation, dtype=np.float64)
    translation = transform[:3, 3]
    euler_deg = np.degrees(rotation_matrix_to_euler_rpy(transform[:3, :3]))
    text_lines = [
        f"{object_name} ICP 6D",
        f"t=({translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}) m",
        f"rpy=({euler_deg[0]:.1f}, {euler_deg[1]:.1f}, {euler_deg[2]:.1f}) deg",
        f"fitness={result.fitness:.3f} rmse={result.inlier_rmse:.4f}m",
    ]

    files = {}
    full_image = crop_dir / "annotated.png"
    if full_image.is_file():
        full_output = crop_dir / "pose_axes_full.png"
        draw_pose_axes(full_image, full_output, transform, intrinsics, crop_offset=(0.0, 0.0), axis_length=axis_length, text_lines=text_lines)
        files["pose_axes_full"] = str(full_output)

    crop_image = crop_dir / "rgb.png"
    if crop_image.is_file():
        bbox = metadata.get("roi", {}).get("bbox_xyxy") or [0.0, 0.0, 0.0, 0.0]
        crop_output = crop_dir / "pose_axes_crop.png"
        draw_pose_axes(crop_image, crop_output, transform, intrinsics, crop_offset=(float(bbox[0]), float(bbox[1])), axis_length=axis_length, text_lines=text_lines)
        files["pose_axes_crop"] = str(crop_output)

    if write_aligned_clouds:
        aligned_path = crop_dir / "icp_aligned_model.ply"
        scene_path = crop_dir / "icp_scene.ply"
        o3d.io.write_point_cloud(str(aligned_path), aligned_model)
        o3d.io.write_point_cloud(str(scene_path), scene_cloud)
        files["icp_aligned_model"] = str(aligned_path)
        files["icp_scene"] = str(scene_path)

    estimate = PoseEstimate(
        object_name=object_name,
        transformation=transform,
        fitness=result.fitness,
        inlier_rmse=result.inlier_rmse,
        selected_candidate=candidate_index,
        selected_stage=selected_stage,
        num_scene_points=len(scene_cloud.points),
        num_model_points=len(model_cloud.points),
        intrinsics=intrinsics,
        files=files,
    )
    data = estimate.to_dict()
    data["raw_scene_points"] = raw_scene_points
    data["model_path"] = str(model_path)
    data["scene_points_path"] = str(scene_points_path)
    data["model_to_scene_distance"] = distance_stats(aligned_model, scene_cloud)
    data["scene_to_model_distance"] = distance_stats(scene_cloud, aligned_model)
    result_path = crop_dir / "icp_pose_result.json"
    result_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    files["icp_pose_result"] = str(result_path)
    data["files"] = files

    if metadata_path.is_file():
        metadata["icp_pose"] = data
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return data
