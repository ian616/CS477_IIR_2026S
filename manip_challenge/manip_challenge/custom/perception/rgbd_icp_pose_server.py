#!/usr/bin/env python3
"""YOLO RGB-D + Open3D ICP 6D pose service."""

import json
import math
from pathlib import Path

import cv2
import numpy as np
import rclpy
import sensor_msgs_py.point_cloud2 as pc2
import std_msgs.msg
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from riro_srvs.srv import StringString
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import String

try:
    from . import icp_pose_estimator as icp
    from .rgbd_crop_server import (
        DEFAULT_MODEL_PATH,
        Detection,
        YoloDetector,
        extract_rgbd_roi,
        get_model,
        known_labels,
        parse_target_label,
    )
except ImportError:
    import icp_pose_estimator as icp
    from rgbd_crop_server import (
        DEFAULT_MODEL_PATH,
        Detection,
        YoloDetector,
        extract_rgbd_roi,
        get_model,
        known_labels,
        parse_target_label,
    )


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ICP_MODEL_DIR = SCRIPT_DIR / "icp_models"


def rotation_matrix_to_quaternion(rotation):
    r = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(r))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r[2, 1] - r[1, 2]) / s
        qy = (r[0, 2] - r[2, 0]) / s
        qz = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        qw = (r[2, 1] - r[1, 2]) / s
        qx = 0.25 * s
        qy = (r[0, 1] + r[1, 0]) / s
        qz = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        qw = (r[0, 2] - r[2, 0]) / s
        qx = (r[0, 1] + r[1, 0]) / s
        qy = 0.25 * s
        qz = (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        qw = (r[1, 0] - r[0, 1]) / s
        qx = (r[0, 2] + r[2, 0]) / s
        qy = (r[1, 2] + r[2, 1]) / s
        qz = 0.25 * s
    quat = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm > 1e-12:
        quat /= norm
    return quat


def estimate_intrinsics_from_cloud(cloud, image_shape, max_points=8000):
    if cloud is None or cloud.ndim != 3 or cloud.shape[2] < 3:
        raise ValueError("An organized cloud with shape HxWx3 is required for projection.")

    image_h, image_w = image_shape[:2]
    cloud_h, cloud_w = cloud.shape[:2]
    yy, xx = np.mgrid[0:cloud_h, 0:cloud_w]
    u = ((xx.astype(np.float64) + 0.5) * (image_w / float(cloud_w)) - 0.5).reshape(-1)
    v = ((yy.astype(np.float64) + 0.5) * (image_h / float(cloud_h)) - 0.5).reshape(-1)
    xyz = cloud.reshape(-1, cloud.shape[2])[:, :3].astype(np.float64)
    valid = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 1e-6)
    xyz = xyz[valid]
    u = u[valid]
    v = v[valid]
    if len(xyz) < 20:
        raise RuntimeError("Not enough valid cloud points to estimate camera intrinsics.")

    if len(xyz) > max_points:
        indices = np.linspace(0, len(xyz) - 1, max_points).astype(np.int64)
        xyz = xyz[indices]
        u = u[indices]
        v = v[indices]

    x_over_z = xyz[:, 0] / xyz[:, 2]
    y_over_z = xyz[:, 1] / xyz[:, 2]
    fx, cx = np.linalg.lstsq(np.c_[x_over_z, np.ones_like(x_over_z)], u, rcond=None)[0]
    fy, cy = np.linalg.lstsq(np.c_[y_over_z, np.ones_like(y_over_z)], v, rcond=None)[0]
    return float(fx), float(fy), float(cx), float(cy)


def project_points(points, fx, fy, cx, cy):
    projected = []
    for point in points:
        x, y, z = point
        if z <= 1e-9:
            projected.append(None)
        else:
            projected.append((int(round(fx * x / z + cx)), int(round(fy * y / z + cy))))
    return projected


def draw_arrow(image, start, end, color, label):
    if start is None or end is None:
        return
    cv2.arrowedLine(image, start, end, color, 3, cv2.LINE_AA, tipLength=0.18)
    cv2.circle(image, start, 4, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.putText(image, label, end, cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)


class IcpModelStore:
    def __init__(self, model_dir, model_voxel_size, normal_radius, normal_max_nn):
        icp.require_deps()
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.model_voxel_size = float(model_voxel_size)
        self.normal_radius = float(normal_radius)
        self.normal_max_nn = int(normal_max_nn)
        self._cache = {}

    def get(self, object_name):
        if object_name not in self._cache:
            model_path = self.model_dir / f"{object_name}.ply"
            if not model_path.is_file():
                raise FileNotFoundError(f"ICP model cloud not found: {model_path}")
            model_cloud = icp.load_model_cloud(model_path)
            model_cloud = icp.preprocess_cloud(
                model_cloud,
                voxel_size=self.model_voxel_size,
                normal_radius=self.normal_radius,
                normal_max_nn=self.normal_max_nn,
                remove_outliers=False,
            )
            self._cache[object_name] = model_cloud
        return self._cache[object_name]


class RgbdIcpPoseServiceNode(Node):
    def __init__(self):
        super().__init__("rgbd_icp_pose_service_node")

        self.declare_parameter("model_path", str(DEFAULT_MODEL_PATH))
        self.declare_parameter("icp_model_dir", str(DEFAULT_ICP_MODEL_DIR))
        self.declare_parameter("confidence", 0.35)
        self.declare_parameter("iou", 0.45)
        self.declare_parameter("target_label", "")
        self.declare_parameter("service_name", "detect_object_icp_pose")
        self.declare_parameter("image_topic", "/wrist_camera/wrist_camera/color/image_raw")
        self.declare_parameter("depth_topic", "/wrist_camera/wrist_camera/depth/color/image_raw")
        self.declare_parameter("points_topic", "/wrist_camera/wrist_camera/depth/color/points")
        self.declare_parameter("camera_frame", "wrist_camera_color_optical_frame")
        self.declare_parameter("bbox_padding_ratio", 0.0)
        self.declare_parameter("min_roi_points", 30)
        self.declare_parameter("component_mode", "center")
        self.declare_parameter("depth_mode", "center")
        self.declare_parameter("color_mask_mode", "auto")
        self.declare_parameter("scene_voxel_size", 0.005)
        self.declare_parameter("model_voxel_size", 0.003)
        self.declare_parameter("normal_radius", 0.015)
        self.declare_parameter("normal_max_nn", 30)
        self.declare_parameter("candidate_mode", "cube")
        self.declare_parameter("coarse_threshold", 0.20)
        self.declare_parameter("refine_threshold", 0.08)
        self.declare_parameter("coarse_iterations", 40)
        self.declare_parameter("refine_iterations", 60)
        self.declare_parameter("coarse_method", "point_to_point")
        self.declare_parameter("refine_method", "point_to_plane")
        self.declare_parameter("distance_thresholds", [0.01, 0.02, 0.04])
        self.declare_parameter("publish_pose_axes_image", True)
        self.declare_parameter("axis_length", 0.08)
        self.declare_parameter("roi_info_topic", "/icp_pose/roi_info")
        self.declare_parameter("scene_points_topic", "/icp_pose/scene_points")
        self.declare_parameter("aligned_model_topic", "/icp_pose/aligned_model_points")
        self.declare_parameter("pose_axes_image_topic", "/icp_pose/pose_axes_image")

        self.target_label = parse_target_label(self.get_parameter("target_label").value)
        self.camera_frame = self.get_parameter("camera_frame").value
        self.bbox_padding_ratio = float(self.get_parameter("bbox_padding_ratio").value)
        self.min_roi_points = int(self.get_parameter("min_roi_points").value)
        self.component_mode = str(self.get_parameter("component_mode").value)
        self.depth_mode = str(self.get_parameter("depth_mode").value)
        self.color_mask_mode = str(self.get_parameter("color_mask_mode").value)
        self.scene_voxel_size = float(self.get_parameter("scene_voxel_size").value)
        self.normal_radius = float(self.get_parameter("normal_radius").value)
        self.normal_max_nn = int(self.get_parameter("normal_max_nn").value)
        self.candidate_mode = str(self.get_parameter("candidate_mode").value)
        self.coarse_threshold = float(self.get_parameter("coarse_threshold").value)
        self.refine_threshold = float(self.get_parameter("refine_threshold").value)
        self.coarse_iterations = int(self.get_parameter("coarse_iterations").value)
        self.refine_iterations = int(self.get_parameter("refine_iterations").value)
        self.coarse_method = str(self.get_parameter("coarse_method").value)
        self.refine_method = str(self.get_parameter("refine_method").value)
        self.distance_thresholds = [float(v) for v in self.get_parameter("distance_thresholds").value]
        self.publish_pose_axes_image = bool(self.get_parameter("publish_pose_axes_image").value)
        self.axis_length = float(self.get_parameter("axis_length").value)

        self.detector = YoloDetector(
            model_path=Path(self.get_parameter("model_path").value).expanduser(),
            confidence=float(self.get_parameter("confidence").value),
            iou=float(self.get_parameter("iou").value),
        )
        self.model_store = IcpModelStore(
            model_dir=self.get_parameter("icp_model_dir").value,
            model_voxel_size=float(self.get_parameter("model_voxel_size").value),
            normal_radius=self.normal_radius,
            normal_max_nn=self.normal_max_nn,
        )

        qos_profile = QoSProfile(depth=10)
        qos_profile.reliability = ReliabilityPolicy.BEST_EFFORT
        self.bridge = CvBridge()
        self.latest_cv_img = None
        self.latest_depth_img = None
        self.latest_cloud = None

        self.create_subscription(Image, self.get_parameter("image_topic").value, self.image_callback, qos_profile)
        self.create_subscription(Image, self.get_parameter("depth_topic").value, self.depth_callback, qos_profile)
        self.create_subscription(PointCloud2, self.get_parameter("points_topic").value, self.points_callback, qos_profile)
        self.create_service(StringString, self.get_parameter("service_name").value, self.detect_callback)

        latched_qos = QoSProfile(depth=10)
        latched_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        latched_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.roi_info_pub = self.create_publisher(String, self.get_parameter("roi_info_topic").value, 10)
        self.scene_pub = self.create_publisher(PointCloud2, self.get_parameter("scene_points_topic").value, latched_qos)
        self.aligned_pub = self.create_publisher(PointCloud2, self.get_parameter("aligned_model_topic").value, latched_qos)
        self.pose_axes_image_pub = self.create_publisher(Image, self.get_parameter("pose_axes_image_topic").value, 10)

        self.get_logger().info(
            "RGB-D ICP pose service ready. "
            f"known_labels={', '.join(known_labels())}, icp_model_dir={self.model_store.model_dir}"
        )

    def image_callback(self, msg):
        self.latest_cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def depth_callback(self, msg):
        self.latest_depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    def points_callback(self, msg):
        raw_cloud = pc2.read_points_numpy(msg, field_names=("x", "y", "z"))
        self.latest_cloud = raw_cloud.reshape(msg.height, msg.width, 3) if msg.height > 1 else raw_cloud

    def detect_callback(self, request, response):
        target_label = parse_target_label(request.data) or self.target_label
        try:
            info = self.detect_pose(target_label, request.data)
        except Exception as exc:
            info = {"ok": False, "error": str(exc), "target": target_label, "known_labels": known_labels()}
            self.get_logger().warn(str(exc))
        response.data = json.dumps(info, sort_keys=True)
        msg = String()
        msg.data = response.data
        self.roi_info_pub.publish(msg)
        return response

    def detect_pose(self, target_label, request_text):
        if self.latest_cv_img is None or self.latest_depth_img is None or self.latest_cloud is None:
            raise RuntimeError("RGB image, depth image, or organized point cloud has not been received yet.")

        target_model = get_model(target_label)
        if target_model is None:
            raise RuntimeError(f"Unknown target='{target_label}'. Choose one of: {known_labels()}")

        detections = self.detector.detect(self.latest_cv_img)
        selected = self.detector.select(detections, target_model.name)
        if selected is None:
            seen = [detection.label for detection in detections]
            raise RuntimeError(f"No YOLO detection matched target='{target_model.name}'. seen={seen}")

        model = get_model(selected.label) or target_model
        padded = selected.padded(self.latest_cv_img.shape, self.bbox_padding_ratio)
        roi = extract_rgbd_roi(
            rgb_image=self.latest_cv_img,
            depth_image=self.latest_depth_img,
            cloud=self.latest_cloud,
            bbox_xyxy=padded.xyxy,
            image_shape=self.latest_cv_img.shape,
            depth_margin_m=model.depth_margin_m,
            min_points=self.min_roi_points,
            component_mode=self.component_mode,
            depth_mode=self.depth_mode,
            label=model.name,
            color_mask_mode=self.color_mask_mode,
        )

        scene_cloud = icp.cloud_from_points(roi.foreground_points)
        scene_cloud = icp.preprocess_cloud(
            scene_cloud,
            voxel_size=self.scene_voxel_size,
            normal_radius=self.normal_radius,
            normal_max_nn=self.normal_max_nn,
            remove_outliers=True,
        )
        model_cloud = self.model_store.get(model.name)
        result = icp.estimate_pose(
            object_name=model.name,
            model_cloud=model_cloud,
            scene_cloud=scene_cloud,
            candidate_mode=self.candidate_mode,
            coarse_threshold=self.coarse_threshold,
            refine_threshold=self.refine_threshold,
            coarse_iterations=self.coarse_iterations,
            refine_iterations=self.refine_iterations,
            coarse_method=self.coarse_method,
            refine_method=self.refine_method,
        )

        aligned_model = icp.transform_cloud(model_cloud, result.transformation)
        model_to_scene = icp.cloud_distance_stats(aligned_model, scene_cloud, self.distance_thresholds)
        scene_to_model = icp.cloud_distance_stats(scene_cloud, aligned_model, self.distance_thresholds)
        pose = np.asarray(result.transformation, dtype=np.float64)
        quat = rotation_matrix_to_quaternion(pose[:3, :3])

        self.publish_cloud(scene_cloud, self.scene_pub)
        self.publish_cloud(aligned_model, self.aligned_pub)
        projection_info = {}
        if self.publish_pose_axes_image:
            axes_image, projection_info = self.draw_pose_axes_image(pose, selected, padded)
            self.publish_pose_axes_image_msg(axes_image)

        info = {
            "ok": True,
            "stage": "icp_pose_estimated",
            "request": request_text,
            "target": target_model.name,
            "detected_label": selected.label,
            "frame_id": self.camera_frame,
            "detection": selected.to_dict(),
            "padded_detection": padded.to_dict(),
            "all_detections": [detection.to_dict() for detection in detections],
            "roi": roi.to_dict(),
            "pose_matrix": pose.tolist(),
            "translation_xyz_m": pose[:3, 3].astype(float).tolist(),
            "quaternion_xyzw": quat.astype(float).tolist(),
            "fitness": float(result.fitness),
            "inlier_rmse_m": float(result.inlier_rmse),
            "icp_stage": result.selected_stage,
            "candidate_index": int(result.candidate_index),
            "num_scene_points": int(result.num_scene_points),
            "num_model_points": int(result.num_model_points),
            "model_to_scene_distance": model_to_scene,
            "scene_to_model_distance": scene_to_model,
            "projection": projection_info,
        }
        self.get_logger().info(
            f"ICP pose {model.name}: t=({pose[0,3]:.3f}, {pose[1,3]:.3f}, {pose[2,3]:.3f}), "
            f"fitness={result.fitness:.3f}, rmse={result.inlier_rmse:.4f}, stage={result.selected_stage}"
        )
        return info

    def draw_pose_axes_image(self, pose, selected, padded):
        image = self.latest_cv_img.copy()
        fx, fy, cx, cy = estimate_intrinsics_from_cloud(self.latest_cloud, image.shape)
        origin = pose[:3, 3]
        rotation = pose[:3, :3]
        axis_length = self.axis_length
        points = [
            origin,
            origin + rotation[:, 0] * axis_length,
            origin + rotation[:, 1] * axis_length,
            origin + rotation[:, 2] * axis_length,
        ]
        projected = project_points(points, fx, fy, cx, cy)

        cv2.rectangle(image, (selected.left, selected.top), (selected.right, selected.bottom), (0, 255, 255), 2)
        if padded.xyxy != selected.xyxy:
            cv2.rectangle(image, (padded.left, padded.top), (padded.right, padded.bottom), (255, 128, 0), 2)

        start = projected[0]
        draw_arrow(image, start, projected[1], (0, 0, 255), "X")
        draw_arrow(image, start, projected[2], (0, 200, 0), "Y")
        draw_arrow(image, start, projected[3], (255, 0, 0), "Z")
        if start is not None:
            cv2.putText(
                image,
                f"t=({origin[0]:.3f}, {origin[1]:.3f}, {origin[2]:.3f})m",
                (max(0, start[0] + 8), max(20, start[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return image, {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "axis_length_m": axis_length,
            "projected_origin_xyz_axes": [list(point) if point is not None else None for point in projected],
        }

    def publish_pose_axes_image_msg(self, image):
        msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.camera_frame
        self.pose_axes_image_pub.publish(msg)

    def publish_cloud(self, cloud, publisher):
        points = np.asarray(cloud.points, dtype=np.float32)
        if len(points) == 0:
            return
        header = std_msgs.msg.Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.camera_frame
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg = PointCloud2(
            header=header,
            height=1,
            width=len(points),
            is_dense=True,
            is_bigendian=False,
            fields=fields,
            point_step=12,
            row_step=12 * len(points),
            data=points.tobytes(),
        )
        publisher.publish(msg)


def main():
    rclpy.init()
    node = RgbdIcpPoseServiceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
