#!/usr/bin/env python3
"""YOLO RGB-D crop service without LINEMOD pose estimation."""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
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
    from ultralytics import YOLO
except ImportError:
    YOLO = None


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "model" / "best.pt"
DEFAULT_SAVE_DIR = Path(
    "/home/lhs/CS477_IIR_2026S/manip_challenge/manip_challenge/custom/perception/rgbd_crops"
)


@dataclass(frozen=True)
class ObjectModel:
    name: str
    aliases: tuple
    dimensions_m: tuple
    depth_margin_m: float
    notes: str

    def to_dict(self):
        data = asdict(self)
        data["aliases"] = list(self.aliases)
        data["dimensions_m"] = list(self.dimensions_m)
        return data


OBJECT_MODELS = {
    "coke_can": ObjectModel("coke_can", ("coke can", "coke", "can"), (0.0670, 0.0670, 0.1239), 0.08, "Cylinder-like object."),
    "strawberry": ObjectModel("strawberry", ("strawberry",), (0.0452, 0.0453, 0.0457), 0.045, "Nearly spherical object."),
    "meat_can": ObjectModel("meat_can", ("meat can", "meat", "spam"), (0.1021, 0.0601, 0.0835), 0.085, "Box/can-like object."),
    "hammer": ObjectModel("hammer", ("hammer",), (0.1335, 0.3348, 0.0336), 0.12, "Elongated object."),
    "banana": ObjectModel("banana", ("banana",), (0.0819, 0.1432, 0.0685), 0.08, "Curved elongated object."),
}


def normalize_label(label):
    return str(label or "").lower().replace("_", " ").replace("-", " ").strip()


def get_model(label):
    normalized = normalize_label(label)
    for model in OBJECT_MODELS.values():
        names = (model.name, *model.aliases)
        if any(normalized == normalize_label(name) for name in names):
            return model
    for model in OBJECT_MODELS.values():
        names = (model.name, *model.aliases)
        if any(normalized in normalize_label(name) or normalize_label(name) in normalized for name in names):
            return model
    return None


def known_labels():
    return sorted(OBJECT_MODELS.keys())


def parse_target_label(text):
    normalized_text = normalize_label(text)
    if not normalized_text:
        return ""

    direct_model = get_model(normalized_text)
    if direct_model is not None:
        return direct_model.name

    for model in OBJECT_MODELS.values():
        for name in (model.name, *model.aliases):
            normalized_name = normalize_label(name)
            if normalized_name and normalized_name in normalized_text:
                return model.name
    return normalized_text


@dataclass
class Detection:
    label: str
    confidence: float
    left: int
    top: int
    right: int
    bottom: int

    @property
    def center(self):
        return (self.left + self.right) // 2, (self.top + self.bottom) // 2

    @property
    def xyxy(self):
        return (self.left, self.top, self.right, self.bottom)

    def padded(self, image_shape, ratio):
        image_h, image_w = image_shape[:2]
        width = self.right - self.left
        height = self.bottom - self.top
        pad_x = int(round(width * ratio))
        pad_y = int(round(height * ratio))
        return Detection(
            label=self.label,
            confidence=self.confidence,
            left=max(0, self.left - pad_x),
            top=max(0, self.top - pad_y),
            right=min(image_w - 1, self.right + pad_x),
            bottom=min(image_h - 1, self.bottom + pad_y),
        )

    def to_dict(self):
        return {
            "label": self.label,
            "confidence": self.confidence,
            "bbox_xyxy": [self.left, self.top, self.right, self.bottom],
            "center_xy": list(self.center),
        }


class YoloDetector:
    def __init__(self, model_path=DEFAULT_MODEL_PATH, confidence=0.35, iou=0.45):
        model_path = Path(model_path).expanduser()
        if not model_path.is_file():
            raise FileNotFoundError(f"YOLO model file not found: {model_path}")
        if YOLO is None:
            raise ImportError("ultralytics is required. Install it with: pip install ultralytics")

        self.model_path = model_path
        self.confidence = confidence
        self.iou = iou
        self.model = YOLO(str(model_path))
        self.class_names = self.model.names

    def detect(self, image):
        result = self.model.predict(source=image, conf=self.confidence, iou=self.iou, verbose=False)[0]
        detections = []
        if result.boxes is None:
            return detections

        image_h, image_w = image.shape[:2]
        for box in result.boxes:
            left, top, right, bottom = box.xyxy[0].detach().cpu().numpy().astype(int).tolist()
            left = int(np.clip(left, 0, image_w - 1))
            right = int(np.clip(right, 0, image_w - 1))
            top = int(np.clip(top, 0, image_h - 1))
            bottom = int(np.clip(bottom, 0, image_h - 1))
            if right <= left or bottom <= top:
                continue

            cls_id = int(box.cls[0].detach().cpu().item())
            label = str(self.class_names.get(cls_id, cls_id))
            confidence = float(box.conf[0].detach().cpu().item())
            detections.append(Detection(label, confidence, left, top, right, bottom))

        detections.sort(key=lambda det: det.confidence, reverse=True)
        return detections

    def select(self, detections, target_label):
        if not detections:
            return None
        if not target_label:
            return detections[0]

        normalized_target = normalize_label(target_label)
        for detection in detections:
            if normalize_label(detection.label) == normalized_target:
                return detection
        for detection in detections:
            normalized_detection = normalize_label(detection.label)
            if normalized_target in normalized_detection or normalized_detection in normalized_target:
                return detection
        return None

    @staticmethod
    def draw(image, detections, selected=None, padded=None):
        vis_img = image.copy()
        selected_xyxy = selected.xyxy if selected is not None else None
        for detection in detections:
            color = (0, 255, 255) if detection.xyxy == selected_xyxy else (0, 255, 0)
            cv2.rectangle(vis_img, (detection.left, detection.top), (detection.right, detection.bottom), color, 2)
            cv2.circle(vis_img, detection.center, 4, (0, 0, 255), -1)
            text_y = max(detection.top - 8, 18)
            cv2.putText(vis_img, f"{detection.label} {detection.confidence:.2f}", (detection.left, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        if padded is not None:
            cv2.rectangle(vis_img, (padded.left, padded.top), (padded.right, padded.bottom), (255, 128, 0), 2)
        return vis_img


@dataclass
class RgbdRoi:
    rgb_crop: np.ndarray
    depth_crop: np.ndarray
    cloud_crop: np.ndarray
    object_mask: np.ndarray
    foreground_points: np.ndarray
    centroid: np.ndarray
    bbox_xyxy: tuple
    cloud_bbox_xyxy: tuple
    stats: dict

    def to_dict(self):
        return {
            "bbox_xyxy": list(self.bbox_xyxy),
            "cloud_bbox_xyxy": list(self.cloud_bbox_xyxy),
            "crop_shape_hw": [int(self.rgb_crop.shape[0]), int(self.rgb_crop.shape[1])],
            "mask_nonzero": int(np.count_nonzero(self.object_mask)),
            "num_foreground_points": int(len(self.foreground_points)),
            "centroid_xyz": self.centroid.astype(float).tolist(),
            "stats": self.stats,
        }


def largest_component(mask):
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    largest_label = int(np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1)
    return (labels == largest_label).astype(np.uint8)


def extract_rgbd_roi(rgb_image, depth_image, cloud, bbox_xyxy, image_shape, depth_margin_m, min_points=30):
    if cloud is None or cloud.ndim != 3:
        raise ValueError("An organized point cloud with shape (H, W, 3) is required.")

    image_h, image_w = image_shape[:2]
    cloud_h, cloud_w = cloud.shape[:2]
    left, top, right, bottom = bbox_xyxy
    left = int(np.clip(left, 0, image_w - 1))
    right = int(np.clip(right, 0, image_w - 1))
    top = int(np.clip(top, 0, image_h - 1))
    bottom = int(np.clip(bottom, 0, image_h - 1))
    if right <= left or bottom <= top:
        raise ValueError(f"Invalid image bbox: {bbox_xyxy}")

    scale_x = cloud_w / float(image_w)
    scale_y = cloud_h / float(image_h)
    cx1 = int(np.clip(round(left * scale_x), 0, cloud_w - 1))
    cx2 = int(np.clip(round(right * scale_x), 0, cloud_w - 1))
    cy1 = int(np.clip(round(top * scale_y), 0, cloud_h - 1))
    cy2 = int(np.clip(round(bottom * scale_y), 0, cloud_h - 1))
    if cx2 <= cx1 or cy2 <= cy1:
        raise ValueError(f"Invalid cloud bbox after scaling: {(cx1, cy1, cx2, cy2)}")

    rgb_crop = rgb_image[top:bottom, left:right].copy()
    depth_crop = depth_image[top:bottom, left:right].copy()
    cloud_crop = cloud[cy1:cy2, cx1:cx2].copy()

    valid_mask = np.isfinite(cloud_crop).all(axis=2) & (cloud_crop[:, :, 2] > 0.0)
    if not np.any(valid_mask):
        raise ValueError("No valid depth points inside ROI.")

    valid_z = cloud_crop[:, :, 2][valid_mask]
    z_min = float(np.min(valid_z))
    z_far = float(np.percentile(valid_z, 95.0))
    foreground_limit = min(z_min + float(depth_margin_m), z_far)
    foreground_mask = valid_mask & (cloud_crop[:, :, 2] <= foreground_limit)
    foreground_mask = largest_component(foreground_mask.astype(np.uint8)).astype(bool)
    foreground_points = cloud_crop[foreground_mask]

    if len(foreground_points) < min_points:
        relaxed_limit = min(z_min + float(depth_margin_m) * 1.5, z_far)
        foreground_mask = valid_mask & (cloud_crop[:, :, 2] <= relaxed_limit)
        foreground_mask = largest_component(foreground_mask.astype(np.uint8)).astype(bool)
        foreground_points = cloud_crop[foreground_mask]

    if len(foreground_points) < min_points:
        raise ValueError(f"Only {len(foreground_points)} foreground points found; need at least {min_points}.")

    return RgbdRoi(
        rgb_crop=rgb_crop,
        depth_crop=depth_crop,
        cloud_crop=cloud_crop,
        object_mask=foreground_mask.astype(np.uint8) * 255,
        foreground_points=foreground_points[:, :3],
        centroid=np.median(foreground_points[:, :3], axis=0),
        bbox_xyxy=(left, top, right, bottom),
        cloud_bbox_xyxy=(cx1, cy1, cx2, cy2),
        stats={
            "z_min": z_min,
            "z_percentile_95": z_far,
            "foreground_z_limit": foreground_limit,
            "valid_points": int(np.count_nonzero(valid_mask)),
            "foreground_points": int(len(foreground_points)),
        },
    )


def make_depth_visual(depth_crop):
    if depth_crop is None or depth_crop.size == 0:
        return None

    depth = depth_crop[:, :, 0] if depth_crop.ndim == 3 else depth_crop
    depth = depth.astype(np.float32, copy=False)
    finite = np.isfinite(depth)
    if not np.any(finite):
        return np.zeros((*depth.shape[:2], 3), dtype=np.uint8)

    finite_depth = depth[finite]
    low = float(np.percentile(finite_depth, 2.0))
    high = float(np.percentile(finite_depth, 98.0))
    if high <= low:
        high = low + 1.0

    normalized = np.zeros(depth.shape[:2], dtype=np.uint8)
    clipped = np.clip(depth, low, high)
    normalized[finite] = ((clipped[finite] - low) * 255.0 / (high - low)).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_JET)


class RgbdCropServiceNode(Node):
    def __init__(self):
        super().__init__("rgbd_crop_service_node")

        self.declare_parameter("model_path", str(DEFAULT_MODEL_PATH))
        self.declare_parameter("confidence", 0.35)
        self.declare_parameter("iou", 0.45)
        self.declare_parameter("display", True)
        self.declare_parameter("display_hz", 5.0)
        self.declare_parameter("target_label", "")
        self.declare_parameter("service_name", "detect_object_rgbd_crop")
        self.declare_parameter("image_topic", "/wrist_camera/wrist_camera/color/image_raw")
        self.declare_parameter("depth_topic", "/wrist_camera/wrist_camera/depth/color/image_raw")
        self.declare_parameter("points_topic", "/wrist_camera/wrist_camera/depth/color/points")
        self.declare_parameter("camera_frame", "wrist_camera_color_optical_frame")
        self.declare_parameter("bbox_padding_ratio", 0.15)
        self.declare_parameter("min_roi_points", 30)
        self.declare_parameter("save_dir", str(DEFAULT_SAVE_DIR))
        self.declare_parameter("annotated_image_topic", "/yolov11/rgbd_crop_detection_image")
        self.declare_parameter("roi_mask_topic", "/yolov11/rgbd_crop_mask")
        self.declare_parameter("roi_info_topic", "/yolov11/rgbd_crop_info")
        self.declare_parameter("roi_points_topic", "/rgbd_crop_filtered_points")

        self.target_label = parse_target_label(self.get_parameter("target_label").value)
        self.display = bool(self.get_parameter("display").value)
        self.camera_frame = self.get_parameter("camera_frame").value
        self.bbox_padding_ratio = float(self.get_parameter("bbox_padding_ratio").value)
        self.min_roi_points = int(self.get_parameter("min_roi_points").value)
        self.save_dir = Path(self.get_parameter("save_dir").value).expanduser()
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.detector = YoloDetector(
            model_path=Path(self.get_parameter("model_path").value).expanduser(),
            confidence=float(self.get_parameter("confidence").value),
            iou=float(self.get_parameter("iou").value),
        )

        qos_profile = QoSProfile(depth=10)
        qos_profile.reliability = ReliabilityPolicy.BEST_EFFORT

        self.bridge = CvBridge()
        self.latest_cv_img = None
        self.latest_depth_img = None
        self.latest_cloud = None
        self.annotated_img = None
        self.mask_img = None

        self.create_subscription(Image, self.get_parameter("image_topic").value, self.image_callback, qos_profile)
        self.create_subscription(Image, self.get_parameter("depth_topic").value, self.depth_callback, qos_profile)
        self.create_subscription(PointCloud2, self.get_parameter("points_topic").value, self.points_callback, qos_profile)

        self.create_service(StringString, self.get_parameter("service_name").value, self.detect_callback)

        latched_qos = QoSProfile(depth=10)
        latched_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        latched_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.roi_pub = self.create_publisher(PointCloud2, self.get_parameter("roi_points_topic").value, latched_qos)
        self.roi_info_pub = self.create_publisher(String, self.get_parameter("roi_info_topic").value, 10)
        self.annotated_image_pub = self.create_publisher(Image, self.get_parameter("annotated_image_topic").value, 10)
        self.roi_mask_pub = self.create_publisher(Image, self.get_parameter("roi_mask_topic").value, 10)

        display_hz = max(float(self.get_parameter("display_hz").value), 0.1)
        self.timer = self.create_timer(1.0 / display_hz, self.timer_callback)
        self.get_logger().info(
            "RGB-D crop service ready. "
            f"model={self.detector.model_path}, save_dir={self.save_dir}, known_labels={', '.join(known_labels())}"
        )

    def image_callback(self, msg):
        self.latest_cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def depth_callback(self, msg):
        self.latest_depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    def points_callback(self, msg):
        raw_cloud = pc2.read_points_numpy(msg, field_names=("x", "y", "z"))
        self.latest_cloud = raw_cloud.reshape(msg.height, msg.width, 3) if msg.height > 1 else raw_cloud

    def timer_callback(self):
        if self.annotated_img is not None:
            self.publish_annotated_image(self.annotated_img)
            if self.display:
                cv2.imshow("RGB-D Crop Detection", self.annotated_img)
                cv2.waitKey(1)
        if self.mask_img is not None:
            self.publish_roi_mask(self.mask_img)

    def detect_callback(self, request, response):
        target_label = parse_target_label(request.data) or self.target_label
        try:
            info = self.detect_and_save(target_label, request.data)
        except Exception as exc:
            info = {"ok": False, "error": str(exc), "target": target_label, "known_labels": known_labels()}
            self.get_logger().warn(str(exc))

        response.data = json.dumps(info, sort_keys=True)
        msg = String()
        msg.data = response.data
        self.roi_info_pub.publish(msg)
        return response

    def detect_and_save(self, target_label, request_text):
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
        )

        request_dir = self.make_request_dir(model.name)
        saved_files = self.save_rgbd_crop(request_dir, model.name, selected, padded, detections, roi, request_text)

        self.publish_roi_cloud(roi.foreground_points)
        self.mask_img = roi.object_mask
        self.annotated_img = self.draw_debug_image(detections, selected, padded, roi)

        info = {
            "ok": True,
            "stage": "rgbd_crop_saved",
            "message": "YOLO bbox, object location, and RGB-D crop were saved. LINEMOD was not run.",
            "request": request_text,
            "target": target_model.name,
            "detected_label": selected.label,
            "frame_id": self.camera_frame,
            "detection": selected.to_dict(),
            "padded_detection": padded.to_dict(),
            "all_detections": [detection.to_dict() for detection in detections],
            "model": model.to_dict(),
            "roi": roi.to_dict(),
            "location_xyz_m": roi.centroid.astype(float).tolist(),
            "save_dir": str(request_dir),
            "files": saved_files,
        }

        self.get_logger().info(
            f"Saved RGB-D crop for {model.name}: bbox={selected.xyxy}, "
            f"centroid=({roi.centroid[0]:.3f}, {roi.centroid[1]:.3f}, {roi.centroid[2]:.3f}), dir={request_dir}"
        )
        return info

    def make_request_dir(self, label):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        request_dir = self.save_dir / f"{timestamp}_{label}"
        request_dir.mkdir(parents=True, exist_ok=False)
        return request_dir

    def save_rgbd_crop(self, request_dir, label, selected, padded, detections, roi, request_text):
        rgb_path = request_dir / "rgb.png"
        depth_npy_path = request_dir / "depth.npy"
        depth_vis_path = request_dir / "depth_vis.png"
        mask_path = request_dir / "mask.png"
        mask_rgb_path = request_dir / "mask_on_rgb_size.png"
        cloud_path = request_dir / "cloud.npy"
        foreground_path = request_dir / "foreground_points.npy"
        annotated_path = request_dir / "annotated.png"
        metadata_path = request_dir / "metadata.json"

        cv2.imwrite(str(rgb_path), roi.rgb_crop)
        np.save(depth_npy_path, roi.depth_crop)
        depth_vis = make_depth_visual(roi.depth_crop)
        if depth_vis is not None:
            cv2.imwrite(str(depth_vis_path), depth_vis)
        cv2.imwrite(str(mask_path), roi.object_mask)
        mask_rgb = cv2.resize(roi.object_mask, (roi.rgb_crop.shape[1], roi.rgb_crop.shape[0]), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(mask_rgb_path), mask_rgb)
        np.save(cloud_path, roi.cloud_crop)
        np.save(foreground_path, roi.foreground_points)

        annotated = self.draw_debug_image(detections, selected, padded, roi)
        cv2.imwrite(str(annotated_path), annotated)

        metadata = {
            "label": label,
            "request": request_text,
            "frame_id": self.camera_frame,
            "detection": selected.to_dict(),
            "padded_detection": padded.to_dict(),
            "all_detections": [detection.to_dict() for detection in detections],
            "roi": roi.to_dict(),
            "location_xyz_m": roi.centroid.astype(float).tolist(),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

        files = {
            "rgb": str(rgb_path),
            "depth_npy": str(depth_npy_path),
            "mask": str(mask_path),
            "mask_rgb_size": str(mask_rgb_path),
            "cloud_npy": str(cloud_path),
            "foreground_points_npy": str(foreground_path),
            "annotated": str(annotated_path),
            "metadata": str(metadata_path),
        }
        if depth_vis is not None:
            files["depth_visualization"] = str(depth_vis_path)
        return files

    def draw_debug_image(self, detections, selected, padded, roi):
        image = self.detector.draw(self.latest_cv_img, detections, selected=selected, padded=padded)
        x1, y1, x2, y2 = roi.bbox_xyxy
        if x2 > x1 and y2 > y1:
            mask = cv2.resize(roi.object_mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
            overlay = np.zeros((y2 - y1, x2 - x1, 3), dtype=np.uint8)
            overlay[:, :, 1] = mask
            region = image[y1:y2, x1:x2]
            if region.size and overlay.shape == region.shape:
                image[y1:y2, x1:x2] = cv2.addWeighted(region, 0.75, overlay, 0.25, 0.0)
        cv2.putText(image, "RGB-D crop saved", (max(0, x1), min(image.shape[0] - 8, y2 + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 128, 0), 2)
        return image

    def publish_annotated_image(self, image):
        msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.camera_frame
        self.annotated_image_pub.publish(msg)

    def publish_roi_mask(self, mask):
        msg = self.bridge.cv2_to_imgmsg(mask, encoding="mono8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.camera_frame
        self.roi_mask_pub.publish(msg)

    def publish_roi_cloud(self, valid_points):
        if valid_points is None or len(valid_points) == 0:
            return

        header = std_msgs.msg.Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.camera_frame
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        points_float32 = valid_points[:, :3].astype(np.float32)
        msg = PointCloud2(
            header=header,
            height=1,
            width=len(points_float32),
            is_dense=True,
            is_bigendian=False,
            fields=fields,
            point_step=12,
            row_step=12 * len(points_float32),
            data=points_float32.tobytes(),
        )
        self.roi_pub.publish(msg)


def main():
    rclpy.init()
    node = RgbdCropServiceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
