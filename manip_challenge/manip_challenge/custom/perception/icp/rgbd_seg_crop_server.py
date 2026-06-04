#!/usr/bin/env python3
"""YOLO segmentation RGB-D crop service.

This server uses polygon masks from a YOLO segmentation model to extract only
the requested object's RGB-D pixels and organized point-cloud samples.
"""

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import rclpy
import sensor_msgs_py.point_cloud2 as pc2
import std_msgs.msg
from cv_bridge import CvBridge, CvBridgeError
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from riro_srvs.srv import StringString
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import String

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    from ..rgbd_crop_server import get_model, known_labels, make_depth_visual, parse_target_label, point_bounds
    from ..pca_bbox_visualization import compute_all_detection_pca_bboxes, draw_pca_bbox_debug
except ImportError:
    import sys

    perception_dir = Path(__file__).resolve().parent.parent
    if str(perception_dir) not in sys.path:
        sys.path.insert(0, str(perception_dir))
    from rgbd_crop_server import get_model, known_labels, make_depth_visual, parse_target_label, point_bounds
    from pca_bbox_visualization import compute_all_detection_pca_bboxes, draw_pca_bbox_debug


SCRIPT_DIR = Path(__file__).resolve().parent
PERCEPTION_DIR = SCRIPT_DIR.parent
CUSTOM_DIR = PERCEPTION_DIR.parent
DEFAULT_MODEL_PATH = CUSTOM_DIR / "best.pt"
DEFAULT_SAVE_DIR = Path(
    "/home/lhs/CS477_IIR_2026S/manip_challenge/manip_challenge/custom/perception/icp/results"
)
SCENE_REQUEST_TARGETS = {"__scene__", "scene", "all", "*"}


@dataclass
class SegDetection:
    label: str
    confidence: float
    left: int
    top: int
    right: int
    bottom: int
    mask: np.ndarray
    polygon_xy: list

    @property
    def center(self):
        return (self.left + self.right) // 2, (self.top + self.bottom) // 2

    @property
    def xyxy(self):
        return (self.left, self.top, self.right, self.bottom)

    def padded_xyxy(self, image_shape, ratio):
        image_h, image_w = image_shape[:2]
        width = self.right - self.left
        height = self.bottom - self.top
        pad_x = int(round(width * ratio))
        pad_y = int(round(height * ratio))
        return (
            max(0, self.left - pad_x),
            max(0, self.top - pad_y),
            min(image_w - 1, self.right + pad_x),
            min(image_h - 1, self.bottom + pad_y),
        )

    def to_dict(self):
        return {
            "label": self.label,
            "confidence": float(self.confidence),
            "bbox_xyxy": [int(self.left), int(self.top), int(self.right), int(self.bottom)],
            "center_xy": [int(self.center[0]), int(self.center[1])],
            "mask_pixels": int(np.count_nonzero(self.mask)),
            "polygon_points": int(len(self.polygon_xy)),
        }


@dataclass
class SegRgbdRoi:
    rgb_crop: np.ndarray
    rgb_masked_crop: np.ndarray
    depth_crop: np.ndarray
    cloud_crop: np.ndarray
    mask_full: np.ndarray
    mask_raw_full: np.ndarray
    mask_rgb_crop: np.ndarray
    mask_raw_rgb_crop: np.ndarray
    mask_depth_crop: np.ndarray
    mask_cloud_crop: np.ndarray
    foreground_points: np.ndarray
    centroid: np.ndarray
    bbox_xyxy: tuple
    depth_bbox_xyxy: tuple
    cloud_bbox_xyxy: tuple
    stats: dict

    def to_dict(self):
        return {
            "bbox_xyxy": [int(v) for v in self.bbox_xyxy],
            "depth_bbox_xyxy": [int(v) for v in self.depth_bbox_xyxy],
            "cloud_bbox_xyxy": [int(v) for v in self.cloud_bbox_xyxy],
            "crop_shape_hw": [int(self.rgb_crop.shape[0]), int(self.rgb_crop.shape[1])],
            "mask_pixels_rgb_crop": int(np.count_nonzero(self.mask_rgb_crop)),
            "mask_pixels_cloud_crop": int(np.count_nonzero(self.mask_cloud_crop)),
            "raw_mask_pixels_rgb_crop": int(np.count_nonzero(self.mask_raw_rgb_crop)),
            "num_foreground_points": int(len(self.foreground_points)),
            "centroid_xyz": self.centroid.astype(float).tolist(),
            "stats": self.stats,
        }


def normalize_label(label):
    return str(label or "").lower().replace("_", " ").replace("-", " ").strip()


def scale_bbox(bbox_xyxy, src_shape, dst_shape):
    src_h, src_w = src_shape[:2]
    dst_h, dst_w = dst_shape[:2]
    left, top, right, bottom = bbox_xyxy
    scale_x = dst_w / float(src_w)
    scale_y = dst_h / float(src_h)
    x1 = int(np.clip(round(left * scale_x), 0, dst_w - 1))
    x2 = int(np.clip(round(right * scale_x), 0, dst_w - 1))
    y1 = int(np.clip(round(top * scale_y), 0, dst_h - 1))
    y2 = int(np.clip(round(bottom * scale_y), 0, dst_h - 1))
    if x2 <= x1:
        x2 = min(dst_w, x1 + 1)
    if y2 <= y1:
        y2 = min(dst_h, y1 + 1)
    return x1, y1, x2, y2


def centered_crop_xyxy(shape, crop_ratio):
    ratio = float(crop_ratio)
    if ratio <= 0.0 or ratio >= 1.0:
        return 0, 0, int(shape[1]), int(shape[0])
    h, w = shape[:2]
    crop_w = max(1, int(round(w * ratio)))
    crop_h = max(1, int(round(h * ratio)))
    left = max(0, (w - crop_w) // 2)
    top = max(0, (h - crop_h) // 2)
    return left, top, min(w, left + crop_w), min(h, top + crop_h)


def crop_array_xyxy(array, bbox_xyxy):
    left, top, right, bottom = bbox_xyxy
    return array[top:bottom, left:right].copy()


def bbox_from_mask(mask):
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise ValueError("Segmentation mask is empty.")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def clean_mask(mask, min_area=20):
    mask = (mask > 0).astype(np.uint8)
    if int(np.count_nonzero(mask)) < min_area:
        return mask * 255
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return (mask > 0).astype(np.uint8) * 255


def depth_to_uint16_mm(depth_crop):
    depth = depth_crop[:, :, 0] if depth_crop.ndim == 3 else depth_crop
    if depth.dtype == np.uint16:
        return depth
    depth = depth.astype(np.float32, copy=False)
    finite = np.isfinite(depth) & (depth > 0.0)
    output = np.zeros(depth.shape[:2], dtype=np.uint16)
    if not np.any(finite):
        return output
    # Floating ROS depth images are normally meters. Values that are already
    # millimeter-like are left in their original scale.
    finite_median = float(np.median(depth[finite]))
    scaled = depth if finite_median > 20.0 else depth * 1000.0
    output[finite] = np.clip(scaled[finite], 0.0, np.iinfo(np.uint16).max).astype(np.uint16)
    return output


def image_msg_to_cv2_fallback(msg, desired_encoding=None):
    encoding = str(msg.encoding or "").lower()
    if msg.width == 0 or msg.height == 0 or len(msg.data) == 0:
        raise ValueError("empty image message")

    specs = {
        "rgb8": (np.uint8, 3),
        "bgr8": (np.uint8, 3),
        "rgba8": (np.uint8, 4),
        "bgra8": (np.uint8, 4),
        "mono8": (np.uint8, 1),
        "8uc1": (np.uint8, 1),
        "8uc3": (np.uint8, 3),
        "8uc4": (np.uint8, 4),
        "mono16": (np.uint16, 1),
        "16uc1": (np.uint16, 1),
        "32fc1": (np.float32, 1),
    }
    if encoding not in specs:
        raise ValueError(f"unsupported image encoding '{msg.encoding}'")

    dtype, channels = specs[encoding]
    itemsize = np.dtype(dtype).itemsize
    min_step = int(msg.width) * channels * itemsize
    row_step = int(msg.step) if int(msg.step) > 0 else min_step
    required = row_step * int(msg.height)
    if len(msg.data) < required:
        raise ValueError(
            f"image data too short for encoding={msg.encoding}, "
            f"got={len(msg.data)}, need={required}, width={msg.width}, height={msg.height}, step={msg.step}"
        )

    flat = np.frombuffer(msg.data, dtype=dtype, count=required // itemsize)
    rows = flat.reshape(int(msg.height), row_step // itemsize)
    pixels = rows[:, : min_step // itemsize]
    if channels == 1:
        image = pixels.reshape(int(msg.height), int(msg.width)).copy()
    else:
        image = pixels.reshape(int(msg.height), int(msg.width), channels).copy()

    if desired_encoding == "bgr8":
        if encoding in ("rgb8", "8uc3"):
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if encoding == "bgr8":
            return image
        if encoding == "rgba8":
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        if encoding == "bgra8":
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if channels == 1 and image.dtype == np.uint8:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        raise ValueError(f"cannot convert encoding '{msg.encoding}' to bgr8")

    return image


def estimate_mask_center_depth(cloud_crop, valid_mask, window_ratio=0.25):
    h, w = valid_mask.shape[:2]
    half_h = max(2, int(round(h * window_ratio * 0.5)))
    half_w = max(2, int(round(w * window_ratio * 0.5)))
    cy = h // 2
    cx = w // 2
    y1 = max(0, cy - half_h)
    y2 = min(h, cy + half_h + 1)
    x1 = max(0, cx - half_w)
    x2 = min(w, cx + half_w + 1)
    center_valid = valid_mask[y1:y2, x1:x2]
    center_z = cloud_crop[y1:y2, x1:x2, 2][center_valid]
    if center_z.size >= 10:
        return float(np.median(center_z)), int(center_z.size), "center_window"
    all_z = cloud_crop[:, :, 2][valid_mask]
    return float(np.median(all_z)), int(all_z.size), "mask_median"


def polygon_to_mask(polygon_xy, image_shape):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    if len(polygon_xy) < 3:
        return mask
    polygon = np.asarray(polygon_xy, dtype=np.float32)
    polygon[:, 0] = np.clip(polygon[:, 0], 0, image_shape[1] - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, image_shape[0] - 1)
    cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 255)
    return mask


def mask_data_to_full(mask_data, image_shape):
    mask = mask_data.detach().cpu().numpy().astype(np.float32)
    mask = cv2.resize(mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_LINEAR)
    return (mask >= 0.5).astype(np.uint8) * 255


class YoloSegDetector:
    def __init__(self, model_path=DEFAULT_MODEL_PATH, confidence=0.35, iou=0.45, device=""):
        model_path = Path(model_path).expanduser()
        if not model_path.is_file():
            raise FileNotFoundError(f"YOLO segmentation model file not found: {model_path}")
        if YOLO is None:
            raise ImportError("ultralytics is required. Install it with: pip install ultralytics")

        self.model_path = model_path
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.device = str(device or "").strip()
        self.model = YOLO(str(model_path))
        self.class_names = self.model.names

    def detect(self, image):
        predict_kwargs = {
            "source": image,
            "conf": self.confidence,
            "iou": self.iou,
            "verbose": False,
        }
        if self.device:
            predict_kwargs["device"] = self.device
        result = self.model.predict(**predict_kwargs)[0]
        if result.boxes is None or result.masks is None:
            return []

        image_h, image_w = image.shape[:2]
        polygons = result.masks.xy if result.masks.xy is not None else []
        detections = []
        for index, box in enumerate(result.boxes):
            cls_id = int(box.cls[0].detach().cpu().item())
            label = str(self.class_names.get(cls_id, cls_id))
            confidence = float(box.conf[0].detach().cpu().item())

            polygon = []
            if index < len(polygons) and len(polygons[index]) >= 3:
                polygon = np.asarray(polygons[index], dtype=np.float32).tolist()

            if result.masks.data is not None and index < len(result.masks.data):
                mask = mask_data_to_full(result.masks.data[index], image.shape)
            elif polygon:
                mask = polygon_to_mask(polygon, image.shape)
            else:
                continue

            mask = clean_mask(mask)
            if int(np.count_nonzero(mask)) == 0:
                continue

            left, top, right, bottom = bbox_from_mask(mask)
            left = int(np.clip(left, 0, image_w - 1))
            right = int(np.clip(right, 0, image_w))
            top = int(np.clip(top, 0, image_h - 1))
            bottom = int(np.clip(bottom, 0, image_h))
            if right <= left or bottom <= top:
                continue

            detections.append(SegDetection(label, confidence, left, top, right, bottom, mask, polygon))

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

    def matches(self, detections, target_label):
        if not detections:
            return []
        if not target_label:
            return list(detections)

        normalized_target = normalize_label(target_label)
        exact = [
            detection
            for detection in detections
            if normalize_label(detection.label) == normalized_target
        ]
        if exact:
            return exact
        return [
            detection
            for detection in detections
            if normalized_target in normalize_label(detection.label)
            or normalize_label(detection.label) in normalized_target
        ]


def extract_seg_rgbd_roi(
    rgb_image,
    depth_image,
    cloud,
    detection,
    bbox_padding_ratio=0.02,
    max_depth_m=1.5,
    depth_filter=True,
    depth_margin_m=0.04,
    min_points=30,
):
    if cloud is None or cloud.ndim != 3:
        raise ValueError("An organized point cloud with shape (H, W, 3) is required.")

    rgb_bbox = detection.padded_xyxy(rgb_image.shape, bbox_padding_ratio)
    left, top, right, bottom = rgb_bbox
    if right <= left or bottom <= top:
        raise ValueError(f"Invalid segmentation bbox: {rgb_bbox}")

    rgb_crop = rgb_image[top:bottom, left:right].copy()
    mask_full = detection.mask.astype(np.uint8)
    mask_rgb_crop = mask_full[top:bottom, left:right].copy()
    mask_raw_full = mask_full.copy()
    mask_raw_rgb_crop = mask_rgb_crop.copy()
    rgb_masked_crop = cv2.bitwise_and(rgb_crop, rgb_crop, mask=mask_rgb_crop)

    depth_mask_full = cv2.resize(mask_full, (depth_image.shape[1], depth_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    depth_bbox = scale_bbox(rgb_bbox, rgb_image.shape, depth_image.shape)
    dx1, dy1, dx2, dy2 = depth_bbox
    depth_crop = depth_image[dy1:dy2, dx1:dx2].copy()
    mask_depth_crop = depth_mask_full[dy1:dy2, dx1:dx2].copy()

    cloud_mask_full = cv2.resize(mask_full, (cloud.shape[1], cloud.shape[0]), interpolation=cv2.INTER_NEAREST)
    cloud_bbox = scale_bbox(rgb_bbox, rgb_image.shape, cloud.shape)
    cx1, cy1, cx2, cy2 = cloud_bbox
    cloud_crop = cloud[cy1:cy2, cx1:cx2].copy()
    mask_cloud_crop = cloud_mask_full[cy1:cy2, cx1:cx2].copy()

    valid_mask = (mask_cloud_crop > 0) & np.isfinite(cloud_crop).all(axis=2) & (cloud_crop[:, :, 2] > 0.0)
    if max_depth_m > 0.0:
        valid_mask &= cloud_crop[:, :, 2] <= float(max_depth_m)

    raw_valid_points = int(np.count_nonzero(valid_mask))
    depth_reference = None
    depth_reference_points = 0
    depth_reference_mode = "disabled"
    foreground_z_min = None
    foreground_z_max = None
    if depth_filter and np.any(valid_mask):
        depth_reference, depth_reference_points, depth_reference_mode = estimate_mask_center_depth(cloud_crop, valid_mask)
        foreground_z_min = depth_reference - float(depth_margin_m)
        foreground_z_max = depth_reference + float(depth_margin_m)
        filtered_mask = valid_mask & (cloud_crop[:, :, 2] >= foreground_z_min) & (cloud_crop[:, :, 2] <= foreground_z_max)
        if int(np.count_nonzero(filtered_mask)) >= min_points:
            valid_mask = filtered_mask

    foreground_points = cloud_crop[valid_mask][:, :3]
    if len(foreground_points) < min_points:
        raise ValueError(f"Only {len(foreground_points)} mask-selected foreground points found; need at least {min_points}.")

    final_mask_cloud_crop = valid_mask.astype(np.uint8) * 255
    final_mask_rgb_crop = mask_rgb_crop
    final_mask_full = mask_full
    if final_mask_cloud_crop.shape[:2] == mask_rgb_crop.shape[:2]:
        final_mask_rgb_crop = final_mask_cloud_crop
        final_mask_full = np.zeros_like(mask_full)
        final_mask_full[top:bottom, left:right] = final_mask_rgb_crop
    final_mask_depth_crop = mask_depth_crop
    if final_mask_cloud_crop.shape[:2] == mask_depth_crop.shape[:2]:
        final_mask_depth_crop = final_mask_cloud_crop

    rgb_masked_crop = cv2.bitwise_and(rgb_crop, rgb_crop, mask=final_mask_rgb_crop)

    z_values = foreground_points[:, 2]
    return SegRgbdRoi(
        rgb_crop=rgb_crop,
        rgb_masked_crop=rgb_masked_crop,
        depth_crop=depth_crop,
        cloud_crop=cloud_crop,
        mask_full=final_mask_full,
        mask_raw_full=mask_raw_full,
        mask_rgb_crop=final_mask_rgb_crop,
        mask_raw_rgb_crop=mask_raw_rgb_crop,
        mask_depth_crop=final_mask_depth_crop,
        mask_cloud_crop=final_mask_cloud_crop,
        foreground_points=foreground_points,
        centroid=np.median(foreground_points, axis=0),
        bbox_xyxy=rgb_bbox,
        depth_bbox_xyxy=depth_bbox,
        cloud_bbox_xyxy=cloud_bbox,
        stats={
            "max_depth_m": float(max_depth_m),
            "raw_mask_valid_points": raw_valid_points,
            "depth_filter": bool(depth_filter),
            "depth_margin_m": float(depth_margin_m),
            "depth_reference_m": depth_reference,
            "depth_reference_points": int(depth_reference_points),
            "depth_reference_mode": depth_reference_mode,
            "depth_filter_z_min": foreground_z_min,
            "depth_filter_z_max": foreground_z_max,
            "foreground_z_min": float(np.min(z_values)),
            "foreground_z_median": float(np.median(z_values)),
            "foreground_z_max": float(np.max(z_values)),
            "foreground_points": int(len(foreground_points)),
            "foreground_bounds": point_bounds(foreground_points),
        },
    )


def draw_seg_debug(image, detections, selected, roi=None):
    debug = image.copy()
    overlay = debug.copy()
    for detection in detections:
        color = (0, 255, 255) if detection is selected else (0, 180, 0)
        contours, _ = cv2.findContours(detection.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, thickness=cv2.FILLED)
        cv2.drawContours(debug, contours, -1, color, thickness=2)
        cv2.rectangle(debug, (detection.left, detection.top), (detection.right, detection.bottom), color, 2)
        cv2.putText(
            debug,
            f"{detection.label} {detection.confidence:.2f}",
            (detection.left, max(18, detection.top - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )
    debug = cv2.addWeighted(debug, 0.78, overlay, 0.22, 0.0)
    if roi is not None:
        x1, y1, x2, y2 = roi.bbox_xyxy
        cv2.rectangle(debug, (x1, y1), (x2, y2), (255, 128, 0), 2)
        cv2.putText(debug, "seg RGB-D crop saved", (x1, min(debug.shape[0] - 8, y2 + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 128, 0), 2)
    return debug


class RgbdSegCropServiceNode(Node):
    NODE_NAME = "rgbd_seg_crop_service_node"
    DEFAULT_SERVICE_NAME = "detect_object_rgbd_seg_crop"
    READY_LOG_NAME = "YOLO segmentation RGB-D crop service"

    def __init__(self, node_name=None, default_params=None):
        super().__init__(node_name or self.NODE_NAME)
        default_params = dict(default_params or {})

        def param_default(name, fallback):
            value = default_params.get(name, fallback)
            if isinstance(fallback, bool) and isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            if isinstance(fallback, float) and isinstance(value, str):
                return float(value)
            if isinstance(fallback, int) and not isinstance(fallback, bool) and isinstance(value, str):
                return int(value)
            return value

        self.declare_parameter("model_path", param_default("model_path", str(DEFAULT_MODEL_PATH)))
        self.declare_parameter("confidence", param_default("confidence", 0.35))
        self.declare_parameter("iou", param_default("iou", 0.45))
        self.declare_parameter("device", param_default("device", ""))
        self.declare_parameter("display", param_default("display", True))
        self.declare_parameter("display_hz", param_default("display_hz", 5.0))
        self.declare_parameter("target_label", param_default("target_label", ""))
        self.declare_parameter("service_name", param_default("service_name", self.DEFAULT_SERVICE_NAME))
        self.declare_parameter("image_topic", param_default("image_topic", "/camera/camera/color/image_raw"))
        self.declare_parameter("depth_topic", param_default("depth_topic", "/camera/camera/depth/color/image_raw"))
        self.declare_parameter("points_topic", param_default("points_topic", "/camera/camera/depth/color/points"))
        self.declare_parameter("input_wait_timeout", param_default("input_wait_timeout", 5.0))
        self.declare_parameter("camera_frame", param_default("camera_frame", "camera_color_optical_frame"))
        self.declare_parameter("bbox_padding_ratio", param_default("bbox_padding_ratio", 0.02))
        self.declare_parameter("input_crop_ratio", param_default("input_crop_ratio", 1.0))
        self.declare_parameter("min_roi_points", param_default("min_roi_points", 30))
        self.declare_parameter("max_depth_m", param_default("max_depth_m", 1.5))
        self.declare_parameter("depth_filter", param_default("depth_filter", True))
        self.declare_parameter("depth_margin_m", param_default("depth_margin_m", 0.04))
        self.declare_parameter("save_dir", param_default("save_dir", str(DEFAULT_SAVE_DIR)))
        self.declare_parameter("annotated_image_topic", param_default("annotated_image_topic", "/yolov11_seg/rgbd_crop_detection_image"))
        self.declare_parameter("roi_mask_topic", param_default("roi_mask_topic", "/yolov11_seg/rgbd_crop_mask"))
        self.declare_parameter("roi_info_topic", param_default("roi_info_topic", "/yolov11_seg/rgbd_crop_info"))
        self.declare_parameter("roi_points_topic", param_default("roi_points_topic", "/yolov11_seg/rgbd_crop_points"))

        self.target_label = parse_target_label(self.get_parameter("target_label").value)
        self.display = bool(self.get_parameter("display").value)
        self.input_wait_timeout = float(self.get_parameter("input_wait_timeout").value)
        self.camera_frame = str(self.get_parameter("camera_frame").value)
        self.bbox_padding_ratio = float(self.get_parameter("bbox_padding_ratio").value)
        self.input_crop_ratio = float(self.get_parameter("input_crop_ratio").value)
        self.min_roi_points = int(self.get_parameter("min_roi_points").value)
        self.max_depth_m = float(self.get_parameter("max_depth_m").value)
        self.depth_filter = bool(self.get_parameter("depth_filter").value)
        self.depth_margin_m = float(self.get_parameter("depth_margin_m").value)
        self.save_dir = Path(self.get_parameter("save_dir").value).expanduser()
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.detector = YoloSegDetector(
            model_path=Path(self.get_parameter("model_path").value).expanduser(),
            confidence=float(self.get_parameter("confidence").value),
            iou=float(self.get_parameter("iou").value),
            device=str(self.get_parameter("device").value),
        )
        yolo_device = self.detector.device or "auto"
        self.get_logger().info(f"YOLO inference device: {yolo_device}")

        qos_profile = QoSProfile(depth=10)
        qos_profile.reliability = ReliabilityPolicy.BEST_EFFORT

        self.bridge = CvBridge()
        self.latest_cv_img = None
        self.latest_depth_img = None
        self.latest_cloud = None
        self.annotated_img = None
        self.mask_img = None
        self.skip_counts = {}
        self.callback_group = ReentrantCallbackGroup()

        self.create_subscription(
            Image,
            self.get_parameter("image_topic").value,
            self.image_callback,
            qos_profile,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            Image,
            self.get_parameter("depth_topic").value,
            self.depth_callback,
            qos_profile,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            PointCloud2,
            self.get_parameter("points_topic").value,
            self.points_callback,
            qos_profile,
            callback_group=self.callback_group,
        )
        self.create_service(
            StringString,
            self.get_parameter("service_name").value,
            self.detect_callback,
            callback_group=self.callback_group,
        )

        latched_qos = QoSProfile(depth=10)
        latched_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        latched_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.roi_pub = self.create_publisher(PointCloud2, self.get_parameter("roi_points_topic").value, latched_qos)
        self.roi_info_pub = self.create_publisher(String, self.get_parameter("roi_info_topic").value, 10)
        self.annotated_image_pub = self.create_publisher(Image, self.get_parameter("annotated_image_topic").value, 10)
        self.roi_mask_pub = self.create_publisher(Image, self.get_parameter("roi_mask_topic").value, 10)

        display_hz = max(float(self.get_parameter("display_hz").value), 0.1)
        self.timer = self.create_timer(1.0 / display_hz, self.timer_callback, callback_group=self.callback_group)
        self.get_logger().info(
            f"{self.READY_LOG_NAME} ready. "
            f"model={self.detector.model_path}, save_dir={self.save_dir}, known_labels={', '.join(known_labels())}"
        )

    def image_callback(self, msg):
        if msg.width == 0 or msg.height == 0 or len(msg.data) == 0:
            self.log_skipped_message("rgb_empty", "Skipping empty RGB image message.")
            return
        if str(msg.encoding or "").lower() in {"rgb8", "bgr8", "rgba8", "bgra8", "mono8", "8uc1", "8uc3", "8uc4"}:
            try:
                image = image_msg_to_cv2_fallback(msg, desired_encoding="bgr8")
            except Exception as exc:
                self.log_skipped_message("rgb_fallback_error", f"Skipping RGB image that fallback could not convert: {exc}")
                return
            if image is None or image.size == 0:
                self.log_skipped_message("rgb_empty_array", "Skipping RGB image converted to an empty array.")
                return
            self.latest_cv_img = image
            return
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            try:
                image = image_msg_to_cv2_fallback(msg, desired_encoding="bgr8")
                self.log_skipped_message(
                    "rgb_bridge_fallback",
                    f"cv_bridge could not convert RGB image, but manual fallback succeeded. encoding={msg.encoding}",
                )
            except Exception as fallback_exc:
                self.log_skipped_message(
                    "rgb_bridge_error",
                    f"Skipping RGB image that neither cv_bridge nor fallback could convert: "
                    f"cv_bridge={exc}; fallback={fallback_exc}",
                )
                return
        if image is None or image.size == 0:
            self.log_skipped_message("rgb_empty_array", "Skipping RGB image converted to an empty array.")
            return
        self.latest_cv_img = image

    def depth_callback(self, msg):
        if msg.width == 0 or msg.height == 0 or len(msg.data) == 0:
            self.log_skipped_message("depth_empty", "Skipping empty depth image message.")
            return
        if str(msg.encoding or "").lower() in {"16uc1", "mono16", "32fc1", "mono8", "8uc1"}:
            try:
                depth = image_msg_to_cv2_fallback(msg)
            except Exception as exc:
                self.log_skipped_message("depth_fallback_error", f"Skipping depth image that fallback could not convert: {exc}")
                return
            if depth is None or depth.size == 0:
                self.log_skipped_message("depth_empty_array", "Skipping depth image converted to an empty array.")
                return
            self.latest_depth_img = depth
            return
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError as exc:
            try:
                depth = image_msg_to_cv2_fallback(msg)
                self.log_skipped_message(
                    "depth_bridge_fallback",
                    f"cv_bridge could not convert depth image, but manual fallback succeeded. encoding={msg.encoding}",
                )
            except Exception as fallback_exc:
                self.log_skipped_message(
                    "depth_bridge_error",
                    f"Skipping depth image that neither cv_bridge nor fallback could convert: "
                    f"cv_bridge={exc}; fallback={fallback_exc}",
                )
                return
        if depth is None or depth.size == 0:
            self.log_skipped_message("depth_empty_array", "Skipping depth image converted to an empty array.")
            return
        self.latest_depth_img = depth

    def points_callback(self, msg):
        if msg.width == 0 or msg.height == 0 or len(msg.data) == 0:
            self.log_skipped_message("points_empty", "Skipping empty point cloud message.")
            return
        raw_cloud = pc2.read_points_numpy(msg, field_names=("x", "y", "z"))
        self.latest_cloud = raw_cloud.reshape(msg.height, msg.width, 3) if msg.height > 1 else raw_cloud

    def log_skipped_message(self, key, message):
        count = self.skip_counts.get(key, 0) + 1
        self.skip_counts[key] = count
        if count == 1:
            self.get_logger().warn(f"{message} Further identical messages will be suppressed.")

    def timer_callback(self):
        if self.annotated_img is not None:
            msg = self.bridge.cv2_to_imgmsg(self.annotated_img, encoding="bgr8")
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.camera_frame
            self.annotated_image_pub.publish(msg)
            if self.display:
                cv2.imshow("YOLO Seg RGB-D Crop", self.annotated_img)
                cv2.waitKey(1)
        if self.mask_img is not None:
            msg = self.bridge.cv2_to_imgmsg(self.mask_img, encoding="mono8")
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.camera_frame
            self.roi_mask_pub.publish(msg)

    @staticmethod
    def request_options(request_text):
        text = str(request_text or "").strip()
        if not text.startswith("{"):
            return {"target": text, "debug": True, "mode": "object"}
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return {"target": text, "debug": True, "mode": "object"}
        target = payload.get("target") or payload.get("label") or payload.get("object") or ""
        mode = payload.get("mode") or ("scene" if str(target).strip().lower() in SCENE_REQUEST_TARGETS else "object")
        return {
            "target": str(target),
            "debug": bool(payload.get("debug", True)),
            "mode": str(mode).strip().lower(),
        }

    def detect_callback(self, request, response):
        options = self.request_options(request.data)
        raw_target = str(options.get("target") or "")
        target_label = parse_target_label(raw_target) or self.target_label
        try:
            if options.get("mode") == "scene" or raw_target.strip().lower() in SCENE_REQUEST_TARGETS:
                info = self.detect_scene(request.data, debug=bool(options.get("debug", True)))
            else:
                info = self.detect_and_save(target_label, request.data, debug=bool(options.get("debug", True)))
        except Exception as exc:
            info = {"ok": False, "error": str(exc), "target": target_label, "known_labels": known_labels()}
            self.get_logger().warn(str(exc))

        response.data = json.dumps(info, sort_keys=True)
        msg = String()
        msg.data = response.data
        self.roi_info_pub.publish(msg)
        return response

    def missing_detection_inputs(self):
        missing = []
        if self.latest_cv_img is None:
            missing.append("RGB image")
        if self.latest_depth_img is None:
            missing.append("depth image")
        if self.latest_cloud is None:
            missing.append("organized point cloud")
        return missing

    def wait_for_detection_inputs(self):
        missing = self.missing_detection_inputs()
        if not missing:
            return

        timeout = max(0.0, self.input_wait_timeout)
        deadline = time.monotonic() + timeout
        self.get_logger().info(f"Waiting up to {timeout:.1f}s for {', '.join(missing)} before detection.")
        while rclpy.ok() and missing and time.monotonic() < deadline:
            time.sleep(0.02)
            missing = self.missing_detection_inputs()
        if missing:
            raise RuntimeError(f"{', '.join(missing)} has not been received yet.")

    def detect_and_save(self, target_label, request_text, debug=True):
        self.wait_for_detection_inputs()

        target_model = get_model(target_label)
        if target_model is None:
            raise RuntimeError(f"Unknown target='{target_label}'. Choose one of: {known_labels()}")

        rgb_image, depth_image, cloud = self.get_detection_inputs()
        started = time.monotonic()
        detections = self.detector.detect(rgb_image)
        yolo_done = time.monotonic()
        matching_detections = self.detector.matches(detections, target_model.name)
        if not matching_detections:
            seen = [detection.label for detection in detections]
            self.annotated_img = draw_seg_debug(rgb_image, detections, selected=None)
            debug_path = self.save_failure_debug_image(target_model.name) if debug else None
            raise RuntimeError(
                f"No YOLO segmentation matched target='{target_model.name}'. "
                f"seen={seen}. debug_image={debug_path}"
            )

        request_dir = self.make_request_dir(target_model.name)
        instances = []
        selected = None
        selected_roi = None
        saved_files = {}
        for index, detection in enumerate(matching_detections):
            instance_name = f"{target_model.name}_{index}"
            candidate_dir = request_dir / instance_name
            candidate_dir.mkdir(parents=True, exist_ok=False)
            try:
                roi = extract_seg_rgbd_roi(
                    rgb_image=rgb_image,
                    depth_image=depth_image,
                    cloud=cloud,
                    detection=detection,
                    bbox_padding_ratio=self.bbox_padding_ratio,
                    max_depth_m=self.max_depth_m,
                    depth_filter=self.depth_filter,
                    depth_margin_m=self.depth_margin_m,
                    min_points=self.min_roi_points,
                )
                files = self.save_rgbd_crop(
                    candidate_dir,
                    target_model.name,
                    detection,
                    detections,
                    roi,
                    request_text,
                    rgb_image,
                    debug=debug,
                )
                instance = {
                    "ok": True,
                    "target": target_model.name,
                    "instance_name": instance_name,
                    "instance_index": int(index),
                    "detected_label": detection.label,
                    "frame_id": self.camera_frame,
                    "detection": detection.to_dict(),
                    "roi": roi.to_dict(),
                    "location_xyz_m": roi.centroid.astype(float).tolist(),
                    "save_dir": str(candidate_dir),
                    "files": files,
                }
                if selected is None:
                    selected = detection
                    selected_roi = roi
                    saved_files = files
                instances.append(instance)
            except Exception as exc:
                instances.append({
                    "ok": False,
                    "target": target_model.name,
                    "instance_name": instance_name,
                    "instance_index": int(index),
                    "detected_label": detection.label,
                    "frame_id": self.camera_frame,
                    "detection": detection.to_dict(),
                    "error": str(exc),
                    "save_dir": str(candidate_dir),
                })
        roi_done = time.monotonic()

        valid_instances = [instance for instance in instances if instance.get("ok")]
        if not valid_instances:
            self.annotated_img = draw_seg_debug(rgb_image, detections, selected=None)
            debug_path = self.save_failure_debug_image(target_model.name) if debug else None
            raise RuntimeError(
                f"YOLO matched target='{target_model.name}', but no instance produced a usable RGB-D ROI. "
                f"errors={[instance.get('error') for instance in instances]}. debug_image={debug_path}"
            )

        pca_bboxes = []
        if debug:
            pca_bboxes = compute_all_detection_pca_bboxes(
                rgb_image=rgb_image,
                depth_image=depth_image,
                cloud=cloud,
                detections=detections,
                bbox_padding_ratio=self.bbox_padding_ratio,
                max_depth_m=self.max_depth_m,
                depth_filter=self.depth_filter,
                depth_margin_m=self.depth_margin_m,
                min_points=self.min_roi_points,
                extract_roi=extract_seg_rgbd_roi,
            )
        position_points = [
            {
                "label": item.get("label"),
                "confidence": item.get("confidence"),
                **item.get("position_point"),
            }
            for item in pca_bboxes
            if item.get("ok") and (item.get("position_point") or {}).get("ok")
        ]

        self.publish_roi_cloud(selected_roi.foreground_points)
        self.mask_img = selected_roi.mask_full
        self.annotated_img = draw_seg_debug(rgb_image, detections, selected, selected_roi)
        selected_info = valid_instances[0]

        info = {
            "ok": True,
            "stage": "rgbd_seg_crop_saved",
            "message": "YOLO segmentation polygon mask and RGB-D crop were saved.",
            "request": request_text,
            "target": target_model.name,
            "detected_label": selected_info["detected_label"],
            "frame_id": self.camera_frame,
            "detection": selected_info["detection"],
            "all_detections": [detection.to_dict() for detection in detections],
            "all_pca_bboxes": pca_bboxes,
            "all_position_points": position_points,
            "input_crop_ratio": float(self.input_crop_ratio),
            "roi": selected_info["roi"],
            "location_xyz_m": selected_info["location_xyz_m"],
            "instances": instances,
            "save_dir": str(request_dir),
            "files": saved_files,
        }
        self.get_logger().info(
            f"Saved {len(valid_instances)}/{len(instances)} seg RGB-D instance crops for {target_model.name}: "
            f"selected_bbox={selected_roi.bbox_xyxy}, points={len(selected_roi.foreground_points)}, "
            f"centroid=({selected_roi.centroid[0]:.3f}, {selected_roi.centroid[1]:.3f}, {selected_roi.centroid[2]:.3f}), "
            f"timing_ms=yolo:{(yolo_done - started) * 1000.0:.1f}, "
            f"roi:{(roi_done - yolo_done) * 1000.0:.1f}, total:{(time.monotonic() - started) * 1000.0:.1f}"
        )
        return info

    def detect_scene(self, request_text, debug=True):
        self.wait_for_detection_inputs()
        rgb_image, depth_image, cloud = self.get_detection_inputs()
        started = time.monotonic()
        detections = self.detector.detect(rgb_image)
        yolo_done = time.monotonic()
        request_dir = self.make_request_dir("scene")

        label_counts: dict[str, int] = {}
        instances = []
        valid_instances = 0
        selected_for_display = None
        selected_roi = None
        for detection in detections:
            target_model = get_model(detection.label)
            if target_model is None:
                continue
            class_name = target_model.name
            index = label_counts.get(class_name, 0)
            label_counts[class_name] = index + 1
            instance_name = f"{class_name}_{index}"
            candidate_dir = request_dir / instance_name
            candidate_dir.mkdir(parents=True, exist_ok=False)
            try:
                roi = extract_seg_rgbd_roi(
                    rgb_image=rgb_image,
                    depth_image=depth_image,
                    cloud=cloud,
                    detection=detection,
                    bbox_padding_ratio=self.bbox_padding_ratio,
                    max_depth_m=self.max_depth_m,
                    depth_filter=self.depth_filter,
                    depth_margin_m=self.depth_margin_m,
                    min_points=self.min_roi_points,
                )
                files = self.save_rgbd_crop(
                    candidate_dir,
                    class_name,
                    detection,
                    detections,
                    roi,
                    request_text,
                    rgb_image,
                    debug=debug,
                )
                instance = {
                    "ok": True,
                    "target": class_name,
                    "instance_name": instance_name,
                    "instance_index": int(index),
                    "detected_label": detection.label,
                    "frame_id": self.camera_frame,
                    "detection": detection.to_dict(),
                    "roi": roi.to_dict(),
                    "location_xyz_m": roi.centroid.astype(float).tolist(),
                    "save_dir": str(candidate_dir),
                    "files": files,
                }
                valid_instances += 1
                if selected_for_display is None:
                    selected_for_display = detection
                    selected_roi = roi
            except Exception as exc:
                instance = {
                    "ok": False,
                    "target": class_name,
                    "instance_name": instance_name,
                    "instance_index": int(index),
                    "detected_label": detection.label,
                    "frame_id": self.camera_frame,
                    "detection": detection.to_dict(),
                    "error": str(exc),
                    "save_dir": str(candidate_dir),
                }
            instances.append(instance)
        roi_done = time.monotonic()

        if selected_roi is not None:
            self.publish_roi_cloud(selected_roi.foreground_points)
            self.mask_img = selected_roi.mask_full
        else:
            self.mask_img = None
        self.annotated_img = draw_seg_debug(rgb_image, detections, selected_for_display, selected_roi)

        pca_bboxes = []
        if debug:
            pca_bboxes = compute_all_detection_pca_bboxes(
                rgb_image=rgb_image,
                depth_image=depth_image,
                cloud=cloud,
                detections=detections,
                bbox_padding_ratio=self.bbox_padding_ratio,
                max_depth_m=self.max_depth_m,
                depth_filter=self.depth_filter,
                depth_margin_m=self.depth_margin_m,
                min_points=self.min_roi_points,
                extract_roi=extract_seg_rgbd_roi,
            )
            (request_dir / "scene_metadata.json").write_text(
                json.dumps(
                    {
                        "request": request_text,
                        "frame_id": self.camera_frame,
                        "all_detections": [detection.to_dict() for detection in detections],
                        "instances": instances,
                        "all_pca_bboxes": pca_bboxes,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            cv2.imwrite(str(request_dir / "scene_annotated.png"), self.annotated_img)

        info = {
            "ok": True,
            "stage": "scene_scan",
            "message": "YOLO scene scan completed.",
            "request": request_text,
            "target": "__scene__",
            "frame_id": self.camera_frame,
            "all_detections": [detection.to_dict() for detection in detections],
            "all_pca_bboxes": pca_bboxes,
            "input_crop_ratio": float(self.input_crop_ratio),
            "instances": instances,
            "save_dir": str(request_dir),
        }
        self.get_logger().info(
            f"Scene scan: yolo_detections={len(detections)}, known_instances={valid_instances}, debug={debug}, "
            f"timing_ms=yolo:{(yolo_done - started) * 1000.0:.1f}, "
            f"roi:{(roi_done - yolo_done) * 1000.0:.1f}, total:{(time.monotonic() - started) * 1000.0:.1f}"
        )
        return info

    def get_detection_inputs(self):
        if self.input_crop_ratio <= 0.0 or self.input_crop_ratio >= 1.0:
            return self.latest_cv_img, self.latest_depth_img, self.latest_cloud

        rgb_bbox = centered_crop_xyxy(self.latest_cv_img.shape, self.input_crop_ratio)
        depth_bbox = scale_bbox(rgb_bbox, self.latest_cv_img.shape, self.latest_depth_img.shape)
        cloud_bbox = scale_bbox(rgb_bbox, self.latest_cv_img.shape, self.latest_cloud.shape)
        rgb_image = crop_array_xyxy(self.latest_cv_img, rgb_bbox)
        depth_image = crop_array_xyxy(self.latest_depth_img, depth_bbox)
        cloud = crop_array_xyxy(self.latest_cloud, cloud_bbox)
        return rgb_image, depth_image, cloud

    def make_request_dir(self, label):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        request_dir = self.save_dir / f"{timestamp}_{label}"
        request_dir.mkdir(parents=True, exist_ok=False)
        return request_dir

    def save_failure_debug_image(self, label):
        if self.annotated_img is None:
            return None
        debug_dir = self.save_dir / "_debug_failures"
        debug_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        debug_path = debug_dir / f"{timestamp}_{label}_detections.png"
        cv2.imwrite(str(debug_path), self.annotated_img)
        return str(debug_path)

    def save_rgbd_crop(
        self,
        request_dir,
        label,
        selected,
        detections,
        roi,
        request_text,
        source_image=None,
        pca_bboxes=None,
        debug=True,
    ):
        paths = {
            "rgb": request_dir / "rgb.png",
            "rgb_masked": request_dir / "rgb_masked.png",
            "depth_npy": request_dir / "depth.npy",
            "depth_png": request_dir / "depth.png",
            "depth_visualization": request_dir / "depth_vis.png",
            "mask": request_dir / "mask.png",
            "mask_full": request_dir / "mask_full.png",
            "mask_raw": request_dir / "mask_raw.png",
            "mask_raw_full": request_dir / "mask_raw_full.png",
            "mask_depth_size": request_dir / "mask_depth_size.png",
            "mask_cloud_size": request_dir / "mask_cloud_size.png",
            "mask_overlay": request_dir / "mask_overlay.png",
            "cloud_npy": request_dir / "cloud.npy",
            "foreground_points_npy": request_dir / "foreground_points.npy",
            "annotated": request_dir / "annotated.png",
            "annotated_pca_bbox": request_dir / "annotated_pca_bbox.png",
            "polygon_json": request_dir / "polygon.json",
            "metadata": request_dir / "metadata.json",
        }

        if not debug:
            np.save(paths["foreground_points_npy"], roi.foreground_points)
            return {
                "foreground_points_npy": str(paths["foreground_points_npy"]),
            }

        cv2.imwrite(str(paths["rgb"]), roi.rgb_crop)
        cv2.imwrite(str(paths["rgb_masked"]), roi.rgb_masked_crop)
        np.save(paths["depth_npy"], roi.depth_crop)
        cv2.imwrite(str(paths["depth_png"]), depth_to_uint16_mm(roi.depth_crop))
        depth_vis = make_depth_visual(roi.depth_crop)
        if depth_vis is not None:
            cv2.imwrite(str(paths["depth_visualization"]), depth_vis)
        cv2.imwrite(str(paths["mask"]), roi.mask_rgb_crop)
        cv2.imwrite(str(paths["mask_full"]), roi.mask_full)
        cv2.imwrite(str(paths["mask_raw"]), roi.mask_raw_rgb_crop)
        cv2.imwrite(str(paths["mask_raw_full"]), roi.mask_raw_full)
        cv2.imwrite(str(paths["mask_depth_size"]), roi.mask_depth_crop)
        cv2.imwrite(str(paths["mask_cloud_size"]), roi.mask_cloud_crop)
        overlay = roi.rgb_crop.copy()
        red = np.zeros_like(overlay)
        red[:, :, 2] = 255
        alpha_mask = (roi.mask_rgb_crop > 0).astype(np.uint8)[:, :, None]
        overlay = np.where(alpha_mask > 0, cv2.addWeighted(overlay, 0.55, red, 0.45, 0.0), overlay)
        cv2.imwrite(str(paths["mask_overlay"]), overlay)
        np.save(paths["cloud_npy"], roi.cloud_crop)
        np.save(paths["foreground_points_npy"], roi.foreground_points)
        if source_image is None:
            source_image = self.latest_cv_img
        cv2.imwrite(str(paths["annotated"]), draw_seg_debug(source_image, detections, selected, roi))
        cv2.imwrite(str(paths["annotated_pca_bbox"]), draw_pca_bbox_debug(source_image, pca_bboxes or []))

        position_points = [
            {
                "label": item.get("label"),
                "confidence": item.get("confidence"),
                **item.get("position_point"),
            }
            for item in (pca_bboxes or [])
            if item.get("ok") and (item.get("position_point") or {}).get("ok")
        ]

        polygon_payload = {
            "label": label,
            "detected_label": selected.label,
            "confidence": float(selected.confidence),
            "polygon_xy": selected.polygon_xy,
            "bbox_xyxy": [int(v) for v in roi.bbox_xyxy],
        }
        paths["polygon_json"].write_text(json.dumps(polygon_payload, indent=2, sort_keys=True), encoding="utf-8")

        metadata = {
            "label": label,
            "request": request_text,
            "frame_id": self.camera_frame,
            "detection": selected.to_dict(),
            "all_detections": [detection.to_dict() for detection in detections],
            "all_pca_bboxes": pca_bboxes or [],
            "all_position_points": position_points,
            "input_crop_ratio": float(self.input_crop_ratio),
            "roi": roi.to_dict(),
            "location_xyz_m": roi.centroid.astype(float).tolist(),
        }
        paths["metadata"].write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

        return {key: str(path) for key, path in paths.items() if path.exists()}

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
    node = RgbdSegCropServiceNode()
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
