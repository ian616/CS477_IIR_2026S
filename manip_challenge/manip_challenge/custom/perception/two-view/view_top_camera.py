#!/usr/bin/env python3
"""Show or save the current top-view RGB camera image."""

import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


DEFAULT_IMAGE_TOPIC = "/camera/camera/color/image_raw"
DEFAULT_SAVE_DIR = Path(__file__).resolve().parent / "results" / "top_camera"


def image_msg_to_bgr(msg):
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
    }
    if encoding not in specs:
        raise ValueError(f"unsupported RGB topic encoding: {msg.encoding}")

    dtype, channels = specs[encoding]
    itemsize = np.dtype(dtype).itemsize
    min_step = int(msg.width) * channels * itemsize
    row_step = int(msg.step) if int(msg.step) > 0 else min_step
    required = row_step * int(msg.height)
    if len(msg.data) < required:
        raise ValueError(f"image data too short: got={len(msg.data)}, need={required}")

    flat = np.frombuffer(msg.data, dtype=dtype, count=required // itemsize)
    rows = flat.reshape(int(msg.height), row_step // itemsize)
    pixels = rows[:, : min_step // itemsize]
    if channels == 1:
        image = pixels.reshape(int(msg.height), int(msg.width)).copy()
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    image = pixels.reshape(int(msg.height), int(msg.width), channels).copy()
    if encoding in ("rgb8", "8uc3"):
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if encoding == "bgr8":
        return image
    if encoding == "rgba8":
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    if encoding in ("bgra8", "8uc4"):
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


class TopCameraViewer(Node):
    def __init__(self, image_topic, save_dir, once=False, no_window=False):
        super().__init__("top_camera_viewer")
        self.save_dir = Path(save_dir).expanduser()
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.once = bool(once)
        self.no_window = bool(no_window)
        self.latest = None
        self.saved_once = False

        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self.create_subscription(Image, image_topic, self.image_callback, qos)
        self.get_logger().info(f"Listening for top camera images on {image_topic}")

    def image_callback(self, msg):
        try:
            image = image_msg_to_bgr(msg)
        except Exception as exc:
            self.get_logger().warn(f"Skipping image: {exc}")
            return

        stamp = self.get_clock().now().to_msg()
        label = f"{msg.header.frame_id or 'camera'}  {msg.width}x{msg.height}  {msg.encoding}"
        cv2.rectangle(image, (0, 0), (image.shape[1], 34), (20, 20, 20), -1)
        cv2.putText(image, label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        self.latest = image

        if self.once and not self.saved_once:
            path = self.save_image(stamp)
            self.get_logger().info(f"Saved top camera image: {path}")
            self.saved_once = True
            return

        if not self.no_window:
            cv2.imshow("Top camera RGB", image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                path = self.save_image(stamp)
                self.get_logger().info(f"Saved top camera image: {path}")
            elif key in (ord("q"), 27):
                raise KeyboardInterrupt

    def save_image(self, stamp):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = self.save_dir / f"{timestamp}_top_camera.png"
        cv2.imwrite(str(path), self.latest)
        return path


def main():
    parser = argparse.ArgumentParser(description="Show or save top-view RGB camera frames.")
    parser.add_argument("--image-topic", default=DEFAULT_IMAGE_TOPIC)
    parser.add_argument("--save-dir", default=str(DEFAULT_SAVE_DIR))
    parser.add_argument("--once", action="store_true", help="Save one frame and exit.")
    parser.add_argument("--no-window", action="store_true", help="Do not open an OpenCV window.")
    args = parser.parse_args()

    rclpy.init()
    node = TopCameraViewer(args.image_topic, args.save_dir, once=args.once, no_window=args.no_window)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.2)
            if args.once and node.saved_once:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
