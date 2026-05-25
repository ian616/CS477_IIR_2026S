#!/usr/bin/env python3
"""Print one-shot metadata for the top-view RGB-D topics."""

import argparse
import time

import rclpy
import sensor_msgs_py.point_cloud2 as pc2
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, PointCloud2


DEFAULT_IMAGE_TOPIC = "/camera/camera/color/image_raw"
DEFAULT_DEPTH_TOPIC = "/camera/camera/depth/color/image_raw"
DEFAULT_POINTS_TOPIC = "/camera/camera/depth/color/points"


class TopicInspector(Node):
    def __init__(self, image_topic, depth_topic, points_topic):
        super().__init__("top_view_topic_inspector")
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self.seen = {}
        self.create_subscription(Image, image_topic, self.image_callback("rgb"), qos)
        self.create_subscription(Image, depth_topic, self.image_callback("depth"), qos)
        self.create_subscription(PointCloud2, points_topic, self.points_callback, qos)

    def image_callback(self, key):
        def callback(msg):
            if key in self.seen:
                return
            self.seen[key] = True
            print(
                f"{key}: width={msg.width}, height={msg.height}, "
                f"encoding={msg.encoding}, step={msg.step}, data_len={len(msg.data)}, "
                f"frame_id={msg.header.frame_id}"
            )

        return callback

    def points_callback(self, msg):
        if "points" in self.seen:
            return
        self.seen["points"] = True
        print(
            f"points: width={msg.width}, height={msg.height}, point_step={msg.point_step}, "
            f"row_step={msg.row_step}, data_len={len(msg.data)}, frame_id={msg.header.frame_id}"
        )
        try:
            points = pc2.read_points_numpy(msg, field_names=("x", "y", "z"))
            print(f"points numpy shape: {points.shape}")
        except Exception as exc:
            print(f"points numpy read failed: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Inspect top camera RGB-D topic metadata.")
    parser.add_argument("--image-topic", default=DEFAULT_IMAGE_TOPIC)
    parser.add_argument("--depth-topic", default=DEFAULT_DEPTH_TOPIC)
    parser.add_argument("--points-topic", default=DEFAULT_POINTS_TOPIC)
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()

    rclpy.init()
    node = TopicInspector(args.image_topic, args.depth_topic, args.points_topic)
    deadline = time.monotonic() + args.timeout
    try:
        while rclpy.ok() and time.monotonic() < deadline and len(node.seen) < 3:
            rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        missing = sorted(set(["rgb", "depth", "points"]) - set(node.seen))
        if missing:
            print(f"missing within {args.timeout:.1f}s: {', '.join(missing)}")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
