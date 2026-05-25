#!/usr/bin/env python3
"""Client for the top-view YOLO segmentation RGB-D crop server."""

import argparse
import json
import sys

import rclpy
from rclpy.node import Node
from riro_srvs.srv import StringString


DEFAULT_SERVICE = "detect_object_top_rgbd_seg_crop"


class TopViewSegClient(Node):
    def __init__(self, service_name):
        super().__init__("top_view_seg_client")
        self.cli = self.create_client(StringString, service_name)
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f"Waiting for service '{service_name}'...")

    def send_request(self, command):
        request = StringString.Request()
        request.data = command
        future = self.cli.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()


def main():
    cli_args = rclpy.utilities.remove_ros_args(args=sys.argv)[1:]
    parser = argparse.ArgumentParser(description="Request a top-view YOLO segmentation RGB-D crop.")
    parser.add_argument("command", nargs="?", default="banana", help="Target or command, e.g. coke_can or 'pick banana'.")
    parser.add_argument("--service-name", default=DEFAULT_SERVICE)
    args = parser.parse_args(cli_args)

    rclpy.init()
    node = TopViewSegClient(args.service_name)
    try:
        response = node.send_request(args.command)
        if response is None:
            node.get_logger().error("Service call failed.")
            return 1
        try:
            print(json.dumps(json.loads(response.data), indent=2, sort_keys=True))
        except json.JSONDecodeError:
            print(response.data)
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
