#!/usr/bin/env python3
"""Client for the YOLO segmentation + ICP 6D pose service."""

import argparse
import json
import sys

import rclpy
from rclpy.node import Node
from riro_srvs.srv import StringString


class RgbdSegIcpPoseClient(Node):
    def __init__(self, service_name):
        super().__init__("rgbd_seg_icp_pose_client")
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
    parser = argparse.ArgumentParser(description="Request YOLO segmentation crop + ICP 6D pose.")
    parser.add_argument("command", nargs="?", default="banana", help="Target or command, e.g. banana or 'pick banana'.")
    parser.add_argument("--service-name", default="detect_object_rgbd_seg_icp_pose")
    args = parser.parse_args(cli_args)

    rclpy.init()
    node = RgbdSegIcpPoseClient(args.service_name)
    try:
        response = node.send_request(args.command)
        if response is None:
            node.get_logger().error("Service call failed.")
            return
        try:
            print(json.dumps(json.loads(response.data), indent=2, sort_keys=True))
        except json.JSONDecodeError:
            print(response.data)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
