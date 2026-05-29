#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


if __package__ in {None, ""}:
    _PDDL_DIR = Path(__file__).resolve().parent
    _PACKAGE_ROOT = _PDDL_DIR.parents[1]
    if str(_PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(_PACKAGE_ROOT))
    from manip_challenge.pddl.utils import ensure_project_paths, ensure_ros_python
else:
    from .utils import ensure_project_paths, ensure_ros_python


ensure_ros_python()
ensure_project_paths()

import rclpy
from rclpy.node import Node
from riro_srvs.srv import StringString
from std_msgs.msg import String


class PddlTampClient(Node):
    def __init__(self, command_topic, service_name):
        super().__init__("pddl_tamp_client")
        self.command_topic = command_topic
        self.publisher = self.create_publisher(String, command_topic, 10)
        self.service_name = service_name
        self.client = self.create_client(StringString, service_name)

    def publish_command(self, command, settle_time=3.0):
        msg = String()
        msg.data = command
        deadline = time.monotonic() + float(settle_time)
        while rclpy.ok() and time.monotonic() < deadline and self.publisher.get_subscription_count() == 0:
            rclpy.spin_once(self, timeout_sec=0.05)
            time.sleep(0.05)
        if self.publisher.get_subscription_count() == 0:
            self.get_logger().warn(
                f"No subscribers on '{self.command_topic}' after {settle_time:.1f}s — "
                "message may be dropped. Is the server running?"
            )
        self.publisher.publish(msg)
        rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info(f"Published command to '{self.command_topic}': {command}")

    def wait_for_server(self, timeout):
        deadline = time.monotonic() + timeout
        while rclpy.ok() and not self.client.wait_for_service(timeout_sec=0.5):
            if time.monotonic() > deadline:
                return False
            self.get_logger().info(f"Waiting for service '{self.service_name}'...")
        return True

    def send(self, command, timeout):
        req = StringString.Request()
        req.data = command
        future = self.client.call_async(req)
        deadline = time.monotonic() + timeout
        while rclpy.ok() and not future.done():
            if time.monotonic() > deadline:
                raise TimeoutError(f"Timed out waiting for '{self.service_name}' response.")
            rclpy.spin_once(self, timeout_sec=0.05)
        return future.result()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Send natural-language commands to the PDDL TAMP server.")
    parser.add_argument("command", nargs="*", help="Natural-language command")
    parser.add_argument("--command-topic", default="/task_commands")
    parser.add_argument("--service", action="store_true", help="Use the legacy service call instead of topic publish.")
    parser.add_argument("--service-name", default="pddl_tamp_command")
    parser.add_argument("--wait-timeout", type=float, default=10.0)
    parser.add_argument("--response-timeout", type=float, default=360.0)
    parser.add_argument("--publish-settle-time", type=float, default=3.0)
    parser.add_argument("--raw", action="store_true")
    return parser.parse_args(rclpy.utilities.remove_ros_args(args=argv or sys.argv)[1:])


def main(argv=None):
    args = parse_args(argv)
    command = " ".join(args.command).strip()
    if not command:
        print("Usage: python3 client.py \"Move the banana to the left storage.\"")
        return 1
    rclpy.init(args=argv)
    node = PddlTampClient(args.command_topic, args.service_name)
    try:
        if not args.service:
            node.publish_command(command, settle_time=args.publish_settle_time)
            return 0
        if not node.wait_for_server(args.wait_timeout):
            node.get_logger().error(f"Service '{args.service_name}' is not available.")
            return 1
        response = node.send(command, args.response_timeout)
        if response is None:
            node.get_logger().error("Service call failed.")
            return 1
        if args.raw:
            print(response.data)
        else:
            print(json.dumps(json.loads(response.data), indent=2, sort_keys=True))
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
