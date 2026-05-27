#!/usr/bin/env python3
"""Client for two_view_grasp_server.py.

Examples:
    python3 two_view_grasp_client.py banana left
    python3 two_view_grasp_client.py "move coke can to right storage"
    python3 two_view_grasp_client.py
    python3 two_view_grasp_client.py home
"""

import argparse
import json
import os
import sys
import time


def ensure_ros_python():
    ros_python = "/usr/bin/python3"
    if sys.version_info[:2] != (3, 10) and os.path.exists(ros_python):
        os.execv(ros_python, [ros_python, *sys.argv])


ensure_ros_python()

import rclpy
from rclpy.node import Node
from riro_srvs.srv import StringString


class TwoViewGraspClient(Node):
    def __init__(self, service_name):
        super().__init__("two_view_grasp_client")
        self.service_name = service_name
        self.client = self.create_client(StringString, service_name)

    def wait_for_server(self, timeout):
        deadline = time.monotonic() + timeout
        while rclpy.ok() and not self.client.wait_for_service(timeout_sec=0.5):
            if time.monotonic() > deadline:
                return False
            self.get_logger().info(f"Waiting for service '{self.service_name}'...")
        return True

    def send(self, command, timeout):
        request = StringString.Request()
        request.data = command
        future = self.client.call_async(request)
        deadline = time.monotonic() + timeout
        while rclpy.ok() and not future.done():
            if time.monotonic() > deadline:
                raise TimeoutError(f"Timed out waiting for '{self.service_name}' response.")
            rclpy.spin_once(self, timeout_sec=0.05)
        return future.result()


def print_grasp_summary(payload):
    selection = payload.get("grasp_selection") or {}
    if not selection:
        return

    method = selection.get("method")
    target = selection.get("target_xyz_m")
    delta = selection.get("delta_from_raw_centroid_m")
    visualization = selection.get("visualization")
    depth_adjustment = selection.get("rgbd_depth_adjustment") or {}

    print()
    if method:
        print(f"grasp method: {method}")
    if target:
        print("selected grasp xyz [source frame]: " + ", ".join(f"{float(v):.4f}" for v in target))
    if delta:
        print("delta from raw centroid [m]: " + ", ".join(f"{float(v):+.4f}" for v in delta))
    if depth_adjustment:
        surface = depth_adjustment.get("measured_surface_depth_m")
        clearance = depth_adjustment.get("clearance_above_surface_m")
        adjusted = depth_adjustment.get("adjusted_target_depth_m")
        if surface is not None and clearance is not None and adjusted is not None:
            print(
                "rgbd surface depth [m]: "
                f"{float(surface):.4f}, clearance: {float(clearance):.4f}, target depth: {float(adjusted):.4f}"
            )
    if visualization:
        print(f"visualization: {visualization}")


def print_response(response, raw=False):
    if raw:
        print(response.data)
        return

    try:
        payload = json.loads(response.data)
    except json.JSONDecodeError:
        print(response.data)
        return

    print(json.dumps(payload, indent=2, sort_keys=True))
    print_grasp_summary(payload)


def command_loop(initial_command=None):
    if initial_command:
        yield initial_command

    while True:
        try:
            command = input("two-view grasp> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not command:
            continue
        if command.lower() in {"exit", "quit", "q"}:
            return
        yield command


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Send pick-and-place commands to the two-view server.")
    parser.add_argument("command", nargs="*", help="Command text, e.g. banana left")
    parser.add_argument("--service-name", default="two_view_grasp_command")
    parser.add_argument("--wait-timeout", type=float, default=10.0)
    parser.add_argument("--response-timeout", type=float, default=240.0)
    parser.add_argument("--loop", action="store_true", help="Keep reading commands after the first request.")
    parser.add_argument("--raw", action="store_true", help="Print raw service response instead of pretty JSON.")
    return parser.parse_args(rclpy.utilities.remove_ros_args(args=argv or sys.argv)[1:])


def main(argv=None):
    args = parse_args(argv)
    initial_command = " ".join(args.command).strip()
    interactive = args.loop or not initial_command

    rclpy.init(args=argv)
    node = TwoViewGraspClient(args.service_name)
    try:
        if not node.wait_for_server(args.wait_timeout):
            node.get_logger().error(f"Service '{args.service_name}' is not available.")
            return 1

        if interactive:
            print("Enter commands like 'banana left' or 'move coke can to right storage'. Type 'quit' to stop.")
            commands = command_loop(initial_command or None)
        else:
            commands = [initial_command]

        for command in commands:
            try:
                response = node.send(command, args.response_timeout)
            except TimeoutError as exc:
                node.get_logger().error(str(exc))
                if not interactive:
                    return 1
                continue

            if response is None:
                node.get_logger().error("Service call failed.")
                if not interactive:
                    return 1
                continue

            print_response(response, args.raw)
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
