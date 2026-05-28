#!/usr/bin/env python3
"""Client for two_view_grasp_server.py.

Publishes pick-and-place commands to the /task_commands topic.
Subscribes to /task_results to print grasp summary when the server finishes.

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
from std_msgs.msg import String


class TwoViewGraspClient(Node):
    def __init__(self, command_topic, result_topic):
        super().__init__("two_view_grasp_client")
        self.publisher = self.create_publisher(String, command_topic, 10)
        self.create_subscription(String, result_topic, self._result_callback, 10)

    def send(self, command):
        deadline = time.monotonic() + 5.0
        while self.publisher.get_subscription_count() == 0:
            if time.monotonic() > deadline:
                self.get_logger().warn("No subscribers discovered, publishing anyway.")
                break
            rclpy.spin_once(self, timeout_sec=0.1)
        msg = String()
        msg.data = command
        self.publisher.publish(msg)
        self.get_logger().info(f"Published: '{command}'")

    def _result_callback(self, msg):
        print_response(msg)


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


def print_response(msg, raw=False):
    if raw:
        print(msg.data)
        return

    try:
        payload = json.loads(msg.data)
    except json.JSONDecodeError:
        print(msg.data)
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
    parser = argparse.ArgumentParser(description="Publish pick-and-place commands to the two-view server.")
    parser.add_argument("command", nargs="*", help="Command text, e.g. banana left")
    parser.add_argument("--command-topic", default="/task_commands")
    parser.add_argument("--result-topic", default="/task_results")
    parser.add_argument("--loop", action="store_true", help="Keep reading commands after the first publish.")
    return parser.parse_args(rclpy.utilities.remove_ros_args(args=argv or sys.argv)[1:])


def main(argv=None):
    args = parse_args(argv)
    initial_command = " ".join(args.command).strip()
    interactive = args.loop or not initial_command

    rclpy.init(args=argv)
    node = TwoViewGraspClient(args.command_topic, args.result_topic)
    try:
        if interactive:
            print("Enter commands like 'banana left' or 'move coke can to right storage'. Type 'quit' to stop.")
            commands = command_loop(initial_command or None)
        else:
            commands = [initial_command]

        for command in commands:
            node.send(command)

        # publish 후 DDS가 실제로 메시지를 전송할 시간을 줌
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)

        if interactive:
            # 인터랙티브 모드: 결과를 백그라운드에서 계속 수신
            try:
                rclpy.spin(node)
            except KeyboardInterrupt:
                pass

        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
