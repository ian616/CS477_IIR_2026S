#!/usr/bin/env python3
import json
import time
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from riro_srvs.srv import StringString
from tf2_ros import Buffer, TransformListener

from assignment_2.move_joint import ArmClient

from .parsing import parse_task_commands
from .grasping import pick


class _TFNode(Node):
    """Dedicated node for TF — runs in background executor so TF never stalls."""
    def __init__(self):
        super().__init__('tf_listener_node')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)


class PerceptionNode(Node):
    """Dedicated node for perception — runs in background executor so
    detection service calls are processed while the arm is moving."""
    def __init__(self):
        super().__init__('perception_node')
        self.cli = self.create_client(StringString, 'detect_object_rgbd_crop')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for perception service...')
        self.get_logger().info('Perception node ready')

    def detect_async(self, obj_name):
        """Send detection request and return a Future immediately."""
        req = StringString.Request()
        req.data = obj_name
        return self.cli.call_async(req)

    def collect(self, future, obj_name):
        """Block until future is resolved (executor handles it in background).
        Returns Pose or None."""
        while not future.done():
            time.sleep(0.01)
        result = future.result()
        if result is None:
            self.get_logger().error(f"No response for '{obj_name}'")
            return None
        info = json.loads(result.data)
        if not info.get('ok'):
            self.get_logger().error(f"Detection failed for '{obj_name}': {info.get('error')}")
            return None
        xyz = info['location_xyz_m']
        pose = Pose()
        pose.position.x = xyz[0]
        pose.position.y = xyz[1]
        pose.position.z = xyz[2]
        return pose


class TaskManagerNode(Node):
    def __init__(self):
        super().__init__('task_manager_node')

        self.task_queue = []
        self.create_subscription(String, '/task_commands', self._command_callback, 10)

        self.get_logger().info('Task Manager Node is Ready!')

    def _command_callback(self, msg):
        self.get_logger().info(f'Command received: {msg.data}')
        tasks = parse_task_commands(msg.data)
        self.get_logger().info(f'Parsed {len(tasks)} tasks: {tasks}')
        self.task_queue.extend(tasks)


def main():
    rclpy.init()

    tf_node = _TFNode()
    perception_node = PerceptionNode()
    node = TaskManagerNode()

    # tf_node and perception_node share the executor so both stay responsive
    # while the arm is blocking in the main thread.
    executor = MultiThreadedExecutor()
    executor.add_node(tf_node)
    executor.add_node(perception_node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    arm = ArmClient()
    try:
        arm.move_joint([0., -np.pi / 2.0, 1., -np.pi / 3., -np.pi / 2., 0.])

        # Holds a pre-fetched detection future for the current task.
        pending_future = None
        pending_obj = None

        while rclpy.ok():
            if not node.task_queue:
                rclpy.spin_once(node, timeout_sec=0.1)
                continue

            obj_name, destination = node.task_queue.pop(0)
            node.get_logger().info(f"Processing '{obj_name}' → '{destination}'")

            # Collect detection: either pre-fetched (ran during last idle return)
            # or start a fresh blocking detection now.
            if pending_future is not None and pending_obj == obj_name:
                pose = perception_node.collect(pending_future, obj_name)
                pending_future = None
                pending_obj = None
            else:
                pose = perception_node.collect(perception_node.detect_async(obj_name), obj_name)

            if pose is None or (
                    pose.position.x == 0.0 and
                    pose.position.y == 0.0 and
                    pose.position.z == 0.0):
                node.get_logger().error(f"Failed to detect '{obj_name}', skipping.")
                continue

            node.get_logger().info(f"Detected '{obj_name}'! Executing pick...")

            # Build the on_before_idle callback: fires after retreat, before idle
            # return — starts detecting the next object while the arm is moving home.
            def make_callback(task_queue, perc_node):
                def on_before_idle():
                    nonlocal pending_future, pending_obj
                    if task_queue:
                        next_obj, _ = task_queue[0]
                        node.get_logger().info(
                            f"[on_before_idle] Starting detection for next: '{next_obj}'")
                        pending_future = perc_node.detect_async(next_obj)
                        pending_obj = next_obj
                return on_before_idle

            callback = make_callback(node.task_queue, perception_node)
            pick(node, tf_node.tf_buffer, arm, pose, destination, obj_name,
                 on_before_idle=callback)

    except KeyboardInterrupt:
        pass

    executor.shutdown()
    node.destroy_node()
    tf_node.destroy_node()
    perception_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
