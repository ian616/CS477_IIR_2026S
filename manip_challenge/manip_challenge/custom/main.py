#!/usr/bin/env python3
import json
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


class TaskManagerNode(Node):
    def __init__(self):
        super().__init__('task_manager_node')

        self.cli = self.create_client(StringString, 'detect_object_rgbd_crop')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for perception service...')

        self.task_queue = []
        self.create_subscription(String, '/task_commands', self._command_callback, 10)

        self.get_logger().info('Task Manager Node is Ready!')

    def _command_callback(self, msg):
        self.get_logger().info(f'Command received: {msg.data}')
        tasks = parse_task_commands(msg.data)
        self.get_logger().info(f'Parsed {len(tasks)} tasks: {tasks}')
        self.task_queue.extend(tasks)

    def _detect_object(self, obj_name):
        req = StringString.Request()
        req.data = obj_name
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if result is None:
            return None
        info = json.loads(result.data)
        if not info.get('ok'):
            self.get_logger().error(f"Detection failed: {info.get('error')}")
            return None
        xyz = info['location_xyz_m']
        pose = Pose()
        pose.position.x = xyz[0]
        pose.position.y = xyz[1]
        pose.position.z = xyz[2]
        return pose


def main():
    rclpy.init()

    tf_node = _TFNode()
    node = TaskManagerNode()

    # Only tf_node goes in the executor so TF stays fresh during blocking calls.
    # node and arm stay outside — library functions (move_gripper, ArmClient.send_goal)
    # call spin_until_future_complete on them directly, which requires they are not
    # already owned by an executor.
    executor = MultiThreadedExecutor()
    executor.add_node(tf_node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    arm = ArmClient()
    try:
        arm.move_joint([0., -np.pi / 2.0, 1., -np.pi / 3., -np.pi / 2., 0.])

        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)

            if not node.task_queue:
                continue

            obj_name, destination = node.task_queue.pop(0)
            node.get_logger().info(f"Picking: '{obj_name}'")

            pose = node._detect_object(obj_name)

            if pose is not None and not (
                    pose.position.x == 0.0 and
                    pose.position.y == 0.0 and
                    pose.position.z == 0.0):
                node.get_logger().info(f"Detected '{obj_name}'! Executing pick...")
                pick(node, tf_node.tf_buffer, arm, pose, destination, obj_name)
            else:
                node.get_logger().error(f"Failed to detect '{obj_name}', skipping.")

    except KeyboardInterrupt:
        pass

    executor.shutdown()
    node.destroy_node()
    tf_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
