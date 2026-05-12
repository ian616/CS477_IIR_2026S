#!/usr/bin/env python3
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from riro_srvs.srv import StringPose
from tf2_ros import Buffer, TransformListener

from assignment_2 import move_joint as mj

from .parsing import parse_task_commands
from .grasping import pick


class TaskManagerNode(Node):
    def __init__(self):
        super().__init__('task_manager_node')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.cli = self.create_client(StringPose, 'detect_objects_with_prompt')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for perception service...')

        self.arm = mj.ArmClient()
        self.req = StringPose.Request()

        self.task_queue = []
        self.create_subscription(String, '/task_commands', self._command_callback, 10)
        self.get_logger().info('Task Manager Node is Ready!')

    def _command_callback(self, msg):
        self.get_logger().info(f'Command received: {msg.data}')
        tasks = parse_task_commands(msg.data)
        self.get_logger().info(f'Parsed {len(tasks)} tasks: {tasks}')
        self.task_queue.extend(tasks)

    def _detect_object(self, obj_name):
        self.req.data = f'Detect a {obj_name} and return [ymin, xmin, ymax, xmax, label]'
        future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()


def main():
    rclpy.init()
    node = TaskManagerNode()
    node.arm.move_joint([0., -np.pi / 2.0, 1., -np.pi / 3., -np.pi / 2., 0.])

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)

            if not node.task_queue:
                continue

            obj_name, destination = node.task_queue.pop(0)
            node.get_logger().info(f"Picking: '{obj_name}'")

            response = node._detect_object(obj_name)
            pose = response.pose if response is not None else None

            if pose is not None and not (
                    pose.position.x == 0.0 and
                    pose.position.y == 0.0 and
                    pose.position.z == 0.0):
                node.get_logger().info(f"Detected '{obj_name}'! Executing pick...")
                pick(node, node.tf_buffer, node.arm, pose)
            else:
                node.get_logger().error(f"Failed to detect '{obj_name}', skipping.")

    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
