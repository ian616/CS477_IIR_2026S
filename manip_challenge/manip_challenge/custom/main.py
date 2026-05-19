#!/usr/bin/env python3
import json
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from riro_srvs.srv import StringString
from tf2_ros import Buffer, TransformListener

from assignment_2.move_joint import ArmClient
from manip_challenge.move_joint import move_joint

from .parsing import parse_task_commands
from .grasping import pick


class TaskManagerNode(Node):
    def __init__(self):
        super().__init__('task_manager_node')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

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
    node = TaskManagerNode()

    arm = ArmClient()
    went_home = False
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)

            if not went_home:
                went_home = True
                move_joint(node, [0., -np.pi / 2.0, 1., -np.pi / 3., -np.pi / 2., 0.])
                continue

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
                pick(node, node.tf_buffer, arm, pose, destination, obj_name)
            else:
                node.get_logger().error(f"Failed to detect '{obj_name}', skipping.")

    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
