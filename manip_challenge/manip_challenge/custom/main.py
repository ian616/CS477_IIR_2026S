#!/usr/bin/env python3
import copy
import json
import math
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
from .grasping import pick, compute_grasp_pose
from .motion import transform_pose, execute_pick_place_sequence


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


class TaskSubscriberNode(Node):
    def __init__(self, task_queue):
        super().__init__('task_subscriber_node')
        self.task_queue = task_queue
        self.create_subscription(String, '/task_commands', self._command_callback, 10)
        self.get_logger().info('Task Subscriber Node is listening...')

    def _command_callback(self, msg):
        self.get_logger().info(f'Command received: {msg.data}')
        tasks = parse_task_commands(msg.data)
        self.get_logger().info(f'Parsed {len(tasks)} tasks: {tasks}')
        self.task_queue.extend(tasks)


class TaskManagerNode(Node):
    def __init__(self):
        super().__init__('task_manager_node')
        self.task_queue = []
        self.get_logger().info('Task Manager Node is Ready!')


def main():
    rclpy.init()

    tf_node = _TFNode()
    perception_node = PerceptionNode()
    node = TaskManagerNode()
    subscriber_node = TaskSubscriberNode(node.task_queue)

    # subscriber_node handles /task_commands in the background executor so
    # topic callbacks fire even while the main thread is blocked on arm motion.
    executor = MultiThreadedExecutor()
    executor.add_node(tf_node)
    executor.add_node(perception_node)
    executor.add_node(subscriber_node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    arm = ArmClient()
    try:
        arm.move_joint([0., -np.pi / 2.0, 1., -np.pi / 3., -np.pi / 2., 0.])

        # pending_future:    async detection future for the next task
        # pending_obj:       object name the future is for
        # pending_pick_done: True when pick phase of next task was already executed inline
        # pending_grasp_pose: base-frame grasp pose used in that inline pick
        pending_future = None
        pending_obj = None
        pending_pick_done = False
        pending_grasp_pose = None

        while rclpy.ok():
            if not node.task_queue:
                time.sleep(0.01)
                continue

            obj_name, destination = node.task_queue.pop(0)
            node.get_logger().info(f"Processing '{obj_name}' → '{destination}'")

            def make_callbacks(task_queue, perc_node, tf_buf, mgr_node, arm_ref):
                def on_before_idle():
                    nonlocal pending_future, pending_obj
                    if task_queue:
                        next_obj, _ = task_queue[0]
                        mgr_node.get_logger().info(
                            f"[on_before_idle] Starting detection for next: '{next_obj}'")
                        pending_future = perc_node.detect_async(next_obj)
                        pending_obj = next_obj

                def get_next_pick_data():
                    """Called after place — collects detection and returns full pick data."""
                    nonlocal pending_future, pending_obj, pending_pick_done, pending_grasp_pose
                    if pending_future is None:
                        mgr_node.get_logger().warn('[get_next_pick_data] pending_future is None')
                        return None
                    if not task_queue:
                        mgr_node.get_logger().warn('[get_next_pick_data] task_queue is empty')
                        return None
                    next_obj, _ = task_queue[0]
                    raw = perc_node.collect(pending_future, next_obj)
                    pending_future = None
                    if raw is None or (raw.position.x == 0.0 and
                                       raw.position.y == 0.0 and
                                       raw.position.z == 0.0):
                        mgr_node.get_logger().warn(f'[get_next_pick_data] detection failed for {next_obj}')
                        pending_obj = None
                        return None
                    pose_base = transform_pose(
                        mgr_node, tf_buf, raw,
                        'camera_color_optical_frame', 'base_link')
                    if pose_base is None:
                        mgr_node.get_logger().warn('[get_next_pick_data] transform failed')
                        pending_obj = None
                        return None
                    goal_pose = compute_grasp_pose(arm_ref, pose_base)
                    approach_pose = copy.deepcopy(goal_pose)
                    approach_pose.position.z += 0.15
                    pan = math.atan2(goal_pose.position.y, goal_pose.position.x)
                    mgr_node.get_logger().info(f'[get_next_pick_data] → {next_obj}, pan={pan:.3f}')
                    pending_pick_done = True
                    pending_grasp_pose = goal_pose
                    return {
                        'pick_joint': [pan, -np.pi / 2.0, 1., -np.pi / 3.0, -np.pi / 2.0, 0.],
                        'approach_pose': approach_pose,
                        'grasp_pose': goal_pose,
                    }

                return on_before_idle, get_next_pick_data

            on_before_idle, get_next_pick_data = make_callbacks(
                node.task_queue, perception_node, tf_node.tf_buffer, node, arm)

            if pending_pick_done and pending_obj == obj_name:
                # Pick phase was already executed inline — go directly to place
                grasp_pose_base = pending_grasp_pose
                pending_pick_done = False
                pending_grasp_pose = None
                pending_obj = None
                node.get_logger().info(f"Pick already done for '{obj_name}', skipping to place.")
                execute_pick_place_sequence(node, arm, grasp_pose_base, destination, obj_name,
                                            on_before_idle=on_before_idle,
                                            get_next_pick_data=get_next_pick_data,
                                            pick_already_done=True)
                continue

            # Detection: pre-fetched future → fresh detection
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
            pick(node, tf_node.tf_buffer, arm, pose, destination, obj_name,
                 on_before_idle=on_before_idle,
                 get_next_pick_data=get_next_pick_data)

    except KeyboardInterrupt:
        pass

    executor.shutdown()
    subscriber_node.destroy_node()
    node.destroy_node()
    tf_node.destroy_node()
    perception_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
