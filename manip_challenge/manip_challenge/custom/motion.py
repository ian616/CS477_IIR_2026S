#!/usr/bin/env python3
import time
import copy

import rclpy
import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, ConnectivityException, ExtrapolationException, LookupException

from manip_challenge import move_gripper

def transform_pose(node, tf_buffer, pose, source_frame, target_frame):
    """Transform a Pose from source_frame to target_frame. Returns None on failure."""
    try:
        t_pose = PoseStamped()
        t_pose.header.frame_id = source_frame
        t_pose.pose = pose
        transformed = tf_buffer.transform(
            t_pose, target_frame, timeout=rclpy.duration.Duration(seconds=1.0))
        node.get_logger().info(f"Transformed to {target_frame}: {transformed.pose.position}")
        return transformed.pose
    except (LookupException, ConnectivityException, ExtrapolationException) as e:
        node.get_logger().error(f"TF transform failed: {e}")
        return None


def wait_for_tf(node, tf_buffer, source_frame, target_frame):
    """Block until the TF transform between two frames becomes available."""
    while rclpy.ok():
        if tf_buffer.can_transform(
                target_frame, source_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)):
            break
        time.sleep(0.1)

def execute_pick_sequence(node, arm, grasp_pose):
    move_gripper.gripper_open(node)

    approach_pose = copy.copy(grasp_pose)
    approach_pose.position.z += 0.1
    arm.move_pose(approach_pose)

    grasp_pose.position.z -= 0.01
    arm.move_pose(grasp_pose)

    move_gripper.gripper_close(node, force=0.5, gripper_close_pos=0.5)

    lift_pose = copy.copy(grasp_pose)
    lift_pose.position.z += 0.2
    arm.move_pose(lift_pose)
