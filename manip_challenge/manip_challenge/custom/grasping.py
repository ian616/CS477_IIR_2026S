#!/usr/bin/env python3
import copy

import numpy as np
from manip_challenge import move_gripper

from .motion import transform_pose, wait_for_tf


def pick(node, tf_buffer, arm, pose):
    """Open gripper, move to object pose, grasp, and lift."""
    move_gripper.gripper_open(node)

    joint_angles = [0., -np.pi / 2.0, 1., -np.pi / 3., -np.pi / 2., 0.]
    pose_base_link_to_gripper = arm.fk_request(joint_angles)

    wait_for_tf(node, tf_buffer, 'wrist_camera_color_optical_frame', 'base_link')

    pose_in_base = transform_pose(
        node, tf_buffer, pose, 'wrist_camera_color_optical_frame', 'base_link')
    if pose_in_base is None:
        node.get_logger().error("Transform failed, skipping pick.")
        return

    goal_pose = copy.copy(pose_in_base)
    goal_pose.orientation = pose_base_link_to_gripper.orientation

    goal_pose.position.y -= 0.01
    goal_pose.position.z += 0.1
    arm.move_pose(goal_pose)

    goal_pose.position.z -= 0.11
    arm.move_pose(goal_pose)

    move_gripper.gripper_close(node, force=0.5, gripper_close_pos=0.5)

    goal_pose.position.z += 0.2
    arm.move_pose(goal_pose)

    move_gripper.gripper_open(node)
