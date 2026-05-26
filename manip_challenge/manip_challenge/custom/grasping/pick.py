#!/usr/bin/env python3
import copy

import numpy as np

from ..motion import execute_pick_place_sequence, transform_pose, wait_for_tf


def pick(node, tf_buffer, arm, pose, destination, obj_name,
         on_before_idle=None, get_next_pick_data=None):
    joint_angles = [0., -np.pi / 2.0, 1., -np.pi / 3., -np.pi / 2., 0.]
    home_pose = arm.fk_request(joint_angles)

    wait_for_tf(node, tf_buffer, 'camera_color_optical_frame', 'base_link')

    pose_in_base = transform_pose(node, tf_buffer, pose, 'camera_color_optical_frame', 'base_link')
    if pose_in_base is None:
        return

    goal_pose = copy.deepcopy(pose_in_base)
    goal_pose.orientation = home_pose.orientation

    # Move to Motion part
    execute_pick_place_sequence(node, arm, goal_pose, destination, obj_name,
                                on_before_idle, get_next_pick_data)
