#!/usr/bin/env python3
import time
import copy
import math
import random
import threading

import rclpy
import tf2_geometry_msgs
from geometry_msgs.msg import Pose, PoseStamped
from tf2_ros import Buffer, ConnectivityException, ExtrapolationException, LookupException

from manip_challenge import move_gripper
from .grasping.grasping_item import grasping_item


PLACE_CONFIGS = {
    "left storage": {
        "range_x": [-0.124, 0.117],
        "range_y": [0.381, 0.717],
        "base_z": 0.06,  # 0.66 - 0.6
        "use_grasp_orientation": True,
    },
    "right storage": {
        "range_x": [-0.124, 0.117],
        "range_y": [-0.717, -0.381],
        "base_z": 0.06,  # 0.66 - 0.6
        "use_grasp_orientation": True,
    },
    "bookshelf": {
        "range_x": [0.846, 1.030],
        "range_y": [-0.416, -0.184],
        "base_z": 0.06, # 0.795 - 0.6
        "use_grasp_orientation": False,
    },
}


# Helper function to dynamically calculate rotation time based on the angle difference
def _calc_rot_time(start_angle, target_angle, sec_per_rad=1.2, min_time=0.8):
    """
    Calculates the required rotation time based on the difference between start_angle and target_angle.
    - sec_per_rad: Time allocated per radian (approx. 57 degrees). Lower = faster rotation.
    - min_time: Minimum guaranteed time to prevent sudden jerks or motor overloads.
    """
    delta = abs(target_angle - start_angle)
    return max(min_time, delta * sec_per_rad)


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


def pick_place_storage(node, arm, grasp_pose, destination, obj_name,
                       on_before_idle=None, get_next_pick_data=None,
                       pick_already_done=False):
    pick_pan_angle = math.atan2(grasp_pose.position.y, grasp_pose.position.x)

    if not pick_already_done:
        # 1. [Pick Phase]
        node.get_logger().info(f"Starting PICK phase for {obj_name}...")
        move_gripper.gripper_open(node)

        current_pan = arm.js_joint_position[0]

        approach_pose = copy.deepcopy(grasp_pose)
        approach_pose.position.z += 0.15

        if abs(pick_pan_angle - current_pan) < 0.05:
            arm.execute_trajectory([approach_pose], durations=[1.5])
        else:
            pick_joint = [pick_pan_angle, -math.pi / 2.0, 1., -math.pi / 3., -math.pi / 2., 0.]
            rot_time_pick = _calc_rot_time(current_pan, pick_pan_angle)
            arm.execute_trajectory(
                [pick_joint, approach_pose],
                durations=[rot_time_pick, 1.5],
            )
        # Move to Grasping part
        grasping_item(node, arm, grasp_pose, obj_name)

    # 2. [Place Phase]
    node.get_logger().info(f"Starting PLACE phase. Destination: {destination}")
    config = PLACE_CONFIGS[destination]

    count_attr = f"storage_{destination.replace(' ', '_')}_count"
    if not hasattr(node, count_attr):
        setattr(node, count_attr, 0)
    count = getattr(node, count_attr)

    # Divide storage into 8 slots
    x_start, x_end = config["range_x"][0], config["range_x"][1]
    x_slots = [x_start + (x_end - x_start) * 0.25, x_start + (x_end - x_start) * 0.75]

    y_start, y_end = config["range_y"][0], config["range_y"][1]
    y_slots = [y_start + (y_end - y_start) * (0.125 + 0.25 * i) for i in range(4)]

    lift_pose = copy.deepcopy(grasp_pose)
    lift_pose.position.z += 0.20

    place_pose = Pose()
    place_pose.position.x = x_slots[(count // 4) % 2]
    place_pose.position.y = y_slots[count % 4]
    place_pose.position.z = config["base_z"]
    place_pose.orientation = lift_pose.orientation

    place_pan_angle = math.atan2(place_pose.position.y, place_pose.position.x)
    place_joint = [place_pan_angle, -math.pi / 2.0, 1., -math.pi / 3., -math.pi / 2., 0.]

    place_approach = copy.deepcopy(place_pose)
    place_approach.position.z += 0.15

    # 💡 Apply dynamic rotation time (Pick angle -> Place angle)
    rot_time_place = _calc_rot_time(pick_pan_angle, place_pan_angle)

    # Lift → rotate → approach → place in one smooth trajectory
    # Fire on_before_idle after place rotation ends (lift + rotate = 1.0 + rot_time_place)
    if on_before_idle is not None:
        threading.Timer(1.0 + rot_time_place, on_before_idle).start()
    arm.execute_trajectory(
        [lift_pose, place_joint, place_approach, place_pose],
        durations=[1.0, rot_time_place, 1.5, 1.0],
    )
    move_gripper.gripper_open(node)

    retreat_pose = copy.deepcopy(place_pose)
    retreat_pose.position.z += 0.20

    next_data = get_next_pick_data() if get_next_pick_data is not None else None
    if next_data is not None:
        next_pick_joint = next_data['pick_joint']
        next_approach = next_data['approach_pose']
        next_grasp = next_data['grasp_pose']
        rot_time_next = _calc_rot_time(place_pan_angle, next_pick_joint[0])
        arm.execute_trajectory(
            [retreat_pose, next_pick_joint, next_approach],
            durations=[1.0, rot_time_next, 1.5],
        )
        # Move to Grasping part
        grasping_item(node, arm, next_grasp, next_data.get('obj_name'))
    else:
        idle_joint = [0., -math.pi / 2.0, 1., -math.pi / 3., -math.pi / 2., 0.]
        rot_time_idle = _calc_rot_time(place_pan_angle, 0.0)
        arm.execute_trajectory([retreat_pose, idle_joint], durations=[1.0, rot_time_idle])

    setattr(node, count_attr, count + 1)
    node.get_logger().info("PICK-and-PLACE sequence completed successfully!")


def pick_place_bookshelf(node, arm, grasp_pose, destination, obj_name,
                         on_before_idle=None, get_next_pick_data=None,
                         pick_already_done=False):
    pick_pan_angle = math.atan2(grasp_pose.position.y, grasp_pose.position.x)

    if not pick_already_done:
        # 1. [Pick Phase]
        node.get_logger().info(f"Starting PICK phase for {obj_name}...")
        move_gripper.gripper_open(node)

        current_pan = arm.js_joint_position[0]

        approach_pose = copy.deepcopy(grasp_pose)
        approach_pose.position.z += 0.15

        if abs(pick_pan_angle - current_pan) < 0.05:
            arm.execute_trajectory([approach_pose], durations=[1.5])
        else:
            pick_joint = [pick_pan_angle, -math.pi / 2.0, 1., -math.pi / 3., -math.pi / 2., 0.]
            rot_time_pick = _calc_rot_time(current_pan, pick_pan_angle)
            arm.execute_trajectory(
                [pick_joint, approach_pose],
                durations=[rot_time_pick, 1.5],
            )
        # Move to Grasping part
        grasping_item(node, arm, grasp_pose, obj_name)

    # 2. [Place Phase]
    node.get_logger().info(f"Starting PLACE phase. Destination: {destination}")
    config = PLACE_CONFIGS[destination]

    if not hasattr(node, 'bookshelf_count'):
        node.bookshelf_count = 0

    lift_pose = copy.deepcopy(grasp_pose)
    lift_pose.position.z += 0.1

    # Generate target pose
    place_pose = Pose()
    place_pose.position.x = 0.925

    y_slots = [-0.215, -0.30, -0.385]
    place_pose.position.y = y_slots[node.bookshelf_count % len(y_slots)]

    # Force all objects to the 2nd floor of the bookshelf
    place_pose.position.z = config["base_z"] + 0.15
    place_pose.orientation.x = 0.5
    place_pose.orientation.y = 0.5
    place_pose.orientation.z = 0.5
    place_pose.orientation.w = 0.5

    place_pan_angle = math.atan2(place_pose.position.y, place_pose.position.x)
    place_joint = [place_pan_angle, -math.pi / 2.0, 1., -math.pi / 3., -math.pi / 2., 0.]

    place_approach = copy.deepcopy(place_pose)
    place_approach.position.x -= 0.2

    rot_time_place = _calc_rot_time(pick_pan_angle, place_pan_angle)

    # Lift → rotate → approach → place in one smooth trajectory
    # Fire on_before_idle after place rotation ends (lift + rotate = 1.0 + rot_time_place)
    if on_before_idle is not None:
        threading.Timer(1.0 + rot_time_place, on_before_idle).start()
    arm.execute_trajectory(
        [lift_pose, place_joint, place_approach, place_pose],
        durations=[1.0, rot_time_place, 1.5, 1.0],
    )
    move_gripper.gripper_open(node)

    retreat_pose = copy.deepcopy(place_pose)
    retreat_pose.position.x -= 0.3

    next_data = get_next_pick_data() if get_next_pick_data is not None else None
    if next_data is not None:
        next_pick_joint = next_data['pick_joint']
        next_approach = next_data['approach_pose']
        next_grasp = next_data['grasp_pose']
        rot_time_next = _calc_rot_time(place_pan_angle, next_pick_joint[0])
        arm.execute_trajectory(
            [retreat_pose, next_pick_joint, next_approach],
            durations=[1.3, rot_time_next, 1.5],
        )
        # Move to Grasping part
        grasping_item(node, arm, next_grasp, next_data.get('obj_name'))
    else:
        idle_joint = [0., -math.pi / 2.0, 1., -math.pi / 3., -math.pi / 2., 0.]
        arm.execute_trajectory([retreat_pose, idle_joint], durations=[1.3, 1.5])

    node.bookshelf_count += 1
    node.get_logger().info("PICK-and-PLACE sequence completed successfully!")


def execute_pick_place_sequence(node, arm, grasp_pose, destination, obj_name,
                                on_before_idle=None, get_next_pick_data=None,
                                pick_already_done=False):
    if destination == "bookshelf":
        pick_place_bookshelf(node, arm, grasp_pose, destination, obj_name,
                             on_before_idle, get_next_pick_data, pick_already_done)
    else:
        pick_place_storage(node, arm, grasp_pose, destination, obj_name,
                           on_before_idle, get_next_pick_data, pick_already_done)
