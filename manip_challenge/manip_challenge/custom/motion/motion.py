#!/usr/bin/env python3
import time
import copy
import math
import random
import threading

import numpy as np

import rclpy
import tf2_geometry_msgs
from geometry_msgs.msg import Pose, PoseStamped
from tf2_ros import Buffer, ConnectivityException, ExtrapolationException, LookupException

from manip_challenge import move_gripper
from ..grasping.grasping_item import (
    grasping_item,
    _load_grasp_database,
    _lookup_object_grasp_configs,
    _select_state_config,
    _extract_grasp_transform,
)
from ..grasping.pose_math import rotate_vector
from ..grasping.perception_features import extract_perception_features


DEFAULT_HOME_JOINTS = [0.0, -math.pi / 2.0, 1.0, -1.0, -math.pi / 2.0, 0.0]
DEFAULT_OBSERVE_JOINTS = list(DEFAULT_HOME_JOINTS)
DEFAULT_OBSERVE_X_OFFSET_M = 0.15
DEFAULT_OBSERVE_RETREAT_DURATION_S = 2.0

_BOOKSHELF_WRIST_FLIP_OBJECTS = frozenset({'banana', 'hammer'})


def _needs_wrist_flip(obj_name):
    base = str(obj_name).strip().lower().split('_')[0]
    return base in _BOOKSHELF_WRIST_FLIP_OBJECTS


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
def _calc_rot_time(start_angle, target_angle, sec_per_rad=1.2, min_time=0.8, max_time=None):
    """
    Calculates the required rotation time based on the difference between start_angle and target_angle.
    - sec_per_rad: Time allocated per radian (approx. 57 degrees). Lower = faster rotation.
    - min_time: Minimum guaranteed time to prevent sudden jerks or motor overloads.
    - max_time: Optional upper clamp.
    """
    delta = abs(target_angle - start_angle)
    t = max(min_time, delta * sec_per_rad)
    return min(t, max_time) if max_time is not None else t


def _observe_joints(node):
    args = getattr(node, "args", None)
    joints = getattr(args, "observe_joints", None)
    return list(joints) if joints is not None else list(DEFAULT_OBSERVE_JOINTS)


def _home_joints(node):
    args = getattr(node, "args", None)
    joints = getattr(args, "home_joints", None)
    return list(joints) if joints is not None else list(DEFAULT_HOME_JOINTS)


def _observe_x_offset(node):
    args = getattr(node, "args", None)
    return float(getattr(args, "observe_x_offset", DEFAULT_OBSERVE_X_OFFSET_M))


def _observe_retreat_duration(node):
    args = getattr(node, "args", None)
    return float(getattr(args, "observe_retreat_duration", DEFAULT_OBSERVE_RETREAT_DURATION_S))


def _move_to_observe_and_notify(node, arm, retreat_pose, place_pan_angle, on_observe_ready, retreat_duration, observe_rot_duration=None):
    observe_joint = _observe_joints(node)
    rot_time_observe = observe_rot_duration if observe_rot_duration is not None else _calc_rot_time(place_pan_angle, observe_joint[0])
    observe_pose = copy.deepcopy(arm.fk_request(observe_joint, attach_tool=True))
    observe_pose.position.x -= _observe_x_offset(node)
    arm.execute_trajectory(
        [retreat_pose, observe_joint, observe_pose],
        durations=[retreat_duration, rot_time_observe, _observe_retreat_duration(node)],
    )
    if on_observe_ready is not None:
        try:
            on_observe_ready()
        except Exception as exc:
            node.get_logger().warn(f"observe-ready callback failed: {exc}")


def _move_to_home_and_notify(node, arm, retreat_pose, place_pan_angle, on_observe_ready, retreat_duration):
    home_joint = _home_joints(node)
    rot_time_home = _calc_rot_time(place_pan_angle, home_joint[0])
    arm.execute_trajectory([retreat_pose, home_joint], durations=[retreat_duration, rot_time_home])
    if on_observe_ready is not None:
        try:
            on_observe_ready()
        except Exception as exc:
            node.get_logger().warn(f"home-ready callback failed: {exc}")


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
                       pick_already_done=False, perception_info=None,
                       on_observe_ready=None, skip_observe_after_place=False):
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

        # [Test] Wait for user confirmation before grasping
        input("Press Enter to GRASP...") 

        # Move to Grasping part
        grasping_item(node, arm, grasp_pose, obj_name, perception_info=perception_info)

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

    slot_count = 1 if obj_name == 'hammer' else count
    place_pose = Pose()
    place_pose.position.x = x_slots[(slot_count // 4) % 2]
    place_pose.position.y = y_slots[slot_count % 4]
    place_pose.position.z = config["base_z"] + grasp_pose.position.z + 0.15
    place_pose.orientation = lift_pose.orientation

    place_pan_angle = math.atan2(place_pose.position.y, place_pose.position.x)
    place_joint = [place_pan_angle, -math.pi / 2.0, 1., -math.pi / 3., -math.pi / 2., 0.]

    place_approach = copy.deepcopy(place_pose)
    place_approach.position.z += 0.15

    # 💡 Apply dynamic rotation time (Pick angle -> Place angle)
    rot_time_place = _calc_rot_time(pick_pan_angle, place_pan_angle)

    # Move deterministically to the storage approach pose first.  Once the arm is
    # actually above storage, trigger the prefetch scan before descending.
    arm.execute_trajectory(
        [lift_pose, place_joint, place_approach],
        durations=[1.0, rot_time_place, 1.5],
    )
    if on_before_idle is not None:
        try:
            on_before_idle()
        except Exception as exc:
            node.get_logger().warn(f"storage-place callback failed: {exc}")
    arm.execute_trajectory([place_pose], durations=[1.0])
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

        # [Test] Wait for user confirmation before grasping
        input("Press Enter to GRASP...") 

        # Move to Grasping part
        grasping_item(
            node,
            arm,
            next_grasp,
            next_data.get('obj_name'),
            perception_info=next_data.get('perception_info'),
        )

    else:
        if skip_observe_after_place:
            node.get_logger().info("Skipping observe pose after storage-place prefetch; retreating above storage.")
            arm.execute_trajectory([retreat_pose], durations=[1.0])
        else:
            _move_to_observe_and_notify(node, arm, retreat_pose, place_pan_angle, on_observe_ready, retreat_duration=1.0)

    setattr(node, count_attr, count + 1)
    node.get_logger().info("PICK-and-PLACE sequence completed successfully!")


def pick_place_bookshelf(node, arm, grasp_pose, destination, obj_name,
                         on_before_idle=None, get_next_pick_data=None,
                         pick_already_done=False, perception_info=None,
                         on_observe_ready=None, skip_observe_after_place=False):
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

        # [Test] Wait for user confirmation before grasping
        input("Press Enter to GRASP...") 
        
        # Move to Grasping part
        grasping_item(node, arm, grasp_pose, obj_name, perception_info=perception_info)

    # 2. [Place Phase]
    node.get_logger().info(f"Starting PLACE phase. Destination: {destination}")
    config = PLACE_CONFIGS[destination]

    if not hasattr(node, 'bookshelf_count'):
        node.bookshelf_count = 0

    lift_pose = copy.deepcopy(grasp_pose)
    lift_pose.position.z += 0.1

    # Generate target pose
    place_pose = Pose()
    place_pose.position.x = 0.970

    y_slots = [-0.215, -0.30, -0.385]
    place_pose.position.y = y_slots[node.bookshelf_count % len(y_slots)]

    # Get database offset (grasp local frame) to correctly adjust place z/y.
    # rotate_vector(place_q, db_offset) converts the grasp-frame offset into
    # world-frame displacement at place time, without relying on grasp_pose ≈ centroid.
    db_offset = np.zeros(3)
    try:
        perception_features = extract_perception_features(perception_info)
        pca_bbox_area = perception_features.get("pca_bbox_area_m2")
        w = perception_features.get("pca_bbox_width_m")
        l = perception_features.get("pca_bbox_length_m")
        if pca_bbox_area is None and w is not None and l is not None:
            pca_bbox_area = float(w) * float(l)
        database = _load_grasp_database()
        _, object_configs = _lookup_object_grasp_configs(database, obj_name)
        _, grasp_config, _ = _select_state_config(object_configs, pca_bbox_area)
        t = _extract_grasp_transform(grasp_config)
        db_offset = np.array([t["x"], t["y"], t["z"]], dtype=float)
    except Exception as e:
        print(f"[bookshelf] db_offset lookup failed: {e}", flush=True)
    print(f"[bookshelf] db_offset={db_offset}", flush=True)

    if _needs_wrist_flip(obj_name):
        place_q = np.array([0.0, 0.7071, 0.0, 0.7071])  # (x,y,z,w)
        rotated = rotate_vector(place_q, db_offset)
        print(f"[bookshelf] wrist_flip rotated={rotated}  -rotated[1]={-rotated[1]:.4f}", flush=True)
        place_pose.position.z = config["base_z"] + 0.16
        place_pose.position.y += float(rotated[1])
        place_pose.orientation.x = 0.0
        place_pose.orientation.y = 0.7071
        place_pose.orientation.z = 0.0
        place_pose.orientation.w = 0.7071
    else:
        place_q = np.array([0.5, 0.5, 0.5, 0.5])  # (x,y,z,w)
        rotated = rotate_vector(place_q, db_offset)
        delta_z = float(rotated[2])
        print(f"[bookshelf] normal rotated={rotated}  delta_z={delta_z:.4f}", flush=True)
        place_pose.position.z = config["base_z"] + 0.18 + delta_z
        place_pose.orientation.x = 0.5
        place_pose.orientation.y = 0.5
        place_pose.orientation.z = 0.5
        place_pose.orientation.w = 0.5

    wrist_angle = math.pi / 2 if _needs_wrist_flip(obj_name) else 0.0
    place_pan_angle = math.atan2(place_pose.position.y, place_pose.position.x)
    place_joint = [place_pan_angle, -math.pi / 2.0, 1., -math.pi / 3., -math.pi / 2., wrist_angle]

    place_approach = copy.deepcopy(place_pose)
    place_approach.position.x -= 0.2

    rot_time_place = _calc_rot_time(pick_pan_angle, place_pan_angle, sec_per_rad=1.8, min_time=1.0, max_time=2.5)

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
        arm.execute_trajectory(
            [retreat_pose, next_pick_joint, next_approach],
            durations=[1.5, 1.5, 1.5],
        )

        # [Test] Wait for user confirmation before grasping
        input("Press Enter to GRASP...") 

        # Move to Grasping part
        grasping_item(
            node,
            arm,
            next_grasp,
            next_data.get('obj_name'),
            perception_info=next_data.get('perception_info'),
        )
        
    else:
        if skip_observe_after_place:
            arm.execute_trajectory([retreat_pose], durations=[1.5])
        else:
            _move_to_observe_and_notify(node, arm, retreat_pose, place_pan_angle, on_observe_ready, retreat_duration=1.5)

    node.bookshelf_count += 1
    node.get_logger().info("PICK-and-PLACE sequence completed successfully!")


def execute_pick_place_sequence(node, arm, grasp_pose, destination, obj_name,
                                on_before_idle=None, get_next_pick_data=None,
                                pick_already_done=False, perception_info=None,
                                on_observe_ready=None, skip_observe_after_place=False):
    if destination == "bookshelf":
        pick_place_bookshelf(node, arm, grasp_pose, destination, obj_name,
                             on_before_idle, get_next_pick_data, pick_already_done,
                             perception_info=perception_info,
                             on_observe_ready=on_observe_ready,
                             skip_observe_after_place=skip_observe_after_place)
    else:
        pick_place_storage(node, arm, grasp_pose, destination, obj_name,
                           on_before_idle, get_next_pick_data, pick_already_done,
                           perception_info=perception_info,
                           on_observe_ready=on_observe_ready,
                           skip_observe_after_place=skip_observe_after_place)
