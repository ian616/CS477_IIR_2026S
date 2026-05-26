#!/usr/bin/env python3

import copy
import json
import math
from pathlib import Path

from ament_index_python.packages import PackageNotFoundError
from ament_index_python.packages import get_package_share_directory

from manip_challenge import move_gripper
from .pose_math import apply_grasp_transform


DEFAULT_GRIPPER_CLOSE_POS = 0.5
GRASP_DATABASE_FILE = "grasp_database.json"
TRANSFORM_FIELDS = ("x", "y", "z", "roll", "pitch", "yaw")


def _database_path():
    try:
        return Path(get_package_share_directory("manip_challenge")) / "config" / GRASP_DATABASE_FILE
    except PackageNotFoundError:
        return Path(__file__).resolve().parents[3] / "config" / GRASP_DATABASE_FILE


def _normalize_obj_name(obj_name):
    return str(obj_name).strip().lower().replace(" ", "_")


def _load_grasp_database():
    with _database_path().open("r", encoding="utf-8") as f:
        return json.load(f)


def _lookup_grasp_config(database, obj_name):
    key = _normalize_obj_name(obj_name)

    if key in database:
        return key, database[key]

    aliases = {
        "meat_can": "meat_can_standing",
    }
    alias_key = aliases.get(key)
    if alias_key in database:
        return alias_key, database[alias_key]

    raise KeyError(key)


def _extract_grasp_transform(config):
    return {field: float(config.get(field, 0.0)) for field in TRANSFORM_FIELDS}


def grasping_item(node, arm, grasp_pose, obj_name, grasp_duration=1.0, timeout=3):
    """Move from approach pose to grasp pose, then close the gripper."""
    gripper_close_pos = DEFAULT_GRIPPER_CLOSE_POS
    corrected_grasp_pose = copy.deepcopy(grasp_pose)

    try:
        database = _load_grasp_database()
        matched_name, grasp_config = _lookup_grasp_config(database, obj_name)
        grasp_transform = _extract_grasp_transform(grasp_config)
        corrected_grasp_pose = apply_grasp_transform(grasp_pose, grasp_transform)

        grasp_value = grasp_config["grasp_value"]
        gripper_close_pos = math.radians(float(grasp_value))

        # Log
        node.get_logger().info(
            f"\n<Grasping> Loaded grasp transform for '{matched_name}': "
            f"x={grasp_transform['x']:.4f}, y={grasp_transform['y']:.4f}, "
            f"z={grasp_transform['z']:.4f}, "
            f"roll={grasp_transform['roll']:.4f}, "
            f"pitch={grasp_transform['pitch']:.4f}, "
            f"yaw={grasp_transform['yaw']:.4f}, "
            f"grasp_value={grasp_value} deg")
    except (FileNotFoundError, KeyError, TypeError, ValueError) as e:
        # Log
        node.get_logger().warn(
            f"<Grasping> Failed to load grasp transform for '{obj_name}' ({e}); "
            f"using original grasp_pose and "
            f"default gripper_close_pos={DEFAULT_GRIPPER_CLOSE_POS}")


    # --- IGNORE ---
    # Remove after the transform values in the database are finalized.
    corrected_grasp_pose = grasp_pose
    # --------------

    # Log
    node.get_logger().info(
        f"<Grasping> Moving to corrected grasp_pose for '{obj_name}': "
        f"x={corrected_grasp_pose.position.x:.4f}, "
        f"y={corrected_grasp_pose.position.y:.4f}, "
        f"z={corrected_grasp_pose.position.z:.4f}")

    # Move the robot arm to the corrected grasp pose
    arm.execute_trajectory([corrected_grasp_pose], durations=[grasp_duration])

    # Log
    node.get_logger().info(
        f"<Grasping> Closing gripper for '{obj_name}' "
        f"with gripper_close_pos={gripper_close_pos}\n")

    # Close the gripper to grasp the item
    return move_gripper.gripper_close(
        node,
        timeout=timeout,
        gripper_close_pos=gripper_close_pos,
    )
