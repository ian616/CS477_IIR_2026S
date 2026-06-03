#!/usr/bin/env python3

import copy
import json
import math
from pathlib import Path

from ament_index_python.packages import PackageNotFoundError
from ament_index_python.packages import get_package_share_directory

from manip_challenge import move_gripper
from .perception_features import extract_perception_features
from .pose_math import apply_grasp_transform


DEFAULT_GRIPPER_CLOSE_POS = 0.5
GRASP_DATABASE_FILE = "grasp_database.json"
TRANSFORM_FIELDS = ("x", "y", "z", "roll", "pitch", "yaw")
STATE_NAMES = ("standing", "lying")


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


def _is_transform_config(config):
    return isinstance(config, dict) and any(field in config for field in TRANSFORM_FIELDS)


def _lookup_object_grasp_configs(database, obj_name):
    key = _normalize_obj_name(obj_name)

    if key in database:
        config = database[key]
        if _is_transform_config(config):
            return key, {"default": config}
        return key, config

    aliases = {
        "meat_can_standing": "meat_can",
        "meat_can_lying": "meat_can",
    }
    alias_key = aliases.get(key)
    if alias_key in database:
        config = database[alias_key]
        if _is_transform_config(config):
            return alias_key, {"default": config}
        return alias_key, config

    state_configs = {
        state: database[f"{key}_{state}"]
        for state in STATE_NAMES
        if f"{key}_{state}" in database
    }
    if state_configs:
        return key, state_configs

    raise KeyError(key)


def _read_bbox_area(config):
    if not isinstance(config, dict):
        return None

    value = config.get("area_m2")
    if value is None:
        value = config.get("bbox_area")
    if value is None:
        return None
    return float(value)


def _select_state_config(object_configs, measured_area):
    if _is_transform_config(object_configs):
        return "default", object_configs, {}

    state_configs = {
        state: object_configs[state]
        for state in STATE_NAMES
        if isinstance(object_configs, dict) and isinstance(object_configs.get(state), dict)
    }
    if not state_configs and isinstance(object_configs, dict):
        state_configs = {
            state: config
            for state, config in object_configs.items()
            if isinstance(config, dict)
        }
    if not state_configs:
        raise KeyError("No standing/lying grasp configs found.")

    bbox_area_by_state = {
        state: bbox_area
        for state, config in state_configs.items()
        for bbox_area in [_read_bbox_area(config)]
        if bbox_area is not None
    }

    if measured_area is not None and bbox_area_by_state:
        measured_area = float(measured_area)
        selected_state = min(
            bbox_area_by_state,
            key=lambda state: (
                abs(measured_area - bbox_area_by_state[state]),
                STATE_NAMES.index(state) if state in STATE_NAMES else 99,
            ),
        )
    elif "standing" in state_configs:
        selected_state = "standing"
    else:
        selected_state = next(iter(state_configs))

    return selected_state, state_configs[selected_state], bbox_area_by_state


def _measured_object_area_from_features(perception_features):
    for key in (
        "mask_projected_area_m2",
        "mask_surface_area_m2",
        "pca_bbox_area_m2",
    ):
        value = perception_features.get(key)
        if value is not None:
            return float(value), key

    width = perception_features.get("pca_bbox_width_m")
    length = perception_features.get("pca_bbox_length_m")
    if width is not None and length is not None:
        return float(width) * float(length), "pca_bbox_width_m*pca_bbox_length_m"

    return None, None


def _extract_grasp_transform(config):
    return {field: float(config.get(field, 0.0)) for field in TRANSFORM_FIELDS}


def grasping_item(node, arm, grasp_pose, obj_name, perception_info=None,
                  grasp_duration=1.0, timeout=3):
    """Move from approach pose to grasp pose, then close the gripper."""
    gripper_close_pos = DEFAULT_GRIPPER_CLOSE_POS
    corrected_grasp_pose = copy.deepcopy(grasp_pose)

    if perception_info and "grasp_selection" not in perception_info and "target_xyz_m" in perception_info:
        perception_features = extract_perception_features({"grasp_selection": perception_info})
    else:
        perception_features = extract_perception_features(perception_info)

    target_position = perception_features["target_position_xyz_m"]
    target_orientation = perception_features["target_orientation_yaw_rad"]
    target_orientation_deg = perception_features["target_orientation_yaw_deg"]
    pca_major_axis = perception_features["pca_major_axis_xy"]
    pca_minor_axis = perception_features["pca_minor_axis_xy"]
    pca_bbox_width = perception_features.get("pca_bbox_width_m")
    pca_bbox_length = perception_features.get("pca_bbox_length_m")
    pca_bbox_area = perception_features.get("pca_bbox_area_m2")
    if pca_bbox_area is None and pca_bbox_width is not None and pca_bbox_length is not None:
        pca_bbox_area = float(pca_bbox_width) * float(pca_bbox_length)

    try:
        database = _load_grasp_database()
        matched_name, object_configs = _lookup_object_grasp_configs(database, obj_name)
        item_state, grasp_config, bbox_area_by_state = _select_state_config(object_configs, pca_bbox_area)
        node.last_grasped_state = item_state
        grasp_transform = _extract_grasp_transform(grasp_config)
        corrected_grasp_pose = apply_grasp_transform(grasp_pose, grasp_transform)

        grasp_value = grasp_config.get("grasp_value")
        if grasp_value is not None:
            gripper_close_pos = math.radians(float(grasp_value))

        # Log
        node.get_logger().info(
            f"\n<Grasping> Loaded grasp transform for '{matched_name}' "
            f"state='{item_state}' using measured_area_m2={measured_area} "
            f"source={measured_area_source}, area_by_state={bbox_area_by_state}: "
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

    # Log
    node.get_logger().info(
        f"<Grasping> Moving to corrected grasp_pose for '{obj_name}': "
        f"x={corrected_grasp_pose.position.x:.4f}, "
        f"y={corrected_grasp_pose.position.y:.4f}, "
        f"z={corrected_grasp_pose.position.z:.4f}")



    # --- IGNORE ---
    # Remove after the transform values in the database are finalized.
    # corrected_grasp_pose = grasp_pose
    # --------------


    # Move the robot arm to the corrected grasp pose
    arm.execute_trajectory([corrected_grasp_pose], durations=[grasp_duration])

    # Log
    node.get_logger().info(
        f"<Grasping> Closing gripper for '{obj_name}' "
        f"with gripper_close_pos={gripper_close_pos}\n")
    
    # [Test] Wait for user confirmation before grasping
    input("Press Enter to Close Gripper...") 

    # Close the gripper to grasp the item
    return move_gripper.gripper_close(
        node,
        timeout=timeout,
        gripper_close_pos=gripper_close_pos,
    )
