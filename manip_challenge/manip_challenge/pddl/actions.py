#!/usr/bin/env python3
from __future__ import annotations

import copy
import math
from typing import Any

from geometry_msgs.msg import Pose
import numpy as np

from manip_challenge.custom.motion import motion as motion_module
from manip_challenge.custom.motion.motion import execute_pick_place_sequence
from manip_challenge.pddl.ros_helpers import pose_to_dict
from .pddl_types import DYNAMIC_BUFFER_LOCATION, LOCATION_TO_DESTINATION, PlanAction


# This file is the main swap point between symbolic PDDL actions and the real
# robot code in custom/.  When you want to change "pick object X" or "move X to
# destination Y", start here before touching planner.py or server.py.
#
# Current low-level reuse:
# - Grasp target binding is prepared in _prepare_grasp().
# - Object-specific grasp correction/gripper close is reused through
#   custom.grasping.grasping_item, called inside execute_pick_place_sequence().
# - Pick/place trajectories and destination placement are reused through
#   custom.motion.motion.execute_pick_place_sequence().
#
# Expected edit flow:
# 1. Change symbolic action schema in domain.pddl / problem generation if needed.
# 2. Add or edit a handler in this file.
# 3. Register the handler in ACTION_HANDLERS.
# 4. Keep new trajectory/gripper code inside custom/grasping or custom/motion
#    when possible, then call it from the handler here.

DYNAMIC_BUFFER_DEFAULT_OBJECT_RADIUS_M = 0.065
DYNAMIC_BUFFER_WALL_CLEARANCE_M = 0.020
DYNAMIC_BUFFER_WORKSPACE_CENTER = (0.550, 0.000)
DYNAMIC_BUFFER_WORKSPACE_SIZE = (0.500, 0.900)
DYNAMIC_BUFFER_WORKSPACE_X = (
    DYNAMIC_BUFFER_WORKSPACE_CENTER[0] - 0.5 * DYNAMIC_BUFFER_WORKSPACE_SIZE[0]
    + DYNAMIC_BUFFER_DEFAULT_OBJECT_RADIUS_M + DYNAMIC_BUFFER_WALL_CLEARANCE_M,
    DYNAMIC_BUFFER_WORKSPACE_CENTER[0] + 0.5 * DYNAMIC_BUFFER_WORKSPACE_SIZE[0]
    - DYNAMIC_BUFFER_DEFAULT_OBJECT_RADIUS_M - DYNAMIC_BUFFER_WALL_CLEARANCE_M,
)
DYNAMIC_BUFFER_WORKSPACE_Y = (
    DYNAMIC_BUFFER_WORKSPACE_CENTER[1] - 0.5 * DYNAMIC_BUFFER_WORKSPACE_SIZE[1]
    + DYNAMIC_BUFFER_DEFAULT_OBJECT_RADIUS_M + DYNAMIC_BUFFER_WALL_CLEARANCE_M,
    DYNAMIC_BUFFER_WORKSPACE_CENTER[1] + 0.5 * DYNAMIC_BUFFER_WORKSPACE_SIZE[1]
    - DYNAMIC_BUFFER_DEFAULT_OBJECT_RADIUS_M - DYNAMIC_BUFFER_WALL_CLEARANCE_M,
)
DYNAMIC_BUFFER_GRID_STEP_M = 0.035
DYNAMIC_BUFFER_MIN_CLEARANCE_M = 0.075
DYNAMIC_BUFFER_BASE_Z = 0.06

BUFFER_CONFIGS = {
    DYNAMIC_BUFFER_LOCATION: {
        "range_x": list(DYNAMIC_BUFFER_WORKSPACE_X),
        "range_y": list(DYNAMIC_BUFFER_WORKSPACE_Y),
        "base_z": DYNAMIC_BUFFER_BASE_Z,
        "use_grasp_orientation": True,
    },
}
# Action editing guide:
# - Add/modify symbolic-to-robot execution handlers in this file.
# - If you add a PDDL action in domain.pddl, add a Python handler below and
#   register it in ACTION_HANDLERS.
# - Keep low-level grasp/move inside existing custom.grasping/custom.motion
#   code when possible; this file should mostly bind symbolic args to those
#   primitives.
# - Buffer place poses are configured through BUFFER_CONFIGS.


def pose_from_xyz(xyz) -> Pose:
    pose = Pose()
    pose.position.x = float(xyz[0])
    pose.position.y = float(xyz[1])
    pose.position.z = float(xyz[2])
    pose.orientation.w = 1.0
    return pose


def _pose_debug_text(pose: Pose) -> str:
    return (
        f"pos=({pose.position.x:.4f}, {pose.position.y:.4f}, {pose.position.z:.4f}) "
        f"quat=({pose.orientation.x:.4f}, {pose.orientation.y:.4f}, "
        f"{pose.orientation.z:.4f}, {pose.orientation.w:.4f})"
    )


def _finite_xyz(values) -> list[float] | None:
    if values is None:
        return None
    try:
        if len(values) < 3:
            return None
        xyz = [float(values[0]), float(values[1]), float(values[2])]
    except (TypeError, ValueError, IndexError):
        return None
    return xyz if all(math.isfinite(v) for v in xyz) else None


def _transform_xyz_to_base(server, xyz, source_frame: str) -> tuple[float, float, float] | None:
    xyz = _finite_xyz(xyz)
    if xyz is None:
        return None
    try:
        pose = server.transform_pose(pose_from_xyz(xyz), source_frame, "base_link")
    except Exception as exc:
        server.get_logger().debug(f"[DynamicBuffer] failed to transform point from {source_frame}: {exc}")
        return None
    return (float(pose.position.x), float(pose.position.y), float(pose.position.z))


def _scene_footprints_from_detection(server, detection: dict | None) -> list[dict[str, Any]]:
    if not detection:
        return []
    source_frame = detection.get("frame_id") or server.args.camera_frame
    footprints: list[dict[str, Any]] = []

    for item in detection.get("all_pca_bboxes") or []:
        if not item.get("ok"):
            continue
        position = item.get("position_point") or {}
        xyz = position.get("grasp_xyz_m") or position.get("target_xyz_m") or position.get("raw_centroid_xyz_m")
        base_xyz = _transform_xyz_to_base(server, xyz, source_frame)
        if base_xyz is None:
            continue
        length = float(item.get("length_m") or 0.0)
        width = float(item.get("width_m") or 0.0)
        radius = max(
            DYNAMIC_BUFFER_DEFAULT_OBJECT_RADIUS_M,
            0.5 * math.hypot(length, width) + DYNAMIC_BUFFER_MIN_CLEARANCE_M,
        )
        footprints.append({
            "label": item.get("label"),
            "xy": base_xyz[:2],
            "radius": radius,
            "source": "all_pca_bboxes",
        })

    if footprints:
        return footprints

    for item in detection.get("all_position_points") or []:
        xyz = item.get("grasp_xyz_m") or item.get("target_xyz_m") or item.get("raw_centroid_xyz_m")
        base_xyz = _transform_xyz_to_base(server, xyz, source_frame)
        if base_xyz is None:
            continue
        footprints.append({
            "label": item.get("label"),
            "xy": base_xyz[:2],
            "radius": DYNAMIC_BUFFER_DEFAULT_OBJECT_RADIUS_M + DYNAMIC_BUFFER_MIN_CLEARANCE_M,
            "source": "all_position_points",
        })

    return footprints


def _scene_footprints(server, state) -> list[dict[str, Any]]:
    footprints: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for obj in state.objects.values():
        detection = obj.detection or {}
        for footprint in _scene_footprints_from_detection(server, detection):
            xy = footprint["xy"]
            key = (
                str(footprint.get("label") or ""),
                int(round(float(xy[0]) * 1000.0)),
                int(round(float(xy[1]) * 1000.0)),
            )
            if key in seen:
                continue
            seen.add(key)
            footprints.append(footprint)

        if obj.centroid_xyz is not None:
            source_frame = detection.get("frame_id") or server.args.camera_frame
            base_xyz = _transform_xyz_to_base(server, obj.centroid_xyz, source_frame)
            if base_xyz is not None:
                footprints.append({
                    "label": obj.class_name or obj.name,
                    "xy": base_xyz[:2],
                    "radius": DYNAMIC_BUFFER_DEFAULT_OBJECT_RADIUS_M + DYNAMIC_BUFFER_MIN_CLEARANCE_M,
                    "source": "object_centroid",
                })
    return footprints


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), float(lower)), float(upper))


def _object_base_xy(server, state, moving_object_name: str, grasp_pose: Pose | None) -> tuple[float, float] | None:
    obj = state.objects.get(moving_object_name)
    if obj is None:
        for candidate in state.objects.values():
            if candidate.class_name == moving_object_name:
                obj = candidate
                break
    if obj is not None:
        detection = obj.detection or {}
        source_frame = detection.get("frame_id") or server.args.camera_frame
        for xyz in (obj.centroid_xyz, obj.grasp_xyz):
            base_xyz = _transform_xyz_to_base(server, xyz, source_frame)
            if base_xyz is not None:
                return base_xyz[:2]
    if grasp_pose is not None:
        return (float(grasp_pose.position.x), float(grasp_pose.position.y))
    return None


def _mirrored_buffer_xy(
    server,
    state,
    moving_object_name: str,
    grasp_pose: Pose | None,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> tuple[list[float] | None, tuple[float, float] | None]:
    object_xy = _object_base_xy(server, state, moving_object_name, grasp_pose)
    if object_xy is None:
        return None, None
    _, center_y = DYNAMIC_BUFFER_WORKSPACE_CENTER
    mirrored = [
        _clamp(object_xy[0], x_range[0], x_range[1]),
        _clamp(2.0 * center_y - object_xy[1], y_range[0], y_range[1]),
    ]
    return mirrored, object_xy


def _candidate_clearance(
    x: float,
    y: float,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    footprints: list[dict[str, Any]],
    avoid_points: list[tuple[float, float]],
) -> tuple[float, float | None]:
    x_min, x_max = x_range
    y_min, y_max = y_range
    edge_clearance = min(x - x_min, x_max - x, y - y_min, y_max - y)
    clearance = edge_clearance
    nearest = None
    for footprint in footprints:
        fx, fy = footprint["xy"]
        distance = math.hypot(x - float(fx), y - float(fy)) - float(footprint["radius"])
        clearance = min(clearance, distance)
        if nearest is None or distance < nearest:
            nearest = distance
    for ax, ay in avoid_points:
        distance = math.hypot(x - ax, y - ay) - DYNAMIC_BUFFER_MIN_CLEARANCE_M
        clearance = min(clearance, distance)
    return float(clearance), float(nearest) if nearest is not None else None


def find_empty_workspace_buffer(server, state, moving_object_name: str, grasp_pose: Pose | None = None) -> dict[str, Any]:
    footprints = _scene_footprints(server, state)
    avoid_points = []
    if grasp_pose is not None:
        avoid_points.append((float(grasp_pose.position.x), float(grasp_pose.position.y)))

    x_range = DYNAMIC_BUFFER_WORKSPACE_X
    y_range = DYNAMIC_BUFFER_WORKSPACE_Y
    x_min, x_max = x_range
    y_min, y_max = y_range
    mirrored_xy, object_xy = _mirrored_buffer_xy(
        server, state, moving_object_name, grasp_pose, x_range, y_range
    )
    if mirrored_xy is not None:
        clearance, nearest = _candidate_clearance(
            mirrored_xy[0], mirrored_xy[1], x_range, y_range, footprints, avoid_points
        )
        if clearance >= 0.0:
            server.get_logger().info(
                "[DynamicBuffer] selected y-mirrored workspace buffer "
                f"{mirrored_xy} from object_xy={object_xy} clearance={clearance:.3f}m"
            )
            return {
                "name": DYNAMIC_BUFFER_LOCATION,
                "moving_object": moving_object_name,
                "policy": "workspace_y_mirror",
                "object_xy": [float(object_xy[0]), float(object_xy[1])] if object_xy is not None else None,
                "mirror_xy": [float(mirrored_xy[0]), float(mirrored_xy[1])],
                "workspace_center": [float(DYNAMIC_BUFFER_WORKSPACE_CENTER[0]), float(DYNAMIC_BUFFER_WORKSPACE_CENTER[1])],
                "workspace_x": [float(x_min), float(x_max)],
                "workspace_y": [float(y_min), float(y_max)],
                "footprint_count": len(footprints),
                "score": float(clearance),
                "clearance_m": float(clearance),
                "nearest_object_clearance_m": nearest,
                "place_xy": [float(mirrored_xy[0]), float(mirrored_xy[1])],
            }

    step = DYNAMIC_BUFFER_GRID_STEP_M
    xs = np.arange(x_min, x_max + 0.5 * step, step, dtype=float)
    ys = np.arange(y_min, y_max + 0.5 * step, step, dtype=float)

    best = None
    for x in xs:
        for y in ys:
            clearance, nearest = _candidate_clearance(x, y, x_range, y_range, footprints, avoid_points)
            if mirrored_xy is not None:
                mirror_distance = math.hypot(x - mirrored_xy[0], y - mirrored_xy[1])
                score = (10.0 if clearance >= 0.0 else 0.0) + clearance - 0.35 * mirror_distance
            else:
                mirror_distance = None
                # Prefer clear spots closer to the robot centerline after satisfying clearance.
                score = clearance - 0.015 * abs(y)
            if best is None or score > best["score"]:
                best = {
                    "score": float(score),
                    "clearance_m": float(clearance),
                    "nearest_object_clearance_m": float(nearest) if nearest is not None else None,
                    "mirror_distance_m": float(mirror_distance) if mirror_distance is not None else None,
                    "place_xy": [float(x), float(y)],
                }

    if best is None:
        raise RuntimeError("Could not sample any dynamic buffer candidate inside workspace.")
    if best["clearance_m"] < 0.0:
        server.get_logger().warn(
            "[DynamicBuffer] no fully clear workspace sample found; using least-crowded "
            f"candidate {best['place_xy']} clearance={best['clearance_m']:.3f}m"
        )
    else:
        server.get_logger().info(
            "[DynamicBuffer] selected workspace buffer "
            f"{best['place_xy']} clearance={best['clearance_m']:.3f}m from {len(footprints)} footprints"
        )
    return {
        "name": DYNAMIC_BUFFER_LOCATION,
        "moving_object": moving_object_name,
        "policy": "workspace_y_mirror_adjusted" if mirrored_xy is not None else "max_clearance",
        "object_xy": [float(object_xy[0]), float(object_xy[1])] if object_xy is not None else None,
        "mirror_xy": [float(mirrored_xy[0]), float(mirrored_xy[1])] if mirrored_xy is not None else None,
        "workspace_center": [float(DYNAMIC_BUFFER_WORKSPACE_CENTER[0]), float(DYNAMIC_BUFFER_WORKSPACE_CENTER[1])],
        "workspace_x": [float(x_min), float(x_max)],
        "workspace_y": [float(y_min), float(y_max)],
        "footprint_count": len(footprints),
        **best,
    }


def dynamic_buffer_config(server, state, object_name: str, grasp_pose: Pose | None = None) -> dict[str, Any]:
    selection = find_empty_workspace_buffer(server, state, object_name, grasp_pose=grasp_pose)
    return {
        "range_x": list(DYNAMIC_BUFFER_WORKSPACE_X),
        "range_y": list(DYNAMIC_BUFFER_WORKSPACE_Y),
        "base_z": DYNAMIC_BUFFER_BASE_Z,
        "use_grasp_orientation": True,
        "place_xy": selection["place_xy"],
        "dynamic_selection": selection,
    }


class ActionContext:
    def __init__(self, server, on_before_idle=None, get_next_pick_data=None,
                 pick_already_done=False, prefetch_grasp_pose=None, prefetch_class_name=None,
                 on_observe_ready=None, skip_observe_after_place=False):
        self.server = server
        self.on_before_idle = on_before_idle
        self.get_next_pick_data = get_next_pick_data
        self.pick_already_done = pick_already_done
        self.prefetch_grasp_pose = prefetch_grasp_pose
        self.prefetch_class_name = prefetch_class_name
        self.on_observe_ready = on_observe_ready
        self.skip_observe_after_place = skip_observe_after_place


def _prepare_grasp(server, object_name: str, fact=None) -> dict[str, Any]:
    # Shared geometric binding step for physical actions:
    #   detection -> grasp target -> TF transform -> grasp pose.
    #
    # CHANGE GRASP SELECTION HERE:
    # - server._compute_grasp_pose() runs the full pipeline: select_grasp_target,
    #   raw-centroid XY correction, PCA axis TF transform, and make_grasp_pose.
    #   All grasp logic lives in server.py / ros_helpers.py.
    #
    # Do not add raw gripper/trajectory code here unless there is no existing
    # primitive to reuse.  Prefer calling custom.grasping/custom.motion helpers.
    detection = fact.detection if fact is not None and fact.detection else server.detect_object(object_name)
    if not detection or not detection.get("ok"):
        raise RuntimeError(f"No usable detection for {object_name}: {(detection or {}).get('error')}")
    grasp_pose, selection = server._compute_grasp_pose(detection, object_name)
    if server.args.debug:
        print(
            f"[grasp] _prepare_grasp for {object_name}: "
            f"source={selection.get('grasp_pose_source')}, "
            f"method={selection.get('method')}, "
            f"grasp_pose={_pose_debug_text(grasp_pose)} ready!",
            flush=True,
        )
    approach_pose = copy.deepcopy(grasp_pose)
    approach_pose.position.z += server.args.approach_height
    source_frame = detection.get("frame_id") or server.args.camera_frame
    detected_pose = pose_from_xyz(selection.get("grasp_pose_xyz_m") or selection["target_xyz_m"])
    return {
        "detection": detection,
        "source_frame": source_frame,
        "grasp_selection": selection,
        "detected_pose": detected_pose,
        "grasp_pose": grasp_pose,
        "approach_pose": approach_pose,
    }


def move_target_to_goal(context: ActionContext, action: PlanAction, state) -> dict[str, Any]:
    # Handler for PDDL action:
    #   (move-target-to-goal ?o ?from ?to)


    object_name, _from, to = action.args
    destination = LOCATION_TO_DESTINATION[to]
    fact = state.objects.get(object_name)
    class_name = fact.class_name if fact is not None and fact.class_name else object_name

    if context.pick_already_done and context.prefetch_grasp_pose is not None:
        # Pick was executed inline at the end of the previous place trajectory.
        if not context.server.args.dry_run:
            execute_pick_place_sequence(
                context.server, context.server.arm, context.prefetch_grasp_pose, destination, class_name,
                on_before_idle=context.on_before_idle,
                get_next_pick_data=context.get_next_pick_data,
                pick_already_done=True,
                on_observe_ready=context.on_observe_ready,
                skip_observe_after_place=context.skip_observe_after_place,
            )
        return {
            "ok": True,
            "action": action.to_dict(),
            "class_name": class_name,
            "destination": destination,
            "pick_already_done": True,
            "dry_run": bool(context.server.args.dry_run),
        }

    prepared = _prepare_grasp(context.server, class_name, fact=fact)
    if not context.server.args.dry_run:
        # 여기다 바뀐 grasp 로직 추가하시면 됩니다!!!
        # You can add the changed grasp logic here!!!
        # if object_name == "banana" 뭐 이런 느낌으로
        if context.server.args.debug:
            print(
                f"[grasp] move_target_to_goal passes pose to motion for {class_name}: "
                f"{_pose_debug_text(prepared['grasp_pose'])}; "
                f"database correction will be applied inside grasping_item ready!",
                flush=True,
            )
        execute_pick_place_sequence(
            context.server, context.server.arm, prepared["grasp_pose"], destination, class_name,
            on_before_idle=context.on_before_idle,
            get_next_pick_data=context.get_next_pick_data,
            on_observe_ready=context.on_observe_ready,
            skip_observe_after_place=context.skip_observe_after_place,
            perception_info=prepared["grasp_selection"],
        )
    return {
        "ok": True,
        "action": action.to_dict(),
        "class_name": class_name,
        "destination": destination,
        "grasp_selection": prepared["grasp_selection"],
        "detected_pose_in_source": pose_to_dict(prepared["detected_pose"]),
        "grasp_pose_in_base": pose_to_dict(prepared["grasp_pose"]),
        "dry_run": bool(context.server.args.dry_run),
    }


def move_obstacle_to_buffer(context: ActionContext, action: PlanAction, state) -> dict[str, Any]:
    # Handler for PDDL action:
    #   (move-obstacle-to-buffer ?o ?from ?buf)
    #
    # 이건 물건 치우는건데 grasp 준비는 move-target-to-goal이랑 거의 비슷할거에요.  재사용하시면 됩니다.
    # This is for moving obstacles, but the grasp preparation should be similar to move-target-to-goal. You can reuse it.
    object_name, _from, buffer_name = action.args
    if buffer_name not in BUFFER_CONFIGS:
        raise RuntimeError(f"Unknown buffer location: {buffer_name}")
    fact = state.objects.get(object_name)
    class_name = fact.class_name if fact is not None and fact.class_name else object_name
    if fact is not None and not fact.graspable:
        raise RuntimeError(f"Obstacle {object_name} is not graspable. Note: {fact.error}")
    if context.pick_already_done and context.prefetch_grasp_pose is not None:
        buffer_config = (
            dynamic_buffer_config(context.server, state, object_name, grasp_pose=context.prefetch_grasp_pose)
            if buffer_name == DYNAMIC_BUFFER_LOCATION else copy.deepcopy(BUFFER_CONFIGS[buffer_name])
        )
        motion_module.PLACE_CONFIGS[buffer_name] = buffer_config
        if not context.server.args.dry_run:
            execute_pick_place_sequence(
                context.server, context.server.arm, context.prefetch_grasp_pose, buffer_name, class_name,
                on_before_idle=context.on_before_idle,
                get_next_pick_data=context.get_next_pick_data,
                pick_already_done=True,
                on_observe_ready=context.on_observe_ready,
                skip_observe_after_place=context.skip_observe_after_place,
            )
        return {
            "ok": True,
            "action": action.to_dict(),
            "class_name": class_name,
            "destination": buffer_name,
            "buffer_config": buffer_config,
            "dynamic_buffer": buffer_config.get("dynamic_selection"),
            "pick_already_done": True,
            "dry_run": bool(context.server.args.dry_run),
        }

    prepared = _prepare_grasp(context.server, class_name, fact=fact)
    buffer_config = (
        dynamic_buffer_config(context.server, state, object_name, grasp_pose=prepared["grasp_pose"])
        if buffer_name == DYNAMIC_BUFFER_LOCATION else copy.deepcopy(BUFFER_CONFIGS[buffer_name])
    )
    motion_module.PLACE_CONFIGS[buffer_name] = buffer_config
    if not context.server.args.dry_run:
        # Low-level implementation reused from custom/motion/motion.py.
        if context.server.args.debug:
            print(
                f"[grasp] move_obstacle_to_buffer passes pose to motion for {class_name}: "
                f"{_pose_debug_text(prepared['grasp_pose'])}; "
                f"database correction will be applied inside grasping_item ready!",
                flush=True,
            )
        execute_pick_place_sequence(
            context.server, context.server.arm, prepared["grasp_pose"], buffer_name, class_name,
            on_before_idle=context.on_before_idle,
            get_next_pick_data=context.get_next_pick_data,
            on_observe_ready=context.on_observe_ready,
            skip_observe_after_place=context.skip_observe_after_place,
            perception_info=prepared["grasp_selection"],
        )
    return {
        "ok": True,
        "action": action.to_dict(),
        "class_name": class_name,
        "destination": buffer_name,
        "buffer_config": buffer_config,
        "dynamic_buffer": buffer_config.get("dynamic_selection"),
        "grasp_selection": prepared["grasp_selection"],
        "detected_pose_in_source": pose_to_dict(prepared["detected_pose"]),
        "grasp_pose_in_base": pose_to_dict(prepared["grasp_pose"]),
        "dry_run": bool(context.server.args.dry_run),
    }


ACTION_HANDLERS = {
    # Register new PDDL action handlers here.  The key must exactly match the
    # action name emitted by domain.pddl / planner.py, for example:
    #   (:action move-target-to-goal ...) -> "move-target-to-goal"
    "move-target-to-goal": move_target_to_goal,
    "move-obstacle-to-buffer": move_obstacle_to_buffer,
}


def execute_action(context: ActionContext, action: PlanAction, state) -> dict[str, Any]:
    handler = ACTION_HANDLERS.get(action.name)
    if handler is None:
        raise RuntimeError(f"No action handler for symbolic action: {action.name}")
    return handler(context, action, state)
