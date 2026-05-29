#!/usr/bin/env python3
from __future__ import annotations

import copy
from typing import Any

from geometry_msgs.msg import Pose

from manip_challenge.custom.motion import motion as motion_module
from manip_challenge.custom.motion.motion import execute_pick_place_sequence
from manip_challenge.pddl.ros_helpers import pose_to_dict
from .pddl_types import LOCATION_TO_DESTINATION, PlanAction


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

BUFFER_CONFIGS = {
    "buffer1": {
        "range_x": [0.300, 0.430],
        "range_y": [0.185, 0.335],
        "base_z": 0.06,
        "use_grasp_orientation": True,
    },
    "buffer2": {
        "range_x": [0.300, 0.430],
        "range_y": [-0.335, -0.185],
        "base_z": 0.06,
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


class ActionContext:
    def __init__(self, server, on_before_idle=None, get_next_pick_data=None,
                 pick_already_done=False, prefetch_grasp_pose=None, prefetch_class_name=None,
                 on_observe_ready=None):
        self.server = server
        self.on_before_idle = on_before_idle
        self.get_next_pick_data = get_next_pick_data
        self.pick_already_done = pick_already_done
        self.prefetch_grasp_pose = prefetch_grasp_pose
        self.prefetch_class_name = prefetch_class_name
        self.on_observe_ready = on_observe_ready


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
        execute_pick_place_sequence(
            context.server, context.server.arm, prepared["grasp_pose"], destination, class_name,
            on_before_idle=context.on_before_idle,
            get_next_pick_data=context.get_next_pick_data,
            on_observe_ready=context.on_observe_ready,
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
    motion_module.PLACE_CONFIGS.setdefault(buffer_name, BUFFER_CONFIGS[buffer_name])
    fact = state.objects.get(object_name)
    class_name = fact.class_name if fact is not None and fact.class_name else object_name
    if fact is not None and not fact.graspable:
        raise RuntimeError(f"Obstacle {object_name} is not graspable. Note: {fact.error}")
    if context.pick_already_done and context.prefetch_grasp_pose is not None:
        if not context.server.args.dry_run:
            execute_pick_place_sequence(
                context.server, context.server.arm, context.prefetch_grasp_pose, buffer_name, class_name,
                on_before_idle=context.on_before_idle,
                get_next_pick_data=context.get_next_pick_data,
                pick_already_done=True,
                on_observe_ready=context.on_observe_ready,
            )
        return {
            "ok": True,
            "action": action.to_dict(),
            "class_name": class_name,
            "destination": buffer_name,
            "pick_already_done": True,
            "dry_run": bool(context.server.args.dry_run),
        }

    prepared = _prepare_grasp(context.server, class_name, fact=fact)
    if not context.server.args.dry_run:
        # Low-level implementation reused from custom/motion/motion.py.
        execute_pick_place_sequence(
            context.server, context.server.arm, prepared["grasp_pose"], buffer_name, class_name,
            on_before_idle=context.on_before_idle,
            get_next_pick_data=context.get_next_pick_data,
            on_observe_ready=context.on_observe_ready,
        )
    return {
        "ok": True,
        "action": action.to_dict(),
        "class_name": class_name,
        "destination": buffer_name,
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
