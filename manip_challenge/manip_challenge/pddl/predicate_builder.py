#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import replace
from typing import Callable

import numpy as np

from .types import BUFFER_LOCATIONS, Goal, ObjectState, PredicateState, TARGET_OBJECTS
from .utils import pddl_name


MIN_CONFIDENCE = 0.12
MIN_MASK_PIXELS = 80
MIN_FOREGROUND_POINTS = 20
BBOX_OVERLAP_THRESHOLD = 0.08
DEPTH_ORDER_EPS_M = 0.008
SAFE_DISTANCE_M = 0.11
SAFE_PIXEL_DISTANCE = 95.0
# Predicate tuning guide:
# This file is the main swap point for symbolic state judgement.  Geometry,
# masks, depth, and safety distances should be converted to boolean PDDL facts
# here, not inside domain.pddl.
#
# CHANGE PREDICATE LOGIC HERE:
# - visible(o): edit MIN_CONFIDENCE, MIN_MASK_PIXELS, MIN_FOREGROUND_POINTS
#   and the `visible = ...` line in object_from_detection().
# - graspable(o): edit object_from_detection(); currently visible + pose_known.
# - blocks(a,b): edit BBOX_OVERLAP_THRESHOLD, DEPTH_ORDER_EPS_M, and the
#   overlap/depth branch in _annotate_relations().
# - near(a,b) / safe(o): edit SAFE_DISTANCE_M, SAFE_PIXEL_DISTANCE, and the
#   distance branch in _annotate_relations().
# - final PDDL predicate strings: edit _build_predicates().


def _bbox_area(bbox) -> float:
    if not bbox:
        return 0.0
    x1, y1, x2, y2 = bbox
    return float(max(0, x2 - x1) * max(0, y2 - y1))


def _bbox_overlap(a, b) -> float:
    if not a or not b:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    return float(inter) / max(1.0, min(_bbox_area(a), _bbox_area(b)))


def _bbox_center(bbox):
    if not bbox:
        return None
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def _distance(a: ObjectState, b: ObjectState) -> float | None:
    if a.centroid_xyz and b.centroid_xyz:
        return float(np.linalg.norm(np.asarray(a.centroid_xyz[:2]) - np.asarray(b.centroid_xyz[:2])))
    ac = _bbox_center(a.bbox_xyxy)
    bc = _bbox_center(b.bbox_xyxy)
    if ac and bc:
        return math.dist(ac, bc)
    return None


def _choose_ambiguous_blocker(a: ObjectState, b: ObjectState) -> tuple[ObjectState, ObjectState] | None:
    candidates = [obj for obj in (a, b) if obj.graspable]
    if not candidates:
        return None
    candidates.sort(key=lambda obj: (obj.is_target, -obj.confidence, obj.name))
    blocker = candidates[0]
    blocked = b if blocker is a else a
    return blocker, blocked


def _roi_stats(detection: dict) -> tuple[int, int, float | None]:
    det = detection.get("detection") or {}
    roi = detection.get("roi") or {}
    stats = roi.get("stats") or {}
    mask_pixels = int(det.get("mask_pixels", roi.get("mask_pixels_rgb_crop", 0)))
    points = int(roi.get("num_foreground_points", stats.get("foreground_points", 0)))
    depth = stats.get("foreground_z_median")
    return mask_pixels, points, float(depth) if depth is not None else None


def object_from_detection(
    name: str,
    detection: dict | None,
    is_target: bool,
    error: str | None = None,
    class_name: str | None = None,
    instance_index: int | None = None,
) -> ObjectState:
    # Builds per-object base predicates from one perception result.
    #
    # CHANGE SINGLE-OBJECT PREDICATES HERE:
    # - visible(o), pose-known(o), graspable(o), clear(o), safe(o) defaults are
    #   created from one top-view perception result in this function.
    # - If the perception service starts returning a better grasp candidate,
    #   approach score, or object pose validity flag, consume it here and update
    #   graspable/pose_known accordingly.
    name = pddl_name(name)
    class_name = pddl_name(class_name or (detection or {}).get("target") or name)
    if not detection or not detection.get("ok"):
        return ObjectState(
            name=name,
            class_name=class_name,
            instance_index=instance_index,
            is_target=is_target,
            detected=False,
            error=error or (detection or {}).get("error"),
            detection=detection,
        )
    det = detection.get("detection") or {}
    roi = detection.get("roi") or {}
    confidence = float(det.get("confidence", 0.0))
    mask_pixels, points, depth_median = _roi_stats(detection)
    bbox = roi.get("bbox_xyxy") or det.get("bbox_xyxy")
    bbox_tuple = tuple(int(v) for v in bbox) if bbox and len(bbox) == 4 else None
    centroid = detection.get("location_xyz_m") or roi.get("centroid_xyz")
    centroid_tuple = tuple(float(v) for v in centroid) if centroid and len(centroid) >= 3 else None
    visible = confidence >= MIN_CONFIDENCE and mask_pixels >= MIN_MASK_PIXELS and points >= MIN_FOREGROUND_POINTS
    pose_known = centroid_tuple is not None and all(np.isfinite(centroid_tuple))
    graspable = visible and pose_known
    return ObjectState(
        name=name,
        class_name=class_name,
        instance_index=instance_index,
        is_target=is_target,
        detected=True,
        visible=visible,
        pose_known=pose_known,
        graspable=graspable,
        clear=graspable,
        safe=True,
        confidence=confidence,
        mask_pixels=mask_pixels,
        foreground_points=points,
        bbox_xyxy=bbox_tuple,
        centroid_xyz=centroid_tuple,
        depth_median=depth_median,
        detection=detection,
    )


def obstacle_from_yolo(detection: dict, instance_index: int | None = None) -> ObjectState:
    # Builds non-target obstacle facts from YOLO's all_detections list.
    #
    # CHANGE NON-TARGET OBSTACLE PREDICATES HERE:
    # - Right now extra objects from all_detections are bbox-only, so they are
    #   visible but not graspable.
    # - If you add a service/path that returns centroid/foreground points for
    #   soap/book/glue/etc., set pose_known=True and graspable=True here so
    #   move-obstacle-to-buffer can execute in actions.py.
    class_name = pddl_name(detection.get("label", "unknown"))
    name = f"{class_name}_{instance_index}" if instance_index is not None else class_name
    bbox = detection.get("bbox_xyxy")
    return ObjectState(
        name=name,
        class_name=class_name,
        instance_index=instance_index,
        is_target=False,
        detected=True,
        visible=float(detection.get("confidence", 0.0)) >= MIN_CONFIDENCE,
        pose_known=False,
        graspable=False,
        clear=False,
        safe=True,
        confidence=float(detection.get("confidence", 0.0)),
        mask_pixels=int(detection.get("mask_pixels", 0)),
        bbox_xyxy=tuple(int(v) for v in bbox) if bbox and len(bbox) == 4 else None,
        error="bbox-only obstacle; perception service did not provide a grasp pose",
    )


def _merge_all_detections(objects: dict[str, ObjectState], detection: dict) -> None:
    # Adds target-external objects seen in the same top-view YOLO frame.
    # Edit here to filter obstacle classes or rename unknown objects.
    label_counts: dict[str, int] = {}
    for raw in detection.get("all_detections") or []:
        class_name = pddl_name(raw.get("label", "unknown"))
        index = label_counts.get(class_name, 0)
        label_counts[class_name] = index + 1
        if class_name in TARGET_OBJECTS:
            continue
        obstacle = obstacle_from_yolo(raw, instance_index=index)
        if obstacle.name in objects:
            continue
        objects[obstacle.name] = obstacle


def _annotate_relations(objects: dict[str, ObjectState]) -> None:
    # Computes relational predicates from object facts.
    #
    # CHANGE RELATIONAL PREDICATES HERE:
    # - blocks(a,b) is decided from bbox overlap plus depth order.
    # - near(a,b) is decided from centroid distance or bbox-center pixel
    #   distance.
    # - clear(o) and safe(o) are derived from those relations.
    # This is the most important place for clutter/occlusion behavior because
    # it decides whether the planner should move a blocker before a target.
    for obj in objects.values():
        obj.blocks.clear()
        obj.blocked_by.clear()
        obj.near.clear()
        obj.safe = True
        obj.clear = obj.graspable

    values = [obj for obj in objects.values() if obj.visible]
    for i, a in enumerate(values):
        for b in values[i + 1:]:
            overlap = _bbox_overlap(a.bbox_xyxy, b.bbox_xyxy)
            if overlap >= BBOX_OVERLAP_THRESHOLD:
                if a.depth_median is not None and b.depth_median is not None:
                    if a.depth_median < b.depth_median - DEPTH_ORDER_EPS_M:
                        a.blocks.add(b.name)
                        b.blocked_by.add(a.name)
                        b.clear = False
                    elif b.depth_median < a.depth_median - DEPTH_ORDER_EPS_M:
                        b.blocks.add(a.name)
                        a.blocked_by.add(b.name)
                        a.clear = False
                    else:
                        ordered = _choose_ambiguous_blocker(a, b)
                        if ordered is None:
                            a.clear = False
                            b.clear = False
                        else:
                            blocker, blocked = ordered
                            blocker.blocks.add(blocked.name)
                            blocked.blocked_by.add(blocker.name)
                            blocked.clear = False
                else:
                    ordered = _choose_ambiguous_blocker(a, b)
                    if ordered is None:
                        a.clear = False
                        b.clear = False
                    else:
                        blocker, blocked = ordered
                        blocker.blocks.add(blocked.name)
                        blocked.blocked_by.add(blocker.name)
                        blocked.clear = False

            distance = _distance(a, b)
            if distance is None:
                continue
            threshold = SAFE_DISTANCE_M if a.centroid_xyz and b.centroid_xyz else SAFE_PIXEL_DISTANCE
            if distance < threshold:
                a.near.add(b.name)
                b.near.add(a.name)
                # Obstacles can be moved away while near. Targets should not be
                # moved to their goal until non-target clutter around them is cleared.
                # Target-target proximity is handled by moving one clear target first.
                if a.is_target and not b.is_target:
                    a.safe = False
                if b.is_target and not a.is_target:
                    b.safe = False


def _build_predicates(state: PredicateState) -> set[str]:
    # Converts ObjectState fields into PDDL init predicates.
    #
    # CHANGE PDDL FACT EMISSION HERE:
    # - If you add a predicate to domain.pddl, emit its true instances here.
    # - The strings are written into problem.pddl by problem_generator.py.
    predicates = {"handempty"}
    for obj in state.objects.values():
        predicates.add(("target" if obj.is_target else "obstacle") + f" {obj.name}")
        predicates.add(f"at {obj.name} {obj.location}")
        if obj.clear:
            predicates.add(f"clear {obj.name}")
        if obj.graspable:
            predicates.add(f"graspable {obj.name}")
        if obj.safe:
            predicates.add(f"safe {obj.name}")
        for blocked in sorted(obj.blocks):
            predicates.add(f"blocks {obj.name} {blocked}")
        for other in sorted(obj.near):
            predicates.add(f"near {obj.name} {other}")
    for goal in state.unfinished_goals():
        predicates.add(f"goal-at {goal.target_name} {goal.location}")
    predicates.update({"storage left_storage", "storage right_storage", "storage bookshelf"})
    predicates.update(f"buffer {buffer}" for buffer in BUFFER_LOCATIONS)
    predicates.update(f"buffer-free {buffer}" for buffer in state.free_buffers())
    return predicates


def _objects_from_detection(class_name: str, detection: dict | None, is_target: bool) -> list[ObjectState]:
    class_name = pddl_name(class_name)
    objects = []
    for index, instance in enumerate((detection or {}).get("instances") or []):
        instance_index = int(instance.get("instance_index", index))
        instance_name = pddl_name(instance.get("instance_name") or f"{class_name}_{instance_index}")
        objects.append(
            object_from_detection(
                instance_name,
                instance,
                is_target=is_target,
                class_name=class_name,
                instance_index=instance_index,
            )
        )
    if objects:
        return objects
    return [
        object_from_detection(
            class_name,
            detection,
            is_target=is_target,
            class_name=class_name,
            instance_index=None,
        )
    ]


def _goal_instance_score(obj: ObjectState) -> tuple:
    ready = obj.graspable and obj.clear and obj.safe
    return (
        int(ready),
        int(obj.graspable and obj.clear),
        int(obj.graspable),
        int(obj.safe),
        -len(obj.blocked_by),
        -len(obj.near),
        -len(obj.blocks),
        obj.confidence,
        -float(obj.instance_index if obj.instance_index is not None else 9999),
    )


def _bind_goals_to_instances(goals: list[Goal], objects: dict[str, ObjectState], completed: set[str]) -> list[Goal]:
    bound_goals = []
    reserved: set[str] = set()
    for goal in goals:
        if goal.object_name in completed:
            bound_goals.append(goal)
            continue
        candidates = [
            obj
            for obj in objects.values()
            if obj.is_target
            and obj.class_name == goal.object_name
            and obj.name not in reserved
            and obj.detected
        ]
        if not candidates:
            bound_goals.append(replace(goal, bound_object_name=None))
            continue
        candidates.sort(key=_goal_instance_score, reverse=True)
        selected = candidates[0]
        reserved.add(selected.name)
        bound_goals.append(replace(goal, bound_object_name=selected.name))
    return bound_goals


def build_predicate_state(
    goals: list[Goal],
    detect_fn: Callable[[str], dict],
    completed: set[str] | None = None,
    occupied_buffers: set[str] | None = None,
    scan_all_targets: bool = True,
) -> PredicateState:
    # Top-level predicate pipeline called by server.py every planning step:
    #   perception service -> ObjectState -> relations -> PDDL predicates.
    # Edit scan behavior here if you want extra perception sources beyond
    # target detections plus YOLO all_detections.
    objects: dict[str, ObjectState] = {}
    completed = set(completed or set())
    occupied_buffers = set(occupied_buffers or set())
    target_names = set(TARGET_OBJECTS if scan_all_targets else [goal.object_name for goal in goals])
    target_names.update(goal.object_name for goal in goals)
    for name in sorted(target_names):
        if name in completed:
            continue
        try:
            detection = detect_fn(name)
            facts = _objects_from_detection(name, detection, is_target=name in TARGET_OBJECTS)
            for fact in facts:
                objects[fact.name] = fact
            if detection and detection.get("ok"):
                _merge_all_detections(objects, detection)
        except Exception as exc:
            fact = object_from_detection(name, None, is_target=name in TARGET_OBJECTS, class_name=name, error=str(exc))
            objects[fact.name] = fact

    _annotate_relations(objects)
    bound_goals = _bind_goals_to_instances(goals, objects, completed)
    state = PredicateState(goals=bound_goals, objects=objects, completed=completed, occupied_buffers=occupied_buffers)
    state.predicates = _build_predicates(state)
    for obj in objects.values():
        if obj.error:
            state.notes.append(f"{obj.name}: {obj.error}")
    return state
