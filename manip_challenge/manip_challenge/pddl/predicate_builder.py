#!/usr/bin/env python3
from __future__ import annotations

import logging
import math
from dataclasses import replace
from typing import Callable

import numpy as np

try:
    import cv2 as _cv2
except ImportError:
    _cv2 = None

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

from .pddl_types import BUFFER_LOCATIONS, Goal, ObjectState, PredicateState, KNOWN_OBJECTS
from .utils import pddl_name, PDDL_DIR


MIN_CONFIDENCE = 0.12
MIN_MASK_PIXELS = 80
MIN_FOREGROUND_POINTS = 20
BBOX_OVERLAP_THRESHOLD = 0.08
DEPTH_ORDER_EPS_M = 0.008
SAFE_DISTANCE_M = 0.11
SAFE_PIXEL_DISTANCE = 10.0
ACTIVE_SCENE_REGIONS = {"active_table", "active_workspace"}
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



def _dist_point_to_bbox(px: float, py: float, bbox) -> float:
    x1, y1, x2, y2 = bbox
    dx = max(x1 - px, 0.0, px - x2)
    dy = max(y1 - py, 0.0, py - y2)
    return math.sqrt(dx * dx + dy * dy)


def _load_mask_pixels(mask_path: str) -> np.ndarray | None:
    """Load a binary mask PNG and return Nx2 float32 array of (x, y) nonzero pixel coords."""
    if not mask_path or _cv2 is None:
        return None
    try:
        mask = _cv2.imread(mask_path, _cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return None
        return np.column_stack([xs, ys]).astype(np.float32)
    except Exception:
        return None


def _grasp_pixel(target: ObjectState) -> tuple[float, float] | None:
    """Grasp point in image pixels: mask centroid of mask_full, fallback to bbox center."""
    files = (target.detection or {}).get("files") or {}
    for key in ("mask_full", "mask"):
        pixels = _load_mask_pixels(files.get(key))
        if pixels is not None:
            return float(np.median(pixels[:, 0])), float(np.median(pixels[:, 1]))
    return _bbox_center(target.bbox_xyxy)


def _dist_point_to_seg(px: float, py: float, mask_path: str) -> float | None:
    """Minimum pixel distance from (px, py) to any nonzero pixel in mask_path."""
    pixels = _load_mask_pixels(mask_path)
    if pixels is None:
        return None
    diff = pixels - np.array([px, py], dtype=np.float32)
    return float(np.min(np.hypot(diff[:, 0], diff[:, 1])))


def _grasp_to_obstacle_dist(target: ObjectState, obstacle: ObjectState) -> float | None:
    """Min distance from target's grasp pixel to obstacle's segmentation mask (or bbox edge)."""
    gp = _grasp_pixel(target)
    if gp is None:
        return None
    files = (obstacle.detection or {}).get("files") or {}
    for key in ("mask_full", "mask"):
        mask_path = files.get(key)
        if mask_path:
            d = _dist_point_to_seg(gp[0], gp[1], mask_path)
            if d is not None:
                return d
    if not obstacle.bbox_xyxy:
        return None
    return _dist_point_to_bbox(gp[0], gp[1], obstacle.bbox_xyxy)


def _put_text(img, text, pos, scale=0.45, color=(255, 255, 255)):
    _cv2.putText(img, text, pos, _cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, _cv2.LINE_AA)
    _cv2.putText(img, text, pos, _cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, _cv2.LINE_AA)


def _visualize_safe_check(objects: dict) -> None:
    """Save safe-check debug image to pddl/debug/safe_check.png."""
    if _cv2 is None:
        return

    visible = [obj for obj in objects.values() if obj.visible]
    if not visible:
        return

    # --- Scene canvas (background) ---
    canvas = None
    for obj in visible:
        files = (obj.detection or {}).get("files") or {}
        bg_path = files.get("annotated") or files.get("mask_overlay")
        if bg_path:
            canvas = _cv2.imread(bg_path)
            if canvas is not None:
                break
    if canvas is None:
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    H, W = canvas.shape[:2]

    def _overlay(path, color_bgr, alpha=0.45):
        if not path:
            return None
        m = _cv2.imread(path, _cv2.IMREAD_GRAYSCALE)
        if m is None:
            return None
        if m.shape[:2] != (H, W):
            m = _cv2.resize(m, (W, H), interpolation=_cv2.INTER_NEAREST)
        where = m > 0
        canvas[where] = np.clip(
            canvas[where].astype(np.float32) * (1 - alpha)
            + np.array(color_bgr, dtype=np.float32) * alpha,
            0, 255,
        ).astype(np.uint8)
        ys, xs = np.nonzero(m)
        return np.column_stack([xs, ys]).astype(np.float32) if len(xs) else None

    # Overlay masks, collect pixel sets and grasp points
    obj_pixels: dict[str, np.ndarray | None] = {}
    obj_grasp: dict[str, tuple[int, int] | None] = {}
    for obj in visible:
        files = (obj.detection or {}).get("files") or {}
        color = (0, 200, 0) if obj.is_target else (0, 0, 220)
        pixels = None
        for key in ("mask_full", "mask"):
            pixels = _overlay(files.get(key), color)
            if pixels is not None:
                break
        if pixels is None and obj.bbox_xyxy:
            x1, y1, x2, y2 = obj.bbox_xyxy
            _cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        obj_pixels[obj.name] = pixels
        if obj.is_target:
            gp = _grasp_pixel(obj)
            obj_grasp[obj.name] = (int(round(gp[0])), int(round(gp[1]))) if gp else None
        else:
            obj_grasp[obj.name] = None

    name_to_obj = {obj.name: obj for obj in visible}

    # Pre-compute distances for all target-obstacle pairs
    # dist_map[(t_name, o_name)] = (d_px, endpoint_px)
    dist_map: dict[tuple[str, str], tuple[float, tuple[int, int]]] = {}
    for t_name, gp in obj_grasp.items():
        if gp is None:
            continue
        for o_name, o_pixels in obj_pixels.items():
            obstacle = name_to_obj.get(o_name)
            if obstacle is None or obstacle.is_target:
                continue
            if o_pixels is not None and len(o_pixels) > 0:
                diff = o_pixels - np.array([gp[0], gp[1]], dtype=np.float32)
                idx = int(np.argmin(np.hypot(diff[:, 0], diff[:, 1])))
                ep = (int(round(o_pixels[idx, 0])), int(round(o_pixels[idx, 1])))
                d_px = float(np.hypot(diff[idx, 0], diff[idx, 1]))
            elif obstacle.bbox_xyxy:
                x1, y1, x2, y2 = obstacle.bbox_xyxy
                ep = (int(np.clip(gp[0], x1, x2)), int(np.clip(gp[1], y1, y2)))
                d_px = _dist_point_to_bbox(gp[0], gp[1], obstacle.bbox_xyxy)
            else:
                continue
            dist_map[(t_name, o_name)] = (d_px, ep)

    # Draw lines only (no text on scene)
    for (t_name, o_name), (d_px, ep) in dist_map.items():
        gp = obj_grasp[t_name]
        if gp is None:
            continue
        target = name_to_obj[t_name]
        is_near = o_name in target.near
        _cv2.line(canvas, gp, ep, (0, 60, 255) if is_near else (60, 220, 60), 2)

    # Grasp point markers
    for t_name, gp in obj_grasp.items():
        if gp is None:
            continue
        is_safe = name_to_obj[t_name].safe
        _cv2.drawMarker(canvas, gp, (0, 255, 255) if is_safe else (0, 80, 255),
                        _cv2.MARKER_CROSS, 20, 2)

    # --- Info panel (right side) ---
    ROW_H = 30
    PANEL_W = 300
    panel_h = max(H, ROW_H * (len(visible) + 3))
    panel = np.full((panel_h, PANEL_W, 3), 25, dtype=np.uint8)

    _put_text(panel, f"thr = {SAFE_PIXEL_DISTANCE:.0f} px", (10, 22), scale=0.48,
              color=(180, 180, 180))
    _cv2.line(panel, (6, 32), (PANEL_W - 6, 32), (70, 70, 70), 1)

    # Collect min distance per obstacle (across all targets)
    obs_min_dist: dict[str, tuple[float, str]] = {}  # o_name -> (min_d, t_name)
    for (t_name, o_name), (d_px, _) in dist_map.items():
        if o_name not in obs_min_dist or d_px < obs_min_dist[o_name][0]:
            obs_min_dist[o_name] = (d_px, t_name)

    for i, obj in enumerate(visible):
        y = 32 + ROW_H * (i + 1)
        tag = "T" if obj.is_target else "O"
        tag_color = (100, 240, 100) if obj.is_target else (80, 100, 255)

        _put_text(panel, f"[{tag}]", (8, y), scale=0.45, color=tag_color)
        _put_text(panel, obj.name, (50, y), scale=0.45, color=(220, 220, 220))

        if obj.is_target:
            status = "SAFE" if obj.safe else "UNSAFE"
            status_color = (100, 240, 100) if obj.safe else (60, 60, 255)
            _put_text(panel, status, (190, y), scale=0.45, color=status_color)
        else:
            # Show distance from nearest target
            if obj.name in obs_min_dist:
                d_px, t_name = obs_min_dist[obj.name]
                target = name_to_obj[t_name]
                is_near = obj.name in target.near
                d_color = (60, 60, 255) if is_near else (80, 220, 80)
                _put_text(panel, f"{d_px:.0f} px", (190, y), scale=0.45, color=d_color)

    # Combine scene + panel
    if panel_h > H:
        canvas = np.pad(canvas, ((0, panel_h - H), (0, 0), (0, 0)), constant_values=30)
    sep = np.full((panel_h, 2, 3), 60, dtype=np.uint8)
    combined = np.hstack([canvas, sep, panel])

    debug_dir = PDDL_DIR / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    _cv2.imwrite(str(debug_dir / "safe_check.png"), combined)


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
    grasp_sel = detection.get("grasp_selection") or {}
    grasp_raw = grasp_sel.get("grasp_pose_xyz_m")
    grasp_tuple = tuple(float(v) for v in grasp_raw) if grasp_raw and len(grasp_raw) >= 3 else None
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
        grasp_xyz=grasp_tuple,
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
        if class_name in KNOWN_OBJECTS:
            continue
        obstacle = obstacle_from_yolo(raw, instance_index=index)
        if obstacle.name in objects:
            continue
        objects[obstacle.name] = obstacle


def _annotate_relations(objects: dict[str, ObjectState], debug: bool = False) -> None:
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
        obj.relation_candidate = (
            obj.visible
            and obj.location == "table"
            and obj.scene_region in ACTIVE_SCENE_REGIONS
        )

    values = [obj for obj in objects.values() if obj.relation_candidate]
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

            # near: target-obstacle pairs only, using grasp pixel → nearest obstacle bbox edge.
            if a.is_target == b.is_target:
                continue
            target, obstacle = (a, b) if a.is_target else (b, a)
            d = _grasp_to_obstacle_dist(target, obstacle)
            is_near = d is not None and d < SAFE_PIXEL_DISTANCE
            if debug:
                print(
                    f"[near] {target.name}(T) <-> {obstacle.name}(O)  "
                    f"d={f'{d:.1f}' if d is not None else 'None'}  "
                    f"thr={SAFE_PIXEL_DISTANCE:.1f}  near={is_near}",
                    flush=True,
                )
            if is_near:
                a.near.add(b.name)
                b.near.add(a.name)

    # safe: derived solely from near. A target is unsafe if any near neighbor is a non-target.
    name_to_obj = {obj.name: obj for obj in objects.values()}
    for obj in objects.values():
        if obj.is_target:
            xyz = obj.grasp_xyz or obj.centroid_xyz
            xyz_str = f"({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f})" if xyz else "None"
            for near_name in obj.near:
                neighbor = name_to_obj.get(near_name)
                if neighbor and not neighbor.is_target:
                    obj.safe = False
                    break
            if debug:
                print(
                    f"[safe] {obj.name}  xyz={xyz_str}  near={sorted(obj.near)}  safe={obj.safe}",
                    flush=True,
                )

    if debug:
        _visualize_safe_check(objects)


def _build_predicates(state: PredicateState) -> set[str]:
    # Converts ObjectState fields into PDDL init predicates.
    #
    # CHANGE PDDL FACT EMISSION HERE:
    # - If you add a predicate to domain.pddl, emit its true instances here.
    # - The strings are written into problem.pddl by problem_generator.py.
    predicates = {"handempty"}
    for obj in state.objects.values():
        if obj.location != "table" or obj.scene_region not in ACTIVE_SCENE_REGIONS:
            continue
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
    parent_grasp_selection = (detection or {}).get("grasp_selection")
    for index, instance in enumerate((detection or {}).get("instances") or []):
        instance_index = int(instance.get("instance_index", index))
        instance_name = pddl_name(instance.get("instance_name") or f"{class_name}_{instance_index}")
        if parent_grasp_selection and "grasp_selection" not in instance:
            instance = {**instance, "grasp_selection": parent_grasp_selection}
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
            and obj.location == "table"
            and obj.scene_region in ACTIVE_SCENE_REGIONS
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
    debug: bool = False,
) -> PredicateState:
    # Top-level predicate pipeline called by server.py every planning step:
    #   perception service -> ObjectState -> relations -> PDDL predicates.
    # Edit scan behavior here if you want extra perception sources beyond
    # target detections plus YOLO all_detections.
    objects: dict[str, ObjectState] = {}
    completed = set(completed or set())
    occupied_buffers = set(occupied_buffers or set())
    goal_names = {goal.object_name for goal in goals}
    scan_names = set(KNOWN_OBJECTS if scan_all_targets else goal_names)
    scan_names.update(goal_names)
    for name in sorted(scan_names):
        try:
            detection = detect_fn(name)
            facts = _objects_from_detection(name, detection, is_target=name in goal_names)
            for fact in facts:
                objects[fact.name] = fact
            if detection and detection.get("ok"):
                _merge_all_detections(objects, detection)
        except Exception as exc:
            fact = object_from_detection(name, None, is_target=name in goal_names, class_name=name, error=str(exc))
            objects[fact.name] = fact

    _annotate_relations(objects, debug=debug)
    bound_goals = _bind_goals_to_instances(goals, objects, completed)
    state = PredicateState(goals=bound_goals, objects=objects, completed=completed, occupied_buffers=occupied_buffers)
    state.relation_input_objects = sorted(obj.name for obj in objects.values() if obj.relation_candidate)
    state.predicates = _build_predicates(state)
    for obj in objects.values():
        if obj.error:
            state.notes.append(f"{obj.name}: {obj.error}")
    return state


def build_predicate_state_from_scene(
    goals: list[Goal],
    scene_detection: dict | None,
    completed: set[str] | None = None,
    occupied_buffers: set[str] | None = None,
    debug: bool = False,
) -> PredicateState:
    objects: dict[str, ObjectState] = {}
    completed = set(completed or set())
    occupied_buffers = set(occupied_buffers or set())
    goal_names = {goal.object_name for goal in goals}

    if scene_detection and scene_detection.get("ok"):
        for index, instance in enumerate(scene_detection.get("instances") or []):
            class_name = pddl_name(instance.get("target") or instance.get("detected_label") or "unknown")
            instance_index = int(instance.get("instance_index", index))
            instance_name = pddl_name(instance.get("instance_name") or f"{class_name}_{instance_index}")
            objects[instance_name] = object_from_detection(
                instance_name,
                instance,
                is_target=class_name in goal_names,
                class_name=class_name,
                instance_index=instance_index,
            )
        _merge_all_detections(objects, scene_detection)
    else:
        error = (scene_detection or {}).get("error") or "scene detection failed"
        for name in sorted(goal_names):
            objects[pddl_name(name)] = object_from_detection(
                name,
                None,
                is_target=True,
                class_name=name,
                error=error,
            )

    for name in sorted(goal_names):
        if not any(obj.class_name == name and obj.is_target for obj in objects.values()):
            objects[pddl_name(name)] = object_from_detection(
                name,
                None,
                is_target=True,
                class_name=name,
                error="target not detected in scene scan",
            )

    _annotate_relations(objects, debug=debug)
    bound_goals = _bind_goals_to_instances(goals, objects, completed)
    state = PredicateState(goals=bound_goals, objects=objects, completed=completed, occupied_buffers=occupied_buffers)
    state.raw_observed_objects = {
        "stage": (scene_detection or {}).get("stage"),
        "detections": len((scene_detection or {}).get("all_detections") or []),
        "instances": len((scene_detection or {}).get("instances") or []),
    }
    state.relation_input_objects = sorted(obj.name for obj in objects.values() if obj.relation_candidate)
    state.predicates = _build_predicates(state)
    for obj in objects.values():
        if obj.error:
            state.notes.append(f"{obj.name}: {obj.error}")
    return state
