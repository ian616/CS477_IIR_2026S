#!/usr/bin/env python3
"""Inspect the initial PDDL predicate state produced from perception.

This script calls the same top-view perception service and predicate builder
used by server.py, then prints every object judgement and every generated
predicate before any robot action is executed.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


if __package__ in {None, ""}:
    _PDDL_DIR = Path(__file__).resolve().parent
    _PACKAGE_ROOT = _PDDL_DIR.parents[1]
    if str(_PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(_PACKAGE_ROOT))
    from manip_challenge.pddl.utils import ensure_project_paths, ensure_ros_python
else:
    from .utils import ensure_project_paths, ensure_ros_python


ensure_ros_python()
ensure_project_paths()

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from riro_srvs.srv import StringString

from manip_challenge.pddl.nlp import parse_goals
from manip_challenge.pddl.predicate_builder import build_predicate_state
from manip_challenge.pddl.ros_helpers import (
    load_top_view_perception_module,
    ros_args_with_embedded_perception_defaults,
)
from manip_challenge.pddl.pddl_types import TARGET_OBJECTS


class PredicateInspectionClient(Node):
    def __init__(self, service_name: str, timeout_sec: float):
        super().__init__("initial_predicate_inspection_client")
        self.timeout_sec = float(timeout_sec)
        self.client = self.create_client(StringString, service_name)
        self.get_logger().info(f"Waiting for perception service '{service_name}'...")
        while rclpy.ok() and not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f"Still waiting for '{service_name}'...")

    def detect_object(self, object_name: str) -> dict:
        request = StringString.Request()
        request.data = object_name
        future = self.client.call_async(request)
        deadline = time.monotonic() + self.timeout_sec
        while rclpy.ok() and not future.done():
            if time.monotonic() > deadline:
                raise TimeoutError(f"Timed out waiting for perception result for '{object_name}'.")
            time.sleep(0.02)
        result = future.result()
        if result is None:
            raise RuntimeError(f"Perception service returned no result for '{object_name}'.")
        return json.loads(result.data)


def jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(jsonable(item) for item in value)
    return value


def detection_summary(detection: dict | None) -> dict:
    if not detection:
        return {}
    det = detection.get("detection") or {}
    roi = detection.get("roi") or {}
    return {
        "ok": detection.get("ok"),
        "target": detection.get("target"),
        "detected_label": detection.get("detected_label"),
        "confidence": det.get("confidence"),
        "bbox_xyxy": det.get("bbox_xyxy") or roi.get("bbox_xyxy"),
        "mask_pixels": det.get("mask_pixels") or roi.get("mask_pixels_rgb_crop"),
        "foreground_points": roi.get("num_foreground_points"),
        "location_xyz_m": detection.get("location_xyz_m") or roi.get("centroid_xyz"),
        "depth_median": ((roi.get("stats") or {}).get("foreground_z_median")),
        "all_detection_count": len(detection.get("all_detections") or []),
        "error": detection.get("error"),
        "save_dir": detection.get("save_dir"),
    }


def state_to_payload(state, include_full_detections=False) -> dict:
    objects = {}
    for name, obj in sorted(state.objects.items()):
        item = jsonable(obj)
        if include_full_detections:
            item["detection_summary"] = detection_summary(obj.detection)
        else:
            item["detection"] = detection_summary(obj.detection)
        objects[name] = item
    return {
        "goals": jsonable(state.goals),
        "objects": objects,
        "predicates": sorted(state.predicates),
        "notes": list(state.notes),
    }


def print_state(payload: dict) -> None:
    print("\n=== OBJECT JUDGEMENTS ===")
    for name, obj in payload["objects"].items():
        print(
            f"{name}: detected={obj['detected']} visible={obj['visible']} "
            f"pose_known={obj['pose_known']} graspable={obj['graspable']} "
            f"clear={obj['clear']} safe={obj['safe']}"
        )
        print(
            f"  conf={obj['confidence']:.3f} mask={obj['mask_pixels']} "
            f"points={obj['foreground_points']} bbox={obj['bbox_xyxy']} "
            f"centroid={obj['centroid_xyz']} depth={obj['depth_median']}"
        )
        if obj["blocks"] or obj["blocked_by"] or obj["near"]:
            print(f"  blocks={obj['blocks']} blocked_by={obj['blocked_by']} near={obj['near']}")
        if obj.get("error"):
            print(f"  error={obj['error']}")
        detection = obj.get("detection") or obj.get("detection_summary") or {}
        if detection:
            print(f"  detection={json.dumps(detection, sort_keys=True)}")

    print("\n=== TRUE PREDICATES ===")
    for predicate in payload["predicates"]:
        print(predicate)

    if payload["notes"]:
        print("\n=== NOTES ===")
        for note in payload["notes"]:
            print(note)


def parse_args(argv=None):
    cli_args = rclpy.utilities.remove_ros_args(args=argv or sys.argv)[1:]
    parser = argparse.ArgumentParser(description="Print the initial PDDL predicates produced from top-view perception.")
    parser.add_argument(
        "command",
        nargs="?",
        default="",
        help="Optional task command. If omitted, all target objects are scanned without goal-at predicates.",
    )
    parser.add_argument("--perception-service", default="detect_object_top_rgbd_seg_crop")
    parser.add_argument("--perception-timeout", type=float, default=15.0)
    parser.add_argument("--external-perception", action="store_true", help="Use an already running perception service.")
    parser.add_argument("--scan-goal-targets-only", action="store_true", help="Only scan objects named in the command goals.")
    parser.add_argument("--json-out", default="", help="Optional path to write the full predicate payload as JSON.")
    parser.add_argument("--full-detections", action="store_true", help="Include full perception dictionaries in JSON output.")
    return parser.parse_args(cli_args)


def main(argv=None):
    args = parse_args(argv)
    top_view_module = None if args.external_perception else load_top_view_perception_module()
    ros_argv = (
        ros_args_with_embedded_perception_defaults(argv or sys.argv, args, top_view_module)
        if top_view_module is not None
        else argv
    )

    rclpy.init(args=ros_argv)
    executor = MultiThreadedExecutor()
    perception_node = None
    client_node = None
    spin_thread = None
    try:
        if top_view_module is not None:
            perception_node = top_view_module.RgbdSegCropServiceNode()
            perception_node.get_logger().info("Embedded top-view perception is running for predicate inspection.")
            executor.add_node(perception_node)

        client_node = PredicateInspectionClient(args.perception_service, args.perception_timeout)
        executor.add_node(client_node)
        spin_thread = threading.Thread(target=executor.spin, daemon=True)
        spin_thread.start()

        goals = parse_goals(args.command, use_gemini=False) if args.command else []
        if args.command and not goals:
            print(f"Warning: no goals were parsed from command: {args.command!r}", file=sys.stderr)
        state = build_predicate_state(
            goals,
            client_node.detect_object,
            scan_all_targets=not args.scan_goal_targets_only,
        )
        payload = state_to_payload(state, include_full_detections=args.full_detections)
        print_state(payload)

        if args.json_out:
            output_path = Path(args.json_out).expanduser()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            print(f"\nWrote JSON: {output_path}")
        return 0
    finally:
        if client_node is not None:
            executor.remove_node(client_node)
            client_node.destroy_node()
        if perception_node is not None:
            executor.remove_node(perception_node)
            perception_node.destroy_node()
        executor.shutdown()
        if spin_thread is not None:
            spin_thread.join(timeout=1.0)
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
