#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
import traceback
from pathlib import Path

try:
    import cv2
except ImportError:
    cv2 = None


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

import numpy as np
if not hasattr(np, "mat"):
    np.mat = np.asmatrix

import rclpy
import tf2_geometry_msgs  # noqa: F401
from geometry_msgs.msg import Pose, PoseStamped
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from riro_srvs.srv import StringString
from std_msgs.msg import String
from tf2_ros import Buffer, ConnectivityException, ExtrapolationException, LookupException, TransformListener

from assignment_2.move_joint import ArmClient
from manip_challenge.pddl.actions import ActionContext, execute_action
from manip_challenge.pddl.nlp import parse_goals
from manip_challenge.pddl.planner import plan
from manip_challenge.pddl.predicate_builder import build_predicate_state
from manip_challenge.pddl.problem_generator import ensure_domain, write_problem
from manip_challenge.pddl.ros_helpers import (
    HOME_JOINTS,
    adjust_target_depth_from_rgbd,
    choose_grasp_target_from_points,
    load_grasp_points_from_detection,
    load_top_view_perception_module,
    patch_move_gripper,
    ros_args_with_embedded_perception_defaults,
    save_grasp_selection_visualization,
    stop_child_processes,
)
from manip_challenge.pddl.types import PlanAction
from manip_challenge.pddl.utils import PDDL_DIR, load_dotenv


patch_move_gripper()


class TfNode(Node):
    def __init__(self):
        super().__init__("pddl_tamp_tf_node")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)


def pose_from_xyz(xyz) -> Pose:
    pose = Pose()
    pose.position.x = float(xyz[0])
    pose.position.y = float(xyz[1])
    pose.position.z = float(xyz[2])
    pose.orientation.w = 1.0
    return pose


def pose_to_dict(pose: Pose) -> dict:
    return {
        "position": {
            "x": float(pose.position.x),
            "y": float(pose.position.y),
            "z": float(pose.position.z),
        },
        "orientation": {
            "x": float(pose.orientation.x),
            "y": float(pose.orientation.y),
            "z": float(pose.orientation.z),
            "w": float(pose.orientation.w),
        },
    }


def object_summary(obj) -> dict:
    files = (obj.detection or {}).get("files") or {}
    return {
        "target": obj.is_target,
        "location": obj.location,
        "detected": obj.detected,
        "visible": obj.visible,
        "pose_known": obj.pose_known,
        "graspable": obj.graspable,
        "clear": obj.clear,
        "safe": obj.safe,
        "confidence": obj.confidence,
        "mask_pixels": obj.mask_pixels,
        "foreground_points": obj.foreground_points,
        "bbox_xyxy": obj.bbox_xyxy,
        "centroid_xyz": obj.centroid_xyz,
        "depth_median": obj.depth_median,
        "blocks": sorted(obj.blocks),
        "blocked_by": sorted(obj.blocked_by),
        "near": sorted(obj.near),
        "error": obj.error,
        "debug_files": {
            key: files[key]
            for key in ("annotated", "mask_overlay", "rgb", "rgb_masked")
            if key in files
        },
    }


class PddlTampServer(Node):
    def __init__(self, args, tf_buffer, arm):
        super().__init__("pddl_tamp_server")
        self.args = args
        self.tf_buffer = tf_buffer
        self.arm = arm
        self.callback_group = ReentrantCallbackGroup()
        self.command_lock = threading.Lock()

        # Attributes consumed by existing gripper/motion wrappers.
        self.gripper_force = args.gripper_force
        self.gripper_close_pos = args.gripper_close_pos
        self.gripper_settle_time = args.gripper_settle_time
        self.gripper_result_timeout_margin = args.gripper_result_timeout_margin
        self.strict_gripper_result = args.strict_gripper_result

        self.perception_client = self.create_client(
            StringString,
            args.perception_service,
            callback_group=self.callback_group,
        )
        self.command_service = self.create_service(
            StringString,
            args.service_name,
            self.handle_command,
            callback_group=self.callback_group,
        )
        self.command_sub = self.create_subscription(
            String,
            args.command_topic,
            self.handle_command_topic,
            10,
            callback_group=self.callback_group,
        )
        self.status_pub = self.create_publisher(String, "/planner_status", 10)
        self.pred_pub = self.create_publisher(String, "/world_predicates", 10)
        self.action_pub = self.create_publisher(String, "/planned_action", 10)
        self.result_pub = self.create_publisher(String, "/action_result", 10)
        self.pddl_log_pub = self.create_publisher(String, args.log_topic, 10)

        self.get_logger().info(f"Waiting for perception service '{args.perception_service}'...")
        while rclpy.ok() and not self.perception_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f"Still waiting for '{args.perception_service}'...")
        self.get_logger().info(
            f"PDDL TAMP server ready. Publish natural-language commands to '{args.command_topic}' "
            f"or call service '{args.service_name}'."
        )

    def publish_json(self, publisher, payload: dict) -> None:
        msg = String()
        msg.data = json.dumps(payload, sort_keys=True)
        publisher.publish(msg)

    def publish_pddl_log(self, event: str, lines: list[str], **_payload) -> None:
        msg = String()
        header = f"[PDDL_LOG] event={event} stamp_sec={time.time():.3f}"
        msg.data = "\n".join([header, *lines])
        self.pddl_log_pub.publish(msg)

    def handle_command(self, request, response):
        try:
            with self.command_lock:
                result = self.execute_command(request.data)
        except Exception as exc:
            self.get_logger().error(f"Command failed: {exc}\n{traceback.format_exc()}")
            result = {"ok": False, "error": str(exc), "command": request.data}
            self.publish_pddl_log(
                "command_error",
                ["[PDDL] command failed", f"error: {exc}", f"command: {request.data}"],
                error=str(exc),
                command=request.data,
            )
        response.data = json.dumps(result, sort_keys=True)
        return response

    def handle_command_topic(self, msg: String) -> None:
        command_text = str(msg.data or "").strip()
        if not command_text:
            self.get_logger().warn("[PDDL] Ignoring empty topic command.")
            return
        self.get_logger().info(f"[PDDL] Topic command received on '{self.args.command_topic}': {command_text}")
        worker = threading.Thread(
            target=self.execute_topic_command,
            args=(command_text,),
            daemon=True,
        )
        worker.start()

    def execute_topic_command(self, command_text: str) -> None:
        try:
            with self.command_lock:
                result = self.execute_command(command_text)
        except Exception as exc:
            self.get_logger().error(f"Topic command failed: {exc}\n{traceback.format_exc()}")
            result = {"ok": False, "error": str(exc), "command": command_text}
            self.publish_pddl_log(
                "topic_command_error",
                ["[PDDL] topic command failed", f"error: {exc}", f"command: {command_text}"],
                error=str(exc),
                command=command_text,
            )
        self.publish_json(
            self.result_pub,
            {
                "event": "topic_command_result",
                "command": command_text,
                "result": result,
            },
        )

    def execute_command(self, command_text: str) -> dict:
        lowered = str(command_text or "").strip().lower()
        if lowered in {"home", "idle", "reset"}:
            if not self.args.dry_run:
                self.arm.move_joint(HOME_JOINTS)
            self.publish_pddl_log(
                "home",
                [f"[PDDL] {lowered}: robot moved to home joints"],
                action=lowered,
                dry_run=bool(self.args.dry_run),
            )
            return {"ok": True, "action": lowered, "message": "Robot moved to home joints."}

        goals = parse_goals(command_text, use_gemini=not self.args.no_gemini)
        if not goals:
            raise ValueError("Could not parse any target goals from command.")
        self.get_logger().info("[PDDL] Parsed goals:")
        for goal in goals:
            self.get_logger().info(f"[PDDL]   {goal.object_name} -> {goal.location}")
        self.publish_json(self.status_pub, {"event": "parsed_goals", "goals": [goal.__dict__ for goal in goals]})
        self.publish_pddl_log(
            "parsed_goals",
            [
                "[PDDL] parsed goals",
                *[f"  {goal.object_name} -> {goal.location}" for goal in goals],
            ],
            goals=[goal.__dict__ for goal in goals],
        )

        if not self.args.dry_run and not self.args.no_home:
            self.get_logger().info("[PDDL] Returning to home before planning.")
            self.arm.move_joint(HOME_JOINTS)

        completed: set[str] = set()
        occupied_buffers: set[str] = set()
        steps = []
        scenes = []
        domain_path = ensure_domain(PDDL_DIR / "domain.pddl")
        problem_path = PDDL_DIR / "problem.pddl"
        context = ActionContext(self)

        for step_idx in range(1, self.args.max_steps + 1):
            self.get_logger().info(f"[PDDL] ===== planning step {step_idx}/{self.args.max_steps} =====")
            state = build_predicate_state(
                goals,
                self.detect_object,
                completed=completed,
                occupied_buffers=occupied_buffers,
                scan_all_targets=not self.args.scan_goals_only,
            )
            scene_summary = {
                "completed": sorted(completed),
                "occupied_buffers": sorted(occupied_buffers),
                "objects": {name: object_summary(obj) for name, obj in state.objects.items()},
                "predicates": sorted(state.predicates),
                "notes": state.notes,
            }
            scenes.append(scene_summary)
            self.log_state(scene_summary)
            self.publish_json(self.pred_pub, scene_summary)
            self.publish_pddl_log(
                "predicate_judgement",
                self.judgement_log_lines(scene_summary, step_idx),
                step=step_idx,
                scene=scene_summary,
            )
            if self.args.debug:
                self.show_debug_view(scene_summary, step_idx)

            if not state.unfinished_goals():
                self.get_logger().info("[PDDL] All goals completed.")
                self.publish_pddl_log(
                    "all_goals_completed",
                    ["[PDDL] all goals completed", f"completed: {', '.join(sorted(completed))}"],
                    step=step_idx,
                    completed=sorted(completed),
                )
                break

            generated_problem = write_problem(state, problem_path)
            self.get_logger().info(f"[PDDL] Generated problem: {generated_problem}")
            actions, raw_output = plan(domain_path, generated_problem, state)
            self.get_logger().info("[PDDL] Planner raw output:\n" + raw_output)
            plan_payload = [action.to_dict() for action in actions]
            self.publish_json(
                self.status_pub,
                {
                    "event": "planner_output",
                    "problem": str(generated_problem),
                    "raw_output": raw_output,
                    "plan": plan_payload,
                },
            )
            self.publish_pddl_log(
                "planner_output",
                self.planner_log_lines(generated_problem, actions, raw_output),
                step=step_idx,
                problem=str(generated_problem),
                raw_output=raw_output,
                plan=plan_payload,
            )
            if not actions:
                self.publish_pddl_log(
                    "planning_failed",
                    ["[PDDL] no executable action found", f"completed: {', '.join(sorted(completed)) or '<none>'}"],
                    step=step_idx,
                    completed=sorted(completed),
                )
                return {
                    "ok": False,
                    "error": "No executable PDDL action found.",
                    "goals": [goal.__dict__ for goal in goals],
                    "completed": sorted(completed),
                    "occupied_buffers": sorted(occupied_buffers),
                    "steps": steps,
                    "scenes": scenes,
                }

            action = self.select_executable_action(actions)
            self.get_logger().info(f"[PDDL] Selected next action: {action.pddl()} from {action.source}")
            self.publish_json(self.action_pub, {"event": "selected_action", "action": action.to_dict()})
            self.publish_pddl_log(
                "selected_action",
                self.selected_action_log_lines(action, state, step_idx),
                step=step_idx,
                action=action.to_dict(),
            )

            self.publish_pddl_log(
                "executing_action",
                self.executing_action_log_lines(action, step_idx),
                step=step_idx,
                action=action.to_dict(),
            )
            result = execute_action(context, action, state)
            steps.append(result)
            self.publish_json(self.result_pub, {"event": "action_result", "result": result})
            self.get_logger().info("[PDDL] Action execution result: " + json.dumps(result, sort_keys=True))
            self.publish_pddl_log(
                "action_result",
                self.action_result_log_lines(action, result, step_idx),
                step=step_idx,
                action=action.to_dict(),
                result=result,
            )

            if action.name == "move-target-to-goal" and result.get("ok"):
                completed.add(action.args[0])
            elif action.name == "move-obstacle-to-buffer" and result.get("ok"):
                occupied_buffers.add(action.args[2])

            self.get_logger().info("[PDDL] Replanning after action execution.")
            self.publish_pddl_log(
                "replan",
                [
                    "[PDDL] replanning after action execution",
                    f"completed: {', '.join(sorted(completed)) or '<none>'}",
                    f"occupied_buffers: {', '.join(sorted(occupied_buffers)) or '<none>'}",
                ],
                step=step_idx,
                completed=sorted(completed),
                occupied_buffers=sorted(occupied_buffers),
            )
            if self.args.one_step:
                break

        ok = len(completed) == len({goal.object_name for goal in goals})
        self.publish_pddl_log(
            "command_finished",
            [
                f"[PDDL] command finished ok={ok}",
                f"completed: {', '.join(sorted(completed)) or '<none>'}",
                f"occupied_buffers: {', '.join(sorted(occupied_buffers)) or '<none>'}",
            ],
            ok=ok,
            completed=sorted(completed),
            occupied_buffers=sorted(occupied_buffers),
        )
        return {
            "ok": ok,
            "action": "pddl_tamp",
            "goals": [goal.__dict__ for goal in goals],
            "completed": sorted(completed),
            "occupied_buffers": sorted(occupied_buffers),
            "steps": steps,
            "scenes": scenes,
            "message": "PDDL TAMP completed." if ok else "PDDL TAMP stopped before all goals completed.",
            "dry_run": bool(self.args.dry_run),
        }

    def select_executable_action(self, actions: list[PlanAction]) -> PlanAction:
        for action in actions:
            if action.name in {"move-target-to-goal", "move-obstacle-to-buffer"}:
                return action
        raise RuntimeError(f"Planner returned no physical action: {[action.pddl() for action in actions]}")

    def log_state(self, summary: dict) -> None:
        self.get_logger().info("[PDDL] Detected objects and predicates:")
        for name, obj in sorted(summary["objects"].items()):
            self.get_logger().info(
                "[PDDL]   "
                f"{name}: target={obj['target']}, detected={obj['detected']}, "
                f"visible={obj['visible']}, pose_known={obj['pose_known']}, "
                f"graspable={obj['graspable']}, clear={obj['clear']}, safe={obj['safe']}, "
                f"conf={obj['confidence']:.3f}, bbox={obj['bbox_xyxy']}, "
                f"centroid={obj['centroid_xyz']}, blocks={obj['blocks']}, "
                f"blocked_by={obj['blocked_by']}, near={obj['near']}, error={obj['error']}"
            )
        self.get_logger().info("[PDDL]   predicates: " + ", ".join(summary["predicates"]))
        for note in summary["notes"]:
            self.get_logger().warn("[PDDL]   note: " + note)

    def judgement_log_lines(self, summary: dict, step_idx: int) -> list[str]:
        lines = [
            f"[PDDL] judgement step {step_idx}",
            "completed: " + (", ".join(summary["completed"]) if summary["completed"] else "<none>"),
            "occupied_buffers: "
            + (", ".join(summary.get("occupied_buffers", [])) if summary.get("occupied_buffers") else "<none>"),
            "objects:",
        ]
        for name, obj in sorted(summary["objects"].items()):
            lines.append(
                f"  {name}: target={obj['target']} detected={obj['detected']} visible={obj['visible']} "
                f"pose_known={obj['pose_known']} graspable={obj['graspable']} clear={obj['clear']} "
                f"safe={obj['safe']} conf={obj['confidence']:.3f}"
            )
            lines.append(
                f"    bbox={obj['bbox_xyxy']} centroid={obj['centroid_xyz']} depth={obj['depth_median']} "
                f"blocks={obj['blocks']} blocked_by={obj['blocked_by']} near={obj['near']}"
            )
            if obj["error"]:
                lines.append(f"    error={obj['error']}")
        lines.append("predicates:")
        lines.extend(f"  {predicate}" for predicate in summary["predicates"])
        if summary["notes"]:
            lines.append("notes:")
            lines.extend(f"  {note}" for note in summary["notes"])
        return lines

    @staticmethod
    def selected_action_log_lines(action: PlanAction, state, step_idx: int) -> list[str]:
        lines = [
            f"[PDDL] selected action at step {step_idx}",
            f"  action: {action.pddl()}",
            f"  source: {action.source}",
            "  unfinished goal objects:",
        ]
        unfinished = {goal.object_name for goal in state.unfinished_goals()}
        for name in sorted(unfinished):
            obj = state.objects.get(name)
            if obj is None:
                lines.append(f"    {name}: not in current perception state")
                continue
            ready = obj.graspable and obj.clear and obj.safe
            lines.append(
                f"    {name}: ready={ready} detected={obj.detected} visible={obj.visible} "
                f"pose_known={obj.pose_known} graspable={obj.graspable} clear={obj.clear} "
                f"safe={obj.safe} conf={obj.confidence:.3f}"
            )
            lines.append(
                f"      bbox={obj.bbox_xyxy} centroid={obj.centroid_xyz} depth={obj.depth_median} "
                f"blocked_by={sorted(obj.blocked_by)} near={sorted(obj.near)}"
            )
        if action.args:
            selected = state.objects.get(action.args[0])
            if selected is not None:
                lines.append(
                    f"  selected object judgement: {selected.name} clear={selected.clear} "
                    f"graspable={selected.graspable} safe={selected.safe} centroid={selected.centroid_xyz}"
                )
        return lines

    @staticmethod
    def executing_action_log_lines(action: PlanAction, step_idx: int) -> list[str]:
        lines = [
            f"[PDDL] executing physical action at step {step_idx}",
            f"  action: {action.pddl()}",
        ]
        if action.name == "move-target-to-goal" and len(action.args) == 3:
            object_name, source, destination = action.args
            lines.append(f"  moving target: {object_name} {source} -> {destination}")
        elif action.name == "move-obstacle-to-buffer" and len(action.args) == 3:
            object_name, source, buffer_name = action.args
            lines.append(f"  moving obstacle: {object_name} {source} -> {buffer_name}")
        lines.append("  motion note: low-level pick-place retreats and returns to idle before replanning")
        return lines

    @staticmethod
    def action_result_log_lines(action: PlanAction, result: dict, step_idx: int) -> list[str]:
        lines = [
            f"[PDDL] action result at step {step_idx}",
            f"  action: {action.pddl()}",
            f"  ok: {result.get('ok')}",
            f"  destination: {result.get('destination', '<none>')}",
        ]
        selection = result.get("grasp_selection") or {}
        if selection:
            lines.append(
                "  grasp_selection: "
                f"method={selection.get('method')} target_xyz_m={selection.get('target_xyz_m')} "
                f"reason={selection.get('reason', '<none>')}"
            )
        grasp_pose = (result.get("grasp_pose_in_base") or {}).get("position") or {}
        if grasp_pose:
            lines.append(
                "  grasp_pose_in_base: "
                f"x={grasp_pose.get('x'):.4f} y={grasp_pose.get('y'):.4f} z={grasp_pose.get('z'):.4f}"
            )
        return lines

    @staticmethod
    def planner_log_lines(problem_path: Path, actions: list[PlanAction], raw_output: str) -> list[str]:
        lines = [
            "[PDDL] planner output",
            f"problem: {problem_path}",
            "plan:",
        ]
        if actions:
            lines.extend(f"  {action.pddl()} [{action.source}]" for action in actions)
        else:
            lines.append("  <empty>")
        lines.append("raw_output:")
        raw_lines = str(raw_output or "").splitlines()
        if raw_lines:
            lines.extend(f"  {line}" for line in raw_lines)
        else:
            lines.append("  <empty>")
        return lines

    def show_debug_view(self, summary: dict, step_idx: int) -> None:
        if cv2 is None:
            self.get_logger().warn("[PDDL] --debug requested, but cv2 is not available.")
            return

        image = self.load_debug_image(summary)
        if image is None:
            image = np.full((720, 900, 3), 245, dtype=np.uint8)
            cv2.putText(
                image,
                "No perception debug image available",
                (28, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (30, 30, 30),
                2,
                cv2.LINE_AA,
            )

        max_h = 720
        scale = min(1.0, max_h / max(1, image.shape[0]))
        if scale < 1.0:
            image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))

        panel_w = 760
        panel_h = max(image.shape[0], 720)
        panel = np.full((panel_h, panel_w, 3), 32, dtype=np.uint8)
        lines = self.debug_lines(summary, step_idx)
        y = 34
        for line, color in lines:
            for chunk in self.wrap_debug_line(line, 78):
                cv2.putText(panel, chunk, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
                y += 22
                if y > panel_h - 20:
                    break
            if y > panel_h - 20:
                cv2.putText(panel, "...", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
                break

        if image.shape[0] < panel_h:
            pad = np.full((panel_h - image.shape[0], image.shape[1], 3), 245, dtype=np.uint8)
            image = np.vstack([image, pad])
        canvas = np.hstack([image, panel])

        debug_dir = PDDL_DIR / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        latest = debug_dir / "latest_debug.png"
        step_path = debug_dir / f"step_{step_idx:02d}_debug.png"
        cv2.imwrite(str(latest), canvas)
        cv2.imwrite(str(step_path), canvas)
        self.get_logger().info(f"[PDDL] Debug view saved: {latest}")

        if self.args.debug_window and self.debug_display_available():
            try:
                cv2.imshow("PDDL TAMP Debug", canvas)
                key = cv2.waitKey(0 if self.args.debug_wait else 1)
                if key in (27, ord("q")):
                    self.args.debug = False
                    self.get_logger().info("[PDDL] Debug window disabled by key press.")
            except Exception as exc:
                self.get_logger().warn(f"[PDDL] Could not open debug window: {exc}")
        elif self.args.debug_window:
            self.get_logger().info("[PDDL] No usable GUI display found; debug image was saved only.")

    def load_debug_image(self, summary: dict):
        for obj in summary["objects"].values():
            files = obj.get("debug_files") or {}
            for key in ("annotated", "mask_overlay", "rgb"):
                path = files.get(key)
                if path and Path(path).is_file():
                    image = cv2.imread(str(path))
                    if image is not None:
                        return image
        return None

    def debug_lines(self, summary: dict, step_idx: int) -> list[tuple[str, tuple[int, int, int]]]:
        lines = [
            (f"PDDL TAMP DEBUG - planning step {step_idx}", (120, 230, 255)),
            ("completed: " + (", ".join(summary["completed"]) if summary["completed"] else "<none>"), (220, 220, 220)),
            ("", (220, 220, 220)),
            ("OBJECT JUDGEMENT", (120, 230, 255)),
        ]
        for name, obj in sorted(summary["objects"].items()):
            status_color = (120, 230, 120) if obj["graspable"] and obj["clear"] and obj["safe"] else (80, 190, 255)
            if obj["error"] or not obj["detected"]:
                status_color = (80, 120, 255)
            lines.append((
                f"{name}: target={obj['target']} det={obj['detected']} vis={obj['visible']} "
                f"pose={obj['pose_known']} grasp={obj['graspable']} clear={obj['clear']} safe={obj['safe']} "
                f"conf={obj['confidence']:.2f}",
                status_color,
            ))
            lines.append((
                f"  bbox={obj['bbox_xyxy']} centroid={obj['centroid_xyz']} depth={obj['depth_median']} "
                f"blocks={obj['blocks']} blocked_by={obj['blocked_by']} near={obj['near']}",
                (205, 205, 205),
            ))
            if obj["error"]:
                lines.append((f"  error={obj['error']}", (80, 120, 255)))
        lines.extend([
            ("", (220, 220, 220)),
            ("TRUE PREDICATES", (120, 230, 255)),
        ])
        for predicate in summary["predicates"]:
            lines.append((predicate, (190, 190, 190)))
        if summary["notes"]:
            lines.extend([("", (220, 220, 220)), ("NOTES", (120, 230, 255))])
            for note in summary["notes"]:
                lines.append((note, (80, 120, 255)))
        return lines

    @staticmethod
    def wrap_debug_line(line: str, limit: int) -> list[str]:
        if not line:
            return [""]
        chunks = []
        text = str(line)
        while len(text) > limit:
            split_at = text.rfind(" ", 0, limit)
            if split_at <= 0:
                split_at = limit
            chunks.append(text[:split_at])
            text = text[split_at:].lstrip()
        chunks.append(text)
        return chunks

    @staticmethod
    def debug_display_available() -> bool:
        wayland = os.environ.get("WAYLAND_DISPLAY")
        if wayland:
            runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
            if runtime_dir and (Path(runtime_dir) / wayland).exists():
                return True
        display = os.environ.get("DISPLAY")
        if not display:
            return False
        if display.startswith(":"):
            display_num = display[1:].split(".")[0]
            return (Path("/tmp/.X11-unix") / f"X{display_num}").exists()
        return True

    def detect_object(self, obj_name: str) -> dict:
        req = StringString.Request()
        req.data = obj_name
        future = self.perception_client.call_async(req)
        deadline = time.monotonic() + self.args.perception_timeout
        while rclpy.ok() and not future.done():
            if time.monotonic() > deadline:
                raise TimeoutError(f"Timed out waiting for perception result for '{obj_name}'.")
            time.sleep(0.02)
        result = future.result()
        if result is None:
            raise RuntimeError(f"Perception service returned no result for '{obj_name}'.")
        return json.loads(result.data)

    def select_grasp_target(self, detection: dict, obj_name: str) -> dict:
        points, points_path = load_grasp_points_from_detection(detection)
        if points is None or len(points) == 0:
            fallback = detection["location_xyz_m"]
            return {
                "method": "perception_centroid_fallback",
                "target_xyz_m": [float(v) for v in fallback],
                "raw_centroid_xyz_m": [float(v) for v in fallback],
                "points_path": points_path,
                "reason": "foreground_points_unavailable",
            }
        label = detection.get("target") or detection.get("detected_label") or obj_name
        selection = choose_grasp_target_from_points(points, label=label)
        selection["points_path"] = points_path
        selection = adjust_target_depth_from_rgbd(
            points,
            selection,
            clearance_m=self.args.grasp_surface_clearance,
            local_radius_m=self.args.grasp_depth_local_radius,
            percentile=self.args.grasp_depth_percentile,
        )
        output_dir = detection.get("save_dir")
        if output_dir:
            try:
                visualization = save_grasp_selection_visualization(points, selection, output_dir, label)
            except Exception as exc:
                visualization = None
                selection["visualization_error"] = str(exc)
            if visualization:
                selection["visualization"] = visualization
        raw = np.asarray(selection.get("raw_centroid_xyz_m", detection["location_xyz_m"]), dtype=float)
        target = np.asarray(selection["target_xyz_m"], dtype=float)
        selection["delta_from_raw_centroid_m"] = (target - raw).astype(float).tolist()
        return selection

    def transform_pose(self, pose: Pose, source_frame: str, target_frame: str) -> Pose:
        deadline = time.monotonic() + self.args.tf_timeout
        while rclpy.ok() and time.monotonic() < deadline:
            if self.tf_buffer.can_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5),
            ):
                break
            time.sleep(0.05)
        else:
            raise TimeoutError(f"TF unavailable: {source_frame} -> {target_frame}")
        stamped = PoseStamped()
        stamped.header.frame_id = source_frame
        stamped.header.stamp = rclpy.time.Time().to_msg()
        stamped.pose = pose
        try:
            return self.tf_buffer.transform(
                stamped,
                target_frame,
                timeout=rclpy.duration.Duration(seconds=1.0),
            ).pose
        except (LookupException, ConnectivityException, ExtrapolationException) as exc:
            raise RuntimeError(f"TF transform failed: {exc}") from exc

    def make_grasp_pose(self, base_pose: Pose) -> Pose:
        home_pose = self.arm.fk_request(HOME_JOINTS)
        grasp_pose = Pose()
        grasp_pose.position = base_pose.position
        grasp_pose.orientation = home_pose.orientation
        grasp_pose.position.y += self.args.grasp_y_offset
        grasp_pose.position.z += self.args.final_z_offset
        return grasp_pose


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="PDDL-based TAMP command server.")
    parser.add_argument("--service-name", default="pddl_tamp_command")
    parser.add_argument("--command-topic", default="/task_commands")
    parser.add_argument("--log-topic", default="/pddl_tamp_log")
    parser.add_argument("--perception-service", default="detect_object_top_rgbd_seg_crop")
    parser.add_argument("--camera-frame", default="camera_color_optical_frame")
    parser.add_argument("--perception-timeout", type=float, default=15.0)
    parser.add_argument("--tf-timeout", type=float, default=5.0)
    parser.add_argument("--max-steps", type=int, default=int(os.environ.get("PDDL_TAMP_MAX_STEPS", "8")))
    parser.add_argument("--one-step", action="store_true", help="Execute only the first selected physical action.")
    parser.add_argument("--scan-goals-only", action="store_true", help="Only scan requested targets instead of all five fixed targets.")
    parser.add_argument("--no-gemini", action="store_true", help="Disable Gemini parsing and use the rule-based parser.")
    parser.add_argument("--debug", action="store_true", help="Show and save a perception+predicate debug view each planning step.")
    parser.add_argument("--debug-window", action="store_true", help="With --debug, also open an OpenCV window when a GUI display is usable.")
    parser.add_argument("--debug-wait", action="store_true", help="With --debug-window, wait for a key press at each planning step.")
    parser.add_argument("--external-perception", action="store_true", help="Use an already running top-view perception service.")
    parser.add_argument("--dry-run", action="store_true", help="Plan and bind actions without moving the arm.")
    parser.add_argument("--no-home", action="store_true", help="Do not return to home before executing a command.")
    parser.add_argument("--approach-height", type=float, default=0.15)
    parser.add_argument("--final-z-offset", type=float, default=0.0)
    parser.add_argument("--grasp-surface-clearance", type=float, default=-0.012)
    parser.add_argument("--grasp-depth-local-radius", type=float, default=0.035)
    parser.add_argument("--grasp-depth-percentile", type=float, default=10.0)
    parser.add_argument("--grasp-y-offset", type=float, default=-0.01)
    parser.add_argument("--gripper-force", type=float, default=0.5)
    parser.add_argument("--gripper-close-pos", type=float, default=0.5)
    parser.add_argument("--gripper-settle-time", type=float, default=0.4)
    parser.add_argument("--gripper-result-timeout-margin", type=float, default=8.0)
    parser.add_argument("--strict-gripper-result", action="store_true")
    return parser.parse_args(rclpy.utilities.remove_ros_args(args=argv or sys.argv)[1:])


def main(argv=None):
    load_dotenv()
    args = parse_args(argv)
    top_view_module = None if args.external_perception else load_top_view_perception_module()
    ros_argv = (
        ros_args_with_embedded_perception_defaults(argv or sys.argv, args, top_view_module)
        if top_view_module is not None
        else argv
    )
    rclpy.init(args=ros_argv)
    perception_node = None
    tf_node = TfNode()
    if top_view_module is not None:
        perception_node = top_view_module.RgbdSegCropServiceNode()
        perception_node.get_logger().info("Embedded top-view perception is running inside PDDL TAMP server.")
    arm = ArmClient()
    server = PddlTampServer(args, tf_node.tf_buffer, arm)
    child_processes = []
    try:
        if not args.no_home:
            arm.move_joint(HOME_JOINTS)
        executor = MultiThreadedExecutor(num_threads=4)
        if perception_node is not None:
            executor.add_node(perception_node)
        executor.add_node(tf_node)
        executor.add_node(server)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if "executor" in locals():
            executor.shutdown()
        server.destroy_node()
        tf_node.destroy_node()
        if perception_node is not None:
            perception_node.destroy_node()
        arm.destroy_node()
        rclpy.shutdown()
        stop_child_processes(child_processes)


if __name__ == "__main__":
    main()
