#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path

from .pddl_types import PlanAction, PredicateState


def parse_plan_text(text: str, source: str = "external") -> list[PlanAction]:
    actions = []
    for line in str(text or "").splitlines():
        stripped = line.split(";", 1)[0].strip().lower()
        if not stripped:
            continue
        match = re.search(r"\(([a-z0-9_-]+)\s+([^)]*)\)", stripped)
        if not match:
            continue
        name = match.group(1)
        args = tuple(arg for arg in match.group(2).split() if arg and not arg.startswith("?"))
        actions.append(PlanAction(name, args, source=source))
    return actions


def run_external_planner(domain_path: Path, problem_path: Path) -> tuple[list[PlanAction], str]:
    command_template = os.environ.get("PDDL_PLANNER_CMD", "").strip()
    if not command_template:
        return [], "PDDL_PLANNER_CMD is not set; using fallback planner."
    command = command_template.format(domain=domain_path, problem=problem_path)
    try:
        completed = subprocess.run(
            shlex.split(command),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=float(os.environ.get("PDDL_PLANNER_TIMEOUT", "30")),
            check=False,
        )
    except Exception as exc:
        return [], f"External planner failed: {exc}"
    actions = parse_plan_text(completed.stdout, source="external")
    return actions, completed.stdout


def _first_free_buffer(state: PredicateState) -> str | None:
    buffers = state.free_buffers()
    return buffers[0] if buffers else None


def fallback_plan(state: PredicateState) -> list[PlanAction]:
    # Fallback action-selection policy when no external PDDL planner is set.
    # Edit here to change "which action should be tried next" without touching
    # the physical execution handlers.
    #
    # Current priority:
    #   1. clear/graspable/safe goal objects, preferring objects blocking goals
    #   2. blockers of an unfinished goal
    #   3. non-target obstacles near an unsafe target
    #   4. recovery: best clear/graspable target even if safe is false
    goals_by_object = {goal.target_name: goal for goal in state.unfinished_goals()}
    unfinished = set(goals_by_object)
    free_buffer = _first_free_buffer(state)

    candidates = []
    for name, obj in state.objects.items():
        if name not in unfinished:
            continue
        if not (obj.graspable and obj.clear and obj.safe):
            continue
        blocking_score = len(obj.blocks & unfinished)
        candidates.append((blocking_score, obj.confidence, name, obj))
    if candidates:
        candidates.sort(reverse=True)
        _, _, name, obj = candidates[0]
        goal = goals_by_object[name]
        return [PlanAction("move-target-to-goal", (name, obj.location, goal.location), source="fallback")]

    for target_name in sorted(unfinished):
        target = state.objects.get(target_name)
        if not target:
            continue
        for blocker_name in sorted(target.blocked_by):
            blocker = state.objects.get(blocker_name)
            if not blocker or not (blocker.graspable and blocker.clear):
                continue
            if blocker_name in goals_by_object:
                goal = goals_by_object[blocker_name]
                return [PlanAction("move-target-to-goal", (blocker_name, blocker.location, goal.location), source="fallback")]
            if free_buffer:
                return [PlanAction("move-obstacle-to-buffer", (blocker_name, blocker.location, free_buffer), source="fallback")]
        if not target.safe:
            for near_name in sorted(target.near):
                near_obj = state.objects.get(near_name)
                if near_obj and not near_obj.is_target and near_obj.graspable and near_obj.clear:
                    if free_buffer:
                        return [PlanAction("move-obstacle-to-buffer", (near_name, near_obj.location, free_buffer), source="fallback")]

    # Recovery: move the clearest visible target even if safe is false; this is
    # useful when the perception thresholds are too conservative.
    recover = []
    for name in unfinished:
        obj = state.objects.get(name)
        if obj and obj.graspable and obj.clear:
            recover.append((obj.confidence, name, obj))
    if recover:
        recover.sort(reverse=True)
        _, name, obj = recover[0]
        goal = goals_by_object[name]
        return [PlanAction("move-target-to-goal", (name, obj.location, goal.location), source="fallback-recovery")]
    return []


def plan(domain_path: Path, problem_path: Path, state: PredicateState) -> tuple[list[PlanAction], str]:
    actions, raw = run_external_planner(domain_path, problem_path)
    if actions:
        return actions, raw
    fallback = fallback_plan(state)
    raw_with_fallback = raw + "\nFallback planner selected:\n" + "\n".join(action.pddl() for action in fallback)
    return fallback, raw_with_fallback
