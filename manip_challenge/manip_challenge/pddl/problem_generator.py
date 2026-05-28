#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from .types import LOCATIONS, PredicateState
from .utils import PDDL_DIR


def predicate_to_lisp(predicate: str) -> str:
    parts = predicate.split()
    return "(" + " ".join(parts) + ")"


def write_problem(state: PredicateState, path: Path | None = None) -> Path:
    # Writes the current planning problem from generated predicates.
    # Edit here if you add object types, locations, goal forms, or want
    # completed/buffer state to persist differently across replans.
    path = path or PDDL_DIR / "problem.pddl"
    object_names = sorted(set(state.objects) | {goal.target_name for goal in state.unfinished_goals()})
    init = sorted(state.predicates)
    goals = [f"(at {goal.target_name} {goal.location})" for goal in state.unfinished_goals()]
    text = [
        "(define (problem manip-generated)",
        "  (:domain manip-tamp)",
        "",
        "  (:objects",
        "    " + " ".join(object_names) + " - item",
        "    " + " ".join(LOCATIONS) + " - location",
        "  )",
        "",
        "  (:init",
    ]
    text.extend("    " + predicate_to_lisp(pred) for pred in init)
    text.extend([
        "  )",
        "",
        "  (:goal",
        "    (and",
    ])
    text.extend("      " + goal for goal in goals)
    text.extend([
        "    )",
        "  )",
        ")",
        "",
    ])
    path.write_text("\n".join(text), encoding="utf-8")
    return path


DOMAIN_TEXT = """(define (domain manip-tamp)
  ;; Action/predicate editing guide:
  ;; - Declare new predicates in :predicates.
  ;; - Emit those predicates in predicate_builder.py::_build_predicates().
  ;; - Add or modify action schemas here.
  ;; - Add matching Python execution handlers in actions.py::ACTION_HANDLERS.
  (:requirements :strips :typing :negative-preconditions)

  (:types
    item location
  )

  (:predicates
    ;; Object role predicates. Generated in predicate_builder.py::_build_predicates().
    (target ?o - item)
    (obstacle ?o - item)
    ;; Location/goal predicates. at(...) and goal-at(...) are generated per object/goal.
    (at ?o - item ?l - location)
    (goal-at ?o - item ?l - location)
    ;; Robot hand state. Current implementation replans after full pick-place actions.
    (handempty)
    (holding ?o - item)
    ;; Manipulation feasibility predicates from predicate_builder.py.
    (clear ?o - item)
    (graspable ?o - item)
    (safe ?o - item)
    ;; Clutter relation predicates from predicate_builder.py::_annotate_relations().
    (blocks ?a - item ?b - item)
    (near ?a - item ?b - item)
    ;; Destination resources. Generated in predicate_builder.py::_build_predicates().
    (buffer ?l - location)
    (buffer-free ?l - location)
    (storage ?l - location)
  )

  ;; Physical action implemented by actions.py::move_target_to_goal.
  (:action move-target-to-goal
    :parameters (?o - item ?from - location ?to - location)
    :precondition (and
      (target ?o)
      (goal-at ?o ?to)
      (handempty)
      (at ?o ?from)
      (clear ?o)
      (graspable ?o)
      (safe ?o)
      (storage ?to)
    )
    :effect (and
      (at ?o ?to)
      (not (at ?o ?from))
      (handempty)
    )
  )

  ;; Physical action implemented by actions.py::move_obstacle_to_buffer.
  (:action move-obstacle-to-buffer
    :parameters (?o - item ?from - location ?buf - location)
    :precondition (and
      (obstacle ?o)
      (handempty)
      (at ?o ?from)
      (clear ?o)
      (graspable ?o)
      (buffer ?buf)
      (buffer-free ?buf)
    )
    :effect (and
      (at ?o ?buf)
      (not (at ?o ?from))
      (not (buffer-free ?buf))
      (handempty)
    )
  )

  ;; Logical relation-cleanup action for external planners.
  ;; In the robot loop, perception usually recomputes this after every action.
  (:action clear-blocking-relation
    :parameters (?blocker - item ?blocked - item)
    :precondition (and
      (blocks ?blocker ?blocked)
      (not (at ?blocker table))
    )
    :effect (and
      (not (blocks ?blocker ?blocked))
      (clear ?blocked)
    )
  )

  ;; Logical relation-cleanup action for external planners.
  ;; In the robot loop, perception usually recomputes this after every action.
  (:action clear-near-relation
    :parameters (?moved - item ?other - item)
    :precondition (and
      (near ?moved ?other)
      (not (at ?moved table))
    )
    :effect (and
      (not (near ?moved ?other))
      (safe ?other)
    )
  )
)
"""


def ensure_domain(path: Path | None = None) -> Path:
    path = path or PDDL_DIR / "domain.pddl"
    if not path.is_file() or path.read_text(encoding="utf-8") != DOMAIN_TEXT:
        path.write_text(DOMAIN_TEXT, encoding="utf-8")
    return path
