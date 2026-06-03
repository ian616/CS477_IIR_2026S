#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


KNOWN_OBJECTS = ("banana", "meat_can", "coke_can", "hammer", "strawberry")
DYNAMIC_BUFFER_LOCATION = "dynamic_buffer"
BUFFER_LOCATIONS = (DYNAMIC_BUFFER_LOCATION,)
LOCATIONS = ("table", "left_storage", "right_storage", "bookshelf", *BUFFER_LOCATIONS)
DESTINATION_TO_LOCATION = {
    "left storage": "left_storage",
    "left_storage": "left_storage",
    "left": "left_storage",
    "right storage": "right_storage",
    "right_storage": "right_storage",
    "right": "right_storage",
    "bookshelf": "bookshelf",
    "shelf": "bookshelf",
    "book shelf": "bookshelf",
}
LOCATION_TO_DESTINATION = {
    "left_storage": "left storage",
    "right_storage": "right storage",
    "bookshelf": "bookshelf",
    "dynamic_buffer": "dynamic_buffer",
}


@dataclass(frozen=True)
class Goal:
    object_name: str
    location: str
    bound_object_name: str | None = None

    @property
    def target_name(self) -> str:
        return self.bound_object_name or self.object_name


@dataclass
class ObjectState:
    name: str
    class_name: str = ""
    instance_index: int | None = None
    is_target: bool = False
    location: str = "table"
    detected: bool = False
    visible: bool = False
    pose_known: bool = False
    graspable: bool = False
    clear: bool = False
    safe: bool = True
    confidence: float = 0.0
    mask_pixels: int = 0
    foreground_points: int = 0
    bbox_xyxy: tuple[int, int, int, int] | None = None
    centroid_xyz: tuple[float, float, float] | None = None
    grasp_xyz: tuple[float, float, float] | None = None
    base_link_xy: tuple[float, float] | None = None
    scene_region: str = "active_table"
    classification_reason: str | None = None
    relation_candidate: bool = True
    depth_median: float | None = None
    blocks: set[str] = field(default_factory=set)
    blocked_by: set[str] = field(default_factory=set)
    near: set[str] = field(default_factory=set)
    detection: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class ActionLedgerEntry:
    action_name: str
    object_name: str
    class_name: str = ""
    source_location: str = ""
    destination_location: str = ""
    dynamic_buffer_place_xy: tuple[float, float] | None = None
    dynamic_buffer_radius_m: float | None = None
    source_base_link_xy: tuple[float, float] | None = None
    result_ok: bool | None = None
    result_status: str = ""
    step: int | None = None
    stamp_sec: float | None = None
    command_id: str | None = None
    anticipated: bool = False
    result: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_name": self.action_name,
            "object_name": self.object_name,
            "class_name": self.class_name,
            "source_location": self.source_location,
            "destination_location": self.destination_location,
            "dynamic_buffer_place_xy": list(self.dynamic_buffer_place_xy) if self.dynamic_buffer_place_xy else None,
            "dynamic_buffer_radius_m": self.dynamic_buffer_radius_m,
            "source_base_link_xy": list(self.source_base_link_xy) if self.source_base_link_xy else None,
            "result_ok": self.result_ok,
            "result_status": self.result_status,
            "step": self.step,
            "stamp_sec": self.stamp_sec,
            "command_id": self.command_id,
            "anticipated": self.anticipated,
            "result": self.result,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ActionLedgerEntry":
        place_xy = payload.get("dynamic_buffer_place_xy")
        source_xy = payload.get("source_base_link_xy")
        return cls(
            action_name=str(payload.get("action_name") or payload.get("name") or ""),
            object_name=str(payload.get("object_name") or ""),
            class_name=str(payload.get("class_name") or ""),
            source_location=str(payload.get("source_location") or ""),
            destination_location=str(payload.get("destination_location") or ""),
            dynamic_buffer_place_xy=tuple(float(v) for v in place_xy[:2]) if place_xy and len(place_xy) >= 2 else None,
            dynamic_buffer_radius_m=(
                float(payload["dynamic_buffer_radius_m"])
                if payload.get("dynamic_buffer_radius_m") is not None else None
            ),
            source_base_link_xy=tuple(float(v) for v in source_xy[:2]) if source_xy and len(source_xy) >= 2 else None,
            result_ok=payload.get("result_ok"),
            result_status=str(payload.get("result_status") or ""),
            step=int(payload["step"]) if payload.get("step") is not None else None,
            stamp_sec=float(payload["stamp_sec"]) if payload.get("stamp_sec") is not None else None,
            command_id=str(payload.get("command_id") or "") or None,
            anticipated=bool(payload.get("anticipated", False)),
            result=payload.get("result"),
        )

    def planning_key(self) -> tuple:
        return (
            self.step,
            self.action_name,
            self.object_name,
            self.class_name,
            self.source_location,
            self.destination_location,
            bool(self.result_ok),
        )


@dataclass
class PredicateState:
    goals: list[Goal]
    objects: dict[str, ObjectState]
    completed: set[str] = field(default_factory=set)
    occupied_buffers: set[str] = field(default_factory=set)
    buffered_obstacles: set[str] = field(default_factory=set)
    action_ledger: list[ActionLedgerEntry] = field(default_factory=list)
    reconciliation: list[dict[str, Any]] = field(default_factory=list)
    raw_observed_objects: dict[str, Any] = field(default_factory=dict)
    ignored_objects: dict[str, str] = field(default_factory=dict)
    relation_input_objects: list[str] = field(default_factory=list)
    predicates: set[str] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)

    def unfinished_goals(self) -> list[Goal]:
        return [goal for goal in self.goals if goal.object_name not in self.completed]

    def free_buffers(self) -> list[str]:
        return [buffer for buffer in BUFFER_LOCATIONS if buffer not in self.occupied_buffers]


@dataclass(frozen=True)
class PlanAction:
    name: str
    args: tuple[str, ...]
    source: str = "fallback"

    def pddl(self) -> str:
        return "(" + " ".join((self.name, *self.args)) + ")"

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "args": list(self.args), "source": self.source, "pddl": self.pddl()}
