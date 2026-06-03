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
    depth_median: float | None = None
    blocks: set[str] = field(default_factory=set)
    blocked_by: set[str] = field(default_factory=set)
    near: set[str] = field(default_factory=set)
    detection: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class PredicateState:
    goals: list[Goal]
    objects: dict[str, ObjectState]
    completed: set[str] = field(default_factory=set)
    occupied_buffers: set[str] = field(default_factory=set)
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
