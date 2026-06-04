#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from typing import Iterable

from .pddl_types import DESTINATION_TO_LOCATION, Goal, KNOWN_OBJECTS
from .utils import pddl_name


OBJECT_ALIASES = {
    "banana": "banana",
    "meat": "meat_can",
    "meat can": "meat_can",
    "meat_can": "meat_can",
    "can of meat": "meat_can",
    "coke": "coke_can",
    "coke can": "coke_can",
    "coke_can": "coke_can",
    "cola": "coke_can",
    "hammer": "hammer",
    "strawberry": "strawberry",
}


def normalize_object(name: str) -> str:
    text = str(name or "").strip().lower()
    text = re.sub(r"^(a|an|the)\s+", "", text)
    text = text.replace("-", " ").replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return OBJECT_ALIASES.get(text, pddl_name(text))


def normalize_location(name: str) -> str:
    text = str(name or "").strip().lower()
    text = re.sub(r"^(a|an|the)\s+", "", text)
    text = text.replace("-", " ").replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return DESTINATION_TO_LOCATION.get(text, DESTINATION_TO_LOCATION.get(text.replace(" ", "_"), pddl_name(text)))


def _dedupe(goals: Iterable[Goal]) -> list[Goal]:
    output = []
    seen = set()
    for goal in goals:
        key = (goal.object_name, goal.location)
        if goal.object_name in KNOWN_OBJECTS and goal.location and key not in seen:
            seen.add(key)
            output.append(goal)
    return output


def parse_task_commands(text: str) -> list[tuple[str, str]]:
    """Parse natural-language instruction into raw (object_name, destination) tuples.

    Example input:
      "Move a book, an eraser, and a soap to the left storage."
    Example output:
      [("book", "left storage"), ("eraser", "left storage"), ("soap", "left storage")]

    This intentionally stays simple so the normal path does not depend on an
    LLM.  parse_goals_rule_based() normalizes these raw names for PDDL.
    """
    tasks = []
    sentence_pattern = r"Move\s+(.*?)\s+(?:to|in|on|into|onto)\s+(?:the\s+)?([\w\s]+?)(?:\.|$)"
    for objects_str, destination in re.findall(sentence_pattern, str(text or ""), re.IGNORECASE):
        destination = destination.strip()
        objects_str = re.sub(r"\band\b", ",", objects_str, flags=re.IGNORECASE)
        for raw_obj in objects_str.split(","):
            obj = re.sub(r"^\s*(a|an|the)\s+", "", raw_obj.strip(), flags=re.IGNORECASE).strip()
            if obj:
                tasks.append((obj, destination))
    return tasks


def parse_goals_rule_based(text: str) -> list[Goal]:
    return _dedupe(
        Goal(normalize_object(object_name), normalize_location(destination))
        for object_name, destination in parse_task_commands(text)
    )


def _gemini_prompt(text: str) -> str:
    objects = ", ".join(KNOWN_OBJECTS)
    locations = "left_storage, right_storage, bookshelf"
    return (
        "Extract pick-and-place goals for a robot. Return only JSON with a top-level goals array. "
        "Each goal must have object and location. Use underscore names. "
        f"Allowed objects: {objects}. Allowed locations: {locations}. "
        f"Command: {text}"
    )


def _parse_json_response(text: str) -> list[Goal]:
    match = re.search(r"\{.*\}", str(text or ""), re.DOTALL)
    payload = match.group(0) if match else str(text or "")
    data = json.loads(payload)
    return _dedupe(
        Goal(normalize_object(item.get("object", "")), normalize_location(item.get("location", item.get("destination", ""))))
        for item in data.get("goals", [])
    )


def parse_goals_with_gemini(text: str, api_key: str = "", model: str = "gemini-2.0-flash") -> list[Goal]:
    api_key = str(api_key or "").strip()
    if not api_key:
        return []
    try:
        from google import genai
    except ImportError:
        return []
    model = str(model or "gemini-2.0-flash").strip()
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=model, contents=_gemini_prompt(text))
    return _parse_json_response(response.text or "")


def parse_goals(
    text: str,
    use_gemini: bool = True,
    gemini_api_key: str = "",
    gemini_model: str = "gemini-2.0-flash",
) -> list[Goal]:
    goals = parse_goals_rule_based(text)
    if goals:
        return goals
    if use_gemini:
        try:
            goals = parse_goals_with_gemini(text, api_key=gemini_api_key, model=gemini_model)
            if goals:
                return goals
        except Exception:
            pass
    return []
