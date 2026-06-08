#!/usr/bin/env python3
import re


DESTINATION_ALIASES = {
    "left": "left storage",
    "left_storage": "left storage",
    "left storage": "left storage",
    "storage a": "left storage",
    "storage_a": "left storage",
    "right": "right storage",
    "right_storage": "right storage",
    "right storage": "right storage",
    "storage b": "right storage",
    "storage_b": "right storage",
    "book": "bookshelf",
    "book shelf": "bookshelf",
    "bookshelf": "bookshelf",
    "shelf": "bookshelf",
}


def normalize_destination(name):
    text = str(name or "").strip().lower()
    text = re.sub(r"^(a|an|the)\s+", "", text)
    text = text.replace("-", " ").replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return DESTINATION_ALIASES.get(text, DESTINATION_ALIASES.get(text.replace(" ", "_"), text))


def parse_task_commands(text):
    """Parse natural-language instruction into list of (object_name, destination) tuples.

    Example input:
      "Move a book, an eraser, and a soap to the left storage."
    Example output:
      [("book", "left storage"), ("eraser", "left storage"), ("soap", "left storage")]
    """
    tasks = []
    sentence_pattern = r'Move\s+(.*?)\s+to\s+(?:the\s+)?([\w\s]+?)(?:\.|$)'
    for objects_str, destination in re.findall(sentence_pattern, text, re.IGNORECASE):
        destination = normalize_destination(destination)
        objects_str = re.sub(r'\band\b', ',', objects_str, flags=re.IGNORECASE)
        for raw_obj in objects_str.split(','):
            obj = re.sub(r'^\s*(a|an|the)\s+', '', raw_obj.strip(), flags=re.IGNORECASE).strip()
            if obj:
                tasks.append((obj, destination))
    return tasks
