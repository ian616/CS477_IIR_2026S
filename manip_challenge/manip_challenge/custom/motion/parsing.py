#!/usr/bin/env python3
import re


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
        destination = destination.strip()
        objects_str = re.sub(r'\band\b', ',', objects_str, flags=re.IGNORECASE)
        for raw_obj in objects_str.split(','):
            obj = re.sub(r'^\s*(a|an|the)\s+', '', raw_obj.strip(), flags=re.IGNORECASE).strip()
            if obj:
                tasks.append((obj, destination))
    return tasks