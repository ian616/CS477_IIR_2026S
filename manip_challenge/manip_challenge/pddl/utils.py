#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import sys
from pathlib import Path


PDDL_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = PDDL_DIR.parent
PACKAGE_ROOT = PACKAGE_DIR.parent
WORKSPACE_ROOT = PACKAGE_ROOT.parent
ASSIGNMENT2_SRC = WORKSPACE_ROOT / "assignment_2"


def ensure_ros_python() -> None:
    ros_python = "/usr/bin/python3"
    if sys.version_info[:2] != (3, 10) and os.path.exists(ros_python):
        os.execv(ros_python, [ros_python, *sys.argv])


def ensure_project_paths() -> None:
    for path in (PACKAGE_ROOT, ASSIGNMENT2_SRC, WORKSPACE_ROOT):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def load_dotenv(path: Path | None = None) -> None:
    env_path = path or PDDL_DIR / ".env"
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def pddl_name(name: str) -> str:
    text = str(name or "").strip().lower()
    text = re.sub(r"^(a|an|the)\s+", "", text)
    text = text.replace("-", "_").replace(" ", "_")
    text = re.sub(r"[^a-z0-9_]", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def lisp_atom(name: str) -> str:
    return pddl_name(name).replace("_", "-")
