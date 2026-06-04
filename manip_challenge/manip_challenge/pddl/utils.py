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
ASSIGNMENT1_SRC = WORKSPACE_ROOT / "assignment_1"
ASSIGNMENT2_SRC = WORKSPACE_ROOT / "assignment_2"
RIRO_KDL_ROOT = WORKSPACE_ROOT / "utils" / "riro-kdl"
PYKDL_UTILS_SRC = RIRO_KDL_ROOT / "pykdl_utils"
HRL_GEOM_SRC = RIRO_KDL_ROOT / "hrl_geom"


def ensure_ros_python() -> None:
    ros_python = "/usr/bin/python3"
    if sys.version_info[:2] != (3, 10) and os.path.exists(ros_python):
        os.execv(ros_python, [ros_python, *sys.argv])


def ensure_project_paths() -> None:
    for path in (
        PACKAGE_ROOT,
        ASSIGNMENT1_SRC,
        ASSIGNMENT2_SRC,
        PYKDL_UTILS_SRC,
        HRL_GEOM_SRC,
        WORKSPACE_ROOT,
    ):
        text = str(path)
        if path.exists() and text not in sys.path:
            sys.path.insert(0, text)


def pddl_name(name: str) -> str:
    text = str(name or "").strip().lower()
    text = re.sub(r"^(a|an|the)\s+", "", text)
    text = text.replace("-", "_").replace(" ", "_")
    text = re.sub(r"[^a-z0-9_]", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def lisp_atom(name: str) -> str:
    return pddl_name(name).replace("_", "-")
