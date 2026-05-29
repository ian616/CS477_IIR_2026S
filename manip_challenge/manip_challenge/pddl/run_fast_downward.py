#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: run_fast_downward.py DOMAIN PROBLEM", file=sys.stderr)
        return 2

    domain_path = Path(sys.argv[1]).resolve()
    problem_path = Path(sys.argv[2]).resolve()
    pddl_dir = Path(__file__).resolve().parent
    repo_root = pddl_dir.parents[2]
    fast_downward = repo_root / "third_party" / "downward" / "fast-downward.py"

    if not fast_downward.is_file():
        print(f"Fast Downward not found: {fast_downward}", file=sys.stderr)
        print("Run: git submodule update --init --recursive", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory(prefix="manip_fd_") as tmpdir:
        plan_file = Path(tmpdir) / "sas_plan"
        cmd = [
            str(fast_downward),
            "--plan-file",
            str(plan_file),
            "--alias",
            "lama-first",
            str(domain_path),
            str(problem_path),
        ]
        completed = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

        plan_files = sorted(plan_file.parent.glob(plan_file.name + "*"))
        if plan_files:
            print(plan_files[-1].read_text(encoding="utf-8"))
        else:
            print(completed.stdout)
        return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
