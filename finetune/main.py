#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPU 版 finetune 入口。这里只做薄薄一层分发。")
    p.add_argument("command", choices=["label", "teacher_prep", "reasoning", "inference"])
    p.add_argument("extra", nargs=argparse.REMAINDER)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    mapping = {
        "label": SCRIPT_DIR / "sft_label.py",
        "teacher_prep": SCRIPT_DIR / "reasoning_teacher_prep.py",
        "reasoning": SCRIPT_DIR / "reasoning_sft.py",
        "inference": SCRIPT_DIR / "inference.py",
    }
    script = mapping[args.command]
    cmd = [sys.executable, str(script), *args.extra]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
