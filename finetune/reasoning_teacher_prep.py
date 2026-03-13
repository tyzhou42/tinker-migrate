#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import common


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="把历史 teacher 产物整理到新的 reasoning run 目录里。")
    p.add_argument("--run-name", type=str, required=True)
    p.add_argument("--source-run-name", type=str, default="reasoning_sft_20260301_114319")
    p.add_argument("--dataset-root", type=Path, default=common.DEFAULT_DATASET_ROOT)
    p.add_argument("--dataset-name", type=str, default="ethos")
    p.add_argument("--runs-root", type=Path, default=common.DEFAULT_RUNS_ROOT)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_paths = common.make_run_paths(args.run_name, args.runs_root)
    run_paths.run_dir.mkdir(parents=True, exist_ok=True)

    splits = common.load_dataset_splits(args.dataset_root, args.dataset_name)
    common.copy_dataset_splits_to_run(run_paths, splits, args.dataset_root, args.dataset_name)
    split_summary = common.build_split_summary(splits, args.dataset_root, args.dataset_name, args.seed)
    common.write_json(run_paths.run_dir / "split_summary.json", split_summary)

    teacher_info = common.materialize_teacher_artifacts(
        source_run_name=args.source_run_name,
        run_paths=run_paths,
        dataset_name=args.dataset_name,
        splits=splits,
        seed=args.seed,
    )
    common.write_json(
        run_paths.run_dir / "teacher_source.json",
        {
            "source_run_name": args.source_run_name,
            "accepted_samples": len(teacher_info["accepted_rows"]),
            "rejected_samples": len(teacher_info["rejected_rows"]),
            "rule_files": teacher_info["rule_files"],
        },
    )


if __name__ == "__main__":
    main()
