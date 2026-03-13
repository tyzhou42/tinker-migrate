#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import common


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="为 label SFT 预先创建 run 目录和元数据。")
    p.add_argument("--run-name", type=str, required=True)
    p.add_argument("--dataset-root", type=Path, default=common.DEFAULT_DATASET_ROOT)
    p.add_argument("--dataset-name", type=str, default="ethos")
    p.add_argument("--runs-root", type=Path, default=common.DEFAULT_RUNS_ROOT)
    p.add_argument("--model-name", type=str, default="Qwen/Qwen3-8B")
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--num-epochs", type=float, default=3.0)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-ratio", type=float, default=0.0)
    p.add_argument("--per-device-train-batch-size", type=int, default=4)
    p.add_argument("--per-device-eval-batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--eval-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=10)
    p.add_argument("--save-total-limit", type=int, default=6)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    p.add_argument("--wandb-project", type=str, default="reasoning-sft-gpu")
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_paths = common.make_run_paths(args.run_name, args.runs_root)
    run_paths.run_dir.mkdir(parents=True, exist_ok=True)
    run_paths.student_dir.mkdir(parents=True, exist_ok=True)

    splits = common.load_dataset_splits(args.dataset_root, args.dataset_name)
    common.copy_dataset_splits_to_run(run_paths, splits, args.dataset_root, args.dataset_name)
    common.write_json(
        run_paths.run_dir / "split_summary.json",
        common.build_split_summary(splits, args.dataset_root, args.dataset_name, args.seed),
    )
    common.write_json(
        run_paths.run_dir / "resolved_config.json",
        common.make_resolved_config(
            {
                "dataset_root_dir": str(args.dataset_root),
                "dataset_name": args.dataset_name,
                "student_model_name": args.model_name,
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "num_epochs": args.num_epochs,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "warmup_ratio": args.warmup_ratio,
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "per_device_eval_batch_size": args.per_device_eval_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "logging_steps": args.logging_steps,
                "eval_steps": args.eval_steps,
                "save_steps": args.save_steps,
                "save_total_limit": args.save_total_limit,
                "max_length": args.max_length,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "dtype": args.dtype,
                "wandb_project": args.wandb_project,
                "wandb_mode": args.wandb_mode,
                "seed": args.seed,
            },
            args.run_name,
        ),
    )


if __name__ == "__main__":
    main()
