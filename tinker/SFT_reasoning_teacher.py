#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from openai import OpenAI
from tqdm import tqdm

import SFT_reasoning as core
try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(dotenv_path: str | os.PathLike[str] | None = None, override: bool = False) -> bool:
        if dotenv_path is None:
            return False
        path = Path(dotenv_path)
        if not path.exists():
            return False
        loaded_any = False
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'\"")
            if not key:
                continue
            if override or key not in os.environ:
                os.environ[key] = value
                loaded_any = True
        return loaded_any


TEACHER_SUBDIR = "teacher_phase"
DEFAULT_DATASET_DIR = "dataset"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reasoning SFT teacher stage (sampling + rejection)")
    p.add_argument("--config", type=str, default="configs/reasoning_sft.example.json")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--dataset-name", "--dataset_name", dest="dataset_name", type=str, default=None)
    p.add_argument("--dataset-root-dir", "--dataset_root_dir", dest="dataset_root_dir", type=str, default=None)
    p.add_argument("--rules-root-dir", "--rules_root_dir", dest="rules_root_dir", type=str, default=None)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)
    p.add_argument("--teacher-k", type=int, default=None)
    p.add_argument("--teacher-workers", "--teacher_workers", dest="teacher_workers", type=int, default=None)
    p.add_argument("--teacher-max-tokens", "--teacher_max_tokens", dest="teacher_max_tokens", type=int, default=None)
    p.add_argument("--teacher-temperature", "--teacher_temperature", dest="teacher_temperature", type=float, default=None)
    p.add_argument("--teacher-model-name", "--teacher_model_name", dest="teacher_model_name", type=str, default=None)
    p.add_argument(
        "--selection-metric",
        choices=["macro_f1", "auroc", "auprc", "accuracy", "loss", "cls_loss"],
        default=None,
    )
    p.add_argument("--max-train-examples", type=int, default=None)
    p.add_argument("--max-val-examples", type=int, default=None)
    p.add_argument("--max-test-examples", type=int, default=None)
    return p.parse_args()


def _first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists() and p.is_file():
            return p
    return None


def _resolve_split_paths(project_root: Path, cfg: dict[str, Any]) -> tuple[dict[str, Path], Path, str]:
    dataset_name = str(cfg.get("dataset_name", "ethos")).strip()
    dataset_root = core.resolve_path(project_root, str(cfg.get("dataset_root_dir", DEFAULT_DATASET_DIR)))
    if dataset_root is None or not dataset_root.exists():
        raise FileNotFoundError(f"Missing dataset root directory: {dataset_root}")

    dataset_dir = dataset_root / dataset_name
    aliases = {
        "train": ["train"],
        "val": ["val", "valid", "validation"],
        "test": ["test"],
    }
    split_paths: dict[str, Path] = {}
    missing_info: list[str] = []

    for split, names in aliases.items():
        candidates: list[Path] = []
        for name in names:
            # Preferred layout: dataset/<dataset_name>/<dataset_name>_<split>.csv
            candidates.append(dataset_dir / f"{dataset_name}_{name}.csv")
            # Also support generic names in dataset subfolder.
            candidates.append(dataset_dir / f"{name}.csv")
            # Also support flat layout under dataset root.
            candidates.append(dataset_root / f"{dataset_name}_{name}.csv")
            candidates.append(dataset_root / f"{name}.csv")

        resolved = _first_existing(candidates)
        if resolved is None:
            pretty = ", ".join(str(p) for p in candidates)
            missing_info.append(f"{split}: [{pretty}]")
        else:
            split_paths[split] = resolved

    if missing_info:
        raise FileNotFoundError(
            "Could not resolve dataset split files for dataset "
            f"'{dataset_name}'. Tried: {' | '.join(missing_info)}"
        )

    return split_paths, dataset_dir, dataset_name


def _resolve_rules_dir(project_root: Path, cfg: dict[str, Any], dataset_name: str) -> Path:
    rules_root = core.resolve_path(project_root, str(cfg.get("rules_root_dir", "rules")))
    if rules_root is not None:
        dataset_rules = rules_root / dataset_name
        if dataset_rules.exists() and dataset_rules.is_dir():
            return dataset_rules

    # Backward compatibility with existing rules_dir behavior.
    legacy_rules = core.resolve_path(project_root, str(cfg.get("rules_dir", "rules")))
    if legacy_rules is not None:
        legacy_dataset_rules = legacy_rules / dataset_name
        if legacy_dataset_rules.exists() and legacy_dataset_rules.is_dir():
            return legacy_dataset_rules
        if legacy_rules.exists() and legacy_rules.is_dir():
            return legacy_rules

    raise FileNotFoundError(
        "Could not resolve rules directory. Tried rules_root_dir/dataset_name and rules_dir variants."
    )


def _load_split_dataset(path: Path, cfg: dict[str, Any], split_name: str):
    attempts: list[tuple[str, str, str | None]] = [
        # Preferred for pre-split files under tinker/dataset/*.
        ("text", "label", None),
        # Fallbacks for custom datasets/configs.
        (str(cfg["text_column"]), str(cfg["label_column"]), None),
        (
            str(cfg["text_column"]),
            str(cfg["label_column"]),
            None if cfg.get("csv_sep") is None else str(cfg.get("csv_sep")),
        ),
    ]
    tried: list[str] = []
    last_exc: Exception | None = None

    for text_col, label_col, sep in attempts:
        key = f"text={text_col},label={label_col},sep={sep}"
        if key in tried:
            continue
        tried.append(key)
        try:
            df = core.load_dataset(
                data_path=path,
                text_column=text_col,
                label_column=label_col,
                csv_sep=sep,
                label_threshold=float(cfg["label_threshold"]),
            )
            logger.info(
                "Loaded {} split from {} using columns ({}, {})",
                split_name,
                path,
                text_col,
                label_col,
            )
            return df
        except Exception as exc:
            last_exc = exc

    raise RuntimeError(
        f"Failed loading split '{split_name}' from {path}. Tried: {tried}. Last error: {last_exc}"
    )


def _example_idx_from_id(example_id: Any) -> int | None:
    s = str(example_id or "").strip()
    if not s.startswith("train_"):
        return None
    tail = s[len("train_") :]
    try:
        return int(tail)
    except Exception:
        return None


def _build_hard_tag_frame(train_df, accepted_rows: list[dict[str, Any]]):
    accepted_idx: set[int] = set()
    for row in accepted_rows:
        idx = _example_idx_from_id(row.get("example_id"))
        if idx is not None:
            accepted_idx.add(idx)

    hard_flags = [0 if i in accepted_idx else 1 for i in range(len(train_df))]
    hard_df = train_df.copy()
    hard_df["hard"] = hard_flags
    hard_df["example_id"] = [f"train_{i}" for i in range(len(train_df))]
    return hard_df


def _safe_div(n: float, d: float) -> float:
    if d == 0:
        return 0.0
    return float(n / d)


def _macro_f1_from_confusion(tp: int, tn: int, fp: int, fn: int) -> float:
    # Matches sklearn f1_score(..., average="macro", labels=[0,1], zero_division=0)
    f1_pos = _safe_div(2 * tp, 2 * tp + fp + fn)
    f1_neg = _safe_div(2 * tn, 2 * tn + fp + fn)
    return float((f1_pos + f1_neg) / 2.0)


def _teacher_label_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    derived = core.binary_metrics(y_true, y_pred)
    return {
        "accuracy": float(derived["accuracy"]),
        "macro_f1": float(_macro_f1_from_confusion(int(derived["tp"]), int(derived["tn"]), int(derived["fp"]), int(derived["fn"]))),
        "precision": float(derived["precision"]),
        "recall": float(derived["recall"]),
        "tp": int(derived["tp"]),
        "fp": int(derived["fp"]),
        "fn": int(derived["fn"]),
        "tn": int(derived["tn"]),
    }


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    cfg = core.load_config(project_root, args)
    prompt_cfg = core.load_prompt_config(project_root, cfg)

    random.seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))

    run_name = cfg["run_name"] or f"reasoning_sft_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = core.resolve_path(project_root, str(cfg["log_dir"]))
    if log_dir is None:
        raise RuntimeError("log_dir resolution failed")
    run_dir = log_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    teacher_dir = run_dir / TEACHER_SUBDIR
    teacher_dir.mkdir(parents=True, exist_ok=True)
    core.setup_logger(teacher_dir)
    logger.info("Run dir: {}", run_dir)
    logger.info("Teacher artifact dir: {}", teacher_dir)

    (run_dir / "resolved_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    env_file = core.resolve_path(project_root, str(cfg["teacher_env_file"]))
    if env_file is None or not env_file.exists():
        raise FileNotFoundError(f"Missing env file: {env_file}")
    load_dotenv(env_file, override=False)

    teacher_api_key = os.environ.get(str(cfg["teacher_api_key_env"]))
    if not teacher_api_key:
        raise RuntimeError(f"Environment variable {cfg['teacher_api_key_env']} is not set")

    wandb_run = None
    if wandb is None:
        logger.warning("wandb import failed; teacher logging will be local-only.")
    elif str(cfg.get("wandb_mode", "disabled")) != "disabled":
        try:
            wandb_run = wandb.init(
                project=str(cfg.get("wandb_project", "reasoning-sft-tinker")),
                entity=cfg.get("wandb_entity"),
                name=f"{run_name}_teacher",
                config=cfg,
                mode=str(cfg.get("wandb_mode", "online")),
                dir=str(run_dir),
            )
        except Exception as exc:
            logger.warning("WandB init failed ({}). Teacher phase will continue without remote logging.", exc)
            wandb_run = None

    split_paths, dataset_dir, dataset_name = _resolve_split_paths(project_root, cfg)
    train_path = split_paths["train"]
    val_path = split_paths["val"]
    test_path = split_paths["test"]
    logger.info(
        "Resolved dataset '{}' splits: train={}, val={}, test={}",
        dataset_name,
        train_path,
        val_path,
        test_path,
    )

    train_df = _load_split_dataset(train_path, cfg, "train")
    val_df = _load_split_dataset(val_path, cfg, "val")
    test_df = _load_split_dataset(test_path, cfg, "test")

    source_summary = {
        "train": {
            "count": int(len(train_df)),
            "label_counts": core.label_counts(train_df),
            "label_distribution": core.label_distribution(train_df),
        },
        "val": {
            "count": int(len(val_df)),
            "label_counts": core.label_counts(val_df),
            "label_distribution": core.label_distribution(val_df),
        },
        "test": {
            "count": int(len(test_df)),
            "label_counts": core.label_counts(test_df),
            "label_distribution": core.label_distribution(test_df),
        },
    }

    train_df = core.stratified_subset(
        train_df,
        int(cfg["max_train_examples"]),
        seed=int(cfg["seed"]),
        split_name="train",
    )
    val_df = core.stratified_subset(
        val_df,
        int(cfg["max_val_examples"]),
        seed=int(cfg["seed"]) + 1,
        split_name="val",
    )
    test_df = core.stratified_subset(
        test_df,
        int(cfg["max_test_examples"]),
        seed=int(cfg["seed"]) + 2,
        split_name="test",
    )

    splits_dir = run_dir / "data_splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(splits_dir / "train.csv", index=False, encoding="utf-8")
    val_df.to_csv(splits_dir / "val.csv", index=False, encoding="utf-8")
    test_df.to_csv(splits_dir / "test.csv", index=False, encoding="utf-8")

    split_summary = {
        "counts": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
            "total": int(len(train_df) + len(val_df) + len(test_df)),
        },
        "subset_limits": {
            "max_train_examples": int(cfg["max_train_examples"]),
            "max_val_examples": int(cfg["max_val_examples"]),
            "max_test_examples": int(cfg["max_test_examples"]),
        },
        "source_split_summary": source_summary,
        "used_split_summary": {
            "train": {
                "count": int(len(train_df)),
                "label_counts": core.label_counts(train_df),
                "label_distribution": core.label_distribution(train_df),
            },
            "val": {
                "count": int(len(val_df)),
                "label_counts": core.label_counts(val_df),
                "label_distribution": core.label_distribution(val_df),
            },
            "test": {
                "count": int(len(test_df)),
                "label_counts": core.label_counts(test_df),
                "label_distribution": core.label_distribution(test_df),
            },
        },
        "seed": int(cfg["seed"]),
        "dataset_name": dataset_name,
        "source_dir": str(dataset_dir),
        "resolved_split_paths": {k: str(v) for k, v in split_paths.items()},
    }
    (run_dir / "split_summary.json").write_text(json.dumps(split_summary, indent=2), encoding="utf-8")
    logger.info("Split sizes: train={}, val={}, test={}", len(train_df), len(val_df), len(test_df))

    rules_dir = _resolve_rules_dir(project_root, cfg, dataset_name)
    rulebook, rule_files = core.read_rulebook(rules_dir, str(cfg["rules_glob"]))
    (teacher_dir / "rulebook.txt").write_text(rulebook, encoding="utf-8")
    (teacher_dir / "rule_files.json").write_text(json.dumps(rule_files, indent=2), encoding="utf-8")
    logger.info("Loaded {} rule files from {} for dataset '{}'", len(rule_files), rules_dir, dataset_name)

    teacher_samples: list[dict[str, Any]] = []
    min_reasoning_chars = int(cfg["min_reasoning_chars"])
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    k = int(cfg["teacher_k"])
    teacher_workers = int(cfg["teacher_workers"])
    logger.info("Teacher sampling config: k={}, workers={}", k, teacher_workers)
    # Policy: per-example first try uses temperature=0.0 (single attempt),
    # and subsequent tries ("retries") use cfg["teacher_temperature"] with max_retries.
    teacher_retry_temperature = float(cfg["teacher_temperature"])
    logger.info(
        "Teacher sampling temperature policy: first_try_temp=0.0 (no internal retries), retry_temp={} (with max_retries={})",
        teacher_retry_temperature,
        int(cfg["teacher_max_retries"]),
    )
    thread_local = threading.local()

    def get_teacher_client() -> OpenAI:
        cli = getattr(thread_local, "teacher_client", None)
        if cli is None:
            cli = OpenAI(api_key=teacher_api_key, base_url=str(cfg["teacher_base_url"]))
            thread_local.teacher_client = cli
        return cli

    def sample_one_example(idx: int, text: str, gold_label: int) -> dict[str, Any]:
        messages = core.build_teacher_messages(prompt_cfg, rulebook, text)
        accepted_for_example = False
        local_teacher_samples: list[dict[str, Any]] = []
        local_accepted: list[dict[str, Any]] = []
        local_rejected: list[dict[str, Any]] = []
        client_for_example = get_teacher_client()

        # Per-example metrics: label from first try (k0) vs label after allowing retries (until accepted or exhausted).
        first_try_pred: int | None = None
        first_try_parse_ok = False
        final_pred: int = 0
        final_parse_ok = False

        for ki in range(k):
            # First attempt: deterministic temp=0, but do not internally retry; if it doesn't parse/meet criteria,
            # subsequent tries use retry temperature and internal retries.
            if ki == 0:
                temperature = 0.0
                max_retries = 1
            else:
                temperature = teacher_retry_temperature
                max_retries = int(cfg["teacher_max_retries"])

            raw_output, parsed, error = core.request_teacher_sample(
                client_for_example,
                messages,
                model=str(cfg["teacher_model"]),
                temperature=float(temperature),
                max_tokens=int(cfg["teacher_max_tokens"]),
                timeout_seconds=int(cfg["teacher_request_timeout"]),
                max_retries=int(max_retries),
                json_mode=bool(cfg["teacher_json_mode"]),
            )

            sample = {
                "sample_id": f"train_{idx}_k{ki}",
                "example_id": f"train_{idx}",
                "split": "train",
                "k_index": ki,
                "temperature": float(temperature),
                "max_retries_used": int(max_retries),
                "text": text,
                "gold_label": gold_label,
                "raw_output": raw_output,
                "parse_ok": parsed is not None,
                "parse_source": None if parsed is None else parsed.source,
                "pred_label": None if parsed is None else parsed.label,
                "reasoning": None if parsed is None else parsed.reasoning,
                "error": error,
            }

            if ki == 0:
                first_try_parse_ok = bool(sample["parse_ok"])
                if first_try_parse_ok and sample.get("pred_label") in {0, 1}:
                    first_try_pred = int(sample["pred_label"])
                else:
                    first_try_pred = 0

            # Rejection sampling with early stop:
            # keep trying this example until first valid trace or max k attempts.
            if not sample["parse_ok"]:
                sample["reject_reason"] = sample.get("error") or "parse_failed"
                local_rejected.append(sample)
                local_teacher_samples.append(sample)
                final_pred = 0
                final_parse_ok = False
            else:
                pred_label = int(sample["pred_label"])
                reasoning = str(sample["reasoning"] or "").strip()
                if pred_label != gold_label:
                    sample["reject_reason"] = "label_mismatch"
                    local_rejected.append(sample)
                    local_teacher_samples.append(sample)
                    final_pred = pred_label
                    final_parse_ok = True
                elif len(reasoning) < min_reasoning_chars:
                    sample["reject_reason"] = "reasoning_too_short"
                    local_rejected.append(sample)
                    local_teacher_samples.append(sample)
                    final_pred = pred_label
                    final_parse_ok = True
                else:
                    local_teacher_samples.append(sample)
                    local_accepted.append(
                        {
                            "sample_id": sample["sample_id"],
                            "example_id": sample["example_id"],
                            "text": sample["text"],
                            "label": gold_label,
                            "reasoning": reasoning,
                            "teacher_pred_label": pred_label,
                            "parse_source": sample.get("parse_source"),
                        }
                    )
                    accepted_for_example = True
                    final_pred = pred_label
                    final_parse_ok = True

            sleep_seconds = float(cfg["teacher_sleep_seconds"])
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

            if accepted_for_example:
                break

        return {
            "idx": idx,
            "teacher_samples": local_teacher_samples,
            "accepted": local_accepted,
            "rejected": local_rejected,
            "first_try_pred_label": 0 if first_try_pred is None else int(first_try_pred),
            "first_try_parse_ok": bool(first_try_parse_ok),
            "final_pred_label": int(final_pred),
            "final_parse_ok": bool(final_parse_ok),
        }

    pbar = tqdm(total=len(train_df), desc="Teacher sampling", unit="example")
    results_by_idx: dict[int, dict[str, Any]] = {}
    if teacher_workers <= 1:
        for idx, row in train_df.iterrows():
            result = sample_one_example(
                idx=int(idx),
                text=str(row["text"]),
                gold_label=int(row["label"]),
            )
            results_by_idx[int(idx)] = result
            pbar.update(1)
    else:
        with ThreadPoolExecutor(max_workers=teacher_workers) as pool:
            futures = {
                pool.submit(
                    sample_one_example,
                    int(idx),
                    str(row["text"]),
                    int(row["label"]),
                ): int(idx)
                for idx, row in train_df.iterrows()
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    raise RuntimeError(f"Teacher sampling failed for example {idx}: {exc}") from exc
                results_by_idx[idx] = result
                pbar.update(1)
    pbar.close()

    for idx in range(len(train_df)):
        result = results_by_idx[idx]
        teacher_samples.extend(result["teacher_samples"])
        accepted.extend(result["accepted"])
        rejected.extend(result["rejected"])

    core.write_jsonl(teacher_dir / "teacher_samples.jsonl", teacher_samples)

    accepted_example_ids = [str(x["example_id"]) for x in accepted]
    if len(set(accepted_example_ids)) != len(accepted_example_ids):
        raise RuntimeError("Internal error: more than one accepted trace found for at least one example")

    core.write_jsonl(teacher_dir / "accepted_samples.jsonl", accepted)
    core.write_jsonl(teacher_dir / "rejected_samples.jsonl", rejected)

    # Teacher label metrics: compare first-try temp=0 vs allowing retries (k attempts) policy.
    y_true = [int(x) for x in train_df["label"].tolist()]
    y_pred_first = [int(results_by_idx[i]["first_try_pred_label"]) for i in range(len(train_df))]
    y_pred_final = [int(results_by_idx[i]["final_pred_label"]) for i in range(len(train_df))]
    first_invalid = sum(1 for i in range(len(train_df)) if not bool(results_by_idx[i].get("first_try_parse_ok", False)))
    final_invalid = sum(1 for i in range(len(train_df)) if not bool(results_by_idx[i].get("final_parse_ok", False)))

    teacher_first_metrics = _teacher_label_metrics(y_true, y_pred_first)
    teacher_with_retries_metrics = _teacher_label_metrics(y_true, y_pred_final)

    # Log to wandb (single step, teacher phase).
    if wandb_run is not None:
        wandb.log(
            {
                "teacher_first_try/accuracy": teacher_first_metrics["accuracy"],
                "teacher_first_try/macro_f1": teacher_first_metrics["macro_f1"],
                "teacher_first_try/precision": teacher_first_metrics["precision"],
                "teacher_first_try/recall": teacher_first_metrics["recall"],
                "teacher_first_try/tp": teacher_first_metrics["tp"],
                "teacher_first_try/fp": teacher_first_metrics["fp"],
                "teacher_first_try/fn": teacher_first_metrics["fn"],
                "teacher_first_try/tn": teacher_first_metrics["tn"],
                "teacher_first_try/invalid_rate": float(first_invalid / max(1, len(train_df))),
                "teacher_with_retries/accuracy": teacher_with_retries_metrics["accuracy"],
                "teacher_with_retries/macro_f1": teacher_with_retries_metrics["macro_f1"],
                "teacher_with_retries/precision": teacher_with_retries_metrics["precision"],
                "teacher_with_retries/recall": teacher_with_retries_metrics["recall"],
                "teacher_with_retries/tp": teacher_with_retries_metrics["tp"],
                "teacher_with_retries/fp": teacher_with_retries_metrics["fp"],
                "teacher_with_retries/fn": teacher_with_retries_metrics["fn"],
                "teacher_with_retries/tn": teacher_with_retries_metrics["tn"],
                "teacher_with_retries/invalid_rate": float(final_invalid / max(1, len(train_df))),
            },
            step=0,
        )

    teacher_summary = {
        "teacher_model": str(cfg["teacher_model"]),
        "teacher_workers": teacher_workers,
        "requested_samples": int(len(teacher_samples)),
        "accepted_samples": int(len(accepted)),
        "rejected_samples": int(len(rejected)),
        "acceptance_rate": float(len(accepted) / max(1, len(teacher_samples))),
        "temperature_policy": {
            "first_try_temperature": 0.0,
            "retry_temperature": float(teacher_retry_temperature),
            "first_try_max_retries": 1,
            "retry_max_retries": int(cfg["teacher_max_retries"]),
        },
        "first_try_metrics": teacher_first_metrics,
        "with_retries_metrics": teacher_with_retries_metrics,
        "first_try_invalid_rate": float(first_invalid / max(1, len(train_df))),
        "with_retries_invalid_rate": float(final_invalid / max(1, len(train_df))),
    }
    (teacher_dir / "teacher_summary.json").write_text(json.dumps(teacher_summary, indent=2), encoding="utf-8")
    logger.info(
        "Teacher samples: requested={}, accepted={}, rejected={}, acceptance_rate={:.4f}",
        teacher_summary["requested_samples"],
        teacher_summary["accepted_samples"],
        teacher_summary["rejected_samples"],
        teacher_summary["acceptance_rate"],
    )
    logger.info(
        "Teacher first try (temp=0) metrics: acc={:.4f} macro_f1={:.4f} precision={:.4f} recall={:.4f} tp={} fp={} fn={} tn={} invalid_rate={:.2%}",
        float(teacher_first_metrics["accuracy"]),
        float(teacher_first_metrics["macro_f1"]),
        float(teacher_first_metrics["precision"]),
        float(teacher_first_metrics["recall"]),
        int(teacher_first_metrics["tp"]),
        int(teacher_first_metrics["fp"]),
        int(teacher_first_metrics["fn"]),
        int(teacher_first_metrics["tn"]),
        float(first_invalid / max(1, len(train_df))),
    )
    logger.info(
        "Teacher w/ retries metrics: acc={:.4f} macro_f1={:.4f} precision={:.4f} recall={:.4f} tp={} fp={} fn={} tn={} invalid_rate={:.2%}",
        float(teacher_with_retries_metrics["accuracy"]),
        float(teacher_with_retries_metrics["macro_f1"]),
        float(teacher_with_retries_metrics["precision"]),
        float(teacher_with_retries_metrics["recall"]),
        int(teacher_with_retries_metrics["tp"]),
        int(teacher_with_retries_metrics["fp"]),
        int(teacher_with_retries_metrics["fn"]),
        int(teacher_with_retries_metrics["tn"]),
        float(final_invalid / max(1, len(train_df))),
    )

    if not accepted:
        raise RuntimeError("No accepted teacher samples after rejection sampling")

    hard_df = _build_hard_tag_frame(train_df, accepted)
    hard_local_path = splits_dir / "train_with_hard_tag.csv"
    hard_df.to_csv(hard_local_path, index=False, encoding="utf-8")

    source_train_count = int(source_summary["train"]["count"])
    full_train_run = int(len(train_df)) == source_train_count
    hard_dataset_path = dataset_dir / "train_with_hard_tag.csv"
    if full_train_run:
        hard_df.to_csv(hard_dataset_path, index=False, encoding="utf-8")
        logger.info(
            "Regenerated hard-tag dataset: {} (hard={}, easy={}, n={})",
            hard_dataset_path,
            int(hard_df["hard"].sum()),
            int((1 - hard_df["hard"]).sum()),
            int(len(hard_df)),
        )
    else:
        logger.warning(
            "Skipped dataset-level hard-tag overwrite because this was not a full train run "
            "(used {} of {} rows). Run-local file written to {}",
            int(len(train_df)),
            source_train_count,
            hard_local_path,
        )

    phase_summary = {
        "run_dir": str(run_dir),
        "teacher_dir": str(teacher_dir),
        "split_summary": split_summary,
        "teacher_summary": teacher_summary,
        "hard_tag_summary": {
            "full_train_run": bool(full_train_run),
            "run_local_path": str(hard_local_path),
            "dataset_path": str(hard_dataset_path),
            "hard_count": int(hard_df["hard"].sum()),
            "easy_count": int((1 - hard_df["hard"]).sum()),
            "total_count": int(len(hard_df)),
        },
    }
    (teacher_dir / "teacher_phase_summary.json").write_text(json.dumps(phase_summary, indent=2), encoding="utf-8")
    logger.info(
        "Teacher phase complete | run={} | requested={} accepted={} rejected={} acceptance={:.2%}",
        run_name,
        teacher_summary["requested_samples"],
        teacher_summary["accepted_samples"],
        teacher_summary["rejected_samples"],
        float(teacher_summary["acceptance_rate"]),
    )
    logger.info("Artifacts saved at: {}", teacher_dir)
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
