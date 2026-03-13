#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import random
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from datasets import Dataset
from peft import LoraConfig
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainerCallback


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "tinker" / "dataset"
DEFAULT_RULES_ROOT = PROJECT_ROOT / "tinker" / "rules"
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "finetune" / "runs"
DEFAULT_MODEL_ROOT = PROJECT_ROOT / "finetune" / "model"
DEFAULT_PROMPT_ROOT = PROJECT_ROOT / "tinker" / "prompts"


@dataclass
class RunPaths:
    run_name: str
    run_dir: Path
    data_splits_dir: Path
    teacher_dir: Path
    student_dir: Path


@dataclass
class ParsedReasoningOutput:
    reasoning: str
    label: int
    source: str


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_yaml_like_prompts(path: Path) -> dict[str, str]:
    import yaml

    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {str(key): str(value) for key, value in loaded.items()}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_pad_token(tokenizer: Any) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token


def infer_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_tokenizer(model_name: str) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    ensure_pad_token(tokenizer)
    return tokenizer


def load_model(model_name: str, *, dtype_name: str, use_gradient_checkpointing: bool) -> Any:
    torch_dtype = infer_dtype(dtype_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model


def make_lora_config(rank: int, alpha: int, dropout: float) -> LoraConfig:
    # 这里用一组对 Qwen 系列比较稳的 target modules，避免再搞复杂的自动推断。
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "gate_proj",
            "down_proj",
        ],
    )


def load_split_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    return df


def load_dataset_splits(dataset_root: Path, dataset_name: str) -> dict[str, pd.DataFrame]:
    dataset_dir = dataset_root / dataset_name
    return {
        "train": load_split_csv(dataset_dir / f"{dataset_name}_train.csv"),
        "val": load_split_csv(dataset_dir / f"{dataset_name}_val.csv"),
        "test": load_split_csv(dataset_dir / f"{dataset_name}_test.csv"),
    }


def copy_dataset_splits_to_run(paths: RunPaths, splits: dict[str, pd.DataFrame], dataset_root: Path, dataset_name: str) -> None:
    paths.data_splits_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = dataset_root / dataset_name
    for split in ("train", "val", "test"):
        src = dataset_dir / f"{dataset_name}_{split}.csv"
        dst = paths.data_splits_dir / f"{split}.csv"
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    hard_src = dataset_dir / "train_with_hard_tag.csv"
    if hard_src.exists():
        hard_dst = paths.data_splits_dir / "train_with_hard_tag.csv"
        hard_dst.write_text(hard_src.read_text(encoding="utf-8"), encoding="utf-8")


def summarize_split(df: pd.DataFrame) -> dict[str, Any]:
    label_counts = df["label"].value_counts().sort_index().to_dict()
    total = int(len(df))
    return {
        "count": total,
        "label_counts": {str(int(k)): int(v) for k, v in label_counts.items()},
        "label_distribution": {str(int(k)): float(v / total) for k, v in label_counts.items()},
    }


def build_split_summary(splits: dict[str, pd.DataFrame], dataset_root: Path, dataset_name: str, seed: int) -> dict[str, Any]:
    source_dir = dataset_root / dataset_name
    source_summary = {split: summarize_split(frame) for split, frame in splits.items()}
    return {
        "counts": {
            "train": int(len(splits["train"])),
            "val": int(len(splits["val"])),
            "test": int(len(splits["test"])),
            "total": int(sum(len(frame) for frame in splits.values())),
        },
        "subset_limits": {
            "max_train_examples": 0,
            "max_val_examples": 0,
            "max_test_examples": 0,
        },
        "source_split_summary": source_summary,
        "used_split_summary": source_summary,
        "seed": seed,
        "dataset_name": dataset_name,
        "source_dir": str(source_dir),
        "resolved_split_paths": {
            "train": str(source_dir / f"{dataset_name}_train.csv"),
            "val": str(source_dir / f"{dataset_name}_val.csv"),
            "test": str(source_dir / f"{dataset_name}_test.csv"),
        },
    }


def make_run_paths(run_name: str, runs_root: Path | None = None) -> RunPaths:
    root = DEFAULT_RUNS_ROOT if runs_root is None else runs_root
    run_dir = root / run_name
    return RunPaths(
        run_name=run_name,
        run_dir=run_dir,
        data_splits_dir=run_dir / "data_splits",
        teacher_dir=run_dir / "teacher_phase",
        student_dir=run_dir / "student_phase",
    )


def default_label_prompt(text: str) -> str:
    return f"Decide whether the following text is hate speech.\nText: {text}"


def default_label_completion(label: int) -> str:
    return "Yes" if int(label) == 1 else "No"


def truncate_label_text(text: str, tokenizer: Any, max_length: int) -> str:
    # 这里对齐 tinker 旧实现：先截断原始文本，确保最终答案 token 不会被挤掉。
    encoded_text = tokenizer.encode(str(text), add_special_tokens=False)
    encoded_text = encoded_text[: max(1, max_length - 10)]
    return tokenizer.decode(encoded_text)


def build_label_prompt(text: str, tokenizer: Any, max_length: int) -> str:
    return default_label_prompt(truncate_label_text(text, tokenizer, max_length))


def default_reasoning_prompt(text: str, *, rulebook: str | None = None) -> str:
    if rulebook:
        return (
            "Decide whether the following text is hate speech. Label 1 for hate speech, and 0 for non-hate speech.\n\n"
            "Use the rulebook below as guidance to reason in details before selecting the label. "
            "Use its ideas to help reasoning, but do not mention the rulebook explicitly.\n"
            "Do not output any explanation after the label. \n\n"
            f"Rulebook:\n{rulebook}\n\n"
            f"Text:\n{text}\n\n"
            "The <label>...</label> block must be the final block of your response. Return exactly in this format:\n"
            "<think>\n"
            "rule-grounded reasoning\n"
            "</think>\n"
            "then exactly one of:\n"
            "<label>\n"
            "0\n"
            "</label>\n"
            "or\n"
            "<label>\n"
            "1\n"
            "</label>"
        )
    return (
        "Decide whether the following text is hate speech. Label 1 for hate speech, and 0 for non-hate speech.\n\n"
        "Reasoning in details. Then select the label.\n"
        "Do not output any explanation after the label. \n\n"
        f"Text:\n{text}\n\n"
        "The <label>...</label> block must be the final block of your response. Return exactly in this format:\n"
        "<think>\n"
        "your reasoning\n"
        "</think>\n"
        "then exactly one of:\n"
        "<label>\n"
        "0\n"
        "</label>\n"
        "or\n"
        "<label>\n"
        "1\n"
        "</label>"
    )


def default_reasoning_completion(reasoning: str, label: int) -> str:
    reasoning = reasoning.strip()
    if not reasoning.startswith("<think>"):
        reasoning = "<think>\n" + reasoning + "\n</think>"
    return f"{reasoning}\n<label>\n{int(label)}\n</label>"


def strip_outer_think_tags(reasoning: str) -> str:
    raw = str(reasoning).strip()
    match = re.fullmatch(r"(?is)<think>\s*(.*?)\s*</think>", raw)
    if match is None:
        return raw
    return match.group(1).strip()


def ensure_think_wrapped_reasoning(reasoning: str) -> str:
    body = strip_outer_think_tags(reasoning)
    if not body:
        return ""
    return f"<think>\n{body}\n</think>"


def build_label_probe_assistant_prefix(reasoning_placeholder: str) -> str:
    clean_reasoning = strip_outer_think_tags(reasoning_placeholder)
    return f"<think>\n{clean_reasoning}\n</think>\n<label>\n"


def _coerce_input_ids(out: Any) -> list[int]:
    ids: Any = out
    if isinstance(out, dict) and "input_ids" in out:
        ids = out["input_ids"]
    elif hasattr(out, "input_ids"):
        ids = getattr(out, "input_ids")
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, tuple):
        ids = list(ids)
    if isinstance(ids, list) and ids and isinstance(ids[0], (list, tuple)):
        ids = list(ids[0])
    if not isinstance(ids, list):
        ids = list(ids)
    return [int(x) for x in ids]


def chat_input_ids(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
    enable_thinking: bool = False,
) -> list[int]:
    if enable_thinking:
        try:
            out = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=True,
            )
            return _coerce_input_ids(out)
        except TypeError as exc:
            if "enable_thinking" not in str(exc):
                raise
    out = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    return _coerce_input_ids(out)


def fit_user_prompt_and_answer_to_max_length(
    tokenizer: Any,
    *,
    user_builder: Callable[[str], str],
    text: str,
    assistant_content: str,
    max_length: int,
    enable_thinking: bool = False,
) -> tuple[list[dict[str, str]], list[int], list[int]] | None:
    text_tokens = tokenizer.encode(str(text), add_special_tokens=False)
    if not text_tokens:
        text_tokens = tokenizer.encode(" ", add_special_tokens=False)

    answer_tokens = tokenizer.encode(assistant_content, add_special_tokens=False)
    if not answer_tokens:
        return None

    best_messages: list[dict[str, str]] | None = None
    best_prompt_tokens: list[int] | None = None

    lo, hi = 1, len(text_tokens)
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate_text = tokenizer.decode(text_tokens[:mid])
        prompt_text = user_builder(candidate_text)
        messages = [
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": assistant_content},
        ]
        prompt_tokens = chat_input_ids(
            tokenizer,
            [{"role": "user", "content": prompt_text}],
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        if len(prompt_tokens) + len(answer_tokens) <= max_length:
            best_messages = messages
            best_prompt_tokens = prompt_tokens
            lo = mid + 1
        else:
            hi = mid - 1

    if best_messages is None or best_prompt_tokens is None:
        return None
    return best_messages, best_prompt_tokens, answer_tokens


def make_prompt_completion_example(prompt_text: str, completion_text: str, label: int) -> dict[str, Any]:
    prompt = [{"role": "user", "content": prompt_text}]
    completion = [{"role": "assistant", "content": completion_text}]
    messages = prompt + completion
    return {
        "prompt": prompt,
        "completion": completion,
        "messages": messages,
        "label": int(label),
    }


def tokenize_prompt_completion_example(
    tokenizer: Any,
    *,
    prompt_text: str,
    completion_text: str,
    label: int,
    max_length: int,
) -> dict[str, Any]:
    prompt_messages = [{"role": "user", "content": prompt_text}]
    completion_messages = [{"role": "assistant", "content": completion_text}]

    # 这里直接复刻 TRL 对 prompt-completion 数据的处理方式，
    # 但我们在进入 Trainer 之前就先做好，避开多进程 dataset.map/caching。
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
    )
    if isinstance(prompt_ids[0], list):
        prompt_ids = prompt_ids[0]

    full_processed = tokenizer.apply_chat_template(
        prompt_messages + completion_messages,
        tokenize=True,
        return_dict=True,
    )
    input_ids = full_processed["input_ids"]
    if isinstance(input_ids[0], list):
        input_ids = input_ids[0]

    attention_mask = [1] * len(input_ids)
    labels = list(input_ids)
    for i in range(min(len(prompt_ids), len(labels))):
        labels[i] = -100

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def tokenize_prompt_completion_dataset(
    dataset: Dataset,
    *,
    tokenizer: Any,
    max_length: int,
) -> Dataset:
    # 这里尽量走 datasets.map 的官方范式，只保留“completion 部分参与 loss”这个任务必需逻辑。
    def _map_fn(example: dict[str, Any]) -> dict[str, Any]:
        return tokenize_prompt_completion_example(
            tokenizer,
            prompt_text=str(example["prompt_text"]),
            completion_text=str(example["completion_text"]),
            label=int(example["label"]),
            max_length=max_length,
        )

    return dataset.map(
        _map_fn,
        remove_columns=dataset.column_names,
        desc="Tokenizing prompt-completion dataset",
    )


def build_label_dataset(frame: pd.DataFrame) -> Dataset:
    rows = []
    for record in frame.to_dict("records"):
        rows.append(
            make_prompt_completion_example(
                prompt_text=default_label_prompt(record["text"]),
                completion_text=default_label_completion(int(record["label"])),
                label=int(record["label"]),
            )
        )
    return Dataset.from_list(rows)


def build_label_tokenized_dataset(frame: pd.DataFrame, tokenizer: Any, max_length: int) -> Dataset:
    raw_rows = []
    for record in frame.to_dict("records"):
        raw_rows.append(
            {
                "prompt_text": build_label_prompt(record["text"], tokenizer, max_length),
                "completion_text": default_label_completion(int(record["label"])),
                "label": int(record["label"]),
            }
        )
    return tokenize_prompt_completion_dataset(
        Dataset.from_list(raw_rows),
        tokenizer=tokenizer,
        max_length=max_length,
    )


def build_reasoning_dataset(
    accepted_rows: list[dict[str, Any]],
    *,
    rulebook: str | None,
    use_rulebook: bool,
) -> Dataset:
    rows = []
    for record in accepted_rows:
        prompt_text = default_reasoning_prompt(
            record["text"],
            rulebook=rulebook if use_rulebook else None,
        )
        completion_text = default_reasoning_completion(record["reasoning"], int(record["label"]))
        item = make_prompt_completion_example(prompt_text, completion_text, int(record["label"]))
        item["sample_id"] = str(record["sample_id"])
        item["example_id"] = str(record["example_id"])
        item["text"] = str(record["text"])
        rows.append(item)
    return Dataset.from_list(rows)


def build_reasoning_tokenized_dataset(
    accepted_rows: list[dict[str, Any]],
    *,
    tokenizer: Any,
    rulebook: str | None,
    use_rulebook: bool,
    max_length: int,
    reasoning_token_weight: float = 1.0,
    label_token_weight: float = 1.0,
    enable_thinking: bool = False,
) -> Dataset:
    prompt_builder = lambda text: default_reasoning_prompt(
        text,
        rulebook=rulebook if use_rulebook else None,
    )
    rows: list[dict[str, Any]] = []
    dropped = 0
    for record in accepted_rows:
        text = str(record["text"])
        label = int(record["label"])
        completion_text = default_reasoning_completion(record["reasoning"], label)
        fit = fit_user_prompt_and_answer_to_max_length(
            tokenizer,
            user_builder=prompt_builder,
            text=text,
            assistant_content=completion_text,
            max_length=max_length,
            enable_thinking=enable_thinking,
        )
        if fit is None:
            dropped += 1
            continue
        _, prompt_tokens, answer_tokens = fit
        full_tokens = prompt_tokens + answer_tokens
        if len(full_tokens) < 2:
            dropped += 1
            continue

        label_suffix = f"<label>\n{label}\n</label>"
        label_token_count = len(tokenizer.encode(label_suffix, add_special_tokens=False))
        if label_token_count <= 0:
            label_token_count = 1
        if label_token_count > len(answer_tokens):
            label_token_count = len(answer_tokens)
        reasoning_token_count = len(answer_tokens) - label_token_count

        labels = list(full_tokens)
        for i in range(len(prompt_tokens)):
            labels[i] = -100

        token_weights = (
            [0.0] * len(prompt_tokens)
            + [float(reasoning_token_weight)] * reasoning_token_count
            + [float(label_token_weight)] * label_token_count
        )
        rows.append(
            {
                "input_ids": full_tokens,
                "attention_mask": [1] * len(full_tokens),
                "labels": labels,
                "token_weights": token_weights,
            }
        )
    if not rows:
        raise RuntimeError("No usable reasoning training rows after tokenization")
    if dropped > 0:
        print(f"[reasoning_dataset] dropped_rows={dropped}", flush=True)
    return Dataset.from_list(rows)


def make_data_collator(tokenizer: Any, model: Any | None = None) -> DataCollatorForSeq2Seq:
    return DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )


def make_reasoning_data_collator(tokenizer: Any, model: Any | None = None) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    base_collator = make_data_collator(tokenizer, model)

    def _collate(features: list[dict[str, Any]]) -> dict[str, Any]:
        token_weights = [list(feature.pop("token_weights")) for feature in features]
        batch = base_collator(features)
        max_len = int(batch["labels"].shape[1])
        padded_weights = []
        for weights in token_weights:
            padded_weights.append(weights + [0.0] * (max_len - len(weights)))
        batch["token_weights"] = torch.tensor(padded_weights, dtype=torch.float32)
        return batch

    return _collate


class ReasoningTrainer(Trainer):
    def __init__(self, *args, min_lr_ratio: float = 0.1, **kwargs) -> None:
        self.min_lr_ratio = float(min_lr_ratio)
        super().__init__(*args, **kwargs)

    def create_scheduler(self, num_training_steps: int, optimizer: Any = None):
        if self.lr_scheduler is None:
            opt = optimizer if optimizer is not None else self.optimizer
            if opt is None:
                raise RuntimeError("Optimizer must exist before creating scheduler.")

            def lr_lambda(step: int) -> float:
                progress = min(max(float(step) / max(1, num_training_steps - 1), 0.0), 1.0)
                return 1.0 - (1.0 - self.min_lr_ratio) * progress

            self.lr_scheduler = LambdaLR(opt, lr_lambda)
        return self.lr_scheduler

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        token_weights = inputs.pop("token_weights", None)
        outputs = model(**inputs)

        if token_weights is None:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
            return (loss, outputs) if return_outputs else loss

        labels = inputs["labels"]
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_weights = token_weights[..., 1:].to(shift_logits.device).contiguous()

        token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view_as(shift_labels)
        valid_mask = shift_labels.ne(-100).to(token_loss.dtype)
        weighted_loss = token_loss * shift_weights * valid_mask
        denom = (shift_weights * valid_mask).sum().clamp_min(1e-12)
        loss = weighted_loss.sum() / denom
        return (loss, outputs) if return_outputs else loss


def export_train_sft_jsonl(dataset: Dataset, output_path: Path) -> None:
    rows = []
    for item in dataset:
        rows.append(
            {
                "messages": item["messages"],
                "label": int(item["label"]),
            }
        )
    write_jsonl(output_path, rows)


def binary_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    for gold, pred in zip(y_true, y_pred, strict=True):
        if gold == 1 and pred == 1:
            tp += 1
        elif gold == 0 and pred == 1:
            fp += 1
        elif gold == 1 and pred == 0:
            fn += 1
        else:
            tn += 1

    total = max(1, tp + fp + fn + tn)
    accuracy = (tp + tn) / total
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)

    # 宏平均 F1 分别对正类和负类算，再取平均。
    neg_precision = tn / max(1, tn + fn)
    neg_recall = tn / max(1, tn + fp)
    neg_f1 = 2 * neg_precision * neg_recall / max(1e-12, neg_precision + neg_recall)

    denom = math.sqrt(max(1e-12, (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = ((tp * tn) - (fp * fn)) / denom
    return {
        "accuracy": float(accuracy),
        "macro_f1": float((f1 + neg_f1) / 2.0),
        "balanced_accuracy": float((recall + specificity) / 2.0),
        "F1": float(f1),
        "mcc": float(mcc),
        "precision": float(precision),
        "recall": float(recall),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def normalize_label_value(value: Any) -> int | None:
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)):
        v = int(value)
        return v if v in {0, 1} else None
    if isinstance(value, (float, np.floating)):
        v = float(value)
        if v in {0.0, 1.0}:
            return int(v)
        return None

    s = str(value).strip().lower()
    if s in {"0", "no", "false", "negative"}:
        return 0
    if s in {"1", "yes", "true", "positive"}:
        return 1
    return None


def iter_json_object_candidates(text: str) -> Iterator[str]:
    depth = 0
    start = -1
    in_string = False
    escape = False
    for idx, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
            continue
        if ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    yield text[start : idx + 1]


def parse_reasoning_label_from_obj(obj: Any) -> ParsedReasoningOutput | None:
    if isinstance(obj, list) and obj:
        obj = obj[0]
    if not isinstance(obj, dict):
        return None

    norm = {str(k).strip().lower(): v for k, v in obj.items()}
    reasoning_keys = ["reasoning", "rationale", "analysis", "explanation", "thought"]
    label_keys = ["label", "pred_label", "prediction", "predicted_label", "class"]

    reasoning_val = None
    for key in reasoning_keys:
        if key in norm:
            reasoning_val = norm[key]
            break
    label_val = None
    for key in label_keys:
        if key in norm:
            label_val = norm[key]
            break

    if label_val is None and "output" in norm and isinstance(norm["output"], dict):
        return parse_reasoning_label_from_obj(norm["output"])

    label = normalize_label_value(label_val)
    if label is None:
        return None
    reasoning = ensure_think_wrapped_reasoning("" if reasoning_val is None else str(reasoning_val).strip())
    if not reasoning:
        return None
    return ParsedReasoningOutput(reasoning=reasoning, label=label, source="json")


def parse_reasoning_output(raw: str) -> tuple[ParsedReasoningOutput | None, str | None]:
    raw = (raw or "").strip()
    if not raw:
        return None, "empty_response"

    think_match = re.search(r"(?is)<think>\s*(.*?)\s*</think>", raw)
    if think_match is not None:
        reasoning = ensure_think_wrapped_reasoning(think_match.group(1))
        label_match = re.search(r"(?is)<label>\s*(0|1|yes|no|true|false)\s*</label>", raw)
        if label_match is not None:
            label = normalize_label_value(label_match.group(1))
            if label is not None and reasoning:
                return ParsedReasoningOutput(reasoning=reasoning, label=label, source="think_xml_label"), None
        label_match = re.search(
            r'(?is)"?\b(?:label|prediction|predicted[_\s]?label)\b"?\s*[:=]\s*(0|1|yes|no|true|false)\b',
            raw,
        )
        if label_match is not None:
            label = normalize_label_value(label_match.group(1))
            if label is not None and reasoning:
                return ParsedReasoningOutput(reasoning=reasoning, label=label, source="think_label"), None

    candidates: list[str] = []
    seen: set[str] = set()

    def push(candidate: str) -> None:
        c = candidate.strip()
        if c and c not in seen:
            seen.add(c)
            candidates.append(c)

    push(raw)
    for candidate in iter_json_object_candidates(raw):
        push(candidate)

    for candidate in candidates:
        parsed_obj = None
        try:
            parsed_obj = json.loads(candidate)
        except Exception:
            parsed_obj = None
        if parsed_obj is None:
            continue
        parsed = parse_reasoning_label_from_obj(parsed_obj)
        if parsed is not None:
            return parsed, None

    return None, "parse_failed"


def parse_label_from_text(text: str) -> tuple[int | None, str]:
    raw = (text or "").strip()
    match = re.match(r"^\s*(yes)\b", raw, flags=re.IGNORECASE)
    if match:
        return 1, "leading_yes"
    match = re.match(r"^\s*(no)\b", raw, flags=re.IGNORECASE)
    if match:
        return 0, "leading_no"
    match = re.search(r'"label"\s*:\s*([01])', raw)
    if match:
        return int(match.group(1)), "json_label"
    match = re.search(r"<label>\s*([01])\s*</label>", raw, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return int(match.group(1)), "xml_label"
    match = re.match(r"^\s*([01])(?:\D|$)", raw)
    if match:
        return int(match.group(1)), "leading_digit"
    return None, "unparsed"


def parse_reasoning_from_text(text: str) -> str:
    raw = (text or "").strip()
    match = re.search(r"<think>\s*(.*?)\s*</think>", raw, flags=re.IGNORECASE | re.DOTALL)
    if match:
        content = match.group(1).strip()
        return "<think>\n" + content + "\n</think>"
    return ""


def _extract_input_ids(maybe_encoded: Any) -> list[int]:
    if isinstance(maybe_encoded, dict):
        value = maybe_encoded["input_ids"]
    else:
        value = maybe_encoded
    if isinstance(value, torch.Tensor):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(x) for x in value]


def _find_last_sublist(haystack: list[int], needle: list[int]) -> int:
    answer_idx = -1
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i:i + len(needle)] == needle:
            answer_idx = i
    if answer_idx < 0:
        raise ValueError("找不到 answer token 在完整消息中的位置。")
    return answer_idx


def score_completion_from_prompt_tokens(model: Any, prompt_tokens: list[int], answer_tokens: list[int]) -> float:
    seq = prompt_tokens + answer_tokens
    input_ids = torch.tensor([seq], device=model.device, dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    logprobs = torch.log_softmax(shift_logits, dim=-1)
    token_logprobs = logprobs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)[0]
    start = len(prompt_tokens) - 1
    end = start + len(answer_tokens)
    return float(token_logprobs[start:end].sum().item())


def score_prompt_completion(model: Any, tokenizer: Any, *, prompt_text: str, completion_text: str) -> float:
    messages = [
        {"role": "user", "content": prompt_text},
        {"role": "assistant", "content": completion_text},
    ]
    encoded_messages = _extract_input_ids(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=False,
        )
    )
    answer_tokens = tokenizer.encode(completion_text, add_special_tokens=False)
    answer_idx = _find_last_sublist(encoded_messages, answer_tokens)
    prompt_tokens = encoded_messages[:answer_idx]
    seq = prompt_tokens + answer_tokens
    input_ids = torch.tensor([seq], device=model.device, dtype=torch.long)

    with torch.no_grad():
        logits = model(input_ids=input_ids).logits
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    logprobs = torch.log_softmax(shift_logits, dim=-1)
    token_logprobs = logprobs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)[0]

    start = len(prompt_tokens) - 1
    end = start + len(answer_tokens)
    return float(token_logprobs[start:end].sum().item())


def normalize_binary_logprobs(lp_one: float, lp_zero: float) -> tuple[float, float]:
    denom = float(np.logaddexp(lp_one, lp_zero))
    p_one = float(np.exp(lp_one - denom))
    p_zero = float(np.exp(lp_zero - denom))
    return p_one, p_zero


def build_reasoning_eval_rows(
    *,
    frame: pd.DataFrame,
    tokenizer: Any,
    max_length: int,
    reasoning_prompt_builder: Callable[[str], str],
    reasoning_placeholder: str,
    enable_thinking: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    probe_prefix = build_label_probe_assistant_prefix(reasoning_placeholder)
    for row in frame.to_dict("records"):
        fit = fit_user_prompt_and_answer_to_max_length(
            tokenizer,
            user_builder=reasoning_prompt_builder,
            text=str(row["text"]),
            assistant_content=probe_prefix,
            max_length=max_length,
            enable_thinking=enable_thinking,
        )
        if fit is None:
            rows.append(
                {
                    "text": str(row["text"]),
                    "label": int(row["label"]),
                    "prompt_tokens": None,
                    "fit_error": "prompt_too_long_after_fit",
                }
            )
            continue
        _, prompt_tokens, probe_tokens = fit
        rows.append(
            {
                "text": str(row["text"]),
                "label": int(row["label"]),
                "prompt_tokens": prompt_tokens + probe_tokens,
                "fit_error": None,
            }
        )
    return rows


def _generate_from_prompt_tokens(
    *,
    model: Any,
    tokenizer: Any,
    prompt_tokens: list[int],
    max_new_tokens: int,
    temperature: float,
) -> tuple[str, list[int]]:
    input_ids = torch.tensor([prompt_tokens], device=model.device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    generate_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        generate_kwargs["temperature"] = temperature
    with torch.no_grad():
        generated = model.generate(**generate_kwargs)
    new_tokens = generated[0][input_ids.shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True), [int(x) for x in new_tokens.tolist()]


def _parse_emitted_binary_label(generated_text: str, generated_ids: list[int], one_tokens: list[int], zero_tokens: list[int]) -> int | None:
    if generated_ids:
        if len(generated_ids) >= len(one_tokens) and generated_ids[: len(one_tokens)] == one_tokens:
            return 1
        if len(generated_ids) >= len(zero_tokens) and generated_ids[: len(zero_tokens)] == zero_tokens:
            return 0
    parsed, _ = parse_reasoning_output(generated_text)
    if parsed is not None:
        return int(parsed.label)
    label, _ = parse_label_from_text(generated_text)
    return label


def run_label_candidate_eval(
    *,
    model: Any,
    tokenizer: Any,
    frame: pd.DataFrame,
    split_name: str,
    build_prompt: Any,
) -> list[dict[str, Any]]:
    model.eval()
    yes_tokens = tokenizer.encode("Yes", add_special_tokens=False)
    no_tokens = tokenizer.encode("No", add_special_tokens=False)
    records: list[dict[str, Any]] = []
    for idx, row in enumerate(frame.to_dict("records")):
        prompt_text = build_prompt(row["text"])
        lp_yes = score_prompt_completion(model, tokenizer, prompt_text=prompt_text, completion_text="Yes")
        lp_no = score_prompt_completion(model, tokenizer, prompt_text=prompt_text, completion_text="No")
        pred_text = "Yes" if lp_yes >= lp_no else "No"
        pred_label = 1 if lp_yes >= lp_no else 0
        records.append(
            {
                "sample_id": f"{split_name}_{idx}_k0",
                "example_id": f"{split_name}_{idx}",
                "split": split_name,
                "k_index": 0,
                "text": row["text"],
                "gold_label": int(row["label"]),
                "raw_output": pred_text,
                "parse_ok": True,
                "parse_source": "yesno_logprob",
                "pred_label": pred_label,
                "reasoning": "",
                "score_yes": float(lp_yes),
                "score_no": float(lp_no),
                "true_answer_len": int(len(yes_tokens) if int(row["label"]) == 1 else len(no_tokens)),
            }
        )
    return records


def format_generation_record(
    *,
    split_name: str,
    idx: int,
    text: str,
    gold_label: int,
    raw_output: str,
    pred_label: int | None,
    parse_source: str,
) -> dict[str, Any]:
    return {
        "sample_id": f"{split_name}_{idx}_k0",
        "example_id": f"{split_name}_{idx}",
        "split": split_name,
        "k_index": 0,
        "text": text,
        "gold_label": int(gold_label),
        "raw_output": raw_output,
        "parse_ok": pred_label is not None,
        "parse_source": parse_source,
        "pred_label": None if pred_label is None else int(pred_label),
        "reasoning": parse_reasoning_from_text(raw_output),
    }


def compute_generation_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    y_true = [int(row["gold_label"]) for row in records]
    invalid = sum(1 for row in records if row["pred_label"] is None)
    y_pred = [0 if row["pred_label"] is None else int(row["pred_label"]) for row in records]
    metrics = binary_metrics(y_true, y_pred)
    metrics["invalid_label_rate"] = float(invalid / max(1, len(records)))
    metrics["decision_threshold"] = 0.5
    metrics["loss"] = math.nan
    metrics["cls_loss"] = math.nan
    metrics["auroc"] = math.nan
    metrics["auprc"] = math.nan
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "invalid_label_rate": metrics["invalid_label_rate"],
        "metrics": metrics,
    }


def compute_label_candidate_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    y_true = [int(row["gold_label"]) for row in records]
    y_pred = [int(row["pred_label"]) for row in records]
    metrics = binary_metrics(y_true, y_pred)

    cls_losses: list[float] = []
    total_nll = 0.0
    total_tokens = 0.0
    for row in records:
        gold = int(row["gold_label"])
        lp_yes = float(row["score_yes"])
        lp_no = float(row["score_no"])
        true_lp = lp_yes if gold == 1 else lp_no
        cls_losses.append(float(np.logaddexp(lp_yes, lp_no) - true_lp))
        total_nll += float(-true_lp)
        total_tokens += float(row["true_answer_len"])

    metrics["invalid_label_rate"] = 0.0
    metrics["decision_threshold"] = 0.5
    metrics["loss"] = float(total_nll / max(1.0, total_tokens))
    metrics["cls_loss"] = float(np.mean(cls_losses)) if cls_losses else math.nan
    metrics["auroc"] = math.nan
    metrics["auprc"] = math.nan
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "invalid_label_rate": metrics["invalid_label_rate"],
        "metrics": metrics,
    }


def compute_reasoning_generation_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    y_true = [int(row["gold_label"]) for row in records]
    y_pred = [int(row["pred_label"]) for row in records]
    metrics = binary_metrics(y_true, y_pred)

    invalid_flags = [bool(row["invalid_flag"]) for row in records]
    invalid_count = sum(1 for flag in invalid_flags if flag)
    p_one_all = [float(row["p_one"]) for row in records]
    p_zero_all = [float(row["p_zero"]) for row in records]
    cls_losses = [float(row["nll"]) for row in records]

    metrics["invalid_label_rate"] = float(invalid_count / max(1, len(records)))
    metrics["decision_threshold"] = 0.5
    metrics["loss"] = float(sum(cls_losses) / max(1, len(cls_losses))) if cls_losses else math.nan
    metrics["cls_loss"] = float(sum(cls_losses) / max(1, len(cls_losses))) if cls_losses else math.nan
    metrics["auroc"] = math.nan
    metrics["auprc"] = math.nan
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "p_one": p_one_all,
        "p_zero": p_zero_all,
        "invalid_label_rate": metrics["invalid_label_rate"],
        "metrics": metrics,
    }


def compute_eval_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    if records and records[0].get("parse_source") == "yesno_logprob":
        return compute_label_candidate_metrics(records)
    if records and "p_one" in records[0] and "invalid_flag" in records[0] and "nll" in records[0]:
        return compute_reasoning_generation_metrics(records)
    return compute_generation_metrics(records)


def run_reasoning_generation_eval(
    *,
    model: Any,
    tokenizer: Any,
    frame: pd.DataFrame,
    split_name: str,
    build_prompt: Callable[[str], str],
    max_length: int,
    max_new_tokens: int,
    temperature: float,
    reasoning_placeholder: str,
    invalid_label_warn_rate: float = 0.1,
    enable_thinking: bool = False,
    eval_batch_size: int = 1,
) -> list[dict[str, Any]]:
    model.eval()
    dummy_answer = "<think>\nx\n</think>\n<label>\n0\n</label>"
    records: list[dict[str, Any] | None] = [None] * len(frame)
    invalid_examples: list[str] = []
    ready_items: list[dict[str, Any]] = []
    for idx, row in enumerate(frame.to_dict("records")):
        text = str(row["text"])
        gold_label = int(row["label"])
        fit = fit_user_prompt_and_answer_to_max_length(
            tokenizer,
            user_builder=build_prompt,
            text=text,
            assistant_content=dummy_answer,
            max_length=max_length,
            enable_thinking=enable_thinking,
        )
        if fit is None:
            records[idx] = (
                {
                    "sample_id": f"{split_name}_{idx}_k0",
                    "example_id": f"{split_name}_{idx}",
                    "split": split_name,
                    "k_index": 0,
                    "text": text,
                    "gold_label": gold_label,
                    "raw_output": "",
                    "parse_ok": False,
                    "parse_source": None,
                    "pred_label": None,
                    "reasoning": None,
                    "error": "prompt_too_long_after_fit",
                }
            )
            continue
        _, prompt_tokens, _ = fit
        ready_items.append(
            {
                "idx": idx,
                "text": text,
                "gold_label": gold_label,
                "prompt_tokens": prompt_tokens,
                "available_new_tokens": max(1, max_length - len(prompt_tokens)),
            }
        )

    if ready_items:
        ready_items.sort(key=lambda item: len(item["prompt_tokens"]), reverse=True)
        batch_size = max(1, int(eval_batch_size))
        total_batches = math.ceil(len(ready_items) / batch_size)
        report_every = max(1, total_batches // 10)
        print(
            f"[reasoning_eval] split={split_name} samples={len(ready_items)} batch_size={batch_size} total_batches={total_batches}",
            flush=True,
        )
        for batch_idx, start in enumerate(range(0, len(ready_items), batch_size), start=1):
            batch_items = ready_items[start : start + batch_size]
            prompt_lists = [list(item["prompt_tokens"]) for item in batch_items]
            padded = tokenizer.pad(
                {"input_ids": prompt_lists},
                padding=True,
                return_tensors="pt",
            )
            input_ids = padded["input_ids"].to(model.device)
            attention_mask = padded["attention_mask"].to(model.device)
            batch_max_new_tokens = max(
                1,
                min(
                    int(max_new_tokens),
                    min(int(item["available_new_tokens"]) for item in batch_items),
                ),
            )
            generate_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": batch_max_new_tokens,
                "do_sample": temperature > 0,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if temperature > 0:
                generate_kwargs["temperature"] = temperature
            with torch.no_grad():
                generated = model.generate(**generate_kwargs)

            for row_i, item in enumerate(batch_items):
                prompt_len = int(attention_mask[row_i].sum().item())
                new_tokens = generated[row_i][prompt_len:]
                raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True)
                parsed_output, parse_error = parse_reasoning_output(raw_output)
                pred = None if parsed_output is None else int(parsed_output.label)
                if pred is None and len(invalid_examples) < 5:
                    invalid_examples.append(
                        f"sample_text={str(item['text'])[:80]!r} emitted={raw_output[:120]!r}"
                    )
                records[int(item["idx"])] = {
                    "sample_id": f"{split_name}_{int(item['idx'])}_k0",
                    "example_id": f"{split_name}_{int(item['idx'])}",
                    "split": split_name,
                    "k_index": 0,
                    "text": str(item["text"]),
                    "gold_label": int(item["gold_label"]),
                    "raw_output": raw_output,
                    "parse_ok": parsed_output is not None,
                    "parse_source": None if parsed_output is None else parsed_output.source,
                    "pred_label": pred,
                    "reasoning": None if parsed_output is None else parsed_output.reasoning,
                    "error": parse_error,
                }

            if batch_idx == 1 or batch_idx == total_batches or batch_idx % report_every == 0:
                print(
                    f"[reasoning_eval] split={split_name} batch={batch_idx}/{total_batches}",
                    flush=True,
                )

    finalized_records = [row for row in records if row is not None]
    invalid_rate = sum(1 for row in finalized_records if row["pred_label"] is None) / max(1, len(finalized_records))
    if invalid_rate >= float(invalid_label_warn_rate):
        print(
            "[reasoning_eval_warn] "
            f"invalid_label_rate={float(invalid_rate):.4f} "
            f"examples={' | '.join(invalid_examples)}",
            flush=True,
        )
    return finalized_records


def run_generation_eval(
    *,
    model: Any,
    tokenizer: Any,
    frame: pd.DataFrame,
    split_name: str,
    build_prompt: Any,
    max_new_tokens: int,
    temperature: float,
) -> list[dict[str, Any]]:
    model.eval()
    records: list[dict[str, Any]] = []
    for idx, row in enumerate(frame.to_dict("records")):
        prompt_messages = [{"role": "user", "content": build_prompt(row["text"])}]
        prompt_inputs = tokenizer.apply_chat_template(
            prompt_messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if hasattr(prompt_inputs, "to"):
            prompt_inputs = prompt_inputs.to(model.device)
        if isinstance(prompt_inputs, torch.Tensor):
            input_ids = prompt_inputs
        else:
            input_ids = prompt_inputs["input_ids"]
        attention_mask = torch.ones_like(input_ids)
        generate_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if temperature > 0:
            generate_kwargs["temperature"] = temperature
        with torch.no_grad():
            generated = model.generate(**generate_kwargs)
        new_tokens = generated[0][input_ids.shape[1] :]
        raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True)
        pred_label, parse_source = parse_label_from_text(raw_output)
        records.append(
            format_generation_record(
                split_name=split_name,
                idx=idx,
                text=row["text"],
                gold_label=int(row["label"]),
                raw_output=raw_output,
                pred_label=pred_label,
                parse_source=parse_source,
            )
        )
    return records


def save_eval_bundle(
    *,
    output_dir: Path,
    checkpoint_path: str,
    checkpoint_name: str,
    split_name: str,
    tag: str,
    step: int,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_out = compute_eval_metrics(records)
    export_name = f"{split_name}_{tag}.json"
    export_path = output_dir / export_name
    write_json(export_path, records)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint_path": checkpoint_path,
        "checkpoint_name": checkpoint_name,
        "split": split_name,
        "tag": tag,
        "step": int(step),
        "dataset_fingerprint": None,
        "config_fingerprint": None,
        "decision_threshold": 0.5,
        "n": int(len(records)),
        "predictions_path": str(export_path),
        "metrics": eval_out["metrics"],
    }
    write_json(output_dir / f"metrics_{split_name}_{tag}.json", payload)
    return payload


def save_model_pointer(
    *,
    model_root: Path,
    run_name: str,
    final_model_path: str,
    best_model_path: str,
    resume_state_path: str | None,
) -> dict[str, Any]:
    model_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": run_name,
        "final_model_path": final_model_path,
        "best_model_path": best_model_path,
        "resume_state_path": resume_state_path,
    }
    write_json(model_root / f"{run_name}.json", payload)
    write_json(model_root / "latest.json", payload)
    (model_root / "latest.txt").write_text(
        "\n".join(
            [
                f"run_name={run_name}",
                f"final_model_path={final_model_path}",
                f"best_model_path={best_model_path}",
                f"resume_state_path={resume_state_path}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return payload


class GenerationEvalCallback(TrainerCallback):
    def __init__(
        self,
        *,
        run_paths: RunPaths,
        tokenizer: Any,
        val_frame: pd.DataFrame,
        test_frame: pd.DataFrame,
        build_prompt: Any,
        max_new_tokens: int,
        temperature: float,
        selection_metric: str = "macro_f1",
        eval_mode: str = "generation",
        max_length: int = 2048,
        reasoning_placeholder: str = "Reasoning intentionally omitted for label scoring.",
        invalid_label_warn_rate: float = 0.1,
        enable_thinking: bool = False,
        eval_batch_size: int = 1,
    ) -> None:
        self.run_paths = run_paths
        self.tokenizer = tokenizer
        self.val_frame = val_frame
        self.test_frame = test_frame
        self.build_prompt = build_prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.selection_metric = selection_metric
        self.eval_mode = eval_mode
        self.max_length = int(max_length)
        self.reasoning_placeholder = str(reasoning_placeholder)
        self.invalid_label_warn_rate = float(invalid_label_warn_rate)
        self.enable_thinking = bool(enable_thinking)
        self.eval_batch_size = int(eval_batch_size)
        self.best_step = 0
        self.best_checkpoint_path: str | None = None
        self.best_val_metrics: dict[str, Any] | None = None
        self.best_test_metrics: dict[str, Any] | None = None

    def _run_eval(self, *, model: Any, frame: pd.DataFrame, split_name: str) -> list[dict[str, Any]]:
        if self.eval_mode == "label_candidate":
            return run_label_candidate_eval(
                model=model,
                tokenizer=self.tokenizer,
                frame=frame,
                split_name=split_name,
                build_prompt=self.build_prompt,
            )
        if self.eval_mode == "reasoning_generation":
            return run_reasoning_generation_eval(
                model=model,
                tokenizer=self.tokenizer,
                frame=frame,
                split_name=split_name,
                build_prompt=self.build_prompt,
                max_length=self.max_length,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                reasoning_placeholder=self.reasoning_placeholder,
                invalid_label_warn_rate=self.invalid_label_warn_rate,
                enable_thinking=self.enable_thinking,
                eval_batch_size=self.eval_batch_size,
            )
        return run_generation_eval(
            model=model,
            tokenizer=self.tokenizer,
            frame=frame,
            split_name=split_name,
            build_prompt=self.build_prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        if not state.is_world_process_zero or model is None:
            return control

        was_training = model.training
        model.eval()
        val_records = self._run_eval(model=model, frame=self.val_frame, split_name="val")
        test_records = self._run_eval(model=model, frame=self.test_frame, split_name="test")
        if was_training:
            model.train()

        step = int(state.global_step)
        val_bundle = save_eval_bundle(
            output_dir=self.run_paths.student_dir,
            checkpoint_path=str(self.run_paths.student_dir / f"checkpoint-{step}"),
            checkpoint_name=f"checkpoint-{step}",
            split_name="val",
            tag=f"step_{step}",
            step=step,
            records=val_records,
        )
        test_bundle = save_eval_bundle(
            output_dir=self.run_paths.student_dir,
            checkpoint_path=str(self.run_paths.student_dir / f"checkpoint-{step}"),
            checkpoint_name=f"checkpoint-{step}",
            split_name="test",
            tag=f"step_{step}",
            step=step,
            records=test_records,
        )

        if wandb.run is not None:
            wandb_payload = {
                "val/loss": val_bundle["metrics"]["loss"],
                "val/cls_loss": val_bundle["metrics"]["cls_loss"],
                "val/accuracy": val_bundle["metrics"]["accuracy"],
                "val/macro_f1": val_bundle["metrics"]["macro_f1"],
                "val/invalid_label_rate": val_bundle["metrics"]["invalid_label_rate"],
                "test/loss": test_bundle["metrics"]["loss"],
                "test/cls_loss": test_bundle["metrics"]["cls_loss"],
                "test/accuracy": test_bundle["metrics"]["accuracy"],
                "test/macro_f1": test_bundle["metrics"]["macro_f1"],
                "test/invalid_label_rate": test_bundle["metrics"]["invalid_label_rate"],
            }
            if metrics is not None and "eval_loss" in metrics:
                wandb_payload["val/trainer_eval_loss"] = float(metrics["eval_loss"])
            wandb.log(wandb_payload, step=step)

        if metrics is not None and "eval_loss" in metrics:
            print(
                "[eval] "
                f"step={step} "
                f"val_loss={float(val_bundle['metrics']['loss']):.6f} "
                f"val_cls_loss={float(val_bundle['metrics']['cls_loss']):.6f} "
                f"trainer_eval_loss={float(metrics['eval_loss']):.6f} "
                f"val_acc={float(val_bundle['metrics']['accuracy']):.6f} "
                f"val_macro_f1={float(val_bundle['metrics']['macro_f1']):.6f} "
                f"val_invalid={float(val_bundle['metrics']['invalid_label_rate']):.4f} "
                f"test_loss={float(test_bundle['metrics']['loss']):.6f} "
                f"test_cls_loss={float(test_bundle['metrics']['cls_loss']):.6f} "
                f"test_acc={float(test_bundle['metrics']['accuracy']):.6f} "
                f"test_macro_f1={float(test_bundle['metrics']['macro_f1']):.6f} "
                f"test_invalid={float(test_bundle['metrics']['invalid_label_rate']):.4f}",
                flush=True,
            )
        else:
            print(
                "[eval] "
                f"step={step} "
                f"val_loss={float(val_bundle['metrics']['loss']):.6f} "
                f"val_cls_loss={float(val_bundle['metrics']['cls_loss']):.6f} "
                f"val_acc={float(val_bundle['metrics']['accuracy']):.6f} "
                f"val_macro_f1={float(val_bundle['metrics']['macro_f1']):.6f} "
                f"val_invalid={float(val_bundle['metrics']['invalid_label_rate']):.4f} "
                f"test_loss={float(test_bundle['metrics']['loss']):.6f} "
                f"test_cls_loss={float(test_bundle['metrics']['cls_loss']):.6f} "
                f"test_acc={float(test_bundle['metrics']['accuracy']):.6f} "
                f"test_macro_f1={float(test_bundle['metrics']['macro_f1']):.6f} "
                f"test_invalid={float(test_bundle['metrics']['invalid_label_rate']):.4f}",
                flush=True,
            )

        metric_value = float(val_bundle["metrics"][self.selection_metric])
        if self.best_val_metrics is None or metric_value > float(self.best_val_metrics[self.selection_metric]):
            self.best_step = step
            self.best_checkpoint_path = str(self.run_paths.student_dir / f"checkpoint-{step}")
            self.best_val_metrics = val_bundle["metrics"]
            self.best_test_metrics = test_bundle["metrics"]
        return control


def load_teacher_source_run(source_run_name: str) -> Path:
    source_run = PROJECT_ROOT / "tinker" / "runs" / source_run_name
    if not source_run.exists():
        raise FileNotFoundError(source_run)
    return source_run


def build_rulebook_from_source(source_run_name: str, dataset_name: str) -> tuple[str, list[str]]:
    source_teacher_dir = load_teacher_source_run(source_run_name) / "teacher_phase"
    rules_root = DEFAULT_RULES_ROOT / dataset_name
    rule_files_path = source_teacher_dir / "rule_files.json"
    if rule_files_path.exists() and rule_files_path.stat().st_size > 0:
        filenames = read_json(rule_files_path)
        rule_files = [str(name) for name in filenames]
    else:
        rule_files = sorted(path.name for path in rules_root.glob("*.txt"))
    chunks = []
    for filename in rule_files:
        chunks.append((rules_root / filename).read_text(encoding="utf-8").strip())
    return "\n\n".join(chunks).strip() + "\n", rule_files


def materialize_teacher_artifacts(
    *,
    source_run_name: str,
    run_paths: RunPaths,
    dataset_name: str,
    splits: dict[str, pd.DataFrame],
    seed: int,
) -> dict[str, Any]:
    source_run = load_teacher_source_run(source_run_name)
    source_teacher_dir = source_run / "teacher_phase"
    accepted_rows = read_jsonl(source_teacher_dir / "accepted_samples.jsonl")
    rejected_path = source_teacher_dir / "rejected_samples.jsonl"
    rejected_rows = read_jsonl(rejected_path) if rejected_path.exists() and rejected_path.stat().st_size > 0 else []
    teacher_summary_path = source_teacher_dir / "teacher_summary.json"
    teacher_summary = read_json(teacher_summary_path) if teacher_summary_path.exists() and teacher_summary_path.stat().st_size > 0 else {
        "source_run_name": source_run_name,
        "accepted_samples": len(accepted_rows),
        "rejected_samples": len(rejected_rows),
    }
    rulebook, rule_files = build_rulebook_from_source(source_run_name, dataset_name)

    run_paths.teacher_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(run_paths.teacher_dir / "accepted_samples.jsonl", accepted_rows)
    write_jsonl(run_paths.teacher_dir / "rejected_samples.jsonl", rejected_rows)
    write_json(run_paths.teacher_dir / "rule_files.json", rule_files)
    (run_paths.teacher_dir / "rulebook.txt").write_text(rulebook, encoding="utf-8")
    write_json(run_paths.teacher_dir / "teacher_summary.json", teacher_summary)

    split_summary = build_split_summary(splits, DEFAULT_DATASET_ROOT, dataset_name, seed)
    phase_summary = {
        "run_dir": str(run_paths.run_dir),
        "teacher_dir": str(run_paths.teacher_dir),
        "split_summary": split_summary,
        "teacher_summary": teacher_summary,
        "source_run_name": source_run_name,
    }
    write_json(run_paths.teacher_dir / "teacher_phase_summary.json", phase_summary)
    return {
        "accepted_rows": accepted_rows,
        "rejected_rows": rejected_rows,
        "teacher_summary": teacher_summary,
        "rulebook": rulebook,
        "rule_files": rule_files,
        "phase_summary": phase_summary,
    }


def load_teacher_source_artifacts(source_run_name: str, dataset_name: str) -> dict[str, Any]:
    source_run = load_teacher_source_run(source_run_name)
    source_teacher_dir = source_run / "teacher_phase"
    accepted_rows = read_jsonl(source_teacher_dir / "accepted_samples.jsonl")
    rejected_path = source_teacher_dir / "rejected_samples.jsonl"
    rejected_rows = read_jsonl(rejected_path) if rejected_path.exists() and rejected_path.stat().st_size > 0 else []
    teacher_summary_path = source_teacher_dir / "teacher_summary.json"
    teacher_summary = read_json(teacher_summary_path) if teacher_summary_path.exists() and teacher_summary_path.stat().st_size > 0 else {
        "source_run_name": source_run_name,
        "accepted_samples": len(accepted_rows),
        "rejected_samples": len(rejected_rows),
    }
    rulebook, rule_files = build_rulebook_from_source(source_run_name, dataset_name)
    return {
        "accepted_rows": accepted_rows,
        "rejected_rows": rejected_rows,
        "teacher_summary": teacher_summary,
        "rulebook": rulebook,
        "rule_files": rule_files,
        "phase_summary": None,
    }


def load_materialized_teacher_artifacts(run_paths: RunPaths) -> dict[str, Any]:
    teacher_dir = run_paths.teacher_dir
    return {
        "accepted_rows": read_jsonl(teacher_dir / "accepted_samples.jsonl"),
        "rejected_rows": read_jsonl(teacher_dir / "rejected_samples.jsonl")
        if (teacher_dir / "rejected_samples.jsonl").exists() and (teacher_dir / "rejected_samples.jsonl").stat().st_size > 0
        else [],
        "teacher_summary": read_json(teacher_dir / "teacher_summary.json"),
        "rulebook": (teacher_dir / "rulebook.txt").read_text(encoding="utf-8"),
        "rule_files": read_json(teacher_dir / "rule_files.json"),
        "phase_summary": read_json(teacher_dir / "teacher_phase_summary.json"),
    }


def latest_checkpoint_dir(output_dir: Path) -> Path | None:
    checkpoints = sorted(output_dir.glob("checkpoint-*"), key=lambda path: int(path.name.split("-")[-1]))
    if not checkpoints:
        return None
    return checkpoints[-1]


def checkpoint_step(path_like: str | Path | None) -> int:
    if path_like is None:
        return 0
    name = Path(path_like).name
    if "-" in name:
        tail = name.split("-")[-1]
        if tail.isdigit():
            return int(tail)
    return 0


def make_resolved_config(payload: dict[str, Any], run_name: str) -> dict[str, Any]:
    resolved = dict(payload)
    resolved["run_name"] = run_name
    return resolved
