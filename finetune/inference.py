#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from peft import PeftModel

import common


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="对已经训练好的 LoRA checkpoint 做 val/test 推理导出。")
    p.add_argument("--checkpoint-path", type=Path, required=True)
    p.add_argument("--model-name", type=str, default="Qwen/Qwen3-8B")
    p.add_argument("--dataset-root", type=Path, default=common.DEFAULT_DATASET_ROOT)
    p.add_argument("--dataset-name", type=str, default="ethos")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--mode", choices=["label", "reasoning"], default="label")
    p.add_argument("--student-use-rulebook", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--source-run-name", type=str, default="reasoning_sft_20260301_114319")
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    return p.parse_args()


def load_model(base_model_name: str, adapter_path: Path, dtype: str):
    base_model = common.load_model(
        base_model_name,
        dtype_name=dtype,
        use_gradient_checkpointing=False,
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    if common.torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    splits = common.load_dataset_splits(args.dataset_root, args.dataset_name)
    tokenizer = common.load_tokenizer(args.model_name)
    model = load_model(args.model_name, args.checkpoint_path, args.dtype)

    if args.mode == "label":
        def prompt_builder(text: str) -> str:
            return common.build_label_prompt(text, tokenizer, args.max_length)
    else:
        teacher_source = common.build_rulebook_from_source(args.source_run_name, args.dataset_name)
        rulebook = teacher_source[0]

        def prompt_builder(text: str) -> str:
            return common.default_reasoning_prompt(
                text,
                rulebook=rulebook if args.student_use_rulebook else None,
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("val", "test"):
        if args.mode == "label":
            records = common.run_label_candidate_eval(
                model=model,
                tokenizer=tokenizer,
                frame=splits[split],
                split_name=split,
                build_prompt=prompt_builder,
            )
        else:
            records = common.run_reasoning_generation_eval(
                model=model,
                tokenizer=tokenizer,
                frame=splits[split],
                split_name=split,
                build_prompt=prompt_builder,
                max_length=args.max_length,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                reasoning_placeholder="Reasoning intentionally omitted for label scoring.",
                invalid_label_warn_rate=0.1,
                enable_thinking=True,
                eval_batch_size=4,
            )
        common.save_eval_bundle(
            output_dir=args.output_dir,
            checkpoint_path=str(args.checkpoint_path),
            checkpoint_name=args.checkpoint_path.name,
            split_name=split,
            tag="manual",
            step=0,
            records=records,
        )


if __name__ == "__main__":
    main()
