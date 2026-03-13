#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Any

import wandb
from peft import PeftModel, get_peft_model
from transformers import TrainingArguments

import common


def log(message: str) -> None:
    print(f"[reasoning_sft] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="在本地 GPU 上训练 reasoning LoRA SFT。")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--source-run-name", type=str, default="reasoning_sft_20260301_114319")
    p.add_argument("--dataset-root", type=Path, default=common.DEFAULT_DATASET_ROOT)
    p.add_argument("--dataset-name", type=str, default="ethos")
    p.add_argument("--model-name", type=str, default="Qwen/Qwen3-8B")
    p.add_argument("--runs-root", type=Path, default=common.DEFAULT_RUNS_ROOT)
    p.add_argument("--model-root", type=Path, default=common.DEFAULT_MODEL_ROOT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--student-use-rulebook", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--num-epochs", type=float, default=3.0)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-ratio", type=float, default=0.0)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--reasoning-token-weight", type=float, default=1.0)
    p.add_argument("--label-token-weight", type=float, default=1.0)
    p.add_argument("--per-device-train-batch-size", type=int, default=2)
    p.add_argument("--per-device-eval-batch-size", type=int, default=2)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--eval-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=10)
    p.add_argument("--save-total-limit", type=int, default=8)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    p.add_argument("--student-enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--eval-reasoning-placeholder", type=str, default="Reasoning intentionally omitted for label scoring.")
    p.add_argument("--invalid-label-warn-rate", type=float, default=0.1)
    p.add_argument("--wandb-project", type=str, default="reasoning-sft-gpu")
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    return p.parse_args()


def build_resolved_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "dataset_root_dir": str(args.dataset_root),
        "dataset_name": args.dataset_name,
        "teacher_source_run_name": args.source_run_name,
        "student_model_name": args.model_name,
        "student_use_rulebook": args.student_use_rulebook,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "min_lr_ratio": args.min_lr_ratio,
        "reasoning_token_weight": args.reasoning_token_weight,
        "label_token_weight": args.label_token_weight,
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
        "student_enable_thinking": args.student_enable_thinking,
        "eval_reasoning_placeholder": args.eval_reasoning_placeholder,
        "invalid_label_warn_rate": args.invalid_label_warn_rate,
        "wandb_project": args.wandb_project,
        "wandb_mode": args.wandb_mode,
        "selection_metric": "macro_f1",
        "seed": args.seed,
    }


def maybe_setup_wandb(args: argparse.Namespace) -> str:
    if args.wandb_mode == "disabled":
        os.environ["WANDB_MODE"] = "disabled"
        return "none"
    os.environ["WANDB_PROJECT"] = args.wandb_project
    os.environ["WANDB_MODE"] = args.wandb_mode
    return "wandb"


def load_adapter_model(base_model_name: str, adapter_path: str, dtype: str) -> Any:
    base_model = common.load_model(
        base_model_name,
        dtype_name=dtype,
        use_gradient_checkpointing=False,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    if common.torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    run_name = args.run_name or f"reasoning_sft_{common.now_ts()}"
    common.seed_everything(args.seed)
    log(f"starting run_name={run_name}")

    run_paths = common.make_run_paths(run_name, args.runs_root)
    run_paths.run_dir.mkdir(parents=True, exist_ok=True)
    log(f"loading dataset from {args.dataset_root / args.dataset_name}")
    splits = common.load_dataset_splits(args.dataset_root, args.dataset_name)

    if run_paths.teacher_dir.exists():
        teacher_info = common.load_materialized_teacher_artifacts(run_paths)
    else:
        teacher_info = common.load_teacher_source_artifacts(args.source_run_name, args.dataset_name)
    log("teacher artifacts ready in memory")
    log(f"teacher artifacts ready accepted={len(teacher_info['accepted_rows'])}")

    if os.environ.get("LOCAL_RANK", "0") == "0":
        common.copy_dataset_splits_to_run(run_paths, splits, args.dataset_root, args.dataset_name)
        common.write_json(
            run_paths.run_dir / "split_summary.json",
            common.build_split_summary(splits, args.dataset_root, args.dataset_name, args.seed),
        )
        common.write_json(
            run_paths.run_dir / "resolved_config.json",
            common.make_resolved_config(build_resolved_config(args), run_name),
        )

    accepted_rows = list(teacher_info["accepted_rows"])
    random.Random(args.seed).shuffle(accepted_rows)

    log(f"loading tokenizer {args.model_name}")
    tokenizer = common.load_tokenizer(args.model_name)
    log("tokenizer loaded")
    train_dataset = common.build_reasoning_tokenized_dataset(
        accepted_rows,
        tokenizer=tokenizer,
        rulebook=teacher_info["rulebook"],
        use_rulebook=args.student_use_rulebook,
        max_length=args.max_length,
        reasoning_token_weight=args.reasoning_token_weight,
        label_token_weight=args.label_token_weight,
        enable_thinking=args.student_enable_thinking,
    )
    eval_dataset = train_dataset.select(range(min(32, len(train_dataset))))
    if os.environ.get("LOCAL_RANK", "0") == "0":
        common.export_train_sft_jsonl(
            common.build_reasoning_dataset(
                accepted_rows,
                rulebook=teacher_info["rulebook"],
                use_rulebook=args.student_use_rulebook,
            ),
            run_paths.run_dir / "train_sft.jsonl",
        )
    log(f"built tokenized datasets train={len(train_dataset)} eval={len(eval_dataset)}")
    log(f"loading model {args.model_name} dtype={args.dtype}")
    model = common.load_model(
        args.model_name,
        dtype_name=args.dtype,
        use_gradient_checkpointing=True,
    )
    log("model loaded")
    peft_config = common.make_lora_config(args.lora_rank, args.lora_alpha, args.lora_dropout)
    model = get_peft_model(model, peft_config)
    log("lora model prepared")

    report_to = maybe_setup_wandb(args)
    def prompt_builder(text: str) -> str:
        return common.default_reasoning_prompt(
            text,
            rulebook=teacher_info["rulebook"] if args.student_use_rulebook else None,
        )

    generation_eval_callback = common.GenerationEvalCallback(
        run_paths=run_paths,
        tokenizer=tokenizer,
        val_frame=splits["val"],
        test_frame=splits["test"],
        build_prompt=prompt_builder,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        selection_metric="macro_f1",
        eval_mode="reasoning_generation",
        max_length=args.max_length,
        reasoning_placeholder=args.eval_reasoning_placeholder,
        invalid_label_warn_rate=args.invalid_label_warn_rate,
        enable_thinking=args.student_enable_thinking,
        eval_batch_size=args.per_device_eval_batch_size,
    )
    trainer_args = TrainingArguments(
        output_dir=str(run_paths.student_dir),
        run_name=run_name,
        report_to=report_to,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type="linear",
        warmup_ratio=args.warmup_ratio,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        logging_steps=args.logging_steps,
        logging_strategy="steps",
        logging_first_step=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=False,
        seed=args.seed,
        bf16=(common.infer_dtype(args.dtype) == common.torch.bfloat16),
        fp16=(common.infer_dtype(args.dtype) == common.torch.float16),
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        save_only_model=True,
        remove_unused_columns=False,
    )

    callbacks = [generation_eval_callback]
    log("constructing trainer")
    trainer = common.ReasoningTrainer(
        model=model,
        args=trainer_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=common.make_reasoning_data_collator(tokenizer, model),
        callbacks=callbacks,
        min_lr_ratio=args.min_lr_ratio,
    )
    log("trainer ready, entering train()")
    trainer.train()
    if not trainer.is_world_process_zero():
        return

    log("main rank starting checkpoint discovery and evaluation")
    best_checkpoint = generation_eval_callback.best_checkpoint_path
    final_checkpoint = common.latest_checkpoint_dir(run_paths.student_dir)
    if final_checkpoint is None:
        raise RuntimeError("训练结束后没有找到任何 checkpoint。")
    if best_checkpoint is None or not Path(best_checkpoint).exists():
        best_checkpoint = str(final_checkpoint)
    best_step = common.checkpoint_step(best_checkpoint)
    final_step = common.checkpoint_step(final_checkpoint)

    final_model = load_adapter_model(args.model_name, str(final_checkpoint), args.dtype)
    best_model = load_adapter_model(args.model_name, str(best_checkpoint), args.dtype)

    final_val_records = common.run_reasoning_generation_eval(
        model=final_model,
        tokenizer=tokenizer,
        frame=splits["val"],
        split_name="val",
        build_prompt=prompt_builder,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        reasoning_placeholder=args.eval_reasoning_placeholder,
        invalid_label_warn_rate=args.invalid_label_warn_rate,
        enable_thinking=args.student_enable_thinking,
        eval_batch_size=args.per_device_eval_batch_size,
    )
    final_test_records = common.run_reasoning_generation_eval(
        model=final_model,
        tokenizer=tokenizer,
        frame=splits["test"],
        split_name="test",
        build_prompt=prompt_builder,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        reasoning_placeholder=args.eval_reasoning_placeholder,
        invalid_label_warn_rate=args.invalid_label_warn_rate,
        enable_thinking=args.student_enable_thinking,
        eval_batch_size=args.per_device_eval_batch_size,
    )
    best_val_records = common.run_reasoning_generation_eval(
        model=best_model,
        tokenizer=tokenizer,
        frame=splits["val"],
        split_name="val",
        build_prompt=prompt_builder,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        reasoning_placeholder=args.eval_reasoning_placeholder,
        invalid_label_warn_rate=args.invalid_label_warn_rate,
        enable_thinking=args.student_enable_thinking,
        eval_batch_size=args.per_device_eval_batch_size,
    )
    best_test_records = common.run_reasoning_generation_eval(
        model=best_model,
        tokenizer=tokenizer,
        frame=splits["test"],
        split_name="test",
        build_prompt=prompt_builder,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        reasoning_placeholder=args.eval_reasoning_placeholder,
        invalid_label_warn_rate=args.invalid_label_warn_rate,
        enable_thinking=args.student_enable_thinking,
        eval_batch_size=args.per_device_eval_batch_size,
    )

    final_val_metrics = common.save_eval_bundle(
        output_dir=run_paths.student_dir,
        checkpoint_path=str(final_checkpoint),
        checkpoint_name=f"{run_name}_final",
        split_name="val",
        tag="final",
        step=final_step,
        records=final_val_records,
    )
    final_test_metrics = common.save_eval_bundle(
        output_dir=run_paths.student_dir,
        checkpoint_path=str(final_checkpoint),
        checkpoint_name=f"{run_name}_final",
        split_name="test",
        tag="final",
        step=final_step,
        records=final_test_records,
    )
    best_val_metrics = common.save_eval_bundle(
        output_dir=run_paths.student_dir,
        checkpoint_path=str(best_checkpoint),
        checkpoint_name=f"{run_name}_bestval",
        split_name="val",
        tag="bestval",
        step=best_step,
        records=best_val_records,
    )
    best_test_metrics = common.save_eval_bundle(
        output_dir=run_paths.student_dir,
        checkpoint_path=str(best_checkpoint),
        checkpoint_name=f"{run_name}_bestval",
        split_name="test",
        tag="bestval",
        step=best_step,
        records=best_test_records,
    )

    pointer = common.save_model_pointer(
        model_root=args.model_root,
        run_name=run_name,
        final_model_path=str(final_checkpoint),
        best_model_path=str(best_checkpoint),
        resume_state_path=str(final_checkpoint),
    )

    run_summary = {
        "run_dir": str(run_paths.run_dir),
        "teacher_dir": str(run_paths.teacher_dir),
        "final_model_path": str(final_checkpoint),
        "best_model_path": str(best_checkpoint),
        "resume_state_path": str(final_checkpoint),
        "best_step": best_step,
        "selection_metric": "macro_f1",
        "best_val_macro_f1": None if generation_eval_callback.best_val_metrics is None else float(generation_eval_callback.best_val_metrics["macro_f1"]),
        "teacher_summary": teacher_info["teacher_summary"],
        "best_val_metrics": best_val_metrics["metrics"],
        "best_test_metrics": best_test_metrics["metrics"],
        "final_test_metrics": final_test_metrics["metrics"],
        "model_pointer": pointer,
    }
    common.write_json(run_paths.run_dir / "run_summary.json", run_summary)

    if wandb.run is not None:
        wandb.log(
            {
                "teacher/accepted_samples": len(teacher_info["accepted_rows"]),
                "teacher/rejected_samples": len(teacher_info["rejected_rows"]),
                "final_test/accuracy": final_test_metrics["metrics"]["accuracy"],
                "final_test/macro_f1": final_test_metrics["metrics"]["macro_f1"],
                "bestval_test/accuracy": best_test_metrics["metrics"]["accuracy"],
                "bestval_test/macro_f1": best_test_metrics["metrics"]["macro_f1"],
            },
            step=int(trainer.state.global_step),
        )
    print(
        "[final_eval] "
        f"best_step={best_step} "
        f"bestval_test_acc={float(best_test_metrics['metrics']['accuracy']):.6f} "
        f"bestval_test_macro_f1={float(best_test_metrics['metrics']['macro_f1']):.6f} "
        f"final_test_acc={float(final_test_metrics['metrics']['accuracy']):.6f} "
        f"final_test_macro_f1={float(final_test_metrics['metrics']['macro_f1']):.6f}",
        flush=True,
    )
    log(f"run complete summary={run_paths.run_dir / 'run_summary.json'}")


if __name__ == "__main__":
    main()
