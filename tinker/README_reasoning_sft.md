# Reasoning SFT (Teacher + Student)

## What it does
- Loads binary text dataset.
- Loads rulebook by concatenating `rules/*.txt` (typically `rules/<dataset_name>/*.txt`).
- Uses `deepseek-chat` teacher to generate `<think>...</think>` plus `<label>0</label>` / `<label>1</label>`.
- Applies rejection sampling (`pred_label == gold_label`, minimum reasoning length).
- Fine-tunes a Tinker LoRA student on accepted reasoning traces.
- Logs training/eval metrics to Weights & Biases.

## Required inputs
1. Put rules under `rules/` as `.txt` files (dataset-specific subfolders supported).
2. Ensure `extraction/.env` contains `DEEPSEEK_API_KEY` and `TINKER_API_KEY`.
3. Adjust config at `configs/reasoning_sft.example.json`.

## Run
```bash
cd <REPO_ROOT>/tinker
python3 SFT_reasoning.py --config configs/reasoning_sft.example.json
```

## Key outputs
- `runs/<run_name>/teacher_samples.jsonl`
- `runs/<run_name>/accepted_samples.jsonl`
- `runs/<run_name>/rejected_samples.jsonl`
- `runs/<run_name>/train_sft.jsonl` (if enabled)
- `runs/<run_name>/train.log`
- `runs/<run_name>/test_report.md`
- `runs/<run_name>/run_summary.json`
