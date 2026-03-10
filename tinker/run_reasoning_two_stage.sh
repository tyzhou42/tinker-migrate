#!/usr/bin/env bash
# NOTE: This script relies on bash features (arrays, [[ ]], process substitution).
# If it's invoked via `sh` (e.g., `sh script.sh` or from a `shell=True` subprocess),
# re-exec under bash so we fail less mysteriously.
if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi

set -euo pipefail

# Resolve tinker root from this script location, so paths work from any cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TINKER_ROOT="${SCRIPT_DIR}"

# ==========================
# Tunable Arguments (Edit Me)
# ==========================
PYTHON_BIN="python3"
CONFIG_PATH="${TINKER_ROOT}/configs/reasoning_sft.example.json"
RUN_NAME="reasoning_sft_$(date +%Y%m%d_%H%M%S)"
WANDB_MODE="online"       # online | offline | disabled
SEED="42"
ENABLE_STAGE1="true"      # true | false (teacher sampling + rejection)
ENABLE_STAGE2="true"      # true | false (student train/eval)
ENABLE_INFERENCE="false"  # true | false (student inference/export)
INFER_REASONING="true"           # true | false (reasoning prompt eval + generation export)
INFER_LABEL_ONLY="true"          # true | false (label-only eval; metrics only)
DATASET_NAME="ethos"
DATASET_ROOT_DIR="${TINKER_ROOT}/dataset"
RULES_ROOT_DIR="${TINKER_ROOT}/rules"
TEACHER_K="4"
TEACHER_WORKERS="${TEACHER_WORKERS:-15}"
TEACHER_MAX_TOKENS="3072"
TEACHER_TEMP="1.0"
INFER_WORKERS="${INFER_WORKERS:-15}"
# Max new tokens for generation-style eval/exports (student eval + inference).
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-3072}"
INFER_TEMPERATURE="0.0"
INFER_FORCE="false"
SELECTION_METRIC="macro_f1"  # macro_f1 | auroc | auprc | accuracy | loss | cls_loss
USE_EMITTED_LABEL_METRICS="false"  # true | false (use raw emitted 0/1 for discrete metrics; skip threshold search)
STUDENT_USE_RULEBOOK="false"  # true | false (whether student SFT prompts include the rulebook)
MAX_TRAIN_EXAMPLES="0"       # 0 means use all
MAX_VAL_EXAMPLES="0"         # 0 means use all
MAX_TEST_EXAMPLES="0"        # 0 means use all
TEACHER_MODEL_NAME="deepseek-chat"  # e.g., deepseek-chat | gemini-3-flash-preview
STUDENT_MODEL_NAME="Qwen/Qwen3-8B"
LORA_RANK="16"
NUM_EPOCHS="3"
BATCH_SIZE="32"
# Per-epoch batch ordering: 'stratified' keeps each batch close to the overall train label ratio.
BATCH_STRATEGY="${BATCH_STRATEGY:-stratified}"  # random | stratified
LEARNING_RATE="0.0002"
REASONING_TOKEN_WEIGHT="1.0"
LABEL_TOKEN_WEIGHT="1.0"
MAX_CONTEXT_LENGTH="${MAX_CONTEXT_LENGTH:-20480}"
EVAL_INTERVAL="16"
MAX_EVAL_SAMPLES="0"          # 0 means full split
EVAL_MAX_CONCURRENCY="${EVAL_MAX_CONCURRENCY:-25}"
# Best practice: avoid test-set peeking during training. Final/bestval test eval still runs at the end.
EVAL_TEST_DURING_TRAIN="${EVAL_TEST_DURING_TRAIN:-false}"  # true | false
# Optional: during periodic eval events, also run end-to-end evaluation on epoch's first/last train batch.
TRAIN_CHECKPOINT_E2E_EVAL="${TRAIN_CHECKPOINT_E2E_EVAL:-true}"  # true | false
# Save val per-example JSON every K eval events during student training (0 disables).
VAL_EXPORT_EVERY_EVALS="${VAL_EXPORT_EVERY_EVALS:-1}"
TTL_SECONDS="7776000"         # 90 days
SAVE_RESUME_CHECKPOINT="true"  # true | false
DOWNLOAD_ADAPTER="false"       # true | false
ADAPTER_OUTPUT_SUBDIR="model"  # saved under: <run_dir>/<ADAPTER_OUTPUT_SUBDIR>/
ADAPTER_DOWNLOAD_TIMEOUT_SECONDS="600"

# Optional: export here if needed
# export GOOGLE_API_KEY="..."
# export TINKER_API_KEY="..."
# export DEEPSEEK_API_KEY="..."

echo "======================================="
echo "Two-Stage Reasoning SFT Run"
echo "======================================="
echo "Python            : ${PYTHON_BIN}"
echo "Config            : ${CONFIG_PATH}"
echo "Run name          : ${RUN_NAME}"
echo "W&B mode          : ${WANDB_MODE}"
echo "Seed              : ${SEED}"
echo "Enable stage1     : ${ENABLE_STAGE1}"
echo "Enable stage2     : ${ENABLE_STAGE2}"
echo "Enable inference  : ${ENABLE_INFERENCE}"
echo "Infer reasoning   : ${INFER_REASONING}"
echo "Infer label-only  : ${INFER_LABEL_ONLY}"
echo "Dataset name      : ${DATASET_NAME}"
echo "Dataset root dir  : ${DATASET_ROOT_DIR}"
echo "Rules root dir    : ${RULES_ROOT_DIR}"
echo "Teacher k         : ${TEACHER_K}"
echo "Teacher workers   : ${TEACHER_WORKERS}"
echo "Teacher max tokens: ${TEACHER_MAX_TOKENS}"
echo "Teacher temp      : ${TEACHER_TEMP}"
echo "Infer workers     : ${INFER_WORKERS}"
echo "Max new tokens    : ${MAX_NEW_TOKENS}"
echo "Infer temp        : ${INFER_TEMPERATURE}"
echo "Infer force       : ${INFER_FORCE}"
echo "Selection metric  : ${SELECTION_METRIC}"
echo "Student rulebook  : ${STUDENT_USE_RULEBOOK}"
echo "Max train rows    : ${MAX_TRAIN_EXAMPLES}"
echo "Max val rows      : ${MAX_VAL_EXAMPLES}"
echo "Max test rows     : ${MAX_TEST_EXAMPLES}"
echo "Teacher model     : ${TEACHER_MODEL_NAME}"
echo "Student model     : ${STUDENT_MODEL_NAME}"
echo "LoRA rank         : ${LORA_RANK}"
echo "Epochs            : ${NUM_EPOCHS}"
echo "Batch size        : ${BATCH_SIZE}"
echo "Batch strategy    : ${BATCH_STRATEGY}"
echo "Learning rate     : ${LEARNING_RATE}"
echo "Reasoning wgt     : ${REASONING_TOKEN_WEIGHT}"
echo "Label wgt         : ${LABEL_TOKEN_WEIGHT}"
echo "Max context tokens: ${MAX_CONTEXT_LENGTH}"
echo "Eval interval     : ${EVAL_INTERVAL}"
echo "Max eval samples  : ${MAX_EVAL_SAMPLES}"
echo "Eval concurrency  : ${EVAL_MAX_CONCURRENCY}"
echo "Mid-run test eval : ${EVAL_TEST_DURING_TRAIN}"
echo "Train e2e eval    : ${TRAIN_CHECKPOINT_E2E_EVAL}"
echo "Val export every  : ${VAL_EXPORT_EVERY_EVALS} evals"
echo "Checkpoint TTL    : ${TTL_SECONDS}"
echo "Save resume ckpt  : ${SAVE_RESUME_CHECKPOINT}"
echo "Download adapter  : ${DOWNLOAD_ADAPTER}"
echo "Adapter subdir    : ${ADAPTER_OUTPUT_SUBDIR}"
echo

RUNS_DIR="$("${PYTHON_BIN}" - "${CONFIG_PATH}" "${TINKER_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

cfg_path = Path(sys.argv[1])
tinker_root = Path(sys.argv[2])
log_dir = "runs"
if cfg_path.exists():
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if isinstance(cfg, dict) and cfg.get("log_dir") is not None:
            log_dir = str(cfg["log_dir"])
    except Exception:
        pass
p = Path(log_dir)
if not p.is_absolute():
    if p.parts and p.parts[0] == "tinker":
        p = tinker_root.parent / p
    else:
        p = tinker_root / p
print(str(p))
PY
)"

has_teacher_artifacts() {
  local run_name="$1"
  local base="${RUNS_DIR}/${run_name}"
  [[ -f "${base}/teacher_phase/accepted_samples.jsonl" ]] && \
  [[ -f "${base}/teacher_phase/teacher_summary.json" ]] && \
  [[ -f "${base}/teacher_phase/rulebook.txt" ]] && \
  [[ -f "${base}/data_splits/train.csv" ]] && \
  [[ -f "${base}/data_splits/val.csv" ]] && \
  [[ -f "${base}/data_splits/test.csv" ]]
}

find_latest_run_with_artifacts() {
  if [[ ! -d "${RUNS_DIR}" ]]; then
    return 1
  fi
  while IFS= read -r candidate_dir; do
    local candidate_name
    candidate_name="$(basename "${candidate_dir}")"
    if has_teacher_artifacts "${candidate_name}"; then
      echo "${candidate_name}"
      return 0
    fi
  done < <(
    find "${RUNS_DIR}" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' \
      | sort -nr \
      | cut -d' ' -f2-
  )
  return 1
}

if [[ "${ENABLE_STAGE1}" == "true" ]]; then
  echo "[1/2] Teacher phase: sampling + rejection"
  "${PYTHON_BIN}" "${TINKER_ROOT}/SFT_reasoning_teacher.py" \
    --config "${CONFIG_PATH}" \
    --seed "${SEED}" \
    --dataset-name "${DATASET_NAME}" \
    --dataset-root-dir "${DATASET_ROOT_DIR}" \
    --rules-root-dir "${RULES_ROOT_DIR}" \
    --run-name "${RUN_NAME}" \
    --wandb-mode "${WANDB_MODE}" \
    --teacher-k "${TEACHER_K}" \
    --teacher-workers "${TEACHER_WORKERS}" \
    --teacher-max-tokens "${TEACHER_MAX_TOKENS}" \
    --teacher-temperature "${TEACHER_TEMP}" \
    --selection-metric "${SELECTION_METRIC}" \
    --max-train-examples "${MAX_TRAIN_EXAMPLES}" \
    --max-val-examples "${MAX_VAL_EXAMPLES}" \
    --max-test-examples "${MAX_TEST_EXAMPLES}" \
    --teacher-model-name "${TEACHER_MODEL_NAME}"
else
  echo "[1/2] Teacher phase skipped (ENABLE_STAGE1=${ENABLE_STAGE1})"
fi

echo
if [[ "${ENABLE_STAGE2}" == "true" ]]; then
  OUTPUT_RUN_NAME="${RUN_NAME}"
  TEACHER_RUN_NAME="${RUN_NAME}"
  if ! has_teacher_artifacts "${TEACHER_RUN_NAME}"; then
    echo "Stage2 could not find stage1 artifacts under run '${TEACHER_RUN_NAME}'." >&2
    latest_run="$(find_latest_run_with_artifacts || true)"
    if [[ -n "${latest_run}" ]]; then
      echo "Sourcing teacher artifacts from latest run: ${latest_run}" >&2
      TEACHER_RUN_NAME="${latest_run}"
    else
      echo "ERROR: No existing run with valid teacher artifacts found under ${RUNS_DIR}" >&2
      exit 1
    fi
  fi

  echo "[2/2] Student phase: train + eval from teacher artifacts"
  echo "Output run name   : ${OUTPUT_RUN_NAME}"
  echo "Teacher run name  : ${TEACHER_RUN_NAME}"
  STUDENT_FLAGS=()
  if [[ "${SAVE_RESUME_CHECKPOINT}" == "true" ]]; then
    STUDENT_FLAGS+=(--save-resume-checkpoint)
  else
    STUDENT_FLAGS+=(--no-save-resume-checkpoint)
  fi
  if [[ "${USE_EMITTED_LABEL_METRICS}" == "true" ]]; then
    STUDENT_FLAGS+=(--use-emitted-label-metrics)
  else
    STUDENT_FLAGS+=(--no-use-emitted-label-metrics)
  fi
  if [[ "${STUDENT_USE_RULEBOOK}" == "true" ]]; then
    STUDENT_FLAGS+=(--student-use-rulebook)
  else
    STUDENT_FLAGS+=(--no-student-use-rulebook)
  fi
  if [[ "${DOWNLOAD_ADAPTER}" == "true" ]]; then
    STUDENT_FLAGS+=(--download-adapter)
  else
    STUDENT_FLAGS+=(--no-download-adapter)
  fi
  if [[ "${EVAL_TEST_DURING_TRAIN}" == "true" ]]; then
    STUDENT_FLAGS+=(--eval-test-during-train)
  else
    STUDENT_FLAGS+=(--no-eval-test-during-train)
  fi
  if [[ "${TRAIN_CHECKPOINT_E2E_EVAL}" == "true" ]]; then
    STUDENT_FLAGS+=(--train-checkpoint-e2e-eval)
  else
    STUDENT_FLAGS+=(--no-train-checkpoint-e2e-eval)
  fi
	  "${PYTHON_BIN}" "${TINKER_ROOT}/SFT_reasoning_student.py" \
    --config "${CONFIG_PATH}" \
    --seed "${SEED}" \
    --run-name "${OUTPUT_RUN_NAME}" \
    --teacher-run-name "${TEACHER_RUN_NAME}" \
    --wandb-mode "${WANDB_MODE}" \
    --selection-metric "${SELECTION_METRIC}" \
    --student-model-name "${STUDENT_MODEL_NAME}" \
    --lora-rank "${LORA_RANK}" \
    --num-epochs "${NUM_EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --batch-strategy "${BATCH_STRATEGY}" \
    --learning-rate "${LEARNING_RATE}" \
	    --reasoning-token-weight "${REASONING_TOKEN_WEIGHT}" \
	    --label-token-weight "${LABEL_TOKEN_WEIGHT}" \
	    --max-length "${MAX_CONTEXT_LENGTH}" \
	    --max-new-tokens "${MAX_NEW_TOKENS}" \
	    --eval-interval "${EVAL_INTERVAL}" \
	    --val-export-every-evals "${VAL_EXPORT_EVERY_EVALS}" \
	    --max-eval-samples "${MAX_EVAL_SAMPLES}" \
	    --eval-max-concurrency "${EVAL_MAX_CONCURRENCY}" \
	    --ttl-seconds "${TTL_SECONDS}" \
    --adapter-output-subdir "${ADAPTER_OUTPUT_SUBDIR}" \
    --adapter-download-timeout-seconds "${ADAPTER_DOWNLOAD_TIMEOUT_SECONDS}" \
    "${STUDENT_FLAGS[@]}"
else
  echo "[2/2] Student phase skipped (ENABLE_STAGE2=${ENABLE_STAGE2})"
fi

echo
if [[ "${ENABLE_INFERENCE}" == "true" ]]; then
  echo "[3/3] Inference/export phase: generate student outputs"
  INFER_RUN_NAME="${OUTPUT_RUN_NAME:-${RUN_NAME}}"
  INFER_TEACHER_RUN_NAME="${TEACHER_RUN_NAME:-${RUN_NAME}}"

  # Prefer the model pointer(s) under tinker/model/ so inference uses the same
  # checkpoint selected by the student training phase (and doesn't depend on
  # config fingerprint matching).
  INFER_POINTER_RUN_NAME="$("${PYTHON_BIN}" - "${TINKER_ROOT}" "${INFER_RUN_NAME}" <<'PY'
import json
import sys
from pathlib import Path

tinker_root = Path(sys.argv[1])
run_name = sys.argv[2]
model_dir = tinker_root / "model"

for fp in (model_dir / f"{run_name}.json", model_dir / "latest.json"):
    if not fp.exists():
        continue
    try:
        obj = json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        continue
    pointer_run = str(obj.get("run_name") or "").strip()
    if pointer_run:
        print(pointer_run)
        raise SystemExit(0)
print("")
PY
)"

  INFER_CHECKPOINT_PATH="$("${PYTHON_BIN}" - "${TINKER_ROOT}" "${INFER_RUN_NAME}" <<'PY'
import json
import sys
from pathlib import Path

tinker_root = Path(sys.argv[1])
run_name = sys.argv[2]
model_dir = tinker_root / "model"

for fp in (model_dir / f"{run_name}.json", model_dir / "latest.json"):
    if not fp.exists():
        continue
    try:
        obj = json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        continue
    ckpt = (obj.get("best_model_path") or obj.get("final_model_path") or "").strip()
    if ckpt:
        print(ckpt)
        raise SystemExit(0)
print("")
PY
)"

  # If stage1/2 are disabled, RUN_NAME is typically a fresh timestamp and the
  # run directory won't exist. In that case, use the run referenced by the model
  # pointer (or fall back to latest run with teacher artifacts).
  if [[ ! -d "${RUNS_DIR}/${INFER_RUN_NAME}" ]]; then
    if [[ -n "${INFER_POINTER_RUN_NAME}" && -d "${RUNS_DIR}/${INFER_POINTER_RUN_NAME}" ]]; then
      echo "Inference-only mode: using existing run dir from model pointer: ${INFER_POINTER_RUN_NAME}" >&2
      INFER_RUN_NAME="${INFER_POINTER_RUN_NAME}"
      INFER_TEACHER_RUN_NAME="${INFER_POINTER_RUN_NAME}"
    else
      latest_run="$(find_latest_run_with_artifacts || true)"
      if [[ -n "${latest_run}" ]]; then
        echo "Inference-only mode: using latest run dir with teacher artifacts: ${latest_run}" >&2
        INFER_RUN_NAME="${latest_run}"
        INFER_TEACHER_RUN_NAME="${latest_run}"
      fi
    fi
  fi

  # Recompute checkpoint for the final INFER_RUN_NAME so run dir + checkpoint stay consistent.
  INFER_CHECKPOINT_PATH="$("${PYTHON_BIN}" - "${TINKER_ROOT}" "${INFER_RUN_NAME}" <<'PY'
import json
import sys
from pathlib import Path

tinker_root = Path(sys.argv[1])
run_name = sys.argv[2]
model_dir = tinker_root / "model"

for fp in (model_dir / f"{run_name}.json", model_dir / "latest.json"):
    if not fp.exists():
        continue
    try:
        obj = json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        continue
    ckpt = (obj.get("best_model_path") or obj.get("final_model_path") or "").strip()
    if ckpt:
        print(ckpt)
        raise SystemExit(0)
print("")
PY
)"

  INFER_FLAGS=()
  if [[ "${INFER_FORCE}" == "true" ]]; then
    INFER_FLAGS+=(--force)
  else
    INFER_FLAGS+=(--no-force)
  fi
  if [[ "${INFER_REASONING}" == "true" ]]; then
    INFER_FLAGS+=(--reasoning)
  else
    INFER_FLAGS+=(--no-reasoning)
  fi
  if [[ "${INFER_LABEL_ONLY}" == "true" ]]; then
    INFER_FLAGS+=(--label-only)
  else
    INFER_FLAGS+=(--no-label-only)
  fi
  if [[ -n "${INFER_CHECKPOINT_PATH}" ]]; then
    INFER_FLAGS+=(--checkpoint-path "${INFER_CHECKPOINT_PATH}")
  fi
	  "${PYTHON_BIN}" "${TINKER_ROOT}/SFT_inference.py" \
    --config "${CONFIG_PATH}" \
    --seed "${SEED}" \
    --run-name "${INFER_RUN_NAME}" \
    --teacher-run-name "${INFER_TEACHER_RUN_NAME}" \
	    --student-model-name "${STUDENT_MODEL_NAME}" \
	    --lora-rank "${LORA_RANK}" \
	    --max-length "${MAX_CONTEXT_LENGTH}" \
	    --eval-max-concurrency "${EVAL_MAX_CONCURRENCY}" \
	    --workers "${INFER_WORKERS}" \
	    --max-new-tokens "${MAX_NEW_TOKENS}" \
	    --temperature "${INFER_TEMPERATURE}" \
	    "${INFER_FLAGS[@]}"
else
  echo "[3/3] Inference/export phase skipped (ENABLE_INFERENCE=${ENABLE_INFERENCE})"
fi

echo
echo "Completed run request: ${RUN_NAME}"
