#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi

set -euo pipefail

export WANDB_API_KEY="tml-cTfkvS9E4qfpeNKQtJh8shzn7EjNsem2zzERqjRkfteeko8N40RnomcKFmJbh4gMWBAAAA"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/mnt/data2/zhongmouhe/conda_envs/ml/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-/mnt/data2/zhongmouhe/conda_envs/ml/bin/accelerate}"
NPROC_PER_NODE="${NPROC_PER_NODE:-6}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-0}"

RUN_NAME="${RUN_NAME:-label_sft_$(date +%Y%m%d_%H%M%S)}"
DATASET_NAME="${DATASET_NAME:-ethos}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-8B}"
WANDB_PROJECT="${WANDB_PROJECT:-reasoning-sft-gpu}"
WANDB_MODE="${WANDB_MODE:-online}"

detect_gpu_count() {
  local raw=""
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    raw="${CUDA_VISIBLE_DEVICES}"
  elif [[ -n "${SLURM_STEP_GPUS:-}" ]]; then
    raw="${SLURM_STEP_GPUS}"
  elif [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
    raw="${SLURM_JOB_GPUS}"
  fi

  if [[ -z "${raw}" ]]; then
    echo "${NPROC_PER_NODE}"
    return 0
  fi

  IFS=',' read -r -a gpu_list <<< "${raw}"
  echo "${#gpu_list[@]}"
}

GPU_COUNT="$(detect_gpu_count)"
if [[ "${GPU_COUNT}" -le 0 ]]; then
  echo "No visible GPUs detected." >&2
  exit 1
fi

LAUNCH_ARGS=(
  launch
  --num_processes "${GPU_COUNT}"
  --num_machines 1
  --main_process_port "${MAIN_PROCESS_PORT}"
  --mixed_precision bf16
  --dynamo_backend no
)
if [[ "${GPU_COUNT}" -gt 1 ]]; then
  LAUNCH_ARGS+=(--multi_gpu)
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

echo "======================================="
echo "GPU Label SFT"
echo "======================================="
echo "Run name      : ${RUN_NAME}"
echo "Model         : ${MODEL_NAME}"
echo "Dataset       : ${DATASET_NAME}"
echo "GPUs          : ${GPU_COUNT}"
echo "Launcher      : accelerate"
echo "Port          : ${MAIN_PROCESS_PORT}"
echo "Python        : ${PYTHON_BIN}"
echo "Unbuffered    : ${PYTHONUNBUFFERED}"
echo "W&B project   : ${WANDB_PROJECT}"
echo "W&B mode      : ${WANDB_MODE}"
echo

"${PYTHON_BIN}" "${SCRIPT_DIR}/label_run_prep.py" \
  --run-name "${RUN_NAME}" \
  --dataset-name "${DATASET_NAME}" \
  --model-name "${MODEL_NAME}" \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-mode "${WANDB_MODE}"

"${ACCELERATE_BIN}" \
  "${LAUNCH_ARGS[@]}" \
  "${SCRIPT_DIR}/sft_label.py" \
  --run-name "${RUN_NAME}" \
  --dataset-name "${DATASET_NAME}" \
  --model-name "${MODEL_NAME}" \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-mode "${WANDB_MODE}" \
  "$@"
