#!/usr/bin/env bash
# run_rl.sh — GRPO RL training wrapper (tinker/RL.py)
# Usage: `bash tinker/run_rl.sh`
# Override vars inline, e.g. `GROUP_SIZE=16 NUM_EPOCHS=2 bash tinker/run_rl.sh`.
set -euo pipefail

# ── User-tunable args (edit here or override via env) ─────────────────────────
RUN_NAME="${RUN_NAME:-grpo_rl_$(date +%Y%m%d_%H%M%S)}"

# Init
INIT_SOURCE="${INIT_SOURCE:-sft}"                 # sft | base
INIT_CHECKPOINT="${INIT_CHECKPOINT:-final}"        # best | final (only for INIT_SOURCE=sft)
BASE_MODEL_NAME="${BASE_MODEL_NAME:-}"            # only for INIT_SOURCE=base
LORA_RANK="${LORA_RANK:-16}"                        # empty = use config lora_rank

# Prompting
RL_PROMPT_FILE="${RL_PROMPT_FILE:-prompts/rl_reasoning.yaml}"
RL_USE_RULEBOOK="${RL_USE_RULEBOOK:-false}"       # true | false

# Core RL
GROUP_SIZE="${GROUP_SIZE:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
LOSS_FN="${LOSS_FN:-ppo}"                         # ppo | importance_sampling | cispo
PPO_CLIP="${PPO_CLIP:-0.2}"
NUM_SUBSTEPS="${NUM_SUBSTEPS:-1}"
NORMALIZE_ADVANTAGES="${NORMALIZE_ADVANTAGES:-true}"  # true | false

# Regularizers / shaping
KL_BETA="${KL_BETA:-0.01}"
REWARD_EMA_DECAY="${REWARD_EMA_DECAY:-0.9}"
FORMAT_REWARD_PENALTY="${FORMAT_REWARD_PENALTY:-1.0}"
THINKING_REWARD_COEF="${THINKING_REWARD_COEF:-0.1}"
THINKING_REWARD_LEN="${THINKING_REWARD_LEN:-512}"      # tokens at which thinking bonus saturates

# Rollout sampling
ROLLOUT_MAX_TOKENS="${ROLLOUT_MAX_TOKENS:-3072}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
ROLLOUT_TOP_K="${ROLLOUT_TOP_K:--1}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
ROLLOUT_STOP="${ROLLOUT_STOP:-}"                       # comma-separated; empty = none

# Eval / checkpointing
EVAL_INTERVAL="${EVAL_INTERVAL:-30}"
EVAL_MAX_CONCURRENCY="${EVAL_MAX_CONCURRENCY:-32}"
SAMPLE_MAX_CONCURRENCY="${SAMPLE_MAX_CONCURRENCY:-32}"
VAL_EXPORT_EVERY_EVALS="${VAL_EXPORT_EVERY_EVALS:-4}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-3}"
SAVE_INTERVAL="${SAVE_INTERVAL:-0}"
TTL_SECONDS="${TTL_SECONDS:-7776000}"

# Optimizer
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.002}"

# ── Directories ───────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  GRPO RL Training"
echo "  Run name  : $RUN_NAME"
echo "  Epochs    : $NUM_EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Group size: $GROUP_SIZE"
echo "  LR        : $LEARNING_RATE"
echo "  Loss fn   : $LOSS_FN"
echo "  KL beta   : $KL_BETA"
echo "  EMA decay : $REWARD_EMA_DECAY"
echo "  Fmt pen   : $FORMAT_REWARD_PENALTY"
echo "  Think rw  : coef=$THINKING_REWARD_COEF len=$THINKING_REWARD_LEN"
echo "  Adv norm  : $NORMALIZE_ADVANTAGES"
echo "  Eval every: $EVAL_INTERVAL steps"
echo "  Eval conc : $EVAL_MAX_CONCURRENCY"
echo "  Samp conc : $SAMPLE_MAX_CONCURRENCY"
echo "  Val export: every $VAL_EXPORT_EVERY_EVALS evals"
echo "  Patience  : $EARLY_STOPPING_PATIENCE evals"
echo "  TTL       : $TTL_SECONDS sec"
if [[ "$INIT_SOURCE" == "base" ]]; then
  echo "  Init from : base model ($BASE_MODEL_NAME)"
else
  echo "  Init from : model/latest.json ($INIT_CHECKPOINT checkpoint)"
fi
echo "  RL prompt : $RL_PROMPT_FILE"
echo "  Rulebook  : $RL_USE_RULEBOOK"
echo "============================================================"

MODEL_FLAGS=()
if [[ "$INIT_SOURCE" == "base" ]]; then
  if [[ -z "$BASE_MODEL_NAME" ]]; then
    echo "BASE_MODEL_NAME must be set when INIT_SOURCE=base" >&2
    exit 2
  fi
  MODEL_FLAGS+=(--base-model-name "$BASE_MODEL_NAME")
fi

RL_PROMPT_FLAGS=()
if [[ "$RL_USE_RULEBOOK" == "true" ]]; then
  RL_PROMPT_FLAGS+=(--rl-use-rulebook)
else
  RL_PROMPT_FLAGS+=(--no-rl-use-rulebook)
fi

ADV_FLAGS=()
if [[ "$NORMALIZE_ADVANTAGES" == "true" ]]; then
  ADV_FLAGS+=(--normalize-advantages)
else
  ADV_FLAGS+=(--no-normalize-advantages)
fi

STOP_FLAGS=()
if [[ -n "${ROLLOUT_STOP}" ]]; then
  STOP_FLAGS+=(--rollout-stop "${ROLLOUT_STOP}")
fi

python3 RL.py \
    --run-name                   "$RUN_NAME"                \
    --init-source                "$INIT_SOURCE"             \
    --init-checkpoint            "$INIT_CHECKPOINT"          \
    --rl-prompt-file             "$RL_PROMPT_FILE"           \
    "${MODEL_FLAGS[@]}"                                     \
    ${LORA_RANK:+--lora-rank "$LORA_RANK"}                  \
    \
    --num-epochs                 "$NUM_EPOCHS"               \
    --batch-size                 "$BATCH_SIZE"               \
    --group-size                 "$GROUP_SIZE"               \
    --learning-rate              "$LEARNING_RATE"            \
    --loss-fn                    "$LOSS_FN"                  \
    --ppo-clip-coef              "$PPO_CLIP"                 \
    --num-substeps               "$NUM_SUBSTEPS"             \
    "${ADV_FLAGS[@]}"                                        \
    \
    --kl-beta                    "$KL_BETA"                  \
    --reward-ema-decay           "$REWARD_EMA_DECAY"          \
    --format-reward-penalty      "$FORMAT_REWARD_PENALTY"     \
    --thinking-reward-coef       "$THINKING_REWARD_COEF"      \
    --thinking-reward-len        "$THINKING_REWARD_LEN"       \
    \
    --rollout-max-tokens         "$ROLLOUT_MAX_TOKENS"        \
    --rollout-temperature        "$ROLLOUT_TEMPERATURE"       \
    --rollout-top-k              "$ROLLOUT_TOP_K"             \
    --rollout-top-p              "$ROLLOUT_TOP_P"             \
    "${STOP_FLAGS[@]}"                                       \
    \
    --eval-interval              "$EVAL_INTERVAL"             \
    --val-export-every-evals     "$VAL_EXPORT_EVERY_EVALS"    \
    --early-stopping-patience    "$EARLY_STOPPING_PATIENCE"   \
    --save-interval              "$SAVE_INTERVAL"             \
    --ttl-seconds                "$TTL_SECONDS"               \
    \
    --grad-clip-norm             "$GRAD_CLIP_NORM"           \
    --weight-decay               "$WEIGHT_DECAY"             \
    \
    --eval-max-concurrency       "$EVAL_MAX_CONCURRENCY"     \
    --sample-max-concurrency     "$SAMPLE_MAX_CONCURRENCY"   \
    "${RL_PROMPT_FLAGS[@]}"

echo "============================================================"
echo "  Training complete. Results in: runs/$RUN_NAME/rl_phase/"
echo "  Model pointer: model/rl_latest.json"
echo "============================================================"
