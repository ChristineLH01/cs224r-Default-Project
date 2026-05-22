#!/bin/bash
# Separate, self-contained SELS RLOO launcher. Does NOT modify the default
# train_rloo_modal.sh. Starts RLOO from the SELS-SFT v3 warm-start and adds
# --inject_state0 so each prompt is seeded with "<state_0> [nums]" (the model
# can't reliably emit the opening state token; the context manager provides it).
#
# Required before running:
#   export WANDB_API_KEY=...   (HF_TOKEN if any gated repo)
#
# All knobs are env-overridable, e.g. a short smoke:
#   BATCH_SIZE=16 GRADIENT_ACCUMULATION_STEPS=16 NUM_TRAINING_STEPS=2 \
#   SAVE_EVERY_N_STEPS=2 WANDB_NAME=sels_rloo_smoke bash rloo_trainer/train_rloo_sels.sh

set -euo pipefail
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export MODAL_GPU="${MODAL_GPU:-H100!}"
export MODAL_VOLUME_NAME="${MODAL_VOLUME_NAME:-default-proj-training}"
export MODAL_TIMEOUT_SECONDS="${MODAL_TIMEOUT_SECONDS:-86400}"
export MODAL_STARTUP_TIMEOUT_SECONDS="${MODAL_STARTUP_TIMEOUT_SECONDS:-1800}"

# SELS warm-start: the v3 checkpoint carries the 10 registered tokens; the
# trainer tokenizer MUST match it (so <state_*>/<backtrack_*> tokenize atomically).
sels_ckpt="${MODEL_NAME:-/vol/checkpoints/sft_model/sels_sft/sels_sft_v3/best_model/model}"
tokenizer_name="${TOKENIZER_NAME:-$sels_ckpt}"
dataset_name="${DATASET_NAME:-asingh15/countdown_tasks_3to4}"
wandb_project="${WANDB_PROJECT:-sels_rloo}"
save_dir="${SAVE_DIR:-/vol/checkpoints/rloo_checkpoints}"

curr_lr="${LR:-1e-5}"
batch_size="${BATCH_SIZE:-128}"
gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-128}"
gradient_clipping="${GRADIENT_CLIPPING:-0.0}"
group_size="${GROUP_SIZE:-16}"
num_training_steps="${NUM_TRAINING_STEPS:-100}"
kl_divergence_coefficient="${KL_DIVERGENCE_COEFFICIENT:-0.001}"
entropy_coefficient="${ENTROPY_COEFFICIENT:-0.001}"
save_every_n_steps="${SAVE_EVERY_N_STEPS:-25}"
lr_schedule="${LR_SCHEDULE:-constant}"
warmup_ratio="${WARMUP_RATIO:-0.0}"
weight_decay="${WEIGHT_DECAY:-1e-4}"
temperature="${TEMPERATURE:-1.0}"
top_k="${TOP_K:--1}"
top_p="${TOP_P:-1.0}"
min_p="${MIN_P:-0.0}"
max_tokens="${MAX_TOKENS:-2048}"            # response length (v2 traces are long)
max_model_len="${MAX_MODEL_LEN:-3072}"      # total context = prompt + max_tokens
wandb_name="${WANDB_NAME:-sels_rloo_v3_gs${group_size}_lr${curr_lr}}"

command=(
    modal run --detach "$PROJECT_ROOT/modal_train.py"
    rloo
    --model_name "$sels_ckpt"
    --ref_model_name "$sels_ckpt"
    --tokenizer_name "$tokenizer_name"
    --dataset_name "$dataset_name"
    --wandb_project "$wandb_project"
    --wandb_name "$wandb_name"
    --learning_rate "$curr_lr"
    --batch_size "$batch_size"
    --gradient_accumulation_steps "$gradient_accumulation_steps"
    --gradient_clipping "$gradient_clipping"
    --group_size "$group_size"
    --entropy_coefficient "$entropy_coefficient"
    --kl_divergence_coefficient "$kl_divergence_coefficient"
    --num_training_steps "$num_training_steps"
    --lr_schedule "$lr_schedule"
    --save_every_n_steps "$save_every_n_steps"
    --save_dir "$save_dir"
    --warmup_ratio "$warmup_ratio"
    --weight_decay "$weight_decay"
    --temperature "$temperature"
    --top_p "$top_p"
    --top_k "$top_k"
    --min_p "$min_p"
    --max_tokens "$max_tokens"
    --max_model_len "$max_model_len"
    --inject_state0
)

printf 'Executing command: '
printf '%q ' "${command[@]}"
printf '\n'
"${command[@]}"
