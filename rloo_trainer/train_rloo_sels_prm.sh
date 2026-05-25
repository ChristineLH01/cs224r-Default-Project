#!/bin/bash
# SELS RLOO launcher WITH structural PRM. Separate from train_rloo_sels.sh
# (baseline, no PRM) so each experiment has its own committed recipe.
#
# Total reward per rollout = compute_score(final) + α · structural_PRM(trace),
# where α = STRUCTURAL_COEF. Calibrated at α = 0.5 — see project_sels_rloo_plan
# for the σ-match derivation. α ≤ 1 preserves "correctness dominance" (every
# correct rollout outscores every wrong rollout regardless of structure).
#
# Required before running:
#   export WANDB_API_KEY=...   (HF_TOKEN if any gated repo)
#
# Hyperparameters are identical to train_rloo_sels.sh except for the added
# `--structural_coef "$structural_coef"` flag, so a clean A/B against the
# baseline `sels_rloo_v5_gs8_lr1e-5_kl0.001_t1024` run on W&B.

set -euo pipefail
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export MODAL_GPU="${MODAL_GPU:-H100!}"
export MODAL_VOLUME_NAME="${MODAL_VOLUME_NAME:-default-proj-training}"
export MODAL_TIMEOUT_SECONDS="${MODAL_TIMEOUT_SECONDS:-86400}"
export MODAL_STARTUP_TIMEOUT_SECONDS="${MODAL_STARTUP_TIMEOUT_SECONDS:-1800}"

# SELS warm-start: the v5 checkpoint carries the 10 registered tokens; the
# trainer tokenizer MUST match it (so <state_*>/<backtrack_*> tokenize atomically).
sels_ckpt="${MODEL_NAME:-/vol/checkpoints/sft_model/sels_sft/sels_sft_v5/best_model/model}"
tokenizer_name="${TOKENIZER_NAME:-$sels_ckpt}"
dataset_name="${DATASET_NAME:-asingh15/countdown_tasks_3to4}"
wandb_project="${WANDB_PROJECT:-sels_rloo}"
save_dir="${SAVE_DIR:-/vol/checkpoints/rloo_checkpoints}"

curr_lr="${LR:-1e-5}"
batch_size="${BATCH_SIZE:-128}"
gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-128}"
gradient_clipping="${GRADIENT_CLIPPING:-0.0}"
group_size="${GROUP_SIZE:-8}"
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
max_tokens="${MAX_TOKENS:-1024}"
max_model_len="${MAX_MODEL_LEN:-2048}"
structural_coef="${STRUCTURAL_COEF:-0.5}"
wandb_name="${WANDB_NAME:-sels_rloo_v5_gs${group_size}_lr${curr_lr}_kl${kl_divergence_coefficient}_t${max_tokens}_prm${structural_coef}_v2}"

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
    --structural_coef "$structural_coef"
)

printf 'Executing command: '
printf '%q ' "${command[@]}"
printf '\n'
"${command[@]}"
