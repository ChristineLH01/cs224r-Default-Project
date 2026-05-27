#!/bin/bash
# Default-substrate RLOO + arithmetic PRM (Design 1: negative-only with
# repetition kicker, capped). Mirrors train_rloo_default.sh exactly except
# for the added `--arithmetic_coef` flag, so this is a clean compute-
# equivalent A/B against the outcome-only default RLOO run (pass@1 = 0.580
# at step 99 in W&B run default_rloo_qwen_sft_gs8_lr1e-5_kl0.001_t1024).
#
# PRM design + calibration: see rloo_trainer/arithmetic_prm.py
#   penalty = β·n_total_mistakes + γ·n_repeated_mistakes,  prm = -min(penalty, cap)
#   β=0.1, γ=0.2, cap=0.5  →  prm ∈ [-0.5, 0]
# arithmetic_coef = 1.0 → max contribution is -0.5 added to compute_score.
#
# Required before running:
#   export WANDB_API_KEY=...

set -euo pipefail
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export MODAL_GPU="${MODAL_GPU:-H100!}"
export MODAL_VOLUME_NAME="${MODAL_VOLUME_NAME:-default-proj-training}"
export MODAL_TIMEOUT_SECONDS="${MODAL_TIMEOUT_SECONDS:-86400}"
export MODAL_STARTUP_TIMEOUT_SECONDS="${MODAL_STARTUP_TIMEOUT_SECONDS:-1800}"

default_ckpt='asingh15/qwen-sft-countdown-defaultproj'
tokenizer_name="$default_ckpt"
dataset_name='asingh15/countdown_tasks_3to4'
wandb_project='rloo_training'
save_dir='/vol/checkpoints/rloo_checkpoints'

curr_lr='1e-5'
batch_size='128'
gradient_accumulation_steps='128'
gradient_clipping='0.0'
group_size='8'
num_training_steps='100'
kl_divergence_coefficient='0.001'
entropy_coefficient='0.001'
save_every_n_steps='25'
lr_schedule='constant'
warmup_ratio='0.0'
weight_decay='1e-4'
temperature='1.0'
top_k='-1'
top_p='1.0'
min_p='0.0'
max_tokens='1024'
max_model_len='2048'
arithmetic_coef='1.0'
wandb_name="default_rloo_qwen_sft_gs${group_size}_lr${curr_lr}_kl${kl_divergence_coefficient}_t${max_tokens}_arithprm${arithmetic_coef}"

command=(
    modal run --detach "$PROJECT_ROOT/modal_train.py"
    rloo
    --model_name "$default_ckpt"
    --ref_model_name "$default_ckpt"
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
    --arithmetic_coef "$arithmetic_coef"
)

printf 'Executing command: '
printf '%q ' "${command[@]}"
printf '\n'
"${command[@]}"
