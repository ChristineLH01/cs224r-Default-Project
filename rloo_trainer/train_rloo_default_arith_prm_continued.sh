#!/bin/bash
# CONTINUATION of the v1 arithmetic PRM run, starting from its final
# checkpoint (latest_checkpoint = step 99 of the original 100-step v1 run).
# Trains for another 100 steps with the same unconditional arithmetic PRM
# design.
#
# v1 mechanism (per evaluation/failure_modes_arithprm_v1.md):
#   - Unconditional arithmetic PRM cut mean n_wrong from 0.89 → 0.55 (-37%)
#   - But induced verification-tax: n_repeats went UP, NO-ANSWER spiked,
#     pass@1 regressed from 0.580 → 0.460
#
# This continuation extends v1's training to 200 total steps with the same
# reward signal. Decisiveness reward DROPPED (true addressable headroom
# was 2.2%, see evaluation/methods_retrospective.md).
#
# This run produces the RLOO foundation for the restart-on-rambling SFT
# stage (see evaluation/restart_on_rambling_plan.md). The verification-tax
# behavior (rambling, NO-ANSWER) is INTENDED to be addressed downstream by
# the SFT stage, not by reward shaping.
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

# Continue from the v1 arith PRM run's final checkpoint.
# (No epoch_0_step_100 was saved — the training loop breaks at
#  global_step == num_training_steps before that step's save fires.
#  latest_checkpoint = step 99, the last update applied.)
v1_final_ckpt='/vol/checkpoints/rloo_checkpoints/rloo_training/default_rloo_qwen_sft_gs8_lr1e-5_kl0.001_t1024_arithprm1.0/latest_checkpoint/model'

# Reference policy: SAME as v1's KL anchor (the original SFT base).
# Keeps the KL signal consistent across the original 100 + continuation 100 steps.
default_sft='asingh15/qwen-sft-countdown-defaultproj'

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
wandb_name="default_rloo_qwen_sft_gs${group_size}_lr${curr_lr}_kl${kl_divergence_coefficient}_t${max_tokens}_arithprm${arithmetic_coef}_continued"

command=(
    modal run --detach "$PROJECT_ROOT/modal_train.py"
    rloo
    --model_name "$v1_final_ckpt"
    --ref_model_name "$default_sft"
    --tokenizer_name "$default_sft"
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
