#!/bin/bash
# v3 sampling for canonical RFT: high-diversity rollouts on the same prompt
# pool as v1 (idx 0-1999) so the comparison isolates the diversity-preservation
# methodology.
#
# Knobs that differ from v1/v2 sampling:
#   temperature 0.6 -> 1.0   (sampling diversity)
#   num_responses  8  -> 16  (more shots per prompt, more chances at distinct traces)
#
# Cost: 2000 prompts * 16 rollouts = 32K rollouts ~30 min H100.

set -euo pipefail
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export MODAL_GPU="${MODAL_GPU:-H100!}"
export MODAL_VOLUME_NAME="${MODAL_VOLUME_NAME:-default-proj-training}"
export MODAL_TIMEOUT_SECONDS="${MODAL_TIMEOUT_SECONDS:-86400}"
export MODAL_STARTUP_TIMEOUT_SECONDS="${MODAL_STARTUP_TIMEOUT_SECONDS:-1800}"

model_path='/vol/checkpoints/rloo_checkpoints/rloo_training/default_rloo_qwen_sft_gs8_lr1e-5_kl0.001_t1024_arithprm1.0_continued/latest_checkpoint/model'
output_name='train_rollouts_arithprm_continued_t1_n16'

command=(
    modal run --detach "$PROJECT_ROOT/modal_train.py"
    sample_train
    --model_path "$model_path"
    --output_name "$output_name"
    --num_responses 16
    --max_prompts 2000
    --start_idx 0
    --temperature 1.0
)

printf 'Running command: '
printf '%q ' "${command[@]}"
printf '\n'
"${command[@]}"
