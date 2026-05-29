#!/bin/bash
# Sample 6000 ADDITIONAL train-split rollouts (prompts 2000-7999) from
# cont. latest, for the v2 RFT dataset. Combined with the existing p1
# (2000 prompts at start_idx 0) this brings the total source pool to 8000
# prompts × 8 rollouts = 64K rollouts.
#
# Same model/sampling params as the original p1 sample so the two files are
# directly mergeable in synthesize_rft_data.py.

set -euo pipefail
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export MODAL_GPU="${MODAL_GPU:-H100!}"
export MODAL_VOLUME_NAME="${MODAL_VOLUME_NAME:-default-proj-training}"
export MODAL_TIMEOUT_SECONDS="${MODAL_TIMEOUT_SECONDS:-86400}"
export MODAL_STARTUP_TIMEOUT_SECONDS="${MODAL_STARTUP_TIMEOUT_SECONDS:-1800}"

model_path='/vol/checkpoints/rloo_checkpoints/rloo_training/default_rloo_qwen_sft_gs8_lr1e-5_kl0.001_t1024_arithprm1.0_continued/latest_checkpoint/model'
output_name='train_rollouts_arithprm_continued_p2'

command=(
    modal run --detach "$PROJECT_ROOT/modal_train.py"
    sample_train
    --model_path "$model_path"
    --output_name "$output_name"
    --num_responses 8
    --max_prompts 6000
    --start_idx 2000
)

printf 'Running command: '
printf '%q ' "${command[@]}"
printf '\n'
"${command[@]}"
