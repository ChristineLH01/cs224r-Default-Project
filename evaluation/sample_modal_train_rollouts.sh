#!/bin/bash
# Stage 2, Step 1: sample N rollouts per training prompt for SFT data generation.
# Output goes to /vol/sft_data/ on the Modal volume.
#
# Default budget: 8 rollouts × 2000 prompts = 16,000 rollouts (~5-10 min H100).
# 8 samples per prompt favors prompt diversity over redundant pairings —
# Stage 2 SFT needs ONE (CORRECT, rambling) pair per prompt, not many.

set -euo pipefail
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export MODAL_GPU="${MODAL_GPU:-H100!}"
export MODAL_VOLUME_NAME="${MODAL_VOLUME_NAME:-default-proj-training}"
export MODAL_TIMEOUT_SECONDS="${MODAL_TIMEOUT_SECONDS:-86400}"
export MODAL_STARTUP_TIMEOUT_SECONDS="${MODAL_STARTUP_TIMEOUT_SECONDS:-1800}"

# Warm-start checkpoint: cont. latest (continuation step 199; pass@1 = 0.560,
# pass@16 = 0.760 — best test pass@16 in the project).
model_path='/vol/checkpoints/rloo_checkpoints/rloo_training/default_rloo_qwen_sft_gs8_lr1e-5_kl0.001_t1024_arithprm1.0_continued/latest_checkpoint/model'
output_name='train_rollouts_arithprm_continued'

command=(
    modal run --detach "$PROJECT_ROOT/modal_train.py"
    sample_train
    --model_path "$model_path"
    --output_name "$output_name"
    --num_responses 8
    --max_prompts 2000
)

printf 'Running command: '
printf '%q ' "${command[@]}"
printf '\n'
"${command[@]}"
