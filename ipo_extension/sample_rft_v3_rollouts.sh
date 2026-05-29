#!/bin/bash
# Sample rollouts from RFT v3 for the anti-ramble IPO data.
#
# Why this exists: IPO's reference policy is the warm-start model (RFT v3).
# The (chosen, rejected) pairs we train on should reflect RFT v3's OWN
# failure modes — sampling from cont. latest would put us in a different
# distribution than the reference, weakening the IPO signal.
#
# Setup mirrors v3 sampling: T=1.0, n=16, idx 0-1999. Output JSONL goes to
# /vol/sft_data/train_rollouts_rft_v3_t1_n16.json.

set -euo pipefail
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export MODAL_GPU="${MODAL_GPU:-H100!}"
export MODAL_VOLUME_NAME="${MODAL_VOLUME_NAME:-default-proj-training}"
export MODAL_TIMEOUT_SECONDS="${MODAL_TIMEOUT_SECONDS:-86400}"
export MODAL_STARTUP_TIMEOUT_SECONDS="${MODAL_STARTUP_TIMEOUT_SECONDS:-1800}"

model_path='/vol/checkpoints/rft/rft_v3/best_model/model'
output_name='train_rollouts_rft_v3_t1_n16'

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
