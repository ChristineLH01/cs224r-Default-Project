#!/bin/bash
# Anti-ramble IPO v2 — refined pair selection on top of RFT v3.
#
# Differences from v1:
#   - chosen pool also filtered: CORRECT + len <= 600 chars + n_arith_mistakes == 0
#   - This gives the model a SHORTER and ARITHMETICALLY CLEAN positive example
#     vs the long failure rejected, sharpening the contrast IPO learns.
#
# Warm-start: RFT v3 (same as v1).
# Loss: IPO (squared), beta=0.1 (same as v1).
#
# Prereqs (run from project root):
#   1) RFT v3 rollouts already sampled at T=1.0 n=16
#      (evaluation/eval_results/train_rollouts_rft_v3_t1_n16.json)
#   2) .venv/bin/python -m ipo_extension.synthesize_ipo_pairs \
#         --input evaluation/eval_results/train_rollouts_rft_v3_t1_n16.json \
#         --output_dir evaluation/eval_results/ipo_anti_ramble_data_v2 \
#         --top_k 2 --min_rambling_length 1200 \
#         --max_chosen_length 600 --clean_chosen
#   3) This script handles upload + train.

set -euo pipefail
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export MODAL_GPU="${MODAL_GPU:-H100!}"
export MODAL_VOLUME_NAME="${MODAL_VOLUME_NAME:-default-proj-training}"
export MODAL_TIMEOUT_SECONDS="${MODAL_TIMEOUT_SECONDS:-86400}"
export MODAL_STARTUP_TIMEOUT_SECONDS="${MODAL_STARTUP_TIMEOUT_SECONDS:-1800}"

model_path='/vol/checkpoints/rft/rft_v3/best_model/model'

local_data_dir="$PROJECT_ROOT/evaluation/eval_results/ipo_anti_ramble_data_v2"
remote_data_dir='/vol/sft_data/ipo_anti_ramble_data_v2'

if [[ ! -d "$local_data_dir" ]]; then
    echo "ERROR: $local_data_dir not found." >&2
    echo "  Run the synth command from the header of this script first." >&2
    exit 1
fi

echo "Uploading $local_data_dir -> $MODAL_VOLUME_NAME:/sft_data/ipo_anti_ramble_data_v2"
"$PROJECT_ROOT/.venv/bin/modal" volume put "$MODAL_VOLUME_NAME" \
    "$local_data_dir" /sft_data/ --force

wandb_project='ipo_anti_ramble'
wandb_name='ipo_v2'

num_epochs='1'
batch_size='16'
gradient_accumulation_steps='4'
learning_rate='5e-6'
warmup_ratio='0.05'
weight_decay='0.01'
max_prompt_length='384'
max_response_length='1024'
beta='0.1'
loss_type='ipo'

command=(
    modal run --detach "$PROJECT_ROOT/modal_train.py"
    ipo_anti_ramble
    --model_name "$model_path"
    --dataset_name "$remote_data_dir"
    --output_dir "/vol/checkpoints"
    --wandb_project "$wandb_project"
    --wandb_name "$wandb_name"
    --num_epochs "$num_epochs"
    --batch_size "$batch_size"
    --gradient_accumulation_steps "$gradient_accumulation_steps"
    --learning_rate "$learning_rate"
    --warmup_ratio "$warmup_ratio"
    --weight_decay "$weight_decay"
    --max_prompt_length "$max_prompt_length"
    --max_response_length "$max_response_length"
    --beta "$beta"
    --loss_type "$loss_type"
)

printf 'Executing command: '
printf '%q ' "${command[@]}"
printf '\n'
"${command[@]}"
