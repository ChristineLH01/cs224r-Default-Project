#!/bin/bash
# Anti-ramble IPO training, stacked on top of RFT v3.
#
# Warm-start AND reference policy: RFT v3 (best test pass@1 so far; the model
# has its own characteristic rambling failure mode that we now target).
#
# Data: pairwise preference pairs from RFT v3's OWN rollouts at T=1.0 / n=16.
# Sampling from the warm-start model itself keeps the rejected examples in the
# reference policy's actual distribution — IPO's signal is strongest when
# (chosen, rejected) reflect what the model currently does.
#
# Pair selection (synthesize_ipo_pairs.py):
#   chosen:   compute_score == 1.0 (CORRECT, any length)
#   rejected: NOT correct AND length >= 1200 chars (rambling failure)
#   Random K=2 distinct from each pool per prompt; pair index-by-index.
#
# Prereqs (run from project root):
#   1) bash ipo_extension/sample_rft_v3_rollouts.sh
#      .venv/bin/modal volume get default-proj-training \
#        /sft_data/train_rollouts_rft_v3_t1_n16.json \
#        evaluation/eval_results/train_rollouts_rft_v3_t1_n16.json --force
#   2) .venv/bin/python -m ipo_extension.synthesize_ipo_pairs \
#        --input evaluation/eval_results/train_rollouts_rft_v3_t1_n16.json \
#        --output_dir evaluation/eval_results/ipo_anti_ramble_data \
#        --top_k 2 --min_rambling_length 1200
#   3) this script handles upload + train.

set -euo pipefail
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export MODAL_GPU="${MODAL_GPU:-H100!}"
export MODAL_VOLUME_NAME="${MODAL_VOLUME_NAME:-default-proj-training}"
export MODAL_TIMEOUT_SECONDS="${MODAL_TIMEOUT_SECONDS:-86400}"
export MODAL_STARTUP_TIMEOUT_SECONDS="${MODAL_STARTUP_TIMEOUT_SECONDS:-1800}"

# Warm-start = RFT v3 (best pass@1 baseline; same model used to generate the pairs).
model_path='/vol/checkpoints/rft/rft_v3/best_model/model'

local_data_dir="$PROJECT_ROOT/evaluation/eval_results/ipo_anti_ramble_data"
remote_data_dir='/vol/sft_data/ipo_anti_ramble_data'

if [[ ! -d "$local_data_dir" ]]; then
    echo "ERROR: $local_data_dir not found." >&2
    echo "  Run the prereq commands from the header of this script first." >&2
    exit 1
fi

echo "Uploading $local_data_dir -> $MODAL_VOLUME_NAME:/sft_data/ipo_anti_ramble_data"
"$PROJECT_ROOT/.venv/bin/modal" volume put "$MODAL_VOLUME_NAME" \
    "$local_data_dir" /sft_data/ --force

wandb_project='ipo_anti_ramble'
wandb_name='ipo_v1'

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
