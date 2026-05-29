#!/bin/bash
# Rejection-sampling Fine-Tuning (RFT) launcher.
#
# Distills cont. latest's pass@8 into pass@1: for each train prompt, take the
# shortest CORRECT rollout (commit-first bias) and SFT the warm-start model
# on those (query, completion) pairs. Reuses sft_trainer/sft.py unchanged.
#
# Prereqs (run from project root):
#   .venv/bin/python -m rft_extension.synthesize_rft_data
#   ^^ produces evaluation/eval_results/rft_data/  (HF DatasetDict)

set -euo pipefail
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export MODAL_GPU="${MODAL_GPU:-H100!}"
export MODAL_VOLUME_NAME="${MODAL_VOLUME_NAME:-default-proj-training}"
export MODAL_TIMEOUT_SECONDS="${MODAL_TIMEOUT_SECONDS:-86400}"
export MODAL_STARTUP_TIMEOUT_SECONDS="${MODAL_STARTUP_TIMEOUT_SECONDS:-1800}"

# Warm-start: cont. latest (RLOO arith-PRM continued; best pass@16 in project).
model_path='/vol/checkpoints/rloo_checkpoints/rloo_training/default_rloo_qwen_sft_gs8_lr1e-5_kl0.001_t1024_arithprm1.0_continued/latest_checkpoint/model'

local_data_dir="$PROJECT_ROOT/evaluation/eval_results/rft_data"
remote_data_dir='/vol/sft_data/rft_data'

if [[ ! -d "$local_data_dir" ]]; then
    echo "ERROR: $local_data_dir not found." >&2
    echo "  Run: .venv/bin/python -m rft_extension.synthesize_rft_data" >&2
    exit 1
fi

# Upload the HF DatasetDict dir to the modal volume (idempotent).
echo "Uploading $local_data_dir -> $MODAL_VOLUME_NAME:/sft_data/rft_data"
"$PROJECT_ROOT/.venv/bin/modal" volume put "$MODAL_VOLUME_NAME" \
    "$local_data_dir" /sft_data/ --force

wandb_project='rft'
wandb_name='rft_v1'

num_epochs='1'
batch_size='16'
gradient_accumulation_steps='4'
learning_rate='5e-6'
warmup_ratio='0.05'
weight_decay='0.01'
max_prompt_length='384'
max_response_length='1024'

command=(
    modal run --detach "$PROJECT_ROOT/modal_train.py"
    sft
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
)

printf 'Executing command: '
printf '%q ' "${command[@]}"
printf '\n'
"${command[@]}"
