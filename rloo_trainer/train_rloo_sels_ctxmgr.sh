#!/bin/bash
# SELS RLOO launcher WITH context-restoration manager (segmented sampling).
# Separate from train_rloo_sels.sh (baseline) and train_rloo_sels_prm.sh
# (PRM only) so each experiment has its own committed recipe.
#
# What this adds on top of train_rloo_sels_prm.sh:
#   --context_restoration  → vLLM sampling stops on <backtrack_K> in addition
#                            to </answer>. On each stop, the failed branch is
#                            truncated, a clean "<backtrack_K>\n<state_K>
#                            [list]\n" marker is appended, and vLLM is
#                            re-prompted with the cleaned prefix. Model never
#                            sees the failed branch in its conditioning
#                            window — see rloo_trainer/context_restoration.py.
#   --max_restoration_rounds 4  → hard cap on segments per completion.
#
# Default keeps PRM ON (structural_coef=0.5) on top of context restoration,
# so this is the "both components" cell of the proposal's 2x2 ablation. For
# the context-restoration-only ablation, set STRUCTURAL_COEF=0 in this file
# and copy to its own launcher (per feedback_one_launcher_per_setting).
#
# Expected cost: each backtrack adds one full re-prompt with cold KV cache.
# With ~1–2 backtracks per rollout on v5, ~2–3× the baseline vLLM sampling
# cost. Update step is unchanged.
#
# Required before running:
#   export WANDB_API_KEY=...   (HF_TOKEN if any gated repo)

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
sels_ckpt="/vol/checkpoints/sft_model/sels_sft/sels_sft_v5/best_model/model"
tokenizer_name="$sels_ckpt"
dataset_name="asingh15/countdown_tasks_3to4"
wandb_project="sels_rloo"
save_dir="/vol/checkpoints/rloo_checkpoints"

curr_lr="1e-5"
batch_size="128"
gradient_accumulation_steps="128"
gradient_clipping="0.0"
group_size="8"
num_training_steps="100"
kl_divergence_coefficient="0.001"
entropy_coefficient="0.001"
save_every_n_steps="25"
lr_schedule="constant"
warmup_ratio="0.0"
weight_decay="1e-4"
temperature="1.0"
top_k="-1"
top_p="1.0"
min_p="0.0"
max_tokens="1024"
max_model_len="2048"
structural_coef="0.5"
max_restoration_rounds="4"
wandb_name="sels_rloo_v5_gs${group_size}_lr${curr_lr}_kl${kl_divergence_coefficient}_t${max_tokens}_prm${structural_coef}_ctxmgr"

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
    --context_restoration
    --max_restoration_rounds "$max_restoration_rounds"
)

printf 'Executing command: '
printf '%q ' "${command[@]}"
printf '\n'
"${command[@]}"
