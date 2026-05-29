#!/bin/bash
# Bigger eval comparing 4 models on a held-out slice of the train split
# (idx 8000-8199 — held out from RFT synthesis; previously seen by the RLOO
# models during RL training, which gives a fair apples-to-apples comparison
# of what RFT adds on top of cont. latest).
#
# Models compared:
#   1) Default RLOO        — bare RLOO baseline, no PRM
#   2) Arith PRM (cont)    — cont. latest; the RFT warm-start
#   3) RFT v1              — 1.2K examples, top_k=1
#   4) RFT v2              — 9.1K examples, top_k=2
#
# Sampling: 200 prompts × 16 rollouts × 4 models = 12.8K rollouts per model.
# Dispatched in parallel via `&` + `wait`. Each modal app uses one H100;
# total wall-clock ≈ slowest of the 4 (~15-20 min).

set -euo pipefail
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export MODAL_GPU="${MODAL_GPU:-H100!}"
export MODAL_VOLUME_NAME="${MODAL_VOLUME_NAME:-default-proj-training}"
export MODAL_TIMEOUT_SECONDS="${MODAL_TIMEOUT_SECONDS:-86400}"
export MODAL_STARTUP_TIMEOUT_SECONDS="${MODAL_STARTUP_TIMEOUT_SECONDS:-1800}"

start_idx='8000'
max_prompts='200'
num_responses='16'

# Output_name: prepended to /vol/sft_data/ by sample_train_rollouts.py.
default_rloo_path='/vol/checkpoints/rloo_checkpoints/rloo_training/default_rloo_qwen_sft_gs8_lr1e-5_kl0.001_t1024/latest_checkpoint/model'
arith_prm_path='/vol/checkpoints/rloo_checkpoints/rloo_training/default_rloo_qwen_sft_gs8_lr1e-5_kl0.001_t1024_arithprm1.0_continued/latest_checkpoint/model'
rft_v1_path='/vol/checkpoints/rft/rft_v1/best_model/model'
rft_v2_path='/vol/checkpoints/rft/rft_v2/best_model/model'

dispatch() {
    local model_path="$1"
    local output_name="$2"
    echo "Dispatching $output_name -> $model_path"
    modal run --detach "$PROJECT_ROOT/modal_train.py" \
        sample_train \
        --model_path "$model_path" \
        --output_name "$output_name" \
        --num_responses "$num_responses" \
        --max_prompts "$max_prompts" \
        --start_idx "$start_idx"
}

dispatch "$default_rloo_path" 'compare_default_rloo'   &
dispatch "$arith_prm_path"    'compare_arith_prm_cont' &
dispatch "$rft_v1_path"       'compare_rft_v1'         &
dispatch "$rft_v2_path"       'compare_rft_v2'         &

wait
echo "All 4 eval jobs completed."
