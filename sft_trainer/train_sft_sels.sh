#!/bin/bash
# Launch SELS SFT training on Modal. Wraps train_sft_modal.sh with the
# SELS-specific dataset path and W&B naming so the recipe is reproducible.
#
# Required before running:
#   export WANDB_API_KEY=...
#   export HF_TOKEN=...
#
# Hyperparameters intentionally match the default-project baseline (cog_behav)
# so the SELS vs default comparison is apples-to-apples — only the dataset and
# the 10 special-token registration differ.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# v3 = chain-heavy warm-start (recommended). Override DATASET_NAME/WANDB_NAME_PREFIX
# to sels_sft_v2 to train the full-structure variant for comparison.
export DATASET_NAME="${DATASET_NAME:-dataset_generation/output/sels_sft_v3}"
export WANDB_PROJECT="${WANDB_PROJECT:-sels_sft}"
export WANDB_NAME_PREFIX="${WANDB_NAME_PREFIX:-sels_sft_v3}"
export LRS="${LRS:-5e-5}"
export EPOCHS="${EPOCHS:-6}"
# SELS extension: register the 10 <state_*>/<backtrack_*> tokens (default SFT leaves this off).
export REGISTER_SELS_TOKENS="${REGISTER_SELS_TOKENS:-1}"

# Everything else (batch_size=64, gradient_accumulation_steps=8,
# warmup_ratio=0.05, weight_decay=0.01, max_prompt_length=512,
# max_response_length=1024, Qwen/Qwen2.5-0.5B, H100 GPU) inherits from
# train_sft_modal.sh defaults.

exec "$SCRIPT_DIR/train_sft_modal.sh"
