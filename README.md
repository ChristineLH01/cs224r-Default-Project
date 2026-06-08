# Countdown Post-Training: Structured Backtracking + Stacked Methods

Post-training of Qwen2.5-0.5B on the **Countdown** arithmetic task (reach a target
by combining given numbers with `+ - * /`). The project implements the full
post-training pipeline — SFT → IPO → RLOO → evaluation — and extends it along two
independent research lines:

- **Line A — SELS** (Structured Exploration with Latent States): teach explicit
  `<state_K>` / `<backtrack_J>` search structure via synthetic SFT, then refine it
  with RLOO + a structural process-reward model.
- **Line B — Stacked post-training**: three methods stacked on the default SFT
  checkpoint — **Arithmetic PRM → RFT → IPO anti-ramble** — each targeting a
  distinct failure axis (arithmetic mistakes, low pass@1, rambling).

Both lines train on `asingh15/countdown_tasks_3to4` and share the evaluation
harness. Training/eval run locally (CUDA GPU) or remotely on Modal via the shared
`modal_train.py` entrypoint.

## Task and scoring

Each example gives a target (e.g. `24`) and numbers (e.g. `[3, 4, 6, 8]`). The
model emits an equation in tags: `<answer>(8 - 4) * 6</answer>`.
`evaluation/countdown.py` scores it: `0.0` no answer, `0.1` answer found but
invalid/wrong, `1.0` valid equation using exactly the provided numbers and
equal to the target.

## Repository layout

```
modal_train.py                 # shared Modal app/entrypoint for all remote train+eval

sft_trainer/                   # SFT (base): train loop + dataset + launchers
  sft.py  sft_dataset.py  train_sft.sh  train_sft_modal.sh
  train_sft_sels.sh            #   [Line A] SFT on the synthetic SELS dataset

ipo_trainer/                   # IPO (base): preference-tuning train loop + dataset
  ipo.py  ipo_dataset.py  train_ipo.sh  train_ipo_modal.sh

rloo_trainer/                  # RLOO (base): orchestration + update/sampling workers
  rloo.py  rloo_update_worker.py  sampling_worker.py  rloo_dataset.py
  process_reward.py            #   [Line A] structural process-reward model (SELS grammar)
  context_restoration.py       #   [Line A] backtrack context manager for rollouts
  arithmetic_prm.py            #   [Line B] arithmetic process-reward model
  train_rloo_sels*.sh          #   [Line A] SELS RLOO (baseline + structural PRM)
  train_rloo_default*.sh       #   [Line B] default RLOO + arithmetic-PRM launchers

dataset_generation/            # [Line A] synthetic SELS SFT-data pipeline
  countdown_solver.py  skeleton_builder.py  narrate_with_gpt.py
  generate_sft_data.py  prompts/              # GPT narrator → HF SELS dataset

rft_extension/                 # [Line B] rejection-sampling fine-tuning
  synthesize_rft_data.py  train_rft.sh  train_rft_v3.sh  eval_*.sh  analyze_compare.py

ipo_extension/                 # [Line B] IPO anti-ramble preference tuning
  synthesize_ipo_pairs.py  train_ipo_anti_ramble.sh  train_ipo_anti_ramble_v2.sh

evaluation/
  countdown.py                 # scorer (answer extraction + validation + reward)
  countdown_eval.py            # batch vLLM eval (stops on </answer>)
  analyze_failure_modes.py     # CORRECT / NO-ANSWER / BAD-NUMS / WRONG-VAL decomposition
  compare_failure_modes.py     # cross-model comparison table
  sample_modal*.sh             # per-model Modal eval launchers
```

The reward models are wired into `rloo_trainer/rloo.py` behind coefficient flags
(`--structural_coef`, `--arithmetic_coef`). With both at `0.0` (the default) RLOO
training is identical to the base behavior; the reward models are pure additions.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -e ".[all]"     # includes modal, vllm, ray
```

Auth (set before launching anything):

```bash
export WANDB_API_KEY=...        # experiment tracking
export HF_TOKEN=...             # model/dataset access
export OPENAI_API_KEY=...       # only for Line A (SELS) data synthesis (GPT narrator)
```

Optional Modal (remote H100). After `modal setup`, the `*_modal.sh` / `sample_modal*.sh`
launchers package the repo and run on Modal. Defaults: `H100!` GPU, 24h timeout,
volume `default-proj-training`. Artifacts land under `/vol`:
`/vol/checkpoints/...` (checkpoints), `/vol/evaluation/eval_results` (eval JSON).
Download with `modal volume get default-proj-training <path> <dest>`.

Invoke tools from the project root via `.venv/bin/<tool>`. To change a run's
hyperparameters or checkpoint paths, edit the `.sh` launcher directly rather than
passing command-line overrides.

## Recreating the experiments

### Base pipeline (milestones)

```bash
bash sft_trainer/train_sft_modal.sh      # SFT
bash ipo_trainer/train_ipo_modal.sh      # IPO
bash rloo_trainer/train_rloo_modal.sh    # RLOO   (MODEL_NAME / DATASET_NAME via the launcher)
```

### Line A — SELS structured backtracking

```bash
.venv/bin/python dataset_generation/generate_sft_data.py   # → dataset_generation/output/sels_sft_v5 (needs OPENAI_API_KEY)
bash sft_trainer/train_sft_sels.sh                         # SFT warm-start on SELS data
bash rloo_trainer/train_rloo_sels.sh                       # RLOO baseline (no PRM)
bash rloo_trainer/train_rloo_sels_prm.sh                   # RLOO + structural PRM (--structural_coef 0.5)
```

The structural PRM (`rloo_trainer/process_reward.py`) scores each rollout against
the SELS grammar (valid state references, forward/backtrack transitions, depth
consistency, well-formed `<think>`/`<answer>`, no leaked special tokens). Its
self-test confirms the checks: `.venv/bin/python rloo_trainer/process_reward.py`
(all v5-SFT sample rows score 1.000).

### Line B — stacked post-training

A single warm-start chain; run the stages in order, each training from the
previous checkpoint.

| # | Model | Launcher | Warm-start from |
|---|-------|----------|-----------------|
| 0 | SFT base | (provided) `asingh15/qwen-sft-countdown-defaultproj` | — |
| 1 | Default RLOO | `rloo_trainer/train_rloo_default.sh` | SFT base |
| 2a | Arith PRM v1 | `rloo_trainer/train_rloo_default_arith_prm.sh` | SFT base |
| 2b | Arith PRM (cont) | `rloo_trainer/train_rloo_default_arith_prm_continued.sh` | Arith PRM v1 |
| 3 | RFT v1 | `rft_extension/train_rft.sh` | Arith PRM (cont) |
| 4 | RFT v3 | `rft_extension/train_rft_v3.sh` | Arith PRM (cont) |
| 5 | IPO v1 (headline) | `ipo_extension/train_ipo_anti_ramble.sh` | RFT v3 |

Data synthesis (run before the stage that consumes it):

```bash
# RFT pairs from correct rollouts (v3 = diversity-preserving; --top_k 1 reproduces v1)
.venv/bin/python rft_extension/synthesize_rft_data.py \
    --input evaluation/eval_results/train_rollouts_arithprm_continued_t1_n16.json \
    --output_dir evaluation/eval_results/rft_data_v3 --top_k_distinct 3

# IPO anti-ramble (chosen, rejected) pairs from RFT-v3 rollouts
.venv/bin/python -m ipo_extension.synthesize_ipo_pairs \
    --input evaluation/eval_results/train_rollouts_rft_v3_t1_n16.json \
    --output_dir evaluation/eval_results/ipo_anti_ramble_data \
    --top_k 2 --min_rambling_length 1200
```

Ablation: `ipo_extension/train_ipo_anti_ramble_v2.sh` reproduces IPO v2, which
validates the v1 design choice (engineered surface contrasts underperform).

### Evaluation (both lines)

Every model is scored identically — 50 test prompts × 16 rollouts, vLLM via
`countdown_eval.py` (temperature 0.6, top_p 0.95, max_tokens 1024, stop on
`</answer>`). Point the matching launcher at a checkpoint, then decompose and
compare:

```bash
bash evaluation/sample_modal_default_rloo.sh       # or sample_modal_default_arith_prm.sh / sample_modal_sels_rloo_prm.sh
.venv/bin/python evaluation/analyze_failure_modes.py   # per-model outcome decomposition
.venv/bin/python evaluation/compare_failure_modes.py   # cross-model table
```

## Dataset schemas

Loaders assume specific columns: **SFT** → `query`, `completion`; **IPO** →
`query`, `response_ws` (chosen), `response_ls` (rejected); **RLOO** → `prompt`,
`ground_truth` (dict with `target`, `numbers`); **eval** → `prompt`, `target`,
`nums`. Preprocess or update the loader if your dataset differs.

## Troubleshooting

- **OOM** — lower `batch_size` / `max_tokens` / `max_model_len` / `group_size`; raise gradient accumulation.
- **vLLM / Ray startup** — check `CUDA_VISIBLE_DEVICES`; ensure `vllm`/`ray` are installed.
- **W&B hangs** — verify `WANDB_API_KEY`; use `WANDB_MODE=offline` for local debugging.
- **Modal** — run `modal setup`; inspect with `modal volume ls default-proj-training /`.

Checkpoints and eval JSONs are regenerable and intentionally untracked (see `.gitignore`).
