# Extension: Stacked Post-Training + Structured Backtracking on Countdown

This document is the entry point for the **project extension**. It explains what
each piece of extension code does and gives the exact script sequence to
recreate every reported result. For environment setup, authentication, and Modal
configuration, see the base [`README.md`](README.md) (sections 6.1–6.3) — the
extension reuses that same environment, the shared `modal_train.py` entrypoint,
and the unmodified `evaluation/countdown.py` scorer.

The extension has **two independent lines of work** that share the Countdown task
and evaluation harness:

- **Line A — Stacked post-training on the default substrate** (the headline
  report). Three methods stacked on top of the provided SFT checkpoint:
  Arithmetic PRM → RFT → IPO anti-ramble. Documented in
  [`evaluation/report.md`](evaluation/report.md).
- **Line B — SELS (Structured Exploration with Latent States)**. A separate
  thread that teaches explicit `<state_K>` / `<backtrack_J>` search structure via
  synthetic SFT, then refines it with RLOO + a structural process-reward model.

Both lines train Qwen2.5-0.5B and evaluate on `asingh15/countdown_tasks_3to4`.

---

## 1. Repository map (extension code)

```
dataset_generation/            # Line B: synthetic SELS SFT data pipeline
  countdown_solver.py          #   exhaustive Countdown solver (search ground truth)
  skeleton_builder.py          #   turns a solve path into a <state_K>/<backtrack_J> skeleton
  narrate_with_gpt.py          #   GPT narration of skeletons into natural traces
  generate_sft_data.py         #   end-to-end driver → HF dataset (sels_sft_v5)
  prompts/narrator_fewshot.txt #   few-shot prompt for the narrator
  README.md                    #   data-format spec and invariants

rft_extension/                 # Line A: rejection-sampling fine-tuning (RFT)
  synthesize_rft_data.py       #   build (query, completion) pairs from correct rollouts
  sample_rft_train_rollouts_v3.sh  # sample rollouts to mine for RFT v3
  train_rft.sh / train_rft_v3.sh   # RFT v1 (length-biased) / v3 (diversity-preserving)
  eval_rft*.sh, eval_compare_4models.sh, analyze_compare.py  # eval + analysis

ipo_extension/                 # Line A: IPO anti-ramble preference tuning
  synthesize_ipo_pairs.py      #   build (chosen, rejected) anti-ramble preference pairs
  ipo_dataset_local.py         #   local IPO dataset adapter
  train_ipo_anti_ramble.sh     #   IPO v1 (headline)
  train_ipo_anti_ramble_v2.sh  #   IPO v2 (ablation; see report §6)
  eval_ipo_anti_ramble*.sh

rloo_trainer/                  # RLOO core (starter) + extension reward models
  arithmetic_prm.py            # Line A: arithmetic process-reward model
  process_reward.py            # Line B: structural process-reward model (SELS)
  context_restoration.py       # Line B: backtrack context manager for RLOO rollouts
  train_rloo_default*.sh       # Line A: default RLOO + arithmetic PRM launchers
  train_rloo_sels*.sh          # Line B: SELS RLOO (baseline + structural PRM) launchers

sft_trainer/
  train_sft_sels.sh            # Line B: SFT on the synthetic SELS dataset

evaluation/
  countdown.py                 # scorer (starter, unmodified)
  countdown_eval.py            # batch vLLM eval (stops on </answer>)
  analyze_failure_modes.py     # CORRECT/NO-ANSWER/BAD-NUMS/WRONG-VAL decomposition
  compare_failure_modes.py     # cross-model failure-mode comparison
  sample_modal_*.sh            # per-model Modal eval launchers
  report.md                    # the detailed results report (Line A)
  view_passk.ipynb             # pass@k / loss-curve notebook
```

`process_reward.py` and `arithmetic_prm.py` are wired into `rloo_trainer/rloo.py`
behind coefficient flags (`--structural_coef`, `--arithmetic_coef`). When both
coefficients are `0.0` (the default), RLOO training is byte-identical to the
starter behavior; the reward models are pure additions.

---

## 2. Line A — Stacked post-training (recreates `report.md`)

All five reported models form a single warm-start chain. Each stage trains from
the previous stage's checkpoint, so run them in order. Every launcher is a thin
wrapper over `modal_train.py`; edit the launcher (not the command line) to change
hyperparameters or checkpoint paths.

| # | Model | Launcher | Warm-start from |
|---|-------|----------|-----------------|
| 0 | SFT base | (provided) `asingh15/qwen-sft-countdown-defaultproj` | — |
| 1 | Default RLOO | `rloo_trainer/train_rloo_default.sh` | SFT base |
| 2a | Arith PRM v1 | `rloo_trainer/train_rloo_default_arith_prm.sh` | SFT base |
| 2b | Arith PRM (cont) | `rloo_trainer/train_rloo_default_arith_prm_continued.sh` | Arith PRM v1 ckpt |
| 3 | RFT v1 | `rft_extension/train_rft.sh` | Arith PRM (cont) |
| 4 | RFT v3 | `rft_extension/train_rft_v3.sh` | Arith PRM (cont) |
| 5 | IPO v1 (headline) | `ipo_extension/train_ipo_anti_ramble.sh` | RFT v3 |

Data-synthesis steps (run before the stage that consumes them):

- **RFT data** — sample rollouts from the warm-start model
  (`rft_extension/sample_rft_train_rollouts_v3.sh`), then build pairs:
  ```bash
  .venv/bin/python rft_extension/synthesize_rft_data.py \
      --input evaluation/eval_results/train_rollouts_arithprm_continued_t1_n16.json \
      --output_dir evaluation/eval_results/rft_data_v3 \
      --top_k_distinct 3          # v3 = diversity-preserving; --top_k 1 reproduces v1
  ```
- **IPO pairs** — sample rollouts from RFT v3
  (`ipo_extension/sample_rft_v3_rollouts.sh`), then build anti-ramble
  (chosen, rejected) preference pairs:
  ```bash
  .venv/bin/python -m ipo_extension.synthesize_ipo_pairs \
      --input evaluation/eval_results/train_rollouts_rft_v3_t1_n16.json \
      --output_dir evaluation/eval_results/ipo_anti_ramble_data \
      --top_k 2 --min_rambling_length 1200
  ```

**Ablation (report §6):** `ipo_extension/train_ipo_anti_ramble_v2.sh` reproduces
IPO v2 (engineered surface contrasts) to validate the v1 design choice.

---

## 3. Line B — SELS structured backtracking

| Step | Command | Output |
|------|---------|--------|
| 1. Synthesize data | `.venv/bin/python dataset_generation/generate_sft_data.py` (needs `OPENAI_API_KEY` — the narrator uses GPT) | `dataset_generation/output/sels_sft_v5` |
| 2. SFT warm-start | `bash sft_trainer/train_sft_sels.sh` | `sels_sft_v5/best_model` |
| 3. RLOO baseline | `bash rloo_trainer/train_rloo_sels.sh` | RLOO, no PRM |
| 4. RLOO + structural PRM | `bash rloo_trainer/train_rloo_sels_prm.sh` | RLOO, `--structural_coef 0.5` |

The structural PRM (`rloo_trainer/process_reward.py`) scores each rollout's search
trace against the SELS grammar (valid state references, forward/backtrack
transitions, depth consistency, well-formed `<think>`/`<answer>`, no leaked special
tokens). Run its self-test to confirm the grammar checks:

```bash
.venv/bin/python rloo_trainer/process_reward.py    # all v5-SFT sample rows score 1.000
```

The SELS data format and invariants are specified in
[`dataset_generation/README.md`](dataset_generation/README.md).

---

## 4. Evaluation (both lines)

Every model is scored identically: 50 test prompts × 16 rollouts, vLLM via
`evaluation/countdown_eval.py` (temperature 0.6, top_p 0.95, max_tokens 1024,
stopping on `</answer>`). Launch a model's eval by pointing the relevant
`evaluation/sample_modal_*.sh` at its checkpoint:

- `sample_modal_default_rloo.sh` — default RLOO baseline
- `sample_modal_default_arith_prm.sh` — arithmetic-PRM models
- `sample_modal_sels_rloo_prm.sh` — SELS models
- `sample_modal.sh` — generic per-checkpoint eval

Then decompose outcomes and compare across models:

```bash
.venv/bin/python evaluation/analyze_failure_modes.py   # per-model CORRECT/NO-ANSWER/BAD-NUMS/WRONG-VAL
.venv/bin/python evaluation/compare_failure_modes.py   # cross-model table
```

The headline numbers these reproduce are in [`evaluation/report.md`](evaluation/report.md) §4.

---

## 5. Reproduction checklist

1. Set up the environment and auth per [`README.md`](README.md) §6 (`pip install -e ".[all]"`, `WANDB_API_KEY`, `HF_TOKEN`, optional `modal setup`).
2. **Line A:** run §2 stages 1 → 5 in order (each warm-starts from the prior checkpoint); synthesize RFT/IPO data at the indicated points.
3. **Line B:** run §3 steps 1 → 4.
4. **Evaluate** every checkpoint with §4 and compare against `evaluation/report.md`.

Checkpoints land on the Modal volume under `/vol/checkpoints/...`; eval JSONs land
under `/vol/evaluation/eval_results`. Both are regenerable and intentionally not
tracked in git (see `.gitignore`).
