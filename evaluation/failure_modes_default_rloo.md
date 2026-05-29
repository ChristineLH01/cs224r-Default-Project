# Default RLOO — Failure-mode analysis

Diagnostic study of the trained default-RLOO Countdown model's rollouts on the
held-out test set, conducted before designing the arithmetic-PRM extension.
Goal: identify *which* failures an arithmetic-PRM can plausibly fix and what
the ceiling impact looks like.

## Setup

- **Checkpoint:** `/vol/checkpoints/rloo_checkpoints/rloo_training/default_rloo_qwen_sft_gs8_lr1e-5_kl0.001_t1024/latest_checkpoint/model` (Qwen SFT base, RLOO-trained 100 steps).
- **W&B run:** `ruichu-xia-stanford-university/rloo_training/default_rloo_qwen_sft_gs8_lr1e-5_kl0.001_t1024` (final `sampling/reward_mean = 0.518` at step 99).
- **Eval:** `evaluation/countdown_eval.py` on `asingh15/countdown_tasks_3to4` test split, 50 prompts × 16 samples = **800 rollouts**, temperature 0.6, top_p 0.95, top_k 20, `max_tokens=1024`, `stop=["</answer>"]`.
- **Launcher:** `evaluation/sample_modal_default_rloo.sh`.
- **Output:** `evaluation/eval_results/default_rloo_qwen_sft.json` (gitignored; on Modal volume).

## Headline numbers

| metric | value |
|---|---|
| pass@1 | **0.580** |
| pass@16 | 0.720 |
| mean per-rollout score | 0.587 |

Pass@1 substantially exceeds SELS-RLOO+PRM (0.322 at step 99 in W&B run `sels_rloo_v5_gs8_lr1e-5_kl0.001_t1024_prm0.5_v2`); the structural-extension substrate clearly underperformed default SFT + RLOO.

## Category distribution

`compute_score` returns 1.0 (correct), 0.1 (well-formed wrong), 0.0 (no answer / malformed). Categories below derive from re-running `extract_solution` + `validate_equation` + `evaluate_equation` (`evaluation/countdown.py`):

| category | rollouts | share |
|---|---|---|
| CORRECT (`equation` evaluates to target) | 456 | **57.0%** |
| **NO-ANSWER** (`<answer>` tag missing) | **207** | **25.9%** |
| BAD-NUMS (answer present, wrong operand multiset) | 87 | 10.9% |
| WRONG-VAL (operands correct, value ≠ target) | 50 | 6.2% |

The dominant failure mode is **NO-ANSWER, not wrong arithmetic.** The model rambles through many search paths and hits the `max_tokens=1024` budget before committing.

## Arithmetic-mistake density

Per-rollout we regex-extract every `a OP b = c` step (`(\d+)\s*([+\-*/])\s*(\d+)\s*=\s*(-?\d+)`) and check `eval(LHS) == RHS`.

| category | ≥ 1 mistake | mean ops / rollout | mean mistakes / rollout |
|---|---|---|---|
| CORRECT | 91/456 (20%) | 13.5 | 0.36 |
| **NO-ANSWER** | **119/207 (57%)** | **67.1** | **2.13** |
| BAD-NUMS | 19/87 (22%) | 18.0 | 0.54 |
| WRONG-VAL | 29/50 (**58%**) | 34.1 | 1.12 |

NO-ANSWER rollouts have ~5× the number of arithmetic operations as CORRECT rollouts (67 vs 13) and ~6× the mean-mistake count. The "rambling, sticky-wrong-fact" failure mode is empirically the longest and most error-dense.

### Mistakes-per-rollout histogram (capped at 5+)

| category | 0 | 1 | 2 | 3 | 4 | 5+ |
|---|---|---|---|---|---|---|
| CORRECT | 365 | 54 | 23 | 5 | 5 | 4 |
| NO-ANSWER | 88 | 31 | 24 | 17 | 10 | **37** |
| BAD-NUMS | 68 | 5 | 7 | 3 | 1 | 3 |
| WRONG-VAL | 21 | 15 | 10 | 0 | 3 | 1 |

37 of 207 NO-ANSWER rollouts have ≥ 5 detectable arithmetic mistakes — the long-tail rambling pattern.

## Three findings that should shape PRM design

### 1. The dominant failure is runaway exploration, not bad arithmetic

NO-ANSWER (26%) outweighs WRONG-VAL (6%) and BAD-NUMS (11%) combined. Arithmetic-PRM only indirectly addresses NO-ANSWER by penalizing the wrong arithmetic that fills these long traces.

### 2. The direct PRM target is small but high-yield

WRONG-VAL + BAD-NUMS = 137 rollouts (17% of total). 58% of WRONG-VAL and 22% of BAD-NUMS contain ≥ 1 detectable arithmetic mistake. If a PRM converts even 30–50% of these to CORRECT, the ceiling impact is **+ 3–7 pts pass@1**. Not transformative; meaningful.

### 3. ~20% of CORRECT rollouts also have arithmetic mistakes

The model occasionally gets the right answer despite sloppy CoT — e.g., one wrong line in a chain that nonetheless terminates with a correct expression. A negative-only arithmetic PRM still penalizes these. Net effect: a slight pressure toward cleaner CoT inside correct rollouts. Acceptable, but worth knowing it isn't a free signal for the wrong-answer cases.

## The "sticky-wrong-fact" pattern (representative trace)

NO-ANSWER rollout, nums = `[63, 95, 96]`, target = 64. Excerpts:

```
3. Looking at 96 and 95:
96 - 95 = 1
1 * 96 = 96 (too big)
…
5. What if we try:
95 - 96 = -1
-1 * 63 = -63 (wrong direction)

6. Let's try:
63 * 95 = 5885    ← actual: 5985    ✗
…
7. Let's try:
96 / 95 ≈ 1.02
63 * 1.0
```

`63 * 95 = 5885` appears **four times** in this single trace; the wrong fact enters context once and the model re-uses it. The outcome reward can't distinguish this from "the search was unlucky" — but an arithmetic-PRM (`-β` per incorrect op) penalizes each re-emission. This is the failure mode the PRM is uniquely positioned to suppress.

## Implications for the next experiment

Two candidate PRM designs to compare against the outcome-only baseline (default RLOO at 0.580):

- **PRM-A (negative-only arithmetic correctness):** total reward `= compute_score - β · #incorrect_ops`. Targets WRONG-VAL + BAD-NUMS directly; secondary "anti-rambling" effect on NO-ANSWER via the sticky-wrong-fact penalty. Cleanest ablation; safest against reward hacking ("emit no arithmetic" hack is bounded by the SFT prior + KL).
- **PRM-B (arithmetic + length penalty):** `compute_score - β · #incorrect_ops - γ · max(0, n_tokens - 300)`. Directly attacks NO-ANSWER. Risk: model truncates to game the length term.

Recommendation: start with PRM-A as the clean experimental cell against default RLOO; measure the NO-ANSWER delta separately. Add length penalty only if NO-ANSWER stays high.

## Reproducing this analysis

```bash
# 1. Eval the checkpoint (~10 min on Modal H100)
bash evaluation/sample_modal_default_rloo.sh

# 2. Fetch the eval JSON locally
.venv/bin/modal volume get default-proj-training \
    /evaluation/eval_results/default_rloo_qwen_sft.json \
    evaluation/eval_results/default_rloo_qwen_sft.json --force

# 3. Re-run the inline analysis from this doc — see commit log /
#    chat transcript for the exact one-shot Python; the categorization
#    + arithmetic-mistake regex live there. analyze_failure_modes.py
#    handles the trajectories.txt variant (SELS eval format).
```
