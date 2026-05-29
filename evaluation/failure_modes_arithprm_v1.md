# Arithmetic PRM v1 — Training and Eval Analysis

Empirical evaluation of the Design-1 arithmetic process reward (negative-only
with sticky-wrong-fact repetition kicker) trained on top of default RLOO on
Countdown. Paired with `evaluation/failure_modes_default_rloo.md` (the
baseline failure-mode reference). Headline finding: the PRM successfully
suppressed arithmetic mistakes (objective achieved) but regressed pass@1 by
12 pts via a **verification tax** mechanism — the model became cleaner per
operation but more verbose, hitting the `max_tokens` budget more often and
producing more NO-ANSWER rollouts.

## Training setup

- **Reward shape:** `total = compute_score + α · arithmetic_prm` where the
  arithmetic PRM is `−min(β·n_total_mistakes + γ·n_repeated_mistakes, cap)`
  with `β=0.1`, `γ=0.2`, `cap=0.5`, `α=1.0` (so PRM contribution is in
  `[−0.5, 0]`). Implementation: `rloo_trainer/arithmetic_prm.py`.
- **Warm start:** `asingh15/qwen-sft-countdown-defaultproj` (same as the
  default-RLOO baseline run at W&B `default_rloo_qwen_sft_gs8_lr1e-5_kl0.001_t1024`).
- **Hyperparameters:** identical to `rloo_trainer/train_rloo_default.sh`
  (gs=8, lr=1e-5, KL=0.001, max_tokens=1024, max_model_len=2048, 100 steps,
  H100, temperature=1.0, top_p=1.0).
- **Launcher:** `rloo_trainer/train_rloo_default_arith_prm.sh`.
- **W&B run:** `default_rloo_qwen_sft_gs8_lr1e-5_kl0.001_t1024_arithprm1.0`
  (project `rloo_training`). 99 training steps completed.
- **Eval recipe:** identical to baseline (`evaluation/sample_modal_default_arith_prm.sh`,
  50 prompts × 16 samples = 800 rollouts, temperature 0.6, `stop=["</answer>"]`).

## Training-time outcomes: the PRM hit its objective

W&B training curve from start to step 99:

| metric | start (~step 0) | end (step 99) | change |
|---|---|---|---|
| `sampling/reward_mean` (raw compute_score) | ~0.29 | 0.522 | +80% |
| `arithmetic/n_total_mistakes_mean` | ~1.80 | 0.589 | **−67%** |
| `arithmetic/rollouts_with_mistake_rate` | ~0.65 | 0.300 | **−54%** |
| `arithmetic/n_repeated_mistakes_mean` | ~0.31 | 0.108 | **−65%** |
| `sampling/arithmetic_reward_mean` | ~−0.17 | −0.060 | mistakes ↓ ⇒ PRM penalty ↓ |

The PRM gradient successfully reshaped the policy — by step 99 the model
produced ~⅔ fewer arithmetic mistakes per rollout, and the sticky-wrong-fact
repetition pattern was cut by 65%. **The PRM did exactly what it was
mathematically designed to do.**

## Headline eval numbers — outcome regressed

50 prompts × 16 samples = 800 rollouts on `asingh15/countdown_tasks_3to4` test.

| metric | baseline (default RLOO) | + arithmetic PRM | Δ |
|---|---|---|---|
| **pass@1** | 0.580 | **0.460** | **−0.120** |
| **pass@16** | 0.720 | 0.680 | −0.040 |
| mean score | 0.587 | 0.510 | −0.077 |

Note pass@16 regressed only 4 pts while pass@1 regressed 12 pts. The model's
*capability* (pass@16) is largely preserved; the *selection* of which sample
to emit first got worse. This is the signature of a budget-driven failure
(NO-ANSWERs vs CORRECTs depend on which sample finishes in time), not a
fundamental capability loss.

## Category distribution: outcome migrated from CORRECT to NO-ANSWER

| category | baseline | + PRM | shift |
|---|---|---|---|
| CORRECT | 57.0% (456) | 49.1% (393) | **−63 rollouts** |
| NO-ANSWER | 25.9% (207) | 32.2% (258) | **+51 rollouts** |
| BAD-NUMS | 10.9% (87) | 11.9% (95) | +8 |
| WRONG-VAL | 6.2% (50) | 6.8% (54) | +4 |

The lost CORRECTs are reasonably explained by the gain in NO-ANSWERs: ~63
rollouts that would have produced a correct answer instead ran out of
`max_tokens` before committing.

## Per-category arithmetic-mistake density — PRM hit its target in every bucket

| category | baseline ≥1 mistake | + PRM ≥1 mistake | mean mistakes baseline | mean mistakes + PRM |
|---|---|---|---|---|
| CORRECT | 20.0% (91/456) | 16.3% (64/393) | 0.36 | 0.23 |
| WRONG-VAL | **58.0%** (29/50) | **29.6%** (16/54) | 1.12 | 0.54 |
| BAD-NUMS | 21.8% (19/87) | 18.9% (18/95) | 0.54 | 0.29 |
| NO-ANSWER | **57.5%** (119/207) | **35.3%** (91/258) | 2.13 | 1.14 |

Across all categories the mistake density dropped 30–50%. The PRM training
worked at the level it was designed for. The regression is in what the model
did *behaviorally* in response to the gradient — not in whether it learned
the signal.

## The verbose-template signature

Trace-length distributions, baseline vs treatment:

| | baseline | + PRM | Δ |
|---|---|---|---|
| mean chars | 914 | 1078 | +18% |
| median chars | 626 | **818** | **+31%** |
| p90 chars | 1764 | 1966 | +11% |
| mean arithmetic ops / rollout | 29.1 | 31.7 | +9% |

The treatment writes **longer traces with more arithmetic operations**, not
fewer. This is the opposite of "silent reasoning" — the model is more
verbose, not less. Eyeballing the traces reveals a distinct learned style
shift:

- **Baseline** uses an inline narrative: `"Let me try a different approach. 35 + 19 = 54. 54 + 44 = 98 (this works!)"`
- **Treatment** uses a numbered-attempt structure: `"First attempt: ... Second attempt: ... Third attempt: ..."` with each attempt fleshed out before pivoting.

Each numbered attempt is longer than the baseline's inline narrative, so
fewer fit in the 1024-token budget. The model converged on a verbose
multi-attempt template under the PRM gradient.

## Disagreement analysis — 9 regressions vs 3 wins on first-sample pass@1

Out of 50 prompts (sample[0] of each):

| | count |
|---|---|
| Both right (no change) | 20 |
| PRM fixed (wrong → right) | 3 |
| PRM regressed (right → wrong) | **9** |
| Both wrong | 18 |
| **Net pass@1 effect** | **−6** |

Three representative regressions (full traces in `evaluation/eval_results/default_rloo_qwen_sft_arithprm.json`):

### Regression 1 — nums `[35, 77, 61]`, target 19

```
BASELINE  (CORRECT):
  "1. 77 - 61 = 16
   35 - 16 = 19 (this works!)"   → commits immediately

TREATMENT (NO-ANSWER):
  "First attempt:
   77 - 61 = 16 (this gets us closer to 19)
   16 + 35 = 51 (too high)

   Second attempt:
   61 - 35 = 26 (this is closer to 19)
   26 - 77 would..."             → keeps deliberating, runs out of tokens
```

Both versions compute the correct intermediate `77 − 61 = 16`. The baseline
naturally tries `35 − 16` next (the answer). The treatment tries `16 + 35`
(wrong direction), abandons to a new approach, and never circles back.

### Regression 2 — nums `[21, 39, 34, 36]`, target 16

```
BASELINE  (CORRECT):
  "39 - 34 = 5 (closer to 16)
   36 - 34 = 2 (closer to 16)
   2 * 39 = 78 ..."             → eventually commits to a correct expression

TREATMENT (NO-ANSWER):
  "First attempt:
   36 - 34 = 2
   2 * 39 = 78 (too big)

   Let's try a different approach.
   ..."                          → many short attempts, no commit
```

### Regression 3 — nums `[49, 41, 73]`, target 17

```
BASELINE  (CORRECT):  "49 - 41 = 8 (close to 17). 73 - 49 = 24..."  → commits
TREATMENT (BAD-NUMS): "Let's try division since it might help..."  → drifts to non-original numbers
```

## Diagnosis: verification tax (Goodhart-style optimization)

The mechanism is straightforward given the data above:

1. The PRM penalizes wrong arithmetic (`−β · n_mistakes`).
2. The policy gradient pushes the model toward producing rollouts with
   fewer arithmetic mistakes.
3. The model's two ways to reduce mistakes are: (a) actually compute
   arithmetic correctly, (b) verify more / try smaller steps to reduce risk
   of mistakes.
4. The model adopts **both** strategies. (a) is what we wanted; (b) is the
   "verification tax" — it consumes tokens.
5. Under a fixed `max_tokens=1024` budget, more verification per attempt
   means fewer attempts complete within budget.
6. Result: cleaner arithmetic per attempt (PRM objective ↓), more
   NO-ANSWERs (real outcome ↓).

This is the classic Goodhart pathology: a proxy metric (mistake count) was
optimized at the expense of the underlying goal (find a correct answer in
budget). The verbose multi-attempt template is the visible manifestation of
the proxy-optimization gradient.

The pass@1 vs pass@16 split (12 pts vs 4 pts regression) confirms the
mechanism: the model's *capability* (correct answer exists in 16 samples)
is largely preserved; only *first-sample-within-budget* degraded.

## Decisiveness diagnostic — implication for next experiment

Running the proposed decisiveness reward retroactively on both eval JSONs
(scoring: `target_reached_legally` requires the trace to compute target via
a sequence of arithmetic steps using only original-or-derived operands):

| category | baseline target-reached | + PRM target-reached |
|---|---|---|
| CORRECT | 97.1% (442/456) | 95.7% (376/393) |
| **BAD-NUMS** | **86.2% (75/87)** | 68.4% (65/95) |
| WRONG-VAL | 30.0% (15/50) | 40.7% (22/54) |
| NO-ANSWER | 16.4% (34/207) | 11.2% (29/258) |

**Striking finding: in the baseline, 86% of BAD-NUMS rollouts (75/87)
already legally computed the target somewhere in the CoT** but then
committed to an `<answer>` that doesn't validate (typically using target
itself as an operand, or otherwise drifting from the original-number
multiset). These are not "model can't solve" failures — they are
"model solved it in CoT but bungled the commit" failures.

Total *headroom* for a decisiveness reward (failed rollouts that legally
computed target → could be flipped to CORRECT by training the policy to
commit when it sees target):

```
NO-ANSWER  target reached: 34/207  (16.4%)
WRONG-VAL  target reached: 15/50   (30.0%)
BAD-NUMS   target reached: 75/87   (86.2%)
total headroom            : 124/800 rollouts = 15.5%
```

This is the bridge to the next experiment: a positive-side reward
(`+α` for "computed target legally AND committed correctly", sparse
elsewhere) avoids the verification-tax pathway by construction (no
ambient "be more careful" gradient), while specifically targeting the
15.5% of rollouts where the model already has the answer but doesn't
commit.

## Implications and lessons

1. **Negative-only reward shaping on outcomes already covered by the outcome
   reward is risky.** `compute_score` already differentiates correct from
   wrong arithmetic at the final-answer level. Adding `−β · n_mistakes` for
   intermediate steps gave the model a *second* "mistakes are bad" signal,
   which it generalized into the verbose-verification policy. The
   intermediate signal added behavioral pressure that the outcome reward
   alone did not.

2. **Pass@1 vs pass@16 gap is a substrate property, not a reward-shape
   property.** The 14-pt gap in the baseline (0.58 vs 0.72) reflects that
   the model already produces the right answer ~72% of the time given 16
   samples; the bottleneck is *first-sample selection*, which is a
   sampling/decoding property, not an arithmetic-quality one. Reward
   shapes that attack arithmetic quality don't address this gap directly.

3. **The BAD-NUMS-with-target-reached pool (75 rollouts, 9.4% of total) is
   the largest single addressable failure pattern.** It is not visible in
   the arithmetic-mistake density analysis (these rollouts often have
   clean arithmetic) and was therefore not targeted by Design 1. A
   commit-decisiveness reward addresses it directly.

4. **The PRM works as a *training* signal but might be better deployed as
   an *inference reranker*** (Lightman et al. 2023's original use case
   for PRMs). On Countdown specifically, `compute_score` is a perfect
   verifier at inference, making learned PRM reranking redundant — but
   the principle generalizes: PRMs trade off training-time behavior
   shaping (where they risk Goodhart) for inference-time selection (where
   they don't influence the policy's training distribution).

## Reproducing this analysis

```bash
# 1. Eval the arithprm checkpoint (~10 min on Modal H100)
bash evaluation/sample_modal_default_arith_prm.sh

# 2. Fetch the eval JSON locally
.venv/bin/modal volume get default-proj-training \
    /evaluation/eval_results/default_rloo_qwen_sft_arithprm.json \
    evaluation/eval_results/default_rloo_qwen_sft_arithprm.json --force

# 3. Side-by-side comparison against the baseline
PYTHONPATH=. .venv/bin/python evaluation/compare_failure_modes.py \
    evaluation/eval_results/default_rloo_qwen_sft.json \
    evaluation/eval_results/default_rloo_qwen_sft_arithprm.json
```

## Cross-references

- Baseline failure-mode analysis: [`evaluation/failure_modes_default_rloo.md`](failure_modes_default_rloo.md)
- PRM implementation: [`rloo_trainer/arithmetic_prm.py`](../rloo_trainer/arithmetic_prm.py)
- Training launcher: [`rloo_trainer/train_rloo_default_arith_prm.sh`](../rloo_trainer/train_rloo_default_arith_prm.sh)
- Eval launcher: [`evaluation/sample_modal_default_arith_prm.sh`](sample_modal_default_arith_prm.sh)
- Comparison analyzer: [`evaluation/compare_failure_modes.py`](compare_failure_modes.py)
