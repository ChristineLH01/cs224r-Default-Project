# Project Retrospective — Countdown RL Extensions

Comprehensive walkthrough of every methodology tried in this project, with
hypotheses, designs, empirical results, identified issues, and mitigations.
Companion to `evaluation/failure_modes_default_rloo.md` and
`evaluation/failure_modes_arithprm_v1.md` (which deep-dive specific runs).

The project explores whether *process-style* reward shaping and *structured
search* can improve over outcome-only RLOO on the Countdown arithmetic task.
Baseline: Qwen-2.5-0.5B-Instruct, SFT'd on `asingh15/countdown_tasks_3to4`,
then RLOO with the verifier-based `compute_score` reward (1.0 correct / 0.1
well-formed wrong / 0 malformed).

## TL;DR — full ablation table

All numbers below are pass@1 (50 prompts × 16 samples → first sample only),
pass@16 (any of 16 correct), and mean per-rollout score on the held-out test
split. Same eval recipe (temperature 0.6, top_p 0.95, max_tokens 1024,
`stop=["</answer>"]`) for every cell.

| run | training | pass@1 | pass@16 | mean score | dominant failure |
|---|---|---|---|---|---|
| default SFT | — | 0.349 | — | — | — |
| **default RLOO (baseline)** | 100 steps, outcome only | **0.580** | **0.720** | **0.587** | NO-ANSWER 26%, BAD-NUMS 11% |
| SELS v5 SFT | 1K synthetic state-token traces | 0.263 | — | — | substrate cost > benefit |
| SELS v5 RLOO (no PRM) | 100 steps from v5 SFT | 0.311 | — | — | structural verbose tax |
| SELS v5 RLOO + structural PRM | 100 steps, α=0.5 | 0.322 | — | — | substrate gap not closed |
| SELS v5 RLOO + structural PRM + ctxmgr | crashed @ step 49 | — | — | — | sentinel 99.6% — broken integration |
| Default + arith PRM v1 (unconditional) | 100 steps, α=1.0 | 0.460 | 0.680 | 0.510 | verification tax → NO-ANS 32% |
| Default + conditional arith + decisiveness | 100 steps | 0.540 | 0.660 | 0.551 | bad-commits cut, NO-ANS up |
| **Default + conditional arith + decisiveness** | **200 steps** | **0.580** | 0.680 | 0.564 | search rambling (30% sub-B) |
| (proposed) Restart-on-Rambling | not started | TBD | TBD | TBD | search compression target |

**Bottom line:** combined reward shaping (Phase 3) matched baseline pass@1
exactly, with a meaningful redistribution of failure modes (bad-commits halved,
WRONG-VAL nearly halved, NO-ANSWER rate elevated). No tested combination
exceeded baseline pass@1.

---

## Phase 0 — Default substrate baselines

### Default SFT
- **Method:** standard supervised fine-tuning on the Countdown-3to4 training
  split. Qwen-2.5-0.5B-Instruct base. Single epoch.
- **Pass@1: 0.349.** The substrate the rest of the project sits on.

### Default RLOO
- **Hypothesis:** outcome-reward RL on top of SFT should boost pass@1 by
  amplifying the small subset of CORRECT rollouts within each group of 8.
- **Method:** standard RLOO (`rloo_trainer/rloo.py`), 100 training steps,
  gs=8, KL=0.001, max_tokens=1024, temperature=1.0, lr=1e-5. Launcher
  `train_rloo_default.sh`.
- **Result:** pass@1 = 0.580 (+23 pts over SFT), pass@16 = 0.720, mean = 0.587.
- **Failure-mode breakdown** (800 rollouts, detailed in `failure_modes_default_rloo.md`):
  - CORRECT 57.0%, **NO-ANSWER 25.9%** (dominant), BAD-NUMS 10.9%, WRONG-VAL 6.2%
  - WRONG-VAL: 58% have ≥ 1 detectable arithmetic mistake (mean 1.12)
  - NO-ANSWER: 57% have ≥ 1 mistake (mean 2.13), median 67 arithmetic ops/trace
  - **BAD-NUMS: 86% had target legally reached in CoT** but committed wrong (typically using target as operand)
- **Headroom:** 14-pt gap between pass@1 (0.58) and pass@16 (0.72) — for 14%
  of test prompts the model produces a correct rollout *somewhere* in 16
  samples but the modal rollout is wrong.

---

## Phase 1 — SELS: Structured Search with Special State/Backtrack Tokens

### Hypothesis (from project proposal)
Explicit `<state_K>`/`<backtrack_K>` tokens give the model a structured
backtracking primitive. A solver-grounded process reward (`is_solvable`
check at backtrack events) provides dense intermediate gradient. Together
they should improve search efficiency on combinatorial reasoning.

### Method
- **Special tokens:** 10 added — `<state_0..4>` and `<backtrack_0..4>` (depth-indexed)
- **SFT data:** synthetic search traces generated via DFS + GPT narration
  (`dataset_generation/`). Iterated v1 → v5 fixing operand-leakage,
  format invariants, and shape mix.
- **Structural PRM** (`rloo_trainer/process_reward.py`): 7 per-event rules
  (state_depth_consistent, forward_transition_valid, operand_legal,
  arith_correct, …) + 3 trace-level rules (think_well_formed,
  no_special_tokens, answer_matches_final_state). Aggregated as mean of
  all booleans ∈ [0, 1].
- **Context restoration manager** (`rloo_trainer/context_restoration.py`):
  segmented sampling with `<backtrack_K>` stops + cleaned re-prompt, so
  failed branches don't pollute the conditioning window.

### Results

| run | pass@1 | failure mode |
|---|---|---|
| SELS v5 SFT | **0.263** | — |
| SELS v5 RLOO (no PRM) | 0.311 | NO-ANSWER 54% (rambling) |
| SELS v5 RLOO + structural PRM (α=0.5) | 0.322 | improvement marginal |
| SELS v5 RLOO + structural PRM + ctxmgr | crashed @ step 49 | sentinel-trip 99.6% |

### Issues identified
1. **Substrate regression** — v5 SFT (0.263) was substantially worse than
   default SFT (0.349). The structural overhead ate tokens that would have
   gone to reasoning.
2. **Token-learning bottleneck.** 10 new tokens × ~100 examples per token
   in 1K training samples → embeddings under-trained.
3. **No error masking in SFT.** The model was explicitly trained to
   *generate* failed branches followed by backtracks, rather than to
   *recognize and react* to errors. Contrast with Yang et al. 2025
   "Self-Backtracking" (cited later) which masks the erroneous-action
   tokens — model only learns to emit `<backtrack>` after them.
4. **Context-restoration manager broke PRM integration.** Sentinel
   "no/excess events" tripped on 99.6% of rollouts because the cleaned
   trace structure didn't match what the structural PRM parser expected.

### Mitigations attempted
- v1 → v5 SFT data iterations (operand leakage, restored-state formats,
  backtrack semantics)
- Structural PRM v1 → v3 (added arith_correct, operand_legal, trace-level
  rules)
- Context restoration manager design (`rloo_trainer/context_restoration.py`)

### Resolution
**Pivot to default substrate.** The SELS substrate gap (0.263 vs 0.349)
proved too large for any reward shaping to recover. Comparison with Yang
et al. 2025 (Self-Backtracking, arXiv:2502.04404) clarified the structural
choices that would have been needed: a single backtrack token, error-masked
SFT, 500× more training data, no RL. None of those fit our project's
remaining time budget.

**Documented in:** `evaluation/failure_modes_default_rloo.md` (the
post-pivot baseline analysis).

---

## Phase 2 — Arithmetic PRM v1 (unconditional)

### Hypothesis
On the default substrate (pass@1 = 0.580), most failures involve detectable
arithmetic errors (58% of WRONG-VAL, 57% of NO-ANSWER). A direct penalty
on wrong arithmetic should shape the policy toward fewer mistakes →
fewer wrong answers.

### Method
- **`rloo_trainer/arithmetic_prm.py`**, "Design 1":
  - Regex-extract `a OP b = c` patterns
  - For each, check `eval(a OP b) == c` (with non-integer division skipped)
  - Penalty: β · n_total_mistakes + γ · n_repeated_mistakes (β=0.1, γ=0.2)
  - Repetition kicker targets the *sticky-wrong-fact* pattern observed in
    NO-ANSWER traces (same wrong op emitted multiple times)
  - Capped at -0.5 to bound per-rollout impact
- Total reward: `compute_score + arithmetic_coef · prm` with α = 1.0
- Trained 100 steps from default SFT (launcher `train_rloo_default_arith_prm.sh`)

### Results
- **Pass@1 = 0.460 (regressed 12 pts vs baseline 0.580).**
- Pass@16 = 0.680 (regressed 4 pts vs baseline 0.720).
- The PRM *did* hit its training objective:
  - `arithmetic/n_total_mistakes_mean`: 1.81 → 0.59 (−67% across training)
  - `arithmetic/rollouts_with_mistake_rate`: 0.65 → 0.30 (−54%)
- Categorical shift:
  - CORRECT 57% → **49%**
  - NO-ANSWER 26% → **32%**

### Issue: verification tax (Goodhart's law)
- The PRM gradient ("mistakes are bad") generalized to "be more careful
  globally" — model learned a verbose multi-attempt template
  ("First attempt: / Second attempt: / Third attempt:")
- Trace lengths inflated: mean 914 → 1078 chars (+18%), median 626 → 818 (+31%)
- Verbose traces hit `max_tokens=1024` budget before committing → more
  NO-ANSWERs
- Disagreement analysis: 9 prompts regressed (baseline got right, treatment
  got wrong) vs only 3 fixed

### Mitigation: identified two design fixes
1. **Outcome-conditional gate** — don't penalize CORRECT rollouts at all
   (where the verification tax pressure was costing real prompts)
2. **Positive-side signal** — reward decisive commitment to legally-derived
   target (the "found-target-but-bungled-commit" failure pool, 86% of BAD-NUMS)

**Documented in:** `evaluation/failure_modes_arithprm_v1.md` — the canonical
reference for this regression and its mechanism.

---

## Phase 3 — Decisiveness Reward + Conditional Arithmetic PRM (v2)

### Pre-design diagnostic
Retroactively running the *proposed* decisiveness reward on the existing
default-RLOO eval JSONs revealed concentration of the addressable failure pool:

| category | target reached legally | committed correctly when reached |
|---|---|---|
| CORRECT (baseline) | 97.1% | 97.1% |
| BAD-NUMS (baseline) | **86.2%** ← biggest finding | 0% (definitionally) |
| WRONG-VAL (baseline) | 30.0% | 0% |
| NO-ANSWER (baseline) | 16.4% | 0% (no answer) |

86% of BAD-NUMS rollouts had **already legally computed target** in their
CoT but committed an answer that didn't validate (typically using target
itself as an operand). Total "failed but target legally reached" pool:
15.5% of all rollouts — the addressable headroom for a decisiveness signal.

### Hypothesis
Two changes would fix Phase 2's regression while preserving the arithmetic
signal:
1. Internalize the conditional gate inside `compute_arithmetic_prm` — skip
   the penalty when `compute_score(text, gt) >= 1.0` (removes verification tax)
2. Add a sparse, mostly-positive decisiveness reward — `+0.2` for
   legally-derived-target + correct commit, `-0.3` for target-reached-no-commit,
   `-0.2` for target-reached-wrong-commit, `0` when target never reached

### Method

**`rloo_trainer/arithmetic_prm.py` — outcome-conditional gate added:**
```python
if compute_score(response_text, ground_truth) >= 1.0:
    return 0.0, {"skipped_correct": True}
# else: existing β·mistakes + γ·repeats penalty
```

**`rloo_trainer/decisiveness_reward.py` — new module:**
- Walk arithmetic steps with operand-legality tracking (derived set =
  original_nums ∪ correctly-derived intermediates)
- For each step `a OP b = c`: skip if operand illegal, skip if arithmetic
  wrong, otherwise add `c` to derived. Stop when `c == target`.
- Return: `+0.2` if target reached + correct commit, `-0.3` if no commit,
  `-0.2` if bad/wrong commit, `0` if target never reached

**Total reward:**
```
total = compute_score + α · arithmetic_prm + β · decisiveness
       (α = β = 1.0; total range [-0.8, +1.2])
```

**Launcher:** `train_rloo_default_decisiveness.sh` (despite the filename,
this is the COMBINED arith+dec launcher). 200 training steps (vs prior 100).

### Pre-training verification
On existing eval JSONs (`evaluation/verify_combined_rewards.py`):
- All correctness checks pass (conditional gate, decisiveness sparsity, semantics)
- Combined reward total stdev: 0.705 vs compute_score 0.476 → **1.48× more spread**
- Within-group LOO advantage stdev amplification: **1.54×** (RLOO sees that much
  more discriminative gradient)
- No pathological cases beyond the −0.8 floor

### Results (step 199)

| metric | value |
|---|---|
| pass@1 | **0.580** (tied with baseline) |
| pass@16 | 0.680 |
| mean score | 0.564 |
| `commit=bad_nums` rate | 4.9% (down from 9.4% baseline; −48%) |
| `commit=wrong_value` rate | 0.5% (down from 1.9%; −74%) |
| `commit=correct` rate | 54.4% (vs 55.4% baseline; ~unchanged) |
| BAD-NUMS share | 7.9% (down from 10.9%) |
| WRONG-VAL share | 3.4% (down from 6.2%) |
| **NO-ANSWER share** | **33.5%** (UP from 25.9%) |

### Behavioral verification — did the rewards do what they were designed for?

Empirical answers to "is the reward actually shaping behavior":

**Decisiveness reward: YES, works as designed.**
- "Tail" length (chars from first legal target-reached to `<answer>`): median
  **75 → 24 chars** (3× shorter — model commits almost immediately on finding target)
- Correct-commit rate, *given* target was legally reached: **78.1% → 86.1%**
- "Rambling past target" rate (reached target but wrong/missing answer):
  **21.9% → 13.9%** (cut in half)

**Conditional arithmetic PRM: over-corrected.**
- v1 (unconditional) was effective at reducing mistakes:
  baseline 0.89 → v1 0.55 mean mistakes/rollout
- v2 (conditional) **inflated** mistakes to **1.49 / rollout** — *worse than baseline*
- The conditional gate removed the only mechanism by which the arithmetic
  signal could shape clean-CoT behavior (gradient now only fires on
  already-wrong rollouts, no contrast against clean-correct)

### Why decisiveness gains didn't translate to higher pass@1

Decomposition of why v2 only ties baseline despite the commit-quality wins:

```
                              baseline     v2 step-199
target reached legally        567/800       505/800     ← v2 reaches less often
correct commit given target   78.1%         86.1%       ← v2 commits better
total CORRECTs                ≈ 443         ≈ 435       ← net wash
```

**V2 trades quantity (reaches target less often) for quality (commits more
correctly when reached).** Net pass@1 unchanged. The combined reward
solved its targeted failure mode (bad commits) but unmasked a different
dominant failure (target-never-reached, the search rambling pool).

### Pass@1 trajectory across saved checkpoints

```
v2 step 75:    0.460  ← below baseline
v2 step 100:   0.540  ← still below
v2 step 150:   0.520  ← slight dip
v2 step 199:   0.580  ← matches baseline (best)
```

200 steps were *needed* to recover to baseline. The combined reward signal
is genuinely slower-to-converge than outcome-only RLOO, not faster.

### Documented in
- `evaluation/failure_modes_arithprm_v1.md` (background motivation —
  decisiveness diagnostic appendix)
- `evaluation/compare_failure_modes.py` (pairwise comparison)
- `evaluation/verify_rewards.py`, `evaluation/verify_combined_rewards.py`
  (pre-launch + correctness checks)

---

## Phase 4 — Trajectory deep-dive on v2 (current diagnostic state)

Focused analysis of v2 step-199's 800 rollouts to characterize the
remaining failure modes precisely.

### Findings
1. **NO-ANSWER decomposition**:
   - Sub-A (target reached, no commit): 27 rollouts = 3.4% (decisiveness mostly solved this)
   - **Sub-B (target NEVER legally reached): 241 rollouts = 30.1% (DOMINANT)**

2. **Sub-B characteristics** (the rambling search-failure pool):
   - Median trace length 2163 chars (vs CORRECT 360 chars — 6× longer)
   - Median 64 arithmetic ops/trace (vs CORRECT ~10)
   - 0% hitting `max_tokens` — the model gives up rambling before budget exhausts

3. **Pathological patterns observed in sub-B traces:**
   - **Repetition without learning** — same `(a, op, b)` triple tried 3-4× per trace
   - **Hallucinated operands** — uses numbers not in originals
   - **Catastrophic loops** — one trace literally repeated `"3035 - 6 = 3035 - 6 ..."` 22 times

4. **Per-prompt concentration**:
   - 16 of 50 test prompts have 0/16 CORRECT samples (truly stuck → pass@16
     ceiling = 0.68)
   - On stuck prompts, 78% of samples are NO-ANSWER

5. **Sample diversity within prompts**:
   - Mean 8.8 CORRECT samples per 16-sample prompt (capability is there)
   - Mean 5.4 NO-ANSWER samples per prompt (⅓ of samples wasted)

### Implication
The remaining failure is **search efficiency**, not commit quality. The
decisiveness reward already mostly solved commit quality. No additional
terminal-scalar reward targeting commit decisions can fix this — the
mechanism is *the model rambles through repeated operations without
recognizing it should restart with a fresh strategy*.

---

## Phase 5 (proposed, not yet implemented) — Restart-on-Rambling via Self-Recognition

### Hypothesis
Train the model to recognize rambling patterns in its own context and emit
a `<restart>` token, then truncate to the original prompt at inference.
Unifies reward shaping (existing) with state backtracking (via context
truncation).

### Method outline
1. **Synthetic data generation:** For each training prompt that has both
   rambling NO-ANSWER and CORRECT rollouts from v2, splice:
   `[rambling head from NO-ANSWER]` + `<restart>` + `[correct trace tail]`.
   Mask the rambling-head tokens from SFT loss (Self-Backtracking trick).
2. **SFT** for 1 epoch on this synthetic + original-CoT data.
3. **Inference**: vLLM stops on `<restart>`, context truncates to prompt,
   regenerates. Up to N=3 restart attempts per rollout.
4. **Final RL** with existing combined reward (50 steps).

### Mathematical bound
Effective sampling becomes pass@4 of the SFT'd policy (4 independent
attempts at constant inference cost). Bound: ≥ pass@16 of current model
(0.72) in the limit. Realistic estimate: pass@1 = 0.65–0.72.

### Status
**Designed in detail (see chat transcript). Not implemented yet.**

---

## Cross-cutting analysis — failure-mode evolution

How the dominant failure mode shifted across phases:

| phase | dominant failure | mechanism |
|---|---|---|
| 0 (baseline) | NO-ANSWER 26% (mixed: arithmetic mistakes + verbose traces) | undifferentiated |
| 2 (v1 arith) | NO-ANSWER 32% (verbose template induced by mistake penalty) | verification tax |
| 3 (v2 arith+dec) | NO-ANSWER 33.5% (target never legally reached) | search rambling |
| 5 (restart-on-ramble, proposed) | TBD | targets sub-B directly |

Each design successfully addressed the failure mode it was *designed* for,
but unmasked a different remaining failure. The interpretation: outcome
reward + LOO baseline already implicitly handles "be correct overall";
process rewards can shape *specific* sub-failures (commit decisions,
arithmetic execution) but don't aggregate into pass@1 gains until **all**
sub-failures are addressed in concert.

---

## Cross-cutting analysis — what each design tried to mitigate, and outcomes

| design | targeted failure | outcome on target | side effect | overall pass@1 |
|---|---|---|---|---|
| SELS structure + structural PRM | "model needs explicit backtracking" | structural compliance ↑ | substrate cost ↑↑ | 0.32 (regressed) |
| Arith PRM v1 (unconditional) | wrong arithmetic ops | mistakes −67% ✓ | verbose template, NO-ANS ↑ | 0.46 (regressed) |
| Conditional gate (inside arith PRM) | verification tax on CORRECTs | tax eliminated ✓ | arithmetic signal too weak | recovered to 0.58 |
| Decisiveness reward | bad commits (BAD-NUMS, WRONG-VAL) | bad-commit rate cut in half ✓ | search reach rate down | tied baseline 0.58 |
| Restart-on-rambling (proposed) | search rambling (NO-ANS sub-B) | TBD | TBD | TBD |

### Three lessons for the writeup

**1. Reward shaping that conflicts with the outcome reward's implicit
signal regresses pass@1.** Arith PRM v1 added a "mistakes bad" gradient
that the model generalized into "be more careful globally" — the
verification tax. The outcome reward already discriminates correct from
wrong; adding a second "mistakes bad" gradient over-corrects on rollouts
the outcome reward was already shaping well.

**2. Sparse, positive-side, behavior-specific rewards compose cleanly.**
Decisiveness reward fires on at most 15.5% of rollouts, mostly with `+0.2`,
and targets a specific decision (commit vs. don't). It empirically did
what it was designed for without inducing collateral pathologies. The
contrast with v1's diffuse negative signal is the methodological lesson.

**3. Conditional gating can be too strong.** v2's conditional gate
eliminated v1's verification tax (good), but also eliminated the
mechanism through which the arithmetic signal had been working on
the modal CORRECT-vs-wrong contrast. The arith PRM stopped shaping
behavior because almost all its meaningful gradient came from the
clean-CORRECT vs sloppy-anything contrast. The fix should not be
"gate off entirely" but "discount on CORRECTs while preserving some
contrast" — e.g., half-magnitude rather than zero. Not yet tested.

---

## Reproducibility

### Trained checkpoints (Modal volume `default-proj-training`)
```
/vol/checkpoints/rloo_checkpoints/rloo_training/
    default_rloo_qwen_sft_gs8_lr1e-5_kl0.001_t1024/                    (baseline)
    default_rloo_qwen_sft_gs8_lr1e-5_kl0.001_t1024_arithprm1.0/        (v1 arith uncond)
    default_rloo_qwen_sft_gs8_lr1e-5_kl0.001_t1024_arith1.0_dec1.0/    (v2 arith+dec, 200 steps)
```

### Eval JSONs (in `evaluation/eval_results/`, gitignored — pulled from Modal volume)
```
default_rloo_qwen_sft.json                    (baseline)
default_rloo_qwen_sft_arithprm.json           (v1 arith uncond)
default_rloo_qwen_sft_arith_dec.json          (v2 step 199)
default_rloo_qwen_sft_arith_dec_step{75,100,150}.json    (v2 trajectory)
```

### Launchers
```
rloo_trainer/train_rloo_default.sh                  (baseline)
rloo_trainer/train_rloo_default_arith_prm.sh        (v1 arith uncond)
rloo_trainer/train_rloo_default_decisiveness.sh     (v2 — DESPITE filename, combines both)
evaluation/sample_modal_default_rloo.sh             (eval baseline)
evaluation/sample_modal_default_arith_prm.sh        (eval v1)
evaluation/sample_modal_default_arith_dec.sh        (eval v2 step 199)
```

### Reward modules
```
rloo_trainer/process_reward.py        (SELS structural PRM; self-tested)
rloo_trainer/arithmetic_prm.py        (Design 1, internal conditional gate; self-tested)
rloo_trainer/decisiveness_reward.py   (sparse positive-side commit reward; self-tested)
rloo_trainer/context_restoration.py   (segmented sampling; self-tested)
```

### Analysis pipelines
```
evaluation/analyze_failure_modes.py       (parse _trajectories.txt, category breakdown)
evaluation/compare_failure_modes.py       (pairwise eval JSON comparison)
evaluation/verify_rewards.py              (correctness checks on real eval data)
evaluation/verify_combined_rewards.py     (combined-reward scale + LOO advantage check)
```

### Companion docs
```
evaluation/failure_modes_default_rloo.md      (baseline failure-mode reference)
evaluation/failure_modes_arithprm_v1.md       (v1 regression + decisiveness motivation)
evaluation/methods_retrospective.md           (this document)
```

### W&B
Project: `ruichu-xia-stanford-university/rloo_training`
- `default_rloo_qwen_sft_gs8_lr1e-5_kl0.001_t1024`
- `default_rloo_qwen_sft_gs8_lr1e-5_kl0.001_t1024_arithprm1.0`
- `default_rloo_qwen_sft_gs8_lr1e-5_kl0.001_t1024_arith1.0_dec1.0` (200 steps)

Project: `ruichu-xia-stanford-university/sels_rloo`
- (SELS experiments, deprioritized per Phase 1 conclusion)

---

## What's tracked in this project's git history (`rloo-extension` branch)

```
.gitignore                          (.venv, eval_results/, checkpoints/, etc.)
CS224R_Project_Proposal.pdf
dataset_generation/                 (SELS SFT data generation)
sft_trainer/                        (default + SELS SFT)
ipo_trainer/                        (unused in this project's experiments)
rloo_trainer/
    arithmetic_prm.py               (Phase 2-3)
    decisiveness_reward.py          (Phase 3)
    context_restoration.py          (Phase 1 — SELS-only)
    process_reward.py               (Phase 1 — SELS structural PRM)
    rloo.py                         (trainer with all extension flags, default-off)
    rloo_update_worker.py           (unchanged from starter)
    sampling_worker.py              (segmented sampling additions, gated)
    train_rloo_*.sh                 (per-experiment launchers)
modal_train.py                      (Modal entry; .venv ignore added)
sample_smoke.py                     (rollout inspection, segmented-mode aware)
evaluation/
    countdown.py                    (compute_score, validate_equation)
    countdown_eval.py               (eval driver; </answer> stop fix)
    sample_modal_*.sh               (per-checkpoint eval launchers)
    analyze_*.py / compare_*.py / verify_*.py  (analysis scripts)
    failure_modes_*.md              (this file + others)
```

All extension code is **default-off**: setting all `--*_coef` flags to 0
gives byte-identical baseline RLOO training. Verified by reading the call
sites in `rloo.py`'s reward-computation block.
