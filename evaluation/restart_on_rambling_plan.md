# Restart-on-Rambling Pipeline — Plan of Record

The plan for the next experiment: replace further reward-shaping experiments
with a behavior-cloning + inference-restart pipeline. Higher-confidence than
PRM design based on empirical track record in this project plus literature
precedent (Self-Backtracking, Yang et al. 2025).

**Status: planned, not yet implemented (as of writing).** Cross-references the
retrospective in `evaluation/methods_retrospective.md` for what's been tried.

## Why this approach over more reward shaping

| Reward shaping (what we've tried) | SFT + restart-on-rambling (this plan) |
|---|---|
| "Penalize behavior X" — model can dodge, side effects (verification tax) | "Show model what to do" — direct behavior cloning |
| Empirically risky: 2× regression-then-recovery cycle in this project (v1 → v2) | Self-Backtracking reports +40% on Countdown with this exact pattern |
| pass@1 effect is gradient-mediated and noisy | pass@1 has a hard bound: → pass@k via filtering + inference sampling |
| One reward, one knob, complex interactions | Behavior cloning of correct trajectories is well-established (STaR, RFT) |
| Net pass@1 effects in this project so far: 0 to negative | Mathematical guarantee under SFT objective + multi-restart sampling |

Self-Backtracking (Yang et al. 2025, arXiv:2502.04404) ran this on Countdown
with Llama-3.2-1B and hit pass@1 = 73%. Our v2 step-199 is at 0.58 with
Qwen-2.5-0.5B. Different model size, similar task — reaching ~0.70 with this
pipeline would be consistent with their result and our pass@16 ceiling (0.72).

## The two metrics this plan reduces

Carried over from prior analysis. Both deterministic, both regex-computable
from a trace + ground truth:

1. **`n_wrong`** (arithmetic incorrectness): count of `a OP b = c` extracted
   from the trace where `eval(a OP b) ≠ c`. Computed by
   `rloo_trainer/arithmetic_prm.py`.
2. **`n_repeats`** (looping/rambling): count of total arithmetic operations
   in the trace minus distinct `(a, op, b)` triples. Each time the model
   writes the same operation beyond the first occurrence, this counter ticks
   up.

Baseline measurements on the default-RLOO trained model (800 eval rollouts):

```
                 mean n_wrong   mean n_repeats   mean n_total_ops   pass@1
baseline         0.89           18.01            29.1               0.580
v1 arith uncond  0.55           21.87 (↑)        31.7               0.460
v2 arith+dec     1.49 (↑)       19.85            29.8               0.580
```

The restart-on-rambling pipeline targets both. Within failures, ~72% involve
rambling; ~58% of NO-ANSWER rollouts have ≥ 1 arithmetic mistake.

## The pipeline

```
┌──────────────────────────────────────────────────────────────┐
│ Step 1: Sample from current best model (v2 step-199)         │
│   For each TRAINING prompt: generate 16 rollouts             │
│   Modal cost: ~$15, ~1 hour                                  │
│   Output: 16 × N_train_prompts rollouts → JSON               │
└─────────────────────────┬────────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────────┐
│ Step 2: Classify rollouts per training prompt                │
│   For each rollout:                                          │
│     - CORRECT? (compute_score == 1.0)                        │
│     - Rambling? (single line repeated ≥ 3× verbatim,         │
│        OR n_repeats > 8; either works as detector)           │
│   Per-prompt: identify prompts that have BOTH at least one   │
│   CORRECT and at least one rambling NO-ANSWER                │
│   Expected: ~300–500 qualifying prompts                      │
│   Cost: ~30 min local Python                                 │
└─────────────────────────┬────────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────────┐
│ Step 3: Synthesize <restart> training examples               │
│   For each qualifying prompt:                                │
│     stuck_head = first 2-3 attempts of a rambling rollout    │
│                  (the visible failure pattern)               │
│     correct_tail = a CORRECT rollout from same prompt        │
│     synthetic_trace = original_prompt                        │
│                       + stuck_head                           │
│                       + "<restart>\n"                        │
│                       + correct_tail                         │
│   Mask the stuck_head tokens from SFT loss (Self-Backtracking│
│   trick — model learns to RECOGNIZE the rambling pattern and │
│   emit <restart>, NOT to generate the rambling itself)       │
│   Cost: half day local script                                │
└─────────────────────────┬────────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────────┐
│ Step 4: Tokenizer modification                                │
│   Add single <restart> token (additive_special_tokens).      │
│   Initialize embedding from a semantically-related existing  │
│   token like "Wait", "Reset", or "Stop".                     │
│   Resize model embedding layer to match.                     │
│   Cost: ~1 hour local                                        │
└─────────────────────────┬────────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────────┐
│ Step 5: SFT for 1 epoch                                      │
│   Train on:                                                  │
│     - Synthetic <restart>-data (with stuck_head masked)      │
│     - Original CoT examples (~1:2 ratio with synthetic) to   │
│       preserve general behavior                              │
│   LR: 1e-5 (low to prevent forgetting RL-learned behaviors)  │
│   Single epoch                                               │
│   Cost: ~4 hours Modal                                       │
└─────────────────────────┬────────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────────┐
│ Step 6: Inference-time restart loop                          │
│   Modify rloo_trainer/sampling_worker.py:                    │
│     stop=["</answer>", "<restart>"]                          │
│   When <restart> token is emitted:                           │
│     Truncate context back to original_prompt + "<restart>\n" │
│     Continue generation                                      │
│     Allow up to N=3 restart attempts per rollout             │
│   Eval the resulting policy.                                 │
│   Cost: half day implementation + ~$5 eval                   │
└─────────────────────────┬────────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────────┐
│ Step 7 (optional): Final RL pass                             │
│   IF Step 6 shows clear pass@1 gains AND we want to refine:  │
│   Run 50–100 steps RLOO with outcome reward only             │
│   (DROP the PRM rewards — we've evaluated them in this       │
│   project and the SFT step does the heavy lifting)           │
│   Cost: ~3 hours Modal                                       │
└──────────────────────────────────────────────────────────────┘
```

**Total: ~2 days of work + ~$80 Modal credit.**

## Concrete design decisions baked in

These have been pre-decided so the pipeline doesn't get stuck in design loops:

- **"Rambling" detector (Step 2)**: trace contains any line appearing ≥ 3×
  verbatim. Empirically simple, deterministic, catches 98.6% of NO-ANSWER
  rollouts per baseline analysis.

- **Single `<restart>` token**, not depth-indexed multi-token (SELS approach
  failed for that reason). Initialize from a semantically related token's
  embedding to avoid random-init learning bottleneck.

- **Error masking on the stuck_head** is critical. The model should NOT be
  trained to *generate* the rambling — only to *emit `<restart>` in response
  to seeing it*. This is the key Self-Backtracking technique.

- **3 restart attempts max** at inference. Effective sampling ≈ pass@4 of the
  SFT'd policy. Bound: ≥ pass@k of the SFT'd model under the iteration.

- **Drop all PRM rewards in Step 7's RL pass.** We've documented their
  issues in `evaluation/methods_retrospective.md`. The SFT step compresses
  behavior; outcome reward alone is sufficient to refine.

- **Train from v2 step-199**, NOT from scratch. We're building on existing
  capability, not replacing it. v2 step-199 is the best-saved checkpoint
  (matches baseline pass@1 = 0.58 with improved commit-quality
  distribution).

## Forecast (honest, with mathematical bound)

```
current v2 step-199:                              pass@1 = 0.580 (measured)

Step 1 alone (sample + filter):                   no model change
  Just data collection.

After Step 5 (SFT only, no inference restart):    pass@1 ≈ 0.60–0.66
  Theoretical bound: → pass@k of v2 step-199 (≈ 0.68)
  Realistic with 1 epoch + small data: captures ~50–70% of the gap

After Step 6 (SFT + inference restart loop):      pass@1 ≈ 0.65–0.72
  With N=3 restarts: effective pass@4 of SFT'd policy
  Bound ≤ pass@k of v2 step-199 = 0.68 (approximately)

After Step 7 (optional final RL):                 pass@1 ≈ 0.68–0.75
  RL refines whatever's left; bound depends on remaining gap to capability ceiling

Capability ceiling (32% truly-stuck prompts):     pass@1 ≤ ~0.85
  Cannot be broken by this pipeline alone — only by solver-distillation SFT
  or a bigger base model. Out of scope for this plan.
```

## What this plan does NOT do

- It does NOT break the 32% truly-stuck-prompt capability ceiling.
  Those prompts have 0/16 CORRECT samples in v2 step-199; the SFT step
  has no examples to learn from for them. To address those, the right
  intervention is `dataset_generation/countdown_solver.py` + synthetic
  solver-traces SFT. That's a separate plan.

- It does NOT promise pass@1 > 0.75. Self-Backtracking got 73% on Llama-3.2-1B
  (a bigger model). Our 0.5B model has a real capability ceiling that this
  pipeline can approach but probably not exceed.

- It does NOT continue iterating reward-shaping experiments. The PRM work
  we've done (`arithmetic_prm.py`, `decisiveness_reward.py`) is documented
  and complete; we're moving past it.

## Implementation order (incremental, inspectable at each step)

Do not commit to Step 5+ until Step 1-2 outputs look sane. Each step's output
is a concrete artifact that can be inspected and reasoned about before moving
on.

1. **Step 1 + 2 first** (~1 hour). Sample on training prompts; classify.
   Gives data inventory before committing.
2. **Step 3-4** (~half day). Materialize the synthetic training set. Inspect
   100 examples by hand to verify the splicing is sane.
3. **Step 5** (~overnight Modal job). SFT. Track loss curve.
4. **Step 6** (~half day implementation). Eval at this point. If pass@1
   moved meaningfully, this is the win.
5. **Step 7 only if** Step 6's eval shows clear gains worth refining and
   compute budget allows.

## Cross-references

- Retrospective of prior methods: `evaluation/methods_retrospective.md`
- Baseline failure analysis: `evaluation/failure_modes_default_rloo.md`
- v1 PRM regression analysis: `evaluation/failure_modes_arithprm_v1.md`
- Self-Backtracking paper: arXiv:2502.04404 (Yang et al. 2025)
- Existing inference-loop scaffold (for adapting Step 6):
  `rloo_trainer/context_restoration.py` (built for SELS; same pattern but
  much simpler here — single `<restart>` token, single truncation rule)
- Existing reward modules (NOT used by this plan, kept for reference):
  `rloo_trainer/arithmetic_prm.py`, `rloo_trainer/decisiveness_reward.py`

## When to revisit this plan

- If pass@1 after Step 6 < 0.62 (below baseline + 4): the SFT didn't take.
  Diagnose: was the synthetic data quality bad? Was the masking incorrect?
  Was the SFT undertrained?
- If pass@1 after Step 6 ≥ 0.65: this validates the approach. Iterate
  Step 7 (additional RL refinement).
- If pass@1 saturates near 0.72 after Step 7: that's the pass@16 ceiling
  of v2 step-199 — the natural stopping point for this pipeline. To go
  further, pivot to solver-distillation SFT (a different plan).
