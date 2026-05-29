# Project Journey: Step-by-Step Documentation

A chronological record of every approach we tried on Countdown — including the failed attempts (SELS, restart-on-rambling, decisiveness, naive RFT, IPO v2 ablation), with their motivations, designs, results, and the lessons each one taught.

All numbers below come from the canonical eval setup: `asingh15/countdown_tasks_3to4` test split (50 prompts), `num_responses = 16` per prompt, sampling at `temperature=0.6, top_p=0.95, top_k=20, max_tokens=1024`.

---

## At-a-glance summary table

| # | Method | Outcome | pass@1 | pass@16 | Key insight |
|---|---|---|---|---|---|
| 0 | SFT base | reference | — | — | Starting point |
| 1 | Default RLOO (no PRM) | baseline | 0.58 | 0.72 | Plain RL on `compute_score` |
| 2 | **SELS extension (v1–v5)** | **failed** | ~0.30 (RL) | — | Special-token search structure doesn't transfer through RL at small scale |
| 3 | **Decisiveness reward** | **failed** | (not trained) | — | Premise wrong: "target value reached ≠ correct answer" in Countdown |
| 4 | Arithmetic PRM v1 (100 steps) | partial | 0.46 | — | Verification tax: penalizing arithmetic made model ramble more |
| 5 | **Arithmetic PRM (continued, 200 steps)** | **worked** | 0.56 | **0.76** | Arithmetic mistakes cut −69% (0.89 → 0.28); best pass@16 in project |
| 6 | **Restart-on-rambling SFT v1** | **failed** | 0.485 | — | New `<restart>` token never fires at inference (cold embedding) |
| 7 | **Restart-on-rambling SFT v2** | **failed** | (similar) | — | System prompt + W=20 loss weight still didn't make token fire |
| 8 | RFT v1 (naive: shortest correct) | inconclusive | 0.60 | 0.70 | Low T + length-biased selection killed diversity |
| 9 | RFT v2 (more data, top-2 shortest) | inconclusive | 0.58 | 0.72 | Same flaw, more of it |
| 10 | **RFT v3 (canonical: T=1.0 + top-K distinct)** | **worked** | 0.60 | 0.74 | Validated diversity-preservation hypothesis |
| 11 | **IPO anti-ramble v1** | **headline win** | **0.62** | 0.74 | Contrastive learning on (correct, long-failure) pairs |
| 12 | IPO v2 (chosen also filtered) | ablation, regressed | 0.60 | 0.72 | Confirms length-shortcut hypothesis — chosen filter introduces premature commits |

Final pipeline (kept): **SFT base → Default RLOO → Arith PRM (continued) → RFT v3 → IPO anti-ramble v1**.

---

## Phase 0: Starting state

**SFT base**: A Qwen2.5-0.5B model fine-tuned on `Asap7772/cog_behav_all_strategies` via `sft_trainer/sft.py`. This was provided by the starter project as the warm-start for all subsequent RL.

**Default RLOO**: Trained via `rloo_trainer/rloo.py` on the Countdown train split using the unmodified `compute_score` reward (1.0 = correct, 0.1 = format-correct but wrong, 0 = no answer). Hyperparameters: gs=8, lr=1e-5, KL=0.001, max_tokens=1024.

**Baseline numbers on test (50 prompts × 16 rollouts)**:
- pass@1 = 0.58
- pass@16 = 0.72
- mean arithmetic mistakes per rollout = **0.89**
- % rollouts containing ≥1 arithmetic mistake = **32.2%**
- NO-ANSWER rate = 25.9%
- sticky-repeat (model re-emits same wrong calculation) = 14.2%

The 0.14 consistency gap (pass@16 − pass@1) and the dominant arithmetic-error pattern were the failure-mode signal that everything afterward tried to attack.

---

## Phase 1: SELS extension — *failed*

### Motivation

Before pivoting to PRM, the original idea was the **SELS (Self-Extracting-then-Local-Search)** extension: explicitly structure the reasoning trace with special tokens that mark exploration states and backtrack points. The hypothesis was that giving the model an explicit search grammar would let RL more effectively shape branching/backtracking behavior.

### Approach (iterated through v1–v5)

Added 10 special tokens to the tokenizer:
```
<state_0>, <state_1>, …, <state_4>      ← reasoning states
<backtrack_0>, …, <backtrack_4>          ← backtrack-to-state markers
```

Generated synthetic SFT data where the reasoning trace was structured around these tokens. SFT'd the base model on this data, then RLOO'd on the SFT'd checkpoint.

Five iterations:
- v1, v2: early prototypes
- v3: better state-grammar design
- v4: cleaner backtrack semantics
- v5: best warm-start (pass@1 ≈ 0.263 from SFT alone with `</answer>` stop fix)

### Result

**SFT v5 (best)**: pass@1 ≈ 0.263 with significant residual no-answer rate from backtrack-looping (≈29% of failures).

**SELS + RLOO**: stuck around **pass@1 ≈ 0.30**, vs default RLOO at **0.55–0.58**. RL on top of the SELS-structured model never closed the gap.

### Why it failed

1. **Self-structured search is hard to teach via SFT alone**: the model could mimic the surface form of `<state_i>` / `<backtrack_j>` patterns but didn't learn to use them as *real* control flow during reasoning.
2. **The SFT-only pass@1 (0.263) was already far below default RLOO (0.58)** — RL had to first close a 30pp deficit *before* it could provide value over baseline, which it couldn't.
3. **New-token learning at small scale is hard** (same root issue we'd hit again in Phase 5, Restart): freshly initialized embeddings for `<state_*>`, `<backtrack_*>` carry near-zero signal at initialization, and limited training data can't move them enough to compete with established tokens during sampling.

### Decision

Pivoted away from SELS entirely. Moved to a **default-substrate PRM extension**: keep the standard token vocabulary, add a reward-shaping signal during RLOO instead of changing the generation grammar.

### Files involved

- `sft_trainer/sft.py` (with `--register_sels_tokens` flag, originally added for this)
- `rloo_trainer/train_rloo_sels.sh`, `train_rloo_sels_prm.sh`
- `evaluation/sample_modal_sels_rloo_prm.sh`

These remain in the repo as historical artifacts of the SELS attempt.

---

## Phase 2: Pivot to default-substrate PRM extension

### 2.1 Failure mode analysis

Before designing the PRM, we ran a detailed analysis on the default RLOO model's rollouts to identify which failure modes were *fixable* and which were structural.

Findings (documented in `evaluation/failure_modes_default_rloo.md`):

- **58% of WRONG-VAL rollouts contain at least one invalid arithmetic operation** (e.g., a line saying `63 * 95 = 5885` when `63 * 95 = 5985`)
- **22% of BAD-NUMS rollouts also contain detectable arithmetic mistakes**
- **57% of NO-ANSWER rollouts have ≥1 arithmetic mistake**; mean 2.13 mistakes per NO-ANSWER trace
- **The dominant pattern in NO-ANSWERs: sticky-wrong-fact** — the model emits a wrong calculation, then re-emits the same wrong calculation 4-8 times within one trace before running out of budget

This last finding (sticky-wrong-fact) became the explicit design target.

### 2.2 Decisiveness reward — *failed before training*

In parallel to designing the arithmetic PRM, we considered a **decisiveness reward**: detect when the model's reasoning had numerically reached the target value but then failed to commit (e.g., the trace contained the line `44 + 19 + 35 = 98` for `target=98` but emitted NO-ANSWER).

Initial estimate from a "walk the trace" analysis: ~15.5% of NO-ANSWER rollouts had this pattern → tempting headroom to claw back via reward shaping.

**Problem caught before training**: Countdown requires using *each input number exactly once*, but the naïve walk just looked for the target value appearing anywhere in the trace. In a Countdown task with `numbers=[44, 19, 35]` and `target=98`, the line `19+44+19=82, +35-19=98` would have its target value hit, but the equation uses 19 twice. With strict provenance tracking (each derived value remembers which subset of original numbers it used), the true "target reached AND uses correct multiset" rate dropped from 15.5% → **2.2%**.

Decided not to train: the addressable headroom (2.2%) was too small to justify a separate reward term, and the metric was easy to game (a model could learn to hallucinate trivial number reuse).

**Removed entirely.** No code shipped.

### 2.3 Arithmetic PRM design

Designed `rloo_trainer/arithmetic_prm.py` with the following formula:

```
For each rollout r, scan all (a, op, b, c) patterns where `a OP b = c` appears.
For each pattern, validate: does (a OP b) actually equal c?

n_total_mistakes(r)      = number of invalid (a, op, b, c) tuples
n_unique_mistakes(r)     = distinct invalid tuples
n_repeated_mistakes(r)   = n_total - n_unique  (the sticky-wrong-fact signature)

penalty = β · n_total_mistakes + γ · n_repeated_mistakes
prm     = -min(penalty, cap)             ∈ [-cap, 0]
```

Hyperparameters: β=0.1, γ=0.2, cap=0.5. Calibration:
- 1 distinct mistake → prm = −0.1 (one format_score worth of pressure)
- 4 identical mistakes (sticky) → prm = −0.5 (β·4 + γ·3 = 1.0, capped)
- 5 distinct mistakes → prm = −0.5

Plugged into `rloo_trainer/rloo.py` via an `arithmetic_coef` flag (default 0.0 = no change to existing behavior).

### 2.4 Arithmetic PRM v1 (100 RLOO steps) — *partial, with verification tax*

Trained for 100 RLOO steps from the default RLOO checkpoint with `--arithmetic_coef 1.0`.

**Result**:
- Arithmetic mistakes per rollout: 0.89 → 0.55 (**−37%**)
- BUT: pass@1 regressed from 0.58 → **0.46**
- NO-ANSWER rate **rose**
- The verification tax: penalizing arithmetic mistakes made the model deliberate longer, increasing the chance of running out of tokens

Diagnosed in `evaluation/failure_modes_arithprm_v1.md`. The PRM correctly identified what to suppress, but the downstream behavior change (more deliberation → more rambling) had a larger negative effect on pass@1 than the gain from cleaner arithmetic.

### 2.5 Arithmetic PRM (continued, 200 RLOO steps total) — *worked*

Continued training for another 100 RLOO steps from the v1 final checkpoint. Same reward formulation, same hyperparameters.

**Result** (this checkpoint becomes "Arith PRM cont. latest" throughout the rest of the project):
- pass@1 = **0.56**
- pass@16 = **0.76**  ← best pass@16 in entire project
- Mean arithmetic mistakes per rollout: **0.285** (−68% vs Default RLOO)
- % rollouts with ≥1 mistake: 32.2% → **18.8%** (−42%)
- % sticky-repeat rollouts: 14.2% → **3.0%** (−79%)
- NO-ANSWER rate: 25.9% → 27.3% (slight regression, the residual verification tax)

The pass@1 was 2pp under Default's 0.58 — the verification tax wasn't fully paid back — but pass@16 went UP +4pp (the model is genuinely better at finding answers given enough tries). This became the **shared warm-start** for everything downstream (Restart, RFT, IPO).

### Files involved

- `rloo_trainer/arithmetic_prm.py` (the PRM scoring function)
- `rloo_trainer/rloo.py` (minimal change: added `--arithmetic_coef` flag)
- `rloo_trainer/train_rloo_default_arith_prm.sh` (v1 launcher)
- `rloo_trainer/train_rloo_default_arith_prm_continued.sh` (continuation launcher)
- `evaluation/failure_modes_default_rloo.md`, `failure_modes_arithprm_v1.md`, `methods_retrospective.md`

---

## Phase 3: Restart-on-rambling SFT — *failed, with diagnosis*

### Motivation

After Arith PRM stabilized at pass@1 = 0.56 with NO-ANSWER still ≈ 27%, the next target was **rambling**. Hypothesis: teach the model an explicit `<restart>` token it can emit when it recognizes its own reasoning is going in circles, then continue with a clean retry.

### 3.1 Setup

1. **Sampled training rollouts** from Arith PRM cont. latest:
   - 2000 train prompts × 8 rollouts each (T=0.6) = 16,000 rollouts
   - File: `evaluation/eval_results/train_rollouts_arithprm_continued.json`
   - Per-prompt analysis: ~62% had ≥1 CORRECT rollout

2. **Synthesized restart pairs** (`evaluation/synthesize_restart_sft.py`):
   - For each prompt with both ≥1 CORRECT and ≥1 rambling NO-ANSWER (max_line_repeat ≥ 5):
   - Splice: `<think>\n[rambling head from NO-ANSWER]\n<restart>\n[clean tail from CORRECT]`
   - Up to 2 pairs per prompt
   - **Result**: 737 examples from 484 qualifying prompts

3. **Added `<restart>` as a special token** to the tokenizer; resized embeddings.

### 3.2 Restart-SFT v1 — uniform loss weight

Trained `restart_sft_extension/train_restart_sft.py` (a thin wrapper around `sft_trainer/sft.py`) with:
- Warm-start from Arith PRM cont. latest
- 3 epochs, lr=5e-6, batch_size=16, ga=4 → 129 optimizer steps
- Loss masked over the prompt + rambling head; trained on `<restart>` + clean tail

**Training metrics looked great**:
- Eval loss = 0.215
- Eval acc (teacher-forced next-token) = **0.946**

**At inference**:
- Probe with 30 train prompts × 8 rollouts at the new checkpoint
- **`<restart>` was emitted in 0 / 240 rollouts** — the token never fired

The model fit the training distribution perfectly in teacher-forced eval but **never autoregressively generated id 151665 during sampling**.

### Diagnosis of v1 failure

The freshly-initialized LM head row `W_out[<restart>]` had near-zero inner product with any context hidden state. Established tokens (`"\n"`, `"Wait"`, etc.) have row norms ~1; the new row was ~0.02 after 129 steps of SFT. At T=0.6 / top_p=0.95 / top_k=20, the new token never made it into the top-k sampled set.

Rough math: per-step contribution to `W_out[151665]` ≈ `lr × (1 − P) × h ≈ 5e-6 × 35 ≈ 1.75e-4`. After 129 SFT steps with one boundary token per row, the total magnitude added is ~0.02 — far below the ~1.0 norm of competing tokens.

### 3.3 Restart-SFT v2 — system prompt + per-token loss weight

Two interventions:

1. **System prompt augmentation**: added a one-line instruction naming the `<restart>` token:
   > "You are a helpful assistant. If you notice you are stuck, repeating yourself, or making no progress toward the answer, emit a single `<restart>` token to abandon the current attempt and start fresh."

   This makes the literal `<restart>` token appear in every input's context, giving the model an in-context attention anchor.

2. **Loss weight on `<restart>` position**: changed `is_response_token` from binary {0, 1} to a float weight where the `<restart>` boundary position got weight **W=20** vs weight 1.0 for tail tokens. With ~50 tail tokens, this made the `<restart>` NLL contribute ~30% of the per-row loss (vs ~2% under uniform weighting).

**Training metrics**:
- Eval loss = 0.931 (higher — expected: the boundary position now dominates)
- Decomposed: tail NLL stayed ~0.21, `<restart>` NLL ≈ 3.8 → P(`<restart>` | training context) ≈ 2.2%

**At inference**:
- **STILL 0 / 240 emissions of `<restart>`**
- Some behavioral change: NO-ANSWER dropped 19% → 3%, but in the wrong way — BAD-NUMS doubled. The model became *over-decisive*, committing to wrong answers instead of emitting `<restart>`.
- pass@1 regressed to **0.485** (worse than cont. latest's 0.56)

### Diagnosis of v2 failure

The augmented system prompt made `<restart>` salient in context, and the per-token loss weight pushed the embedding harder, but **at the actual rambling-state hidden vectors the model produces at inference, `W_out[<restart>] · h` still wasn't competitive** against established tokens. 700 examples × W=20 × 1 epoch wasn't enough.

The decisiveness shift (NO-ANSWER 19% → 3%, BAD-NUMS doubled) was the model finding a *different* surface response to the rambling-context gradient: rather than emit a hard-to-sample new token, it learned to commit to *something* earlier — including wrong answers. This is a classic case of training pushing the policy in a direction that minimizes loss but not in the direction the experimenter intended.

### 3.4 Verification: not a vLLM bug

To rule out an "is sampling stripping the token?" hypothesis:
- Inspected the saved tokenizer files: `<restart>` correctly registered as id 151665 in `added_tokens.json`, `special_tokens_map.json`, and `tokenizer_config.added_tokens_decoder` with `special: True`
- Verified vLLM's `skip_special_tokens=False` setting in our sampling code → tokens are preserved in decoded output
- Round-trip test: encoded a synthetic text containing `<restart>`, decoded with `skip_special_tokens=False`, the literal `<restart>` string was preserved
- Scanned 240 v2 probe rollouts for any restart-adjacent string (`<restart>`, `<restart`, `restart`, `Restart`, `RESTART`) — found nothing in any form

The token was not stripped or hidden — the model genuinely never sampled id 151665.

### Decision

Removed the restart codebase entirely:
- Deleted `restart_sft_extension/` directory
- Removed `run_restart_sft` from `modal_train.py`
- Removed `--augment_system_prompt` from `evaluation/sample_train_rollouts.py`
- Kept `evaluation/restart_on_rambling_plan.md` as documentation of the failed approach
- Deleted training data artifacts (`restart_sft_data.jsonl`) and probe outputs

### Lessons

1. **Teacher-forced eval acc is a misleading metric for new-token training**: 0.946 acc on a balanced position-weighted dataset can correspond to ~0% inference firing rate when the "new token" position is one of many in the loss aggregation.
2. **Inference is dominated by competition between tokens at each position**: getting `P(new_token) > 0.05` in autoregressive sampling requires moving an LM head row to compete with already-aligned rows. SFT on 700 examples isn't sufficient at this model scale.
3. **When a new behavior is hard to teach at sampling time**, the model often finds a *different* surface response to the gradient signal — in our case, becoming over-decisive instead of self-restarting. The training "worked" in that it changed behavior; it just didn't change behavior in the intended way.
4. **The canonical literature solution is embedding initialization** (e.g., init `W_out[<restart>]` from `W_out[<|im_start|>]` or similar). We did not test this — the analysis below (Phase 4) made the pivot to RFT more attractive.

---

## Phase 4: Rejection-sampling Fine-Tuning (RFT)

### Motivation

After Restart's failure, we revisited the underlying goal: **distill pass@16 into pass@1**. The Arith PRM model had a 0.20 consistency gap (pass@16 0.76 − pass@1 0.56) — the model knew how to find the answer in some of the 16 tries but committed to wrong/no answer on the first try.

The canonical method for this gap is **Rejection-sampling Fine-Tuning** (Yuan et al. 2023; STaR; BoN distillation):
1. Sample K rollouts per prompt
2. Filter to correct ones
3. SFT on those (prompt, correct rollout) pairs

This is conceptually simpler than the restart approach (no new tokens, no self-monitoring) and has a clean theoretical motivation: amplify modes that already work.

### 4.1 RFT v1 (naive) — *inconclusive*

**Synthesis** (`rft_extension/synthesize_rft_data.py`):
- Input: existing 16K rollouts from Arith PRM cont. latest (T=0.6, n=8 per prompt)
- Per prompt with ≥1 CORRECT: keep the **shortest** correct rollout
- Result: 1,170 training examples (62% of 2000 prompts had ≥1 correct)

**Training**:
- Warm-start: Arith PRM cont. latest
- 1 epoch, lr=5e-6, batch_size=16, ga=4 → 73 optimizer steps
- Reused `sft_trainer/sft.py` directly (the data format matched its default `query` / `completion` schema)

**Result**:
- pass@1 = 0.60 (vs Arith PRM 0.56, +4pp)
- pass@16 = 0.70 (vs Arith PRM 0.76, **−6pp regression**)
- Median rollout length: 513 chars (vs Arith PRM 632 — model became terser)

### 4.2 RFT v2 (more data, top-2 shortest) — *inconclusive*

Hypothesis: maybe v1's 1170 examples was just too small. Scale up.

- Sampled 6,000 more train prompts (idx 2000-7999) at T=0.6 / n=8 → 48K more rollouts
- Combined p1 + p2 = 64K rollouts from 8000 prompts
- Per prompt: top-2 shortest correct (vs v1's top-1)
- Result: 9,093 training examples (8× v1's size)

**Training**: same hyperparams. 568 optimizer steps.

**Result**:
- pass@1 = 0.58 (vs v1's 0.60, marginally worse)
- pass@16 = 0.72 (vs v1's 0.70, slight recovery)
- 8× the data did NOT improve pass@1

### Diagnosis of RFT v1/v2 failure

Two methodological issues identified after a careful read of the literature and the actual data:

1. **Sampling temperature 0.6 was too low for diversity**: at T=0.6 with 8 rollouts per prompt, the rollouts are token-distinct (median 7/8 unique strings) but their **reasoning content is highly redundant**. Most correct rollouts on a prompt explore the same path with minor variation. We were training on near-duplicate reasoning patterns.

2. **"Shortest correct" selection is anti-diversity by construction**: the shortest correct rollouts tend to be the most templated (fewest exploration steps before committing). Picking them concentrates training on a narrow mode.

The combined effect: even with 9K training examples (v2), the *effective* diversity was ~1 reasoning pattern per prompt, with the model trained on it 8 times over.

Empirical signature: pass@16 dropped (the policy collapsed onto a narrow mode), median rollout length dropped (the model learned the terse template), NO-ANSWER rose (when the template didn't work, the model had nothing else to try).

### 4.3 RFT v3 (canonical) — *worked*

Two changes to address the diagnosis:

1. **Sample at T=1.0 with n=16** (instead of T=0.6 / n=8): forces more diverse trajectories per prompt at the sampling stage
2. **Per-prompt selection: top-K=5 distinct random correct** (instead of "single shortest"): keeps multiple genuinely different reasoning paths

`rft_extension/synthesize_rft_data.py` updated with `--top_k_distinct` flag that:
1. Dedupes correct rollouts by exact text match
2. Randomly samples K from the distinct set (deterministic via `--seed`)
3. No length-based sorting

**Synthesis result**: 5,849 train examples (vs v2's 9093 from same-prompt count) — fewer examples but more diverse. Median 14 of 16 sampled correct rollouts were token-distinct, vs 7 of 8 at T=0.6.

**Training**: same hyperparams as v1/v2.

**Eval**:
- pass@1 = 0.60 (same as v1)
- pass@16 = **0.74** (+4pp recovery over v1's 0.70)
- Median rollout length: 621 chars (natural length restored vs v1's 513)

The methodological insight: **diversity preservation in RFT data construction is a necessary condition, not optional, at small model scale.** The pass@1 didn't increase further over v1, but the pass@16 regression was largely undone — the policy stayed broader.

### Files involved

- `rft_extension/synthesize_rft_data.py` (with `--top_k`, `--top_k_distinct`, `--seed`)
- `rft_extension/sample_rft_train_rollouts_v3.sh` (T=1.0 / n=16 sampling)
- `rft_extension/train_rft.sh`, `train_rft_v2.sh`, `train_rft_v3.sh`
- `rft_extension/eval_rft.sh`, `eval_rft_v2.sh`, `eval_rft_v3.sh`
- `rft_extension/analyze_compare.py` (multi-model comparison tool)

---

## Phase 5: IPO anti-ramble

### Motivation

After RFT v3, pass@1 was at 0.60 and NO-ANSWER was at 29.6% — RFT had partly recovered diversity but didn't fix the rambling failure mode. RFT is a positive-only signal: it tells the model "do this" but never "don't do that."

Switched to **contrastive preference learning**: train on (chosen, rejected) pairs where chosen = correct and rejected = rambling-failure for the same prompt. This is the canonical setup for IPO (Azar et al. 2024) / DPO (Rafailov et al. 2023).

### 5.1 Pair design — critical decision

After extensive discussion, settled on the principled filter:

```
chosen pool   = compute_score == 1.0                      (any length, no further filter)
rejected pool = NOT correct  AND  len(text) >= 1200       (long failure, "rambling")
```

The 1200-char threshold came from empirical length-distribution analysis on 64K rollouts:
- CORRECT p95 length = 931 chars
- NO-ANSWER p10 length = 1695 chars
- There's a natural ~760-char "gap" between successful reasoning and rambling failure
- 1200 sits squarely in the gap

Choice **not** to filter the chosen pool by length: doing so would create a surface "shorter is better" shortcut that IPO would learn instead of the actual rambling signature (this prediction was tested empirically in §5.4 below).

### 5.2 Data synthesis

`ipo_extension/synthesize_ipo_pairs.py`:
1. Sampled RFT v3's *own* rollouts: 2000 prompts × 16 rollouts at T=1.0
   - **Critical**: sampling from RFT v3 itself, not from cont. latest. IPO's reference policy is the warm-start, and the pairs should reflect *that policy's* failure modes.
2. For each prompt with ≥1 chosen + ≥1 rejected:
   - Dedupe both pools by exact text
   - Random sample top-K=2 distinct from each pool
   - Pair index-by-index

**Result**: 1,131 train pairs / 59 test pairs from 720 qualifying prompts (36% of 2000 source prompts). Rejected distribution: 72% NO-ANSWER, 16% BAD-NUMS, 12% WRONG-VAL.

### 5.3 IPO v1 training — *headline win*

`ipo_extension/train_ipo_anti_ramble.py` (wrapper around the unmodified `ipo_trainer/ipo.py`):

- Warm-start (and reference policy): RFT v3
- Loss type: IPO (squared), β = 0.1
- 1 epoch, lr=5e-6, batch_size=16, ga=4 → 70 optimizer steps

**Training metrics**:
- Train loss = 7.88, eval loss = 11.52 (IPO loss target is `(h − 1/(2β))² = (h − 5)²`)
- Eval reward margin (`β·h`) = 0.42 → mean h ≈ 4.2 (target 5) — converging properly

**Eval result** (the headline):
- **pass@1 = 0.62** (best across all 7 trained models)
- pass@16 = 0.74 (preserved from RFT v3, no regression — KL anchor working)
- NO-ANSWER = **25.4%** (lowest of any model; *below* Default RLOO baseline of 25.9%)
- Severe loops (≥10 line-repeats) = **15.1%** (lowest)
- CORRECT % per rollout = **61.1%** (highest)
- Slight arithmetic regression: mean mistakes 0.30 → 0.40 (still −55% vs Default's 0.89)

**Mechanism**: IPO's contrastive gradient explicitly pushed the policy distribution *toward* CORRECT trajectories and *away from* long-rambling failures. The KL anchor to RFT v3 prevented the policy from collapsing while it did so. The arithmetic regression is the cost of more attempts actually committing (fewer NO-ANSWERs → more rollouts that commit, some of those expose latent arithmetic errors).

### 5.4 IPO v2 (chosen also filtered) — *ablation, regressed*

Hypothesis to test: would also engineering a length contrast on the chosen side help (push median rollout length down further)?

`synthesize_ipo_pairs.py` updated with:
- `--max_chosen_length 600` → require chosen rollouts to be ≤ 600 chars (vs CORRECT median of 433, so this keeps the natural shorter tail and drops the longer-correct rollouts)
- `--clean_chosen` → require `n_total_mistakes == 0` per `compute_arithmetic_prm` (so chosen has perfectly clean step-by-step calculations)

**Synthesis stats**: 876 train / 46 test pairs (vs v1's 1131). Only ~5% of CORRECT rollouts had arithmetic mistakes (the warm-start was already Arith-PRM trained), so the length filter is the dominant effect.

**Eval result**:
- pass@1 = 0.60 (**−2pp regression vs v1**)
- pass@16 = 0.72 (−2pp regression)
- median chars 593 → 576 (shorter as designed)
- BUT **WRONG-VAL = 7.6% (vs v1's 4.8%, +2.8pp regression)** — premature commits
- Catastrophic line-repeat went *up* slightly (32.4% → 35.1%, losing the v1 anti-loop gain)
- Arithmetic improved slightly (0.40 → 0.34 mean mistakes — the chosen filter transferred)

**Diagnosis**: this is the length-shortcut failure mode the v1 design avoided. When IPO sees `chosen = always short, rejected = always long`, the dominant gradient signal is "be shorter." The model becomes shorter even on prompts where committing would be wrong. NO-ANSWER barely moved (25.4% → 26.1%) — the model didn't really ramble less; it just committed earlier on the same wrong answers it would have rambled toward.

**Conclusion**: **IPO v1's design choice — filter only the rejected pool, leave chosen unfiltered — was the correct principled call.** v2 is the empirical confirmation that engineering length contrast hurts more than it helps.

### Files involved

- `ipo_extension/synthesize_ipo_pairs.py` (with `--min_rambling_length`, `--max_chosen_length`, `--clean_chosen`)
- `ipo_extension/ipo_dataset_local.py` (subclasses `ipo_trainer.ipo_dataset.IPODataset` to load via `load_from_disk` instead of HF hub)
- `ipo_extension/train_ipo_anti_ramble.py` (wrapper around `ipo_trainer.ipo`)
- `ipo_extension/train_ipo_anti_ramble.sh`, `eval_ipo_anti_ramble.sh` (v1)
- `ipo_extension/train_ipo_anti_ramble_v2.sh`, `eval_ipo_anti_ramble_v2.sh`
- `ipo_extension/sample_rft_v3_rollouts.sh` (sample RFT v3's own rollouts)

---

## Phase 6: Analysis tooling and writeup

### Comparison framework

`rft_extension/analyze_compare.py` — a single script that loads all model eval JSONs from `evaluation/eval_results/` and computes a side-by-side comparison table of:

- pass@1, pass@16, consistency gap
- Outcome distribution (CORRECT / NO-ANSWER / BAD-NUMS / WRONG-VAL)
- Length distribution (median, p95, max)
- Rambling proxies (catastrophic line-repeat at thresholds 2, 5, 10)
- Arithmetic correctness (mean ops, mean mistakes, % w/ mistake, mean repeated mistakes, % sticky-repeat)

This is the source of truth for the comparison tables in this document and in `final_analysis.md`.

### Documentation

- `evaluation/failure_modes_default_rloo.md` — initial failure-mode analysis on Default RLOO
- `evaluation/failure_modes_arithprm_v1.md` — diagnosis of Arith PRM v1's verification tax
- `evaluation/methods_retrospective.md` — early notes on decisiveness reward / restart approaches
- `evaluation/restart_on_rambling_plan.md` — design doc for the (subsequently failed) restart-SFT attempt
- `evaluation/final_analysis.md` — the polished cross-model comparison with mathematical metric definitions
- `evaluation/project_journey.md` — this document

---

## Cross-cutting lessons

Pulling out the recurring themes across the failed attempts:

### Lesson 1: New tokens are very hard to teach at small model scale

Both **SELS** (10 new tokens, never effectively used) and **Restart** (1 new token, never fired at inference) ran into the same wall: a freshly-initialized LM head row needs *substantial* gradient signal to compete with established tokens during sampling. At our scale (700-2000 examples, single-epoch SFT), the gradient signal is ~10-100× too small to make new tokens competitive at sampling.

The canonical literature remedy is **embedding initialization** from a related existing token, which we did not test. This remains an open avenue.

### Lesson 2: Premise-check before optimization

The **decisiveness reward** was a several-day implementation effort that got caught at the design-review stage because the metric ("target value reached") didn't match the actual success criterion ("equation using each number exactly once"). The provenance-strict analysis showed the addressable headroom was 7× smaller than the naive estimate.

Lesson: when defining a reward shaping target, audit whether the proxy metric actually tracks the goal under all the constraints the task imposes.

### Lesson 3: Diversity preservation is a necessary condition for RFT at small scale

**RFT v1 and v2** both used length-biased selection and low-temperature sampling. Both regressed pass@16 by 4-6pp. **RFT v3** swapped these for T=1.0 sampling + top-K distinct selection and recovered most of the regression. The diversity preservation was the difference between a working RFT and a failing one.

### Lesson 4: Engineering surface contrasts in preference data backfires

**IPO v1** filtered only the rejected pool by length, leaving the chosen pool diverse. This gave the cleanest signal: the model learned "don't ramble" without learning "always be short."

**IPO v2** *also* filtered the chosen pool to short + clean. Result: pass@1 regressed by 2pp because the model learned to be short *as the primary signal*, leading to premature commits on uncertain prompts.

Lesson: in preference learning, the gradient finds the most accessible surface contrast between the two pools. If you engineer that contrast (e.g., length), the model will learn the surface shortcut rather than the underlying property you wanted.

### Lesson 5: Failed attempts with diagnoses are themselves contributions

The SELS, Restart, naive-RFT, and IPO-v2 failures each came with a clear mechanistic explanation. For a class project, these diagnoses (and the methodological prescriptions they imply: embedding init for new tokens, diversity preservation for RFT, non-engineered contrast for IPO) are arguably more interesting than the positive results — they generalize beyond the specific task.

---

## Final pipeline summary

```
SFT base (Qwen2.5-0.5B on cog_behav_all_strategies)
   │
   ▼
+ RLOO (default)
   pass@1 0.58,  pass@16 0.72,  arith mistakes 0.89,  NO-ANSWER 25.9%
   │
   ▼
+ Arithmetic PRM (continued, 200 RL steps)               ← Phase 2.5
   pass@1 0.56,  pass@16 0.76,  arith mistakes 0.285 (−68%),  NO-ANSWER 27.3%
   │
   ▼
+ Canonical RFT (T=1.0, n=16, top-K=5 distinct)          ← Phase 4.3
   pass@1 0.60,  pass@16 0.74,  arith mistakes 0.30
   │
   ▼
+ IPO anti-ramble v1 (chosen=CORRECT, rejected=long-fail) ← Phase 5.3
   pass@1 0.62,  pass@16 0.74,  arith mistakes 0.40,  NO-ANSWER 25.4%
   │
   ╔══════════════════════════════════════════════════════╗
   ║   FINAL DELIVERABLE: pass@1 = 0.62, NO-ANSWER 25.4%  ║
   ║   +4pp pass@1 over Default RLOO baseline              ║
   ║   −62% sticky-repeat (Arith PRM contribution)         ║
   ║   −55% mean arith mistakes vs Default                 ║
   ╚══════════════════════════════════════════════════════╝
```

**Failed attempts kept as artifacts** (with `evaluation/*.md` documentation):

- SELS extension (Phase 1)
- Decisiveness reward (Phase 2.2 — caught before training)
- Restart-on-rambling SFT v1 & v2 (Phase 3)
- RFT v1 & v2 (Phase 4.1, 4.2 — superseded by v3)
- IPO v2 (Phase 5.4 — ablation that validates v1 design)

Each was kept in the report rather than deleted because the diagnoses are part of the contribution.
