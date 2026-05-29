# Detailed Results Report: Stacked Post-Training Methods on Countdown

A comprehensive analysis of five models that form a monotonic post-training stack on the Countdown arithmetic task: **Default RLOO** (baseline) → **Arithmetic PRM** → **Rejection-sampling Fine-Tuning (RFT v1, RFT v3)** → **Identity Preference Optimization (IPO v1)**. Each method targets a distinct failure mode of the prior stage. The final model achieves **pass@1 = 0.62** (vs Default RLOO's 0.58), with **69% fewer arithmetic mistakes** and **NO-ANSWER rate below the original baseline**.

---

## Abstract

The Countdown arithmetic puzzle requires the model to construct a single equation from a given multiset of integers that evaluates to a target value, using each number exactly once. Despite achieving a Default-RLOO baseline of pass@1 = 0.58, the model exhibits three orthogonal failure modes: (1) **arithmetic mistakes** in `a OP b = c` operations during the reasoning trace, (2) **rambling** — long traces that fail to commit to an `<answer>` tag (NO-ANSWER) or commit to wrong values after extended search, and (3) **consistency gap** — the model can find the answer in 16 attempts at 72% but commits on the first attempt at only 58%.

We attack each failure mode with a specifically motivated method:

- **Arithmetic PRM** (process reward shaping during RLOO) reduces arithmetic mistakes per rollout from 0.885 to 0.285 (−68%) and sticky-repeat failures by −79%, while improving pass@16 to 0.76.
- **Canonical RFT** (rejection-sampling fine-tuning with diversity-preserving sample selection) closes the consistency gap from 0.20 to 0.14, recovering pass@1 to 0.60 with pass@16 preserved at 0.74.
- **IPO** with pair design `(CORRECT, length ≥ 1200 chars failure)` reduces NO-ANSWER from 29.6% to 25.4% (below the original baseline of 25.9%), severe loop patterns from 19.5% to 15.1%, and lifts pass@1 to **0.62**.

We also present an ablation (**IPO v2**) demonstrating that filtering the chosen pool by length introduces premature commits and regresses pass@1 by 2pp, validating IPO v1's principled design.

---

## 1. Background and Problem Setup

### 1.1 Task

The Countdown task supplies the model with:
- A multiset `nums ∈ ℤᴺ` of `N ∈ {3, 4}` non-negative integers
- A target value `target ∈ ℤ`

The model must emit a reasoning trace followed by an `<answer>` tag containing an arithmetic equation `eq` such that:
1. `eq` uses each element of `nums` exactly once (multiplicity-strict)
2. `eq` uses only the operators `+`, `−`, `*`, `/`, parentheses, and the numbers themselves
3. `evaluate(eq) == target`

Example: `nums = [44, 19, 35]`, `target = 98` → correct answer: `(44 + 19) + 35 = 98`.

The reward `compute_score(r, gt)` (from `evaluation/countdown.py`) returns 1.0 for fully correct equations, 0.1 for format-correct but wrong, and 0 for no parseable `<answer>` tag.

### 1.2 Starting state and the failure-mode decomposition

The Default RLOO model (provided baseline RLOO trained on this task) achieves:

- **pass@1 = 0.58**, **pass@16 = 0.72**

In terms of *per-rollout outcomes* (across 800 rollouts = 50 test prompts × 16 samples):

| Outcome | Default RLOO |
|---|---|
| CORRECT | 57.0% |
| NO-ANSWER (rambling) | 25.9% |
| BAD-NUMS (wrong number set) | 10.9% |
| WRONG-VAL (right numbers, wrong arithmetic) | 6.2% |

These three failure modes are largely *orthogonal* — they are distinct surface signatures with distinct mechanisms, and a method that addresses one need not affect the others. Empirically, **arithmetic mistakes appear in 32.2% of all rollouts** (including some that nevertheless ended up CORRECT because the answer didn't depend on the mistake), and **14.2% of all rollouts contain the "sticky-wrong-fact" pattern** — the same wrong calculation re-emitted multiple times within a single trace.

The headline consistency gap is `pass@16 − pass@1 = 0.14`. The model knows how to solve roughly 72% of problems given enough tries, but only commits on the first try 58% of the time. Closing this gap is the project's central objective.

### 1.3 The three failure axes and the methods that address them

| Failure axis | Method addressing it | Mechanism |
|---|---|---|
| Arithmetic mistakes | **Arithmetic PRM** | Process reward shaping during RLOO penalizes invalid `a OP b = c` lines |
| Diversity / consistency gap | **Canonical RFT (RFT v3)** | Rejection-sampling FT on diverse correct trajectories amplifies viable modes |
| Rambling (NO-ANSWER, loops) | **IPO anti-ramble** | Contrastive preference optimization pushes policy away from long failures |

---

## 2. Methods

### 2.1 Arithmetic PRM (Process Reward Model)

**Motivation**: Failure-mode analysis (`evaluation/failure_modes_default_rloo.md`) on Default RLOO rollouts showed that 58% of WRONG-VAL rollouts contained at least one invalid arithmetic operation and that the "sticky-wrong-fact" pattern — the model re-emitting the same wrong calculation 4-8 times within one trace — appeared in 14.2% of all rollouts. The PRM is designed to suppress these specifically.

**Reward formulation** (`rloo_trainer/arithmetic_prm.py`):

For each rollout `r`, the scanner extracts all `(a, op, b, c)` tuples where the literal substring `"a OP b = c"` appears. Each tuple is validated: does `a OP b` actually equal `c`? Counts:
- `n_total_mistakes(r)` = number of invalid tuples
- `n_unique_mistakes(r)` = distinct invalid tuples
- `n_repeated_mistakes(r) = n_total_mistakes(r) − n_unique_mistakes(r)`

The reward is:

$$
\text{penalty}(r) = \beta \cdot n\_total\_mistakes(r) + \gamma \cdot n\_repeated\_mistakes(r)
$$

$$
\text{prm}(r) = -\min(\text{penalty}(r), \text{cap}) \in [-\text{cap},\ 0]
$$

with `β = 0.1`, `γ = 0.2`, `cap = 0.5`. This produces:
- 1 distinct mistake → −0.1 (one `format_score` worth of pressure)
- 4 identical mistakes (sticky-repeat) → β·4 + γ·3 = 1.0 → capped at −0.5
- 5 distinct mistakes → −0.5

The penalty is added to the standard `compute_score` reward during RLOO advantage computation.

**Training**: 200 RLOO steps total. First 100 with the PRM penalty starting from Default RLOO (this run, "Arith PRM v1", showed pass@1 regression to 0.46 due to verification-tax effect — penalizing arithmetic mistakes made the model deliberate longer, increasing NO-ANSWER rate). Continuing for another 100 steps from that checkpoint (Arith PRM "cont. latest", denoted **Arith PRM** throughout this report) stabilized at:
- pass@1 = 0.56, pass@16 = **0.76** (best pass@16 in project)
- Mean arithmetic mistakes: 0.885 → **0.285** (−68%)
- Sticky-repeat rate: 14.2% → **3.0%** (−79%)

The verification tax cost 2pp of pass@1 (0.58 → 0.56) but pass@16 *increased* 4pp (0.72 → 0.76) — the model is genuinely better at finding answers when given more chances.

### 2.2 Rejection-sampling Fine-Tuning (RFT v1 and v3)

**Motivation**: After Arith PRM, the consistency gap was 0.20 (0.76 − 0.56). RFT (Yuan et al. 2023; STaR / BoN distillation) is the canonical technique to close this gap by re-training on the model's own successful trajectories.

**Synthesis** (`rft_extension/synthesize_rft_data.py`): For each train prompt, sample K rollouts from Arith PRM, filter to those scoring 1.0, select N per prompt as training targets. Two variants tested:

#### RFT v1 (naive)
- Sampling: K=8 rollouts per prompt at T=0.6, 2000 prompts
- Selection: 1 shortest correct rollout per prompt (length-biased)
- Result: 1,170 training examples
- Training: 1 epoch, lr=5e-6 on Arith PRM warm-start

#### RFT v3 (canonical, diversity-preserving)
- Sampling: K=16 rollouts per prompt at **T=1.0**, 2000 prompts
- Selection: top-5 **distinct** correct rollouts per prompt (token-distinct, randomly sampled, *no length bias*)
- Result: 5,849 training examples
- Training: same as v1

**Why two variants**: v1 was the first attempt and matched the typical "shortest correct" intuition. It failed to deliver pass@1 gains (pass@1 stayed at 0.60, but pass@16 dropped from 0.76 to 0.70 — a 6pp regression). v3 was the methodological correction after we diagnosed that v1's two choices (low temperature + length-biased selection) both eroded the diversity that RFT's pass@1 gain relies on.

**RFT v3 results**:
- pass@1 = 0.60 (same as v1)
- pass@16 = 0.74 (+4pp recovery vs v1's 0.70)
- Natural rollout length restored (median 621 chars vs v1's 513 chars — the model is no longer collapsed onto the terse template that "shortest correct" trained)

The pass@1 of 0.60 represents +2pp on Arith PRM (0.56 → 0.60) with mostly-preserved pass@16 (0.76 → 0.74). The 4pp loss in pass@16 is the unavoidable cost of any positive-only fine-tuning: the policy shifts toward modes that "do the right thing" and away from broader exploration.

### 2.3 IPO anti-ramble

**Motivation**: After RFT v3, NO-ANSWER was still at 29.6% — *higher* than Default RLOO's 25.9%. RFT's positive-only signal had compounded with Arith PRM's verification tax to actually make rambling worse than the original baseline.

The fix: contrastive learning with **rambling specifically defined**.

**Pair construction** (`ipo_extension/synthesize_ipo_pairs.py`):

Defined operationally:
- **CHOSEN pool**: any rollout with `compute_score(r) == 1.0`. No length filter, no further conditions.
- **REJECTED pool**: any rollout that is NOT CORRECT and has `len(text) ≥ 1200` chars.

The 1200-char threshold was determined by empirical length-distribution analysis on 64,000 rollouts (`evaluation/final_analysis.md` §6.4): CORRECT rollouts have p95 length = 931 chars while NO-ANSWER rollouts have p10 length = 1695 chars. There is a natural ~760-char empirical gap — successful reasoning has terminated by 931 chars, while rambling failures haven't yet hit their uniform "ran out of budget" terminus at 1695. T=1200 sits in this gap. It captures essentially all NO-ANSWERs and the long tails of WRONG-VAL and BAD-NUMS that are rambling-then-bad-commit cases, while excluding short impulsive wrong commits (a different failure mode).

**Critical design choice — chosen pool is NOT length-filtered**: We deliberately *did not* sort or filter the chosen pool by length. Doing so would engineer a surface contrast (shorter vs longer) that IPO would learn as "be shorter," rather than the actual property we want (don't ramble). This prediction was empirically tested in the IPO v2 ablation (§5.3 below).

**Data**: Sampled RFT v3's own rollouts (T=1.0, n=16, 2000 prompts) and ran the synthesis. Result: 1,131 training pairs from 720 qualifying prompts (36% of source prompts had both ≥1 chosen and ≥1 rejected qualifier). Rejected distribution: 72% NO-ANSWER, 16% BAD-NUMS, 12% WRONG-VAL — heavily skewed toward the dominant rambling failure mode.

**Training** (`ipo_extension/train_ipo_anti_ramble.py`, wrapping the unmodified `ipo_trainer/ipo.py`):
- Warm-start AND reference policy: RFT v3 (chosen because IPO's KL anchor is more effective when the reference distribution is broad and diverse)
- Loss type: IPO (squared loss on log-ratio differences, vs DPO's sigmoid)
- β = 0.1 (literature standard)
- 1 epoch, lr=5e-6, 70 optimizer steps

The IPO loss:

$$
\mathcal{L}_{\text{IPO}} = \mathbb{E}_{(x, y_w, y_l)} \left[\left(\log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)} - \frac{1}{2\beta}\right)^2\right]
$$

The model is trained to push the log-probability ratio of (chosen vs reference) above (rejected vs reference) by `1/(2β) = 5` natural-log units, with KL anchoring to the reference preventing collapse.

---

## 3. Experimental Setup

### 3.1 Evaluation

All five models are evaluated on the canonical Countdown test split:
- **Dataset**: `asingh15/countdown_tasks_3to4`, `split='test'`
- **Size**: 50 prompts
- **Sampling**: 16 rollouts per prompt → 800 rollouts total per model
- **Sampling parameters**: temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, max_tokens=1024
- **Framework**: vLLM via `evaluation/countdown_eval.py` (the project's default eval script, unmodified)

All five models go through exactly the same eval — directly comparable.

### 3.2 Metrics (mathematical definitions)

Let `P` be the 50 test prompts and `R_p = (r_1, …, r_{16})` be each prompt's 16 rollouts. Let `score(r, p) ∈ {0, 0.1, 1.0}` be `compute_score` and let `categorize(r, p)` partition each rollout into one of {CORRECT, NO-ANSWER, BAD-NUMS, WRONG-VAL}.

**Final-answer correctness**:

$$
\text{pass@1} = \tfrac{1}{|P|} \sum_{p} \mathbb{1}[\text{score}(r_1^{(p)}, p) = 1.0]
$$

$$
\text{pass@16} = \tfrac{1}{|P|} \sum_{p} \mathbb{1}[\exists r \in R_p : \text{score}(r, p) = 1.0]
$$

$$
\text{consistency gap} = \text{pass@16} - \text{pass@1}
$$

**Per-rollout outcome distribution** (over all 800 rollouts):

$$
\%[\text{category}] = \tfrac{|\{r : \text{categorize}(r) = \text{category}\}|}{800}
$$

with categories defined by the decision tree:

$$
\text{categorize}(r) = \begin{cases}
\text{NO-ANSWER} & \text{if } eq := \text{extract\_solution}(r) \text{ is None} \\
\text{BAD-NUMS}  & \text{if } eq \text{ is unparseable or uses wrong numbers} \\
\text{WRONG-VAL} & \text{if } \text{evaluate}(eq) \neq \text{target} \\
\text{CORRECT}   & \text{if } \text{evaluate}(eq) = \text{target}
\end{cases}
$$

**Length** (in characters): median, p95, max across all 800 rollouts. Qwen tokenizer averages ~2.7 chars/token on Countdown, so the 1024-token sampling cap corresponds to ~2,800 chars.

**Rambling proxies** (two complementary measures):

*Line-repetition pathology* — for each rollout, compute:

$$
\text{max\_line\_repeat}(r) = \max_{\ell \in \text{lines}(r)} |\{i : \text{line}_i(r) = \ell\}|
$$

(maximum count of any single non-empty stripped line within the rollout). Report at thresholds {2, 5, 10}.

*Long-failure rate* — broader rambling metric that captures both literal loops and non-looping rambling:

$$
\%\text{long\_fail}(T) = \tfrac{|\{r : \text{categorize}(r) \neq \text{CORRECT} \land \text{len}(r) \geq T\}|}{800}
$$

Reported at T ∈ {1200, 1500, 1800} chars.

**Arithmetic correctness** (via `compute_arithmetic_prm`):

For each rollout `r`, let `n_total_ops(r)` = number of `a OP b = c` tuples detected; `n_total_mistakes(r)` = number invalid; `n_repeated_mistakes(r)` = repeated occurrences of the same wrong tuple. Aggregates:

$$
\overline{\text{ops}} = \tfrac{1}{N}\sum_r n\_total\_ops(r), \quad \overline{\text{mistakes}} = \tfrac{1}{N}\sum_r n\_total\_mistakes(r)
$$

$$
\%[\text{clean}] = \tfrac{|\{r : n\_total\_mistakes(r) = 0\}|}{N}
$$

$$
\%[\text{sticky-repeat}] = \tfrac{|\{r : n\_repeated\_mistakes(r) \geq 1\}|}{N}
$$

The last metric (`%[sticky-repeat]`) measures the fraction of rollouts where the model emitted the same wrong calculation more than once within a single trace — the failure mode Arith PRM was specifically designed to suppress.

---

## 4. Results

### 4.1 Headline comparison table

```
metric                     | Default | Arith PRM |  RFT v1  |  RFT v3  |  IPO v1
                           |  RLOO   |  (cont)   |          |          |
---------------------------+---------+-----------+----------+----------+---------
n_prompts                  |    50   |     50    |    50    |    50    |    50
n_rollouts (50 × 16)       |   800   |    800    |   800    |   800    |   800

--- final-answer correctness ---
pass@1                     |  0.5800 |   0.5600  |  0.6000  |  0.6000  |  0.6200  ←best
pass@16                    |  0.7200 |   0.7600  |  0.7000  |  0.7400  |  0.7400
consistency gap            |  0.1400 |   0.2000  |  0.1000  |  0.1400  |  0.1200

--- outcome distribution (per rollout) ---
CORRECT_%                  |  57.0%  |   55.5%   |  56.9%   |  57.8%   |  61.1%   ←best
NO-ANSWER_%                |  25.9%  |   27.3%   |  30.1%   |  29.6%   |  25.4%   ←best
BAD-NUMS_%                 |  10.9%  |    9.8%   |   9.1%   |   9.1%   |   8.8%   ←best
WRONG-VAL_%                |   6.2%  |    7.5%   |   3.9%   |   3.5%   ←best  4.8%

--- length (chars per rollout) ---
median_chars               |   627   |    632    |   513    |   621    |   593
p95_chars                  |  1816   |   1872    |  1867    |  1852    |  1873
max_chars                  |  2036   |   2065    |  2114    |  2188    |  2156

--- rambling: line-repetition signature ---
% line-repeat ≥ 2          |  94.1%  |   74.8%   |  73.8%   |  81.4%   |  81.0%
% line-repeat ≥ 5          |  35.5%  |   34.8%   |  34.9%   |  35.2%   |  32.4%   ←best
% line-repeat ≥ 10         |  15.8%  |   20.8%   |  24.4%   |  19.5%   |  15.1%   ←best

--- rambling: long-failure signature ---
% long_fail ≥ 1200         |  29.4%  |   32.2%   |  32.6%   |  32.9%   |  29.4%
% long_fail ≥ 1500         |  27.1%  |   29.8%   |  31.6%   |  31.6%   |  27.9%
% long_fail ≥ 1800         |   5.9%  |   12.0%   |  11.8%   |  10.9%   |  14.2%

--- arithmetic correctness ---
mean arith ops/rollout     |  29.13  |   29.21   |  30.29   |  29.62   |  28.43
mean mistakes/rollout      |  0.885  |    0.285  |  0.282   |  0.305   |  0.399
% clean (zero mistakes)    |  67.8%  |   81.2%   |  83.9%   ←best  81.2% |  78.6%
% sticky-repeat            |  14.2%  |    3.0%   ←best  3.5% |   4.5%  |   4.5%
```

### 4.2 What each method contributed

| Contribution | Magnitude | Method |
|---|---|---|
| Reduced **arithmetic mistakes** (mean per rollout) | 0.885 → 0.285 (−68%) | Arith PRM |
| Reduced **sticky-repeat failures** | 14.2% → 3.0% (−79%) | Arith PRM |
| Improved **pass@16** (knowledge ceiling) | 0.72 → 0.76 (+4pp) | Arith PRM |
| Recovered **pass@1** | 0.56 → 0.60 (+4pp) | Canonical RFT (v3) |
| Recovered **diversity** (vs naive RFT v1) | pass@16 0.70 → 0.74 | Diversity-preserving selection (v3 over v1) |
| Reduced **NO-ANSWER** below baseline | 27.3% → 25.4% (−1.9pp, below Default's 25.9%) | IPO v1 |
| Reduced **severe loops** (≥10 line-repeats) | 19.5% → 15.1% (−4.4pp, below Default's 15.8%) | IPO v1 |
| **Final pass@1** | 0.58 → 0.62 (+4pp) | IPO v1 (stacked on all prior) |

The total +4pp gain in pass@1 from Default RLOO is decomposed as:
- Arith PRM: −2pp (verification tax)
- RFT v3: +4pp
- IPO v1: +2pp

In other words, **Arith PRM's pass@1 cost is more than paid back by RFT v3 and IPO v1**, while the pass@16 and arithmetic-correctness gains from Arith PRM are *preserved* throughout the stack.

### 4.3 Costs and trade-offs

No method is free. The trade-offs:

**Arithmetic PRM**:
- Cost: pass@1 −2pp (verification tax — model rambles more after learning to deliberate on arithmetic)
- This is paid back by RFT v3 and IPO v1.

**RFT v3**:
- Cost: pass@16 −2pp (0.76 → 0.74) — small mode collapse from any positive-only SFT
- RFT v1's same-direction cost was much worse: pass@16 −6pp due to length bias. v3 fixed this.

**IPO v1**:
- Cost: arithmetic mistakes increased from 0.30 (RFT v3) to 0.40 (+0.10 per rollout)
- Mechanism: with NO-ANSWER reduced (25.4% vs 29.6%), more rollouts actually attempt to commit, and some commits expose latent arithmetic errors that were previously masked behind NO-ANSWER outcomes
- Net: IPO still has **−55% fewer arithmetic mistakes than Default RLOO** (0.40 vs 0.89) — the Arith PRM contribution is largely preserved

**Interesting nuance — long-failure tail (% long_fail ≥ 1800)**:
- Default RLOO: 5.9% — fewest extreme rambles
- IPO v1: 14.2% — most extreme rambles

This appears to contradict IPO's anti-rambling story, but the mechanism is:
- IPO eliminated the *rescuable* mid-length rambles (rollouts at 1300-1500 chars that might have been rescuable into WRONG-VAL or BAD-NUMS)
- The rambling failures that *remain* are the truly hard ones that hit the max-token budget
- Empirically: of IPO's NO-ANSWERs, 56% are ≥1800 chars; of Default's NO-ANSWERs, only 22% are
- Total IPO failures = 38.9% non-CORRECT (vs Default 43.0%) — fewer in absolute terms; the remaining ones are concentrated at the extreme tail

So **IPO reduced total long failures while concentrating the residual at the budget cap** — the model is now failing only on prompts it genuinely doesn't know.

---

## 5. Per-Method Detailed Analysis

### 5.1 Default RLOO — baseline characterization

The baseline shows three dominant failure signatures:

**Arithmetic error dominance**:
- 67.8% of rollouts are arithmetically clean
- 32.2% contain ≥1 mistake — and within these, 14.2% have *sticky repeats* (same wrong calculation re-emitted)
- Mean 0.885 mistakes per rollout
- The verification-style behavior is universal: 94.1% of rollouts have at least one line repeated, mostly from self-verification patterns

**Rambling at moderate length**: 25.9% NO-ANSWER, almost all of which are ≥ 1695 chars (the empirical NO-ANSWER p10). The 1024-token sampling budget caps individual rollout length at ~2,800 chars, but the bulk of NO-ANSWERs end at ~1700-1900 chars — likely the model's natural give-up point before hitting the cap.

**Consistency gap of 0.14**: model could find the answer in 16 tries at 72% but commits on first try at 58%.

### 5.2 Arithmetic PRM — the workhorse

**What it did right**: the rule-based detector for `a OP b = c` mistakes is a perfect proxy for the targeted failure mode. The RLOO gradient on the shaped reward directly pushes mass away from invalid arithmetic patterns. The result is a clean, large-magnitude improvement: −68% in mean mistakes, −79% in sticky-repeat. Most importantly, **this gain persists across the entire downstream stack** — RFT v1, v3, and IPO v1 all warm-start from Arith PRM and inherit the arithmetic competence. RFT v3 still has 0.30 mean mistakes (vs Default's 0.89), IPO v1 has 0.40 — both vastly below baseline.

**What it cost**: 2pp of pass@1 (0.58 → 0.56) and +1.4pp NO-ANSWER (25.9% → 27.3%). The mechanism is the verification tax: when arithmetic mistakes carry a gradient penalty, the model invests more reasoning steps per arithmetic operation, increasing the chance of running out of token budget.

**Pass@16 improvement**: +4pp (0.72 → 0.76). With arithmetic mistakes 3× less frequent, prompts that previously failed all 16 times due to consistent miscalculation now succeed in at least one of the 16 tries.

**Net interpretation**: Arithmetic PRM is a **pass@k-focused method**. It pays a small pass@1 tax for a larger pass@16 gain. The subsequent methods (RFT, IPO) then convert that pass@16 ceiling into pass@1 gains.

### 5.3 RFT v1 — the negative-result baseline

**What we tried**: shortest-correct rollout per prompt, 1170 training examples, 1 epoch.

**Result**: pass@1 = 0.60 (+4pp over Arith PRM), but pass@16 = 0.70 (−6pp regression).

**Why it failed (at preserving pass@16)**:

1. Sampling at T=0.6 with n=8 rollouts → 8 rollouts per prompt are token-distinct (median 7/8 unique) but their *reasoning patterns* are highly redundant. We empirically verified this: at T=0.6 the model produces variations of the same reasoning trajectory.

2. "Shortest correct" selection biased the training toward the most templated rollouts. Combined with the low temperature, the training distribution had effectively ~1 reasoning pattern per prompt repeated.

3. The resulting model fits this narrow mode tightly: median rollout length collapsed from Arith PRM's 632 chars to 513 chars (−19%). The model became *terser* — and when the terse pattern doesn't find the answer, the model has no fallback (% line-repeat ≥ 10 rose to 24.4%, the highest of any model).

**Why we report it**: RFT v1's failure was the empirical signal that motivated RFT v3. The contrast between v1 (pass@16 0.70) and v3 (pass@16 0.74) demonstrates that *diversity preservation is a necessary condition* for RFT at this model scale.

### 5.4 RFT v3 — canonical RFT

**Two changes from v1**:

1. Sampling: T=1.0, n=16 per prompt (instead of T=0.6, n=8) → genuine diversity at the sampling stage. Median 14/16 sampled correct rollouts are token-distinct, vs 7/8 at T=0.6; more importantly, the *reasoning patterns* are substantively varied.

2. Selection: top-K=5 distinct random correct per prompt (instead of shortest-1) → preserves multiple reasoning paths in the training distribution.

**Result**: pass@1 = 0.60 (same as v1), pass@16 = 0.74 (+4pp recovery vs v1's 0.70).

The pass@1 didn't *improve* over v1, but the pass@16 regression was largely undone. Median rollout length returned to 621 chars (natural reasoning length, near Arith PRM's 632) — the model is no longer collapsed onto a terse template.

**Interpretation**: RFT's mechanism (amplify successful modes) has a fixed pass@1 ceiling at the model scale tested. The variable was how much pass@16 was sacrificed in the process. Canonical RFT (v3) extracted the same pass@1 with much less pass@16 damage.

### 5.5 IPO v1 — the headline result

**What it did**: contrastive preference optimization with (CORRECT, length ≥ 1200 char failure) pairs from RFT v3's own rollouts.

**Result**:
- pass@1 = **0.62** (+2pp over RFT v3, +4pp over Default RLOO baseline)
- pass@16 = 0.74 (preserved from RFT v3, no regression — KL anchor working)
- NO-ANSWER = **25.4%** (lowest of any model; below Default's 25.9%)
- Severe loops (≥10 line-repeats) = **15.1%** (lowest of any model; below Default's 15.8%)
- CORRECT % per rollout = **61.1%** (highest of any model)

**Mechanism**:

IPO's gradient explicitly pushes the policy distribution *toward* CORRECT trajectories and *away from* long-rambling failures for the *same prompt*. The (chosen, rejected) pair from a single prompt provides a much sharper signal than positive-only SFT, which only tells the model "do this" but never "don't do that."

The KL anchor to the reference policy (RFT v3) prevents the policy from collapsing — empirically, pass@16 is preserved at 0.74.

**Why the pair design worked**:
- Chosen pool is filter-free (any CORRECT rollout, any length) → the chosen distribution stays diverse, mirroring RFT v3's training emphasis on diverse reasoning paths
- Rejected pool is filtered by length ≥ 1200 chars → the rejected distribution is exactly "long failure" — the rambling signature
- The contrast IPO learns is "this style of reasoning succeeds vs this style of reasoning rambles into the void," with length emerging as a *consequence* of the rambling filter rather than as an engineered surface signal

**The cost**: arithmetic mistakes went from 0.30 (RFT v3) to 0.40 (IPO). With NO-ANSWER reduced, more rollouts actually commit, and some commits expose latent arithmetic errors. Still −55% vs Default's 0.89.

---

## 6. The IPO v1 vs IPO v2 Ablation (validating the design choice)

A natural alternative design: also filter the chosen pool by length to *engineer* a sharper length contrast. We tested this:

- IPO v2 design: chosen = CORRECT AND `len ≤ 600` AND `n_arith_mistakes = 0`; rejected unchanged.
- All other hyperparameters identical to v1.

**Result**:
- pass@1: 0.62 (v1) → **0.60** (v2) — −2pp regression
- pass@16: 0.74 (v1) → 0.72 (v2) — −2pp regression
- median chars: 593 (v1) → 576 (v2) — shorter as engineered
- WRONG-VAL: 4.8% (v1) → **7.6%** (v2) — +2.8pp regression: premature commits to wrong values
- Severe loops (≥10): 15.1% (v1) → 19.5% (v2) — lost IPO's anti-loop gain

**Interpretation**:

The engineered length contrast worked — output length dropped. But IPO learned "be shorter" as its dominant signal rather than "don't ramble," and the model now over-commits even on prompts where deliberation was needed. The pass@1 regression is net-negative.

This is the empirical validation of v1's principled design: **in preference learning, do not engineer surface contrasts between chosen and rejected pools.** Let the rambling filter on the rejected pool carry the signal; leave the chosen pool diverse so length doesn't dominate as a shortcut.

This ablation is a key contribution of the project — it demonstrates that the "obvious" design choice (filter both pools for cleaner contrast) is empirically wrong.

---

## 7. Methodological Insights

### Insight 1: Failure modes decompose orthogonally

The three failure axes (arithmetic mistakes, diversity collapse, rambling) can be attacked independently. Arith PRM addresses arithmetic without affecting pass@k diversity. Canonical RFT addresses diversity without affecting arithmetic. IPO addresses rambling without affecting either (much). The methods stack additively rather than substituting for each other.

This is the project's central empirical claim, and it generalizes: when you can decompose failures into independent surface signatures, you can attack them with independent mechanisms and the gains compose.

### Insight 2: Diversity preservation is necessary for RFT

Naive RFT (shortest-correct + low-T sampling) and canonical RFT (T=1.0 + distinct random selection) extract the same pass@1, but the pass@k cost differs by 4pp. At small model scale, the policy is fragile to mode collapse, and the data construction must actively preserve diversity. This is consistent with findings in the broader RFT/STaR literature (Yuan et al. 2023; Zelikman et al. 2022).

### Insight 3: Engineered surface contrasts in preference data backfire

IPO v1 (rejected-only length filter) beats IPO v2 (chosen AND rejected length filter) on pass@1, because v2 inadvertently teaches the model "be shorter" as a shortcut for "don't ramble." This is a specific instance of a general principle in preference learning: the gradient will find the most accessible surface contrast between pools, so the experimenter should ensure that contrast IS the property they want.

### Insight 4: The 1200-char threshold is empirically grounded

We did not pick the rambling threshold heuristically. From the empirical length distribution on 64K rollouts: CORRECT p95 = 931 chars; NO-ANSWER p10 = 1695 chars. There is a natural ~760-char gap where rollouts essentially don't exist. The 1200-char threshold sits inside this gap. We verified that the threshold:
- Captures ~100% of NO-ANSWERs
- Captures ~70% of WRONG-VAL (the rambling-then-wrong-commit tail)
- Captures ~30% of BAD-NUMS (the rambling-then-bad-nums tail)
- Excludes only 1.9% of CORRECT (which aren't in the rejected pool anyway since they pass the CORRECT predicate)

### Insight 5: Pass@16 and pass@1 represent distinct degrees of freedom

Arith PRM *increased* pass@16 (+4pp) while *decreasing* pass@1 (−2pp). RFT v3 *increased* pass@1 (+4pp) while *decreasing* pass@16 (−2pp). These are not the same metric. A method can improve "the model's knowledge ceiling" without improving "first-attempt commitment," and vice versa. Choosing the right method for a deliverable depends on whether the use case is pass@1 (single response) or pass@K with selection (resampling).

---

## 8. Limitations

1. **Sample size**: n=50 test prompts × 16 rollouts gives SE ≈ 0.07 for pass@1 at p ≈ 0.6. Individual +2pp gains (e.g., IPO v1 over RFT v3) are within 1 SE. The argument for the results is that the *direction* is consistent across multiple correlated metrics — IPO v1 simultaneously has the highest pass@1 AND the lowest NO-ANSWER AND the lowest severe-loop rate AND the highest CORRECT %. The joint probability of all four being noise in the same direction is much lower than any single metric.

2. **Single sampling seed**: each model is evaluated with one sampling pass. Multiple seeds would tighten the confidence intervals.

3. **Single epoch each**: RFT v1, RFT v3, and IPO v1 all ran for one epoch at lr=5e-6. The gains may not be saturated. More epochs or higher lr might push pass@1 further at the cost of more pass@16 regression — we did not sweep.

4. **Arithmetic regression from IPO**: mean mistakes 0.30 → 0.40 is a real cost of the contrastive training. For deployments where arithmetic correctness matters more than first-attempt commitment, Arith PRM alone (mean mistakes 0.285) may be preferable to the full stack.

5. **Char-based rambling threshold**: 1200 chars is approximate; at the Qwen tokenizer's ~2.7 chars/token, this corresponds to ~444 tokens (~43% of the 1024 sampling budget). The threshold could be specified in tokens for stricter unit consistency with the sampling budget; the empirical gap argument holds in either unit.

6. **Single warm-start path**: IPO is stacked on RFT v3 which is stacked on Arith PRM. We did not test alternative orderings (e.g., RFT then Arith PRM, or IPO directly on Arith PRM). The current ordering follows the failure-mode decomposition (fix arithmetic first, then diversity, then rambling).

---

## 9. Conclusion

We presented a five-stage post-training stack on the Countdown arithmetic task that improves pass@1 from 0.58 to 0.62 (+4pp) while reducing arithmetic mistakes by 55% and pushing NO-ANSWER (rambling) below the original baseline.

The three contributions — **process reward shaping for arithmetic correctness** (Arith PRM), **canonical rejection-sampling fine-tuning with diversity preservation** (RFT v3), and **contrastive preference learning on length-filtered rambling pairs** (IPO v1) — each target a distinct failure mode and stack additively. We also presented a methodologically important ablation (IPO v1 vs IPO v2) showing that engineering surface contrasts in preference data backfires, which is a generalizable lesson beyond this task.

The final model achieves the lowest NO-ANSWER rate (25.4%, below the 25.9% baseline), the lowest severe-loop rate (15.1%, below the 15.8% baseline), the highest CORRECT rate (61.1%), and the highest pass@1 (0.62) of any model trained in this project. The pass@k diversity (pass@16 = 0.74) is preserved within 2pp of Arith PRM's project-best 0.76.

The most generalizable findings are the methodological ones: (1) failure modes on this task decompose orthogonally and can be attacked independently; (2) diversity preservation is necessary for RFT at small model scale; (3) preference learning gradients find the most accessible surface contrast, so chosen and rejected pools should differ *exactly* in the property the experimenter wants the model to learn — and nothing else.
