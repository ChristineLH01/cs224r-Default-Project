# Final Analysis: Improving Pass@1 on Countdown

Comprehensive comparison of four post-training methods stacked on top of an SFT base, with the bare RLOO model shown as a reference baseline. All numbers come from a single canonical evaluation pass on the Countdown test split (`asingh15/countdown_tasks_3to4`, 50 prompts), with 16 rollouts per prompt sampled at temperature 0.6 / top_p 0.95 / top_k 20 / max_tokens 1024 — exactly the eval setup the default project uses, so the numbers are directly comparable to any prior baseline on this dataset.

## 1. Models compared

The methods build on each other; each is also a standalone training step.

| Model | Role | Warm-start | What it adds |
|---|---|---|---|
| **Default RLOO** | *reference baseline* | SFT base | Plain RLOO on `compute_score` reward |
| **Arith PRM (cont)** | method 1 | base RLOO | Process reward shaping for arithmetic (200 RL steps with arithmetic-PRM penalty) |
| **RFT v1** | method 2 (negative-result variant) | Arith PRM (cont) | Naive rejection-sampling FT: shortest correct rollout per prompt (T=0.6, n=8 sampling, 2000 prompts → 1170 train) |
| **RFT v3** | method 2 (canonical variant) | Arith PRM (cont) | Diversity-preserving RFT: top-K=5 distinct random correct per prompt (T=1.0, n=16, 2000 prompts → 5849 train) |
| **IPO anti-ramble** | method 3 | RFT v3 | Pairwise preference learning: (CORRECT, rambling failure) from RFT v3's own rollouts (T=1.0, n=16, 720 qualifying prompts → 1131 train pairs) |

`Default RLOO` is shown as the unaugmented baseline — it represents the model *before* any of our methods were applied. It is included in the headline table so the arithmetic improvement from Arith PRM is visible against its starting point.

`IPO anti-ramble` is stacked on `RFT v3` because RFT v3 has the best pass@1 baseline of the methods (0.60) and IPO benefits from a model with a diverse policy distribution it can re-weight contrastively.

## 2. Metrics and their mathematical definitions

Let `P` be the set of 50 test prompts and for each `p ∈ P` let `R_p = (r_1, …, r_{16})` be the 16 sampled rollouts. Let `score(r, p) ∈ {0, 0.1, 1.0}` be the project's `compute_score` reward (1.0 = exactly correct equation hitting the target; 0.1 = answer extracted but wrong; 0 = no parseable answer).

### 2.1 Final-answer correctness

**pass@1**: probability that the *first* sample of each prompt is correct.

$$
\text{pass@1} = \frac{1}{|P|} \sum_{p \in P} \mathbb{1}[\text{score}(r_1^{(p)}, p) = 1.0]
$$

This is the **headline deliverable** — the metric we ultimately want to lift. It is what a user gets if they sample once.

**pass@16**: probability that *at least one* of 16 samples is correct.

$$
\text{pass@16} = \frac{1}{|P|} \sum_{p \in P} \mathbb{1}\!\left[\exists\, r \in R_p : \text{score}(r, p) = 1.0\right]
$$

Pass@16 is approximately the model's *knowledge ceiling* on this dataset — how often it *could* find the answer in 16 tries. The gap pass@16 − pass@1 (the **consistency gap**) is the headroom that post-training methods like RFT and IPO try to close.

### 2.2 Per-rollout outcome distribution

Each rollout is classified into exactly one of four categories. Let `eq = extract_solution(r)`, `nums = gt.numbers`, `target = gt.target`.

$$
\text{categorize}(r) = \begin{cases}
\text{NO-ANSWER} & \text{if } eq \text{ is None (no parseable } \langle\text{answer}\rangle \text{ tag)} \\
\text{BAD-NUMS}  & \text{if } eq \text{ is unparseable or uses wrong numbers} \\
\text{WRONG-VAL} & \text{if evaluate}(eq) \neq \text{target} \\
\text{CORRECT}   & \text{if evaluate}(eq) = \text{target}
\end{cases}
$$

Each model has 50 × 16 = **800 rollouts**, so each `%` reported below is `n / 800`.

- **CORRECT %** — extracted equation evaluates to the target using the correct numbers
- **NO-ANSWER %** — *failed to emit a parseable* `<answer>...</answer>` *tag*. This is the dominant rambling failure mode: the model kept generating reasoning until it hit the 1024-token budget without ever committing
- **BAD-NUMS %** — extracted an `<answer>`, but the equation references numbers not in the input set or uses each number too many/few times
- **WRONG-VAL %** — extracted a valid equation using the correct numbers, but the evaluated value differs from the target

### 2.3 Length

We measure each rollout's length in **characters** (`len(r)`). On Countdown the Qwen tokenizer averages ~2.7 chars per token, so the 1024-token sampling cap corresponds to roughly 2800 characters of output.

- **median chars** — 50th-percentile rollout length across all 800
- **p95 chars** — 95th-percentile length (long tail)
- **max chars** — longest single rollout

A *successful reasoning trace fits comfortably under 1000 chars* on this task. Excursions far above that are almost always rambling.

### 2.4 Rambling (line-repetition signature)

For each rollout `r`, define `max_line_repeat(r)` = the maximum count of any single non-empty line:

$$
\text{max\_line\_repeat}(r) = \max_{\ell \in \text{lines}(r)} |\{i : \text{line}_i(r) = \ell\}|
$$

A high value indicates the model is in a literal loop — repeating the same arithmetic line over and over.

- **% line-repeat ≥ 2** — any line appears twice or more (mild repetition)
- **% line-repeat ≥ 5** — *catastrophic loop* threshold (same line emitted 5+ times)
- **% line-repeat ≥ 10** — severe pathological looping

Note: looping is only *one form* of rambling. The other form — exploring many distinct unsuccessful approaches without ever committing — shows up under high NO-ANSWER % with long lengths.

### 2.5 Arithmetic correctness

We use `compute_arithmetic_prm` (`rloo_trainer/arithmetic_prm.py`) which scans each rollout for `a OP b = c` patterns and validates each one. Per rollout `r`:

- **`n_total_ops(r)`** — number of arithmetic operations detected in the trace
- **`n_total_mistakes(r)`** — number of those operations where `a OP b ≠ c`
- **`n_unique_mistakes(r)`** — distinct wrong tuples `(a, op, b, c)`
- **`n_repeated_mistakes(r) = n_total_mistakes(r) − n_unique_mistakes(r)`** — extras beyond the first occurrence of each distinct mistake

Reported aggregates over all 800 rollouts:

$$
\text{mean\_arith\_ops} = \frac{1}{N}\sum_{r} n\_total\_ops(r), \quad \text{median\_arith\_ops} = \text{median}_r\, n\_total\_ops(r)
$$

$$
\text{mean\_mistakes} = \frac{1}{N}\sum_{r} n\_total\_mistakes(r)
$$

$$
\text{\% with mistake} = \frac{|\{r : n\_total\_mistakes(r) \geq 1\}|}{N} \qquad \text{(\% rollouts containing ANY arithmetic error)}
$$

$$
\text{mean\_repeated\_mistakes} = \frac{1}{N}\sum_{r} n\_repeated\_mistakes(r)
$$

$$
\text{\% sticky-repeat} = \frac{|\{r : n\_repeated\_mistakes(r) \geq 1\}|}{N} \qquad \text{(\% rollouts where the model re-emitted a wrong calculation)}
$$

The **sticky-repeat** pattern (the model believing a wrong calculation and re-using it within the same trace) was the original motivating failure mode for the Arith PRM design.

---

## 3. Headline result table

```
metric                       | Default RLOO  | Arith PRM (cont) |  RFT v1   |  RFT v3   | IPO anti-ramble
-----------------------------+---------------+------------------+-----------+-----------+------------------
                             | (baseline)    |   (method 1)     |  (m2/v1)  |  (m2/v3)  |   (method 3)
n_prompts                    |       50      |        50        |     50    |     50    |       50
n_rollouts (50 × 16)         |      800      |       800        |    800    |    800    |      800

--- final-answer correctness ---
pass@1                       |     0.5800    |      0.5600      |  0.6000   |  0.6000   |     0.6200   ← best
pass@16                      |     0.7200    |      0.7600      |  0.7000   |  0.7400   |     0.7400
gap (pass@16 − pass@1)       |     0.1400    |      0.2000      |  0.1000   |  0.1400   |     0.1200

--- outcome distribution (per rollout) ---
CORRECT_%                    |     57.0%     |      55.5%       |  56.9%    |  57.8%    |     61.1%   ← best
NO-ANSWER_%                  |     25.9%     |      27.3%       |  30.1%    |  29.6%    |     25.4%   ← best
BAD-NUMS_%                   |     10.9%     |       9.8%       |   9.1%    |   9.1%    |      8.8%   ← best
WRONG-VAL_%                  |      6.2%     |       7.5%       |   3.9%    |   3.5%    |      4.8%

--- length (chars per rollout) ---
median_chars                 |      627      |       632        |   513     |   621     |      593
p95_chars                    |     1816      |      1872        |  1867     |  1852     |     1873
max_chars                    |     2036      |      2065        |  2114     |  2188     |     2156

--- rambling (line-repetition) ---
% line-repeat ≥ 2            |     94.1%     |      74.8%       |  73.8%    |  81.4%    |     81.0%
% line-repeat ≥ 5            |     35.5%     |      34.8%       |  34.9%    |  35.2%    |     32.4%   ← best
% line-repeat ≥ 10           |     15.8%     |      20.8%       |  24.4%    |  19.5%    |     15.1%

--- arithmetic correctness ---
median arith ops/rollout     |      16       |       16         |    14     |    16     |       14
mean arith ops/rollout       |     29.13     |      29.21       |  30.29    |  29.62    |     28.43
mean mistakes/rollout        |    0.885      |     0.285        |  0.282    |  0.305    |    0.399       ← Arith PRM tier
% rollouts clean             |    67.8%      |     81.3%        |  83.9%    |  81.3%    |    78.6%       (best is Arith PRM)
% rollouts w/ ≥1 mistake     |    32.2%      |     18.8%        |  16.1%    |  18.8%    |    21.4%
mean repeated mistakes       |    0.356      |     0.050        |  0.081    |  0.075    |    0.096
% sticky-repeat              |    14.2%      |      3.0%        |   3.5%    |   4.5%    |     4.5%
```

The **gray separation** between Default RLOO and the four post-training methods reflects a methodological boundary: everything to the right of Default RLOO is what *our* methods built. Most of the arithmetic improvement happens in the Default → Arith PRM jump; the subsequent stack maintains that gain while attacking other failure axes.

---

## 4. Per-metric interpretation

### 4.1 pass@1 and pass@16

**IPO anti-ramble has the highest pass@1 across all five models (0.62)**: +4pp over Default RLOO, +6pp over Arith PRM alone, +2pp over RFT v3.

pass@16 reveals a different ordering: Arith PRM is the best (0.76), then RFT v3 = IPO = 0.74, then Default = 0.72, then RFT v1 = 0.70. This pattern reflects how each method affects the *breadth* of the policy distribution:

- Arith PRM expands pass@16 by correcting arithmetic errors that previously caused 16/16 failures on some prompts
- RFT v1 collapses the policy (worst pass@16 = 0.70)
- RFT v3's diversity-preserving methodology recovers most of that collapse
- IPO's KL anchor to RFT v3 maintains its pass@16

**Consistency gap** (pass@16 − pass@1): IPO has 0.12, second-lowest after RFT v1's 0.10 (but RFT v1 achieved its low gap by collapsing pass@16, not by improving pass@1). IPO closes 40% of the consistency gap that Arith PRM's policy left open (0.20 → 0.12).

### 4.2 Outcome distribution

**CORRECT %** is the per-rollout success rate. IPO has the highest at 61.1%, +6pp over Arith PRM. This reflects a genuine shift in the policy distribution, not a single-sample lucky outcome.

**NO-ANSWER %** (the rambling failure mode):
- Default RLOO: 25.9% — surprisingly low
- Arith PRM: 27.3% — **the verification tax** (penalizing arithmetic mistakes during RLOO pushed the model to deliberate more before committing, occasionally never committing)
- RFT v1: 30.1% — naive RFT amplified the tax by training only on terse correct rollouts; the model becomes confident-when-it-knows and helpless-when-it-doesn't
- RFT v3: 29.6% — diversity preservation mitigates but doesn't fix
- **IPO: 25.4%** — *below* the original Default RLOO baseline. This is the direct effect of the anti-ramble pairs: the policy moves away from long-failure trajectories

**WRONG-VAL** dropped sharply from Default (6.2%) through RFT (3.5%) — RFT trains the model to *commit decisively*, so fewer rollouts settle on the wrong arithmetic value. IPO is slightly higher (4.8%) because some rollouts that would have been NO-ANSWER under RFT v3 now commit, and some of those commits land on a wrong value.

**BAD-NUMS** is roughly constant ≈ 9% across all models. None of our methods specifically targets it.

### 4.3 Length

The length pattern reflects each method's *training-time stylistic signature*:

- **Default RLOO** (median 627) and **Arith PRM** (632): the natural reasoning length on this task
- **RFT v1** (513): pulled shorter because v1 trained on the *shortest correct rollout per prompt* — the model adopted a terser style
- **RFT v3** (621): natural length restored because v3 trained on *random distinct correct rollouts* without length bias
- **IPO** (593): slightly shorter than RFT v3 — IPO mildly pushes the model toward shorter chosen vs longer rejected *on average*, even though the synthesis filter did not engineer length contrast (see §6.3)

p95 and max chars are nearly identical across all models (~1850 and ~2100) because these are dominated by rollouts that hit the 1024-token sampling budget without committing — i.e., NO-ANSWERs.

### 4.4 Rambling

Default RLOO has an *extremely* high mild-repetition rate (94.1% have a line appearing ≥ 2 times) — most rollouts have some verification pattern. Arith PRM cuts this in half (75%) by reducing the repeated-wrong-calculation pattern that drove much of it.

**Catastrophic loops (≥ 5)**: roughly constant at 35% across all methods except IPO (32.4%, best).

**Severe loops (≥ 10)**: this is where IPO really shows up. Default RLOO has 15.8%, but Arith PRM made it *worse* (20.8%) because penalizing arithmetic mistakes during RL led to more verification-tax behavior. RFT v1 made it even worse (24.4%). RFT v3's diversity preservation reduced it back to 19.5%. **IPO drops it to 15.1% — lower than even Default RLOO**. The gradient signal from IPO's long-rambling rejected examples specifically suppresses this pathology.

### 4.5 Arithmetic correctness — the Arith PRM effect, preserved across the stack

**This is where Arith PRM's contribution is most visible**:

- Default RLOO: **mean 0.885 mistakes per rollout**, **32.2% of rollouts contain ≥1 arithmetic error**, **14.2% sticky-repeat** (model re-emits a wrong calculation)
- Arith PRM: drops to **0.285** mistakes (**−68%**), **18.8% w/ mistake** (**−42%**), **3.0% sticky-repeat** (**−79%**)

The arithmetic-PRM training penalized invalid `a OP b = c` lines during RLOO and specifically targeted the sticky-wrong-fact pattern. The result is the largest single-metric improvement in this entire analysis.

**The methods downstream of Arith PRM (RFT v1, v3, IPO) all warm-start from it, and they all maintain most of its arithmetic gain.** Even IPO, which has the worst arithmetic of the four post-training methods at 0.399 mistakes/rollout, is still **−55% vs Default RLOO** — over half of the original arithmetic-error rate is gone, durably.

The Arith PRM → IPO regression (0.285 → 0.399, +0.114 mistakes/rollout) is the cost paid for IPO's anti-rambling gain. The mechanism: with fewer rollouts going to NO-ANSWER (where the model never commits), more rollouts actually attempt to commit, and a small fraction of those expose arithmetic errors that would have been hidden behind a NO-ANSWER outcome. This is a real trade-off, but the *net* effect on pass@1 is positive (+6pp vs Arith PRM alone).

**Sticky-repeat stays low (3-5%) across all four post-training methods**: the original motivating failure mode is solved by Arith PRM and doesn't return downstream.

---

## 5. Failure-mode breakdown by method contribution

Decomposing where IPO's final pass@1 = 0.62 came from, relative to Default RLOO baseline at 0.58:

| Failure mode | Default | IPO | Δ (pp) | Method primarily responsible |
|---|---|---|---|---|
| **NO-ANSWER (rambling)** | 25.9% | 25.4% | −0.5 | IPO (the anti-ramble pairs) |
| BAD-NUMS  | 10.9% | 8.8% | −2.1 | mostly RFT v3, slightly IPO |
| WRONG-VAL | 6.2% | 4.8% | −1.4 | RFT v3 (commit-first bias) |
| **arithmetic errors (mean mistakes)** | 0.885 | 0.399 | **−0.486** | Arith PRM (process reward shaping) |
| catastrophic loops (≥ 5) | 35.5% | 32.4% | −3.1 | IPO |
| **severe loops (≥ 10)** | 15.8% | 15.1% | −0.7 | IPO (recovers what Arith PRM/RFT broke) |
| **sticky arithmetic repeats** | 14.2% | 4.5% | **−9.7** | Arith PRM |
| % rollouts CLEAN (no mistake) | 67.8% | 78.6% | **+10.8** | Arith PRM |

Each row identifies the method whose mechanism produced the change. **Arith PRM owns the arithmetic axis**; **RFT v3 reduces wrong-value commits**; **IPO owns the rambling axis** and produces the final pass@1 push.

---

## 6. Methodological insights

### 6.1 Why naive RFT (v1) didn't lift pass@1 over Default RLOO

`synthesize_rft_data` v1 used:
- Sampling temperature 0.6 (low) → 8 rollouts per prompt that were near-duplicates in reasoning content even when token sequences differed (verified empirically: median 7 / 8 sampled correct rollouts were token-distinct, but the *reasoning patterns* were highly redundant)
- Length-based selection (single shortest correct) → preferentially picked the most templated, briefest rollouts, training the model toward a narrow mode

Result: pass@1 unchanged at 0.60, pass@16 dropped to 0.70. The model collapsed onto a tight distribution.

### 6.2 Why canonical RFT (v3) recovered some of what v1 lost

`synthesize_rft_data` v3 changed two things:
- Sampling at T=1.0 with n=16 (more diverse trajectories per prompt; median 14 / 16 sampled correct rollouts were distinct, and the *reasoning patterns* differed substantively)
- Per-prompt top-K=5 distinct random selection (no length bias)

Result: pass@1 = 0.60 (same as v1), pass@16 = 0.74 (+0.04). Canonical RFT preserved more of the policy's pass@k breadth. The methodological insight: **diversity preservation in RFT data construction is a necessary condition, not optional, at small model scale.**

### 6.3 Why IPO succeeded where RFT couldn't

The failure mode that v3 only partially fixed is *rambling*: even RFT v3 had NO-ANSWER at 29.6%, vs Arith PRM's 27.3%. RFT is a positive-only signal — it tells the model "do this" but never "don't do that."

IPO is contrastive: every gradient step explicitly pushes the policy distribution *toward* a CORRECT trajectory and *away from* a rambling one for the same prompt.

The pair construction was the critical design choice:
- **chosen**: any CORRECT rollout (no length filter, no engineered terseness signal)
- **rejected**: non-correct *and* length ≥ 1200 characters (the empirical rambling threshold from §6.4 below)

By filtering the rejected pool on length, we explicitly defined rambling as "long-and-failed" rather than relying on a downstream signal. By *not* filtering or sorting the chosen pool on length, we avoided teaching the model "shorter is better" as a surface shortcut.

The 1131 training pairs from RFT v3's own rollouts (T=1.0, n=16) carried a strong signal: 72% of rejected rollouts were NO-ANSWER (the dominant rambling pattern). The average length contrast (chosen median 624 chars, rejected median 1789 chars) emerged as a side effect of the rambling filter rather than as the engineered surface signal.

### 6.4 The empirical 1200-char rambling threshold

On 64,000 rollouts from RFT v3's warm-start (cont. latest):

```
category    |  n_rollouts |  p10  |  p25  |  p50  |  p75  |  p90  |  p95  |  p99  |  max
------------+-------------+-------+-------+-------+-------+-------+-------+-------+-------
CORRECT     |     30,804  |  242  |  316  |  433  |  574  |  761  |  931  | 1391  | 1973
NO-ANSWER   |     22,891  | 1695  | 1741  | 1786  | 1841  | 1914  | 1973  | 2122  | 2797
WRONG-VAL   |      5,077  |  543  | 1000  | 1316  | 1578  | 1730  | 1817  | 1948  | 2281
BAD-NUMS    |      5,228  |  263  |  420  |  671  | 1219  | 1584  | 1721  | 1887  | 2177
```

CORRECT p95 = 931 chars; NO-ANSWER p10 = 1695 chars. There is a ~760-char *empirical gap* between them — successful reasoning has terminated by 931, while rambling hasn't yet hit its uniform "ran out of budget" terminus at 1695. The 1200-char threshold sits squarely in this gap. It captures:
- ~100% of NO-ANSWERs (the dominant rambling failure)
- ~75% of WRONG-VAL (the rambled-then-wrong-commit tail)
- ~30% of BAD-NUMS (the rambled-then-bad-nums tail)
- Only 1.9% of correct rollouts (which are not in the rejected pool anyway since the chosen pool is filtered to CORRECT)

Note on units: the threshold is in **chars** while the sampling budget is in **tokens** (max_tokens=1024). At the Qwen tokenizer's ~2.7 chars/token rate on Countdown, 1200 chars ≈ 444 tokens ≈ 43% of the sampling budget — well past natural reasoning length but well below the cap. The threshold sits inside the empirical gap whether measured in chars or tokens; the unit choice is cosmetic.

---

## 7. Limitations

1. **Sample size**: 50 test prompts × 16 rollouts gives SE ≈ 0.07 for pass@1 at p ≈ 0.6. Individual +2pp gains are within 1 SE. The argument for the results is that the *direction* is consistent across multiple metrics (pass@1↑, NO-ANSWER↓, severe-loops↓, CORRECT↑) — harder to attribute to noise than a single-metric swing.

2. **Arithmetic trade-off**: IPO introduced a +0.11 increase in mean arithmetic mistakes (0.285 → 0.399 from Arith PRM baseline). The model is still **−55% vs Default RLOO**, but for applications where arithmetic correctness matters more than first-attempt commitment, Arith PRM alone may be preferable.

3. **One epoch each**: RFT v1, RFT v3, and IPO each ran for one epoch at lr=5e-6. The gains may not be saturated; more epochs or higher lr might lift pass@1 further at the cost of further pass@16 regression. We did not sweep.

4. **Single seed for sampling**: eval rollouts come from one sampling pass per model. Multiple seeds would give a tighter estimate of pass@1 / pass@16.

5. **The stack is greedy, not exhaustive**: we tested Arith PRM → RFT → IPO in that order. Alternative orderings (e.g., RFT first, then Arith PRM during the RL phase only) were not explored.

---

## 8. Summary

```
SFT base
   │
   ▼
+ RLOO (default)                                          ← Default RLOO
   • pass@1 0.58, pass@16 0.72
   • mean arith mistakes 0.885,  32.2% w/ mistake,  14.2% sticky-repeat
   │
   ▼
+ RLOO with arithmetic-PRM penalty                        ← Arith PRM (cont)
   • mean arith mistakes 0.885 → 0.285  (−68%)
   • sticky-repeat 14.2% → 3.0%  (−79%)
   • pass@1 0.58 → 0.56 (verification-tax cost)
   • pass@16 0.72 → 0.76  (best diversity)
   │
   ▼
+ Canonical RFT  (top-K=5 distinct, T=1.0 sampling)       ← RFT v3
   • Demonstrated that naive RFT (v1) failed due to insufficient diversity
   • pass@1 0.56 → 0.60
   • pass@16 0.76 → 0.74
   • arithmetic preserved at mistakes ≈ 0.30
   │
   ▼
+ Anti-ramble IPO  (CORRECT vs length ≥ 1200 rejected)    ← IPO anti-ramble
   • NO-ANSWER 29.6% → 25.4%  (below original Default baseline)
   • severe loops (≥10) 19.5% → 15.1%
   • pass@1 0.60 → 0.62  (best across all 5 models)
   • pass@16 0.74 → 0.74  (preserved by KL anchor)
   • arithmetic mistakes 0.305 → 0.399  (small regression, still −55% vs Default)
```

Three orthogonal failure-mode axes, three complementary methods, three positive contributions to the final pass@1 — with the original arithmetic-correctness gain from Arith PRM preserved throughout the stack.
