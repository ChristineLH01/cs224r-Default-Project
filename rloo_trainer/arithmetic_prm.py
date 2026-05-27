"""Arithmetic PRM for default-substrate RLOO (no SELS structure tokens).

Single-purpose, terminal-scalar reward shaper that targets two specific failure
modes from `evaluation/failure_modes_default_rloo.md`:

  1. WRONG-VAL / BAD-NUMS rollouts containing detectable arithmetic mistakes
     (58% and 22% of those categories respectively in the default-RLOO eval).
  2. NO-ANSWER rollouts dominated by the sticky-wrong-fact pattern — the same
     wrong `a OP b = c` re-emitted multiple times inside one trace (57% of
     NO-ANSWER rollouts have ≥ 1 mistake, mean 2.13 mistakes per rollout).

Design 1 (negative-only + repetition kicker, capped), UNCONDITIONAL:

    penalty = β · n_total_mistakes + γ · n_repeated_mistakes
    prm     = -min(penalty, cap)       ∈ [-cap, 0]

where
    n_total_mistakes    = #{(a, op, b, c) in trace : eval(a OP b) ≠ c}
                          counted across all occurrences (sticky repeats count
                          again).
    n_repeated_mistakes = n_total_mistakes - n_unique_mistakes
                          (extras beyond the first occurrence of each distinct
                          wrong tuple — directly targets sticky-wrong-fact).

With β=0.1, γ=0.2, cap=0.5 the calibration is:
    1 distinct mistake          →  -0.1   (one format_score worth of pressure)
    4 same mistake (sticky)     →  -0.5   (β·4 + γ·3 = 1.0, capped)
    5 distinct mistakes         →  -0.5   (β·5 = 0.5)

History: an outcome-conditional gate (skip penalty when compute_score >= 1.0)
was tried as v2 to avoid v1's verification-tax pathology. Empirically (see
evaluation/methods_retrospective.md), the gate over-corrected — v2 had MORE
arithmetic mistakes than baseline (mean 1.49 vs 0.89), since the gate
removed the only mechanism by which the reward shaped CORRECT-rollout
cleanliness. v1's unconditional design is restored here; the verification-
tax issue is to be addressed downstream by behavior-cloning / restart
mechanisms (see evaluation/restart_on_rambling_plan.md), not by gating off
the arithmetic signal.

Total reward in `rloo.py` becomes `compute_score + arithmetic_coef · prm`.
Default `arithmetic_coef=0.0` → PRM never called, byte-identical baseline.
"""
from __future__ import annotations

import re
from collections import Counter


# `a OP b = c` with optional whitespace. ASCII operators only — natural-language
# arithmetic ("thirty-five plus nineteen…") is silently ignored (no signal, no
# penalty). Declared value allows leading `-` (model writes "1 * 63 = -63").
_ARITH_RE = re.compile(r"(\d+)\s*([+\-*/])\s*(\d+)\s*=\s*(-?\d+)")

# Reward-shape constants. Free parameters here are intentional — exposing the
# trade-off in code instead of in a wandb sweep config keeps the calibration
# auditable from the file. `rloo.py`'s `--arithmetic_coef α` is a separate
# multiplier on top of these.
_BETA  = 0.1   # per-mistake penalty
_GAMMA = 0.2   # additional per-repeated-mistake penalty (sticky-wrong-fact)
_CAP   = 0.5   # absolute upper bound on total penalty


def _is_mistake(a: int, op: str, b: int, declared: int) -> bool | None:
    """True iff `eval(a OP b) != declared`. None when undefined (div by zero)."""
    if op == "+":
        return (a + b) != declared
    if op == "-":
        return (a - b) != declared
    if op == "*":
        return (a * b) != declared
    if op == "/":
        if b == 0:
            return None
        actual = a / b
        if abs(actual - round(actual)) < 1e-9:
            return int(round(actual)) != declared
        return True                          # non-integer actual, int declared
    return None


def compute_arithmetic_prm(response_text: str, ground_truth: dict,
                           beta: float = _BETA,
                           gamma: float = _GAMMA,
                           cap: float = _CAP,
                           ) -> tuple[float, dict]:
    """Return (prm_scalar in [-cap, 0], metrics dict).

    Metrics keys (always present): n_total_ops, n_total_mistakes,
    n_unique_mistakes, n_repeated_mistakes, prm.

    `ground_truth` is accepted for signature parity with other PRMs but is
    not used — the arithmetic check is intrinsic to the trace.
    """
    del ground_truth

    n_total_ops      = 0
    mistakes: list[tuple[int, str, int, int]] = []
    for m in _ARITH_RE.finditer(response_text):
        a, op, b, c = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))
        is_bad = _is_mistake(a, op, b, c)
        if is_bad is None:
            continue                          # div-by-zero: no signal
        n_total_ops += 1
        if is_bad:
            mistakes.append((a, op, b, c))

    n_total_mistakes    = len(mistakes)
    n_unique_mistakes   = len(set(mistakes))
    n_repeated_mistakes = n_total_mistakes - n_unique_mistakes

    penalty = beta * n_total_mistakes + gamma * n_repeated_mistakes
    prm     = -min(penalty, cap)

    return prm, {
        "n_total_ops":         n_total_ops,
        "n_total_mistakes":    n_total_mistakes,
        "n_unique_mistakes":   n_unique_mistakes,
        "n_repeated_mistakes": n_repeated_mistakes,
        "prm":                 prm,
    }


# --- Self-test --------------------------------------------------------------

if __name__ == "__main__":
    # Default ground truth used when the case has no <answer> tag — compute_score
    # returns 0 in that scenario, so the conditional gate doesn't fire and the
    # PRM logic runs as in v1. The two cases at the end exercise the gate.
    _gt = {"target": 98, "numbers": [44, 19, 35]}

    cases = [
        # (label, trace, ground_truth, expected_prm, expected_n_mistakes, expected_n_repeats)
        ("clean correct CoT — no mistakes",
         "44 + 19 = 63\n63 + 35 = 98\n", _gt, 0.0, 0, 0),

        ("one mistake — gentle penalty",
         "44 + 19 = 64\n", _gt, -0.1, 1, 0),

        ("sticky-wrong-fact (1 wrong × 4 occurrences)",
         "63 * 95 = 5885\n63 * 95 = 5885\n63 * 95 = 5885\n63 * 95 = 5885\n",
         _gt, -0.5, 4, 3),                          # penalty = 0.4 + 0.6 = 1.0 → cap

        ("5 distinct mistakes — caps at 0.5",
         "1 + 1 = 3\n2 + 2 = 5\n3 + 3 = 7\n4 + 4 = 9\n5 + 5 = 11\n",
         _gt, -0.5, 5, 0),

        ("2 same mistakes — repetition kicker applies",
         "1 * 63 = -63\n1 * 63 = -63\n", _gt, -0.4, 2, 1),   # 0.2 + 0.2 = 0.4

        ("3 distinct mistakes — no repeats",
         "1 + 1 = 3\n2 + 2 = 5\n3 + 3 = 7\n", _gt, -0.3, 3, 0),

        ("div-by-zero silently skipped",
         "5 / 0 = 5\n", _gt, 0.0, 0, 0),

        ("non-integer division mistake",
         "44 / 16 = 2\n", _gt, -0.1, 1, 0),

        ("mixed: 1 wrong + 1 correct + 1 div-skip",
         "1 + 1 = 2\n44 + 19 = 64\n5 / 0 = 5\n", _gt, -0.1, 1, 0),

        ("mistake inside SELS-style annotation is NOT counted",
         # The regex requires whitespace between operands and `=`. `19)=64`
         # has `)` between, so this annotation form is silently ignored —
         # the arithmetic PRM is the prose-form complement to structural PRM.
         "<state_1> [(44 + 19)=64, 35]\n", _gt, 0.0, 0, 0),

        # Unconditional design — penalty applies regardless of outcome.
        ("CORRECT rollout with intermediate mistake — PRM fires (no gate)",
         # The answer "(44+19)+35 = 98" is correct, but the trace contains
         # a mistake "44+19=64". With the v1 unconditional design, the
         # penalty applies.
         "44 + 19 = 64 (oops)\n63 + 35 = 98\n<answer> (44 + 19) + 35 </answer>",
         {"target": 98, "numbers": [44, 19, 35]}, -0.1, 1, 0),

        ("WRONG-VAL rollout with mistake — PRM fires",
         "44 + 19 = 64\n<answer> 44 + 19 - 35 </answer>",       # = 28, not 98
         {"target": 98, "numbers": [44, 19, 35]}, -0.1, 1, 0),
    ]
    failed = 0
    for label, text, gt, exp_prm, exp_n_mis, exp_n_rep in cases:
        prm, m = compute_arithmetic_prm(text, gt)
        ok_prm = abs(prm - exp_prm) < 1e-9
        ok_mis = m["n_total_mistakes"]    == exp_n_mis
        ok_rep = m["n_repeated_mistakes"] == exp_n_rep
        marker = "PASS" if (ok_prm and ok_mis and ok_rep) else "FAIL"
        if not (ok_prm and ok_mis and ok_rep):
            failed += 1
        print(f"  {marker}  prm={prm:+.2f}  mistakes={m['n_total_mistakes']}  "
              f"repeats={m['n_repeated_mistakes']}  — {label}")
    print(f"\n{failed} failures out of {len(cases)} cases.")
