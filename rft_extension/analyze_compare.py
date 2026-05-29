"""4-way model comparison on the bigger eval (held-out train prompts).

Reads the 4 JSON files produced by eval_compare_4models.sh and prints a
side-by-side table with:
  - pass@1, pass@16 (final-answer correctness)
  - Outcome distribution: CORRECT / NO-ANSWER / BAD-NUMS / WRONG-VAL
  - Length: median / p95 / max (chars per rollout)
  - Rambling: % rollouts with catastrophic line-repeat (>=5)
  - Arithmetic correctness: mean ops/rollout, mean mistakes/rollout,
    % rollouts with >=1 mistake, mean repeated-mistakes/rollout
"""
from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

from evaluation.countdown import (
    extract_solution, validate_equation, evaluate_equation,
)
from rloo_trainer.arithmetic_prm import compute_arithmetic_prm


MODELS = [
    ("default_rloo_qwen_sft",                          "Default RLOO"),
    ("default_rloo_qwen_sft_arithprm_continued_latest","Arith PRM (cont)"),
    ("rft_v1",                                         "RFT v1"),
    ("rft_v2",                                         "RFT v2"),
    ("rft_v3",                                         "RFT v3 (T=1.0+dedup)"),
    ("ipo_anti_ramble_v1",                             "IPO v1"),
    ("ipo_anti_ramble_v2",                             "IPO v2 (short+clean)"),
]


def categorize(text, gt):
    eq = extract_solution(text)
    if eq is None: return "NO-ANSWER"
    if not validate_equation(eq, gt["numbers"]): return "BAD-NUMS"
    res = evaluate_equation(eq)
    if res is None: return "BAD-NUMS"
    return "CORRECT" if abs(res - gt["target"]) < 1e-5 else "WRONG-VAL"


def max_line_repeat(text):
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    return max(Counter(lines).values()) if lines else 0


def correct_score(s): return abs(s - 1.0) < 1e-6


def analyze(path: Path, label: str) -> dict:
    rows = []
    with path.open() as f:
        for line in f:
            rows.append(json.loads(line))
    n_prompts = len(rows)

    pass1 = pass16 = 0
    cats = Counter()
    lens, reps = [], []
    arith_ops, arith_mist, arith_rep = [], [], []
    for r in rows:
        gt = {"target": r["target"], "numbers": list(r["nums"])}
        scores = r["scores"]
        if correct_score(scores[0]): pass1 += 1
        if any(correct_score(s) for s in scores): pass16 += 1
        for s in r["response"]:
            cats[categorize(s, gt)] += 1
            lens.append(len(s))
            reps.append(max_line_repeat(s))
            _, m = compute_arithmetic_prm(s, gt)
            arith_ops.append(m["n_total_ops"])
            arith_mist.append(m["n_total_mistakes"])
            arith_rep.append(m["n_repeated_mistakes"])

    total = sum(cats.values())
    lens_sorted = sorted(lens)
    return {
        "label": label,
        "n_prompts":               n_prompts,
        "pass@1":                  pass1 / n_prompts,
        "pass@16":                 pass16 / n_prompts,
        "CORRECT_pct":             cats["CORRECT"]   / total * 100,
        "NO-ANSWER_pct":           cats["NO-ANSWER"] / total * 100,
        "BAD-NUMS_pct":            cats["BAD-NUMS"]  / total * 100,
        "WRONG-VAL_pct":           cats["WRONG-VAL"] / total * 100,
        "median_chars":            lens_sorted[len(lens_sorted) // 2],
        "p95_chars":               lens_sorted[int(len(lens_sorted) * 0.95)],
        "max_chars":               lens_sorted[-1],
        "pct_catastrophic_repeat": sum(r >= 5 for r in reps) / total * 100,
        "mean_arith_ops":          statistics.mean(arith_ops),
        "mean_mistakes":           statistics.mean(arith_mist),
        "pct_with_mistake":        sum(m >= 1 for m in arith_mist) / total * 100,
        "mean_repeated_mistakes":  statistics.mean(arith_rep),
    }


def main():
    results = []
    for name, label in MODELS:
        path = Path(f"evaluation/eval_results/{name}.json")
        if not path.exists():
            print(f"missing: {path}  (skip)")
            continue
        results.append(analyze(path, label))

    if not results:
        print("no eval results to compare.")
        return

    # Format spec for each metric
    fmt = {
        "n_prompts":               "{:>16}",
        "pass@1":                  "{:>16.4f}",
        "pass@16":                 "{:>16.4f}",
        "CORRECT_pct":             "{:>15.1f}%",
        "NO-ANSWER_pct":           "{:>15.1f}%",
        "BAD-NUMS_pct":            "{:>15.1f}%",
        "WRONG-VAL_pct":           "{:>15.1f}%",
        "median_chars":            "{:>16}",
        "p95_chars":               "{:>16}",
        "max_chars":               "{:>16}",
        "pct_catastrophic_repeat": "{:>15.1f}%",
        "mean_arith_ops":          "{:>16.2f}",
        "mean_mistakes":           "{:>16.2f}",
        "pct_with_mistake":        "{:>15.1f}%",
        "mean_repeated_mistakes":  "{:>16.2f}",
    }

    metric_order = list(fmt.keys())
    header = f"{'metric':<28} | " + " | ".join(f"{r['label']:>16}" for r in results)
    print(header)
    print("-" * len(header))
    section_breaks = {
        "pass@1":                  "--- final-answer correctness ---",
        "CORRECT_pct":             "--- outcome distribution (per rollout) ---",
        "median_chars":            "--- length (chars per rollout) ---",
        "pct_catastrophic_repeat": "--- rambling ---",
        "mean_arith_ops":          "--- arithmetic correctness ---",
    }
    for m in metric_order:
        if m in section_breaks:
            print()
            print(section_breaks[m])
        cells = " | ".join(fmt[m].format(r[m]) for r in results)
        print(f"{m:<28} | {cells}")


if __name__ == "__main__":
    main()
