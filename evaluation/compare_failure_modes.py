"""Side-by-side comparison of two trained-model eval JSONs.

Usage:
    python evaluation/compare_failure_modes.py \\
        evaluation/eval_results/default_rloo_qwen_sft.json \\
        evaluation/eval_results/default_rloo_qwen_sft_arithprm.json

Each JSON is the per-prompt output of `evaluation/countdown_eval.py`
(JSON-Lines, one prompt per line, fields {prompt, target, nums, response,
scores}).

Reports:
  1. Pass@1 / pass@16 / mean_score, both runs.
  2. Category breakdown (CORRECT / WRONG-VAL / BAD-NUMS / NO-ANSWER), both runs.
  3. Per-category arithmetic-mistake density, both runs.
  4. CORRECT-with-mistake rate (sloppy-correct), both runs — answers "did the
     PRM clean up correct rollouts too?".
  5. Trace-length distribution (mean / median / p90) — silent-reasoning check.
  6. Disagreement analysis: for each prompt, did the two runs differ in
     pass@1 outcome? Lists fixed-by-PRM and regressed-by-PRM cases.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path

from evaluation.countdown import extract_solution, validate_equation, evaluate_equation
from rloo_trainer.arithmetic_prm import compute_arithmetic_prm


_ARITH_RE = re.compile(r"(\d+)\s*([+\-*/])\s*(\d+)\s*=\s*(-?\d+)")


def categorize(text, gt):
    target, nums = gt["target"], gt["numbers"]
    eq = extract_solution(text)
    if eq is None:                      return "NO-ANSWER"
    if not validate_equation(eq, nums): return "BAD-NUMS"
    res = evaluate_equation(eq)
    if res is None:                     return "BAD-NUMS"
    return "CORRECT" if abs(res - target) < 1e-5 else "WRONG-VAL"


def total_ops_and_mistakes(text):
    n_ops, n_mis = 0, 0
    for m in _ARITH_RE.finditer(text):
        a, op, b, c = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))
        if op == "/" and b == 0: continue
        if op == "/":
            actual = a / b
            if abs(actual - round(actual)) < 1e-9:
                actual = int(round(actual))
            else:
                n_ops += 1; n_mis += 1; continue
        else:
            actual = {"+": a+b, "-": a-b, "*": a*b}[op]
        n_ops += 1
        if actual != c: n_mis += 1
    return n_ops, n_mis


def load_eval(path: Path) -> tuple[list[dict], list[dict]]:
    """Return (rollouts, prompts) where rollouts is flat list of per-sample dicts,
    prompts is list of per-prompt dicts."""
    rollouts, prompts = [], []
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            gt = {"target": d["target"], "numbers": list(d["nums"])}
            samples = []
            for r, s in zip(d["response"], d["scores"]):
                n_ops, n_mis = total_ops_and_mistakes(r)
                samples.append({
                    "text": r, "score": s, "category": categorize(r, gt),
                    "n_ops": n_ops, "n_mistakes": n_mis,
                    "length_chars": len(r),
                })
                rollouts.append({"nums": gt["numbers"], "target": gt["target"], **samples[-1]})
            prompts.append({"nums": gt["numbers"], "target": gt["target"], "samples": samples})
    return rollouts, prompts


def pct(n, d): return f"{n/max(1,d):.1%}"


def report_one(name: str, rollouts, prompts):
    print(f"\n=== {name}  ({len(rollouts)} rollouts, {len(prompts)} prompts) ===")
    # pass@k
    pass1 = sum(p["samples"][0]["score"] == 1.0 for p in prompts) / len(prompts)
    pass16 = sum(any(s["score"] == 1.0 for s in p["samples"]) for p in prompts) / len(prompts)
    mean_score = sum(r["score"] for r in rollouts) / len(rollouts)
    print(f"  pass@1     : {pass1:.3f}")
    print(f"  pass@16    : {pass16:.3f}")
    print(f"  mean score : {mean_score:.3f}")
    return {"pass1": pass1, "pass16": pass16, "mean_score": mean_score,
            "rollouts": rollouts, "prompts": prompts}


def report_categories(name, rollouts):
    cats = Counter(r["category"] for r in rollouts)
    N = len(rollouts)
    print(f"\n  {name} category distribution:")
    for c in ("CORRECT", "WRONG-VAL", "BAD-NUMS", "NO-ANSWER"):
        n = cats.get(c, 0)
        print(f"    {c:<11} {n:>4}  ({pct(n, N)})")
    return cats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline_json", type=Path)
    ap.add_argument("treatment_json", type=Path)
    args = ap.parse_args()

    base_r, base_p   = load_eval(args.baseline_json)
    treat_r, treat_p = load_eval(args.treatment_json)

    print("=" * 80)
    print("HEADLINE")
    print("=" * 80)
    baseline = report_one("BASELINE  " + args.baseline_json.name,  base_r,  base_p)
    treat    = report_one("TREATMENT " + args.treatment_json.name, treat_r, treat_p)
    print(f"\n  Δ pass@1     : {treat['pass1'] - baseline['pass1']:+.3f}")
    print(f"  Δ pass@16    : {treat['pass16'] - baseline['pass16']:+.3f}")
    print(f"  Δ mean score : {treat['mean_score'] - baseline['mean_score']:+.3f}")

    print("\n" + "=" * 80)
    print("CATEGORY DISTRIBUTIONS")
    print("=" * 80)
    base_cats  = report_categories("BASELINE",  base_r)
    treat_cats = report_categories("TREATMENT", treat_r)
    print(f"\n  shift (treatment minus baseline):")
    for c in ("CORRECT", "WRONG-VAL", "BAD-NUMS", "NO-ANSWER"):
        delta = treat_cats.get(c, 0) - base_cats.get(c, 0)
        print(f"    {c:<11} {delta:+5}")

    print("\n" + "=" * 80)
    print("ARITHMETIC-MISTAKE DENSITY (by category)")
    print("=" * 80)
    print(f"  {'category':<11} {'baseline':<26} | {'treatment':<26}")
    print(f"  {'':<11} {'mistakes/rollout':<13} {'≥1':<12} | {'mistakes/rollout':<13} {'≥1':<12}")
    for c in ("CORRECT", "WRONG-VAL", "BAD-NUMS", "NO-ANSWER"):
        bcr = [r for r in base_r  if r["category"] == c]
        tcr = [r for r in treat_r if r["category"] == c]
        b_mean = sum(r["n_mistakes"] for r in bcr) / max(1, len(bcr))
        t_mean = sum(r["n_mistakes"] for r in tcr) / max(1, len(tcr))
        b_w    = sum(1 for r in bcr if r["n_mistakes"] > 0)
        t_w    = sum(1 for r in tcr if r["n_mistakes"] > 0)
        print(f"  {c:<11} "
              f"{b_mean:>5.2f}        {pct(b_w, len(bcr)):>5} ({b_w}/{len(bcr)}) | "
              f"{t_mean:>5.2f}        {pct(t_w, len(tcr)):>5} ({t_w}/{len(tcr)})")

    print("\n" + "=" * 80)
    print("TRACE-LENGTH DISTRIBUTION (chars)  — silent-reasoning check")
    print("=" * 80)
    for label, rollouts in (("BASELINE", base_r), ("TREATMENT", treat_r)):
        lens = sorted(r["length_chars"] for r in rollouts)
        mean = sum(lens) / len(lens)
        med  = statistics.median(lens)
        p90  = lens[int(0.9 * len(lens))]
        p10  = lens[int(0.1 * len(lens))]
        print(f"  {label:<10}  mean={mean:>6.0f}  p10={p10:>5}  median={med:>5.0f}  p90={p90:>5}")
        # Mean op count too
        op_mean = sum(r["n_ops"] for r in rollouts) / len(rollouts)
        print(f"  {label:<10}  mean arith ops/rollout = {op_mean:.2f}")

    print("\n" + "=" * 80)
    print("PASS@1 DISAGREEMENTS (per prompt, sample[0])")
    print("=" * 80)
    n_prompts = min(len(base_p), len(treat_p))
    fixed, regressed, both_right, both_wrong = 0, 0, 0, 0
    fixed_examples, regressed_examples = [], []
    for i in range(n_prompts):
        b_right = base_p[i]["samples"][0]["score"]  == 1.0
        t_right = treat_p[i]["samples"][0]["score"] == 1.0
        if b_right and t_right:         both_right += 1
        elif b_right and not t_right:   regressed += 1; regressed_examples.append(i)
        elif not b_right and t_right:   fixed     += 1; fixed_examples.append(i)
        else:                           both_wrong += 1
    print(f"  both right (no change)            : {both_right}")
    print(f"  PRM fixed (baseline wrong → right): {fixed}")
    print(f"  PRM regressed (right → wrong)     : {regressed}")
    print(f"  both wrong                        : {both_wrong}")
    print(f"  net pass@1 effect                 : {fixed - regressed:+}")

    print(f"\n  Sample of PRM-fixed prompts (first 3):")
    for idx in fixed_examples[:3]:
        bp, tp = base_p[idx], treat_p[idx]
        print(f"    nums={bp['nums']}  target={bp['target']}")
        print(f"      baseline   ({bp['samples'][0]['category']:<9}): {bp['samples'][0]['text'][:200]!r}…")
        print(f"      treatment  ({tp['samples'][0]['category']:<9}): {tp['samples'][0]['text'][:200]!r}…")

    print(f"\n  Sample of PRM-regressed prompts (first 3):")
    for idx in regressed_examples[:3]:
        bp, tp = base_p[idx], treat_p[idx]
        print(f"    nums={bp['nums']}  target={bp['target']}")
        print(f"      baseline   ({bp['samples'][0]['category']:<9}): {bp['samples'][0]['text'][:200]!r}…")
        print(f"      treatment  ({tp['samples'][0]['category']:<9}): {tp['samples'][0]['text'][:200]!r}…")

    print("\n" + "=" * 80)
    print("PRM SCORE COMPARISON  (what our arithmetic_prm would score these rollouts)")
    print("=" * 80)
    for label, rollouts in (("BASELINE", base_r), ("TREATMENT", treat_r)):
        prms = [compute_arithmetic_prm(r["text"], {})[0] for r in rollouts]
        mean = sum(prms) / len(prms)
        nz   = sum(1 for p in prms if p < 0)
        capped = sum(1 for p in prms if abs(p - (-0.5)) < 1e-9)
        print(f"  {label:<10}  PRM mean = {mean:+.4f}   "
              f"%nonzero = {pct(nz, len(prms))}   "
              f"%capped@-0.5 = {pct(capped, len(prms))}")


if __name__ == "__main__":
    main()
