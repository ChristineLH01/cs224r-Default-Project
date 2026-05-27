"""Failure-mode analysis for an RLOO-trained checkpoint's trajectories.

Reads the `_trajectories.txt` file produced by `evaluation/countdown_eval.py`
and reports:
  1. Per-category breakdown (CORRECT / WRONG-VAL / BAD-NUMS / NO-ANSWER).
  2. For each category, how often the trace contains a detectable arithmetic
     mistake (regex `\\d+ OP \\d+ = \\d+`, check `eval(LHS) == RHS`).
  3. Histogram of arithmetic-mistakes-per-rollout, separately by category.
  4. A handful of sampled traces per category, with the offending arithmetic
     steps highlighted.

Used to diagnose whether the default-RLOO model's wrong answers stem from
arithmetic execution errors (where an arithmetic-PRM could help) vs. wrong
search (where it can't).

Usage:
    python evaluation/analyze_failure_modes.py <path/to/_trajectories.txt> [--n_samples 3]
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path


# Catches "a OP b = c" with optional whitespace. ASCII only — we accept the
# rare miss on natural-language arithmetic as "no signal" rather than penalty.
_ARITH_RE = re.compile(r"(\d+)\s*([+\-*/])\s*(\d+)\s*=\s*(-?\d+)")

# Trajectory file structure (from evaluation/countdown_eval.py):
#   === nums=[...] target=... gt={...} ===
#   prompt: ...
#   --- response[NN] [CATEGORY] chars=N ---
#   <trace text>
#   --- response[NN] [CATEGORY] chars=N ---
#   ...
_PROMPT_HEADER_RE   = re.compile(r"^=== nums=(\[[^\]]*\])\s+target=(-?\d+)\s+gt=(\{[^}]*\})\s+===$")
_RESPONSE_HEADER_RE = re.compile(r"^--- response\[(\d+)\] \[([A-Z\-]+)\] chars=(\d+) ---$")


def parse_trajectories(path: Path) -> list[dict]:
    """Return list of {nums, target, category, text}.

    One row per (prompt, sample). nums/target are repeated across the
    samples belonging to one prompt for convenience.
    """
    rows: list[dict] = []
    cur_nums: list[int] | None = None
    cur_target: int | None = None
    cur_category: str | None = None
    cur_buf: list[str] = []

    def flush():
        if cur_category is not None and cur_nums is not None:
            rows.append({
                "nums":     cur_nums,
                "target":   cur_target,
                "category": cur_category,
                "text":     "".join(cur_buf).rstrip("\n"),
            })

    with path.open("r") as f:
        for line in f:
            m_prompt = _PROMPT_HEADER_RE.match(line.rstrip("\n"))
            m_resp   = _RESPONSE_HEADER_RE.match(line.rstrip("\n"))
            if m_prompt is not None:
                flush()
                cur_buf = []; cur_category = None
                cur_nums   = eval(m_prompt.group(1))           # safe — only ints
                cur_target = int(m_prompt.group(2))
                continue
            if m_resp is not None:
                flush()
                cur_buf = []
                cur_category = m_resp.group(2)
                continue
            # Skip the "prompt: ..." line so it doesn't pollute the trace text.
            if line.startswith("prompt:"):
                continue
            cur_buf.append(line)
    flush()
    return rows


def find_arith_mistakes(text: str) -> list[tuple[int, str, int, int, int]]:
    """Return list of (a, op, b, declared_c, actual_c) for each mismatched
    `a OP b = c` step in `text`. Skips division-by-zero (no actual value).
    """
    bad: list[tuple[int, str, int, int, int]] = []
    for m in _ARITH_RE.finditer(text):
        a, op, b, c = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))
        if op == "/" and b == 0:
            continue
        actual = {"+": a + b, "-": a - b, "*": a * b, "/": (a / b if op == "/" else None)}[op]
        # tolerate integer-valued floats from division
        if op == "/":
            if abs(actual - round(actual)) < 1e-9:
                actual = int(round(actual))
            else:
                # declared int can never equal a non-integer actual → mistake.
                bad.append((a, op, b, c, actual))
                continue
        if actual != c:
            bad.append((a, op, b, c, actual))
    return bad


def find_all_arith(text: str) -> int:
    """Count of all detected `a OP b = c` steps (including correct ones)."""
    n = 0
    for m in _ARITH_RE.finditer(text):
        a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
        if op == "/" and b == 0:
            continue
        n += 1
    return n


def summarize(rows: list[dict], n_samples: int = 3) -> None:
    cat_counts = Counter(r["category"] for r in rows)
    n_total = len(rows)
    print(f"\n=== category distribution ({n_total} rollouts total) ===")
    for cat, n in cat_counts.most_common():
        print(f"  {cat:<12} {n:>5}  ({n/n_total:.1%})")

    print(f"\n=== arithmetic mistakes per category ===")
    print(f"  {'category':<12} {'rollouts':>9} {'≥1 mistake':>11} {'mean ops':>9} {'mean mistakes':>14}")
    for cat in sorted(cat_counts, key=lambda c: -cat_counts[c]):
        cat_rows = [r for r in rows if r["category"] == cat]
        n_ops_list      = [find_all_arith(r["text"])    for r in cat_rows]
        n_mistakes_list = [len(find_arith_mistakes(r["text"])) for r in cat_rows]
        n_with_mistake  = sum(1 for n in n_mistakes_list if n > 0)
        mean_ops        = sum(n_ops_list) / max(1, len(cat_rows))
        mean_mistakes   = sum(n_mistakes_list) / max(1, len(cat_rows))
        print(f"  {cat:<12} {len(cat_rows):>9} "
              f"{n_with_mistake:>5}/{len(cat_rows):<5} "
              f"{mean_ops:>9.2f} {mean_mistakes:>14.2f}")

    # Per-category mistake histogram (capped at 5+)
    print(f"\n=== mistakes-per-rollout histogram (capped at 5+) ===")
    print(f"  {'category':<12} " + "".join(f"{k:>5}" for k in ("0", "1", "2", "3", "4", "5+")))
    for cat in sorted(cat_counts, key=lambda c: -cat_counts[c]):
        cat_rows = [r for r in rows if r["category"] == cat]
        hist = Counter()
        for r in cat_rows:
            n = len(find_arith_mistakes(r["text"]))
            hist[min(n, 5)] += 1
        print(f"  {cat:<12} " + "".join(f"{hist.get(k, 0):>5}" for k in (0, 1, 2, 3, 4, 5)))

    # Sample traces per category
    print(f"\n=== sampled traces (up to {n_samples} per category) ===")
    for cat in sorted(cat_counts, key=lambda c: -cat_counts[c]):
        cat_rows = [r for r in rows if r["category"] == cat]
        # Prefer rows that have at least one mistake (more interesting)
        with_mistake = [r for r in cat_rows if find_arith_mistakes(r["text"])]
        picks = (with_mistake[:n_samples]
                 if cat in ("WRONG-VAL", "BAD-NUMS", "NO-ANSWER")
                 else cat_rows[:n_samples])
        for r in picks:
            mistakes = find_arith_mistakes(r["text"])
            print(f"\n  --- {cat}  nums={r['nums']}  target={r['target']}  mistakes={len(mistakes)} ---")
            for (a, op, b, c, actual) in mistakes:
                print(f"      MISTAKE: {a} {op} {b} = {c}   (actual: {actual})")
            # Print just the first 600 chars of the trace
            t = r["text"]
            print(f"      trace[0:600]: {t[:600]!r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("trajectories", type=Path,
                    help="Path to _trajectories.txt produced by countdown_eval.py")
    ap.add_argument("--n_samples", type=int, default=3,
                    help="Sampled traces to display per category (default 3)")
    args = ap.parse_args()

    rows = parse_trajectories(args.trajectories)
    summarize(rows, n_samples=args.n_samples)
