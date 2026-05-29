"""Synthesize anti-ramble IPO pairs from rollouts.

Criterion (data-driven; see length distribution analysis):

  CHOSEN pool   = compute_score == 1.0                  (any-length CORRECT)
  REJECTED pool = NOT correct  AND  len(text) >= MIN_RAMBLING_LENGTH
                                                         (long failure = rambling)

Why this is the right cut (empirical, see analysis on 64K cont. latest rollouts):
  - CORRECT p95 length = 931 chars
  - NO-ANSWER p10 length = 1695 chars
  - The 931-1695 gap is the empirical "rambling boundary": successful reasoning
    fits below it, rambling fails above it.
  - Default MIN_RAMBLING_LENGTH = 1200 (sits in the gap):
      ~100% of NO-ANSWERs captured (the dominant rambling failure)
      ~70% of WRONG-VAL captured (rambling tails)
      ~30% of BAD-NUMS captured (rambling tails)
      Only ~1.9% of correct rollouts above this length (false-positive risk
      on the rejected side is low — and chosen pool isn't length-filtered
      anyway).

Why no length sort or "shortest correct" / "longest bad" engineering:
  - Sorting by length on either side creates a surface shortcut: IPO would
    learn "shorter is better" instead of "non-rambling is better".
  - The MIN_RAMBLING_LENGTH filter alone puts rambling in the rejected pool;
    we then sample randomly within filtered pools (top-K distinct, mirroring
    RFT v3 selection).

Pairing strategy:
  - For each prompt with >=1 chosen-qualifier AND >=1 rejected-qualifier
  - Dedupe both pools by exact-text match
  - Random sample top-K from each pool (default K=2; deterministic via --seed)
  - Pair index-by-index -> (chosen[i], rejected[i])

Input:  one or more JSONL rollout files (--input ...) with cols
        prompt, target, nums, ground_truth, response, scores
Output: HF DatasetDict at --output_dir with cols query, response_ws,
        response_ls -- the defaults expected by ipo_trainer.ipo_dataset.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from datasets import Dataset, DatasetDict

from evaluation.countdown import (
    extract_solution, validate_equation, evaluate_equation,
)
from rloo_trainer.arithmetic_prm import compute_arithmetic_prm


_USER_OPEN = "<|im_start|>user\n"
_USER_CLOSE = "<|im_end|>"
_CORRECT_SCORE = 1.0


def extract_user_query(chat_templated_prompt: str) -> str:
    start = chat_templated_prompt.index(_USER_OPEN) + len(_USER_OPEN)
    end = chat_templated_prompt.index(_USER_CLOSE, start)
    return chat_templated_prompt[start:end]


def categorize(text: str, gt: dict) -> str:
    eq = extract_solution(text)
    if eq is None: return "NO-ANSWER"
    if not validate_equation(eq, gt["numbers"]): return "BAD-NUMS"
    res = evaluate_equation(eq)
    if res is None: return "BAD-NUMS"
    return "CORRECT" if abs(res - gt["target"]) < 1e-5 else "WRONG-VAL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, nargs="+", required=True,
                    help="One or more rollouts JSONL files (e.g., RFT v3 sample).")
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--top_k", type=int, default=2,
                    help="Max pairs per prompt. Per prompt, we sample top_k "
                         "distinct from chosen and top_k from rejected.")
    ap.add_argument("--min_rambling_length", type=int, default=1200,
                    help="Char-length threshold for the rejected pool: only "
                         "non-correct rollouts with length >= this count.")
    ap.add_argument("--max_chosen_length", type=int, default=0,
                    help="If >0: keep only CORRECT rollouts with len <= this "
                         "char count in the chosen pool (engineered length "
                         "contrast with rejected). 0 = no length filter (v1 "
                         "default).")
    ap.add_argument("--clean_chosen", action="store_true",
                    help="If set: drop CORRECT rollouts that contain any "
                         "arithmetic mistake (n_total_mistakes > 0 per "
                         "compute_arithmetic_prm). Keeps only traces whose "
                         "step-by-step calculations are all valid.")
    ap.add_argument("--eval_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    n_prompts = 0
    n_with_chosen = 0
    n_with_rejected = 0
    n_qualifying = 0
    n_pairs = 0
    n_correct_seen = 0
    n_correct_dropped_long = 0
    n_correct_dropped_dirty = 0
    rejected_cat_counter = Counter()
    chosen_lens: list[int] = []
    rejected_lens: list[int] = []
    pairs = []

    for path in args.input:
        with path.open() as f:
            for line in f:
                row = json.loads(line)
                n_prompts += 1
                gt = {"target": row["target"], "numbers": list(row["nums"])}
                responses = row["response"]

                chosen_idxs = []
                rejected_idxs = []
                for i, r in enumerate(responses):
                    cat = categorize(r, gt)
                    if cat == "CORRECT":
                        n_correct_seen += 1
                        # Filter 1: length cap on chosen.
                        if args.max_chosen_length > 0 and len(r) > args.max_chosen_length:
                            n_correct_dropped_long += 1
                            continue
                        # Filter 2: require zero arithmetic mistakes.
                        if args.clean_chosen:
                            _, m = compute_arithmetic_prm(r, gt)
                            if m["n_total_mistakes"] > 0:
                                n_correct_dropped_dirty += 1
                                continue
                        chosen_idxs.append(i)
                    elif len(r) >= args.min_rambling_length:
                        # Long non-correct = rambling failure (any subcategory).
                        rejected_idxs.append(i)

                if chosen_idxs: n_with_chosen += 1
                if rejected_idxs: n_with_rejected += 1
                if not chosen_idxs or not rejected_idxs:
                    continue
                n_qualifying += 1

                # Dedupe by exact trace text, then random-sample top_k.
                chosen_unique = list({responses[i] for i in chosen_idxs})
                rejected_unique = list({responses[i] for i in rejected_idxs})
                rng.shuffle(chosen_unique)
                rng.shuffle(rejected_unique)
                k = min(args.top_k, len(chosen_unique), len(rejected_unique))

                query = extract_user_query(row["prompt"])
                for j in range(k):
                    ci_text = chosen_unique[j]
                    rj_text = rejected_unique[j]
                    pairs.append({
                        "query": query,
                        "response_ws": ci_text,
                        "response_ls": rj_text,
                    })
                    rejected_cat_counter[categorize(rj_text, gt)] += 1
                    chosen_lens.append(len(ci_text))
                    rejected_lens.append(len(rj_text))
                    n_pairs += 1

    # Deterministic 95/5 split: shuffle so split isn't biased by file order.
    rng.shuffle(pairs)
    n_eval = max(1, int(len(pairs) * args.eval_frac))
    train_rows = pairs[:-n_eval]
    test_rows  = pairs[-n_eval:]

    dd = DatasetDict({
        "train": Dataset.from_list(train_rows),
        "test":  Dataset.from_list(test_rows),
    })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dd.save_to_disk(str(args.output_dir))

    print(f"=== synthesize_ipo_pairs  ===")
    print(f"input files:                 {[str(p) for p in args.input]}")
    print(f"input prompts (rows):        {n_prompts}")
    print(f"prompts with >=1 chosen:     {n_with_chosen} ({n_with_chosen/n_prompts*100:.1f}%)")
    print(f"prompts with >=1 rejected:   {n_with_rejected} ({n_with_rejected/n_prompts*100:.1f}%)")
    print(f"prompts with both (qualifying): {n_qualifying} ({n_qualifying/n_prompts*100:.1f}%)")
    print(f"top_k (pairs per prompt cap): {args.top_k}")
    print(f"min_rambling_length:         {args.min_rambling_length} chars")
    print(f"max_chosen_length:           {args.max_chosen_length if args.max_chosen_length > 0 else '(no filter)'}")
    print(f"clean_chosen:                {args.clean_chosen}")
    if args.max_chosen_length > 0 or args.clean_chosen:
        print(f"  CORRECT rollouts seen:           {n_correct_seen}")
        print(f"  dropped (length > max):          {n_correct_dropped_long}")
        print(f"  dropped (had arith mistake):     {n_correct_dropped_dirty}")
        kept = n_correct_seen - n_correct_dropped_long - n_correct_dropped_dirty
        print(f"  kept in chosen pool:             {kept} ({kept/n_correct_seen*100:.1f}%)" if n_correct_seen else "")
    print(f"total pairs:                 {n_pairs}")
    print(f"  -> train: {len(train_rows)}  test: {len(test_rows)}")
    print(f"rejected failure-mode distribution:")
    total_rej = sum(rejected_cat_counter.values()) or 1
    for k, v in rejected_cat_counter.most_common():
        print(f"    {k:<11} {v:>5} ({v/total_rej*100:.1f}%)")
    if chosen_lens:
        cl, rl = sorted(chosen_lens), sorted(rejected_lens)
        print(f"chosen   chars: median={cl[len(cl)//2]}  p95={cl[int(len(cl)*0.95)]}  max={cl[-1]}")
        print(f"rejected chars: median={rl[len(rl)//2]}  p95={rl[int(len(rl)*0.95)]}  max={rl[-1]}")
    print(f"saved to {args.output_dir}")


if __name__ == "__main__":
    main()
