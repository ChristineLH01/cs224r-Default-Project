"""Synthesize RFT (rejection-sampling fine-tuning) data from rollouts.

Two selection strategies (mutually exclusive):

  --top_k K                  (v1/v2 default)
      Length-biased: per prompt, keep the K SHORTEST correct rollouts.

  --top_k_distinct K         (v3, canonical RFT)
      Diversity-biased: per prompt, dedupe correct rollouts by exact trace
      match, then randomly cap at K distinct traces. Matches Yuan et al.
      (2023) RFT — multiple distinct reasoning paths per prompt amplify
      pass@1 without collapsing the policy.

Common to both:
  - Filter to rollouts with compute_score == 1.0 (CORRECT)
  - Strip chat-template wrap so sft_dataset.py can re-wrap cleanly

Input:  one or more JSONL files (--input file1.json file2.json ...) with
        cols: prompt, target, nums, ground_truth, response, scores
Output: HF DatasetDict at --output_dir, columns: query, completion
"""
from __future__ import annotations

import json
import argparse
import random
from pathlib import Path

from datasets import Dataset, DatasetDict


_USER_OPEN = "<|im_start|>user\n"
_USER_CLOSE = "<|im_end|>"
_CORRECT_SCORE = 1.0


def extract_user_query(chat_templated_prompt: str) -> str:
    """Pull the user-turn content out of a Qwen chat-templated prompt."""
    start = chat_templated_prompt.index(_USER_OPEN) + len(_USER_OPEN)
    end = chat_templated_prompt.index(_USER_CLOSE, start)
    return chat_templated_prompt[start:end]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, nargs="+", required=True,
                    help="One or more rollouts JSONL files. Each prompt across "
                         "files contributes up to --top_k examples.")
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--top_k", type=int, default=1,
                    help="Per prompt, keep the top_k shortest correct rollouts "
                         "(k=1 = strict argmin; k>1 trades data size for diversity). "
                         "Ignored if --top_k_distinct is set.")
    ap.add_argument("--top_k_distinct", type=int, default=0,
                    help="If >0: per prompt, dedupe correct rollouts by exact-text "
                         "match, then randomly cap at K distinct traces. Mutually "
                         "exclusive with --top_k; overrides if both are set.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Random seed for --top_k_distinct sampling (deterministic).")
    ap.add_argument("--eval_frac", type=float, default=0.05)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    n_total = 0
    n_with_correct = 0
    n_distinct_per_prompt = []  # only populated for --top_k_distinct mode
    pairs = []
    for path in args.input:
        with path.open() as f:
            for line in f:
                row = json.loads(line)
                n_total += 1
                scores = row["scores"]
                responses = row["response"]
                correct = [i for i, s in enumerate(scores) if abs(s - _CORRECT_SCORE) < 1e-6]
                if not correct:
                    continue
                n_with_correct += 1
                query = extract_user_query(row["prompt"])

                if args.top_k_distinct > 0:
                    # v3: dedupe by exact trace, then randomly cap at K distinct.
                    distinct_traces = list({responses[i] for i in correct})
                    n_distinct_per_prompt.append(len(distinct_traces))
                    if len(distinct_traces) > args.top_k_distinct:
                        distinct_traces = rng.sample(distinct_traces, args.top_k_distinct)
                    for trace in distinct_traces:
                        pairs.append({"query": query, "completion": trace})
                else:
                    # v1/v2: top_k SHORTEST.
                    ranked = sorted(correct, key=lambda i: len(responses[i]))
                    for idx in ranked[:args.top_k]:
                        pairs.append({"query": query, "completion": responses[idx]})

    n_eval = max(1, int(len(pairs) * args.eval_frac))
    train_rows = pairs[:-n_eval]
    test_rows = pairs[-n_eval:]

    dd = DatasetDict({
        "train": Dataset.from_list(train_rows),
        "test": Dataset.from_list(test_rows),
    })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dd.save_to_disk(str(args.output_dir))

    print(f"input files:                 {[str(p) for p in args.input]}")
    print(f"input prompts (total rows):  {n_total}")
    print(f"prompts with >=1 correct:    {n_with_correct} ({n_with_correct/n_total*100:.1f}%)")
    if args.top_k_distinct > 0:
        print(f"strategy:                    top_k_distinct={args.top_k_distinct} (seed {args.seed})")
        if n_distinct_per_prompt:
            sd = sorted(n_distinct_per_prompt)
            print(f"distinct correct/prompt:     median={sd[len(sd)//2]}  p95={sd[int(len(sd)*0.95)]}  max={sd[-1]}")
    else:
        print(f"strategy:                    top_k={args.top_k} (length-biased)")
    print(f"total (query, completion) examples: {len(pairs)}")
    print(f"  -> train: {len(train_rows)}  test: {len(test_rows)}")
    print(f"saved to {args.output_dir}")

    lens = sorted(len(p["completion"]) for p in pairs)
    print(f"completion char length — median: {lens[len(lens)//2]}  p95: {lens[int(len(lens)*0.95)]}  max: {lens[-1]}")


if __name__ == "__main__":
    main()
