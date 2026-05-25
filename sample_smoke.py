"""Plain script: sample one batch from a (SELS) checkpoint via the REAL RLOO
SamplingWorker, score it, and dump trajectories for inspection.

Run on Modal exactly like the trainers (subprocess inside the training container):

    modal run modal_train.py smoke --model_path /vol/.../best_model/model
    modal volume get default-proj-training /evaluation/eval_results/rloo_smoke.json .

With context restoration (segmented sampling, SELS-only):

    modal run modal_train.py smoke \\
        --model_path /vol/checkpoints/sft_model/sels_sft/sels_sft_v5/best_model/model \\
        --context_restoration --n_prompts 4 --group_size 4 \\
        --output /vol/evaluation/eval_results/rloo_smoke_ctxmgr.json

Diagnostics printed at the end let you eyeball: end-with-</answer> rate, mean
backtracks per response, mean structural PRM reward, and (when ctx restoration
is on) whether any traces contain orphan failed-branch text — i.e., a
`<backtrack_K>` not immediately followed by `<state_K> [list]`. A working
context manager should have zero orphans.
"""
import argparse
import json
import os
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import ray

PROJECT_ROOT = str(Path(__file__).resolve().parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets import load_dataset

from evaluation.countdown import compute_score
from rloo_trainer.sampling_worker import SamplingWorker
from rloo_trainer.process_reward import compute_process_reward


_BACKTRACK_TOK = re.compile(r"<backtrack_(\d+)>")
# A "good" backtrack is `<backtrack_K>\s*<state_K>` — the restored-state token
# immediately follows. Anything else (text, EOS, a different depth) is orphan.
_GOOD_BACKTRACK = re.compile(r"<backtrack_(\d+)>\s*<state_\1>")


def _diagnose(text: str) -> dict:
    n_bt = len(_BACKTRACK_TOK.findall(text))
    n_good_bt = len(_GOOD_BACKTRACK.findall(text))
    return {
        "n_backtracks":         n_bt,
        "n_orphan_backtracks":  n_bt - n_good_bt,
        "has_answer":           "<answer>" in text and "</answer>" in text,
        "ends_with_answer":     text.rstrip().endswith("</answer>"),
        "length_chars":         len(text),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="/vol/checkpoints/sft_model/sels_sft/sels_sft_v3/best_model/model")
    ap.add_argument("--eval_dataset", default="asingh15/countdown_tasks_3to4")
    ap.add_argument("--n_prompts", type=int, default=6)
    ap.add_argument("--group_size", type=int, default=8)
    ap.add_argument("--inject_state0", action="store_true",
                    help='Seed "<state_0> [nums]" into each prompt (mirrors rloo.py).')
    ap.add_argument("--context_restoration", action="store_true",
                    help='SELS only: route sampling through the segmented '
                         'context-restoration manager (mirrors --context_restoration '
                         'on rloo.py). Default off → standard generate() path.')
    ap.add_argument("--max_restoration_rounds", type=int, default=4)
    ap.add_argument("--output", default="/vol/evaluation/eval_results/rloo_smoke.json")
    args = ap.parse_args()

    ds = load_dataset(args.eval_dataset, split="test")
    prompts = list(ds["prompt"])[: args.n_prompts]
    gts = [
        {"target": int(t), "numbers": list(n)}
        for t, n in zip(ds["target"][: args.n_prompts], ds["nums"][: args.n_prompts])
    ]
    if args.inject_state0:
        prompts = [f"{p}<state_0> {sorted(gt['numbers'])}\n" for p, gt in zip(prompts, gts)]

    ray.init()
    # Gate the new SamplingWorker kwargs so the default-mode constructor call
    # is byte-identical to pre-context-restoration. The worker's own defaults
    # cover the missing kwargs (enable_context_restoration=False).
    worker_kwargs = dict(
        model_path=args.model_path,
        temperature=1.0, top_p=1.0, top_k=-1, min_p=0.0,  # RLOO-style sampling
        max_tokens=1024, group_size=args.group_size,
    )
    if args.context_restoration:
        worker_kwargs["enable_context_restoration"] = True
        worker_kwargs["max_restoration_rounds"] = args.max_restoration_rounds
    worker = SamplingWorker.remote(**worker_kwargs)
    ray.get(worker.load_checkpoint.remote())
    if args.context_restoration:
        responses, logprobs = ray.get(
            worker.generate_with_context_restoration.remote(prompts)
        )
    else:
        responses, logprobs = ray.get(worker.generate.remote(prompts))

    if not args.context_restoration:
        # Default path — byte-identical output shape + stdout to pre-session.
        out = []
        for p, gt, resps, lps in zip(prompts, gts, responses, logprobs):
            out.append({
                "prompt": p,
                "ground_truth": gt,
                "samples": [
                    {"reward": compute_score(r, gt), "logprob": float(lp), "text": r}
                    for r, lp in zip(resps, lps)
                ],
            })
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f)
        print(f"Wrote {len(out)} prompts x {args.group_size} samples to {args.output}")
    else:
        # Context-restoration smoke: enriched JSON ({args, agg, rollouts})
        # with per-sample structural PRM + diagnostic counts + stdout summary.
        out = []
        agg = {"n_samples": 0, "ends_with_answer": 0, "has_answer": 0,
               "n_backtracks": 0, "n_orphan_backtracks": 0,
               "compute_score_sum": 0.0, "structural_reward_sum": 0.0}
        for p, gt, resps, lps in zip(prompts, gts, responses, logprobs):
            sample_dicts = []
            for r, lp in zip(resps, lps):
                r_score = compute_score(r, gt)
                r_struct, _ = compute_process_reward(r, gt)
                diag = _diagnose(r)
                sample_dicts.append({
                    "reward": r_score,
                    "structural_reward": r_struct,
                    "logprob": float(lp),
                    "diag": diag,
                    "text": r,
                })
                agg["n_samples"]              += 1
                agg["ends_with_answer"]       += int(diag["ends_with_answer"])
                agg["has_answer"]             += int(diag["has_answer"])
                agg["n_backtracks"]           += diag["n_backtracks"]
                agg["n_orphan_backtracks"]    += diag["n_orphan_backtracks"]
                agg["compute_score_sum"]      += float(r_score)
                agg["structural_reward_sum"]  += float(r_struct)
            out.append({"prompt": p, "ground_truth": gt, "samples": sample_dicts})

        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"args": vars(args), "agg": agg, "rollouts": out}, f)

        n = max(1, agg["n_samples"])
        print()
        print(f"=== smoke summary  (context_restoration=True) ===")
        print(f"  rollouts                       : {agg['n_samples']}")
        print(f"  ends with </answer>            : {agg['ends_with_answer']}/{n}  ({agg['ends_with_answer']/n:.1%})")
        print(f"  has <answer>...</answer> block : {agg['has_answer']}/{n}  ({agg['has_answer']/n:.1%})")
        print(f"  mean backtracks / rollout      : {agg['n_backtracks']/n:.2f}")
        print(f"  ORPHAN backtracks (must be 0)  : {agg['n_orphan_backtracks']}")
        print(f"  mean compute_score             : {agg['compute_score_sum']/n:.3f}")
        print(f"  mean structural PRM            : {agg['structural_reward_sum']/n:.3f}")
        print(f"  → {args.output}")
