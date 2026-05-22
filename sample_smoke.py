"""Plain script: sample one batch from a (SELS) checkpoint via the REAL RLOO
SamplingWorker, score it, and dump trajectories for inspection.

Run on Modal exactly like the trainers (subprocess inside the training container):

    modal run modal_train.py smoke --model_path /vol/.../best_model/model
    modal volume get default-proj-training /evaluation/eval_results/rloo_smoke.json .
"""
import argparse
import json
import os
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

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="/vol/checkpoints/sft_model/sels_sft/sels_sft_v3/best_model/model")
    ap.add_argument("--eval_dataset", default="asingh15/countdown_tasks_3to4")
    ap.add_argument("--n_prompts", type=int, default=6)
    ap.add_argument("--group_size", type=int, default=8)
    ap.add_argument("--inject_state0", action="store_true",
                    help='Seed "<state_0> [nums]" into each prompt (mirrors rloo.py).')
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
    worker = SamplingWorker.remote(
        model_path=args.model_path,
        temperature=1.0, top_p=1.0, top_k=-1, min_p=0.0,  # RLOO-style sampling
        max_tokens=1024, group_size=args.group_size,
    )
    ray.get(worker.load_checkpoint.remote())
    responses, logprobs = ray.get(worker.generate.remote(prompts))

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
