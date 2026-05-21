"""Orchestrator: solver -> skeleton -> GPT-4o-mini narrator -> HF Dataset.

Usage:
    OPENAI_API_KEY=... python dataset_generation/generate_sft_data.py \\
        --output dataset_generation/output/sels_sft_v3 --concurrency 3 \\
        --chain 900 --bt_inner 200 --bt_to_root 100 --bt_deep 0 --multi_bt 0

For local debugging without spending API credits, use --dry_run, which substitutes
a deterministic stub narrator that emits a minimal-compliant completion.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from pathlib import Path
from typing import Optional

# Allow `python dataset_generation/generate_sft_data.py` to resolve project root imports.
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset_generation.narrate_with_gpt import narrate_async, validate, ValidationError  # noqa: E402
from dataset_generation.skeleton_builder import (  # noqa: E402
    AnswerEvent,
    ArithStep,
    BacktrackEvent,
    ResumeEvent,
    Shape,
    Skeleton,
    StateEvent,
    annotate_expressions,
    arith_fact,
    build_skeleton,
    state_list,
)

# Priority order: most-restrictive shapes first so they grab 4-num puzzles before
# fall-through shapes (chain, bt_inner, bt_to_root) eat into the same pool.
SHAPE_PRIORITY: list[Shape] = ["multi_bt", "bt_deep", "bt_to_root", "bt_inner", "chain"]
# Shapes that require a 4-number puzzle (>= 3 arith steps).
FOUR_NUM_ONLY: set[Shape] = {"bt_deep", "multi_bt"}
# Defaults sized to match Asap7772/cog_behav_all_strategies (1000 train + 200 test = 1200 total).
# Mix exercises every special token: every row uses state_0, ~95% use state_1, ~55%
# use state_2; backtrack destinations are spread across state_0/1/2.
DEFAULT_TARGETS: dict[Shape, int] = {
    "multi_bt": 180,   # 15%  — two BTs at different depths (4-num only)
    "bt_deep": 240,    # 20%  — BT to state_2 (4-num only)
    "bt_to_root": 180, # 15%  — BT to state_0 from a non-root state
    "bt_inner": 360,   # 30%  — BT to state_1 (the canonical SELS pattern)
    "chain": 240,      # 20%  — multi-checkpoint walk, no BTs
}
DEFAULT_NUM_EXAMPLES = 1200
DEFAULT_TEST_SIZE = 200
DEFAULT_MAX_SOURCE_ROWS = 4000  # bumped: multi-state shapes need more candidates


# ---------------------------------------------------------------------------
# Source dataset access
# ---------------------------------------------------------------------------
def _extract_puzzle(row: dict) -> Optional[tuple[str, list[int], int]]:
    """Pull (prompt, nums, target) out of a row, accommodating either schema."""
    prompt = row.get("prompt")
    if prompt is None:
        return None
    if "ground_truth" in row and row["ground_truth"] is not None:
        gt = row["ground_truth"]
        if isinstance(gt, str):
            try:
                gt = json.loads(gt)
            except Exception:
                return None
        nums = gt.get("numbers")
        target = gt.get("target")
    else:
        nums = row.get("nums")
        target = row.get("target")
    if nums is None or target is None:
        return None
    return prompt, list(nums), int(target)


# ---------------------------------------------------------------------------
# Shape downgrade on infeasibility
# ---------------------------------------------------------------------------
def _try_build(
    nums: list[int], target: int, requested: Shape, rng: random.Random
) -> Optional[tuple[Shape, Skeleton]]:
    """Try requested shape; if infeasible, downgrade through SHAPE_PRIORITY."""
    start = SHAPE_PRIORITY.index(requested)
    for shape in SHAPE_PRIORITY[start:]:
        sk = build_skeleton(nums, target, shape, rng)
        if sk is not None:
            return shape, sk
    return None


# ---------------------------------------------------------------------------
# Dry-run stub narrator: deterministic, used for orchestrator testing.
# ---------------------------------------------------------------------------
async def _stub_narrate(sk: Skeleton) -> str:
    """Emit a minimal compliant completion. Used by --dry_run.

    Mirrors the production format: <state_K> markers carry the available slots as
    expressions-over-originals; one <think> block per move holds the arithmetic
    (operands are expressions, with the value substitution shown). The backtrack
    reason is its own <think> just before the <backtrack_K> token.
    """
    annos = annotate_expressions(sk)
    lines: list[str] = []
    for ev, anno in zip(sk.events, annos):
        if isinstance(ev, StateEvent):
            lines.append(f"<state_{ev.state_id}> {state_list(anno.available)}")
        elif isinstance(ev, ArithStep):
            lines.append(f"<think>{arith_fact(ev, anno)}.</think>")
        elif isinstance(ev, BacktrackEvent):
            lines.append(f"<think>{ev.reason}</think>")
            lines.append(f"<backtrack_{ev.target_state_id}>")
        elif isinstance(ev, ResumeEvent):
            lines.append(
                f"<think>Resuming from numbers {state_list(anno.available)}. Trying another path.</think>"
            )
        elif isinstance(ev, AnswerEvent):
            lines.append(f"<answer> {ev.equation_str} </answer>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
async def _process_one(
    prompt: str,
    nums: list[int],
    target: int,
    requested_shape: Shape,
    rng: random.Random,
    *,
    dry_run: bool,
    model: str,
    sem: asyncio.Semaphore,
) -> Optional[dict]:
    built = _try_build(nums, target, requested_shape, rng)
    if built is None:
        return None
    final_shape, sk = built
    async with sem:
        if dry_run:
            completion = await _stub_narrate(sk)
            try:
                validate(completion, sk)
            except ValidationError as e:
                print(f"  [skip] stub narration failed validation: {e}")
                return None
        else:
            completion = await narrate_async(sk, model=model, max_attempts=3)
            if completion is None:
                return None
    return {
        "query": prompt,
        "completion": completion,
        "shape": final_shape,
        "target": target,
        "nums": nums,
    }


async def _run(
    dataset_name: str,
    output_dir: str,
    num_examples: int,
    test_size: int,
    soft_targets: dict[Shape, int],
    max_source_rows: int,
    concurrency: int,
    model: str,
    dry_run: bool,
    seed: int,
) -> None:
    from datasets import Dataset, load_dataset

    print(f"Loading dataset {dataset_name} (split=train)...")
    ds = load_dataset(dataset_name, split="train")
    print(f"Loaded {len(ds)} source rows.")

    output_path = Path(output_dir).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # JSONL journal sits next to the output dir so save_to_disk won't clobber it.
    jsonl_path = output_path.parent / f"{output_path.name}.rows.jsonl"

    # Resume: load previously-accepted rows from the journal.
    rows: list[dict] = []
    seen: set[str] = set()
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                rows.append(row)
                seen.add(row["query"])
        print(f"Resumed {len(rows)} rows from {jsonl_path}")

    accepted_per_shape: dict[str, int] = {}
    for r in rows:
        accepted_per_shape[r["shape"]] = accepted_per_shape.get(r["shape"], 0) + 1
    needed: dict[str, int] = {
        s: max(0, soft_targets.get(s, 0) - accepted_per_shape.get(s, 0))
        for s in SHAPE_PRIORITY
    }

    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    indices = indices[:max_source_rows]

    by_arity: dict[int, list[int]] = {3: [], 4: []}
    for i in indices:
        puzzle = _extract_puzzle(ds[i])
        if puzzle is None:
            continue
        prompt, nums, _ = puzzle
        if prompt in seen:
            continue
        if len(nums) in by_arity:
            by_arity[len(nums)].append(i)
    rng.shuffle(by_arity[3])
    rng.shuffle(by_arity[4])
    print(f"Source pool (after dedup): 4-num={len(by_arity[4])} 3-num={len(by_arity[3])}")
    print(f"Needed per shape: {needed}")

    def pull(shape: str) -> Optional[int]:
        if shape in FOUR_NUM_ONLY:
            return by_arity[4].pop() if by_arity[4] else None
        return by_arity[4].pop() if by_arity[4] else (by_arity[3].pop() if by_arity[3] else None)

    sem = asyncio.Semaphore(concurrency)
    inflight: dict[asyncio.Task, str] = {}

    def remaining(s: str) -> int:
        return needed[s] - sum(1 for x in inflight.values() if x == s)

    def next_shape() -> Optional[str]:
        for s in SHAPE_PRIORITY:
            if remaining(s) > 0 and (by_arity[4] or (s not in FOUR_NUM_ONLY and by_arity[3])):
                return s
        return None

    with open(jsonl_path, "a") as jsonl_f:
        while len(rows) < num_examples and (inflight or next_shape() is not None):
            while len(rows) + len(inflight) < num_examples and len(inflight) < concurrency * 2:
                s = next_shape()
                if s is None:
                    break
                idx = pull(s)
                if idx is None:
                    break
                puzzle = _extract_puzzle(ds[idx])
                if puzzle is None:
                    continue
                prompt, nums, target = puzzle
                task = asyncio.create_task(
                    _process_one(prompt, nums, target, s, rng,
                                 dry_run=dry_run, model=model, sem=sem)
                )
                inflight[task] = s
            if not inflight:
                break
            done, _ = await asyncio.wait(inflight.keys(), return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                requested = inflight.pop(t)
                try:
                    result = t.result()
                except Exception as e:
                    print(f"  [error] {e}")
                    continue
                if result is None:
                    continue  # quota NOT decremented; loop will draw a replacement
                rows.append(result)
                needed[requested] = max(0, needed[requested] - 1)
                jsonl_f.write(json.dumps(result) + "\n")
                jsonl_f.flush()
                if len(rows) % 50 == 0:
                    print(f"  accepted {len(rows)} / {num_examples}")
        for t in inflight:
            t.cancel()

    by_shape = {s: sum(1 for r in rows if r["shape"] == s) for s in SHAPE_PRIORITY}
    print(f"Done. Accepted {len(rows)} rows. Final shape mix: {by_shape}")
    print(f"Resume journal: {jsonl_path}")

    # Deterministic train/test split: sort by query, then shuffle with the seeded RNG
    # so the split is stable across re-runs even when async ordering differs.
    from datasets import DatasetDict
    sorted_rows = sorted(rows, key=lambda r: r["query"])
    split_rng = random.Random(seed)
    split_rng.shuffle(sorted_rows)
    test_rows = sorted_rows[:test_size]
    train_rows = sorted_rows[test_size:]
    dsdict = DatasetDict({
        "train": Dataset.from_list(train_rows),
        "test": Dataset.from_list(test_rows),
    })
    dsdict.save_to_disk(str(output_path))
    print(f"Saved DatasetDict to {output_path}  (train={len(train_rows)}, test={len(test_rows)})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="asingh15/countdown_tasks_3to4")
    parser.add_argument(
        "--output", type=str, default="dataset_generation/output/sels_sft_v3"
    )
    parser.add_argument("--num_examples", type=int, default=DEFAULT_NUM_EXAMPLES,
                        help="Total rows to generate (train + test). Default 1200 matches cog_behav.")
    parser.add_argument("--test_size", type=int, default=DEFAULT_TEST_SIZE,
                        help="Rows held out as the test split (drawn from num_examples). Default 200.")
    parser.add_argument("--max_source_rows", type=int, default=DEFAULT_MAX_SOURCE_ROWS)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--model", type=str, default="gpt-4.1-mini")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Use a deterministic stub narrator instead of calling the OpenAI API.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chain", type=int, default=DEFAULT_TARGETS["chain"])
    parser.add_argument("--bt_inner", type=int, default=DEFAULT_TARGETS["bt_inner"])
    parser.add_argument("--bt_deep", type=int, default=DEFAULT_TARGETS["bt_deep"])
    parser.add_argument("--bt_to_root", type=int, default=DEFAULT_TARGETS["bt_to_root"])
    parser.add_argument("--multi_bt", type=int, default=DEFAULT_TARGETS["multi_bt"])
    args = parser.parse_args()

    if not args.dry_run and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required unless --dry_run is set.")

    soft_targets: dict[Shape, int] = {
        "chain": args.chain,
        "bt_inner": args.bt_inner,
        "bt_deep": args.bt_deep,
        "bt_to_root": args.bt_to_root,
        "multi_bt": args.multi_bt,
    }

    if args.test_size >= args.num_examples:
        raise SystemExit(f"--test_size ({args.test_size}) must be < --num_examples ({args.num_examples})")

    asyncio.run(
        _run(
            dataset_name=args.dataset_name,
            output_dir=args.output,
            num_examples=args.num_examples,
            test_size=args.test_size,
            soft_targets=soft_targets,
            max_source_rows=args.max_source_rows,
            concurrency=args.concurrency,
            model=args.model,
            dry_run=args.dry_run,
            seed=args.seed,
        )
    )


if __name__ == "__main__":
    main()
