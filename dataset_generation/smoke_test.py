"""Self-contained orchestrator smoke test.

Bypasses HF datasets so this runs in a stripped-down Python environment
without `datasets`/`openai` installed. Constructs ~20 inline puzzles and
exercises the build_skeleton -> stub_narrate -> validate pipeline end-to-end.

Run with: `python -m dataset_generation.smoke_test`
"""

from __future__ import annotations

import asyncio
import random
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset_generation.generate_sft_data import _stub_narrate, _try_build  # noqa: E402
from dataset_generation.narrate_with_gpt import validate, ValidationError  # noqa: E402

# Hand-picked solvable Countdown puzzles. Each is (nums, target).
PUZZLES: list[tuple[list[int], int]] = [
    ([3, 4, 6, 8], 24),
    ([44, 19, 35], 98),
    ([2, 3, 5, 7], 47),
    ([5, 5, 5, 5], 30),
    ([1, 4, 7, 9], 28),
    ([6, 7, 8, 9], 42),
    ([2, 3, 4, 5], 19),
    ([3, 5, 7], 35),
    ([10, 25, 50, 75], 100),
    ([1, 2, 3, 4], 24),
    ([6, 10, 12], 8),
    ([8, 8, 8], 64),
    ([3, 6, 9, 12], 24),
    ([10, 20, 30, 40], 50),
    ([4, 5, 6, 7], 13),
    ([2, 4, 8, 16], 32),
    ([100, 50, 25], 75),
    ([1, 5, 10, 20], 25),
    ([3, 3, 3, 3], 12),
    ([7, 11, 13], 31),
]


async def main() -> None:
    rng = random.Random(42)
    requested_shapes = ["multi_bt", "bt_deep", "bt_to_root", "bt_inner", "chain"]

    accepted = 0
    by_shape: dict[str, int] = {}
    fallback_count = 0

    for i, (nums, target) in enumerate(PUZZLES):
        requested = requested_shapes[i % len(requested_shapes)]
        built = _try_build(nums, target, requested, rng)
        if built is None:
            print(f"  [{i}] nums={nums} target={target} requested={requested} -> all shapes infeasible")
            continue
        final_shape, sk = built
        if final_shape != requested:
            fallback_count += 1
        completion = await _stub_narrate(sk)
        try:
            validate(completion, sk)
        except ValidationError as e:
            print(f"  [{i}] FAIL validation: {e}")
            continue
        accepted += 1
        by_shape[final_shape] = by_shape.get(final_shape, 0) + 1
        print(
            f"  [{i}] nums={nums} target={target} requested={requested} "
            f"-> final={final_shape}  events={len(sk.events)}  completion_chars={len(completion)}"
        )

    print()
    print(f"Accepted: {accepted}/{len(PUZZLES)}")
    print(f"Shape mix: {by_shape}")
    print(f"Shape fallbacks (requested != final): {fallback_count}")

    # Sample one full completion
    nums, target = PUZZLES[1]  # the canonical 98 example
    sk = _try_build(nums, target, "bt_inner", rng)[1]
    completion = await _stub_narrate(sk)
    print()
    print("=" * 60)
    print("Sample full completion (stub narration):")
    print("=" * 60)
    print(completion)
    print("=" * 60)

    assert accepted >= 15, f"too few accepted ({accepted}); pipeline may be broken"
    print()
    print("orchestrator smoke_test OK")


if __name__ == "__main__":
    asyncio.run(main())
