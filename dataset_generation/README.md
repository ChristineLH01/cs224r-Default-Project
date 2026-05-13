# SELS Synthetic SFT Data Generation

Produces ~1.5K synthetic Countdown traces with `<state_K>` / `<backtrack_K>` /
`Resuming from numbers [...]` structure, ready to feed into the existing SFT
trainer for the SELS extension.

## Pipeline

```
asingh15/countdown_tasks_3to4
        │
        ▼  (prompt, nums, target)
  countdown_solver.py  ────► is_solvable / find_solution_path / dead_end_reason
        │
        ▼  Skeleton (typed events)
  skeleton_builder.py  ────► build_skeleton(nums, target, shape, rng)
        │
        ▼  text spec
  narrate_with_gpt.py  ────► GPT-4o-mini wraps each step in <think>...</think>
        │                    + 6-check validator + retry
        ▼  completion str
  generate_sft_data.py ────► orchestrate, fan-out async calls, save HF Dataset
        │
        ▼
  dataset_generation/output/sels_sft_v1/  (HF Dataset on disk)
```

## Setup

The orchestrator depends on `datasets` and `openai` from the project's
top-level `pyproject.toml`. From the repo root:

```bash
pip install -e ".[all]"
export OPENAI_API_KEY=sk-...
```

For local pipeline checks without API calls, see `Selftests` below.

## Run

Full production run (1200 rows = 1000 train + 200 test, matches the default
`Asap7772/cog_behav_all_strategies` size exactly; costs ~$1–2 of GPT-4o-mini):

```bash
python dataset_generation/generate_sft_data.py \
    --output dataset_generation/output/sels_sft_v1
```

(Defaults: `--num_examples 1200 --test_size 200`.)

Useful flags:

- `--dry_run` — use a deterministic stub narrator instead of calling OpenAI.
  Faster, free, exercises orchestration logic.
- `--num_examples 1200` — total rows (train + test). Default 1200.
- `--test_size 200` — rows held out as the test split for validation-loss
  tracking during SFT training. Default 200.
- `--concurrency 8` — fewer in-flight requests if rate-limited.
- `--chain N --bt_inner N --bt_deep N --bt_to_root N --multi_bt N` —
  override per-shape soft targets. Defaults: 240 / 360 / 240 / 180 / 180
  (sum to 1200; 20/30/20/15/15). Every shape exercises multi-state
  checkpointing (`<state_0>`, `<state_1>`, often `<state_2>`); backtrack
  destinations are spread across state_0/1/2.
- `--max_source_rows 4000` — oversampling headroom for shape-fallback.
- `--seed 0` — random seed for reproducibility (controls both source sampling
  and the train/test split).

## Output schema

The orchestrator writes a HuggingFace `DatasetDict.save_to_disk(...)` directory
with two splits (`train` and `test`), each with columns:

| Column       | Type         | Notes |
|--------------|--------------|-------|
| `query`      | `str`        | Prompt verbatim from the source dataset. |
| `completion` | `str`        | Narrated trace with all special tokens.  |
| `shape`      | `str`        | One of `chain` / `bt_inner` / `bt_deep` / `bt_to_root` / `multi_bt`. Debug. |
| `target`     | `int`        | Countdown target. Debug. |
| `nums`       | `list[int]`  | Countdown numbers. Debug. |

Load downstream:

```python
from datasets import load_from_disk
ds = load_from_disk("dataset_generation/output/sels_sft_v1")
print(ds)
# DatasetDict({train: Dataset(1000), test: Dataset(200)})
```

Train and test prompts are disjoint by construction (each source prompt
appears in exactly one split). The test split serves the same role as the
`Asap7772/cog_behav_all_strategies` test split: validation-loss tracking
during SFT training and best-checkpoint selection. It is NOT the final
task-performance evaluator — that's `evaluation/countdown_eval.py` against
`asingh15/countdown_tasks_3to4` test split (50 problems).

## Shape design — what every trace looks like

Every trace uses multiple state checkpoints. State IDs represent depth from
root in the current search path: `<state_0>` is the initial numbers,
`<state_1>` is the numbers after one arithmetic step, `<state_2>` after two.
After `<backtrack_K>`, all state IDs deeper than K are invalidated and may be
reused as the model re-explores.

| Shape | Pattern | Arity | Share |
|---|---|---|---|
| `chain` | `state_0 → arith → state_1 → arith → (state_2 → arith) → answer` (no backtracks) | 3 or 4 | 20% |
| `bt_inner` | `state_0 → arith → state_1 → dead → BT_1 → resume_1 → solution → answer` | 3 or 4 | 30% |
| `bt_deep` | `state_0 → arith → state_1 → arith → state_2 → dead → BT_2 → resume_2 → solution → answer` | 4 only | 20% |
| `bt_to_root` | `state_0 → bad → state_1(unsolvable) → BT_0 → resume_0 → solution (state_1 reused with new content)` | 3 or 4 | 15% |
| `multi_bt` | `state_0 → arith → state_1 → dead → BT_1 → resume_1 → arith → state_2 → dead → BT_2 → resume_2 → answer` | 4 only | 15% |

This distribution ensures every special token (`<state_0..2>`,
`<backtrack_0..2>`) gets substantial training signal across the 1200-row
dataset, supporting the proposal's claim of structured state checkpointing
with verifier-grounded backtracks.

## Connecting to SFT training

`sft_trainer/sft_dataset.py` already expects `query` / `completion` columns, so
it consumes the saved dataset directly. Point the trainer at the local path:

```bash
python sft_trainer/sft.py \
    --dataset_name $(pwd)/dataset_generation/output/sels_sft_v1 \
    --model_name asingh15/qwen-sft-countdown-defaultproj \
    --output_dir checkpoints/sels_sft \
    ...
```

**Important — special-token registration.** Before SFT training on this data,
the new tokens must be registered with the tokenizer so they don't fragment.
This is a one-line patch in the trainer setup, listed under "Out of scope"
below since it lives in `sft_trainer/`, not here:

```python
tokenizer.add_special_tokens({"additional_special_tokens": [
    "<state_0>", "<state_1>", "<state_2>", "<state_3>", "<state_4>",
    "<backtrack_0>", "<backtrack_1>", "<backtrack_2>", "<backtrack_3>", "<backtrack_4>",
]})
model.resize_token_embeddings(len(tokenizer))
```

## Selftests (local, no external deps)

Each module has a self-contained selftest you can run without `datasets` or
`openai` installed:

```bash
python -m dataset_generation.countdown_solver
python -m dataset_generation.skeleton_builder
python -m dataset_generation.narrate_with_gpt   # validator only, no API calls
python -m dataset_generation.smoke_test         # end-to-end with stub narrator
```

The smoke test runs the full pipeline (build skeleton → stub narrate →
validate) on 20 inline puzzles and reports the shape mix.

## Out of scope for this module

Documented in
[/Users/ericxia/.claude/plans/can-you-take-curried-pizza.md](../../../.claude/plans/can-you-take-curried-pizza.md):

- Special-token registration in the SFT trainer (one-line patch in
  `sft_trainer/sft.py`).
- Context manager runtime behavior (V1 = no-op + verbal restore; optional V2
  = KV-cache truncation + runtime anchor injection). Same SFT data works
  under either runtime.
- PRM scoring at backtrack events — Christine's component.
- RLOO objective modification to incorporate PRM rewards.

## Token scheme

10 fixed special tokens — N is bounded to keep the trigger atomic for any
future context-manager implementation:

```
<state_0> <state_1> <state_2> <state_3> <state_4>
<backtrack_0> <backtrack_1> <backtrack_2> <backtrack_3> <backtrack_4>
```

State IDs reset to 0 at the start of every prompt — not globally unique
across the dataset.

## Re-anchor template

The string after every `<backtrack_K>` token follows the exact template
`Resuming from numbers [N1, N2, ...]. ` — same comma-space format as
`list(...)` produces. The validator enforces this so the V2 runtime can later
inject the identical string deterministically after a truncation event.
