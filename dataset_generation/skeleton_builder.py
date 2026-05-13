"""Skeleton synthesizer for SELS SFT traces.

Produces a structured `Skeleton` per puzzle that the LLM narrator will then
turn into prose. The skeleton itself fully determines:
- which special tokens appear and in what order,
- the arithmetic facts the narrator must paraphrase faithfully,
- the dead-end justifications,
- the final answer equation string.

The narrator only invents the natural-language wrapping inside `<think>...</think>`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal, Optional

from dataset_generation.countdown_solver import (
    Op,
    all_next_states,
    apply_op,
    dead_end_reason,
    find_dead_end_branches,
    find_solution_path,
    find_solvable_branches,
    is_solvable,
)

Shape = Literal["chain", "bt_inner", "bt_deep", "bt_to_root", "multi_bt"]

MAX_STATE_ID = 4  # bounded vocab from plan; depth max actually used is 2


@dataclass
class StateEvent:
    state_id: int
    numbers: tuple[int, ...]


@dataclass
class ArithStep:
    a: int
    b: int
    op: str
    result: int
    remaining_before: tuple[int, ...]
    remaining_after: tuple[int, ...]


@dataclass
class BacktrackEvent:
    target_state_id: int
    reason: str


@dataclass
class ResumeEvent:
    target_state_id: int
    numbers: tuple[int, ...]


@dataclass
class AnswerEvent:
    equation_str: str
    final_value: int


Event = StateEvent | ArithStep | BacktrackEvent | ResumeEvent | AnswerEvent


@dataclass
class Skeleton:
    numbers: tuple[int, ...]
    target: int
    shape: Shape
    events: list[Event] = field(default_factory=list)

    def __iter__(self):
        return iter(self.events)


# ---------------------------------------------------------------------------
# Expression tracking — used to build the final parenthesized equation.
# ---------------------------------------------------------------------------
class _ExprTracker:
    """Tracks (value, expression-string) pairs as ops are applied.

    Each ArithStep replaces two entries with one combined entry. After all
    solution-path ops are applied exactly one entry remains, whose expression
    is the final answer.
    """

    def __init__(self, numbers: tuple[int, ...]):
        self._entries: list[tuple[int, str]] = [(n, str(n)) for n in numbers]

    def values(self) -> tuple[int, ...]:
        return tuple(sorted(v for v, _ in self._entries))

    def apply(self, a: int, b: int, op: str) -> int:
        # Find first matching entry for a, remove it.
        ia = next(i for i, (v, _) in enumerate(self._entries) if v == a)
        a_expr = self._entries.pop(ia)[1]
        # Find first matching for b in the remaining entries.
        ib = next(i for i, (v, _) in enumerate(self._entries) if v == b)
        b_expr = self._entries.pop(ib)[1]
        r = apply_op(a, b, op)
        if r is None:
            raise ValueError(f"illegal op replay {a} {op} {b}")
        new_expr = f"({a_expr} {op} {b_expr})"
        self._entries.append((r, new_expr))
        return r

    def final_expression(self) -> tuple[int, str]:
        if len(self._entries) != 1:
            raise ValueError(f"expected single residual entry, got {self._entries}")
        return self._entries[0]


# ---------------------------------------------------------------------------
# Shape construction
# ---------------------------------------------------------------------------
def _apply_path(state: tuple[int, ...], path: list[Op]) -> tuple[tuple[int, ...], list[ArithStep]]:
    """Replay a list of ops on `state`, emitting an ArithStep per op."""
    cur = list(state)
    out: list[ArithStep] = []
    for a, b, op in path:
        before = tuple(sorted(cur))
        cur.remove(a)
        cur.remove(b)
        r = apply_op(a, b, op)
        if r is None:
            raise ValueError(f"illegal op in path: {a} {op} {b}")
        cur.append(r)
        after = tuple(sorted(cur))
        out.append(
            ArithStep(a=a, b=b, op=op, result=r, remaining_before=before, remaining_after=after)
        )
    return tuple(sorted(cur)), out


def _state_after_path(state: tuple[int, ...], path: list[Op]) -> tuple[int, ...]:
    final, _ = _apply_path(state, path)
    return final


def build_skeleton(
    numbers: list[int],
    target: int,
    shape: Shape,
    rng: random.Random,
) -> Optional[Skeleton]:
    """Build a Skeleton for the given puzzle and shape.

    Returns None if the requested shape is infeasible for this puzzle (e.g.,
    not enough numbers, no dead-end branches available). Caller is responsible
    for fallback.
    """
    nums = tuple(sorted(numbers))
    if not is_solvable(nums, target):
        return None  # puzzle has no solution at all — skip entirely

    sol_path = find_solution_path(nums, target)
    if sol_path is None:
        return None

    if shape == "chain":
        return _build_chain(nums, target, sol_path)
    if shape == "bt_inner":
        return _build_bt_inner(nums, target, sol_path, rng)
    if shape == "bt_deep":
        return _build_bt_deep(nums, target, sol_path, rng)
    if shape == "bt_to_root":
        return _build_bt_to_root(nums, target, sol_path, rng)
    if shape == "multi_bt":
        return _build_multi_bt(nums, target, sol_path, rng)
    raise ValueError(f"unknown shape {shape!r}")


def _solution_to_arith_and_answer(
    start_nums: tuple[int, ...], sol_path: list[Op], target: int
) -> tuple[list[ArithStep], AnswerEvent]:
    _, steps = _apply_path(start_nums, sol_path)
    tracker = _ExprTracker(start_nums)
    for s in steps:
        tracker.apply(s.a, s.b, s.op)
    final_val, final_expr = tracker.final_expression()
    if final_val != target:
        raise ValueError(f"solution path ends at {final_val}, expected {target}")
    return steps, AnswerEvent(equation_str=final_expr, final_value=final_val)


def _interleave_states(
    arith_steps: list[ArithStep],
    start_state_id: int = 0,
    emit_initial_state: bool = True,
) -> tuple[list[Event], int]:
    """Insert a StateEvent before each ArithStep.

    The state at depth K is named state_K. With emit_initial_state=True we
    emit a state event before the first step (state_{start_state_id} naming
    the initial numbers). With emit_initial_state=False we skip that first
    state (useful when a ResumeEvent already named the same checkpoint just
    before this call).

    Returns (events, next_state_id) so callers can continue numbering.
    """
    events: list[Event] = []
    sid = start_state_id
    for i, step in enumerate(arith_steps):
        if not (i == 0 and not emit_initial_state):
            events.append(StateEvent(sid, step.remaining_before))
            sid += 1
        events.append(step)
    return events, sid


def _pick_dead_branch(
    state: tuple[int, ...], target: int, rng: random.Random, depth: int = 1
) -> Optional[list[Op]]:
    """Pick one dead-end branch of the requested depth, or shorter if unavailable."""
    for d in (depth, max(1, depth - 1)):
        branches = find_dead_end_branches(state, target, depth=d, max_branches=8)
        if branches:
            return rng.choice(branches)
    return None


def _build_chain(nums: tuple[int, ...], target: int, sol_path: list[Op]) -> Skeleton:
    """state_0 → arith → state_1 → arith → (state_2 → arith) → answer.

    Checkpoints at every depth; no backtracks. Demonstrates state progression.
    """
    steps, answer = _solution_to_arith_and_answer(nums, sol_path, target)
    events, _ = _interleave_states(steps, start_state_id=0, emit_initial_state=True)
    events.append(answer)
    return Skeleton(numbers=nums, target=target, shape="chain", events=events)


def _build_bt_inner(
    nums: tuple[int, ...], target: int, sol_path: list[Op], rng: random.Random
) -> Optional[Skeleton]:
    """state_0 → good_step → state_1 → dead → BT_1 → resume_1 → rest_of_solution.

    Backtrack to the nearest non-root checkpoint. Requires >=2 arith steps.
    """
    if len(sol_path) < 2:
        return None
    state_1_nums = _state_after_path(nums, [sol_path[0]])
    dead = _pick_dead_branch(state_1_nums, target, rng, depth=1)
    if dead is None:
        return None
    _, dead_steps = _apply_path(state_1_nums, dead)
    dead_state = dead_steps[-1].remaining_after

    sol_steps, answer = _solution_to_arith_and_answer(nums, sol_path, target)
    rest_steps = sol_steps[1:]
    rest_events, _ = _interleave_states(rest_steps, start_state_id=2, emit_initial_state=False)

    events: list[Event] = [
        StateEvent(0, nums),
        sol_steps[0],
        StateEvent(1, state_1_nums),
        *dead_steps,
        BacktrackEvent(target_state_id=1, reason=dead_end_reason(dead_state, target)),
        ResumeEvent(target_state_id=1, numbers=state_1_nums),
        *rest_events,
        answer,
    ]
    return Skeleton(numbers=nums, target=target, shape="bt_inner", events=events)


def _build_bt_deep(
    nums: tuple[int, ...], target: int, sol_path: list[Op], rng: random.Random
) -> Optional[Skeleton]:
    """state_0 → step1 → state_1 → step2 → state_2 → dead → BT_2 → resume_2 → final.

    Backtrack at the deepest non-root checkpoint. Requires >=3 arith steps (4-num).
    """
    if len(sol_path) < 3:
        return None
    sol_steps, answer = _solution_to_arith_and_answer(nums, sol_path, target)
    state_1_nums = sol_steps[1].remaining_before
    state_2_nums = sol_steps[2].remaining_before
    dead = _pick_dead_branch(state_2_nums, target, rng, depth=1)
    if dead is None:
        return None
    _, dead_steps = _apply_path(state_2_nums, dead)
    dead_state = dead_steps[-1].remaining_after

    events: list[Event] = [
        StateEvent(0, nums),
        sol_steps[0],
        StateEvent(1, state_1_nums),
        sol_steps[1],
        StateEvent(2, state_2_nums),
        *dead_steps,
        BacktrackEvent(target_state_id=2, reason=dead_end_reason(dead_state, target)),
        ResumeEvent(target_state_id=2, numbers=state_2_nums),
        sol_steps[2],
        answer,
    ]
    return Skeleton(numbers=nums, target=target, shape="bt_deep", events=events)


def _build_bt_to_root(
    nums: tuple[int, ...], target: int, sol_path: list[Op], rng: random.Random
) -> Optional[Skeleton]:
    """state_0 → bad_step → state_1(unsolvable) → BT_0 → resume_0 → solution.

    Backtrack from a non-root state all the way to root, then take a fresh
    solution path. state_1 is REUSED after the BT (different content the
    second time). Requires the solver to find a dead-end first op that
    differs from sol_path[0].
    """
    if len(sol_path) < 2:
        return None
    bad_branches = find_dead_end_branches(nums, target, depth=1, max_branches=8)
    # Filter out branches whose first op matches the solution path's first op.
    bad_branches = [b for b in bad_branches if b and b[0] != sol_path[0]]
    if not bad_branches:
        return None
    bad_branch = rng.choice(bad_branches)
    _, bad_steps = _apply_path(nums, bad_branch)
    bad_state_1 = bad_steps[-1].remaining_after

    sol_steps, answer = _solution_to_arith_and_answer(nums, sol_path, target)
    sol_events, _ = _interleave_states(sol_steps, start_state_id=1, emit_initial_state=False)

    events: list[Event] = [
        StateEvent(0, nums),
        *bad_steps,
        StateEvent(1, bad_state_1),
        BacktrackEvent(target_state_id=0, reason=dead_end_reason(bad_state_1, target)),
        ResumeEvent(target_state_id=0, numbers=nums),
        *sol_events,
        answer,
    ]
    return Skeleton(numbers=nums, target=target, shape="bt_to_root", events=events)


def _build_multi_bt(
    nums: tuple[int, ...], target: int, sol_path: list[Op], rng: random.Random
) -> Optional[Skeleton]:
    """BT_1 then BT_2 — two backtracks at different depths. Requires >=3 arith steps."""
    if len(sol_path) < 3:
        return None
    sol_steps, answer = _solution_to_arith_and_answer(nums, sol_path, target)
    state_1_nums = sol_steps[1].remaining_before
    state_2_nums = sol_steps[2].remaining_before

    dead1 = _pick_dead_branch(state_1_nums, target, rng, depth=1)
    if dead1 is None:
        return None
    _, dead1_steps = _apply_path(state_1_nums, dead1)
    dead1_state = dead1_steps[-1].remaining_after

    dead2 = _pick_dead_branch(state_2_nums, target, rng, depth=1)
    if dead2 is None:
        return None
    _, dead2_steps = _apply_path(state_2_nums, dead2)
    dead2_state = dead2_steps[-1].remaining_after

    events: list[Event] = [
        StateEvent(0, nums),
        sol_steps[0],
        StateEvent(1, state_1_nums),
        *dead1_steps,
        BacktrackEvent(target_state_id=1, reason=dead_end_reason(dead1_state, target)),
        ResumeEvent(target_state_id=1, numbers=state_1_nums),
        sol_steps[1],
        StateEvent(2, state_2_nums),
        *dead2_steps,
        BacktrackEvent(target_state_id=2, reason=dead_end_reason(dead2_state, target)),
        ResumeEvent(target_state_id=2, numbers=state_2_nums),
        sol_steps[2],
        answer,
    ]
    return Skeleton(numbers=nums, target=target, shape="multi_bt", events=events)


# ---------------------------------------------------------------------------
# Spec rendering for the narrator
# ---------------------------------------------------------------------------
def skeleton_to_spec(sk: Skeleton) -> str:
    """Serialize a Skeleton into a text spec the narrator will rewrite.

    The spec uses explicit event labels and lists every special token, every
    arithmetic step, and every dead-end reason verbatim. The narrator's job
    is to interleave `<think>...</think>` prose between these fixed pieces
    without altering any of them.
    """
    lines: list[str] = []
    lines.append(f"PUZZLE: numbers={list(sk.numbers)}, target={sk.target}, shape={sk.shape}")
    lines.append("")
    lines.append("EVENTS (in order — render the completion respecting this order):")
    for ev in sk.events:
        if isinstance(ev, StateEvent):
            lines.append(f"- STATE<state_{ev.state_id}> numbers={list(ev.numbers)}")
        elif isinstance(ev, ArithStep):
            lines.append(
                f"- ARITH {ev.a} {ev.op} {ev.b} = {ev.result}  "
                f"(remaining before={list(ev.remaining_before)}, after={list(ev.remaining_after)})"
            )
        elif isinstance(ev, BacktrackEvent):
            lines.append(f"- BACKTRACK<backtrack_{ev.target_state_id}> reason: {ev.reason}")
        elif isinstance(ev, ResumeEvent):
            lines.append(
                f"- RESUME (target state_{ev.target_state_id}) "
                f'must open with exactly: "Resuming from numbers {list(ev.numbers)}. "'
            )
        elif isinstance(ev, AnswerEvent):
            lines.append(f"- ANSWER<answer>{ev.equation_str}</answer>")
        else:
            raise ValueError(f"unknown event {ev!r}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Selftest
# ---------------------------------------------------------------------------
def _selftest() -> None:
    from evaluation.countdown import evaluate_equation, validate_equation
    rng = random.Random(0)
    nums3 = (44, 19, 35); t3 = 98
    nums4 = (3, 4, 6, 8); t4 = 24

    def assert_valid(sk: Skeleton, expected_shape: Shape, source_nums: tuple) -> None:
        assert sk is not None and sk.shape == expected_shape
        answer = sk.events[-1]
        assert isinstance(answer, AnswerEvent)
        assert validate_equation(answer.equation_str, list(source_nums))
        assert abs(evaluate_equation(answer.equation_str) - sk.target) < 1e-5
        # No state_id should exceed MAX_STATE_ID.
        for e in sk.events:
            if isinstance(e, StateEvent):
                assert e.state_id <= MAX_STATE_ID
            if isinstance(e, BacktrackEvent):
                assert e.target_state_id <= MAX_STATE_ID

    # 1. chain (3-num and 4-num)
    sk = build_skeleton(list(nums3), t3, "chain", rng); assert_valid(sk, "chain", nums3)
    sids = [e.state_id for e in sk.events if isinstance(e, StateEvent)]
    assert sids == [0, 1], f"3-num chain should have state_0 and state_1 only, got {sids}"
    sk = build_skeleton(list(nums4), t4, "chain", rng); assert_valid(sk, "chain", nums4)
    sids = [e.state_id for e in sk.events if isinstance(e, StateEvent)]
    assert sids == [0, 1, 2], f"4-num chain should have state_0/1/2, got {sids}"

    # 2. bt_inner (BT_1 to nearest non-root)
    sk = build_skeleton(list(nums3), t3, "bt_inner", rng); assert_valid(sk, "bt_inner", nums3)
    bts = [e.target_state_id for e in sk.events if isinstance(e, BacktrackEvent)]
    assert bts == [1], f"bt_inner should BT to state_1 once, got {bts}"
    sk = build_skeleton(list(nums4), t4, "bt_inner", rng); assert_valid(sk, "bt_inner", nums4)
    bts = [e.target_state_id for e in sk.events if isinstance(e, BacktrackEvent)]
    assert bts == [1]

    # 3. bt_deep (BT_2, 4-num only)
    sk = build_skeleton(list(nums4), t4, "bt_deep", rng)
    if sk is not None:
        assert_valid(sk, "bt_deep", nums4)
        bts = [e.target_state_id for e in sk.events if isinstance(e, BacktrackEvent)]
        assert bts == [2], f"bt_deep should BT to state_2, got {bts}"
        sids = sorted({e.state_id for e in sk.events if isinstance(e, StateEvent)})
        assert sids == [0, 1, 2]
    # 3-num bt_deep is infeasible
    assert build_skeleton(list(nums3), t3, "bt_deep", rng) is None

    # 4. bt_to_root — state_1 should appear TWICE (once unsolvable, once reused after BT)
    sk = build_skeleton(list(nums3), t3, "bt_to_root", rng)
    if sk is not None:
        assert_valid(sk, "bt_to_root", nums3)
        s1_count = sum(1 for e in sk.events if isinstance(e, StateEvent) and e.state_id == 1)
        assert s1_count == 2, f"bt_to_root should have state_1 twice (bad + reused), got {s1_count}"
        bts = [e.target_state_id for e in sk.events if isinstance(e, BacktrackEvent)]
        assert bts == [0]

    # 5. multi_bt (BT_1 then BT_2, 4-num only)
    sk = build_skeleton(list(nums4), t4, "multi_bt", rng)
    if sk is not None:
        assert_valid(sk, "multi_bt", nums4)
        bts = [e.target_state_id for e in sk.events if isinstance(e, BacktrackEvent)]
        assert bts == [1, 2], f"multi_bt should BT to 1 then 2, got {bts}"
    assert build_skeleton(list(nums3), t3, "multi_bt", rng) is None  # infeasible

    # 6. Spec rendering doesn't crash.
    sk = build_skeleton(list(nums4), t4, "multi_bt", rng)
    if sk is not None:
        spec = skeleton_to_spec(sk)
        assert "<state_0>" in spec and "<state_1>" in spec and "<state_2>" in spec
        assert "<backtrack_1>" in spec and "<backtrack_2>" in spec
        assert "Resuming from numbers" in spec

    print("skeleton_builder selftest OK")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    _selftest()
