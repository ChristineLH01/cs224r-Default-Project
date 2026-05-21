"""Exhaustive Countdown solver with strict integer arithmetic.

Used by skeleton_builder to label intermediate states as solvable / dead-end,
and (later) by the PRM at RL time. Stateless and memoized.

Countdown rules enforced here:
- Each input number is used exactly once.
- Subtraction must yield a positive result (no zero or negatives).
- Division must yield a whole integer (no fractions).
- Operations are +, -, *, /.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

Op = tuple[int, int, str]  # (a, b, op_symbol) — a and b are the actual values combined


def apply_op(a: int, b: int, op: str) -> Optional[int]:
    """Apply op to (a, b). Returns None if the move is illegal under Countdown rules."""
    if op == "+":
        return a + b
    if op == "-":
        r = a - b
        return r if r > 0 else None
    if op == "*":
        return a * b
    if op == "/":
        if b == 0 or a % b != 0:
            return None
        return a // b
    raise ValueError(f"unknown op {op!r}")


def _canonical(numbers) -> tuple[int, ...]:
    return tuple(sorted(numbers))


def all_next_states(numbers: tuple[int, ...]) -> list[tuple[Op, tuple[int, ...]]]:
    """Return list of (op, new_multiset) reachable in one binary step.

    Commutativity pruning: for + and *, only emit ordered pairs with a >= b.
    For - and /, emit both orderings (non-commutative).
    """
    out: list[tuple[Op, tuple[int, ...]]] = []
    n = len(numbers)
    seen: set[tuple[Op, tuple[int, ...]]] = set()
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a, b = numbers[i], numbers[j]
            rest = tuple(numbers[k] for k in range(n) if k != i and k != j)
            for op in ("+", "-", "*", "/"):
                if op in ("+", "*") and a < b:
                    continue  # canonical ordering for commutative ops
                r = apply_op(a, b, op)
                if r is None:
                    continue
                new_state = _canonical(rest + (r,))
                entry = ((a, b, op), new_state)
                if entry in seen:
                    continue
                seen.add(entry)
                out.append(entry)
    return out


@lru_cache(maxsize=None)
def is_solvable(numbers: tuple[int, ...], target: int) -> bool:
    """True iff some sequence of ops on `numbers` reaches `target` exactly."""
    numbers = _canonical(numbers)
    if len(numbers) == 1:
        return numbers[0] == target
    for _, new_state in all_next_states(numbers):
        if is_solvable(new_state, target):
            return True
    return False


def find_solution_path(numbers: tuple[int, ...], target: int) -> Optional[list[Op]]:
    """Return one valid op sequence reaching target, or None.

    Ops are returned as (a, b, op_symbol) with the actual values combined at
    each step — callers can render them directly into an equation string.
    """
    numbers = _canonical(numbers)
    if len(numbers) == 1:
        return [] if numbers[0] == target else None
    for op, new_state in all_next_states(numbers):
        if is_solvable(new_state, target):
            tail = find_solution_path(new_state, target)
            if tail is not None:
                return [op, *tail]
    return None


def find_dead_end_branches(
    numbers: tuple[int, ...], target: int, depth: int = 1, max_branches: int = 8
) -> list[list[Op]]:
    """Return op sequences of length `depth` ending in an unsolvable state.

    Useful for picking which wrong branch to take in a single_bt / two_bt
    skeleton. `max_branches` caps the result for sampling efficiency.
    """
    numbers = _canonical(numbers)
    if depth <= 0:
        return [[]] if not is_solvable(numbers, target) else []
    out: list[list[Op]] = []
    for op, new_state in all_next_states(numbers):
        if depth == 1:
            if not is_solvable(new_state, target):
                out.append([op])
        else:
            for tail in find_dead_end_branches(new_state, target, depth - 1, max_branches):
                out.append([op, *tail])
                if len(out) >= max_branches:
                    return out
        if len(out) >= max_branches:
            return out
    return out


def find_solvable_branches(
    numbers: tuple[int, ...], target: int, depth: int = 1, max_branches: int = 8
) -> list[list[Op]]:
    """Return op sequences of length `depth` ending in a *solvable* state.

    Used to construct `deep`-shape skeletons: take a solvable first step,
    create state_1 at the resulting multiset, then dead-end from there.
    """
    numbers = _canonical(numbers)
    if depth <= 0:
        return [[]] if is_solvable(numbers, target) else []
    out: list[list[Op]] = []
    for op, new_state in all_next_states(numbers):
        if not is_solvable(new_state, target):
            continue
        if depth == 1:
            out.append([op])
        else:
            for tail in find_solvable_branches(new_state, target, depth - 1, max_branches):
                out.append([op, *tail])
                if len(out) >= max_branches:
                    return out
        if len(out) >= max_branches:
            return out
    return out


def dead_end_reason(numbers: tuple[int, ...], target: int) -> str:
    """Concise programmatic justification for why `numbers` cannot reach `target`.

    Used to seed BacktrackEvent.reason so the narrator does not invent
    arithmetic. Caller should only invoke when is_solvable(numbers, target)
    is False.
    """
    numbers = _canonical(numbers)
    if len(numbers) == 1:
        return (
            f"remaining=[{numbers[0]}], target={target} — single value {numbers[0]} ≠ {target}"
        )

    # For 2-number tails, enumerate every legal op so the narrator has a
    # complete explanation. For larger tails, give a higher-level statement.
    if len(numbers) == 2:
        a, b = numbers[1], numbers[0]  # sort descending for readability
        lines = []
        for sym, label in (("+", f"{a}+{b}={a + b}"), ("-", None), ("*", f"{a}*{b}={a * b}")):
            if sym == "-":
                if a - b > 0:
                    label = f"{a}-{b}={a - b}"
                else:
                    label = f"{a}-{b}≤0 illegal"
            lines.append(label)
        if b != 0 and a % b == 0:
            lines.append(f"{a}/{b}={a // b}")
        else:
            lines.append(f"{a}/{b} not integer")
        return (
            f"remaining={list(numbers)}, target={target} — no op reaches {target}: "
            + ", ".join(lines)
        )

    return (
        f"remaining={list(numbers)}, target={target} — exhaustive search of all binary "
        f"operations and continuations finds no path to {target}"
    )


# ---------------------------------------------------------------------------
# Module-level sanity checks. Run with `python -m dataset_generation.countdown_solver`.
# ---------------------------------------------------------------------------
def _selftest() -> None:
    # 1. Basic solvability — classic 24-game style cases.
    assert is_solvable((3, 4, 6, 8), 24)
    assert is_solvable((44, 19, 35), 98)
    assert is_solvable((25, 100, 9, 7, 3, 7), 765)  # canonical countdown target
    # 2. Unsolvable cases — degenerate small ones.
    assert not is_solvable((1, 1), 100)
    assert not is_solvable((2, 3), 100)
    # 3. Strict integer division: 8 / 3 must be rejected.
    assert apply_op(8, 3, "/") is None
    assert apply_op(9, 3, "/") == 3
    # 4. Strict positive subtraction: 3 - 5 illegal.
    assert apply_op(3, 5, "-") is None
    assert apply_op(5, 3, "-") == 2
    # 5. Commutativity pruning: + on (3,4) emits exactly one outcome (7), not two.
    plus_results = [r for (op, r) in all_next_states((3, 4)) if op[2] == "+"]
    assert len(plus_results) == 1, plus_results
    mul_results = [r for (op, r) in all_next_states((3, 4)) if op[2] == "*"]
    assert len(mul_results) == 1, mul_results
    # 6. Non-commutativity for - and /: (8,3) produces both orderings.
    minus_results = [op for (op, r) in all_next_states((8, 3)) if op[2] == "-"]
    assert len(minus_results) == 1  # only 8-3 is legal (3-8 negative); but op is still recorded
    # 7. Solution path roundtrip — apply ops in order, expect target.
    path = find_solution_path((3, 4, 6, 8), 24)
    assert path is not None
    state = list((3, 4, 6, 8))
    for a, b, op in path:
        assert a in state and b in state and (a != b or state.count(a) >= 2)
        state.remove(a)
        state.remove(b)
        r = apply_op(a, b, op)
        assert r is not None
        state.append(r)
    assert state == [24]
    # 8. Dead-end branch detector returns sequences whose tail is unsolvable.
    deads = find_dead_end_branches((44, 19, 35), 98, depth=1, max_branches=4)
    for branch in deads:
        s = list((44, 19, 35))
        for a, b, op in branch:
            s.remove(a)
            s.remove(b)
            s.append(apply_op(a, b, op))
        assert not is_solvable(tuple(s), 98)
    # 9. dead_end_reason returns a string mentioning target on a real dead end.
    msg = dead_end_reason((19, 1540), 98)
    assert "98" in msg and "1540" in msg
    print("countdown_solver selftest OK")


if __name__ == "__main__":
    _selftest()
