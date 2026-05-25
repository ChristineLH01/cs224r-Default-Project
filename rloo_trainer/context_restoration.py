"""Segmented-sampling helpers for SELS RLOO with context restoration.

Used only when `--context_restoration` is set on `rloo.py`. Default RL path
never imports this module.

Idea: when a SELS rollout emits `<backtrack_K>`, instead of letting the failed
branch sit in context and corrupt downstream generation, we truncate the kept
trace to the matching `<state_K> [list]` ancestor, append a clean
`<backtrack_K>\n<state_K> [list]\n` marker, and continue generation from that
prefix. From the model's perspective, each segment is generated with no
abandoned-branch tokens in its conditioning context — exactly the proposal's
"context manager restores the context to the target checkpoint" claim.

Pure stdlib so this file can be self-tested with `python -m rloo_trainer.context_restoration`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


_STATE_RE     = re.compile(r"<state_(\d+)>(?:\s*\[([^\]]*)\])?")
_BACKTRACK_RE = re.compile(r"<backtrack_(\d+)>")


@dataclass
class _Event:
    kind: str             # "state" | "backtrack"
    depth: int
    list_raw: str | None  # bracket contents; None for backtracks / bare states
    end_pos: int          # byte position immediately after the full match


def _parse_events(text: str) -> list[_Event]:
    evts: list[_Event] = []
    for m in _STATE_RE.finditer(text):
        evts.append(_Event("state", int(m.group(1)), m.group(2), m.end()))
    for m in _BACKTRACK_RE.finditer(text):
        evts.append(_Event("backtrack", int(m.group(1)), None, m.end()))
    evts.sort(key=lambda e: e.end_pos)
    return evts


def backtrack_stop_strings(max_depth: int = 4) -> list[str]:
    """`<backtrack_0>`..`<backtrack_K>` — the SELS v5 special-token set."""
    return [f"<backtrack_{k}>" for k in range(max_depth + 1)]


def parse_backtrack_stop(stop_reason) -> int | None:
    """Return K from `<backtrack_K>`, else None.

    vLLM populates `CompletionOutput.stop_reason` with the literal stop string
    for stop-string stops, `None` for EOS, and a token-id int for stop-token
    stops (unused here).
    """
    if not isinstance(stop_reason, str):
        return None
    m = re.fullmatch(r"<backtrack_(\d+)>", stop_reason)
    return int(m.group(1)) if m is not None else None


def build_cleanup(full_response: str, target_K: int) -> tuple[str, str] | None:
    """Compute (cleaned_response, restore_marker) for a rollout that just
    stopped on `<backtrack_target_K>`.

    Walks all (state, backtrack) events, maintains the active branch (push on
    forward state; pop deeper-than-K entries on backtrack; same-list state
    after a backtrack refreshes the tip's `end_pos` so we cut at the *latest*
    occurrence of that ancestor — preserving any prior cleanups inline), then
    returns:

        cleaned_response = full_response[:end_of_latest_target_K_ancestor]
        restore_marker   = "\n<backtrack_K>\n<state_K> [<list>]\n"

    Returns None if no valid ancestor at depth `target_K` exists on the active
    branch (malformed trace; caller should fall back to keeping the segment
    as-is and stopping that rollout).
    """
    events = _parse_events(full_response)
    # branch entries: (depth, normalized_key, end_pos, original_list_raw)
    branch: list[tuple[int, str, int, str]] = []
    for ev in events:
        if ev.kind == "state":
            if ev.list_raw is None:
                continue                          # bare <state_K>; ignore
            key = re.sub(r"\s+", "", ev.list_raw)
            if branch and branch[-1][0] == ev.depth and branch[-1][1] == key:
                # post-backtrack restored state: refresh tip end_pos.
                branch[-1] = (ev.depth, key, ev.end_pos, ev.list_raw)
            else:
                branch.append((ev.depth, key, ev.end_pos, ev.list_raw))
        else:  # backtrack
            while branch and branch[-1][0] != ev.depth:
                branch.pop()
    ancestor = next(((d, k, ep, raw) for d, k, ep, raw in reversed(branch)
                     if d == target_K), None)
    if ancestor is None:
        return None
    _, _, end_pos, list_raw = ancestor
    cleaned = full_response[:end_pos]
    restore_marker = f"\n<backtrack_{target_K}>\n<state_{target_K}> [{list_raw}]\n"
    return cleaned, restore_marker


# --- Self-test --------------------------------------------------------------

if __name__ == "__main__":
    cases = [
        ("simple chain, backtrack to root",
         "<think>\n<state_0> [a, b, c]\n<state_1> [(a+b)=ab, c]\n"
         "<state_2> [((a+b)+c)=abc]\n<backtrack_0>",
         0, "<state_0> [a, b, c]",
         "\n<backtrack_0>\n<state_0> [a, b, c]\n", 1),
        ("backtrack to depth 1, drop the depth-2 dead end",
         "<think>\n<state_0> [a, b, c]\n<state_1> [(a+b)=ab, c]\n"
         "<state_2> [((a+b)+c)=abc]\n<backtrack_1>",
         1, "<state_1> [(a+b)=ab, c]",
         "\n<backtrack_1>\n<state_1> [(a+b)=ab, c]\n", 1),
        ("second backtrack after a restored state — cut at LATEST occurrence",
         "<think>\n<state_0> [a, b, c]\n"
         "<state_1> [(a+b)=ab, c]\n<state_2> [((a+b)+c)=abc]\n"
         "<backtrack_1>\n<state_1> [(a+b)=ab, c]\n"          # prior cleanup
         "<state_2> [((a+b)-c)=abz]\n<backtrack_1>",
         1, "<state_1> [(a+b)=ab, c]",
         "\n<backtrack_1>\n<state_1> [(a+b)=ab, c]\n", 2),    # both <state_1> kept
        ("unanchored backtrack to depth that never appeared",
         "<think>\n<state_0> [a, b, c]\n<backtrack_4>",
         4, None, None, None),
        ("backtrack from depth 3 to depth 0 (multi-pop)",
         "<think>\n<state_0> [a, b, c, d]\n<state_1> [(a+b)=ab, c, d]\n"
         "<state_2> [((a+b)+c)=abc, d]\n<state_3> [(((a+b)+c)+d)=abcd]\n<backtrack_0>",
         0, "<state_0> [a, b, c, d]",
         "\n<backtrack_0>\n<state_0> [a, b, c, d]\n", 1),
    ]
    for name, text, K, end_match, exp_restore, exp_count in cases:
        result = build_cleanup(text, K)
        if end_match is None:
            assert result is None, f"{name}: expected None, got {result!r}"
            print(f"  PASS  (None as expected)  — {name}")
            continue
        cleaned, restore = result
        assert cleaned.endswith(end_match), (
            f"{name}: cleaned tail={cleaned[-80:]!r} expected end={end_match!r}")
        assert restore == exp_restore, (
            f"{name}: restore={restore!r} expected={exp_restore!r}")
        got_count = cleaned.count(end_match)
        assert got_count == exp_count, (
            f"{name}: expected {exp_count} occurrences of ancestor in cleaned, "
            f"got {got_count}")
        print(f"  PASS  — {name}")
    print("\nstop strings:", backtrack_stop_strings(4))
    print("parse:", parse_backtrack_stop("<backtrack_2>"),
          parse_backtrack_stop("</answer>"), parse_backtrack_stop(None))
