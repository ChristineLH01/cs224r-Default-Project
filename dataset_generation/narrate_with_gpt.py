"""Narrate a Skeleton into a `completion` string via GPT-4o-mini.

The skeleton already determines all structural pieces (tokens, arithmetic
facts, dead-end reasons, the final equation). The narrator only invents the
prose inside `<think>...</think>` blocks. A strict validator catches any
narrator deviation and triggers retries.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Optional

from dataset_generation.skeleton_builder import (
    AnswerEvent,
    ArithStep,
    BacktrackEvent,
    ResumeEvent,
    Skeleton,
    StateEvent,
    annotate_expressions,
    arith_fact,
    state_list,
    skeleton_to_spec,
)

DEFAULT_MODEL = "gpt-4.1-mini"
_FEWSHOT_PATH = Path(__file__).parent / "prompts" / "narrator_fewshot.txt"


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------
def _fewshot_prompt() -> str:
    return _FEWSHOT_PATH.read_text()


def _build_user_message(sk: Skeleton) -> str:
    return (
        "Rewrite the following structured trace into a single completion string,\n"
        "following every rule from the system prompt.\n\n"
        "INPUT:\n"
        + skeleton_to_spec(sk)
        + "\n\nOUTPUT:\n"
    )


# ---------------------------------------------------------------------------
# Output cleanup before validation
# ---------------------------------------------------------------------------
_FENCE_RE = re.compile(r"^\s*```[a-zA-Z]*\s*\n?|\n?```\s*$")


def _cleanup(text: str) -> str:
    # Strip outer markdown fences if the model wrapped its response.
    cleaned = text.strip()
    # Handle wrapped responses iteratively (some models double-wrap).
    for _ in range(3):
        m_open = re.match(r"^\s*```[a-zA-Z]*\s*\n", cleaned)
        m_close = re.search(r"\n?```\s*$", cleaned)
        if m_open and m_close:
            cleaned = cleaned[m_open.end() : m_close.start()].strip()
        else:
            break
    # Unescape backslash-escaped specials: \<state\_0\> -> <state_0>
    cleaned = cleaned.replace("\\<", "<").replace("\\>", ">").replace("\\_", "_")
    return cleaned


# ---------------------------------------------------------------------------
# Validation — 6 checks per the plan
# ---------------------------------------------------------------------------
class ValidationError(Exception):
    pass


def _expected_tag_counts(sk: Skeleton) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ev in sk.events:
        if isinstance(ev, StateEvent):
            counts[f"<state_{ev.state_id}>"] = counts.get(f"<state_{ev.state_id}>", 0) + 1
        elif isinstance(ev, BacktrackEvent):
            counts[f"<backtrack_{ev.target_state_id}>"] = (
                counts.get(f"<backtrack_{ev.target_state_id}>", 0) + 1
            )
    return counts


def _check_tag_counts(completion: str, sk: Skeleton) -> None:
    expected = _expected_tag_counts(sk)
    for tag, want in expected.items():
        got = completion.count(tag)
        if got != want:
            raise ValidationError(f"tag {tag} count mismatch: want {want}, got {got}")
    # Reject hallucinated tags not in expected.
    for K in range(5):
        for tag in (f"<state_{K}>", f"<backtrack_{K}>"):
            if tag not in expected and tag in completion:
                raise ValidationError(f"unexpected tag {tag} in completion")


def _check_ordering(completion: str, sk: Skeleton) -> None:
    needles: list[str] = []
    for ev in sk.events:
        if isinstance(ev, StateEvent):
            needles.append(f"<state_{ev.state_id}>")
        elif isinstance(ev, BacktrackEvent):
            needles.append(f"<backtrack_{ev.target_state_id}>")
        elif isinstance(ev, AnswerEvent):
            needles.append("<answer>")
    cursor = 0
    for n in needles:
        idx = completion.find(n, cursor)
        if idx < 0:
            raise ValidationError(f"token {n} missing or out of order at cursor {cursor}")
        cursor = idx + len(n)


def _check_arith_lines(completion: str, sk: Skeleton) -> None:
    """Every ArithStep must appear in expression-over-originals form,
    e.g. '(94 - 18) / 19 = 4', so no bare intermediate is ever an operand."""
    annos = annotate_expressions(sk)
    for ev, anno in zip(sk.events, annos):
        if isinstance(ev, ArithStep):
            needle = arith_fact(ev, anno)
            if needle not in completion:
                raise ValidationError(f"arithmetic line {needle!r} missing from completion")


def _check_answer(completion: str, sk: Skeleton) -> None:
    answer = next((e for e in sk.events if isinstance(e, AnswerEvent)), None)
    if answer is None:
        return
    matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", completion, re.DOTALL)
    if len(matches) != 1:
        raise ValidationError(f"expected exactly one <answer>...</answer>, got {len(matches)}")
    got = matches[0].strip()
    want = answer.equation_str.strip()
    # Allow whitespace differences only.
    if re.sub(r"\s+", "", got) != re.sub(r"\s+", "", want):
        raise ValidationError(f"answer mismatch: want {want!r}, got {got!r}")


def _check_equation_correctness(sk: Skeleton) -> None:
    """Independent re-evaluation of the answer equation against the target."""
    from evaluation.countdown import evaluate_equation, validate_equation

    answer = next((e for e in sk.events if isinstance(e, AnswerEvent)), None)
    if answer is None:
        return
    if not validate_equation(answer.equation_str, list(sk.numbers)):
        raise ValidationError(f"equation {answer.equation_str} does not use numbers exactly")
    val = evaluate_equation(answer.equation_str)
    if val is None or abs(val - sk.target) > 1e-5:
        raise ValidationError(
            f"equation {answer.equation_str} evaluates to {val}, target {sk.target}"
        )


def _check_state_lists(completion: str, sk: Skeleton) -> None:
    """Every `<state_K>` token must be immediately followed (after optional
    whitespace) by its EXACT bracket list, e.g. `<state_1> [44, (35 + 19)=54]`.

    Walking in event order with a single cursor handles the same K reappearing
    with different content (e.g. state_2 thrice in multi_bt). This also covers
    the restored checkpoint after a backtrack: in v5 the `<backtrack_K>` token is
    immediately followed by the restored `<state_K> [list]`, with no preamble, so
    the restored state's number list is verified here rather than by a resume
    anchor.
    """
    annos = annotate_expressions(sk)
    cursor = 0
    for ev, anno in zip(sk.events, annos):
        if not isinstance(ev, StateEvent):
            continue
        tok = f"<state_{ev.state_id}>"
        idx = completion.find(tok, cursor)
        if idx < 0:
            raise ValidationError(f"state token {tok} missing or out of order at cursor {cursor}")
        after = completion[idx + len(tok):]
        ws = len(after) - len(after.lstrip())
        want = state_list(anno.available)
        if not after.lstrip().startswith(want):
            got = after.lstrip()[: len(want) + 20]
            raise ValidationError(
                f"{tok} not immediately followed by its list; want {want!r}, got {got!r}"
            )
        cursor = idx + len(tok) + ws + len(want)


def _check_single_think(completion: str) -> None:
    """v4: exactly one <think>...</think> wrapping the search, with <answer> after it."""
    if completion.count("<think>") != 1 or completion.count("</think>") != 1:
        raise ValidationError("expected exactly one <think>...</think> block")
    if "<answer>" in completion and completion.index("</think>") > completion.index("<answer>"):
        raise ValidationError("<answer> must come after </think>")


def validate(completion: str, sk: Skeleton) -> None:
    """Run all checks. Raise ValidationError on first failure."""
    _check_single_think(completion)
    _check_tag_counts(completion, sk)
    _check_ordering(completion, sk)
    _check_arith_lines(completion, sk)
    _check_answer(completion, sk)
    _check_equation_correctness(sk)
    _check_state_lists(completion, sk)


# ---------------------------------------------------------------------------
# OpenAI call (async)
# ---------------------------------------------------------------------------
async def _call_openai_async(system_prompt: str, user_message: str, model: str) -> str:
    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.7,
        max_tokens=1500,
    )
    return resp.choices[0].message.content or ""


async def narrate_async(
    sk: Skeleton, model: str = DEFAULT_MODEL, max_attempts: int = 3
) -> Optional[str]:
    """Narrate a skeleton, retrying up to `max_attempts` times on validation failure.

    Returns the validated completion string, or None if all attempts failed.
    """
    system = _fewshot_prompt()
    user = _build_user_message(sk)
    last_err: Optional[Exception] = None
    for attempt in range(max_attempts):
        try:
            raw = await _call_openai_async(system, user, model)
        except Exception as e:  # network / rate-limit / API errors
            last_err = e
            await asyncio.sleep(2 ** attempt)
            continue
        cleaned = _cleanup(raw)
        try:
            validate(cleaned, sk)
        except ValidationError as e:
            last_err = e
            # Strengthen the next attempt's prompt with the specific failure.
            user = _build_user_message(sk) + (
                f"\n\nPREVIOUS ATTEMPT FAILED VALIDATION: {e}\n"
                "Re-emit a corrected completion that passes all rules."
            )
            continue
        return cleaned
    print(f"  [drop] {sk.shape} {list(sk.numbers)}->{sk.target}: {last_err}")
    return None


def narrate(sk: Skeleton, model: str = DEFAULT_MODEL, max_attempts: int = 3) -> Optional[str]:
    """Synchronous wrapper around narrate_async — convenient for unit testing."""
    return asyncio.run(narrate_async(sk, model=model, max_attempts=max_attempts))


# ---------------------------------------------------------------------------
# Selftest: validation logic only — does not call the API.
# ---------------------------------------------------------------------------
def _selftest() -> None:
    import random
    import sys
    from pathlib import Path

    PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    from dataset_generation.skeleton_builder import build_skeleton

    rng = random.Random(0)
    sk = build_skeleton([44, 19, 35], 98, "bt_inner", rng)
    assert sk is not None

    # Programmatically render a compliant completion from the skeleton events
    # so the test stays in sync with whatever shape we used above.
    # v4: ONE <think> wraps the whole search; <answer> outside.
    annos = annotate_expressions(sk)
    answer = sk.events[-1]
    parts = ["<think>"]
    for ev, anno in zip(sk.events, annos):
        if isinstance(ev, StateEvent):
            parts.append(f"<state_{ev.state_id}> {state_list(anno.available)}")
        elif isinstance(ev, ArithStep):
            parts.append(arith_fact(ev, anno))
        elif isinstance(ev, BacktrackEvent):
            parts.append(ev.reason)
            parts.append(f"<backtrack_{ev.target_state_id}>")
        elif isinstance(ev, ResumeEvent):
            # v5: no preamble — the restored <state_K> (next event) renders the checkpoint.
            continue
    parts.append("</think>")
    parts.append(f"<answer> {answer.equation_str} </answer>")
    good = "\n".join(parts)

    validate(good, sk)
    print("narrate_with_gpt validator passes a hand-crafted good completion")

    # Negative test: a state token not immediately followed by its exact list.
    state0_list = state_list(annos[0].available)
    bad = good.replace(f"<state_0> {state0_list}", "<state_0> [999]", 1)
    try:
        validate(bad, sk)
    except ValidationError:
        print("narrate_with_gpt validator correctly rejects a corrupted state list")
    else:
        raise AssertionError("validator failed to catch corrupted state list")

    # Negative test: wrong arithmetic.
    first_arith = next(e for e in sk.events if isinstance(e, ArithStep))
    bad2 = good.replace(f"{first_arith.a} {first_arith.op} {first_arith.b} = {first_arith.result}", "99 + 99 = 198")
    try:
        validate(bad2, sk)
    except ValidationError:
        print("narrate_with_gpt validator correctly rejects altered arithmetic")
    else:
        raise AssertionError("validator failed to catch altered arithmetic")

    # Negative test: cleanup strips a code fence.
    fenced = "```\n" + good + "\n```"
    assert _cleanup(fenced).startswith("<think>")
    print("narrate_with_gpt cleanup correctly strips code fences")

    # Negative test: cleanup unescapes backslashes.
    escaped = good.replace("<state_0>", "\\<state\\_0\\>")
    assert _cleanup(escaped).count("<state_0>") == good.count("<state_0>")
    print("narrate_with_gpt cleanup correctly unescapes markdown specials")

    print("narrate_with_gpt selftest OK")


if __name__ == "__main__":
    _selftest()
