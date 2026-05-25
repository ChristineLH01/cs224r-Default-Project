"""Structural PRM for SELS v5 traces.

Scores well-formedness AND mathematical legality. Per event (a `<state_K>` or
`<backtrack_J>` token in textual order), label these rules:

  - state_reference_valid       backtrack J refers to a depth seen on this branch;
                                non-root state K references the immediate prior depth.
  - state_depth_consistent      len(state.list) == n_originals - K.
  - forward_transition_valid    state_K -> state_{K+1}: multiset shrinks by 1,
                                exactly 2 prior elements consumed, exactly 1 new
                                expression introduced, whose top-level operands
                                match (lexically, ignoring whitespace) the two
                                consumed elements' expression-parts.
  - backtrack_transition_valid  <backtrack_J> immediately followed by <state_J>
                                whose list multiset equals the most recent
                                ancestor at depth J on the active branch.
  - active_state_consistent     event is well-positioned: a non-root state has a
                                prior state; a backtrack has a non-empty branch.
  - operand_legal               (state events only) the multiset of integer
                                literals across all elements equals the nums
                                multiset. Catches operand drift / reuse — the
                                primary BAD-NUMS failure mode under PRM=structural.
  - arith_correct               (state events only) every (expr)=value annotation
                                has eval(expr) == value (integer arithmetic).

Trace-level rules (one boolean each, added to the aggregate):
  - think_well_formed           exactly one <think>...</think> wrapping the trace.
  - no_special_tokens           no chat-template tokens leaked into the trace.
  - answer_matches_final_state  <answer> text equals last single-element state's
                                expression-part (lexical match).

NOTE: an `answer_evaluates_to_target` rule was prototyped and dropped — its
unique signal vs R_out (the outcome reward already in compute_score) was ~5%
on real eval data, so it amounted to double-counting outcome reward. Operand
legality at every state already constrains the final answer indirectly.

Aggregation:    scalar = mean of all (event, rule) booleans, in [0, 1].
Sentinels (force scalar to 0.0, surfaced in metrics):
  - no <state_*> tokens parsed OR no <answer>...</answer> block.
  - len(events) > 3 * n_originals + 10   (anti backtrack-spam).

Public API:
    compute_process_reward(response_text, ground_truth) -> (float, dict)
"""
from __future__ import annotations

import ast
import re
from collections import Counter
from dataclasses import dataclass


# --- Parsing ----------------------------------------------------------------

# List is OPTIONAL so bare `<state_K>` and `<backtrack_J>` tokens are captured
# as events (with list_elems=None) and penalized rather than silently dropped.
_STATE_RE       = re.compile(r"<state_(\d+)>(?:\s*\[([^\]]*)\])?")
_BACKTRACK_RE   = re.compile(r"<backtrack_(\d+)>(?:\s*\[([^\]]*)\])?")
_ANSWER_RE      = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
# Trace-level structure checks.
_THINK_OPEN_RE  = re.compile(r"<think>")
_THINK_CLOSE_RE = re.compile(r"</think>")
# Special / chat-template tokens that must never appear in a generated trace.
_SPECIAL_RE     = re.compile(r"<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>")


@dataclass
class _Event:
    kind: str                     # "state" | "backtrack"
    depth: int
    list_elems: list[str] | None  # normalized in lexical form; None for backtrack
    span: tuple[int, int]


def _norm(s: str) -> str:
    """Whitespace-strip — our canonical form for lexical match."""
    return re.sub(r"\s+", "", s)


def _split_top_commas(list_raw: str) -> list[str]:
    """Split a bracket list at commas outside any parens, returning normalized elems."""
    out, buf, depth = [], [], 0
    for ch in list_raw:
        if ch == "(":
            depth += 1; buf.append(ch)
        elif ch == ")":
            depth -= 1; buf.append(ch)
        elif ch == "," and depth == 0:
            out.append(_norm("".join(buf))); buf = []
        else:
            buf.append(ch)
    if buf:
        out.append(_norm("".join(buf)))
    return [e for e in out if e]


def _expr_part(elem: str) -> str:
    """Strip the trailing `=value` from `(expr)=value`. `elem` is already _norm'd."""
    depth = 0
    for i, ch in enumerate(elem):
        if   ch == "(": depth += 1
        elif ch == ")": depth -= 1
        elif ch == "=" and depth == 0:
            return elem[:i]
    return elem


def _top_operands(expr: str) -> list[str] | None:
    """Return [LHS, RHS] of an outer-parenthesized binary expression, normalized.

    Returns None on parse failure (e.g., bare integer, unbalanced parens).
    """
    if not (expr.startswith("(") and expr.endswith(")")):
        return None
    inner = expr[1:-1]
    depth = 0
    for i, ch in enumerate(inner):
        if   ch == "(": depth += 1
        elif ch == ")": depth -= 1
        elif depth == 0 and ch in "+-*/" and i > 0:
            return [inner[:i], inner[i + 1:]]
    return None


# --- Integer arithmetic (for operand_legal, arith_correct, answer_evaluates_to_target) ---

def _leaf_integers(expr_str: str) -> list[int]:
    """All integer literals appearing in expr_str (left-to-right)."""
    return [int(s) for s in re.findall(r"\d+", expr_str)]


def _eval_node(node):
    """Recursively evaluate a restricted-AST arithmetic expression. Returns
    int / float on success, None on any failure (unknown op, name, div-by-0).
    """
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, (int, float)) else None
    if isinstance(node, ast.UnaryOp):
        v = _eval_node(node.operand)
        if v is None: return None
        if isinstance(node.op, ast.USub): return -v
        if isinstance(node.op, ast.UAdd): return +v
        return None
    if isinstance(node, ast.BinOp):
        l = _eval_node(node.left); r = _eval_node(node.right)
        if l is None or r is None: return None
        if isinstance(node.op, ast.Add):  return l + r
        if isinstance(node.op, ast.Sub):  return l - r
        if isinstance(node.op, ast.Mult): return l * r
        if isinstance(node.op, ast.Div):
            if r == 0: return None
            return l / r
        return None
    return None


def _eval_integer_expr(expr_str: str) -> int | None:
    """Evaluate expr_str; return int iff result is integer-valued, else None.

    Tolerant of integer-valued floats (e.g. 4.0) so that division paths that
    happen to be integer-divisible still pass; rejects genuine non-integers.
    """
    try:
        tree = ast.parse(expr_str, mode="eval")
    except Exception:
        return None
    r = _eval_node(tree)
    if r is None:
        return None
    if isinstance(r, int):
        return r
    if isinstance(r, float) and abs(r - round(r)) < 1e-9:
        return int(round(r))
    return None


def _check_operand_legal(elems: list[str], nums: list[int]) -> bool:
    """The multiset of integer literals across all elements equals nums.

    State-K's bracket list must use exactly the original numbers (multiset),
    whether bare or wrapped inside (expr). Catches the BAD-NUMS failure mode
    where the PRM-trained model reuses an original or invents a new literal.
    """
    all_literals: list[int] = []
    for e in elems:
        all_literals.extend(_leaf_integers(_expr_part(e)))
    return Counter(all_literals) == Counter(nums)


def _check_arith_correct(elems: list[str]) -> bool:
    """Every (expr)=value annotation in the list has eval(expr) == value.

    Bare integers (no `=value`) are skipped (vacuously valid). Returns False
    on the first annotation whose declared value disagrees with arithmetic.
    """
    for e in elems:
        # Locate the top-level `=` separating (expr) from declared value.
        depth = 0
        eq_idx = None
        for i, ch in enumerate(e):
            if   ch == "(": depth += 1
            elif ch == ")": depth -= 1
            elif ch == "=" and depth == 0:
                eq_idx = i; break
        if eq_idx is None:
            continue                          # bare integer; no annotation
        expr     = e[:eq_idx]
        val_str  = e[eq_idx + 1:]
        try:
            declared = int(val_str)
        except ValueError:
            return False
        computed = _eval_integer_expr(expr)
        if computed is None or computed != declared:
            return False
    return True


def _parse_events(text: str) -> list[_Event]:
    events: list[_Event] = []
    for m in _STATE_RE.finditer(text):
        raw = m.group(2)
        events.append(_Event(
            kind="state", depth=int(m.group(1)),
            list_elems=_split_top_commas(raw) if raw is not None else None,
            span=m.span(),
        ))
    for m in _BACKTRACK_RE.finditer(text):
        raw = m.group(2)
        # list_elems is non-None only when the model emitted a *malformed*
        # `<backtrack_J> [...]` with an attached bracket (v5 puts the bracket
        # on the next line, prefixed by `<state_J>`). We keep the value for
        # diagnostics; the labeler treats its presence as a structural failure.
        events.append(_Event(
            kind="backtrack", depth=int(m.group(1)),
            list_elems=_split_top_commas(raw) if raw is not None else None,
            span=m.span(),
        ))
    events.sort(key=lambda e: e.span[0])
    return events


# --- Rule labelling ---------------------------------------------------------

def _check_forward(prior: list[str], nxt: list[str]) -> bool:
    """Valid forward step prior -> nxt? See module docstring rule 3."""
    if len(nxt) != len(prior) - 1:
        return False
    consumed   = Counter(prior) - Counter(nxt)
    introduced = Counter(nxt)   - Counter(prior)
    if sum(consumed.values()) != 2 or sum(introduced.values()) != 1:
        return False
    new_elem      = next(introduced.elements())
    operands      = _top_operands(_expr_part(new_elem))
    if operands is None:
        return False
    expected_ops  = [_expr_part(c) for c in consumed.elements()]
    return Counter(_norm(o) for o in operands) == Counter(expected_ops)


def _label_events(events: list[_Event], n_originals: int,
                  nums: list[int]) -> list[dict]:
    """Walk events, maintain active-branch stack, emit one label dict per event.

    Branch entry = (depth, tuple(sorted(list_elems))). Forward => push; valid
    backtrack to J => pop until top.depth == J (ancestor remains the new tip).
    """
    labels: list[dict] = []
    branch: list[tuple[int, tuple[str, ...]]] = []

    i = 0
    while i < len(events):
        ev = events[i]

        if ev.kind == "state":
            if ev.list_elems is None:
                # Bare `<state_K>` with no following list. v5 invariant requires
                # every state token to be immediately followed by its bracket
                # list — bare tokens are malformed and fail every state rule.
                # Don't advance the branch (no canonical multiset to push).
                rules = {
                    "state_reference_valid":    False,
                    "state_depth_consistent":   False,
                    "forward_transition_valid": False,
                    "active_state_consistent":  False,
                    "operand_legal":            False,
                    "arith_correct":            False,
                }
                labels.append({"event": "state", "depth": ev.depth, "rules": rules})
                i += 1
                continue

            elems = ev.list_elems
            rules = {}
            if not branch:
                rules["state_reference_valid"]    = (ev.depth == 0)
                rules["state_depth_consistent"]   = (ev.depth == 0
                                                     and len(elems) == n_originals)
                rules["forward_transition_valid"] = True  # n/a at root
                rules["active_state_consistent"]  = (ev.depth == 0)
            else:
                prior_depth, prior_elems = branch[-1]
                rules["state_reference_valid"]    = (ev.depth == prior_depth + 1)
                rules["state_depth_consistent"]   = (len(elems) == n_originals - ev.depth
                                                     and ev.depth >= 0)
                rules["forward_transition_valid"] = _check_forward(list(prior_elems), elems)
                rules["active_state_consistent"]  = True
            rules["operand_legal"] = _check_operand_legal(elems, nums)
            rules["arith_correct"] = _check_arith_correct(elems)
            branch.append((ev.depth, tuple(sorted(elems))))
            labels.append({"event": "state", "depth": ev.depth, "rules": rules})
            i += 1

        else:  # backtrack — compound with next event if it's the matching state
            J = ev.depth
            nxt = events[i + 1] if i + 1 < len(events) else None
            has_attached_bracket = ev.list_elems is not None  # malformed v5 form
            rules = {}
            ancestor = next(((d, els) for d, els in reversed(branch) if d == J), None)
            # Strict ancestor: J must be shallower than the current branch tip.
            # Backtrack-to-current-depth is a no-op / loop exploit; reject it.
            rules["state_reference_valid"]   = (ancestor is not None
                                                and J < branch[-1][0])
            rules["active_state_consistent"] = (len(branch) > 0)

            if has_attached_bracket:
                # `<backtrack_J> [...]` puts the restored list on the same line
                # as the backtrack token. v5 invariant: bracket goes on the
                # next line, prefixed by `<state_J>`. Fail regardless of next.
                rules["backtrack_transition_valid"] = False
                labels.append({"event": "backtrack", "depth": J, "rules": rules})
                i += 1
            elif (nxt is not None and nxt.kind == "state" and nxt.depth == J
                    and nxt.list_elems is not None
                    and ancestor is not None):
                restored = tuple(sorted(nxt.list_elems))
                rules["backtrack_transition_valid"] = (restored == ancestor[1])
                # Pop deeper-than-J entries; ancestor stays as the new tip.
                while branch and branch[-1][0] != J:
                    branch.pop()
                labels.append({"event": "backtrack", "depth": J, "rules": rules})
                i += 2  # consumed both backtrack and its restored state
            else:
                rules["backtrack_transition_valid"] = False
                labels.append({"event": "backtrack", "depth": J, "rules": rules})
                i += 1

    return labels


# --- Public API -------------------------------------------------------------

def compute_process_reward(response_text: str, ground_truth: dict
                           ) -> tuple[float, dict]:
    """Structural well-formedness reward for a single SELS rollout.

    Args:
        response_text: the model's output, post-prompt.
        ground_truth:  dict with key "numbers" (list[int]) and "target" (int).
                       Only "numbers" is read here.

    Returns:
        (scalar in [0, 1], metrics dict)

    Metrics keys (always present): n_events, n_states, n_backtracks, has_answer,
    sentinel_no_events, sentinel_excess_events. Plus, when a rule is evaluable,
    rule_<name>_pass_rate (denominator = number of events for which the rule was
    defined). Plus rule_<name>_n (count).
    """
    nums = ground_truth.get("numbers") or ground_truth.get("nums") or []
    n_originals = len(nums)

    events       = _parse_events(response_text)
    n_states     = sum(1 for e in events if e.kind == "state")
    n_backtracks = sum(1 for e in events if e.kind == "backtrack")
    has_answer   = _ANSWER_RE.search(response_text) is not None

    metrics = {
        "n_events": len(events), "n_states": n_states,
        "n_backtracks": n_backtracks, "has_answer": has_answer,
        "sentinel_no_events": False, "sentinel_excess_events": False,
    }

    if n_states == 0 or not has_answer:
        metrics["sentinel_no_events"] = True
        return 0.0, metrics

    if len(events) > 3 * n_originals + 10:
        metrics["sentinel_excess_events"] = True
        return 0.0, metrics

    labels = _label_events(events, n_originals=n_originals, nums=nums)
    if not labels:
        metrics["sentinel_no_events"] = True
        return 0.0, metrics

    all_bits: list[int] = []
    per_rule: dict[str, list[int]] = {}
    for lab in labels:
        for name, val in lab["rules"].items():
            b = 1 if val else 0
            all_bits.append(b)
            per_rule.setdefault(name, []).append(b)

    # --- Trace-level structural rules (one boolean each, added to aggregate) --

    # think_well_formed: exactly one <think>, exactly one </think>, with the
    # close coming before the <answer> block. Catches "-endthink>" garbage,
    # double <think>, answers inside <think>.
    n_open  = len(_THINK_OPEN_RE.findall(response_text))
    n_close = len(_THINK_CLOSE_RE.findall(response_text))
    think_well_formed = (n_open == 1 and n_close == 1)
    ans_m = _ANSWER_RE.search(response_text)
    if think_well_formed and ans_m is not None:
        op = response_text.index("<think>")
        cp = response_text.index("</think>")
        think_well_formed = (op < cp < ans_m.start())

    # no_special_tokens: model special / chat-template tokens must not appear
    # inside a generated trace (we saw `<|im_end|>` mid-think in real rollouts).
    no_special_tokens = (_SPECIAL_RE.search(response_text) is None)

    # answer_matches_final_state: the trace's last single-element state's
    # expression-part must equal (whitespace-normalized) the <answer> content.
    # Catches "trace abandons structure at the end" — e.g., backtrack with
    # attached bracket then jumps straight to <answer> without a terminal state.
    last_single_state = next(
        (e for e in reversed(events)
         if e.kind == "state" and e.list_elems and len(e.list_elems) == 1),
        None,
    )
    answer_matches_final = False
    if last_single_state is not None and ans_m is not None:
        answer_expr = _norm(ans_m.group(1).strip())
        final_expr  = _expr_part(last_single_state.list_elems[0])
        answer_matches_final = (answer_expr == final_expr)

    for name, val in (("think_well_formed",            think_well_formed),
                      ("no_special_tokens",            no_special_tokens),
                      ("answer_matches_final_state",   answer_matches_final)):
        b = 1 if val else 0
        all_bits.append(b)
        per_rule.setdefault(name, []).append(b)

    # -------------------------------------------------------------------------

    scalar = sum(all_bits) / len(all_bits)
    metrics["reward"] = scalar
    for name, vals in per_rule.items():
        metrics[f"rule_{name}_pass_rate"] = sum(vals) / len(vals)
        metrics[f"rule_{name}_n"] = len(vals)
    return scalar, metrics


# --- Self-test --------------------------------------------------------------

if __name__ == "__main__":
    # (a) v5 SFT row should score 1.0 — sanity that the parser accepts the
    #     ground-truth format. (b) hand-crafted broken traces should score < 1.0
    #     or hit a sentinel.
    from datasets import load_from_disk  # type: ignore

    ds = load_from_disk("dataset_generation/output/sels_sft_v5")["train"]
    print(f"=== sanity: structural reward on v5 SFT (first 5 of each shape) ===")
    seen = {}
    for row in ds:
        shape = row["shape"]
        if seen.get(shape, 0) >= 5:
            continue
        seen[shape] = seen.get(shape, 0) + 1
        gt = {"numbers": row["nums"], "target": row["target"]}
        r, m = compute_process_reward(row["completion"], gt)
        print(f"  {shape:11s} nums={row['nums']!s:25s} target={row['target']:3d}"
              f"  reward={r:.3f}  states={m['n_states']:2d}"
              f"  backtracks={m['n_backtracks']:2d}")

    print("\n=== five real-rollout traces from a user-supplied batch (must score notably < 1.0) ===")
    real_cases = [
        ("trace 1 — depth mislabels + -endthink>",
         "<think>\n<state_0> [1, 36, 54]\n<state_1> [(36 / 1)=36, 54]\n"
         "<state_3> [(54 - (36 / 1))=18]\n<backtrack_1>\n"
         "<state_1> [(36 / 1)=36, 54]\n<state_3> [(54 + (36 / 1))=90]\n"
         "<backtrack_2>\n<state_2> [(54 - (36 / 1))=18]\n"
         "<state_2> [((54 - (36 / 1)) + (36 / 1))=54]\n-endthink>\n"
         "<answer> ((54 - (36 / 1)) + (36 / 1)) </answer>",
         {"numbers": [1, 36, 54], "target": 91}),
        ("trace 2 — state label mismatches w/ backtrack",
         "<think>\n<state_0> [36, 41, 60]\n<state_2> [(41 + 36)=77, 60]\n"
         "<backtrack_2>\n<state_1> [36, 41, 60]\n<state_2> [(41 - 36)=5, 60]\n"
         "<state_3> [(60 + (41 - 36))=65]\n<backtrack_2>\n"
         "<state_2> [(41 - 36)=5, 60]\n<state_3> [(60 / (41 - 36))=12]\n"
         "<state_0> [(60 + (60 / (41 - 36)))=72]\n</think>\n"
         "<answer> (60 + (60 / (41 - 36))) </answer>",
         {"numbers": [36, 41, 60], "target": 55}),
        ("trace 3 — bare <state_1> + backtrack-with-bracket",
         "<think>\n<state_0> [10, 28, 78]\n<state_1> [(28 - 10)=18, 78]\n"
         "<state_2> [(78 + (28 - 10))=96]\n<state_1>\n"
         "<state_1> [(28 - 10)=18, 78]\n"
         "<backtrack_1> [(78 - (28 - 10))=60]\n</think>\n"
         "<answer> (78 - (28 - 10)) </answer>",
         {"numbers": [10, 28, 78], "target": 60}),
        ("trace 4 — bare states + foreign-language gibberish",
         "<think>\n<state_0> [2, 16, 65]\n<state_1> [(16 + 2)=18, 65]\n"
         "<state_2> [(65 + (16 + 2))=83]\n<backtrack_1>\n"
         "<state_2> [(16 + 2)=18, 65]\n<state_2> [(65 - (16 + 2))=47]\n"
         "<state_2> [(65 + (65 - (16 + 2)))=112]\n<backtrack_1>\n"
         "<state_1> [(65 - (16 + 2))=47]\n"
         "<state_2> [(65 - (65 - (16 + 2)))=18]\n<state_1>\n"
         "<state_3> [(65 - (16 + 2))=47]\n"
         "<state_3> [(65 + (65 - (16 + 2)))=112]\nปืน\n"
         "<state_2> [(65 - (16 + 2))=47]\n"
         "<state_2> [(65 * (65 - (16 + 2)))=3005]\n<backtrack_1>\n"
         "<state_2> [(65 - (16 + 2))=47]\n"
         "<state_3> [(65 - (65 - (16 + 2)))=18]\n"
         "<state_2> [(65 + (65 - (16 + 2)))=83]\n</think>\n"
         "<answer> (65 + (65 - (16 + 2))) </answer>",
         {"numbers": [2, 16, 65], "target": 51}),
        ("trace 5 — bare states + <|im_end|> mid-trace",
         "<think>\n<state_0> [47, 50, 57, 65]\n"
         "<state_0> [(57 - 50)=7, 47, 65]\n"
         "<state_2> [47, (65 + (57 - 50))=72]\n<backtrack_2>\n"
         "<state_3> [(57 - 50)=7, 47, 65]\n"
         "<state_2> [(47 + (57 - 50))=54, 65]\n"
         "<state_3> [(65 - (47 + (57 - 50)))=11]\n<state_3>\n"
         "<state_1> [(47 + (57 - 50))=54, 65]\n"
         "<state_3> [(65 + (47 + (57 - 50)))=119]\n<state_3>\n"
         "<|im_end|> [(65 + (47 + (57 - 50)))=119]\n<state_2>\n"
         "<state_2> [(47 + (57 - 50))=54, 65]\n"
         "<state_3> [(65 + (47 + (57 - 50)))=119]\n<state_2>\n"
         "<state_2> [(47 + (57 - 50))=54, 65]\n"
         "<state_1> [(65 - (47 + (57 - 50)))=11]\n<state_3>\n"
         "<state_3> [(65 - (47 + (57 - 50)))=11]\n</think>\n"
         "<answer> (65 - (47 + (57 - 50))) </answer>",
         {"numbers": [47, 50, 57, 65], "target": 70}),
    ]
    for name, text, gt in real_cases:
        r, m = compute_process_reward(text, gt)
        flags = []
        for rule in ("think_well_formed", "no_special_tokens",
                     "answer_matches_final_state"):
            v = m.get(f"rule_{rule}_pass_rate", None)
            if v is not None and v < 1.0:
                flags.append(rule)
        marker = ""
        if m["sentinel_no_events"]:    marker = " (sentinel_no_events)"
        if m["sentinel_excess_events"]: marker = " (sentinel_excess_events)"
        print(f"  reward={r:.3f}{marker}  trace_lvl_fail={flags}  — {name}")

    print("\n=== broken traces (should be < 1.0 or hit sentinel) ===")
    cases = [
        ("only an answer, no states",
         "<answer> (1+2) </answer>", {"numbers": [1,2,3], "target": 3}),
        ("88/88 from a state with one 88 (forward_transition_valid fail)",
         "<think>\n<state_0> [88, 7, 5]\n<state_1> [7, 5, (88/88)=1]\n</think>\n"
         "<answer> ((88/88)+7+5) </answer>", {"numbers": [88,7,5], "target": 13}),
        ("dropped operand (forward_transition_valid fail)",
         "<think>\n<state_0> [3, 7, 9]\n<state_1> [(7+9)=16]\n</think>\n"
         "<answer> (7+9) </answer>", {"numbers": [3,7,9], "target": 16}),
        ("backtrack to never-visited depth (state_reference_valid fail)",
         "<think>\n<state_0> [3, 7, 9]\n<backtrack_5>\n<state_5> [3, 7, 9]\n</think>\n"
         "<answer> (3+7+9) </answer>", {"numbers": [3,7,9], "target": 19}),
        ("backtrack to current depth — no-op loop (state_reference_valid fail post-tightening)",
         "<think>\n<state_0> [48, 74, 94]\n<state_1> [(74-48)=26, 94]\n"
         "<state_2> [(94-(74-48))=68]\n<backtrack_2>\n"
         "<state_2> [(94-(74-48))=68]\n</think>\n<answer> (94-(74-48)) </answer>",
         {"numbers": [48,74,94], "target": 68}),
        ("excess events sentinel",
         "<think>\n<state_0> [1,2,3]\n" + "<backtrack_0>\n<state_0> [1,2,3]\n"*15
         + "</think>\n<answer> (1+2+3) </answer>", {"numbers": [1,2,3], "target": 6}),
    ]
    for name, text, gt in cases:
        r, m = compute_process_reward(text, gt)
        marker = " (sentinel_no_events)" if m["sentinel_no_events"] else \
                 " (sentinel_excess_events)" if m["sentinel_excess_events"] else ""
        print(f"  reward={r:.3f}{marker}  — {name}")
