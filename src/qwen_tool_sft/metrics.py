"""Automatic tool-calling evaluation metrics.

Per-case scoring against ground-truth expectations:
  expect = {"mode": "tool"|"no_tool",
            "calls": [{"name": str, "arguments": {key: value, ...}}]}

Aggregated metrics:
  Tool Selection Accuracy      tool cases with exactly the right tool names
  Argument Exact Match         tool cases whose expected args all match
  Argument Key Accuracy        expected argument keys present (macro avg)
  Valid Tool Call Format Rate  tool cases with >=1 parseable valid call
  No-Tool Accuracy             no-tool cases answered without any call
  Overall Exact Match          all of the above satisfied per case
  Invalid JSON Rate / Wrong Tool Rate / Missing Argument Rate /
  Extra Argument Rate
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .parser import ParseResult


def normalize_value(v: Any) -> Any:
    if isinstance(v, str):
        s = v.strip()
        try:  # numeric string -> number
            f = float(s)
            return f
        except ValueError:
            pass
        return s.lower()
    if isinstance(v, (list, tuple)):
        try:
            return sorted(normalize_value(x) for x in v)
        except TypeError:
            return [normalize_value(x) for x in v]
    if isinstance(v, dict):
        return {k: normalize_value(x) for k, x in v.items()}
    return v


@dataclass
class CaseScore:
    case_id: str
    category: str
    expected_tool: bool
    valid_format: bool = False
    tool_selection_correct: bool = False
    arg_key_accuracy: float | None = None
    arg_exact_match: bool = False
    no_tool_correct: bool | None = None
    overall: bool = False
    invalid_json: bool = False
    wrong_tool: bool = False
    missing_argument: bool = False
    extra_argument: bool = False
    expected_names: list[str] = field(default_factory=list)
    predicted_names: list[str] = field(default_factory=list)


def _match_calls(expected: list[dict], predicted_calls: list):
    """Greedy match expected calls to predicted calls.

    Same-name calls (e.g. two parallel get_stock_price calls) are disambiguated
    by maximizing expected-argument value matches.
    Returns (alignment aligned to expected, unmatched predicted indices).
    """
    used: set[int] = set()
    alignment = []
    for exp in expected:
        best_idx, best_score = -1, -1
        exp_args = exp.get("arguments", {}) or {}
        for i, pc in enumerate(predicted_calls):
            if i in used or pc.name != exp["name"]:
                continue
            score = sum(1 for k, v in exp_args.items()
                        if normalize_value(pc.arguments.get(k)) == normalize_value(v))
            if score > best_score:
                best_idx, best_score = i, score
        alignment.append(best_idx)
        if best_idx >= 0:
            used.add(best_idx)
    unmatched = [i for i in range(len(predicted_calls)) if i not in used]
    return alignment, unmatched


def score_case(case: dict, parsed: ParseResult) -> CaseScore:
    expect = case.get("expect", {})
    mode = expect.get("mode", "tool")
    expected_tool = mode == "tool"
    score = CaseScore(
        case_id=case.get("id", ""),
        category=case.get("category", ""),
        expected_tool=expected_tool,
        expected_names=[c["name"] for c in expect.get("calls", [])],
        predicted_names=parsed.names,
    )

    if not expected_tool:
        score.no_tool_correct = not parsed.has_calls
        score.overall = score.no_tool_correct
        score.wrong_tool = parsed.has_calls
        score.invalid_json = bool(parsed.invalid_blocks) and not parsed.has_calls
        return score

    expected_calls = expect.get("calls", [])
    score.valid_format = any(c.valid_json for c in parsed.calls)
    score.invalid_json = bool(parsed.invalid_blocks) or (
        parsed.has_calls and not all(c.valid_json for c in parsed.calls)
    )

    alignment, unmatched_pred = _match_calls(expected_calls, parsed.calls)
    score.tool_selection_correct = (
        all(i >= 0 for i in alignment) and not unmatched_pred
        and Counter(score.expected_names) == Counter(score.predicted_names)
    )
    score.wrong_tool = not score.tool_selection_correct

    # argument scoring on matched calls
    key_accs, exacts = [], []
    for exp, pidx in zip(expected_calls, alignment):
        exp_args = exp.get("arguments", {}) or {}
        if pidx < 0:
            key_accs.append(0.0)
            exacts.append(False)
            score.missing_argument = True
            continue
        pred_args = parsed.calls[pidx].arguments
        exp_keys = set(exp_args.keys())
        present = exp_keys & set(pred_args.keys())
        key_accs.append(len(present) / len(exp_keys) if exp_keys else 1.0)
        if exp_keys - set(pred_args.keys()):
            score.missing_argument = True
        extra = set(pred_args.keys()) - exp_keys
        if extra:
            score.extra_argument = True
        values_ok = all(
            normalize_value(pred_args.get(k)) == normalize_value(v) for k, v in exp_args.items()
        )
        exacts.append(bool(values_ok) and not extra)
    score.arg_key_accuracy = sum(key_accs) / len(key_accs) if key_accs else 0.0
    score.arg_exact_match = all(exacts) and bool(exacts)
    score.overall = (
        score.valid_format and score.tool_selection_correct and score.arg_exact_match
    )
    return score


def aggregate(scores: list[CaseScore]) -> dict:
    tool = [s for s in scores if s.expected_tool]
    no_tool = [s for s in scores if not s.expected_tool]

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    by_category: dict[str, dict] = {}
    for s in scores:
        d = by_category.setdefault(s.category, {"n": 0, "overall": 0})
        d["n"] += 1
        d["overall"] += int(s.overall)
    for d in by_category.values():
        d["accuracy"] = d["overall"] / d["n"]

    return {
        "n_cases": len(scores),
        "n_tool_cases": len(tool),
        "n_no_tool_cases": len(no_tool),
        "tool_selection_accuracy": mean([int(s.tool_selection_correct) for s in tool]),
        "argument_exact_match": mean([int(s.arg_exact_match) for s in tool]),
        "argument_key_accuracy": mean([s.arg_key_accuracy or 0.0 for s in tool]),
        "valid_tool_call_format_rate": mean([int(s.valid_format) for s in tool]),
        "no_tool_accuracy": mean([int(s.no_tool_correct or 0) for s in no_tool]),
        "overall_exact_match": mean([int(s.overall) for s in scores]),
        "invalid_json_rate": mean([int(s.invalid_json) for s in tool]),
        "wrong_tool_rate": mean([int(s.wrong_tool) for s in tool]),
        "missing_argument_rate": mean([int(s.missing_argument) for s in tool]),
        "extra_argument_rate": mean([int(s.extra_argument) for s in tool]),
        "by_category": by_category,
    }


def metrics_table(metrics_a: dict, name_a: str, metrics_b: dict | None = None,
                  name_b: str = "SFT") -> str:
    rows = [
        ("Tool Selection Accuracy", "tool_selection_accuracy"),
        ("Argument Exact Match", "argument_exact_match"),
        ("Argument Key Accuracy", "argument_key_accuracy"),
        ("Valid Tool Call Format Rate", "valid_tool_call_format_rate"),
        ("No-Tool Accuracy", "no_tool_accuracy"),
        ("Overall Exact Match", "overall_exact_match"),
        ("Invalid JSON Rate", "invalid_json_rate"),
        ("Wrong Tool Rate", "wrong_tool_rate"),
        ("Missing Argument Rate", "missing_argument_rate"),
        ("Extra Argument Rate", "extra_argument_rate"),
    ]
    if metrics_b is None:
        lines = ["| Metric | %s |" % name_a, "|---|---:|"]
        for label, key in rows:
            lines.append(f"| {label} | {metrics_a[key]*100:.1f}% |")
        return "\n".join(lines)
    lines = ["| Metric | %s | %s | Delta |" % (name_a, name_b), "|---|---:|---:|---:|"]
    for label, key in rows:
        a, b = metrics_a[key], metrics_b[key]
        lines.append(f"| {label} | {a*100:.1f}% | {b*100:.1f}% | {(b-a)*100:+.1f}pt |")
    return "\n".join(lines)
