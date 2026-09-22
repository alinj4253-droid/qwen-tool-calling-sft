"""Strict Protocol Success metric (P0-4) and abstention naming."""
from __future__ import annotations

import json

from qwen_tool_sft.metrics import (
    aggregate, prompt_leaked, score_case,
)
from qwen_tool_sft.parser import parse_tool_calls


def native_raw(calls, stop=True):
    out = ""
    for name, args in calls:
        out += "<tool_call>\n" + json.dumps(
            {"name": name, "arguments": args}, ensure_ascii=False) + "\n</tool_call>\n"
    return out + ("<|im_end|>" if stop else "")


def fenced_raw(name, args, stop=True):
    out = "```json\n" + json.dumps(
        {"name": name, "arguments": args}, ensure_ascii=False) + "\n```\n"
    return out + ("<|im_end|>" if stop else "")


def tool_case(cid="c1", category="single_tool", calls=(("get_weather", {"city": "北京"}),)):
    return {"id": cid, "category": category,
            "expect": {"mode": "tool",
                       "calls": [{"name": n, "arguments": a} for n, a in calls]}}


def no_tool_case(cid="n1"):
    return {"id": cid, "category": "no_tool", "expect": {"mode": "no_tool"}}


def test_perfect_native_tool_case_is_strict():
    raw = native_raw([("get_weather", {"city": "北京"})])
    sc = score_case(tool_case(), parse_tool_calls(raw), raw)
    assert sc.overall and sc.canonical_format and sc.clean_stop
    assert sc.strict_protocol_success is True


def test_fenced_json_is_overall_correct_but_not_strict():
    raw = fenced_raw("get_weather", {"city": "北京"})
    sc = score_case(tool_case(), parse_tool_calls(raw), raw)
    assert sc.overall is True           # lenient correctness still holds
    assert sc.canonical_format is False  # protocol wrapper missing
    assert sc.strict_protocol_success is False


def test_correct_call_without_clean_stop_is_not_strict():
    raw = native_raw([("get_weather", {"city": "北京"})], stop=False)
    sc = score_case(tool_case(), parse_tool_calls(raw), raw)
    assert sc.overall is True
    assert sc.clean_stop is False
    assert sc.strict_protocol_success is False


def test_wrong_arguments_not_strict():
    raw = native_raw([("get_weather", {"city": "上海"})])
    sc = score_case(tool_case(), parse_tool_calls(raw), raw)
    assert sc.overall is False
    assert sc.strict_protocol_success is False


def test_parallel_same_name_strict():
    calls = [("get_stock_price", {"symbol": "AAA"}),
             ("get_stock_price", {"symbol": "BBB"})]
    raw = native_raw(calls)
    case = tool_case(category="parallel_same_tool", calls=tuple(calls))
    sc = score_case(case, parse_tool_calls(raw), raw)
    assert sc.tool_selection_correct and sc.arg_exact_match
    assert sc.strict_protocol_success is True


def test_no_tool_abstain_clean_is_strict():
    raw = "这个问题我无法通过工具解决，建议直接查阅资料。<|im_end|>"
    sc = score_case(no_tool_case(), parse_tool_calls(raw), raw)
    assert sc.no_tool_correct is True and sc.clean_stop is True
    assert sc.strict_protocol_success is True


def test_no_tool_abstain_with_prompt_leak_not_strict():
    raw = "<|im_start|>user\n你好<|im_start|>assistant\n"
    assert prompt_leaked(raw)
    sc = score_case(no_tool_case(), parse_tool_calls(raw), raw)
    assert sc.no_tool_correct is True  # decision itself was right
    assert sc.clean_stop is False
    assert sc.strict_protocol_success is False


def test_no_tool_but_model_calls_tool_not_strict():
    raw = native_raw([("get_weather", {"city": "北京"})])
    sc = score_case(no_tool_case(), parse_tool_calls(raw), raw)
    assert sc.no_tool_correct is False
    assert sc.strict_protocol_success is False


def test_aggregate_strict_and_abstention_keys():
    raws = [
        native_raw([("get_weather", {"city": "北京"})]),          # strict tool
        fenced_raw("get_weather", {"city": "北京"}),              # not strict
        "好的，我不需要工具。<|im_end|>",                          # strict no-tool
        native_raw([("get_weather", {"city": "北京"})]),          # wrong case type
    ]
    cases = [tool_case("c1"), tool_case("c2"), no_tool_case("n1"), no_tool_case("n2")]
    scores = [score_case(c, parse_tool_calls(r), r) for c, r in zip(cases, raws)]
    m = aggregate(scores)
    assert m["strict_protocol_success"] == 0.5
    assert m["strict_protocol_success_tool"] == 0.5
    assert m["strict_protocol_success_no_tool"] == 0.5
    # abstention alias kept in sync with legacy key
    assert m["tool_abstention_accuracy"] == m["no_tool_accuracy"] == 0.5
