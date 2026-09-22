"""Parallel tool-call evaluation: multi-block rendering + same-name alignment."""
from __future__ import annotations

import json

from qwen_tool_sft.metrics import aggregate, score_case
from qwen_tool_sft.parser import parse_tool_calls


def block(obj):
    return "<tool_call>\n" + json.dumps(obj, ensure_ascii=False) + "\n</tool_call>"


def parallel_case(expect_calls, category="parallel_same_tool"):
    return {"id": "pc", "category": category,
            "expect": {"mode": "tool",
                       "calls": [{"name": n, "arguments": a} for n, a in expect_calls]}}


def test_qwen_parallel_renders_multiple_blocks():
    # Qwen3 template emits one <tool_call> block per call, NOT a JSON array
    raw = (
        block({"name": "get_stock_price", "arguments": {"symbol": "AAA"}})
        + block({"name": "get_stock_price", "arguments": {"symbol": "BBB"}})
        + "<|im_end|>"
    )
    parsed = parse_tool_calls(raw)
    assert len(parsed.calls) == 2
    assert parsed.all_native
    assert parsed.names == ["get_stock_price", "get_stock_price"]
    assert parsed.calls[0].arguments["symbol"] == "AAA"
    assert parsed.calls[1].arguments["symbol"] == "BBB"


def test_json_array_in_single_block_also_parses():
    raw = "<tool_call>\n" + json.dumps([
        {"name": "f", "arguments": {"x": 1}},
        {"name": "f", "arguments": {"x": 2}},
    ]) + "\n</tool_call><|im_end|>"
    parsed = parse_tool_calls(raw)
    assert len(parsed.calls) == 2
    assert parsed.names == ["f", "f"]


def test_same_name_parallel_correct_order():
    raw = (
        block({"name": "get_stock_price", "arguments": {"symbol": "AAA"}})
        + block({"name": "get_stock_price", "arguments": {"symbol": "BBB"}})
        + "<|im_end|>"
    )
    expect = (("get_stock_price", {"symbol": "AAA"}),
              ("get_stock_price", {"symbol": "BBB"}))
    sc = score_case(parallel_case(expect), parse_tool_calls(raw), raw)
    assert sc.tool_selection_correct and sc.arg_exact_match and sc.overall
    assert sc.strict_protocol_success


def test_same_name_parallel_swapped_order_still_matches():
    raw = (
        block({"name": "get_stock_price", "arguments": {"symbol": "BBB"}})
        + block({"name": "get_stock_price", "arguments": {"symbol": "AAA"}})
        + "<|im_end|>"
    )
    expect = (("get_stock_price", {"symbol": "AAA"}),
              ("get_stock_price", {"symbol": "BBB"}))
    sc = score_case(parallel_case(expect), parse_tool_calls(raw), raw)
    assert sc.tool_selection_correct and sc.arg_exact_match and sc.overall


def test_same_name_one_wrong_argument_fails():
    raw = (
        block({"name": "get_stock_price", "arguments": {"symbol": "AAA"}})
        + block({"name": "get_stock_price", "arguments": {"symbol": "CCC"}})
        + "<|im_end|>"
    )
    expect = (("get_stock_price", {"symbol": "AAA"}),
              ("get_stock_price", {"symbol": "BBB"}))
    sc = score_case(parallel_case(expect), parse_tool_calls(raw), raw)
    assert sc.tool_selection_correct  # names right
    assert not sc.arg_exact_match and not sc.overall


def test_missing_one_parallel_call_fails():
    raw = block({"name": "get_stock_price", "arguments": {"symbol": "AAA"}}) + "<|im_end|>"
    expect = (("get_stock_price", {"symbol": "AAA"}),
              ("get_stock_price", {"symbol": "BBB"}))
    sc = score_case(parallel_case(expect), parse_tool_calls(raw), raw)
    assert not sc.tool_selection_correct and sc.wrong_tool


def test_extra_parallel_call_fails():
    raw = (
        block({"name": "get_stock_price", "arguments": {"symbol": "AAA"}})
        + block({"name": "get_stock_price", "arguments": {"symbol": "BBB"}})
        + block({"name": "get_stock_price", "arguments": {"symbol": "CCC"}})
        + "<|im_end|>"
    )
    expect = (("get_stock_price", {"symbol": "AAA"}),
              ("get_stock_price", {"symbol": "BBB"}))
    sc = score_case(parallel_case(expect), parse_tool_calls(raw), raw)
    assert not sc.tool_selection_correct


def test_parallel_different_tools_category_aggregates():
    raw = (
        block({"name": "get_weather", "arguments": {"city": "北京"}})
        + block({"name": "send_email", "arguments": {"to": "a@b.com"}})
        + "<|im_end|>"
    )
    expect = (("get_weather", {"city": "北京"}),
              ("send_email", {"to": "a@b.com"}))
    case = parallel_case(expect, category="parallel_diff_tool")
    sc = score_case(case, parse_tool_calls(raw), raw)
    assert sc.overall
    m = aggregate([sc])
    cat = m["by_category"]["parallel_diff_tool"]
    assert cat["n"] == 1 and cat["strict_protocol_rate"] == 1.0
