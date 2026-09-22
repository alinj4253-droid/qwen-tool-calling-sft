from qwen_tool_sft.metrics import aggregate, score_case
from qwen_tool_sft.parser import parse_tool_calls


def tool_case(cid="x", calls=None):
    return {"id": cid, "category": "single_tool",
            "expect": {"mode": "tool", "calls": calls or [
                {"name": "get_weather", "arguments": {"city": "Tokyo"}}]}}


def no_tool_case(cid="n"):
    return {"id": cid, "category": "no_tool", "expect": {"mode": "no_tool", "calls": []}}


def test_perfect_tool_case():
    out = '<tool_call>{"name": "get_weather", "arguments": {"city": "tokyo"}}</tool_call>'
    sc = score_case(tool_case(), parse_tool_calls(out))
    assert sc.valid_format
    assert sc.tool_selection_correct
    assert sc.arg_exact_match
    assert sc.overall
    assert sc.arg_key_accuracy == 1.0


def test_numeric_string_equivalence():
    case = tool_case(calls=[{"name": "f", "arguments": {"n": 3}}])
    sc = score_case(case, parse_tool_calls(
        '<tool_call>{"name": "f", "arguments": {"n": "3"}}</tool_call>'))
    assert sc.arg_exact_match


def test_wrong_tool():
    sc = score_case(tool_case(), parse_tool_calls(
        '<tool_call>{"name": "calculator", "arguments": {}}</tool_call>'))
    assert sc.wrong_tool
    assert not sc.tool_selection_correct
    assert not sc.overall


def test_missing_and_extra_args():
    sc = score_case(
        tool_case(calls=[{"name": "get_weather",
                          "arguments": {"city": "Tokyo", "date": "2026-10-01"}}]),
        parse_tool_calls(
            '<tool_call>{"name": "get_weather", "arguments": {"city": "Tokyo", "unit": "celsius"}}</tool_call>'))
    assert sc.missing_argument and sc.extra_argument
    assert sc.arg_key_accuracy == 0.5
    assert not sc.arg_exact_match


def test_no_tool_correct_and_violation():
    ok = score_case(no_tool_case(), parse_tool_calls("I can't browse, but here is advice."))
    assert ok.no_tool_correct and ok.overall
    bad = score_case(no_tool_case(), parse_tool_calls(
        '<tool_call>{"name": "get_weather", "arguments": {}}</tool_call>'))
    assert not bad.no_tool_correct
    assert bad.wrong_tool


def test_parallel_calls_order_insensitive():
    case = {"id": "p", "category": "parallel_calls", "expect": {"mode": "tool", "calls": [
        {"name": "get_stock_price", "arguments": {"symbol": "AAPL"}},
        {"name": "get_stock_price", "arguments": {"symbol": "MSFT"}}]}}
    out = ('<tool_call>[{"name": "get_stock_price", "arguments": {"symbol": "MSFT"}}, '
           '{"name": "get_stock_price", "arguments": {"symbol": "AAPL"}}]</tool_call>')
    sc = score_case(case, parse_tool_calls(out))
    assert sc.tool_selection_correct
    assert sc.overall


def test_aggregate_rates():
    cases = [tool_case("a"), tool_case("b"), no_tool_case()]
    outs = [
        '<tool_call>{"name": "get_weather", "arguments": {"city": "Tokyo"}}</tool_call>',
        '<tool_call>{"name": "other", "arguments": {}}</tool_call>',
        "plain answer",
    ]
    scores = [score_case(c, parse_tool_calls(o)) for c, o in zip(cases, outs)]
    m = aggregate(scores)
    assert m["tool_selection_accuracy"] == 0.5
    assert m["no_tool_accuracy"] == 1.0
    assert 0 <= m["overall_exact_match"] <= 1
