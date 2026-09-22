from qwen_tool_sft.parser import parse_tool_calls


def test_native_single_call():
    text = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Beijing", "date": "2026-10-01"}}\n</tool_call>'
    r = parse_tool_calls(text)
    assert r.names == ["get_weather"]
    assert r.calls[0].arguments["city"] == "Beijing"
    assert r.calls[0].valid_json


def test_native_parallel_calls_array():
    text = ('<tool_call>\n[{"name": "get_stock_price", "arguments": {"symbol": "AAPL"}}, '
            '{"name": "get_stock_price", "arguments": {"symbol": "MSFT"}}]\n</tool_call>')
    r = parse_tool_calls(text)
    assert r.names == ["get_stock_price", "get_stock_price"]
    assert [c.arguments["symbol"] for c in r.calls] == ["AAPL", "MSFT"]


def test_openai_function_shape():
    text = ('<tool_call>\n{"type": "function", "function": {"name": "calculator", '
            '"arguments": {"expression": "1+1"}}}\n</tool_call>')
    r = parse_tool_calls(text)
    assert r.names == ["calculator"]
    assert r.calls[0].arguments == {"expression": "1+1"}


def test_fenced_json():
    text = '```json\n{"name": "search_web", "arguments": {"query": "q"}}\n```'
    r = parse_tool_calls(text)
    assert r.names == ["search_web"]


def test_string_arguments_get_parsed():
    text = '<tool_call>\n{"name": "f", "arguments": "{\\"k\\": 3}"}\n</tool_call>'
    r = parse_tool_calls(text)
    assert r.calls[0].arguments == {"k": 3}


def test_truncated_call_marked_invalid():
    text = '<tool_call>\n{"name": "f", "arguments": {"k": 3'
    r = parse_tool_calls(text)
    assert r.calls == []
    assert r.invalid_blocks


def test_plain_text_no_calls():
    r = parse_tool_calls("Sorry, I cannot help with that.")
    assert not r.has_calls
    assert r.names == []


def test_trailing_comma_tolerated():
    text = '<tool_call>\n{"name": "f", "arguments": {"a": 1,}}\n</tool_call>'
    r = parse_tool_calls(text)
    assert r.names == ["f"]
    assert r.calls[0].arguments == {"a": 1}


def test_multiple_native_blocks():
    text = ('<tool_call>\n{"name": "get_weather", "arguments": {"city": "北京"}}\n</tool_call>\n'
            '<tool_call>\n{"name": "get_weather", "arguments": {"city": "上海"}}\n</tool_call>')
    r = parse_tool_calls(text)
    assert r.names == ["get_weather", "get_weather"]
