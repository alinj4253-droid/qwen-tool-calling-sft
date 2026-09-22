"""Dedup must key on BOTH messages and tools (P0-2 regression)."""
from __future__ import annotations

from qwen_tool_sft.dedup import (
    canonical_json, dedup_samples, messages_tools_fingerprint, sample_fingerprint,
)


def _sample(tools=None):
    return {
        "messages": [
            {"role": "user", "content": "查一下北京天气"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"type": "function",
                 "function": {"name": "get_weather", "arguments": {"city": "北京"}}}]},
        ],
        "tools": tools,
    }


WEATHER_TOOL = [{"type": "function", "function": {
    "name": "get_weather",
    "parameters": {"type": "object",
                   "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]

EMAIL_TOOL = [{"type": "function", "function": {
    "name": "send_email",
    "parameters": {"type": "object",
                   "properties": {"to": {"type": "string"}}, "required": ["to"]}}}]


def test_identical_samples_are_duplicates():
    a, b = _sample(WEATHER_TOOL), _sample(WEATHER_TOOL)
    assert sample_fingerprint(a) == sample_fingerprint(b)
    out, n = dedup_samples([a, b])
    assert n == 1 and len(out) == 1


def test_same_messages_different_tools_are_NOT_duplicates():
    a = _sample(WEATHER_TOOL)
    b = _sample(EMAIL_TOOL)
    assert sample_fingerprint(a) != sample_fingerprint(b)
    out, n = dedup_samples([a, b])
    assert n == 0 and len(out) == 2


def test_same_messages_tools_vs_none_are_different():
    a = _sample(WEATHER_TOOL)
    b = _sample(None)
    assert messages_tools_fingerprint(a["messages"], a["tools"]) != \
        messages_tools_fingerprint(b["messages"], b["tools"])


def test_json_key_order_invariant():
    t1 = [{"type": "function", "function": {
        "name": "f", "parameters": {"a": 1, "b": 2}}}]
    t2 = [{"function": {"parameters": {"b": 2, "a": 1}, "name": "f"},
           "type": "function"}]
    assert canonical_json(t1) == canonical_json(t2)


def test_whitespace_normalization_invariant():
    a = _sample(WEATHER_TOOL)
    b = _sample(WEATHER_TOOL)
    a["messages"][0]["content"] = "check  the weather"   # double space
    b["messages"][0]["content"] = " check the weather "  # leading/trailing
    assert sample_fingerprint(a) == sample_fingerprint(b)


def test_argument_value_order_matters_in_lists():
    # message ORDER is meaningful and must not be collapsed
    a = _sample(WEATHER_TOOL)
    b = _sample(WEATHER_TOOL)
    b["messages"] = list(reversed(b["messages"]))
    assert sample_fingerprint(a) != sample_fingerprint(b)
