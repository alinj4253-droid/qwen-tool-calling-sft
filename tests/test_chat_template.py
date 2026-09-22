"""Integration: tokenizer chat template supports tools and tool_calls.

Skips automatically when the base model has not been downloaded yet.
"""
import pytest

from qwen_tool_sft import paths
from qwen_tool_sft.dataio import normalize_for_template
from qwen_tool_sft.parser import parse_tool_calls

MODEL = "Qwen/Qwen3-4B-Base"


def _model_dir():
    d = paths.model_path(MODEL)
    return d if (d / "config.json").exists() else None


def test_chat_template_with_tools():
    d = _model_dir()
    if d is None:
        pytest.skip("base model not downloaded")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(d)

    tools = [{"type": "function", "function": {
        "name": "get_weather",
        "description": "query weather",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string"}},
                       "required": ["city"]}}}]
    messages = [
        {"role": "user", "content": "北京天气怎么样？"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"type": "function",
             "function": {"name": "get_weather", "arguments": {"city": "北京"}}}]},
    ]
    text = tok.apply_chat_template(normalize_for_template(messages), tools=tools,
                                   tokenize=False, add_generation_prompt=False)
    assert "get_weather" in text
    parsed = parse_tool_calls(text)
    assert parsed.names == ["get_weather"]
    assert parsed.calls[0].arguments == {"city": "北京"}

    # generation prompt renders the tool definitions
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "查东京天气"}], tools=tools,
        tokenize=False, add_generation_prompt=True)
    assert "get_weather" in prompt
    ids = tok(prompt, return_tensors="pt")["input_ids"]
    assert ids.shape[1] > 0
