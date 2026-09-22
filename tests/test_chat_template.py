"""Integration: the downloaded tokenizer must render the Qwen3 tool-calling
chat template (skipped until the model has been downloaded)."""
from __future__ import annotations

import pytest

from qwen_tool_sft.dataio import normalize_for_template, render_text


def test_chat_template_with_tools(tokenizer):
    if tokenizer is None:
        pytest.skip("base model tokenizer not downloaded yet")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]

    # 1) prompt side: tools are exposed to the model
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "北京今天天气怎么样？"}],
        tools=tools,
        add_generation_prompt=True,
        tokenize=False,
    )
    assert "get_weather" in prompt
    assert "<tools>" in prompt and "</tools>" in prompt
    assert "<|im_start|>assistant" in prompt

    # 2) no-tools conversation must not leak tool scaffolding
    plain = tokenizer.apply_chat_template(
        [{"role": "user", "content": "你好"}],
        add_generation_prompt=True,
        tokenize=False,
    )
    assert "<tools>" not in plain

    # 3) completion side: assistant tool-call turn with content=None (upstream
    #    convention) must render after our normalization, not crash the
    #    Qwen3 template (`</think> in message.content`).
    messages = [
        {"role": "user", "content": "北京今天天气怎么样？"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "北京"},
                    },
                }
            ],
        },
        {"role": "tool", "name": "get_weather", "content": "晴，25度"},
    ]
    rendered = render_text(tokenizer, messages, tools)
    assert "<tool_call>" in rendered
    assert "get_weather" in rendered
    assert "北京" in rendered
    assert "<tool_response>" in rendered
    # normalization must never emit python None into the text
    assert "None" not in rendered.split("<tool_call>")[1].split("</tool_call>")[0]


def test_normalize_fills_none_assistant_content():
    msgs = [
        {"role": "assistant", "content": None, "tool_calls": [
            {"type": "function",
             "function": {"name": "f", "arguments": {"a": 1}}}]},
        {"role": "assistant", "content": None},
    ]
    out = normalize_for_template(msgs)
    assert out[0]["content"] == ""
    assert out[1]["content"] == ""
