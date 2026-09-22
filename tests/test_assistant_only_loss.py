"""Assistant-only loss mask: rendering invariance + mask correctness."""
from __future__ import annotations

import pytest

from qwen_tool_sft.loss_mask import (
    AssistantOnlyCollator, audit_masks, patch_chat_template,
    render_identical, tokenize_assistant_only,
)


def _tool(name, props=None, req=None):
    return {"type": "function", "function": {
        "name": name, "description": f"UNIQUE_DESC_{name}",
        "parameters": {"type": "object",
                       "properties": props or {"q": {"type": "string"}},
                       "required": req or list((props or {"q": 1}).keys())}}}


def _call(name, args):
    return {"type": "function",
            "function": {"name": name, "arguments": args}}


PLAIN = ([{"role": "user", "content": "你好"},
          {"role": "assistant", "content": "你好，请问需要什么帮助？"}], None)

SINGLE = (
    [{"role": "system", "content": "SYS_PROMPT_X"},
     {"role": "user", "content": "QUERY_WEATHER_BJ"},
     {"role": "assistant", "content": None,
      "tool_calls": [_call("get_weather", {"city": "Beijing"})]},
     {"role": "tool", "name": "get_weather", "content": "TOOL_RESULT_SUNNY"},
     {"role": "assistant", "content": "FINAL_ANSWER_SUNNY"}],
    [_tool("get_weather", {"city": {"type": "string"}}, ["city"])],
)

PARALLEL = (
    [{"role": "user", "content": "QUERY_THREE_CITIES"},
     {"role": "assistant", "content": None, "tool_calls": [
         _call("get_weather", {"city": "Beijing"}),
         _call("get_weather", {"city": "Shanghai"}),
         _call("get_weather", {"city": "Guangzhou"})]}],
    [_tool("get_weather", {"city": {"type": "string"}}, ["city"])],
)

MULTI_TURN = (
    [{"role": "user", "content": "Q_ONE"}, {"role": "assistant", "content": "A_ONE"},
     {"role": "user", "content": "Q_TWO"}, {"role": "assistant", "content": "A_TWO"}],
    None,
)

NO_TOOL_AVOID = (
    [{"role": "user", "content": "GREETING_USER"},
     {"role": "assistant", "content": "GREETING_ASSISTANT"}],
    [_tool("get_weather")],
)


def test_base_template_lacks_generation_markers(tokenizer):
    assert "{% generation" not in tokenizer.chat_template
    patched = patch_chat_template(tokenizer.chat_template)
    assert patched.count("{%- generation %}") == 1
    assert patched.count("{%- endgeneration %}") == 1


@pytest.mark.parametrize("case", [PLAIN, SINGLE, PARALLEL, MULTI_TURN, NO_TOOL_AVOID])
def test_patched_template_renders_byte_identical(tokenizer, case):
    messages, tools = case
    patched = patch_chat_template(tokenizer.chat_template)
    assert render_identical(tokenizer, messages, tools, patched), (
        "patched template changed rendered text")


def test_plain_text_mask(tokenizer):
    messages, _ = PLAIN
    tok = tokenize_assistant_only(tokenizer, messages, None)
    loss_text = tokenizer.decode(
        [i for i, m in zip(tok["input_ids"], tok["assistant_masks"]) if m])
    masked_text = tokenizer.decode(
        [i for i, m in zip(tok["input_ids"], tok["assistant_masks"]) if not m])
    assert "你好，请问需要什么帮助？" in loss_text
    assert "你好" in masked_text  # user query masked (also appears in no answer)
    # final supervised token of an assistant turn must be <|im_end|> (151645);
    # the separating newline after it is intentionally outside the loss.
    last_labeled = max(i for i, x in enumerate(tok["labels"]) if x != -100)
    assert tok["input_ids"][last_labeled] == 151645
    assert tok["labels"][-1] == -100  # trailing newline masked


def test_single_tool_call_mask(tokenizer):
    messages, tools = SINGLE
    tok = tokenize_assistant_only(tokenizer, messages, tools)
    loss_text = tokenizer.decode(
        [i for i, m in zip(tok["input_ids"], tok["assistant_masks"]) if m])
    masked_text = tokenizer.decode(
        [i for i, m in zip(tok["input_ids"], tok["assistant_masks"]) if not m])
    # tool call JSON + final answer supervised
    assert "get_weather" in loss_text
    assert "Beijing" in loss_text
    assert "FINAL_ANSWER_SUNNY" in loss_text
    # system / user / schema / tool result all masked
    assert "SYS_PROMPT_X" in masked_text
    assert "QUERY_WEATHER_BJ" in masked_text
    assert "UNIQUE_DESC_get_weather" in masked_text
    assert "TOOL_RESULT_SUNNY" in masked_text
    # schema description must never appear in the loss
    assert "UNIQUE_DESC_get_weather" not in loss_text


def test_parallel_three_calls_all_supervised(tokenizer):
    messages, tools = PARALLEL
    tok = tokenize_assistant_only(tokenizer, messages, tools)
    loss_text = tokenizer.decode(
        [i for i, m in zip(tok["input_ids"], tok["assistant_masks"]) if m])
    for city in ("Beijing", "Shanghai", "Guangzhou"):
        assert city in loss_text
    assert "QUERY_THREE_CITIES" not in loss_text
    # Qwen renders parallel calls as multiple native blocks: all three present
    assert loss_text.count("<tool_call>") == 3 or loss_text.count("get_weather") >= 3


def test_multi_turn_both_assistant_turns_supervised(tokenizer):
    messages, _ = MULTI_TURN
    tok = tokenize_assistant_only(tokenizer, messages, None)
    loss_text = tokenizer.decode(
        [i for i, m in zip(tok["input_ids"], tok["assistant_masks"]) if m])
    assert "A_ONE" in loss_text and "A_TWO" in loss_text


def test_no_tool_case_with_tools_available_masks_schema(tokenizer):
    messages, tools = NO_TOOL_AVOID
    r = audit_masks(tokenizer, messages, tools)
    assert r["render_identical_to_original"]
    assert "GREETING_ASSISTANT" in r["loss_text"]
    assert "UNIQUE_DESC_get_weather" in r["masked_text"]
    assert "UNIQUE_DESC_get_weather" not in r["loss_text"]


def test_collator_right_pads():
    import torch
    collator = AssistantOnlyCollator(pad_token_id=0)
    batch = collator([
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1],
         "labels": [-100, 2, 3]},
        {"input_ids": [4, 5], "attention_mask": [1, 1], "labels": [4, 5]},
    ])
    assert batch["input_ids"].shape == (2, 3)
    assert batch["attention_mask"][1].tolist() == [1, 1, 0]
    assert batch["labels"][1].tolist() == [4, 5, -100]
    assert batch["labels"][0].tolist() == [-100, 2, 3]


def test_patch_idempotent_on_fixture(tokenizer):
    once = patch_chat_template(tokenizer.chat_template)
    twice = patch_chat_template(once)
    assert once == twice
