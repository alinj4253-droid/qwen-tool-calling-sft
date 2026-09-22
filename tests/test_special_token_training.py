"""Selective special-token training config (P1): PEFT trainable_token_indices.

The v1 modules_to_save=["embed_tokens","lm_head"] scheme trains ~811M params
(16.8%) and writes a ~1.7GB adapter. PEFT's trainable_token_indices trains
only the 6 untrained chat/tool token rows (~33M params, ~148MB adapter).
"""
from __future__ import annotations

import pytest

peft = pytest.importorskip("peft")
import torch.nn as nn  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
SPECIAL_TOKENS = ["<|im_start|>", "<|im_end|>", "<tool_call>", "</tool_call>",
                  "<tool_response>", "</tool_response>"]


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(100, 8)
        self.lm_head = nn.Linear(8, 100)

    def forward(self, x):
        return self.lm_head(self.embed_tokens(x))


def test_selective_lora_config_builds():
    cfg = LoraConfig(
        r=32, lora_alpha=64, target_modules=TARGET_MODULES,
        task_type="CAUSAL_LM", modules_to_save=None,
        trainable_token_indices=[151644, 151645, 151657, 151658, 151665, 151666],
    )
    assert cfg.trainable_token_indices == [151644, 151645, 151657, 151658, 151665, 151666]


def test_selective_indices_attach_trainable_token_params():
    model = get_peft_model(
        _TinyModel(),
        LoraConfig(r=8, target_modules=["lm_head"], modules_to_save=None,
                   trainable_token_indices=[1, 2, 3]))
    names = [n for n, _ in model.named_parameters() if "trainable_tokens" in n]
    assert names, "expected TrainableTokensWrapper delta parameters"


def test_indices_conflict_with_modules_to_save_raises():
    # PEFT forbids training token rows while also fully saving embeddings
    with pytest.raises(ValueError):
        get_peft_model(
            _TinyModel(),
            LoraConfig(r=8, target_modules=["lm_head"],
                       modules_to_save=["embed_tokens"],
                       trainable_token_indices=[1, 2]))


def test_special_tokens_resolve_to_distinct_ids(tokenizer):
    ids = []
    for t in SPECIAL_TOKENS:
        tid = tokenizer.convert_tokens_to_ids(t)
        assert isinstance(tid, int) and tid >= 0, t
        ids.append(tid)
    assert len(set(ids)) == len(SPECIAL_TOKENS)
    # eos <|endoftext|> is a separate, already-trained token
    eos = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    assert eos not in ids


def test_base_template_uses_all_six_protocol_tokens(tokenizer):
    from qwen_tool_sft.dataio import normalize_for_template
    messages = normalize_for_template([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"type": "function",
             "function": {"name": "f", "arguments": {"x": 1}}}]},
        {"role": "tool", "name": "f", "content": "r"},
        {"role": "assistant", "content": "ok"}])
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=[{"type": "function", "function": {
            "name": "f",
            "parameters": {"type": "object",
                           "properties": {"x": {"type": "integer"}},
                           "required": ["x"]}}}],
        tokenize=False, add_generation_prompt=False)
    for t in SPECIAL_TOKENS:
        assert t in rendered, t
