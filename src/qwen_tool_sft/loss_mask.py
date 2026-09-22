"""Assistant-only loss masking for the Qwen3 tool-calling chat template.

The Qwen3-4B-Base tokenizer ships a chat template WITHOUT Jinja ``generation``
markers, so TRL's ``assistant_only_loss`` cannot locate the supervised spans
(it raises "the chat_template must contain {% generation %}").

``patch_chat_template`` injects ``{% generation %}`` / ``{% endgeneration %}``
markers around every assistant turn (the optional thinking scaffold, the
assistant text, the native ``<tool_call>`` blocks and the closing
``<|im_end|>``).  The injection is *rendering-preserving*: for any
conversation the patched template renders byte-identical text to the
original (asserted by tests/test_assistant_only_loss.py over plain text,
single/multiple/parallel tool calls, multi-turn and no-tool cases).

``tokenize_assistant_only`` then asks the tokenizer for the
``assistant_masks`` and turns non-assistant tokens into ``labels == -100``.
The training script uses ``AssistantOnlyCollator`` so that:

  system prompt / user query / tool schema / tool responses -> condition only
  assistant text + tool-call JSON + <|im_end|>              -> contribute loss
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .dataio import normalize_for_template

GEN_OPEN = "{%- generation %}"
GEN_CLOSE = "{%- endgeneration %}"


def patch_chat_template(template: str) -> str:
    """Return a copy of the Qwen3 Jinja chat template with generation markers.

    The assistant branch is restructured from::

        {{- '<|im_start|>assistant\\n' + <body> }}
        ...tool calls...
        {{- '<|im_end|>\\n' }}

    into::

        {{- '<|im_start|>assistant\\n' }}
        {%- generation %}
        {{- <body> }}
        ...tool calls...
        {{- '<|im_end|>' }}
        {%- endgeneration %}
        {{- '\\n' }}

    Hard assertions fail loudly if the upstream template changes layout.
    """
    if not isinstance(template, str) or "assistant" not in template:
        raise ValueError("patch_chat_template expects a Qwen-style chat template string")
    if "{% generation" in template or "{%- generation" in template:
        return template  # already patched / already supports generation masks

    lines = template.split("\n")

    def find(pred: Any, lo: int = 0) -> int:
        for i in range(lo, len(lines)):
            if pred(lines[i]):
                return i
        raise ValueError("chat template layout not recognized: anchor line missing")

    a_start = find(lambda l: 'message.role == "assistant"' in l)
    a_end = find(lambda l: 'message.role == "tool"' in l, a_start)
    l_if1 = find(lambda l: "if loop.index0 > ns.last_query_index" in l, a_start)
    l_scaf = find(lambda l: "reasoning_content.strip(" in l and "message.role" in l, l_if1)
    l_end = a_end - 1  # assistant branch terminates with the <|im_end|> line

    plain_target = "{{- '<|im_start|>' + message.role + '\\n' + content }}"
    plain = [i for i in range(l_if1, l_end) if plain_target in lines[i]]
    assert len(plain) == 2, f"expected 2 plain assistant header lines, got {len(plain)}"
    assert "{{- '<|im_end|>\\n' }}" in lines[l_end], repr(lines[l_end])

    indent = re.match(r"\s*", lines[l_scaf]).group(0)
    out = lines[:]

    # scaffold line: keep the role header on its own, generation region starts
    # with the (possibly empty) thinking scaffold + content.
    s = out[l_scaf]
    anchor = "+ message.role + '\\n"
    j = s.index(anchor) + len(anchor)
    assert s[:j].endswith("+ '\\n"), repr(s[:j])
    rest = s[j:]
    assert rest.endswith(" }}"), repr(rest)
    out[l_scaf] = indent + "{{- '" + rest

    # the two plain "header + content" lines become content-only
    for i in plain:
        out[i] = re.sub(r"\S.*$", "{{- content }}", lines[i])

    # <|im_end|> stays inside the generation region; the separating newline
    # after it does not need to be supervised.
    out[l_end] = (
        indent + "{{- '<|im_end|>' }}\n"
        + indent + GEN_CLOSE + "\n"
        + indent + "{{- '\\n' }}"
    )

    # role header (never supervised) followed by the generation region open.
    out[l_if1:l_if1] = [
        indent + "{{- '<|im_start|>' + message.role + '\\n' }}",
        indent + GEN_OPEN,
    ]

    patched = "\n".join(out)
    assert patched.count(GEN_OPEN) == 1 and patched.count(GEN_CLOSE) == 1
    return patched


def render_identical(tokenizer, messages: list[dict], tools: list[dict] | None,
                     patched_template: str) -> bool:
    """True when patched and original templates render the same text."""
    norm = normalize_for_template(messages)
    a = tokenizer.apply_chat_template(
        norm, tools=tools if tools else None,
        tokenize=False, add_generation_prompt=False)
    b = tokenizer.apply_chat_template(
        norm, tools=tools if tools else None,
        tokenize=False, add_generation_prompt=False, chat_template=patched_template)
    return a == b


def tokenize_assistant_only(tokenizer, messages: list[dict], tools: list[dict] | None,
                            max_length: int | None = None) -> dict:
    """Tokenize one conversation with assistant-only ``labels``.

    Returns ``{"input_ids", "attention_mask", "labels", "assistant_masks"}``
    where labels are -100 for every token outside assistant turns.
    """
    messages = normalize_for_template(messages)
    patched = patch_chat_template(tokenizer.chat_template)
    enc = tokenizer.apply_chat_template(
        messages,
        tools=tools if tools else None,
        tokenize=True,
        return_assistant_tokens_mask=True,
        return_dict=True,
        add_generation_prompt=False,
        chat_template=patched,
    )
    input_ids = list(enc["input_ids"])
    masks = list(enc["assistant_masks"])
    if len(input_ids) != len(masks):
        raise ValueError("assistant mask length mismatch")
    labels = [int(t) if m == 1 else -100 for t, m in zip(input_ids, masks)]
    attention_mask = [1] * len(input_ids)
    truncated = False
    if max_length is not None and len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]
        masks = masks[:max_length]
        truncated = True
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "assistant_masks": masks,
        "truncated": truncated,
    }


@dataclass
class AssistantOnlyCollator:
    """Pad pre-tokenized assistant-only examples (right padding)."""

    pad_token_id: int

    def __call__(self, features: list[dict]) -> dict:
        import torch

        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            n = max_len - len(f["input_ids"])
            input_ids.append(list(f["input_ids"]) + [self.pad_token_id] * n)
            attention_mask.append(list(f["attention_mask"]) + [0] * n)
            labels.append(list(f["labels"]) + [-100] * n)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def audit_masks(tokenizer, messages: list[dict], tools: list[dict] | None,
                max_length: int | None = None) -> dict:
    """Human-readable audit of which tokens are supervised.

    Outputs decoded LOSS / MASKED spans plus token counts so the loss audit
    script can prove system/user/tool-schema tokens are at -100 while
    assistant text and tool calls are trained.
    """
    from .dataio import normalize_for_template

    norm = normalize_for_template(messages)
    patched = patch_chat_template(tokenizer.chat_template)
    identical = render_identical(tokenizer, norm, tools, patched)
    tok = tokenize_assistant_only(tokenizer, norm, tools, max_length)
    ids, masks = tok["input_ids"], tok["assistant_masks"]
    loss_ids = [i for i, m in zip(ids, masks) if m == 1]
    masked_ids = [i for i, m in zip(ids, masks) if m == 0]
    return {
        "render_identical_to_original": identical,
        "n_tokens": len(ids),
        "n_loss_tokens": len(loss_ids),
        "n_masked_tokens": len(masked_ids),
        "loss_fraction": round(len(loss_ids) / len(ids), 4) if ids else 0.0,
        "loss_text": tokenizer.decode(loss_ids),
        "masked_text": tokenizer.decode(masked_ids),
        "truncated": tok["truncated"],
    }
