"""JSONL IO, sample validation and chat-template rendering."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

LEGAL_ROLES = {"system", "user", "assistant", "tool"}


def read_jsonl(path: str | Path) -> list[dict]:
    out: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)  # let JSON errors surface with line info
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{ln} is not a JSON object")
            out.append(obj)
    return out


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_tool_def(tool: Any) -> bool:
    if not isinstance(tool, dict):
        return False
    if tool.get("type") != "function":
        return False
    func = tool.get("function")
    if not isinstance(func, dict) or not isinstance(func.get("name"), str) or not func["name"]:
        return False
    params = func.get("parameters", {})
    return isinstance(params, dict)


def validate_tool_call(tc: Any) -> bool:
    if not isinstance(tc, dict) or tc.get("type") != "function":
        return False
    func = tc.get("function")
    if not isinstance(func, dict) or not isinstance(func.get("name"), str) or not func["name"]:
        return False
    args = func.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return False
    return isinstance(args, dict)


def validate_sample(sample: dict) -> tuple[bool, str]:
    """Validate one training/eval sample. Returns (ok, reason)."""
    msgs = sample.get("messages")
    if not isinstance(msgs, list) or len(msgs) < 2:
        return False, "messages missing or <2"
    roles = [m.get("role") for m in msgs]
    if any(r not in LEGAL_ROLES for r in roles):
        return False, f"illegal role in {roles}"
    if "user" not in roles or "assistant" not in roles:
        return False, "needs user and assistant"
    tools = sample.get("tools")
    if tools is not None:
        if not isinstance(tools, list) or not all(validate_tool_def(t) for t in tools):
            return False, "invalid tools definitions"
    for m in msgs:
        if m.get("role") == "assistant":
            content = m.get("content")
            tcs = m.get("tool_calls")
            has_content = content is not None and content != ""
            if not has_content and not tcs:
                return False, "assistant with neither content nor tool_calls"
            if tcs is not None:
                if not isinstance(tcs, list) or not tcs:
                    return False, "tool_calls must be non-empty list"
                if not all(validate_tool_call(tc) for tc in tcs):
                    return False, "invalid tool_calls"
        if m.get("role") == "tool":
            if not isinstance(m.get("content"), str):
                return False, "tool message content must be str"
    return True, ""


def normalize_for_template(messages: list[dict]) -> list[dict]:
    """Return a cleaned copy safe for tokenizer.apply_chat_template.

    Mirrors upstream logic: arguments must be dicts; assistant tool-call
    turns use empty-string content (None breaks the Qwen3 template's
    `'</think>' in message.content` guard).
    """
    cleaned: list[dict] = []
    for m in messages:
        msg: dict[str, Any] = {"role": m["role"], "content": m.get("content")}
        if m.get("tool_calls"):
            tcs = []
            for tc in m["tool_calls"]:
                func = tc.get("function", tc)
                args = func.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                if not isinstance(args, dict):
                    args = {"value": args}
                tcs.append({
                    "type": "function",
                    "function": {"name": func.get("name", ""), "arguments": args},
                })
            msg["tool_calls"] = tcs
        # The official Qwen3 chat template runs `'</think>' in message.content`
        # on every assistant turn, so assistant content must never be None
        # (canonical tool-call turns use an empty string).
        if msg["role"] == "assistant" and msg.get("content") is None:
            msg["content"] = ""
        if m.get("name"):
            msg["name"] = m["name"]
        cleaned.append(msg)
    return cleaned


def render_text(tokenizer, messages: list[dict], tools: list[dict] | None) -> str:
    return tokenizer.apply_chat_template(
        normalize_for_template(messages),
        tools=tools if tools else None,
        tokenize=False,
        add_generation_prompt=False,
    )
