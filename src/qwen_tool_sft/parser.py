"""Robust parser that extracts function/tool calls from a model generation.

Handles Qwen3 native format (<tool_call>...</tool_call>), fenced JSON blocks,
bare JSON objects/arrays, truncated JSON and common formatting mistakes.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"


@dataclass
class ParsedCall:
    name: str
    arguments: dict
    valid_json: bool = True
    raw: str = ""


@dataclass
class ParseResult:
    calls: list[ParsedCall] = field(default_factory=list)
    invalid_blocks: list[str] = field(default_factory=list)

    @property
    def has_calls(self) -> bool:
        return bool(self.calls)

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.calls]


def _loose_json_loads(text: str):
    text = text.strip()
    text = re.sub(r",\s*([}\]])", r"\1", text)  # trailing commas
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # extract the first balanced {...} block
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    cand = text[start:i + 1]
                    try:
                        return json.loads(cand)
                    except json.JSONDecodeError:
                        return None
    return None


def _normalize_call_obj(obj) -> ParsedCall | None:
    if not isinstance(obj, dict):
        return None
    if "function" in obj and isinstance(obj["function"], dict):
        func = obj["function"]
        name = func.get("name") or obj.get("name")
        args = func.get("arguments", {})
    else:
        name = obj.get("name")
        args = obj.get("arguments", {})
    if not name or not isinstance(name, str):
        return None
    valid = True
    if isinstance(args, str):
        parsed = _loose_json_loads(args)
        if parsed is None or not isinstance(parsed, dict):
            valid = False
            args = {}
        else:
            args = parsed
    if not isinstance(args, dict):
        valid = False
        args = {"value": args}
    return ParsedCall(name=name, arguments=args, valid_json=valid, raw=json.dumps(obj, ensure_ascii=False))


def _parse_block(block: str) -> tuple[list[ParsedCall], bool]:
    """Return (calls, json_valid)."""
    block = block.strip()
    # strip ```json fences if present
    block = re.sub(r"^```(?:json)?\s*", "", block)
    block = re.sub(r"\s*```$", "", block).strip()
    try:
        obj = json.loads(block)
        objs = obj if isinstance(obj, list) else [obj]
        calls = [c for c in (_normalize_call_obj(o) for o in objs) if c]
        if calls:
            return calls, True
    except json.JSONDecodeError:
        pass
    obj = _loose_json_loads(block)
    if obj is not None:
        objs = obj if isinstance(obj, list) else [obj]
        calls = [c for c in (_normalize_call_obj(o) for o in objs) if c]
        if calls:
            return calls, all(c.valid_json for c in calls)
    return [], False


def parse_tool_calls(text: str) -> ParseResult:
    result = ParseResult()
    if not text:
        return result

    # 1) native <tool_call> blocks (handle unclosed/truncated)
    pattern = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.DOTALL)
    blocks = pattern.findall(text)
    for block in blocks:
        if not block.strip():
            continue
        calls, ok = _parse_block(block)
        if calls:
            result.calls.extend(calls)
            if not ok:
                result.invalid_blocks.append(block)
        else:
            result.invalid_blocks.append(block)

    if result.calls or result.invalid_blocks:
        return result

    # 2) fenced ```json blocks
    for m in re.finditer(r"```(?:json)?\s*(.*?)```", text, re.DOTALL):
        calls, ok = _parse_block(m.group(1))
        if calls:
            result.calls.extend(calls)
            if not ok:
                result.invalid_blocks.append(m.group(1))
    if result.calls:
        return result

    # 3) bare JSON object mentioning a function name/arguments
    bare = _loose_json_loads(text)
    if bare is not None:
        objs = bare if isinstance(bare, list) else [bare]
        for o in objs:
            if isinstance(o, dict) and ("name" in o or "function" in o):
                c = _normalize_call_obj(o)
                if c:
                    result.calls.append(c)
    return result
