"""Converters for the 7 upstream tool-calling datasets.

Adapted from FuzzyFade/qwen35-tool-calling-sft scripts/prepare_data.py (MIT),
with: streaming support, per-source row caps, no hardcoded local paths.
"""
from __future__ import annotations

import json
import re
import time
from typing import Callable

from datasets import load_dataset


def wrap_tool_openai(tool_def: dict) -> dict:
    if "type" in tool_def and tool_def["type"] == "function" and "function" in tool_def:
        return tool_def
    func: dict = {}
    func["name"] = tool_def.get("name", "unknown")
    func["description"] = tool_def.get("description", "")
    params = tool_def.get("parameters", {})
    if isinstance(params, dict) and "type" in params and "properties" in params:
        func["parameters"] = params
    elif isinstance(params, dict):
        properties, required = {}, []
        for k, v in params.items():
            if isinstance(v, dict):
                properties[k] = v
                if v.get("required", False):
                    required.append(k)
                    v_copy = dict(v)
                    v_copy.pop("required", None)
                    properties[k] = v_copy
            else:
                properties[k] = {"type": "string", "description": str(v)}
        func["parameters"] = {"type": "object", "properties": properties}
        if required:
            func["parameters"]["required"] = required
    elif isinstance(params, list):
        properties, required = {}, []
        for p in params:
            if isinstance(p, dict):
                name = p.get("name", p.get("parameter", "unknown"))
                properties[name] = {
                    "type": p.get("type", "string"),
                    "description": p.get("description", ""),
                }
                if p.get("required", False):
                    required.append(name)
        func["parameters"] = {"type": "object", "properties": properties}
        if required:
            func["parameters"]["required"] = required
    else:
        func["parameters"] = {"type": "object", "properties": {}}
    return {"type": "function", "function": func}


def make_tool_call(name: str, arguments) -> dict:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (json.JSONDecodeError, TypeError):
            arguments = {"raw": arguments}
    if not isinstance(arguments, dict):
        arguments = {"value": arguments}
    return {"type": "function", "function": {"name": name, "arguments": arguments}}


def validate_sample(sample: dict) -> bool:
    msgs = sample.get("messages", [])
    if not msgs or len(msgs) < 2:
        return False
    roles = [m.get("role") for m in msgs]
    if "user" not in roles or "assistant" not in roles:
        return False
    for m in msgs:
        if m.get("role") == "assistant":
            has_content = m.get("content") is not None and m.get("content") != ""
            has_tools = bool(m.get("tool_calls"))
            if not has_content and not has_tools:
                return False
    return True


def _load(repo: str, split: str, streaming: bool, attempts: int = 3, **kwargs):
    """load_dataset with light retries (HF mirror can be flaky).

    datasets>=3 removed ``trust_remote_code``; pass only supported kwargs.
    """
    last_exc = None
    for i in range(attempts):
        try:
            return load_dataset(repo, split=split, streaming=streaming, **kwargs)
        except TypeError as e:
            if "trust_remote_code" in str(e):  # pragma: no cover
                return load_dataset(repo, split=split, streaming=streaming, **kwargs)
            last_exc = e
        except Exception as e:  # network / mirror hiccup -> retry
            last_exc = e
            time.sleep(3 * (i + 1))
    raise last_exc


def _cap_iter(rows, cap: int | None):
    if cap is None:
        for r in rows:
            yield r
    else:
        for i, r in enumerate(rows):
            if i >= cap:
                return
            yield r


# ---------------------------------------------------------------- 1 Deepexi
def convert_deepexi(cap: int | None = None, streaming: bool = True) -> list:
    print("[1/7] Deepexi/function-calling-small ...")
    ds = _load("Deepexi/function-calling-small", "train", streaming)
    results = []
    for row in _cap_iter(ds, cap):
        try:
            system_prompt = row.get("systemPrompt", "") or ""
            user_prompt = row.get("userPrompt", "") or ""
            assistant_resp = row.get("assistantResponse", "") or ""
            if not user_prompt or not assistant_resp:
                continue
            tools = []
            found = re.findall(r'\{[^{}]*"function"[^{}]*"description"[^{}]*\}', system_prompt)
            for t in found:
                try:
                    tools.append(wrap_tool_openai(json.loads(t)))
                except json.JSONDecodeError:
                    pass
            if not tools:
                arrays = re.findall(r'\[(\{[^[\]]*\}(?:,\s*\{[^[\]]*\})*)\]', system_prompt)
                for arr_str in arrays:
                    try:
                        arr = json.loads(f"[{arr_str}]")
                        for item in arr:
                            if isinstance(item, dict) and ("function" in item or "name" in item):
                                tools.append(wrap_tool_openai(item))
                    except json.JSONDecodeError:
                        pass
            tool_calls = []
            try:
                resp = json.loads(assistant_resp)
                if isinstance(resp, dict):
                    args = resp.get("arguments", {})
                    if isinstance(args, list) and args:
                        args = args[0] if isinstance(args[0], dict) else {"args": args}
                    elif isinstance(args, list):
                        args = {}
                    tool_calls.append(make_tool_call(resp.get("function", "unknown"), args))
                elif isinstance(resp, list):
                    for r in resp:
                        if isinstance(r, dict) and "function" in r:
                            args = r.get("arguments", {})
                            if isinstance(args, list) and args:
                                args = args[0] if isinstance(args[0], dict) else {"args": args}
                            tool_calls.append(make_tool_call(r["function"], args))
            except json.JSONDecodeError:
                pass
            messages = []
            if tools:
                messages.append({"role": "system", "content": "你是一个有用的助手，可以调用工具来帮助用户完成任务。"})
            messages.append({"role": "user", "content": user_prompt})
            if tool_calls:
                messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls})
            else:
                messages.append({"role": "assistant", "content": assistant_resp})
            sample = {"messages": messages}
            if tools:
                sample["tools"] = tools
            if validate_sample(sample):
                results.append(sample)
        except Exception:
            continue
    print(f"  -> {len(results)}")
    return results


# ---------------------------------------------------------------- 2 glaive zh
def _convert_glaive(print_tag: str, repo: str, sys_text: str, cap, streaming) -> list:
    print(print_tag)
    ds = _load(repo, "train", streaming)
    results = []
    for row in _cap_iter(ds, cap):
        try:
            convs = row.get("conversations", [])
            tools_str = row.get("tools", "[]")
            tools = []
            try:
                raw = json.loads(tools_str) if isinstance(tools_str, str) else tools_str
                if isinstance(raw, list):
                    tools = [wrap_tool_openai(t) for t in raw]
            except (json.JSONDecodeError, TypeError):
                pass
            messages = []
            if tools:
                messages.append({"role": "system", "content": sys_text})
            for conv in convs:
                role_from, value = conv.get("from", ""), conv.get("value", "")
                if role_from == "human":
                    messages.append({"role": "user", "content": value})
                elif role_from == "gpt":
                    messages.append({"role": "assistant", "content": value})
                elif role_from == "function_call":
                    try:
                        fc = json.loads(value) if isinstance(value, str) else value
                        tc = make_tool_call(fc.get("name", "unknown"), fc.get("arguments", {}))
                        messages.append({"role": "assistant", "content": None, "tool_calls": [tc]})
                    except (json.JSONDecodeError, TypeError):
                        messages.append({"role": "assistant", "content": value})
                elif role_from == "observation":
                    name = ""
                    for m in reversed(messages):
                        if m.get("role") == "assistant" and m.get("tool_calls"):
                            name = m["tool_calls"][0]["function"]["name"]
                            break
                    messages.append({
                        "role": "tool",
                        "content": value if isinstance(value, str) else json.dumps(value, ensure_ascii=False),
                        "name": name,
                    })
            sample = {"messages": messages}
            if tools:
                sample["tools"] = tools
            if validate_sample(sample):
                results.append(sample)
        except Exception:
            continue
    print(f"  -> {len(results)}")
    return results


def convert_glaive_zh(cap=None, streaming=True):
    return _convert_glaive(
        "[2/7] llamafactory/glaive_toolcall_zh ...",
        "llamafactory/glaive_toolcall_zh",
        "你是一个有用的助手，可以调用工具来帮助用户完成任务。",
        cap, streaming,
    )


def convert_glaive_v2_sharegpt(cap=None, streaming=True):
    return _convert_glaive(
        "[3/7] hiyouga/glaive-function-calling-v2-sharegpt ...",
        "hiyouga/glaive-function-calling-v2-sharegpt",
        "You are a helpful assistant with access to tools.",
        cap, streaming,
    )


# ---------------------------------------------------------------- 4 hermes
def convert_hermes_fc(cap=None, streaming=True) -> list:
    print("[4/7] NousResearch/hermes-function-calling-v1 ...")
    ds = _load("NousResearch/hermes-function-calling-v1", "train", streaming)
    pat = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
    results = []
    for row in _cap_iter(ds, cap):
        try:
            convs = row.get("conversations", [])
            tools_str = row.get("tools", "[]")
            tools = []
            try:
                raw = json.loads(tools_str) if isinstance(tools_str, str) else tools_str
                if isinstance(raw, list):
                    tools = [wrap_tool_openai(t) for t in raw]
            except (json.JSONDecodeError, TypeError):
                pass
            messages = []
            for conv in convs:
                role_from, value = conv.get("from", ""), conv.get("value", "")
                if role_from == "system":
                    messages.append({"role": "system",
                                     "content": "You are a helpful assistant with access to tools." if tools else value})
                elif role_from == "human":
                    messages.append({"role": "user", "content": value})
                elif role_from == "gpt":
                    matches = pat.findall(value)
                    if matches:
                        calls = []
                        for js in matches:
                            try:
                                d = json.loads(js)
                                calls.append(make_tool_call(d.get("name", "unknown"), d.get("arguments", {})))
                            except json.JSONDecodeError:
                                continue
                        if calls:
                            rest = pat.sub("", value).strip()
                            messages.append({"role": "assistant", "content": rest if rest else None,
                                             "tool_calls": calls})
                        else:
                            messages.append({"role": "assistant", "content": value})
                    else:
                        messages.append({"role": "assistant", "content": value})
            sample = {"messages": messages}
            if tools:
                sample["tools"] = tools
            if validate_sample(sample):
                results.append(sample)
        except Exception:
            continue
    print(f"  -> {len(results)}")
    return results


# ---------------------------------------------------------------- 5 toolace
def convert_toolace_qwen(cap=None, streaming=True) -> list:
    print("[5/7] tryumanshow/ToolACE-Qwen-cleaned ...")
    ds = _load("tryumanshow/ToolACE-Qwen-cleaned", "train", streaming)
    results = []
    for row in _cap_iter(ds, cap):
        try:
            tools_str, convs_str = row.get("tools", "[]"), row.get("conversations", "[]")
            tools = []
            try:
                raw = json.loads(tools_str) if isinstance(tools_str, str) else tools_str
                if isinstance(raw, list):
                    tools = [wrap_tool_openai(t) for t in raw]
            except (json.JSONDecodeError, TypeError):
                pass
            try:
                convs = json.loads(convs_str) if isinstance(convs_str, str) else convs_str
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(convs, list):
                continue
            messages = []
            if tools:
                messages.append({"role": "system", "content": "You are a helpful assistant with access to tools."})
            for conv in convs:
                role, content = conv.get("role", ""), conv.get("content", "")
                if role == "user":
                    messages.append({"role": "user",
                                     "content": content if isinstance(content, str)
                                     else json.dumps(content, ensure_ascii=False)})
                elif role == "assistant":
                    tc_raw = conv.get("tool_calls", [])
                    if tc_raw:
                        calls = []
                        for tc in tc_raw:
                            fd = tc.get("function", {})
                            args = fd.get("arguments", {})
                            if isinstance(args, str):
                                try:
                                    args = json.loads(args)
                                except json.JSONDecodeError:
                                    args = {"raw": args}
                            calls.append(make_tool_call(fd.get("name", "unknown"), args))
                        messages.append({"role": "assistant", "content": None, "tool_calls": calls})
                    else:
                        messages.append({"role": "assistant",
                                         "content": content if isinstance(content, str)
                                         else json.dumps(content, ensure_ascii=False)})
                elif role == "tool":
                    name = conv.get("name", "")
                    if not name:
                        for m in reversed(messages):
                            if m.get("role") == "assistant" and m.get("tool_calls"):
                                name = m["tool_calls"][0]["function"]["name"]
                                break
                    tc_content = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
                    messages.append({"role": "tool", "content": tc_content, "name": name})
            sample = {"messages": messages}
            if tools:
                sample["tools"] = tools
            if validate_sample(sample):
                results.append(sample)
        except Exception:
            continue
    print(f"  -> {len(results)}")
    return results


# ---------------------------------------------------------------- 6 opus
def convert_opus_reasoning(cap=None, streaming=True) -> list:
    print("[6/7] nohurry/Opus-4.6-Reasoning-3000x-filtered ...")
    ds = _load("nohurry/Opus-4.6-Reasoning-3000x-filtered", "train", streaming)
    results = []
    for row in _cap_iter(ds, cap):
        try:
            problem, thinking, solution = row.get("problem", ""), row.get("thinking", ""), row.get("solution", "")
            if not problem or not solution:
                continue
            content = f"<think>\n{thinking}\n</think>\n\n{solution}" if thinking else solution
            sample = {"messages": [
                {"role": "user", "content": problem},
                {"role": "assistant", "content": content},
            ]}
            if validate_sample(sample):
                results.append(sample)
        except Exception:
            continue
    print(f"  -> {len(results)}")
    return results


# ---------------------------------------------------------------- 7 openclaw
def convert_openclaw(cap=None, streaming=True) -> list:
    print("[7/7] bellfire/openclaw-coder-dataset ...")
    results = []
    seen_splits = 0
    for split in ("train", "test"):
        try:
            ds = _load("bellfire/openclaw-coder-dataset", split, streaming)
        except Exception:
            continue
        for row in _cap_iter(ds, cap):
            try:
                raw_messages = row.get("messages", [])
                if not raw_messages:
                    continue
                messages = []
                for m in raw_messages:
                    role, content = m.get("role", ""), m.get("content", "")
                    if role == "system":
                        messages.append({"role": "system",
                                         "content": content if isinstance(content, str)
                                         else json.dumps(content, ensure_ascii=False)})
                    elif role == "user":
                        messages.append({"role": "user",
                                         "content": content if isinstance(content, str)
                                         else json.dumps(content, ensure_ascii=False)})
                    elif role == "assistant":
                        tc_raw = m.get("tool_calls", [])
                        if tc_raw:
                            calls = []
                            for tc in tc_raw:
                                fd = tc.get("function", {})
                                args = fd.get("arguments", {})
                                if isinstance(args, str):
                                    try:
                                        args = json.loads(args)
                                    except json.JSONDecodeError:
                                        args = {"raw": args}
                                calls.append(make_tool_call(fd.get("name", "unknown"), args))
                            messages.append({"role": "assistant", "content": None, "tool_calls": calls})
                        else:
                            messages.append({"role": "assistant",
                                             "content": content if isinstance(content, str)
                                             else json.dumps(content, ensure_ascii=False)})
                    elif role == "tool":
                        name = m.get("name", "")
                        if not name:
                            for prev in reversed(messages):
                                if prev.get("role") == "assistant" and prev.get("tool_calls"):
                                    name = prev["tool_calls"][0]["function"]["name"]
                                    break
                        tc_content = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
                        messages.append({"role": "tool", "content": tc_content, "name": name})
                sample = {"messages": messages}
                if validate_sample(sample):
                    results.append(sample)
            except Exception:
                continue
        seen_splits += 1
    print(f"  -> {len(results)} ({seen_splits} splits)")
    return results


CONVERTERS: list[tuple[str, Callable]] = [
    ("deepexi_zh", convert_deepexi),
    ("glaive_zh", convert_glaive_zh),
    ("glaive_v2_en", convert_glaive_v2_sharegpt),
    ("hermes_en", convert_hermes_fc),
    ("toolace_en", convert_toolace_qwen),
    ("opus_reasoning_en", convert_opus_reasoning),
    ("openclaw_en", convert_openclaw),
]
