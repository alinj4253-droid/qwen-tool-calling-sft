"""Dataset format validation (task book section XI)."""
import json
from pathlib import Path

from qwen_tool_sft.dataio import (
    LEGAL_ROLES,
    iter_jsonl,
    validate_sample,
    validate_tool_call,
    validate_tool_def,
)

FIX = Path(__file__).parent / "fixtures"


def test_fixtures_valid():
    rows = list(iter_jsonl(FIX / "sample_valid.jsonl"))
    assert len(rows) == 3
    for r in rows:
        ok, reason = validate_sample(r)
        assert ok, reason


def test_fixtures_invalid():
    rows = list(iter_jsonl(FIX / "sample_invalid.jsonl"))
    assert len(rows) == 5
    for r in rows:
        ok, _ = validate_sample(r)
        assert ok is False


def test_tool_def_schema():
    good = {"type": "function", "function": {
        "name": "f", "parameters": {"type": "object", "properties": {}}}}
    assert validate_tool_def(good)
    assert not validate_tool_def({"type": "function", "function": {"name": ""}})
    assert not validate_tool_def({"type": "magic"})


def test_tool_call_schema():
    good = {"type": "function", "function": {"name": "f", "arguments": {"a": 1}}}
    assert validate_tool_call(good)
    str_args = {"type": "function", "function": {"name": "f", "arguments": '{"a": 1}'}}
    assert validate_tool_call(str_args)
    bad_args = {"type": "function", "function": {"name": "f", "arguments": "{oops"}}
    assert not validate_tool_call(bad_args)


def test_roles_legal():
    assert {"system", "user", "assistant", "tool"} <= LEGAL_ROLES


def _validate_file(path: Path):
    assert path.exists() and path.stat().st_size > 0
    n = 0
    for i, row in enumerate(iter_jsonl(path), 1):
        assert isinstance(row.get("messages"), list), f"{path}:{i} no messages"
        ok, reason = validate_sample(row)
        assert ok, f"{path}:{i} invalid: {reason}"
        for tc_msg in (m for m in row["messages"] if m.get("tool_calls")):
            for tc in tc_msg["tool_calls"]:
                assert validate_tool_call(tc), f"{path}:{i} bad tool call"
        for tool in row.get("tools") or []:
            assert validate_tool_def(tool), f"{path}:{i} bad tool def"
        n += 1
    return n


def test_smoke_data_if_present(project_root):
    for name in ("train.jsonl", "eval.jsonl"):
        p = project_root / "data" / name
        if p.exists():
            n = _validate_file(p)
            assert n > 0
