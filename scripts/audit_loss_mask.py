#!/usr/bin/env python3
"""Audit the SFT loss mask (task P0-1 acceptance evidence).

Verifies, on representative scenarios AND on real prepared data:
  * patched template renders byte-identical text to the original;
  * system / user / tool-schema / tool-response tokens are masked (-100);
  * assistant text, native <tool_call> JSON and <|im_end|> carry loss;
  * reports the supervised token fraction on real data.

Outputs runs/loss_mask_audit/loss_mask_audit.json and exits non-zero if any
hard assertion fails.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qwen_tool_sft import paths  # noqa: E402
from qwen_tool_sft.config import load_config  # noqa: E402
from qwen_tool_sft.dataio import read_jsonl  # noqa: E402
from qwen_tool_sft.loss_mask import audit_masks, patch_chat_template, tokenize_assistant_only  # noqa: E402


def tc(name, args):
    return {"type": "function", "function": {"name": name, "arguments": args}}


def tool(name, props=None, req=None):
    return {"type": "function", "function": {
        "name": name, "description": f"{name} function",
        "parameters": {"type": "object",
                       "properties": props or {"q": {"type": "string"}},
                       "required": req or list((props or {"q": 1}).keys())}}}


SCENARIOS = {
    "plain_text_no_tools": {
        "messages": [{"role": "user", "content": "你好"},
                     {"role": "assistant", "content": "你好！有什么可以帮你？"}],
        "tools": None,
        "must_be_in_loss": ["你好！有什么可以帮你？"],
        "must_be_masked": ["你好"],
    },
    "single_tool_call": {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "帮我查一下北京天气"},
            {"role": "assistant", "content": None,
             "tool_calls": [tc("get_weather", {"city": "北京"})]},
            {"role": "tool", "name": "get_weather", "content": "晴 25度"},
            {"role": "assistant", "content": "北京今天晴，25度。"},
        ],
        "tools": [tool("get_weather", {"city": {"type": "string"}}, ["city"])],
        "must_be_in_loss": ['"name": "get_weather"', '"city": "北京"', "北京今天晴"],
        "must_be_masked": ["get_weather function", "帮我查一下北京天气", "晴 25度",
                           "You are a helpful assistant"],
    },
    "parallel_multiple_calls": {
        "messages": [
            {"role": "user", "content": "同时查北京、上海、广州天气"},
            {"role": "assistant", "content": None, "tool_calls": [
                tc("get_weather", {"city": "北京"}),
                tc("get_weather", {"city": "上海"}),
                tc("get_weather", {"city": "广州"})]},
        ],
        "tools": [tool("get_weather", {"city": {"type": "string"}}, ["city"])],
        "must_be_in_loss": ['"city": "北京"', '"city": "上海"', '"city": "广州"'],
        "must_be_masked": ["同时查北京、上海、广州天气"],
    },
    "multiple_different_tools": {
        "messages": [
            {"role": "user", "content": "查天气然后发邮件"},
            {"role": "assistant", "content": None, "tool_calls": [
                tc("get_weather", {"city": "X"}),
                tc("send_email", {"to": "a@b.com", "subject": "s"})]},
        ],
        "tools": [tool("get_weather"),
                  tool("send_email", {"to": {"type": "string"}, "subject": {"type": "string"}},
                       ["to", "subject"])],
        "must_be_in_loss": ['"name": "send_email"', '"to": "a@b.com"'],
        "must_be_masked": ["查天气然后发邮件", "send_email function"],
    },
    "multi_turn_assistant_texts": {
        "messages": [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
                     {"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"}],
        "tools": None,
        "must_be_in_loss": ["a1", "a2"],
        "must_be_masked": ["q1", "q2"],
    },
    "no_tool_with_tools_available": {
        "messages": [{"role": "user", "content": "你好吗"},
                     {"role": "assistant", "content": "我很好，谢谢。"}],
        "tools": [tool("get_weather")],
        "must_be_in_loss": ["我很好，谢谢。"],
        "must_be_masked": ["你好吗", "get_weather function"],
    },
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_sft_v2_30k.yaml")
    ap.add_argument("--data", default="data/full_v2/train.jsonl",
                    help="real data file for aggregate stats (optional)")
    ap.add_argument("--sample-n", type=int, default=500)
    ap.add_argument("--out", default="runs/loss_mask_audit")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    cfg = load_config(args.config)
    model_dir = paths.model_path(cfg["model_path"])
    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # sanity: patched template exists and differs only by generation markers
    patched = patch_chat_template(tok.chat_template)
    assert "{%- generation %}" in patched and "{%- endgeneration %}" in patched

    report = {"scenarios": {}, "hard_checks_passed": True}
    for name, spec in SCENARIOS.items():
        r = audit_masks(tok, spec["messages"], spec["tools"])
        loss_text, masked_text = r["loss_text"], r["masked_text"]
        checks = {"render_identical": r["render_identical_to_original"]}
        for s in spec["must_be_in_loss"]:
            checks[f"loss_contains::{s}"] = s in loss_text
        for s in spec["must_be_masked"]:
            checks[f"masked_contains::{s}"] = s in masked_text
        checks["has_loss_tokens"] = r["n_loss_tokens"] > 0
        r["checks"] = checks
        r["checks_all_pass"] = all(checks.values())
        report["scenarios"][name] = r
        report["hard_checks_passed"] &= r["checks_all_pass"]
        print(f"[{'PASS' if r['checks_all_pass'] else 'FAIL'}] {name}: "
              f"{r['n_loss_tokens']}/{r['n_tokens']} loss tokens")
        for k, v in checks.items():
            if not v:
                print(f"    FAILED CHECK: {k}")

    # aggregate stats on real data
    data_path = paths.resolve_path(args.data)
    if data_path.exists():
        rows = read_jsonl(data_path)[: args.sample_n]
        tot = loss = 0
        n_tool_call_turns = 0
        for s in rows:
            t = tokenize_assistant_only(tok, s.get("messages", []), s.get("tools"))
            tot += len(t["input_ids"])
            loss += sum(1 for x in t["labels"] if x != -100)
            if any(len(m.get("tool_calls") or []) for m in s.get("messages", [])):
                n_tool_call_turns += 1
        report["real_data"] = {
            "path": str(data_path), "n": len(rows),
            "total_tokens": tot, "loss_tokens": loss,
            "supervised_fraction": round(loss / max(tot, 1), 4),
            "samples_with_tool_calls": n_tool_call_turns,
        }
        print(f"real data: {loss}/{tot} = {report['real_data']['supervised_fraction']*100:.1f}% supervised")
    else:
        print(f"[skip] real data not found: {data_path}")

    out_dir = paths.resolve_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "loss_mask_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {out_dir / 'loss_mask_audit.json'}")
    if not report["hard_checks_passed"]:
        print("LOSS_MASK_AUDIT_FAILED")
        sys.exit(1)
    print("LOSS_MASK_AUDIT_PASSED")


if __name__ == "__main__":
    main()
