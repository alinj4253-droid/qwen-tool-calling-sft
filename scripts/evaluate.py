#!/usr/bin/env python3
"""Automatic tool-calling evaluation for a base or SFT model.

Outputs (under runs/<run_name>/ or configured output dir):
  predictions.jsonl  -- every case with raw output, parsed calls, per-case score
  metrics.json       -- aggregated metrics
  summary.md         -- markdown metric table (single model)

Then compare base vs SFT:
  python scripts/compare_runs.py runs/eval_base runs/eval_sft
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qwen_tool_sft import paths  # noqa: E402
from qwen_tool_sft.config import load_config, parse_kv  # noqa: E402
from qwen_tool_sft.dataio import iter_jsonl, normalize_for_template  # noqa: E402
from qwen_tool_sft.metrics import (  # noqa: E402
    aggregate, metrics_table, prompt_leaked, score_case,
)
from qwen_tool_sft.parser import parse_tool_calls  # noqa: E402


def build_model(cfg: dict):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    model_dir = paths.model_path(cfg["model_path"])
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if cfg.get("bf16", True) else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype,
                                                 attn_implementation="sdpa")
    adapter = cfg.get("adapter_path")
    if adapter:
        adapter_p = paths.resolve_path(adapter)
        print(f"loading adapter: {adapter_p}")
        model = PeftModel.from_pretrained(model, str(adapter_p))
        model = model.merge_and_unload()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return model, tokenizer, device


def build_eos_ids(model, tokenizer) -> list[int]:
    """Stop on both <|endoftext|> and the chat-template turn end <|im_end|>."""
    ids: set[int] = set()
    gen_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    for e in (gen_eos, tokenizer.eos_token_id):
        if isinstance(e, (list, tuple)):
            ids.update(int(x) for x in e)
        elif isinstance(e, int):
            ids.add(e)
    for tok in ("<|im_end|>",):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tid, int) and tid >= 0:
            ids.add(tid)
    return sorted(ids)


def generate_once(model, tokenizer, device, messages, tools, gen_cfg: dict,
                  eos_ids: list[int] | None = None) -> tuple[str, bool]:
    prompt = tokenizer.apply_chat_template(
        messages,
        tools=tools or None,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=int(gen_cfg.get("max_new_tokens", 1024)),
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_ids,
        )
    gen = out[0][inputs["input_ids"].shape[1]:]
    terminated = int(gen[-1]) in set(eos_ids or [])
    return tokenizer.decode(gen, skip_special_tokens=False), terminated


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config)
    for k, v in parse_kv(args.set).items():
        cfg[k] = v

    run_name = cfg["run_name"]
    out_dir = paths.resolve_path(cfg.get("output_subdir", f"runs/{run_name}"),
                                 base=paths.RUNS_DIR.parent) if "output_subdir" in cfg \
        else paths.RUNS_DIR / run_name
    out_dir = paths.resolve_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_set = paths.resolve_path(cfg.get("eval_set", "eval/tool_calling_eval.jsonl"))
    cases = list(iter_jsonl(eval_set))
    limit = cfg.get("limit")
    if limit:
        cases = cases[:int(limit)]
    print(f"loaded {len(cases)} eval cases from {eval_set}")

    model, tokenizer, device = build_model(cfg)
    gen_cfg = cfg.get("generation", {})
    eos_ids = build_eos_ids(model, tokenizer)
    terminator_strs = tuple(
        t for t in (tokenizer.decode([i], skip_special_tokens=False) for i in eos_ids) if t
    )
    print(f"eos_token_id set: {eos_ids} -> terminators {terminator_strs}")

    predictions, scores = [], []
    t0 = time.time()
    for i, case in enumerate(cases, 1):
        raw, terminated = generate_once(model, tokenizer, device, case["messages"],
                                        case.get("tools"), gen_cfg, eos_ids=eos_ids)
        parsed = parse_tool_calls(raw)
        # Token-level stop signal + string check (last token may be a special
        # stripped differently across tokenizer versions).
        sc = score_case(case, parsed, raw=raw, terminators=terminator_strs)
        sc.clean_stop = (sc.clean_stop or terminated) and not prompt_leaked(raw)
        if i <= 5 or not sc.overall:
            print(f"[{i:02d}/{len(cases)}] {case.get('id')} | {case.get('category')} "
                  f"| overall={sc.overall} pred={parsed.names}")
        predictions.append({
            "id": case.get("id"),
            "category": case.get("category"),
            "expect": case.get("expect"),
            "raw_output": raw,
            "parsed_calls": [{"name": c.name, "arguments": c.arguments, "valid_json": c.valid_json}
                             for c in parsed.calls],
            "invalid_blocks": parsed.invalid_blocks,
            "score": {
                "overall": sc.overall,
                "valid_format": sc.valid_format,
                "canonical_format": sc.canonical_format,
                "clean_stop": sc.clean_stop,
                "terminated": terminated,
                "tool_selection_correct": sc.tool_selection_correct,
                "arg_key_accuracy": sc.arg_key_accuracy,
                "arg_exact_match": sc.arg_exact_match,
                "no_tool_correct": sc.no_tool_correct,
                "invalid_json": sc.invalid_json,
                "wrong_tool": sc.wrong_tool,
                "missing_argument": sc.missing_argument,
                "extra_argument": sc.extra_argument,
                "expected_names": sc.expected_names,
                "predicted_names": sc.predicted_names,
            },
        })
        scores.append(sc)

    metrics = aggregate(scores)
    metrics["run_name"] = run_name
    metrics["model_path"] = cfg["model_path"]
    metrics["adapter_path"] = cfg.get("adapter_path")
    metrics["eval_seconds"] = round(time.time() - t0, 1)

    (out_dir / "predictions.jsonl").write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in predictions) + "\n",
        encoding="utf-8")
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "summary.md").write_text(
        f"# Eval: {run_name}\n\n" + metrics_table(metrics, run_name) + "\n\n"
        f"Cases: {metrics['n_cases']} (tool: {metrics['n_tool_cases']}, "
        f"no-tool: {metrics['n_no_tool_cases']})\n",
        encoding="utf-8")
    print("\n" + metrics_table(metrics, run_name))
    print(f"\nEVAL_DONE -> {out_dir}")


if __name__ == "__main__":
    main()
