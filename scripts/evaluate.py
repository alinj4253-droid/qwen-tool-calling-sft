#!/usr/bin/env python3
"""Evaluate base / SFT model on the tool-calling eval set.

Greedy decoding, fixed prompt, per-case predictions + metrics.json + summary.md.
Clean stop uses BOTH the decoded-string check and the token-level termination
signal; Strict Protocol Success is then derived by CaseScore.

Examples:
  python scripts/evaluate.py --config configs/eval_base.yaml
  python scripts/evaluate.py --config configs/eval_sft_v2_30k.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qwen_tool_sft import paths  # noqa: E402
from qwen_tool_sft.config import load_config, parse_kv  # noqa: E402
from qwen_tool_sft.dataio import read_jsonl, normalize_for_template  # noqa: E402
from qwen_tool_sft.metrics import aggregate, metrics_table, score_case  # noqa: E402
from qwen_tool_sft.parser import parse_tool_calls  # noqa: E402

# eos (151643) and im_end (151645) both count as a clean turn termination
TERMINATOR_IDS = [151643, 151645]
TERMINATOR_STRINGS = ("<|im_end|>",)


def build_prompt(tokenizer, case: dict) -> str:
    messages = normalize_for_template(case["messages"])
    tools = case.get("tools")
    return tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False,
        add_generation_prompt=True,
    )


def load_model(cfg: dict):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    model_dir = paths.model_path(cfg["model_path"])
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if cfg.get("bf16", True) else torch.float16
    attn = "sdpa"
    try:
        import flash_attn  # noqa: F401
        attn = "flash_attention_2"
    except Exception:
        pass
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=dtype, attn_implementation=attn).cuda().eval()

    adapter = cfg.get("adapter_path", "")
    if adapter:
        adapter_dir = paths.resolve_path(adapter)
        if not adapter_dir.exists():
            raise FileNotFoundError(f"adapter not found: {adapter_dir}")
        peft_model = PeftModel.from_pretrained(model, adapter_dir)
        try:
            model = peft_model.merge_and_unload()
            print(f"merged adapter: {adapter_dir}")
        except Exception as e:
            print(f"[warn] merge_and_unload failed ({e}); using PeftModel unmerged")
            model = peft_model.cuda().eval()
    return model, tokenizer


def evaluate(cfg: dict) -> dict:
    from transformers import AutoTokenizer

    model, tokenizer = load_model(cfg)
    # accept both v2 keys (cases/output_dir) and v1 keys (eval_set/output_subdir)
    cases_arg = cfg.get("cases") or cfg.get("eval_set", "eval/tool_calling_eval.jsonl")
    cases_path = paths.resolve_path(cases_arg)
    cases = read_jsonl(cases_path)
    max_cases = int(cfg.get("max_cases", cfg.get("limit", 0)) or 0)
    if max_cases:
        cases = cases[:max_cases]

    out_arg = cfg.get("output_dir") or cfg.get("output_subdir") \
        or f"runs/eval_{cfg.get('run_name', 'model')}"
    out_dir = paths.resolve_path(out_arg)
    out_dir.mkdir(parents=True, exist_ok=True)

    gen_cfg = cfg.get("generation", {})
    max_new = int(cfg.get("max_new_tokens", gen_cfg.get("max_new_tokens", 1024)))
    do_sample = bool(cfg.get("do_sample", False))
    temperature = float(cfg.get("temperature", 0.0))

    predictions, scores = [], []
    t_start = time.time()
    for i, case in enumerate(cases):
        prompt = build_prompt(tokenizer, case)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=TERMINATOR_IDS,
            )
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        raw = tokenizer.decode(gen_ids, skip_special_tokens=False)
        terminated_on_token = int(gen_ids[-1].item()) in TERMINATOR_IDS if gen_ids.numel() else False

        parsed = parse_tool_calls(raw)
        sc = score_case(case, parsed, raw=raw, terminators=TERMINATOR_STRINGS)
        # clean stop iff a terminator token was actually generated and the
        # text shows no prompt leak (token-level signal is authoritative).
        from qwen_tool_sft.metrics import prompt_leaked
        sc.clean_stop = bool(terminated_on_token) and not prompt_leaked(raw)

        pred = {
            "id": case.get("id", f"case-{i}"),
            "category": case.get("category", ""),
            "expected": case.get("expect", {}),
            "raw_output": raw,
            "terminated_on_token": terminated_on_token,
            "last_token_id": int(gen_ids[-1].item()) if gen_ids.numel() else None,
            "parsed_calls": [c.to_dict() for c in parsed.calls],
            "invalid_blocks": parsed.invalid_blocks,
            "parse_errors": parsed.parse_errors,
            "valid_format": sc.valid_format,
            "canonical_format": sc.canonical_format,
            "tool_selection_correct": sc.tool_selection_correct,
            "arg_key_accuracy": sc.arg_key_accuracy,
            "arg_exact_match": sc.arg_exact_match,
            "no_tool_correct": sc.no_tool_correct,
            "clean_stop": sc.clean_stop,
            "strict_protocol_success": sc.strict_protocol_success,
            "overall": sc.overall,
        }
        predictions.append(pred)
        scores.append(sc)
        if (i + 1) % 10 == 0 or i == len(cases) - 1:
            print(f"  [{i+1}/{len(cases)}] elapsed {time.time()-t_start:.0f}s")

    metrics = aggregate(scores)
    metrics["model_label"] = cfg.get("model_label", cfg.get("run_name", "model"))
    metrics["cases_path"] = str(cases_path)
    metrics["adapter_path"] = cfg.get("adapter_path", "")
    metrics["loss_mode"] = cfg.get("loss_mode", "")
    metrics["special_token_training"] = cfg.get("special_token_training", "")
    metrics["max_new_tokens"] = max_new
    metrics["do_sample"] = do_sample
    metrics["eval_duration_sec"] = round(time.time() - t_start, 1)

    (out_dir / "predictions.jsonl").write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in predictions),
        encoding="utf-8")
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    label = cfg.get("model_label", cfg.get("run_name", "model"))
    table = metrics_table(metrics, label)
    cat_lines = ["", "## By category", "",
                 "| Category | N | Overall | Strict Protocol |", "|---|---:|---:|---:|"]
    for cat, d in sorted(metrics["by_category"].items()):
        cat_lines.append(
            f"| {cat} | {d['n']} | {d['accuracy']*100:.1f}% | {d['strict_protocol_rate']*100:.1f}% |")
    summary = [
        f"# Eval summary: {label}", "",
        f"cases: `{cases_path}` ({metrics['n_cases']})",
        f"adapter: `{cfg.get('adapter_path', '(base)')}`",
        f"max_new_tokens={max_new}, do_sample={do_sample}", "",
        table,
    ] + cat_lines
    (out_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(table)
    print(f"\nsaved: {out_dir}")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config)
    cfg.update(parse_kv(args.set))
    evaluate(cfg)


if __name__ == "__main__":
    main()
