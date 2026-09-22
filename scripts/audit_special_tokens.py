#!/usr/bin/env python3
"""Audit special-token embedding norms and template usage (task P1 evidence).

Loads the base model input embeddings and records, for every added token:
  * token string / id
  * L2 norm (flagged LOW_NORM if < 0.5, i.e. effectively untrained)
  * whether the chat template emits it
Outputs runs/special_token_audit/token_audit.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qwen_tool_sft import paths  # noqa: E402
from qwen_tool_sft.config import load_config  # noqa: E402

LOW_NORM_THRESHOLD = 0.5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_sft_v2_30k.yaml")
    ap.add_argument("--out", default="runs/special_token_audit")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from qwen_tool_sft.dataio import normalize_for_template
    cfg = load_config(args.config)
    model_dir = paths.model_path(cfg["model_path"])
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    emb = model.get_input_embeddings().weight.data

    # tokens the chat template actually emits
    probe_messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"type": "function", "function": {"name": "f", "arguments": {"x": 1}}}]},
        {"role": "tool", "name": "f", "content": "r"},
        {"role": "assistant", "content": "done"},
    ]
    probe_tools = [{"type": "function", "function": {
        "name": "f", "description": "f",
        "parameters": {"type": "object", "properties": {"x": {"type": "integer"}},
                       "required": ["x"]}}}]
    rendered = tok.apply_chat_template(normalize_for_template(probe_messages),
                                       tools=probe_tools,
                                       tokenize=False, add_generation_prompt=False)

    rows = []
    low_norm = []
    for token, idx in sorted(tok.added_tokens_encoder.items(), key=lambda kv: kv[1]):
        norm = round(float(emb[idx].float().norm().item()), 4)
        used = token in rendered
        row = {"token": token, "id": idx, "embedding_norm": norm,
               "low_norm": norm < LOW_NORM_THRESHOLD, "used_by_chat_template": used}
        rows.append(row)
        if norm < LOW_NORM_THRESHOLD:
            low_norm.append(row)

    # targeted protocol tokens
    targets = ["<|im_start|>", "<|im_end|>", "<tool_call>", "</tool_call>",
               "<tool_response>", "</tool_response>", "<|endoftext|>"]
    targeted = []
    for t in targets:
        idx = tok.convert_tokens_to_ids(t)
        targeted.append({
            "token": t, "id": int(idx) if isinstance(idx, int) else None,
            "embedding_norm": round(float(emb[idx].float().norm().item()), 4)
            if isinstance(idx, int) and 0 <= idx < emb.shape[0] else None,
            "used_by_chat_template": t in rendered,
        })

    report = {
        "model_path": str(model_dir),
        "tie_word_embeddings": getattr(model.config, "tie_word_embeddings", None),
        "vocab_size": emb.shape[0],
        "hidden_size": emb.shape[1],
        "low_norm_threshold": LOW_NORM_THRESHOLD,
        "n_added_tokens": len(rows),
        "n_low_norm_added_tokens": len(low_norm),
        "targeted_tokens": targeted,
        "low_norm_added_tokens": low_norm,
        "all_added_tokens": rows,
        "selective_training_recommendation":
            [r["id"] for r in targeted if r["used_by_chat_template"]],
    }
    out_dir = paths.resolve_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "token_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ["n_added_tokens", "n_low_norm_added_tokens",
                       "targeted_tokens", "selective_training_recommendation"]},
                     ensure_ascii=False, indent=2))
    print(f"saved: {out_dir / 'token_audit.json'}")


if __name__ == "__main__":
    main()
