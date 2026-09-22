#!/usr/bin/env python3
import sys, json
sys.path.insert(0, "src")
from transformers import AutoTokenizer
from qwen_tool_sft.dataio import read_jsonl, render_text

tok = AutoTokenizer.from_pretrained("models_local/Qwen/Qwen3-4B-Base")
rows = read_jsonl("data/smoke/train.jsonl")
shown = 0
for r in rows:
    if any(len(m.get("tool_calls") or []) >= 2 for m in r["messages"]):
        text = render_text(tok, r["messages"], r.get("tools"))
        i = text.find("<tool_call>")
        print("=" * 90)
        print(text[i:i+700])
        shown += 1
        if shown >= 3:
            break
print("INSPECT_DONE")
