#!/usr/bin/env python3
import sys, json, re
sys.path.insert(0, "src")
from transformers import AutoTokenizer
from qwen_tool_sft.dataio import read_jsonl, render_text

tok = AutoTokenizer.from_pretrained("models_local/Qwen/Qwen3-4B-Base")
rows = read_jsonl("data/smoke/train.jsonl")
shown = 0
for r in rows:
    if any(len(m.get("tool_calls") or []) >= 2 for m in r["messages"]):
        text = render_text(tok, r["messages"], r.get("tools"))
        # find assistant-generated tool blocks: occurrences of <tool_call> AFTER
        # the last <|im_start|>assistant marker containing multiple calls
        for m in re.finditer(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", text, re.DOTALL):
            seg = m.group(1)
            if seg.count('"name"') >= 2:
                print("=" * 90)
                print(seg[:900])
                shown += 1
                break
        if shown >= 3:
            break
print("INSPECT2_DONE")
