#!/usr/bin/env python3
import sys, json
sys.path.insert(0, "src")
from pathlib import Path
from transformers import AutoTokenizer
from qwen_tool_sft.dataio import read_jsonl, render_text

tok = AutoTokenizer.from_pretrained("models_local/Qwen/Qwen3-4B-Base")
for t in ["<tool_call>", "</tool_call>", "<|im_start|>", "<|im_end|>", "𬜯"]:
    ids = tok(t, add_special_tokens=False).input_ids
    print(f"TOKEN {t!r}: ids={ids} decoded={[tok.decode([i]) for i in ids]}")

print("added tokens containing tool_call:")
for k, v in tok.get_added_vocab().items():
    if "tool" in k:
        print(" ", repr(k), v, "->", repr(tok.decode([v])))

rows = read_jsonl("data/smoke/train.jsonl")
s = next(r for r in rows if any(m.get("tool_calls") for m in r["messages"]))
text = render_text(tok, s["messages"], s.get("tools"))
i = text.find("tool_call")
print("RENDER SNIPPET:", repr(text[max(0, i-40):i+120]))
# tokenize the snippet region
ids = tok(text, add_special_tokens=False).input_ids
# find ids around first occurrence of arguments JSON
import re
m = re.search(r'\{"name"', text)
seg = text[:m.start()]
seg_ids = tok(seg, add_special_tokens=False).input_ids
print("last 12 ids before JSON:", seg_ids[-12:])
print("their decode:", [tok.decode([i]) for i in seg_ids[-12:]])
print("DIAG_DONE")
