#!/usr/bin/env python3
import torch, sys
sys.path.insert(0, "src")
from transformers import AutoModelForCausalLM, AutoTokenizer

mp = "models_local/Qwen/Qwen3-4B-Base"
tok = AutoTokenizer.from_pretrained(mp)
model = AutoModelForCausalLM.from_pretrained(mp, dtype=torch.bfloat16,
                                            attn_implementation="sdpa").cuda().eval()
emb = model.get_input_embeddings().weight
head = model.get_output_embeddings().weight
print("vocab", emb.shape[0], "hidden", emb.shape[1])
with torch.no_grad():
    en = emb.float().norm(dim=1)
    hn = head.float().norm(dim=1)
    med_e = en.median().item(); med_h = hn.median().item()
    for tid in [151643, 151644, 151645, 151657, 151658, 151665, 151666, 122588]:
        t = tok.decode([tid])
        print(f"id={tid} {t!r:20s} emb_norm={en[tid].item():.4f} (med {med_e:.3f})  head_norm={hn[tid].item():.4f} (med {med_h:.3f})")

# next-token prediction at assistant generation prefix for a tool-call prompt
msgs = [{"role": "user", "content": "查一下北京明天的天气"}]
tools = [{"type": "function", "function": {"name": "get_weather",
        "description": "query weather", "parameters": {"type": "object",
        "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
prompt = tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True)
print("PROMPT TAIL:", repr(prompt[-260:]))
ids = tok(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    logits = model(**ids).logits[0, -1]
top = torch.topk(logits, 10)
for v, i in zip(top.values.tolist(), top.indices.tolist()):
    print(f"  top {i} {tok.decode([i])!r:12s} {v:.3f}")
for tid in [151657, 122588]:
    print(f"  logit id {tid} {tok.decode([tid])!r}: {logits[tid].item():.3f}")
print("DIAG_EMB_DONE")
