"""Minimal, interpretable GRPO smoke for parallel tool calling (P2 stage).

Gate (task book section 14): only run after SFT-v2, assistant-only loss,
expanded + decontaminated benchmarks, and targeted parallel SFT have all
been exhausted with parallel still failing. This is a *small* smoke
(section 16): no vLLM, no long run, no LLM-as-judge; five explicit reward
components (canonical format, clean stop, tool selection, argument
correctness, parallel completion) computed with the project's own parser
and scorer. The trained LoRA is evaluated afterwards with scripts/evaluate.py
and only retained if the section-17 bar is met.

Start model: the best SFT checkpoint (base + stacked SFT adapters merged
in memory), then a fresh LoRA is trained by GRPO.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from qwen_tool_sft import paths  # noqa: E402
from qwen_tool_sft.config import load_config  # noqa: E402
from qwen_tool_sft.dataio import read_jsonl  # noqa: E402
from qwen_tool_sft.parser import parse_tool_calls  # noqa: E402
from qwen_tool_sft.metrics import score_case, prompt_leaked, cleanly_terminated  # noqa: E402

TERMINATOR_STRINGS = ("<|im_end|>",)


def _assistant_calls(msg: dict) -> list[dict]:
    out = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", tc)
        out.append({"name": fn["name"], "arguments": fn.get("arguments", {}) or {}})
    return out


def row_to_case(row: dict) -> dict | None:
    """Convert one SFT training row to an eval-style scoring case."""
    messages = row["messages"]
    tools = row.get("tools")
    # first assistant turn that issues calls
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            calls = _assistant_calls(msg)
            if not calls:
                return None
            return {
                "id": row.get("id", ""),
                "category": "parallel" if len(calls) >= 2 else "single",
                "messages": messages[:i],
                "tools": tools,
                "expect": {"mode": "tool", "calls": calls},
            }
    # no-call row (trap / pure chat)
    last_asst = max(
        (i for i, m in enumerate(messages) if m.get("role") == "assistant"),
        default=-1,
    )
    if last_asst < 0:
        return None
    return {
        "id": row.get("id", ""),
        "category": "no_tool",
        "messages": messages[:last_asst],
        "tools": tools,
        "expect": {"mode": "no_tool", "calls": []},
    }


def build_smoke_cases(cfg: dict, tokenizer=None, build_prompt_fn=None) -> list[dict]:
    data_dir = paths.resolve_path(cfg["data_dir"])
    rows = read_jsonl(data_dir / "train.jsonl")
    cap = int(cfg.get("max_prompt_tokens", 2048))
    parallel_same, parallel_diff, guard = [], [], []

    def fits(case: dict) -> bool:
        if tokenizer is None or build_prompt_fn is None:
            return True
        text = build_prompt_fn(tokenizer, case)
        return len(tokenizer(text)["input_ids"]) <= cap

    for row in rows:
        case = row_to_case(row)
        if case is None:
            continue
        n = len(case["expect"]["calls"])
        names = [c["name"] for c in case["expect"]["calls"]]
        if n >= 2:
            if len(set(names)) == 1:
                parallel_same.append(case)
            else:
                parallel_diff.append(case)
        elif case["category"] == "no_tool":
            guard.append(case)
    rng = random.Random(cfg.get("seed", 42))
    rng.shuffle(parallel_same)
    rng.shuffle(parallel_diff)
    rng.shuffle(guard)
    n_same = cfg.get("n_parallel_same", 32)
    n_diff = cfg.get("n_parallel_diff", 32)
    n_guard = cfg.get("n_guard", 16)

    def take(pool: list[dict], k: int) -> list[dict]:
        picked = []
        for case in pool:
            if len(picked) >= k:
                break
            if fits(case):
                picked.append(case)
        return picked

    same = take(parallel_same, n_same)
    diff = take(parallel_diff, n_diff)
    guards = take(guard, n_guard)
    cases = same + diff + guards
    rng.shuffle(cases)
    print(f"smoke cases: same={len(same)} diff={len(diff)} guard={len(guards)} "
          f"prompt-token cap={cap} (available same={len(parallel_same)} "
          f"diff={len(parallel_diff)} guard={len(guard)})")
    return cases


# ---------------- reward functions (all interpretable, no LLM judge) -------

def _score_one(text: str, case: dict) -> dict:
    parsed = parse_tool_calls(text)
    sc = score_case(case, parsed, raw=text, terminators=TERMINATOR_STRINGS)
    expected_n = len(case["expect"].get("calls", []))
    predicted_n = len(parsed.calls)
    return {"sc": sc, "parsed": parsed, "expected_n": expected_n,
            "predicted_n": predicted_n, "raw": text}


def reward_canonical_format(completions, **kwargs) -> list[float]:
    cases = [json.loads(x) for x in kwargs["case_json"]]
    out = []
    for text, case in zip(completions, cases):
        p = _score_one(text, case)
        sc, parsed = p["sc"], p["parsed"]
        if not sc.expected_tool:
            out.append(1.0 if not parsed.has_calls else 0.0)
            continue
        if not parsed.has_calls:
            out.append(0.0)
        elif parsed.all_native and not parsed.invalid_blocks:
            out.append(1.0)
        elif any(s == "native" for s in parsed.sources):
            out.append(0.5)
        else:
            out.append(0.0)
    return out


def reward_clean_stop(completions, **kwargs) -> list[float]:
    # TRL decodes completions with skip_special_tokens=True, so the literal
    # <|im_end|> marker is absent; score stopping on content semantics.
    cases = [json.loads(x) for x in kwargs["case_json"]]
    out = []
    for text, case in zip(completions, cases):
        expected_tool = case["expect"].get("mode", "tool") == "tool"
        tail = text.rstrip()
        leaked = ("<|im_start|>" in text) or any(
            line.strip().startswith(mark)
            for line in text.splitlines()
            for mark in ("user:", "assistant:", "system:", "tool:")
        )
        if expected_tool:
            ok = tail.endswith("</tool_call>") and not leaked
        else:
            ok = ("<tool_call>" not in text) and not leaked
        out.append(1.0 if ok else 0.0)
    return out


def reward_tool_selection(completions, **kwargs) -> list[float]:
    cases = [json.loads(x) for x in kwargs["case_json"]]
    out = []
    for text, case in zip(completions, cases):
        p = _score_one(text, case)
        sc = p["sc"]
        if not sc.expected_tool:
            out.append(1.0 if sc.no_tool_correct else 0.0)
            continue
        if p["expected_n"] == 0:
            out.append(0.0)
            continue
        matched = sum(1 for name in sc.expected_names if name in sc.predicted_names)
        # penalize hallucinated extra calls
        extra = max(0, p["predicted_n"] - p["expected_n"])
        val = matched / p["expected_n"]
        val -= 0.2 * extra
        out.append(max(0.0, min(1.0, val)))
    return out


def reward_arguments(completions, **kwargs) -> list[float]:
    cases = [json.loads(x) for x in kwargs["case_json"]]
    out = []
    for text, case in zip(completions, cases):
        p = _score_one(text, case)
        sc = p["sc"]
        if not sc.expected_tool:
            out.append(1.0 if sc.no_tool_correct else 0.0)
        else:
            out.append(float(sc.arg_key_accuracy) if sc.arg_key_accuracy is not None else 0.0)
    return out


def reward_parallel_completion(completions, **kwargs) -> list[float]:
    cases = [json.loads(x) for x in kwargs["case_json"]]
    out = []
    for text, case in zip(completions, cases):
        p = _score_one(text, case)
        if not p["sc"].expected_tool:
            out.append(1.0 if p["predicted_n"] == 0 else 0.0)
            continue
        n = max(1, p["expected_n"])
        gap = abs(p["predicted_n"] - p["expected_n"])
        out.append(max(0.0, 1.0 - gap / n))
    return out


REWARD_FUNCS = [
    reward_canonical_format,
    reward_clean_stop,
    reward_tool_selection,
    reward_arguments,
    reward_parallel_completion,
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs=2, action="append", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config)
    for k, v in args.set:
        try:
            cfg[k] = json.loads(v)
        except json.JSONDecodeError:
            cfg[k] = v

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel, LoraConfig
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    model_dir = paths.model_path(cfg["model_path"])
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    # TRL GRPO hardcodes tokenizer.eos_token_id for rollout stopping; the
    # chat protocol ends assistant turns on <|im_end|> (151645), not base
    # eos (151643). Point eos at <|im_end|> and keep base eos as pad.
    tokenizer.eos_token_id = 151645
    tokenizer.pad_token_id = 151643
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    base_adapters = cfg.get("base_adapter_path", [])
    if isinstance(base_adapters, str):
        base_adapters = [base_adapters]
    for one in base_adapters:
        d = paths.resolve_path(one)
        print(f"merging base adapter: {d}")
        model = PeftModel.from_pretrained(model, d).merge_and_unload()
    model.cuda().train()

    # render prompts exactly like the eval harness
    sys.path.insert(0, str(ROOT / "scripts"))
    from evaluate import build_prompt  # noqa: E402

    cases = build_smoke_cases(cfg, tokenizer=tokenizer, build_prompt_fn=build_prompt)
    rows_out = []
    for case in cases:
        prompt = build_prompt(tokenizer, case)
        rows_out.append({"prompt": prompt, "case_json": json.dumps(case, ensure_ascii=False)})
    n_prompt_tokens = [len(tokenizer(r["prompt"])["input_ids"]) for r in rows_out]
    print(f"prompt tokens: max={max(n_prompt_tokens)} mean={sum(n_prompt_tokens)//len(n_prompt_tokens)}")
    dataset = Dataset.from_list(rows_out)

    peft_config = LoraConfig(
        r=int(cfg.get("lora_r", 16)),
        lora_alpha=int(cfg.get("lora_alpha", 32)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        target_modules=cfg.get("target_modules",
                               ["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"]),
        task_type="CAUSAL_LM",
    )

    out_dir = paths.resolve_path(cfg.get("output_dir", "runs/grpo-smoke"))
    out_dir.mkdir(parents=True, exist_ok=True)
    gen_kwargs = {"eos_token_id": [151643, 151645]}
    grpo_cfg = GRPOConfig(
        output_dir=str(out_dir),
        bf16=True,
        per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 8)),
        num_generations=int(cfg.get("num_generations", 8)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 2)),
        max_completion_length=int(cfg.get("max_completion_length", 512)),
        temperature=float(cfg.get("temperature", 1.0)),
        beta=float(cfg.get("beta", 0.0)),
        learning_rate=float(cfg.get("learning_rate", 1.0e-5)),
        lr_scheduler_type="cosine",
        warmup_steps=int(cfg.get("warmup_steps", 2)),
        max_steps=int(cfg.get("max_steps", 30)),
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=int(cfg.get("seed", 42)),
        use_vllm=False,
        log_completions=False,
        generation_kwargs=gen_kwargs,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=REWARD_FUNCS,
        args=grpo_cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    t0 = time.time()
    trainer.train()
    train_s = time.time() - t0

    final_dir = out_dir / "final_adapter"
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    meta = {
        "run_name": cfg.get("run_name", "grpo-smoke"),
        "base_adapter_path": base_adapters,
        "n_smoke_cases": len(cases),
        "max_steps": int(cfg.get("max_steps", 30)),
        "num_generations": int(cfg.get("num_generations", 8)),
        "learning_rate": float(cfg.get("learning_rate", 1.0e-5)),
        "train_seconds": train_s,
        "reward_components": [f.__name__ for f in REWARD_FUNCS],
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"GRPO smoke done in {train_s:.0f}s, adapter at {final_dir}")


if __name__ == "__main__":
    main()
