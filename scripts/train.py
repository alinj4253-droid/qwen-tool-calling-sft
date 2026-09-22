#!/usr/bin/env python3
"""LoRA / QLoRA SFT for Qwen tool-calling on one or two RTX 4090s.

Launch (single GPU):
  python scripts/train.py --config configs/train_smoke.yaml
Launch (dual GPU DDP):
  accelerate launch --config_file configs/accelerate_dual4090.yaml \
      scripts/train.py --config configs/train_smoke.yaml
"""
from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qwen_tool_sft import paths  # noqa: E402
from qwen_tool_sft.config import load_config, parse_kv  # noqa: E402
from qwen_tool_sft.dataio import read_jsonl, render_text  # noqa: E402


def git_head() -> str:
    try:
        out = subprocess.run(["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip()
    except Exception:
        return ""


def filter_supported(cls, kwargs: dict) -> dict:
    """Drop kwargs the installed SFTConfig does not accept (cross-version safety)."""
    if dataclasses.is_dataclass(cls):
        supported = {f.name for f in dataclasses.fields(cls)}
    else:
        supported = set(inspect.signature(cls).parameters)
    filtered = {k: v for k, v in kwargs.items() if k in supported}
    dropped = set(kwargs) - set(filtered)
    if dropped:
        print(f"[warn] SFTConfig ignores unsupported args on this TRL version: {sorted(dropped)}")
    return filtered


def load_text_dataset(jsonl_path: Path, tokenizer):
    from datasets import Dataset

    rows = read_jsonl(jsonl_path)
    texts, skipped, errors = [], 0, []
    for i, sample in enumerate(rows):
        try:
            text = render_text(tokenizer, sample.get("messages", []), sample.get("tools"))
            if text:
                texts.append(text)
        except Exception as e:
            skipped += 1
            if len(errors) < 5:
                errors.append(f"row {i}: {type(e).__name__}: {e}")
    print(f"  {jsonl_path.name}: {len(texts)} rendered, {skipped} skipped")
    for e in errors:
        print("   ", e)
    return Dataset.from_dict({"text": texts})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides (YAML values)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    for k, v in parse_kv(args.set).items():
        cfg[k] = v

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main = local_rank == 0

    run_name = cfg["run_name"]
    run_dir = paths.resolve_path(cfg.get("output_subdir", f"runs/{run_name}"),
                                 base=paths.RUNS_DIR.parent) if "output_subdir" in cfg \
        else paths.RUNS_DIR / run_name
    run_dir = paths.resolve_path(run_dir)
    if is_main:
        run_dir.mkdir(parents=True, exist_ok=True)

    model_dir = paths.model_path(cfg["model_path"])
    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(f"model not found at {model_dir}; run scripts/download_model.py")

    print("=" * 60)
    print(f"run: {run_name} | world_size={world_size} local_rank={local_rank}")
    print(f"model: {model_dir}")
    print(f"run_dir: {run_dir}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- model ----
    load_in_4bit = bool(cfg.get("load_in_4bit", False))
    dtype = torch.bfloat16 if cfg.get("bf16", True) else torch.float16
    attn = "sdpa"
    try:
        import flash_attn  # noqa: F401
        attn = "flash_attention_2"
    except Exception:
        pass

    model_kwargs = dict(torch_dtype=dtype, attn_implementation=attn)
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = {"": local_rank}
    model = AutoModelForCausalLM.from_pretrained(model_dir, **model_kwargs)
    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=int(cfg.get("lora_r", 16)),
        lora_alpha=int(cfg.get("lora_alpha", 32)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        target_modules=cfg.get("target_modules",
                               ["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"]),
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    if cfg.get("gradient_checkpointing", True):
        # Required so inputs to frozen layers carry grads into LoRA adapters
        # when gradient checkpointing is enabled.
        model.enable_input_require_grads()
    if is_main:
        model.print_trainable_parameters()

    # ---- data ----
    data_dir = paths.resolve_path(cfg.get("data_dir", "data"))
    train_ds = load_text_dataset(data_dir / "train.jsonl", tokenizer)
    cap_train = cfg.get("max_train_samples")
    if cap_train:
        train_ds = train_ds.select(range(min(int(cap_train), len(train_ds))))
        print(f"  capped train to {len(train_ds)}")
    eval_ds = None
    eval_jsonl = data_dir / "eval.jsonl"
    if eval_jsonl.exists() and cfg.get("do_eval", False):
        eval_ds = load_text_dataset(eval_jsonl, tokenizer)
        cap_eval = cfg.get("max_eval_samples")
        if cap_eval:
            eval_ds = eval_ds.select(range(min(int(cap_eval), len(eval_ds))))

    # ---- trainer args (filtered to installed TRL API) ----
    max_steps = int(cfg.get("max_steps", -1))
    sft_kwargs = dict(
        output_dir=str(run_dir / "checkpoints"),
        per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 8)),
        learning_rate=float(cfg.get("learning_rate", 1e-4)),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.03)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        bf16=bool(cfg.get("bf16", True)),
        logging_steps=int(cfg.get("logging_steps", 5)),
        save_steps=int(cfg.get("save_steps", 200)),
        save_total_limit=int(cfg.get("save_total_limit", 2)),
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        num_train_epochs=float(cfg.get("num_train_epochs", 1)),
        max_steps=max_steps,
        seed=int(cfg.get("seed", 42)),
        dataloader_num_workers=int(cfg.get("dataloader_num_workers", 2)),
        optim=cfg.get("optim", "adamw_torch"),
        report_to="none",
        # TRL >=1.0 renamed max_seq_length -> max_length; filter_supported
        # keeps whichever the installed SFTConfig actually accepts.
        max_length=int(cfg.get("max_seq_length", cfg.get("max_length", 2048))),
        max_seq_length=int(cfg.get("max_seq_length", cfg.get("max_length", 2048))),
        packing=bool(cfg.get("packing", True)),
        # With packing off, group similar-length samples to cut padding waste.
        group_by_length=bool(cfg.get("group_by_length", not cfg.get("packing", True))),
        dataset_text_field="text",
        ddp_find_unused_parameters=False,
    )
    if eval_ds is not None:
        sft_kwargs["eval_strategy"] = cfg.get("eval_strategy", "steps")
        sft_kwargs["eval_steps"] = int(cfg.get("eval_steps", 100))
    sft_kwargs = filter_supported(SFTConfig, sft_kwargs)
    sft_args = SFTConfig(**sft_kwargs)

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    t0 = time.time()
    train_result = trainer.train()
    duration = time.time() - t0

    # ---- save ----
    adapter_dir = run_dir / "final_adapter"
    if is_main:
        trainer.save_model(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        print(f"adapter saved: {adapter_dir}")

        history = getattr(train_result, "global_step", None)
        log_history = trainer.state.log_history
        peak_mem = {
            f"cuda{i}": torch.cuda.max_memory_allocated(i) / 1e9
            for i in range(torch.cuda.device_count())
        }
        meta = {
            "run_name": run_name,
            "git_head": git_head(),
            "config": cfg,
            "model_path": str(model_dir),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "world_size": world_size,
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "train_samples": len(train_ds),
            "eval_samples": len(eval_ds) if eval_ds else 0,
            "global_steps": history,
            "duration_sec": round(duration, 1),
            "peak_mem_gb": {k: round(v, 2) for k, v in peak_mem.items()},
            "final_log_history": log_history[-10:],
            "adapter_dir": str(adapter_dir),
        }
        (run_dir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
        print(json.dumps({"steps": history, "duration_sec": meta["duration_sec"],
                          "peak_mem_gb": meta["peak_mem_gb"]}, indent=2))
    trainer.barrier() if hasattr(trainer, "barrier") else None
    print("TRAIN_DONE")


if __name__ == "__main__":
    main()
