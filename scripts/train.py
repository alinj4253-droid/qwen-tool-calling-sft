#!/usr/bin/env python3
"""LoRA SFT for Qwen tool-calling on one or two RTX 4090s.

Two loss modes (config key ``loss_mode``):
  * assistant_only (SFT-v2 default): only assistant turns (text, native
    <tool_call> blocks and <|im_end|>) contribute to the loss; system/user/
    tool-schema/tool-response tokens are masked to -100. See
    qwen_tool_sft.loss_mask.
  * full: v1 behaviour, the whole rendered sequence is trained.

Two special-token strategies (config key ``special_token_training``):
  * selective (SFT-v2 default): PEFT trainable_token_indices trains ONLY the
    chat/tool special-token rows (~6 rows) instead of full embed/lm_head
    copies. ~0.8% trainable params vs ~16.8% for modules_to_save.
  * modules_to_save: v1 behaviour (full trainable embed_tokens/lm_head copies).

Launch (dual GPU DDP):
  accelerate launch --config_file configs/accelerate_dual4090.yaml \
      scripts/train.py --config configs/train_sft_v2_30k.yaml
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
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
from qwen_tool_sft.dataio import read_jsonl, render_text, validate_sample, normalize_for_template  # noqa: E402
from qwen_tool_sft.loss_mask import AssistantOnlyCollator, tokenize_assistant_only  # noqa: E402

# chat/tool protocol tokens whose embedding/lm-head rows are untrained in
# Qwen3-4B-Base and therefore have to be updated selectively.
SPECIAL_TOKEN_STRINGS = [
    "<|im_start|>", "<|im_end|>", "<tool_call>", "</tool_call>",
    "<tool_response>", "</tool_response>",
]


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


def load_assistant_only_dataset(jsonl_path: Path, tokenizer, max_length: int | None):
    """Pre-tokenized dataset with assistant-only labels."""
    from datasets import Dataset

    rows = read_jsonl(jsonl_path)
    feats, n_invalid, n_too_long = [], 0, 0
    total_tokens, total_loss = 0, 0
    for sample in rows:
        ok, reason = validate_sample(sample)
        if not ok:
            n_invalid += 1
            continue
        messages = normalize_for_template(sample.get("messages", []))
        tok = tokenize_assistant_only(
            tokenizer, messages, sample.get("tools"), max_length=None)
        if max_length and len(tok["input_ids"]) > max_length:
            n_too_long += 1
            continue
        feats.append({
            "input_ids": tok["input_ids"],
            "attention_mask": tok["attention_mask"],
            "labels": tok["labels"],
        })
        total_tokens += len(tok["input_ids"])
        total_loss += sum(1 for x in tok["labels"] if x != -100)
    print(f"  {jsonl_path.name}: {len(feats)} assistant-only examples "
          f"(invalid={n_invalid}, dropped_too_long>{max_length}={n_too_long})")
    if feats:
        print(f"  tokens/example={total_tokens/len(feats):.0f} "
              f"loss_tokens/example={total_loss/len(feats):.0f} "
              f"({100*total_loss/max(total_tokens,1):.1f}% supervised)")
    ds = Dataset.from_list(feats)
    stats = {"examples": len(feats), "invalid": n_invalid, "too_long": n_too_long,
             "total_tokens": total_tokens, "loss_tokens": total_loss}
    return ds, stats


def resolve_special_token_indices(tokenizer, cfg: dict) -> list[int]:
    if cfg.get("trainable_token_indices"):
        return [int(x) for x in cfg["trainable_token_indices"]]
    ids = []
    for s in SPECIAL_TOKEN_STRINGS:
        tid = tokenizer.convert_tokens_to_ids(s)
        if isinstance(tid, int) and tid >= 0:
            ids.append(tid)
    return sorted(set(ids))


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
    run_dir = paths.resolve_path(cfg.get("output_subdir", f"runs/{run_name}"))
    if is_main:
        run_dir.mkdir(parents=True, exist_ok=True)

    model_dir = paths.model_path(cfg["model_path"])
    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(f"model not found at {model_dir}; run scripts/download_model.py")

    loss_mode = cfg.get("loss_mode", "full")
    special_mode = cfg.get("special_token_training", "modules_to_save")
    if is_main:
        print("=" * 60)
        print(f"run: {run_name} | world_size={world_size} local_rank={local_rank}")
        print(f"loss_mode={loss_mode} special_token_training={special_mode}")
        print(f"model: {model_dir}\nrun_dir: {run_dir}")
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

    # Optional continued-training stage: merge an existing adapter (e.g. the
    # v2 SFT adapter) into the base in memory, then attach a fresh LoRA.
    base_adapter = cfg.get("base_adapter_path", "")
    if base_adapter:
        from peft import PeftModel
        base_adapter_dir = paths.resolve_path(base_adapter)
        if is_main:
            print(f"continued training from adapter: {base_adapter_dir}")
        model = PeftModel.from_pretrained(model, base_adapter_dir)
        model = model.merge_and_unload()
        model.config.use_cache = False

    special_indices = resolve_special_token_indices(tokenizer, cfg)
    if is_main:
        print("resolved special token rows:", json.dumps(
            {s: tokenizer.convert_tokens_to_ids(s) for s in SPECIAL_TOKEN_STRINGS},
            ensure_ascii=False))

    lora_kwargs = dict(
        r=int(cfg.get("lora_r", 16)),
        lora_alpha=int(cfg.get("lora_alpha", 32)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        target_modules=cfg.get("target_modules",
                               ["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"]),
        task_type="CAUSAL_LM",
        bias="none",
    )
    if special_mode == "selective":
        # PEFT trainable_token_indices: only the selected embedding/lm-head
        # rows are trained; lm_head shares the delta under tied embeddings.
        lora_kwargs["trainable_token_indices"] = special_indices
        lora_kwargs["modules_to_save"] = None
    elif special_mode == "none":
        lora_kwargs["modules_to_save"] = None
    else:
        lora_kwargs["modules_to_save"] = cfg.get(
            "modules_to_save", ["embed_tokens", "lm_head"])
    lora_cfg = LoraConfig(**lora_kwargs)
    model = get_peft_model(model, lora_cfg)
    if cfg.get("gradient_checkpointing", True):
        model.enable_input_require_grads()
    if is_main:
        model.print_trainable_parameters()

    # ---- data ----
    data_dir = paths.resolve_path(cfg.get("data_dir", "data"))
    max_length = int(cfg.get("max_seq_length", cfg.get("max_length", 2048)))
    cap_train = cfg.get("max_train_samples")
    train_stats = {}
    if loss_mode == "assistant_only":
        train_ds, train_stats = load_assistant_only_dataset(
            data_dir / "train.jsonl", tokenizer, max_length)
        if cfg.get("packing", False):
            raise ValueError("assistant_only loss is incompatible with packing")
    elif loss_mode == "full":
        train_ds = load_text_dataset(data_dir / "train.jsonl", tokenizer)
    else:
        raise ValueError(f"unknown loss_mode {loss_mode}")
    if cap_train:
        train_ds = train_ds.select(range(min(int(cap_train), len(train_ds))))
        print(f"  capped train to {len(train_ds)}")

    eval_ds = None
    eval_jsonl = data_dir / "eval.jsonl"
    eval_stats = {}
    if eval_jsonl.exists() and cfg.get("do_eval", False):
        cap_eval = int(cfg.get("max_eval_samples", 0)) or None
        if loss_mode == "assistant_only":
            eval_ds, eval_stats = load_assistant_only_dataset(
                eval_jsonl, tokenizer, max_length)
        else:
            eval_ds = load_text_dataset(eval_jsonl, tokenizer)
        if cap_eval:
            eval_ds = eval_ds.select(range(min(cap_eval, len(eval_ds))))
            print(f"  capped eval to {len(eval_ds)}")

    # ---- trainer args ----
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
        max_length=max_length,
        max_seq_length=max_length,
        packing=False,
        group_by_length=bool(cfg.get("group_by_length", loss_mode == "full")),
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )
    if loss_mode == "full":
        sft_kwargs["dataset_text_field"] = "text"
    if eval_ds is not None:
        sft_kwargs["eval_strategy"] = cfg.get("eval_strategy", "steps")
        sft_kwargs["eval_steps"] = int(cfg.get("eval_steps", 100))
    sft_kwargs = filter_supported(SFTConfig, sft_kwargs)
    sft_args = SFTConfig(**sft_kwargs)

    collator = None
    if loss_mode == "assistant_only":
        collator = AssistantOnlyCollator(pad_token_id=tokenizer.pad_token_id)

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=collator,
    )

    t0 = time.time()
    train_result = trainer.train()
    duration = time.time() - t0

    # ---- per-rank peak memory (every rank writes, rank0 aggregates) ----
    rank_peak = {
        "local_rank": local_rank,
        "world_size": world_size,
        "device_index_visible": torch.cuda.current_device(),
        "device_name": torch.cuda.get_device_name(0),
        "peak_mem_allocated_gb": round(torch.cuda.max_memory_allocated(0) / 1e9, 3),
        "peak_mem_reserved_gb": round(torch.cuda.max_memory_reserved(0) / 1e9, 3),
    }
    (run_dir / f"peak_mem_rank{local_rank}.json").write_text(
        json.dumps(rank_peak, indent=2), encoding="utf-8")
    _barrier(trainer)

    # ---- save ----
    adapter_dir = run_dir / "final_adapter"
    if is_main:
        trainer.save_model(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        print(f"adapter saved: {adapter_dir}")

        # wait briefly for all rank files to be visible on shared fs
        peak_files = sorted(glob.glob(str(run_dir / "peak_mem_rank*.json")))
        for _ in range(50):
            if len(peak_files) >= world_size:
                break
            time.sleep(2)
            peak_files = sorted(glob.glob(str(run_dir / "peak_mem_rank*.json")))
        peak_per_rank = {}
        for pf in peak_files:
            d = json.loads(Path(pf).read_text())
            r = d["local_rank"]
            peak_per_rank[f"rank{r}_allocated_gb"] = d["peak_mem_allocated_gb"]
            peak_per_rank[f"rank{r}_reserved_gb"] = d["peak_mem_reserved_gb"]

        adapter_bytes = sum(
            p.stat().st_size for p in adapter_dir.rglob("*") if p.is_file())
        n_trainable, n_total = trainable_param_counts(model)
        history = getattr(train_result, "global_step", None)
        log_history = trainer.state.log_history
        meta = {
            "run_name": run_name,
            "git_head": git_head(),
            "config": cfg,
            "model_path": str(model_dir),
            "loss_mode": loss_mode,
            "special_token_training": special_mode,
            "trainable_token_indices": special_indices,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "world_size": world_size,
            "gpu_names": [torch.cuda.get_device_name(i) for i in
                          range(torch.cuda.device_count())],
            "train_samples": len(train_ds),
            "train_token_stats": train_stats,
            "eval_samples": len(eval_ds) if eval_ds else 0,
            "eval_token_stats": eval_stats,
            "global_steps": history,
            "duration_sec": round(duration, 1),
            "peak_mem_per_rank": peak_per_rank,
            "trainable_params": n_trainable,
            "all_params": n_total,
            "trainable_pct": round(100 * n_trainable / max(n_total, 1), 4),
            "adapter_bytes": adapter_bytes,
            "adapter_size_mb": round(adapter_bytes / 1e6, 1),
            "final_log_history": log_history[-10:],
            "adapter_dir": str(adapter_dir),
        }
        (run_dir / "run_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"steps": history, "duration_sec": meta["duration_sec"],
                          "peak_mem_per_rank": peak_per_rank,
                          "trainable_params": n_trainable,
                          "adapter_size_mb": meta["adapter_size_mb"]}, indent=2))
    _barrier(trainer)
    print("TRAIN_DONE")


def _barrier(trainer):
    try:
        if trainer is not None and hasattr(trainer, "barrier"):
            trainer.barrier()
        else:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
    except Exception:
        pass


def trainable_param_counts(model) -> tuple[int, int]:
    trainable, total = 0, 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    return trainable, total


if __name__ == "__main__":
    main()
