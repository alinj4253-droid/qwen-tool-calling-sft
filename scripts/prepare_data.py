#!/usr/bin/env python3
"""Prepare unified tool-calling SFT data from the 7 upstream datasets.

All outputs are written INSIDE the project directory (default: <root>/data).

Examples:
  # smoke subset (default)
  python scripts/prepare_data.py --max-train-samples 500 --max-eval-samples 100
  # full data (v2: messages+tools canonical dedup)
  python scripts/prepare_data.py --config configs/data_full_v2.yaml
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qwen_tool_sft import paths  # noqa: E402
from qwen_tool_sft.config import load_config  # noqa: E402
from qwen_tool_sft.converters import CONVERTERS  # noqa: E402
from qwen_tool_sft.dataio import validate_sample, write_jsonl  # noqa: E402
from qwen_tool_sft.dedup import canonical_json, dedup_samples  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="", help="optional YAML config (CLI args override it)")
    ap.add_argument("--output-dir", default="", help="must stay inside the project directory")
    ap.add_argument("--max-train-samples", type=int, default=-1)
    ap.add_argument("--max-eval-samples", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--sources", default="", help="comma separated converter names; default all")
    ap.add_argument("--no-stream", action="store_true",
                    help="download full datasets instead of streaming caps")
    args = ap.parse_args()

    cfg = {}
    if args.config:
        cfg = load_config(args.config)
    output_dir_arg = args.output_dir or cfg.get("output_dir", str(paths.DATA_DIR))
    max_train = args.max_train_samples if args.max_train_samples >= 0 else cfg.get(
        "max_train_samples", 500)
    max_eval = args.max_eval_samples if args.max_eval_samples >= 0 else cfg.get(
        "max_eval_samples", 100)
    seed = args.seed if args.seed >= 0 else cfg.get("seed", 42)
    want_stream = cfg.get("stream", True) and not args.no_stream

    out_dir = paths.resolve_path(output_dir_arg)
    out_dir.mkdir(parents=True, exist_ok=True)

    wanted = {s.strip() for s in args.sources.split(",") if s.strip()}
    converters = [(n, f) for n, f in CONVERTERS if not wanted or n in wanted]

    need = max_train + max_eval
    finite = need > 0
    # Spread the cap evenly across sources (each source yields up to `cap` rows).
    per_source_cap = None
    if finite:
        per_source_cap = max(need, 200)
    streaming = (finite and want_stream)

    print("=" * 60)
    print(f"Qwen tool-calling data prep | streaming={streaming} "
          f"per_source_cap={per_source_cap}")
    print(f"output: {out_dir}")
    print("=" * 60)

    all_samples: list[dict] = []
    stats: dict[str, int] = {}
    failures: dict[str, str] = {}
    for name, fn in converters:
        t0 = time.time()
        try:
            samples = fn(cap=per_source_cap, streaming=streaming)
            stats[name] = len(samples)
            all_samples.extend(samples)
        except Exception as e:  # one source must not kill the whole prep
            stats[name] = 0
            failures[name] = f"{type(e).__name__}: {e}"
            print(f"  !! {name} failed after {time.time()-t0:.0f}s: {e}")

    print(f"\nraw total: {len(all_samples)}")

    # v2 dedup: fingerprint covers BOTH canonical messages AND canonical tools,
    # so "same user query + different tool schemas" samples are NOT collapsed.
    valid_samples = []
    n_invalid = 0
    for s in all_samples:
        ok, reason = validate_sample(s)
        if not ok:
            n_invalid += 1
            continue
        valid_samples.append(s)
    deduped, n_dup = dedup_samples(valid_samples)

    # diagnostic: how often do identical messages appear with DIFFERENT tools
    # (the exact case the v1 messages-only hash wrongly collapsed)?
    by_messages: dict[str, set[str]] = {}
    for s in deduped:
        mk = canonical_json(s.get("messages", []))
        by_messages.setdefault(mk, set()).add(canonical_json(s.get("tools") or []))
    n_same_msg_diff_tools = sum(1 for v in by_messages.values() if len(v) > 1)

    print(f"dedup: {len(all_samples)} -> {len(deduped)} "
          f"(invalid dropped: {n_invalid}, duplicates removed: {n_dup})")
    print(f"same-messages/different-tools groups kept: {n_same_msg_diff_tools}")

    rng = random.Random(seed)
    rng.shuffle(deduped)

    if finite and len(deduped) < need:
        print(f"WARNING: only {len(deduped)} valid samples available (< {need} requested)")

    n_train = max_train if finite else int(len(deduped) * 0.9)
    n_eval = max_eval if finite else len(deduped) - n_train
    train = deduped[:n_train]
    eval_ = deduped[n_train:n_train + n_eval] if finite else deduped[n_train:]

    train_path = out_dir / "train.jsonl"
    eval_path = out_dir / "eval.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(eval_path, eval_)

    with_tools = sum(1 for s in deduped if s.get("tools"))
    with_parallel = sum(
        1 for s in deduped
        if any(len(m.get("tool_calls") or []) >= 2 for m in s.get("messages", []))
    )
    with_same_name_parallel = sum(
        1 for s in deduped if _has_same_name_parallel(s))
    stats_doc = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "seed": seed,
        "dedup": "md5(canonical(messages) || canonical(tools))",
        "streaming": streaming,
        "per_source_cap": per_source_cap,
        "per_source_counts": stats,
        "failures": failures,
        "raw_total": len(all_samples),
        "invalid_dropped": n_invalid,
        "duplicates_removed": n_dup,
        "same_messages_different_tools_groups": n_same_msg_diff_tools,
        "deduped_total": len(deduped),
        "train": len(train),
        "eval": len(eval_),
        "with_tools": with_tools,
        "with_tools_pct": round(100 * with_tools / len(deduped), 2) if deduped else 0,
        "with_parallel_calls": with_parallel,
        "with_parallel_calls_pct": round(100 * with_parallel / len(deduped), 3) if deduped else 0,
        "with_same_name_parallel": with_same_name_parallel,
    }
    (out_dir / "stats.json").write_text(json.dumps(stats_doc, ensure_ascii=False, indent=2),
                                        encoding="utf-8")

    print("\n" + "=" * 60)
    for k, v in stats.items():
        print(f"  {k:24s}: {v:>7d}")
    if failures:
        print("  failed sources:")
        for k, v in failures.items():
            print(f"    {k}: {v}")
    print(f"  train: {len(train)} -> {train_path}")
    print(f"  eval : {len(eval_)} -> {eval_path}")
    print(f"  with tools: {stats_doc['with_tools_pct']}%")
    print(f"  parallel calls: {with_parallel} ({stats_doc['with_parallel_calls_pct']}%)")
    print("=" * 60)


def _has_same_name_parallel(sample: dict) -> bool:
    for m in sample.get("messages", []):
        tcs = m.get("tool_calls") or []
        if len(tcs) >= 2:
            names = [(tc.get("function", tc) or {}).get("name") for tc in tcs]
            if len(set(names)) < len(names):
                return True
    return False


if __name__ == "__main__":
    main()
