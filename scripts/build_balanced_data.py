#!/usr/bin/env python3
"""Build a balanced targeted SFT mix fixing BOTH v2 residual failures.

Evidence from the expanded benchmark (see runs/eval_v2_*):
  * multi-call (parallel) cases are 0% for both v1 and v2 -> upweight
    parallel rows, same-name first;
  * v2 collapses on wrong-tool traps (tools present, must abstain) and
    BFCL irrelevance, i.e. it became over-eager to call tools -> explicitly
    upweight trap-like rows (tools present, no call) and pure-chat rows.

Composition (default, 4000 rows):
  10% parallel (60% same-name, oversampled with replacement),
  10% trap-like (tools present, assistant makes no tool call),
  10% pure chat (no tools exposed),
  70% other ordinary rows.

  python scripts/build_balanced_data.py --src data/full_v2/train.jsonl \
      --out data/targeted_bal --total 4000 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qwen_tool_sft import paths  # noqa: E402
from qwen_tool_sft.dataio import read_jsonl, write_jsonl  # noqa: E402


def call_name_lists(sample):
    out = []
    for m in sample.get("messages", []):
        tcs = m.get("tool_calls") or []
        if len(tcs) >= 2:
            out.append([(tc.get("function", tc) or {}).get("name") for tc in tcs])
    return out


def has_any_call(sample):
    return any(m.get("tool_calls") for m in sample.get("messages", []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/full_v2/train.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--total", type=int, default=4000)
    ap.add_argument("--parallel-fraction", type=float, default=0.10)
    ap.add_argument("--trap-fraction", type=float, default=0.10)
    ap.add_argument("--chat-fraction", type=float, default=0.10)
    ap.add_argument("--same-name-share", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = read_jsonl(paths.resolve_path(args.src))

    parallel, same_name, diff_name = [], [], []
    trap_like, pure_chat, ordinary = [], [], []
    for s in rows:
        names = call_name_lists(s)
        tools = bool(s.get("tools"))
        calls = has_any_call(s)
        if names:
            parallel.append(s)
            (same_name if any(len(set(ns)) < len(ns) for ns in names) else diff_name).append(s)
        elif tools and not calls:
            trap_like.append(s)
        elif not tools:
            pure_chat.append(s)
        else:
            ordinary.append(s)
    print(f"pool: parallel={len(parallel)} (same={len(same_name)}, diff={len(diff_name)}) "
          f"trap={len(trap_like)} pure_chat={len(pure_chat)} ordinary={len(ordinary)}")

    n_same = round(args.total * args.parallel_fraction * args.same_name_share)
    n_diff = round(args.total * args.parallel_fraction) - n_same
    n_trap = round(args.total * args.trap_fraction)
    n_chat = round(args.total * args.chat_fraction)

    def take(pool, n, allow_oversample=False):
        if n == 0:
            return []
        if len(pool) >= n:
            return rng.sample(pool, n)
        if not allow_oversample:
            out = list(pool)
            rng.shuffle(out)
            return out
        out = list(pool)
        out.extend(rng.choices(pool, k=n - len(pool)))
        return out

    picked_same = take(same_name, n_same, allow_oversample=True)
    picked_diff = take(diff_name, n_diff, allow_oversample=True)
    picked_trap = take(trap_like, n_trap)
    picked_chat = take(pure_chat, n_chat)
    reserved = {id(s) for grp in (picked_same, picked_diff, picked_trap, picked_chat) for s in grp}
    rest = [s for s in rows if id(s) not in reserved]
    n_rest = args.total - len(picked_same) - len(picked_diff) - len(picked_trap) - len(picked_chat)
    picked_rest = take(rest, min(n_rest, len(rest)))

    mix = picked_same + picked_diff + picked_trap + picked_chat + picked_rest
    rng.shuffle(mix)

    out_dir = paths.resolve_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "train.jsonl", mix)

    # small distinct eval slice
    eval_rows = (take(diff_name, 10) + take(trap_like, 15) + take(pure_chat, 15))
    rng.shuffle(eval_rows)
    write_jsonl(out_dir / "eval.jsonl", eval_rows)

    def n_parallel(xs):
        return sum(1 for s in xs if call_name_lists(s))

    stats = {
        "source": args.src, "seed": args.seed, "total": len(mix),
        "quota": {"parallel_same_name": n_same, "parallel_diff_name": n_diff,
                  "trap_like": n_trap, "pure_chat": n_chat, "other": n_rest},
        "pool_sizes": {"parallel": len(parallel), "same_name": len(same_name),
                       "diff_name": len(diff_name), "trap_like": len(trap_like),
                       "pure_chat": len(pure_chat), "ordinary": len(ordinary)},
        "oversampled_same_name": max(0, n_same - len(same_name)),
        "actual_parallel": n_parallel(mix),
        "actual_parallel_fraction": round(n_parallel(mix) / max(len(mix), 1), 4),
        "actual_trap_fraction": round(len(picked_trap) / len(mix), 4),
        "actual_pure_chat_fraction": round(len(picked_chat) / len(mix), 4),
        "eval_rows": len(eval_rows),
    }
    (out_dir / "balanced_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
