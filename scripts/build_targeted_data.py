#!/usr/bin/env python3
"""Build a parallel-targeted SFT mix (task section 11).

Holds out the independently generated benchmark for evaluation and builds a
small training mix whose parallel-call share is raised from the natural
~1% to a target fraction (5% then 10%), with same-name parallel calls
(pc-01/pc-02 failure mode) oversampled first.

  python scripts/build_targeted_data.py --src data/full_v2/train.jsonl \
      --out data/targeted_p05 --total 4000 --parallel-fraction 0.05 --seed 42
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


def call_names(sample):
    names = []
    for m in sample.get("messages", []):
        tcs = m.get("tool_calls") or []
        if len(tcs) >= 2:
            row = [(tc.get("function", tc) or {}).get("name") for tc in tcs]
            names.append(row)
    return names


def is_parallel(sample):
    return bool(call_names(sample))


def is_same_name_parallel(sample):
    return any(len(set(ns)) < len(ns) for ns in call_names(sample))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/full_v2/train.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--total", type=int, default=4000)
    ap.add_argument("--parallel-fraction", type=float, default=0.05)
    ap.add_argument("--same-name-share", type=float, default=0.6,
                    help="fraction of parallel slots filled with same-name calls")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = read_jsonl(paths.resolve_path(args.src))
    parallel = [s for s in rows if is_parallel(s)]
    same_name = [s for s in rows if is_same_name_parallel(s)]
    diff_name = [s for s in parallel if not is_same_name_parallel(s)]
    ordinary = [s for s in rows if not is_parallel(s)]
    print(f"source: {len(rows)} | parallel={len(parallel)} "
          f"(same-name={len(same_name)}, diff-name={len(diff_name)}) ordinary={len(ordinary)}")

    n_parallel = round(args.total * args.parallel_fraction)
    n_same = round(n_parallel * args.same_name_share)
    n_diff = n_parallel - n_same

    def sample_pool(pool, n, allow_oversample=True):
        if n == 0:
            return []
        if len(pool) >= n:
            return rng.sample(pool, n)
        if not allow_oversample:
            return list(pool)
        # pool too small: take all, then resample with replacement
        out = list(pool)
        out.extend(rng.choices(pool, k=n - len(pool)))
        rng.shuffle(out)
        return out

    picked_same = sample_pool(same_name, n_same)
    picked_diff = sample_pool(diff_name, n_diff)
    n_ordinary = args.total - len(picked_same) - len(picked_diff)
    picked_ord = sample_pool(ordinary, min(n_ordinary, len(ordinary)),
                             allow_oversample=False)

    mix = picked_same + picked_diff + picked_ord
    rng.shuffle(mix)

    out_dir = paths.resolve_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "train.jsonl", mix)

    # small eval slice drawn from the SAME pools but distinct rows where possible
    picked_same_ids = {id(s) for s in picked_same}
    picked_diff_ids = {id(s) for s in picked_diff}
    eval_same = sample_pool([s for s in same_name if id(s) not in picked_same_ids],
                            min(20, len(same_name)), allow_oversample=False)
    eval_diff = sample_pool([s for s in diff_name if id(s) not in picked_diff_ids],
                            min(20, len(diff_name)), allow_oversample=False)
    eval_ord = sample_pool(ordinary, 20, allow_oversample=False)
    eval_mix = eval_same + eval_diff + eval_ord
    rng.shuffle(eval_mix)
    write_jsonl(out_dir / "eval.jsonl", eval_mix)

    stats = {
        "source": args.src, "seed": args.seed, "total": len(mix),
        "target_parallel_fraction": args.parallel_fraction,
        "same_name_slots": n_same, "diff_name_slots": n_diff,
        "source_pool_sizes": {"all": len(rows), "parallel": len(parallel),
                              "same_name": len(same_name),
                              "diff_name": len(diff_name), "ordinary": len(ordinary)},
        "picked": {"same_name": len(picked_same), "diff_name": len(picked_diff),
                   "ordinary": len(picked_ord)},
        "oversampled_same_name": max(0, n_same - len(same_name)),
        "oversampled_diff_name": max(0, n_diff - len(diff_name)),
        "actual_parallel_fraction": round(
            (len(picked_same) + len(picked_diff)) / max(len(mix), 1), 4),
        "eval_rows": len(eval_mix),
    }
    (out_dir / "targeted_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
