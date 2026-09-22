#!/usr/bin/env python3
"""Train/benchmark decontamination check (task P0-3).

  python scripts/check_contamination.py \
      --train data/full_v2/train.jsonl \
      --benchmark eval/tool_calling_eval.jsonl eval/benchmark_v1.jsonl \
      --out runs/contamination_report --threshold 0.8

Exit code 2 if any exact query match is found; 1 if near duplicates exist;
0 when the benchmark is clean.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qwen_tool_sft import paths  # noqa: E402
from qwen_tool_sft.contamination import run_contamination  # noqa: E402
from qwen_tool_sft.dataio import read_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--benchmark", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    train_path = paths.resolve_path(args.train)
    train_samples = read_jsonl(train_path)
    bench = []
    for bp in args.benchmark:
        p = paths.resolve_path(bp)
        rows = read_jsonl(p)
        for r in rows:
            r.setdefault("benchmark_source", p.name)
        bench.extend(rows)
    print(f"train={len(train_samples)} benchmark={len(bench)}")

    report = run_contamination(train_samples, bench,
                               threshold=args.threshold, top_k=args.top_k)
    out_dir = paths.resolve_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(report["summary"], ensure_ascii=False, indent=2), encoding="utf-8")
    for name in ("exact_matches", "near_duplicates", "top_matches"):
        (out_dir / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in report[name]),
            encoding="utf-8")

    s = report["summary"]
    print(json.dumps(s, ensure_ascii=False, indent=2))
    print(f"saved: {out_dir}")
    if s["n_exact_query_match"] > 0:
        print("CONTAMINATION_EXACT_FOUND")
        sys.exit(2)
    if s["n_near_duplicate_cases"] > 0:
        print("CONTAMINATION_NEAR_FOUND")
        sys.exit(1)
    print("CONTAMINATION_CLEAN")


def benchmark_guard(rows):
    return rows


if __name__ == "__main__":
    main()
