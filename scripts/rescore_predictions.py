#!/usr/bin/env python3
"""Re-score saved predictions.jsonl files with the CURRENT parser/metrics.

Avoids re-running GPU generation when only scoring logic changed. Rewrites
metrics.json and summary.md in each run directory.

Usage:
  python scripts/rescore_predictions.py runs/eval_base runs/eval_sft_smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qwen_tool_sft.dataio import iter_jsonl  # noqa: E402
from qwen_tool_sft.metrics import (  # noqa: E402
    aggregate, metrics_table, prompt_leaked, score_case,
)
from qwen_tool_sft.parser import parse_tool_calls  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--eval-set", default="eval/tool_calling_eval.jsonl")
    ap.add_argument("--model-path", default="models_local/Qwen/Qwen3-4B-Base")
    args = ap.parse_args()

    terminators = ["<|im_end|>"]
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(PROJECT_ROOT / args.model_path)
        for tid in (tok.eos_token_id,):
            t = tok.decode([tid], skip_special_tokens=False)
            if t and t not in terminators:
                terminators.append(t)
    except Exception as e:  # pragma: no cover
        print(f"[warn] could not load tokenizer eos ({e})")
    terminators = tuple(terminators)

    cases = {c["id"]: c for c in iter_jsonl(PROJECT_ROOT / args.eval_set)}

    for rd in args.run_dirs:
        run_dir = PROJECT_ROOT / rd
        pred_path = run_dir / "predictions.jsonl"
        rows = list(iter_jsonl(pred_path))
        scores = []
        for r in rows:
            case = cases.get(r["id"], {"id": r["id"], "category": r.get("category"),
                                       "expect": r.get("expect")})
            raw = r["raw_output"]
            parsed = parse_tool_calls(raw)
            sc = score_case(case, parsed, raw=raw, terminators=terminators)
            stored = r.get("score", {})
            if stored.get("terminated"):
                sc.clean_stop = sc.clean_stop or not prompt_leaked(raw)
            scores.append(sc)
        metrics = aggregate(scores)
        old = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        for keep in ("run_name", "model_path", "adapter_path", "eval_seconds"):
            if keep in old:
                metrics[keep] = old[keep]
        (run_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        name = metrics.get("run_name", run_dir.name)
        (run_dir / "summary.md").write_text(
            f"# Eval: {name}\n\n" + metrics_table(metrics, name) + "\n\n"
            f"Cases: {metrics['n_cases']} (tool: {metrics['n_tool_cases']}, "
            f"no-tool: {metrics['n_no_tool_cases']})\n",
            encoding="utf-8")
        print(f"rescored {rd}: overall={metrics['overall_exact_match']:.3f} "
              f"canonical={metrics['canonical_format_rate']:.3f} "
              f"clean_stop={metrics['clean_stop_rate']:.3f}")
    print("RESCORE_DONE")


if __name__ == "__main__":
    main()
