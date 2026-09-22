#!/usr/bin/env python3
"""Re-score an existing predictions.jsonl after metric/parser changes.

Does NOT re-run generation. Rebuilds cases from the eval set and re-parses
raw outputs, preserving the recorded token-level termination signal.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qwen_tool_sft import paths  # noqa: E402
from qwen_tool_sft.dataio import read_jsonl  # noqa: E402
from qwen_tool_sft.metrics import (  # noqa: E402
    aggregate, cleanly_terminated, metrics_table, prompt_leaked, score_case,
)
from qwen_tool_sft.parser import parse_tool_calls  # noqa: E402

TERMINATOR_STRINGS = ("<|im_end|>",)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--cases", default="eval/tool_calling_eval.jsonl")
    ap.add_argument("--output-dir", default="")
    args = ap.parse_args()

    pred_path = paths.resolve_path(args.predictions)
    cases = {c["id"]: c for c in read_jsonl(paths.resolve_path(args.cases))}
    preds = read_jsonl(pred_path)

    scores, rows = [], []
    for p in preds:
        cid = p["id"]
        case = cases.get(cid)
        if case is None:
            print(f"!! missing case {cid}")
            continue
        raw = p.get("raw_output", "")
        parsed = parse_tool_calls(raw)
        sc = score_case(case, parsed, raw=raw, terminators=TERMINATOR_STRINGS)
        if "terminated_on_token" in p:
            sc.clean_stop = bool(p["terminated_on_token"]) and not prompt_leaked(raw)
        else:
            sc.clean_stop = cleanly_terminated(raw, TERMINATOR_STRINGS)
        scores.append(sc)
        rows.append({
            "id": cid, "category": sc.category,
            "raw_output": raw,
            "terminated_on_token": p.get("terminated_on_token", sc.clean_stop),
            "parsed_names": sc.predicted_names,
            "expected_names": sc.expected_names,
            "valid_format": sc.valid_format,
            "canonical_format": sc.canonical_format,
            "tool_selection_correct": sc.tool_selection_correct,
            "arg_key_accuracy": sc.arg_key_accuracy,
            "arg_exact_match": sc.arg_exact_match,
            "no_tool_correct": sc.no_tool_correct,
            "clean_stop": sc.clean_stop,
            "strict_protocol_success": sc.strict_protocol_success,
            "overall": sc.overall,
        })

    metrics = aggregate(scores)
    out_dir = paths.resolve_path(args.output_dir) if args.output_dir else pred_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics_rescored.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "predictions_rescored.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    print(metrics_table(metrics, "rescored"))
    print(f"\nsaved: {out_dir}/metrics_rescored.json")


if __name__ == "__main__":
    main()
