#!/usr/bin/env python3
"""Build a reproducible offline BFCL v4 subset in this project's eval schema.

The Berkeley Function Calling Leaderboard (BFCL) v4 ships static, offline
categories under gorilla/berkeley-function-call-leaderboard/bfcl_eval/data.
Live / multi-turn categories require external accounts and are excluded.

This script converts a fixed-size, deterministically sampled subset of the
static categories into eval/tool_calling_eval.jsonl-compatible cases, so the
same evaluate.py harness (native <tool_call> format, exact-argument scoring,
Strict Protocol Success) can score Base / SFT-v1 / SFT-v2 on an external,
independently authored benchmark.

Scoring note: BFCL's official checker is lenient (each argument accepts a
list of valid values and optional args may be omitted). This project's
metric is intentionally strict: we canonicalize each answer to its first
non-empty accepted value, require every canonicalized argument exactly, and
treat extra arguments as failures. Numbers therefore are NOT comparable to
official BFCL leaderboard scores; they are a reproducible external yardstick
under the project's own Strict Protocol Success definition.

Inputs (downloaded once, cached under .cache/bfcl):
  BFCL_v4_simple_python.json / BFCL_v4_multiple.json / BFCL_v4_parallel.json /
  BFCL_v4_parallel_multiple.json / BFCL_v4_irrelevance.json
  answers/BFCL_v4_<cat>.json (irrelevance has no answer file: no call expected)

Usage:
  python scripts/build_bfcl_subset.py --cache .cache/bfcl --out eval/bfcl_subset.jsonl
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
from qwen_tool_sft.dataio import write_jsonl  # noqa: E402

# category file -> (our category label, subset size, has ground-truth answers)
CATEGORIES = [
    ("BFCL_v4_simple_python", "single_tool", 60, True),
    ("BFCL_v4_multiple", "multiple_tools", 40, True),
    ("BFCL_v4_parallel", "parallel_same_tool", 50, True),
    ("BFCL_v4_parallel_multiple", "parallel_diff_tool", 30, True),
    ("BFCL_v4_irrelevance", "no_tool", 40, False),
]
SEED = 20260923

UPSTREAM_BASE = (
    "https://api.github.com/repos/ShishirPatil/gorilla/contents/"
    "berkeley-function-call-leaderboard/bfcl_eval/data"
)


def load_jsonl(p: Path) -> list:
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize_property(prop: dict) -> dict:
    """Map a BFCL parameter property to an OpenAI-compatible property."""
    out = {}
    t = prop.get("type")
    if isinstance(t, list):  # union type: keep the first concrete type
        t = next((x for x in t if x not in ("null", "NoneType", None)), t[0])
    t = {"dict": "object", "float": "number", "integer": "integer",
         "string": "string", "boolean": "boolean",
         "array": "array", "list": "array", "bool": "boolean"}.get(t, t)
    if t:
        out["type"] = t
    if "description" in prop:
        out["description"] = prop["description"]
    if "enum" in prop:
        out["enum"] = prop["enum"]
    items = prop.get("items")
    if items is None and "array_item_type" in prop:
        items = {"type": prop["array_item_type"]}
    if items:
        if isinstance(items, dict):
            out["items"] = normalize_property(items)
        else:
            out["items"] = {"type": str(items)}
    return out


def normalize_function(fn: dict) -> dict:
    """Convert a BFCL function definition to an OpenAI tool object."""
    params_in = fn.get("parameters", {}) or {}
    properties = {name: normalize_property(p)
                  for name, p in (params_in.get("properties", {}) or {}).items()}
    required = params_in.get("required", []) or []
    params_out = {"type": "object", "properties": properties}
    if required:
        params_out["required"] = required
    return {"type": "function",
            "function": {"name": fn["name"],
                         "description": fn.get("description", ""),
                         "parameters": params_out}}


def canonical_arg_value(accepted):
    """Pick the canonical value from BFCL's accepted-value list.

    Returns None when the argument is optional-and-omittable (only ""/None
    accepted variants); callers omit such arguments from the strict expect.
    """
    if not isinstance(accepted, list):
        return accepted
    non_empty = [v for v in accepted if v not in ("", None)]
    if not non_empty:
        return None
    return non_empty[0]


def convert_tool_row(row: dict, gt_calls: list, category: str, case_id: str) -> dict:
    tools = [normalize_function(fn) for fn in row.get("function", [])]
    calls = []
    for gt in gt_calls:
        (name, argmap), = gt.items()
        args = {}
        for arg, accepted in (argmap or {}).items():
            val = canonical_arg_value(accepted)
            if val is not None:
                args[arg] = val
        calls.append({"name": name, "arguments": args})
    return {
        "id": case_id,
        "category": category,
        "messages": row["question"][0],
        "tools": tools,
        "expect": {"mode": "tool", "calls": calls},
        "benchmark_source": "BFCL_v4",
    }


def convert_no_tool_row(row: dict, case_id: str) -> dict:
    tools = [normalize_function(fn) for fn in row.get("function", [])]
    return {
        "id": case_id,
        "category": "no_tool",
        "messages": row["question"][0],
        # BFCL irrelevance rows ship an unrelated tool spec: keep it as a
        # distractor so abstention is a real decision.
        "tools": tools,
        "expect": {"mode": "no_tool", "calls": []},
        "benchmark_source": "BFCL_v4",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=".cache/bfcl")
    ap.add_argument("--out", default="eval/bfcl_subset.jsonl")
    args = ap.parse_args()

    cache = paths.resolve_path(args.cache)
    rng = random.Random(SEED)
    cases, manifest_cats = [], {}

    for stem, category, n, has_answers in CATEGORIES:
        rows = load_jsonl(cache / f"{stem}.json")
        answers = {}
        if has_answers:
            for a in load_jsonl(cache / "answers" / f"{stem}.json"):
                answers[a["id"]] = a["ground_truth"]
        order = list(range(len(rows)))
        rng.shuffle(order)
        picked, missing = [], 0
        for idx in order:
            if len(picked) >= n:
                break
            row = rows[idx]
            if has_answers and row["id"] not in answers:
                missing += 1
                continue
            case_id = f"bfcl-{category}-{row['id']}"
            if has_answers:
                picked.append(convert_tool_row(row, answers[row["id"]], category, case_id))
            else:
                picked.append(convert_no_tool_row(row, case_id))
        cases.extend(picked)
        manifest_cats[stem] = {"our_category": category, "pool_rows": len(rows),
                               "selected": len(picked), "skipped_no_answer": missing}
        print(f"{stem}: pool={len(rows)} selected={len(picked)} skipped={missing}")

    out = paths.resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out, cases)

    manifest = {
        "name": "BFCL v4 offline reproducible subset",
        "upstream": "ShishirPatil/gorilla berkeley-function-call-leaderboard/bfcl_eval/data",
        "upstream_base_url": UPSTREAM_BASE,
        "seed": SEED,
        "total_cases": len(cases),
        "categories": manifest_cats,
        "excluded": ["live_* (external accounts)", "multi_turn_* (multi-turn protocol)",
                     "memory", "web_search", "format_sensitivity",
                     "simple_java / simple_javascript (language focus: python)"],
        "scoring_note": (
            "Cases are scored with this project's Strict Protocol Success "
            "(native <tool_call> wrapper, exact arguments, clean stop). BFCL's "
            "official checker is lenient (lists of accepted values, optional "
            "arguments may be omitted); answers are canonicalized to the first "
            "non-empty accepted value. Numbers are NOT comparable to official "
            "BFCL leaderboard scores."),
    }
    manifest_path = out.with_name(out.stem + "_manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    print(f"wrote {len(cases)} cases -> {out}")
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
