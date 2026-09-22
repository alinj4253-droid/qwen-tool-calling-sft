#!/usr/bin/env python3
"""Compare metrics of two eval runs (Base vs SFT).

  python scripts/compare_runs.py runs/eval_base runs/eval_sft
  python scripts/compare_runs.py path/base path/sft --output runs/comparison.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qwen_tool_sft.metrics import metrics_table  # noqa: E402


def load(run_dir: Path) -> tuple[dict, str]:
    m = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    return m, m.get("run_name", run_dir.name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base_run")
    ap.add_argument("sft_run")
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    base, base_name = load(Path(args.base_run))
    sft, sft_name = load(Path(args.sft_run))
    table = metrics_table(base, base_name, sft, sft_name)

    # per-category comparison
    cats = sorted(set(base.get("by_category", {})) | set(sft.get("by_category", {})))
    lines = ["", "## By category (Overall Exact Match)", "",
             f"| Category | {base_name} | {sft_name} | Delta |", "|---|---:|---:|---:|"]
    for c in cats:
        a = base.get("by_category", {}).get(c, {}).get("accuracy", float("nan"))
        b = sft.get("by_category", {}).get(c, {}).get("accuracy", float("nan"))
        lines.append(f"| {c} | {a*100:.1f}% | {b*100:.1f}% | {(b-a)*100:+.1f}pt |")
    report = f"# Base vs SFT: tool-calling evaluation\n\n{table}\n" + "\n".join(lines) + "\n"
    print(report)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"saved: {out}")


if __name__ == "__main__":
    main()
