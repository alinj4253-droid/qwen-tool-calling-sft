#!/usr/bin/env python3
"""Fetch the 7 tool-calling datasets' raw files DIRECTLY from hf-mirror.

The datasets library times out against hf-mirror on this network, while plain
HTTPS file downloads (via aria2c) work reliably. This downloads the exact repo
files into .cache/datasets_raw/<repo>/; converters._load() prefers that local
mirror and falls back to the datasets hub when absent.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qwen_tool_sft import paths  # noqa: E402

MIRROR = "https://hf-mirror.com"

# Explicit file manifest (verified against each repo's tree on 2026-09-22).
MANIFEST: dict[str, list[str]] = {
    "Deepexi/function-calling-small": [
        "function-calling-small_aliyun_openapi_V2.csv",
    ],
    "llamafactory/glaive_toolcall_zh": [
        "glaive_toolcall_zh_1k.json",
    ],
    "hiyouga/glaive-function-calling-v2-sharegpt": [
        "glaive_toolcall.json",
    ],
    "NousResearch/hermes-function-calling-v1": [
        "func-calling.json",
        "func-calling-singleturn.json",
        "glaive-function-calling-5k.json",
    ],
    "tryumanshow/ToolACE-Qwen-cleaned": [
        "data/train-00000-of-00001.parquet",
    ],
    "nohurry/Opus-4.6-Reasoning-3000x-filtered": [
        "distilled_corpus_400k_with_cot-filtered.jsonl",
    ],
    "bellfire/openclaw-coder-dataset": [
        "data/train.jsonl",
        "data/eval.jsonl",
    ],
}


def tree(repo: str) -> list[dict]:
    url = f"{MIRROR}/api/datasets/{repo}/tree/main?recursive=true"
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
        "Accept": "application/json",
    })
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def aria(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "aria2c", "-x8", "-s8", "-k1M", "--file-allocation=none",
        "--continue=true", "--max-tries=8", "--retry-wait=5",
        "--connect-timeout=15", "--timeout=30",
        "--max-connection-per-server=8", "--min-split-size=1M",
        "--console-log-level=warn", "--summary-interval=20",
        "-d", str(dest.parent), "-o", dest.name, url,
    ]
    print("  $", " ".join(cmd[:4]), "...", dest.name, flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="", help="comma-separated repo substrings to fetch")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()

    wanted = {s.strip() for s in args.sources.split(",") if s.strip()}
    raw_root = paths.CACHE_DIR / "datasets_raw"

    for repo, files in MANIFEST.items():
        if wanted and not any(w in repo for w in wanted):
            continue
        print(f"== {repo}")
        try:
            remote = {f["path"]: f.get("size") for f in tree(repo)
                      if f.get("type") == "file"}
        except Exception as e:
            print(f"  !! tree failed: {e}; using manifest only")
            remote = {}
        for rel in files:
            dest = raw_root / repo / rel
            exp = remote.get(rel)
            incomplete = Path(str(dest) + ".aria2").exists()
            if not args.force and dest.exists() and not incomplete and (
                    exp is None or dest.stat().st_size == exp):
                print(f"  ok (exists, {dest.stat().st_size} bytes): {rel}")
                continue
            if incomplete:
                print(f"  resuming incomplete: {rel}")
            if args.verify_only:
                print(f"  MISSING/SIZE MISMATCH: {rel}")
                continue
            url = f"{MIRROR}/datasets/{repo}/resolve/main/{rel}"
            aria(url, dest)
            if exp and dest.stat().st_size != exp:
                print(f"  !! size mismatch {rel}: {dest.stat().st_size} != {exp}")
    print("FETCH_DATASETS_DONE")


if __name__ == "__main__":
    main()
