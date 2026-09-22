#!/usr/bin/env python3
"""Download a base model INTO the project directory (models_local/).

Primary: ModelScope (fast from CN networks). Fallback: HuggingFace mirror.
Files land directly in models_local/<model_id> (deterministic local path).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qwen_tool_sft import paths  # noqa: E402


def via_modelscope(model_id: str, target: Path) -> Path:
    from modelscope import snapshot_download

    print(f"[modelscope] downloading {model_id} -> {target}")
    p = snapshot_download(model_id, local_dir=str(target))
    return Path(p)


def via_hf(model_id: str, target: Path) -> Path:
    from huggingface_hub import snapshot_download

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    print(f"[hf-mirror] downloading {model_id} -> {target}")
    p = snapshot_download(repo_id=model_id, local_dir=str(target))
    return Path(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Base")
    ap.add_argument("--source", choices=["auto", "modelscope", "hf"], default="auto")
    args = ap.parse_args()

    paths.ensure_project_dirs()
    target = paths.MODELS_DIR / args.model
    target.parent.mkdir(parents=True, exist_ok=True)
    if (target / "config.json").exists():
        print(f"already present: {target}")
        print(target)
        return

    try:
        if args.source in ("auto", "modelscope"):
            p = via_modelscope(args.model, target)
        else:
            p = via_hf(args.model, target)
    except Exception as e:
        print(f"modelscope failed: {type(e).__name__}: {e}", file=sys.stderr)
        if args.source == "modelscope":
            raise
        print("falling back to hf-mirror ...", file=sys.stderr)
        p = via_hf(args.model, target)

    if not (Path(p) / "config.json").exists():
        raise RuntimeError(f"download finished but config.json missing under {p}")
    print(f"MODEL_READY {p}")


if __name__ == "__main__":
    main()
