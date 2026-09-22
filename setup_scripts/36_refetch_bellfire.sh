#!/bin/bash
set +e
ROOT=/mnt/ssd2/psf/job/qwen-tool-calling-sft
source "$ROOT/scripts/env.sh"
cd "$ROOT"
LOG="$ROOT/setup_logs/36_refetch_bellfire.log"
exec > >(tee "$LOG") 2>&1
# remove truncated train.jsonl from the killed aria run
rm -f .cache/datasets_raw/bellfire/openclaw-coder-dataset/data/train.jsonl
python scripts/fetch_datasets.py --sources openclaw
echo "REFETCH_EXIT=$?"
ls -la .cache/datasets_raw/bellfire/openclaw-coder-dataset/data/
echo "===== jsonl integrity ====="
wc -l .cache/datasets_raw/bellfire/openclaw-coder-dataset/data/*.jsonl
echo "BELLFIRE_DONE"
