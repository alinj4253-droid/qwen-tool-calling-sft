#!/bin/bash
set +e
ROOT=/mnt/ssd2/psf/job/qwen-tool-calling-sft
source "$ROOT/scripts/env.sh"
cd "$ROOT"
LOG="$ROOT/setup_logs/29_fetch_data.log"
exec > >(tee "$LOG") 2>&1
python scripts/fetch_datasets.py "$@"
echo "FETCH_EXIT=$?"
du -sh .cache/datasets_raw/* 2>/dev/null
echo "FETCH_WRAPPER_DONE"
