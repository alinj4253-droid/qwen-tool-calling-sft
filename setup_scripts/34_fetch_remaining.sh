#!/bin/bash
set +e
ROOT=/mnt/ssd2/psf/job/qwen-tool-calling-sft
source "$ROOT/scripts/env.sh"
cd "$ROOT"
LOG="$ROOT/setup_logs/34_fetch_remaining.log"
exec > >(tee "$LOG") 2>&1
python scripts/fetch_datasets.py --sources nohurry,openclaw
echo "FETCH_REMAINING_EXIT=$?"
du -sh .cache/datasets_raw/nohurry .cache/datasets_raw/bellfire 2>/dev/null
echo "FETCH_REMAINING_DONE"
