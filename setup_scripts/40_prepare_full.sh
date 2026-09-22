#!/bin/bash
set +e
ROOT=/mnt/ssd2/psf/job/qwen-tool-calling-sft
source "$ROOT/scripts/env.sh"
cd "$ROOT"
LOG="$ROOT/setup_logs/40_prepare_full.log"
exec > >(tee "$LOG") 2>&1
rm -rf data/full
python scripts/prepare_data.py --config configs/data_full.yaml
echo "PREPARE_FULL_EXIT=$?"
wc -l data/full/train.jsonl data/full/eval.jsonl 2>/dev/null
cat data/full/stats.json 2>/dev/null
echo
du -sh data/full
echo "PREPARE_FULL_DONE"
