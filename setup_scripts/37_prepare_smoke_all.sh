#!/bin/bash
set +e
ROOT=/mnt/ssd2/psf/job/qwen-tool-calling-sft
source "$ROOT/scripts/env.sh"
cd "$ROOT"
LOG="$ROOT/setup_logs/37_prepare_smoke_all.log"
exec > >(tee "$LOG") 2>&1
rm -rf data/smoke
python scripts/prepare_data.py --config configs/data_smoke.yaml
echo "PREPARE_EXIT=$?"
wc -l data/smoke/train.jsonl data/smoke/eval.jsonl 2>/dev/null
cat data/smoke/stats.json 2>/dev/null
echo
echo "===== train head ====="
head -n 1 data/smoke/train.jsonl | cut -c1-700
echo
echo "PREPARE_SMOKE_ALL_DONE"
