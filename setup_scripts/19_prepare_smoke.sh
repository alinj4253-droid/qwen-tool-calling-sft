#!/bin/bash
set +e
ROOT=/mnt/ssd2/psf/job/qwen-tool-calling-sft
source "$ROOT/scripts/env.sh"
cd "$ROOT"
LOG="$ROOT/setup_logs/19_prepare_smoke.log"
exec > >(tee "$LOG") 2>&1
python scripts/prepare_data.py --config configs/data_smoke.yaml
echo "PREPARE_EXIT=$?"
echo "===== outputs ====="
wc -l data/smoke/train.jsonl data/smoke/eval.jsonl 2>/dev/null
cat data/smoke/stats.json 2>/dev/null
echo
echo "===== head sample ====="
head -c 1200 data/smoke/train.jsonl 2>/dev/null
echo
echo "PREPARE_SMOKE_DONE"
