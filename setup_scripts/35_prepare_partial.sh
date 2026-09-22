#!/bin/bash
set +e
ROOT=/mnt/ssd2/psf/job/qwen-tool-calling-sft
source "$ROOT/scripts/env.sh"
cd "$ROOT"
LOG="$ROOT/setup_logs/35_prepare_partial.log"
exec > >(tee "$LOG") 2>&1
python scripts/prepare_data.py --config configs/data_smoke.yaml \
  --sources deepexi_zh,glaive_zh,glaive_v2_en,hermes_en,toolace_en
echo "PREPARE_EXIT=$?"
wc -l data/smoke/train.jsonl data/smoke/eval.jsonl 2>/dev/null
cat data/smoke/stats.json 2>/dev/null
echo
echo "===== head ====="
head -c 1000 data/smoke/train.jsonl 2>/dev/null
echo
echo "PREPARE_PARTIAL_DONE"
