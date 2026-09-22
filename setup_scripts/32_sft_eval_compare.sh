#!/bin/bash
# Evaluate the smoke SFT adapter and compare Base vs SFT.
set +e
ROOT=/mnt/ssd2/psf/job/qwen-tool-calling-sft
source "$ROOT/scripts/env.sh"
cd "$ROOT"
LOG="$ROOT/setup_logs/32_sft_eval_compare.log"
exec > >(tee "$LOG") 2>&1

echo "===== GPU STATE ====="
nvidia-smi

echo "===== sft smoke eval ====="
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py --config configs/eval_sft.yaml
echo "SFT_EVAL_EXIT=$?"

echo "===== compare base vs sft ====="
python scripts/compare_runs.py runs/eval_base runs/eval_sft_smoke \
  --output runs/comparison_smoke.md
echo "COMPARE_EXIT=$?"
cat runs/comparison_smoke.md
echo "SFT_EVAL_COMPARE_DONE"
