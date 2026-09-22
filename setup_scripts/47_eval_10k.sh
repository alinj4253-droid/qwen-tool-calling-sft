#!/bin/bash
# Evaluate the 10k SFT adapter and compare Base vs SFT-10k.
set +e
ROOT=/mnt/ssd2/psf/job/qwen-tool-calling-sft
source "$ROOT/scripts/env.sh"
cd "$ROOT"
LOG="$ROOT/setup_logs/47_eval_10k.log"
exec > >(tee "$LOG") 2>&1
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py --config configs/eval_sft_10k.yaml
echo "SFT10K_EVAL_EXIT=$?"
python scripts/compare_runs.py runs/eval_base runs/eval_sft_10k \
  --output runs/comparison_10k.md
cat runs/comparison_10k.md
echo "EVAL_10K_DONE"
