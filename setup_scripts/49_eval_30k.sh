#!/bin/bash
# Evaluate the 30k SFT adapter and compare Base vs SFT-30k.
set +e
ROOT=/mnt/ssd2/psf/job/qwen-tool-calling-sft
source "$ROOT/scripts/env.sh"
cd "$ROOT"
LOG="$ROOT/setup_logs/49_eval_30k.log"
exec > >(tee "$LOG") 2>&1
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py --config configs/eval_sft_30k.yaml
echo "SFT30K_EVAL_EXIT=$?"
python scripts/compare_runs.py runs/eval_base runs/eval_sft_30k \
  --output runs/comparison_30k.md
cat runs/comparison_30k.md
echo "EVAL_30K_DONE"
