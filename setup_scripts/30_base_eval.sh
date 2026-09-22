#!/bin/bash
# Base model automatic evaluation on GPU0 (single process).
set +e
ROOT=/mnt/ssd2/psf/job/qwen-tool-calling-sft
source "$ROOT/scripts/env.sh"
cd "$ROOT"
LOG="$ROOT/setup_logs/30_base_eval.log"
exec > >(tee "$LOG") 2>&1

echo "===== GPU STATE BEFORE EVAL ====="
nvidia-smi
echo "===== other GPU compute procs (expect only ours) ====="
nvidia-smi --query-compute-apps=pid,used_memory --format=csv

echo "===== base eval ====="
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py --config configs/eval_base.yaml
echo "BASE_EVAL_EXIT=$?"
echo "BASE_EVAL_DONE"
