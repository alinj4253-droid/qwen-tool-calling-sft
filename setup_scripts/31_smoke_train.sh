#!/bin/bash
# Dual-4090 DDP LoRA smoke training.
set +e
ROOT=/mnt/ssd2/psf/job/qwen-tool-calling-sft
source "$ROOT/scripts/env.sh"
cd "$ROOT"
LOG="$ROOT/setup_logs/31_smoke_train.log"
exec > >(tee "$LOG") 2>&1

echo "===== GPU STATE BEFORE TRAIN ====="
nvidia-smi
echo "===== compute apps ====="
nvidia-smi --query-compute-apps=pid,used_memory --format=csv

echo "===== data check ====="
wc -l data/smoke/train.jsonl data/smoke/eval.jsonl

echo "===== accelerate dual-GPU smoke train ====="
accelerate launch --config_file configs/accelerate_dual4090.yaml \
  scripts/train.py --config configs/train_smoke.yaml
echo "TRAIN_EXIT=$?"

echo "===== artifacts ====="
ls -lah runs/qwen3-4b-smoke/ runs/qwen3-4b-smoke/final_adapter/ 2>/dev/null
echo "SMOKE_TRAIN_DONE"
