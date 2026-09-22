#!/bin/bash
# Stage-2 formal SFT: 30k samples, r=32, dual-4090 DDP.
set +e
ROOT=/mnt/ssd2/psf/job/qwen-tool-calling-sft
source "$ROOT/scripts/env.sh"
cd "$ROOT"
LOG="$ROOT/setup_logs/48_train_30k.log"
exec > >(tee "$LOG") 2>&1

echo "===== GPU STATE BEFORE TRAIN ====="
nvidia-smi
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
wc -l data/full/train.jsonl data/full/eval.jsonl

echo "===== accelerate dual-GPU 30k SFT (r=32) ====="
accelerate launch --config_file configs/accelerate_dual4090.yaml \
  scripts/train.py --config configs/train_sft_30k.yaml
echo "TRAIN30K_EXIT=$?"

ls -lah runs/qwen3-4b-sft-30k-r32/ runs/qwen3-4b-sft-30k-r32/final_adapter/ 2>/dev/null
echo "TRAIN_30K_DONE"
