#!/bin/bash
# Stage-1 formal SFT: 10k samples, dual-4090 DDP LoRA + trained embed/head.
set +e
ROOT=/mnt/ssd2/psf/job/qwen-tool-calling-sft
source "$ROOT/scripts/env.sh"
cd "$ROOT"
LOG="$ROOT/setup_logs/46_train_10k.log"
exec > >(tee "$LOG") 2>&1

echo "===== GPU STATE BEFORE TRAIN ====="
nvidia-smi
echo "===== compute apps ====="
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
echo "===== data check ====="
wc -l data/full/train.jsonl data/full/eval.jsonl

echo "===== accelerate dual-GPU 10k SFT ====="
accelerate launch --config_file configs/accelerate_dual4090.yaml \
  scripts/train.py --config configs/train_sft_10k.yaml
echo "TRAIN10K_EXIT=$?"

echo "===== artifacts ====="
ls -lah runs/qwen3-4b-sft-10k-r16/ runs/qwen3-4b-sft-10k-r16/final_adapter/ 2>/dev/null
echo "TRAIN_10K_DONE"
