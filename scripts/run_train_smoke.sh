#!/usr/bin/env bash
# Dual-RTX-4090 DDP LoRA smoke training.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$PROJECT_ROOT"
nvidia-smi
accelerate launch --config_file configs/accelerate_dual4090.yaml \
  scripts/train.py --config configs/train_smoke.yaml "$@"
