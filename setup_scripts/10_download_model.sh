#!/bin/bash
set +e
ROOT=/mnt/ssd2/psf/job/qwen-tool-calling-sft
source "$ROOT/scripts/env.sh"
cd "$ROOT"
LOG="$ROOT/setup_logs/10_download_model.log"
exec > >(tee "$LOG") 2>&1
python scripts/download_model.py --model Qwen/Qwen3-4B-Base
echo "DL_EXIT=$?"
echo "===== model dir ====="
ls -lah models_local/Qwen/Qwen3-4B-Base | head -40
du -sh models_local/Qwen/Qwen3-4B-Base
echo "DOWNLOAD_DONE"
