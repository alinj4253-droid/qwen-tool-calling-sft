#!/bin/bash
set +e
ROOT=/mnt/ssd2/psf/job/qwen-tool-calling-sft
echo "===== stop current prepare_data (explicit PIDs) ====="
for p in $(pgrep -f "prepare_data.py --config configs/data_smoke.yaml"); do
  kill "$p" 2>/dev/null && echo "killed $p"
done
for p in $(pgrep -f "setup_scripts/19_prepare_smoke.sh"); do
  kill "$p" 2>/dev/null && echo "killed wrapper $p"
done
sleep 2
cd "$ROOT"
setsid bash setup_scripts/19_prepare_smoke.sh >/dev/null 2>&1 < /dev/null &
sleep 5
ps -u wmy -o pid,cmd --no-headers | grep -E 'prepare_data|19_prepare' | grep -v grep
echo RERUN_LAUNCHED
