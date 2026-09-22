#!/bin/bash
set +e
for p in $(pgrep -f "prepare_data.py --config configs/data_smoke.yaml"); do
  kill "$p" 2>/dev/null && echo "killed prepare $p"
done
for p in $(pgrep -f "setup_scripts/19_prepare_smoke.sh"); do
  kill "$p" 2>/dev/null && echo "killed wrapper $p"
done
sleep 2
echo "remaining prepare procs: $(pgrep -f prepare_data.py | wc -l)"
