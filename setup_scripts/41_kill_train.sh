#!/bin/bash
set +e
pkill -f "scripts/train.py"
sleep 3
pkill -9 -f "scripts/train.py"
sleep 2
echo "remaining:"
pgrep -af "train.py" || echo none
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo KILL_DONE
