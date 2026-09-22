#!/usr/bin/env bash
# Smoke data: 500 train / 100 eval via HF mirror streaming.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$PROJECT_ROOT"
python scripts/prepare_data.py --config configs/data_smoke.yaml "$@"
wc -l data/smoke/train.jsonl data/smoke/eval.jsonl
