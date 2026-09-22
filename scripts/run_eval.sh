#!/usr/bin/env bash
# Automatic evaluation: bash scripts/run_eval.sh configs/eval_base.yaml
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$PROJECT_ROOT"
python scripts/evaluate.py --config "${1:-configs/eval_base.yaml}"
