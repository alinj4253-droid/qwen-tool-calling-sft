#!/usr/bin/env bash
# Source before running any project command:
#   source scripts/env.sh
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ "$PROJECT_ROOT" != "/mnt/ssd2/psf/job/qwen-tool-calling-sft" ]; then
  echo "[env.sh] WARNING: unexpected PROJECT_ROOT=$PROJECT_ROOT" >&2
fi

# isolated conda env (never install into base or other existing envs)
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate qwen_tool_sft

# all model/dataset caches stay INSIDE the project directory
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="$PROJECT_ROOT/.cache/hf"
export HUGGINGFACE_HUB_CACHE="$PROJECT_ROOT/.cache/hf/hub"
export HF_DATASETS_CACHE="$PROJECT_ROOT/.cache/hf/datasets"
export MODELSCOPE_CACHE="$PROJECT_ROOT/.cache/modelscope"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export PYTHONHASHSEED=42
# mirror can be slow; don't give up after the 10s default
export HF_HUB_DOWNLOAD_TIMEOUT=60
export HF_HUB_ETAG_TIMEOUT=30
export HF_HUB_REQUEST_TIMEOUT=30

mkdir -p "$HF_HOME/hub" "$HF_HOME/datasets" "$MODELSCOPE_CACHE"
echo "[env.sh] PROJECT_ROOT=$PROJECT_ROOT  CONDA_DEFAULT_ENV=$CONDA_DEFAULT_ENV  HF_ENDPOINT=$HF_ENDPOINT"
