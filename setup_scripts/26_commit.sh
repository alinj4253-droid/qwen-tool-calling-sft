#!/bin/bash
set -e
ROOT=/mnt/ssd2/psf/job/qwen-tool-calling-sft
cd "$ROOT"
GIT="git -c user.name=Mazycity57 -c user.email=wangmengyi57@gmail.com"

$GIT add configs/data_smoke.yaml configs/train_smoke.yaml scripts/env.sh \
  scripts/evaluate.py scripts/prepare_data.py scripts/run_prepare_smoke.sh \
  scripts/train.py src/qwen_tool_sft/config.py src/qwen_tool_sft/converters.py \
  tests/test_config.py
$GIT commit -m "fix: robust config path resolution, mirror retries/timeouts, eval eos set, smoke data dirs"

$GIT add README.md PROJECT_LOG.md setup_scripts
$GIT commit -m "docs: README and PROJECT_LOG; chore: track setup scripts"

$GIT status --short
$GIT log --oneline -5
