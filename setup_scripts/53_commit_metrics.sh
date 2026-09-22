#!/bin/bash
set +e
ROOT=/mnt/ssd2/psf/job/qwen-tool-calling-sft
cd "$ROOT"
git add src/qwen_tool_sft/parser.py src/qwen_tool_sft/metrics.py \
        scripts/evaluate.py scripts/rescore_predictions.py tests/test_metrics.py \
        configs/eval_sft_10k.yaml configs/eval_sft_30k.yaml setup_scripts
git -c user.name=Mazycity57 -c user.email=wangmengyi57@gmail.com \
  commit -m "eval: canonical-format/clean-stop protocol metrics, trailing-punct normalization, offline rescore tool; 10k/30k eval configs"
git log --oneline -3
echo COMMIT_DONE
