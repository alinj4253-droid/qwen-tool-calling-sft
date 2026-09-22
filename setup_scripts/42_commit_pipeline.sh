#!/bin/bash
set +e
ROOT=/mnt/ssd2/psf/job/qwen-tool-calling-sft
cd "$ROOT"
git add src/qwen_tool_sft/parser.py tests/test_eval_parser.py scripts/train.py \
        configs/train_smoke.yaml configs/train_sft_10k.yaml configs/train_sft_30k.yaml \
        .gitignore setup_scripts
git -c user.name=Mazycity57 -c user.email=wangmengyi57@gmail.com \
  commit -m "eval: stricter parser (definitions vs calls, leading parallel bare JSON); train: disable packing under SDPA, input-require-grads, cap in-training eval"
git log --oneline -3
echo COMMIT_DONE
