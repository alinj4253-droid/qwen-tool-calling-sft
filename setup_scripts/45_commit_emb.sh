#!/bin/bash
set +e
ROOT=/mnt/ssd2/psf/job/qwen-tool-calling-sft
cd "$ROOT"
git add scripts/train.py configs/train_smoke.yaml configs/train_sft_10k.yaml \
        configs/train_sft_30k.yaml setup_scripts
git -c user.name=Mazycity57 -c user.email=wangmengyi57@gmail.com \
  commit -m "train: modules_to_save embed_tokens/lm_head (base special-token rows untrained), adamw_8bit"
git log --oneline -3
echo COMMIT_DONE
