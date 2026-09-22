#!/bin/bash
set +e
ROOT=/mnt/ssd2/psf/job/qwen-tool-calling-sft
cd "$ROOT"
git add setup_scripts/54_final_acceptance.sh setup_scripts/55_commit_results.sh
git -c user.name=Mazycity57 -c user.email=wangmengyi57@gmail.com commit -m "results: 10k/30k SFT training and eval artifacts, comparisons, README results, full PROJECT_LOG"
git log --oneline -6
echo COMMIT_FINAL_DONE
