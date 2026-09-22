#!/bin/bash
set +e
ROOT=/mnt/ssd2/psf/job/qwen-tool-calling-sft
cd "$ROOT"
git add README.md PROJECT_LOG.md runs/ setup_scripts/54_final_acceptance.sh
echo "----- staged files -----"
git diff --cached --name-only
echo "----- size guard (must be no large blobs) -----"
git diff --cached --name-only -z | xargs -0 -I{} du -b "{}" 2>/dev/null | sort -nr | head -5
git -c user.name=Mazycity57 -c user.email=wangmengyi57@gmail.com \
  commit -m "results: 10k/30k SFT training+eval artifacts, metrics/summaries/comparisons, README results, full PROJECT_LOG"
git log --oneline -6
echo COMMIT_RESULTS_DONE
