#!/bin/bash
set +e
ROOT=/mnt/ssd2/psf/job/qwen-tool-calling-sft
cd "$ROOT"
cat .gitignore
echo "===== status ====="
git status --short | head -30
git add scripts/fetch_datasets.py src/qwen_tool_sft/converters.py setup_scripts
git -c user.name=Mazycity57 -c user.email=wangmengyi57@gmail.com \
  commit -m "data: direct hf-mirror raw-file fetch + local-mirror converters (all 7 sources verified)"
git log --oneline -3
echo COMMIT_DONE
