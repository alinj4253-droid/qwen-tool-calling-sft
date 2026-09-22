#!/bin/bash
set +e
ROOT=/mnt/ssd2/psf/job/qwen-tool-calling-sft
cd "$ROOT"
git add setup_scripts/60_commit_privacy.sh
git -c user.name=Mazycity57 -c user.email=wangmengyi57@gmail.com commit -m "chore: track privacy commit helper" >/dev/null 2>&1
git status --short
echo TRACK_DONE
