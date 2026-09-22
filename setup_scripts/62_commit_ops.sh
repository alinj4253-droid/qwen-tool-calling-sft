#!/bin/bash
set +e
ROOT=/mnt/ssd2/psf/job/qwen-tool-calling-sft
cd "$ROOT"
git add setup_scripts/22_rerun_prepare.sh setup_scripts/54_final_acceptance.sh setup_scripts/61_track_helper.sh
git -c user.name=Mazycity57 -c user.email=wangmengyi57@gmail.com commit -m "chore: de-identify usernames in ops scripts"
git status --short
echo CLEAN_CHECK_DONE
