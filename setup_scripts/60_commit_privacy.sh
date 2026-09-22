#!/bin/bash
set +e
ROOT=/mnt/ssd2/psf/job/qwen-tool-calling-sft
cd "$ROOT"
git add PRIVACY.md README.md PROJECT_LOG.md tests/test_paths.py \
        setup_scripts/57_gh_auth_check.sh setup_scripts/58_secret_scan.sh \
        setup_scripts/59_sanitize.sh data/smoke/stats.json data/full/stats.json
git status --short
git -c user.name=Mazycity57 -c user.email=wangmengyi57@gmail.com \
  commit -m "privacy: add PRIVACY.md, de-identify usernames/home paths, include data stats for analysis"
echo COMMIT_PRIVACY_DONE
git log --oneline -3
