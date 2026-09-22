#!/bin/bash
set +e
if [ -f "$HOME/.git-credentials" ]; then
  echo "git-credentials hosts:"
  sed -E 's#https://[^:]*:[^@]*@#https://***:***@#g' "$HOME/.git-credentials" | cut -d/ -f1-3
else
  echo "no .git-credentials"
fi
ssh -o StrictHostKeyChecking=accept-new -T git@github.com 2>&1 | head -3
echo AUTHCHECK_DONE
