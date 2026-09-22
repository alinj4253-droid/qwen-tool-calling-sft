#!/bin/bash
set +e
ROOT=/mnt/ssd2/psf/job/qwen-tool-calling-sft
source "$ROOT/scripts/env.sh"
cd "$ROOT"
LOG="$ROOT/setup_logs/38_verify_pytest.log"
exec > >(tee "$LOG") 2>&1
python -m pytest -q 2>&1 | tail -n 15
echo "PYTEST_EXIT=${PIPESTATUS[0]}"
python -m compileall -q src scripts
echo "COMPILE_EXIT=$?"
echo "===== sample tool-call row ====="
python - <<'PYEOF'
import json
n = 0
for line in open("data/smoke/train.jsonl", encoding="utf-8"):
    r = json.loads(line)
    for m in r["messages"]:
        if m.get("tool_calls"):
            print(json.dumps(r, ensure_ascii=False)[:800])
            n += 1
            break
    if n >= 1:
        break
PYEOF
echo "VERIFY_DONE"
