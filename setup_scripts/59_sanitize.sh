#!/bin/bash
# De-identify low-sensitivity environment markers before publishing to GitHub.
set -e
ROOT=/mnt/ssd2/psf/job/qwen-tool-calling-sft
cd "$ROOT"
python - <<'PYEOF'
from pathlib import Path
repls = {
    "PROJECT_LOG.md": [("/home/wmy/anaconda3", "~/anaconda3（/home/<linux-user>/anaconda3）")],
    "tests/test_paths.py": [("/home/wmy/secret", "/home/other-user/secret")],
    "setup_scripts/22_rerun_prepare.sh": [("ps -u wmy ", 'ps -u "$(whoami)" ')],
    "setup_scripts/54_final_acceptance.sh": [("ps -u wmy ", 'ps -u "$(whoami)" ')],
}
for f, pairs in repls.items():
    p = Path(f)
    s = p.read_text(encoding="utf-8")
    for a, b in pairs:
        s = s.replace(a, b)
    p.write_text(s, encoding="utf-8")
    print("sanitized", f)
PYEOF
# small data stats are useful for downstream analysis and contain no PII
git add -f data/smoke/stats.json data/full/stats.json
echo SANITIZE_DONE
