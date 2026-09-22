#!/bin/bash
set +e
ROOT=/mnt/ssd2/psf/job/qwen-tool-calling-sft
source "$ROOT/scripts/env.sh"
cd "$ROOT"
python - <<'PYEOF'
import json
from collections import Counter
for name in ["data/smoke/train.jsonl", "data/full/train.jsonl"]:
    n=0; multi=0; tool_rows=0; turns=Counter()
    with open(name, encoding="utf-8") as f:
        for line in f:
            r=json.loads(line); n+=1
            has=False
            for m in r["messages"]:
                tcs=m.get("tool_calls")
                if tcs:
                    has=True
                    turns[len(tcs)]+=1
                    if len(tcs)>=2: multi+=1
            if has: tool_rows+=1
    print(name, "rows",n,"tool_rows",tool_rows,"rows_with_parallel_turn",multi)
    print("  tool_calls-per-turn distribution:", dict(sorted(turns.items())[:8]))
PYEOF
echo COUNT_DONE
