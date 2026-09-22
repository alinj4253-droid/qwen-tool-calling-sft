#!/bin/bash
set +e
ROOT=/mnt/ssd2/psf/job/qwen-tool-calling-sft
D="$ROOT/models_local/Qwen/Qwen3-4B-Base"
LOG="$ROOT/setup_logs/25_finish_download.log"
exec > >(tee "$LOG") 2>&1
BASE="https://www.modelscope.cn/models/Qwen/Qwen3-4B-Base/resolve/master"

echo "waiting for in-flight aria2c processes..."
while pgrep -f "aria2c.*Qwen3-4B-Base" >/dev/null 2>&1; do
  sleep 10
done
echo "in-flight downloads finished"

finish_shard() {
  local f="$1"
  local n=0
  while [ -f "$D/$f.aria2" ] && [ $n -lt 8 ]; do
    n=$((n+1))
    echo "=== resume attempt $n for $f ==="
    aria2c -x8 -s8 -k1M --file-allocation=none -d "$D" -o "$f" \
      --continue=true --max-tries=5 --retry-wait=5 \
      --summary-interval=30 --console-log-level=warn \
      "$BASE/$f"
    echo "exit=$? for $f"
    sleep 3
  done
  if [ -f "$D/$f.aria2" ]; then
    echo "FAILED to complete $f after attempts"
  else
    echo "COMPLETE $f ($(stat -c%s "$D/$f") bytes)"
  fi
}

finish_shard model-00001-of-00003.safetensors
finish_shard model-00002-of-00003.safetensors
finish_shard model-00003-of-00003.safetensors

echo "===== verify shards against safetensors index ====="
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate qwen_tool_sft
python - <<'PY'
import json, os
from safetensors import safe_open
d = "/mnt/ssd2/psf/job/qwen-tool-calling-sft/models_local/Qwen/Qwen3-4B-Base"
idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))
expected = {}
for k, fn in idx["weight_map"].items():
    expected.setdefault(fn, set()).add(k)
ok = True
for fn, keys in sorted(expected.items()):
    p = os.path.join(d, fn)
    if not os.path.exists(p):
        print("MISSING", fn); ok = False; continue
    with safe_open(p, framework="pt") as st:
        got = set(st.keys())
    missing = keys - got
    extra = got - keys
    print(f"{fn}: expected={len(keys)} got={len(got)} missing={len(missing)} extra={len(extra)}")
    if missing:
        ok = False
        print("  missing examples:", list(missing)[:5])
print("VERIFY_OK" if ok else "VERIFY_FAILED")
PY
echo "FINISH_DONE"
