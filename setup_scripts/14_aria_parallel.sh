#!/bin/bash
set +e
ROOT=/mnt/ssd2/psf/job/qwen-tool-calling-sft
D="$ROOT/models_local/Qwen/Qwen3-4B-Base"
LOG="$ROOT/setup_logs/14_aria_parallel.log"
exec > >(tee "$LOG") 2>&1

echo "===== stop sequential 13 script / aria2 (our own PIDs) ====="
for p in $(pgrep -f "setup_scripts/13_aria_model.sh"); do kill "$p" 2>/dev/null && echo "killed 13-script $p"; done
for p in $(pgrep -f "aria2c.*Qwen3-4B-Base"); do kill "$p" 2>/dev/null && echo "killed aria2 $p"; done
sleep 2
for p in $(pgrep -f "aria2c.*Qwen3-4B-Base"); do kill -9 "$p" 2>/dev/null; done

echo "===== check special_tokens_map in API listing ====="
python3 - <<'PY'
import json
j=json.load(open('/mnt/ssd2/psf/job/qwen-tool-calling-sft/setup_logs/ms_files.json'))
names=[f['Name'] for f in j['Data']['Files']]
print(names)
print("has special_tokens_map:", 'special_tokens_map.json' in names)
PY

cd "$D"
BASE="https://www.modelscope.cn/models/Qwen/Qwen3-4B-Base/resolve/master"

echo "===== parallel aria2: shard1 + shard2 (8 conn each) ====="
aria2c -x8 -s8 -k1M --file-allocation=none -d "$D" -o model-00001-of-00003.safetensors \
  --continue=true --summary-interval=30 --console-log-level=warn \
  "$BASE/model-00001-of-00003.safetensors" > "$ROOT/setup_logs/aria_shard1.log" 2>&1 &
P1=$!
aria2c -x8 -s8 -k1M --file-allocation=none -d "$D" -o model-00002-of-00003.safetensors \
  --continue=true --summary-interval=30 --console-log-level=warn \
  "$BASE/model-00002-of-00003.safetensors" > "$ROOT/setup_logs/aria_shard2.log" 2>&1 &
P2=$!
echo "started aria pids $P1 $P2"
wait $P1; E1=$?
wait $P2; E2=$?
echo "aria exits: shard1=$E1 shard2=$E2"

echo "===== final listing ====="
ls -lah "$D"
du -sh "$D"
echo "PARALLEL_DONE"
