#!/bin/bash
set +e
ROOT=/mnt/ssd2/psf/job/qwen-tool-calling-sft
D="$ROOT/models_local/Qwen/Qwen3-4B-Base"
LOG="$ROOT/setup_logs/13_aria_model.log"
exec > >(tee "$LOG") 2>&1

echo "===== stop our own modelscope downloader (explicit PIDs) ====="
PIDS=$(pgrep -f "download_model.py --model Qwen/Qwen3-4B-Base")
echo "pids: $PIDS"
for p in $PIDS; do kill "$p" 2>/dev/null && echo "killed $p"; done
sleep 3
PIDS2=$(pgrep -f "download_model.py --model Qwen/Qwen3-4B-Base")
for p in $PIDS2; do kill -9 "$p" 2>/dev/null && echo "force killed $p"; done
for p in $(pgrep -f "setup_scripts/10_download_model.sh"); do kill "$p" 2>/dev/null && echo "killed wrapper $p"; done

echo "===== clean incomplete files ====="
cd "$D"
rm -f ./*.incomplete
ls -lah

echo "===== full file list from modelscope API ====="
curl -s --max-time 30 "https://www.modelscope.cn/api/v1/models/Qwen/Qwen3-4B-Base/repo/files?Revision=master" \
  -o "$ROOT/setup_logs/ms_files.json"
python3 - <<'PY'
import json
j=json.load(open('/mnt/ssd2/psf/job/qwen-tool-calling-sft/setup_logs/ms_files.json'))
for f in j['Data']['Files']:
    print(f['Name'], f['Size'], 'LFS' if f['IsLFS'] else '')
PY

echo "===== small files ====="
BASE="https://www.modelscope.cn/models/Qwen/Qwen3-4B-Base/resolve/master"
for f in config.json configuration.json generation_config.json merges.txt \
         model.safetensors.index.json tokenizer_config.json tokenizer.json vocab.json \
         special_tokens_map.json; do
  code=$(curl -sL --max-time 60 -o "$D/$f" -w "%{http_code}" "$BASE/$f")
  if [ "$code" = "200" ]; then
    echo "small ok: $f ($(stat -c%s "$D/$f" 2>/dev/null) bytes)"
  else
    echo "small MISS($code): $f"; rm -f "$D/$f"
  fi
done

echo "===== aria2 large shards (16 connections/file) ====="
for f in model-00001-of-00003.safetensors model-00002-of-00003.safetensors model-00003-of-00003.safetensors; do
  echo "--- aria2 $f ---"
  aria2c -x16 -s16 -k1M --file-allocation=none -d "$D" -o "$f" \
    --continue=true --summary-interval=20 --console-log-level=warn \
    "$BASE/$f"
  echo "aria2 exit=$? for $f"
done

echo "===== final listing ====="
ls -lah "$D"
du -sh "$D"
echo "ARIA_DONE"
