#!/bin/bash
# Final acceptance checks (task book section 24). Read-only + local tests.
set +e
ROOT=/mnt/ssd2/psf/job/qwen-tool-calling-sft
source "$ROOT/scripts/env.sh"
cd "$ROOT" || exit 1

echo "===== [1] git status / log ====="
git status --short
git log --oneline -8

echo "===== [2] conda env / torch CUDA ====="
python - <<'PYEOF'
import torch, transformers, peft, trl, accelerate
print("python ok; torch", torch.__version__,
      "cuda_available", torch.cuda.is_available(),
      "n_gpu", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(" gpu", i, torch.cuda.get_device_name(i))
print("transformers", transformers.__version__, "peft", peft.__version__,
      "trl", trl.__version__, "accelerate", accelerate.__version__)
PYEOF

echo "===== [3] nvidia-smi ====="
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used --format=csv

echo "===== [4] pytest ====="
PYTHONPATH=src python -m pytest -q 2>&1 | tail -3

echo "===== [5] compileall ====="
python -m compileall -q scripts src eval tests && echo COMPILEALL_OK

echo "===== [6] eval artifacts ====="
for d in runs/eval_base runs/eval_sft_smoke runs/eval_sft_10k runs/eval_sft_30k; do
  if [ -f "$d/metrics.json" ]; then
    echo "-- $d"; python -c "import json;m=json.load(open('$d/metrics.json'));print('  overall',m['overall_exact_match'],'canonical',m['canonical_format_rate'],'clean_stop',m['clean_stop_rate'])"
  else
    echo "-- $d MISSING"
  fi
done

echo "===== [7] training artifacts ====="
for r in runs/qwen3-4b-smoke runs/qwen3-4b-sft-10k-r16 runs/qwen3-4b-sft-30k-r32; do
  if [ -f "$r/run_meta.json" ]; then
    echo "-- $r"; ls -la "$r/final_adapter/adapter_model.safetensors" 2>/dev/null | awk '{print "  adapter bytes:",$5}'
  else
    echo "-- $r MISSING"
  fi
done

echo "===== [8] data artifacts ====="
wc -l data/smoke/train.jsonl data/smoke/eval.jsonl data/full/train.jsonl data/full/eval.jsonl 2>/dev/null

echo "===== [9] others' processes untouched (ollama) ====="
ps -u "$(whoami)" -o pid,cmd --no-headers | grep -E "ollama" | grep -v grep | head -5
ss -ltn 2>/dev/null | grep 11434 | head -2

echo "ACCEPTANCE_DONE"
