#!/bin/bash
# Stage 2: training/eval stack on top of torch. Best-effort, logs everything.
set +e
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate qwen_tool_sft
ROOT=/mnt/ssd2/psf/job/qwen-tool-calling-sft
LOG="$ROOT/setup_logs/08_install_stack.log"
exec > >(tee "$LOG") 2>&1
MIRROR="-i https://pypi.tuna.tsinghua.edu.cn/simple"

echo "===== install transformers/trl/peft/accelerate/datasets ====="
pip install --no-cache-dir $MIRROR "transformers>=4.51,<5" "trl>=0.18" "peft>=0.12" "accelerate>=1.0" \
  "datasets>=3.0" safetensors huggingface_hub sentencepiece protobuf pyyaml pytest
echo "PIP_CORE_EXIT=$?"

echo "===== install bitsandbytes (optional, QLoRA only) ====="
pip install --no-cache-dir $MIRROR "bitsandbytes>=0.43"
echo "PIP_BNB_EXIT=$?"

echo "===== install modelscope ====="
pip install --no-cache-dir $MIRROR modelscope
echo "PIP_MS_EXIT=$?"

echo "===== versions ====="
python - <<'PY'
import importlib
mods = ["torch","transformers","trl","peft","accelerate","datasets",
        "bitsandbytes","modelscope","safetensors","huggingface_hub","yaml","pytest"]
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"{m:20s} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"{m:20s} FAIL {type(e).__name__}: {e}")
PY

echo "===== import smoke (trainer path) ====="
python - <<'PY'
import torch, dataclasses
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
fields = {x.name for x in dataclasses.fields(SFTConfig)}
print("SFTConfig supports:",
      [f for f in ("max_seq_length","packing","dataset_text_field","eval_strategy",
                   "evaluation_strategy","gradient_checkpointing_kwargs",
                   "ddp_find_unused_parameters","processing_class","tokenizer") if f in fields])
print("cuda", torch.cuda.is_available(), torch.cuda.device_count())
PY
echo "STAGE2 DONE"
