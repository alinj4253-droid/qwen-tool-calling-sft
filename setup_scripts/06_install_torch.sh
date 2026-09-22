#!/bin/bash
# Stage 1 of dependency install: PyTorch. Logs to project setup_logs.
set -e
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate qwen_tool_sft
ROOT=/mnt/ssd2/psf/job/qwen-tool-calling-sft
mkdir -p "$ROOT/setup_logs"
LOG="$ROOT/setup_logs/06_install_torch.log"
exec > >(tee "$LOG") 2>&1

echo "===== disk before ====="
df -h / /mnt/ssd2
echo "===== python/pip ====="
which python pip
python --version
pip --version

echo "===== upgrade pip ====="
pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "===== install torch (PyPI default CUDA wheel via tuna mirror) ====="
pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "===== verify torch + cuda ====="
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda build", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
print("device count", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
x = torch.randn(1024, 1024, device="cuda:0")
y = (x @ x).sum().item()
print("matmul on cuda:0 ok, sum=", round(y, 2))
PY

echo "===== disk after ====="
df -h / /mnt/ssd2
echo "STAGE1 DONE"
