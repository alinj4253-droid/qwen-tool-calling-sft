# 项目日志（PROJECT_LOG）

> 只追加，不覆盖。记录 Qwen Tool-Calling SFT 复现全过程。

## 2026-09-22

### 操作

#### 1. 安全审计（只读）
- 目标目录确认：`/mnt/ssd2/psf/job`，与任务书一致。
- 目录内已有旧内容：`ollama/`（3.5G）、`models/`（19G，Ollama 模型仓，含 qwen2.5:32b Q4_K_M）、
  `src/extractor.py`、`data/chunks`、`data/kg`（KG 三元组抽取旧实验）、若干 ollama 脚本与日志。
- **发现 Ollama 服务进程仍在运行**（PID 888428/888431/888432，`./ollama/bin/ollama serve`，
  9月18日 setsid 后台启动，端口 11434，`ollama_stdout.log` 今天 11:10 仍有写入）。
- 依据任务书清理规则（有进程占用 / lsof 显示文件被打开 → 不得删除），
  **决定：旧文件与 ollama 服务全部保留不动，不杀进程，不抢占**。
- 其余占用 job 目录 cwd 的进程为用户本人 VSCode Remote-SSH 服务，无影响。
- GPU：2× RTX 4090（24564 MiB），驱动 595.91.07（CUDA 13.2），审计时仅 Xorg/gnome 占用，基本空闲。
- 磁盘：`/mnt/ssd2` 可用 86G；系统盘 `/` 可用 40G（conda 环境位于 `/home/wmy/anaconda3`）。
- Conda：`/home/wmy/anaconda3`，已有 16 个他人/历史环境，**全部只读不改**。

#### 2. 项目初始化
- 新建独立项目目录 `/mnt/ssd2/psf/job/qwen-tool-calling-sft`，所有新操作限定其中。
- `git init`；创建 `AGENTS.md`（8 条协作规则）、`.gitignore`（凭据/大文件/缓存）。
- 克隆上游 `https://github.com/FuzzyFade/qwen35-tool-calling-sft` 到 `upstream-src/`，
  记录上游 commit `aba8ea0006e4df21fa4450a07588074537823fec`（vendored，去除内嵌 .git）。

#### 3. 独立 Conda 环境
- 新建 `qwen_tool_sft`（Python 3.11.16），未改动任何已有环境。
- PyTorch 2.14.0（CUDA 13 wheel 系列），GPU matmul 冒烟通过（cuda:0）。

#### 4. 适配双 4090 的项目代码（不直接照搬上游）
- 上游用 Unsloth + Qwen3.5-9B-Base（混合架构，需 transformers>=5.2）。
- 本阶段按任务书改用 **Qwen3-4B-Base（标准稠密 Qwen3 架构）**，
  训练栈采用更透明、DDP 更稳的 **transformers + PEFT + TRL SFTTrainer**（LoRA/QLoRA 均可）。
- 修复上游硬编码 `/Users/icecee/...`：所有路径由 `src/qwen_tool_sft/paths.py`
  统一解析，相对路径锚定项目根，越界路径直接抛错；HF/ModelScope 缓存全部重定向到项目内 `.cache/`。
- 数据脚本支持 `--config/--max-train-samples/--max-eval-samples/--sources`，
  小样本走 HF 镜像 streaming，避免一次性拉全量。
- 新增 36 题中英双语自动评测集（`eval/tool_calling_eval.jsonl`）与真正的指标体系
  （Tool Selection / Argument EM / Key Accuracy / Valid Format / No-Tool / Overall +
  Invalid JSON / Wrong Tool / Missing / Extra 率），替代上游 Demo 式 eval。
- 单元测试：paths / dataset_format / eval_parser / metrics / config / eval_cases /
  chat_template（模型下载后自动启用）。

### Git Commit
- `679e7ce` chore: initialize repository and collaboration rules
- `6121faa` chore: import upstream reference project (aba8ea00)
- `28b7861` feat: project-local data pipeline, env and data configs
- `6d753f2` test: add unit tests, fixtures and 36-case tool-calling eval set
- `7de8de1` feat: dual-4090 LoRA training, automatic eval and base-vs-sft comparison

### 环境
- Conda env: qwen_tool_sft (Python 3.11.16)
- torch 2.14.0 / CUDA 13 wheels / Driver 595.91.07
- 其余依赖见 `requirements.lock.txt`（安装验证后生成）

### GPU
- 审计时双卡空闲（仅桌面进程）。训练前再次 nvidia-smi 复核。

### 结果
- 待补充（数据准备 / base 评测 / smoke 训练 / 对比评测）。

### 问题
- HuggingFace 直连不通，统一走 `HF_ENDPOINT=https://hf-mirror.com`；模型走 ModelScope。
- Ollama 服务仍在运行（空闲不占显存），训练期间观察其是否被他人调用加载模型。

### 下一步
- 完成依赖安装并 freeze；下载 Qwen3-4B-Base；smoke 数据；base 评测；双卡 smoke 训练。
