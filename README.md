# Qwen3-4B Tool-Calling SFT（双 RTX 4090 复现）

在两台 RTX 4090（单节点双卡）上，用 LoRA / QLoRA 对 **Qwen/Qwen3-4B-Base** 做工具调用
（Function / Tool Calling）监督微调，并提供从数据准备、Base 评测、smoke 训练、
adapter 推理到 Base vs SFT 自动评测对比的完整可复现闭环。

> 本项目是对上游 `FuzzyFade/qwen35-tool-calling-sft`（vendored 于 `upstream-src/`，
> 上游 commit 见 `upstream-src/UPSTREAM_VERSION.txt`）的**适配性重写**：
> 上游使用 Unsloth + Qwen3.5-9B-Base（Gated DeltaNet 混合架构，需 transformers>=5.2），
> 本项目按任务书改用标准稠密架构 Qwen3-4B-Base + transformers/PEFT/TRL，
> 修复上游硬编码个人路径（`/Users/icecee/...`），并把 Demo 式评测替换为
> 36 题中英双语自动评测集与完整指标体系。

---

## 1. 目录结构

```
qwen-tool-calling-sft/
├── AGENTS.md                 # 8 条协作/安全规则（最高优先级）
├── README.md                 # 本文件
├── PROJECT_LOG.md            # 全程操作日志（只追加）
├── requirements.txt          # 直接依赖
├── requirements.lock.txt     # pip freeze 精确版本锁定
├── pytest.ini
├── configs/                  # 所有 YAML 配置（数据/训练/评测/accelerate）
├── scripts/                  # 可执行入口（数据/下载/训练/评测/对比 + env.sh）
├── src/qwen_tool_sft/        # 核心库：路径守卫、配置、数据转换、解析器、指标
├── eval/                     # 36 题自动评测集（eval_cases.py 生成 jsonl）
├── tests/                    # 单元/集成测试（pytest）
├── upstream-src/             # 上游参考项目快照（vendored，不直接运行）
├── data/                     # 生成的数据（git-ignore 内容，仅保留目录）
│   ├── smoke/                # 500/100 smoke 数据
│   └── full/                 # 全量数据
├── runs/                     # 训练 adapter、run_meta.json、评测产物
├── models_local/             # 模型权重（git-ignore，不入库）
├── .cache/                   # HF / ModelScope 缓存，全部在项目内（git-ignore）
├── setup_scripts/            # 环境搭建与运维脚本（不入库）
└── setup_logs/               # 搭建/下载日志（git-ignore）
```

## 2. 硬件与环境

- GPU：2 × NVIDIA RTX 4090（24GB），Driver 595.91.07 / CUDA 13.2
- 独立 conda 环境：`qwen_tool_sft`（Python 3.11），**不改动任何已有环境**
- 关键版本（完整见 `requirements.lock.txt`）：
  torch 2.14.0+cu130、transformers 4.57.x（刻意 <5）、trl 1.13.x、
  peft 0.21.x、accelerate 1.15.x、datasets 5.x、bitsandbytes 0.50.x、modelscope 1.40.x
- 网络：HuggingFace 直连不通，模型走 **ModelScope**，数据集走 **hf-mirror.com**
  （`scripts/env.sh` 已固定 `HF_ENDPOINT` 与更长的超时）

> 注意：TRL 已进入 1.x，`SFTConfig.max_seq_length` 更名为 `max_length`，
> `SFTTrainer` 使用 `processing_class`；`scripts/train.py` 已按安装版本自动过滤兼容参数。

## 3. 一键环境变量

所有命令前先 source（激活 conda、把缓存/PYTHONPATH 固定在项目内）：

```bash
source scripts/env.sh
```

## 4. 下载模型（ModelScope，落盘到项目内）

```bash
python scripts/download_model.py --model Qwen/Qwen3-4B-Base
# 权重位于 models_local/Qwen/Qwen3-4B-Base/
```

## 5. 数据准备

7 个上游工具调用数据集转换器（`src/qwen_tool_sft/converters.py`）：
Deepexi/function-calling-small、glaive 系（中英）、NousResearch hermes-function-calling-v1、
Team-ACE/ToolACE-310k、ServiceNow-AI/function-calling-opus-region-zh、
openbmb/UltraInteract_sft（openclaw/xlam 口径）。统一转换为
`{messages, tools}` ShareGPT/OpenAI 工具调用格式，做合法性校验、去重、混洗、切分。

```bash
# smoke：500 train / 100 eval，流式拉取（每源上限，失败源记录后跳过）
bash scripts/run_prepare_smoke.sh
# 全量（约 10 万，非流式，落 data/full）
python scripts/prepare_data.py --config configs/data_full.yaml
```

产物：`data/<split>/train.jsonl`、`eval.jsonl`、`stats.json`（含每源计数与失败信息）。

## 6. 训练（双卡 DDP LoRA）

```bash
# smoke：LoRA r16、seq 2048、bs1×ga8×2卡（有效 batch 16）、60 steps
bash scripts/run_train_smoke.sh

# 正式：1 万 / 3 万（r16 / r32），数据取 data/full
accelerate launch --config_file configs/accelerate_dual4090.yaml \
  scripts/train.py --config configs/train_sft_10k.yaml
accelerate launch --config_file configs/accelerate_dual4090.yaml \
  scripts/train.py --config configs/train_sft_30k.yaml
```

- LoRA 目标模块：q/k/v/o_proj + gate/up/down_proj；bf16；gradient checkpointing；packing。
- QLoRA：在配置中设 `load_in_4bit: true`（bitsandbytes NF4）。
- 每次训练产物：`runs/<run_name>/final_adapter/`、`checkpoints/`、`run_meta.json`
  （git commit、依赖版本、GPU 型号、样本数、步数、时长、峰值显存、loss 历史）。

## 7. 自动评测

评测集 `eval/tool_calling_eval.jsonl`（36 题，中英双语，7 类能力）：
single_tool、multi_tool_choice、argument_filling、no_tool（纯对话）、
wrong_tool_trap（给了工具但不应调用）、multi_argument、parallel_calls。

指标（`src/qwen_tool_sft/metrics.py`）：
Tool Selection Accuracy、Argument Exact Match、Argument Key Accuracy、
Valid Tool Call Format Rate、No-Tool Accuracy、Overall Exact Match，
以及 Invalid JSON / Wrong Tool / Missing Argument / Extra Argument 率，
并按类别输出准确率。参数比对做数值/布尔/字符串归一，同名并行调用按参数匹配消歧。

```bash
bash scripts/run_eval.sh configs/eval_base.yaml     # runs/eval_base
bash scripts/run_eval.sh configs/eval_sft.yaml      # runs/eval_sft_smoke（加载 adapter）
python scripts/compare_runs.py runs/eval_base runs/eval_sft_smoke \
  --output runs/comparison_smoke.md
```

## 8. 结果（实测数字，跑完后填充）

| 阶段 | Overall EM | Tool Sel | Arg EM | Valid Format | No-Tool |
|---|---:|---:|---:|---:|---:|
| Base（smoke 前） | 待填 | 待填 | 待填 | 待填 | 待填 |
| SFT smoke（500 条 / 60 步） | 待填 | 待填 | 待填 | 待填 | 待填 |
| SFT 10k | 待填 | 待填 | 待填 | 待填 | 待填 |
| SFT 30k | 待填 | 待填 | 待填 | 待填 | 待填 |

训练曲线 / 峰值显存 / 时长见各 `runs/<run_name>/run_meta.json` 与 `PROJECT_LOG.md`。

## 9. 测试与自检

```bash
PYTHONPATH=src python -m pytest -q          # 路径守卫/数据格式/解析/指标/配置/评测集/模板
python -m compileall -q scripts src eval tests
```

## 10. 安全边界（最高优先级）

- 所有写入只发生在 `/mnt/ssd2/psf/job/qwen-tool-calling-sft`；
  `paths.resolve_path` 对越界路径直接报错。
- 不删除/修改/移动该目录之外的任何文件；不改动已有 conda 环境、系统 CUDA/驱动。
- 不杀他人进程。目录中原有的 Ollama 服务（端口 11434）保持运行，
  训练前用 `nvidia-smi` 复核双卡空闲。
- SSH 密码、Token 等凭据一律不写入仓库 / 日志 / 配置。

## 11. 参考与许可

- 上游：https://github.com/FuzzyFade/qwen35-tool-calling-sft（见 `upstream-src/`）
- 模型：Qwen/Qwen3-4B-Base（ModelScope / HuggingFace，Qwen 许可）
- 数据集许可归各发布方所有，详见各数据集卡片。
