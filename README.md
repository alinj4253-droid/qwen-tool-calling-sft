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

7 个上游工具调用数据集（`src/qwen_tool_sft/converters.py` 中的转换器）：
Deepexi/function-calling-small、hiyouga/glaive-function-calling-v2-sharegpt（英）、
llamafactory/glaive_toolcall_zh（中）、NousResearch/hermes-function-calling-v1、
tryumanshow/ToolACE-Qwen-cleaned、nohurry/Opus-4.6-Reasoning-3000x-filtered、
bellfire/openclaw-coder-dataset。统一转换为
`{messages, tools}` ShareGPT/OpenAI 工具调用格式，做合法性校验、去重、混洗、切分。

**抓取方式（实测）**：`datasets` 库经 hf-mirror 在 HEAD/GET 阶段反复超时（死路），
改用自写 `scripts/fetch_datasets.py`：显式 MANIFEST + hf-mirror tree API 取清单 +
aria2c 多连接续传直拉 `/resolve/` 原始文件到 `.cache/datasets_raw/<repo>/`，
按 tree 字节数校验（`.aria2` 控制文件存在视为未完成）；转换器优先加载本地镜像
（csv/parquet/json/jsonl 均支持）。

```bash
# 1) 抓取原始文件到项目内 .cache/datasets_raw/
python scripts/fetch_datasets.py
# 2) smoke：500 train / 100 eval（bash scripts/run_prepare_smoke.sh）
# 3) 全量：105381 去重后 94842 train / 10539 eval，落 data/full
python scripts/prepare_data.py --config configs/data_full.yaml
```

产物：`data/<split>/train.jsonl`、`eval.jsonl`、`stats.json`（含每源计数与失败信息）。
实测 smoke 500 条中 67.4% 带工具；全量 63.47% 行带工具；并行调用样本
smoke 11.6%、全量 1.1%。

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

- LoRA 目标模块：q/k/v/o_proj + gate/up/down_proj；bf16；gradient checkpointing；
  **packing 关闭**（未装 flash-attn，SDPA 下 TRL 警告 packing 会跨样本注意力污染），
  改 group_by_length；优化器 `adamw_8bit`（bitsandbytes）控显存。
- **关键：`modules_to_save=["embed_tokens","lm_head"]`**。Qwen3-4B-Base 的
  `<|im_start|>/<|im_end|>/<tool_call>/</tool_call>` 等 chat/tool 特殊 token 行
  在 embed/lm_head 中是未训练的默认初始化（范数约 0.36 vs 已训练 token 中位 1.16），
  纯 LoRA 冻结 embed/head 时模型在该位置输出邻近范数的罕见汉字 `𬜯`（实测故障）；
  把 embed/head 设为全量可训练副本后协议 token 正常学会（可训练参数 8.11 亿，占 16.78%，
  双卡峰值约 21GB，adapter 约 1.7GB）。
- QLoRA：在配置中设 `load_in_4bit: true`（bitsandbytes NF4；本复现未启用）。
- 每次训练产物：`runs/<run_name>/final_adapter/`、`checkpoints/`、`run_meta.json`
  （git commit、依赖版本、GPU 型号、样本数、步数、时长、峰值显存、loss 历史）。

## 7. 自动评测

评测集 `eval/tool_calling_eval.jsonl`（36 题，中英双语，7 类能力）：
single_tool、multi_tool_choice、argument_filling、no_tool（纯对话）、
wrong_tool_trap（给了工具但不应调用）、multi_argument、parallel_calls。

指标（`src/qwen_tool_sft/metrics.py`）：
Tool Selection Accuracy、Argument Exact Match、Argument Key Accuracy、
Valid Tool Call Format Rate、**Canonical `<tool_call>` Format Rate**（调用必须来自
原生 `<tool_call>` 块，裸 JSON/fenced JSON 不计）、No-Tool Accuracy、
**Clean Stop Rate**（以 `<|im_end|>`/eos 干净结束且无 prompt 泄漏）、Overall Exact Match，
以及 Invalid JSON / Wrong Tool / Missing Argument / Extra Argument 率，
并按类别输出准确率。参数比对做数值/布尔/字符串归一（字符串值忽略末尾中英文句读标点），
同名并行调用按参数匹配消歧。解析器区分调用来源（native/fenced/bare）并拒绝
"工具定义形"对象（有 parameters 无 arguments）。

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py --config configs/eval_base.yaml      # runs/eval_base
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py --config configs/eval_sft.yaml       # runs/eval_sft_smoke
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py --config configs/eval_sft_10k.yaml   # runs/eval_sft_10k
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py --config configs/eval_sft_30k.yaml   # runs/eval_sft_30k
python scripts/compare_runs.py runs/eval_base runs/eval_sft_30k --output runs/comparison_30k.md
# 评分逻辑变更后无需重跑 GPU，可用保存的 predictions.jsonl 离线重打分：
python scripts/rescore_predictions.py runs/eval_base runs/eval_sft_smoke runs/eval_sft_10k
```

## 8. 结果（全部为真实运行数字，36 题评测集，greedy）

| 阶段 | Overall EM | Tool Sel | Arg EM | Arg Key | Valid Format | **Canonical** | No-Tool | **Clean Stop** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base Qwen3-4B-Base | 91.7% | 96.2% | 92.3% | 98.1% | 100% | **0.0%** | 90.0% | **66.7%** |
| SFT smoke（500 条 / 60 步，r16） | 91.7% | 92.3% | 88.5% | 96.2% | 100% | **100%** | 100% | **88.9%** |
| SFT 10k（r16，625 步 / 43.8 分钟） | 91.7% | 92.3% | 92.3% | 96.2% | 100% | **100%** | 90.0% | **100%** |
| SFT 30k（r32，1875 步 / 127.2 分钟） | **94.4%** | 92.3% | 92.3% | 96.2% | 100% | **100%** | **100%** | **100%** |

分类准确率（Overall EM，%）：

| 类别（题数） | Base | smoke | 10k | 30k |
|---|---:|---:|---:|---:|
| single_tool (6) | 83.3 | 100 | 100 | 100 |
| multi_tool_choice (6) | 100 | 100 | 100 | 100 |
| argument_filling (8) | 100 | 87.5 | 100 | 100 |
| no_tool (10) | 100 | 100 | 100 | 100 |
| wrong_tool_trap (4) | 75 | 100 | 75 | 100 |
| multi_argument (4) | 100 | 100 | 100 | 100 |
| parallel_calls (2) | 50 | 0 | 0 | 0 |

- 30k run 36 题中仅 pc-01、pc-02（同名工具并行调用）失败，其余全对；
  训练侧 train_loss 均值 0.391、训练中 eval_loss 0.404→0.368（step 400→1600）、
  mean_token_accuracy 0.930，峰值显存 cuda0 20.42GB。
- Base 评测 430.7s（不停重复 prompt、常撞 1024 上限），30k adapter 评测仅 46.0s（干净停止）。
- 已知局限：36 题对 Base 偏易、Overall 天花板约 92%（每题 2.78pp 粒度）；
  SFT 的核心增益在**协议合规**（canonical wrapper 0→100%、干净停止 66.7→100%、
  陷阱拒答与 single_tool 满分）；并行调用未学会，与全量训练集中并行样本仅 1.1%
  （同名工具并行更少）一致，后续可上采样并行样本并扩充评测集。

训练曲线 / 峰值显存 / 时长见各 `runs/<run_name>/run_meta.json` 与 `PROJECT_LOG.md`。

## 9. 测试与自检

```bash
PYTHONPATH=src python -m pytest -q          # 47 个测试：路径守卫/数据格式/解析/指标/配置/评测集/模板
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
