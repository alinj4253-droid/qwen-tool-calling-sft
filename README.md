# Qwen3-4B Tool-Calling SFT（双 RTX 4090 复现，含 SFT-v2 / Targeted / GRPO-smoke）

在两台 RTX 4090（单节点双卡）上，用 LoRA 对 **Qwen/Qwen3-4B-Base** 做工具调用
（Function / Tool Calling）监督微调，提供数据准备、Base 评测、两代 SFT、
assistant-only loss、selective special-token 训练、去污染、三套评测集、
targeted parallel 微调、最小 GRPO smoke 与最终模型选型的完整可复现闭环。

> 本项目是对上游 `FuzzyFade/qwen35-tool-calling-sft`（上游 commit 记录于
> `upstream-src/UPSTREAM_VERSION.txt`，仅作本地参考、不随仓库重新分发）的适配性重写：
> 上游使用 Unsloth + Qwen3.5-9B-Base（需 transformers>=5.2），本项目改用标准稠密
> Qwen3-4B-Base + transformers/PEFT/TRL，修复硬编码个人路径，并把 Demo 式评测替换为
> 三套共 536 题、带 Strict Protocol Success 口径的自动评测。

**最终模型（v1-bal）**：Base + SFT-v1-30k LoRA + targeted-v1-bal LoRA 两层堆叠，
Strict Protocol Success 在 36/280/220 题三套评测上分别为 **94.4% / 56.4% / 35.0%**。
完整判定与 28 个必答问题见 [FINAL_DECISION.md](FINAL_DECISION.md)，
可用于简历的真实指标见 [RESUME_METRICS.md](RESUME_METRICS.md)。

---

## 1. 目录结构

```
qwen-tool-calling-sft/
├── AGENTS.md                 # 协作/安全强制规则
├── README.md / FINAL_DECISION.md / RESUME_METRICS.md / PRIVACY.md
├── PROJECT_LOG.md            # 全程操作日志（只追加，含完整失败→修复逻辑链）
├── LICENSE / THIRD_PARTY_NOTICES.md
├── requirements.txt / requirements.lock.txt / pytest.ini
├── configs/                  # 数据/训练/评测/accelerate/RL 全部 YAML
├── scripts/                  # 入口：下载/数据/训练/评测/去污染/去重/GRPO smoke
├── src/qwen_tool_sft/        # 路径守卫、配置、转换、解析、指标、loss mask、去污染
├── eval/                     # 36 题回归集、280 题扩展集、220 题 BFCL 子集 + manifest
├── tests/                    # pytest（路径守卫/模板 mask/去重/去污染/Strict 指标/并行评测…）
├── data/                     # 生成数据（git-ignore；仅 stats 入库）
├── runs/                     # adapter 元数据、metrics.json、predictions.jsonl、报告（权重不入库）
├── models_local/ .cache/ setup_logs/ tmp/   # 全部 git-ignore
└── .github/workflows/ci.yml  # 轻量 CI：compileall + pytest（不在 CI 中训练）
```

## 2. 硬件与环境

- GPU：2 × NVIDIA RTX 4090（24GB），Driver 595.91.07 / CUDA 13.2（torch 报 13.0）
- 独立 conda 环境 `qwen_tool_sft`（Python 3.11.16），不改动任何已有环境
- 关键版本（完整见 `requirements.lock.txt`）：torch 2.14.0+cu130、transformers 4.57.6
  （刻意 <5）、trl 1.13.0、peft 0.21.0、accelerate 1.15.0、datasets 5.0.1、
  bitsandbytes 0.50.2、pytest 9.1.1
- 网络：HuggingFace 直连不通，模型走 ModelScope，数据集走 hf-mirror.com
  （`scripts/env.sh` 固定 `HF_ENDPOINT`）；GitHub 仅 api.github.com 可达。

```bash
source scripts/env.sh        # 激活环境变量（conda 需先 conda activate qwen_tool_sft）
```

## 3. 模型与数据准备

```bash
python scripts/download_model.py --model Qwen/Qwen3-4B-Base   # models_local/Qwen/Qwen3-4B-Base/
python scripts/fetch_datasets.py                               # 7 个数据源原件 -> .cache/datasets_raw/
python scripts/prepare_data.py --config configs/data_full.yaml     # v1 池 data/full（94,842/10,539）
python scripts/prepare_data.py --config configs/data_full_v2.yaml  # v2 池 data/full_v2（124,684/13,854）
```

7 个公开数据集经 `src/qwen_tool_sft/converters.py` 统一转成 `{messages, tools}` 格式，
做合法性校验、**messages+tools 联合去重**（v1 去重只看 messages，v2 已修复）、混洗、切分。
数据池真实组成（统计脚本可复现）：

| 池 | n | 带工具 | 带调用 | 陷阱样例（有工具无调用） | 纯闲聊 | 单消息真并行块 |
|---|---:|---:|---:|---:|---:|---:|
| v1 full | 94,842 | 63.4% | 50.4% | 13.36% | 36.6% | 1,060 条（0.85%，同名仅 108） |
| v2 full_v2 | 124,684 | 72.2% | 60.6% | 11.82% | 27.8% | 同上（按消息内容计） |

targeted 数据（确定性 seed，git-ignore，可重建）：

```bash
# 5% / 10% 并行上采样池（各 4,000 条）
python scripts/build_targeted_data.py --src data/full_v2/train.jsonl \
  --out data/targeted_p05 --total 4000 --parallel-fraction 0.05 --same-name-share 0.6 --seed 42
python scripts/build_targeted_data.py --src data/full_v2/train.jsonl \
  --out data/targeted_p10 --total 4000 --parallel-fraction 0.10 --same-name-share 0.6 --seed 42
# 平衡池：10% 并行 + 10% 陷阱 + 10% 纯闲聊
python scripts/build_balanced_data.py --src data/full_v2/train.jsonl \
  --out data/targeted_bal --total 4000 --parallel-fraction 0.10 \
  --trap-fraction 0.10 --chat-fraction 0.10 --seed 42
```

## 4. 两代 SFT 训练

```bash
# v1（full-sequence loss，modules_to_save embed/head，r32/α64，双卡）
accelerate launch --config_file configs/accelerate_dual4090.yaml scripts/train.py \
  --config configs/train_sft_30k.yaml
# v2（assistant-only loss + selective special-token LoRA，r32/α64，双卡）
accelerate launch --config_file configs/accelerate_dual4090.yaml scripts/train.py \
  --config configs/train_sft_v2_30k.yaml
# targeted（在某个 SFT adapter 内存 merge 后挂 fresh LoRA；配置内 base_adapter_path 指定）
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 scripts/train.py \
  --config configs/train_targeted_v1_bal.yaml   # 双卡时改用 accelerate_dual4090.yaml
```

SFT-v2 的两个核心修复（均有单测与审计证据，见 PROJECT_LOG 第 13–17 节）：

1. **Assistant-only loss**：对 chat template 行级注入 `{% generation %}…{% endgeneration %}`
   （8 场景渲染与原模板逐字节一致），自研 ChunkedNLLSFTTrainer（ce_chunk=256）只对
   assistant token 计 NLL；v2-30k 实测 72.5M 总 token 中 29.8M（41.1%）被监督，
   system/user/tool 与工具 schema 全部 mask。
2. **Selective special-token LoRA**：PEFT 0.21
   `trainable_token_indices=[151644,151645,151657,151658,151665,151666]`
   只训练 6 个协议 token（im_start/im_end、tool_call 开闭、tool_response 开闭）的
   embedding/lm_head 行，替代 v1 的 embed/head 全量副本：可训练参数 810,942,464（16.78%）
   → **66,075,648（1.6161%）**，adapter 1.69GB → **280.3MB**，v1 的故障 token
   “𬜯”（id 122588）消失。

v2-30k 实测：1,875 步 / 6,916.9s（115.3 分钟），train_loss 0.2576，训练中 eval_loss
0.2707→0.2583；峰值显存 rank0 17.68/21.08GB（alloc/reserved）。

## 5. 评测体系（三套，greedy，max_new_tokens=1024，同一解析器）

| 评测集 | 题数 | 类别 | 作用 |
|---|---:|---|---|
| Internal Regression `eval/tool_calling_eval.jsonl` | 36 | 7 类（含 pc-01/02 并行） | 冒烟与回归 |
| Extended `eval/benchmark_extended.jsonl`（seed 20260922） | 280 | single/multiple/parallel-same(50)/parallel-diff(20)/multi-arg/distractor/wrong-tool-trap/no-tool | 主 benchmark |
| BFCL 子集 `eval/bfcl_subset.jsonl`（seed 20260923） | 220 | single 60/multiple 40/parallel-same 50/parallel-diff 30/no-tool 40 | 外部标尺（见 manifest 口径声明） |

BFCL 子集由 `scripts/build_bfcl_subset.py` 从 Gorilla BFCL v4 五个静态类别确定性抽样；
官方宽松答案（每参数可接受值列表）被 canonicalize 为首个非空值，**不与官方 BFCL
榜单可比**，只在本项目 Strict 口径下使用。

**Strict Protocol Success** = 任务正确（工具选择 + 参数精确匹配，无多余/缺失参数）
且所有调用来自原生 `<tool_call>` 块、以 `<|im_end|>`/eos 干净停止；裸 JSON/fenced JSON
即使内容正确也不计 Strict（单独以 overall/canonical/clean-stop 列示）。

```bash
# 评测任意模型；targeted/RL 配置用 base_adapter_path（支持列表，按顺序 merge 多层 adapter）
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py --config configs/eval_targeted_v1_bal_benchmark.yaml
python scripts/rescore_predictions.py runs/eval_xxx    # 改评分逻辑后离线重打分，无需 GPU
```

去污染（评测集构建后必做，exact + near 阈值 0.8）：

```bash
python scripts/check_contamination.py --train data/full_v2/train.jsonl \
  --benchmark eval/bfcl_subset.jsonl --out runs/contamination_report/bfcl_subset --threshold 0.8
# 三套评测集对 v1、v2 两个训练池均为 0 exact / 0 near，报告在 runs/contamination_report*/
```

### 主结果（Strict Protocol Success，%；括号为 Base 的 overall 任务正确率）

| 模型 | Regression 36 | Extended 280 | BFCL 220 |
|---|---:|---:|---:|
| Base | 11.1（overall 91.7） | 10.7（overall 67.1） | 1.8（overall 63.2） |
| SFT-v1-30k | 94.4 | 56.1 | 30.0 |
| SFT-v2-30k | 86.1 | 52.1 | 13.2 |
| p05（v2+5% 并行，弱配方） | 83.3 | 52.1 | 16.8 |
| p10（v2+10% 并行，弱配方） | 83.3 | 51.4 | 17.7 |
| bal（v2+平衡数据，弱配方） | 86.1 | 51.4 | 16.4 |
| bal-strong（v2+平衡，lr1e-4/3ep） | 80.6 | 50.0 | 18.2 |
| **v1-bal（最终）** | **94.4** | **56.4** | **35.0** |
| grpo-smoke（未保留） | 94.4 | 57.1 | 35.9 |

关键分类别结论（完整数字在各 `runs/eval_*/metrics.json`）：

- SFT 的核心增益是协议合规：canonical 格式率 Base 0% → SFT 100%，干净停止约 64–70% → 100%；
- v2 把 Extended 单工具从 81.7% 抬到 95.0%，但陷阱拒答 80.8%→3.8%、BFCL 拒答 90%→10%
  （过度调用回归）；v1 路线的陷阱/拒答始终保持（80.8%/91.7%/90%）；
- **并行是稳定失败模式**：v1/v2 及 v2 上 4 个 targeted arm 在 Extended 100 个多调用题上
  全为 0；Base 却能用裸 JSON 完成多调用（BFCL same/diff 任务正确率 64%/73.3%），证明能力
  在基座、被 SFT 的“单块即停”先验压制；v1-bal 首次把规范多块输出迁移到 BFCL
  （同名 2%→20%、异名 0%→6.7%，逐条预测已核验），但未泛化到自然语言并行，未达
  “内部 parallel regression ≥90%”门槛。

## 6. GRPO 最小 smoke（P2，已按门槛淘汰）

任务书第十四节 8 个准入条件全部满足后才进入，且先跑最小实验：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/rl_grpo_smoke.py --config configs/rl_grpo_smoke.yaml
```

80 个 prompt（32 同名并行 + 32 异名并行 + 16 陷阱/闲聊）、num_generations=4、30 步、
transformers rollout（无 vLLM）、5 个可解释奖励（canonical format / clean stop /
tool selection / argument / parallel completion，无 LLM-as-judge）。结果相对 v1-bal
仅 +0.7/+0.9pp（各 2 题，噪声级），Extended/内部并行仍为 0，并出现 1 例非规范输出，
未达“parallel +5pp、其他降 ≤2pp”保留标准，记录 **RL attempted but not retained**，
最终模型继续使用 SFT。脚本与配置保留用于复现这一结论，但不宣称“已支持 RL 训练流程”。

## 7. 测试与 CI

```bash
PYTHONPATH=src python -m pytest -q          # 服务器 96 passed（含需 torch/模型的用例）
python -m compileall -q src scripts tests
```

GitHub Actions（`.github/workflows/ci.yml`）只安装 pytest/pyyaml 跑 compileall + pytest，
重型依赖与模型相关用例在缺 torch/模型时自动 skip，**CI 中不训练、不下载模型**。
在干净 venv（仅 pytest+pyyaml）中本地模拟 CI：82 passed、14 skipped、0 error。

## 8. 安全边界与隐私

- 所有写入只发生在项目目录内；`src/qwen_tool_sft/paths.py` 对越界路径直接报错；
  共享服务器他人目录、conda 环境、Ollama 服务（11434）全程未动，训练前 `nvidia-smi` 复核。
- 模型权重、原始数据、adapter safetensors、缓存、日志、凭据文件均由 `.gitignore` 排除；
  推送前对全部 git 历史做密钥特征扫描，详见 [PRIVACY.md](PRIVACY.md)。
- 仓库不含 SSH 密码、Token、私钥、内网地址；评测样例均为虚构天气/股票/邮件场景。

## 9. 参考与许可

- 上游参考：FuzzyFade/qwen35-tool-calling-sft（上游无 LICENSE，仅本地参考、不重新分发）
- 模型：Qwen/Qwen3-4B-Base（Apache-2.0，归 Qwen）；本仓库自有代码 MIT，见 [LICENSE](LICENSE)
- 训练数据与 BFCL/Gorilla 的归属与许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
