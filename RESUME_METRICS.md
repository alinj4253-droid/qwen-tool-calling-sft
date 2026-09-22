# 可用于简历的真实指标（RESUME_METRICS）

只收录可在 `runs/*/metrics.json`、`run_meta.json`、训练日志与 `runs/contamination_report*/` 中复现的数字。评测口径：greedy 解码、`max_new_tokens=1024`、同一解析器、同一 Strict Protocol Success 定义；最终模型 = Base + v1-30k LoRA + targeted-v1-bal LoRA 两层堆叠。

## 数据

- 训练数据规模：
  - SFT-v1 池：94,842 条训练 / 10,539 条验证（7 个公开工具调用/通用指令数据集转换混合）；
  - SFT-v2 池（messages+tools 去重修复后）：124,684 条训练 / 13,854 条验证；其中 123,527 条在 max_length=2048 内有效（1,157 条超长丢弃）；
  - targeted 平衡池：4,000 条（同名并行 240 槽位/108 条唯一、异名并行 160、陷阱样例 400、纯闲聊 400、其余常规 2,800；实测并行 10.45%、陷阱 10%、纯闲聊 10%）。
- 数据来源：7 个公开数据集（清单与许可见 `THIRD_PARTY_NOTICES.md`，原始文件不随仓库分发，由 `scripts/fetch_datasets.py` 按 MANIFEST 重建）。
- 外部 Benchmark：BFCL v4 静态类别确定性抽样子集 **220 题**（single 60 / multiple 40 / parallel-same 50 / parallel-diff 30 / no-tool 40，seed 20260923；官方宽松答案被 canonicalize，**不与官方 BFCL 榜单可比**，仅作本项目 Strict 口径外部标尺）。
- Internal Regression：36 题（`eval/tool_calling_eval.jsonl`，含 pc-01/pc-02 并行用例）。
- 扩展 Benchmark：280 题（`eval/benchmark_extended.jsonl`，8 个类别，seed 20260922）。
- Decontamination 结果：三套评测集对 v1、v2 两个训练池的 exact duplicate 与 near duplicate（相似度阈值 0.8）均为 **0 / 0**。

## 训练

- Base Model：Qwen/Qwen3-4B-Base（4.0B，tie_word_embeddings，词表 151,936）。
- GPU：2 × NVIDIA RTX 4090（24GB），bf16，gradient checkpointing，DDP（accelerate）。
- SFT 方法：LoRA/QLoRA 监督微调；v1 为 full-sequence loss + modules_to_save（embed/head）；v2 为 assistant-only NLL loss（ChunkedNLL，ce_chunk=256）+ selective special-token LoRA；最终模型再经一层 targeted 平衡数据 LoRA。
- Loss Mask：assistant-only（system/user/tool 消息与工具 schema 全部 mask，仅监督 assistant token；v2 实测 loss token 29,790,590 / 总 72,507,257 = 41.1%）。
- LoRA rank：v1/v2 主训练 r32/α64；targeted 与 GRPO r16/α32；目标模块 q/k/v/o/gate/up/down 7 个线性层；v2 额外训练 6 个协议 token（im_start/im_end、tool_call 开闭、tool_response 开闭）的 embedding/lm_head 行。
- 可训练参数：v1 810,942,464（16.78%）；**v2 66,075,648 / 4,088,543,744 = 1.6161%**；targeted r16 33,030,144。
- Adapter 大小：v1 1.69GB；**v2 280.3MB（缩小约 6 倍）**；targeted 148.1MB。
- 训练时间：v2-30k 双卡 1,875 步 / 6,916.9s（115.3 分钟，3.65s/step）；targeted v1-bal 双卡 494 步 / 约 30.7 分钟；GRPO smoke 单卡 30 步 / 978s。
- Peak VRAM：v2-30k rank0 17.68GB allocated / 21.08GB reserved（物理峰值约 20.7GB），rank1 17.67 / 21.24GB。

## 结果

Strict Protocol Success（%，greedy；括号为 Base 的 overall 任务正确率）：

| 模型 | Regression 36 | Extended 280 | BFCL 220 |
|---|---:|---:|---:|
| Base | 11.1（overall 91.7） | 10.7（overall 67.1） | 1.8（overall 63.2） |
| SFT-v1-30k | 94.4 | 56.1 | 30.0 |
| SFT-v2-30k | 86.1 | 52.1 | 13.2 |
| **最终模型 v1-bal** | **94.4** | **56.4** | **35.0** |
| GRPO smoke（未保留） | 94.4 | 57.1 | 35.9 |

- Task Correctness（最终模型，分类别）：Regression 单工具/多工具选择/参数填充/多参数/无工具/陷阱 86.7–100%；Extended 单工具 86.7%、多参数 90.0%、干扰工具 86.7%、无工具 100%、陷阱拒答 80.8%。
- Canonical Tool Format：所有 SFT 模型 **100%**（Base 为 0%，其调用以裸 JSON/代码块表达）。
- Clean Stop：所有 SFT 模型 **100%**（Base 约 64–70%，常泄漏 prompt 或撞长度上限）。
- Strict Protocol Success：最终模型 94.4 / 56.4 / 35.0（三套，见上表）。
- Tool Abstention（给了不相关工具时不调用 / 无工具题）：最终模型 Regression 100%、Extended 91.7%、BFCL 90.0%（Base BFCL 仅 27.5%）。
- Parallel Tool Calling：**未根治**。BFCL 同名并行 v1 2%→最终 20%、异名 0%→6.7%（GRPO smoke 24%/6.7%）；Extended 100 个自然语言多调用题与内部 pc-01/pc-02 仍为 0%。Base 以非规范裸 JSON 在 BFCL 同名/异名并行上可达 64%/73.3% 任务正确率。

## 关键问题与解决

- Assistant-only masking：发现 v1 对 system/工具 schema 也计算 loss；通过行级 `{% generation %}` 模板注入（8 场景渲染逐字节一致）+ 自研 ChunkedNLLSFTTrainer 实现仅 assistant token 受监督，单测逐 token 验证，实测监督比例 41.1%。
- Special token training：发现 Base 特殊 token 行未被训练且 v1 的 modules_to_save 方案把可训练参数推到 8.1 亿；改用 PEFT trainable_token_indices 只训练 6 个协议 token 行，可训练参数降至 1.62%，故障 token（id 122588）消失，adapter 缩小 6 倍。
- Parallel failure：定位为数据分布问题（单消息真并行样本仅 0.85%）+ SFT 单块停止先验；排除模板渲染、mask、解析器 bug；5 个 targeted SFT arm 与 1 个 GRPO smoke 证明模板化并行可部分修复（BFCL +18pp 同名），自然语言并行未泛化。
- Data decontamination：修复 messages+tools 联合去重（v1 漏去重导致池虚增/泄漏风险），三套评测集对两个训练池 exact/near 均为 0。
- RL 是否使用：按任务书 8 条件准入，跑了 30 步 GRPO smoke（5 个可解释奖励：canonical format / clean stop / tool selection / argument / parallel completion，无 LLM judge）；parallel 未达 +5pp 门槛、总分增益在噪声内，记录 “RL attempted but not retained”，主模型仍为 SFT。

## 不建议在简历中声称

- “解决了 parallel / multiple tool calling”——自然语言并行在 Extended 与内部集上仍为 0%。
- “在 BFCL 榜单上达到 X 分”——220 题是自建 Strict 口径子集，与官方榜单不可比。
- “RL（GRPO/GSPO）显著提升性能”——仅 30 步 smoke，未保留，增益不显著。
- “SFT-v2 全面优于 v1”——v2 单工具与参数效率更优，但陷阱拒答与 BFCL 大幅回退，最终模型建立在 v1 之上。
- 任何超出上述评测集与口径的泛化结论（生产工具链、真实用户流量、其他基座）。
