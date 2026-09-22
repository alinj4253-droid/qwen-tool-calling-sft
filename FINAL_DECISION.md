# FINAL_DECISION.md — Qwen3-4B Tool-Calling SFT 第二阶段最终判定

所有数字均来自服务器 `runs/*/metrics.json`、`runs/*/run_meta.json` 与训练/评测日志，可逐项复现；评测口径统一为 greedy、`max_new_tokens=1024`、同一套解析器与 Strict Protocol Success 指标。

模型简称：

- **Base**：Qwen/Qwen3-4B-Base（未微调）
- **v1**：SFT-v1-30k（full-sequence loss + LoRA r32 + modules_to_save embed/head，94,842 条训练样本）
- **v2**：SFT-v2-30k（assistant-only loss + selective special-token LoRA r32，124,684 条训练样本）
- **p05 / p10 / bal / bal-strong**：在 v2 之上继续的 targeted LoRA（并行占比 5% / 10% / 平衡 10% 弱配方 / 平衡 10% 强配方）
- **v1-bal（最终模型）**：在 v1 之上继续的 targeted LoRA（10% 并行 + 10% 陷阱 + 10% 纯闲聊平衡数据，lr 1e-4，2 epochs）
- **grpo-smoke**：在 v1-bal 之上的 30 步 GRPO 最小实验

三套评测集：Internal Regression 36 题；Extended Benchmark 280 题（seed 20260922）；BFCL 子集 220 题（seed 20260923，本项目 Strict 口径，**不与官方 BFCL 榜单可比**）。

Strict Protocol Success 总表（%）：

| 模型 | Regression 36 | Extended 280 | BFCL 220 |
|---|---:|---:|---:|
| Base（overall 括号内） | 11.1（overall 91.7） | 10.7（overall 67.1） | 1.8（overall 63.2） |
| v1 | 94.4 | 56.1 | 30.0 |
| v2 | 86.1 | 52.1 | 13.2 |
| p05 | 83.3 | 52.1 | 16.8 |
| p10 | 83.3 | 51.4 | 17.7 |
| bal（弱） | 86.1 | 51.4 | 16.4 |
| bal-strong（3ep） | 80.6 | 50.0 | 18.2 |
| **v1-bal（最终）** | **94.4** | **56.4** | **35.0** |
| grpo-smoke（未保留） | 94.4 | 57.1 | 35.9 |

---

## 第十八节：17 个必答问题

### 1. SFT-v2 是否成功？

**部分成功（YES on engineering, NO as replacement for v1）。** 工程目标全部达成并验证：assistant-only loss 真实生效、selective special-token LoRA 跑通、去重与去污染完成、Strict 指标与扩大评测集建立、30k 正式训练完成且可复现。但模型选型意义上 v2 没有取代 v1：v2 在三套 Strict 总分上为 86.1 / 52.1 / 13.2，低于 v1 的 94.4 / 56.1 / 30.0，原因是 v2 出现两类真实回归（见第 8、9 条）。因此 v2 是“方法学验证成功、配方不是最优”的一代。

### 2. Assistant-only loss 是否真实生效？证据是什么？

**是。** 证据链：

1. 对 Qwen3-4B-Base chat template 做行级 `{% generation %}…{% endgeneration %}` 注入，8 组场景渲染文本与原模板逐字节一致（`tests/test_assistant_only_loss.py`）；
2. 单测逐 token 校验 mask：纯闲聊、单调用、三调用并行、多轮、有工具但拒答五类场景中，system/user/tool 与工具 schema 全部为 -100，assistant 内容（含两个并行 `<tool_call>` 块）全部被监督；
3. v2-30k 实测 token 统计：72,507,257 个总 token 中 29,790,590 个为 loss token（41.1%），符合“只监督约 4 成 assistant token”的预期；
4. 自研 ChunkedNLLSFTTrainer 解决 TRL chunked_nll 与 TrainableTokensWrapper 不兼容问题，双卡 DDP 梯度 allreduce 探针证明两卡梯度一致。

### 3. Selective special-token training 是否成功？

**成功。** PEFT 0.21 `LoraConfig(trainable_token_indices=[151644,151645,151657,151658,151665,151666], modules_to_save=None)` 被正确接受：r16 探针下可训练参数 33,045,504（r32 正式 run 为 66,075,648），未出现 v1 时期的故障 token “𬜯”（id 122588）；v2 adapter 仅 280.3MB（v1 为 1.69GB），reload 探针输出标准 `<tool_call>…</tool_call><|im_end|>`、无故障 token、干净停止。

### 4. 最终可训练参数是多少？

- v1-30k：810,942,464（16.78%，LoRA + embed/lm_head 全量副本）；
- v2-30k：66,075,648 / 4,088,543,744 = **1.6161%**（r32/α64，7 个线性层 + 6 个协议 token 行）；
- targeted 各 arm（r16/α32，special=none）：33,030,144；
- grpo-smoke（r16/α32）：33,030,144。

### 5. Adapter 大小是多少？

- v1-30k final_adapter：1,688,012,760 字节（约 1.69GB，含 embed/head 副本）；
- v2-30k final_adapter：280,265,746 字节（约 280.3MB，du 268MB）；
- 各 targeted / grpo-smoke final_adapter：约 148.1MB（du 142MB，r16）。

### 6. Benchmark 是否完成去污染？

**是。** 三套评测集（regression36、benchmark_extension 280、bfcl_subset 220）均对 v2 训练池（full_v2，124,684）与 v1 训练池（full，94,842）做了 exact + near（归一化后序列相似度阈值 0.8）去污染，报告在 `runs/contamination_report/` 与 `runs/contamination_report_v1/` 下。

### 7. Exact duplicate / Near duplicate 分别多少？

三套评测集对两个训练池均为 **exact = 0，near = 0**（阈值 0.8）。

### 8. SFT-v2 相比 SFT-v1 提升了什么？

正向：

- Extended 单工具调用 81.7% → **95.0%**；参数精确匹配（argument exact）46.4% → 50.5%；
- 可训练参数 810.9M → 66.1M（12.3 倍缩减），adapter 1.69GB → 280.3MB（6 倍缩减）；
- 训练口径更正确：只监督 assistant token，工具 schema 不再被学习，协议 token 行被显式训练；
- 数据管线修复后训练池更完整（94,842 → 124,682 条有效样本，去重逻辑修正）。

负向（如实记录）：

- 陷阱拒答（wrong_tool_trap，Extended）80.8% → **3.8%**；BFCL irrelevance 拒答 90% → **10%**；Regression trap 100% → 50%；
- BFCL Strict 30.0% → 13.2%；Regression Strict 94.4% → 86.1%；
- 多调用能力没有任何改善（Extended 100 个多调用题仍为 0）。

机制推断（标注为推断）：assistant-only + 协议 token 行训练叠加 v2 池中带调用样本占比上升（50.4%→60.6%）、纯闲聊/陷阱占比下降，使“优先调用工具”的先验显著增强，造成过度调用与拒答崩塌。

### 9. Parallel 是否仍为主要失败模式？

**是，且是唯一贯穿全部模型的硬失败。** v1、v2 及在 v2 上的全部 4 个 targeted arm，在 Extended 的 100 个多调用题（multiple 30、same-tool 50、diff-tool 20）上多块输出正确率全部为 0；Regression 的 pc-01/pc-02 全部为 0。已证实不是 parser/格式/数据渲染 bug：模板把并行调用渲染成两个独立原生 `<tool_call>` 块，assistant-only mask 对两块都监督（有单测与渲染探针）。Base 反而能用非规范裸 JSON 完成多调用（BFCL same-tool 任务正确率 64%、diff-tool 73.3%、Extended same 36%、diff 35%、multiple 46.7%），说明能力存在于基座，SFT 的“单块即停”先验把它压掉了；训练池中单条 assistant 消息含 ≥2 个调用的样本仅 1,060 条（0.85%，同名仅 108 条）。

### 10. 是否执行 targeted parallel SFT？

**是，共 5 个 arm**：p05（5% 并行，lr2e-5/1ep）、p10（10%，弱配方）、bal（10% 并行+10% 陷阱+10% 闲聊，弱配方）、bal-strong（同平衡数据，lr1e-4/3ep）、v1-bal（在 v1 上用平衡数据 lr1e-4/2ep，双卡）。数据由 `scripts/build_targeted_data.py`、`scripts/build_balanced_data.py` 确定性生成并记录统计。

### 11. Targeted SFT 是否解决 parallel？

**没有达到任务书门槛（内部 parallel regression ≥90%），但出现了真实的局部迁移：**

- 在 v2 上的 4 个 arm：Extended 多调用始终 0/100，陷阱拒答也未恢复（弱配方欠拟合，强配方 3ep 反而使 Regression 降到 80.6%）；
- v1-bal：v1 的陷阱/拒答能力完全保留（Extended trap 80.8%、BFCL abstention 90%、Regression 94.4 均不降），**BFCL 同名并行 2%→20%、异名 0%→6.7%**（预测中确有 2–4 个原生 `<tool_call>` 块，经 predictions 逐条核验），但 Extended 自然语言并行仍为 0、pc-01/02 仍为 0。
- 结论：targeted SFT 能让模型学会高度模板化的并行模式（BFCL 风格），但没有泛化到自然语言、多工具组合的并行意图。门槛未达成。

### 12. 是否需要进入 GRPO / GSPO？

任务书第十四节 8 个条件逐项核对：1) v2 完整跑通 ✔；2) assistant-only 已验证 ✔；3) benchmark 已扩充 ✔；4) contamination 已检查 ✔；5) targeted SFT 已尝试（5 个 arm）✔；6) parallel 仍明显落后 ✔；7) 已排除 parser/格式/数据 bug ✔（模板渲染、mask、预测样本均核验）；8) GPU/磁盘资源允许 ✔（双卡空闲、69GB 可用）。**条件全部满足，因此按要求执行了最小 GRPO smoke**，而没有直接长时间 RL。

### 13. RL 是否真正优于 targeted SFT？

**没有。** GRPO smoke（80 题、G=4、30 步、lr1e-5、5 个可解释奖励项、无 vLLM、无 LLM judge）结果相对 v1-bal：Regression 94.4→94.4、Extended 56.4→57.1（+0.7pp，2 题）、BFCL 35.0→35.9（+0.9pp，2 题）；核心目标 Extended/内部并行仍为 0；canonical 99.5%（出现 1 个非规范样本）。提升在评测噪声范围内、未达到“parallel 绝对 +5pp”的保留门槛，按任务书第十七节记录为 **“RL attempted but not retained”**，最终模型继续使用 SFT。

### 14. 最终保留哪个模型？

**v1-bal**：Base + `runs/qwen3-4b-sft-30k-r32/final_adapter` + `runs/qwen3-4b-targeted-v1-bal/final_adapter` 两层 LoRA 顺序合并加载（评测配置 `configs/eval_targeted_v1_bal_*.yaml` 已固化该堆叠顺序）。

### 15. 为什么它是最终最佳模型？

它在三套 Strict 总分上是全部 SFT 模型中最高的（94.4 / 56.4 / 35.0），并且：

- 相对 v1：Regression 持平 94.4，Extended +0.3pp，BFCL **+5.0pp**，且首次在外部集上出现规范的多块并行输出（同名 20%）；
- 陷阱拒答（80.8%/100%）、无工具拒答（100%/90%）、canonical 与 clean stop（均 100%）全部保持，没有 v2 路线上的过度调用回归；
- 其他类波动均在任务书 2pp 容忍带附近（multi_argument 93.3→90.0、distractor 83.3→80.0，各 −3.3pp，已在局限中明示；single_tool 81.7→86.7）。
- grpo-smoke 的 +0.7/+0.9pp 是噪声级且伴随 1 例格式退化，按规则不保留。

### 16. 现在项目是否适合直接写简历？

**可以，但必须按 `RESUME_METRICS.md` 的口径写。** 可写的真实亮点：完整的失败定位→假设→数据/损失/参数高效微调修复→标准化评测→决策闭环；assistant-only loss 与 selective special-token LoRA 的工程实现与验证；三套共 536 题、含外部 BFCL 子集的去污染评测体系；严格的协议合规指标；在 2×4090 上把可训练参数降到 1.62%、adapter 降到 280MB；targeted SFT 在 BFCL 并行子集上的真实局部修复；一次按门槛进入、按证据淘汰的 GRPO smoke。不能夸大为“解决了 parallel tool calling”。

### 17. 仍有哪些必须如实说明的局限？

1. 自然语言 parallel/multiple tool calling 未解决：Extended 100 题与 pc-01/02 全程为 0；仅 BFCL 模板化并行有 20–24% 的局部迁移；
2. BFCL 子集（220 题）是官方数据的确定性抽样 + 宽松答案 canonicalize，**不与官方 BFCL 榜单可比**，只是本项目 Strict 口径下的外部标尺；
3. 全部评测为 greedy 单次解码，未做多次采样置信区间；280 题上 1 题 ≈0.36pp，2 题内的差异不应作为结论；
4. 训练数据为 7 个公开数据集的混合与转换，分布偏合成；v2 池与 v1 池组成不同（去重修复所致），v1/v2 对比不是纯单变量实验，文档已标注；
5. 只验证了 LoRA/QLoRA 路线与 Qwen3-4B-Base，未做全量微调、未换基座；
6. RL 只跑了 30 步 transformers-rollout smoke，不能据此断言“RL 对该问题无效”，只能说在该规模与奖励设计下没有稳定增益；
7. SFT 会压制基座以非规范形式表现出的多调用能力，如何在不牺牲协议合规的前提下释放该能力是后续工作。

---

## 第二十九节：给用户的 11 个问题

**1. 当前最终最好的模型是哪一个？**
v1-bal：Qwen3-4B-Base + v1-30k LoRA + targeted-v1-bal LoRA（两层堆叠）。

**2. 它相比 Base 提升了什么？**
Strict Protocol Success：Regression 11.1%→94.4%、Extended 10.7%→56.4%、BFCL 1.8%→35.0%；canonical `<tool_call>` 格式率 0%→100%、干净停止率约 64–70%→100%；BFCL 无工具拒答 27.5%→90%；单工具、参数填充、多参数等类稳定在 86.7–100%。Base 的 overall 任务正确率（67.1%/63.2%）大量依赖非规范裸 JSON，无法被工具链直接消费。

**3. 它相比 SFT-v1 提升了什么？**
Regression 持平（94.4）；Extended +0.3pp 且单工具 81.7%→86.7%；BFCL Strict 30.0%→35.0（+5pp），同名并行 2%→20%、异名 0%→6.7%，multiple 20%→22.5%；陷阱/拒答/格式指标全部保持不退化。

**4. Parallel Tool Calling 解决了吗？**
没有完全解决。模板化外部场景（BFCL）出现 20–24% 的规范多块输出；自然语言并行（Extended 100 题、内部 pc-01/02）仍为 0。

**5. 如果解决，是数据/SFT 解决的还是 RL 解决的？**
仅有的局部改善来自数据/SFT（v1-bal 的平衡 targeted 数据）；RL（GRPO smoke）未产生达标增益，未保留。

**6. 如果没解决，原因是什么？**
证据支持的解释：训练池中单消息真并行样本仅 0.85%（同名 108 条），SFT 形成了“一个 `<tool_call>` 块后立即 `<|im_end|>`”的强停止先验；5 个 targeted arm（含 lr1e-4、3 epochs 强配方）都不能把该行为泛化到自然语言并行，弱配方欠拟合、强配方轻度过拟合；基座本身具备多调用能力但以裸 JSON 表达，说明瓶颈在行为先验与数据分布，而非模型容量。模板渲染、loss mask、解析器均已用单测和逐条预测排除了 bug。

**7. 最终是否使用了 GRPO / GSPO？**
执行了 GRPO 最小 smoke（`scripts/rl_grpo_smoke.py`，30 步），未保留其产物作为主模型；未使用 GSPO，未做长时间 RL。

**8. 为什么使用 / 为什么没有扩大？**
使用 smoke 是因为任务书第十四节 8 个准入条件全部满足；没有扩大是因为 smoke 在核心目标（parallel）上零提升、总分仅 +0.7–0.9pp（噪声级）且出现 1 例格式退化，未达到第十七节“parallel 绝对 +5pp 且其他能力降 ≤2pp”的保留标准。按规则不硬保留 RL 模型。

**9. 项目现在是否已经适合写入简历？**
适合，定位为“工具调用 SFT 的严谨复现与诊断项目”：包含两代 SFT 对照、损失掩码与 selective token 训练、去污染评测体系、targeted 数据修复、一次完整的 RL 准入/淘汰决策。不适合包装成“parallel tool calling 已解决”或“RL 提升了性能”。

**10. 简历可以使用哪些真实指标？**
见 `RESUME_METRICS.md`，核心数字：最终模型 Strict 94.4%/56.4%/35.0%（36/280/220 题）；canonical 格式与干净停止 100%；v2 可训练参数 1.62%、adapter 280.3MB（较 v1 缩小 6 倍）；训练 115.3 分钟 / 双卡 4090 / 峰值 21.1GB；三套评测集对两个训练池去污染均 0 exact / 0 near；BFCL 同名并行 2%→20%（v1-bal）。

**11. 还有哪些会影响面试表述的局限？**
同第 17 条：并行未根治、BFCL 子集非官方榜单口径、greedy 单次评测、合成数据分布、仅 LoRA/单基座、RL 仅 smoke、SFT 对基座非规范多调用能力的压制。建议主动说明这些局限与后续方向（更高质量的自然语言并行数据、多轮/并行混合训练、更大规模 RL 需 vLLM 基础设施）。
