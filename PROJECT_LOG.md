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
- 磁盘：`/mnt/ssd2` 可用 86G；系统盘 `/` 可用 40G（conda 环境位于 `~/anaconda3（/home/<linux-user>/anaconda3）`）。
- Conda：`~/anaconda3（/home/<linux-user>/anaconda3）`，已有 16 个他人/历史环境，**全部只读不改**。

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

---

## 2026-09-22（续）：环境冻结、数据管线、Base 评测、三次 SFT 训练与对比评测

### 5. 依赖安装与版本冻结
- 在独立环境 `qwen_tool_sft`（Python 3.11.16）内安装并 freeze 到 `requirements.lock.txt`：
  torch 2.14.0+cu130、transformers 4.57.6（刻意 <5，与 TRL/PEFT 稳定组合）、
  trl 1.13.0、peft 0.21.0、accelerate 1.15.0、datasets 5.0.1、bitsandbytes 0.50.2、
  modelscope 1.40.x、pytest 9.1.1。未改动 base 与其余 16 个已有 conda 环境。
- TRL 1.x 适配点：`SFTConfig.max_seq_length → max_length`；`Trainer` 的
  tokenizer 参数改名 `processing_class`；`scripts/train.py` 按安装版本自动过滤兼容参数。

### 6. 模型下载与校验
- HuggingFace 直连不通；ModelScope SDK 与单文件多流均被限速（~2.5MB/s，服务器出口总带宽约 2.5MB/s），
  最终用 ModelScope 取得文件清单后由 aria2c 多连接续传下载 Qwen/Qwen3-4B-Base
  （3 个 safetensors 分片 + index + tokenizer），落盘到项目内 `models_local/`。
- 逐张量加载校验通过（VERIFY_OK），vocab 151936、hidden 2560。

### 7. 数据管线（关键工程：datasets 库不可用 → 显式直链抓取）
- `datasets` 库经 hf-mirror 在 HEAD/GET 阶段反复超时（多次重试均失败，确认为死路）；
  hf-mirror 的 `/api/parquet` 会回 huggingface.co 直链，同样不可达；ModelScope 无 Deepexi 镜像。
- 自写 `scripts/fetch_datasets.py`：显式 MANIFEST 登记 7 个数据源的原始文件，
  hf-mirror `/api/datasets/<repo>/tree` 取文件清单（偶发 403，加浏览器 UA 缓解；
  tree 失败时退化为 manifest-only），aria2c（-x8 -s8、--continue、
  --connect-timeout 15 --timeout 30）直拉 `/resolve/` 原始文件到
  `.cache/datasets_raw/<repo>/`；`.aria2` 控制文件存在即视为未完成，
  完成后按 tree 返回的字节数校验。
- 踩坑：bellfire train.jsonl 曾因中途 kill aria2 留下 147456 字节截断文件且无 .aria2
  （被误判完成），转换器报 JSON 错后删除重下，最终 4809094 字节精确一致。
- 7 源全部核验（字节数）：Deepexi CSV 111809738；hiyouga glaive_toolcall.json 250561134；
  llamafactory zh 2378465；hermes 三文件 20554130 / 17793686 / 20416419；
  ToolACE parquet 8247862；nohurry jsonl 7501968；bellfire train 4809094 + eval 535985。
- `converters.py` 新增 `_load_local`（csv/parquet/json builder；hermes 三个 JSON 列结构不同，
  逐文件 load 后 IterableDataset 合并；openclaw 按 split 映射 train.jsonl/eval.jsonl），
  `_load` 优先使用本地镜像。7 个转换器注册名：
  deepexi_zh、glaive_zh、glaive_v2_en、hermes_en、toolace_en、opus_reasoning_en、openclaw_en。
- smoke 数据：3112 条去重后分层抽 500 train / 100 eval（67.4% 带工具，7 源全部有产出）。
- 全量数据：105381 条去重后 **94842 train / 10539 eval**（63.47% 行带工具调用，约 268MB，`data/full`）。
- 并行调用样本占比：smoke 58/500 行（11.6%）；全量 1043/94842 行（1.1%；
  每 turn 调用数分布 {1:67683, 2:402, 3:495, 4:121, 5:14, 6:2, 7:2, 9:4}）。
  训练文本确认并行调用渲染为多个独立 `<tool_call>` 块（每块一个 JSON 对象），
  assistant turn 以 `` 思考块开头（常为空）。

### 8. Base 评测与解析器两轮修复
- 首轮 Base Overall 83.3% 高得可疑，拉取 predictions 人工审查发现两个解析器缺陷：
  ①模型回显工具**定义**（含 parameters/description）被误判成调用；
  ②答案开头连续的两个裸 JSON（并行调用）只取了第一个。
- 修复 parser：拒绝"定义形"对象（有 parameters 无 arguments）；
  裸 JSON 只从答案开头连续解析若干个 JSON 值（遇到 prompt 泄漏标记即停），
  支持并行多对象与数组；`ParseResult` 记录每个调用的来源
  （native `<tool_call>` / fenced ```json / bare 裸 JSON）。
- 修复后重跑 Base（36 题，430.7s）：Overall 91.7%（标点归一后口径，见第 11 节）、
  ToolSel 96.2%、ArgEM 92.3%、ArgKey 98.1%、ValidFormat 100%、NoTool 90%、
  **Canonical `<tool_call>` 格式率 0%、Clean Stop 率 66.7%**。
- 分类：single_tool 5/6、multi_tool_choice 6/6、argument_filling 8/8、no_tool 10/10、
  wrong_tool_trap 3/4、multi_argument 4/4、parallel_calls 1/2。
- 行为诊断（真实结论）：Qwen3-4B-Base 靠模板模式匹配直接吐裸 JSON，解析器宽松时"能用"，
  但不会干净停止（常重复 prompt、输出到 1024 token 上限，多语言乱码后接 system/user 标记）、
  不使用 XML wrapper、并行调用与陷阱题弱。

### 9. smoke 训练第一次失败 → 定位根因（特殊 token 未训练）
- 第一次双卡 DDP（LoRA r16，约 33M 可训练参数）loss 正常下降，但 SFT 评测 Overall 仅 27.8%、
  格式率 0：模型在 `<tool_call>` 位置输出罕见汉字 `𬜯`（token 122588）。
- 诊断（setup_scripts/43、44）锁定根因：**Qwen3-4B-Base 的 chat/tool 特殊 token 行
  （151644 `<|im_start|>`、151645 `<|im_end|>`、151657 `<tool_call>`、151658 `</tool_call>`、
  1665/151666 tool_response）在 embed_tokens/lm_head 中是未训练的默认初始化行**
  （范数约 0.3566，已训练 token 行中位数 1.162；eos 151643 范数 1.223 正常；
  122588 `𬜯` 范数 0.3621，故模型落到邻近范数 token）。
  纯 LoRA 冻结 embed/head，永远学不会这些 token。
- 修复：`LoraConfig(modules_to_save=["embed_tokens","lm_head"])`
  （adapter 内保存全量可训练的 embed/head 副本），优化器改 `adamw_8bit`（bitsandbytes）控显存；
  关闭 packing（未装 flash-attn，SDPA 下 TRL 明确警告 packing 会跨样本注意力污染），
  改 group_by_length；加 `model.enable_input_require_grads()`；
  10k/30k 配置加 `max_eval_samples: 500`（训练中 loss eval 不用全部 10.5k eval）。
- 修复后可训练参数 810,942,464（16.78%，LoRA 约 33M + embed/head 约 778M），
  双卡各占 20.6–21.3GB（24.5GB 上限内，无 OOM），
  adapter_model.safetensors 1,688,012,760 字节。

### 10. 三次训练（全部真实运行）
| Run | 数据 | LoRA | 步数 | 时长 | 有效 batch | 峰值显存 | loss |
|---|---|---|---:|---:|---:|---:|---|
| qwen3-4b-smoke | smoke 500 | r16 α32 | 60 | 271.1s（4.5 分钟） | 16 | cuda0 19.91GB | 均值 0.764 |
| qwen3-4b-sft-10k-r16 | full 前 10k | r16 α32 | 625 | 2625.7s（43.8 分钟） | 16 | cuda0 19.91GB | 均值 0.425，末段约 0.325；step600 eval_loss 0.4011；mean_token_accuracy 0.9346 |
| qwen3-4b-sft-30k-r32 | full 前 30k | r32 α64 | 1875 | 7631.3s（127.2 分钟） | 16 | cuda0 20.42GB | 均值 0.391，末段约 0.32；训练中 eval_loss 0.404/0.386/0.374/0.368（step400/800/1200/1600，500 样本）；mean_token_accuracy 0.9296 |

- 统一：bf16、gradient checkpointing、双卡 DDP（accelerate，端口 29501）、
  lr 1e-4 cosine、max_length 2048、packing 关闭、adamw_8bit、modules_to_save embed/head。
- 每个 run 落盘 `final_adapter/`（adapter + tokenizer）、`checkpoints/`、`run_meta.json`
  （git_head、配置、依赖版本、GPU、样本数、步数、时长、峰值显存、完整 loss history）。

### 11. 指标体系增强（协议合规指标 + 离线重打分）
- 36 题对 Base 偏易（Overall 天花板约 89–92%，每题 2.78pp 粒度），宽松解析器把 Base 的裸 JSON
  也计为格式正确，掩盖了 SFT 的真实增益。新增两个"协议合规"指标：
  - **Canonical `<tool_call>` Format Rate**：工具题中所有调用均来自原生 `<tool_call>` 块
    （bare/fenced JSON 不计）；
  - **Clean Stop Rate**：以 `<|im_end|>`/eos 结束且无 prompt 泄漏
    （泄漏正则匹配 `<|im_start|>` 或行首 system/user/assistant 标记）。
- 参数 EM 对字符串值做末尾中英文句读标点归一（rstrip `。.!！?？;；,，`），
  消除"会议。 vs 会议"这类伪失败。
- 新增 `scripts/rescore_predictions.py`：对已有 predictions.jsonl 用最新 parser/metrics
  离线重打分并重写 metrics.json/summary.md（无需 GPU）。新增 3 个单测，测试总数 47 全绿。

### 12. Base vs SFT 自动评测对比（数字均来自真实 metrics.json）
| 指标 | Base | SFT smoke（500/60步） | SFT 10k | SFT 30k |
|---|---:|---:|---:|---:|
| Tool Selection Accuracy | 96.2% | 92.3% | 92.3% | 92.3% |
| Argument Exact Match | 92.3% | 88.5% | 92.3% | 92.3% |
| Argument Key Accuracy | 98.1% | 96.2% | 96.2% | 96.2% |
| Valid Tool Call Format Rate | 100% | 100% | 100% | 100% |
| **Canonical `<tool_call>` Format Rate** | **0.0%** | **100%** | **100%** | **100%** |
| No-Tool Accuracy | 90.0% | 100% | 90.0% | **100%** |
| **Clean Stop Rate** | **66.7%** | **88.9%** | **100%** | **100%** |
| Overall Exact Match | 91.7% | 91.7% | 91.7% | **94.4%** |

分类（Overall EM）：
- Base：single 83.3 / multi_choice 100 / argument_filling 100 / no_tool 100 / trap 75 / multi_arg 100 / parallel 50
- smoke：single 100 / multi_choice 100 / af 87.5 / no_tool 100 / trap 100 / multi_arg 100 / parallel 0
- 10k：single 100 / multi_choice 100 / af 100 / no_tool 100 / trap 75 / multi_arg 100 / parallel 0
- 30k：single 100 / multi_choice 100 / af 100 / no_tool 100 / trap 100 / multi_arg 100 / parallel 0
  （36 题中仅 pc-01、pc-02 两题并行调用失败，其余全对；评测耗时 46.0s）

结论（如实记录）：
1. SFT 的主要增益是**协议合规**：canonical wrapper 0%→100%、干净停止 66.7%→100%、
   single_tool 83.3%→100%、陷阱拒答 75%→100%（30k）、no_tool 90%→100%（30k）；
   输出从"裸 JSON + 重复 prompt + 撞 1024 上限"变为规范的
   `<tool_call>{...}</tool_call><|im_end|>`，评测耗时也从 430.7s 降到 46.0s。
2. Overall EM：Base 91.7% → smoke/10k 91.7%（36 题上饱和，每题 2.78pp 粒度）→ 30k 94.4%。
3. 未解决：parallel_calls（pc-01/02，三个 SFT run 都只发一个调用，pc-01 还多给 unit 参数；
   全量训练集并行样本仅 1.1%，同名工具并行更稀少）。后续可上采样并行调用样本或扩充评测集。

### Git Commit（续）
- `c34f09d` data: 数据集直链抓取（aria2c + MANIFEST）与本地镜像加载
- `53e96f6` eval: 解析器加严（定义 vs 调用、开头并行裸 JSON）；训练关 packing、input-require-grads、训练中 eval 限样本
- `ceef850` train: modules_to_save embed_tokens/lm_head（Base 特殊 token 行未训练），adamw_8bit
- `8f0f9d0` eval: canonical-format/clean-stop 协议指标、参数末尾标点归一、离线 rescore 工具、10k/30k eval 配置
- 30k 训练/评测与文档收尾的 commit 见 git log。

### 安全边界复核
- 全程写入仅发生在 `/mnt/ssd2/psf/job/qwen-tool-calling-sft`；job 目录内他人/历史的
  ollama 二进制、models/（19G）、src/、data/ 旧实验文件全部原样保留。
- Ollama 服务（PID 888428/888431/888432，端口 11434）始终未杀未碰；
  服务器上他人历史 python 进程未动；每次训练前均 nvidia-smi 复核双卡空闲。
- SSH 密码等凭据未写入任何入库文件 / 日志 / 配置。
