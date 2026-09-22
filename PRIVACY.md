# 隐私与敏感信息说明（PRIVACY）

本仓库为双 RTX 4090 上复现 Qwen3-4B Tool-Calling SFT 的实验工程（含 SFT-v2、
targeted parallel SFT 与 GRPO smoke）。发布前已做脱敏处理，请按以下说明理解仓库中的
路径、账号与产物信息。

## 1. 不含的内容（发布前已核验）

- **SSH 密码、私钥、Token、API Key 一律不在仓库中**。推送前对**完整 git 历史**做了
  特征串扫描（`REDACTED_PASSWORD`、服务器内网地址、`ghp_`、`github_pat_`、`sk-`、
  `PEM 私钥头标记` 等），并重建了不含敏感提交的干净历史；早期误入库的一个含密码
  字面量的运维脚本已从跟踪与历史中移除。
- 连接服务器所需的凭据仅存在于受控的控制端，从未写入任何入库脚本、日志、配置或提交。
- 以下大文件/私有数据按 `.gitignore` 排除，**不在 GitHub 上**：
  - `models_local/`：Qwen3-4B-Base 权重（请从 ModelScope/HuggingFace 自行下载）；
  - `.cache/datasets_raw/`：7 个数据源原始文件（按 `scripts/fetch_datasets.py` 的
    MANIFEST 自行抓取）；`.cache/bfcl/`：BFCL 原始下载文件；
  - `data/smoke/`、`data/full/`、`data/full_v2/`、`data/targeted_*/`：生成的
    train/eval jsonl（可用 `prepare_data.py`、`build_targeted_data.py`、
    `build_balanced_data.py` 重建；仅入库 `stats.json` 类统计摘要）；
  - `runs/**/final_adapter/*.safetensors`、`*.bin`、`checkpoints/`：LoRA adapter 权重
    （v1 约 1.7GB、v2 约 280MB、targeted 约 148MB，超过或接近 GitHub 单文件限制，
    可按 README 用仓库内配置重新训练复现）；
  - `setup_logs/`、`tmp/`、`*.log`：运行日志与临时探针（可能含环境细节）。

## 2. 已做的脱敏标记

- Linux 用户名、家目录路径已泛化：文档统一写 `~/anaconda3`；运维脚本用 `$(whoami)`
  代替硬编码用户名；测试用例中的越权路径样例使用通用的 `/home/other-user/secret`。
- 仓库中保留的绝对路径（如 `/mnt/ssd2/psf/job/qwen-tool-calling-sft`）仅为多人共享
  服务器上的实验目录约定，不含主机地址、账号或凭据；在其他机器复现时以项目根目录为锚点
  （`src/qwen_tool_sft/paths.py` 会拒绝越界路径）。
- Git 提交身份（`user.name/user.email`）为实验者主动配置的公开提交身份，非隐私泄露。
- 共享服务器上他人的进程、目录、conda 环境、Ollama 服务（端口 11434）均未触碰，
  相关信息不入库。

## 3. 仓库可见性建议

- 建议保持 **Private**；如需 Public，建议再次对完整历史运行密钥特征扫描，
  并确认 `runs/*/predictions.jsonl` 中不涉及真实业务数据。本仓库全部评测样例均为
  人工编写/公开数据抽样的虚构天气、股票、邮件、数学工具场景，无真实用户数据。
- 本仓库不包含、也不应添加任何 CI 密钥；GitHub Actions 只运行 compileall 与 pytest，
  不下载模型、不训练、不访问内网资源。

## 4. 第三方数据与许可

- 模型权重、7 个训练数据集、BFCL/Gorilla 评测数据的许可归各发布方所有，
  详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 与 README 第 9 节；
  仓库仅包含转换/抓取/抽样代码、统计信息与一个 220 题的 BFCL 抽样评测集
  （来源与口径限制见 `eval/bfcl_subset_manifest.json`），不重新分发原始训练数据。
