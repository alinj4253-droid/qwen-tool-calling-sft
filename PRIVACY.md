# 隐私与敏感信息说明（PRIVACY）

本仓库为双 RTX 4090 上复现 Qwen3-4B Tool-Calling SFT 的实验工程。发布前已做脱敏处理，
请按以下说明理解仓库中的路径、账号与产物信息。

## 1. 不含的内容（发布前已核验）

- **SSH 密码、私钥、Token、API Key 一律不在仓库中**（含完整 git 历史，已用
  `git log --all -p` + 特征串扫描核验：无登录密码、无 `ghp_/github_pat_/sk-` 等令牌、
  无私钥块、无服务器内网 IP）。
- 连接服务器所需的凭据仅存在于受控的控制端环境变量中，从未写入任何脚本、日志或提交。
- 以下大文件/私有数据按 `.gitignore` 排除，**不在 GitHub 上**：
  - `models_local/`：Qwen3-4B-Base 权重（请从 ModelScope/HuggingFace 自行下载）；
  - `.cache/datasets_raw/`：7 个数据源的原始文件（数百 MB，按 `scripts/fetch_datasets.py`
    的 MANIFEST 自行抓取）；
  - `data/smoke/`、`data/full/` 的 train/eval jsonl（生成数据，可用 `prepare_data.py` 重建；
    仅入库两个 `stats.json` 统计摘要）；
  - `runs/**/final_adapter/*.safetensors`、`*.bin`、`checkpoints/`：LoRA adapter 权重
    （含 embed/lm_head 副本，每个 1.6–1.8GB，超过 GitHub 单文件限制，需要可按
    README 用仓库内配置重新训练复现）；
  - `setup_logs/`、`*.log`：运行日志（可能含环境细节）。

## 2. 已做的脱敏标记

- Linux 用户名、家目录路径已泛化：日志中的 `/home/<真实用户>/anaconda3` 写为
  `~/anaconda3（/home/<linux-user>/anaconda3）`；运维脚本用 `$(whoami)` 代替硬编码用户名；
  测试用例中的越权路径样例改为通用的 `/home/other-user/secret`。
- 仓库中保留的绝对路径（如 `/mnt/ssd2/psf/job/qwen-tool-calling-sft`）仅为**多人共享服务器
  上的实验目录约定**，不含主机地址、账号或凭据；在其他机器上复现时请以项目根目录为锚点
  （代码中的 `src/qwen_tool_sft/paths.py` 会拒绝越界路径）。
- Git 提交身份（`user.name/user.email`）为实验者主动配置的公开提交身份，非隐私泄露。
- 共享服务器上**他人的进程、目录、conda 环境、Ollama 服务**均未触碰，相关信息不入库。

## 3. 仓库可见性建议

- 建议保持 **Private**；如需公开（Public），建议再次运行
  `bash setup_scripts/58_secret_scan.sh`（如保留了该运维脚本）或等效扫描，
  并确认 `runs/*/predictions.jsonl` 中自造的评测样例不涉及真实业务数据
  （本仓库评测集为人工编写的虚构天气/股票/邮件场景，无真实用户数据）。

## 4. 第三方数据与许可

- 模型权重与 7 个训练数据集的许可归各发布方所有，详见各数据集卡片与 README 第 11 节；
  仓库仅包含转换/抓取代码与统计信息，不重新分发原始数据。
