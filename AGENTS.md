# 项目协作规则

本文件是本仓库（Qwen3-4B Tool-Calling SFT，双 RTX 4090 复现实验工程）对所有协作者与自动化代理的强制约束。

## 注意事项

1. 每次改动完成后，都必须创建一个对应的 Git Commit，以便后续追踪和回滚；提交前先运行 `pytest -q`、`python -m compileall .` 并检查 `git status`。
2. 每次改动后，都必须编写或更新相关测试，并在交付给用户前，确保所有测试和验证全部通过；测试失败时禁止合并、禁止宣称完成。
3. 所有项目文件只能位于本项目目录（共享服务器上为 `/mnt/ssd2/psf/job/qwen-tool-calling-sft`）之内；临时文件放项目内 `tmp/`，产物放 `runs/`，数据放 `data/`。
4. 禁止修改服务器已有 Conda 环境；本项目统一使用 `qwen_tool_sft` 环境，缺依赖时先在项目内解决（venv/源码目录），确需改环境必须记录并说明。
5. 禁止删除或修改本项目目录之外的任何文件；`/mnt/ssd2/psf/job` 下其他目录（ollama、models、src、data 等）属于他人，一律只读。
6. 每次训练前必须记录 GPU 状态（`nvidia-smi`，确认两张 4090 空闲）、环境（`pip freeze`）、代码 Commit（git HEAD）和训练参数（YAML 配置），并写入 `runs/<run>/run_meta.json`；训练后保留 final_adapter、metrics、predictions、日志。
7. 任何可能影响其他用户的操作必须停止：不得 kill 他人进程、不得占用他人正在使用的 GPU、不得修改 Ollama 等共享服务；双卡被占用时等待或与用户确认，不抢卡。
8. 禁止将 SSH 密码、API Key、Token、私有凭据、服务器地址相关敏感信息提交到 Git；推 GitHub 前必须做密钥扫描（含全部 git 历史），`.gitignore` 必须排除模型权重、数据、缓存、日志和凭据文件。
9. 任何实验结论都必须有可复现的证据文件支撑：指标来自 `metrics.json`、逐条输出来自 `predictions.jsonl`、训练过程来自 `run_meta.json` 与训练日志；文档中每个关键数字都必须能在这些产物中找到来源，禁止凭印象写数字。
10. 禁止把未跑成功、未验证或仅计划中的功能写成“已支持/已完成”；失败的实验必须如实记录为失败并写明原因与排查过程；指标名称必须与口径一致（如 no-tool 类只称 Tool Abstention Accuracy，不得称为回答正确率）。

## 第二阶段补充约定（SFT-v2 / targeted / RL）

- 训练开关必须显式记录：`loss_mode`（full / assistant_only）、`special_token_training`（full / selective / none）、`trainable_token_indices`、LoRA rank/alpha 与目标模块。
- 新增评测集必须先对训练池做去污染（exact + near 阈值），报告写入 `runs/contamination_report*/`，并在 manifest 中标注数据来源、采样种子与口径限制。
- Targeted SFT 与 RL 的保留/淘汰只按任务书门槛（内部 parallel regression ≥90%、其他类绝对下降 ≤2pp；RL parallel 绝对提升 ≥5pp）由指标决定；未达标模型即使跑完也不得作为最终模型。
- RL（GRPO/GSPO）只有在任务书第十四节 8 个条件全部满足时才允许进入，且必须先跑最小 smoke；奖励项必须可解释，禁止以 LLM-as-a-Judge 作为唯一奖励。
- 每次 commit 前：`pytest -q` → `python -m compileall src scripts tests` → `git status`，全部通过再提交。
