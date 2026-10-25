# Qwen3-4B Tool-Calling SFT — a reproducible dual-RTX-4090 fine-tuning study

LoRA supervised fine-tuning of **Qwen/Qwen3-4B-Base** for tool / function calling, run on a single node with 2 × RTX 4090 (24 GB): two SFT generations, assistant-only loss masking, **selective special-token LoRA (810M → 66M trainable params)**, a 124k-sample data pool with joint dedup and benchmark decontamination, and **three evaluation suites (36 / 280 / 220)** including a BFCL v4 subset — with every number reproducible and every failure documented.

English | [简体中文](README.zh-CN.md)

[![Python](https://img.shields.io/badge/Python-3.11-3776AB.svg)](https://www.python.org/)
[![torch 2.14 / transformers 4.57 / TRL 1.13 / PEFT 0.21](https://img.shields.io/badge/transformers-4.57_·_TRL-1.13_·_PEFT-0.21-009688.svg)](#2-hardware--environment)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/alinj4253-droid/qwen-tool-calling-sft/actions/workflows/ci.yml/badge.svg)](https://github.com/alinj4253-droid/qwen-tool-calling-sft/actions/workflows/ci.yml)

## Headline results (Strict Protocol Success %)

| Model | Regression (36) | Extended (280) | BFCL v4 subset (220) |
|---|---:|---:|---:|
| Base | 11.1 | 10.7 | **1.8** |
| SFT-v1-30k | 94.4 | 56.1 | 30.0 |
| SFT-v2-30k | 86.1 | 52.1 | 13.2 |
| **v1-bal (final)** | **94.4** | **56.4** | **35.0** |
| grpo-smoke (not retained) | 94.4 | 57.1 | 35.9 |

- **Canonical `<tool_call>` format rate: 0% → 100%** — the core gain of SFT is protocol compliance.
- Final model = Base + SFT-v1-30k LoRA + targeted-v1-bal LoRA (two stacked adapters).
- Full 9-model comparison across all suites below.

<p align="center">
  <img src="docs/images/results-strict.png" alt="Strict protocol success by suite" width="820">
</p>
<p align="center">
  <img src="docs/images/protocol-compliance.png" alt="Protocol compliance Base vs SFT" width="720">
</p>

All figures come from `runs/eval_*/metrics.json` committed in this repository. **Strict Protocol Success** requires correct tool selection + exact argument match, all calls emitted in native `<tool_call>` blocks, and a clean stop via `<|im_end|>`/eos — bare JSON doesn't count even when the content is right.

## What this project actually studies

<p align="center">
  <img src="docs/images/training-pipeline.png" alt="Training and evaluation pipeline" width="820">
</p>

1. **Data.** 7 public datasets converted to a unified `{messages, tools}` format, validated, jointly deduped (messages + tools — a v1 bug fixed in v2), shuffled and split: v1 pool 94,842 / v2 pool 124,684 (72.2% with tools, including trap samples and pure chat). Decontamination checks (exact + 0.8 near-threshold) against both pools: **0 matches**.
2. **SFT-v1.** Full-sequence loss with full embed/head `modules_to_save` — 810M trainable params (16.8%), 1.69 GB adapter.
3. **SFT-v2, the interesting one.** Two fixes, each with unit tests and audit evidence (see [PROJECT_LOG.md](PROJECT_LOG.md) §13–17):
   - **Assistant-only loss** — `{% generation %}` blocks injected into the chat template (byte-identical rendering in 8 scenarios); a custom `ChunkedNLLSFTTrainer` (ce_chunk=256) computes NLL only on assistant tokens: 41.1% of 72.5M tokens supervised.
   - **Selective special-token LoRA** — PEFT `trainable_token_indices` trains the embedding/lm_head rows of only **6 protocol tokens** (im_start/im_end, tool_call/tool_response delimiters): trainable params **810M (16.8%) → 66M (1.6%)**, adapter **1.69 GB → 280 MB**, and the broken token "𬜯" from v1 disappears.
4. **Targeted arms.** Parallel-call upsampling (p05/p10) and balanced mixes (parallel + trap + chat), 4,000 samples each, stacked-adapter merging on top of a base SFT adapter.
5. **Evaluation.** Internal regression (36) · Extended benchmark (280: single/multi/parallel/distractor/wrong-tool-trap/no-tool) · BFCL v4 subset (220, deterministically sampled — **not comparable to official BFCL leaderboards**), all greedy, max_new_tokens=1024, one shared parser.

## Honest findings

- **SFT's main gain is protocol compliance, not task ability**: canonical format 0% → 100%, clean stop 64–70% → 100%.
- **Parallel multi-call is a persistent failure mode**: 0/100 on Extended multi-call items across v1/v2 and all four targeted arms — the capability exists in the base model (bare JSON) but is suppressed by SFT's stop-after-one-block prior. v1-bal is the first variant that partially transfers canonical multi-block output to BFCL (same-name 2%→20%).
- **v2 caused an over-calling regression** on trap/no-tool prompts (Extended tool-abstention 80.8% → 3.8%); the v1 route keeps abstention intact — which is why v1-bal won.
- **GRPO smoke: attempted but not retained.** 80 prompts × 30 steps × 5 interpretable rewards produced +0.7/+0.9 pp (noise-level) and one non-canonical output; it failed the retention threshold and the final model stays SFT. The scripts and configs are kept to reproduce this conclusion.

Every claim above is backed by committed artifacts: `runs/eval_*/metrics.json`, `runs/eval_*/predictions.jsonl`, `runs/comparison_*.md`, and the append-only [PROJECT_LOG.md](PROJECT_LOG.md) (failures → fixes). Final-selection reasoning lives in [FINAL_DECISION.md](FINAL_DECISION.md).

## Repo layout

```
configs/        all YAML: data / training / eval / accelerate / RL
scripts/        download / data / train / eval / decontaminate / GRPO smoke
src/qwen_tool_sft/   path guards, converters, parsing, metrics, loss masking
eval/           36 + 280 + 220 case files + BFCL manifest
runs/           adapter metadata, metrics.json, predictions.jsonl (no weights)
tests/          pytest (path guards, template masking, dedup, decontamination, Strict metric, ...)
docs/images/    charts & pipeline diagram (generated from real runs/ data)
```

## Reproduce

Hardware: 2 × RTX 4090 (24 GB); conda env `qwen_tool_sft` (Python 3.11). Key versions in `requirements.lock.txt`: torch 2.14.0+cu130, transformers 4.57.6 (deliberately <5), trl 1.13.0, peft 0.21.0. HuggingFace unreachable networks: models via ModelScope, datasets via hf-mirror (`scripts/env.sh`).

```bash
source scripts/env.sh

# 1. model + data
python scripts/download_model.py --model Qwen/Qwen3-4B-Base
python scripts/fetch_datasets.py
python scripts/prepare_data.py --config configs/data_full.yaml      # v1 pool
python scripts/prepare_data.py --config configs/data_full_v2.yaml   # v2 pool

# 2. SFT (dual-GPU)
accelerate launch --config_file configs/accelerate_dual4090.yaml scripts/train.py \
  --config configs/train_sft_30k.yaml          # v1
accelerate launch --config_file configs/accelerate_dual4090.yaml scripts/train.py \
  --config configs/train_sft_v2_30k.yaml       # v2

# 3. targeted arms (example)
python scripts/build_balanced_data.py --src data/full_v2/train.jsonl \
  --out data/targeted_bal --total 4000 --parallel-fraction 0.10 --trap-fraction 0.10 --chat-fraction 0.10 --seed 42

# 4. evaluate
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py --config configs/eval_targeted_v1_bal_benchmark.yaml
python scripts/rescore_predictions.py runs/eval_xxx   # offline re-scoring, no GPU needed

# 5. decontamination (mandatory after building eval sets)
python scripts/check_contamination.py --train data/full_v2/train.jsonl \
  --benchmark eval/bfcl_subset.jsonl --out runs/contamination_report/bfcl_subset --threshold 0.8
```

Tests (no GPU / models / keys required — CI runs exactly this):

```bash
PYTHONPATH=src python -m pytest -q        # heavy torch-dependent cases auto-skip
python -m compileall -q src scripts tests
```

## Safety & privacy

- All writes are confined to the project directory (`src/qwen_tool_sft/paths.py` raises on out-of-tree paths); no credentials, tokens, private keys or internal addresses in the repo — see [PRIVACY.md](PRIVACY.md) (git-history secret scan).
- Evaluation cases are fictional weather/stock/email scenarios.

## License & attribution

- Own code: MIT ([LICENSE](LICENSE)). Upstream reference `FuzzyFade/qwen35-tool-calling-sft` (no license) is locally referenced only, never redistributed (`upstream-src/UPSTREAM_VERSION.txt`).
- Model: Qwen/Qwen3-4B-Base (Apache-2.0). Dataset and BFCL/Gorilla attributions: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

中文说明见 [README.zh-CN.md](README.zh-CN.md) · 简历可用指标 [RESUME_METRICS.md](RESUME_METRICS.md)
