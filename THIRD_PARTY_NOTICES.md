# Third-Party Notices

This repository's original code is released under the MIT license (`LICENSE`).
The artifacts below are third-party property and are **not** covered by that
license. Large artifacts (model weights, dataset raw files, trained adapters)
are excluded from git via `.gitignore`; the repository only contains small
evaluation sets, conversion code, and metadata needed for reproduction.

## Model

- **Qwen/Qwen3-4B-Base** — Apache License 2.0 per the official Qwen3 model
  card / model card shipped with the weights. Downloaded here via ModelScope
  into `models_local/` (git-ignored). Copyright Qwen team, Alibaba Cloud.

## Training datasets (raw files git-ignored under `.cache/` and `data/`)

Converted from seven publicly released tool-calling datasets; each remains
under the license set by its publisher (see the corresponding dataset card):

- Deepexi/function-calling-small
- hiyouga/glaive-function-calling-v2-sharegpt
- llamafactory/glaive_toolcall_zh
- NousResearch/hermes-function-calling-v1
- tryumanshow/ToolACE-Qwen-cleaned
- nohurry/Opus-4.6-Reasoning-3000x-filtered
- bellfire/openclaw-coder-dataset

No dataset raw content is redistributed in this repository; regeneration is
fully offline from cached original files via `scripts/fetch_datasets.py` and
`scripts/prepare_data.py`.

## External evaluation benchmark

- **Berkeley Function Calling Leaderboard (BFCL) v4**, static offline
  categories (`simple_python`, `multiple`, `parallel`, `parallel_multiple`,
  `irrelevance`) from
  `ShishirPatil/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data`,
  Apache License 2.0 (Gorilla project, UC Berkeley). The 220-case subset in
  `eval/bfcl_subset.jsonl` is a converted derivative work; conversion rules,
  category sizes and the strict-scoring caveat are recorded in
  `eval/bfcl_subset_manifest.json`. Official BFCL scores use a lenient
  checker; numbers reported here use this project's Strict Protocol Success
  definition and are not comparable to BFCL leaderboard numbers.

## Reference project

- The implementation was bootstrapped as an adaptation study of
  `FuzzyFade/qwen35-tool-calling-sft` (upstream commit recorded in
  `upstream-src/UPSTREAM_VERSION.txt`). At the time of writing that
  repository shipped no LICENSE file; its code is therefore retained only as
  a local, non-redistributed reference snapshot (git-ignored) and is not
  included in the published repository. All shipped code in `src/`,
  `scripts/` and `tests/` was rewritten for Qwen3-4B-Base with the standard
  transformers / PEFT / TRL stack.

## Open-source libraries

This work builds on, among others: PyTorch (BSD-3), Hugging Face transformers
(Apache-2.0), PEFT / TRL / accelerate / datasets (Apache-2.0), bitsandbytes
(MIT), ModelScope (Apache-2.0), pytest (MIT), NumPy (BSD-3). Exact installed
versions are recorded in `runs/environment_v1_freeze.txt` and
`requirements.lock.txt`; each library carries its own license in its
distribution.
