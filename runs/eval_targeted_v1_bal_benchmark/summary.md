# Eval summary: SFT-v1-targeted-bal

cases: `/mnt/ssd2/psf/job/qwen-tool-calling-sft/eval/benchmark_extended.jsonl` (280)
adapter: `runs/qwen3-4b-targeted-v1-bal/final_adapter`
max_new_tokens=1024, do_sample=False

| Metric | SFT-v1-targeted-bal |
|---|---:|
| Tool Selection Accuracy | 54.1% |
| Argument Exact Match | 46.8% |
| Argument Key Accuracy | 74.0% |
| Valid Tool Call Format Rate | 99.5% |
| Canonical Tool Format Rate | 99.5% |
| Tool Abstention Accuracy | 91.7% |
| Clean Stop Rate | 100.0% |
| Strict Protocol Success | 56.4% |
| Overall Exact Match | 56.4% |
| Invalid JSON Rate | 0.0% |
| Wrong Tool Rate | 45.9% |
| Missing Argument Rate | 45.9% |
| Extra Argument Rate | 3.2% |

## By category

| Category | N | Overall | Strict Protocol |
|---|---:|---:|---:|
| distractor | 30 | 80.0% | 80.0% |
| multi_argument | 30 | 90.0% | 90.0% |
| multiple_tools | 30 | 0.0% | 0.0% |
| no_tool | 34 | 100.0% | 100.0% |
| parallel_diff_tool | 20 | 0.0% | 0.0% |
| parallel_same_tool | 50 | 0.0% | 0.0% |
| single_tool | 60 | 86.7% | 86.7% |
| wrong_tool_trap | 26 | 80.8% | 80.8% |