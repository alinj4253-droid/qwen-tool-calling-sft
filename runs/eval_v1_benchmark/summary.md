# Eval summary: SFT-v1-30k

cases: `/mnt/ssd2/psf/job/qwen-tool-calling-sft/eval/benchmark_extended.jsonl` (280)
adapter: `runs/qwen3-4b-sft-30k-r32/final_adapter`
max_new_tokens=1024, do_sample=False

| Metric | SFT-v1-30k |
|---|---:|
| Tool Selection Accuracy | 54.5% |
| Argument Exact Match | 46.4% |
| Argument Key Accuracy | 73.9% |
| Valid Tool Call Format Rate | 100.0% |
| Canonical Tool Format Rate | 100.0% |
| Tool Abstention Accuracy | 91.7% |
| Clean Stop Rate | 100.0% |
| Strict Protocol Success | 56.1% |
| Overall Exact Match | 56.1% |
| Invalid JSON Rate | 0.0% |
| Wrong Tool Rate | 45.5% |
| Missing Argument Rate | 47.3% |
| Extra Argument Rate | 5.0% |

## By category

| Category | N | Overall | Strict Protocol |
|---|---:|---:|---:|
| distractor | 30 | 83.3% | 83.3% |
| multi_argument | 30 | 93.3% | 93.3% |
| multiple_tools | 30 | 0.0% | 0.0% |
| no_tool | 34 | 100.0% | 100.0% |
| parallel_diff_tool | 20 | 0.0% | 0.0% |
| parallel_same_tool | 50 | 0.0% | 0.0% |
| single_tool | 60 | 81.7% | 81.7% |
| wrong_tool_trap | 26 | 80.8% | 80.8% |