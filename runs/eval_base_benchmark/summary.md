# Eval summary: Base

cases: `/mnt/ssd2/psf/job/qwen-tool-calling-sft/eval/benchmark_extended.jsonl` (280)
adapter: `(base)`
max_new_tokens=1024, do_sample=False

| Metric | Base |
|---|---:|
| Tool Selection Accuracy | 85.0% |
| Argument Exact Match | 60.5% |
| Argument Key Accuracy | 88.5% |
| Valid Tool Call Format Rate | 94.5% |
| Canonical Tool Format Rate | 0.0% |
| Tool Abstention Accuracy | 91.7% |
| Clean Stop Rate | 64.3% |
| Strict Protocol Success | 10.7% |
| Overall Exact Match | 67.1% |
| Invalid JSON Rate | 0.0% |
| Wrong Tool Rate | 15.0% |
| Missing Argument Rate | 15.0% |
| Extra Argument Rate | 17.3% |

## By category

| Category | N | Overall | Strict Protocol |
|---|---:|---:|---:|
| distractor | 30 | 63.3% | 0.0% |
| multi_argument | 30 | 93.3% | 0.0% |
| multiple_tools | 30 | 46.7% | 0.0% |
| no_tool | 34 | 100.0% | 58.8% |
| parallel_diff_tool | 20 | 35.0% | 0.0% |
| parallel_same_tool | 50 | 36.0% | 0.0% |
| single_tool | 60 | 78.3% | 0.0% |
| wrong_tool_trap | 26 | 80.8% | 38.5% |