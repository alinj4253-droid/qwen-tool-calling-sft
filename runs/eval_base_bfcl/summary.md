# Eval summary: Base

cases: `/mnt/ssd2/psf/job/qwen-tool-calling-sft/eval/bfcl_subset.jsonl` (220)
adapter: `(base)`
max_new_tokens=1024, do_sample=False

| Metric | Base |
|---|---:|
| Tool Selection Accuracy | 90.6% |
| Argument Exact Match | 73.9% |
| Argument Key Accuracy | 94.2% |
| Valid Tool Call Format Rate | 98.3% |
| Canonical Tool Format Rate | 0.0% |
| Tool Abstention Accuracy | 27.5% |
| Clean Stop Rate | 70.5% |
| Strict Protocol Success | 1.8% |
| Overall Exact Match | 63.2% |
| Invalid JSON Rate | 0.0% |
| Wrong Tool Rate | 9.4% |
| Missing Argument Rate | 12.8% |
| Extra Argument Rate | 0.6% |

## By category

| Category | N | Overall | Strict Protocol |
|---|---:|---:|---:|
| multiple_tools | 40 | 72.5% | 0.0% |
| no_tool | 40 | 27.5% | 10.0% |
| parallel_diff_tool | 30 | 73.3% | 0.0% |
| parallel_same_tool | 50 | 64.0% | 0.0% |
| single_tool | 60 | 75.0% | 0.0% |