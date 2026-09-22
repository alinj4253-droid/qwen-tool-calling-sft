# Eval summary: GRPO-smoke

cases: `/mnt/ssd2/psf/job/qwen-tool-calling-sft/eval/bfcl_subset.jsonl` (220)
adapter: `runs/grpo-smoke/final_adapter`
max_new_tokens=1024, do_sample=False

| Metric | GRPO-smoke |
|---|---:|
| Tool Selection Accuracy | 36.1% |
| Argument Exact Match | 23.9% |
| Argument Key Accuracy | 38.2% |
| Valid Tool Call Format Rate | 100.0% |
| Canonical Tool Format Rate | 100.0% |
| Tool Abstention Accuracy | 90.0% |
| Clean Stop Rate | 100.0% |
| Strict Protocol Success | 35.9% |
| Overall Exact Match | 35.9% |
| Invalid JSON Rate | 0.0% |
| Wrong Tool Rate | 63.9% |
| Missing Argument Rate | 73.3% |
| Extra Argument Rate | 0.0% |

## By category

| Category | N | Overall | Strict Protocol |
|---|---:|---:|---:|
| multiple_tools | 40 | 22.5% | 22.5% |
| no_tool | 40 | 90.0% | 90.0% |
| parallel_diff_tool | 30 | 6.7% | 6.7% |
| parallel_same_tool | 50 | 24.0% | 24.0% |
| single_tool | 60 | 33.3% | 33.3% |