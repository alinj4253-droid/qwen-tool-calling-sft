# Eval summary: SFT-v1-targeted-bal

cases: `/mnt/ssd2/psf/job/qwen-tool-calling-sft/eval/bfcl_subset.jsonl` (220)
adapter: `runs/qwen3-4b-targeted-v1-bal/final_adapter`
max_new_tokens=1024, do_sample=False

| Metric | SFT-v1-targeted-bal |
|---|---:|
| Tool Selection Accuracy | 33.3% |
| Argument Exact Match | 22.8% |
| Argument Key Accuracy | 36.9% |
| Valid Tool Call Format Rate | 100.0% |
| Canonical Tool Format Rate | 100.0% |
| Tool Abstention Accuracy | 90.0% |
| Clean Stop Rate | 100.0% |
| Strict Protocol Success | 35.0% |
| Overall Exact Match | 35.0% |
| Invalid JSON Rate | 0.0% |
| Wrong Tool Rate | 66.7% |
| Missing Argument Rate | 75.0% |
| Extra Argument Rate | 0.0% |

## By category

| Category | N | Overall | Strict Protocol |
|---|---:|---:|---:|
| multiple_tools | 40 | 22.5% | 22.5% |
| no_tool | 40 | 90.0% | 90.0% |
| parallel_diff_tool | 30 | 6.7% | 6.7% |
| parallel_same_tool | 50 | 20.0% | 20.0% |
| single_tool | 60 | 33.3% | 33.3% |