# Eval summary: SFT-v1-30k

cases: `/mnt/ssd2/psf/job/qwen-tool-calling-sft/eval/bfcl_subset.jsonl` (220)
adapter: `runs/qwen3-4b-sft-30k-r32/final_adapter`
max_new_tokens=1024, do_sample=False

| Metric | SFT-v1-30k |
|---|---:|
| Tool Selection Accuracy | 26.7% |
| Argument Exact Match | 16.7% |
| Argument Key Accuracy | 32.7% |
| Valid Tool Call Format Rate | 100.0% |
| Canonical Tool Format Rate | 100.0% |
| Tool Abstention Accuracy | 90.0% |
| Clean Stop Rate | 100.0% |
| Strict Protocol Success | 30.0% |
| Overall Exact Match | 30.0% |
| Invalid JSON Rate | 0.0% |
| Wrong Tool Rate | 73.3% |
| Missing Argument Rate | 80.6% |
| Extra Argument Rate | 0.0% |

## By category

| Category | N | Overall | Strict Protocol |
|---|---:|---:|---:|
| multiple_tools | 40 | 20.0% | 20.0% |
| no_tool | 40 | 90.0% | 90.0% |
| parallel_diff_tool | 30 | 0.0% | 0.0% |
| parallel_same_tool | 50 | 2.0% | 2.0% |
| single_tool | 60 | 35.0% | 35.0% |