# Eval summary: SFT-v2-30k

cases: `/mnt/ssd2/psf/job/qwen-tool-calling-sft/eval/bfcl_subset.jsonl` (220)
adapter: `runs/qwen3-4b-sft-v2-30k-r32/final_adapter`
max_new_tokens=1024, do_sample=False

| Metric | SFT-v2-30k |
|---|---:|
| Tool Selection Accuracy | 26.1% |
| Argument Exact Match | 13.9% |
| Argument Key Accuracy | 31.6% |
| Valid Tool Call Format Rate | 100.0% |
| Canonical Tool Format Rate | 100.0% |
| Tool Abstention Accuracy | 10.0% |
| Clean Stop Rate | 100.0% |
| Strict Protocol Success | 13.2% |
| Overall Exact Match | 13.2% |
| Invalid JSON Rate | 0.0% |
| Wrong Tool Rate | 73.9% |
| Missing Argument Rate | 83.3% |
| Extra Argument Rate | 0.0% |

## By category

| Category | N | Overall | Strict Protocol |
|---|---:|---:|---:|
| multiple_tools | 40 | 17.5% | 17.5% |
| no_tool | 40 | 10.0% | 10.0% |
| parallel_diff_tool | 30 | 0.0% | 0.0% |
| parallel_same_tool | 50 | 0.0% | 0.0% |
| single_tool | 60 | 30.0% | 30.0% |