# Eval summary: SFT-v2-targeted-p05

cases: `/mnt/ssd2/psf/job/qwen-tool-calling-sft/eval/bfcl_subset.jsonl` (220)
adapter: `runs/qwen3-4b-targeted-p05/final_adapter`
max_new_tokens=1024, do_sample=False

| Metric | SFT-v2-targeted-p05 |
|---|---:|
| Tool Selection Accuracy | 30.0% |
| Argument Exact Match | 17.8% |
| Argument Key Accuracy | 34.2% |
| Valid Tool Call Format Rate | 100.0% |
| Canonical Tool Format Rate | 100.0% |
| Tool Abstention Accuracy | 12.5% |
| Clean Stop Rate | 100.0% |
| Strict Protocol Success | 16.8% |
| Overall Exact Match | 16.8% |
| Invalid JSON Rate | 0.0% |
| Wrong Tool Rate | 70.0% |
| Missing Argument Rate | 80.0% |
| Extra Argument Rate | 0.0% |

## By category

| Category | N | Overall | Strict Protocol |
|---|---:|---:|---:|
| multiple_tools | 40 | 20.0% | 20.0% |
| no_tool | 40 | 12.5% | 12.5% |
| parallel_diff_tool | 30 | 0.0% | 0.0% |
| parallel_same_tool | 50 | 10.0% | 10.0% |
| single_tool | 60 | 31.7% | 31.7% |