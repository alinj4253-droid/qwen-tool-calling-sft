# Eval summary: SFT-v2-targeted-p10

cases: `/mnt/ssd2/psf/job/qwen-tool-calling-sft/eval/bfcl_subset.jsonl` (220)
adapter: `runs/qwen3-4b-targeted-p10/final_adapter`
max_new_tokens=1024, do_sample=False

| Metric | SFT-v2-targeted-p10 |
|---|---:|
| Tool Selection Accuracy | 32.8% |
| Argument Exact Match | 19.4% |
| Argument Key Accuracy | 35.9% |
| Valid Tool Call Format Rate | 100.0% |
| Canonical Tool Format Rate | 100.0% |
| Tool Abstention Accuracy | 10.0% |
| Clean Stop Rate | 100.0% |
| Strict Protocol Success | 17.7% |
| Overall Exact Match | 17.7% |
| Invalid JSON Rate | 0.0% |
| Wrong Tool Rate | 67.2% |
| Missing Argument Rate | 78.3% |
| Extra Argument Rate | 0.0% |

## By category

| Category | N | Overall | Strict Protocol |
|---|---:|---:|---:|
| multiple_tools | 40 | 20.0% | 20.0% |
| no_tool | 40 | 10.0% | 10.0% |
| parallel_diff_tool | 30 | 3.3% | 3.3% |
| parallel_same_tool | 50 | 12.0% | 12.0% |
| single_tool | 60 | 33.3% | 33.3% |