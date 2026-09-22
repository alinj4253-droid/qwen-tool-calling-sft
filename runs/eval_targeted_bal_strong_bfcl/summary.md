# Eval summary: SFT-v2-targeted-bal-strong

cases: `/mnt/ssd2/psf/job/qwen-tool-calling-sft/eval/bfcl_subset.jsonl` (220)
adapter: `runs/qwen3-4b-targeted-bal-strong/final_adapter`
max_new_tokens=1024, do_sample=False

| Metric | SFT-v2-targeted-bal-strong |
|---|---:|
| Tool Selection Accuracy | 29.4% |
| Argument Exact Match | 19.4% |
| Argument Key Accuracy | 34.2% |
| Valid Tool Call Format Rate | 100.0% |
| Canonical Tool Format Rate | 100.0% |
| Tool Abstention Accuracy | 12.5% |
| Clean Stop Rate | 100.0% |
| Strict Protocol Success | 18.2% |
| Overall Exact Match | 18.2% |
| Invalid JSON Rate | 0.0% |
| Wrong Tool Rate | 70.6% |
| Missing Argument Rate | 78.9% |
| Extra Argument Rate | 0.0% |

## By category

| Category | N | Overall | Strict Protocol |
|---|---:|---:|---:|
| multiple_tools | 40 | 22.5% | 22.5% |
| no_tool | 40 | 12.5% | 12.5% |
| parallel_diff_tool | 30 | 3.3% | 3.3% |
| parallel_same_tool | 50 | 8.0% | 8.0% |
| single_tool | 60 | 35.0% | 35.0% |