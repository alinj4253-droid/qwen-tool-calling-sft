# Eval summary: SFT-v2-targeted-bal

cases: `/mnt/ssd2/psf/job/qwen-tool-calling-sft/eval/bfcl_subset.jsonl` (220)
adapter: `runs/qwen3-4b-targeted-bal/final_adapter`
max_new_tokens=1024, do_sample=False

| Metric | SFT-v2-targeted-bal |
|---|---:|
| Tool Selection Accuracy | 30.6% |
| Argument Exact Match | 17.2% |
| Argument Key Accuracy | 34.3% |
| Valid Tool Call Format Rate | 100.0% |
| Canonical Tool Format Rate | 100.0% |
| Tool Abstention Accuracy | 12.5% |
| Clean Stop Rate | 100.0% |
| Strict Protocol Success | 16.4% |
| Overall Exact Match | 16.4% |
| Invalid JSON Rate | 0.0% |
| Wrong Tool Rate | 69.4% |
| Missing Argument Rate | 80.6% |
| Extra Argument Rate | 0.0% |

## By category

| Category | N | Overall | Strict Protocol |
|---|---:|---:|---:|
| multiple_tools | 40 | 20.0% | 20.0% |
| no_tool | 40 | 12.5% | 12.5% |
| parallel_diff_tool | 30 | 0.0% | 0.0% |
| parallel_same_tool | 50 | 10.0% | 10.0% |
| single_tool | 60 | 30.0% | 30.0% |