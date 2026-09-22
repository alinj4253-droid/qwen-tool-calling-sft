# Eval summary: SFT-v2-30k

cases: `/mnt/ssd2/psf/job/qwen-tool-calling-sft/eval/benchmark_extended.jsonl` (280)
adapter: `runs/qwen3-4b-sft-v2-30k-r32/final_adapter`
max_new_tokens=1024, do_sample=False

| Metric | SFT-v2-30k |
|---|---:|
| Tool Selection Accuracy | 54.5% |
| Argument Exact Match | 50.5% |
| Argument Key Accuracy | 74.5% |
| Valid Tool Call Format Rate | 100.0% |
| Canonical Tool Format Rate | 100.0% |
| Tool Abstention Accuracy | 58.3% |
| Clean Stop Rate | 100.0% |
| Strict Protocol Success | 52.1% |
| Overall Exact Match | 52.1% |
| Invalid JSON Rate | 0.0% |
| Wrong Tool Rate | 45.5% |
| Missing Argument Rate | 45.5% |
| Extra Argument Rate | 4.5% |

## By category

| Category | N | Overall | Strict Protocol |
|---|---:|---:|---:|
| distractor | 30 | 86.7% | 86.7% |
| multi_argument | 30 | 93.3% | 93.3% |
| multiple_tools | 30 | 0.0% | 0.0% |
| no_tool | 34 | 100.0% | 100.0% |
| parallel_diff_tool | 20 | 0.0% | 0.0% |
| parallel_same_tool | 50 | 0.0% | 0.0% |
| single_tool | 60 | 95.0% | 95.0% |
| wrong_tool_trap | 26 | 3.8% | 3.8% |