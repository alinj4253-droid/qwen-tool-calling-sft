# Eval summary: Base

cases: `/mnt/ssd2/psf/job/qwen-tool-calling-sft/eval/tool_calling_eval.jsonl` (36)
adapter: `(base)`
max_new_tokens=1024, do_sample=False

| Metric | Base |
|---|---:|
| Tool Selection Accuracy | 96.2% |
| Argument Exact Match | 92.3% |
| Argument Key Accuracy | 98.1% |
| Valid Tool Call Format Rate | 100.0% |
| Canonical Tool Format Rate | 0.0% |
| Tool Abstention Accuracy | 90.0% |
| Clean Stop Rate | 66.7% |
| Strict Protocol Success | 11.1% |
| Overall Exact Match | 91.7% |
| Invalid JSON Rate | 0.0% |
| Wrong Tool Rate | 3.8% |
| Missing Argument Rate | 3.8% |
| Extra Argument Rate | 7.7% |

## By category

| Category | N | Overall | Strict Protocol |
|---|---:|---:|---:|
| argument_filling | 8 | 100.0% | 0.0% |
| multi_argument | 4 | 100.0% | 0.0% |
| multi_tool_choice | 6 | 100.0% | 0.0% |
| no_tool | 6 | 100.0% | 33.3% |
| parallel_calls | 2 | 50.0% | 0.0% |
| single_tool | 6 | 83.3% | 0.0% |
| wrong_tool_trap | 4 | 75.0% | 50.0% |