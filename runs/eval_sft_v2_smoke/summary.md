# Eval summary: SFT-v2-smoke

cases: `/mnt/ssd2/psf/job/qwen-tool-calling-sft/eval/tool_calling_eval.jsonl` (36)
adapter: `runs/qwen3-4b-sft-v2-smoke/final_adapter`
max_new_tokens=1024, do_sample=False

| Metric | SFT-v2-smoke |
|---|---:|
| Tool Selection Accuracy | 96.2% |
| Argument Exact Match | 92.3% |
| Argument Key Accuracy | 98.1% |
| Valid Tool Call Format Rate | 100.0% |
| Canonical Tool Format Rate | 100.0% |
| Tool Abstention Accuracy | 90.0% |
| Clean Stop Rate | 100.0% |
| Strict Protocol Success | 91.7% |
| Overall Exact Match | 91.7% |
| Invalid JSON Rate | 0.0% |
| Wrong Tool Rate | 3.8% |
| Missing Argument Rate | 3.8% |
| Extra Argument Rate | 3.8% |

## By category

| Category | N | Overall | Strict Protocol |
|---|---:|---:|---:|
| argument_filling | 8 | 100.0% | 100.0% |
| multi_argument | 4 | 100.0% | 100.0% |
| multi_tool_choice | 6 | 100.0% | 100.0% |
| no_tool | 6 | 100.0% | 100.0% |
| parallel_calls | 2 | 0.0% | 0.0% |
| single_tool | 6 | 100.0% | 100.0% |
| wrong_tool_trap | 4 | 75.0% | 75.0% |