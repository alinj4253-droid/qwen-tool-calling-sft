# Eval summary: SFT-v1-targeted-bal

cases: `/mnt/ssd2/psf/job/qwen-tool-calling-sft/eval/tool_calling_eval.jsonl` (36)
adapter: `runs/qwen3-4b-targeted-v1-bal/final_adapter`
max_new_tokens=1024, do_sample=False

| Metric | SFT-v1-targeted-bal |
|---|---:|
| Tool Selection Accuracy | 92.3% |
| Argument Exact Match | 92.3% |
| Argument Key Accuracy | 96.2% |
| Valid Tool Call Format Rate | 100.0% |
| Canonical Tool Format Rate | 100.0% |
| Tool Abstention Accuracy | 100.0% |
| Clean Stop Rate | 100.0% |
| Strict Protocol Success | 94.4% |
| Overall Exact Match | 94.4% |
| Invalid JSON Rate | 0.0% |
| Wrong Tool Rate | 7.7% |
| Missing Argument Rate | 7.7% |
| Extra Argument Rate | 0.0% |

## By category

| Category | N | Overall | Strict Protocol |
|---|---:|---:|---:|
| argument_filling | 8 | 100.0% | 100.0% |
| multi_argument | 4 | 100.0% | 100.0% |
| multi_tool_choice | 6 | 100.0% | 100.0% |
| no_tool | 6 | 100.0% | 100.0% |
| parallel_calls | 2 | 0.0% | 0.0% |
| single_tool | 6 | 100.0% | 100.0% |
| wrong_tool_trap | 4 | 100.0% | 100.0% |