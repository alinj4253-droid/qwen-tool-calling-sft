# Eval summary: SFT-v2-targeted-p10

cases: `/mnt/ssd2/psf/job/qwen-tool-calling-sft/eval/tool_calling_eval.jsonl` (36)
adapter: `runs/qwen3-4b-targeted-p10/final_adapter`
max_new_tokens=1024, do_sample=False

| Metric | SFT-v2-targeted-p10 |
|---|---:|
| Tool Selection Accuracy | 92.3% |
| Argument Exact Match | 84.6% |
| Argument Key Accuracy | 96.2% |
| Valid Tool Call Format Rate | 100.0% |
| Canonical Tool Format Rate | 100.0% |
| Tool Abstention Accuracy | 80.0% |
| Clean Stop Rate | 100.0% |
| Strict Protocol Success | 83.3% |
| Overall Exact Match | 83.3% |
| Invalid JSON Rate | 0.0% |
| Wrong Tool Rate | 7.7% |
| Missing Argument Rate | 7.7% |
| Extra Argument Rate | 7.7% |

## By category

| Category | N | Overall | Strict Protocol |
|---|---:|---:|---:|
| argument_filling | 8 | 87.5% | 87.5% |
| multi_argument | 4 | 100.0% | 100.0% |
| multi_tool_choice | 6 | 100.0% | 100.0% |
| no_tool | 6 | 100.0% | 100.0% |
| parallel_calls | 2 | 0.0% | 0.0% |
| single_tool | 6 | 83.3% | 83.3% |
| wrong_tool_trap | 4 | 50.0% | 50.0% |