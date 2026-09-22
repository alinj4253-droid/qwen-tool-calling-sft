# Base vs SFT: tool-calling evaluation

| Metric | base | sft-30k | Delta |
|---|---:|---:|---:|
| Tool Selection Accuracy | 96.2% | 92.3% | -3.8pt |
| Argument Exact Match | 92.3% | 92.3% | +0.0pt |
| Argument Key Accuracy | 98.1% | 96.2% | -1.9pt |
| Valid Tool Call Format Rate | 100.0% | 100.0% | +0.0pt |
| Canonical <tool_call> Format Rate | 0.0% | 100.0% | +100.0pt |
| No-Tool Accuracy | 90.0% | 100.0% | +10.0pt |
| Clean Stop Rate | 66.7% | 100.0% | +33.3pt |
| Overall Exact Match | 91.7% | 94.4% | +2.8pt |
| Invalid JSON Rate | 0.0% | 0.0% | +0.0pt |
| Wrong Tool Rate | 3.8% | 7.7% | +3.8pt |
| Missing Argument Rate | 3.8% | 7.7% | +3.8pt |
| Extra Argument Rate | 7.7% | 3.8% | -3.8pt |

## By category (Overall Exact Match)

| Category | base | sft-30k | Delta |
|---|---:|---:|---:|
| argument_filling | 100.0% | 100.0% | +0.0pt |
| multi_argument | 100.0% | 100.0% | +0.0pt |
| multi_tool_choice | 100.0% | 100.0% | +0.0pt |
| no_tool | 100.0% | 100.0% | +0.0pt |
| parallel_calls | 50.0% | 0.0% | -50.0pt |
| single_tool | 83.3% | 100.0% | +16.7pt |
| wrong_tool_trap | 75.0% | 100.0% | +25.0pt |
