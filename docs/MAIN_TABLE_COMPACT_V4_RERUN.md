# Compact-v4 Main Test Rerun

Rerun tag: `main_table_rerun_20260505`

All rows below were regenerated on the updated compact-v4 test set only. The evaluation uses the same structured grounding and execution evaluator for every method.

Test rows: 140; retail rows: 77; telecom rows: 63.

| # | executor | method | overall succ | retail succ | tool EM | retail tool EM | pred exec ok | arg value EM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Qwen2.5-7B | Raw LLM direct | 0.1214 | 0.2208 | 0.1286 | 0.2338 | 0.9412 | 0.4308 |
| 2 | Qwen2.5-7B | Compact-state SFT w/o FM prior | 0.6571 | 0.3766 | 0.6714 | 0.4026 | 0.9268 | 0.4387 |
| 3 | Qwen2.5-7B | ToolRL GRPO | 0.4786 | 0.5065 | 0.4929 | 0.5325 | 0.9643 | 0.5257 |
| 4 | Qwen2.5-7B | Retrieval-augmented LLM executor | 0.5214 | 0.4935 | 0.5214 | 0.4935 | 1.0000 | 0.3478 |
| 5 | Qwen2.5-7B | FlowPlanner | 0.7643 | 0.5714 | 0.7714 | 0.5844 | 0.9619 | 0.5929 |
| 6 | Llama-3.2-3B | Raw LLM direct | 0.2000 | 0.0779 | 0.2071 | 0.0909 | 0.8060 | 0.1225 |
| 7 | Llama-3.2-3B | Compact-state SFT w/o FM prior | 0.6214 | 0.3117 | 0.6286 | 0.3247 | 0.9318 | 0.4348 |
| 8 | Llama-3.2-3B | ToolRL GRPO | 0.1357 | 0.1169 | 0.1500 | 0.1429 | 0.9362 | 0.1028 |
| 9 | Llama-3.2-3B | Retrieval-augmented LLM executor | 0.5286 | 0.5065 | 0.5500 | 0.5455 | 0.9091 | 0.3241 |
| 10 | Llama-3.2-3B | FlowPlanner | 0.7571 | 0.5584 | 0.7714 | 0.5844 | 0.9524 | 0.5573 |

Notes:

- `Raw LLM direct` uses the base instruct model without LoRA.
- `ToolRL GRPO` uses the available ToolRL checkpoint for each executor and the same updated compact-v4 test rows.
- `Retrieval-augmented LLM executor` injects a lexical-nearest training-row prior into the same no-prior SFT executor.
- `FlowPlanner` uses the state-conditioned FM text hint plus the conservative retail operation selector.
