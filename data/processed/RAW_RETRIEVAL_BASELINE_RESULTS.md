# Raw LLM and Retrieval-Augmented Executor Baselines

Setting: clean compact-v4 test split, same JSON parser, state grounding, and closed-loop evaluator as the main experiments.

| method | executor | overall succ | retail succ | tool EM | retail tool EM | pred exec ok | arg value EM |
|---|---|---:|---:|---:|---:|---:|---:|
| Raw LLM direct | Qwen2.5-7B | 0.1214 | 0.2208 | 0.1286 | 0.2338 | 0.9412 | 0.4308 |
| Raw LLM direct | Llama-3.2-3B | 0.2214 | 0.0909 | 0.2286 | 0.1039 | 0.8406 | 0.1304 |
| Retrieval-augmented LLM executor | Qwen2.5-7B | 0.5214 | 0.4935 | 0.5214 | 0.4935 | 1.0000 | 0.3478 |
| Retrieval-augmented LLM executor | Llama-3.2-3B | 0.5357 | 0.5195 | 0.5571 | 0.5584 | 0.9104 | 0.3241 |

Interpretation:

- Raw LLM direct is far below compact-state SFT and FlowPlanner, showing that closed-loop executable tool use requires benchmark adaptation.
- Retrieval hints substantially improve over raw LLM but remain below compact-state SFT on Qwen and far below FlowPlanner on both executors.
- Retrieval-augmented executor uses lexical nearest train-row next-tool/stop hints, the corresponding no-prior SFT executor, and the same state grounding/evaluator.  It is therefore a cleaner baseline than the earlier lexical-nearest diagnostic planner.

Artifacts:

- Retrieval hints: `data/processed/predictions/replan_retrieval_hints_compact_v4/`
- Summary JSON: `data/processed/RAW_RETRIEVAL_BASELINE_RESULTS.json`
- Closed-loop summaries: `data/processed/closed_loop/replan_exec_*_compact_v4.test.state_grounded_v4.summary.json`
