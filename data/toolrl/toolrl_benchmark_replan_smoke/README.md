# ToolRL Benchmark Data

VERL/ToolRL RLHF parquet data derived from this benchmark's chat-style SFT rows.

Columns:
- `data_source`: should route to the benchmark reward function.
- `prompt`: list of chat messages consumed by VERL `RLHFDataset`.
- `reward_model.ground_truth`: JSON string with gold stop/actions and available tools.
- `extra_info`: ids and split/domain metadata.

Use with `scripts/toolrl_benchmark_reward.py` or copy that reward into ToolRL's `verl/utils/reward_score` package.

- train: 16 rows at `['data/processed/toolrl_benchmark_replan_smoke/train.jsonl']`
- dev: 8 rows at `['data/processed/toolrl_benchmark_replan_smoke/dev.jsonl']`
- test: 8 rows at `['data/processed/toolrl_benchmark_replan_smoke/test.jsonl']`
