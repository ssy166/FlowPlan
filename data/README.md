# Data Files

This directory contains the open-source data slice needed for the current FlowPlanner experiments.

## Benchmark

`benchmark/` contains the unified processed benchmark:

- `tools.jsonl`: normalized tool schemas.
- `tasks.jsonl`: normalized task records.
- `gold_plans.jsonl`: gold workflow/action plans.
- `sources_manifest.json`: raw-source provenance.
- `retail_expansion_sources.json`: tau2 retail expansion source summary.

## Replan SFT

`replan_sft/compact_v4_ci_state_hint/` is the main clean feedback-conditioned dataset.

It contains:

- `train.jsonl`
- `dev.jsonl`
- `test.jsonl`
- `manifest.json`

Each row is a chat-style SFT sample whose assistant target is a strict JSON next-action or stop decision.

## ToolRL

`toolrl/toolrl_benchmark_replan_smoke/` is a small JSONL smoke pack for validating the ToolRL data/reward bridge.

The original full ToolRL parquet files and model checkpoints are intentionally not included.

## Evaluation Summaries

`eval_summaries/` contains compact JSON summaries used by the report scripts. Full prediction JSONL files and long traces are omitted from this open-source snapshot.

## Compatibility Copy

`processed/` mirrors the original experiment path convention for report scripts:

- `processed/closed_loop/`
- `processed/predictions/`
- `processed/task_splits/`
- `processed/compute_efficiency/`
- top-level report and benchmark files

This is intentionally a lightweight summary copy, not the full working `data/processed` tree.

`processed/compute_efficiency/` contains the source JSON and generated PNG/PDF figures for the ToolRL-vs-FlowPlanner compute-efficiency comparison.
