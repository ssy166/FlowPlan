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

## Generated Artifacts

Generated result summaries, prediction files, report markdown, and compute-efficiency figures are intentionally not tracked in the clean release.

Local experiment runs may create:

- `data/processed/`
- `data/eval_summaries/`
- `results/`

These paths are ignored by Git.
