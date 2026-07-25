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

`replan_sft/test500/` is an expanded same-domain tau2 retail/telecom test set for reviewer-response evaluation:

- `test.jsonl`: 500 chat-style replan SFT rows.
- `manifest.json`: construction notes and count summary.
- `model_eval_pack.jsonl`: derived prompt pack for `scripts/run_toolrl_inference.py`.

The split has 105 retail rows and 395 telecom rows. It keeps the legacy telecom replan rows available locally and expands with tau2 test-set gold-prefix and terminal stop states. The retail part is rebuilt from tau2 retail test workflows because the exact legacy 77-row retail artifact was not present locally.

`replan_sft/test800/` is the larger same-domain evaluation split:

- `test.jsonl`: 800 chat-style replan SFT rows.
- `manifest.json`: construction notes and count summary.
- `model_eval_pack.jsonl`: derived prompt pack for `scripts/run_toolrl_inference.py`.

It has 105 retail rows and 695 telecom rows. Retail is capped by the available tau2 retail test workflow prefix/terminal rows; the added scale comes from sampled tau2 telecom test prefix rows plus terminal stop states.

## ToolRL

`toolrl/toolrl_benchmark_replan_smoke/` is a small JSONL smoke pack for validating the ToolRL data/reward bridge.

The original full ToolRL parquet files and model checkpoints are intentionally not included.

## Generated Artifacts

Generated result summaries, prediction files, report markdown, and compute-efficiency figures are intentionally not tracked in the clean release.

Local experiment runs may create:

- `data/processed/`
- `data/eval_summaries/`
- `results/`
- `outputs/`

These paths are ignored by Git.
