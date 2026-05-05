# FlowPlanner

<p align="center">
  <strong>Feedback-conditioned executable tool-use planning with FM priors and LLM executors</strong>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue">
  <img alt="Benchmark" src="https://img.shields.io/badge/Benchmark-compact--v4-success">
  <img alt="Models" src="https://img.shields.io/badge/Executors-Qwen%20%7C%20Llama-purple">
  <img alt="Reports" src="https://img.shields.io/badge/Reproducible-reports-green">
</p>

FlowPlanner is a benchmark and method package for **feedback-conditioned executable tool use**. It studies how a noisy continuous planning prior can guide an LLM executor while structured grounding keeps tool calls valid.

The current release focuses on a clean compact-v4 replan setting:

- 🧭 A flow-matching (FM) planning prior provides a state-conditioned planning signal.
- 🤖 A Qwen/Llama-style LLM executor emits strict JSON next-actions or stop decisions.
- 🔗 State/tool-result grounding fills executable arguments.
- 🛡️ A conservative retail operation selector handles the hardest DB-backed decisions.

This directory is a curated open-source snapshot. It intentionally excludes large model checkpoints, raw private server paths, process notes and transient experiment ledgers.

## 🔎 Quick Links

| item | path |
|---|---|
| 📊 Main closed-loop results | `bash scripts/reproduce_paper_results.sh main` |
| 🧪 Component ablations and cost | `bash scripts/reproduce_paper_results.sh reports` |
| 📈 Significance tests | `bash scripts/reproduce_paper_results.sh reports` |
| 🔁 Baseline reproducibility | `docs/BASELINE_REPRODUCIBILITY.md` |
| 🗂️ Baseline result index | `docs/BASELINE_REPRODUCIBILITY.md` |
| ⚡ Compute-efficiency figures | `data/processed/compute_efficiency/COMPUTE_EFFICIENCY.md` |

## 🧱 Layout

```text
flowplanner/
  scripts/                         Core method, baseline, training, and evaluation scripts
  configs/                         Optional experiment config placeholders
  data/benchmark/                  Processed benchmark files: tools, tasks, gold plans, source manifests
  data/replan_sft/                 Compact-v4 feedback-conditioned SFT data
  data/toolrl/                     ToolRL-format smoke data
  data/eval_summaries/             Summary JSONs needed to regenerate result tables
  data/processed/                  Compatibility copy used by report scripts
  docs/                            Reproducibility notes
```

## 📊 Main Results

Generated paper-facing reports are written to `results/` by:

```bash
bash scripts/reproduce_paper_results.sh reports
```

The clean release does not track the generated `results/` directory. The summary JSON/MD files needed to regenerate the main tables are kept under `data/processed/`.

Headline compact-v4 closed-loop test result:

| method | overall success | retail success | retail tool EM | arg value EM |
|---|---:|---:|---:|---:|
| no-prior SFT | 0.6571 | 0.3766 | 0.4026 | 0.4387 |
| FM `c_i + compact state` SFT | 0.7357 | 0.5195 | 0.5325 | 0.5613 |
| FlowPlanner final selector | 0.7643 | 0.5714 | 0.5844 | 0.5929 |

## 🚀 Main Experiment Entry Points

Run commands from the `flowplanner/` repository root.

| goal | command or entry file | primary outputs |
|---|---|---|
| 📦 Regenerate all paper-facing reports and compute figures | `bash scripts/reproduce_paper_results.sh reports` | `results/*.md`, `results/*.json`, `data/processed/compute_efficiency/*` |
| 📊 Regenerate only the closed-loop main table | `bash scripts/reproduce_paper_results.sh main` | `results/MAIN_EXPERIMENT_RESULTS.md`, `results/MAIN_EXPERIMENT_RESULTS.json` |
| 🧾 Main compact-v4 result table | `python scripts/write_main_experiment_results.py --root . --out-md results/MAIN_EXPERIMENT_RESULTS.md --out-json results/MAIN_EXPERIMENT_RESULTS.json` | Qwen/Llama Raw LLM, SFT, ToolRL, retrieval executor, and FlowPlanner comparisons |
| 🧪 Main component ablation and training/selector sensitivity | `python scripts/write_ablation_cost_report.py --root . --out-md results/ABLATION_PARAMETER_COSTS.md --out-json results/ABLATION_PARAMETER_COSTS.json` | 4-row component ablation plus appendix sensitivity/cost tables |
| 📈 Paired row-level uncertainty | `python scripts/write_significance_report.py --root . --out-md results/SIGNIFICANCE_REPORT.md --out-json results/SIGNIFICANCE_REPORT.json` | paired bootstrap CI and sign tests |
| 🔍 Raw LLM and retrieval-augmented executor baselines | `results/RAW_RETRIEVAL_BASELINE_RESULTS.md`; commands in `docs/BASELINE_REPRODUCIBILITY.md` | raw prompting and retrieval-prior baseline summaries |
| 🎯 ToolRL GRPO baselines | `scripts/build_toolrl_benchmark_data.py`, `scripts/run_toolrl_lora_full_3gpu.sh`, `scripts/run_toolrl_llama32_3b_3gpu.sh` | Qwen and Llama ToolRL summaries indexed in `results/BASELINE_RESULT_INDEX.md` |
| 🔁 Llama-3.2-3B transfer check | `scripts/run_llama32_main_experiment.sh` | `results/LLAMA32_MAIN_RESULTS.json` and entries in the main table |
| 🤖 FlowPlanner compact-v4 executor training | `scripts/train_fm_prefix_sft.py` | LoRA SFT executor checkpoints under `outputs/conditioned_sft/` |
| 🧭 Schema-time unseen-tool planner diagnostic | `python scripts/write_unseen_tool_generalization_report.py --root . --split all --mode native --out-md results/UNSEEN_TOOL_GENERALIZATION.native.all.md --out-json results/UNSEEN_TOOL_GENERALIZATION.native.all.json` | planner-level unseen-tool analysis |
| ⚡ Compute-efficiency curves | `python scripts/write_compute_efficiency_report.py --root . --out-dir data/processed/compute_efficiency` | compute-efficiency markdown plus PNG/PDF figures |

For exact model paths, adapter placeholders, and evaluation steps, use `docs/BASELINE_REPRODUCIBILITY.md`.

## 🗃️ Data

Core benchmark files:

- `data/benchmark/tools.jsonl`
- `data/benchmark/tasks.jsonl`
- `data/benchmark/gold_plans.jsonl`

Main replan SFT data:

- `data/replan_sft/compact_v4_ci_state_hint/train.jsonl`
- `data/replan_sft/compact_v4_ci_state_hint/dev.jsonl`
- `data/replan_sft/compact_v4_ci_state_hint/test.jsonl`

The compact-v4 data removes oracle `remaining_prior_tool_names` and uses feedback-conditioned compact state plus FM text/state hints.

## 🛠️ Common Commands

Regenerate all paper-facing reports and compute-efficiency figures from included summary files:

```bash
bash scripts/reproduce_paper_results.sh reports
```

See `docs/BASELINE_REPRODUCIBILITY.md` for exact commands for Raw LLM, no-prior SFT, retrieval-augmented executor, ToolRL GRPO, and FlowPlanner final.

Build lightweight baselines:

```bash
python scripts/build_replan_baseline_predictions.py \
  --data-dir data/replan_sft/compact_v4_ci_state_hint \
  --out-dir outputs/predictions/replan_light_baselines_compact_v4
```

Evaluate generated replan predictions with state grounding:

```bash
python scripts/ground_replan_predictions.py \
  --data data/replan_sft/compact_v4_ci_state_hint/test.jsonl \
  --pred outputs/predictions/replan_light_baselines_compact_v4/baseline_state_key_majority.test.jsonl \
  --out outputs/predictions/replan_light_baselines_compact_v4/baseline_state_key_majority.test.state_grounded_v4.jsonl

python scripts/evaluate_replan_execution.py \
  --data data/replan_sft/compact_v4_ci_state_hint/test.jsonl \
  --pred outputs/predictions/replan_light_baselines_compact_v4/baseline_state_key_majority.test.state_grounded_v4.jsonl \
  --out-details outputs/closed_loop/baseline_state_key_majority.test.details.jsonl \
  --summary-out outputs/closed_loop/baseline_state_key_majority.test.summary.json \
  --result-grounding
```

Regenerate reports from included summaries:

```bash
python scripts/write_main_experiment_results.py \
  --root . \
  --out-md results/MAIN_EXPERIMENT_RESULTS.md \
  --out-json results/MAIN_EXPERIMENT_RESULTS.json

python scripts/write_ablation_cost_report.py \
  --root . \
  --out-md results/ABLATION_PARAMETER_COSTS.md \
  --out-json results/ABLATION_PARAMETER_COSTS.json

python scripts/write_compute_efficiency_report.py \
  --root . \
  --out-dir data/processed/compute_efficiency
```

Train a compact-v4 LoRA SFT executor:

```bash
torchrun --nproc_per_node 3 scripts/train_fm_prefix_sft.py \
  --no-prefix \
  --train data/replan_sft/compact_v4_ci_state_hint/train.jsonl \
  --dev data/replan_sft/compact_v4_ci_state_hint/dev.jsonl \
  --model /path/to/Qwen2.5-7B-Instruct \
  --out outputs/conditioned_sft/compact_v4_ci_state_hint_lora \
  --epochs 1 \
  --batch-size 2 \
  --grad-accum 4 \
  --max-length 2048 \
  --lora-r 4 \
  --lora-alpha 8
```

## 🧪 Baselines Included

- SFT no-prior and FM-hint executor code.
- Lightweight closed-loop baselines: always stop, global majority, domain majority, state-key majority, lexical nearest.
- FM path/planner baselines: nearest path, learned path decoder, pointer decoder, next-tool prior.
- ToolRL/VERL bridge scripts and reward code for LoRA GRPO experiments.
- Report scripts for main results, significance tests, unseen-tool diagnostics, and compute-efficiency figures.

## 📝 Notes

- Raw tau2/toolbench/api-bank/GTA sources are not duplicated here. The processed benchmark files and manifests are included.
- Full model checkpoints are not included. Reports record the paths and configurations used in the original experiments.
- Some evaluation scripts require the tau2 runtime and retail/telecom DB files if you want real tool execution rather than row-level metrics.
- `data/processed/compute_efficiency/success_vs_compute_frontier.png` is the recommended compute-efficiency figure draft; `success_by_training_stage.png` is the clearest internal diagnostic version.
