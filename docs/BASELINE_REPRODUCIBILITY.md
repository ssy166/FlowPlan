# Baseline Reproducibility

This document explains how to reproduce the paper-facing FlowPlanner tables, baseline summaries, and compute-efficiency figures from the open-source snapshot.

The snapshot contains lightweight benchmark files, SFT data, summary JSONs, and all scripts needed to regenerate reports. Large model checkpoints, LoRA adapters, raw tau2 databases, ToolRL checkpoints, and temporary run files are intentionally not included.

## 1. Setup

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

For real executable tau2 evaluation, install tau2-bench separately and make the tau2 data directory visible to the evaluator. The included summary JSONs let you regenerate all paper-facing tables without tau2.

Recommended environment variables for full reruns:

```bash
export FLOWPLANNER_ROOT=$PWD
export QWEN_MODEL=/path/to/Qwen2.5-7B-Instruct
export LLAMA32_MODEL=/path/to/Llama-3.2-3B-Instruct
export TAU2_DATA_DIR=/path/to/tau2-bench/repo/data
```

## 2. Regenerate Paper Reports From Included Summaries

These commands do not require model checkpoints.

```bash
bash scripts/reproduce_paper_results.sh reports
```

Equivalent individual commands:

```bash
python scripts/write_main_experiment_results.py \
  --root . \
  --out-md results/MAIN_EXPERIMENT_RESULTS.md \
  --out-json results/MAIN_EXPERIMENT_RESULTS.json

python scripts/write_ablation_cost_report.py \
  --root . \
  --out-md results/ABLATION_PARAMETER_COSTS.md \
  --out-json results/ABLATION_PARAMETER_COSTS.json

python scripts/write_significance_report.py \
  --root . \
  --out-md results/SIGNIFICANCE_REPORT.md \
  --out-json results/SIGNIFICANCE_REPORT.json

python scripts/write_compute_efficiency_report.py \
  --root . \
  --out-dir data/processed/compute_efficiency
```

Main outputs:

- `results/MAIN_EXPERIMENT_RESULTS.md`
- `results/RAW_RETRIEVAL_BASELINE_RESULTS.md`
- `results/SIGNIFICANCE_REPORT.md`
- `results/GROUNDING_COMPONENT_ABLATION.md`
- `data/processed/compute_efficiency/COMPUTE_EFFICIENCY.md`
- `data/processed/compute_efficiency/success_vs_compute_frontier.png`
- `data/processed/compute_efficiency/success_by_training_stage.png`

## 3. Baseline Inventory

### 3.1 Raw LLM direct

Purpose: direct-prompt baseline without benchmark SFT/RL.

Artifacts included:

- `data/processed/closed_loop/replan_exec_qwen25_7b_raw_llm_direct_compact_v4.test.state_grounded_v4.summary.json`
- `data/processed/closed_loop/replan_exec_llama32_3b_raw_llm_direct_compact_v4.test.state_grounded_v4.summary.json`

To rerun generation, use the same generation interface as the SFT executors, with the base model path and the compact-v4 test data. Ground and evaluate with:

```bash
python scripts/ground_replan_predictions.py \
  --data data/replan_sft/compact_v4_ci_state_hint/test.jsonl \
  --pred outputs/predictions/raw_llm_direct.test.jsonl \
  --out outputs/predictions/raw_llm_direct.test.state_grounded_v4.jsonl

python scripts/evaluate_replan_execution.py \
  --data data/replan_sft/compact_v4_ci_state_hint/test.jsonl \
  --pred outputs/predictions/raw_llm_direct.test.state_grounded_v4.jsonl \
  --out-details outputs/closed_loop/raw_llm_direct.test.details.jsonl \
  --summary-out outputs/closed_loop/raw_llm_direct.test.summary.json \
  --result-grounding
```

### 3.2 Compact-state SFT without FM prior

Purpose: clean supervised executor baseline. It sees task, available tools, executed prefix, and compact state, but not FM prior hints.

Train:

```bash
torchrun --nproc_per_node 3 scripts/train_fm_prefix_sft.py \
  --no-prefix \
  --train data/replan_sft/compact_v4_noprior/train.jsonl \
  --dev data/replan_sft/compact_v4_noprior/dev.jsonl \
  --model "$QWEN_MODEL" \
  --out outputs/conditioned_sft/compact_v4_noprior_lora \
  --epochs 1 \
  --batch-size 2 \
  --grad-accum 4 \
  --max-length 2048 \
  --lora-r 4 \
  --lora-alpha 8
```

If `compact_v4_noprior/` is not present in the lightweight snapshot, rebuild it from the processed compact-v4 source using `scripts/build_replan_conditioned_sft_data.py` or use the included summaries for report regeneration.

### 3.3 Retrieval-augmented LLM executor

Purpose: test whether FM prior can be replaced by a nearest-row retrieved prior.

Artifacts included:

- `results/RAW_RETRIEVAL_BASELINE_RESULTS.md`
- `data/processed/closed_loop/replan_exec_qwen25_7b_retrieval_aug_executor_compact_v4.test.state_grounded_v4.summary.json`
- `data/processed/closed_loop/replan_exec_llama32_3b_retrieval_aug_executor_compact_v4.test.state_grounded_v4.summary.json`

To rebuild lightweight retrieval predictions:

```bash
python scripts/build_replan_baseline_predictions.py \
  --data-dir data/replan_sft/compact_v4_ci_state_hint \
  --out-dir outputs/predictions/replan_light_baselines_compact_v4
```

Then ground/evaluate a prediction file:

```bash
python scripts/ground_replan_predictions.py \
  --data data/replan_sft/compact_v4_ci_state_hint/test.jsonl \
  --pred outputs/predictions/replan_light_baselines_compact_v4/baseline_lexical_nearest.test.jsonl \
  --out outputs/predictions/replan_light_baselines_compact_v4/baseline_lexical_nearest.test.state_grounded_v4.jsonl

python scripts/evaluate_replan_execution.py \
  --data data/replan_sft/compact_v4_ci_state_hint/test.jsonl \
  --pred outputs/predictions/replan_light_baselines_compact_v4/baseline_lexical_nearest.test.state_grounded_v4.jsonl \
  --out-details outputs/closed_loop/baseline_lexical_nearest.test.details.jsonl \
  --summary-out outputs/closed_loop/baseline_lexical_nearest.test.summary.json \
  --result-grounding
```

### 3.4 ToolRL GRPO baseline

Purpose: RL-route comparison using the ToolRL/VERL training loop and the same executable replan reward bridge.

Build ToolRL-format data:

```bash
python scripts/build_toolrl_benchmark_data.py \
  --data-dir data/replan_sft/compact_v4_ci_state_hint \
  --out-dir data/toolrl/toolrl_benchmark_replan
```

Train Qwen ToolRL:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 bash scripts/run_toolrl_lora_full_3gpu.sh
```

Train Llama-3.2-3B ToolRL:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
MODEL_PATH="$LLAMA32_MODEL" \
bash scripts/run_toolrl_llama32_3b_3gpu.sh
```

Generate from a ToolRL checkpoint:

```bash
python scripts/run_toolrl_lora_generation.py \
  --model-path "$QWEN_MODEL" \
  --adapter outputs/toolrl_lora_grpo/qwen7b_full3gpu/actor/global_step_582 \
  --data data/toolrl/toolrl_benchmark_replan/test.jsonl \
  --out outputs/predictions/toolrl_grpo_step582.test.jsonl \
  --raw-out outputs/predictions/toolrl_grpo_step582.test.raw.jsonl
```

Ground and evaluate:

```bash
python scripts/ground_replan_predictions.py \
  --data data/toolrl/toolrl_benchmark_replan/test.jsonl \
  --pred outputs/predictions/toolrl_grpo_step582.test.jsonl \
  --out outputs/predictions/toolrl_grpo_step582.test.state_grounded_v4.jsonl

python scripts/evaluate_replan_execution.py \
  --data data/toolrl/toolrl_benchmark_replan/test.jsonl \
  --pred outputs/predictions/toolrl_grpo_step582.test.state_grounded_v4.jsonl \
  --out-details outputs/closed_loop/toolrl_grpo_step582.test.details.jsonl \
  --summary-out outputs/closed_loop/toolrl_grpo_step582.test.summary.json \
  --result-grounding
```

Included ToolRL summaries:

- Qwen step100 / step200 / step582
- Llama-3.2-3B step100

### 3.5 FlowPlanner final

Purpose: main method, state-conditioned FM hint plus structured grounding and strict operation selector.

Key training stages:

1. Train or load FM prior.
2. Build FM/state-hint SFT data.
3. Train LoRA SFT executor.
4. Generate strict JSON predictions.
5. Apply structured grounding.
6. Apply strict operation selector.
7. Evaluate with real tau2 tools.

Representative SFT command:

```bash
torchrun --nproc_per_node 3 scripts/train_fm_prefix_sft.py \
  --no-prefix \
  --train data/replan_sft/compact_v4_ci_state_hint/train.jsonl \
  --dev data/replan_sft/compact_v4_ci_state_hint/dev.jsonl \
  --model "$QWEN_MODEL" \
  --out outputs/conditioned_sft/compact_v4_ci_state_hint_lora \
  --epochs 1 \
  --batch-size 2 \
  --grad-accum 4 \
  --max-length 2048 \
  --lora-r 4 \
  --lora-alpha 8
```

Selector/evaluation entry points:

- `scripts/apply_retail_operation_rule_override.py`
- `scripts/train_retail_operation_selector.py`
- `scripts/ground_replan_predictions.py`
- `scripts/evaluate_replan_execution.py`

## 4. Compute-efficiency Figures

Regenerate:

```bash
python scripts/write_compute_efficiency_report.py \
  --root . \
  --out-dir data/processed/compute_efficiency
```

Recommended figures:

- Paper draft: `data/processed/compute_efficiency/success_vs_compute_frontier.png`
- Internal diagnostic: `data/processed/compute_efficiency/success_by_training_stage.png`
- True compute axis with compression: `data/processed/compute_efficiency/success_vs_end_to_end_gpu_steps_log.png`

## 5. Notes On Comparability

- The main compact-v4 table uses the 140-row no-leak closed-loop test set.
- ToolRL compute-efficiency curves use the 83-row ToolRL-adapted executable split.
- Diagnostic baselines such as always-stop, global-majority, domain-majority, and state-key-majority are sanity checks, not mature competing methods.
- Unseen-tool results are planner-level schema-time analyses, not closed-loop zero-shot executable tool-use claims.
