# Scripts

The scripts are grouped by role rather than by package module.

## Benchmark/Data Builders

- `build_seed_benchmark.py`
- `build_workflow_records.py`
- `build_replan_records.py`
- `build_tau2_gold_replan_records.py`
- `build_replan_conditioned_sft_data.py`
- `build_planner_hint_sft_data.py`
- `merge_sft_splits.py`

## FM Planner And Decoder

- `train_fm_model.py`
- `build_fm_text_data.py`
- `build_replan_fm_text_data.py`
- `train_fm_next_tool_prior.py`
- `decode_fm_nearest_path.py`
- `train_fm_path_decoder.py`
- `train_fm_ar_decoder.py`
- `train_fm_pointer_decoder.py`
- `train_fm_tool_reranker.py`

## LLM Executor

- `train_fm_prefix_sft.py`
- `run_fm_prefix_sft_generation.py`
- `convert_sft_predictions.py`
- `run_test500_experiment.ps1`
- `build_test500_eval_pack.py`
- `evaluate_test500_predictions.py`
- `audit_test500.py`

## Grounding And Closed-Loop Evaluation

- `ground_replan_predictions.py`
- `evaluate_replan_execution.py`
- `evaluate_plans.py`
- `summarize_closed_loop_by_decision.py`
- `check_acceptance.py`

## Baselines

- `build_replan_baseline_predictions.py`
- `make_baseline_predictions.py`
- `evaluate_replan_baselines.py`
- `apply_retail_operation_rule_override.py`

## ToolRL Bridge

- `build_toolrl_benchmark_data.py`
- `toolrl_benchmark_reward.py`
- `run_toolrl_lora_generation.py`
- `run_toolrl_lora_full_3gpu.sh`
- `run_toolrl_lora_smoke.sh`
- `run_toolrl_llama32_3b_3gpu.sh`
- `run_llama32_main_experiment.sh`

## Reports

- `write_main_experiment_results.py`
- `write_ablation_cost_report.py`
- `write_significance_report.py`
- `write_unseen_tool_generalization_report.py`
- `write_compute_efficiency_report.py`
- `reproduce_paper_results.sh`
