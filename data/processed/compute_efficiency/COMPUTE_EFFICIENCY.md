# Compute Efficiency Curves

This report evaluates FlowPlanner and ToolRL checkpoints on the same 83-row executable ToolRL test split. The curves are intended as compute-efficiency diagnostics, not as a replacement for the clean compact-v4 140-row main table.

Cost conventions:

- `LLM GPU-steps` counts only LLM executor optimization steps multiplied by GPU count.
- `End-to-end GPU-steps` adds the FM prior training/rollout cost used by FlowPlanner. The strict selector is deterministic and counted as zero GPU cost.
- ToolRL points use GRPO checkpoint steps multiplied by 3 GPUs.
- FlowPlanner SFT points use 80 optimizer steps on 3 GPUs; FM prior cost is 280 single-GPU steps.

Figures:

- `success_vs_llm_gpu_steps.png` / `.pdf`
- `success_vs_end_to_end_gpu_steps.png` / `.pdf`
- `success_vs_end_to_end_gpu_steps_log.png` / `.pdf`
- `success_vs_compute_frontier.png` / `.pdf`
- `success_by_training_stage.png` / `.pdf`

| series | method | executor | n | LLM GPU-steps | end-to-end GPU-steps | overall success | retail success | tool EM | pred exec ok | arg value EM |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FlowPlanner-Qwen | Compact-state SFT w/o FM | Qwen2.5-7B | 83 | 240 | 240 | 0.8072 | 0.2000 | 0.8193 | 0.9111 | 0.5391 |
| FlowPlanner-Qwen | FM prior text hint | Qwen2.5-7B | 83 | 240 | 520 | 0.8675 | 0.4500 | 0.8795 | 0.8958 | 0.6406 |
| FlowPlanner-Qwen | State-conditioned FM hint | Qwen2.5-7B | 83 | 240 | 520 | 0.9036 | 0.6000 | 0.9157 | 0.9167 | 0.6953 |
| FlowPlanner-Qwen | State-conditioned FM hint + strict selector | Qwen2.5-7B | 83 | 240 | 520 | 0.9036 | 0.6000 | 0.9157 | 0.9167 | 0.6953 |
| ToolRL-Qwen | ToolRL GRPO step100 | Qwen2.5-7B | 83 | 300 | 300 | 0.2410 | 0.3500 | 0.2410 | 0.9242 | 0.5391 |
| ToolRL-Qwen | ToolRL GRPO step200 | Qwen2.5-7B | 83 | 600 | 600 | 0.5422 | 0.5000 | 0.5422 | 0.9583 | 0.6406 |
| ToolRL-Qwen | ToolRL GRPO step582 | Qwen2.5-7B | 83 | 1746 | 1746 | 0.5422 | 0.5000 | 0.5542 | 0.9792 | 0.6562 |
| ToolRL-Llama | ToolRL GRPO step100 | Llama-3.2-3B | 83 | 300 | 300 | 0.3735 | 0.0000 | 0.3855 | 0.7647 | 0.2500 |

Reading guide:

- The left panel in each figure reports overall next-action/stop success.
- The right panel reports retail-only success, which is the harder DB-backed operation subset.
- FlowPlanner reaches the strongest point after one 3-GPU SFT pass plus the lightweight FM prior. ToolRL improves early but plateaus between step200 and step582 on this split.
