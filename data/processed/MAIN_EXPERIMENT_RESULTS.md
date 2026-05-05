# Main Experiment Results

This report freezes the current clean compact-v4 closed-loop evaluation tables.
Primary acceptance uses six metrics: overall success, retail success, overall tool EM, retail tool EM, predicted execution ok, and argument value EM. A result passes if at least four metrics strictly improve and safety thresholds hold.

## Main Test Comparison
| method | group | overall succ | retail succ | telecom succ | tool EM | retail tool EM | pred exec ok | arg value EM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SFT no-prior compact-v4 | clean supervised baseline | 0.6571 | 0.3766 | 1.0000 | 0.6714 | 0.4026 | 0.9268 | 0.4387 |
| SFT + FM c_i state hint v8 | current clean baseline | 0.7357 | 0.5195 | 1.0000 | 0.7429 | 0.5325 | 0.9619 | 0.5613 |
| Llama-3.2-3B no-prior compact-v4 | second-base-model baseline | 0.6214 | 0.3117 | 1.0000 | 0.6214 | 0.3117 | 0.9186 | 0.4229 |
| Llama-3.2-3B + FM c_i state hint | second-base-model FM hint | 0.7286 | 0.5065 | 1.0000 | 0.7429 | 0.5325 | 0.9524 | 0.5296 |
| Llama-3.2-3B FM hint + strict selector | second-base-model final | 0.7571 | 0.5584 | 1.0000 | 0.7714 | 0.5844 | 0.9524 | 0.5573 |

## Second Base Model: Llama-3.2-3B
| split | method | overall succ | retail succ | tool EM | retail tool EM | pred exec ok | arg value EM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dev | Llama-3.2-3B no-prior compact-v4 | 0.6942 | 0.3019 | 0.7107 | 0.3396 | 0.8916 | 0.6111 |
| dev | Llama-3.2-3B + FM c_i state hint | 0.7603 | 0.4528 | 0.7686 | 0.4717 | 0.9302 | 0.6167 |
| dev | Llama-3.2-3B FM hint + strict selector | 0.7686 | 0.4717 | 0.7851 | 0.5094 | 0.9186 | 0.6278 |
| test | Llama-3.2-3B no-prior compact-v4 | 0.6214 | 0.3117 | 0.6214 | 0.3117 | 0.9186 | 0.4229 |
| test | Llama-3.2-3B + FM c_i state hint | 0.7286 | 0.5065 | 0.7429 | 0.5325 | 0.9524 | 0.5296 |
| test | Llama-3.2-3B FM hint + strict selector | 0.7571 | 0.5584 | 0.7714 | 0.5844 | 0.9524 | 0.5573 |

This table is a transfer check on the same compact-v4 protocol. It tests whether the FM/state-hint interface remains usable when the executor base model is changed from Qwen2.5-7B-Instruct to Llama-3.2-3B-Instruct.

## Light Baselines On Compact-v4
| baseline | overall succ | retail succ | telecom succ | tool EM | action succ | stop succ | pred exec ok | arg value EM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Always stop | - | - | - | - | - | - | - | - |
| Global majority | - | - | - | - | - | - | - | - |
| Domain majority | - | - | - | - | - | - | - | - |
| State-key majority | - | - | - | - | - | - | - | - |
| Lexical nearest train row | - | - | - | - | - | - | - | - |

## Ablations And Upper Bounds
| method | group | overall succ | retail succ | retail tool EM | pred exec ok | arg value EM |
| --- | --- | --- | --- | --- | --- | --- |
| SFT + FM c_i state hint v8 | current clean baseline | 0.7357 | 0.5195 | 0.5325 | 0.9619 | 0.5613 |
| FM-hint SFT + strict retail op selector v3 | main method | - | - | - | - | - |
| Retail op-rule SFT v1 | trained ablation | - | - | - | - | - |
| Oracle retail decision/stage | oracle diagnostic | - | - | - | - | - |
| Oracle retail operation tool | oracle diagnostic | - | - | - | - | - |

## Retail Decision Cases
| method | case | n | success | tool EM | pred exec ok | arg value EM |
| --- | --- | --- | --- | --- | --- | --- |

## Acceptance
- `final_vs_v8`: passed=False, wins=0, ties=0, losses=0; replay_ok=-, pred_exec_ok=-.
- `final_vs_no_prior`: passed=False, wins=0, ties=0, losses=0; replay_ok=-, pred_exec_ok=-.

## Notes
- ToolRL is retained as a secondary RL baseline. Its step582 closed-loop evaluation is on the original replan rows, so it should not replace compact-v4 acceptance comparisons.
- The light baselines are intentionally cheap non-LLM references: always-stop, majority, state-key majority, and lexical nearest. Their role is to show that closed-loop success is not explained by stop bias or memorized shallow state keys.
- Oracle decision/stage and oracle operation-tool rows are diagnostics. They are useful for locating the bottleneck but are not clean baselines.

## Artifact Index
- JSON table: `data/processed/MAIN_EXPERIMENT_RESULTS.json`
- Light baseline predictions: `data/processed/predictions/replan_light_baselines_compact_v4/`
- Light baseline closed-loop summaries: `data/processed/closed_loop/replan_light_baselines_compact_v4/`
