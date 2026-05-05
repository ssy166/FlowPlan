# Ablation, Parameter, And Cost Report

This report separates module ablations from parameter sweeps. The table records reproducible GPU-step cost; wall-clock time is included only for curated runs.

## Module Ablations
| method | module | overall succ | retail succ | retail tool EM | pred exec ok | arg value EM | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| no-prior SFT | FM prior | 0.6571 | 0.3766 | 0.4026 | 0.9268 | 0.4387 |  |
| c_i hint | FM prior | 0.6786 | 0.4156 | 0.4286 | 0.9381 | 0.5099 |  |
| c_i + compact-state hint v8 | FM prior | 0.7357 | 0.5195 | 0.5325 | 0.9619 | 0.5613 |  |
| final + strict op selector v3 | operation selector | 0.7643 | 0.5714 | 0.5844 | 0.9619 | 0.5929 |  |
| retail op-rule SFT | operation selector | 0.7500 | 0.5455 | 0.5584 | 0.9515 | 0.5731 |  |
| oracle decision/stage | oracle diagnostic | 0.7643 | 0.5714 | 0.5974 | 0.9510 | 0.5494 |  |
| oracle operation tool | oracle diagnostic | 0.8071 | 0.6494 | 0.6753 | 0.9406 | 0.6206 |  |

## Grounding / Data Ablations
| method | module | overall succ | retail succ | retail tool EM | pred exec ok | arg value EM |
| --- | --- | --- | --- | --- | --- | --- |
| compact-v2 raw generation | grounding | 0.7214 | 0.4935 | 0.7532 | 0.7327 | 0.5613 |
| compact-v2 list fix | grounding | 0.7357 | 0.5195 | 0.7532 | 0.7525 | 0.5613 |
| compact-v2 state grounding v2 | grounding | 0.8071 | 0.6494 | 0.7532 | 0.8713 | 0.6443 |
| compact-v2 state grounding v3 | grounding | 0.8643 | 0.7532 | 0.7532 | 0.9802 | 0.6996 |
| compact-v3 oracle prior + DB grounding v4 | grounding/oracle | 0.9714 | 0.9481 | 0.9481 | 0.9902 | 0.8103 |
| compact-v4 clean DB grounding v4 | grounding/clean | 0.6571 | 0.3766 | 0.4026 | 0.9268 | 0.4387 |

## FM Injection Ablations
| method | module | overall succ/tool EM | row tool EM | retail succ | retail tool EM | arg value EM |
| --- | --- | --- | --- | --- | --- | --- |
| soft-prefix y_hat | injection | - | 0.8916 | - | 0.5500 | 0.7698 |
| text hint c_i | injection | - | 0.8916 | - | 0.5500 | 0.7599 |
| structured hint | injection | - | 0.8916 | - | 0.5500 | 0.7867 |
| compact-v4 c_i text hint | injection | 0.6786 | 0.6857 | 0.4156 | 0.4286 | 0.5099 |
| compact-v4 c_i state text hint | injection | 0.7357 | 0.7429 | 0.5195 | 0.5325 | 0.5613 |
| multi-candidate hint | injection | 0.7000 | 0.7000 | 0.4545 | 0.4545 | 0.4585 |
| noisy-prior wording | injection | 0.6714 | 0.6786 | 0.4026 | 0.4156 | 0.4901 |

## Parameter Analysis: SFT / Data / Selector
| setting | family | overall succ | retail succ | retail tool EM | pred exec ok | arg value EM |
| --- | --- | --- | --- | --- | --- | --- |
| state hint mixed/dropout 1 epoch | SFT epoch/data | 0.7357 | 0.5195 | 0.5325 | 0.9619 | 0.5613 |
| state hint mixed/dropout 2 epoch | SFT epoch/data | 0.7286 | 0.5065 | 0.5195 | 0.9500 | 0.5415 |
| predicted-only hint | SFT epoch/data | 0.7357 | 0.5195 | 0.5325 | 0.9524 | 0.5573 |
| predicted-only + stop hint | SFT epoch/data | 0.7357 | 0.5195 | 0.5325 | 0.9619 | 0.5455 |
| retail oversampling v1 | data weighting | 0.7357 | 0.5195 | 0.5325 | 0.9615 | 0.5494 |
| retail oversampling v2 | data weighting | 0.7357 | 0.5195 | 0.5325 | 0.9612 | 0.5455 |
| retail op oversampling v3 | data weighting | 0.7143 | 0.4805 | 0.4935 | 0.9583 | 0.5336 |
| selector threshold t=0.9 | selector threshold | 0.7571 | 0.5584 | 0.5714 | 0.9524 | 0.5929 |
| selector threshold t=0.9 + legality v3 | selector threshold | 0.7643 | 0.5714 | 0.5844 | 0.9619 | 0.5929 |

## Parameter Analysis: FM Prior
| setting | family | train acc | dev acc | dev retail | test acc | test retail |
| --- | --- | --- | --- | --- | --- | --- |
| v8 c_i + compact state | FM prior parameter | - | - | - | - | - |
| seed11 | FM prior parameter | - | - | - | - | - |
| seed13 | FM prior parameter | - | - | - | - | - |
| result/intent tokens v11 | FM prior parameter | - | - | - | - | - |
| weighted rw2/post2/op1.5 | FM prior parameter | - | - | - | - | - |
| stage aux v13c | FM prior parameter | - | - | - | - | - |
| candidate reranker v1 | FM prior parameter | - | - | - | - | - |

## Time And GPU Cost
| run | type | train rows | epochs | steps | GPUs | GPU-steps | wall h | GPU h | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FM endpoint rollout v4 | FM / shallow prior | - | 20 | 280 | 1 | 280 | - | - | single-GPU/lightweight run; wall time not timestamped |
| FM replan small-noise v2 | FM / shallow prior | - | - | - | 1 | - | - | - | single-GPU/lightweight run; wall time not timestamped |
| compact-v4 no-prior SFT | LoRA SFT | 1987 | 0.9653 | 80 | 3 | 240 | - | - | wall time not included in the release snapshot |
| v8 state-hint SFT | LoRA SFT | 1987 | 0.9653 | 80 | 3 | 240 | - | - | wall time not included in the release snapshot |
| v8 state-hint 2 epoch | LoRA SFT | 1987 | 1.9291 | 160 | 3 | 480 | - | - | wall time not included in the release snapshot |
| candidate-hint SFT | LoRA SFT | 1987 | 0.9653 | 80 | 3 | 240 | - | - | wall time not included in the release snapshot |
| retail op-rule SFT | LoRA SFT | 1987 | 0.9653 | 80 | 3 | 240 | - | - | wall time not included in the release snapshot |
| ToolRL GRPO full step582 | GRPO | - | 1 | 582 | 3 | 1746 | 1.7665 | 5.2995 | wall time copied from the curated cost summary; excludes separate generation evaluation |

## Cost Interpretation
- Main compact-v4 LoRA SFT runs use 3 H800 GPUs, LoRA r=4/alpha=8, max length 2048, global batch 24.
- A one-epoch compact-v4 SFT is 80 optimizer steps, or 240 GPU-steps. The 2-epoch check doubles this to 160 steps / 480 GPU-steps and did not improve test metrics.
- ToolRL GRPO used 3 GPUs, LoRA r=8/alpha=16/dropout=0.05, rollout n=2, and 582 training steps.
- Shallow FM/prior/selector experiments are low-cost relative to 7B SFT/GRPO; they are best reported by accuracy and used to motivate the final selector rather than as expensive training claims.
