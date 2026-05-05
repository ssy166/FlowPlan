# Qwen vs Llama Prediction Disagreement

Setting: compact-v4 test split. Raw output/tool agreement is computed on state-conditioned FM hint predictions before selector; final tool/argument agreement is computed after strict selector and state grounding.

| group | n | raw output exact | raw parsed exact | raw tool agreement | raw arg agreement | final tool agreement | final arg agreement |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 140 | 0.8286 | 0.8286 | 1.0000 | 0.8286 | 1.0000 | 0.8714 |
| retail | 77 | 0.6883 | 0.6883 | 1.0000 | 0.6883 | 1.0000 | 0.7662 |
| telecom | 63 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Interpretation: Qwen and Llama state-hint outputs are highly aligned, especially on saturated telecom rows.  This means the Llama result should be used only as an interface transfer check, not as strong independent evidence that a different base model reasoned its way to the same policy.  The shared training data, hint format, grounding, selector, and evaluator likely explain much of the agreement.
