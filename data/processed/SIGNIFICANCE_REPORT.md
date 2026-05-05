# Significance And Paired Difference Report

This report uses paired row-level comparisons on the compact-v4 test set. The confidence interval is a paired bootstrap over rows; the p-value is an exact two-sided sign test over discordant rows. It is intended to separate robust prior-injection gains from small selector gains.

| comparison | metric | n | base | cand | diff | ci95 | wins | losses | p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen state-hint vs no-prior | success | 140 | 0.6571 | 0.7357 | 0.0786 | [0.0000, 0.1571] | 23 | 12 | 0.0895 |
| Qwen state-hint vs no-prior | tool_em | 140 | 0.6714 | 0.7429 | 0.0714 | [-0.0071, 0.1571] | 23 | 13 | 0.1325 |
| Qwen final vs state-hint | success | 140 | 0.7357 | 0.7643 | 0.0286 | [0.0071, 0.0571] | 4 | 0 | 0.1250 |
| Qwen final vs state-hint | tool_em | 140 | 0.7429 | 0.7714 | 0.0286 | [0.0071, 0.0571] | 4 | 0 | 0.1250 |
| Qwen final vs no-prior | success | 140 | 0.6571 | 0.7643 | 0.1071 | [0.0286, 0.1857] | 24 | 9 | 0.0135 |
| Qwen final vs no-prior | tool_em | 140 | 0.6714 | 0.7714 | 0.1000 | [0.0214, 0.1786] | 24 | 10 | 0.0243 |
| Llama state-hint vs no-prior | success | 140 | 0.6214 | 0.7286 | 0.1071 | [0.0357, 0.1786] | 22 | 7 | 0.0081 |
| Llama state-hint vs no-prior | tool_em | 140 | 0.6214 | 0.7429 | 0.1214 | [0.0500, 0.2000] | 24 | 7 | 0.0033 |
| Llama final vs state-hint | success | 140 | 0.7286 | 0.7571 | 0.0286 | [0.0071, 0.0571] | 4 | 0 | 0.1250 |
| Llama final vs state-hint | tool_em | 140 | 0.7429 | 0.7714 | 0.0286 | [0.0071, 0.0571] | 4 | 0 | 0.1250 |
| Llama final vs no-prior | success | 140 | 0.6214 | 0.7571 | 0.1357 | [0.0643, 0.2071] | 25 | 6 | 0.0009 |
| Llama final vs no-prior | tool_em | 140 | 0.6214 | 0.7714 | 0.1500 | [0.0714, 0.2286] | 27 | 6 | 0.0003 |

Interpretation:

- The no-prior -> state-hint comparisons are the cleanest evidence for FM/state-prior injection because they share the same evaluator, grounding, and no selector.
- The state-hint -> final-selector comparisons are targeted but small; report them as a conservative retail-operation improvement, not as the main statistical claim.
- Llama transfer repeats the direction of the state-hint gain, but because the evaluator/grounder is shared, it should be framed as an interface transfer check rather than independent proof of model reasoning.
