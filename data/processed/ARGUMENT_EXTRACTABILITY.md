# Argument Extractability

Counts how often tau2 gold action argument values are directly visible in the task prompt.

| split | tasks | actions | args | visible args | visible arg ratio | value atoms | visible atoms | visible atom ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 4861 | 6108 | 14668 | 3084 | 0.210 | 14898 | 3120 | 0.209 |
| train | 3444 | 4354 | 10427 | 2177 | 0.209 | 10621 | 2209 | 0.208 |
| dev | 691 | 840 | 2029 | 433 | 0.213 | 2054 | 437 | 0.213 |
| test | 726 | 914 | 2212 | 474 | 0.214 | 2223 | 474 | 0.213 |

Notes:
- Hidden arguments usually require tool results, database state, or workflow context.
- Direct prompt extraction is therefore only a partial argument-grounding baseline.
