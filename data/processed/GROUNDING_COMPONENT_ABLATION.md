# Grounding Component Ablation

评测设置：compact-v4 Qwen FM state-hint SFT，test split，不使用 strict selector。该实验固定 planner/executor 的工具预测，只隔离 argument/state grounding 层。

| setting | enabled grounding components | overall succ | retail succ | tool EM | retail tool EM | pred exec ok | arg value EM |
|---|---|---:|---:|---:|---:|---:|---:|
| no state grounding | none | 0.7071 | 0.4675 | 0.7429 | 0.5325 | 0.8762 | 0.5455 |
| full grounding | schema normalization + entity binding + field mapping + DB/state validation | 0.7357 | 0.5195 | 0.7429 | 0.5325 | 0.9619 | 0.5613 |
| w/o schema normalization | entity binding + field mapping + DB/state validation | 0.7357 | 0.5195 | 0.7429 | 0.5325 | 0.9619 | 0.5613 |
| w/o entity binding | schema normalization + field mapping + DB/state validation | 0.7357 | 0.5195 | 0.7429 | 0.5325 | 0.9619 | 0.5613 |
| w/o field mapping | schema normalization + entity binding + DB/state validation | 0.7357 | 0.5195 | 0.7429 | 0.5325 | 0.9524 | 0.5613 |
| w/o DB/state validation | schema normalization + entity binding + field mapping | 0.7071 | 0.4675 | 0.7429 | 0.5325 | 0.8857 | 0.5455 |

解读：

- Tool EM 不变，因为该消融只改变参数 grounding，不改变工具选择。
- DB/state constraint validation 是当前 test slice 中最关键的组件：去掉后 predicted execution success 从 0.9619 降到 0.8857，retail success 从 0.5195 降到 0.4675。
- Schema normalization、entity binding、field mapping 仍保留为方法组件，因为它们定义 typed tool arguments 的通用接口；当前 split 中这些字段较稀疏，所以单独去掉时影响较小。

产物：

- summaries：`data/processed/closed_loop/grounding_components_compact_v4/*.summary.json`
- grounded predictions：`data/processed/predictions/grounding_components_compact_v4/*.jsonl`
- fill counts：`data/processed/predictions/grounding_components_compact_v4/*.fill_counts.json`
