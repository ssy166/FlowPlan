import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def metric(summary: dict[str, Any] | None, section: str, key: str) -> Any:
    if not summary:
        return None
    if section == "overall":
        return (summary.get("overall") or {}).get(key)
    return ((summary.get("by_domain") or {}).get(section) or {}).get(key)


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return str(value)


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def row_from_summary(root: Path, spec: dict[str, str]) -> dict[str, Any]:
    path = root / spec["summary"]
    summary = load_json(path)
    row: dict[str, Any] = {
        "name": spec["name"],
        "group": spec.get("group", ""),
        "split": spec.get("split", "test"),
        "summary": spec["summary"],
        "note": spec.get("note", ""),
        "exists": summary is not None,
    }
    for section in ["overall", "retail", "telecom"]:
        prefix = "" if section == "overall" else f"{section}_"
        row[prefix + "success"] = metric(summary, section, "next_action_success")
        row[prefix + "tool_em"] = metric(summary, section, "tool_exact_match")
        row[prefix + "action_success"] = metric(summary, section, "action_success")
        row[prefix + "stop_success"] = metric(summary, section, "stop_success")
        row[prefix + "pred_exec_ok"] = metric(summary, section, "predicted_action_execution_ok")
        row[prefix + "arg_key_recall"] = metric(summary, section, "argument_key_recall")
        row[prefix + "arg_value_em"] = metric(summary, section, "argument_value_exact_match")
        row[prefix + "count"] = metric(summary, section, "count")
    return row


def primary_wins(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    keys = [
        ("success", "overall success"),
        ("retail_success", "retail success"),
        ("tool_em", "overall tool EM"),
        ("retail_tool_em", "retail tool EM"),
        ("pred_exec_ok", "pred exec ok"),
        ("arg_value_em", "arg value EM"),
    ]
    items = []
    wins = 0
    ties = 0
    losses = 0
    for key, label in keys:
        c = candidate.get(key)
        b = baseline.get(key)
        if c is None or b is None:
            outcome = "missing"
        elif c > b:
            outcome = "win"
            wins += 1
        elif c == b:
            outcome = "tie"
            ties += 1
        else:
            outcome = "loss"
            losses += 1
        items.append({"metric": label, "candidate": c, "baseline": b, "outcome": outcome})
    safety = {
        "replay_execution_ok": candidate.get("replay_execution_ok"),
        "pred_exec_ok": candidate.get("pred_exec_ok"),
        "replay_ok_pass": (candidate.get("replay_execution_ok") or 0.0) >= 0.98,
        "pred_exec_pass": (candidate.get("pred_exec_ok") or 0.0) >= 0.90,
    }
    return {
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "passed": wins >= 4 and safety["replay_ok_pass"] and safety["pred_exec_pass"],
        "metrics": items,
        "safety": safety,
    }


def enrich_replay(summary_path: Path, row: dict[str, Any]) -> None:
    summary = load_json(summary_path)
    row["replay_execution_ok"] = metric(summary, "overall", "replay_execution_ok")


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> list[str]:
    lines = []
    lines.append("| " + " | ".join(title for title, _ in columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(key)) for _, key in columns) + " |")
    return lines


def decision_rows(root: Path) -> list[dict[str, Any]]:
    specs = [
        ("v8 state hint", "data/processed/closed_loop/decision_breakdown.v8_state_hint.test.json"),
        (
            "final strict v3",
            "data/processed/closed_loop/decision_breakdown.decision_selector_op_rule_override_strict_t09_v3.test.json",
        ),
        (
            "oracle operation tool",
            "data/processed/closed_loop/decision_breakdown.retail_operation_tool_oracle.test.json",
        ),
    ]
    rows = []
    for name, rel_path in specs:
        data = load_json(root / rel_path)
        if not data:
            continue
        for case, values in sorted((data.get("by_decision_case") or {}).items()):
            rows.append(
                {
                    "method": name,
                    "decision_case": case,
                    "count": values.get("count"),
                    "success": values.get("next_action_success"),
                    "tool_em": values.get("tool_exact_match"),
                    "arg_value_em": values.get("argument_value_exact_match"),
                    "pred_exec_ok": values.get("predicted_action_execution_ok"),
                }
            )
    return rows


def method_specs(split: str) -> list[dict[str, str]]:
    suffix = f"{split}.state_grounded_v4.summary.json"
    base = "data/processed/closed_loop"
    return [
        {
            "name": "SFT no-prior compact-v4",
            "group": "clean supervised baseline",
            "split": split,
            "summary": f"{base}/replan_exec_sft_mixed_retail_gold_v4_compact_noprior_3gpu.{suffix}",
        },
        {
            "name": "SFT + FM c_i state hint v8",
            "group": "current clean baseline",
            "split": split,
            "summary": f"{base}/replan_exec_sft_mixed_retail_gold_v4_compact_noprior_ci_state_hint_3gpu.{suffix}",
        },
        {
            "name": "FM-hint SFT + strict retail op selector v3",
            "group": "main method",
            "split": split,
            "summary": (
                f"{base}/replan_exec_sft_mixed_retail_gold_v4_compact_noprior_ci_state_hint_"
                f"decision_selector_op_rule_override_strict_t09_v3.{suffix}"
            ),
        },
        {
            "name": "Retail op-rule SFT v1",
            "group": "trained ablation",
            "split": split,
            "summary": f"{base}/replan_exec_sft_mixed_retail_gold_v4_compact_noprior_ci_state_hint_retail_op_rule_3gpu.{suffix}",
        },
        {
            "name": "Oracle retail decision/stage",
            "group": "oracle diagnostic",
            "split": split,
            "summary": f"{base}/replan_exec_sft_mixed_retail_gold_v4_compact_noprior_ci_state_hint_retail_decision_oracle_3gpu.{suffix}",
        },
        {
            "name": "Oracle retail operation tool",
            "group": "oracle diagnostic",
            "split": split,
            "summary": f"{base}/replan_exec_sft_mixed_retail_gold_v4_compact_noprior_ci_state_hint_retail_operation_tool_oracle_3gpu.{suffix}",
        },
    ]


def light_baseline_specs(split: str) -> list[dict[str, str]]:
    base = "data/processed/closed_loop/replan_light_baselines_compact_v4"
    labels = [
        ("baseline_always_stop", "Always stop"),
        ("baseline_global_majority", "Global majority"),
        ("baseline_domain_majority", "Domain majority"),
        ("baseline_state_key_majority", "State-key majority"),
        ("baseline_lexical_nearest", "Lexical nearest train row"),
    ]
    return [
        {
            "name": label,
            "group": "light baseline",
            "split": split,
            "summary": f"{base}/{slug}.{split}.state_grounded_v4.summary.json",
        }
        for slug, label in labels
    ]


def toolrl_specs(split: str) -> list[dict[str, str]]:
    return [
        {
            "name": "ToolRL GRPO step582",
            "group": "RL baseline",
            "split": split,
            "summary": f"data/processed/closed_loop/replan_exec_toolrl_lora_grpo_qwen7b_step582.{split}.summary.json",
            "note": "Evaluated on original replan rows; use as secondary same-data RL baseline, not compact-v4 acceptance baseline.",
        }
    ]


def llama_specs(split: str) -> list[dict[str, str]]:
    base = "data/processed/closed_loop"
    suffix = f"{split}.state_grounded_v4.summary.json"
    return [
        {
            "name": "Llama-3.2-3B no-prior compact-v4",
            "group": "second-base-model baseline",
            "split": split,
            "summary": f"{base}/replan_exec_llama32_3b_compact_v4_noprior.{suffix}",
            "note": "Same compact-v4 no-prior SFT protocol using Llama-3.2-3B-Instruct.",
        },
        {
            "name": "Llama-3.2-3B + FM c_i state hint",
            "group": "second-base-model FM hint",
            "split": split,
            "summary": f"{base}/replan_exec_llama32_3b_compact_v4_ci_state_hint.{suffix}",
            "note": "Same compact-v4 c_i + compact-state hint protocol using Llama-3.2-3B-Instruct.",
        },
        {
            "name": "Llama-3.2-3B FM hint + strict selector",
            "group": "second-base-model final",
            "split": split,
            "summary": f"{base}/replan_exec_llama32_3b_compact_v4_ci_state_hint_decision_selector_op_rule_override_strict_t09_v3.{suffix}",
            "note": "Llama state-hint predictions with the same strict retail operation selector.",
        },
    ]


def write_report(root: Path, out_md: Path, out_json: Path) -> None:
    rows = []
    for split in ["dev", "test"]:
        for spec in method_specs(split) + light_baseline_specs(split) + toolrl_specs(split) + llama_specs(split):
            row = row_from_summary(root, spec)
            enrich_replay(root / spec["summary"], row)
            rows.append(row)

    by_name_split = {(row["name"], row["split"]): row for row in rows}
    final = by_name_split.get(("FM-hint SFT + strict retail op selector v3", "test"), {})
    v8 = by_name_split.get(("SFT + FM c_i state hint v8", "test"), {})
    no_prior = by_name_split.get(("SFT no-prior compact-v4", "test"), {})
    acceptance = {
        "final_vs_v8": primary_wins(final, v8) if final and v8 else {},
        "final_vs_no_prior": primary_wins(final, no_prior) if final and no_prior else {},
    }
    decisions = decision_rows(root)

    payload = {
        "rows": rows,
        "acceptance": acceptance,
        "decision_rows": decisions,
        "artifacts": {
            "light_baselines": "data/processed/predictions/replan_light_baselines_compact_v4/",
            "closed_loop_light_baselines": "data/processed/closed_loop/replan_light_baselines_compact_v4/",
            "main_summary": (
                "data/processed/closed_loop/replan_exec_sft_mixed_retail_gold_v4_compact_noprior_ci_state_hint_"
                "decision_selector_op_rule_override_strict_t09_v3.test.state_grounded_v4.summary.json"
            ),
        },
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Main Experiment Results",
        "",
        "This report freezes the current clean compact-v4 closed-loop evaluation tables.",
        "Primary acceptance uses six metrics: overall success, retail success, overall tool EM, retail tool EM, predicted execution ok, and argument value EM. A result passes if at least four metrics strictly improve and safety thresholds hold.",
        "",
        "## Main Test Comparison",
    ]
    main_test = [
        row
        for row in rows
        if row["split"] == "test"
        and row["group"]
        in {
            "clean supervised baseline",
            "current clean baseline",
            "main method",
            "RL baseline",
            "second-base-model baseline",
            "second-base-model FM hint",
            "second-base-model final",
        }
        and row["exists"]
    ]
    lines.extend(
        markdown_table(
            main_test,
            [
                ("method", "name"),
                ("group", "group"),
                ("overall succ", "success"),
                ("retail succ", "retail_success"),
                ("telecom succ", "telecom_success"),
                ("tool EM", "tool_em"),
                ("retail tool EM", "retail_tool_em"),
                ("pred exec ok", "pred_exec_ok"),
                ("arg value EM", "arg_value_em"),
            ],
        )
    )
    lines.extend(["", "## Second Base Model: Llama-3.2-3B"])
    llama_rows = [row for row in rows if row["split"] in {"dev", "test"} and row["group"].startswith("second-base-model") and row["exists"]]
    lines.extend(
        markdown_table(
            llama_rows,
            [
                ("split", "split"),
                ("method", "name"),
                ("overall succ", "success"),
                ("retail succ", "retail_success"),
                ("tool EM", "tool_em"),
                ("retail tool EM", "retail_tool_em"),
                ("pred exec ok", "pred_exec_ok"),
                ("arg value EM", "arg_value_em"),
            ],
        )
    )
    if llama_rows:
        lines.extend(
            [
                "",
                "This table is a transfer check on the same compact-v4 protocol. It tests whether the FM/state-hint interface remains usable when the executor base model is changed from Qwen2.5-7B-Instruct to Llama-3.2-3B-Instruct.",
            ]
        )
    lines.extend(["", "## Light Baselines On Compact-v4"])
    light_test = [row for row in rows if row["split"] == "test" and row["group"] == "light baseline"]
    lines.extend(
        markdown_table(
            light_test,
            [
                ("baseline", "name"),
                ("overall succ", "success"),
                ("retail succ", "retail_success"),
                ("telecom succ", "telecom_success"),
                ("tool EM", "tool_em"),
                ("action succ", "action_success"),
                ("stop succ", "stop_success"),
                ("pred exec ok", "pred_exec_ok"),
                ("arg value EM", "arg_value_em"),
            ],
        )
    )
    lines.extend(["", "## Ablations And Upper Bounds"])
    ablations = [
        row
        for row in rows
        if row["split"] == "test" and row["group"] in {"trained ablation", "oracle diagnostic", "main method", "current clean baseline"}
    ]
    lines.extend(
        markdown_table(
            ablations,
            [
                ("method", "name"),
                ("group", "group"),
                ("overall succ", "success"),
                ("retail succ", "retail_success"),
                ("retail tool EM", "retail_tool_em"),
                ("pred exec ok", "pred_exec_ok"),
                ("arg value EM", "arg_value_em"),
            ],
        )
    )
    lines.extend(["", "## Retail Decision Cases"])
    lines.extend(
        markdown_table(
            decisions,
            [
                ("method", "method"),
                ("case", "decision_case"),
                ("n", "count"),
                ("success", "success"),
                ("tool EM", "tool_em"),
                ("pred exec ok", "pred_exec_ok"),
                ("arg value EM", "arg_value_em"),
            ],
        )
    )
    lines.extend(["", "## Acceptance"])
    for name, result in acceptance.items():
        if not result:
            continue
        lines.append(
            f"- `{name}`: passed={result['passed']}, wins={result['wins']}, ties={result['ties']}, losses={result['losses']}; "
            f"replay_ok={fmt(result['safety']['replay_execution_ok'])}, pred_exec_ok={fmt(result['safety']['pred_exec_ok'])}."
        )
    lines.extend(
        [
            "",
            "## Notes",
            "- ToolRL is retained as a secondary RL baseline. Its step582 closed-loop evaluation is on the original replan rows, so it should not replace compact-v4 acceptance comparisons.",
            "- The light baselines are intentionally cheap non-LLM references: always-stop, majority, state-key majority, and lexical nearest. Their role is to show that closed-loop success is not explained by stop bias or memorized shallow state keys.",
            "- Oracle decision/stage and oracle operation-tool rows are diagnostics. They are useful for locating the bottleneck but are not clean baselines.",
            "",
            "## Artifact Index",
            f"- JSON table: `{rel(out_json, root)}`",
            "- Light baseline predictions: `data/processed/predictions/replan_light_baselines_compact_v4/`",
            "- Light baseline closed-loop summaries: `data/processed/closed_loop/replan_light_baselines_compact_v4/`",
        ]
    )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write compact main experiment result tables.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--out-md", default="data/processed/MAIN_EXPERIMENT_RESULTS.md")
    parser.add_argument("--out-json", default="data/processed/MAIN_EXPERIMENT_RESULTS.json")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    write_report(root, root / args.out_md, root / args.out_json)
    print(json.dumps({"out_md": args.out_md, "out_json": args.out_json}, ensure_ascii=False))


if __name__ == "__main__":
    main()
