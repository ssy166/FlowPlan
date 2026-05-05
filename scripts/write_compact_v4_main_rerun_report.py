import argparse
import json
from pathlib import Path
from typing import Any


METRICS = [
    ("overall_success", "overall succ"),
    ("retail_success", "retail succ"),
    ("tool_em", "tool EM"),
    ("retail_tool_em", "retail tool EM"),
    ("pred_exec_ok", "pred exec ok"),
    ("arg_value_em", "arg value EM"),
]


METHODS = [
    {
        "executor": "Qwen2.5-7B",
        "method": "Raw LLM direct",
        "tag": "qwen25_7b_raw_llm_direct_compact_v4_rerun",
        "data": "compact-v4 no-prior test",
    },
    {
        "executor": "Qwen2.5-7B",
        "method": "Compact-state SFT w/o FM prior",
        "tag": "qwen25_7b_compact_v4_noprior_sft_rerun",
        "data": "compact-v4 no-prior test",
    },
    {
        "executor": "Qwen2.5-7B",
        "method": "ToolRL GRPO",
        "tag": "toolrl_grpo_qwen25_7b_step582_compact_v4_rerun",
        "data": "compact-v4 state-hint test format",
    },
    {
        "executor": "Qwen2.5-7B",
        "method": "Retrieval-augmented LLM executor",
        "tag": "qwen25_7b_retrieval_aug_executor_compact_v4_rerun",
        "data": "compact-v4 no-prior test + lexical retrieval hint",
    },
    {
        "executor": "Qwen2.5-7B",
        "method": "FlowPlanner",
        "tag": "qwen25_7b_flowplanner_final_strict_selector_compact_v4_rerun",
        "data": "compact-v4 state-conditioned FM hint test",
    },
    {
        "executor": "Llama-3.2-3B",
        "method": "Raw LLM direct",
        "tag": "llama32_3b_raw_llm_direct_compact_v4_rerun",
        "data": "compact-v4 no-prior test",
    },
    {
        "executor": "Llama-3.2-3B",
        "method": "Compact-state SFT w/o FM prior",
        "tag": "llama32_3b_compact_v4_noprior_sft_rerun",
        "data": "compact-v4 no-prior test",
    },
    {
        "executor": "Llama-3.2-3B",
        "method": "ToolRL GRPO",
        "tag": "toolrl_grpo_llama32_3b_step100_compact_v4_rerun",
        "data": "compact-v4 state-hint test format",
    },
    {
        "executor": "Llama-3.2-3B",
        "method": "Retrieval-augmented LLM executor",
        "tag": "llama32_3b_retrieval_aug_executor_compact_v4_rerun",
        "data": "compact-v4 no-prior test + lexical retrieval hint",
    },
    {
        "executor": "Llama-3.2-3B",
        "method": "FlowPlanner",
        "tag": "llama32_3b_flowplanner_final_strict_selector_compact_v4_rerun",
        "data": "compact-v4 state-conditioned FM hint test",
    },
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def section(summary: dict[str, Any], name: str) -> dict[str, Any]:
    if name == "overall":
        return summary.get("overall") or {}
    return (summary.get("by_domain") or {}).get(name) or {}


def fmt(value: Any, key: str | None = None) -> str:
    if value is None:
        return "-"
    if key == "#":
        return str(value)
    if isinstance(value, (float, int)):
        return f"{value:.4f}"
    return str(value)


def summarize(path: Path) -> dict[str, Any]:
    summary = load_json(path)
    overall = section(summary, "overall")
    retail = section(summary, "retail")
    telecom = section(summary, "telecom")
    return {
        "summary_path": str(path),
        "row_count": overall.get("count"),
        "retail_count": retail.get("count"),
        "telecom_count": telecom.get("count"),
        "overall_success": overall.get("next_action_success"),
        "retail_success": retail.get("next_action_success"),
        "telecom_success": telecom.get("next_action_success"),
        "tool_em": overall.get("tool_exact_match"),
        "retail_tool_em": retail.get("tool_exact_match"),
        "pred_exec_ok": overall.get("predicted_action_execution_ok"),
        "arg_value_em": overall.get("argument_value_exact_match"),
        "replay_exec_ok": overall.get("replay_execution_ok"),
    }


def build_rows(summary_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for idx, spec in enumerate(METHODS, 1):
        path = summary_dir / f"{spec['tag']}.test.state_grounded_v4.summary.json"
        values = summarize(path)
        rows.append({"#": idx, **spec, **values})
    return rows


def markdown(rows: list[dict[str, Any]], tag: str) -> str:
    lines = [
        "# Compact-v4 Main Test Rerun",
        "",
        f"Rerun tag: `{tag}`",
        "",
        "All rows below were regenerated on the updated compact-v4 test set only. The evaluation uses the same structured grounding and execution evaluator for every method.",
        "",
    ]
    counts = {(row.get("row_count"), row.get("retail_count"), row.get("telecom_count")) for row in rows}
    if len(counts) == 1:
        row_count, retail_count, telecom_count = next(iter(counts))
        lines.append(f"Test rows: {row_count}; retail rows: {retail_count}; telecom rows: {telecom_count}.")
        lines.append("")
    columns = [
        ("#", "#"),
        ("executor", "executor"),
        ("method", "method"),
        ("overall succ", "overall_success"),
        ("retail succ", "retail_success"),
        ("tool EM", "tool_em"),
        ("retail tool EM", "retail_tool_em"),
        ("pred exec ok", "pred_exec_ok"),
        ("arg value EM", "arg_value_em"),
    ]
    lines.append("| " + " | ".join(title for title, _ in columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(key), key) for _, key in columns) + " |")
    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- `Raw LLM direct` uses the base instruct model without LoRA.",
            "- `ToolRL GRPO` uses the available ToolRL checkpoint for each executor and the same updated compact-v4 test rows.",
            "- `Retrieval-augmented LLM executor` injects a lexical-nearest training-row prior into the same no-prior SFT executor.",
            "- `FlowPlanner` uses the state-conditioned FM text hint plus the conservative retail operation selector.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", required=True)
    parser.add_argument("--tag", default="main_table_rerun_20260505")
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    rows = build_rows(Path(args.summary_dir))
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"tag": args.tag, "rows": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(markdown(rows, args.tag), encoding="utf-8")


if __name__ == "__main__":
    main()
