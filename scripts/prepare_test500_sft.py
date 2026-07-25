import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STOP = "<stop>"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def shorten(text: str, max_chars: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def load_tools(path: str | Path) -> dict[str, dict[str, Any]]:
    return {row["tool_id"]: row for row in read_jsonl(path)}


def compact_tool(tool: dict[str, Any], max_desc_chars: int) -> dict[str, Any]:
    schema = tool.get("schema") or {}
    return {
        "tool_id": tool["tool_id"],
        "name": tool.get("name") or tool["tool_id"].rsplit("::", 1)[-1],
        "description": shorten(tool.get("description") or "", max_desc_chars),
        "parameters": schema.get("parameters")
        or schema.get("required_parameters")
        or schema.get("input_parameters")
        or schema.get("inputs")
        or [],
        "phase": tool.get("phase", "unknown"),
    }


def normalize_action(action: dict[str, Any] | None, tools: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    if not action:
        return None
    tool_id = action.get("tool_id")
    if not tool_id:
        return None
    tool = tools.get(tool_id) or {}
    return {
        "tool_id": tool_id,
        "tool_name": action.get("tool_name") or tool.get("name") or tool_id.rsplit("::", 1)[-1],
        "arguments": action.get("arguments") or {},
    }


def build_state(row: dict[str, Any], tools: dict[str, dict[str, Any]], max_desc_chars: int) -> dict[str, Any]:
    payload = json.loads(row["input_text"])
    task = payload.get("task") or {}
    available_ids = row.get("available_tools") or []
    state = {
        "available_tools": [
            compact_tool(tools[tool_id], max_desc_chars)
            for tool_id in available_ids
            if tool_id in tools
        ],
        "domain": row.get("domain"),
        "executed_prefix": payload.get("executed_prefix") or [],
        "init_feedback": payload.get("init_feedback") or payload.get("initialization_feedback") or [],
        "record_id": row.get("record_id"),
        "replan_reason": payload.get("replan_reason") or {},
        "source": row.get("source"),
        "step": (row.get("metadata") or {}).get("workflow_step_idx") or row.get("replan_step_idx"),
        "task": task.get("prompt") or payload.get("task") or "",
        "task_id": row.get("task_id") or task.get("task_id"),
        "target_mode": "next_action_or_stop",
        "stop_target": bool(row.get("stop_target")),
    }
    return state


def user_content(state: dict[str, Any]) -> str:
    return "\n".join(
        [
            "You are a feedback-conditioned tool execution model.",
            "Given the compact state, output the next tool call as strict JSON.",
            'Use schema {"stop": false, "actions": [{"tool_name": "...", "arguments": {}}]} or {"stop": true, "actions": []}.',
            "Choose only from available_tools. Use executed_prefix and init_feedback to ground ids/status/items/payments.",
            "",
            "Compact state:",
            json.dumps(state, ensure_ascii=False, sort_keys=True, default=str),
        ]
    )


def convert_row(row: dict[str, Any], tools: dict[str, dict[str, Any]], max_desc_chars: int, format_name: str) -> dict[str, Any]:
    gold = json.loads(row["gold_tool_call_text"])
    action = normalize_action((gold.get("actions") or [None])[0], tools)
    actions = [] if row.get("stop_target") else ([action] if action else [])
    assistant = {"actions": actions, "stop": bool(row.get("stop_target"))}
    metadata = dict(row.get("metadata") or {})
    metadata.update(
        {
            "format": f"{format_name}_chat_sft_v1",
            "source": row.get("source"),
            "domain": row.get("domain"),
            "split": row.get("split"),
            "task_id": row.get("task_id"),
            "record_id": row.get("record_id"),
            "next_tool": row.get("next_tool"),
            "stop": bool(row.get("stop_target")),
            "available_tool_ids": row.get("available_tools") or [],
        }
    )
    return {
        "id": row["record_id"],
        "messages": [
            {"role": "user", "content": user_content(build_state(row, tools, max_desc_chars))},
            {"role": "assistant", "content": json.dumps(assistant, ensure_ascii=False, sort_keys=True, default=str)},
        ],
        "metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a row-level replan pack into FlowPlan chat SFT format.")
    parser.add_argument("--source-row-level", required=True)
    parser.add_argument("--tools", default=str(ROOT / "data" / "benchmark" / "tools.jsonl"))
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "replan_sft" / "test500"))
    parser.add_argument("--format-name", default="test500")
    parser.add_argument("--max-desc-chars", type=int, default=320)
    args = parser.parse_args()

    tools = load_tools(args.tools)
    rows = [convert_row(row, tools, args.max_desc_chars, args.format_name) for row in read_jsonl(args.source_row_level)]
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    write_jsonl(out_dir / "test.jsonl", rows)
    domain_counts = Counter((row.get("metadata") or {}).get("domain") for row in rows)
    source_counts = Counter((row.get("metadata") or {}).get("source_kind") for row in rows)
    stop_counts = Counter((row.get("metadata") or {}).get("domain") for row in rows if (row.get("metadata") or {}).get("stop"))
    manifest = {
        "format": f"{args.format_name}_chat_sft_v1",
        "rows": len(rows),
        "path": str((out_dir / "test.jsonl").relative_to(ROOT)),
        "source_row_level": Path(args.source_row_level).name,
        "counts": {
            "domains": dict(sorted(domain_counts.items())),
            "source_kinds": dict(sorted(source_counts.items())),
            "stop_by_domain": dict(sorted(stop_counts.items())),
            "stop_total": sum(stop_counts.values()),
            "unique_tasks": len({(row.get("metadata") or {}).get("task_id") for row in rows}),
        },
        "construction_notes": [
            "Same-domain tau2 retail+telecom test rows only.",
            "Preserves the legacy telecom replan rows available locally.",
            "Retail rows are rebuilt from tau2 test workflow gold prefixes and terminal stop states because the exact legacy 77-row retail artifact was not present locally.",
            "Targets are next-action/stop only; each assistant target has zero or one action.",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
