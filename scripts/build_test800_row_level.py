import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STOP_TOOL = "<stop>"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def stable_int(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()
    return int(digest, 16)


def compact_action(action: dict[str, Any] | None) -> dict[str, Any] | None:
    if not action:
        return None
    return {
        "tool_id": action.get("tool_id"),
        "tool_name": action.get("tool_name"),
        "arguments": action.get("arguments") or {},
    }


def available_tool_ids(row: dict[str, Any]) -> list[str]:
    out = []
    for tool in row.get("available_tools") or []:
        if isinstance(tool, dict) and tool.get("tool_id"):
            out.append(tool["tool_id"])
        elif isinstance(tool, str):
            out.append(tool)
    return out


def gold_executed_prefix(prefix_actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prefix = []
    for idx, action in enumerate(prefix_actions):
        compact = compact_action(action)
        prefix.append(
            {
                "step_idx": idx,
                "predicted_action": compact,
                "execution": {
                    "ok": True,
                    "tool_name": compact["tool_name"] if compact else None,
                    "arguments": compact["arguments"] if compact else {},
                    "result": "gold_prefix_replay",
                },
                "tool_match": True,
                "db_hash_after": None,
            }
        )
    return prefix


def build_input_text(
    task_id: str,
    domain: str,
    prompt: str,
    executed_prefix: list[dict[str, Any]],
    source_kind: str,
    stop_target: bool,
) -> str:
    payload = {
        "task": {"task_id": task_id, "domain": domain, "prompt": prompt},
        "initialization_feedback": [],
        "executed_prefix": executed_prefix,
        "replan_reason": {
            "execution_ok": True,
            "error_type": None,
            "has_predicted_action_at_step": bool(executed_prefix),
            "tool_match": True,
            "source_kind": source_kind,
        },
        "instruction": "Predict the next tool call from this compact state. Return an empty action list if the workflow should stop.",
        "target_mode": "next_action_or_stop",
        "stop_target": stop_target,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def workflow_to_compact(row: dict[str, Any], source_kind: str) -> dict[str, Any]:
    target = row.get("target") or {}
    next_action = compact_action(target.get("next_action"))
    actions = [next_action] if next_action else []
    future_tools = [next_action["tool_id"]] if next_action else []
    next_tool = future_tools[0] if future_tools else STOP_TOOL
    executed_prefix = gold_executed_prefix(row.get("prefix_actions") or [])
    return {
        "record_id": f"{row.get('record_id')}::test800",
        "task_id": row.get("task_id"),
        "source": row.get("source"),
        "domain": row.get("domain"),
        "split": row.get("split"),
        "input_text": build_input_text(
            row.get("task_id"),
            row.get("domain"),
            row.get("prompt"),
            executed_prefix,
            source_kind,
            not bool(future_tools),
        ),
        "gold_tool_call_text": json.dumps({"actions": actions}, ensure_ascii=False, sort_keys=True, default=str),
        "future_tools": future_tools,
        "next_tool": next_tool,
        "available_tools": available_tool_ids(row),
        "stop_target": not bool(future_tools),
        "metadata": {
            "format": "test800",
            "source_kind": source_kind,
            "base_record_id": row.get("record_id"),
            "workflow_step_idx": row.get("step_idx"),
            "executed_prefix_len": len(executed_prefix),
            "target_remaining_len": len(future_tools),
            "legacy_140_member": False,
        },
    }


def workflow_terminal_to_compact(rows_for_task: list[dict[str, Any]], source_kind: str) -> dict[str, Any]:
    rows_for_task = sorted(rows_for_task, key=lambda row: int(row.get("step_idx") or 0))
    last = rows_for_task[-1]
    full_actions = (last.get("target") or {}).get("full_actions") or []
    executed_prefix = gold_executed_prefix(full_actions)
    return {
        "record_id": f"{last.get('task_id')}::terminal::test800",
        "task_id": last.get("task_id"),
        "source": last.get("source"),
        "domain": last.get("domain"),
        "split": last.get("split"),
        "input_text": build_input_text(
            last.get("task_id"),
            last.get("domain"),
            last.get("prompt"),
            executed_prefix,
            source_kind,
            True,
        ),
        "gold_tool_call_text": json.dumps({"actions": []}, ensure_ascii=False, sort_keys=True),
        "future_tools": [],
        "next_tool": STOP_TOOL,
        "available_tools": available_tool_ids(last),
        "stop_target": True,
        "metadata": {
            "format": "test800",
            "source_kind": source_kind,
            "base_record_id": last.get("record_id"),
            "workflow_step_idx": int(last.get("step_idx") or 0) + 1,
            "executed_prefix_len": len(executed_prefix),
            "target_remaining_len": 0,
            "legacy_140_member": False,
        },
    }


def normalize_legacy(row: dict[str, Any], source_kind: str) -> dict[str, Any]:
    out = dict(row)
    out["record_id"] = f"{row.get('record_id')}::test800"
    future_tools = list(out.get("future_tools") or [])
    if len(future_tools) > 1:
        future_tools = future_tools[:1]
        out["future_tools"] = future_tools
        out["next_tool"] = future_tools[0]
        payload = json.loads(out["gold_tool_call_text"])
        payload["actions"] = (payload.get("actions") or [])[:1]
        out["gold_tool_call_text"] = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    input_payload = json.loads(out["input_text"])
    input_payload["target_mode"] = "next_action_or_stop"
    input_payload["stop_target"] = bool(out.get("stop_target"))
    input_payload["instruction"] = (
        "Predict the next tool call from this compact state. "
        "Return an empty action list if the workflow should stop."
    )
    out["input_text"] = json.dumps(input_payload, ensure_ascii=False, sort_keys=True, default=str)
    out.setdefault("metadata", {})
    out["metadata"] = dict(out["metadata"])
    out["metadata"].update(
        {
            "format": "test800",
            "source_kind": source_kind,
            "base_record_id": row.get("record_id"),
            "legacy_140_member": source_kind == "legacy_telecom_replan",
        }
    )
    return out


def by_task(rows: list[dict[str, Any]], domain: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("domain") == domain:
            grouped.setdefault(row["task_id"], []).append(row)
    return grouped


def stable_sample(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: stable_int(row["record_id"]))[:n]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a row-level tau2 retail/telecom test800 replan pack.")
    parser.add_argument("--workflows", default=str(ROOT / "data" / "processed" / "workflows" / "test.jsonl"))
    parser.add_argument("--legacy-telecom", default=str(ROOT / "data" / "processed" / "replan_text" / "test.telecom.jsonl"))
    parser.add_argument("--out", default=str(ROOT / "outputs" / "test800" / "row_level_source.jsonl"))
    parser.add_argument("--target-size", type=int, default=800)
    parser.add_argument("--target-stop-rate", type=float, default=0.25)
    args = parser.parse_args()

    workflow_test = read_jsonl(args.workflows)
    legacy_telecom = [normalize_legacy(row, "legacy_telecom_replan") for row in read_jsonl(args.legacy_telecom)]
    retail_workflow = [
        workflow_to_compact(row, "retail_gold_prefix")
        for row in workflow_test
        if row.get("source") == "tau2" and row.get("domain") == "retail"
    ]
    retail_terminal = [
        workflow_terminal_to_compact(rows, "retail_gold_terminal")
        for _, rows in sorted(by_task(workflow_test, "retail").items())
    ]
    target_stop_total = round(args.target_size * args.target_stop_rate)
    existing_stop = sum(1 for row in legacy_telecom + retail_terminal if row.get("stop_target"))
    telecom_terminal_needed = max(0, target_stop_total - existing_stop)
    telecom_terminal_all = [
        workflow_terminal_to_compact(rows, "telecom_gold_terminal")
        for _, rows in sorted(by_task(workflow_test, "telecom").items())
    ]
    telecom_terminal = stable_sample(telecom_terminal_all, telecom_terminal_needed)

    base_rows = legacy_telecom + retail_workflow + retail_terminal + telecom_terminal
    needed = args.target_size - len(base_rows)
    if needed < 0:
        raise ValueError(f"base rows exceed target size: {len(base_rows)} > {args.target_size}")
    used_ids = {row["record_id"] for row in base_rows}
    telecom_workflow_candidates = [
        workflow_to_compact(row, "telecom_gold_prefix")
        for row in workflow_test
        if row.get("source") == "tau2" and row.get("domain") == "telecom"
    ]
    telecom_workflow_candidates = [row for row in telecom_workflow_candidates if row["record_id"] not in used_ids]
    if needed > len(telecom_workflow_candidates):
        raise ValueError(f"not enough telecom workflow candidates: need {needed}, have {len(telecom_workflow_candidates)}")
    telecom_workflow = stable_sample(telecom_workflow_candidates, needed)
    rows = base_rows + telecom_workflow
    rows = sorted(rows, key=lambda row: (row.get("domain") != "retail", stable_int(row["record_id"])))

    if len(rows) != args.target_size:
        raise ValueError(f"expected {args.target_size} rows, got {len(rows)}")
    if len({row["record_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate record_id in test800")

    write_jsonl(args.out, rows)
    source_counts = Counter(row["metadata"]["source_kind"] for row in rows)
    domain_counts = Counter(row["domain"] for row in rows)
    stop_counts = Counter(row["domain"] for row in rows if row.get("stop_target"))
    report = {
        "format": "test800_row_level_v1",
        "out": args.out,
        "rows": len(rows),
        "domains": dict(sorted(domain_counts.items())),
        "source_kinds": dict(sorted(source_counts.items())),
        "stop_by_domain": dict(sorted(stop_counts.items())),
        "stop_total": sum(stop_counts.values()),
        "unique_tasks": len({row["task_id"] for row in rows}),
        "notes": [
            "Same-domain tau2 retail+telecom test rows only.",
            "All tau2 retail test gold-prefix and terminal rows are included.",
            "Legacy telecom replan rows are preserved.",
            "Telecom terminal rows are sampled to keep the stop rate near the requested target.",
            "Remaining capacity is filled with sampled tau2 telecom test gold-prefix rows.",
        ],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
