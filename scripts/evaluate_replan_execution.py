import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_telecom_tool_traces import (  # noqa: E402
    action_tool_id,
    action_with_result_grounding,
    arg_overlap,
    assistant_initialization_actions,
    callable_name,
    execute_action,
    load_domain_tools,
    update_grounding_context,
)


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


def find_balanced_json(text: str, start: int = 0) -> str | None:
    first = text.find("{", start)
    if first < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for idx in range(first, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[first : idx + 1]
    return None


def parse_state(row: dict[str, Any]) -> dict[str, Any]:
    content = ""
    messages = row.get("messages") or row.get("prompt") or []
    for message in messages:
        if message.get("role") == "user":
            content = message.get("content") or ""
            break
    markers = ["Feedback-conditioned state:", "Compact state:"]
    start = -1
    for marker in markers:
        marker_start = content.find(marker)
        if marker_start >= 0:
            start = content.find("{", marker_start)
            break
    if start < 0:
        start = content.find("{")
    candidate = find_balanced_json(content, start)
    if not candidate:
        return {}
    try:
        state = json.loads(candidate)
    except Exception:
        return {}
    metadata = row_metadata(row)
    for key in ["source", "domain", "task_id"]:
        if metadata.get(key) and not state.get(key):
            state[key] = metadata[key]
    return state


def reward_ground_truth(row: dict[str, Any]) -> dict[str, Any]:
    reward_model = row.get("reward_model") or {}
    ground_truth = reward_model.get("ground_truth")
    if not ground_truth:
        return {}
    if isinstance(ground_truth, dict):
        return ground_truth
    try:
        return json.loads(ground_truth)
    except Exception:
        return {}


def row_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") or row.get("extra_info") or {}
    gt = reward_ground_truth(row)
    if gt:
        merged = dict(gt)
        merged.update(metadata)
        return merged
    return metadata


def row_id(row: dict[str, Any]) -> str | None:
    metadata = row_metadata(row)
    return row.get("id") or metadata.get("id") or metadata.get("record_id") or metadata.get("task_id")


def parse_target(row: dict[str, Any]) -> tuple[bool, list[dict[str, Any]]]:
    gt = reward_ground_truth(row)
    if gt:
        if "gold_stop" in gt or "gold_actions" in gt:
            return bool(gt.get("gold_stop", False)), gt.get("gold_actions") or []
        target = gt.get("target_json")
        if target:
            try:
                parsed = json.loads(target)
                return bool(parsed.get("stop", False)), parsed.get("actions") or []
            except Exception:
                pass
    for message in reversed(row.get("messages") or []):
        if message.get("role") == "assistant":
            target = json.loads(message.get("content") or "{}")
            return bool(target.get("stop", False)), target.get("actions") or []
    return False, []


def pred_key(row: dict[str, Any]) -> str | None:
    return row.get("record_id") or (row.get("metadata") or {}).get("record_id")


def replay_state(task: dict[str, Any], state: dict[str, Any], result_grounding: bool) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    db, tools = load_domain_tools(task.get("domain") or "telecom")
    grounding_context = {"user_ids": [], "order_ids": [], "order_cursor": 0}
    replay = []
    for init_action in assistant_initialization_actions(task):
        execution = execute_action(tools, init_action)
        execution["func_name"] = callable_name(init_action)
        update_grounding_context(grounding_context, init_action, execution)
        replay.append({"kind": "assistant_initialization", "action": init_action, "execution": execution})
    for step in state.get("executed_prefix") or []:
        action = step.get("predicted_action") or step.get("action")
        grounded_action = action_with_result_grounding(action, grounding_context) if result_grounding else action
        if grounded_action:
            execution = execute_action(tools, grounded_action)
            update_grounding_context(grounding_context, grounded_action, execution)
        else:
            execution = {"ok": False, "error_type": "MissingReplayAction"}
        replay.append(
            {
                "kind": "executed_prefix",
                "step_idx": step.get("step_idx"),
                "action": grounded_action,
                "execution": execution,
            }
        )
    return tools, grounding_context, replay


def evaluate_row(
    row: dict[str, Any],
    pred: dict[str, Any] | None,
    task: dict[str, Any],
    result_grounding: bool,
) -> dict[str, Any]:
    state = parse_state(row)
    gold_stop, gold_actions = parse_target(row)
    gold_action = gold_actions[0] if gold_actions else None
    pred_actions = (pred or {}).get("actions") or []
    pred_action = pred_actions[0] if pred_actions else None
    pred_stop = not pred_actions
    tools, grounding_context, replay = replay_state(task, state, result_grounding)
    grounded_pred_action = (
        action_with_result_grounding(pred_action, grounding_context) if result_grounding else pred_action
    )
    if grounded_pred_action:
        execution = execute_action(tools, grounded_pred_action)
        update_grounding_context(grounding_context, grounded_pred_action, execution)
    else:
        execution = {"ok": bool(gold_stop), "error_type": None if gold_stop else "MissingPrediction"}
    overlap = arg_overlap(grounded_pred_action, gold_action)
    tool_match = bool(action_tool_id(grounded_pred_action) == action_tool_id(gold_action) and grounded_pred_action and gold_action)
    stop_match = pred_stop == gold_stop
    next_action_success = bool((gold_stop and pred_stop) or (not gold_stop and tool_match and execution.get("ok", False)))
    metadata = row.get("metadata") or {}
    metadata = row_metadata(row)
    rid = row_id(row)
    return {
        "id": rid,
        "task_id": metadata.get("task_id") or rid,
        "source": metadata.get("source"),
        "domain": metadata.get("domain"),
        "prediction_type": (pred or {}).get("plan_type"),
        "gold_stop": gold_stop,
        "pred_stop": pred_stop,
        "stop_exact_match": stop_match,
        "gold_tool_id": action_tool_id(gold_action),
        "pred_tool_id": action_tool_id(grounded_pred_action),
        "tool_exact_match": tool_match if not gold_stop else pred_stop,
        "next_action_success": next_action_success,
        "predicted_action_execution_ok": bool(execution.get("ok", False)) if grounded_pred_action else None,
        "execution": execution,
        "gold_action": gold_action,
        "predicted_action": grounded_pred_action,
        "replay_execution_ok": all(item.get("execution", {}).get("ok", False) for item in replay),
        "replay": replay,
        **overlap,
    }


def div(num: float, den: float) -> float | None:
    return num / den if den else None


def summarize(details: list[dict[str, Any]]) -> dict[str, Any]:
    if not details:
        return {"count": 0}
    action_rows = [row for row in details if not row["gold_stop"]]
    stop_rows = [row for row in details if row["gold_stop"]]
    predicted_action_rows = [row for row in details if row["pred_tool_id"]]
    gold_arg_count = sum(row["gold_arg_count"] for row in details)
    pred_arg_count = sum(row["pred_arg_count"] for row in details)
    arg_key_hits = sum(row["arg_key_hits"] for row in details)
    arg_value_hits = sum(row["arg_value_hits"] for row in details)
    return {
        "count": len(details),
        "action_rows": len(action_rows),
        "stop_rows": len(stop_rows),
        "stop_exact_match": sum(row["stop_exact_match"] for row in details) / len(details),
        "tool_exact_match": sum(row["tool_exact_match"] for row in details) / len(details),
        "action_tool_exact_match": div(sum(row["tool_exact_match"] for row in action_rows), len(action_rows)),
        "next_action_success": sum(row["next_action_success"] for row in details) / len(details),
        "action_success": div(sum(row["next_action_success"] for row in action_rows), len(action_rows)),
        "stop_success": div(sum(row["next_action_success"] for row in stop_rows), len(stop_rows)),
        "predicted_action_count": len(predicted_action_rows),
        "predicted_action_execution_ok": div(
            sum(row["predicted_action_execution_ok"] for row in predicted_action_rows),
            len(predicted_action_rows),
        ),
        "replay_execution_ok": sum(row["replay_execution_ok"] for row in details) / len(details),
        "argument_key_recall": div(arg_key_hits, gold_arg_count),
        "argument_key_precision": div(arg_key_hits, pred_arg_count),
        "argument_value_exact_match": div(arg_value_hits, gold_arg_count),
    }


def summarize_groups(details: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in details:
        groups.setdefault(str(row.get(key)), []).append(row)
    return {name: summarize(rows) for name, rows in sorted(groups.items())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute/evaluate generated replan next-actions against tau2 tools.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--tasks", default=str(ROOT / "data" / "processed" / "tasks.jsonl"))
    parser.add_argument("--out-details", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--domain", choices=["telecom", "retail", "all"], default="all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--result-grounding", action="store_true")
    args = parser.parse_args()

    tasks = {row["task_id"]: row for row in read_jsonl(args.tasks)}
    preds = {pred_key(row): row for row in read_jsonl(args.pred) if pred_key(row)}
    details = []
    for row in read_jsonl(args.data):
        metadata = row_metadata(row)
        if args.domain != "all" and metadata.get("domain") != args.domain:
            continue
        task = tasks.get(metadata.get("task_id"))
        if not task or task.get("source") != "tau2":
            continue
        pred = preds.get(row_id(row))
        details.append(evaluate_row(row, pred, task, result_grounding=args.result_grounding))
        if args.limit and len(details) >= args.limit:
            break

    summary = {
        "data": args.data,
        "pred": args.pred,
        "domain": args.domain,
        "limit": args.limit,
        "result_grounding": args.result_grounding,
        "overall": summarize(details),
        "by_domain": summarize_groups(details, "domain"),
        "by_prediction_type": summarize_groups(details, "prediction_type"),
    }
    write_jsonl(args.out_details, details)
    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_out).write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
