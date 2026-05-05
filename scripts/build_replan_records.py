import argparse
import json
from pathlib import Path
from typing import Any


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


def first_replan_step(trace: dict[str, Any]) -> dict[str, Any] | None:
    for step in trace.get("steps") or []:
        if step.get("needs_replan"):
            return step
    return None


def build_record(trace: dict[str, Any], split: str) -> dict[str, Any] | None:
    step = first_replan_step(trace)
    if step is None:
        return None
    step_idx = int(step.get("step_idx", 0))
    gold_actions = []
    for item in trace.get("steps") or []:
        gold_action = item.get("gold_action")
        if gold_action is not None:
            gold_actions.append(gold_action)

    if step.get("tool_match") and not step.get("execution", {}).get("ok", False):
        target_start = step_idx
    else:
        target_start = step_idx
    target_actions = gold_actions[target_start:]

    prefix_steps = (trace.get("steps") or [])[: step_idx + 1]
    return {
        "task_id": trace.get("task_id"),
        "split": split,
        "source": trace.get("source"),
        "domain": trace.get("domain"),
        "prompt": trace.get("prompt"),
        "prediction_type": trace.get("prediction_type"),
        "replan_step_idx": step_idx,
        "replan_reason": {
            "tool_match": step.get("tool_match"),
            "execution_ok": step.get("execution", {}).get("ok", False),
            "error_type": step.get("execution", {}).get("error_type"),
            "has_gold_action_at_step": step.get("gold_action") is not None,
            "has_predicted_action_at_step": step.get("predicted_action") is not None,
        },
        "initialization_feedback": trace.get("assistant_initialization_actions") or [],
        "initial_db_hash_after_init": trace.get("initial_db_hash_after_init"),
        "executed_prefix": [
            {
                "step_idx": item.get("step_idx"),
                "predicted_action": item.get("predicted_action"),
                "execution": item.get("execution"),
                "db_hash_after": item.get("db_hash_after"),
                "tool_match": item.get("tool_match"),
            }
            for item in prefix_steps
        ],
        "gold_prefix_actions": [
            item.get("gold_action") for item in prefix_steps if item.get("gold_action") is not None
        ],
        "target_remaining_actions": target_actions,
        "target_remaining_tool_ids": [action.get("tool_id") for action in target_actions],
        "original_pred_remaining_tool_ids": trace.get("pred_tool_ids", [])[step_idx + 1 :],
        "metadata": {
            "format": "closed_loop_replan_record_v1",
            "target_semantics": "gold_remaining_from_first_replan_step",
        },
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"count": 0}
    empty_targets = sum(not row.get("target_remaining_actions") for row in records)
    exec_failures = sum(not row.get("replan_reason", {}).get("execution_ok", False) for row in records)
    tool_mismatches = sum(not row.get("replan_reason", {}).get("tool_match", False) for row in records)
    return {
        "count": len(records),
        "empty_target_rate": empty_targets / len(records),
        "execution_failure_rate": exec_failures / len(records),
        "tool_mismatch_rate": tool_mismatches / len(records),
        "avg_target_remaining_len": sum(len(row.get("target_remaining_actions") or []) for row in records) / len(records),
        "avg_executed_prefix_len": sum(len(row.get("executed_prefix") or []) for row in records) / len(records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build feedback-conditioned replanning records from executed traces.")
    parser.add_argument("--traces", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", default=None)
    parser.add_argument("--split", required=True)
    args = parser.parse_args()

    records = []
    for trace in read_jsonl(args.traces):
        record = build_record(trace, args.split)
        if record:
            records.append(record)

    write_jsonl(args.out, records)
    summary = summarize(records)
    summary.update({"traces": args.traces, "out": args.out, "split": args.split})
    if args.summary_out:
        Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_out).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
