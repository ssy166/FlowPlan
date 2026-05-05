import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

try:
    from make_baseline_predictions import raw_tau2_item
except Exception:  # pragma: no cover
    raw_tau2_item = None


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def action_key(action):
    return action.get("tool_id")


def normalize_args(action):
    args = action.get("arguments") or {}
    return args if isinstance(args, dict) else {}


def arg_overlap(pred_action, gold_action):
    pred_args = normalize_args(pred_action)
    gold_args = normalize_args(gold_action)
    gold_keys = set(gold_args)
    pred_keys = set(pred_args)
    matched = gold_keys & pred_keys
    value_hits = 0
    for key in matched:
        if json.dumps(pred_args.get(key), sort_keys=True, ensure_ascii=False) == json.dumps(
            gold_args.get(key), sort_keys=True, ensure_ascii=False
        ):
            value_hits += 1
    return {
        "gold_arg_count": len(gold_keys),
        "pred_arg_count": len(pred_keys),
        "arg_key_hits": len(matched),
        "arg_value_hits": value_hits,
    }


def initial_state_summary(task):
    if not raw_tau2_item or task.get("source") != "tau2":
        return {}
    item = raw_tau2_item(task)
    initial_state = item.get("initial_state") or {}
    return {
        "has_initial_state": bool(initial_state),
        "initialization_action_count": len(initial_state.get("initialization_actions") or []),
        "has_initialization_data": bool(initial_state.get("initialization_data")),
    }


def build_trace(task, gold, pred, max_steps):
    gold_actions = gold.get("actions") or []
    pred_actions = pred.get("actions") or []
    total = max(len(gold_actions), len(pred_actions))
    if max_steps:
        total = min(total, max_steps)
    steps = []
    prefix_tool_matches = 0
    needs_replan_at = None
    for idx in range(total):
        gold_action = gold_actions[idx] if idx < len(gold_actions) else None
        pred_action = pred_actions[idx] if idx < len(pred_actions) else None
        tool_match = bool(gold_action and pred_action and action_key(gold_action) == action_key(pred_action))
        if tool_match and needs_replan_at is None and idx == prefix_tool_matches:
            prefix_tool_matches += 1
        elif needs_replan_at is None:
            needs_replan_at = idx
        overlap = arg_overlap(pred_action or {}, gold_action or {}) if gold_action and pred_action else {
            "gold_arg_count": len(normalize_args(gold_action or {})),
            "pred_arg_count": len(normalize_args(pred_action or {})),
            "arg_key_hits": 0,
            "arg_value_hits": 0,
        }
        steps.append(
            {
                "step_idx": idx,
                "state_context": {
                    "prefix_predicted_tools": [a.get("tool_id") for a in pred_actions[:idx]],
                    "prefix_gold_tools": [a.get("tool_id") for a in gold_actions[:idx]],
                },
                "predicted_action": pred_action,
                "gold_action": gold_action,
                "simulated_feedback": {
                    "tool_match": tool_match,
                    "argument_key_hits": overlap["arg_key_hits"],
                    "argument_value_hits": overlap["arg_value_hits"],
                    "gold_arg_count": overlap["gold_arg_count"],
                    "pred_arg_count": overlap["pred_arg_count"],
                    "needs_replan": not tool_match,
                },
            }
        )
    return {
        "task_id": task["task_id"],
        "source": task.get("source"),
        "domain": task.get("domain"),
        "prompt": task.get("prompt"),
        "prediction_type": pred.get("plan_type"),
        "initial_state_summary": initial_state_summary(task),
        "gold_tool_ids": gold.get("tool_ids") or [],
        "pred_tool_ids": pred.get("tool_ids") or [],
        "full_tool_match": (pred.get("tool_ids") or []) == (gold.get("tool_ids") or []),
        "first_tool_match": bool((pred.get("tool_ids") or [None])[0] == (gold.get("tool_ids") or [None])[0]),
        "prefix_tool_match_length": prefix_tool_matches,
        "needs_replan_at": needs_replan_at,
        "steps": steps,
    }


def summarize(traces):
    if not traces:
        return {"count": 0}
    return {
        "count": len(traces),
        "full_tool_match": sum(t["full_tool_match"] for t in traces) / len(traces),
        "first_tool_match": sum(t["first_tool_match"] for t in traces) / len(traces),
        "avg_prefix_tool_match_length": sum(t["prefix_tool_match_length"] for t in traces) / len(traces),
        "needs_replan_rate": sum(t["needs_replan_at"] is not None for t in traces) / len(traces),
    }


def main():
    parser = argparse.ArgumentParser(description="Build offline closed-loop replanning traces from decoded predictions.")
    parser.add_argument("--pred", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", default=None)
    parser.add_argument("--tasks", default=str(ROOT / "data" / "processed" / "tasks.jsonl"))
    parser.add_argument("--gold", default=str(ROOT / "data" / "processed" / "gold_plans.jsonl"))
    parser.add_argument("--source", default="tau2")
    parser.add_argument("--domain", default="telecom")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=8)
    args = parser.parse_args()

    tasks = {row["task_id"]: row for row in read_jsonl(args.tasks)}
    gold = {row["task_id"]: row for row in read_jsonl(args.gold)}
    preds = read_jsonl(args.pred)
    traces = []
    for pred in preds:
        task = tasks.get(pred.get("task_id"))
        gold_plan = gold.get(pred.get("task_id"))
        if not task or not gold_plan:
            continue
        if args.source and task.get("source") != args.source:
            continue
        if args.domain and task.get("domain") != args.domain:
            continue
        if not gold_plan.get("tool_ids"):
            continue
        traces.append(build_trace(task, gold_plan, pred, args.max_steps))
        if args.limit and len(traces) >= args.limit:
            break
    write_jsonl(Path(args.out), traces)
    summary = summarize(traces)
    summary["pred"] = args.pred
    summary["source"] = args.source
    summary["domain"] = args.domain
    summary["limit"] = args.limit
    if args.summary_out:
        Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_out).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
