import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def load_gold(path):
    return {row["task_id"]: row for row in read_jsonl(path)}


def phase_for_tool(tool_id, tools):
    return tools.get(tool_id, {}).get("phase", "unknown")


def actions_from_plan(plan):
    actions = plan.get("actions")
    if actions:
        return actions
    rows = []
    for tool_id in plan.get("tool_ids") or []:
        rows.append({"tool_id": tool_id, "tool_name": tool_id.rsplit("::", 1)[-1], "arguments": {}})
    return rows


def normalize_value(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def argument_scores(pred_actions, gold_actions):
    gold_key_count = 0
    pred_key_count = 0
    key_hits = 0
    value_hits = 0
    comparable = 0
    for idx, gold_action in enumerate(gold_actions):
        pred_action = pred_actions[idx] if idx < len(pred_actions) else {}
        gold_args = gold_action.get("arguments") or {}
        pred_args = pred_action.get("arguments") or {}
        if not isinstance(gold_args, dict):
            gold_args = {}
        if not isinstance(pred_args, dict):
            pred_args = {}
        gold_keys = set(gold_args)
        pred_keys = set(pred_args)
        gold_key_count += len(gold_keys)
        pred_key_count += len(pred_keys)
        matched = gold_keys & pred_keys
        key_hits += len(matched)
        comparable += len(gold_keys)
        for key in matched:
            if normalize_value(pred_args.get(key)) == normalize_value(gold_args.get(key)):
                value_hits += 1
    return {
        "gold_argument_count": gold_key_count,
        "predicted_argument_count": pred_key_count,
        "argument_key_hits": key_hits,
        "argument_value_hits": value_hits,
        "argument_key_recall": key_hits / gold_key_count if gold_key_count else None,
        "argument_key_precision": key_hits / pred_key_count if pred_key_count else None,
        "argument_value_exact_match": value_hits / comparable if comparable else None,
    }


def evaluate(predictions, gold, tools, include_empty_gold=False):
    scores = []
    for pred in predictions:
        task_id = pred["task_id"]
        if task_id not in gold:
            continue
        gold_plan = gold[task_id]
        pred_tool_ids = pred.get("tool_ids") or []
        gold_tool_ids = gold_plan.get("tool_ids") or []
        if not include_empty_gold and not gold_tool_ids:
            continue
        pred_phases = pred.get("phase_names") or [phase_for_tool(t, tools) for t in pred_tool_ids]
        gold_phases = gold_plan.get("phase_names") or []
        tool_ed = edit_distance(pred_tool_ids, gold_tool_ids)
        phase_ed = edit_distance(pred_phases, gold_phases)
        exact_tool = pred_tool_ids == gold_tool_ids
        exact_phase = pred_phases == gold_phases
        redundant = max(0, len(pred_tool_ids) - len(gold_tool_ids))
        arg_scores = argument_scores(actions_from_plan(pred), actions_from_plan(gold_plan))
        scores.append(
            {
                "task_id": task_id,
                "source": gold_plan.get("source"),
                "domain": gold_plan.get("domain"),
                "has_unseen_tool": bool(gold_plan.get("has_unseen_tool")),
                "tool_edit_distance": tool_ed,
                "phase_edit_distance": phase_ed,
                "tool_exact_match": exact_tool,
                "phase_exact_match": exact_phase,
                "predicted_tool_count": len(pred_tool_ids),
                "gold_tool_count": len(gold_tool_ids),
                "redundant_tool_count": redundant,
                **arg_scores,
            }
        )
    return scores


def summarize(scores):
    if not scores:
        return {"count": 0}
    n = len(scores)
    arg_recall_rows = [s for s in scores if s.get("argument_key_recall") is not None]
    arg_precision_rows = [s for s in scores if s.get("argument_key_precision") is not None]
    arg_value_rows = [s for s in scores if s.get("argument_value_exact_match") is not None]
    return {
        "count": n,
        "tool_exact_match": sum(s["tool_exact_match"] for s in scores) / n,
        "phase_exact_match": sum(s["phase_exact_match"] for s in scores) / n,
        "avg_tool_edit_distance": sum(s["tool_edit_distance"] for s in scores) / n,
        "avg_phase_edit_distance": sum(s["phase_edit_distance"] for s in scores) / n,
        "avg_predicted_tool_count": sum(s["predicted_tool_count"] for s in scores) / n,
        "avg_redundant_tool_count": sum(s["redundant_tool_count"] for s in scores) / n,
        "argument_task_count": len(arg_recall_rows),
        "avg_argument_key_recall": sum(s["argument_key_recall"] for s in arg_recall_rows) / len(arg_recall_rows)
        if arg_recall_rows
        else None,
        "avg_argument_key_precision": sum(s["argument_key_precision"] for s in arg_precision_rows) / len(arg_precision_rows)
        if arg_precision_rows
        else None,
        "avg_argument_value_exact_match": sum(s["argument_value_exact_match"] for s in arg_value_rows) / len(arg_value_rows)
        if arg_value_rows
        else None,
    }


def summarize_groups(scores, key):
    groups = {}
    for score in scores:
        groups.setdefault(str(score.get(key)), []).append(score)
    return {group: summarize(rows) for group, rows in sorted(groups.items())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", default=str(ROOT / "data" / "processed" / "gold_plans.jsonl"))
    parser.add_argument("--gold", default=str(ROOT / "data" / "processed" / "gold_plans.jsonl"))
    parser.add_argument("--tasks", default=str(ROOT / "data" / "processed" / "tasks.jsonl"))
    parser.add_argument("--tools", default=str(ROOT / "data" / "processed" / "tools.jsonl"))
    parser.add_argument("--details", default=None)
    parser.add_argument("--include-empty-gold", action="store_true")
    args = parser.parse_args()

    tools = {row["tool_id"]: row for row in read_jsonl(args.tools)}
    # Build domain lookup from tasks (gold_plans don't carry domain)
    task_domain = {row["task_id"]: row.get("domain") for row in read_jsonl(args.tasks)}
    gold = load_gold(args.gold)
    # Backfill domain into gold plans from tasks
    for task_id, plan in gold.items():
        if "domain" not in plan and task_id in task_domain:
            plan["domain"] = task_domain[task_id]
    predictions = read_jsonl(args.pred)
    scores = evaluate(predictions, gold, tools, include_empty_gold=args.include_empty_gold)
    report = {
        "overall": summarize(scores),
        "by_source": summarize_groups(scores, "source"),
        "by_domain": summarize_groups(scores, "domain"),
        "by_has_unseen_tool": summarize_groups(scores, "has_unseen_tool"),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.details:
        with Path(args.details).open("w", encoding="utf-8") as f:
            for score in scores:
                f.write(json.dumps(score, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
