import argparse
import json
from pathlib import Path


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def safe_mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def as_float_bool(value):
    if value is None:
        return None
    return 1.0 if bool(value) else 0.0


def summarize(rows):
    action_rows = [row for row in rows if not row.get("gold_stop")]
    return {
        "count": len(rows),
        "action_rows": len(action_rows),
        "next_action_success": safe_mean(as_float_bool(row.get("next_action_success")) for row in rows),
        "tool_exact_match": safe_mean(
            as_float_bool(row.get("pred_tool_id") == row.get("gold_tool_id")) for row in action_rows
        ),
        "predicted_action_execution_ok": safe_mean(
            as_float_bool(row.get("predicted_action_execution_ok")) for row in rows if row.get("predicted_action") is not None
        ),
        "argument_key_recall": safe_mean(
            (row.get("arg_key_hits") / row.get("gold_arg_count"))
            for row in action_rows
            if isinstance(row.get("gold_arg_count"), int) and row.get("gold_arg_count") > 0
        ),
        "argument_value_exact_match": safe_mean(
            (row.get("arg_value_hits") / row.get("gold_arg_count"))
            for row in action_rows
            if isinstance(row.get("gold_arg_count"), int) and row.get("gold_arg_count") > 0
        ),
    }


def build_report(details, decisions, top_failures):
    decision_by_id = {row["id"]: row for row in decisions}
    groups = {}
    unmatched = []
    for row in details:
        if row.get("domain") != "retail":
            continue
        decision = decision_by_id.get(row.get("id"))
        if not decision:
            unmatched.append(row.get("id"))
            case = "missing_decision_case"
        else:
            case = decision.get("decision_case") or "unknown"
        enriched = dict(row)
        if decision:
            enriched["decision_case"] = case
            enriched["target_stage"] = (decision.get("target") or {}).get("stage")
            enriched["target_tool_name"] = (decision.get("target") or {}).get("tool_name")
        groups.setdefault(case, []).append(enriched)

    failures = []
    for case, rows in groups.items():
        for row in rows:
            if row.get("next_action_success"):
                continue
            failures.append(
                {
                    "id": row.get("id"),
                    "decision_case": case,
                    "gold_tool_id": row.get("gold_tool_id"),
                    "pred_tool_id": row.get("pred_tool_id"),
                    "gold_stop": row.get("gold_stop"),
                    "pred_stop": row.get("pred_stop"),
                    "execution_ok": row.get("predicted_action_execution_ok"),
                }
            )

    failures.sort(key=lambda item: (item["decision_case"], item["id"] or ""))
    return {
        "overall_retail": summarize([row for rows in groups.values() for row in rows]),
        "by_decision_case": {case: summarize(rows) for case, rows in sorted(groups.items())},
        "unmatched_detail_rows": len(unmatched),
        "failure_examples": failures[:top_failures],
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize closed-loop replan details by retail decision case.")
    parser.add_argument("--details", required=True)
    parser.add_argument("--decision", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--top-failures", type=int, default=20)
    args = parser.parse_args()

    report = build_report(read_jsonl(args.details), read_jsonl(args.decision), args.top_failures)
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, default=str)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
