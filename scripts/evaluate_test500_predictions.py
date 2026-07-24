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


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def norm_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def gold_action(row: dict[str, Any]) -> dict[str, Any] | None:
    for message in reversed(row.get("messages") or []):
        if message.get("role") != "assistant":
            continue
        payload = json.loads(message.get("content") or "{}")
        actions = payload.get("actions") or []
        return actions[0] if actions else None
    return None


def pred_action(row: dict[str, Any] | None) -> dict[str, Any] | None:
    actions = (row or {}).get("actions") or []
    return actions[0] if actions else None


def index_predictions(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed = {}
    for row in rows:
        keys = [row.get("record_id"), row.get("task_id"), (row.get("metadata") or {}).get("record_id")]
        for key in keys:
            if key:
                indexed.setdefault(str(key), row)
    return indexed


def score_row(gold_row: dict[str, Any], pred_row: dict[str, Any] | None) -> dict[str, Any]:
    metadata = gold_row.get("metadata") or {}
    gold = gold_action(gold_row)
    pred = pred_action(pred_row)
    gold_tool = metadata.get("next_tool") or ((gold or {}).get("tool_id") if gold else STOP)
    pred_tool = (pred or {}).get("tool_id") or STOP
    tool_hit = pred_tool == gold_tool
    gold_args = (gold or {}).get("arguments") or {}
    pred_args = (pred or {}).get("arguments") or {}
    if not isinstance(gold_args, dict):
        gold_args = {}
    if not isinstance(pred_args, dict):
        pred_args = {}
    gold_keys = set(gold_args)
    pred_keys = set(pred_args)
    key_hits = gold_keys & pred_keys
    value_hits = sum(norm_value(gold_args[key]) == norm_value(pred_args.get(key)) for key in key_hits)
    arg_value_em = None if gold_tool == STOP or not gold_keys else value_hits / len(gold_keys)
    return {
        "record_id": gold_row["id"],
        "task_id": metadata.get("task_id"),
        "domain": metadata.get("domain"),
        "source_kind": metadata.get("source_kind"),
        "gold_tool": gold_tool,
        "pred_tool": pred_tool,
        "tool_em": tool_hit,
        "success_static": tool_hit and (gold_tool == STOP or arg_value_em == 1.0),
        "gold_arg_count": len(gold_keys),
        "pred_arg_count": len(pred_keys),
        "arg_key_recall": None if not gold_keys else len(key_hits) / len(gold_keys),
        "arg_value_em": arg_value_em,
        "missing_prediction": pred_row is None,
    }


def summarize(scores: list[dict[str, Any]]) -> dict[str, Any]:
    if not scores:
        return {"count": 0}
    arg_rows = [row for row in scores if row["arg_value_em"] is not None]
    return {
        "count": len(scores),
        "tool_em": sum(row["tool_em"] for row in scores) / len(scores),
        "success_static": sum(row["success_static"] for row in scores) / len(scores),
        "arg_rows": len(arg_rows),
        "arg_key_recall": sum(row["arg_key_recall"] for row in arg_rows) / len(arg_rows) if arg_rows else None,
        "arg_value_em": sum(row["arg_value_em"] for row in arg_rows) / len(arg_rows) if arg_rows else None,
        "missing_predictions": sum(row["missing_prediction"] for row in scores),
    }


def group_summary(scores: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in scores:
        grouped.setdefault(str(row.get(key)), []).append(row)
    return {group: summarize(rows) for group, rows in sorted(grouped.items())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate row-level predictions on test500.")
    parser.add_argument("--gold", default=str(ROOT / "data" / "replan_sft" / "test500" / "test.jsonl"))
    parser.add_argument("--pred", required=True)
    parser.add_argument("--out", default=str(ROOT / "outputs" / "test500" / "eval.json"))
    parser.add_argument("--details", default=str(ROOT / "outputs" / "test500" / "eval_details.jsonl"))
    args = parser.parse_args()

    gold_rows = read_jsonl(args.gold)
    pred_rows = index_predictions(read_jsonl(args.pred))
    scores = [score_row(row, pred_rows.get(row["id"])) for row in gold_rows]
    report = {
        "format": "test500_static_eval_v1",
        "gold": args.gold,
        "pred": args.pred,
        "overall": summarize(scores),
        "by_domain": group_summary(scores, "domain"),
        "by_source_kind": group_summary(scores, "source_kind"),
        "pred_tool_top20": Counter(row["pred_tool"] for row in scores).most_common(20),
    }
    write_json(args.out, report)
    details = Path(args.details)
    details.parent.mkdir(parents=True, exist_ok=True)
    with details.open("w", encoding="utf-8") as f:
        for row in scores:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
