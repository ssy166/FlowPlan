import argparse
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


INTENT_TO_TOOL = [
    (("exchange",), "tau2::retail::exchange_delivered_order_items"),
    (("return", "refund"), "tau2::retail::return_delivered_order_items"),
    (("cancel",), "tau2::retail::cancel_pending_order"),
    (("address", "shipping address", "delivery address"), "tau2::retail::modify_pending_order_address"),
    (("item", "quantity", "size", "color"), "tau2::retail::modify_pending_order_items"),
]


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def find_balanced_json(text, start=0):
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


def parse_state(row):
    content = ""
    for message in row.get("messages") or []:
        if message.get("role") == "user":
            content = message.get("content") or ""
            break
    marker = "Feedback-conditioned state:"
    start = content.find(marker)
    start = content.find("{", start) if start >= 0 else content.find("{")
    candidate = find_balanced_json(content, start)
    if not candidate:
        return {}
    try:
        parsed = json.loads(candidate)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def assistant_target(row):
    for message in reversed(row.get("messages") or []):
        if message.get("role") == "assistant":
            try:
                return json.loads(message.get("content") or "{}")
            except Exception:
                return {}
    return {}


def gold_tool_id(row):
    target = assistant_target(row)
    actions = target.get("actions") or []
    if target.get("stop") or not actions:
        return None
    return actions[0].get("tool_id")


def state_text(row, state):
    parts = [state.get("user_prompt") or ""]
    metadata = row.get("metadata") or {}
    parts.append(metadata.get("task_id") or "")
    for step in state.get("executed_prefix") or []:
        parts.append(json.dumps(step, ensure_ascii=False, sort_keys=True))
    for item in state.get("assistant_initialization_feedback") or []:
        parts.append(json.dumps(item, ensure_ascii=False, sort_keys=True))
    return "\n".join(parts).lower()


def has_grounded_retail_state(state):
    for step in state.get("executed_prefix") or []:
        execution = step.get("execution") or {}
        if not execution.get("ok", False):
            continue
        action = step.get("predicted_action") or {}
        tool_name = (
            action.get("tool_name")
            or action.get("tool_id", "").rsplit("::", 1)[-1]
            or execution.get("tool_name")
            or execution.get("func_name")
            or ""
        )
        blob = json.dumps(execution.get("result"), ensure_ascii=False, sort_keys=True).lower()
        if tool_name.startswith("find_user_id") and "user_id" in blob:
            return True
        if "order_id" in blob or "order_item_id" in blob:
            return True
    return False


def available_tool_ids(state):
    return {tool.get("tool_id") for tool in state.get("available_tools") or [] if tool.get("tool_id")}


def infer_retail_intent_tool(row, state):
    if (row.get("metadata") or {}).get("domain") != "retail":
        return None
    if not has_grounded_retail_state(state):
        return None
    text = state_text(row, state)
    available = available_tool_ids(state)
    for keywords, tool_id in INTENT_TO_TOOL:
        if tool_id in available and any(keyword in text for keyword in keywords):
            return tool_id
    return None


def pred_key(row):
    return row.get("record_id") or (row.get("metadata") or {}).get("record_id") or row.get("task_id")


def tool_name(tool_id):
    return tool_id.rsplit("::", 1)[-1] if tool_id else None


def should_override(pred_tool_id, args):
    if not pred_tool_id:
        return False
    if args.only_lookup:
        return pred_tool_id.startswith("tau2::retail::find_user_id")
    return pred_tool_id.startswith("tau2::retail::")


def apply_override(rows, preds, args):
    by_key = {pred_key(row): row for row in preds if pred_key(row)}
    out = []
    events = []
    for row in rows:
        row_id = row.get("id") or (row.get("metadata") or {}).get("record_id")
        pred = json.loads(json.dumps(by_key.get(row_id, {"record_id": row_id, "tool_ids": [], "actions": []})))
        state = parse_state(row)
        pred_tool_ids = pred.get("tool_ids") or []
        pred_tool_id = pred_tool_ids[0] if pred_tool_ids else None
        override_tool_id = infer_retail_intent_tool(row, state)
        if override_tool_id and should_override(pred_tool_id, args):
            pred["tool_ids"] = [override_tool_id]
            pred["actions"] = [{"tool_id": override_tool_id, "tool_name": tool_name(override_tool_id), "arguments": {}}]
            pred["plan_type"] = f"{pred.get('plan_type') or 'prediction'}_retail_intent_override"
            metadata = dict(pred.get("metadata") or {})
            metadata.update(
                {
                    "retail_intent_override": True,
                    "original_tool_id": pred_tool_id,
                    "override_tool_id": override_tool_id,
                }
            )
            pred["metadata"] = metadata
            events.append((gold_tool_id(row), pred_tool_id, override_tool_id))
        out.append(pred)
    return out, events


def summarize(rows, preds):
    by_key = {pred_key(row): row for row in preds if pred_key(row)}
    groups = {"overall": [], "retail": [], "telecom": []}
    for row in rows:
        row_id = row.get("id") or (row.get("metadata") or {}).get("record_id")
        pred = by_key.get(row_id) or {}
        pred_ids = pred.get("tool_ids") or []
        pred_tool_id = pred_ids[0] if pred_ids else None
        gold = gold_tool_id(row)
        ok = pred_tool_id == gold
        domain = (row.get("metadata") or {}).get("domain")
        groups["overall"].append(ok)
        if domain in groups:
            groups[domain].append(ok)
    return {name: (sum(vals) / len(vals) if vals else None) for name, vals in groups.items()}


def main():
    parser = argparse.ArgumentParser(description="Apply a diagnostic retail intent override to next-tool prior predictions.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--only-lookup", action="store_true", default=True)
    args = parser.parse_args()

    rows = read_jsonl(args.data)
    preds = read_jsonl(args.pred)
    before = summarize(rows, preds)
    out, events = apply_override(rows, preds, args)
    after = summarize(rows, out)
    summary = {
        "before": before,
        "after": after,
        "override_count": len(events),
        "events": [
            {"gold_tool_id": gold, "original_tool_id": original, "override_tool_id": override}
            for gold, original, override in events
        ],
        "gold_original_override_counts": {
            " | ".join(str(x) for x in key): value
            for key, value in Counter(events).most_common()
        },
    }
    write_jsonl(args.out, out)
    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_out).write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
