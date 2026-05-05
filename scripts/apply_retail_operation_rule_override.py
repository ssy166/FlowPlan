import argparse
import json
import re
from pathlib import Path


RETAIL_LOOKUP_TOOLS = {
    "find_user_id_by_email",
    "find_user_id_by_name_zip",
    "get_user_details",
    "get_order_details",
    "get_product_details",
    "get_item_details",
    "list_all_product_types",
}
ZIP_RE = re.compile(r"\b\d{5}\b")


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
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def find_balanced_json(text, start=0):
    first = text.find("{", max(0, start))
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
        elif ch == '"':
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
    starts = []
    for marker in ["Compact state:", "Feedback-conditioned state:", "Conditioned workflow state:"]:
        pos = content.find(marker)
        if pos >= 0:
            starts.append(pos)
    starts.append(0)
    for start in starts:
        candidate = find_balanced_json(content, start)
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def has_word(text, *words):
    return any(re.search(rf"\b{re.escape(word)}\b", text) for word in words)


def has_address_payload(text):
    text = str(text or "")
    low = text.lower()
    return bool(ZIP_RE.search(text) and ("," in text or "address" in low or "shipping" in low or "deliver" in low))


def has_item_modification_intent(text):
    return has_word(
        str(text or "").lower(),
        "item",
        "items",
        "product",
        "products",
        "color",
        "size",
        "material",
        "backpack",
        "lamp",
        "desk",
    )


def operation_rule_from_state(state):
    text = str(state.get("task") or "").lower()
    progress = state.get("grounding_progress") or {}
    eligibility = {str(item) for item in (progress.get("operation_eligibility") or [])}
    if has_word(text, "return", "refund"):
        candidates = ["return_delivered_order_items"]
    elif has_word(text, "exchange", "replace", "replacement"):
        candidates = ["exchange_delivered_order_items"]
    elif has_word(text, "cancel"):
        candidates = ["cancel_pending_order"]
    elif has_word(text, "modify", "change", "update", "adjust") and has_item_modification_intent(text):
        candidates = ["modify_pending_order_items", "modify_pending_order_address", "modify_pending_order_payment"]
    elif has_word(text, "address", "shipping"):
        candidates = ["modify_pending_order_address", "modify_user_address"]
    elif has_word(text, "payment", "card", "paypal"):
        candidates = ["modify_pending_order_payment"]
    elif has_word(text, "modify", "change", "update"):
        candidates = ["modify_pending_order_items", "modify_pending_order_address", "modify_pending_order_payment"]
    else:
        candidates = []
    if "modify_pending_order_address" in candidates and not has_address_payload(state.get("task") or ""):
        candidates = [candidate for candidate in candidates if candidate != "modify_pending_order_address"]
    for candidate in candidates:
        if not eligibility or candidate in eligibility:
            return candidate
    # If the state already exposes operation eligibility, never fall back to
    # an ineligible keyword candidate. That fallback caused valid lookup rows
    # to be overwritten by non-executable operations on pending/delivered
    # status mismatches.
    return None if eligibility else (candidates[0] if candidates else None)


def tool_name(tool_id):
    return str(tool_id or "").rsplit("::", 1)[-1]


def should_override(pred_tool_id, state, suggested):
    if not suggested:
        return False
    if state.get("domain") != "retail":
        return False
    progress = state.get("grounding_progress") or {}
    if not progress.get("order_grounded"):
        return False
    if not progress.get("operation_eligibility"):
        return False
    current = tool_name(pred_tool_id)
    return (not pred_tool_id) or current in RETAIL_LOOKUP_TOOLS


def override_row(row, state, suggested):
    row = json.loads(json.dumps(row, ensure_ascii=False))
    tool_id = f"tau2::retail::{suggested}"
    action = {
        "tool_id": tool_id,
        "tool_name": suggested,
        "arguments": {},
        "metadata": {"retail_operation_rule_override": True},
    }
    row["tool_ids"] = [tool_id]
    row["tool_names"] = [suggested]
    row["actions"] = [action]
    metadata = dict(row.get("metadata") or {})
    metadata["retail_operation_rule_override"] = {
        "suggested_tool": suggested,
        "order_status": (state.get("grounding_progress") or {}).get("resolved_order_status"),
    }
    row["metadata"] = metadata
    return row


def load_selector_predictions(path):
    if not path:
        return {}
    out = {}
    for row in read_jsonl(path):
        out[str(row.get("id"))] = row
    return out


def main():
    parser = argparse.ArgumentParser(description="Apply a conservative retail operation-rule override to prediction JSONL.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--selector-pred", default=None)
    parser.add_argument("--selector-threshold", type=float, default=0.0)
    args = parser.parse_args()

    states = {row["id"]: parse_state(row) for row in read_jsonl(args.data)}
    selector_predictions = load_selector_predictions(args.selector_pred)
    rows = []
    stats = {"rows": 0, "overrides": 0, "override_by_tool": {}, "selector_filtered": 0}
    for row in read_jsonl(args.pred):
        stats["rows"] += 1
        row_id = row.get("record_id") or (row.get("metadata") or {}).get("record_id") or row.get("task_id")
        state = states.get(row_id) or {}
        pred_tool_id = (row.get("tool_ids") or [None])[0]
        suggested = operation_rule_from_state(state)
        selector_ok = True
        if selector_predictions:
            selector = selector_predictions.get(str(row_id)) or {}
            selector_ok = (
                selector.get("prediction") == "post_order_operation"
                and float(selector.get("confidence") or 0.0) >= args.selector_threshold
            )
            if not selector_ok:
                stats["selector_filtered"] += 1
        if selector_ok and should_override(pred_tool_id, state, suggested):
            row = override_row(row, state, suggested)
            stats["overrides"] += 1
            stats["override_by_tool"][suggested] = stats["override_by_tool"].get(suggested, 0) + 1
        rows.append(row)
    write_jsonl(args.out, rows)
    print(json.dumps(stats, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
