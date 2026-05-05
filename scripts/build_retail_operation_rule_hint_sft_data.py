import argparse
import json
import re
from pathlib import Path


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


def operation_rule(decision):
    text = str(decision.get("task") or "").lower()
    eligibility = {str(item) for item in (decision.get("grounding_progress") or {}).get("operation_eligibility") or []}

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
    if "modify_pending_order_address" in candidates and not has_address_payload(decision.get("task") or ""):
        candidates = [candidate for candidate in candidates if candidate != "modify_pending_order_address"]

    for candidate in candidates:
        if not eligibility or candidate in eligibility:
            return candidate
    return None if eligibility else (candidates[0] if candidates else None)


def operation_hint(decision, prediction):
    suggested = operation_rule(decision)
    if not suggested:
        return None
    progress = decision.get("grounding_progress") or {}
    payload = {
        "decision_case": decision.get("decision_case"),
        "selector": "retail_operation_rule_v1",
        "suggested_operation_tool": suggested,
        "fm_prior_tool": prediction,
        "resolved_order_status": progress.get("resolved_order_status"),
        "operation_eligibility": progress.get("operation_eligibility") or [],
        "instruction": "If this conflicts with the planning prior, prefer this operation selector for operation-ready retail rows.",
    }
    return "Retail operation selector prior:\n" + json.dumps(payload, ensure_ascii=False, sort_keys=True)


def load_prediction_map(path):
    if not Path(path).exists():
        return {}
    out = {}
    for row in read_jsonl(path):
        key = row.get("record_id") or (row.get("metadata") or {}).get("record_id") or row.get("task_id")
        tool_ids = row.get("tool_ids") or []
        if key and tool_ids:
            out[str(key)] = tool_ids[0]
    return out


def apply_hint(row, hint):
    row = json.loads(json.dumps(row, ensure_ascii=False))
    if not hint:
        return row
    for message in reversed(row.get("messages") or []):
        if message.get("role") == "user":
            message["content"] = (message.get("content") or "") + "\n\n" + hint
            return row
    row.setdefault("messages", []).insert(-1, {"role": "user", "content": hint})
    return row


def build_split(input_rows, decision_rows, predictions):
    decisions = {row["id"]: row for row in decision_rows}
    out = []
    hinted = 0
    correct = 0
    by_gold = {}
    for row in input_rows:
        decision = decisions.get(row.get("id"))
        hint = None
        if decision and decision.get("decision_case") == "post_order_operation":
            gold = (decision.get("target") or {}).get("tool_name")
            suggested = operation_rule(decision)
            by_gold[gold] = by_gold.get(gold, 0) + 1
            correct += int(bool(suggested) and suggested == gold)
            hint = operation_hint(decision, predictions.get(row.get("id")))
        if hint:
            hinted += 1
        out.append(apply_hint(row, hint))
    return out, {
        "rows": len(out),
        "operation_hinted": hinted,
        "operation_rule_correct": correct,
        "operation_rule_accuracy": correct / hinted if hinted else None,
        "operation_gold_distribution": by_gold,
    }


def main():
    parser = argparse.ArgumentParser(description="Append non-oracle retail operation-rule hints to replan SFT rows.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--decision-dir", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    manifest = {
        "input_dir": args.input_dir,
        "decision_dir": args.decision_dir,
        "pred_dir": args.pred_dir,
        "hint": "retail_operation_rule_v1_non_oracle",
        "splits": {},
    }
    for split in args.splits:
        pred_map = load_prediction_map(Path(args.pred_dir) / f"{split}.pred.jsonl")
        rows, stats = build_split(
            read_jsonl(Path(args.input_dir) / f"{split}.jsonl"),
            read_jsonl(Path(args.decision_dir) / f"{split}.jsonl"),
            pred_map,
        )
        write_jsonl(out_dir / f"{split}.jsonl", rows)
        manifest["splits"][split] = stats
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
