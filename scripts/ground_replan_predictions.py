import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_replan_execution import parse_state, row_id  # noqa: E402


ZIP_RE = re.compile(r"\b(\d{5})\b")
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
USER_ID_RE = re.compile(r"\b[a-z]+_[a-z]+_\d+\b")
ORDER_ID_RE = re.compile(r"#?W\d{7}\b")
NAME_PATTERNS = [
    re.compile(r"\bcalled\s+([A-Z][a-z]+)\s+([A-Z][a-z]+)\b"),
    re.compile(r"\bname is\s+([A-Z][a-z]+)\s+([A-Z][a-z]+)\b"),
    re.compile(r"\bname(?:s)?\s+is\s+([A-Z][a-z]+)\s+([A-Z][a-z]+)\b"),
    re.compile(r"\bYou are\s+(?:an? [^.]*? called\s+)?([A-Z][a-z]+)\s+([A-Z][a-z]+)\b"),
]
RETAIL_DB_PATH = ROOT / "data" / "raw" / "tau2-bench" / "repo" / "data" / "tau2" / "domains" / "retail" / "db.json"
STATE_ABBR = {
    "alabama": "AL",
    "alaska": "AK",
    "arizona": "AZ",
    "arkansas": "AR",
    "california": "CA",
    "colorado": "CO",
    "connecticut": "CT",
    "florida": "FL",
    "georgia": "GA",
    "illinois": "IL",
    "indiana": "IN",
    "new york": "NY",
    "nyc": "NY",
    "texas": "TX",
    "washington": "WA",
}
CITY_STATE = {
    "new york": "NY",
    "fort worth": "TX",
    "indianapolis": "IN",
    "seattle": "WA",
    "denver": "CO",
}


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


def unique(values: list[Any]) -> list[str]:
    out = []
    seen = set()
    for value in values:
        if value in (None, ""):
            continue
        text = str(value)
        if text not in seen:
            out.append(text)
            seen.add(text)
    return out


def load_retail_db() -> dict[str, Any]:
    if not RETAIL_DB_PATH.exists():
        return {}
    try:
        with RETAIL_DB_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def collect_summaries(state: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[str]]:
    orders = {}
    products = []
    user = {}
    user_ids = []
    for step in state.get("executed_prefix") or []:
        summary = step.get("result_summary")
        if isinstance(summary, str):
            user_ids.extend(USER_ID_RE.findall(summary))
        if isinstance(summary, dict):
            if summary.get("order_id"):
                orders[str(summary["order_id"])] = summary
            if summary.get("variant_item_ids"):
                products.append(summary)
            if summary.get("user_id") and "address" in summary:
                user = summary
            if summary.get("user_id"):
                user_ids.append(str(summary["user_id"]))
    user_ids.extend(USER_ID_RE.findall(json.dumps(state, ensure_ascii=False, sort_keys=True, default=str)))
    return orders, products, user, unique(user_ids)


def extract_name(task: str) -> tuple[str | None, str | None]:
    for pattern in NAME_PATTERNS:
        match = pattern.search(task or "")
        if match:
            return match.group(1), match.group(2)
    return None, None


def state_from_text(text: str) -> str | None:
    low = text.lower()
    for name, abbr in STATE_ABBR.items():
        if name in low:
            return abbr
    return None


def extract_address(task: str) -> dict[str, str]:
    task = " ".join((task or "").split())
    patterns = [
        # "... live in 445 Maple Drive, Suite 394, Fort Worth, Texas, 76165"
        re.compile(
            r"live in (?P<address1>[^,]+),\s*(?P<address2>[^,]+),\s*(?P<city>[^,]+),\s*(?P<state>[A-Za-z ]+),\s*(?P<zip>\d{5})",
            re.I,
        ),
        # "... to be 101 Highway, New York, 10001"
        re.compile(r"to be (?P<address1>[^,]+),\s*(?P<city>[^,]+),\s*(?P<zip>\d{5})", re.I),
    ]
    for pattern in patterns:
        match = pattern.search(task)
        if not match:
            continue
        data = {k: v.strip() for k, v in match.groupdict().items() if v is not None}
        state = data.get("state")
        city = data.get("city", "").lower()
        if state:
            data["state"] = STATE_ABBR.get(state.lower(), state)
        elif city in CITY_STATE:
            data["state"] = CITY_STATE[city]
        data.setdefault("address2", "")
        data.setdefault("country", "USA")
        return data
    return {}


def zip_from_task(task: str) -> str | None:
    for pattern in [
        r"zip(?: code)?\s*(\d{5})",
        r"live(?:s)? in [^.]*?\b(\d{5})\b",
        r"living in [^.]*?\b(\d{5})\b",
    ]:
        match = re.search(pattern, task or "", re.I)
        if match:
            return match.group(1)
    matches = ZIP_RE.findall(task or "")
    return matches[-1] if matches else None


def ids_for_order(order_id: str | None, orders: dict[str, dict[str, Any]], progress: dict[str, Any]) -> tuple[list[str], list[str]]:
    if order_id and order_id in orders:
        summary = orders[order_id]
        return unique(summary.get("item_ids") or []), unique(summary.get("payment_method_ids") or [])
    return unique(progress.get("resolved_order_item_ids") or []), unique(progress.get("resolved_payment_method_ids") or [])


def product_variant_ids(products: list[dict[str, Any]], exclude: set[str]) -> list[str]:
    values = []
    for product in products:
        for item_id in product.get("variant_item_ids") or []:
            if str(item_id) not in exclude:
                values.append(str(item_id))
    return unique(values)


def variant_options(db: dict[str, Any], item_id: str) -> tuple[str | None, dict[str, Any]]:
    for product_id, product in (db.get("products") or {}).items():
        variants = product.get("variants") or {}
        if item_id in variants:
            return str(product_id), variants[item_id]
    return None, {}


def best_variant_for_item(item_id: str, task: str, db: dict[str, Any]) -> str | None:
    product_id, old_variant = variant_options(db, item_id)
    if not product_id:
        return None
    product = (db.get("products") or {}).get(product_id) or {}
    candidates = []
    low = (task or "").lower()
    for candidate_id, variant in (product.get("variants") or {}).items():
        if candidate_id == item_id or not variant.get("available"):
            continue
        options = variant.get("options") or {}
        score = 0.0
        if "easy" in low or "easiest" in low or "little kid" in low or "kid" in low:
            difficulty = str(options.get("difficulty level") or "").lower()
            score += {"beginner": 4, "intermediate": 2, "expert": 0}.get(difficulty, 0)
        if "fewest" in low or "few" in low or "less" in low:
            try:
                score -= float(options.get("pieces") or 0) / 1000.0
            except Exception:
                pass
        for value in options.values():
            if str(value).lower() in low:
                score += 1.0
        # Prefer available variants closest in price when task text gives no option clue.
        try:
            score -= abs(float(variant.get("price") or 0) - float(old_variant.get("price") or 0)) / 100.0
        except Exception:
            pass
        candidates.append((score, str(candidate_id)))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def align_new_items(item_ids: list[Any], new_item_ids: list[Any], task: str, db: dict[str, Any]) -> list[str]:
    out = []
    for idx, item in enumerate(item_ids):
        item_id = str(item)
        current = str(new_item_ids[idx]) if idx < len(new_item_ids) else ""
        old_product, _ = variant_options(db, item_id)
        new_product, new_variant = variant_options(db, current) if current else (None, {})
        if current and old_product and new_product == old_product and new_variant.get("available"):
            out.append(current)
            continue
        replacement = best_variant_for_item(item_id, task, db)
        if replacement:
            out.append(replacement)
        elif current:
            out.append(current)
    return unique(out)


def relevant_order_items(order_id: str | None, orders: dict[str, dict[str, Any]], task: str) -> list[str]:
    if not order_id or order_id not in orders:
        return []
    summary = orders[order_id]
    low = (task or "").lower()
    names = [str(x) for x in summary.get("item_names") or []]
    ids = [str(x) for x in summary.get("item_ids") or []]
    hits = []
    for item_id, name in zip(ids, names):
        name_low = name.lower()
        tokens = [tok for tok in re.split(r"[^a-z0-9]+", name_low) if len(tok) > 2]
        if name_low in low or any(tok in low for tok in tokens):
            hits.append(item_id)
    return unique(hits)


def normalize_order_id(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value)
    match = ORDER_ID_RE.search(text)
    if not match:
        return text
    order_id = match.group(0)
    return order_id if order_id.startswith("#") else f"#{order_id}"


def valid_order_id(value: Any, db: dict[str, Any]) -> bool:
    order_id = normalize_order_id(value)
    return bool(order_id and order_id in (db.get("orders") or {}))


def valid_payment_id(value: Any, db: dict[str, Any]) -> bool:
    if value in (None, ""):
        return False
    needle = str(value)
    for user in (db.get("users") or {}).values():
        if needle in (user.get("payment_methods") or {}):
            return True
    return False


def valid_item_id(value: Any, db: dict[str, Any]) -> bool:
    if value in (None, ""):
        return False
    return bool(variant_options(db, str(value))[0])


def find_user_from_task(task: str, db: dict[str, Any]) -> dict[str, Any] | None:
    low = (task or "").lower()
    users = db.get("users") or {}
    email_match = EMAIL_RE.search(task or "")
    if email_match:
        email = email_match.group(0).lower()
        for user in users.values():
            if str(user.get("email") or "").lower() == email:
                return user
    first, last = extract_name(task)
    zip_code = zip_from_task(task)
    for user in users.values():
        name = user.get("name") or {}
        address = user.get("address") or {}
        if first and last:
            if str(name.get("first_name") or "").lower() != first.lower():
                continue
            if str(name.get("last_name") or "").lower() != last.lower():
                continue
            if zip_code and str(address.get("zip") or "") != zip_code:
                continue
            return user
    for user in users.values():
        user_id = str(user.get("user_id") or "")
        if user_id and user_id.lower() in low:
            return user
    return None


def db_orders_for_state(state: dict[str, Any], db: dict[str, Any]) -> list[dict[str, Any]]:
    progress = state.get("grounding_progress") or {}
    order_ids = [normalize_order_id(x) for x in progress.get("resolved_order_ids") or []]
    orders = [db["orders"][oid] for oid in order_ids if oid in (db.get("orders") or {})]
    if orders:
        return orders
    user = find_user_from_task(state.get("task") or state.get("user_prompt") or "", db)
    if not user:
        return []
    return [(db.get("orders") or {}).get(order_id) for order_id in user.get("orders") or [] if order_id in (db.get("orders") or {})]


def desired_status_for_tool(name: str) -> str | None:
    if name in {"cancel_pending_order", "modify_pending_order_address", "modify_pending_order_items", "modify_pending_order_payment"}:
        return "pending"
    if name in {"return_delivered_order_items", "exchange_delivered_order_items"}:
        return "delivered"
    return None


def order_text_score(order: dict[str, Any], task: str, name: str) -> float:
    low = (task or "").lower()
    score = 0.0
    status = str(order.get("status") or "").lower()
    desired = desired_status_for_tool(name)
    if desired and desired in status:
        score += 5.0
    if str(order.get("order_id") or "").lower() in low:
        score += 10.0
    for item in order.get("items") or []:
        item_name = str(item.get("name") or "").lower()
        tokens = [tok for tok in re.split(r"[^a-z0-9]+", item_name) if len(tok) > 2]
        if item_name and item_name in low:
            score += 3.0
        score += sum(1.0 for tok in tokens if tok in low)
    address = order.get("address") or {}
    for key in ["city", "state", "zip"]:
        value = str(address.get(key) or "").lower()
        if value and value in low:
            score += 0.5
    return score


def infer_db_order_id(name: str, state: dict[str, Any], db: dict[str, Any]) -> str | None:
    orders = [order for order in db_orders_for_state(state, db) if order]
    if not orders:
        return None
    scored = [(order_text_score(order, state.get("task") or state.get("user_prompt") or "", name), order) for order in orders]
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1].get("order_id") if scored else None


def db_order_items(order_id: str | None, db: dict[str, Any]) -> list[dict[str, Any]]:
    order_id = normalize_order_id(order_id)
    if not order_id:
        return []
    order = (db.get("orders") or {}).get(order_id) or {}
    return order.get("items") or []


def db_payment_ids(order_id: str | None, db: dict[str, Any]) -> list[str]:
    order_id = normalize_order_id(order_id)
    if not order_id:
        return []
    order = (db.get("orders") or {}).get(order_id) or {}
    return unique([payment.get("payment_method_id") for payment in order.get("payment_history") or [] if isinstance(payment, dict)])


def relevant_db_item_ids(order_id: str | None, task: str, db: dict[str, Any]) -> list[str]:
    items = db_order_items(order_id, db)
    low = (task or "").lower()
    hits = []
    for item in items:
        item_name = str(item.get("name") or "").lower()
        tokens = [tok for tok in re.split(r"[^a-z0-9]+", item_name) if len(tok) > 2]
        if item_name in low or any(tok in low for tok in tokens):
            hits.append(str(item.get("item_id")))
    return unique(hits or [item.get("item_id") for item in items])


def product_id_for_item(item_id: Any, orders: dict[str, dict[str, Any]]) -> str | None:
    if item_id in (None, ""):
        return None
    item_id = str(item_id)
    for summary in orders.values():
        item_ids = [str(x) for x in summary.get("item_ids") or []]
        product_ids = [str(x) for x in summary.get("product_ids") or []]
        if item_id in item_ids:
            idx = item_ids.index(item_id)
            if idx < len(product_ids):
                return product_ids[idx]
    return None


def cancel_reason(task: str) -> str:
    low = (task or "").lower()
    if "ordered by mistake" in low or "mistake" in low:
        return "ordered by mistake"
    return "no longer needed"


def fill_action(
    action: dict[str, Any],
    state: dict[str, Any],
    retail_db: dict[str, Any] | None = None,
    schema_normalization: bool = True,
    entity_binding: bool = True,
    field_mapping: bool = True,
    db_validation: bool = True,
) -> tuple[dict[str, Any], dict[str, str]]:
    action = json.loads(json.dumps(action, ensure_ascii=False))
    args = dict(action.get("arguments") or {})
    name = action.get("tool_name") or str(action.get("tool_id") or "").rsplit("::", 1)[-1]
    task = state.get("task") or state.get("user_prompt") or ""
    progress = state.get("grounding_progress") or {} if entity_binding else {}
    orders, products, user, state_user_ids = collect_summaries(state) if entity_binding else ({}, [], {}, [])
    retail_db = retail_db or {} if db_validation else {}
    fills = {}

    normalized_order = normalize_order_id(args.get("order_id"))
    if schema_normalization and normalized_order and normalized_order != args.get("order_id"):
        args["order_id"] = normalized_order
        fills["order_id"] = "normalized"

    if field_mapping and name == "find_user_id_by_name_zip":
        first_name, last_name = extract_name(task)
        if not args.get("first_name") and first_name:
            args["first_name"] = first_name
            fills["first_name"] = "task_text"
        if not args.get("last_name") and last_name:
            args["last_name"] = last_name
            fills["last_name"] = "task_text"
        if not args.get("zip"):
            zip_code = zip_from_task(task)
            if zip_code:
                args["zip"] = zip_code
                fills["zip"] = "task_text"

    if name in {"get_user_details", "modify_user_address"} and not args.get("user_id"):
        user_ids = progress.get("resolved_user_ids") or state_user_ids or ([user.get("user_id")] if user.get("user_id") else [])
        if user_ids:
            args["user_id"] = user_ids[0]
            fills["user_id"] = "grounding_progress"

    needs_order = name in {
        "get_order_details",
        "cancel_pending_order",
        "modify_pending_order_address",
        "modify_pending_order_items",
        "modify_pending_order_payment",
        "return_delivered_order_items",
        "exchange_delivered_order_items",
    }
    if needs_order and (not args.get("order_id") or (retail_db and not valid_order_id(args.get("order_id"), retail_db))):
        order_ids = progress.get("resolved_order_ids") or list(orders)
        if order_ids:
            args["order_id"] = normalize_order_id(order_ids[0]) or order_ids[0]
            fills["order_id"] = "grounding_progress"
        elif db_validation and retail_db:
            db_order_id = infer_db_order_id(name, state, retail_db)
            if db_order_id:
                args["order_id"] = db_order_id
                fills["order_id"] = "retail_db_task"

    if name == "get_product_details" and (not args.get("product_id") or (retail_db and args.get("product_id") not in (retail_db.get("products") or {}))) and args.get("item_id"):
        product_id = product_id_for_item(args.get("item_id"), orders)
        if product_id:
            args["product_id"] = product_id
            fills["product_id"] = "order_summary"
    if db_validation and name == "get_product_details" and retail_db and (not args.get("product_id") or args.get("product_id") not in (retail_db.get("products") or {})):
        low = task.lower()
        best = None
        best_score = 0
        for product_id, product in (retail_db.get("products") or {}).items():
            product_name = str(product.get("name") or "").lower()
            tokens = [tok for tok in re.split(r"[^a-z0-9]+", product_name) if len(tok) > 2]
            score = int(product_name in low) * 3 + sum(1 for tok in tokens if tok in low)
            if score > best_score:
                best = product_id
                best_score = score
        if best:
            args["product_id"] = best
            fills["product_id"] = "retail_db_task"

    if field_mapping and name in {"modify_user_address", "modify_pending_order_address"}:
        address = extract_address(task)
        if not address and isinstance(user.get("address"), dict):
            address = dict(user["address"])
        if address:
            for key in ["address1", "address2", "city", "state", "country", "zip"]:
                if key not in args or args.get(key) in (None, ""):
                    if key in address:
                        args[key] = address[key]
                        fills[key] = "task_or_user_address"
            if not args.get("state"):
                inferred = state_from_text(task)
                if inferred:
                    args["state"] = inferred
                    fills["state"] = "task_text"

    if name in {
        "modify_pending_order_payment",
        "modify_pending_order_items",
        "return_delivered_order_items",
        "exchange_delivered_order_items",
    }:
        order_id = args.get("order_id")
        item_ids, payment_ids = ids_for_order(order_id, orders, progress)
        if retail_db and order_id:
            db_items = relevant_db_item_ids(order_id, task, retail_db)
            db_payments = db_payment_ids(order_id, retail_db)
            if db_items:
                item_ids = db_items
            if db_payments:
                payment_ids = db_payments
        existing_items = args.get("item_ids") or []
        invalid_items = retail_db and existing_items and any(not valid_item_id(item, retail_db) for item in existing_items)
        if name != "modify_pending_order_payment" and (not existing_items or invalid_items) and item_ids:
            relevant = relevant_order_items(order_id, orders, task) if field_mapping else []
            args["item_ids"] = relevant or item_ids
            fills["item_ids"] = "task_order_summary" if relevant else "order_or_db_summary"
        if (not args.get("payment_method_id") or (retail_db and not valid_payment_id(args.get("payment_method_id"), retail_db))) and payment_ids:
            args["payment_method_id"] = payment_ids[0]
            fills["payment_method_id"] = "order_or_db_summary"
        if db_validation and name in {"modify_pending_order_items", "exchange_delivered_order_items"} and args.get("item_ids"):
            aligned = align_new_items(args.get("item_ids") or [], args.get("new_item_ids") or [], task, retail_db)
            if aligned and aligned != [str(x) for x in (args.get("new_item_ids") or [])]:
                args["new_item_ids"] = aligned
                fills["new_item_ids"] = "db_variant_options"
        if entity_binding and name in {"modify_pending_order_items", "exchange_delivered_order_items"} and not args.get("new_item_ids"):
            exclude = set(str(x) for x in (args.get("item_ids") or []))
            candidates = product_variant_ids(products, exclude)
            if candidates:
                count = max(1, len(args.get("item_ids") or [None]))
                args["new_item_ids"] = candidates[:count]
                fills["new_item_ids"] = "product_summary"

    if field_mapping and name == "cancel_pending_order" and args.get("reason") not in {"no longer needed", "ordered by mistake"}:
        args["reason"] = cancel_reason(task)
        fills["reason"] = "task_text"

    if fills:
        action["arguments"] = args
        metadata = dict(action.get("metadata") or {})
        metadata["state_grounded_args"] = fills
        action["metadata"] = metadata
    return action, fills


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill missing replan prediction arguments from compact state.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--disable-schema-normalization", action="store_true")
    parser.add_argument("--disable-entity-binding", action="store_true")
    parser.add_argument("--disable-field-mapping", action="store_true")
    parser.add_argument("--disable-db-validation", action="store_true")
    args = parser.parse_args()

    states = {row_id(row): parse_state(row) for row in read_jsonl(args.data) if row_id(row)}
    retail_db = load_retail_db()
    rows = []
    fill_counts = {}
    for row in read_jsonl(args.pred):
        row = json.loads(json.dumps(row, ensure_ascii=False))
        state = states.get(row.get("record_id") or row.get("id")) or {}
        actions = []
        row_fills = {}
        if state.get("domain") == "retail":
            for action in row.get("actions") or []:
                grounded, fills = fill_action(
                    action,
                    state,
                    retail_db,
                    schema_normalization=not args.disable_schema_normalization,
                    entity_binding=not args.disable_entity_binding,
                    field_mapping=not args.disable_field_mapping,
                    db_validation=not args.disable_db_validation,
                )
                actions.append(grounded)
                for key in fills:
                    fill_counts[key] = fill_counts.get(key, 0) + 1
                    row_fills[key] = fills[key]
            row["actions"] = actions
            row["tool_ids"] = [action.get("tool_id") for action in actions]
            row["tool_names"] = [action.get("tool_name") for action in actions]
            if row_fills:
                metadata = dict(row.get("metadata") or {})
                metadata["state_grounded_args"] = row_fills
                metadata["grounding_components"] = {
                    "schema_normalization": not args.disable_schema_normalization,
                    "entity_binding": not args.disable_entity_binding,
                    "field_mapping": not args.disable_field_mapping,
                    "db_validation": not args.disable_db_validation,
                }
                row["metadata"] = metadata
        rows.append(row)
    write_jsonl(args.out, rows)
    print(json.dumps({"rows": len(rows), "fill_counts": fill_counts}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
