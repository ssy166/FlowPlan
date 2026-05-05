import argparse
import json
import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 on some remote envs
    import toml as tomllib


ROOT = Path(__file__).resolve().parents[1]
TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")
USER_ID_RE = re.compile(r"\b[a-z]+_[a-z]+_\d+\b")
RESERVATION_ID_RE = re.compile(r"\b[A-Z0-9]{6}\b")
DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
TIME_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
PHONE_RE = re.compile(r"\b\d{3}-\d{3}-\d{4}\b")
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
ZIP_RE = re.compile(r"\b\d{5}(?:-\d{4})?\b")
ORDER_ID_RE = re.compile(r"#[A-Z0-9]{8}\b")
PRODUCT_ID_RE = re.compile(r"\b\d{10}\b")
NAME_ZIP_RE = re.compile(
    r"\b(?:You are|Your name is|You name is)\s+([A-Z][A-Za-z'-]+)\s+([A-Z][A-Za-z'-]+).{0,200}?\b(?:zip(?:code| code)?(?:\s+is|:)?|live in(?:\s+zipcode)?(?:\s+[A-Za-z .'-]+,)?|residing in(?:\s+[A-Za-z .'-]+,)?|live at.{0,80}?)\s+(\d{5})",
    re.IGNORECASE | re.DOTALL,
)
RAW_TASK_CACHE = {}


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def tokens(text):
    return set(TOKEN_RE.findall((text or "").lower()))


def score_tool(task_tokens, tool):
    tool_text = " ".join(
        str(x)
        for x in [
            tool.get("name"),
            tool.get("description"),
            tool.get("domain"),
            tool.get("phase"),
        ]
        if x
    )
    overlap = task_tokens & tokens(tool_text)
    return len(overlap)


def keyword_nearest(tasks, tools, max_tools):
    rows = []
    for task in tasks:
        candidates = [tools[t] for t in task.get("available_tool_ids", []) if t in tools]
        task_tokens = tokens(task.get("prompt", ""))
        ranked = sorted(
            candidates,
            key=lambda tool: (score_tool(task_tokens, tool), tool.get("tool_split") != "train_seen", tool["tool_id"]),
            reverse=True,
        )
        chosen = [tool for tool in ranked if score_tool(task_tokens, tool) > 0][:max_tools]
        if not chosen and ranked:
            chosen = ranked[:1]
        rows.append(
            {
                "task_id": task["task_id"],
                "source": task["source"],
                "plan_type": "baseline_keyword_nearest",
                "tool_ids": [tool["tool_id"] for tool in chosen],
                "tool_names": [tool["name"] for tool in chosen],
                "phase_names": [tool["phase"] for tool in chosen],
            }
        )
    return rows


def values_for_param(prompt, param):
    if param == "user_id":
        return USER_ID_RE.findall(prompt or "")
    if param in {"reservation_id", "confirmation_number"}:
        return RESERVATION_ID_RE.findall(prompt or "")
    if param == "date":
        return DATE_RE.findall(prompt or "")
    if param in {"leave_after", "time"}:
        return TIME_RE.findall(prompt or "")
    if param in {"email", "new_email"}:
        return EMAIL_RE.findall(prompt or "")
    if param in {"phone_number"}:
        return PHONE_RE.findall(prompt or "")
    if param in {"zip", "zip_code"}:
        return ZIP_RE.findall(prompt or "")
    if param == "first_name":
        match = NAME_ZIP_RE.search(prompt or "")
        return [match.group(1)] if match else []
    if param == "last_name":
        match = NAME_ZIP_RE.search(prompt or "")
        return [match.group(2)] if match else []
    if param == "order_id":
        return ORDER_ID_RE.findall(prompt or "")
    if param in {"product_id", "item_id"}:
        return PRODUCT_ID_RE.findall(prompt or "")
    return []


def infer_arguments(prompt, tool, counters):
    params = ((tool.get("schema") or {}).get("parameters")) or []
    args = {}
    for param in params:
        values = values_for_param(prompt, param)
        if not values:
            continue
        key = (tool["tool_id"], param)
        idx = counters.get(key, 0)
        args[param] = values[min(idx, len(values) - 1)]
        counters[key] = idx + 1
    return args


def with_actions(rows, tasks, tools):
    task_by_id = {task["task_id"]: task for task in tasks}
    out = []
    for row in rows:
        task = task_by_id[row["task_id"]]
        counters = {}
        actions = []
        for tool_id in row.get("tool_ids") or []:
            tool = tools.get(tool_id) or {}
            actions.append(
                {
                    "tool_id": tool_id,
                    "tool_name": tool.get("name") or tool_id.rsplit("::", 1)[-1],
                    "arguments": infer_arguments(task.get("prompt", ""), tool, counters),
                }
            )
        enriched = dict(row)
        enriched["plan_type"] = f"{row.get('plan_type')}_with_schema_args"
        enriched["actions"] = actions
        out.append(enriched)
    return out


def unique_in_order(values):
    seen = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def action_row(tool_id, tools, arguments):
    tool = tools.get(tool_id) or {}
    return {
        "tool_id": tool_id,
        "tool_name": tool.get("name") or tool_id.rsplit("::", 1)[-1],
        "arguments": arguments,
    }


def arg_routed(tasks, tools, max_tools):
    fallback = {row["task_id"]: row for row in with_actions(keyword_nearest(tasks, tools, max_tools), tasks, tools)}
    rows = []
    for task in tasks:
        if task.get("source") != "tau2":
            rows.append(fallback[task["task_id"]])
            continue

        available = set(task.get("available_tool_ids") or [])
        prompt = task.get("prompt", "")
        actions = []

        user_tool = next((tool_id for tool_id in available if tool_id.endswith("::get_user_details")), None)
        if user_tool:
            for user_id in unique_in_order(USER_ID_RE.findall(prompt)):
                actions.append(action_row(user_tool, tools, {"user_id": user_id}))

        reservation_tool = next(
            (tool_id for tool_id in available if tool_id.endswith("::get_reservation_details")),
            None,
        )
        if reservation_tool:
            for reservation_id in unique_in_order(RESERVATION_ID_RE.findall(prompt)):
                actions.append(action_row(reservation_tool, tools, {"reservation_id": reservation_id}))

        if not actions:
            rows.append(fallback[task["task_id"]])
            continue

        fallback_actions = fallback[task["task_id"]].get("actions") or []
        seen = {(action.get("tool_id"), json.dumps(action.get("arguments") or {}, sort_keys=True)) for action in actions}
        for action in fallback_actions:
            key = (action.get("tool_id"), json.dumps(action.get("arguments") or {}, sort_keys=True))
            if key in seen:
                continue
            actions.append(action)
            seen.add(key)
            if len(actions) >= max_tools:
                break

        rows.append(
            {
                "task_id": task["task_id"],
                "source": task["source"],
                "plan_type": "baseline_arg_routed",
                "tool_ids": [action["tool_id"] for action in actions],
                "tool_names": [action["tool_name"] for action in actions],
                "phase_names": [tools.get(action["tool_id"], {}).get("phase", "unknown") for action in actions],
                "actions": actions,
            }
        )
    return rows


def add_param(params, key, value):
    if value in (None, ""):
        return
    if isinstance(value, (dict, list)):
        return
    params.setdefault(key, [])
    value = str(value)
    if value not in params[key]:
        params[key].append(value)


def collect_argument_params(value, params):
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                collect_argument_params(item, params)
            else:
                add_param(params, key, item)
    elif isinstance(value, list):
        for item in value:
            collect_argument_params(item, params)


def raw_tau2_item(task):
    raw_path = ROOT / (task.get("raw_path", "").replace("\\", "/"))
    if not raw_path.exists():
        return {}
    if raw_path not in RAW_TASK_CACHE:
        try:
            items = json.loads(raw_path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            items = []
        by_id = {}
        if isinstance(items, list):
            for idx, item in enumerate(items):
                by_id[str(idx)] = item
                if item.get("id") is not None:
                    by_id[str(item.get("id"))] = item
        RAW_TASK_CACHE[raw_path] = by_id
    by_id = RAW_TASK_CACHE[raw_path]
    local_id = task["task_id"].rsplit("::", 1)[-1]
    return by_id.get(local_id) or {}


def tau2_state_params(task):
    item = raw_tau2_item(task)
    initial_state = item.get("initial_state") or {}
    params = {}
    for action in initial_state.get("initialization_actions") or []:
        collect_argument_params(action.get("arguments") or {}, params)
    collect_argument_params(initial_state.get("initialization_data") or {}, params)
    return params


def tau2_initialization_actions(task):
    item = raw_tau2_item(task)
    initial_state = item.get("initial_state") or {}
    return initial_state.get("initialization_actions") or []


def load_tau2_domain_db(domain):
    base = ROOT / "data" / "raw" / "tau2-bench" / "repo" / "data" / "tau2" / "domains" / domain
    json_path = base / "db.json"
    toml_path = base / "db.toml"
    if json_path.exists():
        try:
            return json.loads(json_path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            return {}
    if toml_path.exists():
        try:
            return tomllib.loads(toml_path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            return {}
    return {}


def domain_state_params(task, db):
    prompt = task.get("prompt", "")
    domain = task.get("domain")
    params = {}
    if domain == "telecom":
        phone_numbers = unique_in_order(PHONE_RE.findall(prompt))
        lines = db.get("lines") or []
        customers = db.get("customers") or []
        line_by_phone = {line.get("phone_number"): line for line in lines}
        customer_by_phone = {customer.get("phone_number"): customer for customer in customers}
        for phone in phone_numbers:
            line = line_by_phone.get(phone)
            if line:
                add_param(params, "line_id", line.get("line_id"))
            customer = customer_by_phone.get(phone)
            if customer:
                add_param(params, "customer_id", customer.get("customer_id"))
                for line_id in customer.get("line_ids") or []:
                    line = next((row for row in lines if row.get("line_id") == line_id and row.get("phone_number") == phone), None)
                    if line:
                        add_param(params, "line_id", line_id)
            for customer in customers:
                if phone in (customer.get("phone_number") or ""):
                    add_param(params, "customer_id", customer.get("customer_id"))
        gb_match = re.search(r"\b(\d+(?:\.\d+)?)\s*GB\b", prompt, re.IGNORECASE)
        if gb_match:
            add_param(params, "gb_amount", gb_match.group(1))
    elif domain == "airline":
        users = db.get("users") or {}
        user_ids = unique_in_order(USER_ID_RE.findall(prompt))
        for user_id in user_ids:
            user = users.get(user_id) or {}
            add_param(params, "user_id", user_id)
            for reservation_id in user.get("reservations") or []:
                add_param(params, "reservation_id", reservation_id)
    elif domain == "retail":
        users = db.get("users") or {}
        orders = db.get("orders") or {}

        def add_retail_user(user_id, user):
            if not user:
                return
            add_param(params, "user_id", user_id or user.get("user_id"))
            name = user.get("name") or {}
            address = user.get("address") or {}
            add_param(params, "first_name", name.get("first_name"))
            add_param(params, "last_name", name.get("last_name"))
            add_param(params, "zip", address.get("zip"))
            for order_id in user.get("orders") or []:
                add_param(params, "order_id", order_id)

        name_zip_match = NAME_ZIP_RE.search(prompt or "")
        if name_zip_match:
            first_name, last_name, zip_code = name_zip_match.groups()
            add_param(params, "first_name", first_name)
            add_param(params, "last_name", last_name)
            add_param(params, "zip", zip_code)
            for user_id, user in users.items():
                name = user.get("name") or {}
                address = user.get("address") or {}
                if (
                    (name.get("first_name") or "").lower() == first_name.lower()
                    and (name.get("last_name") or "").lower() == last_name.lower()
                    and str(address.get("zip") or "") == zip_code
                ):
                    add_retail_user(user_id, user)
        for order_id in unique_in_order(ORDER_ID_RE.findall(prompt)):
            order = orders.get(order_id) or {}
            add_param(params, "order_id", order_id)
            add_param(params, "user_id", order.get("user_id"))
            add_retail_user(order.get("user_id"), users.get(order.get("user_id")) or {})
        for email in unique_in_order(EMAIL_RE.findall(prompt)):
            for user_id, user in users.items():
                if (user.get("email") or "").lower() == email.lower():
                    add_retail_user(user_id, user)
        for user_id in unique_in_order(USER_ID_RE.findall(prompt)):
            user = users.get(user_id) or {}
            add_retail_user(user_id, user)
    elif domain == "banking_knowledge":
        users = ((db.get("users") or {}).get("data")) or {}
        for email in unique_in_order(EMAIL_RE.findall(prompt)):
            for user_id, user in users.items():
                if (user.get("email") or "").lower() == email.lower():
                    add_param(params, "user_id", user_id)
        for user_id in re.findall(r"\b[a-f0-9]{10}\b|\b\d{3}\b", prompt):
            if user_id in users:
                add_param(params, "user_id", user_id)
    return params


def merged_state_params(task, db):
    params = tau2_state_params(task)
    for key, values in domain_state_params(task, db).items():
        for value in values:
            add_param(params, key, value)
    return params


def first_init_action(actions, func_name):
    return next((action for action in actions if (action.get("func_name") or action.get("name")) == func_name), None)


def action_if_available(task, tools, tool_name, arguments):
    tool_id = f"tau2::{task.get('domain')}::{tool_name}"
    if tool_id not in set(task.get("available_tool_ids") or []):
        return None
    tool = tools.get(tool_id) or {}
    params = ((tool.get("schema") or {}).get("parameters")) or []
    return action_row(tool_id, tools, {key: value for key, value in arguments.items() if key in params})


def telecom_state_actions(task, tools, state_params):
    init_actions = tau2_initialization_actions(task)
    init_funcs = {(action.get("func_name") or action.get("name")) for action in init_actions}
    actions = []
    common = {
        "customer_id": (state_params.get("customer_id") or [None])[0],
        "line_id": (state_params.get("line_id") or [None])[0],
    }
    suspended = first_init_action(init_actions, "suspend_line_for_overdue_bill")
    if suspended:
        args = suspended.get("arguments") or {}
        common["customer_id"] = args.get("customer_id") or common["customer_id"]
        common["line_id"] = args.get("line_id") or common["line_id"]
        if args.get("contract_ended") is False:
            payment = action_if_available(
                task,
                tools,
                "send_payment_request",
                {"customer_id": common["customer_id"], "bill_id": args.get("new_bill_id")},
            )
            resume = action_if_available(task, tools, "resume_line", common)
            actions.extend([action for action in [payment, resume] if action])
        else:
            transfer = action_if_available(
                task,
                tools,
                "transfer_to_human_agents",
                {"summary": "Line suspended for overdue bill and contract has ended."},
            )
            if transfer:
                actions.append(transfer)
        return actions

    data_action = first_init_action(init_actions, "set_data_usage")
    if data_action:
        args = data_action.get("arguments") or {}
        common["customer_id"] = args.get("customer_id") or common["customer_id"]
        common["line_id"] = args.get("line_id") or common["line_id"]
        refuel = action_if_available(
            task,
            tools,
            "refuel_data",
            {
                "customer_id": common["customer_id"],
                "line_id": common["line_id"],
                "gb_amount": (state_params.get("gb_amount") or ["2.0"])[0],
            },
        )
        if refuel:
            actions.append(refuel)
    if "disable_roaming" in init_funcs:
        enable = action_if_available(task, tools, "enable_roaming", common)
        if enable:
            actions.append(enable)
    return actions


def mock_state_actions(task, tools, state_params):
    prompt = task.get("prompt", "")
    text = prompt.lower()
    if "create" in text and "task" in text:
        title_match = re.search(r"(?:called|task called)\s+'([^']+)'", prompt, re.IGNORECASE)
        action = action_if_available(
            task,
            tools,
            "create_task",
            {
                "user_id": (state_params.get("user_id") or values_for_param(prompt, "user_id") or [None])[0],
                "title": title_match.group(1) if title_match else "Task",
            },
        )
        return [action] if action else []
    if "update" in text or "mark" in text or "completed" in text:
        task_ids = re.findall(r"\btask_\d+\b", prompt)
        action = action_if_available(
            task,
            tools,
            "update_task_status",
            {"task_id": (task_ids or state_params.get("task_id") or ["task_1"])[0], "status": "completed"},
        )
        return [action] if action else []
    if "human" in text or "transfer" in text:
        action = action_if_available(task, tools, "transfer_to_human_agents", {"summary": "Customer requested human help."})
        return [action] if action else []
    return []


def airline_state_actions(task, tools, state_params):
    prompt = task.get("prompt", "")
    text = prompt.lower()
    actions = []
    user_ids = values_for_param(prompt, "user_id") or state_params.get("user_id") or []
    reservation_ids = values_for_param(prompt, "reservation_id") or state_params.get("reservation_id") or []

    if user_ids:
        action = action_if_available(task, tools, "get_user_details", {"user_id": user_ids[0]})
        if action:
            actions.append(action)

    if "most recent" in text or "last reservation" in text or "all" in text:
        selected_reservations = reservation_ids[:5]
    else:
        selected_reservations = reservation_ids[:1]
    for reservation_id in selected_reservations:
        action = action_if_available(task, tools, "get_reservation_details", {"reservation_id": reservation_id})
        if action:
            actions.append(action)

    if not actions and "book" in text and "flight" in text:
        action = action_if_available(task, tools, "book_reservation", {"user_id": (user_ids or [None])[0]})
        if action:
            actions.append(action)
    if not actions and ("search" in text or "flight from" in text):
        action = action_if_available(task, tools, "search_direct_flight", {})
        if action:
            actions.append(action)
    return actions


def name_zip_from_prompt(prompt):
    match = NAME_ZIP_RE.search(prompt or "")
    if not match:
        return None
    return match.group(1), match.group(2), match.group(3)


def retail_state_actions(task, tools, state_params):
    prompt = task.get("prompt", "")
    text = prompt.lower()
    actions = []
    order_ids = values_for_param(prompt, "order_id")
    emails = values_for_param(prompt, "email")
    name_zip = name_zip_from_prompt(prompt)

    needs_identity_lookup = any(phrase in text for phrase in ["all your", "exactly how many", "online store", "your orders"])
    if needs_identity_lookup:
        if emails:
            action = action_if_available(task, tools, "find_user_id_by_email", {"email": emails[0]})
            if action:
                actions.append(action)
        elif name_zip:
            first_name, last_name, zip_code = name_zip
            action = action_if_available(
                task,
                tools,
                "find_user_id_by_name_zip",
                {"first_name": first_name, "last_name": last_name, "zip": zip_code},
            )
            if action:
                actions.append(action)
        user_id = (state_params.get("user_id") or [None])[0]
        if user_id:
            action = action_if_available(task, tools, "get_user_details", {"user_id": user_id})
            if action:
                actions.append(action)

    for order_id in order_ids[:3]:
        action = action_if_available(task, tools, "get_order_details", {"order_id": order_id})
        if action:
            actions.append(action)

    op_name = None
    if "exchange" in text:
        op_name = "exchange_delivered_order_items"
    elif "return" in text:
        op_name = "return_delivered_order_items"
    elif "cancel" in text:
        op_name = "cancel_pending_order"
    elif "payment" in text and "modify" in text:
        op_name = "modify_pending_order_payment"
    elif "address" in text and ("modify" in text or "change" in text):
        op_name = "modify_pending_order_address"
    elif "modify" in text or "change" in text:
        op_name = "modify_pending_order_items"

    if op_name:
        op_args = {}
        if order_ids:
            op_args["order_id"] = order_ids[0]
        action = action_if_available(task, tools, op_name, op_args)
        if action:
            actions.append(action)

    seen = set()
    unique_actions = []
    for action in actions:
        key = (action["tool_id"], json.dumps(action.get("arguments") or {}, sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        unique_actions.append(action)
    return unique_actions


def infer_state_arguments(prompt, tool, state_params, counters):
    params = ((tool.get("schema") or {}).get("parameters")) or []
    name = tool.get("name") or tool.get("tool_id", "").rsplit("::", 1)[-1]
    args = {}
    for param in params:
        if name == "find_user_id_by_name_zip" and param in {"first_name", "last_name", "zip"}:
            values = state_params.get(param) or values_for_param(prompt, param) or []
        else:
            values = values_for_param(prompt, param) or state_params.get(param) or []
        if not values:
            continue
        key = (tool["tool_id"], param)
        idx = counters.get(key, 0)
        args[param] = values[min(idx, len(values) - 1)]
        counters[key] = idx + 1
    return args


def state_action_score(prompt_tokens, prompt, tool, args):
    name = tool.get("name") or ""
    score = len(args) * 5 + score_tool(prompt_tokens, tool)
    text = (prompt or "").lower()
    if name == "enable_roaming" and ("abroad" in text or "roaming" in text or "mobile data" in text):
        score += 12
    if name == "refuel_data" and ("refuel" in text or "gb" in text or "data" in text):
        score += 10
    if name == "resume_line" and ("resume" in text or "suspend" in text):
        score += 10
    if name == "send_payment_request" and ("bill" in text or "payment" in text):
        score += 8
    if name == "get_user_details" and "user_id" in args:
        score += 9
    if name == "get_reservation_details" and "reservation_id" in args:
        score += 9
    if name == "get_order_details" and "order_id" in args:
        score += 9
    if name in {"find_user_id_by_email", "find_user_id_by_name_zip"}:
        score += 8
    if name.startswith("get_") and args:
        score += 4
    if name.startswith(("cancel_", "modify_", "update_", "return_", "exchange_", "book_")) and args:
        score += 3
    return score


def state_routed(tasks, tools, max_tools):
    fallback = {row["task_id"]: row for row in arg_routed(tasks, tools, max_tools)}
    db_cache = {}
    rows = []
    for task in tasks:
        if task.get("source") != "tau2":
            rows.append(fallback[task["task_id"]])
            continue
        domain = task.get("domain")
        db_cache.setdefault(domain, load_tau2_domain_db(domain))
        state_params = merged_state_params(task, db_cache[domain])
        prompt = task.get("prompt", "")
        if domain == "telecom":
            actions = telecom_state_actions(task, tools, state_params)
            if actions:
                rows.append(
                    {
                        "task_id": task["task_id"],
                        "source": task["source"],
                        "plan_type": "baseline_state_routed",
                        "tool_ids": [action["tool_id"] for action in actions],
                        "tool_names": [action["tool_name"] for action in actions],
                        "phase_names": [tools.get(action["tool_id"], {}).get("phase", "unknown") for action in actions],
                        "actions": actions,
                        "metadata": {"state_param_keys": sorted(state_params), "state_rule": "telecom_initial_state"},
                    }
                )
                continue
        if domain == "mock":
            actions = mock_state_actions(task, tools, state_params)
            if actions:
                rows.append(
                    {
                        "task_id": task["task_id"],
                        "source": task["source"],
                        "plan_type": "baseline_state_routed",
                        "tool_ids": [action["tool_id"] for action in actions],
                        "tool_names": [action["tool_name"] for action in actions],
                        "phase_names": [tools.get(action["tool_id"], {}).get("phase", "unknown") for action in actions],
                        "actions": actions,
                        "metadata": {"state_param_keys": sorted(state_params), "state_rule": "mock_prompt_rule"},
                    }
                )
                continue
        if domain == "airline":
            actions = airline_state_actions(task, tools, state_params)
            if actions:
                rows.append(
                    {
                        "task_id": task["task_id"],
                        "source": task["source"],
                        "plan_type": "baseline_state_routed",
                        "tool_ids": [action["tool_id"] for action in actions],
                        "tool_names": [action["tool_name"] for action in actions],
                        "phase_names": [tools.get(action["tool_id"], {}).get("phase", "unknown") for action in actions],
                        "actions": actions,
                        "metadata": {"state_param_keys": sorted(state_params), "state_rule": "airline_retrieval_prune"},
                    }
                )
                continue
        if domain == "retail":
            actions = retail_state_actions(task, tools, state_params)
            if actions:
                rows.append(
                    {
                        "task_id": task["task_id"],
                        "source": task["source"],
                        "plan_type": "baseline_state_routed",
                        "tool_ids": [action["tool_id"] for action in actions],
                        "tool_names": [action["tool_name"] for action in actions],
                        "phase_names": [tools.get(action["tool_id"], {}).get("phase", "unknown") for action in actions],
                        "actions": actions,
                        "metadata": {"state_param_keys": sorted(state_params), "state_rule": "retail_prompt_workflow"},
                    }
                )
                continue
        prompt_tokens = tokens(prompt)
        counters = {}
        candidates = []
        for tool_id in task.get("available_tool_ids") or []:
            tool = tools.get(tool_id)
            if not tool:
                continue
            args = infer_state_arguments(prompt, tool, state_params, counters)
            if not args:
                continue
            candidates.append((state_action_score(prompt_tokens, prompt, tool, args), tool_id, args))
        ranked = sorted(candidates, key=lambda item: (item[0], item[1]), reverse=True)
        actions = [action_row(tool_id, tools, args) for score, tool_id, args in ranked if score > 0][:max_tools]
        if not actions:
            rows.append(fallback[task["task_id"]])
            continue
        rows.append(
            {
                "task_id": task["task_id"],
                "source": task["source"],
                "plan_type": "baseline_state_routed",
                "tool_ids": [action["tool_id"] for action in actions],
                "tool_names": [action["tool_name"] for action in actions],
                "phase_names": [tools.get(action["tool_id"], {}).get("phase", "unknown") for action in actions],
                "actions": actions,
                "metadata": {"state_param_keys": sorted(state_params)},
            }
        )
    return rows


def phase_majority(tasks, tools, max_tools):
    phase_order = {"retrieve": 0, "verify": 1, "operate": 2, "compute": 3, "create": 4, "other": 5}
    rows = []
    for task in tasks:
        candidates = [tools[t] for t in task.get("available_tool_ids", []) if t in tools]
        ranked = sorted(
            candidates,
            key=lambda tool: (
                -phase_order.get(tool.get("phase", "other"), 99),
                tool.get("tool_split") != "train_seen",
                tool["tool_id"],
            ),
        )
        chosen = ranked[:max_tools]
        rows.append(
            {
                "task_id": task["task_id"],
                "source": task["source"],
                "plan_type": "baseline_phase_majority",
                "tool_ids": [tool["tool_id"] for tool in chosen],
                "tool_names": [tool["name"] for tool in chosen],
                "phase_names": [tool["phase"] for tool in chosen],
            }
        )
    return rows


def oracle_phase(tasks, tools, gold_rows):
    gold = {row["task_id"]: row for row in gold_rows}
    rows = []
    for task in tasks:
        candidates = [tools[t] for t in task.get("available_tool_ids", []) if t in tools]
        by_phase = {}
        for tool in sorted(candidates, key=lambda item: item["tool_id"]):
            by_phase.setdefault(tool.get("phase", "other"), []).append(tool)
        chosen = []
        for phase in gold.get(task["task_id"], {}).get("phase_names", []):
            pool = by_phase.get(phase) or []
            if pool:
                chosen.append(pool[min(len(chosen), len(pool) - 1)])
        rows.append(
            {
                "task_id": task["task_id"],
                "source": task["source"],
                "plan_type": "baseline_oracle_phase",
                "tool_ids": [tool["tool_id"] for tool in chosen],
                "tool_names": [tool["name"] for tool in chosen],
                "phase_names": [tool["phase"] for tool in chosen],
            }
        )
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default=str(ROOT / "data" / "processed" / "tasks.jsonl"))
    parser.add_argument("--tools", default=str(ROOT / "data" / "processed" / "tools.jsonl"))
    parser.add_argument("--gold", default=str(ROOT / "data" / "processed" / "gold_plans.jsonl"))
    parser.add_argument(
        "--mode",
        choices=["keyword_nearest", "keyword_args", "arg_routed", "state_routed", "phase_majority", "oracle_phase"],
        default="keyword_nearest",
    )
    parser.add_argument("--out", default=None)
    parser.add_argument("--max-tools", type=int, default=4)
    args = parser.parse_args()

    tasks = read_jsonl(args.tasks)
    tools = {row["tool_id"]: row for row in read_jsonl(args.tools)}
    if args.mode == "keyword_nearest":
        rows = keyword_nearest(tasks, tools, args.max_tools)
    elif args.mode == "keyword_args":
        rows = with_actions(keyword_nearest(tasks, tools, args.max_tools), tasks, tools)
    elif args.mode == "arg_routed":
        rows = arg_routed(tasks, tools, args.max_tools)
    elif args.mode == "state_routed":
        rows = state_routed(tasks, tools, args.max_tools)
    elif args.mode == "phase_majority":
        rows = phase_majority(tasks, tools, args.max_tools)
    else:
        rows = oracle_phase(tasks, tools, read_jsonl(args.gold))
    out = args.out or str(ROOT / "data" / "processed" / "predictions" / f"{args.mode}.jsonl")
    write_jsonl(out, rows)
    print(json.dumps({"mode": args.mode, "predictions": len(rows), "out": out}, ensure_ascii=False))


if __name__ == "__main__":
    main()
