import argparse
import ast
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


OPERATION_TOOLS = {
    "cancel_pending_order",
    "exchange_delivered_order_items",
    "modify_pending_order_address",
    "modify_pending_order_items",
    "modify_pending_order_payment",
    "modify_user_address",
    "return_delivered_order_items",
}


LOOKUP_TOOLS = {
    "find_user_id_by_name_zip",
    "find_user_id_by_email",
    "get_order_details",
    "get_product_details",
    "get_item_details",
    "get_user_details",
    "list_all_product_types",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def tool_args_from_source(path):
    module = ast.parse(Path(path).read_text(encoding="utf-8"))
    out = {}
    for node in module.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for fn in node.body:
            if not isinstance(fn, ast.FunctionDef):
                continue
            decorators = [ast.unparse(dec) for dec in fn.decorator_list]
            if not any("is_tool" in dec for dec in decorators):
                continue
            out[fn.name] = [arg.arg for arg in fn.args.args[1:]]
    return out


def order_status_counts(db):
    counts = Counter()
    orders = db.get("orders") or {}
    for order in orders.values():
        counts[str(order.get("status") or "unknown")] += 1
    return dict(sorted(counts.items()))


def action_counts(tasks):
    counts = Counter()
    task_final_action = Counter()
    operation_tasks = Counter()
    lookup_then_operation = 0
    for task in tasks:
        actions = ((task.get("evaluation_criteria") or {}).get("actions") or [])
        names = [action.get("name") for action in actions if action.get("name")]
        counts.update(names)
        if names:
            task_final_action[names[-1]] += 1
        if any(name in OPERATION_TOOLS for name in names):
            operation_tasks["tasks_with_operation"] += 1
            if any(name in LOOKUP_TOOLS for name in names) and names[-1] in OPERATION_TOOLS:
                lookup_then_operation += 1
    return {
        "gold_action_counts": dict(counts.most_common()),
        "final_action_counts": dict(task_final_action.most_common()),
        "tasks_with_operation": operation_tasks["tasks_with_operation"],
        "lookup_then_operation_tasks": lookup_then_operation,
    }


def split_counts(split_tasks):
    return {split: len(ids) for split, ids in sorted(split_tasks.items()) if isinstance(ids, list)}


def main():
    parser = argparse.ArgumentParser(description="Summarize tau2 retail raw sources for targeted dataset expansion.")
    parser.add_argument(
        "--tau2-root",
        default=str(ROOT / "data" / "raw" / "tau2-bench" / "repo"),
        help="Path to the tau2-bench repo root.",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    tau2_root = Path(args.tau2_root)
    data_dir = tau2_root / "data" / "tau2" / "domains" / "retail"
    src_dir = tau2_root / "src" / "tau2" / "domains" / "retail"
    tasks_path = data_dir / "tasks.json"
    split_path = data_dir / "split_tasks.json"
    db_path = data_dir / "db.json"
    policy_path = data_dir / "policy.md"
    tools_path = src_dir / "tools.py"
    data_model_path = src_dir / "data_model.py"

    tasks = read_json(tasks_path)
    split_tasks = read_json(split_path)
    db = read_json(db_path)
    tools = tool_args_from_source(tools_path)
    action_summary = action_counts(tasks)

    products = db.get("products") or {}
    users = db.get("users") or {}
    orders = db.get("orders") or {}
    item_count = 0
    for product in products.values():
        item_count += len(product.get("variants") or {})

    summary = {
        "source_name": "tau2-bench retail",
        "purpose": "Targeted expansion for retail lookup-vs-operation replan failures.",
        "paths": {
            "tasks": str(tasks_path),
            "split_tasks": str(split_path),
            "db": str(db_path),
            "policy": str(policy_path),
            "tools": str(tools_path),
            "data_model": str(data_model_path),
            "task_issues_dir": str(data_dir / "task_issues"),
            "final_result_dir": str(tau2_root / "data" / "tau2" / "results" / "final"),
        },
        "counts": {
            "tasks": len(tasks),
            "split_tasks": split_counts(split_tasks),
            "products": len(products),
            "items": item_count,
            "users": len(users),
            "orders": len(orders),
            "order_status": order_status_counts(db),
            **action_summary,
        },
        "tool_args": {name: tools[name] for name in sorted(tools)},
        "recommended_expansion_targets": {
            "operation_tools": sorted(OPERATION_TOOLS & set(tools)),
            "lookup_tools": sorted(LOOKUP_TOOLS & set(tools)),
            "state_features": [
                "resolved_user_id",
                "resolved_order_id",
                "resolved_order_status",
                "resolved_order_item_ids",
                "resolved_payment_method_ids",
                "last_successful_lookup_tool",
                "operation_eligibility",
            ],
        },
        "expansion_protocol": [
            "Keep the original tau2 split ids as provenance; generated variants should inherit source_task_id.",
            "Generate new replan rows only from executable DB-backed states.",
            "Preserve gold action arguments from DB/policy-valid tool calls.",
            "Label whether lookup is incomplete or operation-ready before adding operation hints.",
            "Evaluate row-level tool EM and real-tool execution success separately.",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
