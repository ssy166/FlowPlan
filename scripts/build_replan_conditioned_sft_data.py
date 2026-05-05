import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPLAN_DIR = ROOT / "data" / "processed" / "closed_loop"
DEFAULT_OUT_DIR = ROOT / "data" / "processed" / "fm_replan_sft_next"


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
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


def shorten(text, max_chars):
    text = " ".join(str(text or "").split())
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def tool_parameters(tool):
    params = tool.get("parameters")
    if params:
        return params
    schema = tool.get("schema") or {}
    return schema.get("parameters") or []


def compact_tool(tool, max_desc_chars):
    out = {
        "tool_id": tool.get("tool_id"),
        "name": tool.get("name") or str(tool.get("tool_id", "")).rsplit("::", 1)[-1],
        "phase": tool.get("phase"),
        "parameters": tool_parameters(tool),
    }
    if max_desc_chars:
        out["description"] = shorten(tool.get("description") or "", max_desc_chars)
    return out


def compact_action(action, include_tool_id=True):
    if not action:
        return None
    out = {
        "tool_name": action.get("tool_name") or str(action.get("tool_id", "")).rsplit("::", 1)[-1],
        "arguments": action.get("arguments") or {},
    }
    if include_tool_id and action.get("tool_id"):
        out["tool_id"] = action["tool_id"]
    return out


def summarize_result(result):
    if isinstance(result, list):
        return [summarize_result(item) for item in result[:8]]
    if not isinstance(result, dict):
        return result
    if "order_id" in result and ("items" in result or "payment_history" in result or "status" in result):
        items = result.get("items") or []
        payments = result.get("payment_history") or []
        fulfillments = result.get("fulfillments") or []
        return {
            "order_id": result.get("order_id"),
            "user_id": result.get("user_id"),
            "status": result.get("status"),
            "item_ids": [item.get("item_id") for item in items if isinstance(item, dict) and item.get("item_id")],
            "product_ids": [item.get("product_id") for item in items if isinstance(item, dict) and item.get("product_id")],
            "item_names": [item.get("name") for item in items if isinstance(item, dict) and item.get("name")],
            "payment_method_ids": [
                item.get("payment_method_id") for item in payments if isinstance(item, dict) and item.get("payment_method_id")
            ],
            "fulfillment_item_ids": [
                item_id
                for fulfillment in fulfillments
                if isinstance(fulfillment, dict)
                for item_id in (fulfillment.get("item_ids") or [])
            ],
        }
    if "user_id" in result and ("orders" in result or "payment_methods" in result):
        payment_methods = result.get("payment_methods") or {}
        return {
            "user_id": result.get("user_id"),
            "email": result.get("email"),
            "orders": result.get("orders") or [],
            "payment_method_ids": list(payment_methods) if isinstance(payment_methods, dict) else [],
            "address": result.get("address"),
        }
    if "product_id" in result and "variants" in result:
        variants = result.get("variants") or {}
        return {
            "product_id": result.get("product_id"),
            "name": result.get("name"),
            "variant_item_ids": list(variants)[:24] if isinstance(variants, dict) else [],
        }
    return result


def compact_result(result, max_chars, summarize=False):
    if not max_chars or result in (None, ""):
        return summarize_result(result) if summarize else result
    if summarize:
        result = summarize_result(result)
    text = json.dumps(result, ensure_ascii=False, sort_keys=True, default=str)
    if len(text) <= max_chars:
        return result
    return {"_truncated_json": shorten(text, max_chars)}


def compact_execution_step(step, args):
    execution = step.get("execution") or {}
    return {
        "step_idx": step.get("step_idx"),
        "predicted_action": compact_action(step.get("predicted_action"), include_tool_id=True),
        "execution": {
            "ok": execution.get("ok"),
            "tool_name": execution.get("tool_name"),
            "arguments": execution.get("arguments") or {},
            "dropped_arguments": execution.get("dropped_arguments") or {},
            "error_type": execution.get("error_type"),
            "error": shorten(execution.get("error") or "", 300),
            "result": compact_result(execution.get("result"), args.max_result_chars, args.summarize_results),
        },
        "tool_match": step.get("tool_match"),
    }


def compact_execution_step_v2(step):
    execution = step.get("execution") or {}
    result = execution.get("result")
    return {
        "step_idx": step.get("step_idx"),
        "action": compact_action(step.get("predicted_action"), include_tool_id=False),
        "ok": execution.get("ok"),
        "tool_name": execution.get("tool_name"),
        "arguments": execution.get("arguments") or {},
        "error_type": execution.get("error_type"),
        "error": shorten(execution.get("error") or "", 160),
        "result_summary": summarize_result(result),
        "tool_match": step.get("tool_match"),
    }


def compact_feedback_v2(item):
    if not isinstance(item, dict):
        return item
    result = item.get("result")
    return {
        "ok": item.get("ok"),
        "tool_name": item.get("tool_name") or item.get("func_name"),
        "arguments": item.get("arguments") or {},
        "error_type": item.get("error_type"),
        "result_summary": summarize_result(result),
    }


ORDER_ID_RE = re.compile(r"#W\d+")


def collect_key_values(obj, keys):
    found = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in keys and value not in (None, ""):
                found.append(value)
            found.extend(collect_key_values(value, keys))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(collect_key_values(item, keys))
    return found


def collect_order_ids(obj):
    values = collect_key_values(obj, {"order_id", "order_ids"})
    text = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
    values.extend(ORDER_ID_RE.findall(text))
    out = []
    for value in values:
        if isinstance(value, list):
            out.extend(str(item) for item in value)
        else:
            out.append(str(value))
    return out


def unique_keep_order(values):
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


def action_name(action, execution):
    action = action or {}
    execution = execution or {}
    return (
        action.get("tool_name")
        or str(action.get("tool_id") or "").rsplit("::", 1)[-1]
        or execution.get("tool_name")
        or execution.get("func_name")
        or ""
    )


def update_retail_grounding(progress, action, execution):
    if not execution or not execution.get("ok", False):
        return
    name = action_name(action, execution)
    blob = {"arguments": execution.get("arguments") or {}, "result": execution.get("result")}
    user_ids = collect_key_values(blob, {"user_id", "user_ids"})
    order_ids = collect_order_ids(blob)
    item_ids = collect_key_values(blob, {"item_id", "item_ids", "order_item_id", "order_item_ids"})
    payment_ids = collect_key_values(blob, {"payment_method_id", "payment_method_ids"})
    statuses = collect_key_values(blob, {"status", "order_status"})
    progress["resolved_user_ids"].extend(str(value) for value in user_ids)
    progress["resolved_order_ids"].extend(str(value) for value in order_ids)
    for value in item_ids:
        if isinstance(value, list):
            progress["resolved_order_item_ids"].extend(str(item) for item in value)
        else:
            progress["resolved_order_item_ids"].append(str(value))
    for value in payment_ids:
        if isinstance(value, list):
            progress["resolved_payment_method_ids"].extend(str(item) for item in value)
        else:
            progress["resolved_payment_method_ids"].append(str(value))
    if statuses:
        progress["resolved_order_status"] = str(statuses[-1])
    if name.startswith(("find_user_id", "get_user_details", "get_order_details", "get_product_details", "get_item_details")):
        progress["last_successful_lookup_tool"] = name


def retail_grounding_progress(record):
    if record.get("domain") != "retail":
        return None
    progress = {
        "resolved_user_ids": [],
        "resolved_order_ids": [],
        "resolved_order_status": None,
        "resolved_order_item_ids": [],
        "resolved_payment_method_ids": [],
        "last_successful_lookup_tool": None,
        "lookup_completed": False,
        "order_grounded": False,
        "operation_eligibility": [],
    }
    for item in record.get("initialization_feedback") or []:
        update_retail_grounding(progress, {"tool_name": item.get("tool_name") or item.get("func_name")}, item)
    for step in record.get("executed_prefix") or []:
        update_retail_grounding(progress, step.get("predicted_action") or {}, step.get("execution") or {})
    progress["resolved_user_ids"] = unique_keep_order(progress["resolved_user_ids"])
    progress["resolved_order_ids"] = unique_keep_order(progress["resolved_order_ids"])
    progress["resolved_order_item_ids"] = unique_keep_order(progress["resolved_order_item_ids"])
    progress["resolved_payment_method_ids"] = unique_keep_order(progress["resolved_payment_method_ids"])
    progress["lookup_completed"] = bool(progress["resolved_user_ids"])
    progress["order_grounded"] = bool(progress["resolved_order_ids"])
    status = (progress.get("resolved_order_status") or "").lower()
    if status == "pending":
        progress["operation_eligibility"] = [
            "cancel_pending_order",
            "modify_pending_order_address",
            "modify_pending_order_items",
            "modify_pending_order_payment",
        ]
    elif status == "delivered":
        progress["operation_eligibility"] = [
            "return_delivered_order_items",
            "exchange_delivered_order_items",
        ]
    return progress


def load_tasks(path):
    return {row["task_id"]: row for row in read_jsonl(path)}


def load_tools(path):
    return {row["tool_id"]: row for row in read_jsonl(path)}


def tools_for_task(task_id, tasks, tools, max_desc_chars):
    task = tasks.get(task_id) or {}
    out = []
    for tool_id in task.get("available_tool_ids") or []:
        if tool_id in tools:
            out.append(compact_tool(tools[tool_id], max_desc_chars))
    return out


def target_for_record(record, include_tool_id):
    actions = record.get("target_remaining_actions") or []
    if not actions:
        return {"stop": True, "actions": []}
    return {"stop": False, "actions": [compact_action(actions[0], include_tool_id=include_tool_id)]}


def build_user_content(record, available_tools, args):
    reason = record.get("replan_reason") or {}
    observable_reason = {
        "execution_ok": reason.get("execution_ok"),
        "error_type": reason.get("error_type"),
        "has_predicted_action_at_step": reason.get("has_predicted_action_at_step"),
        "tool_match": reason.get("tool_match"),
    }
    fm_condition = {
        "kind": "external_soft_prefix_optional",
        "note": "This replan SFT pack is valid for no-prefix LLM training now. Encoder-derived FM replan latents can be attached later with the same record_id.",
    }
    state = {
        "task_id": record.get("task_id"),
        "record_id": f"{record.get('task_id')}::replan::{record.get('replan_step_idx')}",
        "source": record.get("source"),
        "domain": record.get("domain"),
        "replan_step_idx": record.get("replan_step_idx"),
        "user_prompt": record.get("prompt") or "",
        "available_tools": available_tools,
        "assistant_initialization_feedback": record.get("initialization_feedback") or [],
        "executed_prefix": [compact_execution_step(step, args) for step in record.get("executed_prefix") or []],
        "original_pred_remaining_tool_ids": record.get("original_pred_remaining_tool_ids") or [],
        "replan_reason": observable_reason,
        "fm_condition": fm_condition,
    }
    if args.include_grounding_progress:
        progress = retail_grounding_progress(record)
        if progress is not None:
            state["grounding_progress"] = progress
    instruction = [
        "You are a feedback-conditioned tool execution model.",
        "Use the task, available tools, executed prefix, tool results, and replan reason.",
        "Emit exactly one next tool call, or stop if no more tool call should be made.",
        "Emit strict JSON only with this schema:",
        '{"stop": false, "actions": [{"tool_name": "...", "arguments": {}}]}',
        "If the workflow should stop now, emit:",
        '{"stop": true, "actions": []}',
        "Do not call tools that are absent from available_tools.",
    ]
    return "\n".join(instruction + ["", "Feedback-conditioned state:", json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True, default=str)])


def build_user_content_compact_v2(record, available_tools, args):
    reason = record.get("replan_reason") or {}
    observable_reason = {
        "execution_ok": reason.get("execution_ok"),
        "error_type": reason.get("error_type"),
        "tool_match": reason.get("tool_match"),
        "gold_prefix": reason.get("gold_prefix"),
    }
    remaining_prior_tool_names = [
        str(tool_id).rsplit("::", 1)[-1] for tool_id in record.get("original_pred_remaining_tool_ids") or []
    ]
    state = {
        "task_id": record.get("task_id"),
        "record_id": f"{record.get('task_id')}::replan::{record.get('replan_step_idx')}",
        "source": record.get("source"),
        "domain": record.get("domain"),
        "step": record.get("replan_step_idx"),
        "task": record.get("prompt") or "",
        "available_tools": available_tools,
        "init_feedback": [compact_feedback_v2(item) for item in record.get("initialization_feedback") or []],
        "executed_prefix": [compact_execution_step_v2(step) for step in record.get("executed_prefix") or []],
        "replan_reason": observable_reason,
    }
    if not args.omit_remaining_prior:
        state["remaining_prior_tool_names"] = remaining_prior_tool_names
    if args.include_grounding_progress:
        progress = retail_grounding_progress(record)
        if progress is not None:
            state["grounding_progress"] = progress
    instruction = [
        "You are a feedback-conditioned tool execution model.",
        "Given the compact state, output the next tool call as strict JSON.",
        'Use schema {"stop": false, "actions": [{"tool_name": "...", "arguments": {}}]} or {"stop": true, "actions": []}.',
        "Choose only from available_tools. Use grounding_progress and result_summary for ids/status/items/payments.",
    ]
    return "\n".join(instruction + ["", "Compact state:", json.dumps(state, ensure_ascii=False, separators=(",", ":"), sort_keys=True, default=str)])


def build_example(record, split, tasks, tools, args):
    task_id = record.get("task_id")
    target = target_for_record(record, include_tool_id=args.include_tool_id_in_target)
    record_id = f"{task_id}::replan::{record.get('replan_step_idx')}"
    return {
        "id": record_id,
        "messages": [
            {
                "role": "user",
                "content": (
                    build_user_content_compact_v2(record, tools_for_task(task_id, tasks, tools, args.max_desc_chars), args)
                    if args.state_format == "compact_v2"
                    else build_user_content(record, tools_for_task(task_id, tasks, tools, args.max_desc_chars), args)
                ),
            },
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False, sort_keys=True)},
        ],
        "metadata": {
            "task_id": task_id,
            "record_id": record_id,
            "source": record.get("source"),
            "domain": record.get("domain"),
            "split": split,
            "replan_step_idx": record.get("replan_step_idx"),
            "target_remaining_count": len(record.get("target_remaining_actions") or []),
            "stop": target["stop"],
            "available_tool_count": len(tools_for_task(task_id, tasks, tools, args.max_desc_chars)),
            "conditioning": "feedback_replan_next_action_v1",
            "state_format": args.state_format,
            "grounding_progress": bool(args.include_grounding_progress),
            "prediction_type": record.get("prediction_type"),
            "replan_reason": record.get("replan_reason") or {},
        },
    }


def interleave_by_domain(records):
    buckets = {}
    order = []
    for record in records:
        domain = record.get("domain") or "unknown"
        if domain not in buckets:
            buckets[domain] = []
            order.append(domain)
        buckets[domain].append(record)
    out = []
    while any(buckets.values()):
        for domain in order:
            if buckets[domain]:
                out.append(buckets[domain].pop(0))
    return out


def validate_rows(rows):
    stats = {
        "rows": len(rows),
        "json_parse_errors": 0,
        "bad_schema": 0,
        "stop_rows": 0,
        "action_rows": 0,
        "max_user_chars": 0,
        "max_target_chars": 0,
    }
    errors = []
    for row in rows:
        stats["max_user_chars"] = max(stats["max_user_chars"], len(row["messages"][0]["content"]))
        stats["max_target_chars"] = max(stats["max_target_chars"], len(row["messages"][-1]["content"]))
        try:
            target = json.loads(row["messages"][-1]["content"])
        except Exception as exc:
            stats["json_parse_errors"] += 1
            errors.append(f"{row.get('id')}: target JSON error {exc}")
            continue
        if not isinstance(target, dict) or not isinstance(target.get("stop"), bool) or not isinstance(target.get("actions"), list):
            stats["bad_schema"] += 1
            errors.append(f"{row.get('id')}: bad target schema")
            continue
        if target["stop"]:
            stats["stop_rows"] += 1
        else:
            stats["action_rows"] += 1
        for action in target["actions"]:
            if not isinstance(action, dict) or not isinstance(action.get("tool_name"), str) or not isinstance(action.get("arguments"), dict):
                stats["bad_schema"] += 1
                errors.append(f"{row.get('id')}: bad action schema")
                break
    return stats, errors


def write_readme(out_dir, manifest):
    lines = [
        "# Feedback Replan Next-Action SFT Data",
        "",
        "Chat-style SFT data derived from closed-loop replan records.",
        "",
        "Each row conditions on task prompt, available tools, executed prefix, tool results, and replan reason.",
        'Target schema: `{"stop": bool, "actions": [{"tool_name": "...", "arguments": {}}]}`.',
        "Target mode: next action only, with explicit stop rows.",
        "",
        "This pack is currently most useful for no-prefix executor SFT. Encoder-derived replan latents can be attached later.",
        "",
        "## Counts",
        "",
    ]
    for split, info in manifest["splits"].items():
        lines.append(f"- {split}: {info['rows']} rows, stop {info['stop_rows']}, action {info['action_rows']}")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Build next-action SFT data from closed-loop replan records.")
    parser.add_argument("--replan-dir", default=str(DEFAULT_REPLAN_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--tasks", default=str(ROOT / "data" / "processed" / "tasks.jsonl"))
    parser.add_argument("--tools", default=str(ROOT / "data" / "processed" / "tools.jsonl"))
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--domains", nargs="+", default=["telecom", "retail"])
    parser.add_argument("--max-desc-chars", type=int, default=240)
    parser.add_argument("--max-result-chars", type=int, default=0)
    parser.add_argument("--summarize-results", action="store_true")
    parser.add_argument("--state-format", choices=["full", "compact_v2"], default="full")
    parser.add_argument("--max-rows-per-split", type=int, default=0)
    parser.add_argument("--include-tool-id-in-target", action="store_true")
    parser.add_argument("--include-grounding-progress", action="store_true")
    parser.add_argument("--interleave-domains-before-limit", action="store_true")
    parser.add_argument(
        "--omit-remaining-prior",
        action="store_true",
        help="Do not include remaining_prior_tool_names in compact state; useful for no-leak supervised baselines.",
    )
    args = parser.parse_args()

    replan_dir = Path(args.replan_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(args.tasks)
    tools = load_tools(args.tools)

    manifest = {
        "format": "feedback_replan_next_action_sft_v1",
        "replan_dir": str(replan_dir),
        "domains": args.domains,
        "include_tool_id_in_target": args.include_tool_id_in_target,
        "max_desc_chars": args.max_desc_chars,
        "max_result_chars": args.max_result_chars,
        "summarize_results": args.summarize_results,
        "state_format": args.state_format,
        "omit_remaining_prior": args.omit_remaining_prior,
        "splits": {},
    }
    all_errors = []
    for split in args.splits:
        records = []
        domain_counts = {}
        for domain in args.domains:
            domain_records = read_jsonl(replan_dir / f"replan_records.{split}.{domain}.jsonl")
            records.extend(domain_records)
            domain_counts[domain] = len(domain_records)
        if args.interleave_domains_before_limit:
            records = interleave_by_domain(records)
        if args.max_rows_per_split:
            records = records[: args.max_rows_per_split]
        rows = [build_example(record, split, tasks, tools, args) for record in records]
        stats, errors = validate_rows(rows)
        if errors:
            all_errors.extend(errors[:20])
        write_jsonl(out_dir / f"{split}.jsonl", rows)
        source_counts = {}
        out_domain_counts = {}
        for row in rows:
            md = row["metadata"]
            source_counts[md["source"]] = source_counts.get(md["source"], 0) + 1
            out_domain_counts[md["domain"]] = out_domain_counts.get(md["domain"], 0) + 1
        manifest["splits"][split] = {
            **stats,
            "path": str((out_dir / f"{split}.jsonl").relative_to(ROOT) if out_dir.is_relative_to(ROOT) else out_dir / f"{split}.jsonl"),
            "input_domain_counts": domain_counts,
            "source_counts": source_counts,
            "domain_counts": out_domain_counts,
        }
    if all_errors:
        raise ValueError("Validation failed: " + "; ".join(all_errors[:10]))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_readme(out_dir, manifest)
    print(json.dumps({"out_dir": str(out_dir), "splits": {k: v["rows"] for k, v in manifest["splits"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
