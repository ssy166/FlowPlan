import argparse
import inspect
import json
import re
import sys
import types
from pathlib import Path
from typing import Any, get_args, get_origin


ROOT = Path(__file__).resolve().parents[1]
TAU2_SRC = ROOT / "data" / "raw" / "tau2-bench" / "repo" / "src"
TELECOM_DB_PATH = (
    ROOT / "data" / "raw" / "tau2-bench" / "repo" / "data" / "tau2" / "domains" / "telecom" / "db.toml"
)
RETAIL_DB_PATH = (
    ROOT / "data" / "raw" / "tau2-bench" / "repo" / "data" / "tau2" / "domains" / "retail" / "db.json"
)
USER_ID_RE = re.compile(r"\b[a-z]+_[a-z]+_\d+\b")
ORDER_ID_RE = re.compile(r"#[A-Z0-9]{8}\b")

sys.path.insert(0, str(ROOT / "scripts"))
from make_baseline_predictions import raw_tau2_item  # noqa: E402


def install_light_tau2_package() -> None:
    """Avoid importing tau2.__init__, which pulls in the full runner/LLM stack."""
    if "tau2" in sys.modules:
        return
    tau2 = types.ModuleType("tau2")
    tau2.__path__ = [str(TAU2_SRC / "tau2")]
    sys.modules["tau2"] = tau2


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


def load_domain_tools(domain: str):
    install_light_tau2_package()
    if domain == "telecom":
        from tau2.domains.telecom.data_model import TelecomDB
        from tau2.domains.telecom.tools import TelecomTools

        db = TelecomDB.load(TELECOM_DB_PATH)
        return db, TelecomTools(db)
    if domain == "retail":
        from tau2.domains.retail.data_model import RetailDB
        from tau2.domains.retail.tools import RetailTools

        db = RetailDB.load(RETAIL_DB_PATH)
        return db, RetailTools(db)
    raise ValueError(f"Unsupported domain: {domain}")


def serialize(value: Any, depth: int = 0) -> Any:
    if depth > 5:
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        return serialize(value.model_dump(mode="json"), depth + 1)
    if isinstance(value, dict):
        return {str(k): serialize(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize(v, depth + 1) for v in value]
    return str(value)


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def coerce_value(value: Any, annotation: Any) -> Any:
    if value is None or annotation is inspect.Signature.empty:
        return value
    origin = get_origin(annotation)
    if origin is not None:
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if origin in {list, tuple}:
            item_annotation = args[0] if args else inspect.Signature.empty
            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                    value = parsed if isinstance(parsed, list) else [value]
                except Exception:
                    value = [value]
            if not isinstance(value, (list, tuple)):
                value = [value]
            return [coerce_value(item, item_annotation) for item in value]
        if args:
            return coerce_value(value, args[0])
    if annotation in {str, "str"}:
        return str(value)
    if annotation in {int, "int"}:
        return int(value)
    if annotation in {float, "float"}:
        return float(value)
    if annotation in {bool, "bool"}:
        return coerce_bool(value)
    return value


def callable_name(action: dict[str, Any]) -> str:
    tool_id = action.get("tool_id") or ""
    return action.get("func_name") or action.get("name") or action.get("tool_name") or tool_id.rsplit("::", 1)[-1]


def execute_action(tools: Any, action: dict[str, Any]) -> dict[str, Any]:
    name = callable_name(action)
    if not name or not hasattr(tools, name):
        return {
            "ok": False,
            "tool_name": name,
            "error_type": "UnknownTool",
            "error": f"TelecomTools has no callable named {name}",
        }
    fn = getattr(tools, name)
    signature = inspect.signature(fn)
    raw_args = action.get("arguments") or {}
    kwargs = {}
    dropped = {}
    for key, value in raw_args.items():
        if key not in signature.parameters:
            dropped[key] = value
            continue
        kwargs[key] = coerce_value(value, signature.parameters[key].annotation)
    try:
        result = fn(**kwargs)
        return {
            "ok": True,
            "tool_name": name,
            "arguments": serialize(kwargs),
            "dropped_arguments": serialize(dropped),
            "result": serialize(result),
        }
    except Exception as exc:
        return {
            "ok": False,
            "tool_name": name,
            "arguments": serialize(kwargs),
            "dropped_arguments": serialize(dropped),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def add_unique(values: list[str], value: Any) -> None:
    if value in (None, ""):
        return
    value = str(value)
    if value not in values:
        values.append(value)


def collect_grounding_values(value: Any, context: dict[str, Any]) -> None:
    if value is None:
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "user_id":
                add_unique(context.setdefault("user_ids", []), item)
            elif key == "order_id":
                add_unique(context.setdefault("order_ids", []), item)
            collect_grounding_values(item, context)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            collect_grounding_values(item, context)
        return
    if isinstance(value, str):
        for user_id in USER_ID_RE.findall(value):
            add_unique(context.setdefault("user_ids", []), user_id)
        for order_id in ORDER_ID_RE.findall(value):
            add_unique(context.setdefault("order_ids", []), order_id)


def action_with_result_grounding(action: dict[str, Any] | None, context: dict[str, Any]) -> dict[str, Any] | None:
    if not action:
        return action
    grounded = dict(action)
    args = dict(grounded.get("arguments") or {})
    filled = {}
    name = callable_name(grounded)

    if name == "get_user_details" and not args.get("user_id") and context.get("user_ids"):
        args["user_id"] = context["user_ids"][0]
        filled["user_id"] = "previous_tool_result"

    needs_order_id = (
        name == "get_order_details"
        or name.startswith("cancel_pending_order")
        or name.startswith("modify_pending_order")
        or name.startswith("return_delivered_order")
        or name.startswith("exchange_delivered_order")
    )
    if needs_order_id and not args.get("order_id") and context.get("order_ids"):
        idx = context.setdefault("order_cursor", 0)
        order_ids = context["order_ids"]
        args["order_id"] = order_ids[min(idx, len(order_ids) - 1)]
        context["order_cursor"] = idx + 1
        filled["order_id"] = "previous_tool_result"

    if filled:
        grounded["arguments"] = args
        metadata = dict(grounded.get("metadata") or {})
        metadata["result_grounded_args"] = filled
        grounded["metadata"] = metadata
    return grounded


def update_grounding_context(context: dict[str, Any], action: dict[str, Any] | None, execution: dict[str, Any]) -> None:
    collect_grounding_values((action or {}).get("arguments") or {}, context)
    if execution.get("ok"):
        collect_grounding_values(execution.get("result"), context)


def assistant_initialization_actions(task: dict[str, Any]) -> list[dict[str, Any]]:
    item = raw_tau2_item(task)
    initial_state = item.get("initial_state") or {}
    actions = []
    for action in initial_state.get("initialization_actions") or []:
        if action.get("env_type") == "assistant":
            actions.append(action)
    return actions


def action_tool_id(action: dict[str, Any] | None) -> str | None:
    if not action:
        return None
    return action.get("tool_id")


def arg_overlap(pred_action: dict[str, Any] | None, gold_action: dict[str, Any] | None) -> dict[str, int]:
    pred_args = (pred_action or {}).get("arguments") or {}
    gold_args = (gold_action or {}).get("arguments") or {}
    if not isinstance(pred_args, dict):
        pred_args = {}
    if not isinstance(gold_args, dict):
        gold_args = {}
    key_hits = set(pred_args) & set(gold_args)
    value_hits = sum(
        json.dumps(pred_args[k], sort_keys=True, default=str) == json.dumps(gold_args[k], sort_keys=True, default=str)
        for k in key_hits
    )
    return {
        "gold_arg_count": len(gold_args),
        "pred_arg_count": len(pred_args),
        "arg_key_hits": len(key_hits),
        "arg_value_hits": value_hits,
    }


def build_trace(
    task: dict[str, Any],
    gold: dict[str, Any],
    pred: dict[str, Any],
    max_steps: int,
    result_grounding: bool = False,
) -> dict[str, Any]:
    db, tools = load_domain_tools(task.get("domain") or "telecom")
    init_results = []
    grounding_context = {"user_ids": [], "order_ids": [], "order_cursor": 0}
    for init_action in assistant_initialization_actions(task):
        result = execute_action(tools, init_action)
        result["func_name"] = callable_name(init_action)
        update_grounding_context(grounding_context, init_action, result)
        init_results.append(result)

    gold_actions = gold.get("actions") or []
    pred_actions = pred.get("actions") or []
    total = min(max(len(gold_actions), len(pred_actions)), max_steps)
    steps = []
    prefix_tool_matches = 0
    needs_replan_at = None

    for idx in range(total):
        gold_action = gold_actions[idx] if idx < len(gold_actions) else None
        pred_action = pred_actions[idx] if idx < len(pred_actions) else None
        tool_match = bool(action_tool_id(gold_action) == action_tool_id(pred_action) and gold_action and pred_action)
        if tool_match and needs_replan_at is None and idx == prefix_tool_matches:
            prefix_tool_matches += 1
        elif needs_replan_at is None:
            needs_replan_at = idx
        grounded_pred_action = (
            action_with_result_grounding(pred_action, grounding_context) if result_grounding else pred_action
        )
        overlap = arg_overlap(grounded_pred_action, gold_action)
        execution = (
            execute_action(tools, grounded_pred_action or {})
            if grounded_pred_action
            else {"ok": False, "error_type": "MissingPrediction"}
        )
        update_grounding_context(grounding_context, grounded_pred_action, execution)
        steps.append(
            {
                "step_idx": idx,
                "predicted_action": grounded_pred_action,
                "gold_action": gold_action,
                "tool_match": tool_match,
                "argument_key_hits": overlap["arg_key_hits"],
                "argument_value_hits": overlap["arg_value_hits"],
                "gold_arg_count": overlap["gold_arg_count"],
                "pred_arg_count": overlap["pred_arg_count"],
                "execution": execution,
                "db_hash_after": db.get_hash(),
                "needs_replan": not tool_match or not execution.get("ok", False),
            }
        )

    return {
        "task_id": task["task_id"],
        "source": task.get("source"),
        "domain": task.get("domain"),
        "prompt": task.get("prompt"),
        "prediction_type": pred.get("plan_type"),
        "result_grounding": result_grounding,
        "assistant_initialization_actions": init_results,
        "initialization_ok": all(row.get("ok") for row in init_results),
        "initial_db_hash_after_init": db.get_hash(),
        "gold_tool_ids": gold.get("tool_ids") or [],
        "pred_tool_ids": pred.get("tool_ids") or [],
        "full_tool_match": (pred.get("tool_ids") or []) == (gold.get("tool_ids") or []),
        "first_tool_match": bool((pred.get("tool_ids") or [None])[0] == (gold.get("tool_ids") or [None])[0]),
        "prefix_tool_match_length": prefix_tool_matches,
        "needs_replan_at": needs_replan_at,
        "steps": steps,
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    if not traces:
        return {"count": 0}
    step_count = sum(len(trace["steps"]) for trace in traces)
    ok_steps = sum(step["execution"].get("ok", False) for trace in traces for step in trace["steps"])
    exception_steps = sum(
        bool(step["execution"].get("error_type")) for trace in traces for step in trace["steps"]
    )
    return {
        "count": len(traces),
        "step_count": step_count,
        "initialization_ok_rate": sum(trace["initialization_ok"] for trace in traces) / len(traces),
        "tool_execution_ok_rate": ok_steps / step_count if step_count else 0.0,
        "tool_execution_exception_rate": exception_steps / step_count if step_count else 0.0,
        "full_tool_match": sum(trace["full_tool_match"] for trace in traces) / len(traces),
        "first_tool_match": sum(trace["first_tool_match"] for trace in traces) / len(traces),
        "avg_prefix_tool_match_length": sum(trace["prefix_tool_match_length"] for trace in traces) / len(traces),
        "needs_replan_rate": sum(
            any(step.get("needs_replan") for step in trace["steps"]) for trace in traces
        )
        / len(traces),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute tau2 predicted traces against real domain tools.")
    parser.add_argument("--pred", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", default=None)
    parser.add_argument("--tasks", default=str(ROOT / "data" / "processed" / "tasks.jsonl"))
    parser.add_argument("--gold", default=str(ROOT / "data" / "processed" / "gold_plans.jsonl"))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--domain", default="telecom", choices=["telecom", "retail"])
    parser.add_argument("--result-grounding", action="store_true")
    args = parser.parse_args()

    tasks = {row["task_id"]: row for row in read_jsonl(args.tasks)}
    gold = {row["task_id"]: row for row in read_jsonl(args.gold)}
    traces = []
    for pred in read_jsonl(args.pred):
        task = tasks.get(pred.get("task_id"))
        gold_plan = gold.get(pred.get("task_id"))
        if not task or not gold_plan:
            continue
        if task.get("source") != "tau2" or task.get("domain") != args.domain:
            continue
        if not gold_plan.get("tool_ids"):
            continue
        traces.append(build_trace(task, gold_plan, pred, args.max_steps, result_grounding=args.result_grounding))
        if args.limit and len(traces) >= args.limit:
            break

    write_jsonl(args.out, traces)
    summary = summarize(traces)
    summary.update(
        {
            "pred": args.pred,
            "out": args.out,
            "limit": args.limit,
            "max_steps": args.max_steps,
            "domain": args.domain,
            "result_grounding": args.result_grounding,
            "execution_backend": f"tau2.domains.{args.domain}.tools",
        }
    )
    if args.summary_out:
        Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_out).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
