import argparse
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_fm_next_tool_prior import (  # noqa: E402
    RETAIL_INTENT_PATTERNS,
    STOP,
    label_for_row,
    load_tools,
    parse_conditioned_state,
    read_jsonl,
    stage_for_tool_id,
    write_jsonl,
)


def action_target(row, tools):
    label = label_for_row(row)
    if label == STOP:
        return {"tool_id": STOP, "tool_name": STOP, "stage": stage_for_tool_id(STOP, tools), "arguments": {}}
    tool = tools.get(label) or {}
    actions, _ = row_actions(row)
    arguments = actions[0].get("arguments") if actions else {}
    return {
        "tool_id": label,
        "tool_name": tool.get("name") or label.rsplit("::", 1)[-1],
        "stage": stage_for_tool_id(label, tools),
        "arguments": arguments or {},
    }


def row_actions(row):
    content = ((row.get("messages") or [])[-1] or {}).get("content") or "{}"
    try:
        target = json.loads(content)
    except Exception:
        return [], False
    return target.get("actions") or [], bool(target.get("stop"))


def intent_tags(task_text):
    text = str(task_text or "").lower()
    tags = []
    for name, pattern in RETAIL_INTENT_PATTERNS:
        if re.search(pattern, text):
            tags.append(name)
    return tags


def prefix_summary(prefix):
    out = []
    for step in prefix or []:
        if not isinstance(step, dict):
            continue
        result = step.get("result_summary")
        out.append(
            {
                "tool_name": step.get("tool_name") or (step.get("action") or {}).get("tool_name"),
                "ok": step.get("ok"),
                "tool_match": step.get("tool_match"),
                "result_status": result.get("status") if isinstance(result, dict) else None,
                "has_order_id": bool(result.get("order_id")) if isinstance(result, dict) else False,
                "item_count": len(result.get("item_ids") or []) if isinstance(result, dict) else 0,
                "product_count": len(result.get("product_ids") or []) if isinstance(result, dict) else 0,
                "payment_count": len(result.get("payment_method_ids") or []) if isinstance(result, dict) else 0,
            }
        )
    return out


def decision_case(target, state):
    progress = state.get("grounding_progress") or {}
    stage = target["stage"]
    if stage == "stop":
        return "stop"
    if progress.get("order_grounded") and stage == "retail_operation":
        return "post_order_operation"
    if progress.get("order_grounded") and stage in {"retail_order_lookup", "retail_product_lookup", "retail_item_lookup"}:
        return "post_order_detail_lookup"
    if progress.get("lookup_completed") and not progress.get("order_grounded"):
        return "post_user_order_lookup"
    if stage.endswith("_lookup"):
        return "initial_lookup"
    return "other"


def build_row(row, tools):
    state = parse_conditioned_state(row)
    target = action_target(row, tools)
    progress = state.get("grounding_progress") or {}
    metadata = row.get("metadata") or {}
    return {
        "id": row.get("id"),
        "split": metadata.get("split"),
        "task_id": metadata.get("task_id"),
        "domain": metadata.get("domain"),
        "task": state.get("task") or "",
        "intent_tags": intent_tags(state.get("task") or ""),
        "step": state.get("step"),
        "available_tool_names": [tool.get("name") for tool in state.get("available_tools") or [] if isinstance(tool, dict)],
        "prefix": prefix_summary(state.get("executed_prefix") or []),
        "grounding_progress": {
            "lookup_completed": progress.get("lookup_completed"),
            "order_grounded": progress.get("order_grounded"),
            "last_successful_lookup_tool": progress.get("last_successful_lookup_tool"),
            "resolved_order_status": progress.get("resolved_order_status"),
            "resolved_order_ids": progress.get("resolved_order_ids") or [],
            "resolved_order_item_ids": progress.get("resolved_order_item_ids") or [],
            "resolved_payment_method_ids": progress.get("resolved_payment_method_ids") or [],
            "operation_eligibility": progress.get("operation_eligibility") or [],
        },
        "target": target,
        "decision_case": decision_case(target, state),
    }


def summarize(rows):
    out = {"rows": len(rows), "by_target_stage": {}, "by_decision_case": {}, "by_target_tool": {}}
    for row in rows:
        for key, value in [
            ("by_target_stage", row["target"]["stage"]),
            ("by_decision_case", row["decision_case"]),
            ("by_target_tool", row["target"]["tool_name"]),
        ]:
            out[key][value] = out[key].get(value, 0) + 1
    return out


def main():
    parser = argparse.ArgumentParser(description="Build a compact retail decision dataset from replan SFT rows.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--tools", default=str(ROOT / "data" / "processed" / "tools.jsonl"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    args = parser.parse_args()

    tools = load_tools(args.tools)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"source_data": args.data_dir, "splits": {}}
    for split in args.splits:
        rows = [
            build_row(row, tools)
            for row in read_jsonl(Path(args.data_dir) / f"{split}.jsonl")
            if (row.get("metadata") or {}).get("domain") == "retail"
        ]
        write_jsonl(out_dir / f"{split}.jsonl", rows)
        manifest["splits"][split] = summarize(rows)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out_dir": str(out_dir), "splits": {k: v["rows"] for k, v in manifest["splits"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
