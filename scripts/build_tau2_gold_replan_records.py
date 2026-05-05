import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_telecom_tool_traces import (  # noqa: E402
    assistant_initialization_actions,
    callable_name,
    execute_action,
    load_domain_tools,
    update_grounding_context,
    write_jsonl,
)


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_split_ids(split_dir):
    out = {}
    split_dir = Path(split_dir)
    for split in ["train", "dev", "test"]:
        path = split_dir / f"{split}.txt"
        out[split] = {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    return out


def action_tool_id(action):
    return action.get("tool_id") or (
        f"tau2::{action.get('domain')}::{action.get('name')}" if action.get("domain") and action.get("name") else None
    )


def normalize_action(action, domain):
    tool_name = action.get("tool_name") or action.get("name") or str(action.get("tool_id") or "").rsplit("::", 1)[-1]
    tool_id = action.get("tool_id") or f"tau2::{domain}::{tool_name}"
    return {
        "action_id": action.get("action_id"),
        "tool_id": tool_id,
        "tool_name": tool_name,
        "arguments": action.get("arguments") or {},
        "requestor": action.get("requestor"),
    }


def execute_prefix(task, actions, prefix_len):
    db, tools = load_domain_tools(task.get("domain") or "retail")
    initialization_feedback = []
    grounding_context = {"user_ids": [], "order_ids": [], "order_cursor": 0}
    for init_action in assistant_initialization_actions(task):
        execution = execute_action(tools, init_action)
        execution["func_name"] = callable_name(init_action)
        update_grounding_context(grounding_context, init_action, execution)
        initialization_feedback.append(execution)

    executed_prefix = []
    for idx, action in enumerate(actions[:prefix_len]):
        execution = execute_action(tools, action)
        update_grounding_context(grounding_context, action, execution)
        executed_prefix.append(
            {
                "step_idx": idx,
                "predicted_action": action,
                "gold_action": action,
                "tool_match": True,
                "execution": execution,
                "db_hash_after": db.get_hash(),
                "needs_replan": not execution.get("ok", False),
            }
        )
        if not execution.get("ok", False):
            break
    return db, initialization_feedback, executed_prefix


def build_records_for_task(task, gold, split, args):
    domain = task.get("domain")
    actions = [normalize_action(action, domain) for action in gold.get("actions") or []]
    records = []
    max_prefix = len(actions) + (1 if args.include_stop_rows else 0)
    for prefix_len in range(max_prefix):
        db, initialization_feedback, executed_prefix = execute_prefix(task, actions, prefix_len)
        if args.require_successful_prefix and not all((step.get("execution") or {}).get("ok", False) for step in executed_prefix):
            continue
        remaining = actions[prefix_len:]
        record_id = f"{task['task_id']}::gold_replan::{prefix_len}"
        records.append(
            {
                "task_id": task["task_id"],
                "source": task.get("source"),
                "domain": domain,
                "split": split,
                "prompt": task.get("prompt") or "",
                "prediction_type": "gold_prefix_replan_v1",
                "replan_step_idx": prefix_len,
                "record_id": record_id,
                "initialization_feedback": initialization_feedback,
                "initial_db_hash_after_init": db.get_hash(),
                "executed_prefix": executed_prefix,
                "gold_prefix_actions": actions[:prefix_len],
                "target_remaining_actions": remaining,
                "target_remaining_tool_ids": [action.get("tool_id") for action in remaining],
                "original_pred_remaining_tool_ids": [action.get("tool_id") for action in remaining],
                "replan_reason": {
                    "execution_ok": all((step.get("execution") or {}).get("ok", False) for step in executed_prefix),
                    "error_type": None,
                    "has_gold_action_at_step": bool(remaining),
                    "has_predicted_action_at_step": prefix_len > 0,
                    "tool_match": True,
                    "gold_prefix": True,
                },
                "metadata": {
                    "format": "gold_prefix_replan_record_v1",
                    "target_semantics": "gold_remaining_from_gold_prefix",
                    "source_task_id": task["task_id"],
                },
            }
        )
    return records


def main():
    parser = argparse.ArgumentParser(description="Build tau2 gold-prefix replan records with executable successful prefixes.")
    parser.add_argument("--tasks", default=str(ROOT / "data" / "processed" / "tasks.jsonl"))
    parser.add_argument("--gold-plans", default=str(ROOT / "data" / "processed" / "gold_plans.jsonl"))
    parser.add_argument("--split-dir", default=str(ROOT / "data" / "processed" / "task_splits"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--domain", choices=["retail", "telecom"], default="retail")
    parser.add_argument("--include-stop-rows", action="store_true")
    parser.add_argument("--require-successful-prefix", action="store_true")
    args = parser.parse_args()

    tasks = {row["task_id"]: row for row in read_jsonl(args.tasks)}
    gold = {row["task_id"]: row for row in read_jsonl(args.gold_plans)}
    split_ids = read_split_ids(args.split_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "format": "gold_prefix_replan_record_v1",
        "domain": args.domain,
        "include_stop_rows": args.include_stop_rows,
        "require_successful_prefix": args.require_successful_prefix,
        "splits": {},
    }
    for split, ids in split_ids.items():
        records = []
        task_count = 0
        skipped_no_gold = 0
        for task_id in sorted(ids):
            task = tasks.get(task_id)
            if not task or task.get("source") != "tau2" or task.get("domain") != args.domain:
                continue
            plan = gold.get(task_id)
            if not plan:
                skipped_no_gold += 1
                continue
            task_count += 1
            records.extend(build_records_for_task(task, plan, split, args))
        write_jsonl(out_dir / f"replan_records.{split}.{args.domain}.jsonl", records)
        manifest["splits"][split] = {
            "tasks": task_count,
            "records": len(records),
            "action_rows": sum(1 for row in records if row.get("target_remaining_actions")),
            "stop_rows": sum(1 for row in records if not row.get("target_remaining_actions")),
            "skipped_no_gold": skipped_no_gold,
        }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
