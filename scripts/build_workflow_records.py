import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "data" / "processed" / "workflows"


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_split_ids(split):
    path = ROOT / "data" / "processed" / "task_splits" / f"{split}.txt"
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def shorten(text, max_chars):
    text = " ".join((text or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def schema_parameters(tool):
    schema = tool.get("schema") or {}
    return (
        schema.get("parameters")
        or schema.get("required_parameters")
        or schema.get("input_parameters")
        or schema.get("inputs")
        or []
    )


def compact_tool(tool, max_desc_chars):
    return {
        "tool_id": tool["tool_id"],
        "name": tool.get("name") or tool["tool_id"].rsplit("::", 1)[-1],
        "description": shorten(tool.get("description") or "", max_desc_chars),
        "parameters": schema_parameters(tool),
        "phase": tool.get("phase", "unknown"),
    }


def normalized_actions(plan):
    actions = plan.get("actions") or []
    if actions:
        out = []
        for idx, action in enumerate(actions):
            tool_id = action.get("tool_id")
            tool_name = action.get("tool_name") or (tool_id.rsplit("::", 1)[-1] if tool_id else None)
            if not tool_name:
                continue
            out.append(
                {
                    "action_index": idx,
                    "tool_id": tool_id,
                    "tool_name": tool_name,
                    "arguments": action.get("arguments") or {},
                    "requestor": action.get("requestor"),
                }
            )
        return out
    out = []
    for idx, tool_id in enumerate(plan.get("tool_ids") or []):
        out.append(
            {
                "action_index": idx,
                "tool_id": tool_id,
                "tool_name": tool_id.rsplit("::", 1)[-1],
                "arguments": {},
                "requestor": None,
            }
        )
    return out


def action_tool_ids(actions):
    return [action.get("tool_id") for action in actions if action.get("tool_id")]


def build_condition_text(task, available_tools, prefix_actions, tool_feedback, max_chars):
    spec = {
        "task_id": task["task_id"],
        "source": task.get("source"),
        "domain": task.get("domain"),
        "user_prompt": task.get("prompt") or "",
        "available_tools": available_tools,
        "prefix_actions": prefix_actions,
        "tool_feedback": tool_feedback,
    }
    text = "\n".join(
        [
            "Plan the remaining global tool workflow.",
            "Use the task, available tools, previous tool calls, and tool feedback.",
            "The target is the future tool-call path, not only the next tool.",
            "",
            "Workflow state:",
            json.dumps(spec, ensure_ascii=False, indent=2),
        ]
    )
    return shorten(text, max_chars) if max_chars else text


def build_record(task, plan, tools, split, step_idx, actions, max_desc_chars, max_condition_chars):
    available_tools = [
        compact_tool(tools[tool_id], max_desc_chars)
        for tool_id in task.get("available_tool_ids", [])
        if tool_id in tools
    ]
    prefix_actions = actions[:step_idx]
    remaining_actions = actions[step_idx:]
    next_action = remaining_actions[0]
    tool_feedback = []
    condition_text = build_condition_text(
        task,
        available_tools,
        prefix_actions,
        tool_feedback,
        max_condition_chars,
    )
    return {
        "workflow_id": task["task_id"],
        "record_id": f"{task['task_id']}::step::{step_idx}",
        "task_id": task["task_id"],
        "source": task.get("source"),
        "domain": task.get("domain"),
        "split": split,
        "step_idx": step_idx,
        "total_steps": len(actions),
        "prompt": task.get("prompt") or "",
        "available_tools": available_tools,
        "prefix_actions": prefix_actions,
        "tool_feedback": tool_feedback,
        "state_context": {
            "has_initial_state": bool((task.get("metadata") or {}).get("has_initial_state")),
            "raw_path": task.get("raw_path"),
        },
        "target": {
            "next_action": next_action,
            "remaining_actions": remaining_actions,
            "future_tools": [action["tool_name"] for action in remaining_actions],
            "future_tool_ids": action_tool_ids(remaining_actions),
            "full_actions": actions,
            "full_tool_ids": action_tool_ids(actions),
        },
        "condition_text": condition_text,
        "metadata": {
            "gold_action_count": len(actions),
            "available_tool_count": len(task.get("available_tool_ids") or []),
            "has_unseen_tool": bool(plan.get("has_unseen_tool")),
        },
    }


def keep_task(task, plan, split_ids, sources, max_actions, include_empty_gold):
    if task["task_id"] not in split_ids:
        return False
    if task.get("source") not in sources:
        return False
    actions = normalized_actions(plan)
    if not actions and not include_empty_gold:
        return False
    if max_actions and len(actions) > max_actions:
        return False
    return True


def write_readme(out_dir, manifest):
    lines = [
        "# Workflow Records",
        "",
        "Step-level records for global workflow generation.",
        "",
        "Each gold plan is decomposed into records over action prefixes. A record conditions on the task, available tools, prefix actions, and tool feedback, then targets the remaining global workflow path.",
        "",
        "These JSONL files are the intermediate layer before flow-matching tensor extraction. They are not model-ready `.pt` tensors.",
        "",
        "## Files",
        "",
        "- `train.jsonl`",
        "- `dev.jsonl`",
        "- `test.jsonl`",
        "- `manifest.json`",
        "",
        "## Counts",
        "",
    ]
    for split in ["train", "dev", "test"]:
        data = manifest["splits"][split]
        lines.append(f"- {split}: {data['records']} records from {data['workflows']} workflows")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Build step-level workflow records from gold tool plans.")
    parser.add_argument("--tasks", default=str(ROOT / "data" / "processed" / "tasks.jsonl"))
    parser.add_argument("--tools", default=str(ROOT / "data" / "processed" / "tools.jsonl"))
    parser.add_argument("--gold", default=str(ROOT / "data" / "processed" / "gold_plans.jsonl"))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--sources", default="tau2,toolbench")
    parser.add_argument("--max-actions", type=int, default=32)
    parser.add_argument("--max-desc-chars", type=int, default=240)
    parser.add_argument("--max-condition-chars", type=int, default=0)
    parser.add_argument("--include-empty-gold", action="store_true")
    args = parser.parse_args()

    tasks = {row["task_id"]: row for row in read_jsonl(args.tasks)}
    tools = {row["tool_id"]: row for row in read_jsonl(args.tools)}
    plans = {row["task_id"]: row for row in read_jsonl(args.gold)}
    sources = {source.strip() for source in args.sources.split(",") if source.strip()}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "format": "step_level_workflow_records",
        "sources": sorted(sources),
        "max_actions": args.max_actions,
        "max_desc_chars": args.max_desc_chars,
        "max_condition_chars": args.max_condition_chars,
        "splits": {},
    }

    for split in ["train", "dev", "test"]:
        split_ids = load_split_ids(split)
        rows = []
        workflow_count = 0
        skipped = {"missing_task_or_plan": 0, "filtered": 0, "empty_gold": 0}
        source_counts = {}
        domain_counts = {}
        step_histogram = {}
        for task_id in sorted(split_ids):
            task = tasks.get(task_id)
            plan = plans.get(task_id)
            if not task or not plan:
                skipped["missing_task_or_plan"] += 1
                continue
            actions = normalized_actions(plan)
            if not actions:
                skipped["empty_gold"] += 1
            if not keep_task(task, plan, split_ids, sources, args.max_actions, args.include_empty_gold):
                skipped["filtered"] += 1
                continue
            workflow_count += 1
            source = task.get("source")
            domain = f"{source}::{task.get('domain')}"
            source_counts[source] = source_counts.get(source, 0) + 1
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            step_histogram[str(len(actions))] = step_histogram.get(str(len(actions)), 0) + 1
            for step_idx in range(len(actions)):
                rows.append(
                    build_record(
                        task,
                        plan,
                        tools,
                        split,
                        step_idx,
                        actions,
                        args.max_desc_chars,
                        args.max_condition_chars,
                    )
                )
        write_jsonl(out_dir / f"{split}.jsonl", rows)
        manifest["splits"][split] = {
            "path": str((out_dir / f"{split}.jsonl").relative_to(ROOT)),
            "records": len(rows),
            "workflows": workflow_count,
            "skipped": skipped,
            "source_counts": source_counts,
            "domain_counts": domain_counts,
            "gold_action_count_histogram": step_histogram,
        }

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    write_readme(out_dir, manifest)
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "records": {split: data["records"] for split, data in manifest["splits"].items()},
                "workflows": {split: data["workflows"] for split, data in manifest["splits"].items()},
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
