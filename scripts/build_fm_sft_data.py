import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "data" / "processed" / "fm_sft"
BANKING_TOOLS_PY = (
    ROOT / "data" / "raw" / "tau2-bench" / "repo" / "src" / "tau2"
    / "domains" / "banking_knowledge" / "tools.py"
)


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


def load_banking_discoverable_tools(max_desc_chars=160):
    """Parse banking tools.py to extract all @is_discoverable_tool methods."""
    if not BANKING_TOOLS_PY.exists():
        return []
    src = BANKING_TOOLS_PY.read_text(encoding="utf-8")
    names = re.findall(r"@is_discoverable_tool[^\n]*\n\s+def (\w+)\(", src)
    tools = []
    for name in names:
        if "example" in name:
            continue
        m = re.search(r"def " + re.escape(name) + r"\(.*?\).*?:\s+\"\"\"(.*?)\"\"\"", src, re.DOTALL)
        desc, params = "", []
        if m:
            doc_lines = [l.strip() for l in m.group(1).strip().splitlines()]
            desc = shorten(doc_lines[0] if doc_lines else "", max_desc_chars)
            in_args = False
            for line in doc_lines:
                if line.startswith("Args:"):
                    in_args = True
                    continue
                if line.startswith("Returns:") or line.startswith("Raises:"):
                    in_args = False
                if in_args:
                    pm = re.match(r"(\w+)\s*\((\w+)\):\s*(.*)", line)
                    if pm:
                        params.append({"name": pm.group(1), "type": pm.group(2)})
        tools.append({"name": name, "description": desc, "parameters": params})
    return tools


_BANKING_DISCOVERABLE_TOOLS = None


def get_banking_discoverable_tools(max_desc_chars=160):
    global _BANKING_DISCOVERABLE_TOOLS
    if _BANKING_DISCOVERABLE_TOOLS is None:
        _BANKING_DISCOVERABLE_TOOLS = load_banking_discoverable_tools(max_desc_chars)
    return _BANKING_DISCOVERABLE_TOOLS


def compact_tool(tool, max_desc_chars):
    return {
        "tool_id": tool["tool_id"],
        "name": tool.get("name") or tool["tool_id"].rsplit("::", 1)[-1],
        "description": shorten(tool.get("description") or "", max_desc_chars),
        "parameters": schema_parameters(tool),
    }


def build_user_content(task, tools, max_desc_chars):
    available_tools = [
        compact_tool(tools[tool_id], max_desc_chars)
        for tool_id in task.get("available_tool_ids", [])
        if tool_id in tools
    ]
    spec = {
        "task_id": task["task_id"],
        "source": task["source"],
        "domain": task.get("domain"),
        "user_prompt": task.get("prompt") or "",
        "available_tools": available_tools,
    }
    # Banking tasks use a discoverable-tool mechanism: the agent unlocks and calls
    # tools by name via unlock_discoverable_agent_tool / call_discoverable_agent_tool.
    # Include the full catalog so the model can predict the correct tool names and args.
    if task.get("domain") == "banking_knowledge" and task.get("source") == "tau2":
        disc = get_banking_discoverable_tools(max_desc_chars)
        if disc:
            spec["discoverable_agent_tools"] = disc
    instruction_lines = [
        "You are a tool-planning model.",
        "Choose the assistant tool calls needed to solve the user task.",
        "Use only tools from available_tools.",
        "For banking tasks, unlock and call discoverable tools by name using",
        "unlock_discoverable_agent_tool and call_discoverable_agent_tool.",
        "The catalog of available discoverable tools is in discoverable_agent_tools.",
        "Return strict JSON only, with this schema:",
        '{"actions":[{"tool_name":"...", "arguments":{}}]}',
        "If no assistant tool call is needed, return {\"actions\":[]}.",
    ] if task.get("domain") == "banking_knowledge" else [
        "You are a tool-planning model.",
        "Choose the assistant tool calls needed to solve the user task.",
        "Use only tools from available_tools.",
        "Return strict JSON only, with this schema:",
        '{"actions":[{"tool_name":"...", "arguments":{}}]}',
        "If no assistant tool call is needed, return {\"actions\":[]}.",
    ]
    return "\n".join(instruction_lines + ["", "Task:", json.dumps(spec, ensure_ascii=False, indent=2)])


def gold_actions(plan):
    actions = plan.get("actions") or []
    if actions:
        return [
            {
                "tool_name": action.get("tool_name") or action.get("tool_id", "").rsplit("::", 1)[-1],
                "arguments": action.get("arguments") or {},
            }
            for action in actions
            if action.get("tool_name") or action.get("tool_id")
        ]
    return [
        {
            "tool_name": tool_id.rsplit("::", 1)[-1],
            "arguments": {},
        }
        for tool_id in plan.get("tool_ids") or []
    ]


def build_example(task, plan, tools, split, max_desc_chars):
    actions = gold_actions(plan)
    return {
        "id": task["task_id"],
        "messages": [
            {"role": "user", "content": build_user_content(task, tools, max_desc_chars)},
            {"role": "assistant", "content": json.dumps({"actions": actions}, ensure_ascii=False, sort_keys=True)},
        ],
        "metadata": {
            "source": task.get("source"),
            "domain": task.get("domain"),
            "split": split,
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
    action_count = len(plan.get("actions") or plan.get("tool_ids") or [])
    if action_count == 0 and not include_empty_gold:
        return False
    if max_actions and action_count > max_actions:
        return False
    return True


def write_readme(out_dir, manifest):
    body = [
        "# FM SFT Data",
        "",
        "Chat-style supervised data for offline tool-plan generation.",
        "",
        "Input: task prompt plus available tool schemas.",
        "",
        'Output: strict JSON `{"actions":[{"tool_name":"...", "arguments":{}}]}`.',
        "",
        "This is a plain SFT pack. It does not include database state summaries or closed-loop tool execution traces.",
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
        body.append(f"- {split}: {manifest['splits'][split]['rows']} rows")
    body.extend(
        [
            "",
            "## Suggested First Training Target",
            "",
            "Train a Qwen-style chat model or LoRA to produce the assistant message from the user message.",
            "Evaluate generated predictions with `scripts/evaluate_plans.py` after converting model outputs to prediction JSONL.",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(body) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Build plain FM SFT data for tool-plan generation.")
    parser.add_argument("--tasks", default=str(ROOT / "data" / "processed" / "tasks.jsonl"))
    parser.add_argument("--tools", default=str(ROOT / "data" / "processed" / "tools.jsonl"))
    parser.add_argument("--gold", default=str(ROOT / "data" / "processed" / "gold_plans.jsonl"))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--sources", default="tau2,toolbench")
    parser.add_argument("--max-actions", type=int, default=16)
    parser.add_argument("--max-desc-chars", type=int, default=320)
    parser.add_argument("--include-empty-gold", action="store_true")
    args = parser.parse_args()

    tasks = {row["task_id"]: row for row in read_jsonl(args.tasks)}
    tools = {row["tool_id"]: row for row in read_jsonl(args.tools)}
    plans = {row["task_id"]: row for row in read_jsonl(args.gold)}
    sources = {item.strip() for item in args.sources.split(",") if item.strip()}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "format": "chat_messages_jsonl",
        "sources": sorted(sources),
        "include_empty_gold": args.include_empty_gold,
        "max_actions": args.max_actions,
        "max_desc_chars": args.max_desc_chars,
        "splits": {},
    }

    for split in ["train", "dev", "test"]:
        split_ids = load_split_ids(split)
        rows = []
        skipped = {"missing_task_or_plan": 0, "filtered": 0}
        for task_id in sorted(split_ids):
            task = tasks.get(task_id)
            plan = plans.get(task_id)
            if not task or not plan:
                skipped["missing_task_or_plan"] += 1
                continue
            if not keep_task(task, plan, split_ids, sources, args.max_actions, args.include_empty_gold):
                skipped["filtered"] += 1
                continue
            rows.append(build_example(task, plan, tools, split, args.max_desc_chars))
        write_jsonl(out_dir / f"{split}.jsonl", rows)
        source_counts = {}
        domain_counts = {}
        action_counts = {}
        for row in rows:
            metadata = row["metadata"]
            source_counts[metadata["source"]] = source_counts.get(metadata["source"], 0) + 1
            domain = f"{metadata['source']}::{metadata['domain']}"
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            count_key = str(metadata["gold_action_count"])
            action_counts[count_key] = action_counts.get(count_key, 0) + 1
        manifest["splits"][split] = {
            "rows": len(rows),
            "skipped": skipped,
            "source_counts": source_counts,
            "domain_counts": domain_counts,
            "gold_action_count_histogram": action_counts,
            "path": str((out_dir / f"{split}.jsonl").relative_to(ROOT)),
        }

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    write_readme(out_dir, manifest)
    print(json.dumps({"out_dir": str(out_dir), "splits": {k: v["rows"] for k, v in manifest["splits"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
