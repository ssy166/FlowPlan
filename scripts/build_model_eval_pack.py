import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "processed" / "model_eval_pack.jsonl"
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


def load_split_ids(split):
    if split == "all":
        return None
    path = ROOT / "data" / "processed" / "task_splits" / f"{split}.txt"
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def shorten(text, max_chars):
    text = " ".join((text or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def load_banking_discoverable_tools(max_desc_chars=160):
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
            raw_desc = doc_lines[0] if doc_lines else ""
            raw_desc = " ".join(raw_desc.split())
            desc = raw_desc[: max_desc_chars - 3] + "..." if len(raw_desc) > max_desc_chars else raw_desc
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
    schema = tool.get("schema") or {}
    return {
        "tool_id": tool["tool_id"],
        "name": tool.get("name") or tool["tool_id"].rsplit("::", 1)[-1],
        "description": shorten(tool.get("description") or "", max_desc_chars),
        "parameters": schema.get("parameters")
        or schema.get("required_parameters")
        or schema.get("input_parameters")
        or schema.get("inputs")
        or [],
    }


def build_prompt(task, tools, max_desc_chars):
    available_tools = [compact_tool(tools[tool_id], max_desc_chars) for tool_id in task.get("available_tool_ids", []) if tool_id in tools]
    spec = {
        "task_id": task["task_id"],
        "source": task["source"],
        "domain": task.get("domain"),
        "user_prompt": task.get("prompt") or "",
        "available_tools": available_tools,
    }
    # Match the SFT training format: banking tasks include the discoverable tool catalog.
    is_banking = task.get("source") == "tau2" and task.get("domain") == "banking_knowledge"
    if is_banking:
        disc = get_banking_discoverable_tools(max_desc_chars)
        if disc:
            spec["discoverable_agent_tools"] = disc
    instruction_lines = (
        [
            "You are a tool-planning model.",
            "Choose the assistant tool calls needed to solve the user task.",
            "Use only tools from available_tools.",
            "For banking tasks, unlock and call discoverable tools by name using",
            "unlock_discoverable_agent_tool and call_discoverable_agent_tool.",
            "The catalog of available discoverable tools is in discoverable_agent_tools.",
            "Return strict JSON only, with this schema:",
            '{"actions":[{"tool_name":"...", "arguments":{}}]}',
            "If no assistant tool call is needed, return {\"actions\":[]}.",
        ]
        if is_banking
        else [
            "You are a tool-planning model.",
            "Choose the assistant tool calls needed to solve the user task.",
            "Use only tools from available_tools.",
            "Return strict JSON only, with this schema:",
            '{"actions":[{"tool_name":"...", "arguments":{}}]}',
            "If no assistant tool call is needed, return {\"actions\":[]}.",
        ]
    )
    return "\n".join(instruction_lines + ["", "Task:", json.dumps(spec, ensure_ascii=False, indent=2)])


def task_allowed(task, split_ids, source, domain, include_empty_gold, gold):
    if split_ids is not None and task["task_id"] not in split_ids:
        return False
    if source != "all" and task.get("source") != source:
        return False
    if domain != "all" and task.get("domain") != domain:
        return False
    if not include_empty_gold and not (gold.get(task["task_id"], {}).get("tool_ids") or []):
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Build a JSONL prompt pack for model tool-plan inference.")
    parser.add_argument("--tasks", default=str(ROOT / "data" / "processed" / "tasks.jsonl"))
    parser.add_argument("--tools", default=str(ROOT / "data" / "processed" / "tools.jsonl"))
    parser.add_argument("--gold", default=str(ROOT / "data" / "processed" / "gold_plans.jsonl"))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--split", choices=["all", "train", "dev", "test"], default="dev")
    parser.add_argument("--source", default="tau2", help="Use 'all' to include every source.")
    parser.add_argument("--domain", default="all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-desc-chars", type=int, default=320)
    parser.add_argument("--include-empty-gold", action="store_true")
    args = parser.parse_args()

    tasks = read_jsonl(args.tasks)
    tools = {row["tool_id"]: row for row in read_jsonl(args.tools)}
    gold = {row["task_id"]: row for row in read_jsonl(args.gold)}
    split_ids = load_split_ids(args.split)

    rows = []
    for task in tasks:
        if not task_allowed(task, split_ids, args.source, args.domain, args.include_empty_gold, gold):
            continue
        row = {
            "task_id": task["task_id"],
            "source": task["source"],
            "domain": task.get("domain"),
            "split": args.split,
            "prompt": build_prompt(task, tools, args.max_desc_chars),
            "available_tool_ids": task.get("available_tool_ids") or [],
        }
        rows.append(row)
        if args.limit and len(rows) >= args.limit:
            break

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"out": str(out), "rows": len(rows), "split": args.split, "source": args.source}, ensure_ascii=False))


if __name__ == "__main__":
    main()
