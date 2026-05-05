import csv
import hashlib
import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"


PHASE_RULES = [
    ("verify", ["assert", "check", "health", "status", "validate", "verify", "test", "can_"]),
    ("retrieve", ["get", "search", "query", "list", "read", "lookup", "find", "fetch", "ocr", "detail"]),
    ("compute", ["calculate", "calculator", "solver", "count", "math", "plot"]),
    ("operate", ["book", "cancel", "update", "modify", "add", "delete", "return", "exchange", "send", "enable", "toggle", "transfer", "reset", "set_"]),
    ("create", ["create", "generate", "draw", "write", "export", "caption", "stylization", "docx", "pdf", "pptx", "xlsx", "csv"]),
]


def stable_bucket(text, modulo=100):
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo


def infer_phase(name, description="", domain=""):
    text = f"{name} {description} {domain}".lower()
    for phase, keywords in PHASE_RULES:
        if any(keyword in text for keyword in keywords):
            return phase
    return "other"


def assign_tool_split(tool_id):
    bucket = stable_bucket(tool_id)
    if bucket < 70:
        return "train_seen"
    if bucket < 85:
        return "dev_seen"
    return "test_unseen"


def assign_task_split(task_id):
    bucket = stable_bucket(task_id)
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "dev"
    return "test"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{row}\n" for row in rows), encoding="utf-8")


def add_tool(tools, source, domain, name, description="", schema=None, metadata=None):
    if not name:
        return
    tool_id = f"{source}::{domain}::{name}"
    phase = infer_phase(name, description, domain)
    if tool_id in tools:
        existing = tools[tool_id]
        existing["description"] = existing.get("description") or description or ""
        if schema and not existing.get("schema"):
            existing["schema"] = schema
        existing.setdefault("metadata", {}).update(metadata or {})
        existing["phase"] = existing.get("phase") or phase
        existing["tool_split"] = existing.get("tool_split") or assign_tool_split(tool_id)
        return
    tools[tool_id] = {
        "tool_id": tool_id,
        "source": source,
        "domain": domain,
        "name": name,
        "description": description or "",
        "phase": phase,
        "tool_split": assign_tool_split(tool_id),
        "schema": schema or {},
        "metadata": metadata or {},
    }


def norm_action_name(action):
    return action.get("name") or action.get("func_name") or action.get("action_id")


def norm_plan_action(source, domain, action):
    name = norm_action_name(action)
    return {
        "action_id": action.get("action_id"),
        "tool_name": name,
        "tool_id": f"{source}::{domain}::{name}" if name else None,
        "arguments": action.get("arguments") or {},
        "requestor": action.get("requestor"),
    }


def is_is_tool_decorator(decorator):
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    return isinstance(target, ast.Name) and target.id == "is_tool"


def collect_tau2_domain_tools(tools):
    domain_tools = {}
    base = RAW / "tau2-bench" / "repo" / "src" / "tau2" / "domains"
    for tool_file in sorted(base.glob("*/tools.py")):
        domain = tool_file.parent.name
        domain_tools.setdefault(domain, set())
        try:
            tree = ast.parse(tool_file.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if not any(is_is_tool_decorator(d) for d in node.decorator_list):
                continue
            args = [arg.arg for arg in node.args.args if arg.arg != "self"]
            description = ast.get_docstring(node) or ""
            add_tool(
                tools,
                "tau2",
                domain,
                node.name,
                description=description,
                schema={"parameters": args},
                metadata={"from_domain_definition": True, "path": str(tool_file.relative_to(ROOT))},
            )
            domain_tools[domain].add(node.name)
    return domain_tools


def enrich_tasks_and_plans(tasks, plans, tools):
    task_by_id = {task["task_id"]: task for task in tasks}
    for plan in plans:
        phase_names = []
        unseen_count = 0
        for tool_id in plan.get("tool_ids", []):
            tool = tools.get(tool_id)
            if not tool:
                phase_names.append("unknown")
                continue
            phase_names.append(tool.get("phase", "other"))
            if tool.get("tool_split") == "test_unseen":
                unseen_count += 1
        plan["phase_names"] = phase_names
        plan["has_unseen_tool"] = unseen_count > 0
        task = task_by_id.get(plan["task_id"])
        if task is not None:
            task.setdefault("metadata", {})["gold_tool_count"] = len(plan.get("tool_ids", []))
            task["metadata"]["gold_phase_count"] = len(phase_names)
            task["metadata"]["has_unseen_gold_tool"] = unseen_count > 0
            task["metadata"]["requires_multi_tool_plan"] = len(plan.get("tool_ids", [])) >= 2


def collect_tau2(tools, tasks, plans):
    base = RAW / "tau2-bench" / "repo" / "data" / "tau2" / "domains"
    domain_tool_names = collect_tau2_domain_tools(tools)
    for task_file in sorted(base.glob("*/tasks*.json")):
        domain = task_file.parent.name
        split = task_file.stem
        try:
            items = read_json(task_file)
        except Exception:
            continue
        if not isinstance(items, list):
            continue
        for item in items:
            local_id = str(item.get("id") or len(tasks))
            task_id = f"tau2::{domain}-{split}::{local_id}"
            scenario = (item.get("user_scenario") or {}).get("instructions") or {}
            if isinstance(scenario, dict):
                prompt_parts = [
                    scenario.get("reason_for_call"),
                    scenario.get("known_info"),
                    scenario.get("task_instructions"),
                    item.get("ticket"),
                ]
            else:
                prompt_parts = [scenario, item.get("ticket")]
            prompt = "\n".join(str(x) for x in prompt_parts if x)
            evaluation = item.get("evaluation_criteria") or {}
            initial_state = item.get("initial_state") or {}
            eval_actions = evaluation.get("actions") or []
            init_actions = initial_state.get("initialization_actions") or []
            assistant_eval_actions = [
                a for a in eval_actions if (a.get("requestor") in (None, "assistant")) and norm_action_name(a)
            ]
            user_eval_actions = [
                a for a in eval_actions if (a.get("requestor") not in (None, "assistant")) and norm_action_name(a)
            ]
            env_assertions = evaluation.get("env_assertions") or []
            gold = [norm_action_name(a) for a in assistant_eval_actions]
            action_tools = {norm_action_name(a) for a in init_actions + eval_actions if norm_action_name(a)}
            available = sorted(domain_tool_names.get(domain) or action_tools)
            for name in available:
                add_tool(tools, "tau2", domain, name, metadata={"from_task_actions": True})
            tasks.append(
                {
                    "task_id": task_id,
                    "source": "tau2",
                    "domain": domain,
                    "split": split,
                    "prompt": prompt,
                    "available_tool_ids": [f"tau2::{domain}::{name}" for name in available],
                    "raw_path": str(task_file.relative_to(ROOT)),
                    "metadata": {
                        "workflow_category": item.get("description", {}).get("purpose"),
                        "has_initial_state": bool(item.get("initial_state")),
                        "assistant_eval_action_count": len(assistant_eval_actions),
                        "user_eval_action_count": len(user_eval_actions),
                        "env_assertion_count": len(env_assertions),
                    },
                }
            )
            plans.append(
                {
                    "task_id": task_id,
                    "source": "tau2",
                    "plan_type": "evaluation_actions",
                    "tool_names": gold,
                    "tool_ids": [f"tau2::{domain}::{name}" for name in gold],
                    "actions": [norm_plan_action("tau2", domain, action) for action in assistant_eval_actions],
                }
            )


def collect_toolbench(tools, tasks, plans):
    base = RAW / "toolbench" / "repo" / "data_example" / "instruction"
    for query_file in sorted(base.glob("G*_query.json")):
        group = query_file.stem.replace("_query", "")
        items = read_json(query_file)
        for item in items:
            local_id = str(item.get("query_id"))
            task_id = f"toolbench::{group}::{local_id}"
            api_list = item.get("api_list") or []
            available_ids = []
            for api in api_list:
                category = api.get("category_name") or "unknown"
                name = f"{api.get('tool_name')}::{api.get('api_name')}"
                add_tool(
                    tools,
                    "toolbench",
                    category,
                    name,
                    description=api.get("api_description") or "",
                    schema={
                        "required_parameters": api.get("required_parameters") or [],
                        "optional_parameters": api.get("optional_parameters") or [],
                        "method": api.get("method"),
                    },
                )
                available_ids.append(f"toolbench::{category}::{name}")
            relevant = item.get("relevant APIs") or []
            gold_names = [f"{pair[0]}::{pair[1]}" for pair in relevant if len(pair) == 2]
            tasks.append(
                {
                    "task_id": task_id,
                    "source": "toolbench",
                    "domain": group,
                    "split": "example",
                    "prompt": item.get("query") or "",
                    "available_tool_ids": available_ids,
                    "raw_path": str(query_file.relative_to(ROOT)),
                    "metadata": {"query_id": item.get("query_id")},
                }
            )
            plans.append(
                {
                    "task_id": task_id,
                    "source": "toolbench",
                    "plan_type": "relevant_apis",
                    "tool_names": gold_names,
                    "tool_ids": [
                        tool_id
                        for tool_id in available_ids
                        if tool_id.rsplit("::", 1)[-1] in gold_names
                    ],
                }
            )
    hf = RAW / "toolbench" / "hf"
    if hf.exists():
        import pandas as pd

        for query_file in sorted((hf / "benchmark").glob("*.parquet")):
            split = query_file.stem.replace("-00000-of-00001", "")
            df = pd.read_parquet(query_file)
            for _, row in df.iterrows():
                local_id = str(row.get("query_id"))
                task_id = f"toolbench::{split}::{local_id}"
                try:
                    api_list = json.loads(row.get("api_list") or "[]")
                except Exception:
                    api_list = []
                try:
                    relevant = json.loads(row.get("relevant_apis") or "[]")
                except Exception:
                    relevant = []
                available_ids = []
                for api in api_list:
                    category = api.get("category_name") or "unknown"
                    name = f"{api.get('tool_name')}::{api.get('api_name')}"
                    add_tool(
                        tools,
                        "toolbench",
                        category,
                        name,
                        description=api.get("api_description") or "",
                        schema={
                            "required_parameters": api.get("required_parameters") or [],
                            "optional_parameters": api.get("optional_parameters") or [],
                            "method": api.get("method"),
                            "template_response": api.get("template_response"),
                        },
                    )
                    available_ids.append(f"toolbench::{category}::{name}")
                gold_names = [f"{pair[0]}::{pair[1]}" for pair in relevant if len(pair) == 2]
                tasks.append(
                    {
                        "task_id": task_id,
                        "source": "toolbench",
                        "domain": split,
                        "split": "benchmark",
                        "prompt": row.get("query") or "",
                        "available_tool_ids": available_ids,
                        "raw_path": str(query_file.relative_to(ROOT)),
                        "metadata": {"query_id": local_id},
                    }
                )
                plans.append(
                    {
                        "task_id": task_id,
                        "source": "toolbench",
                        "plan_type": "relevant_apis",
                        "tool_names": gold_names,
                        "tool_ids": [
                            tool_id
                            for tool_id in available_ids
                            if tool_id.rsplit("::", 1)[-1] in gold_names
                        ],
                    }
                )

        validation = hf / "data" / "validation-00000-of-00001.parquet"
        if validation.exists():
            df = pd.read_parquet(validation)
            action_pat = re.compile(r"^Action:\s*([A-Za-z0-9_./-]+)", re.MULTILINE)
            for idx, row in df.iterrows():
                task_id = f"toolbench::validation::{idx}"
                conv = row.get("conversations")
                values = list(conv.get("value", [])) if hasattr(conv, "get") else []
                prompt = str(row.get("id") or "")
                actions = []
                for value in values:
                    actions.extend(action_pat.findall(str(value)))
                actions = [a for a in actions if a != "Finish"]
                for name in actions:
                    add_tool(tools, "toolbench", "validation", name, metadata={"from_trajectory": True})
                tasks.append(
                    {
                        "task_id": task_id,
                        "source": "toolbench",
                        "domain": "validation",
                        "split": "trajectory",
                        "prompt": prompt,
                        "available_tool_ids": [f"toolbench::validation::{name}" for name in sorted(set(actions))],
                        "raw_path": str(validation.relative_to(ROOT)),
                        "metadata": {"row": int(idx), "trajectory_turns": len(values)},
                    }
                )
                plans.append(
                    {
                        "task_id": task_id,
                        "source": "toolbench",
                        "plan_type": "trajectory_actions",
                        "tool_names": actions,
                        "tool_ids": [f"toolbench::validation::{name}" for name in actions],
                    }
                )


def collect_api_bank(tools):
    csv_path = RAW / "api-bank" / "repo" / "api-bank" / "data" / "all_apis.csv"
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            class_name = row.get("绫诲悕") or row.get("class_name") or row.get("API名称") or row.get("API鍚嶇О")
            scenario = row.get("搴旂敤鍦烘櫙") or "api"
            add_tool(
                tools,
                "api-bank",
                scenario,
                class_name,
                description=row.get("api_info") or "",
                schema={"expression": row.get("expressions"), "input_parameters": row.get("input_parameters")},
                metadata={"path": row.get("璺緞")},
            )


def collect_gta(tools, tasks, plans):
    base = RAW / "gta" / "gta_workflow_dataset"
    toolmeta = base / "toolmeta.json"
    if toolmeta.exists():
        for name, meta in read_json(toolmeta).items():
            add_tool(
                tools,
                "gta",
                "workflow",
                name,
                description=meta.get("description") or "",
                schema={"inputs": meta.get("inputs") or [], "outputs": meta.get("outputs") or []},
            )
    end_json = base / "end.json"
    if end_json.exists():
        items = read_json(end_json)
        for local_id, item in items.items():
            task_id = f"gta::workflow::{local_id}"
            dialogs = item.get("dialogs") or []
            prompt = "\n".join(d.get("content", "") for d in dialogs if d.get("role") == "user")
            tool_names = item.get("tools") or []
            tasks.append(
                {
                    "task_id": task_id,
                    "source": "gta",
                    "domain": item.get("workflow_category_6class") or "workflow",
                    "split": "workflow_v2",
                    "prompt": prompt,
                    "available_tool_ids": [f"gta::workflow::{name}" for name in tool_names],
                    "raw_path": str(end_json.relative_to(ROOT)),
                    "metadata": {
                        "files": item.get("files") or [],
                        "sub_task_count": len(item.get("sub_tasks") or []),
                    },
                }
            )
            plans.append(
                {
                    "task_id": task_id,
                    "source": "gta",
                    "plan_type": "listed_tools",
                    "tool_names": tool_names,
                    "tool_ids": [f"gta::workflow::{name}" for name in tool_names],
                }
            )


def build_manifest(tools, tasks, plans):
    phase_counts = {}
    tool_split_counts = {}
    for tool in tools.values():
        phase_counts[tool["phase"]] = phase_counts.get(tool["phase"], 0) + 1
        tool_split_counts[tool["tool_split"]] = tool_split_counts.get(tool["tool_split"], 0) + 1
    task_split_counts = {}
    for task in tasks:
        task_split = assign_task_split(task["task_id"])
        task_split_counts[task_split] = task_split_counts.get(task_split, 0) + 1
    sources = {}
    for source in ["tau2-bench", "api-bank", "toolbench", "gta"]:
        path = RAW / source
        sources[source] = {
            "raw_path": str(path.relative_to(ROOT)),
            "present": path.exists(),
        }
    return {
        "outputs": {
            "tools": "data/processed/tools.jsonl",
            "tasks": "data/processed/tasks.jsonl",
            "gold_plans": "data/processed/gold_plans.jsonl",
            "task_splits": {
                "train": "data/processed/task_splits/train.txt",
                "dev": "data/processed/task_splits/dev.txt",
                "test": "data/processed/task_splits/test.txt",
            },
        },
        "counts": {
            "tools": len(tools),
            "tasks": len(tasks),
            "gold_plans": len(plans),
            "tool_phases": phase_counts,
            "tool_splits": tool_split_counts,
            "task_splits": task_split_counts,
        },
        "sources": sources,
        "notes": [
            "Seed index only; raw files remain authoritative.",
            "Phase labels and seen/unseen tool splits are heuristic and embedded in JSONL rows.",
            "Task split files contain task ids only; task rows are not duplicated.",
        ],
    }


def main():
    tools = {}
    tasks = []
    plans = []
    collect_tau2(tools, tasks, plans)
    collect_toolbench(tools, tasks, plans)
    collect_api_bank(tools)
    collect_gta(tools, tasks, plans)
    enrich_tasks_and_plans(tasks, plans, tools)
    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / "tools.jsonl", tools.values())
    write_jsonl(OUT / "tasks.jsonl", tasks)
    write_jsonl(OUT / "gold_plans.jsonl", plans)
    task_splits = {"train": [], "dev": [], "test": []}
    for task in sorted(tasks, key=lambda row: row["task_id"]):
        task_splits[assign_task_split(task["task_id"])].append(task["task_id"])
    for name, task_ids in task_splits.items():
        write_lines(OUT / "task_splits" / f"{name}.txt", task_ids)
    (OUT / "sources_manifest.json").write_text(
        json.dumps(build_manifest(tools, tasks, plans), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(build_manifest(tools, tasks, plans)["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
