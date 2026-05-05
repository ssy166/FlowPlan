import argparse
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

try:
    from make_baseline_predictions import infer_state_arguments, load_tau2_domain_db, merged_state_params
except Exception:  # pragma: no cover - optional local import for state-aware enrichment
    infer_state_arguments = None
    load_tau2_domain_db = None
    merged_state_params = None

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


def schema_parameters(tool):
    schema = tool.get("schema") or {}
    return (
        schema.get("parameters")
        or schema.get("required_parameters")
        or schema.get("input_parameters")
        or schema.get("inputs")
        or []
    )


def values_for_param(prompt, param):
    if param == "user_id":
        return USER_ID_RE.findall(prompt or "")
    if param in {"reservation_id", "confirmation_number"}:
        return RESERVATION_ID_RE.findall(prompt or "")
    if param in {"date", "birth_date"}:
        return DATE_RE.findall(prompt or "")
    if param in {"leave_after", "time"}:
        return TIME_RE.findall(prompt or "")
    if param in {"email", "new_email"}:
        return EMAIL_RE.findall(prompt or "")
    if param in {"phone_number", "phone"}:
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
    args = {}
    for param in schema_parameters(tool):
        values = values_for_param(prompt, param)
        if not values:
            continue
        key = (tool.get("tool_id"), param)
        idx = counters.get(key, 0)
        args[param] = values[min(idx, len(values) - 1)]
        counters[key] = idx + 1
    return args


def enrich(predictions, tasks, tools, use_state=False):
    task_by_id = {task["task_id"]: task for task in tasks}
    db_cache = {}
    rows = []
    for pred in predictions:
        task = task_by_id.get(pred["task_id"], {})
        prompt = task.get("prompt", "")
        counters = {}
        state_params = {}
        if use_state and task.get("source") == "tau2" and merged_state_params and load_tau2_domain_db:
            domain = task.get("domain")
            db_cache.setdefault(domain, load_tau2_domain_db(domain))
            state_params = merged_state_params(task, db_cache[domain])
        actions = []
        for tool_id in pred.get("tool_ids") or []:
            tool = tools.get(tool_id) or {"tool_id": tool_id}
            if use_state and state_params and infer_state_arguments:
                arguments = infer_state_arguments(prompt, tool, state_params, counters)
            else:
                arguments = infer_arguments(prompt, tool, counters)
            actions.append(
                {
                    "tool_id": tool_id,
                    "tool_name": tool.get("name") or tool_id.rsplit("::", 1)[-1],
                    "arguments": arguments,
                }
            )
        row = dict(pred)
        row["actions"] = actions
        row["tool_names"] = [action["tool_name"] for action in actions]
        row["plan_type"] = f"{pred.get('plan_type', 'prediction')}_prompt_args"
        metadata = dict(row.get("metadata") or {})
        metadata["argument_recovery"] = "state_schema_v1" if use_state else "prompt_schema_regex_v1"
        if state_params:
            metadata["state_param_keys"] = sorted(state_params)
        row["metadata"] = metadata
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Enrich decoded tool-path predictions with prompt/schema regex arguments.")
    parser.add_argument("--pred", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tasks", default=str(ROOT / "data" / "processed" / "tasks.jsonl"))
    parser.add_argument("--tools", default=str(ROOT / "data" / "processed" / "tools.jsonl"))
    parser.add_argument("--use-state", action="store_true")
    args = parser.parse_args()

    predictions = read_jsonl(args.pred)
    tasks = read_jsonl(args.tasks)
    tools = {tool["tool_id"]: tool for tool in read_jsonl(args.tools)}
    rows = enrich(predictions, tasks, tools, use_state=args.use_state)
    write_jsonl(Path(args.out), rows)
    print(json.dumps({"rows": len(rows), "out": args.out}, ensure_ascii=False))


if __name__ == "__main__":
    main()
