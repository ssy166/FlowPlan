"""
Convert FM SFT model chat output to prediction JSONL for evaluate_plans.py.

The input is the fm_sft/dev.jsonl (or test.jsonl) with model-generated assistant messages
injected into a "generation" field, or a separate raw generation file.

Two usage modes:

1. Inline mode: the fm_sft JSONL already has a "generation" field added by the inference run.
   python scripts/convert_sft_predictions.py \
       --input data/processed/fm_sft/dev.jsonl \
       --tools data/processed/tools.jsonl \
       --out data/processed/predictions/sft_lora_v1.dev.jsonl \
       --model-name sft_lora_v1

2. Separate raw file mode: raw generations are in a separate JSONL with {"id": ..., "generation": ...}.
   python scripts/convert_sft_predictions.py \
       --input data/processed/fm_sft/dev.jsonl \
       --raw data/processed/predictions/sft_lora_v1.dev_raw.jsonl \
       --tools data/processed/tools.jsonl \
       --out data/processed/predictions/sft_lora_v1.dev.jsonl \
       --model-name sft_lora_v1
"""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def extract_json_object(text):
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except Exception:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start: end + 1])
        except Exception:
            return None
    return None


def fallback_tool_names(text, by_name):
    hits = []
    lowered = text.lower()
    for name in by_name:
        if name.lower() in lowered:
            hits.append(name)
    return hits


def normalize_actions(generation, available_tool_ids, tools):
    by_name = {}
    for tool_id in available_tool_ids:
        tool = tools.get(tool_id) or {}
        name = tool.get("name") or tool_id.rsplit("::", 1)[-1]
        by_name.setdefault(name, tool_id)
        by_name.setdefault(tool_id, tool_id)

    parsed = extract_json_object(generation)
    raw_actions = []
    if isinstance(parsed, dict):
        raw_actions = parsed.get("actions") or parsed.get("tool_calls") or []
    elif isinstance(parsed, list):
        raw_actions = parsed

    actions = []
    if isinstance(raw_actions, list):
        for raw in raw_actions:
            if not isinstance(raw, dict):
                continue
            name = raw.get("tool_name") or raw.get("name") or raw.get("tool") or raw.get("tool_id")
            tool_id = by_name.get(name)
            if not tool_id:
                continue
            actions.append({
                "tool_id": tool_id,
                "tool_name": tools.get(tool_id, {}).get("name") or tool_id.rsplit("::", 1)[-1],
                "arguments": raw.get("arguments") or raw.get("args") or {},
            })

    if not actions:
        for name in fallback_tool_names(generation, by_name):
            tool_id = by_name[name]
            actions.append({
                "tool_id": tool_id,
                "tool_name": tools.get(tool_id, {}).get("name") or tool_id.rsplit("::", 1)[-1],
                "arguments": {},
            })

    seen = set()
    unique = []
    for action in actions:
        key = (action["tool_id"], json.dumps(action.get("arguments") or {}, sort_keys=True))
        if key not in seen:
            seen.add(key)
            unique.append(action)
    return unique


def get_available_tool_ids(sft_row, tools_by_name):
    """Extract available tool IDs from the SFT row's user message JSON spec."""
    user_content = ""
    for msg in sft_row.get("messages") or []:
        if msg.get("role") == "user":
            user_content = msg.get("content") or ""
            break
    # The user content ends with the JSON spec — find the last JSON object
    start = user_content.rfind('{"task_id"')
    if start < 0:
        start = user_content.find("{")
    if start >= 0:
        try:
            spec = json.loads(user_content[start:])
            available = spec.get("available_tools") or []
            ids = []
            for t in available:
                tid = t.get("tool_id")
                if tid:
                    ids.append(tid)
                else:
                    # try lookup by name
                    name = t.get("name")
                    if name and name in tools_by_name:
                        ids.append(tools_by_name[name])
            return ids
        except Exception:
            pass
    return []


def main():
    parser = argparse.ArgumentParser(description="Convert SFT model output to prediction JSONL.")
    parser.add_argument("--input", required=True, help="fm_sft split JSONL (with messages field)")
    parser.add_argument("--raw", default=None, help="Separate raw generation JSONL {id, generation}")
    parser.add_argument("--tools", default=str(ROOT / "data" / "processed" / "tools.jsonl"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-name", default="sft_lora")
    args = parser.parse_args()

    tools = {row["tool_id"]: row for row in read_jsonl(args.tools)}
    tools_by_name = {}
    for tool_id, tool in tools.items():
        name = tool.get("name") or tool_id.rsplit("::", 1)[-1]
        tools_by_name.setdefault(name, tool_id)

    sft_rows = read_jsonl(args.input)

    # Build generation map
    gen_map = {}
    if args.raw:
        for row in read_jsonl(args.raw):
            gen_map[row["id"]] = row.get("generation") or ""
    else:
        for row in sft_rows:
            if "generation" in row:
                gen_map[row["id"]] = row["generation"]

    predictions = []
    skipped = 0
    for row in sft_rows:
        row_id = row.get("id") or row.get("task_id") or ""
        generation = gen_map.get(row_id)
        if generation is None:
            skipped += 1
            continue

        available_tool_ids = get_available_tool_ids(row, tools_by_name)
        metadata = row.get("metadata") or {}
        source = metadata.get("source") or ""

        actions = normalize_actions(generation, available_tool_ids, tools)
        predictions.append({
            "task_id": row_id,
            "source": source,
            "plan_type": f"model_{args.model_name}",
            "tool_ids": [a["tool_id"] for a in actions],
            "tool_names": [a["tool_name"] for a in actions],
            "phase_names": [tools.get(a["tool_id"], {}).get("phase", "unknown") for a in actions],
            "actions": actions,
        })

    write_jsonl(args.out, predictions)
    print(json.dumps({
        "out": args.out,
        "rows": len(predictions),
        "skipped_no_generation": skipped,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
