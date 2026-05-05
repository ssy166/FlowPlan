import argparse
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def find_balanced_json(text, start=0):
    first = text.find("{", start)
    if first < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for idx in range(first, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[first : idx + 1]
    return None


def extract_json_object(text):
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    fenced_start = text.find("```")
    if fenced_start >= 0:
        fenced_json = find_balanced_json(text, fenced_start)
        if fenced_json:
            try:
                return json.loads(fenced_json)
            except Exception:
                pass
    candidate = find_balanced_json(text, text.find("{"))
    if candidate:
        try:
            return json.loads(candidate)
        except Exception:
            pass
    return None


def parse_conditioned_state(row):
    content = ""
    messages = row.get("messages") or row.get("prompt") or []
    for message in messages:
        if message.get("role") == "user":
            content = message.get("content") or ""
            break
    marker = "Conditioned workflow state:"
    start = content.find(marker)
    start = content.find("{", start) if start >= 0 else content.find("{")
    candidate = find_balanced_json(content, start)
    if not candidate:
        return {}
    try:
        return json.loads(candidate)
    except Exception:
        return {}


def reward_ground_truth(row):
    reward_model = row.get("reward_model") or {}
    ground_truth = reward_model.get("ground_truth")
    if not ground_truth:
        return {}
    if isinstance(ground_truth, dict):
        return ground_truth
    try:
        return json.loads(ground_truth)
    except Exception:
        return {}


def row_metadata(row):
    metadata = row.get("metadata") or row.get("extra_info") or {}
    gt = reward_ground_truth(row)
    if gt:
        merged = dict(gt)
        merged.update(metadata)
        return merged
    return metadata


def row_id(row):
    metadata = row_metadata(row)
    return row.get("id") or metadata.get("id") or metadata.get("record_id") or metadata.get("task_id")


def gold_from_assistant(row):
    gt = reward_ground_truth(row)
    if gt:
        actions = gt.get("gold_actions")
        stop = gt.get("gold_stop")
        if actions is not None or stop is not None:
            return actions or [], bool(stop)
        target = extract_json_object(gt.get("target_json") or "")
        if isinstance(target, dict):
            return target.get("actions") or [], bool(target.get("stop", False))
    for message in reversed(row.get("messages") or []):
        if message.get("role") == "assistant":
            parsed = extract_json_object(message.get("content") or "")
            if isinstance(parsed, dict):
                return parsed.get("actions") or [], bool(parsed.get("stop", False))
    return [], False


def prompt_messages(row):
    messages = row.get("messages") or row.get("prompt") or []
    cut = len(messages)
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") == "assistant":
            cut = idx
            break
    return messages[:cut]


def render_prompt(tokenizer, row):
    messages = prompt_messages(row)
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "\n\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages) + "\n\nassistant:"


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def normalize_value(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def argument_scores(pred_actions, gold_actions):
    gold_key_count = 0
    pred_key_count = 0
    key_hits = 0
    value_hits = 0
    comparable = 0
    for idx, gold_action in enumerate(gold_actions):
        pred_action = pred_actions[idx] if idx < len(pred_actions) else {}
        gold_args = gold_action.get("arguments") or {}
        pred_args = pred_action.get("arguments") or {}
        if not isinstance(gold_args, dict):
            gold_args = {}
        if not isinstance(pred_args, dict):
            pred_args = {}
        gold_keys = set(gold_args)
        pred_keys = set(pred_args)
        gold_key_count += len(gold_keys)
        pred_key_count += len(pred_keys)
        matched = gold_keys & pred_keys
        key_hits += len(matched)
        comparable += len(gold_keys)
        for key in matched:
            if normalize_value(pred_args.get(key)) == normalize_value(gold_args.get(key)):
                value_hits += 1
    return {
        "gold_argument_count": gold_key_count,
        "predicted_argument_count": pred_key_count,
        "argument_key_hits": key_hits,
        "argument_value_hits": value_hits,
        "argument_key_recall": key_hits / gold_key_count if gold_key_count else None,
        "argument_key_precision": key_hits / pred_key_count if pred_key_count else None,
        "argument_value_exact_match": value_hits / comparable if comparable else None,
    }


def available_tool_maps(row, tools):
    state = parse_conditioned_state(row)
    gt = reward_ground_truth(row)
    available = state.get("available_tools") or gt.get("available_tools") or []
    by_name = {}
    available_ids = []
    for item in available:
        tool_id = item.get("tool_id")
        name = item.get("name") or (tool_id.rsplit("::", 1)[-1] if tool_id else None)
        if not tool_id:
            continue
        available_ids.append(tool_id)
        by_name.setdefault(tool_id, tool_id)
        if name:
            by_name.setdefault(name, tool_id)
    if not available_ids:
        metadata = row_metadata(row)
        source = metadata.get("source")
        domain = metadata.get("domain")
        for tool_id, tool in tools.items():
            if source and not tool_id.startswith(f"{source}::"):
                continue
            if domain and f"::{domain}::" not in tool_id:
                continue
            available_ids.append(tool_id)
            by_name.setdefault(tool_id, tool_id)
            by_name.setdefault(tool.get("name") or tool_id.rsplit("::", 1)[-1], tool_id)
    return available_ids, by_name


def normalize_actions(parsed, row, tools):
    available_ids, by_name = available_tool_maps(row, tools)
    available_set = set(available_ids)
    if isinstance(parsed, dict):
        raw_actions = parsed.get("actions") or parsed.get("tool_calls") or []
        stop = bool(parsed.get("stop", False))
    elif isinstance(parsed, list):
        raw_actions = parsed
        stop = False
    else:
        raw_actions = []
        stop = False
    actions = []
    invalid = []
    if isinstance(raw_actions, list):
        for raw in raw_actions:
            if not isinstance(raw, dict):
                invalid.append(str(raw))
                continue
            name = raw.get("tool_id") or raw.get("tool_name") or raw.get("name") or raw.get("tool")
            tool_id = by_name.get(name) or (name if name in available_set else None)
            if not tool_id:
                invalid.append(str(name))
                continue
            args = raw.get("arguments")
            if args is None:
                args = raw.get("args") or {}
            if not isinstance(args, dict):
                args = {}
            tool = tools.get(tool_id) or {}
            actions.append(
                {
                    "tool_id": tool_id,
                    "tool_name": tool.get("name") or tool_id.rsplit("::", 1)[-1],
                    "arguments": args,
                }
            )
    return actions, stop, invalid


def summarize_rows(rows):
    n = len(rows)
    if not n:
        return {"count": 0}
    arg_recall = [row["argument_key_recall"] for row in rows if row.get("argument_key_recall") is not None]
    arg_precision = [row["argument_key_precision"] for row in rows if row.get("argument_key_precision") is not None]
    arg_value = [row["argument_value_exact_match"] for row in rows if row.get("argument_value_exact_match") is not None]
    return {
        "count": n,
        "json_parse_rate": sum(row["json_parse_ok"] for row in rows) / n,
        "schema_action_rate": sum(row["schema_action_ok"] for row in rows) / n,
        "valid_available_tool_rate": sum(row["available_tool_ok"] for row in rows) / n,
        "stop_exact_match": sum(row["stop_exact_match"] for row in rows) / n,
        "tool_exact_match": sum(row["tool_exact_match"] for row in rows) / n,
        "avg_tool_edit_distance": sum(row["tool_edit_distance"] for row in rows) / n,
        "avg_predicted_tool_count": sum(row["predicted_tool_count"] for row in rows) / n,
        "avg_gold_tool_count": sum(row["gold_tool_count"] for row in rows) / n,
        "argument_task_count": len(arg_recall),
        "avg_argument_key_recall": sum(arg_recall) / len(arg_recall) if arg_recall else None,
        "avg_argument_key_precision": sum(arg_precision) / len(arg_precision) if arg_precision else None,
        "avg_argument_value_exact_match": sum(arg_value) / len(arg_value) if arg_value else None,
    }


def summarize_groups(rows, key):
    groups = {}
    for row in rows:
        groups.setdefault(str(row.get(key)), []).append(row)
    return {group: summarize_rows(items) for group, items in sorted(groups.items())}


def pick_dtype(name):
    return {"auto": "auto", "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def load_model(model_path, adapter_dir, dtype):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=pick_dtype(dtype),
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def generate_batch(tokenizer, model, prompts, max_new_tokens):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    outputs = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated = outputs[:, inputs["input_ids"].shape[1] :]
    return tokenizer.batch_decode(generated, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a ToolRL LoRA adapter on benchmark replan SFT rows.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--data", default=str(ROOT / "data" / "processed" / "fm_replan_sft_next" / "dev.jsonl"))
    parser.add_argument("--tools", default=str(ROOT / "data" / "processed" / "tools.jsonl"))
    parser.add_argument("--out-raw", required=True)
    parser.add_argument("--out-pred", required=True)
    parser.add_argument("--out-details", default=None)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--model-name", default="toolrl_lora_grpo")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="bf16")
    args = parser.parse_args()

    rows = read_jsonl(args.data)
    if args.limit:
        rows = rows[: args.limit]
    tools = {row["tool_id"]: row for row in read_jsonl(args.tools)}
    tokenizer, model = load_model(args.model_path, args.adapter_dir, args.dtype)

    raw_rows = []
    pred_rows = []
    detail_rows = []
    prompts = [render_prompt(tokenizer, row) for row in rows]
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        generations = generate_batch(tokenizer, model, prompts[start : start + args.batch_size], args.max_new_tokens)
        for row, generation in zip(batch_rows, generations):
            parsed = extract_json_object(generation)
            actions, stop, invalid_tools = normalize_actions(parsed, row, tools)
            gold_actions, gold_stop = gold_from_assistant(row)
            pred_tool_ids = [action["tool_id"] for action in actions]
            gold_tool_ids = [action.get("tool_id") for action in gold_actions]
            arg = argument_scores(actions, gold_actions)
            metadata = row_metadata(row)
            rid = row_id(row)
            detail = {
                "id": rid,
                "task_id": metadata.get("task_id") or rid,
                "source": metadata.get("source"),
                "domain": metadata.get("domain"),
                "step_idx": metadata.get("step_idx"),
                "json_parse_ok": isinstance(parsed, (dict, list)),
                "schema_action_ok": isinstance(parsed, dict) and isinstance(parsed.get("actions"), list),
                "available_tool_ok": not invalid_tools,
                "invalid_tools": invalid_tools,
                "stop_exact_match": stop == gold_stop,
                "tool_exact_match": pred_tool_ids == gold_tool_ids,
                "tool_edit_distance": edit_distance(pred_tool_ids, gold_tool_ids),
                "predicted_tool_count": len(pred_tool_ids),
                "gold_tool_count": len(gold_tool_ids),
                **arg,
            }
            raw_rows.append(
                {
                    "id": rid,
                    "task_id": detail["task_id"],
                    "generation": generation,
                    "parsed": parsed,
                    "gold": {"stop": gold_stop, "actions": gold_actions},
                    "metadata": metadata,
                }
            )
            pred_rows.append(
                {
                    "task_id": detail["task_id"],
                    "record_id": rid,
                    "source": metadata.get("source"),
                    "domain": metadata.get("domain"),
                    "plan_type": f"model_{args.model_name}",
                    "tool_ids": pred_tool_ids,
                    "tool_names": [action["tool_name"] for action in actions],
                    "phase_names": [tools.get(action["tool_id"], {}).get("phase", "unknown") for action in actions],
                    "actions": actions,
                    "metadata": {"record_id": row.get("id"), "step_idx": metadata.get("step_idx")},
                }
            )
            detail_rows.append(detail)
        print(json.dumps({"generated": min(start + len(batch_rows), len(rows)), "rows": len(rows)}, ensure_ascii=False), flush=True)

    summary = {
        "model_name": args.model_name,
        "adapter_dir": args.adapter_dir,
        "data": args.data,
        "limit": args.limit,
        "overall": summarize_rows(detail_rows),
        "by_source": summarize_groups(detail_rows, "source"),
        "by_domain": summarize_groups(detail_rows, "domain"),
    }
    write_jsonl(args.out_raw, raw_rows)
    write_jsonl(args.out_pred, pred_rows)
    if args.out_details:
        write_jsonl(args.out_details, detail_rows)
    write_json(args.summary_out, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
