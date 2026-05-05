import argparse
import json
import re
from pathlib import Path


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


def load_tools(path):
    return {row["tool_id"]: row for row in read_jsonl(path)}


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
            return json.loads(text[start : end + 1])
        except Exception:
            return None
    return None


def fallback_tool_names(text, available_names):
    hits = []
    lowered = text.lower()
    for name in available_names:
        if name.lower() in lowered:
            hits.append(name)
    return hits


def normalize_actions(generation, pack_row, tools):
    available_ids = pack_row.get("available_tool_ids") or []
    by_name = {}
    for tool_id in available_ids:
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
            actions.append(
                {
                    "tool_id": tool_id,
                    "tool_name": tools.get(tool_id, {}).get("name") or tool_id.rsplit("::", 1)[-1],
                    "arguments": raw.get("arguments") or raw.get("args") or {},
                }
            )

    if not actions:
        for name in fallback_tool_names(generation, by_name):
            tool_id = by_name[name]
            if tool_id not in available_ids:
                continue
            actions.append(
                {
                    "tool_id": tool_id,
                    "tool_name": tools.get(tool_id, {}).get("name") or tool_id.rsplit("::", 1)[-1],
                    "arguments": {},
                }
            )

    seen = set()
    unique_actions = []
    for action in actions:
        key = (action["tool_id"], json.dumps(action.get("arguments") or {}, sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        unique_actions.append(action)
    return unique_actions


def prediction_row(pack_row, generation, tools, model_name):
    actions = normalize_actions(generation, pack_row, tools)
    return {
        "task_id": pack_row["task_id"],
        "source": pack_row.get("source"),
        "plan_type": f"model_{model_name}",
        "tool_ids": [action["tool_id"] for action in actions],
        "tool_names": [action["tool_name"] for action in actions],
        "phase_names": [tools.get(action["tool_id"], {}).get("phase", "unknown") for action in actions],
        "actions": actions,
    }


def load_model(model_path, dtype):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # decoder-only models need left-padding for batch inference
    torch_dtype = {"auto": "auto", "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


def render_prompt(tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def generate_batch(tokenizer, model, prompts, max_new_tokens):
    import torch

    rendered = [render_prompt(tokenizer, prompt) for prompt in prompts]
    inputs = tokenizer(rendered, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[:, inputs["input_ids"].shape[1] :]
    return tokenizer.batch_decode(generated, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Run ToolRL/Qwen-style local HF inference for model eval pack.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input", default=str(ROOT / "data" / "processed" / "model_eval_pack.jsonl"))
    parser.add_argument("--tools", default=str(ROOT / "data" / "processed" / "tools.jsonl"))
    parser.add_argument("--out", default=str(ROOT / "data" / "processed" / "predictions" / "toolrl_qwen25_7b.jsonl"))
    parser.add_argument("--raw-out", default=str(ROOT / "data" / "processed" / "predictions" / "toolrl_qwen25_7b_raw.jsonl"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--model-name", default="toolrl_qwen25_7b")
    args = parser.parse_args()

    pack = read_jsonl(args.input)
    if args.limit:
        pack = pack[: args.limit]
    tools = load_tools(args.tools)
    tokenizer, model = load_model(args.model_path, args.dtype)

    predictions = []
    raw_rows = []
    for start in range(0, len(pack), args.batch_size):
        batch = pack[start : start + args.batch_size]
        generations = generate_batch(tokenizer, model, [row["prompt"] for row in batch], args.max_new_tokens)
        for row, generation in zip(batch, generations):
            raw_rows.append({"task_id": row["task_id"], "generation": generation})
            predictions.append(prediction_row(row, generation, tools, args.model_name))
        print(json.dumps({"done": min(start + len(batch), len(pack)), "total": len(pack)}, ensure_ascii=False), flush=True)

    write_jsonl(args.raw_out, raw_rows)
    write_jsonl(args.out, predictions)
    print(json.dumps({"out": args.out, "raw_out": args.raw_out, "rows": len(predictions)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
