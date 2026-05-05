import argparse
import json
from pathlib import Path


def read_jsonl(path, limit=0):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def validate_target(row):
    target = json.loads(row["messages"][-1]["content"])
    if not isinstance(target.get("stop"), bool):
        raise ValueError("target.stop must be bool")
    if not isinstance(target.get("actions"), list):
        raise ValueError("target.actions must be list")
    for action in target["actions"]:
        if not isinstance(action.get("tool_name"), str):
            raise ValueError("action.tool_name must be str")
        if not isinstance(action.get("arguments"), dict):
            raise ValueError("action.arguments must be dict")
    return target


def token_lengths(rows, model_path):
    if not model_path:
        return None
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    lengths = []
    for row in rows:
        messages = row["messages"]
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        else:
            text = "\n".join(f"<|{m['role'].upper()}|>\n{m['content']}" for m in messages)
        lengths.append(len(tokenizer(text, add_special_tokens=False)["input_ids"]))
    lengths.sort()
    if not lengths:
        return {"count": 0}
    return {
        "count": len(lengths),
        "min": lengths[0],
        "p50": lengths[len(lengths) // 2],
        "p90": lengths[int(len(lengths) * 0.9)],
        "p95": lengths[int(len(lengths) * 0.95)],
        "max": lengths[-1],
    }


def main():
    parser = argparse.ArgumentParser(description="Validate FM-conditioned SFT data and optionally run tokenizer length smoke.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model-path", default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    report = {}
    for split in args.splits:
        rows = read_jsonl(data_dir / f"{split}.jsonl", limit=args.limit)
        stats = {
            "rows_checked": len(rows),
            "json_parse_errors": 0,
            "bad_schema": 0,
            "stop_rows": 0,
            "action_rows": 0,
            "max_user_chars": 0,
            "max_target_chars": 0,
        }
        for row in rows:
            try:
                target = validate_target(row)
            except Exception:
                stats["bad_schema"] += 1
                continue
            if target["stop"]:
                stats["stop_rows"] += 1
            else:
                stats["action_rows"] += 1
            stats["max_user_chars"] = max(stats["max_user_chars"], len(row["messages"][0]["content"]))
            stats["max_target_chars"] = max(stats["max_target_chars"], len(row["messages"][-1]["content"]))
        if args.model_path:
            stats["token_lengths"] = token_lengths(rows, args.model_path)
        report[split] = stats
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
