import argparse
import json
from pathlib import Path


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
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def operation_hint(decision):
    target = decision.get("target") or {}
    progress = decision.get("grounding_progress") or {}
    payload = {
        "decision_case": decision.get("decision_case"),
        "target_stage": target.get("stage"),
        "target_tool_id": target.get("tool_id"),
        "target_tool_name": target.get("tool_name"),
        "operation_eligibility": progress.get("operation_eligibility") or [],
        "resolved_order_status": progress.get("resolved_order_status"),
    }
    return "Oracle retail operation-tool prior:\n" + json.dumps(payload, ensure_ascii=False, sort_keys=True)


def apply_hint(row, hint):
    row = json.loads(json.dumps(row, ensure_ascii=False))
    for message in reversed(row.get("messages") or []):
        if message.get("role") == "user":
            message["content"] = (message.get("content") or "") + "\n\n" + hint
            return row
    row.setdefault("messages", []).insert(-1, {"role": "user", "content": hint})
    return row


def build_split(input_rows, decision_rows):
    decisions = {row["id"]: row for row in decision_rows}
    out = []
    hinted = 0
    by_tool = {}
    for row in input_rows:
        decision = decisions.get(row.get("id"))
        if not decision or decision.get("decision_case") != "post_order_operation":
            out.append(row)
            continue
        tool_name = (decision.get("target") or {}).get("tool_name") or "unknown"
        by_tool[tool_name] = by_tool.get(tool_name, 0) + 1
        hinted += 1
        out.append(apply_hint(row, operation_hint(decision)))
    return out, {"rows": len(out), "hinted": hinted, "by_operation_tool": by_tool}


def main():
    parser = argparse.ArgumentParser(description="Append oracle exact retail operation-tool hints to operation-ready rows only.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--decision-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    manifest = {
        "input_dir": args.input_dir,
        "decision_dir": args.decision_dir,
        "hint": "oracle_exact_retail_operation_tool_for_post_order_operation_rows_only",
        "splits": {},
    }
    for split in args.splits:
        rows, stats = build_split(
            read_jsonl(Path(args.input_dir) / f"{split}.jsonl"),
            read_jsonl(Path(args.decision_dir) / f"{split}.jsonl"),
        )
        write_jsonl(out_dir / f"{split}.jsonl", rows)
        manifest["splits"][split] = stats
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
