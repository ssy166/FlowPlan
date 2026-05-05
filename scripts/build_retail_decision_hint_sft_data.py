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


def decision_hint(decision):
    target = decision.get("target") or {}
    progress = decision.get("grounding_progress") or {}
    payload = {
        "decision_case": decision.get("decision_case"),
        "target_stage": target.get("stage"),
        "lookup_completed": progress.get("lookup_completed"),
        "order_grounded": progress.get("order_grounded"),
        "resolved_order_status": progress.get("resolved_order_status"),
        "operation_eligibility": progress.get("operation_eligibility") or [],
    }
    return "Retail decision prior:\n" + json.dumps(payload, ensure_ascii=False, sort_keys=True)


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
    by_case = {}
    for row in input_rows:
        decision = decisions.get(row.get("id"))
        if not decision:
            out.append(row)
            continue
        case = decision.get("decision_case") or "unknown"
        by_case[case] = by_case.get(case, 0) + 1
        hinted += 1
        out.append(apply_hint(row, decision_hint(decision)))
    return out, {"rows": len(out), "hinted": hinted, "by_decision_case": by_case}


def main():
    parser = argparse.ArgumentParser(description="Append oracle retail decision/stage hints to replan SFT rows.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--decision-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    manifest = {
        "input_dir": args.input_dir,
        "decision_dir": args.decision_dir,
        "hint": "oracle_retail_decision_case_and_target_stage_without_tool_id",
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
