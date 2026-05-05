import argparse
import json
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


def parse_domains(raw):
    return {part.strip() for part in (raw or "").split(",") if part.strip()}


def main():
    parser = argparse.ArgumentParser(description="Mix two prediction files by source/domain gates.")
    parser.add_argument("--base", required=True, help="Default prediction file.")
    parser.add_argument("--override", required=True, help="Prediction file used for selected domains/sources.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--tasks", default=str(ROOT / "data" / "processed" / "tasks.jsonl"))
    parser.add_argument("--override-domains", default="")
    parser.add_argument("--override-sources", default="")
    parser.add_argument("--plan-type", default=None)
    args = parser.parse_args()

    tasks = {row["task_id"]: row for row in read_jsonl(args.tasks)}
    override_by_id = {row["task_id"]: row for row in read_jsonl(args.override)}
    override_domains = parse_domains(args.override_domains)
    override_sources = parse_domains(args.override_sources)
    rows = []
    counts = {"base": 0, "override": 0}

    for base_row in read_jsonl(args.base):
        task = tasks.get(base_row["task_id"], {})
        use_override = task.get("domain") in override_domains or task.get("source") in override_sources
        row = dict(override_by_id.get(base_row["task_id"], base_row) if use_override else base_row)
        metadata = dict(row.get("metadata") or {})
        metadata["mix_base"] = Path(args.base).name
        metadata["mix_override"] = Path(args.override).name
        metadata["mix_route"] = "override" if use_override else "base"
        row["metadata"] = metadata
        if args.plan_type:
            row["plan_type"] = args.plan_type
        rows.append(row)
        counts["override" if use_override else "base"] += 1

    write_jsonl(args.out, rows)
    print(json.dumps({"out": args.out, "rows": len(rows), **counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
