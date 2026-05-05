import argparse
import json
import random
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


def parse_multiplier(items):
    out = {}
    for item in items or []:
        key, value = item.split("=", 1)
        out[key] = int(value)
    return out


def load_decision_cases(path):
    return {row["id"]: row.get("decision_case") for row in read_jsonl(path)}


def clone_row(row, copy_idx, reason):
    cloned = json.loads(json.dumps(row, ensure_ascii=False))
    cloned["id"] = f"{row.get('id')}::oversample::{reason}::{copy_idx}"
    metadata = cloned.setdefault("metadata", {})
    metadata["oversampled_from_id"] = row.get("id")
    metadata["oversample_reason"] = reason
    metadata["oversample_copy_idx"] = copy_idx
    return cloned


def build_train(rows, decision_cases, multipliers, seed):
    out = []
    stats = {"input_rows": len(rows), "output_rows": 0, "extra_rows": 0, "by_decision_case": {}, "extra_by_decision_case": {}}
    for row in rows:
        out.append(row)
        row_id = row.get("id")
        decision_case = decision_cases.get(row_id)
        stats["by_decision_case"][decision_case or "non_retail_or_unknown"] = stats["by_decision_case"].get(decision_case or "non_retail_or_unknown", 0) + 1
        copies = max(1, int(multipliers.get(decision_case, 1)))
        for copy_idx in range(1, copies):
            out.append(clone_row(row, copy_idx, decision_case or "unknown"))
            stats["extra_rows"] += 1
            stats["extra_by_decision_case"][decision_case or "unknown"] = stats["extra_by_decision_case"].get(decision_case or "unknown", 0) + 1
    random.Random(seed).shuffle(out)
    stats["output_rows"] = len(out)
    return out, stats


def main():
    parser = argparse.ArgumentParser(description="Oversample SFT train rows by retail decision case.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--decision-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--case-multiplier", action="append", default=[])
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    multipliers = parse_multiplier(args.case_multiplier)
    input_dir = Path(args.input_dir)
    decision_dir = Path(args.decision_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "input_dir": str(input_dir),
        "decision_dir": str(decision_dir),
        "case_multipliers": multipliers,
        "seed": args.seed,
        "splits": {},
    }
    for split in ["train", "dev", "test"]:
        rows = read_jsonl(input_dir / f"{split}.jsonl")
        if split == "train":
            decision_cases = load_decision_cases(decision_dir / "train.jsonl")
            rows, stats = build_train(rows, decision_cases, multipliers, args.seed)
        else:
            stats = {"input_rows": len(rows), "output_rows": len(rows), "extra_rows": 0}
        write_jsonl(out_dir / f"{split}.jsonl", rows)
        manifest["splits"][split] = stats
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
