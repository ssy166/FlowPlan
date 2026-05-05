import argparse
import json
from pathlib import Path


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
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


def row_key(row):
    return row.get("id") or (row.get("metadata") or {}).get("record_id")


def tag_rows(rows, source_name):
    out = []
    for row in rows:
        row = json.loads(json.dumps(row, ensure_ascii=False))
        metadata = dict(row.get("metadata") or {})
        metadata["sft_source_pack"] = source_name
        row["metadata"] = metadata
        out.append(row)
    return out


def main():
    parser = argparse.ArgumentParser(description="Merge chat SFT split directories with optional dedup by row id.")
    parser.add_argument("--input", action="append", nargs=2, metavar=("NAME", "DIR"), required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--dedup", action="store_true")
    parser.add_argument(
        "--dedup-policy",
        choices=["first", "last"],
        default="first",
        help="When --dedup is set, keep the first occurrence or let later inputs replace earlier rows.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"inputs": [{"name": name, "dir": directory} for name, directory in args.input], "splits": {}}
    for split in args.splits:
        merged = []
        input_counts = {}
        seen = set()
        key_to_index = {}
        duplicate_count = 0
        for name, directory in args.input:
            rows = tag_rows(read_jsonl(Path(directory) / f"{split}.jsonl"), name)
            input_counts[name] = len(rows)
            for row in rows:
                key = row_key(row)
                if args.dedup and key in seen:
                    duplicate_count += 1
                    if args.dedup_policy == "last" and key in key_to_index:
                        merged[key_to_index[key]] = row
                    continue
                if key:
                    seen.add(key)
                    key_to_index[key] = len(merged)
                merged.append(row)
        write_jsonl(out_dir / f"{split}.jsonl", merged)
        by_domain = {}
        by_source_pack = {}
        for row in merged:
            md = row.get("metadata") or {}
            domain = md.get("domain") or "unknown"
            pack = md.get("sft_source_pack") or "unknown"
            by_domain[domain] = by_domain.get(domain, 0) + 1
            by_source_pack[pack] = by_source_pack.get(pack, 0) + 1
        manifest["splits"][split] = {
            "rows": len(merged),
            "input_counts": input_counts,
            "duplicate_count": duplicate_count,
            "dedup_policy": args.dedup_policy if args.dedup else None,
            "domain_counts": by_domain,
            "source_pack_counts": by_source_pack,
        }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
