import argparse
import json
from pathlib import Path

import torch


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def attach(input_pt, text_jsonl, output_pt):
    payload = torch_load(input_pt)
    rows = read_jsonl(text_jsonl)
    samples = payload.get("samples") if isinstance(payload, dict) else payload
    if not isinstance(samples, list):
        raise ValueError(f"{input_pt} must contain a sample list or a dict with samples.")
    if len(samples) != len(rows):
        raise ValueError(f"sample/text length mismatch: {len(samples)} != {len(rows)}")
    for sample, row in zip(samples, rows):
        sample["record_id"] = row["record_id"]
        sample.setdefault("metadata", {})
        sample["metadata"]["record_id"] = row["record_id"]
        sample["metadata"]["task_id"] = row.get("task_id")
        sample["metadata"]["domain"] = row.get("domain")
        sample["metadata"]["split"] = row.get("split")
    if isinstance(payload, dict):
        payload["samples"] = samples
        payload.setdefault("metadata", {})
        payload["metadata"]["record_ids_attached"] = True
        payload["metadata"]["record_id_source_jsonl"] = str(text_jsonl)
        out = payload
    else:
        out = samples
    output_pt = Path(output_pt)
    output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, output_pt)
    return len(samples)


def main():
    parser = argparse.ArgumentParser(description="Attach record_id fields to FM tensor samples using the source text JSONL order.")
    parser.add_argument("--input-pt", required=True)
    parser.add_argument("--text-jsonl", required=True)
    parser.add_argument("--output-pt", required=True)
    args = parser.parse_args()
    count = attach(args.input_pt, args.text_jsonl, args.output_pt)
    print(json.dumps({"output_pt": args.output_pt, "samples": count}, ensure_ascii=False))


if __name__ == "__main__":
    main()
