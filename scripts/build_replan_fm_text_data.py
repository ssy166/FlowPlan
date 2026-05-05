import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SFT_DIR = ROOT / "data" / "processed" / "fm_replan_sft_next"
DEFAULT_OUT_DIR = ROOT / "data" / "processed" / "replan_fm_text"
STOP_TOOL = "<stop>"


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


def parse_target(row):
    target = json.loads(row["messages"][-1]["content"])
    actions = target.get("actions") or []
    tool_ids = [action.get("tool_id") for action in actions if action.get("tool_id")]
    if target.get("stop") or not tool_ids:
        return [STOP_TOOL], STOP_TOOL
    return tool_ids, tool_ids[0]


def normalize_row(row):
    metadata = row.get("metadata") or {}
    future_tools, next_tool = parse_target(row)
    record_id = row.get("id") or metadata.get("record_id")
    return {
        "record_id": record_id,
        "trajectory_id": metadata.get("task_id") or record_id,
        "task_id": metadata.get("task_id") or record_id,
        "source": metadata.get("source") or "",
        "domain": metadata.get("domain") or "",
        "split": metadata.get("split") or "",
        "step_idx": int(metadata.get("replan_step_idx") or 0),
        "total_steps": 1,
        "input_text": row["messages"][0]["content"],
        "gold_tool_call_text": row["messages"][-1]["content"],
        "future_tools": future_tools,
        "next_tool": next_tool,
        "available_tools": [],
        "metadata": {
            "record_id": record_id,
            "conditioning": metadata.get("conditioning"),
            "stop": metadata.get("stop"),
            "target_remaining_count": metadata.get("target_remaining_count"),
            "replan_reason": metadata.get("replan_reason") or {},
        },
    }


def write_readme(out_dir, manifest):
    lines = [
        "# Replan FM Text Data",
        "",
        "FM text records derived from feedback-replan next-action SFT rows.",
        "",
        "`future_tools` is the next target tool, or `<stop>` for stop rows, so encoder-derived `y_i` can represent stop decisions.",
        "After running ToolRL's encoder extractor, run `scripts/attach_record_ids_to_fm_tensor.py` so SFT rows can align with tensor samples by `record_id`.",
        "",
        "## Counts",
        "",
    ]
    for split, info in manifest["splits"].items():
        lines.append(f"- {split}: {info['records']} records at `{info['path']}`")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Build FM text JSONL records from replan next-action SFT rows.")
    parser.add_argument("--sft-dir", default=str(DEFAULT_SFT_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    args = parser.parse_args()

    sft_dir = Path(args.sft_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"format": "replan_fm_text_v1", "source_dir": str(sft_dir), "splits": {}}
    for split in args.splits:
        rows = [normalize_row(row) for row in read_jsonl(sft_dir / f"{split}.jsonl")]
        path = out_dir / f"{split}.jsonl"
        write_jsonl(path, rows)
        manifest["splits"][split] = {"path": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path), "records": len(rows)}
        print(f"{split}: wrote {path} with {len(rows)} records")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_readme(out_dir, manifest)


if __name__ == "__main__":
    main()
