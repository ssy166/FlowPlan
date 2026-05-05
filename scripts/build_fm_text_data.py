import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKFLOW_DIR = ROOT / "data" / "processed" / "workflows"
DEFAULT_OUT_DIR = ROOT / "data" / "processed" / "fm_text"


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def compact_action(action):
    return {
        "tool_id": action.get("tool_id"),
        "tool_name": action.get("tool_name"),
        "arguments": action.get("arguments") or {},
    }


def gold_tool_call_text(record):
    actions = [compact_action(action) for action in (record.get("target") or {}).get("remaining_actions") or []]
    return json.dumps({"actions": actions}, ensure_ascii=False, sort_keys=True)


def normalize_record(record):
    target = record.get("target") or {}
    next_action = target.get("next_action") or {}
    future_tool_ids = target.get("future_tool_ids") or []
    return {
        "record_id": record.get("record_id"),
        "trajectory_id": record.get("workflow_id") or record.get("task_id"),
        "task_id": record.get("task_id"),
        "source": record.get("source"),
        "domain": record.get("domain"),
        "split": record.get("split"),
        "step_idx": record.get("step_idx"),
        "total_steps": record.get("total_steps"),
        "input_text": record.get("condition_text") or record.get("prompt") or "",
        "gold_tool_call_text": gold_tool_call_text(record),
        "future_tools": future_tool_ids,
        "next_tool": next_action.get("tool_id") or next_action.get("tool_name"),
        "available_tools": [tool.get("tool_id") for tool in record.get("available_tools") or [] if tool.get("tool_id")],
        "metadata": {
            "next_tool_name": next_action.get("tool_name"),
            "future_tool_names": target.get("future_tools") or [],
        },
    }


def build_split(workflow_dir, split, limit=None):
    rows = []
    for record in read_jsonl(workflow_dir / f"{split}.jsonl"):
        rows.append(normalize_record(record))
        if limit and len(rows) >= limit:
            break
    return rows


def write_readme(out_dir, manifest):
    lines = [
        "# FM Text Data",
        "",
        "Step-level text records for encoder-derived FM tensor extraction.",
        "",
        "These JSONL files adapt `data/processed/workflows/*.jsonl` to the input format expected by ToolRL's `dataset/external_fm/build_fm_tensor_dataset.py`.",
        "",
        "They are not final tensors; run the remote encoder extractor to produce `.pt` samples.",
        "",
        "## Counts",
        "",
    ]
    for split, info in manifest["splits"].items():
        lines.append(f"- {split}: {info['records']} records at `{info['path']}`")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Build FM text JSONL records from workflow records.")
    parser.add_argument("--workflow-dir", default=str(DEFAULT_WORKFLOW_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--limit-train", type=int, default=512)
    parser.add_argument("--limit-dev", type=int, default=128)
    parser.add_argument("--limit-test", type=int, default=128)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    workflow_dir = Path(args.workflow_dir)
    out_dir = Path(args.out_dir)
    limits = {
        "train": None if args.full else args.limit_train,
        "dev": None if args.full else args.limit_dev,
        "test": None if args.full else args.limit_test,
    }
    suffix = "" if args.full else "_smoke"
    manifest = {"format": "fm_text_v1", "source_dir": str(workflow_dir), "splits": {}}
    for split in ["train", "dev", "test"]:
        rows = build_split(workflow_dir, split, limit=limits[split])
        path = out_dir / f"{split}{suffix}.jsonl"
        write_jsonl(path, rows)
        manifest["splits"][split] = {"path": str(path.relative_to(ROOT)), "records": len(rows)}
        print(f"{split}: wrote {path} with {len(rows)} records")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_readme(out_dir, manifest)


if __name__ == "__main__":
    main()
