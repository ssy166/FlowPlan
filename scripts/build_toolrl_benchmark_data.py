import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "data" / "processed" / "fm_replan_sft_next"
DEFAULT_OUT_DIR = ROOT / "data" / "processed" / "toolrl_benchmark"


def read_jsonl(path, limit=0):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def find_balanced_json(text, start=0):
    first = text.find("{", start)
    if first < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for idx in range(first, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[first : idx + 1]
    return None


def parse_state(row):
    user_content = row["messages"][0]["content"]
    marker_positions = [
        user_content.find("Feedback-conditioned state:"),
        user_content.find("Conditioned workflow state:"),
    ]
    starts = [pos for pos in marker_positions if pos >= 0]
    start = user_content.find("{", min(starts) if starts else 0)
    candidate = find_balanced_json(user_content, start)
    if not candidate:
        return {}
    try:
        return json.loads(candidate)
    except Exception:
        return {}


def parse_target(row):
    target = json.loads(row["messages"][-1]["content"])
    actions = target.get("actions") or []
    return bool(target.get("stop", False)), actions


def build_ground_truth(row):
    metadata = row.get("metadata") or {}
    state = parse_state(row)
    gold_stop, gold_actions = parse_target(row)
    return {
        "task_id": metadata.get("task_id") or row.get("id"),
        "record_id": row.get("id") or metadata.get("record_id"),
        "source": metadata.get("source"),
        "domain": metadata.get("domain"),
        "split": metadata.get("split"),
        "gold_stop": gold_stop,
        "gold_actions": gold_actions,
        "available_tools": state.get("available_tools") or [],
        "target_json": row["messages"][-1]["content"],
        "reward_version": "toolrl_benchmark_reward_v1",
    }


def build_prompt(row, args):
    messages = row.get("messages") or []
    prompt_messages = [msg for msg in messages if msg.get("role") != "assistant"]
    if args.system_prompt:
        prompt_messages = [{"role": "system", "content": args.system_prompt}] + prompt_messages
    return prompt_messages


def convert_row(row, idx, split, args):
    metadata = row.get("metadata") or {}
    ground_truth = build_ground_truth(row)
    return {
        "data_source": args.data_source,
        "prompt": build_prompt(row, args),
        "reward_model": {
            "style": "rule",
            "ground_truth": json.dumps(ground_truth, ensure_ascii=False, sort_keys=True),
        },
        "extra_info": {
            "index": idx,
            "split": split,
            "id": row.get("id"),
            "task_id": metadata.get("task_id"),
            "record_id": metadata.get("record_id") or row.get("id"),
            "source": metadata.get("source"),
            "domain": metadata.get("domain"),
            "stop": metadata.get("stop"),
            "conditioning": metadata.get("conditioning"),
        },
    }


def write_parquet(path, rows):
    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError("pandas is required for parquet output") from exc
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def write_readme(out_dir, manifest):
    lines = [
        "# ToolRL Benchmark Data",
        "",
        "VERL/ToolRL RLHF parquet data derived from this benchmark's chat-style SFT rows.",
        "",
        "Columns:",
        "- `data_source`: should route to the benchmark reward function.",
        "- `prompt`: list of chat messages consumed by VERL `RLHFDataset`.",
        "- `reward_model.ground_truth`: JSON string with gold stop/actions and available tools.",
        "- `extra_info`: ids and split/domain metadata.",
        "",
        "Use with `scripts/toolrl_benchmark_reward.py` or copy that reward into ToolRL's `verl/utils/reward_score` package.",
        "",
    ]
    for split, info in manifest["splits"].items():
        lines.append(f"- {split}: {info['rows']} rows at `{info['path']}`")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Convert benchmark SFT JSONL rows to ToolRL/VERL RLHF data.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--format", choices=["parquet", "jsonl", "both"], default="parquet")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-dev", type=int, default=0)
    parser.add_argument("--limit-test", type=int, default=0)
    parser.add_argument("--data-source", default="rlla_tooluse_benchmark")
    parser.add_argument("--system-prompt", default="")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    limits = {"train": args.limit_train, "dev": args.limit_dev, "test": args.limit_test}
    manifest = {
        "format": "toolrl_benchmark_rlhf_v1",
        "input_dir": str(input_dir),
        "data_source": args.data_source,
        "splits": {},
    }
    for split in args.splits:
        source_rows = read_jsonl(input_dir / f"{split}.jsonl", limit=limits.get(split, 0))
        rows = [convert_row(row, idx, split, args) for idx, row in enumerate(source_rows)]
        written = []
        if args.format in {"jsonl", "both"}:
            path = out_dir / f"{split}.jsonl"
            write_jsonl(path, rows)
            written.append(str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path))
        if args.format in {"parquet", "both"}:
            path = out_dir / f"{split}.parquet"
            write_parquet(path, rows)
            written.append(str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path))
        manifest["splits"][split] = {"rows": len(rows), "path": written}
        print(json.dumps({"split": split, "rows": len(rows), "path": written}, ensure_ascii=False))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_readme(out_dir, manifest)


if __name__ == "__main__":
    main()
