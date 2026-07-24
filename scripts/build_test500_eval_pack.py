import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def find_balanced_json(text: str, start: int = 0) -> str | None:
    first = text.find("{", max(0, start))
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
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[first : idx + 1]
    return None


def parse_state(user_text: str) -> dict[str, Any]:
    for marker in ["Compact state:", "Feedback-conditioned state:", "Conditioned workflow state:"]:
        marker_pos = user_text.find(marker)
        if marker_pos < 0:
            continue
        candidate = find_balanced_json(user_text, marker_pos)
        if candidate:
            return json.loads(candidate)
    candidate = find_balanced_json(user_text, 0)
    return json.loads(candidate) if candidate else {}


def user_prompt(row: dict[str, Any]) -> str:
    for message in row.get("messages") or []:
        if message.get("role") == "user":
            return message.get("content") or ""
    return ""


def available_tool_ids(row: dict[str, Any]) -> list[str]:
    metadata = row.get("metadata") or {}
    if metadata.get("available_tool_ids"):
        return list(metadata["available_tool_ids"])
    state = parse_state(user_prompt(row))
    return [tool["tool_id"] for tool in state.get("available_tools") or [] if isinstance(tool, dict) and tool.get("tool_id")]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a row-level model eval pack from data/replan_sft/test500.")
    parser.add_argument("--input", default=str(ROOT / "data" / "replan_sft" / "test500" / "test.jsonl"))
    parser.add_argument("--out", default=str(ROOT / "data" / "replan_sft" / "test500" / "model_eval_pack.jsonl"))
    args = parser.parse_args()

    rows = []
    for row in read_jsonl(args.input):
        metadata = row.get("metadata") or {}
        rows.append(
            {
                "task_id": row["id"],
                "source": metadata.get("source"),
                "domain": metadata.get("domain"),
                "split": metadata.get("split", "test"),
                "prompt": user_prompt(row),
                "available_tool_ids": available_tool_ids(row),
                "metadata": {
                    "format": "test500_model_eval_pack_v1",
                    "original_task_id": metadata.get("task_id"),
                    "record_id": row["id"],
                    "source_kind": metadata.get("source_kind"),
                },
            }
        )
    write_jsonl(args.out, rows)
    print(json.dumps({"out": args.out, "rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
