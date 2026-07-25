import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STOP = "<stop>"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


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


def user_content(row: dict[str, Any]) -> str:
    for message in row.get("messages") or []:
        if message.get("role") == "user":
            return message.get("content") or ""
    return ""


def assistant_payload(row: dict[str, Any]) -> dict[str, Any]:
    for message in reversed(row.get("messages") or []):
        if message.get("role") != "assistant":
            continue
        payload = json.loads(message.get("content") or "{}")
        if isinstance(payload, dict):
            return payload
    return {}


def parse_state(row: dict[str, Any]) -> dict[str, Any]:
    content = user_content(row)
    starts = []
    for marker in ["Compact state:", "Feedback-conditioned state:", "Conditioned workflow state:"]:
        marker_pos = content.find(marker)
        if marker_pos >= 0:
            starts.append(marker_pos)
    starts.append(0)
    for start in starts:
        candidate = find_balanced_json(content, start)
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Static audit for data/replan_sft/test500.")
    parser.add_argument("--input", default=str(ROOT / "data" / "replan_sft" / "test500" / "test.jsonl"))
    parser.add_argument("--expected-rows", type=int, default=500)
    parser.add_argument("--expected-format", default="test500_chat_sft_v1")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    errors: list[str] = []
    warnings: list[str] = []
    ids = Counter()
    domains = Counter()
    source_kinds = Counter()
    stop_by_domain = Counter()
    next_tools = Counter()

    for idx, row in enumerate(rows):
        row_id = row.get("id") or f"row-{idx}"
        ids[row_id] += 1
        metadata = row.get("metadata") or {}
        domains[metadata.get("domain")] += 1
        source_kinds[metadata.get("source_kind")] += 1
        next_tools[metadata.get("next_tool")] += 1
        if metadata.get("stop"):
            stop_by_domain[metadata.get("domain")] += 1

        if not row.get("id"):
            errors.append(f"{row_id}: missing id")
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            errors.append(f"{row_id}: missing chat messages")
            continue
        if messages[0].get("role") != "user" or messages[-1].get("role") != "assistant":
            errors.append(f"{row_id}: first/last message roles must be user/assistant")
        if metadata.get("format") != args.expected_format:
            errors.append(f"{row_id}: unexpected metadata format {metadata.get('format')}")
        if metadata.get("source") != "tau2":
            errors.append(f"{row_id}: non-tau2 source {metadata.get('source')}")
        if metadata.get("domain") not in {"retail", "telecom"}:
            errors.append(f"{row_id}: out-of-domain {metadata.get('domain')}")
        if metadata.get("split") != "test":
            errors.append(f"{row_id}: split is {metadata.get('split')}, expected test")

        state = parse_state(row)
        if not state:
            errors.append(f"{row_id}: cannot parse compact state")
            continue
        available = state.get("available_tools") or []
        available_ids = [tool.get("tool_id") for tool in available if isinstance(tool, dict) and tool.get("tool_id")]
        if not available_ids:
            errors.append(f"{row_id}: empty available_tools")
        if state.get("target_mode") != "next_action_or_stop":
            errors.append(f"{row_id}: target_mode is not next_action_or_stop")
        if state.get("source") != "tau2":
            errors.append(f"{row_id}: state source mismatch")
        if state.get("domain") != metadata.get("domain"):
            errors.append(f"{row_id}: state domain mismatch")

        try:
            target = assistant_payload(row)
        except Exception as exc:
            errors.append(f"{row_id}: assistant JSON parse error {exc}")
            continue
        actions = target.get("actions")
        if not isinstance(actions, list):
            errors.append(f"{row_id}: assistant actions is not a list")
            continue
        if len(actions) > 1:
            errors.append(f"{row_id}: target has more than one action")
        stop = bool(target.get("stop")) or not actions
        if stop != bool(metadata.get("stop")):
            errors.append(f"{row_id}: assistant stop and metadata stop disagree")
        next_tool = metadata.get("next_tool")
        if stop and next_tool != STOP:
            errors.append(f"{row_id}: stop row next_tool is not {STOP}")
        if not stop:
            if not actions:
                errors.append(f"{row_id}: non-stop row has no action")
            else:
                tool_id = actions[0].get("tool_id")
                if tool_id != next_tool:
                    errors.append(f"{row_id}: action tool and metadata next_tool disagree")
                if tool_id not in available_ids:
                    errors.append(f"{row_id}: action tool not in available_tools")

    duplicates = [row_id for row_id, count in ids.items() if count > 1]
    if duplicates:
        errors.append(f"duplicate id count: {len(duplicates)}")
    if len(rows) != args.expected_rows:
        errors.append(f"row count {len(rows)} != expected {args.expected_rows}")
    if set(domains) != {"retail", "telecom"}:
        errors.append(f"domain set mismatch: {sorted(domains)}")
    if domains.get("retail", 0) < 100:
        warnings.append("retail rows below 100")

    report = {
        "input": args.input,
        "rows": len(rows),
        "domains": dict(sorted(domains.items())),
        "source_kinds": dict(sorted(source_kinds.items())),
        "stop_by_domain": dict(sorted(stop_by_domain.items())),
        "stop_total": sum(stop_by_domain.values()),
        "next_tool_top20": next_tools.most_common(20),
        "errors": errors,
        "warnings": warnings,
        "ok": not errors,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
