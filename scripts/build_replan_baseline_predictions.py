import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"[a-zA-Z0-9_#]+")
STOP = "<stop>"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")


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


def target(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    for message in reversed(row.get("messages") or []):
        if message.get("role") != "assistant":
            continue
        try:
            payload = json.loads(message.get("content") or "{}")
        except Exception:
            return STOP, {}
        actions = payload.get("actions") or []
        if payload.get("stop") or not actions:
            return STOP, {}
        action = actions[0]
        tool_id = action.get("tool_id") or ""
        return tool_id, action
    return STOP, {}


def tool_name(tool_id: str) -> str:
    return str(tool_id or "").rsplit("::", 1)[-1]


def prediction_row(row: dict[str, Any], label: str, action: dict[str, Any] | None, plan_type: str) -> dict[str, Any]:
    metadata = row.get("metadata") or {}
    if label == STOP or not label:
        actions = []
    else:
        # Keep baseline arguments empty by default. State/result grounding can
        # fill executable IDs later where supported; copying train-row args
        # would leak entity values across tasks.
        actions = [
            {
                "tool_id": label,
                "tool_name": tool_name(label),
                "arguments": {},
                "metadata": {"baseline_gold_action_available": bool(action)},
            }
        ]
    return {
        "task_id": metadata.get("task_id") or row.get("id"),
        "record_id": row.get("id"),
        "source": metadata.get("source"),
        "domain": metadata.get("domain"),
        "plan_type": plan_type,
        "tool_ids": [] if label == STOP else [label],
        "tool_names": [] if label == STOP else [tool_name(label)],
        "phase_names": [],
        "actions": actions,
        "metadata": {"record_id": row.get("id"), "baseline": plan_type},
    }


def state_key(row: dict[str, Any]) -> tuple[Any, ...]:
    metadata = row.get("metadata") or {}
    state = parse_state(row)
    progress = state.get("grounding_progress") or {}
    prefix = state.get("executed_prefix") or []
    last_tool = None
    if prefix:
        last = prefix[-1] or {}
        last_tool = (
            (last.get("predicted_action") or {}).get("tool_id")
            or (last.get("predicted_action") or {}).get("tool_name")
            or last.get("tool_name")
        )
    return (
        metadata.get("domain"),
        bool(progress.get("lookup_completed")),
        bool(progress.get("order_grounded")),
        progress.get("resolved_order_status"),
        tuple(progress.get("operation_eligibility") or []),
        tool_name(last_tool) if last_tool else None,
        "stopish" if (state.get("step") or 0) >= 4 else "early",
    )


def tokens_for_row(row: dict[str, Any]) -> set[str]:
    state = parse_state(row)
    parts = [state.get("task") or "", json.dumps(state.get("grounding_progress") or {}, sort_keys=True)]
    parts.extend(tool.get("name") or "" for tool in state.get("available_tools") or [] if isinstance(tool, dict))
    prefix = state.get("executed_prefix") or []
    parts.extend(str((step.get("predicted_action") or {}).get("tool_name") or step.get("tool_name") or "") for step in prefix if isinstance(step, dict))
    return set(TOKEN_RE.findall(" ".join(parts).lower()))


def majority(counter: Counter[str], fallback: str) -> str:
    return counter.most_common(1)[0][0] if counter else fallback


def build_baselines(train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    train_targets = {row["id"]: target(row) for row in train_rows}
    global_counter = Counter(label for label, _ in train_targets.values())
    domain_counter: dict[str, Counter[str]] = defaultdict(Counter)
    key_counter: dict[tuple[Any, ...], Counter[str]] = defaultdict(Counter)
    train_by_domain: dict[str, list[tuple[dict[str, Any], set[str], str, dict[str, Any]]]] = defaultdict(list)
    for row in train_rows:
        metadata = row.get("metadata") or {}
        domain = metadata.get("domain") or "unknown"
        label, action = train_targets[row["id"]]
        domain_counter[domain][label] += 1
        key_counter[state_key(row)][label] += 1
        train_by_domain[domain].append((row, tokens_for_row(row), label, action))

    global_label = majority(global_counter, STOP)
    predictions = {
        "baseline_always_stop": [],
        "baseline_global_majority": [],
        "baseline_domain_majority": [],
        "baseline_state_key_majority": [],
        "baseline_lexical_nearest": [],
    }
    for row in eval_rows:
        metadata = row.get("metadata") or {}
        domain = metadata.get("domain") or "unknown"
        domain_label = majority(domain_counter.get(domain, Counter()), global_label)
        key_label = majority(key_counter.get(state_key(row), Counter()), domain_label)
        eval_tokens = tokens_for_row(row)
        best = None
        for train_row, train_tokens, label, action in train_by_domain.get(domain, []):
            union = eval_tokens | train_tokens
            score = len(eval_tokens & train_tokens) / len(union) if union else 0.0
            if best is None or score > best[0]:
                best = (score, label, action, train_row.get("id"))
        nearest_label = best[1] if best else domain_label
        nearest_action = best[2] if best else {}
        predictions["baseline_always_stop"].append(prediction_row(row, STOP, None, "baseline_always_stop"))
        predictions["baseline_global_majority"].append(
            prediction_row(row, global_label, None, "baseline_global_majority")
        )
        predictions["baseline_domain_majority"].append(
            prediction_row(row, domain_label, None, "baseline_domain_majority")
        )
        predictions["baseline_state_key_majority"].append(
            prediction_row(row, key_label, None, "baseline_state_key_majority")
        )
        pred = prediction_row(row, nearest_label, nearest_action, "baseline_lexical_nearest")
        pred["metadata"]["nearest_train_id"] = best[3] if best else None
        pred["metadata"]["nearest_score"] = best[0] if best else None
        predictions["baseline_lexical_nearest"].append(pred)
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Build lightweight next-action baselines for replan SFT rows.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["dev", "test"])
    args = parser.parse_args()

    train_rows = read_jsonl(Path(args.data_dir) / "train.jsonl")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"data_dir": args.data_dir, "splits": {}, "baselines": []}
    for split in args.splits:
        eval_rows = read_jsonl(Path(args.data_dir) / f"{split}.jsonl")
        predictions = build_baselines(train_rows, eval_rows)
        manifest["baselines"] = sorted(predictions)
        manifest["splits"][split] = {"rows": len(eval_rows)}
        for name, rows in predictions.items():
            write_jsonl(out_dir / f"{name}.{split}.jsonl", rows)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out_dir": str(out_dir), "baselines": manifest["baselines"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
