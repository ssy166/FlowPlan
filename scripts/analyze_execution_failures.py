import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def summarize(traces: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    error_types = Counter()
    error_messages = Counter()
    tools = Counter()
    dropped_args = Counter()
    failed_by_tool = defaultdict(Counter)
    examples = []
    total_steps = 0
    failed_steps = 0
    for trace in traces:
        for step in trace.get("steps") or []:
            total_steps += 1
            execution = step.get("execution") or {}
            if execution.get("ok"):
                continue
            failed_steps += 1
            tool = execution.get("tool_name") or ((step.get("predicted_action") or {}).get("tool_name")) or "<missing_tool>"
            error_type = execution.get("error_type") or "UnknownError"
            error = execution.get("error") or ""
            error_types[error_type] += 1
            error_messages[error] += 1
            tools[tool] += 1
            failed_by_tool[tool][error] += 1
            for key in (execution.get("dropped_arguments") or {}):
                dropped_args[f"{tool}.{key}"] += 1
            if len(examples) < top_k:
                examples.append(
                    {
                        "task_id": trace.get("task_id"),
                        "step_idx": step.get("step_idx"),
                        "tool": tool,
                        "predicted_action": step.get("predicted_action"),
                        "gold_action": step.get("gold_action"),
                        "execution": execution,
                    }
                )
    return {
        "trace_count": len(traces),
        "total_steps": total_steps,
        "failed_steps": failed_steps,
        "failure_rate": failed_steps / total_steps if total_steps else 0.0,
        "error_types": error_types.most_common(top_k),
        "error_messages": error_messages.most_common(top_k),
        "failed_tools": tools.most_common(top_k),
        "dropped_args": dropped_args.most_common(top_k),
        "failed_by_tool": {
            tool: counter.most_common(min(top_k, 5))
            for tool, counter in sorted(failed_by_tool.items(), key=lambda item: sum(item[1].values()), reverse=True)[:top_k]
        },
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze failed real-tool executions in closed-loop traces.")
    parser.add_argument("--traces", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()
    summary = summarize(read_jsonl(args.traces), args.top_k)
    summary["traces"] = args.traces
    write_json(args.out, summary)
    print(json.dumps({"out": args.out, "failure_rate": summary["failure_rate"], "failed_steps": summary["failed_steps"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
