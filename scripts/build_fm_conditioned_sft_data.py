import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKFLOW_DIR = ROOT / "data" / "processed" / "workflows"
DEFAULT_OUT_DIR = ROOT / "data" / "processed" / "fm_conditioned_sft"


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def shorten(text, max_chars):
    text = " ".join(str(text or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def compact_tool(tool, max_desc_chars):
    return {
        "tool_id": tool.get("tool_id"),
        "name": tool.get("name") or str(tool.get("tool_id", "")).rsplit("::", 1)[-1],
        "phase": tool.get("phase"),
        "description": shorten(tool.get("description") or "", max_desc_chars),
        "parameters": tool.get("parameters") or [],
    }


def compact_action(action, include_tool_id=True):
    out = {
        "tool_name": action.get("tool_name") or str(action.get("tool_id", "")).rsplit("::", 1)[-1],
        "arguments": action.get("arguments") or {},
    }
    if include_tool_id and action.get("tool_id"):
        out["tool_id"] = action["tool_id"]
    return out


def build_target(record, include_tool_id, target_mode):
    actions = [
        compact_action(action, include_tool_id=include_tool_id)
        for action in (record.get("target") or {}).get("remaining_actions") or []
    ]
    if target_mode == "next":
        actions = actions[:1]
    return {"stop": len(actions) == 0, "actions": actions}


def build_user_content(record, args):
    fm_condition = {
        "kind": "external_soft_prefix",
        "latent_key": record.get("record_id"),
        "checkpoint": args.fm_checkpoint,
        "latent_dim": args.latent_dim,
        "note": "The FM latent is provided out-of-band to the model as gated soft prefix tokens; it is not serialized as text.",
    }
    state = {
        "task_id": record.get("task_id"),
        "record_id": record.get("record_id"),
        "source": record.get("source"),
        "domain": record.get("domain"),
        "step_idx": record.get("step_idx"),
        "total_steps": record.get("total_steps"),
        "user_prompt": record.get("prompt") or "",
        "available_tools": [compact_tool(tool, args.max_desc_chars) for tool in record.get("available_tools") or []],
        "prefix_actions": [compact_action(action, include_tool_id=True) for action in record.get("prefix_actions") or []],
        "tool_feedback": record.get("tool_feedback") or [],
        "fm_condition": fm_condition,
    }
    if args.include_state_context:
        state["state_context"] = record.get("state_context") or {}
    instruction = [
        "You are an FM-conditioned tool execution model.",
        "A learned FM planner latent is attached as soft prefix tokens outside the text.",
        "Use the soft prefix together with the task, available tools, executed prefix, and tool feedback.",
        "Emit strict JSON only with this schema:",
        '{"stop": false, "actions": [{"tool_name": "...", "arguments": {}}]}',
        "If the workflow should stop now, emit:",
        '{"stop": true, "actions": []}',
        "Do not call tools that are absent from available_tools.",
    ]
    return "\n".join(instruction + ["", "Conditioned workflow state:", json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True)])


def build_example(record, split, args):
    target = build_target(record, include_tool_id=args.include_tool_id_in_target, target_mode=args.target_mode)
    return {
        "id": record.get("record_id"),
        "messages": [
            {"role": "user", "content": build_user_content(record, args)},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False, sort_keys=True)},
        ],
        "metadata": {
            "task_id": record.get("task_id"),
            "record_id": record.get("record_id"),
            "workflow_id": record.get("workflow_id"),
            "source": record.get("source"),
            "domain": record.get("domain"),
            "split": split,
            "step_idx": record.get("step_idx"),
            "total_steps": record.get("total_steps"),
            "remaining_action_count": len(target["actions"]),
            "stop": target["stop"],
            "available_tool_count": len(record.get("available_tools") or []),
            "conditioning": "fm_external_soft_prefix_v1",
            "target_mode": args.target_mode,
            "fm_checkpoint": args.fm_checkpoint,
        },
    }


def validate_rows(rows):
    errors = []
    stats = {"rows": len(rows), "json_parse_errors": 0, "bad_schema": 0, "stop_rows": 0, "action_rows": 0}
    for row in rows:
        try:
            target = json.loads(row["messages"][-1]["content"])
        except Exception as exc:
            stats["json_parse_errors"] += 1
            errors.append(f"{row.get('id')}: assistant content is not JSON: {exc}")
            continue
        if not isinstance(target, dict) or not isinstance(target.get("stop"), bool) or not isinstance(target.get("actions"), list):
            stats["bad_schema"] += 1
            errors.append(f"{row.get('id')}: target does not match stop/actions schema")
            continue
        if target["stop"]:
            stats["stop_rows"] += 1
        else:
            stats["action_rows"] += 1
        for action in target["actions"]:
            if not isinstance(action, dict) or not isinstance(action.get("tool_name"), str) or not isinstance(action.get("arguments"), dict):
                stats["bad_schema"] += 1
                errors.append(f"{row.get('id')}: bad action schema")
                break
    return stats, errors


def write_readme(out_dir, manifest):
    lines = [
        "# FM-Conditioned SFT Data",
        "",
        "Chat-style SFT data for the FM-conditioned LLM executor.",
        "",
        "Input text contains task/tool/prefix/feedback context plus an external `fm_condition.latent_key`.",
        "The actual FM latent should be injected out-of-band as gated soft prefix tokens.",
        "",
        'Target schema: `{"stop": bool, "actions": [{"tool_name": "...", "arguments": {}}]}`.',
        f"Target mode: `{manifest['target_mode']}`.",
        "",
        "This pack is for SFT smoke first. Do not use it as an RL claim.",
        "",
        "## Counts",
        "",
    ]
    for split, info in manifest["splits"].items():
        lines.append(f"- {split}: {info['rows']} rows, stop {info['stop_rows']}, action {info['action_rows']}")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Build FM-conditioned SFT data with external soft-prefix latent references.")
    parser.add_argument("--workflow-dir", default=str(DEFAULT_WORKFLOW_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--fm-checkpoint", default="<project_root>/outputs/fm_full/encoder_v3_cosopt/best.pt")
    parser.add_argument("--latent-dim", type=int, default=768)
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--max-desc-chars", type=int, default=240)
    parser.add_argument("--max-rows-per-split", type=int, default=0)
    parser.add_argument("--include-tool-id-in-target", action="store_true")
    parser.add_argument("--include-state-context", action="store_true")
    parser.add_argument("--target-mode", choices=["remaining", "next"], default="remaining")
    args = parser.parse_args()

    workflow_dir = Path(args.workflow_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "format": "fm_conditioned_chat_sft_v1",
        "workflow_dir": str(workflow_dir),
        "fm_checkpoint": args.fm_checkpoint,
        "latent_dim": args.latent_dim,
        "max_desc_chars": args.max_desc_chars,
        "include_tool_id_in_target": args.include_tool_id_in_target,
        "include_state_context": args.include_state_context,
        "target_mode": args.target_mode,
        "splits": {},
    }
    all_errors = []
    for split in args.splits:
        records = read_jsonl(workflow_dir / f"{split}.jsonl")
        if args.max_rows_per_split:
            records = records[: args.max_rows_per_split]
        rows = [build_example(record, split, args) for record in records]
        stats, errors = validate_rows(rows)
        if errors:
            all_errors.extend(errors[:20])
        write_jsonl(out_dir / f"{split}.jsonl", rows)
        source_counts = {}
        domain_counts = {}
        for row in rows:
            md = row["metadata"]
            source_counts[md["source"]] = source_counts.get(md["source"], 0) + 1
            key = f"{md['source']}::{md['domain']}"
            domain_counts[key] = domain_counts.get(key, 0) + 1
        manifest["splits"][split] = {
            **stats,
            "path": str((out_dir / f"{split}.jsonl").relative_to(ROOT) if out_dir.is_relative_to(ROOT) else out_dir / f"{split}.jsonl"),
            "source_counts": source_counts,
            "domain_counts": domain_counts,
        }
    if all_errors:
        raise ValueError("Validation failed: " + "; ".join(all_errors[:10]))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_readme(out_dir, manifest)
    print(json.dumps({"out_dir": str(out_dir), "splits": {k: v["rows"] for k, v in manifest["splits"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
