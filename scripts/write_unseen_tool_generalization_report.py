import argparse
import collections
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_split_ids(path):
    return {line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()}


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def summarize(rows):
    if not rows:
        return {"count": 0}
    n = len(rows)
    unseen_gold = sum(len(row["unseen_tool_ids"]) for row in rows)
    unseen_family_gold = sum(len(row["unseen_families"]) for row in rows)
    return {
        "count": n,
        "tool_exact_match": sum(row["tool_exact_match"] for row in rows) / n,
        "phase_exact_match": sum(row["phase_exact_match"] for row in rows) / n,
        "schema_valid_rate": sum(row["schema_valid"] for row in rows) / n,
        "avg_tool_edit_distance": sum(row["tool_edit_distance"] for row in rows) / n,
        "avg_gold_tool_count": sum(row["gold_tool_count"] for row in rows) / n,
        "avg_predicted_tool_count": sum(row["predicted_tool_count"] for row in rows) / n,
        "unseen_tool_recall": (
            sum(row["unseen_tool_hit_count"] for row in rows) / unseen_gold if unseen_gold else None
        ),
        "unseen_family_recall": (
            sum(row["unseen_family_hit_count"] for row in rows) / unseen_family_gold
            if unseen_family_gold
            else None
        ),
    }


def fmt(value):
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def table(rows, columns):
    lines = [
        "| " + " | ".join(label for label, _ in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(key)) for _, key in columns) + " |")
    return "\n".join(lines)


def tool_family(tool_id):
    parts = tool_id.split("::")
    source = parts[0] if parts else ""
    domain = parts[1] if len(parts) > 1 else ""
    name = parts[-1] if parts else tool_id
    if "_for_" in name:
        return f"{source}::{domain}::{name.split('_for_', 1)[1]}"
    for prefix in (
        "get_",
        "list_",
        "find_",
        "search_",
        "create_",
        "update_",
        "delete_",
        "cancel_",
        "modify_",
        "return_",
        "exchange_",
        "book_",
    ):
        if name.startswith(prefix):
            return f"{source}::{domain}::{prefix[:-1]}"
    return f"{source}::{domain}::{name}"


def unseen_tools_for_mode(gold_tools, gold_plan, train_tools, train_families, tools, mode):
    if mode in {"native", "test-unseen-tool-split"}:
        unseen_tools = sorted({tool for tool in gold_tools if tools.get(tool, {}).get("tool_split") == "test_unseen"})
        if mode == "native" and gold_plan.get("has_unseen_tool") and not unseen_tools:
            unseen_tools = sorted(set(gold_tools))
        return unseen_tools
    if mode == "strict-train-heldout":
        return sorted({tool for tool in gold_tools if tool not in train_tools})
    if mode == "strict-family-heldout":
        return sorted({tool for tool in gold_tools if tool_family(tool) not in train_families})
    raise ValueError(f"Unsupported mode: {mode}")


def read_eval_ids(processed, split):
    if split == "all":
        return (
            read_split_ids(processed / "task_splits" / "dev.txt")
            | read_split_ids(processed / "task_splits" / "test.txt")
        )
    return read_split_ids(processed / "task_splits" / f"{split}.txt")


def prediction_paths(processed, split, filename_pattern):
    if split != "all":
        return [processed / "predictions" / filename_pattern.format(split=split)]
    paths = []
    for part in ("dev", "test"):
        path = processed / "predictions" / filename_pattern.format(split=part)
        if path.exists():
            paths.append(path)
    return paths


def read_prediction_paths(paths):
    rows = []
    for path in paths:
        if path.exists():
            rows.extend(read_jsonl(path))
    return rows


def evaluate_predictions(predictions, gold, tasks, eval_ids, train_tools, train_families, tools, mode):
    pred_by_task = {row["task_id"]: row for row in predictions}
    rows = []
    for task_id in sorted(eval_ids):
        gold_plan = gold.get(task_id)
        pred = pred_by_task.get(task_id)
        if not gold_plan or not pred:
            continue
        gold_tools = gold_plan.get("tool_ids") or []
        if not gold_tools:
            continue
        pred_tools = pred.get("tool_ids") or []
        gold_phases = gold_plan.get("phase_names") or []
        pred_phases = pred.get("phase_names") or []
        unseen_tools = unseen_tools_for_mode(gold_tools, gold_plan, train_tools, train_families, tools, mode)
        is_unseen = bool(unseen_tools)
        unseen_families = sorted({tool_family(tool) for tool in unseen_tools})
        pred_families = {tool_family(tool) for tool in pred_tools}
        task = tasks.get(task_id) or {}
        available = set(task.get("available_tool_ids") or [])
        schema_valid = bool(pred_tools) and (not available or all(tool in available for tool in pred_tools))
        rows.append(
            {
                "task_id": task_id,
                "source": gold_plan.get("source"),
                "domain": task.get("domain"),
                "has_unseen_tool": is_unseen,
                "unseen_tool_ids": unseen_tools,
                "unseen_families": unseen_families,
                "tool_exact_match": pred_tools == gold_tools,
                "phase_exact_match": pred_phases == gold_phases,
                "schema_valid": schema_valid,
                "unseen_tool_hit_count": len(set(unseen_tools) & set(pred_tools)),
                "unseen_family_hit_count": len(set(unseen_families) & pred_families),
                "tool_edit_distance": edit_distance(pred_tools, gold_tools),
                "phase_edit_distance": edit_distance(pred_phases, gold_phases),
                "gold_tool_count": len(gold_tools),
                "predicted_tool_count": len(pred_tools),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Report workflow-planner generalization on split-held-out tools.")
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--split", default="test", choices=["dev", "test", "all"])
    parser.add_argument(
        "--mode",
        default="native",
        choices=["native", "test-unseen-tool-split", "strict-train-heldout", "strict-family-heldout"],
        help=(
            "native/test-unseen use benchmark tool_split metadata; strict-train-heldout uses tools absent "
            "from train gold plans; strict-family-heldout uses tool families absent from train gold plans."
        ),
    )
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    processed = root / "data" / "processed"
    gold = {row["task_id"]: row for row in read_jsonl(processed / "gold_plans.jsonl")}
    tasks = {row["task_id"]: row for row in read_jsonl(processed / "tasks.jsonl")}
    tools = {row["tool_id"]: row for row in read_jsonl(processed / "tools.jsonl")}
    train_ids = read_split_ids(processed / "task_splits" / "train.txt")
    eval_ids = read_eval_ids(processed, args.split)
    train_tools = {
        tool
        for task_id in train_ids
        for tool in (gold.get(task_id, {}).get("tool_ids") or [])
    }
    train_families = {tool_family(tool) for tool in train_tools}

    pred_specs = [
        ("keyword nearest", [processed / "predictions" / "keyword_nearest.jsonl"]),
        ("phase majority", [processed / "predictions" / "phase_majority.jsonl"]),
        ("FM nearest", prediction_paths(processed, args.split, "fm_nearest_path_encoder_v1.{split}.jsonl")),
        (
            "FM constrained nearest",
            prediction_paths(processed, args.split, "fm_nearest_path_encoder_v1_constrained.{split}.jsonl"),
        ),
        ("FM AR available-mask", prediction_paths(processed, args.split, "fm_ar_pointer_decoder_v1.{split}.jsonl")),
        (
            "FM optimized constrained",
            prediction_paths(processed, args.split, "fm_nearest_path_encoder_v2_optimized_constrained.{split}.jsonl"),
        ),
    ]

    methods = []
    for name, paths in pred_specs:
        paths = [path for path in paths if path.exists()]
        if not paths:
            continue
        rows = evaluate_predictions(read_prediction_paths(paths), gold, tasks, eval_ids, train_tools, train_families, tools, args.mode)
        seen = [row for row in rows if not row["has_unseen_tool"]]
        unseen = [row for row in rows if row["has_unseen_tool"]]
        methods.append(
            {
                "method": name,
                "prediction_file": ", ".join(str(path.relative_to(root)) for path in paths),
                "overall": summarize(rows),
                "seen": summarize(seen),
                "unseen": summarize(unseen),
            }
        )

    eval_gold_rows = [
        gold[task_id]
        for task_id in sorted(eval_ids)
        if task_id in gold and (gold[task_id].get("tool_ids") or [])
    ]
    unseen_tool_ids = sorted(
        {
            tool
            for row in eval_gold_rows
            for tool in unseen_tools_for_mode(
                row.get("tool_ids") or [], row, train_tools, train_families, tools, args.mode
            )
        }
    )
    unseen_families = sorted({tool_family(tool) for tool in unseen_tool_ids})
    domain_counts = collections.Counter()
    source_counts = collections.Counter()
    for row in eval_gold_rows:
        row_unseen = unseen_tools_for_mode(row.get("tool_ids") or [], row, train_tools, train_families, tools, args.mode)
        if row_unseen:
            task = tasks.get(row["task_id"]) or {}
            domain_counts[task.get("domain") or row.get("domain") or "unknown"] += 1
            source_counts[row.get("source") or "unknown"] += 1
    definitions = {
        "native": "A native unseen-tool row follows the benchmark has_unseen_tool label and uses tool_split=test_unseen tools as the target subset.",
        "test-unseen-tool-split": "A test-unseen row has at least one gold tool whose tool_split is test_unseen.",
        "strict-train-heldout": "A strict held-out-tool row has at least one gold tool id that never appears in the official train split gold plans.",
        "strict-family-heldout": "A strict held-out-family row has at least one gold tool family absent from the official train split gold plans.",
    }
    summary = {
        "definition": definitions[args.mode],
        "mode": args.mode,
        "split": args.split,
        "train_task_count": len(train_ids),
        "eval_task_count": len(eval_ids),
        "train_tool_count": len(train_tools),
        "train_family_count": len(train_families),
        "eval_nonempty_workflow_count": len(eval_gold_rows),
        "unseen_tool_count": len(unseen_tool_ids),
        "unseen_family_count": len(unseen_families),
        "unseen_tool_ids": unseen_tool_ids,
        "unseen_families": unseen_families,
        "unseen_domain_counts": dict(sorted(domain_counts.items())),
        "unseen_source_counts": dict(sorted(source_counts.items())),
        "methods": methods,
    }

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.out_md:
        rows = []
        for method in methods:
            rows.append(
                {
                    "method": method["method"],
                    "overall_n": method["overall"]["count"],
                    "overall_tool_em": method["overall"].get("tool_exact_match"),
                    "seen_n": method["seen"]["count"],
                    "seen_tool_em": method["seen"].get("tool_exact_match"),
                    "unseen_n": method["unseen"]["count"],
                    "unseen_tool_em": method["unseen"].get("tool_exact_match"),
                    "unseen_schema_valid": method["unseen"].get("schema_valid_rate"),
                    "unseen_tool_recall": method["unseen"].get("unseen_tool_recall"),
                    "unseen_family_recall": method["unseen"].get("unseen_family_recall"),
                    "unseen_avg_ed": method["unseen"].get("avg_tool_edit_distance"),
                }
            )
        md = [
            "# Unseen Tool Generalization",
            "",
            f"Split: `{args.split}`.",
            "",
            f"Definition: {definitions[args.mode]}",
            "",
            f"- Train tasks: {len(train_ids)}",
            f"- Train gold tools: {len(train_tools)}",
            f"- Train gold tool families: {len(train_families)}",
            f"- Eval tasks: {len(eval_ids)}",
            f"- Eval non-empty workflows: {len(eval_gold_rows)}",
            f"- Eval unseen tool ids: {len(unseen_tool_ids)}",
            f"- Eval unseen tool families: {len(unseen_families)}",
            f"- Unseen source counts: {dict(sorted(source_counts.items()))}",
            f"- Unseen domain counts: {dict(sorted(domain_counts.items()))}",
            "",
            table(
                rows,
                [
                    ("method", "method"),
                    ("overall n", "overall_n"),
                    ("overall tool EM", "overall_tool_em"),
                    ("seen n", "seen_n"),
                    ("seen tool EM", "seen_tool_em"),
                    ("unseen n", "unseen_n"),
                    ("unseen tool EM", "unseen_tool_em"),
                    ("unseen schema valid", "unseen_schema_valid"),
                    ("unseen tool recall", "unseen_tool_recall"),
                    ("unseen family recall", "unseen_family_recall"),
                    ("unseen avg ED", "unseen_avg_ed"),
                ],
            ),
            "",
            "Interpretation: this is a workflow-planning generalization check, separate from tau2 closed-loop execution. Tool EM measures full workflow equality; unseen tool/family recall measures whether the decoder at least selects the held-out schema element; schema valid measures whether predicted tools stay inside the task's available-tool catalog.",
            "",
            "Representative unseen tool ids:",
            "",
        ]
        for tool in unseen_tool_ids[:30]:
            md.append(f"- `{tool}`")
        md.extend(["", "Representative unseen tool families:", ""])
        for family in unseen_families[:30]:
            md.append(f"- `{family}`")
        out_md = Path(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
