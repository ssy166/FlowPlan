import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "ARGUMENT_EXTRACTABILITY.md"


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def flatten_values(value):
    if isinstance(value, dict):
        out = []
        for child in value.values():
            out.extend(flatten_values(child))
        return out
    if isinstance(value, list):
        out = []
        for child in value:
            out.extend(flatten_values(child))
        return out
    if value is None:
        return []
    return [str(value)]


def visible_in_prompt(value, prompt):
    value = str(value).strip()
    if not value:
        return False
    return value.lower() in (prompt or "").lower()


def summarize(rows):
    if not rows:
        return {"actions": 0, "args": 0, "visible_args": 0, "value_atoms": 0, "visible_value_atoms": 0}
    return {
        "actions": sum(row["actions"] for row in rows),
        "args": sum(row["args"] for row in rows),
        "visible_args": sum(row["visible_args"] for row in rows),
        "value_atoms": sum(row["value_atoms"] for row in rows),
        "visible_value_atoms": sum(row["visible_value_atoms"] for row in rows),
    }


def pct(num, den):
    return f"{(num / den):.3f}" if den else "-"


def table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def main():
    tasks = {row["task_id"]: row for row in read_jsonl(ROOT / "data" / "processed" / "tasks.jsonl")}
    gold_rows = list(read_jsonl(ROOT / "data" / "processed" / "gold_plans.jsonl"))
    split_ids = {}
    for split in ["train", "dev", "test"]:
        path = ROOT / "data" / "processed" / "task_splits" / f"{split}.txt"
        split_ids[split] = {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}

    records = []
    for gold in gold_rows:
        if gold.get("source") != "tau2":
            continue
        task = tasks.get(gold["task_id"]) or {}
        prompt = task.get("prompt", "")
        actions = gold.get("actions") or []
        args = 0
        visible_args = 0
        atoms = 0
        visible_atoms = 0
        for action in actions:
            action_args = action.get("arguments") or {}
            if not isinstance(action_args, dict):
                continue
            for value in action_args.values():
                args += 1
                values = flatten_values(value)
                atoms += len(values)
                visible = [visible_in_prompt(atom, prompt) for atom in values]
                if values and all(visible):
                    visible_args += 1
                visible_atoms += sum(visible)
        records.append(
            {
                "task_id": gold["task_id"],
                "split": next((name for name, ids in split_ids.items() if gold["task_id"] in ids), "unknown"),
                "actions": len(actions),
                "args": args,
                "visible_args": visible_args,
                "value_atoms": atoms,
                "visible_value_atoms": visible_atoms,
            }
        )

    rows = []
    for split in ["all", "train", "dev", "test"]:
        selected = records if split == "all" else [row for row in records if row["split"] == split]
        summary = summarize(selected)
        rows.append(
            [
                split,
                len(selected),
                summary["actions"],
                summary["args"],
                summary["visible_args"],
                pct(summary["visible_args"], summary["args"]),
                summary["value_atoms"],
                summary["visible_value_atoms"],
                pct(summary["visible_value_atoms"], summary["value_atoms"]),
            ]
        )

    body = [
        "# Argument Extractability",
        "",
        "Counts how often tau2 gold action argument values are directly visible in the task prompt.",
        "",
        table(
            [
                "split",
                "tasks",
                "actions",
                "args",
                "visible args",
                "visible arg ratio",
                "value atoms",
                "visible atoms",
                "visible atom ratio",
            ],
            rows,
        ),
        "",
        "Notes:",
        "- Hidden arguments usually require tool results, database state, or workflow context.",
        "- Direct prompt extraction is therefore only a partial argument-grounding baseline.",
    ]
    OUT.write_text("\n".join(body) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT), "tau2_tasks": len(records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
