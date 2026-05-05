import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASKS = ROOT / "data" / "processed" / "tasks.jsonl"
DEFAULT_GTA_END = ROOT / "data" / "raw" / "gta" / "gta_workflow_dataset" / "end.json"
DEFAULT_OUT = ROOT / "data" / "processed" / "gta_eval_pack.json"
GTA_SCORER = ROOT / "data" / "raw" / "gta" / "repo" / "agent_app_eval" / "score_with_gpt52.py"


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8", errors="replace"))


def rel_path(path):
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def resolve_path(value):
    path = Path(value)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def collect_files(files_dir):
    if not files_dir.exists():
        return []
    if not files_dir.is_dir():
        raise ValueError(f"Expected files directory, got file: {files_dir}")
    return [rel_path(path) for path in sorted(files_dir.rglob("*")) if path.is_file()]


def index_processed_gta_tasks(tasks_path):
    tasks = {}
    for row in read_jsonl(tasks_path):
        if row.get("source") != "gta":
            continue
        local_id = row["task_id"].rsplit("::", 1)[-1]
        tasks[local_id] = row
    return tasks


def build_pack(tasks_path, gta_end_path, result_dir, out_pack, task_ids=None, allow_missing=False):
    processed = index_processed_gta_tasks(tasks_path)
    raw_items = read_json(gta_end_path)
    if not isinstance(raw_items, dict):
        raise ValueError("Unsupported GTA end.json format: expected an object keyed by task id")

    selected_ids = [str(task_id) for task_id in task_ids] if task_ids else sorted(raw_items, key=lambda x: int(x))
    missing = []
    packed_tasks = []

    for sample_idx, local_id in enumerate(selected_ids, start=1):
        if local_id not in raw_items:
            raise KeyError(f"GTA task id not found in end.json: {local_id}")
        if local_id not in processed:
            raise KeyError(f"GTA task id not found in processed tasks: {local_id}")

        task_dir = result_dir / local_id
        final_path = task_dir / "final.txt"
        if not final_path.exists():
            missing.append(str(final_path))
            if allow_missing:
                continue
            raise FileNotFoundError(f"Missing final output: {final_path}")

        processed_task = processed[local_id]
        raw_item = raw_items[local_id]
        final_text = final_path.read_text(encoding="utf-8", errors="replace")

        packed_tasks.append(
            {
                "task_id": int(local_id),
                "sample_idx": sample_idx,
                "origin_prompt": [processed_task.get("prompt") or ""],
                "assistant_outputs": [[{"role": "assistant", "content": final_text}]],
                "attached_files": collect_files(task_dir / "files"),
                "full_tree": raw_item.get("full_tree") or raw_item.get("sub_tasks"),
                "meta": {
                    "benchmark_task_id": processed_task["task_id"],
                    "domain": processed_task.get("domain"),
                    "result_dir": rel_path(result_dir),
                    "raw_path": processed_task.get("raw_path"),
                },
            }
        )

    pack = {
        "schema_version": 1,
        "generated_from": {
            "script": rel_path(Path(__file__)),
            "tasks": rel_path(tasks_path),
            "dataset_end_json": rel_path(gta_end_path),
            "result_dir": rel_path(result_dir),
        },
        "summary": {
            "requested_tasks": len(selected_ids),
            "packed_tasks": len(packed_tasks),
            "missing_final_outputs": len(missing),
        },
        "missing_final_outputs": missing,
        "tasks": packed_tasks,
    }

    out_pack.parent.mkdir(parents=True, exist_ok=True)
    out_pack.write_text(json.dumps(pack, ensure_ascii=False, indent=2), encoding="utf-8")
    return pack


def parse_task_ids(values):
    if not values:
        return None
    task_ids = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                task_ids.append(part)
    return task_ids


def run_dry_score(out_pack):
    if not GTA_SCORER.exists():
        raise FileNotFoundError(f"Missing GTA scorer: {GTA_SCORER}")
    subprocess.run(
        [sys.executable, str(GTA_SCORER), "--in-pack", str(out_pack), "--dry-run", "--overwrite"],
        cwd=ROOT,
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Build a GTA-Workflow final-output eval pack.")
    parser.add_argument("--tasks", default=str(DEFAULT_TASKS))
    parser.add_argument("--gta-end", default=str(DEFAULT_GTA_END))
    parser.add_argument("--result-dir", required=True, help="Directory with <gta_task_id>/final.txt and optional files/")
    parser.add_argument("--out-pack", default=str(DEFAULT_OUT))
    parser.add_argument("--task-id", action="append", default=None, help="GTA local task id; repeat or comma-separate")
    parser.add_argument("--allow-missing", action="store_true", help="Write a partial pack and list missing final.txt files")
    parser.add_argument("--dry-score", action="store_true", help="Run the upstream GTA scorer in --dry-run mode")
    args = parser.parse_args()

    tasks_path = resolve_path(args.tasks)
    gta_end_path = resolve_path(args.gta_end)
    result_dir = resolve_path(args.result_dir)
    out_pack = resolve_path(args.out_pack)
    task_ids = parse_task_ids(args.task_id)

    pack = build_pack(
        tasks_path=tasks_path,
        gta_end_path=gta_end_path,
        result_dir=result_dir,
        out_pack=out_pack,
        task_ids=task_ids,
        allow_missing=args.allow_missing,
    )

    if args.dry_score:
        run_dry_score(out_pack)

    print(
        json.dumps(
            {
                "out_pack": str(out_pack),
                "requested_tasks": pack["summary"]["requested_tasks"],
                "packed_tasks": pack["summary"]["packed_tasks"],
                "missing_final_outputs": pack["summary"]["missing_final_outputs"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
