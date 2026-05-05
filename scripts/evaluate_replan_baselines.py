import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STOP = "<stop>"


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


def target_path(record: dict[str, Any]) -> tuple[str, ...]:
    tools = record.get("target_remaining_tool_ids") or []
    return tuple(tools) if tools else (STOP,)


def decode_path(path: tuple[str, ...]) -> list[str]:
    return [] if path == (STOP,) else list(path)


def edit_distance(a: list[str], b: list[str]) -> int:
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j
    for i, av in enumerate(a, start=1):
        for j, bv in enumerate(b, start=1):
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + (0 if av == bv else 1),
            )
    return dp[-1][-1]


def metrics(records: list[dict[str, Any]], pred_path: tuple[str, ...]) -> dict[str, Any]:
    pred = decode_path(pred_path)
    hits = 0
    stop_hits = 0
    total_ed = 0
    for record in records:
        gold = decode_path(target_path(record))
        hits += int(pred == gold)
        stop_hits += int((not pred) == (not gold))
        total_ed += edit_distance(pred, gold)
    count = len(records)
    return {
        "count": count,
        "prediction": list(pred_path),
        "path_em": hits / count if count else 0.0,
        "stop_acc": stop_hits / count if count else 0.0,
        "avg_edit_distance": total_ed / count if count else 0.0,
    }


def split_summary(records: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    counter = Counter(target_path(record) for record in records)
    count = len(records)
    return {
        "count": count,
        "unique_targets": len(counter),
        "stop_rate": counter[(STOP,)] / count if count else 0.0,
        "top_targets": [
            {"target": list(path), "count": target_count, "rate": target_count / count if count else 0.0}
            for path, target_count in counter.most_common(top_k)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate simple replan baselines and target distributions.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--out", default=str(ROOT / "data" / "processed" / "closed_loop" / "replan_baselines.telecom.json"))
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    splits = {
        "train": read_jsonl(args.train),
        "dev": read_jsonl(args.dev),
        "test": read_jsonl(args.test),
    }
    train_counter = Counter(target_path(record) for record in splits["train"])
    majority_path = train_counter.most_common(1)[0][0] if train_counter else (STOP,)
    baselines = {
        "train_majority_target": majority_path,
        "always_stop": (STOP,),
    }
    output = {
        "format": "replan_baseline_eval_v1",
        "inputs": {"train": args.train, "dev": args.dev, "test": args.test},
        "target_distribution": {split: split_summary(rows, args.top_k) for split, rows in splits.items()},
        "baselines": {},
    }
    for name, pred_path in baselines.items():
        output["baselines"][name] = {split: metrics(rows, pred_path) for split, rows in splits.items()}

    write_json(args.out, output)
    print(json.dumps({"out": args.out, "majority_target": list(majority_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
