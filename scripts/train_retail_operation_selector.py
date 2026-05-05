import argparse
import hashlib
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


WORD_RE = re.compile(r"[a-zA-Z0-9_]+")


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def stable_hash(text):
    digest = hashlib.blake2b(str(text).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def add_token(tokens, name, value=True):
    if value is None:
        return
    if isinstance(value, bool):
        tokens.append(f"{name}:{str(value).lower()}")
    elif isinstance(value, (int, float)):
        tokens.append(f"{name}:{value}")
    elif isinstance(value, str) and value:
        tokens.append(f"{name}:{value.lower()}")


def row_tokens(row):
    tokens = []
    task = row.get("task") or ""
    for word in WORD_RE.findall(task.lower()):
        if len(word) >= 2:
            tokens.append(f"word:{word}")
    for tag in row.get("intent_tags") or []:
        tokens.append(f"intent:{tag}")
    for tool in row.get("available_tool_names") or []:
        add_token(tokens, "available_tool", tool)
    progress = row.get("grounding_progress") or {}
    for key in ["lookup_completed", "order_grounded", "last_successful_lookup_tool", "resolved_order_status"]:
        add_token(tokens, key, progress.get(key))
    for key in ["operation_eligibility", "resolved_order_ids", "resolved_order_item_ids", "resolved_payment_method_ids"]:
        values = progress.get(key) or []
        tokens.append(f"{key}_count:{len(values)}")
        for value in values[:16]:
            add_token(tokens, key, value)
    prefix = row.get("prefix") or []
    tokens.append(f"prefix_len:{len(prefix)}")
    for step in prefix[-4:]:
        if not isinstance(step, dict):
            continue
        add_token(tokens, "prefix_tool", step.get("tool_name"))
        add_token(tokens, "prefix_ok", step.get("ok"))
        add_token(tokens, "prefix_result_status", step.get("result_status"))
        add_token(tokens, "prefix_has_order_id", step.get("has_order_id"))
    return tokens


def vectorize(row, dim):
    vec = torch.zeros(dim, dtype=torch.float32)
    for token in row_tokens(row):
        idx = stable_hash(token) % dim
        sign = 1.0 if stable_hash(f"sign::{token}") % 2 == 0 else -1.0
        vec[idx] += sign
    norm = vec.norm(p=2)
    return vec / norm if norm > 0 else vec


def operation_rows(rows):
    return [row for row in rows if row.get("decision_case") == "post_order_operation"]


def target_name(row):
    return (row.get("target") or {}).get("tool_name") or "unknown"


def summarize(pred_rows):
    if not pred_rows:
        return {"count": 0, "accuracy": None, "by_gold": {}}
    by_gold = {}
    correct = 0
    for row in pred_rows:
        if row["predicted_tool_name"] == row["gold_tool_name"]:
            correct += 1
        item = by_gold.setdefault(row["gold_tool_name"], {"count": 0, "correct": 0})
        item["count"] += 1
        item["correct"] += int(row["predicted_tool_name"] == row["gold_tool_name"])
    for item in by_gold.values():
        item["accuracy"] = item["correct"] / item["count"] if item["count"] else None
    return {"count": len(pred_rows), "accuracy": correct / len(pred_rows), "by_gold": by_gold}


def main():
    parser = argparse.ArgumentParser(description="Train a small hashed linear selector for retail operation tools.")
    parser.add_argument("--decision-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    decision_dir = Path(args.decision_dir)
    splits = {split: operation_rows(read_jsonl(decision_dir / f"{split}.jsonl")) for split in ["train", "dev", "test"]}
    labels = sorted({target_name(row) for row in splits["train"]})
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    model = nn.Linear(args.dim, len(labels))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    x_train = torch.stack([vectorize(row, args.dim) for row in splits["train"]])
    y_train = torch.tensor([label_to_idx[target_name(row)] for row in splits["train"]], dtype=torch.long)
    class_counts = torch.bincount(y_train, minlength=len(labels)).float()
    class_weights = class_counts.sum() / class_counts.clamp_min(1.0)
    class_weights = class_weights / class_weights.mean()

    for epoch in range(1, args.epochs + 1):
        model.train()
        logits = model(x_train)
        loss = F.cross_entropy(logits, y_train, weight=class_weights)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch in {1, 10, 50, 100, args.epochs}:
            acc = (logits.argmax(dim=-1) == y_train).float().mean().item()
            print(json.dumps({"epoch": epoch, "loss": float(loss.detach()), "train_accuracy": acc}, sort_keys=True))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"labels": labels, "splits": {}, "config": vars(args)}
    for split, rows in splits.items():
        pred_rows = []
        if rows:
            x = torch.stack([vectorize(row, args.dim) for row in rows])
            probs = torch.softmax(model(x), dim=-1)
            pred_idx = probs.argmax(dim=-1)
            for row, idx, prob in zip(rows, pred_idx.tolist(), probs.tolist()):
                pred_rows.append(
                    {
                        "id": row["id"],
                        "split": split,
                        "task_id": row.get("task_id"),
                        "gold_tool_name": target_name(row),
                        "predicted_tool_name": labels[idx],
                        "confidence": max(prob),
                        "decision_case": row.get("decision_case"),
                    }
                )
        write_jsonl(out_dir / f"{split}.pred.jsonl", pred_rows)
        manifest["splits"][split] = summarize(pred_rows)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest["splits"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
