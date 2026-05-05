import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


LABELS = [
    "initial_lookup",
    "post_order_detail_lookup",
    "post_order_operation",
    "post_user_order_lookup",
    "other",
    "stop",
]


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


def add(tokens, name, value=True):
    if value is None:
        return
    if isinstance(value, bool):
        tokens.append(f"{name}:{str(value).lower()}")
    elif isinstance(value, (int, float)):
        tokens.append(f"{name}:{value}")
    elif isinstance(value, str) and value:
        tokens.append(f"{name}:{value.lower()}")


def bucket(name, count):
    if count <= 0:
        value = "0"
    elif count == 1:
        value = "1"
    elif count <= 3:
        value = "2-3"
    else:
        value = "4+"
    return f"{name}:{value}"


def tokens_for(row):
    tokens = []
    add(tokens, "step", min(int(row.get("step") or 0), 8))
    for tag in row.get("intent_tags") or []:
        add(tokens, "intent", tag)
    task = str(row.get("task") or "").lower()
    for word in [
        "return",
        "refund",
        "exchange",
        "replace",
        "cancel",
        "modify",
        "change",
        "address",
        "payment",
        "item",
        "product",
        "delivered",
        "pending",
        "order",
    ]:
        if word in task:
            add(tokens, "word", word)
    progress = row.get("grounding_progress") or {}
    for key in ["lookup_completed", "order_grounded"]:
        add(tokens, key, progress.get(key))
    for key in ["last_successful_lookup_tool", "resolved_order_status"]:
        add(tokens, key, progress.get(key))
    for key in ["resolved_order_ids", "resolved_order_item_ids", "resolved_payment_method_ids", "operation_eligibility"]:
        values = progress.get(key) or []
        tokens.append(bucket(key, len(values)))
        for value in values[:12]:
            add(tokens, key, value)
    for tool in row.get("available_tool_names") or []:
        add(tokens, "available_tool", tool)
    prefix = row.get("prefix") or []
    tokens.append(bucket("prefix_len", len(prefix)))
    for idx, step in enumerate(prefix[-4:]):
        add(tokens, f"prefix_tool_{idx}", step.get("tool_name"))
        add(tokens, "prefix_any_tool", step.get("tool_name"))
        add(tokens, "prefix_ok", step.get("ok"))
        add(tokens, "prefix_status", step.get("result_status"))
        add(tokens, "prefix_has_order_id", step.get("has_order_id"))
        for key in ["item_count", "product_count", "payment_count"]:
            tokens.append(bucket(f"prefix_{key}", int(step.get(key) or 0)))
    return tokens


def vectorize(row, dim):
    vec = torch.zeros(dim, dtype=torch.float32)
    for token in tokens_for(row):
        idx = stable_hash(token) % dim
        sign = 1.0 if stable_hash(f"sign::{token}") % 2 == 0 else -1.0
        vec[idx] += sign
    norm = vec.norm()
    if norm > 0:
        vec = vec / norm
    return vec


class DecisionDataset(Dataset):
    def __init__(self, rows, dim):
        self.rows = rows
        self.dim = dim
        self.label_to_idx = {label: idx for idx, label in enumerate(LABELS)}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        label = self.label_to_idx.get(row.get("decision_case"), self.label_to_idx["other"])
        return vectorize(row, self.dim), torch.tensor(label), row.get("id")


class Selector(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, len(LABELS)))

    def forward(self, x):
        return self.net(x)


def class_weights(rows):
    counts = {label: 0 for label in LABELS}
    for row in rows:
        counts[row.get("decision_case") if row.get("decision_case") in counts else "other"] += 1
    total = sum(counts.values())
    return torch.tensor([total / max(1, counts[label]) for label in LABELS], dtype=torch.float32)


@torch.no_grad()
def evaluate(model, rows, dim, batch_size, out_path=None):
    ds = DecisionDataset(rows, dim)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    label_to_idx = {label: idx for idx, label in enumerate(LABELS)}
    confusion = {}
    pred_rows = []
    correct = 0
    total = 0
    op_tp = op_fp = op_fn = 0
    for x, y, ids in loader:
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)
        pred = probs.argmax(dim=-1)
        correct += int((pred == y).sum())
        total += int(y.numel())
        for idx, row_id in enumerate(ids):
            gold = LABELS[int(y[idx])]
            guess = LABELS[int(pred[idx])]
            confidence = float(probs[idx, pred[idx]])
            confusion[f"{gold}->{guess}"] = confusion.get(f"{gold}->{guess}", 0) + 1
            op_tp += int(gold == "post_order_operation" and guess == "post_order_operation")
            op_fp += int(gold != "post_order_operation" and guess == "post_order_operation")
            op_fn += int(gold == "post_order_operation" and guess != "post_order_operation")
            pred_rows.append({"id": row_id, "gold": gold, "prediction": guess, "confidence": confidence})
    if out_path:
        write_jsonl(out_path, pred_rows)
    return {
        "accuracy": correct / total if total else 0.0,
        "count": total,
        "operation_precision": op_tp / (op_tp + op_fp) if op_tp + op_fp else None,
        "operation_recall": op_tp / (op_tp + op_fn) if op_tp + op_fn else None,
        "operation_tp": op_tp,
        "operation_fp": op_fp,
        "operation_fn": op_fn,
        "confusion": dict(sorted(confusion.items())),
    }


def main():
    parser = argparse.ArgumentParser(description="Train a small hashed-feature retail decision-case selector.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dim", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_rows = read_jsonl(Path(args.data_dir) / "train.jsonl")
    dev_rows = read_jsonl(Path(args.data_dir) / "dev.jsonl")
    test_rows = read_jsonl(Path(args.data_dir) / "test.jsonl")
    model = Selector(args.dim)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(train_rows))
    loader = DataLoader(DecisionDataset(train_rows, args.dim), batch_size=args.batch_size, shuffle=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        for x, y, _ in loader:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += float(loss) * int(y.numel())
            seen += int(y.numel())
        if epoch % 10 == 0 or epoch == args.epochs:
            dev = evaluate(model, dev_rows, args.dim, args.batch_size)
            history.append({"epoch": epoch, "train_loss": total / max(1, seen), "dev": dev})
            print(json.dumps(history[-1], ensure_ascii=False, sort_keys=True))
    metrics = {
        "train": evaluate(model, train_rows, args.dim, args.batch_size, out_dir / "train.pred.jsonl"),
        "dev": evaluate(model, dev_rows, args.dim, args.batch_size, out_dir / "dev.pred.jsonl"),
        "test": evaluate(model, test_rows, args.dim, args.batch_size, out_dir / "test.pred.jsonl"),
        "labels": LABELS,
        "dim": args.dim,
        "history": history,
    }
    torch.save({"model": model.state_dict(), "dim": args.dim, "labels": LABELS}, out_dir / "retail_decision_selector.pt")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
