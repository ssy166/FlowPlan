import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_fm_prefix_sft_generation import available_tool_maps, read_jsonl, write_jsonl  # noqa: E402
from train_fm_next_tool_prior import (  # noqa: E402
    STOP,
    label_for_row,
    load_composed_latents,
    load_tools,
    sample_weight,
    stable_hash,
    state_feature_vector,
)


def add_hash(vec, token, scale=1.0):
    idx = stable_hash(token) % vec.numel()
    sign = 1.0 if stable_hash(f"sign::{token}") % 2 == 0 else -1.0
    vec[idx] += sign * scale


def tool_feature_vector(tool_id, tools, dim):
    vec = torch.zeros(dim, dtype=torch.float32)
    if tool_id == STOP:
        add_hash(vec, "decision:stop")
        return vec
    tool = tools.get(tool_id) or {}
    name = tool.get("name") or tool_id.rsplit("::", 1)[-1]
    add_hash(vec, f"tool_id:{tool_id}")
    add_hash(vec, f"tool_name:{name}")
    add_hash(vec, f"phase:{tool.get('phase') or 'unknown'}")
    if "::retail::" in tool_id:
        add_hash(vec, "domain:retail")
        for marker in ["get_", "find_", "return_", "exchange_", "modify_", "cancel_"]:
            if name.startswith(marker):
                add_hash(vec, f"retail_tool_family:{marker.rstrip('_')}")
        for marker in ["order", "product", "user", "payment", "address", "item"]:
            if marker in name:
                add_hash(vec, f"retail_tool_topic:{marker}")
    schema = tool.get("schema") or {}
    params = schema.get("parameters") or schema.get("required_parameters") or tool.get("parameters") or []
    if isinstance(params, dict):
        params = sorted(params)
    for param in params or []:
        if isinstance(param, dict):
            param = param.get("name")
        if isinstance(param, str) and param:
            add_hash(vec, f"param:{param}")
    norm = vec.norm(p=2)
    if norm > 0:
        vec = vec / norm
    return vec


def load_items(path, tensor_path, tools, args, device):
    rows = read_jsonl(path)
    latents = load_composed_latents(tensor_path, args.latent_key, args, device)
    items = []
    missing = 0
    for row in rows:
        row_id = row.get("id")
        latent = latents.get(row_id)
        if latent is None:
            missing += 1
            continue
        latent = latent.float().reshape(-1)
        state_vec = state_feature_vector(row, args)
        if state_vec is not None:
            latent = torch.cat([latent, state_vec])
        label = label_for_row(row)
        available_ids, _ = available_tool_maps(row, tools)
        candidates = [STOP] + [tool_id for tool_id in available_ids if tool_id in tools]
        if label not in candidates:
            label = STOP
        items.append((row, latent, candidates, label))
    if missing:
        print(json.dumps({"missing_latent": missing, "path": str(path)}), flush=True)
    return items


class RerankDataset(Dataset):
    def __init__(self, items, tools, tool_features, args):
        self.items = items
        self.tools = tools
        self.tool_features = tool_features
        self.args = args

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        row, latent, candidates, label = self.items[idx]
        label_idx = candidates.index(label)
        cand_features = torch.stack([self.tool_features[candidate] for candidate in candidates])
        return {
            "latent": latent,
            "candidate_features": cand_features,
            "label": torch.tensor(label_idx, dtype=torch.long),
            "weight": torch.tensor(sample_weight(row, label, self.tools, self.args), dtype=torch.float32),
            "candidates": candidates,
            "row": row,
        }


def collate(batch):
    max_candidates = max(item["candidate_features"].shape[0] for item in batch)
    feature_dim = batch[0]["candidate_features"].shape[1]
    candidate_features = []
    candidate_mask = []
    candidates = []
    for item in batch:
        count = item["candidate_features"].shape[0]
        pad = max_candidates - count
        if pad:
            candidate_features.append(torch.cat([item["candidate_features"], torch.zeros(pad, feature_dim)], dim=0))
        else:
            candidate_features.append(item["candidate_features"])
        candidate_mask.append(torch.cat([torch.ones(count, dtype=torch.bool), torch.zeros(pad, dtype=torch.bool)]))
        candidates.append(item["candidates"] + [STOP] * pad)
    return {
        "latent": torch.stack([item["latent"] for item in batch]),
        "candidate_features": torch.stack(candidate_features),
        "candidate_mask": torch.stack(candidate_mask),
        "label": torch.stack([item["label"] for item in batch]),
        "weight": torch.stack([item["weight"] for item in batch]),
        "candidates": candidates,
        "rows": [item["row"] for item in batch],
    }


class ToolReranker(nn.Module):
    def __init__(self, context_dim, tool_dim, hidden_dim, dropout):
        super().__init__()
        self.context = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.tool = nn.Sequential(
            nn.LayerNorm(tool_dim),
            nn.Linear(tool_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.scorer = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, latent, candidate_features, candidate_mask):
        context = self.context(latent)
        tool = self.tool(candidate_features)
        context_expanded = context.unsqueeze(1).expand_as(tool)
        pair = torch.cat([context_expanded, tool, context_expanded * tool], dim=-1)
        logits = self.scorer(pair).squeeze(-1)
        return logits.masked_fill(~candidate_mask, -1e9)


def summarize(preds, labels, rows):
    by_domain = {}
    for pred, label, row in zip(preds, labels, rows):
        domain = (row.get("metadata") or {}).get("domain") or "unknown"
        stats = by_domain.setdefault(domain, {"count": 0, "hits": 0})
        stats["count"] += 1
        stats["hits"] += int(pred == label)
    return {
        "count": len(labels),
        "accuracy": sum(int(p == y) for p, y in zip(preds, labels)) / max(len(labels), 1),
        "by_domain": {
            domain: {"count": stats["count"], "accuracy": stats["hits"] / max(stats["count"], 1)}
            for domain, stats in sorted(by_domain.items())
        },
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels, rows = [], [], []
    loss_sum = 0.0
    for batch in loader:
        logits = model(
            batch["latent"].to(device),
            batch["candidate_features"].to(device),
            batch["candidate_mask"].to(device),
        )
        label = batch["label"].to(device)
        loss = F.cross_entropy(logits, label)
        pred = logits.argmax(dim=-1)
        loss_sum += float(loss.item()) * label.numel()
        preds.extend(pred.cpu().tolist())
        labels.extend(label.cpu().tolist())
        rows.extend(batch["rows"])
    out = summarize(preds, labels, rows)
    out["loss"] = loss_sum / max(len(labels), 1)
    return out


@torch.no_grad()
def write_predictions(model, dataset, out_path, device, tools, model_name):
    loader = DataLoader(dataset, batch_size=128, shuffle=False, collate_fn=collate)
    rows_out = []
    model.eval()
    for batch in loader:
        logits = model(
            batch["latent"].to(device),
            batch["candidate_features"].to(device),
            batch["candidate_mask"].to(device),
        )
        pred = logits.argmax(dim=-1).cpu().tolist()
        probs = torch.softmax(logits, dim=-1).max(dim=-1).values.cpu().tolist()
        for row, candidates, pred_idx, prob in zip(batch["rows"], batch["candidates"], pred, probs):
            tool_id = candidates[pred_idx]
            metadata = row.get("metadata") or {}
            if tool_id == STOP:
                tool_ids = []
                actions = []
            else:
                tool = tools.get(tool_id) or {}
                tool_ids = [tool_id]
                actions = [{"tool_id": tool_id, "tool_name": tool.get("name") or tool_id.rsplit("::", 1)[-1], "arguments": {}}]
            rows_out.append(
                {
                    "task_id": metadata.get("task_id") or row.get("id"),
                    "record_id": row.get("id"),
                    "source": metadata.get("source"),
                    "domain": metadata.get("domain"),
                    "plan_type": model_name,
                    "tool_ids": tool_ids,
                    "actions": actions,
                    "metadata": {"record_id": row.get("id"), "confidence": prob, "decoder": model_name},
                }
            )
    write_jsonl(out_path, rows_out)
    print(json.dumps({"predictions": len(rows_out), "out": str(out_path)}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train a candidate-aware FM next-tool reranker.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--tensor-dir", required=True)
    parser.add_argument("--tools", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--latent-key", choices=["c_i", "y_i", "proxy_c_i", "y_hat"], default="c_i")
    parser.add_argument("--feature-mode", choices=["single", "ci_yhat_delta"], default="single")
    parser.add_argument("--state-feature-mode", choices=["none", "compact"], default="compact")
    parser.add_argument("--state-feature-detail", choices=["basic", "result"], default="basic")
    parser.add_argument("--state-feature-dim", type=int, default=512)
    parser.add_argument("--state-feature-scale", type=float, default=1.0)
    parser.add_argument("--tool-feature-dim", type=int, default=256)
    parser.add_argument("--retail-loss-weight", type=float, default=1.0)
    parser.add_argument("--post-lookup-loss-weight", type=float, default=1.0)
    parser.add_argument("--operation-loss-weight", type=float, default=1.0)
    parser.add_argument("--stop-loss-weight", type=float, default=1.0)
    parser.add_argument("--fm-checkpoint", default=None)
    parser.add_argument("--fm-inference-steps", type=int, default=16)
    parser.add_argument("--fm-noise-std", type=float, default=0.01)
    parser.add_argument("--fm-sample-batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    tools = load_tools(args.tools)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tool_features = {STOP: tool_feature_vector(STOP, tools, args.tool_feature_dim)}
    for tool_id in tools:
        tool_features[tool_id] = tool_feature_vector(tool_id, tools, args.tool_feature_dim)

    split_items = {
        split: load_items(
            Path(args.data_dir) / f"{split}.jsonl",
            Path(args.tensor_dir) / f"{split}.pt",
            tools,
            args,
            device,
        )
        for split in ["train", "dev", "test"]
    }
    datasets = {split: RerankDataset(items, tools, tool_features, args) for split, items in split_items.items()}
    loaders = {
        split: DataLoader(ds, batch_size=args.batch_size, shuffle=(split == "train"), collate_fn=collate)
        for split, ds in datasets.items()
    }
    context_dim = split_items["train"][0][1].numel()
    model = ToolReranker(context_dim, args.tool_feature_dim, args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = None
    best_acc = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in loaders["train"]:
            logits = model(
                batch["latent"].to(device),
                batch["candidate_features"].to(device),
                batch["candidate_mask"].to(device),
            )
            label = batch["label"].to(device)
            losses = F.cross_entropy(logits, label, reduction="none")
            weight = batch["weight"].to(device)
            loss = (losses * weight).sum() / weight.sum().clamp_min(1e-6)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        metrics = {"epoch": epoch, "dev": evaluate(model, loaders["dev"], device)}
        history.append(metrics)
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps(metrics, ensure_ascii=False, sort_keys=True), flush=True)
        if metrics["dev"]["accuracy"] > best_acc:
            best_acc = metrics["dev"]["accuracy"]
            best = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    if best is not None:
        model.load_state_dict(best)
    final_metrics = {split: evaluate(model, loaders[split], device) for split in ["train", "dev", "test"]}
    payload = {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "config": vars(args), "metrics": final_metrics, "history": history}
    torch.save(payload, out_dir / "tool_reranker.pt")
    (out_dir / "metrics.json").write_text(json.dumps(final_metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"final": final_metrics}, ensure_ascii=False, sort_keys=True), flush=True)
    for split in ["train", "dev", "test"]:
        write_predictions(model, datasets[split], out_dir / f"{split}.pred.jsonl", device, tools, "fm_tool_reranker_v1")


if __name__ == "__main__":
    main()
