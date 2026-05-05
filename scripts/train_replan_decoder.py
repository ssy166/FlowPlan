import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


SPECIALS = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
STOP_TOOL = "<stop>"


def torch_load(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_seq_vocab(*payloads: dict[str, Any]) -> dict[str, int]:
    vocab = dict(SPECIALS)
    tools = {STOP_TOOL}
    for payload in payloads:
        for item in payload.get("metadata") or []:
            tools.update(item.get("future_tool_ids") or [])
            if item.get("next_tool_id"):
                tools.add(item["next_tool_id"])
        tools.update((payload.get("tool_vocab") or {}).keys())
    for tool in sorted(tool for tool in tools if tool not in {"<pad>", "<unk>"}):
        if tool not in vocab:
            vocab[tool] = len(vocab)
    return vocab


def encode_target(metadata: dict[str, Any], vocab: dict[str, int], max_len: int) -> list[int]:
    tools = metadata.get("future_tool_ids") or []
    if not tools:
        tools = [STOP_TOOL]
    ids = [vocab["<bos>"]]
    ids.extend(vocab.get(tool, vocab["<unk>"]) for tool in tools[: max_len - 2])
    ids.append(vocab["<eos>"])
    ids = ids[:max_len]
    if len(ids) < max_len:
        ids.extend([vocab["<pad>"]] * (max_len - len(ids)))
    return ids


def decode_target(ids: list[int], id_to_tool: dict[int, str]) -> list[str]:
    tools = []
    for idx in ids:
        token = id_to_tool.get(int(idx), "<unk>")
        if token == "<eos>":
            break
        if token in SPECIALS:
            continue
        if token == STOP_TOOL:
            return []
        tools.append(token)
    return tools


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


class ReplanDataset(Dataset):
    def __init__(self, payload: dict[str, Any], vocab: dict[str, int], max_len: int):
        self.x = payload["c_i"].float()
        self.metadata = payload.get("metadata") or []
        self.targets = torch.tensor([encode_target(item, vocab, max_len) for item in self.metadata], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids = self.targets[idx]
        return {
            "x": self.x[idx],
            "input_ids": ids[:-1],
            "labels": ids[1:],
        }


class ARDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, vocab_size: int, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim * num_layers)
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout), nn.Linear(hidden_dim, vocab_size))
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def init_hidden(self, x: torch.Tensor) -> torch.Tensor:
        h = self.latent_to_hidden(x).view(x.shape[0], self.num_layers, self.hidden_dim)
        return h.transpose(0, 1).contiguous()

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(self.token_embed(input_ids), self.init_hidden(x))
        return self.out(out)

    @torch.no_grad()
    def generate(self, x: torch.Tensor, bos_id: int, eos_id: int, max_len: int) -> torch.Tensor:
        self.eval()
        h = self.init_hidden(x)
        cur = torch.full((x.shape[0], 1), bos_id, dtype=torch.long, device=x.device)
        seq = []
        for _ in range(max_len - 1):
            out, h = self.gru(self.token_embed(cur), h)
            nxt = self.out(out[:, -1]).argmax(dim=-1)
            seq.append(nxt.cpu())
            cur = nxt.unsqueeze(1)
        return torch.stack(seq, dim=1)


def evaluate(model: ARDecoder, payload: dict[str, Any], vocab: dict[str, int], args: argparse.Namespace, device: torch.device) -> dict[str, float]:
    id_to_tool = {idx: tool for tool, idx in vocab.items()}
    ds = ReplanDataset(payload, vocab, args.max_len)
    dl = DataLoader(ds, batch_size=args.batch_size)
    pred_paths = []
    gold_paths = []
    token_hits = total_tokens = seq_hits = total_seq = 0
    total_loss = 0.0
    model.eval()
    with torch.no_grad():
        for batch in dl:
            x = batch["x"].to(device)
            inp = batch["input_ids"].to(device)
            y = batch["labels"].to(device)
            logits = model(x, inp)
            loss = F.cross_entropy(logits.reshape(-1, len(vocab)), y.reshape(-1), ignore_index=vocab["<pad>"])
            pred = logits.argmax(dim=-1)
            mask = y != vocab["<pad>"]
            token_hits += ((pred == y) & mask).sum().item()
            total_tokens += mask.sum().item()
            seq_hits += ((pred == y) | ~mask).all(dim=-1).sum().item()
            total_seq += y.shape[0]
            total_loss += loss.item() * mask.sum().item()
            gen = model.generate(x, vocab["<bos>"], vocab["<eos>"], args.max_len)
            pred_paths.extend(decode_target(ids, id_to_tool) for ids in gen.tolist())
    gold_paths = [item.get("future_tool_ids") or [] for item in payload.get("metadata") or []]
    path_hits = sum(pred == gold for pred, gold in zip(pred_paths, gold_paths))
    stop_hits = sum((not pred) == (not gold) for pred, gold in zip(pred_paths, gold_paths))
    total_ed = sum(edit_distance(pred, gold) for pred, gold in zip(pred_paths, gold_paths))
    return {
        "loss": total_loss / max(total_tokens, 1),
        "token_acc": token_hits / max(total_tokens, 1),
        "teacher_forced_seq_exact": seq_hits / max(total_seq, 1),
        "path_em": path_hits / max(len(gold_paths), 1),
        "avg_edit_distance": total_ed / max(len(gold_paths), 1),
        "stop_acc": stop_hits / max(len(gold_paths), 1),
        "count": len(gold_paths),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small feedback-conditioned replan decoder.")
    parser.add_argument("--train-pt", required=True)
    parser.add_argument("--dev-pt", required=True)
    parser.add_argument("--test-pt", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--max-len", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    torch.manual_seed(args.seed)
    train_payload = torch_load(args.train_pt)
    dev_payload = torch_load(args.dev_pt)
    test_payload = torch_load(args.test_pt) if args.test_pt else None
    vocab = build_seq_vocab(train_payload, dev_payload, test_payload or {})
    train_ds = ReplanDataset(train_payload, vocab, args.max_len)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    model = ARDecoder(train_payload["c_i"].shape[1], args.hidden_dim, len(vocab), args.num_layers, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_metric = -1.0
    history = []
    for epoch in range(args.epochs):
        model.train()
        for batch in train_dl:
            x = batch["x"].to(device)
            inp = batch["input_ids"].to(device)
            y = batch["labels"].to(device)
            logits = model(x, inp)
            loss = F.cross_entropy(logits.reshape(-1, len(vocab)), y.reshape(-1), ignore_index=vocab["<pad>"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            opt.step()
        dev_metrics = evaluate(model, dev_payload, vocab, args, device)
        row = {"epoch": epoch + 1, "dev": dev_metrics}
        history.append(row)
        print(json.dumps(row), flush=True)
        metric = dev_metrics["path_em"] + dev_metrics["stop_acc"]
        if metric > best_metric:
            best_metric = metric
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if best_state:
        model.load_state_dict(best_state)
    final = {"dev": evaluate(model, dev_payload, vocab, args, device)}
    if test_payload:
        final["test"] = evaluate(model, test_payload, vocab, args, device)
    torch.save({"model": best_state or model.state_dict(), "vocab": vocab, "config": vars(args), "final": final}, out_dir / "replan_decoder.pt")
    write_json(out_dir / "metrics.json", {"history": history, "final": final, "vocab_size": len(vocab)})
    print(json.dumps({"final": final, "out_dir": str(out_dir), "vocab_size": len(vocab)}), flush=True)


if __name__ == "__main__":
    main()
