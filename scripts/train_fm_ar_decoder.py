import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


SPECIALS = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}


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


def torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_samples(path):
    payload = torch_load(path, map_location="cpu")
    samples = payload.get("samples") if isinstance(payload, dict) else payload
    if not isinstance(samples, list):
        raise ValueError(f"{path} must contain samples.")
    return samples


def sample_key(sample):
    return (sample.get("trajectory_id"), int(sample.get("step_idx") or 0))


def text_key(row):
    return (row.get("trajectory_id"), int(row.get("step_idx") or 0))


def load_text(path):
    return {text_key(row): row for row in read_jsonl(path)} if path else {}


def collect_step0(samples):
    return [sample for sample in samples if int(sample.get("step_idx") or 0) == 0]


def build_vocab(samples):
    vocab = dict(SPECIALS)
    for tool in sorted({tool for sample in samples for tool in sample.get("future_tools") or []}):
        vocab[tool] = len(vocab)
    return vocab


def encode_path(tools, vocab, max_len):
    ids = [vocab["<bos>"]]
    ids.extend(vocab.get(tool, vocab["<unk>"]) for tool in (tools or [])[: max_len - 2])
    ids.append(vocab["<eos>"])
    ids = ids[:max_len]
    if len(ids) < max_len:
        ids.extend([vocab["<pad>"]] * (max_len - len(ids)))
    return ids


def decode_path(ids, id_to_tool):
    out = []
    for idx in ids:
        token = id_to_tool.get(int(idx), "<unk>")
        if token == "<eos>":
            break
        if token in SPECIALS:
            continue
        out.append(token)
    return out


class ARDataset(Dataset):
    def __init__(self, samples, vocab, max_len, input_keys):
        self.items = [(sample, input_key) for sample in samples for input_key in input_keys]
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        sample, input_key = self.items[idx]
        ids = torch.tensor(encode_path(sample.get("future_tools") or [], self.vocab, self.max_len), dtype=torch.long)
        return {
            "x": sample[input_key].float().reshape(-1),
            "input_ids": ids[:-1],
            "labels": ids[1:],
        }


class ARDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, vocab_size, num_layers=1, dropout=0.1):
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

    def init_hidden(self, x):
        h = self.latent_to_hidden(x).view(x.shape[0], self.num_layers, self.hidden_dim)
        return h.transpose(0, 1).contiguous()

    def forward(self, x, input_ids):
        h0 = self.init_hidden(x)
        emb = self.token_embed(input_ids)
        out, _ = self.gru(emb, h0)
        return self.out(out)

    @torch.no_grad()
    def generate(self, x, bos_id, eos_id, max_len, allowed_ids=None, eos_bias=0.0, min_len=0):
        self.eval()
        h = self.init_hidden(x)
        cur = torch.full((x.shape[0], 1), bos_id, dtype=torch.long, device=x.device)
        seq = []
        for step_idx in range(max_len - 1):
            out, h = self.gru(self.token_embed(cur), h)
            logits = self.out(out[:, -1])
            if allowed_ids is not None:
                mask = torch.full_like(logits, -1e9)
                for row_idx, ids in enumerate(allowed_ids):
                    mask[row_idx, ids] = 0.0
                logits = logits + mask
            if eos_bias:
                logits[:, eos_id] = logits[:, eos_id] + float(eos_bias)
            if step_idx < min_len:
                logits[:, eos_id] = -1e9
            nxt = logits.argmax(dim=-1)
            seq.append(nxt.cpu())
            cur = nxt.unsqueeze(1)
        return torch.stack(seq, dim=1)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = int(embed_dim)

    def forward(self, t):
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        half_dim = self.embed_dim // 2
        freq = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, device=t.device, dtype=t.dtype)
            / max(half_dim - 1, 1)
        )
        emb = torch.cat([torch.sin(t * freq.unsqueeze(0)), torch.cos(t * freq.unsqueeze(0))], dim=-1)
        return torch.cat([emb, t], dim=-1) if self.embed_dim % 2 else emb


class FMVelocityModel(nn.Module):
    def __init__(self, x_dim, cond_dim, time_embed_dim, target_dim, hidden_dims, dropout=0.0):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.x_norm = nn.LayerNorm(x_dim)
        self.cond_norm = nn.LayerNorm(cond_dim)
        dims = [x_dim + cond_dim + time_embed_dim] + [int(x) for x in hidden_dims]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.GELU()]
            if dropout:
                layers.append(nn.Dropout(dropout))
        self.backbone = nn.Sequential(*layers)
        self.velocity_head = nn.Linear(dims[-1], target_dim)

    def forward(self, x_t, t, c_i):
        return self.velocity_head(self.backbone(torch.cat([self.x_norm(x_t), self.cond_norm(c_i), self.time_embed(t)], dim=-1)))


def cfg(config: dict[str, Any], dotted, default):
    cur = config
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def build_fm(ckpt, device):
    state = ckpt["model"]
    config = ckpt.get("config") or {}
    target_dim = state["x_norm.weight"].numel()
    cond_dim = state["cond_norm.weight"].numel()
    model = FMVelocityModel(
        target_dim,
        cond_dim,
        int(cfg(config, "model.time_embed_dim", 128)),
        target_dim,
        cfg(config, "model.hidden_dims", [1024, 1024]),
        float(cfg(config, "model.dropout", 0.0)),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def sample_fm(fm, samples, args, device, key):
    for start in range(0, len(samples), args.predict_batch_size):
        batch = samples[start : start + args.predict_batch_size]
        c_i = torch.stack([s["c_i"].float().reshape(-1) for s in batch]).to(device)
        gen = torch.Generator(device=device)
        gen.manual_seed(args.seed + start)
        x_t = torch.randn(c_i.shape[0], fm.x_norm.normalized_shape[0], generator=gen, device=device) * args.fm_noise_std
        dt = 1.0 / args.inference_steps
        for step in range(args.inference_steps):
            t = torch.full((c_i.shape[0], 1), step * dt, device=device)
            x_t = x_t + dt * fm(x_t, t, c_i)
        for sample, vec in zip(batch, x_t.cpu()):
            sample[key] = vec


def eval_ar(model, loader, vocab_size, pad_id, device):
    model.eval()
    token_hits = total_tokens = seq_hits = total_seq = 0
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            inp = batch["input_ids"].to(device)
            y = batch["labels"].to(device)
            logits = model(x, inp)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1), ignore_index=pad_id)
            pred = logits.argmax(dim=-1)
            mask = y != pad_id
            token_hits += ((pred == y) & mask).sum().item()
            total_tokens += mask.sum().item()
            seq_hits += ((pred == y) | ~mask).all(dim=-1).sum().item()
            total_seq += y.shape[0]
            total_loss += loss.item() * mask.sum().item()
    return {
        "loss": total_loss / max(total_tokens, 1),
        "token_acc": token_hits / max(total_tokens, 1),
        "sequence_exact": seq_hits / max(total_seq, 1),
    }


def train(args, device):
    train_samples = collect_step0(load_samples(args.train_pt))
    dev_samples = collect_step0(load_samples(args.dev_pt))
    input_keys = ["y_i"]
    if args.train_input in {"fm", "both"}:
        fm = build_fm(torch_load(args.fm_checkpoint, map_location=device), device)
        sample_fm(fm, train_samples, args, device, "fm_y_hat")
        sample_fm(fm, dev_samples, args, device, "fm_y_hat")
        input_keys = ["fm_y_hat"] if args.train_input == "fm" else ["y_i", "fm_y_hat"]
    vocab = build_vocab(train_samples)
    train_ds = ARDataset(train_samples, vocab, args.max_len, input_keys)
    dev_ds = ARDataset(dev_samples, vocab, args.max_len, ["fm_y_hat"] if "fm_y_hat" in input_keys else ["y_i"])
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    dev_dl = DataLoader(dev_ds, batch_size=args.batch_size)
    model = ARDecoder(train_samples[0][input_keys[0]].numel(), args.hidden_dim, len(vocab), args.num_layers, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = None
    best_metric = -1
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
        metrics = eval_ar(model, dev_dl, len(vocab), vocab["<pad>"], device)
        print(json.dumps({"epoch": epoch + 1, **metrics}), flush=True)
        if metrics["sequence_exact"] > best_metric:
            best_metric = metrics["sequence_exact"]
            best = {k: v.cpu() for k, v in model.state_dict().items()}
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": best or model.state_dict(), "vocab": vocab, "config": vars(args), "input_keys": input_keys},
        Path(args.out_dir) / "ar_decoder.pt",
    )


def allowed_ids_for(row, vocab):
    ids = {vocab["<eos>"]}
    for tool in row.get("available_tools") or []:
        if tool in vocab:
            ids.add(vocab[tool])
    return sorted(ids)


def predict(args, device):
    tools = {row["tool_id"]: row for row in read_jsonl(args.tools)}
    payload = torch_load(Path(args.out_dir) / "ar_decoder.pt", map_location=device)
    vocab = payload["vocab"]
    id_to_tool = {idx: tool for tool, idx in vocab.items()}
    model = ARDecoder(768, args.hidden_dim, len(vocab), args.num_layers, args.dropout).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    fm = build_fm(torch_load(args.fm_checkpoint, map_location=device), device)
    samples = collect_step0(load_samples(args.eval_pt))
    text = load_text(args.eval_text)
    rows = []
    for start in range(0, len(samples), args.predict_batch_size):
        batch = samples[start : start + args.predict_batch_size]
        sample_fm(fm, batch, args, device, "fm_y_hat")
        x = torch.stack([s["fm_y_hat"].float() for s in batch]).to(device)
        allowed = None
        if args.mask_available:
            allowed = [allowed_ids_for(text.get(sample_key(s), {}), vocab) for s in batch]
        pred = model.generate(
            x,
            vocab["<bos>"],
            vocab["<eos>"],
            args.max_len,
            allowed_ids=allowed,
            eos_bias=args.eos_bias,
            min_len=args.min_decode_len,
        )
        for sample, ids in zip(batch, pred.tolist()):
            tool_ids = decode_path(ids, id_to_tool)
            rows.append(
                {
                    "task_id": sample.get("trajectory_id"),
                    "source": sample.get("source"),
                    "plan_type": "fm_ar_pointer_decoder_v1",
                    "tool_ids": tool_ids,
                    "tool_names": [tools.get(t, {}).get("name") or str(t).rsplit("::", 1)[-1] for t in tool_ids],
                    "phase_names": [tools.get(t, {}).get("phase", "unknown") for t in tool_ids],
                    "actions": [{"tool_id": t, "tool_name": tools.get(t, {}).get("name") or str(t).rsplit("::", 1)[-1], "arguments": {}} for t in tool_ids],
                    "metadata": {"decoder": "ar_gru_available_mask", "mask_available": args.mask_available},
                }
            )
    write_jsonl(Path(args.pred_out), rows)
    print(json.dumps({"predictions": len(rows), "out": args.pred_out}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "predict", "train_predict"], default="train_predict")
    parser.add_argument("--train-pt", required=True)
    parser.add_argument("--dev-pt", required=True)
    parser.add_argument("--eval-pt")
    parser.add_argument("--eval-text")
    parser.add_argument("--tools")
    parser.add_argument("--fm-checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--pred-out")
    parser.add_argument("--train-input", choices=["gold", "fm", "both"], default="fm")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--predict-batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--max-len", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--fm-noise-std", type=float, default=0.01)
    parser.add_argument("--inference-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--mask-available", action="store_true")
    parser.add_argument("--eos-bias", type=float, default=0.0)
    parser.add_argument("--min-decode-len", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    torch.manual_seed(args.seed)
    if args.mode in {"train", "train_predict"}:
        train(args, device)
    if args.mode in {"predict", "train_predict"}:
        predict(args, device)


if __name__ == "__main__":
    main()
