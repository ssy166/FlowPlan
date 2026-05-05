import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


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
        raise ValueError(f"{path} must contain a sample list or a dict with samples.")
    return samples


def sample_key(sample):
    return (sample.get("trajectory_id"), int(sample.get("step_idx") or 0))


def text_key(row):
    return (row.get("trajectory_id"), int(row.get("step_idx") or 0))


def load_text_metadata(path):
    if not path:
        return {}
    return {text_key(row): row for row in read_jsonl(path)}


def collect_step0(samples):
    return [sample for sample in samples if int(sample.get("step_idx") or 0) == 0]


def build_vocab(samples):
    tools = sorted({tool for sample in samples for tool in sample.get("future_tools") or []})
    vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2}
    for tool in tools:
        vocab[tool] = len(vocab)
    return vocab


def encode_path(tools, vocab, max_len):
    ids = [vocab.get(tool, vocab["<unk>"]) for tool in (tools or [])[: max_len - 1]]
    ids.append(vocab["<eos>"])
    ids = ids[:max_len]
    if len(ids) < max_len:
        ids.extend([vocab["<pad>"]] * (max_len - len(ids)))
    return ids


class PathDataset(Dataset):
    def __init__(self, samples, vocab, max_len, input_key="y_i", noise_std=0.0):
        self.samples = list(samples)
        self.vocab = vocab
        self.max_len = int(max_len)
        self.input_key = input_key
        self.noise_std = float(noise_std)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        x = sample[self.input_key].float().reshape(-1)
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        y = torch.tensor(encode_path(sample.get("future_tools") or [], self.vocab, self.max_len), dtype=torch.long)
        return {"x": x, "y": y}


class PathDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, max_len, vocab_size, dropout=0.1):
        super().__init__()
        self.max_len = int(max_len)
        self.vocab_size = int(vocab_size)
        self.net = nn.Sequential(
            nn.LayerNorm(int(latent_dim)),
            nn.Linear(int(latent_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.max_len * self.vocab_size),
        )

    def forward(self, x):
        return self.net(x).view(x.shape[0], self.max_len, self.vocab_size)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = int(embed_dim)

    def forward(self, t):
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t = t.float()
        half_dim = self.embed_dim // 2
        if half_dim == 0:
            return t
        freq = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, device=t.device, dtype=t.dtype)
            / max(half_dim - 1, 1)
        )
        angles = t * freq.unsqueeze(0)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if self.embed_dim % 2 == 1:
            emb = torch.cat([emb, t], dim=-1)
        return emb


class FMVelocityModel(nn.Module):
    def __init__(self, x_dim, cond_dim, time_embed_dim, target_dim, hidden_dims, dropout=0.0):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.x_norm = nn.LayerNorm(int(x_dim))
        self.cond_norm = nn.LayerNorm(int(cond_dim))
        in_dim = int(x_dim) + int(cond_dim) + int(time_embed_dim)
        dims = [in_dim] + [int(dim) for dim in hidden_dims]
        layers = []
        for dim_in, dim_out in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(dim_in, dim_out))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.velocity_head = nn.Linear(dims[-1], target_dim)

    def forward(self, x_t, t, c_i):
        t_embed = self.time_embed(t)
        model_in = torch.cat([self.x_norm(x_t), self.cond_norm(c_i), t_embed], dim=-1)
        hidden = self.backbone(model_in)
        return self.velocity_head(hidden)


def get_config_value(config: dict[str, Any], dotted: str, default):
    cur: Any = config
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def build_fm_model(checkpoint, device):
    config = checkpoint.get("config") or {}
    state = checkpoint["model"]
    target_dim = int(state["x_norm.weight"].numel()) if "x_norm.weight" in state else 768
    cond_dim = int(state["cond_norm.weight"].numel()) if "cond_norm.weight" in state else target_dim
    model = FMVelocityModel(
        x_dim=target_dim,
        cond_dim=cond_dim,
        time_embed_dim=int(get_config_value(config, "model.time_embed_dim", 128)),
        target_dim=target_dim,
        hidden_dims=get_config_value(config, "model.hidden_dims", [1024, 1024]),
        dropout=float(get_config_value(config, "model.dropout", 0.0)),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def sample_latent(model, c_i, inference_steps, noise_std, seed):
    generator = torch.Generator(device=c_i.device)
    generator.manual_seed(int(seed))
    x_t = torch.randn(c_i.shape[0], model.x_norm.normalized_shape[0], generator=generator, device=c_i.device) * noise_std
    dt = 1.0 / float(max(1, inference_steps))
    for step in range(max(1, inference_steps)):
        t = torch.full((c_i.shape[0], 1), step * dt, device=c_i.device, dtype=c_i.dtype)
        velocity = model(x_t=x_t, t=t, c_i=c_i)
        x_t = x_t + dt * velocity
    return x_t


def decode_ids(ids, id_to_tool):
    out = []
    for idx in ids:
        token = id_to_tool.get(int(idx), "<unk>")
        if token == "<eos>":
            break
        if token in {"<pad>", "<unk>"}:
            continue
        out.append(token)
    return out


def phase_names(tool_ids, tools):
    return [tools.get(tool_id, {}).get("phase", "unknown") for tool_id in tool_ids]


def compact_actions(tool_ids, tools):
    return [
        {
            "tool_id": tool_id,
            "tool_name": tools.get(tool_id, {}).get("name") or str(tool_id).rsplit("::", 1)[-1],
            "arguments": {},
        }
        for tool_id in tool_ids
    ]


def train_decoder(args, device):
    train_samples = collect_step0(load_samples(args.train_pt)) if args.step0_only else load_samples(args.train_pt)
    dev_samples = collect_step0(load_samples(args.dev_pt)) if args.step0_only else load_samples(args.dev_pt)
    if args.train_input == "fm":
        if not args.fm_checkpoint:
            raise ValueError("--train-input fm requires --fm-checkpoint")
        fm = build_fm_model(torch_load(args.fm_checkpoint, map_location=device), device=device)
        add_fm_latents(train_samples, fm, args, device, key="fm_y_hat")
        add_fm_latents(dev_samples, fm, args, device, key="fm_y_hat")
        input_key = "fm_y_hat"
    else:
        input_key = "y_i"
    vocab = build_vocab(train_samples)
    train_ds = PathDataset(train_samples, vocab, args.max_len, input_key=input_key, noise_std=args.noise_std)
    dev_ds = PathDataset(dev_samples, vocab, args.max_len, input_key=input_key, noise_std=0.0)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    dev_dl = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False)

    latent_dim = int(train_samples[0]["y_i"].numel())
    model = PathDecoder(latent_dim, args.hidden_dim, args.max_len, len(vocab), dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pad_id = vocab["<pad>"]
    best_dev_acc = -1.0
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        for batch in train_dl:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, len(vocab)), y.reshape(-1), ignore_index=pad_id)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            opt.step()
            nonpad = (y != pad_id).sum().item()
            total_loss += float(loss.item()) * nonpad
            total_tokens += nonpad
        metrics = evaluate_decoder(model, dev_dl, vocab, device)
        train_loss = total_loss / max(total_tokens, 1)
        print(json.dumps({"epoch": epoch + 1, "train_loss": train_loss, **metrics}), flush=True)
        if metrics["sequence_exact"] > best_dev_acc:
            best_dev_acc = metrics["sequence_exact"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    output = {
        "model": best_state or model.state_dict(),
        "vocab": vocab,
        "config": vars(args),
        "latent_dim": latent_dim,
        "max_len": args.max_len,
    }
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out_dir) / "path_decoder.pt"
    torch.save(output, out_path)
    return out_path


@torch.no_grad()
def add_fm_latents(samples, fm, args, device, key):
    for start in range(0, len(samples), args.predict_batch_size):
        batch = samples[start : start + args.predict_batch_size]
        c_i = torch.stack([sample["c_i"].float().reshape(-1) for sample in batch], dim=0).to(device)
        latent = sample_latent(fm, c_i, args.inference_steps, args.fm_noise_std, args.seed + start).cpu()
        for sample, vec in zip(batch, latent):
            sample[key] = vec


@torch.no_grad()
def evaluate_decoder(model, dataloader, vocab, device):
    model.eval()
    pad_id = vocab["<pad>"]
    total_tokens = 0
    token_hits = 0
    total_seq = 0
    seq_hits = 0
    for batch in dataloader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        pred = model(x).argmax(dim=-1)
        mask = y != pad_id
        token_hits += ((pred == y) & mask).sum().item()
        total_tokens += mask.sum().item()
        seq_hits += ((pred == y) | ~mask).all(dim=-1).sum().item()
        total_seq += y.shape[0]
    return {
        "token_acc": token_hits / max(total_tokens, 1),
        "sequence_exact": seq_hits / max(total_seq, 1),
    }


def apply_available_filter(tool_ids, available):
    if not available:
        return tool_ids
    allowed = set(available)
    return [tool_id for tool_id in tool_ids if tool_id in allowed]


@torch.no_grad()
def predict(args, device):
    tools = {row["tool_id"]: row for row in read_jsonl(args.tools)}
    train_payload = torch_load(Path(args.out_dir) / "path_decoder.pt", map_location=device)
    vocab = train_payload["vocab"]
    id_to_tool = {idx: tool for tool, idx in vocab.items()}
    latent_dim = int(train_payload["latent_dim"])
    decoder = PathDecoder(latent_dim, args.hidden_dim, int(train_payload["max_len"]), len(vocab), dropout=args.dropout).to(device)
    decoder.load_state_dict(train_payload["model"])
    decoder.eval()

    fm = build_fm_model(torch_load(args.fm_checkpoint, map_location=device), device=device)
    eval_samples = collect_step0(load_samples(args.eval_pt))
    eval_text = load_text_metadata(args.eval_text)
    rows = []
    for start in range(0, len(eval_samples), args.predict_batch_size):
        batch = eval_samples[start : start + args.predict_batch_size]
        c_i = torch.stack([sample["c_i"].float().reshape(-1) for sample in batch], dim=0).to(device)
        latent = sample_latent(fm, c_i, args.inference_steps, args.fm_noise_std, args.seed + start)
        pred_ids = decoder(latent).argmax(dim=-1).cpu()
        for sample, ids in zip(batch, pred_ids):
            meta = eval_text.get(sample_key(sample), {})
            tool_ids = decode_ids(ids.tolist(), id_to_tool)
            if args.filter_available:
                tool_ids = apply_available_filter(tool_ids, meta.get("available_tools") or [])
            rows.append(
                {
                    "task_id": sample.get("trajectory_id"),
                    "source": sample.get("source"),
                    "plan_type": "fm_learned_path_decoder_v1",
                    "tool_ids": tool_ids,
                    "tool_names": [tools.get(tool_id, {}).get("name") or str(tool_id).rsplit("::", 1)[-1] for tool_id in tool_ids],
                    "phase_names": phase_names(tool_ids, tools),
                    "actions": compact_actions(tool_ids, tools),
                    "metadata": {
                        "decoder": "learned_position_classifier",
                        "checkpoint": str(Path(args.out_dir) / "path_decoder.pt"),
                        "fm_checkpoint": args.fm_checkpoint,
                        "filter_available": args.filter_available,
                    },
                }
            )
    write_jsonl(Path(args.pred_out), rows)
    print(json.dumps({"predictions": len(rows), "out": args.pred_out}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train and apply a learned FM latent path-to-tool decoder.")
    parser.add_argument("--train-pt", required=True)
    parser.add_argument("--dev-pt", required=True)
    parser.add_argument("--eval-pt", default=None)
    parser.add_argument("--eval-text", default=None)
    parser.add_argument("--tools", default=None)
    parser.add_argument("--fm-checkpoint", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--pred-out", default=None)
    parser.add_argument("--mode", choices=["train", "predict", "train_predict"], default="train_predict")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--predict-batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--max-len", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--fm-noise-std", type=float, default=0.01)
    parser.add_argument("--train-input", choices=["gold", "fm"], default="gold")
    parser.add_argument("--inference-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--step0-only", action="store_true")
    parser.add_argument("--filter-available", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    torch.manual_seed(args.seed)

    if args.mode in {"train", "train_predict"}:
        train_decoder(args, device)
    if args.mode in {"predict", "train_predict"}:
        missing = [name for name in ["eval_pt", "eval_text", "tools", "fm_checkpoint", "pred_out"] if not getattr(args, name)]
        if missing:
            raise ValueError(f"prediction mode missing args: {missing}")
        predict(args, device)


if __name__ == "__main__":
    main()
