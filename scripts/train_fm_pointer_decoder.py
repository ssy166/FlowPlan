import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


SPECIAL_LABEL_EOS = -1
SPECIAL_LABEL_PAD = -100


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


def stable_hash(text, modulo):
    digest = hashlib.blake2b(str(text).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % modulo


def token_pieces(text):
    text = str(text or "").replace("\n", " ")
    out = []
    for piece in text.split(" "):
        piece = piece.strip().lower()
        if piece:
            out.append(piece)
    return out


def hashed_feature(items, dim):
    vec = torch.zeros(dim, dtype=torch.float32)
    count = 0
    for item in items:
        if item in (None, ""):
            continue
        idx = stable_hash(item, dim)
        sign = 1.0 if stable_hash(f"{item}::sign", 2) == 0 else -1.0
        vec[idx] += sign
        count += 1
    if count:
        vec = vec / max(float(vec.norm(p=2).item()), 1.0)
    return vec


def tool_feature(tool_id, tool, dim):
    schema = tool.get("schema") or {}
    params = schema.get("parameters") or schema.get("required_parameters") or []
    pieces = [
        f"id:{tool_id}",
        f"name:{tool.get('name')}",
        f"domain:{tool.get('domain')}",
        f"phase:{tool.get('phase')}",
    ]
    pieces.extend(f"param:{param}" for param in params)
    pieces.extend(f"desc:{piece}" for piece in token_pieces(tool.get("description"))[:96])
    return hashed_feature(pieces, dim)


def build_tool_features(tools, dim):
    features = {tool_id: tool_feature(tool_id, tool, dim) for tool_id, tool in tools.items()}
    features["<eos>"] = hashed_feature(["<eos>"], dim)
    features["<bos>"] = hashed_feature(["<bos>"], dim)
    features["<unk>"] = hashed_feature(["<unk>"], dim)
    return features


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


def encode_labels(future_tools, available_tools, max_len):
    index = {tool_id: idx for idx, tool_id in enumerate(available_tools)}
    labels = []
    for tool_id in (future_tools or [])[: max_len - 1]:
        labels.append(index.get(tool_id, SPECIAL_LABEL_EOS))
    labels.append(SPECIAL_LABEL_EOS)
    if len(labels) < max_len:
        labels.extend([SPECIAL_LABEL_PAD] * (max_len - len(labels)))
    return labels[:max_len]


class PointerDataset(Dataset):
    def __init__(self, samples, text_by_key, tool_features, args, input_keys):
        self.items = [(sample, input_key) for sample in samples for input_key in input_keys]
        self.text_by_key = text_by_key
        self.tool_features = tool_features
        self.args = args

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        sample, input_key = self.items[idx]
        meta = self.text_by_key.get(sample_key(sample), {})
        available = list(meta.get("available_tools") or [])
        available = available[: self.args.max_available_tools]
        candidate_feats = [self.tool_features.get(tool_id, self.tool_features["<unk>"]) for tool_id in available]
        candidate_mask = [1] * len(candidate_feats)
        while len(candidate_feats) < self.args.max_available_tools:
            candidate_feats.append(torch.zeros(self.args.tool_feature_dim))
            candidate_mask.append(0)
        labels = encode_labels(sample.get("future_tools") or [], available, self.args.max_len)
        prev_feats = [self.tool_features["<bos>"]]
        for label in labels[:-1]:
            if label >= 0 and label < len(available):
                prev_feats.append(self.tool_features.get(available[label], self.tool_features["<unk>"]))
            else:
                prev_feats.append(self.tool_features["<eos>"])
        return {
            "x": sample[input_key].float().reshape(-1),
            "candidate_feats": torch.stack(candidate_feats),
            "candidate_mask": torch.tensor(candidate_mask, dtype=torch.bool),
            "prev_feats": torch.stack(prev_feats[: self.args.max_len]),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class PointerDecoder(nn.Module):
    def __init__(self, latent_dim, tool_feature_dim, hidden_dim, num_layers=1, dropout=0.1):
        super().__init__()
        self.cond = nn.Sequential(nn.LayerNorm(latent_dim), nn.Linear(latent_dim, hidden_dim), nn.GELU())
        self.tool_proj = nn.Sequential(nn.LayerNorm(tool_feature_dim), nn.Linear(tool_feature_dim, hidden_dim), nn.GELU())
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.init_hidden = nn.Linear(hidden_dim, hidden_dim * num_layers)
        self.query = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))
        self.eos = nn.Linear(hidden_dim, 1)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def forward(self, x, candidate_feats, candidate_mask, prev_feats):
        cond = self.cond(x)
        cand = self.tool_proj(candidate_feats)
        prev = self.tool_proj(prev_feats)
        h0 = self.init_hidden(cond).view(x.shape[0], self.num_layers, self.hidden_dim).transpose(0, 1).contiguous()
        states, _ = self.gru(prev, h0)
        q = self.query(states)
        tool_logits = torch.einsum("bth,bah->bta", q, cand) / math.sqrt(self.hidden_dim)
        tool_logits = tool_logits.masked_fill(~candidate_mask.unsqueeze(1), -1e9)
        eos_logits = self.eos(q)
        return torch.cat([tool_logits, eos_logits], dim=-1)

    @torch.no_grad()
    def generate(self, x, candidate_feats, candidate_mask, bos_feat, eos_feat, max_len, min_len=0, no_repeat=True):
        self.eval()
        cond = self.cond(x)
        cand = self.tool_proj(candidate_feats)
        h = self.init_hidden(cond).view(x.shape[0], self.num_layers, self.hidden_dim).transpose(0, 1).contiguous()
        cur = self.tool_proj(bos_feat).unsqueeze(1)
        used = torch.zeros_like(candidate_mask)
        seq = []
        for step_idx in range(max_len):
            out, h = self.gru(cur, h)
            q = self.query(out[:, -1])
            tool_logits = torch.einsum("bh,bah->ba", q, cand) / math.sqrt(self.hidden_dim)
            mask = candidate_mask
            if no_repeat:
                mask = mask & ~used
            tool_logits = tool_logits.masked_fill(~mask, -1e9)
            eos_logits = self.eos(q)
            if step_idx < min_len:
                eos_logits[:] = -1e9
            logits = torch.cat([tool_logits, eos_logits], dim=-1)
            nxt = logits.argmax(dim=-1)
            seq.append(nxt.cpu())
            is_tool = nxt < candidate_mask.shape[1]
            if no_repeat:
                used[torch.arange(x.shape[0], device=x.device)[is_tool], nxt[is_tool]] = True
            next_feats = []
            for row_idx, token in enumerate(nxt.tolist()):
                if token < candidate_feats.shape[1]:
                    next_feats.append(candidate_feats[row_idx, token])
                else:
                    next_feats.append(eos_feat[row_idx])
            cur = self.tool_proj(torch.stack(next_feats).to(x.device)).unsqueeze(1)
        return torch.stack(seq, dim=1)


def pointer_loss(logits, labels):
    target = labels.clone()
    eos_index = logits.shape[-1] - 1
    target[target == SPECIAL_LABEL_EOS] = eos_index
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=SPECIAL_LABEL_PAD)


def eval_pointer(model, loader, device):
    model.eval()
    loss_sum = token_hits = total_tokens = seq_hits = total_seq = 0.0
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            cand = batch["candidate_feats"].to(device)
            mask = batch["candidate_mask"].to(device)
            prev = batch["prev_feats"].to(device)
            labels = batch["labels"].to(device)
            logits = model(x, cand, mask, prev)
            loss = pointer_loss(logits, labels)
            target = labels.clone()
            eos_index = logits.shape[-1] - 1
            target[target == SPECIAL_LABEL_EOS] = eos_index
            pred = logits.argmax(dim=-1)
            valid = target != SPECIAL_LABEL_PAD
            token_hits += ((pred == target) & valid).sum().item()
            total_tokens += valid.sum().item()
            seq_hits += ((pred == target) | ~valid).all(dim=-1).sum().item()
            total_seq += target.shape[0]
            loss_sum += loss.item() * max(valid.sum().item(), 1)
    return {
        "loss": loss_sum / max(total_tokens, 1),
        "token_acc": token_hits / max(total_tokens, 1),
        "sequence_exact": seq_hits / max(total_seq, 1),
    }


def train(args, device):
    tools = {row["tool_id"]: row for row in read_jsonl(args.tools)}
    tool_features = build_tool_features(tools, args.tool_feature_dim)
    train_samples = collect_step0(load_samples(args.train_pt))
    dev_samples = collect_step0(load_samples(args.dev_pt))
    input_keys = ["y_i"]
    if args.train_input in {"fm", "both"}:
        fm = build_fm(torch_load(args.fm_checkpoint, map_location=device), device)
        sample_fm(fm, train_samples, args, device, "fm_y_hat")
        sample_fm(fm, dev_samples, args, device, "fm_y_hat")
        input_keys = ["fm_y_hat"] if args.train_input == "fm" else ["y_i", "fm_y_hat"]
    train_ds = PointerDataset(train_samples, load_text(args.train_text), tool_features, args, input_keys)
    dev_ds = PointerDataset(dev_samples, load_text(args.dev_text), tool_features, args, ["fm_y_hat"] if "fm_y_hat" in input_keys else ["y_i"])
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    dev_dl = DataLoader(dev_ds, batch_size=args.batch_size)
    latent_dim = int(train_samples[0][input_keys[0]].numel())
    model = PointerDecoder(latent_dim, args.tool_feature_dim, args.hidden_dim, args.num_layers, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = None
    best_metric = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_dl:
            x = batch["x"].to(device)
            cand = batch["candidate_feats"].to(device)
            mask = batch["candidate_mask"].to(device)
            prev = batch["prev_feats"].to(device)
            labels = batch["labels"].to(device)
            loss = pointer_loss(model(x, cand, mask, prev), labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            opt.step()
        metrics = eval_pointer(model, dev_dl, device)
        metrics["epoch"] = epoch
        history.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False, sort_keys=True), flush=True)
        if metrics["sequence_exact"] > best_metric:
            best_metric = metrics["sequence_exact"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": best_state or {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "config": vars(args),
            "history": history,
        },
        out_dir / "pointer_decoder.pt",
    )


def predict(args, device):
    tools = {row["tool_id"]: row for row in read_jsonl(args.tools)}
    tool_features = build_tool_features(tools, args.tool_feature_dim)
    payload = torch_load(Path(args.out_dir) / "pointer_decoder.pt", map_location=device)
    config = payload.get("config") or {}
    hidden_dim = int(config.get("hidden_dim", args.hidden_dim))
    num_layers = int(config.get("num_layers", args.num_layers))
    dropout = float(config.get("dropout", args.dropout))
    latent_dim = 768
    model = PointerDecoder(latent_dim, args.tool_feature_dim, hidden_dim, num_layers, dropout).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    fm = build_fm(torch_load(args.fm_checkpoint, map_location=device), device)
    samples = collect_step0(load_samples(args.eval_pt))
    text = load_text(args.eval_text)
    bos_feat = tool_features["<bos>"]
    eos_feat = tool_features["<eos>"]
    rows = []
    for start in range(0, len(samples), args.predict_batch_size):
        batch = samples[start : start + args.predict_batch_size]
        sample_fm(fm, batch, args, device, "fm_y_hat")
        x = torch.stack([s["fm_y_hat"].float().reshape(-1) for s in batch]).to(device)
        avail_lists = [list(text.get(sample_key(s), {}).get("available_tools") or [])[: args.max_available_tools] for s in batch]
        cand_feats = []
        cand_mask = []
        for available in avail_lists:
            feats = [tool_features.get(tool_id, tool_features["<unk>"]) for tool_id in available]
            mask = [1] * len(feats)
            while len(feats) < args.max_available_tools:
                feats.append(torch.zeros(args.tool_feature_dim))
                mask.append(0)
            cand_feats.append(torch.stack(feats))
            cand_mask.append(torch.tensor(mask, dtype=torch.bool))
        cand_feats = torch.stack(cand_feats).to(device)
        cand_mask = torch.stack(cand_mask).to(device)
        bos = bos_feat.repeat(len(batch), 1).to(device)
        eos = eos_feat.repeat(len(batch), 1).to(device)
        pred = model.generate(x, cand_feats, cand_mask, bos, eos, args.max_len, min_len=args.min_decode_len, no_repeat=not args.allow_repeat)
        for sample, ids, available in zip(batch, pred.tolist(), avail_lists):
            tool_ids = []
            eos_idx = args.max_available_tools
            for idx in ids:
                if idx == eos_idx:
                    break
                if idx < len(available):
                    tool_ids.append(available[idx])
            rows.append(
                {
                    "task_id": sample.get("trajectory_id"),
                    "source": sample.get("source"),
                    "plan_type": "fm_schema_pointer_decoder_v1",
                    "tool_ids": tool_ids,
                    "tool_names": [tools.get(t, {}).get("name") or str(t).rsplit("::", 1)[-1] for t in tool_ids],
                    "phase_names": [tools.get(t, {}).get("phase", "unknown") for t in tool_ids],
                    "actions": [
                        {"tool_id": t, "tool_name": tools.get(t, {}).get("name") or str(t).rsplit("::", 1)[-1], "arguments": {}}
                        for t in tool_ids
                    ],
                    "metadata": {"decoder": "schema_pointer_available_tools", "fm_checkpoint": str(args.fm_checkpoint)},
                }
            )
    write_jsonl(args.pred_out, rows)
    print(json.dumps({"predictions": len(rows), "out": args.pred_out}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "predict", "train_predict"], default="train_predict")
    parser.add_argument("--train-pt", required=True)
    parser.add_argument("--dev-pt", required=True)
    parser.add_argument("--eval-pt")
    parser.add_argument("--train-text", required=True)
    parser.add_argument("--dev-text", required=True)
    parser.add_argument("--eval-text")
    parser.add_argument("--tools", required=True)
    parser.add_argument("--fm-checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--pred-out")
    parser.add_argument("--train-input", choices=["gold", "fm", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--predict-batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--max-len", type=int, default=8)
    parser.add_argument("--max-available-tools", type=int, default=128)
    parser.add_argument("--tool-feature-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--fm-noise-std", type=float, default=0.01)
    parser.add_argument("--inference-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--min-decode-len", type=int, default=0)
    parser.add_argument("--allow-repeat", action="store_true")
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
