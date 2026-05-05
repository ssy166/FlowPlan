import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


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


class FMDataset(Dataset):
    def __init__(self, samples):
        self.samples = list(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "c_i": sample["c_i"].float().reshape(-1),
            "y_i": sample["y_i"].float().reshape(-1),
        }


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
        dims = [int(x_dim) + int(cond_dim) + int(time_embed_dim)] + [int(dim) for dim in hidden_dims]
        layers = []
        for dim_in, dim_out in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(dim_in, dim_out))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.velocity_head = nn.Linear(dims[-1], int(target_dim))

    def forward(self, x_t, t, c_i):
        return self.velocity_head(
            self.backbone(torch.cat([self.x_norm(x_t), self.cond_norm(c_i), self.time_embed(t)], dim=-1))
        )


def make_model(args, sample):
    cond_dim = int(sample["c_i"].numel())
    target_dim = int(sample["y_i"].numel())
    return FMVelocityModel(
        x_dim=target_dim,
        cond_dim=cond_dim,
        time_embed_dim=args.time_embed_dim,
        target_dim=target_dim,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
    )


def get_config_value(config: dict[str, Any], dotted: str, default):
    cur: Any = config
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def model_from_checkpoint(checkpoint, device):
    state = checkpoint["model"]
    config = checkpoint.get("config") or {}
    target_dim = int(state["x_norm.weight"].numel())
    cond_dim = int(state["cond_norm.weight"].numel())
    model = FMVelocityModel(
        x_dim=target_dim,
        cond_dim=cond_dim,
        time_embed_dim=int(get_config_value(config, "model.time_embed_dim", 128)),
        target_dim=target_dim,
        hidden_dims=get_config_value(config, "model.hidden_dims", [1024, 1024]),
        dropout=float(get_config_value(config, "model.dropout", 0.0)),
    ).to(device)
    model.load_state_dict(state)
    return model


def flow_batch(model, batch, args, device, generator=None):
    c_i = batch["c_i"].to(device)
    y_i = batch["y_i"].to(device)
    if generator is None:
        x0 = torch.randn_like(y_i) * args.noise_std
        t = torch.rand(y_i.shape[0], 1, device=device)
    else:
        x0 = torch.randn(y_i.shape, generator=generator, device=device, dtype=y_i.dtype) * args.noise_std
        t = torch.rand(y_i.shape[0], 1, generator=generator, device=device, dtype=y_i.dtype)
    x_t = (1.0 - t) * x0 + t * y_i
    target_v = y_i - x0
    pred_v = model(x_t, t, c_i)
    return pred_v, target_v


@torch.no_grad()
def sample_endpoint(model, c_i, args, device, seed_offset=0):
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed + seed_offset))
    x_t = torch.randn(
        c_i.shape[0],
        model.x_norm.normalized_shape[0],
        generator=generator,
        device=device,
        dtype=c_i.dtype,
    ) * args.noise_std
    dt = 1.0 / max(1, args.inference_steps)
    for step in range(max(1, args.inference_steps)):
        t = torch.full((c_i.shape[0], 1), step * dt, device=device, dtype=c_i.dtype)
        x_t = x_t + dt * model(x_t, t, c_i)
    return x_t


def sample_endpoint_for_loss(model, c_i, args, device):
    steps = max(1, int(args.endpoint_loss_steps or args.inference_steps))
    x_t = torch.randn(
        c_i.shape[0],
        model.x_norm.normalized_shape[0],
        device=device,
        dtype=c_i.dtype,
    ) * args.noise_std
    dt = 1.0 / steps
    for step in range(steps):
        t = torch.full((c_i.shape[0], 1), step * dt, device=device, dtype=c_i.dtype)
        x_t = x_t + dt * model(x_t, t, c_i)
    return x_t


@torch.no_grad()
def evaluate(model, loader, args, device, split):
    model.eval()
    loss_sum = 0.0
    flow_cos_sum = 0.0
    mse_sum = 0.0
    cos_sum = 0.0
    count = 0
    for batch_idx, batch in enumerate(loader):
        c_i = batch["c_i"].to(device)
        y_i = batch["y_i"].to(device)
        pred_v, target_v = flow_batch(model, batch, args, device)
        loss = F.mse_loss(pred_v, target_v, reduction="none").mean(dim=-1)
        flow_cos = F.cosine_similarity(pred_v, target_v, dim=-1)
        y_hat = sample_endpoint(model, c_i, args, device, seed_offset=batch_idx * args.eval_batch_size)
        endpoint_mse = F.mse_loss(y_hat, y_i, reduction="none").mean(dim=-1)
        endpoint_cos = F.cosine_similarity(y_hat, y_i, dim=-1)
        batch_n = y_i.shape[0]
        loss_sum += float(loss.sum().item())
        flow_cos_sum += float(flow_cos.sum().item())
        mse_sum += float(endpoint_mse.sum().item())
        cos_sum += float(endpoint_cos.sum().item())
        count += batch_n
    return {
        f"{split}_flow_loss": loss_sum / max(count, 1),
        f"{split}_flow_cosine": flow_cos_sum / max(count, 1),
        f"{split}_endpoint_mse": mse_sum / max(count, 1),
        f"{split}_endpoint_cosine": cos_sum / max(count, 1),
    }


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def config_dict(args):
    return {
        "data": {
            "train_files": args.train_pt,
            "val_files": args.dev_pt,
            "test_files": args.test_pt,
            "batch_size": args.batch_size,
            "val_batch_size": args.eval_batch_size,
        },
        "model": {
            "time_embed_dim": args.time_embed_dim,
            "hidden_dims": args.hidden_dims,
            "dropout": args.dropout,
            "noise_std": args.noise_std,
            "inference_steps": args.inference_steps,
        },
        "optim": {
            "lr": args.lr,
            "betas": [args.beta1, args.beta2],
            "weight_decay": args.weight_decay,
            "warmup_steps_ratio": args.warmup_ratio,
            "clip_grad": args.clip_grad,
            "velocity_cos_weight": args.velocity_cos_weight,
            "endpoint_loss_weight": args.endpoint_loss_weight,
            "endpoint_cos_weight": args.endpoint_cos_weight,
            "endpoint_loss_steps": args.endpoint_loss_steps,
        },
        "trainer": {
            "default_local_dir": args.out_dir,
            "total_epochs": args.epochs,
            "seed": args.seed,
            "device": args.device,
            "best_by": args.best_by,
        },
    }


def save_checkpoint(path, model, args, metrics, epoch):
    payload = {
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "config": config_dict(args),
        "metrics": metrics,
        "epoch": epoch,
    }
    torch.save(payload, path)


def metric_is_better(metrics, best_value, key):
    value = metrics[key]
    if "loss" in key or "mse" in key:
        return best_value is None or value < best_value
    return best_value is None or value > best_value


def train(args, device):
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_samples = load_samples(args.train_pt)
    dev_samples = load_samples(args.dev_pt)
    test_samples = load_samples(args.test_pt) if args.test_pt else []

    train_loader = DataLoader(FMDataset(train_samples), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    dev_loader = DataLoader(FMDataset(dev_samples), batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(FMDataset(test_samples), batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers) if test_samples else None

    model = make_model(args, train_samples[0]).to(device)
    if args.init_checkpoint:
        checkpoint = torch_load(args.init_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    total_steps = max(1, args.epochs * len(train_loader))
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step):
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    history = []
    best_value = None
    global_step = 0

    (out_dir / "run_config.json").write_text(json.dumps(config_dict(args), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "start", "train": len(train_samples), "dev": len(dev_samples), "test": len(test_samples), "device": str(device)}), flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch in train_loader:
            pred_v, target_v = flow_batch(model, batch, args, device)
            mse_loss = F.mse_loss(pred_v, target_v)
            loss = mse_loss
            if args.velocity_cos_weight:
                cos_loss = 1.0 - F.cosine_similarity(pred_v, target_v, dim=-1).mean()
                loss = loss + args.velocity_cos_weight * cos_loss
            if args.endpoint_loss_weight or args.endpoint_cos_weight:
                c_i = batch["c_i"].to(device)
                y_i = batch["y_i"].to(device)
                y_hat = sample_endpoint_for_loss(model, c_i, args, device)
                if args.endpoint_loss_weight:
                    loss = loss + args.endpoint_loss_weight * F.mse_loss(y_hat, y_i)
                if args.endpoint_cos_weight:
                    endpoint_cos_loss = 1.0 - F.cosine_similarity(y_hat, y_i, dim=-1).mean()
                    loss = loss + args.endpoint_cos_weight * endpoint_cos_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            scheduler.step()
            batch_n = batch["y_i"].shape[0]
            train_loss_sum += float(loss.item()) * batch_n
            train_count += batch_n
            global_step += 1
        metrics = {
            "epoch": epoch,
            "global_step": global_step,
            "lr": scheduler.get_last_lr()[0],
            "train_flow_loss": train_loss_sum / max(train_count, 1),
        }
        metrics.update(evaluate(model, dev_loader, args, device, "dev"))
        if test_loader is not None and (epoch == args.epochs or epoch % args.eval_test_every == 0):
            metrics.update(evaluate(model, test_loader, args, device, "test"))
        history.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False, sort_keys=True), flush=True)

        save_checkpoint(out_dir / "last.pt", model, args, metrics, epoch)
        if args.save_every_n_epochs and epoch % args.save_every_n_epochs == 0:
            save_checkpoint(out_dir / f"epoch_{epoch}.pt", model, args, metrics, epoch)
        best_key = f"dev_{args.best_by}"
        if metric_is_better(metrics, best_value, best_key):
            best_value = metrics[best_key]
            save_checkpoint(out_dir / "best.pt", model, args, metrics, epoch)

    (out_dir / "metrics.json").write_text(json.dumps({"history": history, "best_by": args.best_by}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def evaluate_checkpoint(args, device):
    if not args.checkpoint:
        raise ValueError("--mode eval requires --checkpoint")
    model = model_from_checkpoint(torch_load(args.checkpoint, map_location=device), device)
    model.eval()
    metrics = {"checkpoint": args.checkpoint}
    if args.train_pt:
        train_samples = load_samples(args.train_pt)
        train_loader = DataLoader(FMDataset(train_samples), batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
        metrics.update(evaluate(model, train_loader, args, device, "train"))
    if args.dev_pt:
        dev_samples = load_samples(args.dev_pt)
        dev_loader = DataLoader(FMDataset(dev_samples), batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
        metrics.update(evaluate(model, dev_loader, args, device, "dev"))
    if args.test_pt:
        test_samples = load_samples(args.test_pt)
        test_loader = DataLoader(FMDataset(test_samples), batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
        metrics.update(evaluate(model, test_loader, args, device, "test"))
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True), flush=True)
    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "eval_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Train an encoder-derived flow-matching workflow planner.")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--train-pt")
    parser.add_argument("--dev-pt")
    parser.add_argument("--test-pt")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[1024, 1024])
    parser.add_argument("--time-embed-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--inference-steps", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--velocity-cos-weight", type=float, default=0.0)
    parser.add_argument("--endpoint-loss-weight", type=float, default=0.0)
    parser.add_argument("--endpoint-cos-weight", type=float, default=0.0)
    parser.add_argument("--endpoint-loss-steps", type=int, default=0, help="Differentiable rollout steps for endpoint loss; default uses --inference-steps.")
    parser.add_argument("--best-by", choices=["flow_loss", "endpoint_mse", "endpoint_cosine"], default="endpoint_cosine")
    parser.add_argument("--eval-test-every", type=int, default=5)
    parser.add_argument("--save-every-n-epochs", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.mode == "train" and (not args.train_pt or not args.dev_pt):
        raise ValueError("--mode train requires --train-pt and --dev-pt")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if args.mode == "train":
        train(args, device)
    else:
        evaluate_checkpoint(args, device)


if __name__ == "__main__":
    main()
