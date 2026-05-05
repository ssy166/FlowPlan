import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


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


def get_config_value(config: dict[str, Any], dotted: str, default):
    cur: Any = config
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def build_model(checkpoint, device):
    config = checkpoint.get("config") or {}
    state = checkpoint["model"]
    first_key = next(iter(state))
    target_dim = int(state["x_norm.weight"].numel()) if "x_norm.weight" in state else 768
    cond_dim = int(state["cond_norm.weight"].numel()) if "cond_norm.weight" in state else target_dim
    hidden_dims = get_config_value(config, "model.hidden_dims", [1024, 1024])
    time_embed_dim = int(get_config_value(config, "model.time_embed_dim", 128))
    dropout = float(get_config_value(config, "model.dropout", 0.0))
    model = FMVelocityModel(
        x_dim=target_dim,
        cond_dim=cond_dim,
        time_embed_dim=time_embed_dim,
        target_dim=target_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, config


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


def collect_step0(samples):
    return [sample for sample in samples if int(sample.get("step_idx") or 0) == 0]


def build_train_index(samples, normalize=True):
    vectors = torch.stack([sample["y_i"].float().reshape(-1) for sample in samples], dim=0)
    if normalize:
        vectors = F.normalize(vectors, dim=-1)
    return vectors


def build_candidate_mask(eval_meta, train_samples, train_meta, require_same_domain, require_available_subset):
    allowed = []
    eval_available = set(eval_meta.get("available_tools") or [])
    eval_source = eval_meta.get("source")
    eval_domain = eval_meta.get("domain")
    for idx, sample in enumerate(train_samples):
        meta = train_meta.get(sample_key(sample), {})
        if require_same_domain:
            if eval_source and meta.get("source") != eval_source:
                continue
            if eval_domain and meta.get("domain") != eval_domain:
                continue
        if require_available_subset and eval_available:
            future_tools = set(sample.get("future_tools") or [])
            if not future_tools or not future_tools.issubset(eval_available):
                continue
        allowed.append(idx)
    return allowed


def compact_actions(tool_ids, tools):
    actions = []
    for tool_id in tool_ids:
        actions.append(
            {
                "tool_id": tool_id,
                "tool_name": tools.get(tool_id, {}).get("name") or str(tool_id).rsplit("::", 1)[-1],
                "arguments": {},
            }
        )
    return actions


def phase_names(tool_ids, tools):
    return [tools.get(tool_id, {}).get("phase", "unknown") for tool_id in tool_ids]


def decode_split(split_samples, train_samples, train_index, tools, model, args, device, train_text_meta, eval_text_meta):
    rows = []
    details = []
    source_samples = collect_step0(split_samples)
    for start in range(0, len(source_samples), args.batch_size):
        batch_samples = source_samples[start : start + args.batch_size]
        c_i = torch.stack([sample["c_i"].float().reshape(-1) for sample in batch_samples], dim=0).to(device)
        y_hat = sample_latent(
            model=model,
            c_i=c_i,
            inference_steps=args.inference_steps,
            noise_std=args.noise_std,
            seed=args.seed + start,
        ).cpu()
        y_query = F.normalize(y_hat, dim=-1)
        scores = y_query @ train_index.T
        best_scores = []
        best_indices = []
        for row_idx, sample in enumerate(batch_samples):
            eval_meta = eval_text_meta.get(sample_key(sample), {})
            allowed = build_candidate_mask(
                eval_meta=eval_meta,
                train_samples=train_samples,
                train_meta=train_text_meta,
                require_same_domain=args.require_same_domain,
                require_available_subset=args.require_available_subset,
            )
            if allowed:
                allowed_tensor = torch.tensor(allowed, dtype=torch.long)
                allowed_scores = scores[row_idx, allowed_tensor]
                best_allowed_pos = int(allowed_scores.argmax().item())
                best_indices.append(int(allowed_tensor[best_allowed_pos].item()))
                best_scores.append(float(allowed_scores[best_allowed_pos].item()))
            else:
                score, nn_idx = scores[row_idx].max(dim=-1)
                best_indices.append(int(nn_idx.item()))
                best_scores.append(float(score.item()))
        for sample, score, nn_idx in zip(batch_samples, best_scores, best_indices):
            nearest = train_samples[int(nn_idx)]
            tool_ids = list(nearest.get("future_tools") or [])
            task_id = sample.get("trajectory_id")
            eval_meta = eval_text_meta.get(sample_key(sample), {})
            nearest_meta = train_text_meta.get(sample_key(nearest), {})
            row = {
                "task_id": task_id,
                "source": sample.get("source"),
                "plan_type": "fm_nearest_path_encoder_v1",
                "tool_ids": tool_ids,
                "tool_names": [tools.get(tool_id, {}).get("name") or str(tool_id).rsplit("::", 1)[-1] for tool_id in tool_ids],
                "phase_names": phase_names(tool_ids, tools),
                "actions": compact_actions(tool_ids, tools),
                "metadata": {
                    "decoder": "nearest_train_y_i",
                    "nearest_trajectory_id": nearest.get("trajectory_id"),
                    "nearest_step_idx": nearest.get("step_idx"),
                    "nearest_domain": nearest_meta.get("domain"),
                    "eval_domain": eval_meta.get("domain"),
                    "nearest_score": score,
                    "checkpoint": str(args.checkpoint),
                    "require_same_domain": args.require_same_domain,
                    "require_available_subset": args.require_available_subset,
                },
            }
            rows.append(row)
            details.append(
                {
                    "task_id": task_id,
                    "nearest_score": score,
                    "nearest_trajectory_id": nearest.get("trajectory_id"),
                    "pred_tool_ids": tool_ids,
                    "gold_future_tools": sample.get("future_tools"),
                }
            )
    return rows, details


def main():
    parser = argparse.ArgumentParser(description="Decode FM checkpoint outputs using nearest training workflow-path latent.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-pt", required=True)
    parser.add_argument("--eval-pt", required=True)
    parser.add_argument("--tools", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--details", default=None)
    parser.add_argument("--train-text", default=None)
    parser.add_argument("--eval-text", default=None)
    parser.add_argument("--require-same-domain", action="store_true")
    parser.add_argument("--require-available-subset", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--inference-steps", type=int, default=32)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    tools = {row["tool_id"]: row for row in read_jsonl(args.tools)}
    checkpoint = torch_load(args.checkpoint, map_location=device)
    model, _ = build_model(checkpoint, device=device)
    train_samples = load_samples(args.train_pt)
    eval_samples = load_samples(args.eval_pt)
    train_text_meta = load_text_metadata(args.train_text)
    eval_text_meta = load_text_metadata(args.eval_text)
    train_index = build_train_index(train_samples).to("cpu")
    rows, details = decode_split(
        eval_samples,
        train_samples,
        train_index,
        tools,
        model,
        args,
        device,
        train_text_meta=train_text_meta,
        eval_text_meta=eval_text_meta,
    )
    write_jsonl(Path(args.out), rows)
    if args.details:
        write_jsonl(Path(args.details), details)
    print(json.dumps({"predictions": len(rows), "out": args.out, "details": args.details}, ensure_ascii=False))


if __name__ == "__main__":
    main()
