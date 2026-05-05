import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_fm_model import model_from_checkpoint, sample_endpoint, torch_load  # noqa: E402


def load_samples(path):
    payload = torch_load(path, map_location="cpu")
    samples = payload.get("samples") if isinstance(payload, dict) else payload
    if not isinstance(samples, list):
        raise ValueError(f"{path} must contain a sample list or a dict with samples")
    return samples


def group_value(sample, key):
    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    return metadata.get(key) or sample.get(key) or "unknown"


def summarize(values):
    if not values:
        return {"count": 0}
    keys = values[0].keys()
    out = {"count": len(values)}
    for key in keys:
        out[key] = sum(row[key] for row in values) / len(values)
    return out


@torch.no_grad()
def evaluate_grouped(model, samples, args, device):
    model.eval()
    grouped = defaultdict(list)
    for start in range(0, len(samples), args.batch_size):
        batch = samples[start : start + args.batch_size]
        c_i = torch.stack([sample["c_i"].float().reshape(-1) for sample in batch]).to(device)
        y_i = torch.stack([sample["y_i"].float().reshape(-1) for sample in batch]).to(device)
        y_hat = sample_endpoint(model, c_i, args, device, seed_offset=start)
        endpoint_mse = F.mse_loss(y_hat, y_i, reduction="none").mean(dim=-1).detach().cpu().tolist()
        endpoint_cos = F.cosine_similarity(y_hat, y_i, dim=-1).detach().cpu().tolist()
        for sample, mse, cos in zip(batch, endpoint_mse, endpoint_cos):
            grouped[group_value(sample, args.group_key)].append({"endpoint_mse": mse, "endpoint_cosine": cos})
    return {group: summarize(rows) for group, rows in sorted(grouped.items())}


def main():
    parser = argparse.ArgumentParser(description="Evaluate an FM checkpoint with metrics grouped by sample metadata.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pt", required=True)
    parser.add_argument("--group-key", default="domain")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--inference-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    checkpoint = torch_load(args.checkpoint, map_location=device)
    model = model_from_checkpoint(checkpoint, device)
    samples = load_samples(args.pt)
    result = {
        "checkpoint": args.checkpoint,
        "pt": args.pt,
        "group_key": args.group_key,
        "groups": evaluate_grouped(model, samples, args, device),
    }
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
