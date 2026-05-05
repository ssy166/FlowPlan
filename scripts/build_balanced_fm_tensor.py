import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch


def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def sample_group(sample, key):
    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    return metadata.get(key) or sample.get(key) or "unknown"


def main():
    parser = argparse.ArgumentParser(description="Build an upsampled/balanced FM tensor sample list.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--group-key", default="domain")
    parser.add_argument("--target-count", type=int, default=0, help="Per-group target count; default is max group size.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    payload = torch_load(args.input)
    samples = payload.get("samples") if isinstance(payload, dict) else payload
    if not isinstance(samples, list):
        raise ValueError(f"{args.input} must be a sample list or a dict with samples")

    buckets = defaultdict(list)
    for sample in samples:
        buckets[sample_group(sample, args.group_key)].append(sample)
    target = args.target_count or max(len(items) for items in buckets.values())
    rng = random.Random(args.seed)
    balanced = []
    for group in sorted(buckets):
        items = buckets[group]
        if len(items) >= target:
            chosen = rng.sample(items, target)
        else:
            chosen = items + [rng.choice(items) for _ in range(target - len(items))]
            rng.shuffle(chosen)
        balanced.extend(chosen)
    rng.shuffle(balanced)

    out_payload = dict(payload) if isinstance(payload, dict) else {"samples": samples}
    out_payload["samples"] = balanced
    metadata = dict(out_payload.get("metadata") or {})
    metadata["balanced_from"] = str(args.input)
    metadata["balance_group_key"] = args.group_key
    metadata["balance_seed"] = args.seed
    metadata["balance_target_count"] = target
    metadata["original_counts"] = dict(Counter(sample_group(sample, args.group_key) for sample in samples))
    metadata["balanced_counts"] = dict(Counter(sample_group(sample, args.group_key) for sample in balanced))
    out_payload["metadata"] = metadata

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_payload, output)
    print(
        json.dumps(
            {
                "input": args.input,
                "output": str(output),
                "group_key": args.group_key,
                "target_count": target,
                "original_counts": metadata["original_counts"],
                "balanced_counts": metadata["balanced_counts"],
                "samples": len(balanced),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
