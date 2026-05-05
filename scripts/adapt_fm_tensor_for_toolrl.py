import argparse
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN_DIR = ROOT / "data" / "processed" / "fm_tensor"
DEFAULT_OUT_DIR = ROOT / "data" / "processed" / "fm_tensor_toolrl"


def invert_vocab(vocab):
    return {int(idx): tool_id for tool_id, idx in vocab.items()}


def non_pad_names(ids, id_to_tool, pad_id=0):
    names = []
    for item in ids.tolist():
        item = int(item)
        if item == pad_id:
            continue
        names.append(id_to_tool.get(item, "<unk>"))
    return names


def adapt_split(input_path, output_path, limit=None):
    payload = torch.load(input_path, map_location="cpu", weights_only=False)
    tool_vocab = payload["tool_vocab"]
    id_to_tool = invert_vocab(tool_vocab)
    pad_id = int(tool_vocab.get("<pad>", 0))
    size = len(payload["metadata"])
    if limit is not None:
        size = min(size, int(limit))

    samples = []
    for idx in range(size):
        meta = dict(payload["metadata"][idx])
        next_tool_id = int(payload["next_tool"][idx].item())
        next_tool = id_to_tool.get(next_tool_id, meta.get("next_tool_id") or "<unk>")
        sample = {
            "c_i": payload["c_i"][idx].clone(),
            "y_i": payload["y_i"][idx].clone(),
            "input_ids": payload["input_ids"][idx].clone(),
            "attention_mask": payload["attention_mask"][idx].clone(),
            "gold_tool_call_ids": payload["gold_tool_call_ids"][idx].clone(),
            "next_tool": next_tool,
            "future_tools": non_pad_names(payload["future_tools"][idx], id_to_tool, pad_id=pad_id),
            "available_tools": non_pad_names(payload["available_tools"][idx], id_to_tool, pad_id=pad_id),
            "trajectory_id": meta.get("workflow_id") or meta.get("task_id"),
            "step_idx": meta.get("step_idx"),
            "total_steps": meta.get("total_steps"),
            "source": meta.get("source"),
            "metadata": meta,
            "should_call": True,
        }
        samples.append(sample)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "samples": samples,
            "metadata": {
                "source_pt": str(input_path),
                "schema_version": payload.get("schema_version"),
                "adapter": "toolrl_samples_v1",
                "num_samples": len(samples),
                "tool_vocab_size": len(tool_vocab),
            },
        },
        output_path,
    )
    return len(samples)


def main():
    parser = argparse.ArgumentParser(description="Adapt batched FM tensors to ToolRL fm_sft_trainer sample-list PT files.")
    parser.add_argument("--in-dir", default=str(DEFAULT_IN_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--limit-train", type=int, default=512)
    parser.add_argument("--limit-dev", type=int, default=128)
    parser.add_argument("--full", action="store_true", help="Export full train/dev/test splits instead of smoke-test subsets.")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    manifest = {"format": "toolrl_samples_v1", "splits": {}}
    limits = {
        "train": None if args.full else args.limit_train,
        "dev": None if args.full else args.limit_dev,
        "test": None if args.full else args.limit_dev,
    }
    for split in ["train", "dev", "test"]:
        suffix = "" if args.full else "_smoke"
        out_path = out_dir / f"{split}{suffix}.pt"
        count = adapt_split(in_dir / f"{split}.pt", out_path, limit=limits[split])
        manifest["splits"][split] = {"path": str(out_path.relative_to(ROOT)), "samples": count}
        print(f"{split}: wrote {out_path} with {count} samples")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
