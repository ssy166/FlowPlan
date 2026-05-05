import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKFLOW_DIR = ROOT / "data" / "processed" / "workflows"
DEFAULT_OUT_DIR = ROOT / "data" / "processed" / "fm_tensor"


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def stable_hash(text, modulo):
    digest = hashlib.blake2b(str(text).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % modulo


def token_pieces(text):
    text = str(text or "").replace("\n", " ")
    return [piece for piece in text.split(" ") if piece]


def hash_token_ids(text, max_len, vocab_size):
    ids = [stable_hash(piece, vocab_size - 1) + 1 for piece in token_pieces(text)[:max_len]]
    mask = [1] * len(ids)
    if len(ids) < max_len:
        pad = max_len - len(ids)
        ids.extend([0] * pad)
        mask.extend([0] * pad)
    return ids, mask


def hash_vector(items, dim):
    vec = torch.zeros(dim, dtype=torch.float32)
    count = 0
    for item in items:
        if item is None:
            continue
        idx = stable_hash(item, dim)
        sign = 1.0 if stable_hash(f"{item}::sign", 2) == 0 else -1.0
        vec[idx] += sign
        count += 1
    if count:
        vec = vec / max(vec.norm(p=2), torch.tensor(1.0))
    return vec


def action_signature(action):
    args = action.get("arguments") or {}
    arg_keys = sorted(args.keys())
    return [
        action.get("tool_id"),
        action.get("tool_name"),
        *[f"arg:{key}" for key in arg_keys],
    ]


def condition_features(record):
    pieces = [
        f"source:{record.get('source')}",
        f"domain:{record.get('domain')}",
        f"step:{record.get('step_idx')}",
        f"total:{record.get('total_steps')}",
    ]
    pieces.extend(token_pieces(record.get("prompt")))
    for tool in record.get("available_tools") or []:
        pieces.append(f"tool:{tool.get('tool_id')}")
        pieces.append(f"phase:{tool.get('phase')}")
        for param in tool.get("parameters") or []:
            pieces.append(f"param:{tool.get('tool_id')}::{param}")
    for action in record.get("prefix_actions") or []:
        pieces.extend(f"prefix:{part}" for part in action_signature(action))
    for feedback in record.get("tool_feedback") or []:
        pieces.extend(token_pieces(json.dumps(feedback, ensure_ascii=False, sort_keys=True)))
    return pieces


def target_features(record):
    target = record.get("target") or {}
    pieces = []
    for idx, action in enumerate(target.get("remaining_actions") or []):
        pieces.append(f"future_idx:{idx}")
        pieces.extend(f"future:{part}" for part in action_signature(action))
    return pieces


def build_tool_vocab(workflow_dir):
    counter = Counter()
    for split in ["train", "dev", "test"]:
        path = workflow_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        for record in read_jsonl(path):
            for tool in record.get("available_tools") or []:
                if tool.get("tool_id"):
                    counter[tool["tool_id"]] += 1
            target = record.get("target") or {}
            for tool_id in target.get("full_tool_ids") or []:
                counter[tool_id] += 1
    vocab = {"<pad>": 0, "<unk>": 1}
    for tool_id, _ in sorted(counter.items()):
        vocab[tool_id] = len(vocab)
    return vocab


def encode_id_list(items, vocab, max_len):
    ids = [vocab.get(item, vocab["<unk>"]) for item in items[:max_len]]
    if len(ids) < max_len:
        ids.extend([vocab["<pad>"]] * (max_len - len(ids)))
    return ids


def encode_split(path, tool_vocab, args):
    records = list(read_jsonl(path))
    tensors = {
        "c_i": [],
        "y_i": [],
        "input_ids": [],
        "attention_mask": [],
        "gold_tool_call_ids": [],
        "next_tool": [],
        "future_tools": [],
        "available_tools": [],
    }
    metadata = []

    for record in records:
        target = record.get("target") or {}
        future_tool_ids = target.get("future_tool_ids") or []
        full_tool_ids = target.get("full_tool_ids") or []
        available_tool_ids = [tool["tool_id"] for tool in record.get("available_tools") or [] if tool.get("tool_id")]
        next_tool_id = (target.get("next_action") or {}).get("tool_id")

        input_ids, attention_mask = hash_token_ids(record.get("condition_text") or "", args.max_input_len, args.token_vocab_size)
        tensors["c_i"].append(hash_vector(condition_features(record), args.latent_dim))
        tensors["y_i"].append(hash_vector(target_features(record), args.latent_dim))
        tensors["input_ids"].append(torch.tensor(input_ids, dtype=torch.long))
        tensors["attention_mask"].append(torch.tensor(attention_mask, dtype=torch.long))
        tensors["gold_tool_call_ids"].append(torch.tensor(encode_id_list(full_tool_ids, tool_vocab, args.max_actions), dtype=torch.long))
        tensors["next_tool"].append(torch.tensor(tool_vocab.get(next_tool_id, tool_vocab["<unk>"]), dtype=torch.long))
        tensors["future_tools"].append(torch.tensor(encode_id_list(future_tool_ids, tool_vocab, args.max_actions), dtype=torch.long))
        tensors["available_tools"].append(torch.tensor(encode_id_list(available_tool_ids, tool_vocab, args.max_available_tools), dtype=torch.long))
        metadata.append(
            {
                "record_id": record.get("record_id"),
                "workflow_id": record.get("workflow_id"),
                "task_id": record.get("task_id"),
                "source": record.get("source"),
                "domain": record.get("domain"),
                "split": record.get("split"),
                "step_idx": record.get("step_idx"),
                "total_steps": record.get("total_steps"),
                "next_tool_id": next_tool_id,
                "future_tool_ids": future_tool_ids,
            }
        )

    output = {key: torch.stack(value) if value else torch.empty(0) for key, value in tensors.items()}
    output["metadata"] = metadata
    output["schema_version"] = "fm_tensor_hash_v1"
    output["tool_vocab"] = tool_vocab
    return output


def validate_split(data):
    size = len(data["metadata"])
    required = ["c_i", "y_i", "input_ids", "attention_mask", "gold_tool_call_ids", "next_tool", "future_tools", "available_tools"]
    errors = []
    for key in required:
        if key not in data:
            errors.append(f"missing key: {key}")
            continue
        if data[key].shape[0] != size:
            errors.append(f"{key} first dimension {data[key].shape[0]} != metadata {size}")
    if "c_i" in data and "y_i" in data and data["c_i"].shape != data["y_i"].shape:
        errors.append(f"c_i shape {tuple(data['c_i'].shape)} != y_i shape {tuple(data['y_i'].shape)}")
    return errors


def write_readme(out_dir, manifest):
    lines = [
        "# FM Tensor Dataset",
        "",
        "ToolRL-style flow matching tensor samples derived from `data/processed/workflows/*.jsonl`.",
        "",
        "This is a first runnable tensorization pass. `c_i` and `y_i` use deterministic hashed continuous features so the training and validation pipeline can be exercised before replacing them with LLM encoder latents.",
        "",
        "## Files",
        "",
        "- `train.pt`",
        "- `dev.pt`",
        "- `test.pt`",
        "- `tool_vocab.json`",
        "- `manifest.json`",
        "",
        "## Counts",
        "",
    ]
    for split in ["train", "dev", "test"]:
        info = manifest["splits"][split]
        lines.append(f"- {split}: {info['records']} samples, c_i/y_i shape {info['latent_shape']}")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Build FM tensor samples from workflow records.")
    parser.add_argument("--workflow-dir", default=str(DEFAULT_WORKFLOW_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--latent-dim", type=int, default=768)
    parser.add_argument("--max-input-len", type=int, default=2048)
    parser.add_argument("--token-vocab-size", type=int, default=32000)
    parser.add_argument("--max-actions", type=int, default=32)
    parser.add_argument("--max-available-tools", type=int, default=128)
    args = parser.parse_args()

    workflow_dir = Path(args.workflow_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tool_vocab = build_tool_vocab(workflow_dir)
    (out_dir / "tool_vocab.json").write_text(json.dumps(tool_vocab, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest = {
        "format": "fm_tensor_hash_v1",
        "source_dir": str(workflow_dir.relative_to(ROOT) if workflow_dir.is_relative_to(ROOT) else workflow_dir),
        "latent_dim": args.latent_dim,
        "max_input_len": args.max_input_len,
        "token_vocab_size": args.token_vocab_size,
        "max_actions": args.max_actions,
        "max_available_tools": args.max_available_tools,
        "tool_vocab_size": len(tool_vocab),
        "splits": {},
    }

    for split in ["train", "dev", "test"]:
        data = encode_split(workflow_dir / f"{split}.jsonl", tool_vocab, args)
        errors = validate_split(data)
        if errors:
            raise ValueError(f"{split} validation failed: {'; '.join(errors)}")
        out_path = out_dir / f"{split}.pt"
        torch.save(data, out_path)
        manifest["splits"][split] = {
            "path": str(out_path.relative_to(ROOT)),
            "records": len(data["metadata"]),
            "latent_shape": list(data["c_i"].shape),
            "input_shape": list(data["input_ids"].shape),
            "gold_tool_call_shape": list(data["gold_tool_call_ids"].shape),
            "available_tools_shape": list(data["available_tools"].shape),
        }
        print(f"{split}: wrote {out_path} with {len(data['metadata'])} samples")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_readme(out_dir, manifest)
    print(f"manifest: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
