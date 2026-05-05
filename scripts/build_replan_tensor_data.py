import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPLAN_DIR = ROOT / "data" / "processed" / "closed_loop"
DEFAULT_TEXT_DIR = ROOT / "data" / "processed" / "replan_text"
DEFAULT_TENSOR_DIR = ROOT / "data" / "processed" / "replan_tensor"
STOP_TOOL = "<stop>"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def stable_hash(text: Any, modulo: int) -> int:
    digest = hashlib.blake2b(str(text).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % modulo


def token_pieces(text: Any) -> list[str]:
    return [piece for piece in str(text or "").replace("\n", " ").split(" ") if piece]


def hash_token_ids(text: str, max_len: int, vocab_size: int) -> tuple[list[int], list[int]]:
    ids = [stable_hash(piece, vocab_size - 1) + 1 for piece in token_pieces(text)[:max_len]]
    mask = [1] * len(ids)
    if len(ids) < max_len:
        pad = max_len - len(ids)
        ids.extend([0] * pad)
        mask.extend([0] * pad)
    return ids, mask


def hash_vector(items: list[Any], dim: int) -> torch.Tensor:
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


def compact_action(action: dict[str, Any] | None) -> dict[str, Any] | None:
    if not action:
        return None
    return {
        "tool_id": action.get("tool_id"),
        "tool_name": action.get("tool_name"),
        "arguments": action.get("arguments") or {},
    }


def load_tasks(path: str | Path) -> dict[str, dict[str, Any]]:
    return {row["task_id"]: row for row in read_jsonl(path)}


def build_input_text(record: dict[str, Any]) -> str:
    reason = record.get("replan_reason") or {}
    observable_reason = {
        "execution_ok": reason.get("execution_ok"),
        "error_type": reason.get("error_type"),
        "has_predicted_action_at_step": reason.get("has_predicted_action_at_step"),
        "tool_match": reason.get("tool_match"),
    }
    payload = {
        "task": {
            "task_id": record.get("task_id"),
            "domain": record.get("domain"),
            "prompt": record.get("prompt"),
        },
        "initialization_feedback": record.get("initialization_feedback") or [],
        "executed_prefix": record.get("executed_prefix") or [],
        "replan_reason": observable_reason,
        "instruction": "Predict the remaining tool workflow from this state. Return an empty action list if the workflow should stop.",
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def build_gold_text(record: dict[str, Any]) -> str:
    actions = [compact_action(action) for action in record.get("target_remaining_actions") or []]
    return json.dumps({"actions": actions}, ensure_ascii=False, sort_keys=True, default=str)


def normalize_record(record: dict[str, Any], tasks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    task = tasks.get(record.get("task_id"), {})
    target_tool_ids = record.get("target_remaining_tool_ids") or []
    next_tool = target_tool_ids[0] if target_tool_ids else STOP_TOOL
    return {
        "record_id": f"{record.get('task_id')}::replan::{record.get('replan_step_idx')}",
        "task_id": record.get("task_id"),
        "source": record.get("source"),
        "domain": record.get("domain"),
        "split": record.get("split"),
        "replan_step_idx": record.get("replan_step_idx"),
        "input_text": build_input_text(record),
        "gold_tool_call_text": build_gold_text(record),
        "future_tools": target_tool_ids,
        "next_tool": next_tool,
        "available_tools": task.get("available_tool_ids") or [],
        "stop_target": not bool(target_tool_ids),
        "metadata": {
            "format": "replan_text_v1",
            "prediction_type": record.get("prediction_type"),
            "replan_reason": record.get("replan_reason") or {},
            "executed_prefix_len": len(record.get("executed_prefix") or []),
            "target_remaining_len": len(target_tool_ids),
        },
    }


def build_tool_vocab(text_rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    counter = Counter()
    counter[STOP_TOOL] += 1
    for rows in text_rows_by_split.values():
        for row in rows:
            for tool_id in row.get("available_tools") or []:
                counter[tool_id] += 1
            for tool_id in row.get("future_tools") or []:
                counter[tool_id] += 1
            counter[row.get("next_tool") or STOP_TOOL] += 1
    vocab = {"<pad>": 0, "<unk>": 1}
    for tool_id, _ in sorted(counter.items()):
        vocab[tool_id] = len(vocab)
    return vocab


def encode_id_list(items: list[str], vocab: dict[str, int], max_len: int) -> list[int]:
    ids = [vocab.get(item, vocab["<unk>"]) for item in items[:max_len]]
    if len(ids) < max_len:
        ids.extend([vocab["<pad>"]] * (max_len - len(ids)))
    return ids


def condition_features(row: dict[str, Any]) -> list[Any]:
    pieces = [
        f"source:{row.get('source')}",
        f"domain:{row.get('domain')}",
        f"replan_step:{row.get('replan_step_idx')}",
    ]
    pieces.extend(token_pieces(row.get("input_text")))
    for tool_id in row.get("available_tools") or []:
        pieces.append(f"available:{tool_id}")
    return pieces


def target_features(row: dict[str, Any]) -> list[Any]:
    future_tools = row.get("future_tools") or []
    if not future_tools:
        return [STOP_TOOL]
    pieces = []
    for idx, tool_id in enumerate(future_tools):
        pieces.extend([f"future_idx:{idx}", f"future:{tool_id}"])
    pieces.extend(token_pieces(row.get("gold_tool_call_text")))
    return pieces


def encode_split(rows: list[dict[str, Any]], tool_vocab: dict[str, int], args: argparse.Namespace) -> dict[str, Any]:
    tensors = {
        "c_i": [],
        "y_i": [],
        "input_ids": [],
        "attention_mask": [],
        "gold_tool_call_ids": [],
        "next_tool": [],
        "future_tools": [],
        "available_tools": [],
        "stop_target": [],
    }
    metadata = []
    for row in rows:
        future_tools = row.get("future_tools") or []
        target_ids = future_tools if future_tools else [STOP_TOOL]
        input_ids, attention_mask = hash_token_ids(row.get("input_text") or "", args.max_input_len, args.token_vocab_size)
        tensors["c_i"].append(hash_vector(condition_features(row), args.latent_dim))
        tensors["y_i"].append(hash_vector(target_features(row), args.latent_dim))
        tensors["input_ids"].append(torch.tensor(input_ids, dtype=torch.long))
        tensors["attention_mask"].append(torch.tensor(attention_mask, dtype=torch.long))
        tensors["gold_tool_call_ids"].append(torch.tensor(encode_id_list(target_ids, tool_vocab, args.max_actions), dtype=torch.long))
        tensors["next_tool"].append(torch.tensor(tool_vocab.get(row.get("next_tool"), tool_vocab["<unk>"]), dtype=torch.long))
        tensors["future_tools"].append(torch.tensor(encode_id_list(target_ids, tool_vocab, args.max_actions), dtype=torch.long))
        tensors["available_tools"].append(torch.tensor(encode_id_list(row.get("available_tools") or [], tool_vocab, args.max_available_tools), dtype=torch.long))
        tensors["stop_target"].append(torch.tensor(1 if row.get("stop_target") else 0, dtype=torch.long))
        metadata.append(
            {
                "record_id": row.get("record_id"),
                "task_id": row.get("task_id"),
                "source": row.get("source"),
                "domain": row.get("domain"),
                "split": row.get("split"),
                "replan_step_idx": row.get("replan_step_idx"),
                "next_tool_id": row.get("next_tool"),
                "future_tool_ids": future_tools,
                "stop_target": row.get("stop_target"),
            }
        )
    output = {key: torch.stack(value) if value else torch.empty(0) for key, value in tensors.items()}
    output["metadata"] = metadata
    output["schema_version"] = "replan_tensor_hash_v1"
    output["tool_vocab"] = tool_vocab
    return output


def validate_split(data: dict[str, Any]) -> list[str]:
    size = len(data["metadata"])
    required = ["c_i", "y_i", "input_ids", "attention_mask", "gold_tool_call_ids", "next_tool", "future_tools", "available_tools", "stop_target"]
    errors = []
    for key in required:
        if key not in data:
            errors.append(f"missing key: {key}")
        elif data[key].shape[0] != size:
            errors.append(f"{key} first dimension {data[key].shape[0]} != metadata {size}")
    if "c_i" in data and "y_i" in data and data["c_i"].shape != data["y_i"].shape:
        errors.append(f"c_i shape {tuple(data['c_i'].shape)} != y_i shape {tuple(data['y_i'].shape)}")
    return errors


def write_readme(text_dir: Path, tensor_dir: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Replan Tensor Data",
        "",
        "Feedback-conditioned replanning data derived from executed tau2 telecom traces.",
        "",
        "Each sample conditions on prompt, assistant initialization feedback, executed prefix, tool result, DB hash, and replan reason. The target is the remaining gold workflow from the replan point, or `<stop>` when no further tool should be called.",
        "",
        "This is a hash-feature bootstrap for local shape checks. A stronger version should reuse the Qwen encoder extraction path used by `fm_tensor_encoder_full`.",
        "",
        "## Splits",
        "",
    ]
    for split, info in manifest["splits"].items():
        lines.append(f"- {split}: {info['records']} records, stop rate {info['stop_rate']:.4f}")
    lines.extend(
        [
            "",
            "## Paths",
            "",
            f"- text: `{display_path(text_dir)}`",
            f"- tensor: `{display_path(tensor_dir)}`",
        ]
    )
    (tensor_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def display_path(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build text and hash tensor data from closed-loop replan records.")
    parser.add_argument("--replan-dir", default=str(DEFAULT_REPLAN_DIR))
    parser.add_argument("--text-dir", default=str(DEFAULT_TEXT_DIR))
    parser.add_argument("--tensor-dir", default=str(DEFAULT_TENSOR_DIR))
    parser.add_argument("--tasks", default=str(ROOT / "data" / "processed" / "tasks.jsonl"))
    parser.add_argument("--splits", nargs="+", default=["dev", "test"])
    parser.add_argument("--domain", default="telecom")
    parser.add_argument("--latent-dim", type=int, default=768)
    parser.add_argument("--max-input-len", type=int, default=2048)
    parser.add_argument("--token-vocab-size", type=int, default=32000)
    parser.add_argument("--max-actions", type=int, default=32)
    parser.add_argument("--max-available-tools", type=int, default=128)
    args = parser.parse_args()

    replan_dir = Path(args.replan_dir)
    text_dir = Path(args.text_dir)
    tensor_dir = Path(args.tensor_dir)
    text_dir.mkdir(parents=True, exist_ok=True)
    tensor_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_tasks(args.tasks)
    text_rows_by_split = {}
    for split in args.splits:
        records = read_jsonl(replan_dir / f"replan_records.{split}.{args.domain}.jsonl")
        rows = [normalize_record(record, tasks) for record in records]
        text_rows_by_split[split] = rows
        write_jsonl(text_dir / f"{split}.{args.domain}.jsonl", rows)

    tool_vocab = build_tool_vocab(text_rows_by_split)
    (tensor_dir / "tool_vocab.json").write_text(json.dumps(tool_vocab, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest = {
        "format": "replan_tensor_hash_v1",
        "source_dir": display_path(replan_dir),
        "text_dir": display_path(text_dir),
        "latent_dim": args.latent_dim,
        "max_input_len": args.max_input_len,
        "max_actions": args.max_actions,
        "max_available_tools": args.max_available_tools,
        "tool_vocab_size": len(tool_vocab),
        "stop_tool": STOP_TOOL,
        "splits": {},
    }

    for split, rows in text_rows_by_split.items():
        data = encode_split(rows, tool_vocab, args)
        errors = validate_split(data)
        if errors:
            raise ValueError(f"{split} validation failed: {'; '.join(errors)}")
        out_path = tensor_dir / f"{split}.{args.domain}.pt"
        torch.save(data, out_path)
        stop_count = sum(row.get("stop_target") for row in rows)
        manifest["splits"][split] = {
            "text_path": display_path(text_dir / f"{split}.{args.domain}.jsonl"),
            "tensor_path": display_path(out_path),
            "records": len(rows),
            "stop_rate": stop_count / len(rows) if rows else 0.0,
            "latent_shape": list(data["c_i"].shape),
            "input_shape": list(data["input_ids"].shape),
            "future_tools_shape": list(data["future_tools"].shape),
            "available_tools_shape": list(data["available_tools"].shape),
        }
        print(f"{split}: wrote {out_path} with {len(rows)} samples")

    (tensor_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (text_dir / "manifest.json").write_text(json.dumps({"format": "replan_text_v1", "splits": manifest["splits"]}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_readme(text_dir, tensor_dir, manifest)
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
