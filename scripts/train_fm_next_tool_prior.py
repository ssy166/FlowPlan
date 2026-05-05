import argparse
import json
import sys
from pathlib import Path
import hashlib
import re

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_fm_prefix_sft_generation import actions_from_assistant, available_tool_maps, read_jsonl, write_jsonl  # noqa: E402
from train_fm_prefix_sft import load_latents, load_sampled_fm_latents, torch_load  # noqa: E402


STOP = "<stop>"
STAGE_OTHER = "other"
STAGE_STOP = "stop"
STAGE_RETAIL_USER_LOOKUP = "retail_user_lookup"
STAGE_RETAIL_ORDER_LOOKUP = "retail_order_lookup"
STAGE_RETAIL_PRODUCT_LOOKUP = "retail_product_lookup"
STAGE_RETAIL_ITEM_LOOKUP = "retail_item_lookup"
STAGE_RETAIL_COMPUTE = "retail_compute"
STAGE_RETAIL_OPERATION = "retail_operation"
STAGE_RETAIL_OTHER = "retail_other"
STAGE_VOCAB = [
    STAGE_OTHER,
    STAGE_STOP,
    STAGE_RETAIL_USER_LOOKUP,
    STAGE_RETAIL_ORDER_LOOKUP,
    STAGE_RETAIL_PRODUCT_LOOKUP,
    STAGE_RETAIL_ITEM_LOOKUP,
    STAGE_RETAIL_COMPUTE,
    STAGE_RETAIL_OPERATION,
    STAGE_RETAIL_OTHER,
]
RETAIL_INTENT_KEYWORDS = [
    "cancel",
    "exchange",
    "return",
    "modify",
    "change",
    "address",
    "refund",
    "replace",
    "delivered",
    "pending",
    "order",
    "item",
    "payment",
    "product",
]
RETAIL_INTENT_PATTERNS = [
    ("intent_return", r"\breturn|refund\b"),
    ("intent_exchange", r"\bexchange|replace|replacement\b"),
    ("intent_cancel", r"\bcancel\b"),
    ("intent_modify", r"\bmodify|change|update\b"),
    ("intent_address", r"\baddress|shipping\b"),
    ("intent_payment", r"\bpayment|card|paypal|gift card\b"),
    ("intent_delivered", r"\bdelivered|received\b"),
    ("intent_pending", r"\bpending|not shipped|before it ships\b"),
    ("intent_item", r"\bitem|product|size|color|variant\b"),
]


def load_tools(path):
    return {row["tool_id"]: row for row in read_jsonl(path)}


def label_for_row(row):
    actions, stop = actions_from_assistant(row)
    if stop or not actions:
        return STOP
    return actions[0].get("tool_id") or STOP


def build_vocab(tools):
    return [STOP] + sorted(tools)


def stage_for_tool_id(tool_id, tools):
    if tool_id == STOP:
        return STAGE_STOP
    tool = tools.get(tool_id) or {}
    if "::retail::" not in str(tool_id):
        return STAGE_OTHER
    name = tool.get("name") or str(tool_id).rsplit("::", 1)[-1]
    if name.startswith("find_user_id") or name == "get_user_details":
        return STAGE_RETAIL_USER_LOOKUP
    if name == "get_order_details":
        return STAGE_RETAIL_ORDER_LOOKUP
    if name in {"get_product_details", "list_all_product_types"}:
        return STAGE_RETAIL_PRODUCT_LOOKUP
    if name == "get_item_details":
        return STAGE_RETAIL_ITEM_LOOKUP
    if name == "calculate":
        return STAGE_RETAIL_COMPUTE
    if name.startswith(("cancel_", "exchange_", "modify_", "return_")) or name == "transfer_to_human_agents":
        return STAGE_RETAIL_OPERATION
    return STAGE_RETAIL_OTHER


def stable_hash(text):
    digest = hashlib.blake2b(str(text).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def find_balanced_json(text, start):
    first = text.find("{", max(start, 0))
    if first < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx, ch in enumerate(text[first:], start=first):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[first : idx + 1]
    return None


def parse_conditioned_state(row):
    content = ""
    for message in row.get("messages") or []:
        if message.get("role") == "user":
            content = message.get("content") or ""
            break
    starts = []
    for marker in ["Compact state:", "Feedback-conditioned state:", "Conditioned workflow state:"]:
        marker_pos = content.find(marker)
        if marker_pos >= 0:
            starts.append(marker_pos)
    starts.append(0)
    for start in starts:
        candidate = find_balanced_json(content, start)
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def bucket_count(name, value):
    try:
        count = len(value)
    except TypeError:
        count = int(bool(value))
    if count <= 0:
        bucket = "0"
    elif count == 1:
        bucket = "1"
    elif count <= 3:
        bucket = "2-3"
    elif count <= 8:
        bucket = "4-8"
    else:
        bucket = "9+"
    return f"{name}:{bucket}"


def add_token(tokens, name, value=True):
    if value is None:
        return
    if isinstance(value, bool):
        tokens.append(f"{name}:{str(value).lower()}")
    elif isinstance(value, (int, float)):
        tokens.append(f"{name}:{value}")
    elif isinstance(value, str) and value:
        tokens.append(f"{name}:{value}")


def add_result_summary_tokens(tokens, prefix, result_summary):
    if not isinstance(result_summary, dict):
        return
    add_token(tokens, f"{prefix}_status", result_summary.get("status"))
    for key in [
        "orders",
        "order_ids",
        "item_ids",
        "fulfillment_item_ids",
        "product_ids",
        "payment_method_ids",
        "new_item_ids",
    ]:
        values = result_summary.get(key)
        if isinstance(values, list):
            tokens.append(bucket_count(f"{prefix}_{key}", values))
    for key in ["order_id", "user_id", "product_id"]:
        add_token(tokens, f"{prefix}_has_{key}", bool(result_summary.get(key)))
    item_names = result_summary.get("item_names")
    if isinstance(item_names, list):
        for name in item_names[:8]:
            if isinstance(name, str) and name:
                tokens.append(f"{prefix}_item_name:{name.lower()}")


def add_retail_intent_tokens(tokens, task_text):
    for keyword in RETAIL_INTENT_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", task_text):
            tokens.append(f"task_keyword:{keyword}")
    matched = set()
    for name, pattern in RETAIL_INTENT_PATTERNS:
        if re.search(pattern, task_text):
            tokens.append(name)
            matched.add(name)
    for first, second in [
        ("intent_return", "intent_delivered"),
        ("intent_exchange", "intent_delivered"),
        ("intent_modify", "intent_pending"),
        ("intent_modify", "intent_item"),
        ("intent_modify", "intent_address"),
        ("intent_modify", "intent_payment"),
        ("intent_cancel", "intent_pending"),
    ]:
        if first in matched and second in matched:
            tokens.append(f"{first}+{second}")


def compact_state_tokens(row, args):
    state = parse_conditioned_state(row)
    metadata = row.get("metadata") or {}
    tokens = []
    domain = metadata.get("domain") or state.get("domain")
    add_token(tokens, "domain", domain)
    add_token(tokens, "source", metadata.get("source") or state.get("source"))
    add_token(tokens, "stop_meta", metadata.get("stop"))
    step = metadata.get("replan_step_idx", state.get("step"))
    if isinstance(step, int):
        add_token(tokens, "step", min(step, 6))
        add_token(tokens, "step_is_zero", step == 0)
    reason = state.get("replan_reason") or metadata.get("replan_reason") or {}
    if isinstance(reason, dict):
        for key in ["execution_ok", "error_type", "tool_match", "gold_prefix"]:
            add_token(tokens, f"reason_{key}", reason.get(key))
    available = state.get("available_tools") or []
    if isinstance(available, list):
        tokens.append(bucket_count("available_tools", available))
        for tool in available:
            if not isinstance(tool, dict):
                continue
            add_token(tokens, "available_tool", tool.get("name") or str(tool.get("tool_id", "")).rsplit("::", 1)[-1])
            add_token(tokens, "available_phase", tool.get("phase"))
    prefix = state.get("executed_prefix") or []
    if isinstance(prefix, list):
        tokens.append(bucket_count("prefix_len", prefix))
        for idx, step_row in enumerate(prefix[-4:]):
            if not isinstance(step_row, dict):
                continue
            tool_name = step_row.get("tool_name") or (step_row.get("action") or {}).get("tool_name")
            add_token(tokens, f"prefix_tool_{idx - min(len(prefix), 4)}", tool_name)
            add_token(tokens, "prefix_any_tool", tool_name)
            add_token(tokens, "prefix_ok", step_row.get("ok"))
            add_token(tokens, "prefix_tool_match", step_row.get("tool_match"))
            if args.state_feature_detail == "result":
                add_result_summary_tokens(tokens, f"prefix_result_{idx - min(len(prefix), 4)}", step_row.get("result_summary"))
        if prefix:
            last = prefix[-1] if isinstance(prefix[-1], dict) else {}
            add_token(tokens, "last_tool", last.get("tool_name") or (last.get("action") or {}).get("tool_name"))
            add_token(tokens, "last_ok", last.get("ok"))
            if args.state_feature_detail == "result":
                add_result_summary_tokens(tokens, "last_result", last.get("result_summary"))
    progress = state.get("grounding_progress") or {}
    if isinstance(progress, dict):
        add_token(tokens, "lookup_completed", progress.get("lookup_completed"))
        add_token(tokens, "order_grounded", progress.get("order_grounded"))
        add_token(tokens, "last_successful_lookup_tool", progress.get("last_successful_lookup_tool"))
        add_token(tokens, "resolved_order_status", progress.get("resolved_order_status"))
        for key in [
            "resolved_user_ids",
            "resolved_order_ids",
            "resolved_order_item_ids",
            "resolved_payment_method_ids",
            "operation_eligibility",
        ]:
            values = progress.get(key) or []
            if isinstance(values, list):
                tokens.append(bucket_count(key, values))
                for value in values[:16]:
                    add_token(tokens, key, value)
    task_text = str(state.get("task") or "").lower()
    if domain == "retail":
        if args.state_feature_detail == "result":
            add_retail_intent_tokens(tokens, task_text)
        else:
            for keyword in RETAIL_INTENT_KEYWORDS:
                if re.search(rf"\b{re.escape(keyword)}\b", task_text):
                    tokens.append(f"task_keyword:{keyword}")
    return tokens


def state_feature_vector(row, args):
    if args.state_feature_mode == "none":
        return None
    tokens = compact_state_tokens(row, args)
    dim = int(args.state_feature_dim)
    vec = torch.zeros(dim, dtype=torch.float32)
    for token in tokens:
        idx = stable_hash(token) % dim
        sign = 1.0 if (stable_hash(f"sign::{token}") % 2 == 0) else -1.0
        vec[idx] += sign
    norm = vec.norm(p=2)
    if norm > 0:
        vec = vec / norm
    return vec * float(args.state_feature_scale)


def load_composed_latents(tensor_path, latent_key, args, device):
    if args.feature_mode == "single":
        if latent_key == "y_hat":
            return load_sampled_fm_latents(
                tensor_path,
                args.fm_checkpoint,
                device,
                args.fm_sample_batch_size,
                args.fm_inference_steps,
                args.fm_noise_std,
                args.seed,
            )
        return load_latents(tensor_path, latent_key)
    if args.feature_mode != "ci_yhat_delta":
        raise ValueError(f"Unknown feature mode: {args.feature_mode}")
    c_i = load_latents(tensor_path, "c_i")
    y_hat = load_sampled_fm_latents(
        tensor_path,
        args.fm_checkpoint,
        device,
        args.fm_sample_batch_size,
        args.fm_inference_steps,
        args.fm_noise_std,
        args.seed,
    )
    out = {}
    for key, c_vec in c_i.items():
        y_vec = y_hat.get(key)
        if y_vec is None:
            continue
        out[key] = torch.cat([c_vec.float().reshape(-1), y_vec.float().reshape(-1), (y_vec - c_vec).float().reshape(-1)])
    return out


def keep_domain(row, domain_filter):
    if not domain_filter or domain_filter == "all":
        return True
    return (row.get("metadata") or {}).get("domain") == domain_filter


def load_rows_with_latents(path, tensor_path, latent_key, args, device, domain_filter="all"):
    rows = read_jsonl(path)
    latents = load_composed_latents(tensor_path, latent_key, args, device)
    items = []
    missing = 0
    for row in rows:
        if not keep_domain(row, domain_filter):
            continue
        row_id = row.get("id")
        latent = latents.get(row_id)
        if latent is None:
            missing += 1
            continue
        latent = latent.float().reshape(-1)
        state_vec = state_feature_vector(row, args)
        if state_vec is not None:
            latent = torch.cat([latent, state_vec])
        items.append((row, latent))
    if missing:
        print(json.dumps({"missing_latent": missing, "path": str(path)}), flush=True)
    return items


def sample_weight(row, label, tools, args):
    weight = 1.0
    metadata = row.get("metadata") or {}
    domain = metadata.get("domain")
    if domain == "retail":
        weight *= float(args.retail_loss_weight)
    state = parse_conditioned_state(row)
    progress = state.get("grounding_progress") or {}
    if domain == "retail" and isinstance(progress, dict):
        if progress.get("lookup_completed") or progress.get("order_grounded"):
            weight *= float(args.post_lookup_loss_weight)
    tool = tools.get(label) or {}
    if domain == "retail" and label != STOP and tool.get("phase") in {"operate", "verify"}:
        weight *= float(args.operation_loss_weight)
    if label == STOP:
        weight *= float(args.stop_loss_weight)
    return weight


class NextToolDataset(Dataset):
    def __init__(self, items, tools, vocab_index, args):
        self.items = items
        self.tools = tools
        self.vocab_index = vocab_index
        self.stage_index = {label: idx for idx, label in enumerate(STAGE_VOCAB)}
        self.args = args

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        row, latent = self.items[idx]
        label = label_for_row(row)
        label_idx = self.vocab_index.get(label, self.vocab_index[STOP])
        stage_idx = self.stage_index[stage_for_tool_id(label, self.tools)]
        available_ids, _ = available_tool_maps(row, self.tools)
        allowed = [self.vocab_index[STOP]]
        allowed.extend(self.vocab_index[tool_id] for tool_id in available_ids if tool_id in self.vocab_index)
        weight = sample_weight(row, label, self.tools, self.args)
        return {
            "latent": latent,
            "label": torch.tensor(label_idx, dtype=torch.long),
            "stage_label": torch.tensor(stage_idx, dtype=torch.long),
            "allowed": torch.tensor(sorted(set(allowed)), dtype=torch.long),
            "weight": torch.tensor(weight, dtype=torch.float32),
            "row": row,
        }


def collate(batch):
    max_allowed = max(item["allowed"].numel() for item in batch)
    allowed = []
    allowed_mask = []
    for item in batch:
        pad = max_allowed - item["allowed"].numel()
        allowed.append(torch.cat([item["allowed"], torch.zeros(pad, dtype=torch.long)]))
        allowed_mask.append(torch.cat([torch.ones(item["allowed"].numel(), dtype=torch.bool), torch.zeros(pad, dtype=torch.bool)]))
    return {
        "latent": torch.stack([item["latent"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
        "stage_label": torch.stack([item["stage_label"] for item in batch]),
        "weight": torch.stack([item["weight"] for item in batch]),
        "allowed": torch.stack(allowed),
        "allowed_mask": torch.stack(allowed_mask),
        "rows": [item["row"] for item in batch],
    }


class NextToolPrior(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, dropout, stage_output_dim=0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.tool_head = nn.Linear(hidden_dim, output_dim)
        self.stage_head = nn.Linear(hidden_dim, stage_output_dim) if stage_output_dim else None

    def forward(self, latent):
        hidden = self.encoder(latent)
        stage_logits = self.stage_head(hidden) if self.stage_head is not None else None
        return self.tool_head(hidden), stage_logits


def masked_logits(logits, allowed, allowed_mask):
    mask = torch.zeros_like(logits, dtype=torch.bool)
    valid_allowed = allowed.masked_fill(~allowed_mask, 0)
    mask.scatter_(1, valid_allowed, True)
    return logits.masked_fill(~mask, -1e9)


def apply_stage_logits(tool_logits, stage_logits, tool_stage_ids, stage_logit_weight):
    if stage_logits is None or not stage_logit_weight:
        return tool_logits
    stage_scores = stage_logits[:, tool_stage_ids.to(stage_logits.device)]
    return tool_logits + stage_scores * float(stage_logit_weight)


def summarize(preds, labels, rows, vocab, stage_preds=None, stage_labels=None):
    total = len(labels)
    hits = sum(int(p == y) for p, y in zip(preds, labels))
    by_domain = {}
    for pred, label, row in zip(preds, labels, rows):
        domain = (row.get("metadata") or {}).get("domain") or "unknown"
        stats = by_domain.setdefault(domain, {"count": 0, "hits": 0})
        stats["count"] += 1
        stats["hits"] += int(pred == label)
    out = {
        "count": total,
        "accuracy": hits / max(total, 1),
        "by_domain": {
            domain: {"count": stats["count"], "accuracy": stats["hits"] / max(stats["count"], 1)}
            for domain, stats in sorted(by_domain.items())
        },
        "pred_stop_rate": sum(1 for p in preds if vocab[p] == STOP) / max(total, 1),
        "gold_stop_rate": sum(1 for y in labels if vocab[y] == STOP) / max(total, 1),
    }
    if stage_preds is not None and stage_labels is not None:
        out["stage_accuracy"] = sum(int(p == y) for p, y in zip(stage_preds, stage_labels)) / max(total, 1)
        by_stage = {}
        for pred, label in zip(stage_preds, stage_labels):
            stage = STAGE_VOCAB[label]
            stats = by_stage.setdefault(stage, {"count": 0, "hits": 0})
            stats["count"] += 1
            stats["hits"] += int(pred == label)
        out["by_stage"] = {
            stage: {"count": stats["count"], "accuracy": stats["hits"] / max(stats["count"], 1)}
            for stage, stats in sorted(by_stage.items())
        }
    return out


@torch.no_grad()
def evaluate(model, loader, device, vocab, tool_stage_ids, args):
    model.eval()
    labels = []
    preds = []
    stage_labels = []
    stage_preds = []
    rows = []
    loss_sum = 0.0
    for batch in loader:
        latent = batch["latent"].to(device)
        label = batch["label"].to(device)
        raw_logits, stage_logits = model(latent)
        logits = apply_stage_logits(raw_logits, stage_logits, tool_stage_ids, args.stage_logit_weight)
        logits = masked_logits(logits, batch["allowed"].to(device), batch["allowed_mask"].to(device))
        loss = F.cross_entropy(logits, label)
        pred = logits.argmax(dim=-1)
        loss_sum += float(loss.item()) * label.numel()
        labels.extend(label.cpu().tolist())
        preds.extend(pred.cpu().tolist())
        if stage_logits is not None:
            stage_label = batch["stage_label"].to(device)
            stage_labels.extend(stage_label.cpu().tolist())
            stage_preds.extend(stage_logits.argmax(dim=-1).cpu().tolist())
        rows.extend(batch["rows"])
    out = summarize(preds, labels, rows, vocab, stage_preds or None, stage_labels or None)
    out["loss"] = loss_sum / max(len(labels), 1)
    return out


@torch.no_grad()
def write_predictions(model, dataset, out_path, device, vocab, tools, model_name, tool_stage_ids, args):
    loader = DataLoader(dataset, batch_size=128, shuffle=False, collate_fn=collate)
    rows_out = []
    for batch in loader:
        latent = batch["latent"].to(device)
        raw_logits, stage_logits = model(latent)
        logits = apply_stage_logits(raw_logits, stage_logits, tool_stage_ids, args.stage_logit_weight)
        logits = masked_logits(logits, batch["allowed"].to(device), batch["allowed_mask"].to(device))
        pred = logits.argmax(dim=-1).cpu().tolist()
        probs = torch.softmax(logits, dim=-1).max(dim=-1).values.cpu().tolist()
        stage_pred = stage_logits.argmax(dim=-1).cpu().tolist() if stage_logits is not None else [None] * len(pred)
        for row, pred_idx, prob, stage_idx in zip(batch["rows"], pred, probs, stage_pred):
            tool_id = vocab[pred_idx]
            metadata = row.get("metadata") or {}
            if tool_id == STOP:
                tool_ids = []
                actions = []
            else:
                tool = tools.get(tool_id) or {}
                tool_ids = [tool_id]
                actions = [{"tool_id": tool_id, "tool_name": tool.get("name") or tool_id.rsplit("::", 1)[-1], "arguments": {}}]
            rows_out.append(
                {
                    "task_id": metadata.get("task_id") or row.get("id"),
                    "record_id": row.get("id"),
                    "source": metadata.get("source"),
                    "domain": metadata.get("domain"),
                    "plan_type": model_name,
                    "tool_ids": tool_ids,
                    "actions": actions,
                    "metadata": {
                        "record_id": row.get("id"),
                        "confidence": prob,
                        "decoder": model_name,
                        "stage_prediction": STAGE_VOCAB[stage_idx] if stage_idx is not None else None,
                    },
                }
            )
    write_jsonl(out_path, rows_out)
    print(json.dumps({"predictions": len(rows_out), "out": str(out_path)}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train a lightweight FM-latent next-tool/stop prior.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--tensor-dir", required=True)
    parser.add_argument("--tools", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--latent-key", choices=["c_i", "y_i", "proxy_c_i", "y_hat"], default="y_hat")
    parser.add_argument("--feature-mode", choices=["single", "ci_yhat_delta"], default="single")
    parser.add_argument("--state-feature-mode", choices=["none", "compact"], default="none")
    parser.add_argument("--state-feature-detail", choices=["basic", "result"], default="basic")
    parser.add_argument("--state-feature-dim", type=int, default=256)
    parser.add_argument("--state-feature-scale", type=float, default=1.0)
    parser.add_argument("--retail-loss-weight", type=float, default=1.0)
    parser.add_argument("--post-lookup-loss-weight", type=float, default=1.0)
    parser.add_argument("--operation-loss-weight", type=float, default=1.0)
    parser.add_argument("--stop-loss-weight", type=float, default=1.0)
    parser.add_argument("--aux-stage-loss-weight", type=float, default=0.0)
    parser.add_argument("--stage-logit-weight", type=float, default=0.0)
    parser.add_argument("--fm-checkpoint", default=None)
    parser.add_argument("--fm-inference-steps", type=int, default=16)
    parser.add_argument("--fm-noise-std", type=float, default=0.01)
    parser.add_argument("--fm-sample-batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--domain-filter", choices=["all", "telecom", "retail"], default="all")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    tools = load_tools(args.tools)
    vocab = build_vocab(tools)
    vocab_index = {label: idx for idx, label in enumerate(vocab)}
    stage_index = {label: idx for idx, label in enumerate(STAGE_VOCAB)}
    tool_stage_ids = torch.tensor([stage_index[stage_for_tool_id(tool_id, tools)] for tool_id in vocab], dtype=torch.long)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_items = {
        split: load_rows_with_latents(
            Path(args.data_dir) / f"{split}.jsonl",
            Path(args.tensor_dir) / f"{split}.pt",
            args.latent_key,
            args,
            device,
            args.domain_filter,
        )
        for split in ["train", "dev", "test"]
    }
    if not split_items["train"]:
        raise ValueError(f"No training rows after domain filter: {args.domain_filter}")
    datasets = {split: NextToolDataset(items, tools, vocab_index, args) for split, items in split_items.items()}
    loaders = {
        split: DataLoader(ds, batch_size=args.batch_size, shuffle=(split == "train"), collate_fn=collate)
        for split, ds in datasets.items()
    }
    latent_dim = split_items["train"][0][1].numel()
    stage_output_dim = len(STAGE_VOCAB) if args.aux_stage_loss_weight or args.stage_logit_weight else 0
    model = NextToolPrior(latent_dim, args.hidden_dim, len(vocab), args.dropout, stage_output_dim=stage_output_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = None
    best_acc = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in loaders["train"]:
            latent = batch["latent"].to(device)
            label = batch["label"].to(device)
            raw_logits, stage_logits = model(latent)
            logits = apply_stage_logits(raw_logits, stage_logits, tool_stage_ids, args.stage_logit_weight)
            logits = masked_logits(logits, batch["allowed"].to(device), batch["allowed_mask"].to(device))
            losses = F.cross_entropy(logits, label, reduction="none")
            weight = batch["weight"].to(device)
            loss = (losses * weight).sum() / weight.sum().clamp_min(1e-6)
            if stage_logits is not None and args.aux_stage_loss_weight:
                stage_label = batch["stage_label"].to(device)
                stage_losses = F.cross_entropy(stage_logits, stage_label, reduction="none")
                loss = loss + float(args.aux_stage_loss_weight) * (stage_losses * weight).sum() / weight.sum().clamp_min(1e-6)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        metrics = {"epoch": epoch, "dev": evaluate(model, loaders["dev"], device, vocab, tool_stage_ids, args)}
        history.append(metrics)
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps(metrics, ensure_ascii=False, sort_keys=True), flush=True)
        if metrics["dev"]["accuracy"] > best_acc:
            best_acc = metrics["dev"]["accuracy"]
            best = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    if best is not None:
        model.load_state_dict(best)
    final_metrics = {split: evaluate(model, loaders[split], device, vocab, tool_stage_ids, args) for split in ["train", "dev", "test"]}
    payload = {
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "config": vars(args),
        "vocab": vocab,
        "stage_vocab": STAGE_VOCAB,
        "metrics": final_metrics,
        "history": history,
    }
    torch.save(payload, out_dir / "next_tool_prior.pt")
    (out_dir / "metrics.json").write_text(json.dumps(final_metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"final": final_metrics}, ensure_ascii=False, sort_keys=True), flush=True)
    for split in ["train", "dev", "test"]:
        write_predictions(
            model,
            datasets[split],
            out_dir / f"{split}.pred.jsonl",
            device,
            vocab,
            tools,
            "fm_next_tool_prior_v1",
            tool_stage_ids,
            args,
        )


if __name__ == "__main__":
    main()
