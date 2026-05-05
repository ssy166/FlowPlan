import argparse
import json
import math
import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


ROOT = Path(__file__).resolve().parents[1]


def ddp_info():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return world_size, rank, local_rank


def rank0_print(rank, payload):
    if rank == 0:
        print(payload)


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def sample_key(sample):
    if sample.get("record_id"):
        return str(sample["record_id"])
    metadata = sample.get("metadata")
    if isinstance(metadata, dict) and metadata.get("record_id"):
        return str(metadata["record_id"])
    return f"{sample.get('trajectory_id')}::step::{int(sample.get('step_idx') or 0)}"


def load_tensor_samples(path):
    payload = torch_load(path, map_location="cpu")
    samples = payload.get("samples") if isinstance(payload, dict) else payload
    if not isinstance(samples, list):
        raise ValueError(f"{path} must contain a sample list or a dict with samples.")
    return samples


def load_latents(path, latent_key):
    samples = load_tensor_samples(path)
    out = {}
    for sample in samples:
        key = sample_key(sample)
        if latent_key not in sample:
            raise KeyError(f"{latent_key} missing in tensor sample {key}")
        out[key] = sample[latent_key].float().reshape(-1)
    return out


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
        return self.velocity_head(
            self.backbone(torch.cat([self.x_norm(x_t), self.cond_norm(c_i), self.time_embed(t)], dim=-1))
        )


def cfg(config, dotted, default):
    cur = config
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def build_fm_model(checkpoint_path, device):
    ckpt = torch_load(checkpoint_path, map_location=device)
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
def load_sampled_fm_latents(path, checkpoint_path, device, batch_size, inference_steps, noise_std, seed):
    samples = load_tensor_samples(path)
    fm = build_fm_model(checkpoint_path, device)
    out = {}
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        c_i = torch.stack([sample["c_i"].float().reshape(-1) for sample in batch]).to(device)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed + start)
        x_t = torch.randn(
            c_i.shape[0],
            fm.x_norm.normalized_shape[0],
            generator=generator,
            device=device,
        ) * noise_std
        dt = 1.0 / inference_steps
        for step in range(inference_steps):
            t = torch.full((c_i.shape[0], 1), step * dt, device=device)
            x_t = x_t + dt * fm(x_t, t, c_i)
        for sample, vec in zip(batch, x_t.cpu()):
            out[sample_key(sample)] = vec.float().reshape(-1)
    del fm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def format_messages(tokenizer, messages):
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    parts = []
    for message in messages:
        parts.append(f"<|{message['role'].upper()}|>\n{message['content']}")
    return "\n".join(parts) + "\n"


def format_prompt(tokenizer, messages):
    prompt_messages = messages[:-1]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    parts = []
    for message in prompt_messages:
        parts.append(f"<|{message['role'].upper()}|>\n{message['content']}")
    parts.append("<|ASSISTANT|>\n")
    return "\n".join(parts)


class PrefixSFTDataset(Dataset):
    def __init__(self, rows, latents, tokenizer, max_length):
        self.items = []
        self.latents = latents
        self.tokenizer = tokenizer
        self.max_length = max_length
        missing = 0
        for row in rows:
            row_id = row.get("id")
            if row_id not in latents:
                missing += 1
                continue
            self.items.append(row)
        if missing:
            print(json.dumps({"skipped_missing_latent": missing}, ensure_ascii=False))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        row = self.items[idx]
        messages = row["messages"]
        prompt_text = format_prompt(self.tokenizer, messages)
        answer_text = messages[-1]["content"] + (self.tokenizer.eos_token or "")
        prompt_ids = self.tokenizer(prompt_text, truncation=False, add_special_tokens=False)["input_ids"]
        answer_ids = self.tokenizer(answer_text, truncation=False, add_special_tokens=False)["input_ids"]
        if len(answer_ids) >= self.max_length:
            answer_ids = answer_ids[: self.max_length]
            prompt_ids = []
        elif len(prompt_ids) + len(answer_ids) > self.max_length:
            prompt_budget = self.max_length - len(answer_ids)
            prompt_ids = prompt_ids[-prompt_budget:]
        input_ids = torch.tensor(prompt_ids + answer_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        prompt_len = min(len(prompt_ids), labels.numel())
        labels[:prompt_len] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "fm_latent": self.latents[row["id"]],
            "id": row["id"],
        }


def collate(batch, pad_token_id):
    max_len = max(item["input_ids"].numel() for item in batch)
    input_ids = []
    attention_mask = []
    labels = []
    latents = []
    for item in batch:
        pad = max_len - item["input_ids"].numel()
        input_ids.append(torch.cat([item["input_ids"], torch.full((pad,), pad_token_id, dtype=torch.long)]))
        attention_mask.append(torch.cat([item["attention_mask"], torch.zeros(pad, dtype=torch.long)]))
        labels.append(torch.cat([item["labels"], torch.full((pad,), -100, dtype=torch.long)]))
        latents.append(item["fm_latent"])
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
        "fm_latent": torch.stack(latents),
    }


class PrefixProjector(nn.Module):
    def __init__(self, latent_dim, llm_hidden_dim, prefix_len, projector_hidden_dim=512, dropout=0.0, gate_init=-4.0):
        super().__init__()
        self.latent_norm = nn.LayerNorm(latent_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, projector_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projector_hidden_dim, prefix_len * llm_hidden_dim),
        )
        self.prefix_len = prefix_len
        self.hidden_dim = llm_hidden_dim
        self.projector_hidden_dim = projector_hidden_dim
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, fm_latent):
        prefix = self.net(self.latent_norm(fm_latent)).view(-1, self.prefix_len, self.hidden_dim)
        return torch.sigmoid(self.gate_logit) * prefix


class PrefixCausalLM(nn.Module):
    def __init__(self, base_model, projector=None):
        super().__init__()
        self.base_model = base_model
        self.projector = projector
        self.prefix_len = projector.prefix_len if projector is not None else 0

    def forward(self, input_ids, attention_mask, labels, fm_latent):
        if self.projector is None:
            return self.base_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        token_embeds = self.base_model.get_input_embeddings()(input_ids)
        prefix = self.projector(fm_latent.float()).to(token_embeds.dtype)
        inputs_embeds = torch.cat([prefix, token_embeds], dim=1)
        prefix_mask = torch.ones((attention_mask.shape[0], prefix.shape[1]), device=attention_mask.device, dtype=attention_mask.dtype)
        attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        prefix_labels = torch.full((labels.shape[0], prefix.shape[1]), -100, device=labels.device, dtype=labels.dtype)
        labels = torch.cat([prefix_labels, labels], dim=1)
        return self.base_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)


def token_accuracy(logits, labels):
    shifted_logits = logits[:, :-1].argmax(dim=-1)
    shifted_labels = labels[:, 1:]
    mask = shifted_labels.ne(-100)
    if not mask.any():
        return None
    return shifted_logits.eq(shifted_labels).masked_select(mask).float().mean().item()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    losses = []
    accs = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        losses.append(float(out.loss.detach().cpu()))
        labels = batch["labels"]
        if model.prefix_len:
            labels = torch.cat([
                torch.full((batch["labels"].shape[0], model.prefix_len), -100, device=device, dtype=batch["labels"].dtype),
                batch["labels"],
            ], dim=1)
        acc = token_accuracy(out.logits.detach(), labels)
        if acc is not None:
            accs.append(acc)
    model.train()
    return {
        "eval_loss": sum(losses) / max(len(losses), 1),
        "eval_mean_token_accuracy": sum(accs) / max(len(accs), 1) if accs else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Minimal FM soft-prefix SFT trainer.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "processed" / "fm_conditioned_sft_smoke"))
    parser.add_argument("--tensor-dir", default=str(ROOT / "data" / "processed" / "fm_tensor_encoder_full"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--latent-key", default="y_i", choices=["c_i", "y_i", "proxy_c_i", "y_hat"])
    parser.add_argument("--fm-checkpoint", default=str(ROOT / "outputs" / "fm_full" / "encoder_v3_cosopt" / "best.pt"))
    parser.add_argument("--fm-inference-steps", type=int, default=16)
    parser.add_argument("--fm-noise-std", type=float, default=1.0)
    parser.add_argument("--fm-sample-batch-size", type=int, default=64)
    parser.add_argument("--no-prefix", action="store_true", help="Run matched assistant-only LoRA ablation without FM prefix.")
    parser.add_argument("--prefix-len", type=int, default=8)
    parser.add_argument("--projector-hidden-dim", type=int, default=512)
    parser.add_argument("--prefix-dropout", type=float, default=0.05)
    parser.add_argument("--gate-init", type=float, default=-4.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=6144)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-dev-rows", type=int, default=0)
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    args = parser.parse_args()

    world_size, rank, local_rank = ddp_info()
    distributed = world_size > 1
    if distributed:
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel

        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    else:
        dist = None
        DistributedDataParallel = None

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = Path(args.out_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_rows = read_jsonl(Path(args.data_dir) / "train.jsonl")
    dev_rows = read_jsonl(Path(args.data_dir) / "dev.jsonl")
    if args.max_train_rows:
        train_rows = train_rows[: args.max_train_rows]
    if args.max_dev_rows:
        dev_rows = dev_rows[: args.max_dev_rows]
    if args.no_prefix:
        train_latents = {row["id"]: torch.zeros(1) for row in train_rows}
        dev_latents = {row["id"]: torch.zeros(1) for row in dev_rows}
    elif args.latent_key == "y_hat":
        train_latents = load_sampled_fm_latents(
            Path(args.tensor_dir) / "train.pt",
            args.fm_checkpoint,
            device,
            args.fm_sample_batch_size,
            args.fm_inference_steps,
            args.fm_noise_std,
            args.seed,
        )
        dev_latents = load_sampled_fm_latents(
            Path(args.tensor_dir) / "dev.pt",
            args.fm_checkpoint,
            device,
            args.fm_sample_batch_size,
            args.fm_inference_steps,
            args.fm_noise_std,
            args.seed + 1_000_000,
        )
    else:
        train_latents = load_latents(Path(args.tensor_dir) / "train.pt", args.latent_key)
        dev_latents = load_latents(Path(args.tensor_dir) / "dev.pt", args.latent_key)

    train_ds = PrefixSFTDataset(train_rows, train_latents, tokenizer, args.max_length)
    dev_ds = PrefixSFTDataset(dev_rows, dev_latents, tokenizer, args.max_length)
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed) if distributed else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
    )
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: collate(b, tokenizer.pad_token_id))
    rank0_print(
        rank,
        json.dumps(
            {
                "train": len(train_ds),
                "dev": len(dev_ds),
                "latent_key": None if args.no_prefix else args.latent_key,
                "no_prefix": args.no_prefix,
                "distributed": distributed,
                "world_size": world_size,
            },
            ensure_ascii=False,
        ),
    )

    base = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    base.config.use_cache = False
    if not args.no_gradient_checkpointing:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    base = get_peft_model(base, lora_config)
    hidden_dim = int(base.get_input_embeddings().embedding_dim)
    latent_dim = next(iter(train_latents.values())).numel()
    projector = None
    if not args.no_prefix:
        projector = PrefixProjector(
            latent_dim,
            hidden_dim,
            args.prefix_len,
            args.projector_hidden_dim,
            args.prefix_dropout,
            args.gate_init,
        )
    model = PrefixCausalLM(base, projector).to(device)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    total_steps = math.ceil(len(train_loader) * args.epochs / max(args.grad_accum, 1))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    log_history = []
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / args.grad_accum
            loss.backward()
            if step % args.grad_accum == 0 or step == len(train_loader):
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % args.logging_steps == 0 or global_step == 1:
                    unwrapped = model.module if distributed else model
                    log = {
                        "step": global_step,
                        "epoch": epoch + step / max(len(train_loader), 1),
                        "loss": float((loss.detach() * args.grad_accum).cpu()),
                        "grad_norm": float(grad_norm.detach().cpu()),
                        "prefix_gate": float(torch.sigmoid(unwrapped.projector.gate_logit).detach().cpu()) if unwrapped.projector is not None else None,
                    }
                    if rank == 0:
                        print(json.dumps(log, ensure_ascii=False))
                        log_history.append(log)

    if distributed:
        dist.barrier()
    unwrapped = model.module if distributed else model
    if rank == 0:
        eval_metrics = evaluate(unwrapped, dev_loader, device)
        eval_metrics["prefix_gate"] = float(torch.sigmoid(unwrapped.projector.gate_logit).detach().cpu()) if unwrapped.projector is not None else None
        print(json.dumps(eval_metrics, ensure_ascii=False))
    else:
        eval_metrics = {}
    if distributed:
        dist.barrier()

    if rank == 0:
        final_dir = out_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        unwrapped.base_model.save_pretrained(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        if unwrapped.projector is not None:
            torch.save(
                {
                    "projector": unwrapped.projector.state_dict(),
                    "config": {
                    "latent_key": args.latent_key,
                    "fm_checkpoint": args.fm_checkpoint if args.latent_key == "y_hat" else None,
                    "fm_inference_steps": args.fm_inference_steps if args.latent_key == "y_hat" else None,
                    "fm_noise_std": args.fm_noise_std if args.latent_key == "y_hat" else None,
                    "latent_dim": latent_dim,
                        "hidden_dim": hidden_dim,
                        "projector_hidden_dim": args.projector_hidden_dim,
                        "prefix_len": args.prefix_len,
                        "prefix_dropout": args.prefix_dropout,
                        "gate_init": args.gate_init,
                    },
                },
                final_dir / "fm_prefix_projector.pt",
            )
        metrics = {
            "train": len(train_ds),
            "dev": len(dev_ds),
            "latent_key": None if args.no_prefix else args.latent_key,
            "fm_checkpoint": args.fm_checkpoint if args.latent_key == "y_hat" else None,
            "fm_inference_steps": args.fm_inference_steps if args.latent_key == "y_hat" else None,
            "fm_noise_std": args.fm_noise_std if args.latent_key == "y_hat" else None,
            "log_history": log_history,
            "gradient_checkpointing": not args.no_gradient_checkpointing,
            "distributed": distributed,
            "world_size": world_size,
            "global_batch_size": args.batch_size * args.grad_accum * world_size,
            **eval_metrics,
        }
        (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"saved": str(final_dir)}, ensure_ascii=False))
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
