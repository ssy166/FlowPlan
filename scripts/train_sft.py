"""
SFT training script for tool-plan generation using TRL SFTTrainer + LoRA.

Designed for instruction-tuned causal LMs such as Qwen2.5-7B-Instruct.

Example:
    python scripts/train_sft.py \
        --model-path /path/to/models/Qwen2.5-7B-Instruct \
        --data-dir <project_root>/data/processed/fm_sft \
        --out-dir <project_root>/checkpoints/sft_lora_v1 \
        --epochs 3 --batch-size 4 --grad-accum 4

After training, use run_toolrl_inference.py --model-path <out-dir>/merged
to generate predictions, then evaluate with evaluate_plans.py.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def to_hf_dataset(rows, tokenizer, max_length):
    from datasets import Dataset

    def format_row(row):
        messages = row["messages"]
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        else:
            parts = []
            for m in messages:
                role = m["role"].upper()
                parts.append(f"<|{role}|>\n{m['content']}")
            parts.append("")
            text = "\n".join(parts)
        return {"text": text, "id": row.get("id", "")}

    formatted = [format_row(r) for r in rows]
    return Dataset.from_list(formatted)


def build_lora_config(r, alpha, dropout, target_modules):
    from peft import LoraConfig, TaskType

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules or ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )


def main():
    parser = argparse.ArgumentParser(description="SFT training for tool-plan generation.")
    parser.add_argument("--model-path", required=True, help="HF model path (e.g. Qwen2.5-7B-Instruct)")
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "processed" / "fm_sft"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device train batch size")
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-targets", default=None, help="Comma-separated list of LoRA target modules")
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--no-lora", action="store_true", help="Full fine-tune (no LoRA)")
    parser.add_argument("--merge-adapter", action="store_true", help="Merge LoRA weights and save to <out-dir>/merged")
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import SFTTrainer, SFTConfig

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # SFTTrainer expects right-padding during training

    # Load data
    data_dir = Path(args.data_dir)
    train_rows = read_jsonl(data_dir / "train.jsonl")
    dev_rows = read_jsonl(data_dir / "dev.jsonl")
    print(json.dumps({"train": len(train_rows), "dev": len(dev_rows)}, ensure_ascii=False))

    train_ds = to_hf_dataset(train_rows, tokenizer, args.max_length)
    dev_ds = to_hf_dataset(dev_rows, tokenizer, args.max_length)

    # Load model — no device_map so DDP / torchrun can handle placement
    torch_dtype = {"auto": "auto", "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    # LoRA
    peft_config = None
    if not args.no_lora:
        target_modules = [m.strip() for m in args.lora_targets.split(",")] if args.lora_targets else None
        peft_config = build_lora_config(args.lora_r, args.lora_alpha, args.lora_dropout, target_modules)

    # Training config
    training_args = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=(args.dtype == "bf16"),
        fp16=(args.dtype == "fp16"),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=args.seed,
        report_to="none",
        max_length=args.max_length,       # trl >= 1.0: max_seq_length renamed to max_length
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tokenizer,       # trl >= 1.0: tokenizer renamed to processing_class
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(str(out_dir / "final"))
    tokenizer.save_pretrained(str(out_dir / "final"))
    print(json.dumps({"saved": str(out_dir / "final")}, ensure_ascii=False))

    if args.merge_adapter and not args.no_lora:
        merged_dir = out_dir / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(str(merged_dir))
        tokenizer.save_pretrained(str(merged_dir))
        print(json.dumps({"merged": str(merged_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
