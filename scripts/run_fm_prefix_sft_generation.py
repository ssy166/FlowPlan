import argparse
import json
import math
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_fm_prefix_sft import (  # noqa: E402
    PrefixProjector,
    format_prompt,
    load_latents,
    load_sampled_fm_latents,
    read_jsonl,
    torch_load,
)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_prediction_hints(path):
    if not path:
        return {}
    hints = {}
    for row in read_jsonl(path):
        keys = [
            row.get("record_id"),
            (row.get("metadata") or {}).get("record_id"),
            row.get("task_id"),
        ]
        for key in keys:
            if key:
                hints[str(key)] = row
    return hints


def find_balanced_json(text, start=0):
    first = text.find("{", start)
    if first < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for idx in range(first, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[first : idx + 1]
    return None


def extract_json_object(text):
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    fenced_start = text.find("```")
    if fenced_start >= 0:
        fenced_json = find_balanced_json(text, fenced_start)
        if fenced_json:
            try:
                return json.loads(fenced_json)
            except Exception:
                pass
    for start in [0, text.find("{")]:
        if start < 0:
            continue
        candidate = find_balanced_json(text, start)
        if candidate:
            try:
                return json.loads(candidate)
            except Exception:
                pass
    return None


def parse_conditioned_state(row):
    content = ""
    for message in row.get("messages") or []:
        if message.get("role") == "user":
            content = message.get("content") or ""
            break
    starts = []
    for marker in ["Feedback-conditioned state:", "Conditioned workflow state:"]:
        marker_pos = content.find(marker)
        if marker_pos >= 0:
            brace_pos = content.find("{", marker_pos)
            if brace_pos >= 0:
                starts.append(brace_pos)
    starts.append(content.find("{"))
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


def actions_from_assistant(row):
    for message in reversed(row.get("messages") or []):
        if message.get("role") == "assistant":
            parsed = extract_json_object(message.get("content") or "")
            if isinstance(parsed, dict):
                return parsed.get("actions") or [], bool(parsed.get("stop", False))
    return [], False


def messages_with_planner_hint(row, tools, hint_mode, prediction_hints=None):
    messages = [dict(message) for message in row.get("messages") or []]
    if hint_mode == "none" or not messages:
        return messages
    gold_actions, gold_stop = actions_from_assistant(row)
    structured = hint_mode in {"gold_structured_next_tool", "prediction_structured_action_tool"}
    if hint_mode in {"gold_next_tool", "gold_structured_next_tool"}:
        if gold_stop or not gold_actions:
            hint = structured_planner_hint(row, None, {}, None, include_stop=True) if structured else 'Planning prior: the next decision should be {"stop": true, "actions": []}.'
        else:
            action = gold_actions[0]
            tool_id = action.get("tool_id")
            tool = tools.get(tool_id) or {}
            tool_name = tool.get("name") or action.get("tool_name") or (tool_id.rsplit("::", 1)[-1] if tool_id else "")
            hint = structured_planner_hint(row, tool_id, tool, None) if structured else f"Planning prior: the next tool should be {tool_id} ({tool_name})."
    elif hint_mode in {"prediction_next_tool", "prediction_action_tool", "prediction_structured_action_tool"}:
        metadata = row.get("metadata") or {}
        pred = (prediction_hints or {}).get(str(row.get("id"))) or (prediction_hints or {}).get(str(metadata.get("record_id"))) or {}
        tool_ids = pred.get("tool_ids") or []
        if not tool_ids:
            if hint_mode in {"prediction_action_tool", "prediction_structured_action_tool"}:
                return messages
            hint = 'Planning prior: the next decision should be {"stop": true, "actions": []}.'
        else:
            tool_id = tool_ids[0]
            tool = tools.get(tool_id) or {}
            tool_name = tool.get("name") or tool_id.rsplit("::", 1)[-1]
            hint = structured_planner_hint(row, tool_id, tool, pred) if structured else f"Planning prior: the next tool should be {tool_id} ({tool_name})."
    else:
        raise ValueError(f"Unknown planner hint mode: {hint_mode}")
    for message in reversed(messages):
        if message.get("role") == "user":
            message["content"] = (message.get("content") or "") + "\n\n" + hint
            return messages
    messages.append({"role": "user", "content": hint})
    return messages


def tool_parameters(tool):
    schema = tool.get("schema") or {}
    params = schema.get("parameters") or schema.get("required_parameters") or []
    if isinstance(params, dict):
        return sorted(params)
    if isinstance(params, list):
        out = []
        for item in params:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict) and item.get("name"):
                out.append(item["name"])
        return out
    return []


def structured_planner_hint(row, tool_id, tool, pred=None, include_stop=False):
    state = parse_conditioned_state(row)
    metadata = row.get("metadata") or {}
    if not tool_id:
        if not include_stop:
            return None
        payload = {
            "planner_prior": {
                "decision": "stop",
                "actions": [],
                "source": "oracle" if pred is None else "predicted",
            }
        }
    else:
        tool_name = tool.get("name") or tool_id.rsplit("::", 1)[-1]
        executed_prefix = state.get("executed_prefix") or []
        last_execution = executed_prefix[-1] if executed_prefix else {}
        confidence = None
        if isinstance(pred, dict):
            confidence = (pred.get("metadata") or {}).get("confidence")
        payload = {
            "planner_prior": {
                "decision": "action",
                "tool_id": tool_id,
                "tool_name": tool_name,
                "phase": tool.get("phase") or "unknown",
                "required_arguments": tool_parameters(tool),
                "confidence": confidence,
                "domain": metadata.get("domain") or state.get("domain"),
                "replan_step_idx": metadata.get("replan_step_idx") or state.get("replan_step_idx"),
                "replan_reason": state.get("replan_reason") or metadata.get("replan_reason"),
                "last_executed_tool": (
                    (last_execution.get("predicted_action") or {}).get("tool_id")
                    or (last_execution.get("execution") or {}).get("tool_name")
                    or last_execution.get("tool_name")
                ),
            }
        }
    return "Planning prior:\n" + json.dumps(payload, ensure_ascii=False, sort_keys=True)


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def normalize_value(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def argument_scores(pred_actions, gold_actions):
    gold_key_count = 0
    pred_key_count = 0
    key_hits = 0
    value_hits = 0
    comparable = 0
    for idx, gold_action in enumerate(gold_actions):
        pred_action = pred_actions[idx] if idx < len(pred_actions) else {}
        gold_args = gold_action.get("arguments") or {}
        pred_args = pred_action.get("arguments") or {}
        if not isinstance(gold_args, dict):
            gold_args = {}
        if not isinstance(pred_args, dict):
            pred_args = {}
        gold_keys = set(gold_args)
        pred_keys = set(pred_args)
        gold_key_count += len(gold_keys)
        pred_key_count += len(pred_keys)
        matched = gold_keys & pred_keys
        key_hits += len(matched)
        comparable += len(gold_keys)
        for key in matched:
            if normalize_value(pred_args.get(key)) == normalize_value(gold_args.get(key)):
                value_hits += 1
    return {
        "gold_argument_count": gold_key_count,
        "predicted_argument_count": pred_key_count,
        "argument_key_hits": key_hits,
        "argument_value_hits": value_hits,
        "argument_key_recall": key_hits / gold_key_count if gold_key_count else None,
        "argument_key_precision": key_hits / pred_key_count if pred_key_count else None,
        "argument_value_exact_match": value_hits / comparable if comparable else None,
    }


def available_tool_maps(row, tools):
    state = parse_conditioned_state(row)
    available = state.get("available_tools") or []
    by_name = {}
    available_ids = []
    for item in available:
        tool_id = item.get("tool_id")
        name = item.get("name") or (tool_id.rsplit("::", 1)[-1] if tool_id else None)
        if not tool_id:
            continue
        available_ids.append(tool_id)
        by_name.setdefault(tool_id, tool_id)
        if name:
            by_name.setdefault(name, tool_id)
    if not available_ids:
        metadata = row.get("metadata") or {}
        domain = metadata.get("domain")
        source = metadata.get("source")
        for tool_id, tool in tools.items():
            if source and not tool_id.startswith(f"{source}::"):
                continue
            if domain and f"::{domain}::" not in tool_id:
                continue
            available_ids.append(tool_id)
            by_name.setdefault(tool_id, tool_id)
            by_name.setdefault(tool.get("name") or tool_id.rsplit("::", 1)[-1], tool_id)
    return available_ids, by_name


def normalize_actions(parsed, row, tools):
    available_ids, by_name = available_tool_maps(row, tools)
    available_set = set(available_ids)
    if isinstance(parsed, dict):
        raw_actions = parsed.get("actions") or parsed.get("tool_calls") or []
        stop = bool(parsed.get("stop", False))
    elif isinstance(parsed, list):
        raw_actions = parsed
        stop = False
    else:
        raw_actions = []
        stop = False
    actions = []
    invalid = []
    if isinstance(raw_actions, list):
        for raw in raw_actions:
            if not isinstance(raw, dict):
                invalid.append(str(raw))
                continue
            name = raw.get("tool_id") or raw.get("tool_name") or raw.get("name") or raw.get("tool")
            tool_id = by_name.get(name) or (name if name in available_set else None)
            if not tool_id:
                invalid.append(str(name))
                continue
            args = raw.get("arguments")
            if args is None:
                args = raw.get("args") or {}
            if not isinstance(args, dict):
                args = {}
            tool = tools.get(tool_id) or {}
            actions.append(
                {
                    "tool_id": tool_id,
                    "tool_name": tool.get("name") or tool_id.rsplit("::", 1)[-1],
                    "arguments": args,
                }
            )
    return actions, stop, invalid


def summarize_rows(rows):
    n = len(rows)
    if not n:
        return {"count": 0}
    arg_recall = [row["argument_key_recall"] for row in rows if row.get("argument_key_recall") is not None]
    arg_precision = [row["argument_key_precision"] for row in rows if row.get("argument_key_precision") is not None]
    arg_value = [row["argument_value_exact_match"] for row in rows if row.get("argument_value_exact_match") is not None]
    return {
        "count": n,
        "json_parse_rate": sum(row["json_parse_ok"] for row in rows) / n,
        "schema_action_rate": sum(row["schema_action_ok"] for row in rows) / n,
        "valid_available_tool_rate": sum(row["available_tool_ok"] for row in rows) / n,
        "stop_exact_match": sum(row["stop_exact_match"] for row in rows) / n,
        "tool_exact_match": sum(row["tool_exact_match"] for row in rows) / n,
        "avg_tool_edit_distance": sum(row["tool_edit_distance"] for row in rows) / n,
        "avg_predicted_tool_count": sum(row["predicted_tool_count"] for row in rows) / n,
        "avg_gold_tool_count": sum(row["gold_tool_count"] for row in rows) / n,
        "argument_task_count": len(arg_recall),
        "avg_argument_key_recall": sum(arg_recall) / len(arg_recall) if arg_recall else None,
        "avg_argument_key_precision": sum(arg_precision) / len(arg_precision) if arg_precision else None,
        "avg_argument_value_exact_match": sum(arg_value) / len(arg_value) if arg_value else None,
    }


def summarize_groups(rows, key):
    groups = {}
    for row in rows:
        groups.setdefault(str(row.get(key)), []).append(row)
    return {group: summarize_rows(items) for group, items in sorted(groups.items())}


def load_projector(path, model, device, dtype):
    payload = torch_load(path, map_location="cpu")
    config = payload.get("config") or {}
    hidden_dim = int(model.get_input_embeddings().embedding_dim)
    projector = PrefixProjector(
        int(config.get("latent_dim")),
        hidden_dim,
        int(config.get("prefix_len", 8)),
        int(config.get("projector_hidden_dim", 512)),
        0.0,
        float(config.get("gate_init", -4.0)),
    )
    projector.load_state_dict(payload["projector"])
    # Keep the projector in fp32 as trained. Its output is cast to the LLM
    # embedding dtype at the call site.
    projector.to(device=device)
    projector.eval()
    return projector, config


@torch.no_grad()
def generate_one(model, tokenizer, prompt, device, max_new_tokens, projector=None, latent=None):
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    if projector is None:
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        new_ids = out[0, input_ids.shape[1] :]
    else:
        token_embeds = model.get_input_embeddings()(input_ids)
        prefix = projector(latent.to(device=device).float().unsqueeze(0)).to(token_embeds.dtype)
        inputs_embeds = torch.cat([prefix, token_embeds], dim=1)
        prefix_mask = torch.ones((1, prefix.shape[1]), dtype=attention_mask.dtype, device=device)
        full_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        out = model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        new_ids = out[0]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def pick_dtype(name):
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def main():
    parser = argparse.ArgumentParser(description="Run generation-time evaluation for FM-prefix conditioned SFT.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--data", default=str(ROOT / "data" / "processed" / "fm_conditioned_sft_smoke" / "dev.jsonl"))
    parser.add_argument("--tensor", default=str(ROOT / "data" / "processed" / "fm_tensor_encoder_full" / "dev.pt"))
    parser.add_argument("--tools", default=str(ROOT / "data" / "processed" / "tools.jsonl"))
    parser.add_argument("--projector-path", default=None)
    parser.add_argument("--projector-gate-logit", type=float, default=None, help="Override the trained prefix gate logit at generation time.")
    parser.add_argument("--latent-key", default="y_i", choices=["c_i", "y_i", "proxy_c_i", "y_hat"])
    parser.add_argument("--fm-checkpoint", default=None)
    parser.add_argument("--fm-inference-steps", type=int, default=16)
    parser.add_argument("--fm-noise-std", type=float, default=1.0)
    parser.add_argument("--fm-sample-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-prefix", action="store_true")
    parser.add_argument(
        "--planner-hint",
        choices=[
            "none",
            "gold_next_tool",
            "gold_structured_next_tool",
            "prediction_next_tool",
            "prediction_action_tool",
            "prediction_structured_action_tool",
        ],
        default="none",
    )
    parser.add_argument("--planner-hint-predictions", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-raw", required=True)
    parser.add_argument("--out-pred", required=True)
    parser.add_argument("--out-details", default=None)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--model-name", default="fm_prefix_sft")
    args = parser.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = pick_dtype(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model = PeftModel.from_pretrained(base, args.adapter_dir).to(device)
    model.eval()

    rows = read_jsonl(args.data)
    if args.limit:
        rows = rows[: args.limit]
    tools = {row["tool_id"]: row for row in read_jsonl(args.tools)}
    prediction_hints = read_prediction_hints(args.planner_hint_predictions)

    projector = None
    projector_config = {}
    latents = {}
    if not args.no_prefix:
        projector_path = args.projector_path or str(Path(args.adapter_dir) / "fm_prefix_projector.pt")
        projector, projector_config = load_projector(projector_path, model, device, dtype)
        if args.projector_gate_logit is not None:
            projector.gate_logit.data.fill_(float(args.projector_gate_logit))
        if args.latent_key == "y_hat":
            fm_checkpoint = args.fm_checkpoint or projector_config.get("fm_checkpoint")
            if not fm_checkpoint:
                raise ValueError("--fm-checkpoint is required for --latent-key y_hat")
            latents = load_sampled_fm_latents(
                args.tensor,
                fm_checkpoint,
                device,
                args.fm_sample_batch_size,
                args.fm_inference_steps,
                args.fm_noise_std,
                args.seed,
            )
        else:
            latents = load_latents(args.tensor, args.latent_key)

    raw_rows = []
    pred_rows = []
    detail_rows = []
    for idx, row in enumerate(rows, start=1):
        row_id = row.get("id")
        if not args.no_prefix and row_id not in latents:
            continue
        prompt_messages = messages_with_planner_hint(row, tools, args.planner_hint, prediction_hints)
        prompt = format_prompt(tokenizer, prompt_messages)
        generation = generate_one(
            model,
            tokenizer,
            prompt,
            device,
            args.max_new_tokens,
            projector=projector,
            latent=None if args.no_prefix else latents[row_id],
        )
        parsed = extract_json_object(generation)
        actions, stop, invalid_tools = normalize_actions(parsed, row, tools)
        gold_actions, gold_stop = actions_from_assistant(row)
        pred_tool_ids = [action["tool_id"] for action in actions]
        gold_tool_ids = [action.get("tool_id") for action in gold_actions]
        arg = argument_scores(actions, gold_actions)
        metadata = row.get("metadata") or {}
        detail = {
            "id": row_id,
            "task_id": metadata.get("task_id") or row_id,
            "source": metadata.get("source"),
            "domain": metadata.get("domain"),
            "step_idx": metadata.get("step_idx"),
            "json_parse_ok": isinstance(parsed, (dict, list)),
            "schema_action_ok": isinstance(parsed, dict) and isinstance(parsed.get("actions"), list),
            "available_tool_ok": not invalid_tools,
            "invalid_tools": invalid_tools,
            "stop_exact_match": stop == gold_stop,
            "tool_exact_match": pred_tool_ids == gold_tool_ids,
            "tool_edit_distance": edit_distance(pred_tool_ids, gold_tool_ids),
            "predicted_tool_count": len(pred_tool_ids),
            "gold_tool_count": len(gold_tool_ids),
            **arg,
        }
        raw_rows.append(
            {
                "id": row_id,
                "task_id": detail["task_id"],
                "generation": generation,
                "parsed": parsed,
                "gold": {"stop": gold_stop, "actions": gold_actions},
                "metadata": metadata,
            }
        )
        pred_rows.append(
            {
                "task_id": detail["task_id"],
                "record_id": row_id,
                "source": metadata.get("source"),
                "domain": metadata.get("domain"),
                "plan_type": f"model_{args.model_name}",
                "tool_ids": pred_tool_ids,
                "tool_names": [action["tool_name"] for action in actions],
                "phase_names": [tools.get(action["tool_id"], {}).get("phase", "unknown") for action in actions],
                "actions": actions,
                "metadata": {"record_id": row_id, "step_idx": metadata.get("step_idx")},
            }
        )
        detail_rows.append(detail)
        if idx % 8 == 0 or idx == len(rows):
            print(json.dumps({"generated": idx, "rows": len(rows)}, ensure_ascii=False))

    summary = {
        "model_name": args.model_name,
        "adapter_dir": args.adapter_dir,
        "projector_path": None if args.no_prefix else (args.projector_path or str(Path(args.adapter_dir) / "fm_prefix_projector.pt")),
        "latent_key": None if args.no_prefix else args.latent_key,
        "planner_hint": args.planner_hint,
        "planner_hint_predictions": args.planner_hint_predictions,
        "projector_gate_logit": None if args.no_prefix else float(projector.gate_logit.detach().cpu()),
        "projector_gate": None if args.no_prefix else float(torch.sigmoid(projector.gate_logit).detach().cpu()),
        "limit": args.limit,
        "overall": summarize_rows(detail_rows),
        "by_source": summarize_groups(detail_rows, "source"),
        "by_domain": summarize_groups(detail_rows, "domain"),
    }
    write_jsonl(args.out_raw, raw_rows)
    write_jsonl(args.out_pred, pred_rows)
    if args.out_details:
        write_jsonl(args.out_details, detail_rows)
    write_json(args.summary_out, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
