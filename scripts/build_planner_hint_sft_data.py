import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def stable_float(text):
    digest = hashlib.blake2b(str(text).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(2**64 - 1)


def load_predictions(path):
    if not path or not Path(path).exists():
        return {}
    out = {}
    for row in read_jsonl(path):
        for key in [row.get("record_id"), (row.get("metadata") or {}).get("record_id"), row.get("task_id")]:
            if key:
                out[str(key)] = row
    return out


def find_balanced_json(text, start):
    if start is None or start < 0:
        return None
    first = text.find("{", start)
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


def assistant_target(row):
    for message in reversed(row.get("messages") or []):
        if message.get("role") == "assistant":
            try:
                return json.loads(message.get("content") or "{}")
            except Exception:
                return {}
    return {}


def oracle_tool_id(row):
    target = assistant_target(row)
    actions = target.get("actions") or []
    if target.get("stop") or not actions:
        return None
    return actions[0].get("tool_id")


def predicted_tool_id(pred):
    if not pred:
        return None
    tool_ids = pred.get("tool_ids") or []
    if tool_ids:
        return tool_ids[0]
    actions = pred.get("actions") or []
    if actions:
        return actions[0].get("tool_id")
    return None


def prediction_confidence(pred):
    if not isinstance(pred, dict):
        return None
    confidence = (pred.get("metadata") or {}).get("confidence")
    if confidence is None:
        return None
    try:
        return float(confidence)
    except (TypeError, ValueError):
        return None


def hint_from_tool(tool_id, tools, include_stop=False):
    if not tool_id:
        if not include_stop:
            return None
        return 'Planning prior: the next decision should be {"stop": true, "actions": []}.'
    tool = tools.get(tool_id) or {}
    name = tool.get("name") or tool_id.rsplit("::", 1)[-1]
    return f"Planning prior: the next tool should be {tool_id} ({name})."


def noisy_hint_from_tool(tool_id, tools, include_stop=False):
    if not tool_id:
        if not include_stop:
            return None
        return 'Noisy planning prior: suggested decision is {"stop": true, "actions": []}. Use it only if it is consistent with the compact state.'
    tool = tools.get(tool_id) or {}
    name = tool.get("name") or tool_id.rsplit("::", 1)[-1]
    return f"Noisy planning prior: suggested next tool is {tool_id} ({name}). Use it only if it is consistent with the compact state and available_tools."


def compact_result_counts(result):
    if not isinstance(result, dict):
        return {}
    return {
        "status": result.get("status"),
        "has_order_id": bool(result.get("order_id")),
        "item_count": len(result.get("item_ids") or []),
        "product_count": len(result.get("product_ids") or []),
        "payment_count": len(result.get("payment_method_ids") or []),
    }


def retail_evidence_hint_from_tool(row, tool_id, tools, pred=None, include_stop=False):
    state = parse_conditioned_state(row)
    domain = (row.get("metadata") or {}).get("domain") or state.get("domain")
    if domain != "retail":
        return structured_hint_from_tool(row, tool_id, tools, pred=pred, include_stop=include_stop)

    progress = state.get("grounding_progress") or {}
    prefix = state.get("executed_prefix") or []
    last = prefix[-1] if prefix else {}
    result_counts = compact_result_counts(last.get("result_summary"))
    if tool_id:
        tool = tools.get(tool_id) or {}
        prior = {
            "decision": "action",
            "tool_id": tool_id,
            "tool_name": tool.get("name") or tool_id.rsplit("::", 1)[-1],
            "required_arguments": tool_parameters(tool),
            "confidence": prediction_confidence(pred),
            "source": "oracle" if pred is None else "predicted",
        }
    elif include_stop:
        prior = {"decision": "stop", "tool_id": None, "tool_name": None, "required_arguments": [], "confidence": prediction_confidence(pred), "source": "oracle" if pred is None else "predicted"}
    else:
        return None

    evidence = {
        "lookup_completed": progress.get("lookup_completed"),
        "order_grounded": progress.get("order_grounded"),
        "last_successful_lookup_tool": progress.get("last_successful_lookup_tool"),
        "resolved_order_status": progress.get("resolved_order_status"),
        "resolved_order_count": len(progress.get("resolved_order_ids") or []),
        "resolved_item_count": len(progress.get("resolved_order_item_ids") or []),
        "resolved_payment_count": len(progress.get("resolved_payment_method_ids") or []),
        "operation_eligibility": progress.get("operation_eligibility") or [],
        "last_tool": last.get("tool_name") or (last.get("action") or {}).get("tool_name"),
        "last_result": result_counts,
    }
    payload = {"planner_prior": prior, "retail_decision_evidence": evidence}
    return "Planning prior and retail decision evidence:\n" + json.dumps(payload, ensure_ascii=False, sort_keys=True)


def retail_evidence_simple_hint_from_tool(row, tool_id, tools, pred=None, include_stop=False):
    base = hint_from_tool(tool_id, tools, include_stop=include_stop)
    if not base:
        return None
    state = parse_conditioned_state(row)
    domain = (row.get("metadata") or {}).get("domain") or state.get("domain")
    if domain != "retail":
        return base
    progress = state.get("grounding_progress") or {}
    prefix = state.get("executed_prefix") or []
    last = prefix[-1] if prefix else {}
    evidence = {
        "lookup_completed": progress.get("lookup_completed"),
        "order_grounded": progress.get("order_grounded"),
        "last_successful_lookup_tool": progress.get("last_successful_lookup_tool"),
        "resolved_order_status": progress.get("resolved_order_status"),
        "resolved_order_count": len(progress.get("resolved_order_ids") or []),
        "resolved_item_count": len(progress.get("resolved_order_item_ids") or []),
        "operation_eligibility": progress.get("operation_eligibility") or [],
        "last_tool": last.get("tool_name") or (last.get("action") or {}).get("tool_name"),
        "last_result": compact_result_counts(last.get("result_summary")),
    }
    return base + "\nRetail decision evidence:\n" + json.dumps(evidence, ensure_ascii=False, sort_keys=True)


def tool_parameters(tool):
    schema = tool.get("schema") or {}
    params = schema.get("parameters") or schema.get("required_parameters") or tool.get("parameters") or []
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


def structured_hint_from_tool(row, tool_id, tools, pred=None, include_stop=False):
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
        return "Planning prior:\n" + json.dumps(payload, ensure_ascii=False, sort_keys=True)

    tool = tools.get(tool_id) or {}
    state = parse_conditioned_state(row)
    metadata = row.get("metadata") or {}
    executed_prefix = state.get("executed_prefix") or []
    last_execution = executed_prefix[-1] if executed_prefix else {}
    confidence = None
    if isinstance(pred, dict):
        confidence = (pred.get("metadata") or {}).get("confidence")
    payload = {
        "planner_prior": {
            "decision": "action",
            "tool_id": tool_id,
            "tool_name": tool.get("name") or tool_id.rsplit("::", 1)[-1],
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
            "source": "oracle" if pred is None else "predicted",
        }
    }
    return "Planning prior:\n" + json.dumps(payload, ensure_ascii=False, sort_keys=True)


def candidate_hint_from_predictions(row, prediction_sets, tools, args, use_oracle=False):
    candidates = {}
    if use_oracle:
        oracle_id = oracle_tool_id(row)
        if oracle_id or args.include_stop_hint:
            candidates[oracle_id] = {"tool_id": oracle_id, "votes": 1, "confidence": 1.0, "source": "oracle"}

    for pred in prediction_sets:
        tool_id = predicted_tool_id(pred)
        if not tool_id and not args.include_stop_hint:
            continue
        item = candidates.setdefault(tool_id, {"tool_id": tool_id, "votes": 0, "confidence": None, "source": "predicted"})
        item["votes"] += 1
        confidence = prediction_confidence(pred)
        if confidence is not None:
            if item["confidence"] is None or confidence > item["confidence"]:
                item["confidence"] = confidence

    if not candidates:
        return None

    ranked = sorted(
        candidates.values(),
        key=lambda item: (
            item.get("source") != "oracle",
            -item["votes"],
            -(item["confidence"] if item["confidence"] is not None else -1.0),
            str(item["tool_id"]),
        ),
    )[: args.max_candidates]
    payload = []
    for item in ranked:
        tool_id = item["tool_id"]
        if not tool_id:
            payload.append(
                {
                    "decision": "stop",
                    "tool_id": None,
                    "tool_name": None,
                    "votes": item["votes"],
                    "confidence": item["confidence"],
                    "source": item["source"],
                }
            )
            continue
        tool = tools.get(tool_id) or {}
        payload.append(
            {
                "decision": "action",
                "tool_id": tool_id,
                "tool_name": tool.get("name") or tool_id.rsplit("::", 1)[-1],
                "phase": tool.get("phase") or "unknown",
                "required_arguments": tool_parameters(tool),
                "votes": item["votes"],
                "confidence": item["confidence"],
                "source": item["source"],
            }
        )
    return "Planning prior candidates:\n" + json.dumps(
        {"planner_prior_candidates": payload},
        ensure_ascii=False,
        sort_keys=True,
    )


def make_hint(row, tool_id, tools, args, pred=None):
    if args.hint_format == "structured":
        return structured_hint_from_tool(row, tool_id, tools, pred=pred, include_stop=args.include_stop_hint)
    if args.hint_format == "noisy":
        return noisy_hint_from_tool(tool_id, tools, include_stop=args.include_stop_hint)
    if args.hint_format == "retail_evidence":
        return retail_evidence_hint_from_tool(row, tool_id, tools, pred=pred, include_stop=args.include_stop_hint)
    if args.hint_format == "retail_evidence_simple":
        return retail_evidence_simple_hint_from_tool(row, tool_id, tools, pred=pred, include_stop=args.include_stop_hint)
    return hint_from_tool(tool_id, tools, include_stop=args.include_stop_hint)


def apply_hint(row, hint):
    row = json.loads(json.dumps(row, ensure_ascii=False))
    if not hint:
        return row
    for message in reversed(row.get("messages") or []):
        if message.get("role") == "user":
            message["content"] = (message.get("content") or "") + "\n\n" + hint
            return row
    row.setdefault("messages", []).insert(-1, {"role": "user", "content": hint})
    return row


def build_split(rows, prediction_maps, tools, args, split):
    out = []
    hinted = 0
    for row in rows:
        row_id = row.get("id") or (row.get("metadata") or {}).get("record_id")
        if split == "train" and args.hint_dropout and stable_float(row_id) < args.hint_dropout:
            out.append(apply_hint(row, None))
            continue
        prediction_sets = [predictions.get(str(row_id), {}) for predictions in prediction_maps]
        pred = prediction_sets[0] if prediction_sets else {}
        pred_for_hint = pred
        use_oracle = False
        if args.hint_source == "oracle":
            tool_id = oracle_tool_id(row)
            pred_for_hint = None
            use_oracle = True
        elif args.hint_source == "predicted":
            tool_id = predicted_tool_id(pred)
        elif args.hint_source == "mixed":
            use_oracle = split == "train" and stable_float(f"{row_id}::oracle") < args.oracle_mix_rate
            tool_id = oracle_tool_id(row) if use_oracle else predicted_tool_id(pred)
            pred_for_hint = None if use_oracle else pred
        else:
            raise ValueError(args.hint_source)
        if args.hint_format == "candidates":
            hint = candidate_hint_from_predictions(row, prediction_sets, tools, args, use_oracle=use_oracle)
        else:
            hint = make_hint(row, tool_id, tools, args, pred=pred_for_hint)
        if hint:
            hinted += 1
        out.append(apply_hint(row, hint))
    return out, hinted


def main():
    parser = argparse.ArgumentParser(description="Append planner prior hints to replan SFT data.")
    parser.add_argument("--input-dir", default=str(ROOT / "data" / "processed" / "fm_replan_sft_next"))
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--extra-pred-dir", action="append", default=[])
    parser.add_argument("--tools", default=str(ROOT / "data" / "processed" / "tools.jsonl"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--hint-source", choices=["predicted", "oracle", "mixed"], default="predicted")
    parser.add_argument(
        "--hint-format",
        choices=["simple", "structured", "candidates", "noisy", "retail_evidence", "retail_evidence_simple"],
        default="simple",
    )
    parser.add_argument("--oracle-mix-rate", type=float, default=0.0)
    parser.add_argument("--hint-dropout", type=float, default=0.2)
    parser.add_argument("--include-stop-hint", action="store_true")
    parser.add_argument("--max-candidates", type=int, default=4)
    args = parser.parse_args()

    tools = {row["tool_id"]: row for row in read_jsonl(args.tools)}
    out_dir = Path(args.out_dir)
    pred_dirs = [args.pred_dir] + list(args.extra_pred_dir or [])
    manifest = {
        "hint_source": args.hint_source,
        "hint_format": args.hint_format,
        "oracle_mix_rate": args.oracle_mix_rate,
        "hint_dropout": args.hint_dropout,
        "include_stop_hint": args.include_stop_hint,
        "max_candidates": args.max_candidates,
        "pred_dirs": pred_dirs,
        "splits": {},
    }
    for split in ["train", "dev", "test"]:
        rows = read_jsonl(Path(args.input_dir) / f"{split}.jsonl")
        prediction_maps = [load_predictions(Path(pred_dir) / f"{split}.pred.jsonl") for pred_dir in pred_dirs]
        built, hinted = build_split(rows, prediction_maps, tools, args, split)
        write_jsonl(out_dir / f"{split}.jsonl", built)
        manifest["splits"][split] = {
            "rows": len(built),
            "hinted": hinted,
            "predictions": [len(predictions) for predictions in prediction_maps],
        }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
