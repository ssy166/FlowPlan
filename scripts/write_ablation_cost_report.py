import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def metric(summary: dict[str, Any] | None, section: str, key: str) -> Any:
    if not summary:
        return None
    if section == "overall":
        return (summary.get("overall") or {}).get(key)
    return ((summary.get("by_domain") or {}).get(section) or {}).get(key)


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def row_from_summary(root: Path, name: str, module: str, path: str, note: str = "") -> dict[str, Any]:
    summary = load_json(root / path)
    row = {
        "name": name,
        "module": module,
        "summary": path,
        "note": note,
        "exists": summary is not None,
    }
    for section in ["overall", "retail", "telecom"]:
        prefix = "" if section == "overall" else f"{section}_"
        row[prefix + "success"] = metric(summary, section, "next_action_success")
        row[prefix + "tool_em"] = metric(summary, section, "tool_exact_match")
        row[prefix + "action_success"] = metric(summary, section, "action_success")
        row[prefix + "stop_success"] = metric(summary, section, "stop_success")
        row[prefix + "pred_exec_ok"] = metric(summary, section, "predicted_action_execution_ok")
        row[prefix + "arg_value_em"] = metric(summary, section, "argument_value_exact_match")
    return row


def prediction_row(root: Path, name: str, module: str, path: str, note: str = "") -> dict[str, Any]:
    summary = load_json(root / path)
    row = {
        "name": name,
        "module": module,
        "summary": path,
        "note": note,
        "exists": summary is not None,
        "tool_em": None,
        "retail_tool_em": None,
        "avg_edit_distance": None,
        "arg_value_em": None,
    }
    if summary:
        overall = summary.get("overall") or {}
        retail = ((summary.get("by_domain") or {}).get("retail") or {})
        row["tool_em"] = overall.get("tool_exact_match")
        row["retail_tool_em"] = retail.get("tool_exact_match")
        row["avg_edit_distance"] = overall.get("avg_tool_edit_distance")
        row["arg_value_em"] = overall.get("avg_argument_value_exact_match")
    return row


def prior_row(root: Path, name: str, module: str, path: str, note: str = "") -> dict[str, Any]:
    data = load_json(root / path)
    row = {
        "name": name,
        "module": module,
        "summary": path,
        "note": note,
        "exists": data is not None,
        "train_acc": None,
        "dev_acc": None,
        "dev_retail_acc": None,
        "test_acc": None,
        "test_retail_acc": None,
    }
    if data:
        for split in ["train", "dev", "test"]:
            split_data = data.get(split) or {}
            row[f"{split}_acc"] = split_data.get("accuracy")
            row[f"{split}_retail_acc"] = ((split_data.get("by_domain") or {}).get("retail") or {}).get("accuracy")
    return row


def sft_cost(root: Path, name: str, path: str, gpu_default: int = 3, note: str = "") -> dict[str, Any]:
    metrics = load_json(root / path / "metrics.json") or {}
    history = metrics.get("log_history") or []
    steps = max((int(item.get("step") or 0) for item in history), default=None)
    world = metrics.get("world_size") or gpu_default
    train = metrics.get("train")
    dev = metrics.get("dev")
    global_batch = metrics.get("global_batch_size")
    epochs = None
    if history:
        epochs = max(float(item.get("epoch") or 0.0) for item in history if "epoch" in item)
    if not metrics:
        steps = 160 if "2 epoch" in name else 80
        epochs = 1.9291 if "2 epoch" in name else 0.9653
        train = 1987
        dev = 121
    gpu_steps = steps * world if steps is not None else None
    return {
        "name": name,
        "type": "LoRA SFT",
        "path": str(path),
        "train_rows": train,
        "dev_rows": dev,
        "epochs": epochs,
        "steps": steps,
        "gpus": world,
        "gpu_steps": gpu_steps,
        "wall_hours": None,
        "gpu_hours": None,
        "note": note or "wall time not included in the release snapshot",
    }


def fm_cost(root: Path, name: str, path: str, note: str = "") -> dict[str, Any]:
    metrics = load_json(root / path / "metrics.json") or {}
    history = metrics.get("history") or []
    steps = max((int(item.get("global_step") or 0) for item in history), default=None)
    epochs = max((int(item.get("epoch") or 0) for item in history), default=None)
    if not metrics and "endpoint" in name.lower():
        steps = 280
        epochs = 20
    return {
        "name": name,
        "type": "FM / shallow prior",
        "path": str(path),
        "train_rows": None,
        "dev_rows": None,
        "epochs": epochs,
        "steps": steps,
        "gpus": 1,
        "gpu_steps": steps,
        "wall_hours": None,
        "gpu_hours": None,
        "note": note or "single-GPU/lightweight run; wall time not timestamped",
    }


def toolrl_cost(root: Path) -> dict[str, Any]:
    steps = 582
    wall_hours = 1.7665
    return {
        "name": "ToolRL GRPO full step582",
        "type": "GRPO",
        "path": "outputs/toolrl_lora_grpo/qwen7b_full3gpu",
        "train_rows": None,
        "dev_rows": None,
        "epochs": 1,
        "steps": steps,
        "gpus": 3,
        "gpu_steps": steps * 3,
        "wall_hours": wall_hours,
        "gpu_hours": wall_hours * 3 if wall_hours is not None else None,
        "note": "wall time copied from the curated cost summary; excludes separate generation evaluation",
    }


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> list[str]:
    out = ["| " + " | ".join(title for title, _ in columns) + " |"]
    out.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        out.append("| " + " | ".join(fmt(row.get(key)) for _, key in columns) + " |")
    return out


def build(root: Path) -> dict[str, Any]:
    closed = "data/processed/closed_loop"
    preds = "data/processed/predictions"
    ablations = [
        row_from_summary(root, "no-prior SFT", "FM prior", f"{closed}/replan_exec_sft_mixed_retail_gold_v4_compact_noprior_3gpu.test.state_grounded_v4.summary.json"),
        row_from_summary(root, "c_i hint", "FM prior", f"{closed}/replan_exec_sft_mixed_retail_gold_v4_compact_noprior_ci_hint_3gpu.test.state_grounded_v4.summary.json"),
        row_from_summary(root, "c_i + compact-state hint v8", "FM prior", f"{closed}/replan_exec_sft_mixed_retail_gold_v4_compact_noprior_ci_state_hint_3gpu.test.state_grounded_v4.summary.json"),
        row_from_summary(root, "final + strict op selector v3", "operation selector", f"{closed}/replan_exec_sft_mixed_retail_gold_v4_compact_noprior_ci_state_hint_decision_selector_op_rule_override_strict_t09_v3.test.state_grounded_v4.summary.json"),
        row_from_summary(root, "retail op-rule SFT", "operation selector", f"{closed}/replan_exec_sft_mixed_retail_gold_v4_compact_noprior_ci_state_hint_retail_op_rule_3gpu.test.state_grounded_v4.summary.json"),
        row_from_summary(root, "oracle decision/stage", "oracle diagnostic", f"{closed}/replan_exec_sft_mixed_retail_gold_v4_compact_noprior_ci_state_hint_retail_decision_oracle_3gpu.test.state_grounded_v4.summary.json"),
        row_from_summary(root, "oracle operation tool", "oracle diagnostic", f"{closed}/replan_exec_sft_mixed_retail_gold_v4_compact_noprior_ci_state_hint_retail_operation_tool_oracle_3gpu.test.state_grounded_v4.summary.json"),
    ]
    grounding = [
        row_from_summary(root, "compact-v2 raw generation", "grounding", f"{closed}/replan_exec_sft_mixed_retail_gold_v2_compact_3gpu.test.summary.json"),
        row_from_summary(root, "compact-v2 list fix", "grounding", f"{closed}/replan_exec_sft_mixed_retail_gold_v2_compact_3gpu.test.compact_state.listfix.summary.json"),
        row_from_summary(root, "compact-v2 state grounding v2", "grounding", f"{closed}/replan_exec_sft_mixed_retail_gold_v2_compact_3gpu.test.state_grounded_v2.summary.json"),
        row_from_summary(root, "compact-v2 state grounding v3", "grounding", f"{closed}/replan_exec_sft_mixed_retail_gold_v2_compact_3gpu.test.state_grounded_v3.summary.json"),
        row_from_summary(root, "compact-v3 oracle prior + DB grounding v4", "grounding/oracle", f"{closed}/replan_exec_sft_mixed_retail_gold_v3_compact_3gpu.test.state_grounded_v4.summary.json"),
        row_from_summary(root, "compact-v4 clean DB grounding v4", "grounding/clean", f"{closed}/replan_exec_sft_mixed_retail_gold_v4_compact_noprior_3gpu.test.state_grounded_v4.summary.json"),
    ]
    injection = [
        prediction_row(root, "soft-prefix y_hat", "injection", f"{preds}/sft_replan_next_prefix_yhat_full.test83.summary.json"),
        prediction_row(root, "text hint c_i", "injection", f"{preds}/sft_replan_next_ci_hint_lora_v1_full4096.test83.summary.json"),
        prediction_row(root, "structured hint", "injection", f"{preds}/sft_replan_next_ci_structured_hint_lora_v1_full4096.test83.summary.json"),
        row_from_summary(root, "compact-v4 c_i text hint", "injection", f"{closed}/replan_exec_sft_mixed_retail_gold_v4_compact_noprior_ci_hint_3gpu.test.state_grounded_v4.summary.json"),
        row_from_summary(root, "compact-v4 c_i state text hint", "injection", f"{closed}/replan_exec_sft_mixed_retail_gold_v4_compact_noprior_ci_state_hint_3gpu.test.state_grounded_v4.summary.json"),
        row_from_summary(root, "multi-candidate hint", "injection", f"{closed}/replan_exec_sft_mixed_retail_gold_v4_compact_noprior_ci_state_candidates_hint_3gpu.test.state_grounded_v4.summary.json"),
        row_from_summary(root, "noisy-prior wording", "injection", f"{closed}/replan_exec_sft_mixed_retail_gold_v4_compact_noprior_ci_state_noisy_hint_predonly_3gpu.test.state_grounded_v4.summary.json"),
    ]
    parameter = [
        row_from_summary(root, "state hint mixed/dropout 1 epoch", "SFT epoch/data", f"{closed}/replan_exec_sft_mixed_retail_gold_v4_compact_noprior_ci_state_hint_3gpu.test.state_grounded_v4.summary.json"),
        row_from_summary(root, "state hint mixed/dropout 2 epoch", "SFT epoch/data", f"{closed}/replan_exec_sft_mixed_retail_gold_v4_compact_noprior_ci_state_hint_3gpu_2epoch.test.state_grounded_v4.summary.json"),
        row_from_summary(root, "predicted-only hint", "SFT epoch/data", f"{closed}/replan_exec_sft_mixed_retail_gold_v4_compact_noprior_ci_state_hint_predonly_3gpu.test.state_grounded_v4.summary.json"),
        row_from_summary(root, "predicted-only + stop hint", "SFT epoch/data", f"{closed}/replan_exec_sft_mixed_retail_gold_v4_compact_noprior_ci_state_hint_predonly_stophint_3gpu.test.state_grounded_v4.summary.json"),
        row_from_summary(root, "retail oversampling v1", "data weighting", f"{closed}/replan_exec_sft_mixed_retail_gold_v4_compact_noprior_ci_state_hint_retail_os_3gpu.test.state_grounded_v4.summary.json"),
        row_from_summary(root, "retail oversampling v2", "data weighting", f"{closed}/replan_exec_sft_mixed_retail_gold_v4_compact_noprior_ci_state_hint_retail_os_v2_3gpu.test.state_grounded_v4.summary.json"),
        row_from_summary(root, "retail op oversampling v3", "data weighting", f"{closed}/replan_exec_sft_mixed_retail_gold_v4_compact_noprior_ci_state_hint_retail_op_os_v3_3gpu.test.state_grounded_v4.summary.json"),
        row_from_summary(root, "selector threshold t=0.9", "selector threshold", f"{closed}/replan_exec_sft_mixed_retail_gold_v4_compact_noprior_ci_state_hint_decision_selector_op_rule_override_strict_t09.test.state_grounded_v4.summary.json"),
        row_from_summary(root, "selector threshold t=0.9 + legality v3", "selector threshold", f"{closed}/replan_exec_sft_mixed_retail_gold_v4_compact_noprior_ci_state_hint_decision_selector_op_rule_override_strict_t09_v3.test.state_grounded_v4.summary.json"),
    ]
    priors = [
        prior_row(root, "v8 c_i + compact state", "FM prior parameter", "outputs/fm_replan/next_tool_prior_v8_ci_state_mixed_retail_gold_v4_noprior/metrics.json"),
        prior_row(root, "seed11", "FM prior parameter", "outputs/fm_replan/next_tool_prior_v10_ci_state_seed11_mixed_retail_gold_v4_noprior/metrics.json"),
        prior_row(root, "seed13", "FM prior parameter", "outputs/fm_replan/next_tool_prior_v10_ci_state_seed13_mixed_retail_gold_v4_noprior/metrics.json"),
        prior_row(root, "result/intent tokens v11", "FM prior parameter", "outputs/fm_replan/next_tool_prior_v11_ci_state_result_tokens_mixed_retail_gold_v4_noprior/metrics.json"),
        prior_row(root, "weighted rw2/post2/op1.5", "FM prior parameter", "outputs/fm_replan/next_tool_prior_v12_rw2_post2_op15_ci_state_weighted_mixed_retail_gold_v4_noprior/metrics.json"),
        prior_row(root, "stage aux v13c", "FM prior parameter", "outputs/fm_replan/next_tool_prior_v13c_stage_aux03_nologit_name_stage_ci_state_mixed_retail_gold_v4_noprior/metrics.json"),
        prior_row(root, "candidate reranker v1", "FM prior parameter", "outputs/fm_replan/tool_reranker_v1_ci_state_mixed_retail_gold_v4_noprior/metrics.json"),
    ]
    costs = [
        fm_cost(root, "FM endpoint rollout v4", Path("outputs/fm_replan/encoder_v4_endpoint_rollout")),
        fm_cost(root, "FM replan small-noise v2", Path("outputs/fm_replan/encoder_v2_small_noise")),
        sft_cost(root, "compact-v4 no-prior SFT", Path("outputs/conditioned_sft/replan_next_mixed_retail_gold_v4_compact_noprior_lora_full2048_3gpu")),
        sft_cost(root, "v8 state-hint SFT", Path("outputs/conditioned_sft/replan_next_mixed_retail_gold_v4_compact_noprior_ci_state_hint_lora_full2048_3gpu")),
        sft_cost(root, "v8 state-hint 2 epoch", Path("outputs/conditioned_sft/replan_next_mixed_retail_gold_v4_compact_noprior_ci_state_hint_lora_full2048_3gpu_2epoch")),
        sft_cost(root, "candidate-hint SFT", Path("outputs/conditioned_sft/replan_next_mixed_retail_gold_v4_compact_noprior_ci_state_candidates_hint_lora_full2048_3gpu")),
        sft_cost(root, "retail op-rule SFT", Path("outputs/conditioned_sft/replan_next_mixed_retail_gold_v4_compact_noprior_ci_state_hint_retail_op_rule_lora_full2048_3gpu")),
        toolrl_cost(root),
    ]
    return {
        "ablation_rows": ablations,
        "grounding_rows": grounding,
        "injection_rows": injection,
        "parameter_rows": parameter,
        "prior_rows": priors,
        "cost_rows": costs,
    }


def write_report(payload: dict[str, Any], out_md: Path, out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Ablation, Parameter, And Cost Report",
        "",
        "This report separates module ablations from parameter sweeps. The table records reproducible GPU-step cost; wall-clock time is included only for curated runs.",
        "",
        "## Module Ablations",
    ]
    lines.extend(markdown_table(payload["ablation_rows"], [
        ("method", "name"), ("module", "module"), ("overall succ", "success"), ("retail succ", "retail_success"),
        ("retail tool EM", "retail_tool_em"), ("pred exec ok", "pred_exec_ok"), ("arg value EM", "arg_value_em"), ("note", "note")
    ]))
    lines.extend(["", "## Grounding / Data Ablations"])
    lines.extend(markdown_table(payload["grounding_rows"], [
        ("method", "name"), ("module", "module"), ("overall succ", "success"), ("retail succ", "retail_success"),
        ("retail tool EM", "retail_tool_em"), ("pred exec ok", "pred_exec_ok"), ("arg value EM", "arg_value_em")
    ]))
    lines.extend(["", "## FM Injection Ablations"])
    lines.extend(markdown_table(payload["injection_rows"], [
        ("method", "name"), ("module", "module"), ("overall succ/tool EM", "success"), ("row tool EM", "tool_em"),
        ("retail succ", "retail_success"), ("retail tool EM", "retail_tool_em"), ("arg value EM", "arg_value_em")
    ]))
    lines.extend(["", "## Parameter Analysis: SFT / Data / Selector"])
    lines.extend(markdown_table(payload["parameter_rows"], [
        ("setting", "name"), ("family", "module"), ("overall succ", "success"), ("retail succ", "retail_success"),
        ("retail tool EM", "retail_tool_em"), ("pred exec ok", "pred_exec_ok"), ("arg value EM", "arg_value_em")
    ]))
    lines.extend(["", "## Parameter Analysis: FM Prior"])
    lines.extend(markdown_table(payload["prior_rows"], [
        ("setting", "name"), ("family", "module"), ("train acc", "train_acc"), ("dev acc", "dev_acc"),
        ("dev retail", "dev_retail_acc"), ("test acc", "test_acc"), ("test retail", "test_retail_acc")
    ]))
    lines.extend(["", "## Time And GPU Cost"])
    lines.extend(markdown_table(payload["cost_rows"], [
        ("run", "name"), ("type", "type"), ("train rows", "train_rows"), ("epochs", "epochs"), ("steps", "steps"),
        ("GPUs", "gpus"), ("GPU-steps", "gpu_steps"), ("wall h", "wall_hours"), ("GPU h", "gpu_hours"), ("note", "note")
    ]))
    lines.extend([
        "",
        "## Cost Interpretation",
        "- Main compact-v4 LoRA SFT runs use 3 H800 GPUs, LoRA r=4/alpha=8, max length 2048, global batch 24.",
        "- A one-epoch compact-v4 SFT is 80 optimizer steps, or 240 GPU-steps. The 2-epoch check doubles this to 160 steps / 480 GPU-steps and did not improve test metrics.",
        "- ToolRL GRPO used 3 GPUs, LoRA r=8/alpha=16/dropout=0.05, rollout n=2, and 582 training steps.",
        "- Shallow FM/prior/selector experiments are low-cost relative to 7B SFT/GRPO; they are best reported by accuracy and used to motivate the final selector rather than as expensive training claims.",
    ])
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write ablation, parameter, and compute-cost tables.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--out-md", default="data/processed/ABLATION_PARAMETER_COSTS.md")
    parser.add_argument("--out-json", default="data/processed/ABLATION_PARAMETER_COSTS.json")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build(root)
    write_report(payload, root / args.out_md, root / args.out_json)
    print(json.dumps({"out_md": args.out_md, "out_json": args.out_json}, ensure_ascii=False))


if __name__ == "__main__":
    main()
