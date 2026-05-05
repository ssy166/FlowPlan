#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(pwd)}"
PY="${PY:-python}"
MODEL="${MODEL:-${LLAMA32_MODEL:-/path/to/Llama-3.2-3B-Instruct}}"
GPUS="${GPUS:-0,1}"
GEN_GPU="${GEN_GPU:-0}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-192}"

cd "$ROOT"

if [ ! -d "$MODEL" ]; then
  echo "Missing model directory: $MODEL" >&2
  exit 2
fi

NOPRIOR_DATA="data/processed/fm_replan_sft_next_mixed_retail_gold_v4_compact_noprior"
STATE_DATA="data/processed/fm_replan_sft_next_mixed_retail_gold_v4_compact_noprior_ci_state_hint_v1"

NOPRIOR_OUT="outputs/conditioned_sft/replan_next_llama32_3b_compact_v4_noprior_lora_full2048_2gpu"
STATE_OUT="outputs/conditioned_sft/replan_next_llama32_3b_compact_v4_ci_state_hint_lora_full2048_2gpu"

train_one() {
  local data_dir="$1"
  local out_dir="$2"
  if [ -f "$out_dir/final/adapter_model.safetensors" ]; then
    echo "Skip existing adapter: $out_dir/final"
    return
  fi
  CUDA_VISIBLE_DEVICES="$GPUS" "$PY" -m torch.distributed.run --nproc_per_node=2 \
    scripts/train_fm_prefix_sft.py \
    --model-path "$MODEL" \
    --data-dir "$data_dir" \
    --tensor-dir data/processed/replan_fm_tensor_encoder \
    --out-dir "$out_dir" \
    --no-prefix \
    --epochs 1 \
    --batch-size 1 \
    --grad-accum 8 \
    --lr 1e-4 \
    --max-length "$MAX_LENGTH" \
    --lora-r 4 \
    --lora-alpha 8 \
    --lora-dropout 0.05 \
    --dtype bf16 \
    --logging-steps 5
}

generate_one() {
  local data_dir="$1"
  local adapter_dir="$2/final"
  local tag="$3"
  local split="$4"
  local data_file="$data_dir/$split.jsonl"
  local pred="data/processed/predictions/${tag}.${split}.jsonl"
  local raw="data/processed/predictions/${tag}.${split}.raw.jsonl"
  local details="data/processed/predictions/${tag}.${split}.details.jsonl"
  local summary="data/processed/predictions/${tag}.${split}.summary.json"
  if [ -f "$summary" ] && [ -f "$pred" ]; then
    echo "Skip existing generation: $summary"
    return
  fi
  CUDA_VISIBLE_DEVICES="$GEN_GPU" "$PY" scripts/run_fm_prefix_sft_generation.py \
    --model-path "$MODEL" \
    --adapter-dir "$adapter_dir" \
    --data "$data_file" \
    --tools data/processed/tools.jsonl \
    --no-prefix \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --dtype bf16 \
    --device cuda \
    --out-raw "$raw" \
    --out-pred "$pred" \
    --out-details "$details" \
    --summary-out "$summary" \
    --model-name "$tag"
}

ground_and_eval() {
  local data_dir="$1"
  local tag="$2"
  local split="$3"
  local pred="data/processed/predictions/${tag}.${split}.jsonl"
  local grounded="data/processed/predictions/${tag}.${split}.state_grounded_v4.jsonl"
  local details="data/processed/closed_loop/replan_exec_${tag}.${split}.state_grounded_v4.details.jsonl"
  local summary="data/processed/closed_loop/replan_exec_${tag}.${split}.state_grounded_v4.summary.json"
  "$PY" scripts/ground_replan_predictions.py \
    --data "$data_dir/$split.jsonl" \
    --pred "$pred" \
    --out "$grounded"
  "$PY" scripts/evaluate_replan_execution.py \
    --data "$data_dir/$split.jsonl" \
    --pred "$grounded" \
    --out-details "$details" \
    --summary-out "$summary" \
    --result-grounding
}

apply_override_and_eval() {
  local tag="$1"
  local split="$2"
  local base="data/processed/predictions/${tag}.${split}.jsonl"
  local override_tag="${tag}_decision_selector_op_rule_override_strict_t09_v3"
  local override_pred="data/processed/predictions/${override_tag}.${split}.jsonl"
  "$PY" scripts/apply_retail_operation_rule_override.py \
    --data "$STATE_DATA/$split.jsonl" \
    --pred "$base" \
    --selector-pred "outputs/fm_replan/retail_decision_selector_v1/${split}.pred.jsonl" \
    --selector-threshold 0.9 \
    --out "$override_pred"
  "$PY" scripts/ground_replan_predictions.py \
    --data "$STATE_DATA/$split.jsonl" \
    --pred "$override_pred" \
    --out "data/processed/predictions/${override_tag}.${split}.state_grounded_v4.jsonl"
  "$PY" scripts/evaluate_replan_execution.py \
    --data "$STATE_DATA/$split.jsonl" \
    --pred "data/processed/predictions/${override_tag}.${split}.state_grounded_v4.jsonl" \
    --out-details "data/processed/closed_loop/replan_exec_${override_tag}.${split}.state_grounded_v4.details.jsonl" \
    --summary-out "data/processed/closed_loop/replan_exec_${override_tag}.${split}.state_grounded_v4.summary.json" \
    --result-grounding
}

train_one "$NOPRIOR_DATA" "$NOPRIOR_OUT"
train_one "$STATE_DATA" "$STATE_OUT"

for split in dev test; do
  generate_one "$NOPRIOR_DATA" "$NOPRIOR_OUT" "llama32_3b_compact_v4_noprior" "$split"
  ground_and_eval "$NOPRIOR_DATA" "llama32_3b_compact_v4_noprior" "$split"

  generate_one "$STATE_DATA" "$STATE_OUT" "llama32_3b_compact_v4_ci_state_hint" "$split"
  ground_and_eval "$STATE_DATA" "llama32_3b_compact_v4_ci_state_hint" "$split"
  apply_override_and_eval "llama32_3b_compact_v4_ci_state_hint" "$split"
done

"$PY" - <<'PY'
import json
from pathlib import Path

paths = {
    "llama_noprior": "data/processed/closed_loop/replan_exec_llama32_3b_compact_v4_noprior.test.state_grounded_v4.summary.json",
    "llama_state_hint": "data/processed/closed_loop/replan_exec_llama32_3b_compact_v4_ci_state_hint.test.state_grounded_v4.summary.json",
    "llama_final_override": "data/processed/closed_loop/replan_exec_llama32_3b_compact_v4_ci_state_hint_decision_selector_op_rule_override_strict_t09_v3.test.state_grounded_v4.summary.json",
}

def metric(summary, section, key):
    if section == "overall":
        return (summary.get("overall") or {}).get(key)
    return ((summary.get("by_domain") or {}).get(section) or {}).get(key)

rows = []
for name, path in paths.items():
    p = Path(path)
    if not p.exists():
        continue
    data = json.loads(p.read_text(encoding="utf-8"))
    rows.append({
        "method": name,
        "overall_success": metric(data, "overall", "next_action_success"),
        "retail_success": metric(data, "retail", "next_action_success"),
        "tool_em": metric(data, "overall", "tool_exact_match"),
        "retail_tool_em": metric(data, "retail", "tool_exact_match"),
        "pred_exec_ok": metric(data, "overall", "predicted_action_execution_ok"),
        "arg_value_em": metric(data, "overall", "argument_value_exact_match"),
        "summary": path,
    })
out = Path("data/processed/LLAMA32_MAIN_RESULTS.json")
out.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(rows, ensure_ascii=False, indent=2))
PY
