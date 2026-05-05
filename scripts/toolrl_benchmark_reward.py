import argparse
import json
from pathlib import Path


def find_balanced_json(text, start=0):
    first = text.find("{", start)
    if first < 0:
        return None, -1
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
                return text[first : idx + 1], idx + 1
    return None, -1


def iter_json_objects(text):
    pos = 0
    while pos < len(text or ""):
        obj_text, end = find_balanced_json(text, pos)
        if not obj_text:
            break
        try:
            yield json.loads(obj_text)
        except Exception:
            pass
        pos = max(end, pos + 1)


def extract_prediction(text):
    candidates = [obj for obj in iter_json_objects(text or "") if isinstance(obj, dict)]
    for obj in reversed(candidates):
        if "stop" in obj and "actions" in obj:
            return obj
    return None


def load_ground_truth(ground_truth):
    if isinstance(ground_truth, dict):
        return ground_truth
    if isinstance(ground_truth, bytes):
        ground_truth = ground_truth.decode("utf-8")
    if hasattr(ground_truth, "item"):
        try:
            ground_truth = ground_truth.item()
        except Exception:
            pass
    if isinstance(ground_truth, str):
        return json.loads(ground_truth)
    raise TypeError(f"Unsupported ground_truth type: {type(ground_truth)}")


def normalize_value(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def build_tool_maps(ground_truth):
    by_name = {}
    available = set()
    for tool in ground_truth.get("available_tools") or []:
        if isinstance(tool, str):
            tool_id = tool
            name = tool.rsplit("::", 1)[-1]
        else:
            tool_id = tool.get("tool_id")
            name = tool.get("name") or (tool_id.rsplit("::", 1)[-1] if tool_id else None)
        if not tool_id:
            continue
        available.add(tool_id)
        by_name[tool_id] = tool_id
        if name:
            by_name[name] = tool_id
    for action in ground_truth.get("gold_actions") or []:
        tool_id = action.get("tool_id")
        name = action.get("tool_name") or (tool_id.rsplit("::", 1)[-1] if tool_id else None)
        if tool_id:
            by_name[tool_id] = tool_id
            if name:
                by_name[name] = tool_id
    return by_name, available


def normalize_actions(prediction, ground_truth):
    if not isinstance(prediction, dict):
        return [], False, False
    by_name, available = build_tool_maps(ground_truth)
    raw_actions = prediction.get("actions")
    if raw_actions is None:
        raw_actions = prediction.get("tool_calls")
    if not isinstance(raw_actions, list):
        return [], bool(prediction.get("stop", False)), False
    actions = []
    available_ok = True
    schema_ok = isinstance(prediction.get("stop"), bool)
    for raw in raw_actions:
        if not isinstance(raw, dict):
            schema_ok = False
            continue
        name = raw.get("tool_id") or raw.get("tool_name") or raw.get("name") or raw.get("tool")
        tool_id = by_name.get(name)
        if not tool_id:
            available_ok = False
            continue
        if available and tool_id not in available:
            available_ok = False
            continue
        args = raw.get("arguments")
        if args is None:
            args = raw.get("args") or {}
        if not isinstance(args, dict):
            schema_ok = False
            args = {}
        actions.append(
            {
                "tool_id": tool_id,
                "tool_name": tool_id.rsplit("::", 1)[-1],
                "arguments": args,
            }
        )
    return actions, bool(prediction.get("stop", False)), schema_ok and available_ok


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def tool_score(pred_actions, gold_actions):
    pred_ids = [action.get("tool_id") for action in pred_actions]
    gold_ids = [action.get("tool_id") for action in gold_actions]
    if pred_ids == gold_ids:
        return 1.0
    denom = max(len(pred_ids), len(gold_ids), 1)
    return max(0.0, 1.0 - edit_distance(pred_ids, gold_ids) / denom)


def argument_score(pred_actions, gold_actions):
    gold_key_count = 0
    key_hits = 0
    value_hits = 0
    for idx, gold in enumerate(gold_actions):
        pred = pred_actions[idx] if idx < len(pred_actions) else {}
        gold_args = gold.get("arguments") or {}
        pred_args = pred.get("arguments") or {}
        if not isinstance(gold_args, dict):
            gold_args = {}
        if not isinstance(pred_args, dict):
            pred_args = {}
        gold_keys = set(gold_args)
        gold_key_count += len(gold_keys)
        for key in gold_keys & set(pred_args):
            key_hits += 1
            if normalize_value(pred_args.get(key)) == normalize_value(gold_args.get(key)):
                value_hits += 1
    if gold_key_count == 0:
        return 1.0
    key_recall = key_hits / gold_key_count
    value_em = value_hits / gold_key_count
    return 0.35 * key_recall + 0.65 * value_em


def score_prediction(prediction, ground_truth):
    gold_actions = ground_truth.get("gold_actions") or []
    gold_stop = bool(ground_truth.get("gold_stop", not gold_actions))
    pred_actions, pred_stop, valid_schema = normalize_actions(prediction, ground_truth)

    format_score = 1.0 if valid_schema and isinstance(prediction, dict) else 0.0
    stop_score = 1.0 if pred_stop == gold_stop else 0.0
    if gold_stop:
        tool_match = 1.0 if pred_stop and not pred_actions else 0.0
        arg_match = 1.0
    else:
        tool_match = tool_score(pred_actions, gold_actions)
        arg_match = argument_score(pred_actions, gold_actions)
    correctness = 0.2 * stop_score + 0.55 * tool_match + 0.25 * arg_match
    length_score = 1.0 if len(pred_actions) <= max(len(gold_actions) + 1, 1) else 0.0
    final = 0.15 * format_score + 0.8 * correctness + 0.05 * length_score
    if not valid_schema:
        final *= 0.5
    return {
        "score": float(final),
        "format_score": float(format_score),
        "correctness_score": float(correctness),
        "length_score": float(length_score),
        "tool_score": float(tool_match),
        "argument_score": float(arg_match),
        "stop_score": float(stop_score),
    }


def compute_score(solution_str, ground_truth, step=0):
    gt = load_ground_truth(ground_truth)
    pred = extract_prediction(solution_str or "")
    if pred is None:
        return 0.0, 0.0, 0.0, 0.0
    scores = score_prediction(pred, gt)
    return (
        scores["score"],
        scores["format_score"],
        scores["correctness_score"],
        scores["length_score"],
    )


def main():
    parser = argparse.ArgumentParser(description="Smoke ToolRL benchmark reward on raw generations.")
    parser.add_argument("--prediction", required=True, help="Prediction text or path to a text file.")
    parser.add_argument("--ground-truth", required=True, help="Ground-truth JSON string or path to JSON.")
    args = parser.parse_args()
    pred_path = Path(args.prediction)
    gt_path = Path(args.ground_truth)
    pred = pred_path.read_text(encoding="utf-8") if len(args.prediction) < 4096 and pred_path.exists() else args.prediction
    gt = gt_path.read_text(encoding="utf-8") if len(args.ground_truth) < 4096 and gt_path.exists() else args.ground_truth
    print(json.dumps(score_prediction(extract_prediction(pred), load_ground_truth(gt)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
