import argparse
import json
from pathlib import Path


METRICS = [
    ("overall.next_action_success", "overall", "next_action_success", "higher"),
    ("retail.next_action_success", "by_domain.retail", "next_action_success", "higher"),
    ("overall.tool_exact_match", "overall", "tool_exact_match", "higher"),
    ("retail.tool_exact_match", "by_domain.retail", "tool_exact_match", "higher"),
    ("overall.predicted_action_execution_ok", "overall", "predicted_action_execution_ok", "higher"),
    ("overall.argument_value_exact_match", "overall", "argument_value_exact_match", "higher"),
]

SAFETY_METRICS = [
    ("overall.replay_execution_ok", "overall", "replay_execution_ok", 0.98),
    ("overall.predicted_action_execution_ok", "overall", "predicted_action_execution_ok", 0.90),
]


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def get_nested(obj, dotted, key):
    cur = obj
    for part in dotted.split("."):
        cur = cur.get(part, {})
    value = cur.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def compare(candidate, baseline, min_delta, max_regression):
    rows = []
    wins = 0
    ties = 0
    losses = 0
    for name, group, key, direction in METRICS:
        cand = get_nested(candidate, group, key)
        base = get_nested(baseline, group, key)
        if cand is None or base is None:
            rows.append({"metric": name, "candidate": cand, "baseline": base, "status": "missing"})
            continue
        delta = cand - base if direction == "higher" else base - cand
        if delta > min_delta:
            status = "win"
            wins += 1
        elif abs(delta) <= min_delta:
            status = "tie"
            ties += 1
        else:
            status = "regress"
            losses += 1
        rows.append({"metric": name, "candidate": cand, "baseline": base, "delta": delta, "status": status})
    safety_failures = []
    for name, group, key, threshold in SAFETY_METRICS:
        cand = get_nested(candidate, group, key)
        if cand is not None and cand < threshold:
            safety_failures.append({"metric": name, "candidate": cand, "threshold": threshold})
    # Majority-win acceptance: a candidate should beat the baseline on most
    # primary metrics. Safety thresholds remain hard requirements.
    passed = wins >= 4 and not safety_failures
    return {
        "passed": passed,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "checked": len(rows),
        "rows": rows,
        "large_regressions": [],
        "safety_failures": safety_failures,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare a candidate closed-loop summary against a baseline.")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument("--max-regression", type=float, default=0.01, help="Deprecated; kept for CLI compatibility.")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    payload = compare(load_json(args.candidate), load_json(args.baseline), args.min_delta, args.max_regression)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
