import argparse
import json
import math
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def success(row):
    return 1 if row.get("next_action_success") else 0


def tool_em(row):
    return 1 if row.get("tool_exact_match") else 0


def pred_exec(row):
    value = row.get("predicted_action_execution_ok")
    return None if value is None else (1 if value else 0)


def metric_value(row, metric):
    if metric == "success":
        return success(row)
    if metric == "tool_em":
        return tool_em(row)
    if metric == "pred_exec":
        return pred_exec(row)
    raise ValueError(metric)


def sign_test_pvalue(wins, losses):
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    cdf = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * cdf)


def bootstrap_ci(diffs, iterations, seed):
    if not diffs:
        return [None, None]
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(iterations):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(0.025 * iterations)]
    hi = means[min(iterations - 1, int(0.975 * iterations))]
    return [lo, hi]


def summarize_pair(name, a_path, b_path, metric, iterations, seed):
    a_rows = {row["id"]: row for row in read_jsonl(a_path)}
    b_rows = {row["id"]: row for row in read_jsonl(b_path)}
    ids = sorted(set(a_rows) & set(b_rows))
    diffs = []
    wins = losses = ties = 0
    changed = []
    for row_id in ids:
        a = metric_value(a_rows[row_id], metric)
        b = metric_value(b_rows[row_id], metric)
        if a is None or b is None:
            continue
        diff = b - a
        diffs.append(diff)
        if diff > 0:
            wins += 1
            changed.append({"id": row_id, "direction": "improved"})
        elif diff < 0:
            losses += 1
            changed.append({"id": row_id, "direction": "regressed"})
        else:
            ties += 1
    mean_a = sum(metric_value(a_rows[row_id], metric) for row_id in ids if metric_value(a_rows[row_id], metric) is not None) / len(diffs)
    mean_b = sum(metric_value(b_rows[row_id], metric) for row_id in ids if metric_value(b_rows[row_id], metric) is not None) / len(diffs)
    return {
        "comparison": name,
        "metric": metric,
        "n": len(diffs),
        "baseline_mean": mean_a,
        "candidate_mean": mean_b,
        "diff": mean_b - mean_a,
        "bootstrap_ci95": bootstrap_ci(diffs, iterations, seed),
        "paired_wins": wins,
        "paired_losses": losses,
        "paired_ties": ties,
        "sign_test_pvalue": sign_test_pvalue(wins, losses),
        "changed_count": wins + losses,
        "changed_examples": changed[:20],
    }


def fmt(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def table(rows):
    columns = [
        ("comparison", "comparison"),
        ("metric", "metric"),
        ("n", "n"),
        ("base", "baseline_mean"),
        ("cand", "candidate_mean"),
        ("diff", "diff"),
        ("ci95", "ci"),
        ("wins", "paired_wins"),
        ("losses", "paired_losses"),
        ("p", "sign_test_pvalue"),
    ]
    lines = [
        "| " + " | ".join(title for title, _ in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        item = dict(row)
        item["ci"] = f"[{fmt(row['bootstrap_ci95'][0])}, {fmt(row['bootstrap_ci95'][1])}]"
        lines.append("| " + " | ".join(fmt(item.get(key)) for _, key in columns) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Write paired bootstrap/sign-test report for closed-loop comparisons.")
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--out-md", default="data/processed/SIGNIFICANCE_REPORT.md")
    parser.add_argument("--out-json", default="data/processed/SIGNIFICANCE_REPORT.json")
    args = parser.parse_args()

    root = Path(args.root)
    closed = root / "data" / "processed" / "closed_loop"
    qwen_no = closed / "replan_exec_sft_mixed_retail_gold_v4_compact_noprior_3gpu.test.state_grounded_v4.details.jsonl"
    qwen_state = closed / "replan_exec_sft_mixed_retail_gold_v4_compact_noprior_ci_state_hint_3gpu.test.state_grounded_v4.details.jsonl"
    qwen_final = closed / "replan_exec_sft_mixed_retail_gold_v4_compact_noprior_ci_state_hint_decision_selector_op_rule_override_strict_t09_v3.test.state_grounded_v4.details.jsonl"
    llama_no = closed / "replan_exec_llama32_3b_compact_v4_noprior.test.state_grounded_v4.details.jsonl"
    llama_state = closed / "replan_exec_llama32_3b_compact_v4_ci_state_hint.test.state_grounded_v4.details.jsonl"
    llama_final = closed / "replan_exec_llama32_3b_compact_v4_ci_state_hint_decision_selector_op_rule_override_strict_t09_v3.test.state_grounded_v4.details.jsonl"

    specs = [
        ("Qwen state-hint vs no-prior", qwen_no, qwen_state),
        ("Qwen final vs state-hint", qwen_state, qwen_final),
        ("Qwen final vs no-prior", qwen_no, qwen_final),
        ("Llama state-hint vs no-prior", llama_no, llama_state),
        ("Llama final vs state-hint", llama_state, llama_final),
        ("Llama final vs no-prior", llama_no, llama_final),
    ]
    rows = []
    for name, base, cand in specs:
        if not base.exists() or not cand.exists():
            continue
        rows.append(summarize_pair(name, base, cand, "success", args.iterations, args.seed))
        rows.append(summarize_pair(name, base, cand, "tool_em", args.iterations, args.seed + 1))

    payload = {
        "method": "paired row-level bootstrap for mean difference plus exact two-sided sign test over discordant rows",
        "iterations": args.iterations,
        "seed": args.seed,
        "rows": rows,
    }
    out_json = root / args.out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    md = [
        "# Significance And Paired Difference Report",
        "",
        "This report uses paired row-level comparisons on the compact-v4 test set. The confidence interval is a paired bootstrap over rows; the p-value is an exact two-sided sign test over discordant rows. It is intended to separate robust prior-injection gains from small selector gains.",
        "",
        table(rows),
        "",
        "Interpretation:",
        "",
        "- The no-prior -> state-hint comparisons are the cleanest evidence for FM/state-prior injection because they share the same evaluator, grounding, and no selector.",
        "- The state-hint -> final-selector comparisons are targeted but small; report them as a conservative retail-operation improvement, not as the main statistical claim.",
        "- Llama transfer repeats the direction of the state-hint gain, but because the evaluator/grounder is shared, it should be framed as an interface transfer check rather than independent proof of model reasoning.",
    ]
    out_md = root / args.out_md
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
