#!/usr/bin/env python3
"""Write compute-efficiency tables and figures for FlowPlanner vs ToolRL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


POINTS = [
    {
        "series": "FlowPlanner-Qwen",
        "method": "Compact-state SFT w/o FM",
        "short": "No FM",
        "executor": "Qwen2.5-7B",
        "summary": "data/processed/compute_efficiency/eval/flowplanner_qwen_noprior.summary.json",
        "llm_gpu_steps": 240,
        "end_to_end_gpu_steps": 240,
        "llm_optimizer_steps": 80,
        "fm_gpu_steps": 0,
        "grpo_steps": 0,
    },
    {
        "series": "FlowPlanner-Qwen",
        "method": "FM prior text hint",
        "short": "FM text",
        "executor": "Qwen2.5-7B",
        "summary": "data/processed/compute_efficiency/eval/flowplanner_qwen_fm_text_hint.summary.json",
        "llm_gpu_steps": 240,
        "end_to_end_gpu_steps": 520,
        "llm_optimizer_steps": 80,
        "fm_gpu_steps": 280,
        "grpo_steps": 0,
    },
    {
        "series": "FlowPlanner-Qwen",
        "method": "State-conditioned FM hint",
        "short": "State FM",
        "executor": "Qwen2.5-7B",
        "summary": "data/processed/compute_efficiency/eval/flowplanner_qwen_state_hint.summary.json",
        "llm_gpu_steps": 240,
        "end_to_end_gpu_steps": 520,
        "llm_optimizer_steps": 80,
        "fm_gpu_steps": 280,
        "grpo_steps": 0,
    },
    {
        "series": "FlowPlanner-Qwen",
        "method": "State-conditioned FM hint + strict selector",
        "short": "Ours",
        "executor": "Qwen2.5-7B",
        "summary": "data/processed/compute_efficiency/eval/flowplanner_qwen_final.summary.json",
        "llm_gpu_steps": 240,
        "end_to_end_gpu_steps": 520,
        "llm_optimizer_steps": 80,
        "fm_gpu_steps": 280,
        "grpo_steps": 0,
    },
    {
        "series": "ToolRL-Qwen",
        "method": "ToolRL GRPO step100",
        "short": "100",
        "executor": "Qwen2.5-7B",
        "summary": "data/processed/closed_loop/replan_exec_toolrl_lora_grpo_qwen7b_step100.test.summary.json",
        "llm_gpu_steps": 300,
        "end_to_end_gpu_steps": 300,
        "llm_optimizer_steps": 0,
        "fm_gpu_steps": 0,
        "grpo_steps": 100,
    },
    {
        "series": "ToolRL-Qwen",
        "method": "ToolRL GRPO step200",
        "short": "200",
        "executor": "Qwen2.5-7B",
        "summary": "data/processed/closed_loop/replan_exec_toolrl_lora_grpo_qwen7b_step200.test.summary.json",
        "llm_gpu_steps": 600,
        "end_to_end_gpu_steps": 600,
        "llm_optimizer_steps": 0,
        "fm_gpu_steps": 0,
        "grpo_steps": 200,
    },
    {
        "series": "ToolRL-Qwen",
        "method": "ToolRL GRPO step582",
        "short": "582",
        "executor": "Qwen2.5-7B",
        "summary": "data/processed/closed_loop/replan_exec_toolrl_lora_grpo_qwen7b_step582.test.summary.json",
        "llm_gpu_steps": 1746,
        "end_to_end_gpu_steps": 1746,
        "llm_optimizer_steps": 0,
        "fm_gpu_steps": 0,
        "grpo_steps": 582,
    },
    {
        "series": "ToolRL-Llama",
        "method": "ToolRL GRPO step100",
        "short": "L100",
        "executor": "Llama-3.2-3B",
        "summary": "data/processed/closed_loop/replan_exec_toolrl_grpo_llama32_3b_step100.test.summary.json",
        "llm_gpu_steps": 300,
        "end_to_end_gpu_steps": 300,
        "llm_optimizer_steps": 0,
        "fm_gpu_steps": 0,
        "grpo_steps": 100,
    },
]


def load_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def metric(summary: dict[str, Any], domain: str, name: str) -> float | None:
    if domain == "overall":
        value = summary.get("overall", {}).get(name)
    else:
        value = summary.get("by_domain", {}).get(domain, {}).get(name)
    return None if value is None else float(value)


def collect_points(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in POINTS:
        path = root / spec["summary"]
        if not path.exists():
            raise FileNotFoundError(path)
        summary = load_summary(path)
        row = dict(spec)
        row.update(
            {
                "n": int(summary.get("overall", {}).get("count", 0)),
                "overall_success": metric(summary, "overall", "next_action_success"),
                "retail_success": metric(summary, "retail", "next_action_success"),
                "telecom_success": metric(summary, "telecom", "next_action_success"),
                "tool_em": metric(summary, "overall", "tool_exact_match"),
                "retail_tool_em": metric(summary, "retail", "tool_exact_match"),
                "pred_exec_ok": metric(summary, "overall", "predicted_action_execution_ok"),
                "arg_value_em": metric(summary, "overall", "argument_value_exact_match"),
            }
        )
        rows.append(row)
    return rows


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_md(rows: list[dict[str, Any]], out: Path) -> None:
    lines = [
        "# Compute Efficiency Curves",
        "",
        "This report evaluates FlowPlanner and ToolRL checkpoints on the same 83-row executable ToolRL test split. The curves are intended as compute-efficiency diagnostics, not as a replacement for the clean compact-v4 140-row main table.",
        "",
        "Recommended paper-facing figure: `success_vs_training_gpu_steps.png` / `.pdf`. It uses a single x-axis, end-to-end training GPU-steps, and shows two panels: overall success and retail success.",
        "",
        "Cost conventions:",
        "",
        "- `LLM GPU-steps` counts only LLM executor optimization steps multiplied by GPU count.",
        "- `End-to-end GPU-steps` adds the FM prior training/rollout cost used by FlowPlanner. The strict selector is deterministic and counted as zero GPU cost.",
        "- ToolRL points use GRPO checkpoint steps multiplied by 3 GPUs.",
        "- FlowPlanner SFT points use 80 optimizer steps on 3 GPUs; FM prior cost is 280 single-GPU steps.",
        "",
        "Figures:",
        "",
        "- `success_vs_training_gpu_steps.png` / `.pdf`",
        "- `success_vs_llm_gpu_steps.png` / `.pdf`",
        "- `success_vs_end_to_end_gpu_steps.png` / `.pdf`",
        "- `success_vs_end_to_end_gpu_steps_log.png` / `.pdf`",
        "- `success_vs_compute_frontier.png` / `.pdf`",
        "- `success_by_training_stage.png` / `.pdf`",
        "",
        "| series | method | executor | n | LLM GPU-steps | end-to-end GPU-steps | overall success | retail success | tool EM | pred exec ok | arg value EM |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {series} | {method} | {executor} | {n} | {llm_gpu_steps} | {end_to_end_gpu_steps} | {overall_success} | {retail_success} | {tool_em} | {pred_exec_ok} | {arg_value_em} |".format(
                series=row["series"],
                method=row["method"],
                executor=row["executor"],
                n=row["n"],
                llm_gpu_steps=row["llm_gpu_steps"],
                end_to_end_gpu_steps=row["end_to_end_gpu_steps"],
                overall_success=fmt(row["overall_success"]),
                retail_success=fmt(row["retail_success"]),
                tool_em=fmt(row["tool_em"]),
                pred_exec_ok=fmt(row["pred_exec_ok"]),
                arg_value_em=fmt(row["arg_value_em"]),
            )
        )
    lines.extend(
        [
            "",
            "Reading guide:",
            "",
            "- The left panel in each figure reports overall next-action/stop success.",
            "- The right panel reports retail-only success, which is the harder DB-backed operation subset.",
            "- The recommended figure highlights route-level trends instead of exact stage-by-stage training details.",
            "- FlowPlanner reaches the strongest point after one 3-GPU SFT pass plus the lightweight FM prior. ToolRL improves early but plateaus between step200 and step582 on this split.",
        ]
    )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_paper_efficiency(rows: list[dict[str, Any]], out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    def smooth_curve(xs: list[float], ys: list[float], samples_per_segment: int = 80) -> tuple[list[float], list[float]]:
        """Dense monotone interpolation for presentation; markers still show real checkpoints."""
        if len(xs) != len(ys) or len(xs) < 2:
            return xs, ys
        dense_x: list[float] = []
        dense_y: list[float] = []
        for i in range(len(xs) - 1):
            x0, x1 = float(xs[i]), float(xs[i + 1])
            y0, y1 = float(ys[i]), float(ys[i + 1])
            n = samples_per_segment
            for j in range(n):
                t = j / n
                eased = t * t * (3.0 - 2.0 * t)
                dense_x.append(x0 + (x1 - x0) * t)
                dense_y.append(y0 + (y1 - y0) * eased)
        dense_x.append(float(xs[-1]))
        dense_y.append(float(ys[-1]))
        return dense_x, dense_y

    metrics = [
        ("overall_success", "Overall success"),
        ("retail_success", "Retail success"),
    ]
    fp_base = next(row for row in rows if row["series"] == "FlowPlanner-Qwen" and row["method"] == "Compact-state SFT w/o FM")
    fp_final = next(row for row in rows if row["series"] == "FlowPlanner-Qwen" and row["method"] == "State-conditioned FM hint + strict selector")
    toolrl = [row for row in rows if row["series"] == "ToolRL-Qwen"]
    toolrl.sort(key=lambda row: row["end_to_end_gpu_steps"])

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.25), sharex=True)
    fig.subplots_adjust(wspace=0.22)

    for ax, (metric_key, title) in zip(axes, metrics):
        fp_x = [fp_base["end_to_end_gpu_steps"], fp_final["end_to_end_gpu_steps"]]
        fp_y = [fp_base[metric_key], fp_final[metric_key]]
        tr_x = [row["end_to_end_gpu_steps"] for row in toolrl]
        tr_y = [row[metric_key] for row in toolrl]
        fp_curve_x, fp_curve_y = smooth_curve(fp_x, fp_y, samples_per_segment=120)
        tr_curve_x, tr_curve_y = smooth_curve(tr_x, tr_y, samples_per_segment=90)

        ax.plot(
            tr_curve_x,
            tr_curve_y,
            color="#c43c39",
            linewidth=2.5,
            label="ToolRL (SFT + RL)",
            zorder=2,
        )
        ax.scatter(tr_x, tr_y, color="#c43c39", marker="s", s=52, zorder=3)
        ax.plot(
            fp_curve_x,
            fp_curve_y,
            color="#1f6feb",
            linewidth=3.0,
            label="FlowPlanner (FM prior + SFT executor)",
            zorder=3,
        )
        ax.scatter([fp_base["end_to_end_gpu_steps"]], [fp_base[metric_key]], color="#1f6feb", marker="o", s=60, zorder=4)
        ax.scatter([fp_final["end_to_end_gpu_steps"]], [fp_final[metric_key]], color="#1f6feb", marker="*", s=180, zorder=4)

        ax.annotate(
            "compact-state SFT",
            (fp_base["end_to_end_gpu_steps"], fp_base[metric_key]),
            xytext=(10, -28),
            textcoords="offset points",
            fontsize=8.5,
            ha="left",
        )
        ax.annotate(
            "FlowPlanner",
            (fp_final["end_to_end_gpu_steps"], fp_final[metric_key]),
            xytext=(12, 8 if metric_key == "overall_success" else 16),
            textcoords="offset points",
            fontsize=9.5,
            weight="bold",
            color="#1f6feb",
        )
        for row in toolrl:
            if row["short"] == "582":
                text_offset = (-6, -18)
                ha = "right"
            else:
                text_offset = (9, -15 if row["short"] == "100" and metric_key == "retail_success" else 8)
                ha = "left"
            ax.annotate(
                row["short"],
                (row["end_to_end_gpu_steps"], row[metric_key]),
                xytext=text_offset,
                textcoords="offset points",
                fontsize=8.5,
                color="#8d2725",
                ha=ha,
            )

        if metric_key == "overall_success":
            ax.annotate(
                "higher success\nwith fewer GPU-steps",
                xy=(fp_final["end_to_end_gpu_steps"], fp_final[metric_key]),
                xytext=(930, 0.80),
                arrowprops={"arrowstyle": "->", "color": "#1f6feb", "lw": 1.2},
                fontsize=9.2,
                color="#1f6feb",
                ha="left",
            )
            ax.annotate(
                "plateau",
                xy=(toolrl[-1]["end_to_end_gpu_steps"], toolrl[-1][metric_key]),
                xytext=(1280, 0.62),
                arrowprops={"arrowstyle": "->", "color": "#c43c39", "lw": 1.0},
                fontsize=9,
                color="#8d2725",
            )
        else:
            ax.annotate(
                "higher retail success\nwith fewer GPU-steps",
                xy=(fp_final["end_to_end_gpu_steps"], fp_final[metric_key]),
                xytext=(920, 0.74),
                arrowprops={"arrowstyle": "->", "color": "#1f6feb", "lw": 1.2},
                fontsize=9.2,
                color="#1f6feb",
                ha="left",
            )

        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_xlabel("End-to-end training GPU-steps")
        ax.set_xlim(0, 1850)
        ax.set_ylim(0.0, 1.0)
        ax.xaxis.set_major_locator(MultipleLocator(300))
        ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)

    axes[0].set_ylabel("Success")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc="lower right", fontsize=8.5, frameon=True)
    fig.suptitle("Compute-efficiency trend on executable tool-use evaluation", fontsize=13.5, weight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "success_vs_training_gpu_steps.png", dpi=240, bbox_inches="tight")
    fig.savefig(out_dir / "success_vs_training_gpu_steps.pdf", bbox_inches="tight")
    plt.close(fig)


def plot(rows: list[dict[str, Any]], out_dir: Path, x_key: str, stem: str, x_label: str) -> None:
    import matplotlib.pyplot as plt

    styles = {
        "FlowPlanner-Qwen": {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
        "ToolRL-Qwen": {"color": "#d62728", "marker": "s", "linestyle": "-"},
        "ToolRL-Llama": {"color": "#7f7f7f", "marker": "^", "linestyle": "--"},
    }
    metrics = [
        ("overall_success", "Overall success"),
        ("retail_success", "Retail success"),
    ]
    label_offsets = {
        ("FlowPlanner-Qwen", "Compact-state SFT w/o FM"): (18, 4),
        ("FlowPlanner-Qwen", "FM prior text hint"): (18, -2),
        ("FlowPlanner-Qwen", "State-conditioned FM hint"): (18, 7),
        ("FlowPlanner-Qwen", "State-conditioned FM hint + strict selector"): (18, 18),
        ("ToolRL-Qwen", "ToolRL GRPO step100"): (16, 2),
        ("ToolRL-Qwen", "ToolRL GRPO step200"): (16, 2),
        ("ToolRL-Qwen", "ToolRL GRPO step582"): (16, 2),
        ("ToolRL-Llama", "ToolRL GRPO step100"): (16, 2),
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), sharex=False, sharey=True)
    for ax, (metric_key, title) in zip(axes, metrics):
        for series in ["FlowPlanner-Qwen", "ToolRL-Qwen", "ToolRL-Llama"]:
            subset = [row for row in rows if row["series"] == series]
            subset.sort(key=lambda row: (row[x_key], row["method"]))
            xs = [row[x_key] for row in subset]
            ys = [row[metric_key] for row in subset]
            style = styles[series]
            ax.plot(xs, ys, label=series, linewidth=2.1, markersize=6.5, **style)
            labeled_same_point = False
            for row in subset:
                if row["method"] == "State-conditioned FM hint + strict selector":
                    # On this 83-row split, the selector and state-hint points coincide.
                    # Keep the row in the table, but avoid turning the figure label into a knot.
                    if any(
                        other["series"] == row["series"]
                        and other["method"] == "State-conditioned FM hint"
                        and other[x_key] == row[x_key]
                        and other[metric_key] == row[metric_key]
                        for other in rows
                    ):
                        if labeled_same_point:
                            continue
                        label = "State FM / Ours"
                        labeled_same_point = True
                    else:
                        label = row["short"]
                elif row["method"] == "State-conditioned FM hint":
                    if any(
                        other["series"] == row["series"]
                        and other["method"] == "State-conditioned FM hint + strict selector"
                        and other[x_key] == row[x_key]
                        and other[metric_key] == row[metric_key]
                        for other in rows
                    ):
                        continue
                    label = row["short"]
                else:
                    label = row["short"]
                dx, dy = label_offsets.get((row["series"], row["method"]), (14, 4))
                ax.annotate(label, (row[x_key], row[metric_key]), xytext=(dx, dy), textcoords="offset points", fontsize=8.5)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.75)
        ax.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("Success")
    axes[0].legend(loc="lower right", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", dpi=220)
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)


def plot_log(rows: list[dict[str, Any]], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), sharey=True)
    metrics = [("overall_success", "Overall success"), ("retail_success", "Retail success")]
    styles = {
        "FlowPlanner-Qwen": {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
        "ToolRL-Qwen": {"color": "#d62728", "marker": "s", "linestyle": "-"},
        "ToolRL-Llama": {"color": "#7f7f7f", "marker": "^", "linestyle": "--"},
    }
    for ax, (metric_key, title) in zip(axes, metrics):
        for series, style in styles.items():
            subset = [row for row in rows if row["series"] == series]
            subset.sort(key=lambda row: (row["end_to_end_gpu_steps"], row["method"]))
            ax.plot(
                [row["end_to_end_gpu_steps"] for row in subset],
                [row[metric_key] for row in subset],
                label=series,
                linewidth=2.1,
                markersize=6.5,
                **style,
            )
        ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel("End-to-end GPU-steps (log scale)")
        ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.75, which="both")
        ax.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("Success")
    axes[0].legend(loc="lower right", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / "success_vs_end_to_end_gpu_steps_log.png", dpi=220)
    fig.savefig(out_dir / "success_vs_end_to_end_gpu_steps_log.pdf")
    plt.close(fig)


def plot_frontier(rows: list[dict[str, Any]], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), sharey=True)
    metrics = [("overall_success", "Overall success"), ("retail_success", "Retail success")]
    toolrl = [row for row in rows if row["series"] == "ToolRL-Qwen"]
    toolrl.sort(key=lambda row: row["end_to_end_gpu_steps"])
    fp_aux = [row for row in rows if row["series"] == "FlowPlanner-Qwen" and row["method"] != "State-conditioned FM hint + strict selector"]
    fp_final = next(row for row in rows if row["method"] == "State-conditioned FM hint + strict selector" and row["series"] == "FlowPlanner-Qwen")
    llama = [row for row in rows if row["series"] == "ToolRL-Llama"]
    for ax, (metric_key, title) in zip(axes, metrics):
        ax.plot(
            [row["end_to_end_gpu_steps"] for row in toolrl],
            [row[metric_key] for row in toolrl],
            color="#d62728",
            marker="s",
            linewidth=2.1,
            markersize=6.5,
            label="ToolRL-Qwen",
        )
        ax.scatter(
            [row["end_to_end_gpu_steps"] for row in fp_aux],
            [row[metric_key] for row in fp_aux],
            color="#1f77b4",
            marker="o",
            s=50,
            alpha=0.35,
            label="FlowPlanner ablations",
        )
        ax.scatter(
            [fp_final["end_to_end_gpu_steps"]],
            [fp_final[metric_key]],
            color="#1f77b4",
            marker="*",
            s=180,
            label="FlowPlanner final",
            zorder=4,
        )
        for row in toolrl:
            ax.annotate(row["short"], (row["end_to_end_gpu_steps"], row[metric_key]), xytext=(12, 3), textcoords="offset points", fontsize=8.5)
        ax.annotate("Ours", (fp_final["end_to_end_gpu_steps"], fp_final[metric_key]), xytext=(14, 8), textcoords="offset points", fontsize=9)
        for row in llama:
            ax.scatter(row["end_to_end_gpu_steps"], row[metric_key], color="#7f7f7f", marker="^", s=70, label="ToolRL-Llama step100")
        ax.set_title(title)
        ax.set_xlabel("End-to-end GPU-steps")
        ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.75)
        ax.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("Success")
    handles, labels = axes[0].get_legend_handles_labels()
    dedup: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        dedup.setdefault(label, handle)
    axes[0].legend(dedup.values(), dedup.keys(), loc="lower right", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / "success_vs_compute_frontier.png", dpi=220)
    fig.savefig(out_dir / "success_vs_compute_frontier.pdf")
    plt.close(fig)


def plot_stage(rows: list[dict[str, Any]], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    stage_rows = [
        next(row for row in rows if row["method"] == "Compact-state SFT w/o FM"),
        next(row for row in rows if row["method"] == "FM prior text hint"),
        next(row for row in rows if row["method"] == "State-conditioned FM hint"),
        next(row for row in rows if row["method"] == "State-conditioned FM hint + strict selector"),
        next(row for row in rows if row["series"] == "ToolRL-Qwen" and row["method"] == "ToolRL GRPO step100"),
        next(row for row in rows if row["series"] == "ToolRL-Qwen" and row["method"] == "ToolRL GRPO step200"),
        next(row for row in rows if row["series"] == "ToolRL-Qwen" and row["method"] == "ToolRL GRPO step582"),
    ]
    xs = [0, 1, 2, 3, 5, 6, 7]
    labels = [
        "No FM\n240",
        "FM text\n520",
        "State FM\n520",
        "Ours\n520",
        "ToolRL 100\n300",
        "ToolRL 200\n600",
        "ToolRL 582\n1746",
    ]
    metrics = [("overall_success", "Overall success"), ("retail_success", "Retail success")]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3), sharey=True)
    for ax, (metric_key, title) in zip(axes, metrics):
        fp_x = xs[:4]
        fp_y = [row[metric_key] for row in stage_rows[:4]]
        tr_x = xs[4:]
        tr_y = [row[metric_key] for row in stage_rows[4:]]
        ax.plot(fp_x, fp_y, color="#1f77b4", marker="o", linewidth=2.1, markersize=6.5, label="FlowPlanner stages")
        ax.plot(tr_x, tr_y, color="#d62728", marker="s", linewidth=2.1, markersize=6.5, label="ToolRL checkpoints")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("Stage / end-to-end GPU-steps")
        ax.grid(True, axis="y", linestyle=":", linewidth=0.8, alpha=0.75)
        ax.axvline(4, color="#aaaaaa", linewidth=1, linestyle="--")
        ax.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("Success")
    axes[0].legend(loc="lower right", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / "success_by_training_stage.png", dpi=220)
    fig.savefig(out_dir / "success_by_training_stage.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "processed" / "compute_efficiency"))
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_points(root)
    (out_dir / "compute_efficiency.json").write_text(
        json.dumps({"points": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_md(rows, out_dir / "COMPUTE_EFFICIENCY.md")
    plot_paper_efficiency(rows, out_dir)
    plot(rows, out_dir, "llm_gpu_steps", "success_vs_llm_gpu_steps", "LLM GPU-steps")
    plot(rows, out_dir, "end_to_end_gpu_steps", "success_vs_end_to_end_gpu_steps", "End-to-end GPU-steps")
    plot_log(rows, out_dir)
    plot_frontier(rows, out_dir)
    plot_stage(rows, out_dir)
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
