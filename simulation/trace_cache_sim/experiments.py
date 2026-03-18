"""Experiment runner for the trace-driven sparse prefix caching simulator."""

from __future__ import annotations

import os
from pathlib import Path

from .policies import (
    BoundedCachePolicy,
    BranchOnlyPolicy,
    BranchPlusGeometricPolicy,
    DenseEveryKPolicy,
    ExponentialHistogramPolicy,
    GeometricBaseRPolicy,
    GeometricEpsPolicy,
    NoCachePolicy,
    OfflineDPOptimalPolicy,
    OnlineHistogramPolicy,
    UniformBudgetMPolicy,
)
from .simulator import SimulationConfig, SimulationResult, simulate_trace
from .toy_recurrence import checkpoint_resume_error
from .traces import Request, build_trace_families


REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
(REPO_ROOT / ".mplconfig").mkdir(exist_ok=True)

import matplotlib
import numpy as np
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import ticker  # noqa: E402


POLICY_ORDER = [
    "no_cache",
    "dense_every_k",
    "uniform_budget_m",
    "geometric_base_r",
    "geometric_eps",
    "branch_only",
    "branch_plus_geometric",
    "offline_dp_optimal",
    "online_histogram_policy",
    "exp_histogram_policy",
]


def build_policies(chunk_size: int) -> list:
    return [
        NoCachePolicy(chunk_size=chunk_size),
        DenseEveryKPolicy(every_k=32, chunk_size=chunk_size),
        UniformBudgetMPolicy(budget=16, chunk_size=chunk_size),
        GeometricBaseRPolicy(base=2.0, chunk_size=chunk_size),
        GeometricEpsPolicy(eps=0.35, chunk_size=chunk_size),
        BranchOnlyPolicy(chunk_size=chunk_size),
        BranchPlusGeometricPolicy(base=2.0, chunk_size=chunk_size),
        OfflineDPOptimalPolicy(budget=16, chunk_size=chunk_size),
        OnlineHistogramPolicy(budget=16, update_every=8, chunk_size=chunk_size),
        ExponentialHistogramPolicy(budget=16, update_every=8, decay=0.95, chunk_size=chunk_size),
    ]


def policy_label(name: str) -> str:
    return {
        "no_cache": "none",
        "dense_every_k": "dense-k",
        "uniform_budget_m": "uniform",
        "geometric_base_r": "geom-r",
        "geometric_eps": "geom-eps",
        "branch_only": "branch",
        "branch_plus_geometric": "branch+geom",
        "offline_dp_optimal": "offline-dp",
        "online_histogram_policy": "online-hist",
        "exp_histogram_policy": "exp-hist",
    }[name]


def policy_color_map() -> dict[str, tuple[float, float, float]]:
    colors = sns.color_palette("tab10", n_colors=len(POLICY_ORDER))
    return {policy: color for policy, color in zip(POLICY_ORDER, colors)}


def short_bytes_label(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        rounded = value / 1_000
        if rounded >= 100:
            return f"{rounded:.0f}k"
        if rounded >= 10:
            return f"{rounded:.0f}k"
        return f"{rounded:.1f}k"
    return f"{value:.0f}"


def byte_tick_formatter(value: float, _: float) -> str:
    return short_bytes_label(value)


def exactness_suite(traces: dict[str, list[Request]]) -> dict[str, float]:
    errors: dict[str, float] = {}
    for family, requests in traces.items():
        sample = requests[min(2, len(requests) - 1)].tokens
        checkpoint = max(1, len(sample) // 2)
        errors[family] = checkpoint_resume_error(sample, checkpoint)
    return errors


def eviction_sweep(
    traces: dict[str, list[Request]],
    config: SimulationConfig,
    figure_dir: Path,
) -> list[dict[str, float | str | int]]:
    """Run cache eviction sweep: vary max cache entries for key policies."""
    base_specs: list[tuple[str, type, dict]] = [
        ("branch_only", BranchOnlyPolicy, {"chunk_size": config.chunk_size}),
        ("offline_dp", OfflineDPOptimalPolicy, {"budget": 16, "chunk_size": config.chunk_size}),
        ("dense_every_k", DenseEveryKPolicy, {"every_k": 32, "chunk_size": config.chunk_size}),
    ]
    cache_budgets = [4, 8, 16, 32, 64, 128]
    records: list[dict[str, float | str | int]] = []

    for family, requests in traces.items():
        for label, cls, kwargs in base_specs:
            base = cls(**kwargs)
            result = simulate_trace(requests, base, config)
            records.append({
                "family": family,
                "policy": label,
                "max_entries": 9999,
                "avg_token_hit": result.summary.avg_token_hit_ratio,
                "avg_hybrid_cost": result.summary.avg_hybrid_cost,
            })
            for max_ent in cache_budgets:
                base = cls(**kwargs)
                bounded = BoundedCachePolicy(base, max_entries=max_ent, chunk_size=config.chunk_size)
                result = simulate_trace(requests, bounded, config)
                records.append({
                    "family": family,
                    "policy": label,
                    "max_entries": max_ent,
                    "avg_token_hit": result.summary.avg_token_hit_ratio,
                    "avg_hybrid_cost": result.summary.avg_hybrid_cost,
                })

    plot_eviction_sweep(records, list(traces.keys()), figure_dir / "trace_eviction_sweep.pdf")
    return records


def plot_eviction_sweep(
    records: list[dict[str, float | str | int]],
    families: list[str],
    filename: Path,
) -> None:
    policy_colors = {"branch_only": "#0a9396", "offline_dp": "#ee9b00", "dense_every_k": "#ca6702"}
    families_plus = families + ["all_families"]
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 7.6))
    axes_flat = axes.flatten()
    legend_handles = []

    for ax, family in zip(axes_flat, families_plus):
        family_rows = records if family == "all_families" else [r for r in records if r["family"] == family]
        for policy_name in ["branch_only", "offline_dp", "dense_every_k"]:
            rows = sorted(
                [r for r in family_rows if r["policy"] == policy_name],
                key=lambda r: int(r["max_entries"]),
            )
            if not rows:
                continue
            xs = [int(r["max_entries"]) for r in rows]
            ys = [float(r["avg_token_hit"]) for r in rows]
            handle, = ax.plot(
                xs, ys, marker="o", markersize=5, linewidth=2,
                label=policy_name.replace("_", " "), color=policy_colors[policy_name],
            )
            if family == families_plus[0]:
                legend_handles.append(handle)
        title = "all families" if family == "all_families" else family.replace("_", "\n")
        ax.set_title(title, fontsize=13)
        ax.set_xscale("log")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        if family == "all_families":
            ax.set_facecolor("#f7f7f7")

    for ax in axes[0, :]:
        ax.tick_params(labelbottom=False)
    if len(families_plus) < len(axes_flat):
        for ax in axes_flat[len(families_plus):]:
            ax.set_visible(False)

    fig.legend(
        legend_handles,
        ["branch only", "offline dp", "dense every-k"],
        loc="lower center", ncol=3, frameon=False, fontsize=10.5, bbox_to_anchor=(0.5, 0.01),
    )
    fig.supxlabel("Max cache entries (log scale)", fontsize=13, y=0.07)
    fig.supylabel("Average token-hit ratio", fontsize=13, x=0.04)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.91, bottom=0.17, hspace=0.16, wspace=0.18)
    plt.savefig(filename)
    plt.close(fig)


def ssm_fraction_sweep(
    traces: dict[str, list[Request]],
    figure_dir: Path,
) -> list[dict[str, float | str]]:
    """Sweep SSM layer fraction and measure sparse caching benefit."""
    fractions = np.linspace(0.1, 1.0, 10)
    records: list[dict[str, float | str]] = []

    for ssm_frac in fractions:
        config = SimulationConfig(
            a_rec=1.0 * ssm_frac,
            a_attn=0.002 * (1.0 - ssm_frac),
        )
        for family, requests in traces.items():
            no_cache_result = simulate_trace(
                requests, NoCachePolicy(chunk_size=config.chunk_size), config,
            )
            dp_result = simulate_trace(
                requests, OfflineDPOptimalPolicy(budget=16, chunk_size=config.chunk_size), config,
            )
            branch_result = simulate_trace(
                requests, BranchOnlyPolicy(chunk_size=config.chunk_size), config,
            )
            no_cost = no_cache_result.summary.avg_hybrid_cost
            best_cost = min(dp_result.summary.avg_hybrid_cost, branch_result.summary.avg_hybrid_cost)
            reduction = 1.0 - best_cost / max(no_cost, 1e-12) if no_cost > 1e-12 else 0.0
            records.append({
                "family": family,
                "ssm_fraction": float(ssm_frac),
                "no_cache_cost": no_cost,
                "best_sparse_cost": best_cost,
                "cost_reduction": reduction,
            })

    plot_ssm_fraction_sweep(records, list(traces.keys()), figure_dir / "trace_ssm_fraction_sweep.pdf")
    return records


def plot_ssm_fraction_sweep(
    records: list[dict[str, float | str]],
    families: list[str],
    filename: Path,
) -> None:
    plt.figure(figsize=(8.5, 5.4))
    colors = sns.color_palette("Set2", n_colors=len(families))
    for color, family in zip(colors, families):
        rows = sorted(
            [r for r in records if r["family"] == family],
            key=lambda r: float(r["ssm_fraction"]),
        )
        xs = [float(r["ssm_fraction"]) for r in rows]
        ys = [100.0 * float(r["cost_reduction"]) for r in rows]
        plt.plot(xs, ys, marker="o", label=family.replace("_", " "), color=color, linewidth=2, markersize=5)

    plt.xlabel("SSM layer fraction")
    plt.ylabel("Cost reduction from sparse caching (%)")
    plt.title("Sparse checkpoint benefit scales with SSM fraction")
    plt.legend(frameon=False, fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def run_all_experiments() -> tuple[list[dict[str, float | str]], dict[str, float]]:
    sns.set_theme(style="whitegrid", context="talk")
    figure_dir = REPO_ROOT / "figures"
    figure_dir.mkdir(exist_ok=True)
    summary_path = REPO_ROOT / "trace_driven_summary.md"

    config = SimulationConfig()
    traces = build_trace_families(seed=42)
    exactness = exactness_suite(traces)

    records: list[dict[str, float | str]] = []
    all_results: dict[tuple[str, str], SimulationResult] = {}
    for family, requests in traces.items():
        for policy in build_policies(config.chunk_size):
            result = simulate_trace(requests, policy, config)
            all_results[(family, policy.name)] = result
            summary = result.summary
            records.append(
                {
                    "family": family,
                    "policy": policy.name,
                    "policy_label": policy_label(policy.name),
                    "avg_recompute_tokens": summary.avg_recompute_tokens,
                    "max_recompute_tokens": summary.max_recompute_tokens,
                    "avg_token_hit_ratio": summary.avg_token_hit_ratio,
                    "avg_alignment_error": summary.avg_alignment_error,
                    "avg_hybrid_cost": summary.avg_hybrid_cost,
                    "avg_recurrent_cost": summary.avg_recurrent_cost,
                    "avg_checkpoint_count": summary.avg_checkpoint_count,
                    "max_checkpoint_count": summary.max_checkpoint_count,
                    "max_recurrent_bytes": summary.max_recurrent_bytes,
                    "max_total_bytes": summary.max_total_bytes,
                    "avg_bytes_written_total": summary.avg_bytes_written_total,
                    "avg_bytes_read_total": summary.avg_bytes_read_total,
                    "exactness_error": exactness[family],
                }
            )

    no_cache_map = {family: all_results[(family, "no_cache")] for family in traces}
    branch_map = {family: all_results[(family, "branch_only")] for family in traces}
    for record in records:
        family = str(record["family"])
        policy = str(record["policy"])
        metrics = all_results[(family, policy)].request_metrics
        no_metrics = no_cache_map[family].request_metrics
        branch_metrics = branch_map[family].request_metrics
        wins_vs_no = sum(metric.hybrid_cost < base.hybrid_cost for metric, base in zip(metrics, no_metrics))
        wins_vs_branch = sum(metric.hybrid_cost < base.hybrid_cost for metric, base in zip(metrics, branch_metrics))
        record["win_rate_vs_no_cache"] = wins_vs_no / len(metrics)
        record["win_rate_vs_branch_only"] = wins_vs_branch / len(metrics)
        record["hybrid_cost_ratio_vs_no_cache"] = record["avg_hybrid_cost"] / max(no_cache_map[family].summary.avg_hybrid_cost, 1e-12)

    plot_memory_vs_metric(
        records,
        families=list(traces.keys()),
        x_key="max_recurrent_bytes",
        y_key="avg_token_hit_ratio",
        y_label="Average token-hit ratio",
        filename=figure_dir / "trace_memory_vs_token_hit.pdf",
        y_limits=(0.5, 1.0),
    )
    plot_memory_vs_metric(
        records,
        families=list(traces.keys()),
        x_key="max_recurrent_bytes",
        y_key="avg_recompute_tokens",
        y_label="Average recompute tokens",
        filename=figure_dir / "trace_memory_vs_expected_recompute.pdf",
        y_limits=(0.0, 100.0),
    )
    plot_memory_vs_metric(
        records,
        families=list(traces.keys()),
        x_key="max_total_bytes",
        y_key="avg_hybrid_cost",
        y_label="Average hybrid proxy TTFT cost",
        filename=figure_dir / "trace_memory_vs_estimated_ttft.pdf",
        y_limits=(0.0, 250.0),
        per_family_y_limits={"agent_tree": (0.0, 60.0)},
    )
    plot_trace_family_heatmap(records, list(traces.keys()), figure_dir / "trace_family_comparisons.pdf")

    eviction_records = eviction_sweep(traces, config, figure_dir)
    ssm_records = ssm_fraction_sweep(traces, figure_dir)

    write_markdown_summary(records, exactness, summary_path)
    print_summary(records, exactness)
    print()
    print("Eviction sweep (sample):")
    for row in eviction_records:
        if row["family"] == "exact_hot_prefix":
            print(f"  {row['policy']:>15} max_entries={int(row['max_entries']):>5} hit={float(row['avg_token_hit']):.3f}")
    print()
    print("SSM fraction sweep (sample):")
    for row in ssm_records:
        if row["family"] == "diffuse_cutpoints":
            print(f"  ssm_frac={float(row['ssm_fraction']):.2f} reduction={100*float(row['cost_reduction']):.1f}%")
    return records, exactness


def plot_memory_vs_metric(
    records: list[dict[str, float | str]],
    families: list[str],
    x_key: str,
    y_key: str,
    y_label: str,
    filename: Path,
    y_limits: tuple[float, float] | None = None,
    per_family_y_limits: dict[str, tuple[float, float]] | None = None,
) -> None:
    families_with_all = families + ["all_families"]
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 7.6))
    axes_flat = axes.flatten()
    colors = policy_color_map()
    legend_handles = []
    per_family_y_limits = per_family_y_limits or {}

    for axis, family in zip(axes_flat, families_with_all):
        family_rows = records if family == "all_families" else [row for row in records if row["family"] == family]
        for policy in POLICY_ORDER:
            matching_rows = [row for row in family_rows if row["policy"] == policy]
            if not matching_rows:
                continue
            color = colors[policy]
            if family == "all_families":
                xs = [float(row[x_key]) for row in matching_rows]
                ys = [float(row[y_key]) for row in matching_rows]
                handle = axis.scatter(xs, ys, s=55, color=color, alpha=0.75, edgecolors="white", linewidths=0.4)
            else:
                row = matching_rows[0]
                x = float(row[x_key])
                y = float(row[y_key])
                handle = axis.scatter(x, y, s=80, color=color, edgecolors="white", linewidths=0.5)
            if family == families_with_all[0]:
                legend_handles.append(handle)

        title = "all families" if family == "all_families" else family.replace("_", "\n")
        axis.set_title(title, fontsize=13)
        axis.tick_params(labelsize=11)
        axis.set_xscale("log")
        axis.xaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=4))
        axis.xaxis.set_major_formatter(ticker.FuncFormatter(byte_tick_formatter))
        axis.xaxis.set_minor_locator(ticker.NullLocator())
        axis.ticklabel_format(axis="y", style="plain")
        axis.grid(alpha=0.3)
        if family in per_family_y_limits:
            axis.set_ylim(*per_family_y_limits[family])
        elif y_limits is not None:
            axis.set_ylim(*y_limits)
        if family == "all_families":
            axis.set_facecolor("#f7f7f7")

    for axis in axes[0, :]:
        axis.tick_params(labelbottom=False)

    fig.legend(
        legend_handles,
        [policy_label(policy) for policy in POLICY_ORDER],
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=10.5,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.supxlabel("Max recurrent bytes" if x_key == "max_recurrent_bytes" else "Max total bytes", fontsize=13, y=0.07)
    fig.supylabel(y_label, fontsize=13, x=0.04)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.91, bottom=0.17, hspace=0.16, wspace=0.18)
    plt.savefig(filename)
    plt.close(fig)


def plot_trace_family_heatmap(records: list[dict[str, float | str]], families: list[str], filename: Path) -> None:
    policies = POLICY_ORDER
    matrix = np.zeros((len(policies), len(families)))
    for i, policy in enumerate(policies):
        for j, family in enumerate(families):
            row = next(item for item in records if item["family"] == family and item["policy"] == policy)
            matrix[i, j] = 100.0 * float(row["win_rate_vs_branch_only"])
    with sns.plotting_context("paper", font_scale=1.05):
        plt.figure(figsize=(7.7, 4.2))
        ax = sns.heatmap(
            matrix,
            annot=True,
            fmt=".1f",
            cmap="YlOrBr",
            xticklabels=[family.replace("_", "\n") for family in families],
            yticklabels=[policy_label(name) for name in policies],
            annot_kws={"size": 7.5},
            cbar_kws={"label": "Request win-rate vs branch-only (%)"},
        )
    ax.set_title("Per-family win-rate against the branch-only baseline", fontsize=10.5, pad=6)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelrotation=25, labelsize=7.5)
    ax.tick_params(axis="y", labelrotation=0, labelsize=7.5)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def write_markdown_summary(records: list[dict[str, float | str]], exactness: dict[str, float], path: Path) -> None:
    best_by_family: list[str] = []
    for family in sorted(exactness):
        family_rows = [row for row in records if row["family"] == family and row["policy"] != "no_cache"]
        best = min(family_rows, key=lambda row: float(row["avg_hybrid_cost"]))
        best_by_family.append(
            f"- `{family}`: best hybrid proxy cost from `{best['policy']}` with "
            f"token-hit {100.0 * float(best['avg_token_hit_ratio']):.1f}% and "
            f"alignment error {float(best['avg_alignment_error']):.2f}."
        )

    lines = [
        "# Trace-Driven Sparse Prefix Caching Summary",
        "",
        "## Assumptions",
        "",
        "- Requests are explicit token sequences; overlap is exact longest common prefix.",
        "- Checkpoints are exact recurrent states; resume always recomputes the missing suffix exactly.",
        "- Memory is tracked separately for recurrent checkpoints and dense attention KV storage induced by cached prefix depth.",
        "",
        "## Main Findings",
        "",
        *best_by_family,
        "",
        "## Exactness",
        "",
    ]
    for family, error in exactness.items():
        lines.append(f"- `{family}`: max checkpoint-resume error `{error:.3e}`.")
    lines.extend(
        [
            "",
            "## Publication signal",
            "",
            "- The trace-driven simulator suggests sparse checkpoint placement is not just a tuning knob.",
            "- Branch-aware caching matters on tree-like workloads.",
            "- Total hybrid memory can remain large even when recurrent checkpoint memory is sparse.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def print_summary(records: list[dict[str, float | str]], exactness: dict[str, float]) -> None:
    print("Trace-driven sparse prefix caching experiments")
    print("=" * 52)
    for row in records:
        print(
            f"[{row['family']}] {str(row['policy']).rjust(22)} | "
            f"hit={100.0 * float(row['avg_token_hit_ratio']):6.2f}% | "
            f"recompute={float(row['avg_recompute_tokens']):7.2f} | "
            f"hybrid={float(row['avg_hybrid_cost']):9.2f} | "
            f"recurrent_mem={int(row['max_recurrent_bytes']):8d} | "
            f"total_mem={int(row['max_total_bytes']):9d} | "
            f"align={float(row['avg_alignment_error']):5.2f}"
        )
    print()
    print("Toy exactness:")
    for family, error in exactness.items():
        print(f"  {family}: max error={error:.3e}")


if __name__ == "__main__":
    run_all_experiments()
