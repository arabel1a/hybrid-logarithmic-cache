#!/usr/bin/env python3
"""Sparse prefix caching simulation and theoretical validation for SSM checkpoints."""

from __future__ import annotations

import argparse
import math
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
(REPO_ROOT / ".mplconfig").mkdir(exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns


SEED = 42
EPS = 1e-12


@dataclass(frozen=True)
class StrategyResult:
    name: str
    checkpoints: list[int]
    expected_cost: float
    monte_carlo_cost: float
    savings: float
    relative_error: float


def dedupe_sorted_positions(positions: Iterable[int], prefix_length: int, budget: int) -> list[int]:
    clipped = [min(prefix_length, max(1, int(p))) for p in positions]
    unique = sorted(set(clipped))
    if budget <= 0:
        return []
    if len(unique) <= budget:
        return unique
    # Preserve overall coverage when rounding produces too many candidates.
    indices = np.linspace(0, len(unique) - 1, num=budget)
    return sorted({unique[int(round(i))] for i in indices})


class CheckpointStrategy:
    """Base class for checkpoint placement strategies."""

    name = "base"

    def place_checkpoints(self, prefix_length: int, budget: int) -> list[int]:
        raise NotImplementedError


class UniformStrategy(CheckpointStrategy):
    """Place checkpoints at evenly spaced positions."""

    name = "uniform"

    def place_checkpoints(self, prefix_length: int, budget: int) -> list[int]:
        if budget <= 0:
            return []
        if budget >= prefix_length:
            return list(range(1, prefix_length + 1))
        positions = np.linspace(1, prefix_length, num=budget + 2)[1:-1]
        return dedupe_sorted_positions(np.rint(positions), prefix_length, budget)


class PowerOfTwoStrategy(CheckpointStrategy):
    """Place checkpoints on a geometric grid; powers-of-two are recovered near log2(N) budget."""

    name = "power_of_two"

    def place_checkpoints(self, prefix_length: int, budget: int) -> list[int]:
        if budget <= 0:
            return []
        if budget >= prefix_length:
            return list(range(1, prefix_length + 1))
        if budget == 1:
            return [1]
        positions = np.geomspace(1, prefix_length, num=budget)
        return dedupe_sorted_positions(np.rint(positions), prefix_length, budget)


class SqrtStrategy(CheckpointStrategy):
    """Place checkpoints on the canonical sqrt(N)-spaced grid, capped or densified to match the budget."""

    name = "sqrt"

    def place_checkpoints(self, prefix_length: int, budget: int) -> list[int]:
        if budget <= 0:
            return []
        if budget >= prefix_length:
            return list(range(1, prefix_length + 1))
        step = max(1, int(round(math.sqrt(prefix_length))))
        natural = np.arange(step, prefix_length + 1, step)
        if len(natural) >= budget:
            indices = np.linspace(0, len(natural) - 1, num=budget)
            return dedupe_sorted_positions(np.rint(natural[indices.astype(int)]), prefix_length, budget)
        extra = np.linspace(1, prefix_length, num=budget)
        positions = np.concatenate([natural, extra])
        return dedupe_sorted_positions(np.rint(positions), prefix_length, budget)


class ConvexHullTrick:
    """Monotone lower hull for lines queried at nondecreasing x."""

    def __init__(self) -> None:
        self.lines: deque[tuple[float, float, int]] = deque()

    @staticmethod
    def _value(line: tuple[float, float, int], x: float) -> float:
        m, b, _ = line
        return m * x + b

    @staticmethod
    def _is_redundant(
        l1: tuple[float, float, int],
        l2: tuple[float, float, int],
        l3: tuple[float, float, int],
    ) -> bool:
        m1, b1, _ = l1
        m2, b2, _ = l2
        m3, b3, _ = l3
        return (b3 - b1) * (m1 - m2) <= (b2 - b1) * (m1 - m3) + EPS

    def add_line(self, slope: float, intercept: float, arg: int) -> None:
        line = (slope, intercept, arg)
        while len(self.lines) >= 2 and self._is_redundant(self.lines[-2], self.lines[-1], line):
            self.lines.pop()
        self.lines.append(line)

    def query(self, x: float) -> tuple[float, int]:
        while len(self.lines) >= 2 and self._value(self.lines[0], x) >= self._value(self.lines[1], x) - EPS:
            self.lines.popleft()
        best = self.lines[0]
        return self._value(best, x), best[2]


class AdaptiveStrategy(CheckpointStrategy):
    """Find optimal checkpoint positions for an empirical divergence distribution via exact DP."""

    name = "adaptive"

    def __init__(self, divergence_distribution: np.ndarray):
        distribution = np.asarray(divergence_distribution, dtype=float)
        if distribution.ndim != 1 or distribution.size < 2:
            raise ValueError("distribution must be a 1D array indexed from 0..N")
        total = distribution[1:].sum()
        if total <= 0:
            raise ValueError("distribution mass over positions 1..N must be positive")
        self.p = distribution.copy()
        self.p[0] = 0.0
        self.p[1:] /= total

    def place_checkpoints(self, prefix_length: int, budget: int) -> list[int]:
        if budget <= 0:
            return []
        budget = min(budget, prefix_length)
        if prefix_length != len(self.p) - 1:
            raise ValueError("prefix_length must match the adaptive distribution length")
        if budget >= prefix_length:
            return list(range(1, prefix_length + 1))

        prefix_prob = np.zeros(prefix_length + 1, dtype=float)
        prefix_tp = np.zeros(prefix_length + 1, dtype=float)
        positions = np.arange(prefix_length + 1, dtype=float)
        prefix_prob[1:] = np.cumsum(self.p[1:])
        prefix_tp[1:] = np.cumsum(self.p[1:] * positions[1:])

        dp_prev = prefix_tp.copy()
        backpointers = np.zeros((budget + 1, prefix_length + 1), dtype=np.int32)

        for m in range(1, budget + 1):
            dp_curr = np.zeros(prefix_length + 1, dtype=float)
            dp_curr[0] = 0.0
            hull = ConvexHullTrick()
            hull.add_line(-1.0, dp_prev[0], 1)
            for j in range(1, prefix_length + 1):
                s = j
                intercept = dp_prev[s - 1] - prefix_tp[s - 1] + s * prefix_prob[s - 1]
                hull.add_line(-float(s), float(intercept), s)
                line_value, argmin_s = hull.query(prefix_prob[j])
                dp_curr[j] = prefix_tp[j] + line_value
                backpointers[m, j] = argmin_s
            dp_prev = dp_curr

        checkpoints: list[int] = []
        j = prefix_length
        for m in range(budget, 0, -1):
            s = int(backpointers[m, j])
            if s <= 0:
                break
            checkpoints.append(s)
            j = s - 1
        return sorted(set(checkpoints))


def normalize_distribution(probabilities: np.ndarray) -> np.ndarray:
    distribution = np.asarray(probabilities, dtype=float)
    distribution[0] = 0.0
    mass = distribution[1:].sum()
    if mass <= 0:
        raise ValueError("distribution has zero probability mass")
    distribution[1:] /= mass
    return distribution


def uniform_distribution(prefix_length: int) -> np.ndarray:
    distribution = np.zeros(prefix_length + 1, dtype=float)
    distribution[1:] = 1.0 / prefix_length
    return distribution


def zipf_distribution(prefix_length: int, alpha: float = 1.2) -> np.ndarray:
    positions = np.arange(1, prefix_length + 1, dtype=float)
    distribution = np.zeros(prefix_length + 1, dtype=float)
    distribution[1:] = 1.0 / np.power(positions, alpha)
    return normalize_distribution(distribution)


def gaussian_bump(prefix_length: int, center: float, width: float) -> np.ndarray:
    positions = np.arange(1, prefix_length + 1, dtype=float)
    return np.exp(-0.5 * np.square((positions - center) / max(width, 1.0)))


def bimodal_distribution(prefix_length: int) -> np.ndarray:
    early = gaussian_bump(prefix_length, 0.2 * prefix_length, 0.06 * prefix_length)
    late = gaussian_bump(prefix_length, 0.82 * prefix_length, 0.05 * prefix_length)
    distribution = np.zeros(prefix_length + 1, dtype=float)
    distribution[1:] = 0.55 * early + 0.45 * late
    return normalize_distribution(distribution)


def realistic_workload_distribution(prefix_length: int, shared_system_prompt: int = 500) -> np.ndarray:
    positions = np.arange(1, prefix_length + 1, dtype=float)
    shoulder = gaussian_bump(prefix_length, min(shared_system_prompt + 120, prefix_length), 90)
    long_context = gaussian_bump(prefix_length, 0.7 * prefix_length, 0.12 * prefix_length)
    tail = np.exp(-(prefix_length - positions) / max(prefix_length / 8.0, 1.0))
    distribution = np.zeros(prefix_length + 1, dtype=float)
    distribution[1:] = 0.5 * shoulder + 0.3 * long_context + 0.2 * tail
    return normalize_distribution(distribution)


def late_peak_distribution(prefix_length: int) -> np.ndarray:
    distribution = np.zeros(prefix_length + 1, dtype=float)
    distribution[1:] = gaussian_bump(prefix_length, 0.88 * prefix_length, 0.05 * prefix_length)
    return normalize_distribution(distribution)


def expected_recomputation_cost(distribution: np.ndarray, checkpoints: list[int]) -> float:
    prefix_length = len(distribution) - 1
    latest = np.zeros(prefix_length + 1, dtype=int)
    if checkpoints:
        checkpoints_array = np.asarray(sorted(set(checkpoints)), dtype=int)
        latest[checkpoints_array] = checkpoints_array
        latest = np.maximum.accumulate(latest)
    positions = np.arange(prefix_length + 1, dtype=float)
    costs = positions - latest
    return float(np.dot(distribution[1:], costs[1:]))


def monte_carlo_cost(
    distribution: np.ndarray,
    checkpoints: list[int],
    rng: np.random.Generator,
    samples: int,
) -> float:
    prefix_length = len(distribution) - 1
    positions = np.arange(prefix_length + 1)
    draws = rng.choice(positions[1:], size=samples, p=distribution[1:])
    latest = np.zeros(prefix_length + 1, dtype=int)
    if checkpoints:
        checkpoints_array = np.asarray(sorted(set(checkpoints)), dtype=int)
        latest[checkpoints_array] = checkpoints_array
        latest = np.maximum.accumulate(latest)
    costs = draws - latest[draws]
    return float(np.mean(costs))


def baseline_cost(distribution: np.ndarray) -> float:
    positions = np.arange(len(distribution), dtype=float)
    return float(np.dot(distribution[1:], positions[1:]))


def empirical_distribution_from_draws(draws: np.ndarray, prefix_length: int) -> np.ndarray:
    counts = np.bincount(draws, minlength=prefix_length + 1).astype(float)
    return normalize_distribution(counts)


def compute_savings(distribution: np.ndarray, checkpoints: list[int]) -> float:
    base = baseline_cost(distribution)
    return 1.0 - expected_recomputation_cost(distribution, checkpoints) / max(base, EPS)


def theoretical_uniform_savings(memory: np.ndarray) -> np.ndarray:
    return memory / (memory + 1.0)


def evaluate_strategy(
    strategy: CheckpointStrategy,
    distribution: np.ndarray,
    prefix_length: int,
    budget: int,
    rng: np.random.Generator,
    samples: int,
) -> StrategyResult:
    checkpoints = strategy.place_checkpoints(prefix_length, budget)
    expected_cost = expected_recomputation_cost(distribution, checkpoints)
    monte_carlo = monte_carlo_cost(distribution, checkpoints, rng, samples=samples)
    base = baseline_cost(distribution)
    savings = 1.0 - expected_cost / max(base, EPS)
    relative_error = abs(monte_carlo - expected_cost) / max(expected_cost, EPS)
    return StrategyResult(
        name=strategy.name,
        checkpoints=checkpoints,
        expected_cost=expected_cost,
        monte_carlo_cost=monte_carlo,
        savings=savings,
        relative_error=relative_error,
    )


def format_checkpoints(checkpoints: list[int], limit: int = 8) -> str:
    if len(checkpoints) <= limit:
        return str(checkpoints)
    head = ", ".join(str(x) for x in checkpoints[: limit // 2])
    tail = ", ".join(str(x) for x in checkpoints[-(limit // 2) :])
    return f"[{head}, ..., {tail}]"


def latex_results_table(rows: list[dict[str, object]]) -> str:
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Distribution & Strategy & Budget & Used & Savings (\%) & RelErr (\%) \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['distribution']} & {row['strategy']} & {row['budget']} & {row['used']} & "
            f"{100.0 * row['savings']:.2f} & {100.0 * row['relative_error']:.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def exact_match_savings(distribution: np.ndarray) -> float:
    return float(distribution[-1])


def plot_pareto_frontier(
    figure_dir: Path,
    prefix_length: int,
    budgets: np.ndarray,
    strategies: list[CheckpointStrategy],
    distribution: np.ndarray,
) -> list[dict[str, float | str]]:
    records: list[dict[str, float | str]] = []
    for strategy in strategies:
        for budget in budgets:
            checkpoints = strategy.place_checkpoints(prefix_length, int(budget))
            records.append(
                {
                    "strategy": strategy.name,
                    "budget": int(budget),
                    "used_memory": len(checkpoints),
                    "savings": 100.0 * compute_savings(distribution, checkpoints),
                }
            )

    plt.figure(figsize=(8.5, 5.4))
    palette = sns.color_palette("Set2", n_colors=len(strategies))
    for color, strategy in zip(palette, strategies):
        xs = [r["used_memory"] for r in records if r["strategy"] == strategy.name]
        ys = [r["savings"] for r in records if r["strategy"] == strategy.name]
        plt.plot(xs, ys, marker="o", linewidth=2, markersize=4, label=strategy.name, color=color)

    theoretical_x = np.unique(np.array([r["used_memory"] for r in records if r["used_memory"] > 0], dtype=float))
    plt.plot(
        theoretical_x,
        100.0 * theoretical_uniform_savings(theoretical_x),
        linestyle="--",
        linewidth=2,
        color="#333333",
        label="uniform theory",
    )
    plt.axhline(50.0, linestyle="--", linewidth=1.5, color="#9b2226", label="power-of-two 50% bound")
    plt.scatter([prefix_length], [100.0], color="#000000", marker="*", s=100, label="full cache")
    plt.xscale("log")
    plt.xlabel("Checkpoint slots used")
    plt.ylabel("Compute savings (%)")
    plt.title("Pareto frontier under uniform divergence")
    plt.ylim(0, 103)
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(figure_dir / "pareto_frontier_uniform.pdf")
    plt.close()
    return records


def plot_strategy_heatmap(
    figure_dir: Path,
    prefix_length: int,
    budget: int,
    strategies: list[CheckpointStrategy],
    distributions: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    matrix: dict[str, dict[str, float]] = {strategy.name: {} for strategy in strategies}
    for strategy in strategies:
        for dist_name, distribution in distributions.items():
            checkpoints = strategy.place_checkpoints(prefix_length, budget)
            score = compute_savings(distribution, checkpoints) / max(len(checkpoints), 1)
            matrix[strategy.name][dist_name] = 100.0 * score

    heatmap_data = np.array(
        [[matrix[strategy.name][dist_name] for dist_name in distributions] for strategy in strategies],
        dtype=float,
    )
    plt.figure(figsize=(7.4, 4.2))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        xticklabels=list(distributions.keys()),
        yticklabels=[strategy.name for strategy in strategies],
        cbar_kws={"label": "Savings per checkpoint slot (%)"},
    )
    plt.title(f"Strategy efficiency at budget M={budget}")
    plt.tight_layout()
    plt.savefig(figure_dir / "strategy_heatmap.pdf")
    plt.close()
    return matrix


def simulate_online_adaptation(
    true_distribution: np.ndarray,
    prefix_length: int,
    budget: int,
    horizon: int,
    update_every: int,
    trials: int,
    rng_seed: int,
    decay: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    oracle_checkpoints = AdaptiveStrategy(true_distribution).place_checkpoints(prefix_length, budget)
    oracle_cost = np.arange(prefix_length + 1) - np.maximum.accumulate(
        np.bincount(oracle_checkpoints, weights=oracle_checkpoints, minlength=prefix_length + 1).astype(int)
    )
    all_regrets = np.zeros((trials, horizon), dtype=float)

    for trial in range(trials):
        rng = np.random.default_rng(rng_seed + trial)
        empirical_counts = np.ones(prefix_length + 1, dtype=float)
        current_strategy: CheckpointStrategy = PowerOfTwoStrategy()
        current_checkpoints = current_strategy.place_checkpoints(prefix_length, budget)
        latest_current = np.zeros(prefix_length + 1, dtype=int)
        latest_current[current_checkpoints] = current_checkpoints
        latest_current = np.maximum.accumulate(latest_current)

        latest_oracle = np.zeros(prefix_length + 1, dtype=int)
        latest_oracle[oracle_checkpoints] = oracle_checkpoints
        latest_oracle = np.maximum.accumulate(latest_oracle)

        cumulative = 0.0
        for t in range(horizon):
            draw = int(rng.choice(np.arange(1, prefix_length + 1), p=true_distribution[1:]))
            observed_cost = draw - latest_current[draw]
            best_cost = draw - latest_oracle[draw]
            cumulative += observed_cost - best_cost
            all_regrets[trial, t] = cumulative
            if decay < 1.0:
                empirical_counts *= decay
            empirical_counts[draw] += 1.0

            if (t + 1) % update_every == 0:
                empirical_distribution = normalize_distribution(empirical_counts.copy())
                current_checkpoints = AdaptiveStrategy(empirical_distribution).place_checkpoints(prefix_length, budget)
                latest_current = np.zeros(prefix_length + 1, dtype=int)
                latest_current[current_checkpoints] = current_checkpoints
                latest_current = np.maximum.accumulate(latest_current)

    return all_regrets.mean(axis=0), all_regrets.std(axis=0)


def plot_online_adaptation(
    figure_dir: Path,
    distribution: np.ndarray,
    prefix_length: int,
    budget: int,
) -> tuple[float, float, float, float]:
    common = dict(
        true_distribution=distribution,
        prefix_length=prefix_length,
        budget=budget,
        horizon=2000,
        update_every=50,
        trials=24,
        rng_seed=SEED,
    )
    mean_flat, std_flat = simulate_online_adaptation(**common, decay=1.0)
    mean_exp, std_exp = simulate_online_adaptation(**common, decay=0.99)

    xs = np.arange(1, len(mean_flat) + 1)
    plt.figure(figsize=(8.2, 4.8))
    plt.plot(xs, mean_flat, linewidth=2.2, color="#005f73", label="flat histogram")
    plt.fill_between(xs, mean_flat - std_flat, mean_flat + std_flat, color="#94d2bd", alpha=0.25)
    plt.plot(xs, mean_exp, linewidth=2.2, color="#ae2012", label="exponential (\u03b3=0.99)")
    plt.fill_between(xs, mean_exp - std_exp, mean_exp + std_exp, color="#ee9b00", alpha=0.25)
    plt.xlabel("Requests processed")
    plt.ylabel("Cumulative regret")
    plt.title("Online adaptation: exponential weighting reduces regret")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(figure_dir / "online_adaptation_convergence.pdf")
    plt.close()
    return float(mean_flat[-1]), float(std_flat[-1]), float(mean_exp[-1]), float(std_exp[-1])


def plot_empirical_oracle_convergence(
    figure_dir: Path,
    distribution: np.ndarray,
    prefix_length: int,
    budget: int,
    sample_sizes: list[int] | None = None,
    trials: int = 40,
    rng_seed: int = SEED,
) -> dict[str, object]:
    if sample_sizes is None:
        sample_sizes = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

    oracle_checkpoints = AdaptiveStrategy(distribution).place_checkpoints(prefix_length, budget)
    oracle_cost = expected_recomputation_cost(distribution, oracle_checkpoints)
    support = np.arange(1, prefix_length + 1)

    mean_l1: list[float] = []
    std_l1: list[float] = []
    mean_gap: list[float] = []
    std_gap: list[float] = []

    for size_index, sample_size in enumerate(sample_sizes):
        l1_errors = np.zeros(trials, dtype=float)
        suboptimality_gaps = np.zeros(trials, dtype=float)
        for trial in range(trials):
            rng = np.random.default_rng(rng_seed + 10_000 * size_index + trial)
            draws = rng.choice(support, size=sample_size, p=distribution[1:])
            empirical_distribution = empirical_distribution_from_draws(draws, prefix_length)
            l1_errors[trial] = float(np.abs(empirical_distribution[1:] - distribution[1:]).sum())
            empirical_checkpoints = AdaptiveStrategy(empirical_distribution).place_checkpoints(prefix_length, budget)
            plugin_cost = expected_recomputation_cost(distribution, empirical_checkpoints)
            suboptimality_gaps[trial] = max(0.0, plugin_cost - oracle_cost)

        mean_l1.append(float(l1_errors.mean()))
        std_l1.append(float(l1_errors.std()))
        mean_gap.append(float(suboptimality_gaps.mean()))
        std_gap.append(float(suboptimality_gaps.std()))

    xs = np.asarray(sample_sizes, dtype=float)
    mean_l1_arr = np.asarray(mean_l1, dtype=float)
    std_l1_arr = np.asarray(std_l1, dtype=float)
    mean_gap_arr = np.asarray(mean_gap, dtype=float)
    std_gap_arr = np.asarray(std_gap, dtype=float)
    reference_l1 = mean_l1_arr[0] * np.sqrt(xs[0] / xs)
    reference_gap = max(mean_gap_arr[0], EPS) * np.sqrt(xs[0] / xs)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4))

    axes[0].plot(xs, mean_l1_arr, marker="o", linewidth=2.2, color="#005f73", label=r"mean $\|\hat p-p\|_1$")
    axes[0].fill_between(
        xs,
        np.maximum(mean_l1_arr - std_l1_arr, EPS),
        mean_l1_arr + std_l1_arr,
        color="#94d2bd",
        alpha=0.25,
    )
    axes[0].plot(xs, reference_l1, linestyle="--", linewidth=1.8, color="#9b2226", label=r"$n^{-1/2}$ reference")
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Observed overlaps")
    axes[0].set_ylabel(r"$\ell_1$ estimation error")
    axes[0].set_title("Empirical law estimation")
    axes[0].legend(frameon=False)

    axes[1].plot(xs, mean_gap_arr, marker="o", linewidth=2.2, color="#bb3e03", label="mean plug-in gap")
    axes[1].fill_between(
        xs,
        np.maximum(mean_gap_arr - std_gap_arr, EPS),
        mean_gap_arr + std_gap_arr,
        color="#ee9b00",
        alpha=0.25,
    )
    axes[1].plot(xs, reference_gap, linestyle="--", linewidth=1.8, color="#0a9396", label=r"$n^{-1/2}$ reference")
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Observed overlaps")
    axes[1].set_ylabel("Suboptimality gap")
    axes[1].set_title("Empirical oracle under true law")
    axes[1].legend(frameon=False)

    fig.suptitle("Stationary sample complexity of the empirical histogram oracle", y=1.02)
    fig.tight_layout()
    fig.savefig(figure_dir / "empirical_oracle_convergence.pdf")
    plt.close(fig)

    return {
        "sample_sizes": sample_sizes,
        "mean_l1": mean_l1,
        "std_l1": std_l1,
        "mean_gap": mean_gap,
        "std_gap": std_gap,
    }


def simulate_drift_tracking(
    schedule: list[np.ndarray],
    prefix_length: int,
    budget: int,
    update_every: int,
    trials: int,
    rng_seed: int,
    decay: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    horizon = len(schedule)
    support = np.arange(1, prefix_length + 1)

    oracle_cache: dict[int, float] = {}
    for distribution in schedule:
        key = id(distribution)
        if key not in oracle_cache:
            oracle_checkpoints = AdaptiveStrategy(distribution).place_checkpoints(prefix_length, budget)
            oracle_cache[key] = expected_recomputation_cost(distribution, oracle_checkpoints)

    l1_curves = np.zeros((trials, horizon), dtype=float)
    gap_curves = np.zeros((trials, horizon), dtype=float)

    for trial in range(trials):
        rng = np.random.default_rng(rng_seed + trial)
        counts = np.ones(prefix_length + 1, dtype=float)
        current_l1 = 0.0
        current_gap = 0.0

        for t, distribution in enumerate(schedule):
            draw = int(rng.choice(support, p=distribution[1:]))
            if decay < 1.0:
                counts *= decay
            counts[draw] += 1.0

            if t == 0 or (t + 1) % update_every == 0:
                empirical_distribution = normalize_distribution(counts.copy())
                checkpoints = AdaptiveStrategy(empirical_distribution).place_checkpoints(prefix_length, budget)
                oracle_cost = oracle_cache[id(distribution)]
                current_l1 = float(np.abs(empirical_distribution[1:] - distribution[1:]).sum())
                current_gap = max(0.0, expected_recomputation_cost(distribution, checkpoints) - oracle_cost)

            l1_curves[trial, t] = current_l1
            gap_curves[trial, t] = current_gap

    return (
        l1_curves.mean(axis=0),
        l1_curves.std(axis=0),
        gap_curves.mean(axis=0),
        gap_curves.std(axis=0),
    )


def plot_drift_tracking(
    figure_dir: Path,
    prefix_length: int,
    budget: int,
    update_every: int = 25,
    segment_length: int = 800,
    trials: int = 20,
) -> dict[str, float | list[int]]:
    phases = [
        ("zipf", zipf_distribution(prefix_length, alpha=1.25)),
        ("bimodal", bimodal_distribution(prefix_length)),
        ("late", late_peak_distribution(prefix_length)),
    ]
    schedule = [distribution for _, distribution in phases for _ in range(segment_length)]
    horizon = len(schedule)
    change_points = [segment_length, 2 * segment_length]

    flat_mean_l1, flat_std_l1, flat_mean_gap, flat_std_gap = simulate_drift_tracking(
        schedule=schedule,
        prefix_length=prefix_length,
        budget=budget,
        update_every=update_every,
        trials=trials,
        rng_seed=SEED + 2000,
        decay=1.0,
    )
    exp_mean_l1, exp_std_l1, exp_mean_gap, exp_std_gap = simulate_drift_tracking(
        schedule=schedule,
        prefix_length=prefix_length,
        budget=budget,
        update_every=update_every,
        trials=trials,
        rng_seed=SEED + 4000,
        decay=0.99,
    )

    xs = np.arange(1, horizon + 1)
    fig, axes = plt.subplots(2, 1, figsize=(10.2, 6.8), sharex=True)

    axes[0].plot(xs, flat_mean_l1, linewidth=2.0, color="#005f73", label="flat histogram")
    axes[0].fill_between(xs, flat_mean_l1 - flat_std_l1, flat_mean_l1 + flat_std_l1, color="#94d2bd", alpha=0.22)
    axes[0].plot(xs, exp_mean_l1, linewidth=2.0, color="#ae2012", label="exponential ($\\gamma=0.99$)")
    axes[0].fill_between(xs, exp_mean_l1 - exp_std_l1, exp_mean_l1 + exp_std_l1, color="#ee9b00", alpha=0.18)
    axes[0].set_ylabel(r"Current-law $\ell_1$ error")
    axes[0].set_title("Tracking a drifting overlap law")
    axes[0].legend(frameon=False, ncol=2)

    axes[1].plot(xs, flat_mean_gap, linewidth=2.0, color="#005f73", label="flat histogram")
    axes[1].fill_between(xs, np.maximum(flat_mean_gap - flat_std_gap, 0.0), flat_mean_gap + flat_std_gap, color="#94d2bd", alpha=0.22)
    axes[1].plot(xs, exp_mean_gap, linewidth=2.0, color="#ae2012", label="exponential ($\\gamma=0.99$)")
    axes[1].fill_between(xs, np.maximum(exp_mean_gap - exp_std_gap, 0.0), exp_mean_gap + exp_std_gap, color="#ee9b00", alpha=0.18)
    axes[1].set_xlabel("Requests processed")
    axes[1].set_ylabel("Current-law plug-in gap")

    for ax in axes:
        for change_point in change_points:
            ax.axvline(change_point, linestyle="--", linewidth=1.2, color="#6c757d", alpha=0.9)

    fig.tight_layout()
    fig.savefig(figure_dir / "drift_tracking_tradeoff.pdf")
    plt.close(fig)

    transition_window = 150
    post_shift_mask = np.zeros(horizon, dtype=bool)
    for change_point in change_points:
        start = change_point
        stop = min(horizon, change_point + transition_window)
        post_shift_mask[start:stop] = True

    return {
        "flat_mean_l1": float(flat_mean_l1.mean()),
        "exp_mean_l1": float(exp_mean_l1.mean()),
        "flat_mean_gap": float(flat_mean_gap.mean()),
        "exp_mean_gap": float(exp_mean_gap.mean()),
        "flat_post_shift_gap": float(flat_mean_gap[post_shift_mask].mean()),
        "exp_post_shift_gap": float(exp_mean_gap[post_shift_mask].mean()),
        "change_points": change_points,
    }


def simulate_nonstationary_dynamic_regret(
    schedule: list[np.ndarray],
    prefix_length: int,
    budget: int,
    update_every: int,
    trials: int,
    rng_seed: int,
    decay: float,
) -> tuple[np.ndarray, np.ndarray]:
    horizon = len(schedule)
    support = np.arange(1, prefix_length + 1)
    oracle_costs = np.zeros(horizon, dtype=float)
    oracle_cache: dict[int, float] = {}

    for t, distribution in enumerate(schedule):
        key = id(distribution)
        if key not in oracle_cache:
            oracle_checkpoints = AdaptiveStrategy(distribution).place_checkpoints(prefix_length, budget)
            oracle_cache[key] = expected_recomputation_cost(distribution, oracle_checkpoints)
        oracle_costs[t] = oracle_cache[key]

    regret_curves = np.zeros((trials, horizon), dtype=float)
    for trial in range(trials):
        rng = np.random.default_rng(rng_seed + trial)
        counts = np.ones(prefix_length + 1, dtype=float)
        current_checkpoints = PowerOfTwoStrategy().place_checkpoints(prefix_length, budget)
        cumulative = 0.0

        for t, distribution in enumerate(schedule):
            draw = int(rng.choice(support, p=distribution[1:]))
            if decay < 1.0:
                counts *= decay
            counts[draw] += 1.0

            if t == 0 or (t + 1) % update_every == 0:
                empirical_distribution = normalize_distribution(counts.copy())
                current_checkpoints = AdaptiveStrategy(empirical_distribution).place_checkpoints(prefix_length, budget)

            cumulative += expected_recomputation_cost(distribution, current_checkpoints) - oracle_costs[t]
            regret_curves[trial, t] = cumulative

    return regret_curves.mean(axis=0), regret_curves.std(axis=0)


def plot_nonstationary_dynamic_regret(
    figure_dir: Path,
    prefix_length: int,
    budget: int,
    update_every: int = 25,
    segment_length: int = 600,
    trials: int = 12,
) -> dict[str, float | list[int]]:
    phases = [
        ("zipf", zipf_distribution(prefix_length, alpha=1.25)),
        ("bimodal", bimodal_distribution(prefix_length)),
        ("late", late_peak_distribution(prefix_length)),
    ]
    schedule = [distribution for _, distribution in phases for _ in range(segment_length)]
    horizon = len(schedule)
    change_points = [segment_length, 2 * segment_length]

    flat_mean, flat_std = simulate_nonstationary_dynamic_regret(
        schedule=schedule,
        prefix_length=prefix_length,
        budget=budget,
        update_every=update_every,
        trials=trials,
        rng_seed=SEED + 6000,
        decay=1.0,
    )
    exp_mean, exp_std = simulate_nonstationary_dynamic_regret(
        schedule=schedule,
        prefix_length=prefix_length,
        budget=budget,
        update_every=update_every,
        trials=trials,
        rng_seed=SEED + 8000,
        decay=0.99,
    )

    xs = np.arange(1, horizon + 1)
    plt.figure(figsize=(10.0, 4.8))
    plt.plot(xs, flat_mean, linewidth=2.2, color="#005f73", label="flat histogram")
    plt.fill_between(xs, np.maximum(flat_mean - flat_std, 0.0), flat_mean + flat_std, color="#94d2bd", alpha=0.22)
    plt.plot(xs, exp_mean, linewidth=2.2, color="#ae2012", label="exponential ($\\gamma=0.99$)")
    plt.fill_between(xs, np.maximum(exp_mean - exp_std, 0.0), exp_mean + exp_std, color="#ee9b00", alpha=0.18)
    for change_point in change_points:
        plt.axvline(change_point, linestyle="--", linewidth=1.2, color="#6c757d", alpha=0.9)
    plt.xlabel("Requests processed")
    plt.ylabel("Cumulative dynamic regret")
    plt.title("Dynamic regret under piecewise-stationary overlap drift")
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(figure_dir / "drift_dynamic_regret.pdf")
    plt.close()

    return {
        "flat_final_regret": float(flat_mean[-1]),
        "exp_final_regret": float(exp_mean[-1]),
        "improvement": float(1.0 - exp_mean[-1] / max(flat_mean[-1], EPS)),
        "change_points": change_points,
    }


def plot_realistic_workload_bars(
    figure_dir: Path,
    prefix_length: int,
    budgets: list[int],
    strategies: list[CheckpointStrategy],
    distribution: np.ndarray,
) -> dict[int, dict[str, float]]:
    data: dict[int, dict[str, float]] = {}
    for budget in budgets:
        data[budget] = {}
        for strategy in strategies:
            checkpoints = strategy.place_checkpoints(prefix_length, budget)
            data[budget][strategy.name] = 100.0 * compute_savings(distribution, checkpoints)
        data[budget]["full_cache"] = 100.0
        data[budget]["exact_match"] = 100.0 * exact_match_savings(distribution)

    labels = [strategy.name for strategy in strategies] + ["full_cache", "exact_match"]
    x = np.arange(len(labels))
    width = 0.22
    plt.figure(figsize=(9.2, 5.2))
    colors = ["#0a9396", "#ee9b00", "#ca6702"]
    for idx, budget in enumerate(budgets):
        heights = [data[budget][label] for label in labels]
        plt.bar(x + (idx - 1) * width, heights, width=width, label=f"M={budget}", color=colors[idx])
    plt.xticks(x, labels)
    plt.ylabel("TTFT reduction proxy (%)")
    plt.title("Realistic workload comparison at fixed memory budgets")
    plt.ylim(0, 103)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(figure_dir / "realistic_workload_bars.pdf")
    plt.close()
    return data


def print_experiment_summary(
    results: list[dict[str, object]],
    realistic_data: dict[int, dict[str, float]],
    final_regret: float,
    final_regret_std: float,
    final_regret_exp: float | None = None,
    final_regret_exp_std: float | None = None,
    empirical_oracle_stats: dict[str, object] | None = None,
    drift_tracking_stats: dict[str, float | list[int]] | None = None,
    dynamic_regret_stats: dict[str, float | list[int]] | None = None,
) -> None:
    print("Sparse Prefix Caching for SSM/Hybrid Models")
    print("=" * 48)
    for row in results:
        print(
            f"[{row['distribution']}] {row['strategy']:>12} | budget={row['budget']:>3} | "
            f"used={row['used']:>3} | savings={100.0 * row['savings']:.2f}% | "
            f"expected={row['expected_cost']:.3f} | mc={row['monte_carlo_cost']:.3f} | "
            f"relerr={100.0 * row['relative_error']:.2f}% | checkpoints={row['checkpoints']}"
        )
    print()
    print("Realistic workload TTFT proxy reduction:")
    for budget, metrics in realistic_data.items():
        ordered = ", ".join(f"{name}={value:.2f}%" for name, value in metrics.items())
        print(f"  M={budget}: {ordered}")
    print()
    print(f"Online adaptation final cumulative regret (flat): {final_regret:.2f} +/- {final_regret_std:.2f}")
    if final_regret_exp is not None:
        print(f"Online adaptation final cumulative regret (exp):  {final_regret_exp:.2f} +/- {final_regret_exp_std:.2f}")
    if empirical_oracle_stats is not None:
        sample_sizes = empirical_oracle_stats["sample_sizes"]
        mean_l1 = empirical_oracle_stats["mean_l1"]
        mean_gap = empirical_oracle_stats["mean_gap"]
        print(
            "Empirical oracle convergence:"
            f" n={sample_sizes[0]} -> l1={mean_l1[0]:.4f}, gap={mean_gap[0]:.4f};"
            f" n={sample_sizes[-1]} -> l1={mean_l1[-1]:.4f}, gap={mean_gap[-1]:.4f}"
        )
    if drift_tracking_stats is not None:
        print(
            "Drift tracking:"
            f" flat gap={drift_tracking_stats['flat_mean_gap']:.3f},"
            f" exp gap={drift_tracking_stats['exp_mean_gap']:.3f};"
            f" post-shift flat={drift_tracking_stats['flat_post_shift_gap']:.3f},"
            f" post-shift exp={drift_tracking_stats['exp_post_shift_gap']:.3f}"
        )
    if dynamic_regret_stats is not None:
        print(
            "Dynamic regret under drift:"
            f" flat={dynamic_regret_stats['flat_final_regret']:.1f},"
            f" exp={dynamic_regret_stats['exp_final_regret']:.1f},"
            f" improvement={100.0 * dynamic_regret_stats['improvement']:.1f}%"
        )
    print()
    print("LaTeX table:")
    print(latex_results_table(results))


def run_experiments(prefix_length: int, validation_budget: int, monte_carlo_samples: int) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    figure_dir = REPO_ROOT / "figures"
    figure_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(SEED)
    distributions = {
        "uniform": uniform_distribution(prefix_length),
        "zipf": zipf_distribution(prefix_length),
        "bimodal": bimodal_distribution(prefix_length),
    }
    realistic_distribution = realistic_workload_distribution(prefix_length)

    fixed_strategies: list[CheckpointStrategy] = [
        UniformStrategy(),
        PowerOfTwoStrategy(),
        SqrtStrategy(),
    ]
    results: list[dict[str, object]] = []

    for dist_name, distribution in distributions.items():
        strategy_set = fixed_strategies + [AdaptiveStrategy(distribution)]
        for strategy in strategy_set:
            outcome = evaluate_strategy(
                strategy=strategy,
                distribution=distribution,
                prefix_length=prefix_length,
                budget=validation_budget,
                rng=rng,
                samples=monte_carlo_samples,
            )
            results.append(
                {
                    "distribution": dist_name,
                    "strategy": strategy.name,
                    "budget": validation_budget,
                    "used": len(outcome.checkpoints),
                    "savings": outcome.savings,
                    "expected_cost": outcome.expected_cost,
                    "monte_carlo_cost": outcome.monte_carlo_cost,
                    "relative_error": outcome.relative_error,
                    "checkpoints": format_checkpoints(outcome.checkpoints),
                }
            )

    pareto_budgets = np.unique(np.rint(np.geomspace(1, prefix_length, num=18)).astype(int))
    plot_pareto_frontier(
        figure_dir=figure_dir,
        prefix_length=prefix_length,
        budgets=pareto_budgets,
        strategies=fixed_strategies + [AdaptiveStrategy(distributions["uniform"])],
        distribution=distributions["uniform"],
    )
    plot_strategy_heatmap(
        figure_dir=figure_dir,
        prefix_length=prefix_length,
        budget=validation_budget,
        strategies=fixed_strategies + [AdaptiveStrategy(distributions["bimodal"])],
        distributions=distributions,
    )
    final_regret, final_regret_std, final_regret_exp, final_regret_exp_std = plot_online_adaptation(
        figure_dir=figure_dir,
        distribution=distributions["bimodal"],
        prefix_length=prefix_length,
        budget=validation_budget,
    )
    empirical_oracle_stats = plot_empirical_oracle_convergence(
        figure_dir=figure_dir,
        distribution=distributions["bimodal"],
        prefix_length=prefix_length,
        budget=validation_budget,
    )
    drift_tracking_stats = plot_drift_tracking(
        figure_dir=figure_dir,
        prefix_length=prefix_length,
        budget=validation_budget,
    )
    dynamic_regret_stats = plot_nonstationary_dynamic_regret(
        figure_dir=figure_dir,
        prefix_length=prefix_length,
        budget=validation_budget,
    )
    realistic_data = plot_realistic_workload_bars(
        figure_dir=figure_dir,
        prefix_length=prefix_length,
        budgets=[10, 50, 100],
        strategies=fixed_strategies + [AdaptiveStrategy(realistic_distribution)],
        distribution=realistic_distribution,
    )
    print_experiment_summary(
        results,
        realistic_data,
        final_regret,
        final_regret_std,
        final_regret_exp,
        final_regret_exp_std,
        empirical_oracle_stats,
        drift_tracking_stats,
        dynamic_regret_stats,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix-length", type=int, default=1024, help="Maximum cached prefix length N.")
    parser.add_argument("--budget", type=int, default=32, help="Checkpoint budget M for validation experiments.")
    parser.add_argument(
        "--samples",
        type=int,
        default=100_000,
        help="Monte Carlo samples per strategy/distribution pair.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.prefix_length < 8:
        raise ValueError("prefix length must be at least 8")
    if args.budget < 1:
        raise ValueError("budget must be positive")
    run_experiments(
        prefix_length=args.prefix_length,
        validation_budget=args.budget,
        monte_carlo_samples=args.samples,
    )


if __name__ == "__main__":
    main()
