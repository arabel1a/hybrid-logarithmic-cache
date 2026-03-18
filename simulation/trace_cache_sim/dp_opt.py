"""Offline dynamic programming for sparse checkpoint placement."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


EPS = 1e-12


def normalize_distribution(distribution: np.ndarray) -> np.ndarray:
    dist = np.asarray(distribution, dtype=float).copy()
    dist[0] = 0.0
    mass = dist[1:].sum()
    if mass <= 0:
        return dist
    dist[1:] /= mass
    return dist


def histogram_from_lengths(lengths: list[int], max_length: int) -> np.ndarray:
    histogram = np.zeros(max_length + 1, dtype=float)
    for length in lengths:
        if 0 <= length <= max_length:
            histogram[length] += 1.0
    return normalize_distribution(histogram)


def expected_gap_from_lengths(lengths: list[int], checkpoints: list[int]) -> float:
    if not lengths:
        return 0.0
    ordered = sorted(set(int(c) for c in checkpoints if c > 0))
    total = 0.0
    for length in lengths:
        eligible = [c for c in ordered if c <= length]
        previous = eligible[-1] if eligible else 0
        total += length - previous
    return total / len(lengths)


@dataclass
class Line:
    slope: float
    intercept: float
    arg: int


class MonotoneHull:
    def __init__(self) -> None:
        self.lines: deque[Line] = deque()

    @staticmethod
    def _value(line: Line, x: float) -> float:
        return line.slope * x + line.intercept

    @staticmethod
    def _redundant(a: Line, b: Line, c: Line) -> bool:
        return (c.intercept - a.intercept) * (a.slope - b.slope) <= (b.intercept - a.intercept) * (a.slope - c.slope) + EPS

    def add_line(self, line: Line) -> None:
        while len(self.lines) >= 2 and self._redundant(self.lines[-2], self.lines[-1], line):
            self.lines.pop()
        self.lines.append(line)

    def query(self, x: float) -> tuple[float, int]:
        while len(self.lines) >= 2 and self._value(self.lines[0], x) >= self._value(self.lines[1], x) - EPS:
            self.lines.popleft()
        best = self.lines[0]
        return self._value(best, x), best.arg


def offline_optimal_checkpoints(max_length: int, budget: int, distribution: np.ndarray) -> tuple[list[int], float]:
    if budget <= 0 or max_length <= 0:
        return [], float(np.dot(np.arange(len(distribution)), distribution))
    budget = min(budget, max_length)
    distribution = normalize_distribution(distribution)
    prefix_prob = np.zeros(max_length + 1, dtype=float)
    prefix_tp = np.zeros(max_length + 1, dtype=float)
    positions = np.arange(max_length + 1, dtype=float)
    prefix_prob[1:] = np.cumsum(distribution[1:])
    prefix_tp[1:] = np.cumsum(distribution[1:] * positions[1:])

    dp_prev = prefix_tp.copy()
    back = np.zeros((budget + 1, max_length + 1), dtype=np.int32)

    for used in range(1, budget + 1):
        dp_curr = np.zeros(max_length + 1, dtype=float)
        hull = MonotoneHull()
        hull.add_line(Line(slope=-1.0, intercept=float(dp_prev[0]), arg=1))
        for end in range(1, max_length + 1):
            start = end
            intercept = float(dp_prev[start - 1] - prefix_tp[start - 1] + start * prefix_prob[start - 1])
            hull.add_line(Line(slope=-float(start), intercept=intercept, arg=start))
            value, argmin = hull.query(float(prefix_prob[end]))
            dp_curr[end] = prefix_tp[end] + value
            back[used, end] = argmin
        dp_prev = dp_curr

    checkpoints: list[int] = []
    end = max_length
    for used in range(budget, 0, -1):
        start = int(back[used, end])
        if start <= 0:
            break
        checkpoints.append(start)
        end = start - 1
    return sorted(set(checkpoints)), float(dp_prev[max_length])

