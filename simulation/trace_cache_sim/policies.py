"""Cache policy definitions for trace-driven sparse checkpointing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dp_opt import histogram_from_lengths, normalize_distribution, offline_optimal_checkpoints
from .traces import PrefixTrie, Request


Prefix = tuple[int, ...]


@dataclass(frozen=True)
class CacheSnapshot:
    raw_prefixes: frozenset[Prefix]
    aligned_prefixes: frozenset[Prefix]


def align_length(length: int, chunk_size: int) -> int:
    if chunk_size <= 1:
        return max(0, length)
    return max(0, (length // chunk_size) * chunk_size)


def uniform_lengths(max_length: int, budget: int) -> list[int]:
    if budget <= 0 or max_length <= 0:
        return []
    if budget >= max_length:
        return list(range(1, max_length + 1))
    return sorted({max(1, min(max_length, round(i * (max_length + 1) / (budget + 1)))) for i in range(1, budget + 1)})


def geometric_base_lengths(max_length: int, base: float) -> list[int]:
    if max_length <= 0:
        return []
    points = [1]
    while points[-1] < max_length:
        next_point = max(points[-1] + 1, int(points[-1] * base))
        points.append(min(max_length, next_point))
        if points[-1] == max_length:
            break
    return sorted(set(points))


def geometric_eps_lengths(max_length: int, eps: float) -> list[int]:
    if max_length <= 0:
        return []
    current = 1
    points = [current]
    factor = 1.0 + eps
    while current < max_length:
        current = max(current + 1, int(current * factor))
        points.append(min(max_length, current))
        if current >= max_length:
            break
    return sorted(set(points))


def collect_prefixes_for_lengths(requests: list[Request], lengths: list[int]) -> set[Prefix]:
    prefixes: set[Prefix] = set()
    positive_lengths = [length for length in lengths if length > 0]
    for request in requests:
        for length in positive_lengths:
            if length <= len(request.tokens):
                prefixes.add(request.tokens[:length])
    return prefixes


def apply_alignment(prefixes: set[Prefix], chunk_size: int) -> set[Prefix]:
    aligned: set[Prefix] = set()
    for prefix in prefixes:
        length = align_length(len(prefix), chunk_size)
        if length > 0:
            aligned.add(prefix[:length])
    return aligned


def deepest_cached_depth(tokens: tuple[int, ...], prefixes: frozenset[Prefix]) -> int:
    for length in range(len(tokens), 0, -1):
        if tokens[:length] in prefixes:
            return length
    return 0


class CachePolicy:
    name = "base"

    def __init__(self, chunk_size: int = 1) -> None:
        self.chunk_size = chunk_size

    def reset(self, full_trace: list[Request] | None = None) -> None:
        return None

    def observe(self, seen_requests: list[Request], trie: PrefixTrie, overlap_history: list[int]) -> None:
        return None

    def raw_prefixes(self, seen_requests: list[Request], trie: PrefixTrie, overlap_history: list[int]) -> set[Prefix]:
        return set()

    def snapshot(self, seen_requests: list[Request], trie: PrefixTrie, overlap_history: list[int]) -> CacheSnapshot:
        raw = self.raw_prefixes(seen_requests, trie, overlap_history)
        return CacheSnapshot(raw_prefixes=frozenset(raw), aligned_prefixes=frozenset(apply_alignment(raw, self.chunk_size)))


class NoCachePolicy(CachePolicy):
    name = "no_cache"


class DenseEveryKPolicy(CachePolicy):
    name = "dense_every_k"

    def __init__(self, every_k: int, chunk_size: int = 1) -> None:
        super().__init__(chunk_size=chunk_size)
        self.every_k = every_k

    def raw_prefixes(self, seen_requests: list[Request], trie: PrefixTrie, overlap_history: list[int]) -> set[Prefix]:
        if self.every_k <= 0:
            return set()
        max_length = max((len(request.tokens) for request in seen_requests), default=0)
        lengths = list(range(self.every_k, max_length + 1, self.every_k))
        return collect_prefixes_for_lengths(seen_requests, lengths)


class UniformBudgetMPolicy(CachePolicy):
    name = "uniform_budget_m"

    def __init__(self, budget: int, chunk_size: int = 1) -> None:
        super().__init__(chunk_size=chunk_size)
        self.budget = budget

    def raw_prefixes(self, seen_requests: list[Request], trie: PrefixTrie, overlap_history: list[int]) -> set[Prefix]:
        max_length = max((len(request.tokens) for request in seen_requests), default=0)
        return collect_prefixes_for_lengths(seen_requests, uniform_lengths(max_length, self.budget))


class GeometricBaseRPolicy(CachePolicy):
    name = "geometric_base_r"

    def __init__(self, base: float, chunk_size: int = 1) -> None:
        super().__init__(chunk_size=chunk_size)
        self.base = base

    def raw_prefixes(self, seen_requests: list[Request], trie: PrefixTrie, overlap_history: list[int]) -> set[Prefix]:
        max_length = max((len(request.tokens) for request in seen_requests), default=0)
        return collect_prefixes_for_lengths(seen_requests, geometric_base_lengths(max_length, self.base))


class GeometricEpsPolicy(CachePolicy):
    name = "geometric_eps"

    def __init__(self, eps: float, chunk_size: int = 1) -> None:
        super().__init__(chunk_size=chunk_size)
        self.eps = eps

    def raw_prefixes(self, seen_requests: list[Request], trie: PrefixTrie, overlap_history: list[int]) -> set[Prefix]:
        max_length = max((len(request.tokens) for request in seen_requests), default=0)
        return collect_prefixes_for_lengths(seen_requests, geometric_eps_lengths(max_length, self.eps))


class BranchOnlyPolicy(CachePolicy):
    name = "branch_only"

    def raw_prefixes(self, seen_requests: list[Request], trie: PrefixTrie, overlap_history: list[int]) -> set[Prefix]:
        prefixes = set(trie.branch_prefixes())
        prefixes.update(request.tokens for request in seen_requests)
        return prefixes


class BranchPlusGeometricPolicy(CachePolicy):
    name = "branch_plus_geometric"

    def __init__(self, base: float, chunk_size: int = 1) -> None:
        super().__init__(chunk_size=chunk_size)
        self.base = base

    def raw_prefixes(self, seen_requests: list[Request], trie: PrefixTrie, overlap_history: list[int]) -> set[Prefix]:
        prefixes = BranchOnlyPolicy().raw_prefixes(seen_requests, trie, overlap_history)
        max_length = max((len(request.tokens) for request in seen_requests), default=0)
        prefixes.update(collect_prefixes_for_lengths(seen_requests, geometric_base_lengths(max_length, self.base)))
        return prefixes


class OfflineDPOptimalPolicy(CachePolicy):
    name = "offline_dp_optimal"

    def __init__(self, budget: int, chunk_size: int = 1) -> None:
        super().__init__(chunk_size=chunk_size)
        self.budget = budget
        self.schedule: list[int] = []

    def reset(self, full_trace: list[Request] | None = None) -> None:
        if not full_trace:
            self.schedule = []
            return
        from .traces import sequential_overlap_lengths

        overlaps = sequential_overlap_lengths(full_trace)
        max_length = max(overlaps, default=0)
        histogram = histogram_from_lengths(overlaps, max_length=max_length)
        self.schedule, _ = offline_optimal_checkpoints(max_length, self.budget, histogram)

    def raw_prefixes(self, seen_requests: list[Request], trie: PrefixTrie, overlap_history: list[int]) -> set[Prefix]:
        return collect_prefixes_for_lengths(seen_requests, self.schedule)


class OnlineHistogramPolicy(CachePolicy):
    name = "online_histogram_policy"

    def __init__(self, budget: int, update_every: int, chunk_size: int = 1) -> None:
        super().__init__(chunk_size=chunk_size)
        self.budget = budget
        self.update_every = update_every
        self.schedule: list[int] = []

    def reset(self, full_trace: list[Request] | None = None) -> None:
        self.schedule = []

    def observe(self, seen_requests: list[Request], trie: PrefixTrie, overlap_history: list[int]) -> None:
        if not overlap_history or len(overlap_history) % self.update_every != 0:
            return
        max_length = max(overlap_history)
        histogram = histogram_from_lengths(overlap_history, max_length=max_length)
        self.schedule, _ = offline_optimal_checkpoints(max_length, self.budget, histogram)

    def raw_prefixes(self, seen_requests: list[Request], trie: PrefixTrie, overlap_history: list[int]) -> set[Prefix]:
        return collect_prefixes_for_lengths(seen_requests, self.schedule)


class ExponentialHistogramPolicy(CachePolicy):
    """Online policy with exponentially weighted overlap histogram.

    Recent observations receive more weight, allowing faster adaptation
    to non-stationary workloads.
    """

    name = "exp_histogram_policy"

    def __init__(self, budget: int, update_every: int, decay: float = 0.95, chunk_size: int = 1) -> None:
        super().__init__(chunk_size=chunk_size)
        self.budget = budget
        self.update_every = update_every
        self.decay = decay
        self.schedule: list[int] = []

    def reset(self, full_trace: list[Request] | None = None) -> None:
        self.schedule = []

    def observe(self, seen_requests: list[Request], trie: PrefixTrie, overlap_history: list[int]) -> None:
        if not overlap_history or len(overlap_history) % self.update_every != 0:
            return
        max_length = max(overlap_history)
        if max_length <= 0:
            return
        n = len(overlap_history)
        histogram = np.zeros(max_length + 1, dtype=float)
        for i, length in enumerate(overlap_history):
            weight = self.decay ** (n - 1 - i)
            if 0 <= length <= max_length:
                histogram[length] += weight
        histogram = normalize_distribution(histogram)
        self.schedule, _ = offline_optimal_checkpoints(max_length, self.budget, histogram)

    def raw_prefixes(self, seen_requests: list[Request], trie: PrefixTrie, overlap_history: list[int]) -> set[Prefix]:
        return collect_prefixes_for_lengths(seen_requests, self.schedule)


class BoundedCachePolicy(CachePolicy):
    """Wraps any base policy and enforces a maximum cache size with LRU eviction."""

    def __init__(self, base_policy: CachePolicy, max_entries: int, chunk_size: int = 1) -> None:
        super().__init__(chunk_size=chunk_size)
        self.base = base_policy
        self.max_entries = max_entries
        self.name = f"{base_policy.name}+lru{max_entries}"
        self._cache: dict[tuple[int, ...], int] = {}
        self._step: int = 0

    def reset(self, full_trace: list[Request] | None = None) -> None:
        self.base.reset(full_trace)
        self._cache = {}
        self._step = 0

    def observe(self, seen_requests: list[Request], trie: PrefixTrie, overlap_history: list[int]) -> None:
        self.base.observe(seen_requests, trie, overlap_history)
        self._step = len(overlap_history)
        desired = self.base.raw_prefixes(seen_requests, trie, overlap_history)
        for prefix in desired:
            self._cache[prefix] = self._step
        while len(self._cache) > self.max_entries:
            oldest_key = min(self._cache, key=self._cache.get)
            del self._cache[oldest_key]

    def raw_prefixes(self, seen_requests: list[Request], trie: PrefixTrie, overlap_history: list[int]) -> set[Prefix]:
        return set(self._cache.keys())

