"""Trace-driven simulation of sparse recurrent-state prefix caching."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .policies import CachePolicy, CacheSnapshot, align_length, deepest_cached_depth
from .traces import PrefixTrie, Request


@dataclass(frozen=True)
class SimulationConfig:
    recurrent_state_bytes: int = 2048
    attention_bytes_per_token: int = 256
    a_rec: float = 1.0
    a_attn: float = 0.002
    a_load: float = 0.0001
    a_write: float = 0.00002
    chunk_size: int = 16


@dataclass
class RequestMetrics:
    overlap_length: int
    raw_reuse_depth: int
    reuse_depth: int
    recompute_tokens: int
    alignment_error: int
    token_hit_ratio: float
    recurrent_cost: float
    hybrid_cost: float
    bytes_read_recurrent: int
    bytes_read_total: int
    bytes_written_recurrent: int
    bytes_written_total: int
    checkpoint_count: int
    recurrent_bytes: int
    total_bytes: int


@dataclass
class SimulationSummary:
    family: str
    policy: str
    requests: int
    avg_recompute_tokens: float
    max_recompute_tokens: int
    avg_token_hit_ratio: float
    avg_alignment_error: float
    avg_recurrent_cost: float
    avg_hybrid_cost: float
    avg_bytes_read_total: float
    avg_bytes_written_total: float
    avg_checkpoint_count: float
    max_checkpoint_count: int
    avg_recurrent_bytes: float
    max_recurrent_bytes: int
    avg_total_bytes: float
    max_total_bytes: int


@dataclass
class SimulationResult:
    summary: SimulationSummary
    request_metrics: list[RequestMetrics]


def recurrent_bytes(snapshot: CacheSnapshot, config: SimulationConfig) -> int:
    return len(snapshot.aligned_prefixes) * config.recurrent_state_bytes


def attention_bytes(snapshot: CacheSnapshot, config: SimulationConfig) -> int:
    return sum(len(prefix) * config.attention_bytes_per_token for prefix in snapshot.aligned_prefixes)


def total_bytes(snapshot: CacheSnapshot, config: SimulationConfig) -> int:
    return recurrent_bytes(snapshot, config) + attention_bytes(snapshot, config)


def bytes_for_new_prefixes(prefixes: set[tuple[int, ...]], config: SimulationConfig) -> tuple[int, int]:
    rec = len(prefixes) * config.recurrent_state_bytes
    attn = sum(len(prefix) * config.attention_bytes_per_token for prefix in prefixes)
    return rec, rec + attn


def simulate_trace(requests: list[Request], policy: CachePolicy, config: SimulationConfig) -> SimulationResult:
    trie = PrefixTrie()
    seen_requests: list[Request] = []
    overlap_history: list[int] = []
    policy.reset(full_trace=requests)

    metrics: list[RequestMetrics] = []
    family = requests[0].family if requests else "unknown"

    for request in requests:
        before = policy.snapshot(seen_requests, trie, overlap_history)
        overlap = trie.longest_prefix(request.tokens)
        raw_reuse = min(overlap, deepest_cached_depth(request.tokens, before.raw_prefixes))
        reuse = min(overlap, deepest_cached_depth(request.tokens, before.aligned_prefixes))
        alignment_error = raw_reuse - align_length(raw_reuse, config.chunk_size)
        recompute = overlap - reuse
        token_hit = reuse / overlap if overlap > 0 else 0.0

        read_recurrent = config.recurrent_state_bytes if reuse > 0 else 0
        read_total = read_recurrent + reuse * config.attention_bytes_per_token if reuse > 0 else 0

        recurrent_cost = config.a_rec * recompute + config.a_load * read_recurrent
        hybrid_cost = config.a_rec * recompute + config.a_attn * (overlap * overlap - reuse * reuse) + config.a_load * read_recurrent

        trie.insert(request.tokens)
        seen_requests.append(request)
        overlap_history.append(overlap)
        policy.observe(seen_requests, trie, overlap_history)
        after = policy.snapshot(seen_requests, trie, overlap_history)

        new_prefixes = set(after.aligned_prefixes) - set(before.aligned_prefixes)
        written_recurrent, written_total = bytes_for_new_prefixes(new_prefixes, config)
        recurrent_cost += config.a_write * written_recurrent
        hybrid_cost += config.a_write * written_total

        metrics.append(
            RequestMetrics(
                overlap_length=overlap,
                raw_reuse_depth=raw_reuse,
                reuse_depth=reuse,
                recompute_tokens=recompute,
                alignment_error=alignment_error,
                token_hit_ratio=token_hit,
                recurrent_cost=recurrent_cost,
                hybrid_cost=hybrid_cost,
                bytes_read_recurrent=read_recurrent,
                bytes_read_total=read_total,
                bytes_written_recurrent=written_recurrent,
                bytes_written_total=written_total,
                checkpoint_count=len(after.aligned_prefixes),
                recurrent_bytes=recurrent_bytes(after, config),
                total_bytes=total_bytes(after, config),
            )
        )

    summary = SimulationSummary(
        family=family,
        policy=policy.name,
        requests=len(metrics),
        avg_recompute_tokens=float(np.mean([item.recompute_tokens for item in metrics])) if metrics else 0.0,
        max_recompute_tokens=max((item.recompute_tokens for item in metrics), default=0),
        avg_token_hit_ratio=float(np.mean([item.token_hit_ratio for item in metrics])) if metrics else 0.0,
        avg_alignment_error=float(np.mean([item.alignment_error for item in metrics])) if metrics else 0.0,
        avg_recurrent_cost=float(np.mean([item.recurrent_cost for item in metrics])) if metrics else 0.0,
        avg_hybrid_cost=float(np.mean([item.hybrid_cost for item in metrics])) if metrics else 0.0,
        avg_bytes_read_total=float(np.mean([item.bytes_read_total for item in metrics])) if metrics else 0.0,
        avg_bytes_written_total=float(np.mean([item.bytes_written_total for item in metrics])) if metrics else 0.0,
        avg_checkpoint_count=float(np.mean([item.checkpoint_count for item in metrics])) if metrics else 0.0,
        max_checkpoint_count=max((item.checkpoint_count for item in metrics), default=0),
        avg_recurrent_bytes=float(np.mean([item.recurrent_bytes for item in metrics])) if metrics else 0.0,
        max_recurrent_bytes=max((item.recurrent_bytes for item in metrics), default=0),
        avg_total_bytes=float(np.mean([item.total_bytes for item in metrics])) if metrics else 0.0,
        max_total_bytes=max((item.total_bytes for item in metrics), default=0),
    )
    return SimulationResult(summary=summary, request_metrics=metrics)
