"""Trace generation and prefix trie utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Request:
    request_id: int
    family: str
    tokens: tuple[int, ...]


def longest_common_prefix(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    limit = min(len(a), len(b))
    index = 0
    while index < limit and a[index] == b[index]:
        index += 1
    return index


@dataclass
class TrieNode:
    depth: int
    children: dict[int, "TrieNode"] = field(default_factory=dict)
    pass_count: int = 0
    terminal_count: int = 0


class PrefixTrie:
    """Simple prefix trie used for overlap queries and branch-point extraction."""

    def __init__(self) -> None:
        self.root = TrieNode(depth=0)

    def insert(self, tokens: tuple[int, ...]) -> None:
        node = self.root
        node.pass_count += 1
        for token in tokens:
            child = node.children.get(token)
            if child is None:
                child = TrieNode(depth=node.depth + 1)
                node.children[token] = child
            node = child
            node.pass_count += 1
        node.terminal_count += 1

    def longest_prefix(self, tokens: tuple[int, ...]) -> int:
        node = self.root
        matched = 0
        for token in tokens:
            child = node.children.get(token)
            if child is None:
                break
            matched += 1
            node = child
        return matched

    def branch_prefixes(self) -> set[tuple[int, ...]]:
        results: set[tuple[int, ...]] = set()

        def dfs(node: TrieNode, prefix: list[int]) -> None:
            continuation_count = len(node.children) + (1 if node.terminal_count > 0 else 0)
            if node.depth > 0 and node.pass_count >= 2 and continuation_count >= 2:
                results.add(tuple(prefix))
            for token, child in node.children.items():
                prefix.append(token)
                dfs(child, prefix)
                prefix.pop()

        dfs(self.root, [])
        return results


def _random_tokens(rng: np.random.Generator, length: int, low: int = 1_000, high: int = 1_000_000) -> tuple[int, ...]:
    if length <= 0:
        return ()
    return tuple(int(x) for x in rng.integers(low, high, size=length))


def exact_hot_prefix(num_requests: int = 72, prefix_length: int = 320, suffix_length: int = 48, seed: int = 42) -> list[Request]:
    rng = np.random.default_rng(seed)
    shared_prefix = _random_tokens(rng, prefix_length)
    requests: list[Request] = []
    for index in range(num_requests):
        suffix = _random_tokens(rng, suffix_length)
        requests.append(Request(index, "exact_hot_prefix", shared_prefix + suffix))
    return requests


def append_only_chat(num_requests: int = 64, system_length: int = 48, turn_length: int = 10, seed: int = 42) -> list[Request]:
    rng = np.random.default_rng(seed + 1)
    prefix = list(_random_tokens(rng, system_length))
    requests: list[Request] = []
    for index in range(num_requests):
        prefix.extend(_random_tokens(rng, turn_length))
        requests.append(Request(index, "append_only_chat", tuple(prefix)))
    return requests


def diffuse_cutpoints(num_requests: int = 72, document_length: int = 512, tail_length: int = 24, seed: int = 42) -> list[Request]:
    rng = np.random.default_rng(seed + 2)
    document = _random_tokens(rng, document_length)
    requests: list[Request] = []
    for index in range(num_requests):
        cut = int(rng.integers(document_length // 6, document_length + 1))
        suffix = _random_tokens(rng, tail_length)
        requests.append(Request(index, "diffuse_cutpoints", document[:cut] + suffix))
    return requests


def agent_tree(branching_factor: int = 3, depth: int = 4, segment_length: int = 20, seed: int = 42) -> list[Request]:
    rng = np.random.default_rng(seed + 3)
    segment_bank: dict[tuple[int, ...], tuple[int, ...]] = {}
    requests: list[Request] = []

    def make_segment(path: tuple[int, ...]) -> tuple[int, ...]:
        if path not in segment_bank:
            segment_bank[path] = _random_tokens(rng, segment_length)
        return segment_bank[path]

    def walk(path: tuple[int, ...], level: int) -> None:
        if level == depth:
            tokens: tuple[int, ...] = ()
            for stop in range(1, len(path) + 1):
                tokens += make_segment(path[:stop])
            tokens += _random_tokens(rng, 8)
            requests.append(Request(len(requests), "agent_tree", tokens))
            return
        for child in range(branching_factor):
            walk(path + (child,), level + 1)

    walk((), 0)
    return requests


def adversarial_uniform_overlap(num_requests: int = 72, max_overlap: int = 384, suffix_length: int = 32, seed: int = 42) -> list[Request]:
    rng = np.random.default_rng(seed + 4)
    backbone = _random_tokens(rng, max_overlap)
    requests: list[Request] = [Request(0, "adversarial_uniform_overlap", backbone + _random_tokens(rng, suffix_length))]
    for index in range(1, num_requests):
        overlap = int(rng.integers(1, max_overlap + 1))
        suffix = _random_tokens(rng, suffix_length)
        requests.append(Request(index, "adversarial_uniform_overlap", backbone[:overlap] + suffix))
    return requests


def build_trace_families(seed: int = 42) -> dict[str, list[Request]]:
    return {
        "exact_hot_prefix": exact_hot_prefix(seed=seed),
        "append_only_chat": append_only_chat(seed=seed),
        "diffuse_cutpoints": diffuse_cutpoints(seed=seed),
        "agent_tree": agent_tree(seed=seed),
        "adversarial_uniform_overlap": adversarial_uniform_overlap(seed=seed),
    }


def sequential_overlap_lengths(requests: Iterable[Request]) -> list[int]:
    trie = PrefixTrie()
    overlaps: list[int] = []
    for request in requests:
        overlaps.append(trie.longest_prefix(request.tokens))
        trie.insert(request.tokens)
    return overlaps

