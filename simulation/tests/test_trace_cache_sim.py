from __future__ import annotations

import unittest

from trace_cache_sim.policies import deepest_cached_depth, geometric_base_lengths, uniform_lengths
from trace_cache_sim.toy_recurrence import checkpoint_resume_error
from trace_cache_sim.traces import longest_common_prefix


class TraceCacheSimTests(unittest.TestCase):
    def test_longest_common_prefix(self) -> None:
        self.assertEqual(longest_common_prefix((1, 2, 3), (1, 2, 4)), 2)
        self.assertEqual(longest_common_prefix((1, 2), (3, 4)), 0)
        self.assertEqual(longest_common_prefix((1, 2, 3), (1, 2, 3)), 3)

    def test_geometric_guarantee_sanity(self) -> None:
        checkpoints = geometric_base_lengths(128, base=2.0)
        prefixes = frozenset(tuple(range(length)) for length in checkpoints)
        for length in range(1, 129):
            tokens = tuple(range(length))
            cached = deepest_cached_depth(tokens, prefixes)
            gap = length - cached
            self.assertLessEqual(gap, length / 2 + 1)

    def test_uniform_additive_gap_sanity(self) -> None:
        schedule = uniform_lengths(120, 12)
        prefixes = frozenset(tuple(range(length)) for length in schedule)
        max_gap = 0
        for length in range(1, 121):
            tokens = tuple(range(length))
            cached = deepest_cached_depth(tokens, prefixes)
            max_gap = max(max_gap, length - cached)
        self.assertLessEqual(max_gap, 12)

    def test_toy_recurrence_exactness(self) -> None:
        tokens = tuple(range(1, 40))
        error = checkpoint_resume_error(tokens, checkpoint_depth=17)
        self.assertLess(error, 1e-10)


if __name__ == "__main__":
    unittest.main()
