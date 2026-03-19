"""Caching strategy for prefix checkpoint evaluation."""
import math

STRATEGIES = [
    "no_cache",
    "balanced_fix_blocksize",
    "balanced_fix_nblocks",
    "sqrt",
    "dyadic"
]

def balanced_positions(seq_len, block_size=None, n_blocks=None):
    assert (block_size is None) != (n_blocks is None)
    if n_blocks is not None:
        block_size = seq_len // n_blocks
    return list(range(block_size, seq_len + 1, block_size))

def sqrt_positions(seq_len):
    block_size = int(math.sqrt(seq_len))
    return balanced_positions(seq_len, block_size=block_size)

def diadic_positions(seq_len: int) -> list[int]:
    positions = []
    i = 0
    while (1 << i) <= seq_len:
        positions.append(1 << i)
        i += 1
    return positions

def checkpoint_positions(strategy, seq_len, block_size=None, n_blocks=None):
    """Return list of positions where checkpoints should be captured."""
    if strategy == "no_cache":
        return []
    if strategy in ("block", "balanced_fix_blocksize"):
        return balanced_positions(seq_len, block_size=block_size)
    if strategy == "balanced_fix_nblocks":
        return balanced_positions(seq_len, n_blocks=n_blocks)
    if strategy == "sqrt":
        return sqrt_positions(seq_len)
    if strategy in ("log", "dyadic"):
        return diadic_positions(seq_len)
    raise ValueError(f"Unknown strategy: {strategy}")
