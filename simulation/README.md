# Simulation Code for Sparse Prefix Caching

This directory contains the distributional and trace-driven simulators that
validate the theoretical results in the paper *Sparse Prefix Caching for
Hybrid and Recurrent LLM Serving*.

## Directory structure

```
simulation/
├── sparse_prefix_caching_sim.py      # Distributional simulator (main entry point)
├── trace_cache_sim/                   # Trace-driven simulator package
│   ├── __init__.py
│   ├── dp_opt.py                      # O(NM) DP solver with convex hull trick
│   ├── policies.py                    # Cache policies (uniform, geometric, DP, branch-aware, online)
│   ├── simulator.py                   # Trace-driven simulation engine
│   ├── traces.py                      # Trace generation & prefix trie utilities
│   ├── experiments.py                 # Experiment runner & figure generation
│   └── toy_recurrence.py             # Toy recurrence for checkpoint-resume exactness check
├── tests/
│   ├── __init__.py
│   └── test_trace_cache_sim.py        # Unit tests
└── README.md                          # This file
```

## Requirements

```
numpy
matplotlib
seaborn
```

Install with:
```bash
pip install numpy matplotlib seaborn
```

## Distributional simulator

The distributional simulator evaluates checkpoint placement strategies under
four synthetic overlap distributions (uniform, Zipf, bimodal, realistic proxy)
and validates theoretical bounds from the paper.

**Run all experiments and generate figures:**

```bash
python sparse_prefix_caching_sim.py
```

Options:
- `--prefix-length N` — maximum cached prefix length (default: 1024)
- `--budget M` — checkpoint budget for validation experiments (default: 32)
- `--samples S` — Monte Carlo samples per strategy/distribution (default: 100,000)

**Experiments included:**
1. Strategy comparison on synthetic distributions (Table 1 in paper)
2. Pareto frontier: memory vs. compute trade-off
3. Savings heatmap across strategies and distributions
4. Realistic workload TTFT-proxy reduction at varying budgets
5. Online adaptation convergence (flat vs. exponential histogram)
6. Empirical histogram oracle convergence (validates Theorem 6)
7. Drift tracking under piecewise-stationary overlap shifts (validates Theorem 7)
8. Dynamic regret under drift (validates Theorem 8)
9. Full non-stationary sweep over γ × H × drift regime (validates Corollary 10)

Output figures are written to a `figures/` directory next to the script.

## Trace-driven simulator

The trace-driven simulator evaluates caching policies on five synthetic request
families that cover qualitatively different sharing patterns:

| Family | Description |
|--------|-------------|
| `exact-hot-prefix` | Many requests share one long prefix, diverge after it |
| `append-only-chat` | Each request extends a previously completed chat trace |
| `diffuse-cutpoints` | Requests share a long document but branch at varying positions |
| `agent-tree` | Branching workload with multiple depths |
| `adversarial-uniform-overlap` | Overlaps spread uniformly over a wide depth range |

**Run the trace-driven experiments:**

```bash
python -m trace_cache_sim.experiments
```

This runs all five trace families against all policies and generates:
- Per-family comparison tables
- Memory vs. token-hit-ratio, recompute, and TTFT figures
- Eviction sweep and SSM fraction sweep analyses
- Checkpoint-resume exactness verification (zero numerical error)

**Policies implemented:**
- `NoCachePolicy` — baseline
- `DenseEveryKPolicy` — dense checkpoints every k positions
- `UniformBudgetMPolicy` — evenly spaced within budget
- `GeometricBaseRPolicy` / `GeometricEpsPolicy` — geometric schedules
- `BranchOnlyPolicy` / `BranchPlusGeometricPolicy` — branch-aware
- `OfflineDPOptimalPolicy` — offline O(NM) DP on empirical histogram
- `OnlineHistogramPolicy` — periodic replanning on flat histogram
- `ExponentialHistogramPolicy` — periodic replanning with exponential decay
- `BoundedCachePolicy` — LRU-bounded wrapper for finite cache capacity

## Offline DP solver

`trace_cache_sim/dp_opt.py` implements the O(NM) offline checkpoint placement
solver from Theorem 5 (the "monotone convex hull trick" DP). It can be used
standalone:

```python
from trace_cache_sim.dp_opt import offline_optimal_checkpoints
import numpy as np

# Example: bimodal distribution on N=1024
N = 1024
dist = np.zeros(N + 1)
dist[200:220] = 1.0
dist[830:850] = 1.0
dist /= dist.sum()

checkpoints, cost = offline_optimal_checkpoints(N, budget=32, distribution=dist)
print(f"Optimal checkpoints: {checkpoints}")
print(f"Expected recompute cost: {cost:.4f}")
```

## Tests

```bash
cd simulation
python -m pytest tests/ -v
# or
python -m unittest tests.test_trace_cache_sim -v
```

## Key theoretical results validated

| Result | Simulator |
|--------|-----------|
| Balanced spacing optimal & minimax under uniform (Thm 1–2) | Distributional |
| Geometric schedule ≥ 1/r savings (Thm 3) | Distributional |
| Lower bounds Ω(N/M) (Thm 4) | Distributional |
| Stability / Lipschitz (Thm 5) | Both |
| Empirical oracle n^{-1/2} convergence (Thm 6) | Distributional |
| Exponential histogram drift tracking (Thm 7) | Distributional |
| Dynamic regret decomposition (Thm 8) | Distributional |
| Optimal γ* under bounded drift (Cor 10) | Distributional |
| Exact DP O(NM) solver (Thm 9) | Both |
| Checkpoint-resume exactness (Thm 10) | Trace-driven |
| Branch-aware policies for tree workloads | Trace-driven |
| LRU eviction robustness | Trace-driven |
| SSM fraction scaling | Trace-driven |
