# Plan: Add empirical plots to checkpoint_cache.py

## Goal
Update `checkpoint_cache.py` to produce plots similar to `plot_flops_cache.py` but with REAL measured latency from running a single layer group with Qwen3.5-27B dimensions.

## Steps
1. [x] Update `conf/config.yaml` with HF-compatible model fields for `make_model_config` (single layer group with 27B dims) — 05158c6
2. [x] Add `plot_benchmark()` function to `checkpoint_cache.py` that sweeps seq lengths, measures latency & cache memory, and produces 2-panel plots (latency vs N, cache memory vs N) — 05158c6
3. [x] Update `main.py` to call `plot_benchmark` — 05158c6
4. [x] Test run — verified on MPS, correctness passes, plots generated — 05158c6
5. [x] Add Qwen3.5-0.8B config, run theoretical FLOPs + empirical benchmark (up to 8k, MPS RAM limit), update README tables — 88528b3
6. [x] Add benchmark_baselines.py: empirical latency for all 6 caching strategies (no-cache, attn-only, block, log, block+attn, log+attn) — a3b5bcf
7. [x] Add cache size panel + theoretical FLOPs/cache dashed overlays to validate theory matches practice — db614b9
8. [x] Add benchmark_e2e.py: ShareGPT trace evaluation with FIFO cache, CPU offloading, 4 strategies — 3127041
9. [x] E2E: per-request JSONL logging + fix ordering (turns arrive in order within conversations, conversations interleaved randomly) — 19c0e52
10. [x] Fix e2e double forward pass on cache miss (was running prefill_baseline + prefill_and_capture_at = 2x compute) — 29f425d
11. [x] Track KV tokens saved per request; truncate oversize stores instead of dropping; document walltime discrepancy — f505e2e
