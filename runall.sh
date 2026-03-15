uv sync
uv run python plot_flops_cache.py 
uv run python plot_flops_cache.py --config-name config_27b.yaml
uv run python benchmark_baselines.py 
uv run python benchmark_e2e.py 
