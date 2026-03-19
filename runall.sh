set -e

uv sync

echo "toy run"
uv run python scripts/prepare_data.py --config-name=toy
uv run python scripts/benchmark_single.py --config-name=toy
uv run python scripts/benchmark_e2e.py --config-name=toy
uv run python scripts/plot_results.py --config-name=toy
