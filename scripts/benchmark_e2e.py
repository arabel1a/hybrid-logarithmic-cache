"""End-to-end prefix caching evaluation on real conversation traces.

Runs no_cache, block, and log strategies with FIFO prefix cache.
Attention cost is measured once (no_cache full vs GDN-only) and added
to checkpoint strategies per-request.

Results saved as per-request JSONL and accumulated summary.
"""
import spase_cache
import json
import logging
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from tqdm.auto import tqdm

from spase_cache.checkpoint_cache import (
    prefill_from_checkpoint,
)
from spase_cache.utils import (
    setup_output_dir,
    make_model,
    prefill_baseline,
    _model_device,
    _sync_device,
    disable_attention_layers,
    prefill_and_capture_at,
    PrefixCache,
    _save_jsonl,
)
from spase_cache.strategies import checkpoint_positions

log = logging.getLogger(__name__)

def load_requests(processed_path, vocab_size):
    """Load pre-tokenized requests from prepare_data.py output.

    Token IDs are clamped to vocab_size since the benchmark model may have
    a smaller vocabulary than the tokenizer (timing is token-value-independent).
    """
    data = torch.load(processed_path, weights_only=False)
    requests = []
    for conv_id, n_turns, ts, token_ids in data:
        input_ids = torch.tensor([token_ids], dtype=torch.long) % vocab_size
        requests.append((conv_id, n_turns, input_ids))
    log.info("Loaded %d requests from %s", len(requests), processed_path)
    return requests


def interleave(requests, seed):
    """Interleave conversations with Poisson arrival times, preserving turn order."""
    by_conv = defaultdict(list)
    for req in requests:
        by_conv[req[0]].append(req)

    rng = np.random.RandomState(seed)
    conv_ids = list(by_conv.keys())
    rng.shuffle(conv_ids)

    queues = {cid: list(by_conv[cid]) for cid in conv_ids}
    arrival = {cid: rng.exponential(1.0) for cid in conv_ids}
    ordered = []
    while queues:
        cid = min(queues, key=lambda c: arrival[c])
        ordered.append(queues[cid].pop(0))
        if queues[cid]:
            arrival[cid] += rng.exponential(1.0)
        else:
            del queues[cid]
            del arrival[cid]
    return ordered


def simulate(model, requests, strategy, cache_budget, block_size,
             skip_attention=False, progress=False):
    """Run all requests through the model with given caching strategy."""
    dev = _model_device(model)
    uses_cache = strategy != "no_cache"
    cache = PrefixCache(cache_budget) if uses_cache else None

    ctx = disable_attention_layers(model) if skip_attention else nullcontext()
    per_request = []

    with ctx:
        for conv_id, n_turns, input_ids in tqdm(requests, desc=strategy, disable=not progress):
            input_ids = input_ids.to(dev)
            seq_len = input_ids.shape[1]

            hit = False
            tokens_saved = 0
            cached_store = None

            if uses_cache:
                cached_store, _ = cache.find_best_prefix(conv_id, n_turns)

            if cached_store is not None:
                hit = True
                cached_store.to(dev)
                _sync_device(dev)
                t0 = time.perf_counter()
                prefill_from_checkpoint(model, input_ids, cached_store)
                _sync_device(dev)
                dt = time.perf_counter() - t0
                cached_store.to("cpu")
                ckpt = cached_store.best_checkpoint(seq_len)
                if ckpt:
                    tokens_saved = ckpt.position
            else:
                _sync_device(dev)
                t0 = time.perf_counter()
                prefill_baseline(model, input_ids)
                _sync_device(dev)
                dt = time.perf_counter() - t0

            # Capture checkpoints for cache
            if uses_cache:
                positions = checkpoint_positions(strategy, seq_len, block_size)
                store = prefill_and_capture_at(model, input_ids, positions)
                if skip_attention:
                    store.kv_cache_keys = {}
                    store.kv_cache_values = {}
                size = store.memory_bytes()
                store.to("cpu")
                cache.put((conv_id, n_turns), store, size)

            per_request.append({
                "conv_id": str(conv_id), "n_turns": n_turns, "seq_len": seq_len,
                "time_s": dt, "hit": hit, "tokens_saved": tokens_saved,
            })

    return {
        "total_time": sum(e["time_s"] for e in per_request),
        "hits": sum(1 for e in per_request if e["hit"]),
        "tokens_saved": sum(e["tokens_saved"] for e in per_request),
        "tokens_total": sum(e["seq_len"] for e in per_request),
        "cache_entries": cache.n_entries if cache else 0,
        "per_request": per_request,
    }


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    out_dir = setup_output_dir(cfg)
    model = make_model(cfg)
    dev = _model_device(model)
    config = model.config
    e2e = cfg.benchmark_e2e

    strategies = list(e2e.strategies)
    cache_budget = int(e2e.cache_budget_gb * 1e9)
    B = cfg.baseline.block_size
    progress = e2e.get("progress", True)

    # Load and interleave requests
    requests = load_requests(cfg.data.processed, config.vocab_size)
    requests = interleave(requests, cfg.seed)

    total_tokens = sum(ids.shape[1] for _, _, ids in requests)
    log.info("Total requests: %d, tokens: %d, cache budget: %.1f GB",
             len(requests), total_tokens, e2e.cache_budget_gb)

    # Warmup
    log.info("Warming up...")
    dummy = torch.randint(0, config.vocab_size, (1, 512)).to(dev)
    for _ in range(3):
        prefill_baseline(model, dummy)
    with disable_attention_layers(model):
        for _ in range(3):
            prefill_baseline(model, dummy)
    _sync_device(dev)

    summary = {
        "model_name": cfg.model.name,
        "block_size": B,
        "n_requests": len(requests),
        "total_tokens": total_tokens,
        "cache_budget_gb": e2e.cache_budget_gb,
        "strategies": {},
    }
    summary_path = out_dir / "e2e_summary.json"

    # Measure per-request attention cost (no_cache full vs GDN-only)
    has_ckpt_strategies = any(s != "no_cache" for s in strategies)
    res_full = None
    attn_costs = None

    if has_ckpt_strategies:
        log.info("Measuring attention cost (no_cache full vs GDN-only)...")
        res_full = simulate(model, requests, "no_cache", 0, B,
                            skip_attention=False, progress=progress)
        res_gdn = simulate(model, requests, "no_cache", 0, B,
                           skip_attention=True, progress=progress)
        attn_costs = [
            max(f["time_s"] - g["time_s"], 0)
            for f, g in zip(res_full["per_request"], res_gdn["per_request"])
        ]

    for strat in strategies:
        log.info("Strategy: %s", strat)

        if strat == "no_cache":
            if res_full is not None:
                res = res_full
            else:
                res = simulate(model, requests, "no_cache", 0, B, progress=progress)
        else:
            # Run with skipped attention, add per-request attn cost
            res = simulate(model, requests, strat, cache_budget, B,
                           skip_attention=True, progress=progress)
            for i, entry in enumerate(res["per_request"]):
                entry["time_s"] += attn_costs[i]
            res["total_time"] = sum(e["time_s"] for e in res["per_request"])

        log.info("  Total time: %.1fs", res["total_time"])
        if res["hits"] > 0:
            log.info("  Cache hits: %d/%d (%.1f%%)",
                     res["hits"], len(requests), res["hits"] / len(requests) * 100)

        # Save JSONL
        jsonl_path = out_dir / f"e2e_{strat}.jsonl"
        _save_jsonl(jsonl_path, res["per_request"])

        # Update summary
        summary["strategies"][strat] = {
            "total_time": res["total_time"],
            "hits": res["hits"],
            "tokens_saved": res["tokens_saved"],
            "tokens_total": res["tokens_total"],
            "cache_entries": res["cache_entries"],
        }
        summary_path.write_text(json.dumps(summary, indent=2))

    log.info("Done. Results saved to %s", out_dir)

if __name__ == "__main__":
    main()
