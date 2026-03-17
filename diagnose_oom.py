"""
diagnose_oom.py - Why does log+attn OOM at 32767?

Part 1: Raw kernel memory (xformers SDPA, chunk_gated_delta_rule)
       → checks if Q!=K or initial_state changes memory behavior
Part 2: Model forward with different cache/store configurations
       → isolates whether store on GPU or fragmentation causes OOM

Run: python diagnose_oom.py
"""
import hybrid_logarithmic_cache
import gc, torch, hydra
from omegaconf import DictConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache
from hybrid_logarithmic_cache.checkpoint_cache import (
    apply_patch, make_model, prefill_baseline, prefill_from_checkpoint,
    PrefixCheckpointStore, RecurrentCheckpoint,
    _model_device, _get_linear_layers, _get_attention_layers,
)

def mb():  return torch.cuda.memory_allocated() / 1024**2
def pk():  return torch.cuda.max_memory_allocated() / 1024**2
def fr():  gc.collect(); torch.cuda.empty_cache()
def rs():  torch.cuda.reset_peak_memory_stats()


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    apply_patch()
    dev = torch.device(cfg.device)
    dtype = getattr(torch, cfg.dtype)
    N = 32767
    R = 16384  # log resume position (2^14)

    print(f"GPU: {torch.cuda.get_device_name()} — "
          f"{torch.cuda.get_device_properties(0).total_mem/1024**2:.0f}MB total")

    # ============================================================
    # Part 1: Raw kernel memory
    # ============================================================
    print("\n===== PART 1: Kernel memory by shape =====\n")

    from torch.nn.attention import sdpa_kernel, SDPBackend
    n_heads_attn = cfg.model.num_attention_heads  # 8 (post-GQA expansion)
    d_a = cfg.model.head_dim                      # 256
    n_v = cfg.model.linear_num_value_heads        # 16
    d_h = cfg.model.linear_value_head_dim         # 128

    # --- xformers SDPA ---
    print("xformers SDPA (post-GQA expansion: 8 heads, d=256):")
    for label, q_len, k_len in [
        ("full Q=32767, K=32767", N, N),
        ("log  Q=16383, K=32767", N - R, N),
        ("blk  Q=15,    K=32767", 15, N),
    ]:
        fr(); rs()
        Q = torch.randn(1, n_heads_attn, q_len, d_a, device=dev, dtype=dtype)
        K = torch.randn(1, n_heads_attn, k_len, d_a, device=dev, dtype=dtype)
        V = torch.randn(1, n_heads_attn, k_len, d_a, device=dev, dtype=dtype)
        inp = mb(); rs()
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            out = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
        print(f"  {label}  inp={inp:6.0f}MB  peak={pk():6.0f}MB  kernel={pk()-inp:5.0f}MB")
        del Q, K, V, out; fr()

    # --- GDN kernel (chunk_gated_delta_rule) ---
    print("\nchunk_gated_delta_rule (16 heads, d=128):")
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        for label, seq, use_init in [
            ("full  32767, no init ", N, False),
            ("log   16383, w/ init ", N - R, True),
            ("blk   15,    w/ init ", 15, True),
        ]:
            fr(); rs()
            q = torch.randn(1, seq, n_v, d_h, device=dev, dtype=dtype)
            k = torch.randn(1, seq, n_v, d_h, device=dev, dtype=dtype)
            v = torch.randn(1, seq, n_v, d_h, device=dev, dtype=dtype)
            beta = torch.randn(1, seq, n_v, d_h, device=dev, dtype=dtype).sigmoid()
            g = -torch.rand(1, seq, n_v, device=dev, dtype=torch.float32).abs()
            init = torch.randn(1, n_v, d_h, d_h, device=dev, dtype=torch.float32) if use_init else None
            inp = mb(); rs()
            out, state = chunk_gated_delta_rule(
                q, k, v, beta=beta, g=g,
                initial_state=init, output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            print(f"  {label}  inp={inp:6.0f}MB  peak={pk():6.0f}MB  kernel={pk()-inp:5.0f}MB")
            del q, k, v, g, beta, init, out, state; fr()
    except Exception as e:
        print(f"  skipped: {e}")

    # ============================================================
    # Part 2: Model-level memory
    # ============================================================
    print("\n===== PART 2: Model-level memory =====\n")

    model = make_model(cfg)
    ll = _get_linear_layers(model.config)
    al = _get_attention_layers(model.config)
    torch.manual_seed(42)
    ids = torch.randint(0, cfg.model.vocab_size, (1, N)).to(dev)

    # warmup
    prefill_baseline(model, ids[:, :1024]); fr()
    print(f"Base GPU (model+ids): {mb():.0f}MB\n")

    # --- Build store on CPU ---
    print("Building log store...")
    cache = Qwen3_5DynamicCache(config=model.config)
    with torch.no_grad():
        out = model(ids, past_key_values=cache, use_cache=True,
                    cache_position=torch.arange(N, device=dev))
    kv_k = {li: out.past_key_values.key_cache[li].cpu() for li in al}
    kv_v = {li: out.past_key_values.value_cache[li].cpu() for li in al}
    del out, cache; fr()

    cache = Qwen3_5DynamicCache(config=model.config)
    with torch.no_grad():
        out = model(ids[:, :R], past_key_values=cache, use_cache=True,
                    cache_position=torch.arange(R, device=dev))
    rec = {li: out.past_key_values.recurrent_states[li].cpu() for li in ll}
    conv = {li: out.past_key_values.conv_states[li].cpu() for li in ll}
    del out, cache; fr()

    store = PrefixCheckpointStore(
        prefix_tokens=ids.cpu(),
        checkpoints={R: RecurrentCheckpoint(R, rec, conv)},
        kv_cache_keys=kv_k, kv_cache_values=kv_v,
    )
    print(f"Store: {store.memory_bytes()/1024**2:.0f}MB\n")

    # A) Full forward (no cache) — known to work
    fr(); rs()
    ret = prefill_baseline(model, ids)
    print(f"A) prefill_baseline({N}):           peak={pk():.0f}MB")
    del ret; fr()

    # B) Incremental: build cache to R, forward remaining
    #    Same kernel shapes as capture's last segment
    fr(); rs()
    cache = Qwen3_5DynamicCache(config=model.config)
    with torch.no_grad():
        out = model(ids[:, :R], past_key_values=cache, use_cache=True,
                    cache_position=torch.arange(R, device=dev))
    cache = out.past_key_values; del out
    pre = mb(); rs()
    with torch.no_grad():
        out = model(ids[:, R:], past_key_values=cache, use_cache=True,
                    cache_position=torch.arange(R, N, device=dev))
    print(f"B) Incremental (→{R}, fwd {N-R}):   pre={pre:.0f}MB  peak={pk():.0f}MB")
    del out, cache; fr()

    # C) prefill_from_checkpoint (single run, clean GPU)
    fr(); rs()
    store.to(dev)
    pre = mb(); rs()
    ret = prefill_from_checkpoint(model, ids, store)
    print(f"C) from_checkpoint (1x, clean):     pre={pre:.0f}MB  peak={pk():.0f}MB")
    del ret; store.to("cpu"); fr()

    # D) 3x from_checkpoint (like _time n_runs=3)
    fr(); rs()
    store.to(dev)
    pre = mb()
    for i in range(3):
        rs()
        ret = prefill_from_checkpoint(model, ids, store)
        print(f"D.{i}) from_checkpoint run {i}:       peak={pk():.0f}MB  now={mb():.0f}MB")
        del ret
    store.to("cpu"); fr()

    # E) Simulate benchmark sequence: alloc/free cycles before log+attn
    #    Tests CUDA fragmentation from prior operations
    print(f"\nE) Simulated benchmark sequence (fragmentation test):")
    fr(); rs()

    # no_cache
    ret = prefill_baseline(model, ids)
    print(f"   no_cache done:     peak={pk():.0f}MB  now={mb():.0f}MB")
    del ret; fr(); rs()

    # attn_only (with disabled attn)
    ret = prefill_baseline(model, ids)
    del ret; fr(); rs()

    # block+attn (15 remaining tokens)
    store.to(dev)
    ret = prefill_from_checkpoint(model, ids, store)
    del ret; fr(); rs()
    # ^ store still on GPU here (like block_store in benchmark)

    # simulate block_store.to("cpu")
    store.to("cpu"); fr(); rs()

    # NOW: log+attn — the problematic step
    print(f"   before log+attn:   gpu={mb():.0f}MB")
    store.to(dev)
    print(f"   store on gpu:      gpu={mb():.0f}MB")
    rs()
    ret = prefill_from_checkpoint(model, ids, store)
    print(f"   log+attn done:     peak={pk():.0f}MB  now={mb():.0f}MB")
    del ret; store.to("cpu"); fr()

    print("\nDone.")


if __name__ == "__main__":
    main()
