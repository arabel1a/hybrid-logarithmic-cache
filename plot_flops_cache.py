"""Theoretical FLOP and cache-size plots for prefix caching strategies."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    m = cfg.model
    ma = cfg.memory_analysis
    out= ma.out

    d, d_ff = m.hidden_size, m.intermediate_size
    n_q, n_kv, d_a = m.num_attention_heads, m.num_key_value_heads, m.head_dim
    n_v, n_qk, d_h = m.linear_num_value_heads, m.linear_num_key_heads, m.linear_value_head_dim
    gdn_per_group, ga_per_group = m.gdn_layers, m.ga_layers

    num_gdn_layers = m.gdn_layers
    num_attn_layers = m.ga_layers
    bpe = ma.bytes_per_element
    seq_lens = ma.seq_lens

    # FLOPs per token per layer group
    n_layers = gdn_per_group + ga_per_group  # layers per group
    flop_ffn = n_layers * 3 * d * d_ff
    flop_ga_proj = 3 * d * n_q * d_a + 2 * d * n_kv * d_a
    flop_gdn_proj = gdn_per_group * (3 * d * n_v * d_h
                     + 2 * d * n_qk * d_h
                     + 2 * d * n_v)
    flop_gdn_rec = gdn_per_group * n_v * d_h ** 2

    # GA quadratic attention: n_q * d_a * N per token (sequence-length dependent)
    flop_ga_quad_per_tok = n_q * d_a  # multiply by N later
    flop_ga_linear = flop_ffn * ga_per_group / n_layers + flop_ga_proj

    # G * GDN recompute: GDN layers + their FFNs
    gdn_ffn_frac = gdn_per_group / n_layers
    flop_gdn_recompute = flop_ffn * gdn_ffn_frac + flop_gdn_proj + flop_gdn_rec

    # Cache sizes per element
    attn_kv_per_token = n_kv * d_a * 2 * bpe
    gdn_state_per_ckpt = gdn_per_group * n_v * d_h ** 2 * bpe

    # Sequence lengths from config
    N = np.arange(min(seq_lens), max(seq_lens) + 1)

    # FLOP recomputation 
    flops_no_cache = N * flop_ga_linear + N * flop_gdn_recompute + N * N * flop_ga_quad_per_tok
    
    # Flop saved:
    flops_saved_attn_only = N * N * flop_ga_quad_per_tok + N * flop_ga_linear

    def flops_saved_block(N, B):
        return (N - N%B) * flop_gdn_recompute

    def recompute_saved_log(N):
        k = np.floor(np.log2(N)).astype(int)
        return 2 ** k

    flops_saved_log = recompute_saved_log(N) * flop_gdn_recompute

    # Cache sizes (full model → GB)
    to_gb = lambda x: x  / 1e9

    cache_attn_only = to_gb(N * attn_kv_per_token)
    cache_log = to_gb(np.floor(np.log2(N) + 1) * gdn_state_per_ckpt)

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 12))

    ax1.plot(N, flops_no_cache, label="No caching", color="black", lw=2)
    ax1.plot(N, flops_no_cache - flops_saved_attn_only, label="Attention-only KV cache", color="tab:red", lw=2)
    for B, lbl, c in [(16, "CUDA", "tab:blue")]:# , (32, "AMD", "tab:cyan"), (128, "Ascend", "tab:purple")]:
        ax1.plot(N, flops_no_cache - flops_saved_block(N, B), label=f"Block hybrid B={B} ({lbl})", color=c, lw=1.5, alpha=0.8)
    ax1.plot(N, flops_no_cache - flops_saved_log, label="Logarithmic", color="tab:orange", lw=2, ls="--")

    ax1.plot(N, flops_no_cache - flops_saved_attn_only - flops_saved_block(N, B), label="Attention + Block hybrid", color="tab:green", lw=2)
    ax1.plot(N, flops_no_cache - flops_saved_attn_only - flops_saved_log, label="Attention + Logarithnic", color="tab:purple", lw=2)
    ax1.set_xlabel("Cached prefix length")
    ax1.set_ylabel("Recomputation FLOPs (full model)")
    ax1.set_title(f"Prefix recomputation cost — {m.name}")
    # ax1.set_xscale("log", base=2)
    # ax1.set_yscale("log")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, max(seq_lens))
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    ax2.plot(N, cache_attn_only, label="Attention-only KV", color="tab:red", lw=2)
    for B, lbl, c in [(16, "CUDA default", "tab:blue")]:# , (32, "AMD", "tab:cyan"), (128, "Ascend", "tab:purple")]:
        ax2.plot(N, to_gb(np.ceil(N/B) * gdn_state_per_ckpt) , label=f"Block hybrid B={B} ({lbl})", color=c, lw=1.5)
    ax2.plot(N, cache_log, label="Logarithmic", color="tab:orange", lw=2)
    ax2.plot(N, cache_attn_only + to_gb(np.ceil(N/B) * gdn_state_per_ckpt), label="Attention + Block hybrid", color="tab:green", lw=2)
    ax2.plot(N, cache_attn_only + cache_log, label="Attention + Logarithnic", color="tab:purple", lw=2)

    ax2.set_xlabel("Cached sequence length")
    ax2.set_ylabel("Cache size (GB, bf16, full model)")
    ax2.set_title(f"Cache memory per sequence — {m.name}")
    #ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(True, alpha=0.3)
    # ax2.set_xlim(1, max(seq_lens))
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
