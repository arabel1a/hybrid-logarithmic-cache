"""Tokenize raw conversation data and compute prefix overlap statistics.

Usage:
  python prepare_data.py
  python prepare_data.py output_dir=outputs/overlap_statistics
"""
import json
import logging
from pathlib import Path

import numpy as np
import polars as pl
import hydra
from omegaconf import DictConfig
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from spase_cache.utils import setup_output_dir

log = logging.getLogger(__name__)


_HASH_PRIME = 2654435761
_HASH_MASK = (1 << 63) - 1

def filter(raw_path, max_rounds, max_rows):
    log.info(f"Loading {raw_path}...")
    df = pl.read_csv(raw_path)
    df = df.with_columns(
        pl.col("message_create_time").str.replace("ts:", "").cast(pl.Float64).alias("ts")
    )

    # drop rows without timestamp
    df = df.filter(pl.col("ts").is_not_null())

    # keep only first max_rounds messages per conversation
    df = df.sort(["url", "message_index"])
    df = df.with_columns(
        pl.col("message_index").rank("ordinal").over("url").alias("_rank")
    ).filter(pl.col("_rank") <= max_rounds).drop("_rank")

    # limit total rows
    df = df.head(max_rows)

    log.info(f"Filtered: {df.n_unique('url')} conversations, {len(df)} rows")
    return df


def tokenize_messages(df, tokenizer_name, max_seq_len, chunk_size=4096):
    """Tokenize each message individually and add tokens column.

    Also formats each message as '<|role|> text' and concatenates consecutive
    same-role messages. Adds cumulative token length per conversation for
    truncation.

    Returns a DataFrame with columns: url, role, ts, message_index, tokens, cum_tokens.
    Only keeps rows where cum_tokens <= max_seq_len.
    """
    log.info(f"Loading tokenizer {tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    # format text as '<|role|> text' with \n separator between messages
    df = df.with_columns(
        (pl.col("message_index") == pl.col("message_index").min().over("url")).alias("_is_first")
    )
    df = df.with_columns(
        (pl.when(pl.col("_is_first"))
         .then(pl.lit("<|") + pl.col("role") + pl.lit("|> ") + pl.col("plain_text").fill_null(""))
         .otherwise(pl.lit("\n<|") + pl.col("role") + pl.lit("|> ") + pl.col("plain_text").fill_null(""))
        ).alias("_formatted")
    ).drop("_is_first")

    # tokenize in chunks, converting to numpy immediately to avoid Python int overhead
    # (Python int = 28 bytes each vs numpy int32 = 4 bytes)
    log.info("Tokenizing messages...")
    formatted = df["_formatted"]
    all_tokens = []
    for i in tqdm(range(0, len(formatted), chunk_size), desc="tokenizing chunks"):
        chunk_texts = formatted[i:i + chunk_size].to_list()
        for ids in tokenizer(chunk_texts, add_special_tokens=False)["input_ids"]:
            all_tokens.append(np.array(ids, dtype=np.int32))
        del chunk_texts
    df = df.drop("_formatted", "plain_text").with_columns(pl.Series("tokens", all_tokens))
    del all_tokens

    # cumulative token count per conversation for truncation
    df = df.with_columns(
        pl.col("tokens").list.len().cum_sum().over("url").alias("cum_tokens")
    )
    n_before = len(df)
    df = df.filter(pl.col("cum_tokens") <= max_seq_len)
    log.info(f"Tokenized {len(df)} messages ({n_before - len(df)} truncated)")
    return df


def _extract_overlap_data(df):
    """Extract lightweight data needed for overlap computation."""
    log.info("Computing prefix overlap distribution...")

    user_rows = df.filter(pl.col("role") == "user").sort("ts")
    n_conversations = user_rows.n_unique("url")
    log.info(f"{len(user_rows)} requests from {n_conversations} conversations")

    conv_tokens = {}
    for url in df["url"].unique(maintain_order=True).to_list():
        conv = df.filter(pl.col("url") == url).sort("message_index")
        conv_tokens[url] = (
            conv["message_index"].to_list(),
            [np.array(t, dtype=np.int32) for t in conv["tokens"].to_list()],
        )

    iter_urls = user_rows["url"].to_list()
    iter_msg_indices = user_rows["message_index"].to_list()
    return conv_tokens, iter_urls, iter_msg_indices, n_conversations


def compute_overlap(conv_tokens, iter_urls, iter_msg_indices, n_conversations, out_dir):
    """Compute prefix overlap using rolling hash over tokenized requests."""
    seen = set()  # rolling hashes at each token position
    lcp_lengths = []

    log.info("Simulating non-deleting prefix cache...")
    for url, msg_idx in tqdm(
        zip(iter_urls, iter_msg_indices),
        total=len(iter_urls),
    ):
        msg_indices, token_lists = conv_tokens[url]
        # find how many messages to include (up to and including msg_idx)
        n_msgs = 0
        for mi in msg_indices:
            n_msgs += 1
            if mi == msg_idx:
                break

        h = 0
        lcp = 0
        matched = True

        for tokens in token_lists[:n_msgs]:
            for t in tokens:
                h = (h * _HASH_PRIME + int(t) + 1) & _HASH_MASK
                if matched and h in seen:
                    lcp += 1
                else:
                    matched = False
                seen.add(h)

        lcp_lengths.append(lcp)

    overlap_path = out_dir / "overlap_lcp.json"
    overlap_path.write_text(json.dumps({
        "lcp_lengths": lcp_lengths,
        "n_requests": len(iter_urls),
        "n_conversations": n_conversations,
    }))

    n_hits = sum(1 for l in lcp_lengths if l > 0)
    log.info(f"{n_hits}/{len(lcp_lengths)} requests had cache hits "
             f"({n_hits/len(lcp_lengths)*100:.1f}%)")
    log.info(f"Overlap results saved to {overlap_path}")


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    out_dir = setup_output_dir(cfg)

    df = filter(cfg.data.raw_path, cfg.data.max_rounds, cfg.data.max_rows)
    df = tokenize_messages(df, cfg.model.tokenizer, cfg.data.max_seq_len, cfg.data.tokenizer_chunk_size)

    # save for downstream benchmark_e2e
    prepared_path = Path(cfg.data.processed)
    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(prepared_path)
    log.info(f"Saved {len(df)} messages to {prepared_path}")

    overlap_data = _extract_overlap_data(df)
    del df
    compute_overlap(*overlap_data, out_dir)

if __name__ == "__main__":
    main()
