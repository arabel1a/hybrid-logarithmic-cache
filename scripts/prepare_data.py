"""Tokenize raw conversation data and compute prefix overlap statistics.

Usage:
  python prepare_data.py
  python prepare_data.py output_dir=outputs/overlap_statistics
"""
import json
import logging
from pathlib import Path

import polars as pl
import hydra
from omegaconf import DictConfig
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from spase_cache.utils import setup_output_dir

log = logging.getLogger(__name__)


class _TrieNode:
    __slots__ = ["c"]
    def __init__(self):
        self.c = {}

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


def build_requests(df):
    """Stack subsequent messages into cumulative LLM requests, keep only user-ending ones.

    Returns ts-sorted list of (conv_id, n_turns, ts, text).
    """
    requests = []
    for url in df["url"].unique(maintain_order=True).to_list():
        conv = df.filter(pl.col("url") == url).sort("message_index")
        roles = conv["role"].to_list()
        texts = conv["plain_text"].fill_null("").to_list()
        tss = conv["ts"].to_list()

        accumulated = ""
        for i, (role, text, ts) in enumerate(zip(roles, texts, tss)):
            if accumulated:
                accumulated += "\n"
            accumulated += f"<|{role}|> {text}"
            if role == "user":
                requests.append((url, i + 1, ts, accumulated))

    requests.sort(key=lambda x: x[2])
    log.info(f"Built {len(requests)} requests from {df.n_unique('url')} conversations")
    return requests


def tokenize(requests, tokenizer_name, max_seq_len):
    """Tokenize requests, truncate to max_seq_len, remove duplicates from truncation."""
    log.info(f"Loading tokenizer {tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    tokenized = []
    last_len = {}
    log.info("Tokenizing...")
    for conv_id, n_turns, ts, text in tqdm(requests):
        # if previous turn already hit max_seq_len, subsequent turns are identical after truncation
        if last_len.get(conv_id, 0) >= max_seq_len:
            continue
        enc = tokenizer(
            text,
            add_special_tokens=False,
            max_length=max_seq_len,
            truncation=True,
        )["input_ids"]
        tokenized.append((conv_id, n_turns, ts, enc))
        last_len[conv_id] = len(enc)

    log.info(f"Tokenized: {len(tokenized)} requests ({len(requests) - len(tokenized)} duplicates removed)")
    return tokenized


def compute_overlap(tokenized, out_dir):
    """Compute prefix overlap using trie over tokenized requests.

    tokenized: ts-sorted list of (conv_id, n_turns, ts, token_ids)
    """
    log.info("Computing prefix overlap distribution...")

    n_conversations = len(set(r[0] for r in tokenized))
    log.info(f"{len(tokenized)} chronological requests from {n_conversations} conversations")

    root = _TrieNode()
    lcp_lengths = []

    log.info("Simulating non-deleting prefix cache...")
    for conv_id, n_turns, ts, token_ids in tqdm(tokenized):
        node = root
        lcp = 0
        matched = True

        for t in token_ids:
            t_int = int(t)
            if matched and t_int in node.c:
                lcp += 1
                node = node.c[t_int]
            else:
                matched = False
                new_node = _TrieNode()
                node.c[t_int] = new_node
                node = new_node

        lcp_lengths.append(lcp)

    overlap_path = out_dir / "overlap_lcp.json"
    overlap_path.write_text(json.dumps({
        "lcp_lengths": lcp_lengths,
        "n_requests": len(tokenized),
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
    requests = build_requests(df)
    del df
    tokenized = tokenize(requests, cfg.model.tokenizer, cfg.data.max_seq_len)
    del requests

    # save tokenized requests for downstream LLM evaluation
    prepared_path = Path(cfg.data.processed)
    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    import torch
    torch.save(
        [(conv_id, n_turns, ts, token_ids) for conv_id, n_turns, ts, token_ids in tokenized],
        prepared_path,
    )
    log.info(f"Saved {len(tokenized)} tokenized requests to {prepared_path}")

    compute_overlap(tokenized, out_dir)

if __name__ == "__main__":
    main()
