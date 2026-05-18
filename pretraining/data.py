"""Streaming FineWebEdu loader. Tokenizes documents with tiktoken GPT-2 in
worker processes and yields fixed-length packed sequences.

We deliberately don't pre-tokenize to disk — 1B tokens of GPT-2 BPE fits in
~2GB so streaming is fast enough, and it keeps the cluster setup trivial."""

from __future__ import annotations

import os
import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import tiktoken
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info


@dataclass
class DataConfig:
    source: str
    source_config: str | None
    split: str
    text_column: str
    streaming: bool
    tokenizer: str
    add_eos: bool
    seq_len: int
    shuffle_buffer: int
    seed: int


class _Llama3TokenizerAdapter:
    """Adapt HF `AutoTokenizer` (Llama-3.1) to the tiktoken-ish surface that
    `PackedFineWebEdu` expects: `.encode_ordinary(text) -> list[int]` and
    `.eot_token: int`. Llama-3 lacks a single EOS BPE token equivalent to
    GPT-2's <|endoftext|>; we use the model's defined `eos_token_id`
    (`<|end_of_text|>`, id 128001 for Llama-3.1)."""

    def __init__(self, model_id: str = "meta-llama/Llama-3.1-8B"):
        from transformers import AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if self._tok.eos_token_id is None:
            raise RuntimeError(f"{model_id} tokenizer has no eos_token_id")
        self.eot_token = int(self._tok.eos_token_id)

    def encode_ordinary(self, text: str) -> list[int]:
        # `add_special_tokens=False` keeps BOS/EOS out of the doc body; the
        # packer inserts an EOS between documents itself.
        return self._tok.encode(text, add_special_tokens=False)


def _get_tokenizer(name: str):
    if name == "gpt2":
        return tiktoken.get_encoding("gpt2")
    if name == "llama3":
        return _Llama3TokenizerAdapter("meta-llama/Llama-3.1-8B")
    raise ValueError(f"Unknown tokenizer: {name}")


class PackedFineWebEdu(IterableDataset):
    """Stream FineWebEdu and emit `(seq_len + 1,)` int64 windows for causal LM.

    Each worker pulls a disjoint slice of the HF stream (rank-shard +
    worker-shard) so DDP and DataLoader workers don't double-consume rows."""

    def __init__(self, cfg: DataConfig, rank: int = 0, world_size: int = 1):
        super().__init__()
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size

    def _iter_documents(self):
        from datasets import load_dataset

        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1

        ds = load_dataset(
            self.cfg.source,
            name=self.cfg.source_config,
            split=self.cfg.split,
            streaming=self.cfg.streaming,
        )
        # Shard across (rank * num_workers + worker_id) of (world_size * num_workers).
        global_workers = self.world_size * num_workers
        global_idx = self.rank * num_workers + worker_id
        if global_workers > 1:
            ds = ds.shard(num_shards=global_workers, index=global_idx)

        if self.cfg.shuffle_buffer > 0 and self.cfg.streaming:
            ds = ds.shuffle(buffer_size=self.cfg.shuffle_buffer, seed=self.cfg.seed + global_idx)

        for row in ds:
            text = row.get(self.cfg.text_column)
            if text:
                yield text

    def __iter__(self):
        tok = _get_tokenizer(self.cfg.tokenizer)
        eos_id = tok.eot_token  # 50256 for GPT-2
        seq_len = self.cfg.seq_len
        window = seq_len + 1   # one extra token for the shift-by-one label

        buf: deque[int] = deque()
        for text in self._iter_documents():
            ids = tok.encode_ordinary(text)
            buf.extend(ids)
            if self.cfg.add_eos:
                buf.append(eos_id)

            while len(buf) >= window:
                chunk = [buf.popleft() for _ in range(window)]
                # Put the last token back so it becomes the first token of the
                # next window — standard "stride by seq_len, overlap by 1" pack.
                buf.appendleft(chunk[-1])
                yield torch.tensor(chunk, dtype=torch.long)


def build_dataloader(
    cfg: dict,
    seq_len: int,
    micro_batch_size: int,
    rank: int = 0,
    world_size: int = 1,
    num_workers: int = 4,
    seed: int = 0,
) -> DataLoader:
    d = cfg["data"]
    data_cfg = DataConfig(
        source=d["source"],
        source_config=d.get("source_config"),
        split=d["split"],
        text_column=d["text_column"],
        streaming=d["streaming"],
        tokenizer=d["tokenizer"],
        add_eos=d["add_eos"],
        seq_len=seq_len,
        shuffle_buffer=d.get("shuffle_buffer", 0),
        seed=seed,
    )
    dataset = PackedFineWebEdu(data_cfg, rank=rank, world_size=world_size)
    return DataLoader(
        dataset,
        batch_size=micro_batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
