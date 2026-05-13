"""PyTorch dataloader over a packed `.bin` shard.

The shard is a flat array of `uint16` token IDs. Sample `i` reads
`sequence_length + 1` tokens starting at `i * sequence_length`; the leading
`sequence_length` form `input_ids`, and the trailing `sequence_length`
(shifted by one) form `labels`. This is the standard nanoGPT layout and
makes seek/read O(1) without any per-sample copy of the underlying buffer.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class PackedDataset(Dataset):
    """Memory-mapped causal-LM dataset over a packed `.bin` file."""

    def __init__(
        self,
        bin_path: str | Path,
        sequence_length: int,
        dtype: np.dtype = np.uint16,
    ) -> None:
        self.bin_path = Path(bin_path)
        self.sequence_length = int(sequence_length)
        self.dtype = np.dtype(dtype)

        # mmap so workers (DataLoader num_workers > 0) share the page cache.
        self._data = np.memmap(self.bin_path, dtype=self.dtype, mode="r")

        n = self._data.shape[0]
        if n < self.sequence_length + 1:
            raise ValueError(
                f"{self.bin_path} has {n} tokens, need >= {self.sequence_length + 1}"
            )
        self._num_sequences = (n - 1) // self.sequence_length

    def __len__(self) -> int:
        return self._num_sequences

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0:
            idx += self._num_sequences
        if not 0 <= idx < self._num_sequences:
            raise IndexError(idx)
        start = idx * self.sequence_length
        # Cast uint16 -> int64 here; embedding layers require int64 indices.
        chunk = self._data[start : start + self.sequence_length + 1].astype(np.int64)
        chunk = torch.from_numpy(chunk)
        return {
            "input_ids": chunk[:-1].contiguous(),
            "labels": chunk[1:].contiguous(),
        }


def build_dataloader(
    bin_path: str | Path,
    sequence_length: int,
    batch_size: int,
    *,
    shuffle: bool = False,
    num_workers: int = 0,
    seed: Optional[int] = None,
    drop_last: bool = True,
    dtype: np.dtype = np.uint16,
) -> DataLoader:
    """Build a deterministic DataLoader.

    `shuffle=False` (default) yields sequences in on-disk order, which is
    what every BCNorm-vs-replacement comparison should use unless the
    experiment explicitly opts in.
    """
    dataset = PackedDataset(bin_path, sequence_length, dtype=dtype)

    generator = None
    if shuffle:
        if seed is None:
            raise ValueError("shuffle=True requires an explicit seed for reproducibility")
        generator = torch.Generator()
        generator.manual_seed(int(seed))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        generator=generator,
        pin_memory=False,
    )
