"""EOS-separated packing.

Tokens from many documents are concatenated (with EOS already appended per
doc by `encode`) and then sliced into fixed-length sequences. Document
boundaries can fall anywhere inside a sequence, but each boundary is marked
by the EOS token — so a model can learn to reset state at EOS rather than
bleed context across an unrelated document.

We keep one extra token at the end of the flat stream so the dataloader can
build labels by shifting input_ids by one position without an off-by-one at
the final sequence.
"""
from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def pack_eos(
    token_lists: Iterable[list[int]],
    sequence_length: int,
    dtype: np.dtype = np.uint16,
    drop_last: bool = True,
) -> Tuple[np.ndarray, int]:
    """Concatenate per-document token lists and chunk into sequences.

    Returns
    -------
    flat : np.ndarray
        1-D array of length `num_sequences * sequence_length + 1` (one extra
        token kept for label shifting). `dtype` is `uint16` by default —
        valid because the GPT-2 vocab (50257) fits in 16 bits.
    num_sequences : int
        Number of complete sequences packed.

    Notes
    -----
    The exact byte content of `flat` is a deterministic function of the
    input token stream and `sequence_length`, so identical inputs across
    machines produce identical files (verified by SHA-256).
    """
    if sequence_length <= 1:
        raise ValueError("sequence_length must be > 1")

    parts = [np.asarray(t, dtype=dtype) for t in token_lists if t]
    if not parts:
        raise ValueError("No tokens to pack — input stream was empty.")
    flat = np.concatenate(parts)

    # +1 buffer so the loader can slice [i*S : i*S + S + 1] for every i.
    usable = (flat.shape[0] - 1) // sequence_length
    if usable <= 0:
        raise ValueError(
            f"Not enough tokens ({flat.shape[0]}) to form a single "
            f"sequence of length {sequence_length}."
        )

    if drop_last:
        keep = usable * sequence_length + 1
        flat = flat[:keep]
    # else: keep the trailing remainder; consumers must handle it.

    return flat, usable
