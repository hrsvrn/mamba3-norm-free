"""Derf reference stabilizer.

Reference: Chen et al., *Stronger Normalization-Free Transformers* (2026).
arXiv:2512.10938v2.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class Derf(nn.Module):
    """Dynamic error-function stabilizer.

    Element-wise stabilizer:

        y = erf(alpha * x + s)

    where erf is the rescaled Gaussian cumulative distribution function.
    ``alpha`` is learned per channel and controls the input scale (init 0.5).
    ``s`` is a learned scalar shift (init 0.0) — a per-channel vector shows
    no benefit over a scalar (Table 15 of the Derf paper).  No multiplicative
    gamma or additive beta are included because the Mamba-3 B/C path already
    applies a learned per-channel bias after this slot.

    Properties satisfied (Chen et al. Section 3):
        - zero-centred (balanced around origin)
        - bounded (output in [-1, 1])
        - centre-sensitive (responsive near x = 0)
        - monotonic (preserves input ordering)
    """

    def __init__(self, d: int, alpha_init: float = 0.5, device=None):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((d,), float(alpha_init), device=device))
        self.s = nn.Parameter(torch.tensor(0.0, device=device))

    def forward(self, x: Tensor) -> Tensor:
        alpha = self.alpha.to(dtype=x.dtype)
        s = self.s.to(dtype=x.dtype)
        return torch.erf(alpha * x + s)
