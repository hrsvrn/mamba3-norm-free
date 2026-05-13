"""DyT reference stabilizer."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DyT(nn.Module):
    """Dynamic tanh stabilizer.

    Element-wise stabilizer:

        y = alpha * tanh(x)

    `alpha` is learned per channel. There is no additive beta here because the
    Mamba-3 B/C path already applies a learned B/C bias after this slot.
    """

    def __init__(self, d: int, alpha_init: float = 1.0, device=None):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((d,), float(alpha_init), device=device))

    def forward(self, x: Tensor) -> Tensor:
        return self.alpha.to(dtype=x.dtype) * torch.tanh(x)
