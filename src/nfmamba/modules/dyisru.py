"""DyISRU reference stabilizer."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DyISRU(nn.Module):
    """Dynamic inverse-square-root unit.

    Element-wise stabilizer:

        y = x / sqrt(1 + alpha * x^2)

    `alpha` is learned per channel and constrained positive with an exp
    parameterization. The derivative at zero is 1, so it starts close to an
    identity map for small B/C activations while bounding large magnitudes.
    """

    def __init__(self, d: int, alpha_init: float = 1.0, device=None):
        super().__init__()
        if alpha_init <= 0:
            raise ValueError("alpha_init must be positive")
        init = torch.full((d,), float(alpha_init), device=device)
        self.log_alpha = nn.Parameter(init.log())

    @property
    def alpha(self) -> Tensor:
        return self.log_alpha.exp()

    def forward(self, x: Tensor) -> Tensor:
        alpha = self.alpha.to(dtype=x.dtype)
        return x * torch.rsqrt(1.0 + alpha * x.pow(2))
