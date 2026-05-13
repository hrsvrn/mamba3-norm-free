"""RMSNorm/BCNorm reference module for external ablations."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ExternalRMSNorm(nn.Module):
    """RMSNorm equivalent used outside the reference implementation."""

    def __init__(self, d: int, eps: float = 1e-5, device=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d, device=device))

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
