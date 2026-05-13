"""DyPowerSign reference stabilizer.

A parametric family of element-wise squashing functions:

    f(x) = x / (beta + |alpha * x|^p)^(1/p)

Decouples small-signal slope (1 / beta^(1/p)) from the asymptotic bound
(±1 / alpha), addressing the coupling limitation of DyISRU where both
are controlled by a single parameter.

Special cases:
    p=1 → DySoftSign     x / (beta + |alpha * x|)
    p=2 → DyISRU          x / sqrt(beta + alpha * x^2)   (beta=1 recovers exact DyISRU)
    p=3 → DyCubic         x / cbrt(beta + |alpha * x|^3)

The constraint p=q ensures simultaneous boundedness and monotonicity;
setting p ≠ q breaks monotonicity and violates the Derf-paper properties.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _validate_power(p: float) -> None:
    if p <= 0:
        raise ValueError(f"Power p must be positive, got {p}")


class DyPowerSign(nn.Module):
    """Element-wise power-sign squashing function.

        y = x / (beta + |alpha * x|^p)^(1/p)

    ``alpha`` is learned per channel and controls the asymptote (±1/α).
    ``beta`` is learned per channel and controls the small-signal slope
    (derivative at zero is 1 / beta^(1/p)).  ``beta`` is log-parameterised
    to guarantee positivity.  ``p`` is fixed at construction time.

    No gamma or beta multiplier is included — the Mamba-3 B/C path applies
    a learned bias after this slot.
    """

    def __init__(
        self,
        d: int,
        *,
        p: float = 2.0,
        alpha_init: float = 1.0,
        beta_init: float = 1.0,
        device=None,
    ):
        super().__init__()
        _validate_power(p)
        self._p = float(p)

        self.alpha = nn.Parameter(torch.full((d,), float(alpha_init), device=device))

        if beta_init <= 0:
            raise ValueError("beta_init must be positive")
        init = torch.full((d,), float(beta_init), device=device)
        self.log_beta = nn.Parameter(init.log())

    @property
    def beta(self) -> Tensor:
        return self.log_beta.exp()

    @property
    def p(self) -> float:
        return self._p

    def forward(self, x: Tensor) -> Tensor:
        alpha = self.alpha.to(dtype=x.dtype)
        beta = self.beta.to(dtype=x.dtype)
        p = self._p

        denominator = (beta + (alpha * x).abs().pow(p)).pow(1.0 / p)
        return x / denominator
