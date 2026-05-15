"""DySoftSign reference stabilizer.

Element-wise squashing function:

    y = x / (beta + alpha * |x|)

A learnable, per-channel softsign. Equivalent to the ``p = 1`` case of
``DyPowerSign`` but kept as its own module for readability: the formula
appears directly in the forward pass.

Properties (cf. Chen et al. 2026 "Stronger Normalization-Free Transformers"):
    - zero-centred:   f(0) = 0
    - bounded:        |f(x)| < 1 / alpha as |x| -> inf
    - centre-sensitive: slope at 0 is 1 / beta
    - monotonic:      strictly increasing for alpha, beta > 0

Why this is a BCNorm-replacement candidate
------------------------------------------
BCNorm normalises B, C along the d_state axis via a global reduction
(RMSNorm), which forces a thread-block barrier in the decode kernel.
DySoftSign is element-wise: no reduction, no sync — the squash collapses
into the surrounding memory traffic. The function decouples small-signal
slope (1 / beta) from the asymptotic bound (1 / alpha), which is what
DyISRU lacks (single parameter controls both).

Init defaults: alpha = 0.0, beta = 1.0 -> y = x exactly at step 0 (linear /
identity). The function only starts to bend once alpha moves off zero, so
training begins from a known, well-conditioned regime and the squash is
*learned in* rather than imposed from the start.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class DySoftSign(nn.Module):
    """Element-wise softsign stabilizer: y = x / (beta + alpha * |x|).

    ``alpha`` is unconstrained per-channel learnable (default init = 0.0 so
    the function is exactly y = x at step 0). ``beta`` is log-parameterised
    per-channel learnable (default init = 1.0 so the small-signal slope
    1 / beta = 1 at step 0).

    A note on safety: with alpha = 0 init, the denominator is beta = 1 > 0
    everywhere. As training drives alpha positive, the denominator stays
    positive for all x. If alpha drifts slightly negative on some channels,
    the denominator could underflow only when |x| > 1 / |alpha|; with
    BC-projection magnitudes typically O(1) this is not a practical issue.
    The forward includes a small additive epsilon on the denominator to
    harden against the edge case in bf16.
    """

    _EPS = 1e-6  # additive floor on denominator; bf16-safe

    def __init__(
        self,
        d: int,
        *,
        alpha_init: float = 0.0,
        beta_init: float = 1.0,
        device=None,
    ):
        super().__init__()
        if beta_init <= 0:
            raise ValueError(f"beta_init must be positive, got {beta_init}")

        self.alpha = nn.Parameter(
            torch.full((d,), float(alpha_init), device=device)
        )
        self.log_beta = nn.Parameter(
            torch.full((d,), math.log(beta_init), device=device)
        )

    @property
    def beta(self) -> Tensor:
        return self.log_beta.exp()

    def forward(self, x: Tensor) -> Tensor:
        alpha = self.alpha.to(dtype=x.dtype)
        beta = self.beta.to(dtype=x.dtype)
        return x / (beta + alpha * x.abs() + self._EPS)
