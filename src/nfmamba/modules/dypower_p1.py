"""DyPowerP1 stabilizer.

Element-wise squashing function:

    y = x / (beta + alpha * |x|)

A per-channel softsign variant. Same functional form as ``DySoftSign`` (the
p = 1 case of ``DyPowerSign``), but with two differences chosen for this
ablation slot:

    1. ``alpha`` and ``beta`` are stored as plain ``nn.Parameter`` tensors
       (no log-parameterisation). This keeps the optimiser's view of the
       parameters identical to a standard affine and lets the squash relax
       to exact linearity when ``alpha = 0``.
    2. ``beta`` is initialised below 1 so the operator behaves as a
       ~4-5x uniform amplifier of its input at step 0, matching BCNorm's
       effective gain on typical B/C activations (RMSNorm divides by an
       RMS of ~0.2 -> ~5x gain, with the learnable scale at 1).

The function only starts to bend once ``alpha`` moves off zero, so training
begins from a known, well-conditioned linear regime and the squash is
*learned in* rather than imposed from the start.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DyPowerP1(nn.Module):
    """Element-wise stabilizer: y = x / (beta + alpha * |x|).

    Parameters
    ----------
    dim:
        Feature dimension (last axis of the input).
    alpha_init:
        Per-channel initial value for ``alpha``. Default 0.0 -> the operator
        is exactly linear at step 0.
    beta_init:
        Per-channel initial value for ``beta``. Default 0.2 -> the linear
        gain at step 0 is 1 / 0.2 = 5x, approximating BCNorm's effective
        scaling on typical B/C activations.
    """

    _EPS = 1e-6  # additive floor on denominator; bf16-safe

    def __init__(
        self,
        dim: int,
        *,
        alpha_init: float = 0.0,
        beta_init: float = 0.2,
        device=None,
        **kwargs,
    ):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((dim,), float(alpha_init), device=device))
        self.beta = nn.Parameter(torch.full((dim,), float(beta_init), device=device))

    def forward(self, x: Tensor) -> Tensor:
        alpha = self.alpha.to(dtype=x.dtype)
        beta = self.beta.to(dtype=x.dtype)
        return x / (beta + alpha * x.abs() + self._EPS)
