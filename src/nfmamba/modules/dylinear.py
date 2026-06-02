"""DyLinear reference stabilizer.

Element-wise norm-free operator:

    y = alpha * x

``alpha`` is per-channel learnable. There is no squashing — the operator is
*purely linear*. The reason this lives in the ablation set is that it isolates
the contribution of the squashing nonlinearity itself: every other stabilizer
in this directory (DyT, Derf, DyISRU, DySoftSign, DyPowerP1) bends the curve
at large |x|; DyLinear does not.

Why this is interesting as a control
------------------------------------
1.  Removes the *only* nonlinearity introduced at the B/C slot. If a model
    trained with DyLinear matches one trained with DyT/Derf/etc., the
    squashing was never load-bearing — a learned channel gain alone suffices
    to replace BCNorm's per-token reduction.
2.  Provides a cheap lower bound. If DyLinear *diverges* but the squashing
    variants stay stable, that pins the role of saturation in keeping B/C
    activations from running away above 1.3B parameters (cf. Jamba report).
3.  Element-wise and reduction-free, so it preserves the decode-latency win
    that motivates this ablation track: no thread-block sync, no global
    reduction, just a per-channel multiply that fuses into surrounding I/O.

Init default
------------
``alpha_init = 1.0`` — exactly y = x at step 0 (identity), matching the DyT
convention. The operator is a learned channel gain; training writes the gain
in rather than imposing it from the start. Override via the registry
(``make_stabilizer("dylinear", d, alpha_init=5.0)``) for the BCNorm-matching
~5x gain start point.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DyLinear(nn.Module):
    """Per-channel learnable linear gain: y = alpha * x.

    Parameters
    ----------
    dim:
        Feature dimension (last axis of the input).
    alpha_init:
        Per-channel initial value for ``alpha``. Default 1.0 -> identity at
        step 0.
    """

    def __init__(
        self,
        dim: int,
        *,
        alpha_init: float = 1.0,
        device=None,
        **kwargs,
    ):
        super().__init__()
        self.alpha = nn.Parameter(
            torch.full((dim,), float(alpha_init), device=device)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.alpha.to(dtype=x.dtype) * x
