"""Placement wrappers for BC stabilizer installation.

Preserves the reference implementation while allowing experiments to vary
where the stabilizer sits relative to BC bias:

    squash_before_bias=False  (default) → projection → norm → bias → RoPE → SSD
    squash_before_bias=True               → projection → bias → norm → RoPE → SSD

SISO only; MIMO has per-head broadcast bias semantics that do not commute
cleanly with an element-wise operation applied at the bc_dim level.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class PreBiasStabilizer(nn.Module):
    """Stabilizer that adds the original BC bias *before* squashing.

    The default Mamba‑3 pipeline is ``norm(bc) + bias``.  Swapping the order
    tests whether the element‑wise function and the bias addition commute
    (they do not, in general, because bias breaks zero‑centering).

    In Mamba‑3 SISO the B/C tensor before norm is ``(batch, seqlen, d_state)``
    — shared across heads.  The per‑head bias ``(nheads, d_state)`` is
    normally added *after* the norm via broadcasting.  To apply it before
    the norm we broadcast to ``(batch, seqlen, nheads, d_state)``, add the
    bias, average over heads, then apply the stabilizer over the resulting
    ``(batch, seqlen, d_state)`` tensor.  After installation the original
    ``mixer.B_bias`` / ``mixer.C_bias`` parameters are zeroed so the later
    addition in the SSM forward is a no‑op.
    """

    def __init__(
        self,
        stabilizer: nn.Module,
        bias_param: nn.Parameter,
        nheads: int,
    ) -> None:
        super().__init__()
        self.stabilizer = stabilizer
        self.nheads = nheads
        # Register a read‑only copy of the bias (per‑head).
        self.register_buffer("bias", bias_param.data.clone())

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, seqlen, d_state)  — per‑head copy, not yet expanded
        x = x.unsqueeze(2)                 # (b, l, 1, d_state)
        x = x + self.bias                  # + (nheads, d_state) → (b, l, nheads, d_state)
        x = x.mean(dim=2)                  # (b, l, d_state) — collapse heads
        return self.stabilizer(x)
