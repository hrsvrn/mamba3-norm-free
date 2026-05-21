"""String registry for B/C stabilizer modules."""

from __future__ import annotations

from torch import nn

from .derf import Derf
from .dypower_p1 import DyPowerP1
from .dyisru import DyISRU
from .dysoftsign import DySoftSign
from .dyt import DyT
from .identity import IdentityStabilizer
from .rmsnorm import ExternalRMSNorm


def make_stabilizer(name: str, d: int, *, device=None, **kwargs) -> nn.Module:
    """Create a B/C stabilizer module by experiment name."""

    name = name.lower()
    if name in {"rmsnorm", "bcnorm"}:
        return ExternalRMSNorm(d, device=device)
    if name == "identity":
        return IdentityStabilizer()
    if name == "derf":
        return Derf(d, device=device)
    if name == "dypower_p1":
        return DyPowerP1(d, device=device, **kwargs)
    if name == "dyisru":
        return DyISRU(d, device=device)
    if name in {"dysoftsign", "dysign", "softsign"}:
        return DySoftSign(d, device=device, **kwargs)
    if name == "dyt":
        return DyT(d, device=device)
    raise ValueError(f"Unsupported B/C stabilizer: {name}")
