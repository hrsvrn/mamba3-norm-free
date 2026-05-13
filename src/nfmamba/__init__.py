"""Experiment-side Mamba-3 ablation package.

This package must not modify the `mamba3-minimal` reference implementation.
Use these helpers to adapt a constructed reference model for ablations.
"""

from .adapters import InstallReport, PreBiasStabilizer, install_bc_stabilizer
from .modules import Derf, DyPowerSign, DyISRU, DyT, ExternalRMSNorm, IdentityStabilizer, make_stabilizer

__all__ = [
    "Derf",
    "DyPowerSign",
    "DyISRU",
    "DyT",
    "ExternalRMSNorm",
    "IdentityStabilizer",
    "InstallReport",
    "PreBiasStabilizer",
    "install_bc_stabilizer",
    "make_stabilizer",
]
