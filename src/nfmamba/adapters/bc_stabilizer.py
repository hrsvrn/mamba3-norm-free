"""External B/C stabilizer installer for Mamba-3 ablations.

The reference implementation in `mamba3-minimal/` stays untouched. Experiments
can build a stock model, then call `install_bc_stabilizer(...)` to replace the
`B_norm` and/or `C_norm` modules on that model instance.

Supports three placement modes:

    squash_before_bias=False  (default) → projection → norm → bias → RoPE → SSD
    squash_before_bias=True               → projection → bias → norm → RoPE → SSD

SISO only for the bias‑before‑norm path; MIMO has per‑head broadcast semantics
that do not commute cleanly with element‑wise operations at the bc_dim level.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterator

import torch
from torch import nn

from nfmamba.modules import make_stabilizer

from .placement import PreBiasStabilizer


@dataclass
class InstallReport:
    """Summary of an experiment-side B/C stabilizer install."""

    name: str
    replaced: int
    stabilize_b: bool
    stabilize_c: bool
    squash_before_bias: bool = False


def _iter_mixers(model: nn.Module) -> Iterator[nn.Module]:
    layers = getattr(getattr(model, "backbone", model), "layers", None)
    if layers is None:
        raise AttributeError(
            "Expected a Mamba3LMHeadModel-shaped object with "
            "`model.backbone.layers[i].mixer`."
        )
    for layer in layers:
        mixer = getattr(layer, "mixer", None)
        if mixer is None:
            raise AttributeError("Layer is missing `.mixer`.")
        yield mixer


def _module_device(module: nn.Module):
    try:
        return next(module.parameters()).device
    except StopIteration:
        return None


def _bc_dim(mixer: nn.Module) -> int:
    if hasattr(mixer, "bc_dim"):
        return int(mixer.bc_dim)
    norm = getattr(mixer, "B_norm", None) or getattr(mixer, "C_norm", None)
    weight = getattr(norm, "weight", None)
    if weight is not None:
        return int(weight.shape[-1])
    raise AttributeError("Could not infer B/C feature dimension from mixer.")


def _mixer_is_mimo(mixer: nn.Module) -> bool:
    return bool(getattr(mixer, "args", None) and mixer.args.use_mimo)


def install_bc_stabilizer(
    model: nn.Module,
    name: str,
    *,
    stabilize_b: bool = True,
    stabilize_c: bool = True,
    squash_before_bias: bool = False,
) -> InstallReport:
    """Replace B/C stabilizers on an already-constructed model instance.

    Parameters
    ----------
    model:
        A ``Mamba3LMHeadModel`` (constructed by ``mamba3-minimal``).
    name:
        Stabilizer name recognised by ``make_stabilizer()``
        (``"bcnorm"``, ``"dyt"``, ``"derf"``, ``"dyisru"``, ``"dysoftsign"``,
        ``"dypower_p1"``, ``"identity"``).
    stabilize_b:
        When ``False``, leave ``mixer.B_norm`` untouched.
    stabilize_c:
        When ``False``, leave ``mixer.C_norm`` untouched.
    squash_before_bias:
        When ``True``, add the original BC bias *before* the stabilizer and
        zero out the original ``mixer.B_bias`` / ``mixer.C_bias`` parameters.
        SISO only (MIMO will emit a warning and fall back to default order).
    """
    name = name.lower()

    if not stabilize_b and not stabilize_c:
        return InstallReport(
            name=name,
            replaced=0,
            stabilize_b=False,
            stabilize_c=False,
            squash_before_bias=False,
        )

    if squash_before_bias:
        for mixer in _iter_mixers(model):
            if _mixer_is_mimo(mixer):
                warnings.warn(
                    "squash_before_bias is not supported for MIMO — "
                    "falling back to norm‑before‑bias order for the first "
                    "MIMO layer.  All MIMO layers will use the default order.",
                    stacklevel=2,
                )
                break
        else:
            # All mixers are SISO — proceed with bias‑before‑norm
            pass

    replaced = 0
    for mixer in _iter_mixers(model):
        d = _bc_dim(mixer)
        device = _module_device(mixer)
        is_mimo = _mixer_is_mimo(mixer)

        if stabilize_b:
            stab = make_stabilizer(name, d, device=device)
            if squash_before_bias and not is_mimo:
                nheads = mixer.args.nheads
                stab = PreBiasStabilizer(stab, mixer.B_bias, nheads)
                with torch.no_grad():
                    mixer.B_bias.zero_()
            mixer.B_norm = stab
            replaced += 1

        if stabilize_c:
            stab = make_stabilizer(name, d, device=device)
            if squash_before_bias and not is_mimo:
                nheads = mixer.args.nheads
                stab = PreBiasStabilizer(stab, mixer.C_bias, nheads)
                with torch.no_grad():
                    mixer.C_bias.zero_()
            mixer.C_norm = stab
            replaced += 1

    # squash_before_bias is False if MIMO forced fallback
    effective_sbb = (
        squash_before_bias
        and not any(_mixer_is_mimo(m) for m in _iter_mixers(model))
    )

    return InstallReport(
        name=name,
        replaced=replaced,
        stabilize_b=stabilize_b,
        stabilize_c=stabilize_c,
        squash_before_bias=effective_sbb,
    )
