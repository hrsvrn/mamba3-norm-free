"""
bc_cosine.py — log cosine similarity between B and C in every Mamba-3 SSM layer.

Why this signal matters
-----------------------
B and C are the SSM analogues of "key" and "query": B writes information into
the recurrent state, C reads it out. After QK-Normalization (Mamba-3 §3.4) they
live on a unit sphere per token, so their cosine similarity is a clean,
scale-invariant measure of how much the two projections have collapsed onto the
same direction. A persistent drift toward `cos ≈ 1` would mean the layer has
degenerated into "read what you just wrote" — i.e. lost selectivity.

This probe is a first-class diagnostic for the BCNorm-replacement ablation: if
DyT/Derf/DyISRU/DySN cause the B/C geometry to drift differently from BCNorm,
we will see it here per-layer per-step long before it shows up in loss.

Design
------
Non-intrusive: the probe attaches forward hooks to `B_norm` and `C_norm` of
every SSM mixer in a stock `mamba3-minimal` model. Nothing in the reference
implementation is modified. This mirrors the pattern in
`nfmamba.diagnostics.probe_bc`.

Comparison point: post-QK-Norm, pre-bias, pre-RoPE.
  - Post-QK-Norm: the natural "query vs key" position; both vectors are unit-RMS.
  - Pre-bias: `BC_bias` is a data-independent additive shift; including it would
    mask the projection-level alignment we want to monitor.
  - Pre-RoPE: B and C share rotation angles, so RoPE preserves <B,C> exactly.
    Whether we measure pre- or post-RoPE is mathematically equivalent for the
    inner product; pre-RoPE is cheaper and avoids spurious dependence on the
    angle cache.

Reduction: per-token cosine, then mean over (batch, seqlen) → one scalar/layer.

Usage
-----
    from nfmamba.diagnostics import BCCosineProbe

    probe = BCCosineProbe(model)        # attaches hooks
    with torch.no_grad():
        logits, _ = model(input_ids)
    cos = probe.values()                # {layer_idx: float}
    probe.reset()                       # clear before next step
    # ... or probe.detach() to remove hooks entirely

Standalone smoke run:
    python -m nfmamba.diagnostics.bc_cosine
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# Resolve mamba3-minimal import without touching it.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "mamba3-minimal") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "mamba3-minimal"))


# ──────────────────────────────────────────────────────────────────────────────
# Probe
# ──────────────────────────────────────────────────────────────────────────────

class BCCosineProbe:
    """Attach forward hooks to every (B_norm, C_norm) pair in a Mamba-3 model.

    For each layer, captures the post-QK-Norm B and C tensors as they exit
    `B_norm` / `C_norm`, then computes mean cosine similarity reduced over
    (batch, seqlen). Results are accumulated across forward passes since the
    last `reset()` so callers can decide their own averaging window.

    Parameters
    ----------
    model : Mamba3LMHeadModel
        Any model exposing `.backbone.layers[i].mixer.{B_norm, C_norm}`.
    fp32 : bool, default True
        Cast B and C to fp32 before the cosine computation. Recommended:
        cosine of two near-orthogonal bf16 vectors loses meaningful bits in
        the dot product when feature dim is large.
    eps : float, default 1e-8
        Stability epsilon for `F.cosine_similarity`.
    """

    def __init__(self, model: nn.Module, *, fp32: bool = True, eps: float = 1e-8):
        self.fp32 = fp32
        self.eps = eps

        # Per-layer accumulators: list of (sum_cos, count) so multiple forward
        # passes can be averaged without losing token weighting.
        self._sum: dict[int, float] = {}
        self._count: dict[int, int] = {}

        # Per-layer scratch: stash B's post-norm output until C arrives (or vice
        # versa) within the same forward pass, then compute cosine and clear.
        self._scratch: dict[int, dict[str, Tensor]] = {}

        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._n_layers: int = 0
        self._attach(model)

    # ── public API ────────────────────────────────────────────────────────────

    def values(self) -> dict[int, float]:
        """Mean cosine per layer over all forward passes since last reset."""
        return {
            i: (self._sum[i] / self._count[i]) if self._count.get(i, 0) > 0 else float("nan")
            for i in range(self._n_layers)
        }

    def tensor(self, device: torch.device | str | None = None) -> Tensor:
        """Same as `values()` but as a 1-D tensor of shape (n_layers,)."""
        v = self.values()
        return torch.tensor([v[i] for i in range(self._n_layers)], device=device)

    def reset(self) -> None:
        """Drop all accumulated stats. Hooks remain attached."""
        self._sum.clear()
        self._count.clear()
        self._scratch.clear()

    def detach(self) -> None:
        """Remove all hooks. The probe is no longer usable after this."""
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._scratch.clear()

    # Context-manager sugar so callers can scope the probe to a single step.
    def __enter__(self) -> "BCCosineProbe":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.detach()

    # ── internals ─────────────────────────────────────────────────────────────

    def _attach(self, model: nn.Module) -> None:
        layers = getattr(getattr(model, "backbone", model), "layers", None)
        if layers is None:
            raise AttributeError(
                "BCCosineProbe expects a Mamba3LMHeadModel-shaped object "
                "(model.backbone.layers[i].mixer.{B_norm, C_norm})."
            )
        for i, layer in enumerate(layers):
            mixer = layer.mixer
            self._handles.append(
                mixer.B_norm.register_forward_hook(self._make_hook(i, "B"))
            )
            self._handles.append(
                mixer.C_norm.register_forward_hook(self._make_hook(i, "C"))
            )
            self._n_layers = i + 1

    def _make_hook(self, layer_idx: int, name: str):
        def hook(_module: nn.Module, _inp: tuple, out: Tensor) -> None:
            # Detach immediately: this is purely diagnostic, never on the graph.
            t = out.detach()
            if self.fp32:
                t = t.float()

            slot = self._scratch.setdefault(layer_idx, {})
            slot[name] = t

            if "B" in slot and "C" in slot:
                B, C = slot.pop("B"), slot.pop("C")
                # Cosine along the feature dim. B and C are (batch, seqlen, bc_dim)
                # in both SISO and MIMO (rank R is flattened into bc_dim here).
                cos = F.cosine_similarity(B, C, dim=-1, eps=self.eps)
                # Token-weighted mean so multi-pass accumulation is unbiased.
                self._sum[layer_idx] = self._sum.get(layer_idx, 0.0) + cos.sum().item()
                self._count[layer_idx] = self._count.get(layer_idx, 0) + cos.numel()

        return hook


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: one-shot summary for a single forward pass
# ──────────────────────────────────────────────────────────────────────────────

def bc_cosine_summary(model: nn.Module, input_ids: Tensor) -> dict[int, float]:
    """Run one forward pass under the probe and return per-layer mean cosine.

    Hooks are removed before returning, so this is safe to call ad-hoc inside
    eval scripts. For training-loop logging, prefer holding the probe live and
    calling `.values()` / `.reset()` per step.
    """
    model.eval()
    with BCCosineProbe(model) as probe:
        with torch.no_grad():
            model(input_ids)
        return probe.values()


# ──────────────────────────────────────────────────────────────────────────────
# Standalone smoke run
# ──────────────────────────────────────────────────────────────────────────────

def _main() -> None:
    from mamba3 import create_toy_model, get_device  # type: ignore[import-not-found]

    torch.manual_seed(42)
    device = get_device()
    print(f"Device: {device}")

    model = create_toy_model(d_model=128, n_layer=4, vocab_size=256, device=device)
    model.eval()
    args = model.args
    print(
        f"Model: d_model={args.d_model}, n_layer={args.n_layer}, "
        f"d_state={args.d_state}, bc_dim="
        f"{args.d_state * args.mimo_rank if args.use_mimo else args.d_state}"
    )

    batch, seqlen = 2, 128
    input_ids = torch.randint(0, args.vocab_size, (batch, seqlen), device=device)

    probe = BCCosineProbe(model)
    t0 = time.perf_counter()
    with torch.no_grad():
        model(input_ids)
    dt_ms = (time.perf_counter() - t0) * 1000

    cos = probe.values()
    probe.detach()

    print(f"\nForward + probe: {dt_ms:.1f} ms")
    print("\nLayer | mean cos(B, C)")
    print("------+---------------")
    for i in sorted(cos):
        print(f"{i:>5} | {cos[i]:+.4f}")

    # Sanity: at random init with QK-Norm, B and C are independent unit-RMS
    # vectors of dim bc_dim. Their expected cosine is ~0 with std ≈ 1/sqrt(bc_dim).
    bc_dim = args.d_state * args.mimo_rank if args.use_mimo else args.d_state
    expected_std = 1.0 / (bc_dim ** 0.5)
    max_abs = max(abs(v) for v in cos.values())
    if max_abs > 6 * expected_std:
        print(
            f"\n[WARN] max |cos| = {max_abs:.4f} exceeds 6σ ≈ {6*expected_std:.4f} "
            "for random init. Suspect a hook misfire or model misconfiguration."
        )
    else:
        print(
            f"\n[PASS] cosines consistent with random init (max |cos| = {max_abs:.4f}, "
            f"6σ ≈ {6*expected_std:.4f})"
        )

    log_dir = _REPO_ROOT / "experiments" / "pilot_logs" / "week1_smoke"
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = log_dir / "bc_cosine.json"
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": str(device),
        "model": {
            "d_model": args.d_model,
            "n_layer": args.n_layer,
            "d_state": args.d_state,
            "use_mimo": args.use_mimo,
            "mimo_rank": args.mimo_rank,
            "bc_dim": bc_dim,
        },
        "input": {"batch": batch, "seqlen": seqlen},
        "forward_ms": round(dt_ms, 2),
        "bc_cosine_per_layer": {str(k): v for k, v in cos.items()},
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nLogged to: {out_path.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    _main()
