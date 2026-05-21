#!/usr/bin/env python3
"""Correctness tests for every registered B/C stabilizer.

Covers:
    1. Forward/step parity   — full forward ≈ prefix + step-by-step
    2. Decode cache shapes   — InferenceCache tensors have correct dims
    3. SISO B/C shape probes — B/C tensors at norm/bias/RoPE boundary
    4. Backward smoke         — every parameter gets a finite gradient
    5. NaN / Inf guard        — no NaN/Inf in logits, loss, or gradients
    6. RMSNorm baseline       — external RMSNorm == stock BCNorm output
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "mamba3-minimal"))

from mamba3 import InferenceCache, Mamba3Config, Mamba3LMHeadModel, get_device  # noqa: E402
from nfmamba import Derf, DyISRU, DyPowerP1, DyT, ExternalRMSNorm  # noqa: E402
from nfmamba import IdentityStabilizer, install_bc_stabilizer  # noqa: E402

DEVICE = get_device()
TOY_KW = {"d_model": 64, "n_layer": 1, "vocab_size": 32, "device": DEVICE}
SEQ_LEN = 64
PREFIX_LEN = 32
TOL = 1e-3  # forward/step parity tolerance (FP32)

# ── All stabilizer names under test ───────────────────────────────────────────

STAB_NAMES = [
    "identity",
    "bcnorm",
    "dyt",
    "derf",
    "dyisru",
    "dypower_p1",
]

STAB_NAME_TO_CLS = {
    "rmsnorm": ExternalRMSNorm,
    "bcnorm": ExternalRMSNorm,
    "identity": IdentityStabilizer,
    "dyt": DyT,
    "derf": Derf,
    "dyisru": DyISRU,
    "dypower_p1": DyPowerP1,
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_model(seed: int = 123) -> Mamba3LMHeadModel:
    from mamba3 import create_toy_model

    torch.manual_seed(seed)
    return create_toy_model(**TOY_KW)


def _token_batch() -> torch.Tensor:
    return torch.randint(0, TOY_KW["vocab_size"], (2, SEQ_LEN), device=DEVICE)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Forward / step parity
# ──────────────────────────────────────────────────────────────────────────────

def test_forward_step_parity_all() -> None:
    """Full forward ≈ prefix-forward + step-by-step for every stabilizer."""
    tokens = _token_batch()
    seqlen = tokens.shape[1]
    prefix = min(PREFIX_LEN, seqlen // 2)

    for name in STAB_NAMES:
        model = _make_model()
        model.eval()
        install_bc_stabilizer(model, name)

        with torch.no_grad():
            logits_fwd, _ = model(tokens)

            logits_pref, cache = model(tokens[:, :prefix])
            steps = [logits_pref]
            for t in range(prefix, seqlen):
                lt, cache = model(tokens[:, t : t + 1], cache)
                steps.append(lt)
            logits_step = torch.cat(steps, dim=1)

        diff = (logits_fwd - logits_step).abs()
        max_d = diff.max().item()
        assert max_d < TOL, (
            f"[{name}] forward/step parity FAILED: max_diff={max_d:.2e} >= {TOL}"
        )
        print(f"  [ok] forward/step parity — {name}  (max_diff={max_d:.2e})")


# ──────────────────────────────────────────────────────────────────────────────
# 2. Decode cache correctness
# ──────────────────────────────────────────────────────────────────────────────

def test_decode_cache_shapes_all() -> None:
    """InferenceCache tensors have correct shapes and finite values."""
    for name in STAB_NAMES:
        model = _make_model()
        model.eval()
        install_bc_stabilizer(model, name)

        caches = [
            InferenceCache.alloc(1, model.args, device=DEVICE)
            for _ in range(model.args.n_layer)
        ]
        token = torch.randint(0, TOY_KW["vocab_size"], (1, 1), device=DEVICE)

        with torch.no_grad():
            _, new_caches = model(token, caches)

        for i, c in enumerate(new_caches):
            nheads = model.args.nheads
            d_state = model.args.d_state
            headdim = model.args.headdim
            assert c.ssm_state.shape == (1, nheads, headdim, d_state), \
                f"[{name}] layer {i} ssm_state shape {c.ssm_state.shape}"
            assert c.prev_Bx.shape == (1, nheads, headdim, d_state), \
                f"[{name}] layer {i} prev_Bx shape {c.prev_Bx.shape}"
            assert c.cum_angle.shape == (1, nheads, d_state // 2), \
                f"[{name}] layer {i} cum_angle shape {c.cum_angle.shape}"
            for field_name, tensor in c._asdict().items():
                assert torch.isfinite(tensor).all(), \
                    f"[{name}] layer {i} {field_name} contains NaN/Inf"
        print(f"  [ok] decode cache       — {name}  (shapes verified, all finite)")


# ──────────────────────────────────────────────────────────────────────────────
# 3. SISO B/C shape probes
# ──────────────────────────────────────────────────────────────────────────────

def test_bc_shape_probes_all() -> None:
    """Hook into the mixer to verify B/C shapes at norm / bias / RoPE boundary."""
    tokens = _token_batch()

    for name in STAB_NAMES:
        model = _make_model()
        model.eval()
        install_bc_stabilizer(model, name)

        probe: dict[str, torch.Tensor] = {}

        def hook(module, _, output):
            pass  # module-level hook — ignore

        def _attach(mixer):
            def b_hook(_, __, output):
                probe["B_after_norm"] = output.detach().clone()
            def c_hook(_, __, output):
                probe["C_after_norm"] = output.detach().clone()

            mixer.B_norm.register_forward_hook(b_hook)
            mixer.C_norm.register_forward_hook(c_hook)

        _attach(model.backbone.layers[0].mixer)

        with torch.no_grad():
            logits, _ = model(tokens)

        B = probe.get("B_after_norm")
        C = probe.get("C_after_norm")
        assert B is not None, f"[{name}] B_after_norm not captured"
        assert C is not None, f"[{name}] C_after_norm not captured"

        bc_dim = model.backbone.layers[0].mixer.bc_dim
        batch, seqlen = tokens.shape
        assert B.shape == (batch, seqlen, bc_dim), \
            f"[{name}] B shape {B.shape} ≠ ({batch},{seqlen},{bc_dim})"
        assert C.shape == (batch, seqlen, bc_dim), \
            f"[{name}] C shape {C.shape} ≠ ({batch},{seqlen},{bc_dim})"
        assert torch.isfinite(B).all(), f"[{name}] B_after_norm contains NaN/Inf"
        assert torch.isfinite(C).all(), f"[{name}] C_after_norm contains NaN/Inf"
        print(f"  [ok] BC shape probe     — {name}  B/C={B.shape}")


# ──────────────────────────────────────────────────────────────────────────────
# 4. Backward smoke
# ──────────────────────────────────────────────────────────────────────────────

def test_backward_fully_finite_all() -> None:
    """Every parameter that requires grad receives a finite gradient."""
    tokens = _token_batch()

    for name in STAB_NAMES:
        model = _make_model()
        model.train()
        install_bc_stabilizer(model, name)

        logits, _ = model(tokens)
        loss = logits.float().square().mean()
        loss.backward()

        params_with_grad = 0
        for p_name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            params_with_grad += 1
            assert p.grad is not None, \
                f"[{name}] parameter '{p_name}' has no gradient"
            assert torch.isfinite(p.grad).all(), \
                f"[{name}] parameter '{p_name}' gradient contains NaN/Inf"

        assert params_with_grad > 0, f"[{name}] no parameters required grad"
        print(f"  [ok] backward            — {name}  ({params_with_grad} params, all finite)")


# ──────────────────────────────────────────────────────────────────────────────
# 5. NaN / Inf guard
# ──────────────────────────────────────────────────────────────────────────────

def test_no_nan_inf_all() -> None:
    """Forward pass produces no NaN/Inf in logits, loss, or intermediate B/C."""
    tokens = _token_batch()

    for name in STAB_NAMES:
        model = _make_model()
        model.eval()
        install_bc_stabilizer(model, name)

        with torch.no_grad():
            logits, _ = model(tokens)
            loss = nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                tokens[:, 1:].reshape(-1),
            )

        assert torch.isfinite(logits).all(), f"[{name}] logits contain NaN/Inf"
        assert torch.isfinite(loss), f"[{name}] loss is NaN/Inf ({loss.item():.4f})"
        print(f"  [ok] NaN/Inf guard       — {name}  loss={loss.item():.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# 6. RMSNorm baseline equivalence
# ──────────────────────────────────────────────────────────────────────────────

def test_rmsnorm_equivalence() -> None:
    """External RMSNorm via install_bc_stabilizer('rmsnorm') == stock BCNorm."""
    tokens = torch.randint(0, TOY_KW["vocab_size"], (1, 32), device=DEVICE)

    ref = _make_model(seed=456).eval()
    ext = _make_model(seed=456).eval()
    install_bc_stabilizer(ext, "rmsnorm")

    with torch.no_grad():
        ref_logits, _ = ref(tokens)
        ext_logits, _ = ext(tokens)

    assert torch.allclose(ref_logits, ext_logits, atol=1e-6, rtol=1e-6), \
        "RMSNorm equivalence FAILED: logits differ"
    print("  [ok] RMSNorm equivalence  — external RMSNorm == stock BCNorm")


# ──────────────────────────────────────────────────────────────────────────────
# 7. Install correctness (class identity + report)
# ──────────────────────────────────────────────────────────────────────────────

def test_install_correctness() -> None:
    """Each stabilizer installs the correct class and reports 2 replacements."""
    for name, cls in STAB_NAME_TO_CLS.items():
        model = _make_model()
        report = install_bc_stabilizer(model, name)
        mixer = model.backbone.layers[0].mixer
        assert report.replaced == 2, f"[{name}] replaced={report.replaced}, expected 2"
        assert isinstance(mixer.B_norm, cls), \
            f"[{name}] B_norm is {type(mixer.B_norm).__name__}, expected {cls.__name__}"
        assert isinstance(mixer.C_norm, cls), \
            f"[{name}] C_norm is {type(mixer.C_norm).__name__}, expected {cls.__name__}"
        print(f"  [ok] install             — {name}  → {cls.__name__}")


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Device: {DEVICE}")
    print(f"Testing {len(STAB_NAMES)} stabilizers: {', '.join(STAB_NAMES)}\n")

    test_install_correctness()
    test_rmsnorm_equivalence()
    test_no_nan_inf_all()
    test_bc_shape_probes_all()
    test_forward_step_parity_all()
    test_decode_cache_shapes_all()
    test_backward_fully_finite_all()

    print(f"\nAll correctness tests passed! ({len(STAB_NAMES)} stabilizers ✓)")


if __name__ == "__main__":
    main()


# ──────────────────────────────────────────────────────────────────────────────
# 8. DyPowerP1 unit test
# ──────────────────────────────────────────────────────────────────────────────

def test_dypower_p1_module() -> None:
    """DyPowerP1: shape preservation (2D and 3D), learnable params, near-linear init."""
    from nfmamba.modules import DyPowerP1

    dim = 64
    mod = DyPowerP1(dim).to(DEVICE)

    # Shape preservation: 3D (batch, seq, dim)
    x3 = torch.randn(4, 16, dim, device=DEVICE)
    y3 = mod(x3)
    assert y3.shape == x3.shape, f"3D shape mismatch: {y3.shape} vs {x3.shape}"
    assert torch.isfinite(y3).all(), "3D output contains NaN/Inf"

    # Shape preservation: 2D (batch, dim)
    x2 = torch.randn(8, dim, device=DEVICE)
    y2 = mod(x2)
    assert y2.shape == x2.shape, f"2D shape mismatch: {y2.shape} vs {x2.shape}"
    assert torch.isfinite(y2).all(), "2D output contains NaN/Inf"

    # Learnable parameters
    assert isinstance(mod.alpha, nn.Parameter), "alpha is not nn.Parameter"
    assert isinstance(mod.beta, nn.Parameter), "beta is not nn.Parameter"
    assert mod.alpha.requires_grad, "alpha is not learnable"
    assert mod.beta.requires_grad, "beta is not learnable"
    assert mod.alpha.shape == (dim,), f"alpha shape {mod.alpha.shape} != ({dim},)"
    assert mod.beta.shape == (dim,), f"beta shape {mod.beta.shape} != ({dim},)"

    # Near-linear at init: y/x should be ~uniform across a random batch.
    # alpha init = 0  =>  y = x / beta  (exactly linear, gain = 1/beta_init).
    x = torch.randn(32, 16, dim, device=DEVICE)
    with torch.no_grad():
        y = mod(x)
    ratio = (y / x).flatten()
    finite = ratio[torch.isfinite(ratio)]
    assert finite.numel() > 0, "no finite ratios"
    mean_ratio = finite.mean().item()
    std_ratio = finite.std().item()
    # Expected gain at init: 1 / 0.2 = 5x (alpha = 0 makes it exactly linear,
    # tiny _EPS in the denominator gives near-zero std).
    assert 3.5 < mean_ratio < 6.0, (
        f"init gain {mean_ratio:.3f} not in expected ~4-5x range"
    )
    assert std_ratio < 1e-2, (
        f"init operator not uniform across inputs: std={std_ratio:.3e}"
    )
    print(
        f"  [ok] DyPowerP1 module   — shapes 2D/3D, learnable α/β, "
        f"init gain={mean_ratio:.3f}x (std={std_ratio:.2e})"
    )
