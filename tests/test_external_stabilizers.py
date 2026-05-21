"""Smoke tests for experiment-side B/C stabilizers.

These tests intentionally construct a stock `mamba3-minimal` model first and
then install stabilizers from `nfmamba`. The reference implementation stays
untouched.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "mamba3-minimal"))

from mamba3 import create_toy_model, get_device  # noqa: E402
from nfmamba import Derf, DyPowerP1, DyISRU, DyT, ExternalRMSNorm, IdentityStabilizer, install_bc_stabilizer  # noqa: E402


DEVICE = get_device()
TOY_KW = {"d_model": 64, "n_layer": 1, "vocab_size": 32, "device": DEVICE}


def first_mixer(model):
    return model.backbone.layers[0].mixer


def make_model(seed: int = 123):
    torch.manual_seed(seed)
    model = create_toy_model(**TOY_KW)
    model.train()
    return model


def test_install_known_stabilizers():
    expected = {
        "rmsnorm": ExternalRMSNorm,
        "bcnorm": ExternalRMSNorm,
        "identity": IdentityStabilizer,
        "derf": Derf,
        "dypower_p1": DyPowerP1,
        "dyisru": DyISRU,
        "dyt": DyT,
    }
    for name, cls in expected.items():
        model = make_model()
        report = install_bc_stabilizer(model, name)
        mixer = first_mixer(model)
        assert report.replaced == 2
        assert isinstance(mixer.B_norm, cls)
        assert isinstance(mixer.C_norm, cls)


def test_external_rmsnorm_matches_reference_at_init():
    tokens = torch.randint(0, TOY_KW["vocab_size"], (1, 32), device=DEVICE)

    reference = make_model(seed=456).eval()
    external = make_model(seed=456).eval()
    install_bc_stabilizer(external, "rmsnorm")

    with torch.no_grad():
        ref_logits, _ = reference(tokens)
        ext_logits, _ = external(tokens)

    assert torch.allclose(ref_logits, ext_logits, atol=1e-6, rtol=1e-6)


def test_variants_forward_backward_smoke():
    tokens = torch.randint(0, TOY_KW["vocab_size"], (2, 32), device=DEVICE)
    for name in ("identity", "derf", "dypower_p1", "dyisru", "dyt"):
        model = make_model(seed=789)
        install_bc_stabilizer(model, name)

        logits, _ = model(tokens)
        assert logits.shape == (2, 32, model.args.vocab_size)
        assert torch.isfinite(logits).all()

        loss = logits.float().square().mean()
        loss.backward()

        grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert any(g is not None and torch.isfinite(g).all() for g in grads)


if __name__ == "__main__":
    test_install_known_stabilizers()
    test_external_rmsnorm_matches_reference_at_init()
    test_variants_forward_backward_smoke()
    print("external stabilizer tests passed!")
