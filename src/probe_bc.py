"""
probe_bc.py — Day 1 diagnostic for B/C tensor statistics.

What this script does:
  1. Instantiates a toy Mamba-3 model (create_toy_model).
  2. Registers forward hooks on B_norm and C_norm in every SSM layer to
     capture tensors both BEFORE and AFTER BCNorm.
  3. Runs a single forward pass with a random token batch.
  4. Prints per-layer shape / mean / std / max-abs for pre- and post-norm.
  5. Logs the same statistics to experiments/pilot_logs/week1_smoke/bc_stats.json.
  6. Runs three sanity checks: shape consistency, NaN/Inf guard, forward reproducibility.

Run from the repo root:
  python src/probe_bc.py
"""

import json
import math
import sys
import time
from pathlib import Path

import torch

# ── resolve imports ────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "mamba3-minimal"))
from mamba3 import Mamba3Config, Mamba3LMHeadModel, create_toy_model, get_device  # noqa: E402

LOG_DIR = REPO_ROOT / "experiments" / "pilot_logs" / "week1_smoke"
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Hook infrastructure
# ──────────────────────────────────────────────────────────────────────────────

class BCProbe:
    """
    Attaches forward hooks to every (B_norm, C_norm) pair in the model.
    Each hook records the tensor arriving at the module (pre-norm) and the
    tensor leaving it (post-norm) so we can measure what BCNorm actually does.
    """

    def __init__(self):
        self.records: list[dict] = []
        self._handles: list = []

    def attach(self, model: Mamba3LMHeadModel) -> None:
        for layer_idx, layer in enumerate(model.backbone.layers):
            mixer = layer.mixer
            self._attach_one(mixer.B_norm, layer_idx, "B")
            self._attach_one(mixer.C_norm, layer_idx, "C")

    def _attach_one(self, module, layer_idx: int, name: str) -> None:
        # pre-norm: captured via the input tuple of the forward hook
        def hook(mod, inp, out):
            pre = inp[0].detach().float()  # input to RMSNorm
            post = out.detach().float()    # output of RMSNorm
            self.records.append({
                "layer": layer_idx,
                "tensor": name,
                "shape": list(pre.shape),
                # ── pre-BCNorm stats ──
                "pre_mean":    pre.mean().item(),
                "pre_std":     pre.std().item(),
                "pre_max_abs": pre.abs().max().item(),
                # ── post-BCNorm stats ──
                "post_mean":    post.mean().item(),
                "post_std":     post.std().item(),
                "post_max_abs": post.abs().max().item(),
                # ── norm of each token-position vector (sanity: should be ≈1 after RMSNorm) ──
                "post_rms_mean": post.pow(2).mean(-1).sqrt().mean().item(),
            })
        h = module.register_forward_hook(hook)
        self._handles.append(h)

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ──────────────────────────────────────────────────────────────────────────────
# Pretty printing
# ──────────────────────────────────────────────────────────────────────────────

def _fmt(val: float, width: int = 9) -> str:
    return f"{val:>{width}.4f}"


def print_stats(records: list[dict]) -> None:
    header = (
        f"{'Layer':>5}  {'T':>1}  {'Shape':>22}  "
        f"{'pre_mean':>9}  {'pre_std':>9}  {'pre_max_abs':>11}  "
        f"{'post_mean':>9}  {'post_std':>9}  {'post_max_abs':>11}  {'post_rms':>8}"
    )
    print("\n" + "=" * len(header))
    print("B/C TENSOR STATISTICS (pre → post BCNorm)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in records:
        shape_str = "×".join(str(d) for d in r["shape"])
        print(
            f"{r['layer']:>5}  {r['tensor']:>1}  {shape_str:>22}  "
            f"{_fmt(r['pre_mean'])}  {_fmt(r['pre_std'])}  {_fmt(r['pre_max_abs'], 11)}  "
            f"{_fmt(r['post_mean'])}  {_fmt(r['post_std'])}  {_fmt(r['post_max_abs'], 11)}  "
            f"{_fmt(r['post_rms_mean'], 8)}"
        )
    print("=" * len(header) + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Sanity checks
# ──────────────────────────────────────────────────────────────────────────────

def check_shapes(records: list[dict], args: Mamba3Config) -> None:
    """Every B and C tensor should be (batch, seqlen, bc_dim)."""
    bc_dim = args.d_state * args.mimo_rank if args.use_mimo else args.d_state
    failures = []
    for r in records:
        expected_last = bc_dim
        actual_last = r["shape"][-1]
        if actual_last != expected_last:
            failures.append(
                f"Layer {r['layer']} {r['tensor']}: expected last dim {expected_last}, got {actual_last}"
            )
    if failures:
        for f in failures:
            print(f"[FAIL] shape check: {f}")
        raise AssertionError("Shape consistency check failed.")
    print("[PASS] shape consistency: all B/C tensors have correct bc_dim")


def check_nan(records: list[dict]) -> None:
    """Neither pre- nor post-BCNorm values should contain NaN or Inf."""
    failures = []
    for r in records:
        for key in ("pre_mean", "pre_std", "pre_max_abs", "post_mean", "post_std", "post_max_abs"):
            v = r[key]
            if math.isnan(v) or math.isinf(v):
                failures.append(f"Layer {r['layer']} {r['tensor']} {key} = {v}")
    if failures:
        for f in failures:
            print(f"[FAIL] NaN/Inf detected: {f}")
        raise AssertionError("Numerical stability check failed.")
    print("[PASS] NaN/Inf check: all values are finite")


def check_rms_norm_invariant(records: list[dict], tol: float = 0.05) -> None:
    """
    After RMSNorm the expected RMS of each vector is the norm weight (≈1 at init).
    Check that post_rms_mean is within tol of 1.0 for all layers.
    Failure here means the norm weights have drifted far from init or the norm
    is not being applied correctly.
    """
    failures = []
    for r in records:
        rms = r["post_rms_mean"]
        if abs(rms - 1.0) > tol:
            failures.append(f"Layer {r['layer']} {r['tensor']}: post_rms_mean = {rms:.4f} (expected ≈1.0)")
    if failures:
        for f in failures:
            print(f"[WARN] RMSNorm invariant: {f}")
        # Warning only — at init, norm weights are 1 but float scatter is expected
    else:
        print("[PASS] RMSNorm invariant: post_rms_mean ≈ 1.0 across all layers")


def check_forward_reproducibility(model, input_ids, device) -> None:
    """
    Two forward passes with torch.no_grad() on the same input must produce
    identical logits (bit-for-bit). Non-determinism here would undermine all
    future ablations.
    """
    model.eval()
    with torch.no_grad():
        logits_a, _ = model(input_ids)
        logits_b, _ = model(input_ids)
    max_diff = (logits_a - logits_b).abs().max().item()
    if max_diff > 0.0:
        print(f"[FAIL] reproducibility: max logit diff = {max_diff} (expected 0.0)")
        raise AssertionError("Forward pass is non-deterministic.")
    print(f"[PASS] reproducibility: max logit diff = {max_diff:.2e}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(42)
    device = get_device()
    print(f"Device: {device}")

    # ── Build toy model ────────────────────────────────────────────────────────
    # 4 layers, d_model=128, d_state=64, headdim=32 → bc_dim=64 per head
    model = create_toy_model(d_model=128, n_layer=4, vocab_size=256, device=device)
    model.eval()
    args: Mamba3Config = model.args
    print(
        f"Model: d_model={args.d_model}, n_layer={args.n_layer}, "
        f"d_state={args.d_state}, nheads={args.nheads}, headdim={args.headdim}, "
        f"bc_dim={args.d_state}, chunk_size={args.chunk_size}"
    )

    # ── Build input: batch=2, seqlen=128 (must be multiple of chunk_size=32) ──
    batch, seqlen = 2, 128
    input_ids = torch.randint(0, args.vocab_size, (batch, seqlen), device=device)

    # ── Attach probes ──────────────────────────────────────────────────────────
    probe = BCProbe()
    probe.attach(model)

    # ── Single forward pass ───────────────────────────────────────────────────
    t0 = time.perf_counter()
    with torch.no_grad():
        logits, _ = model(input_ids)
    elapsed = time.perf_counter() - t0
    print(f"Forward pass: {elapsed*1000:.1f} ms  |  logits shape: {list(logits.shape)}")

    probe.detach()

    # ── Print statistics ──────────────────────────────────────────────────────
    print_stats(probe.records)

    # ── Sanity checks ─────────────────────────────────────────────────────────
    print("Running sanity checks...")
    check_shapes(probe.records, args)
    check_nan(probe.records)
    check_rms_norm_invariant(probe.records)
    check_forward_reproducibility(model, input_ids, device)

    # ── Log to JSON ───────────────────────────────────────────────────────────
    run_meta = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": str(device),
        "model": {
            "d_model": args.d_model,
            "n_layer": args.n_layer,
            "d_state": args.d_state,
            "nheads": args.nheads,
            "headdim": args.headdim,
            "chunk_size": args.chunk_size,
            "use_mimo": args.use_mimo,
        },
        "input": {"batch": batch, "seqlen": seqlen},
        "forward_ms": round(elapsed * 1000, 2),
        "bc_stats": probe.records,
    }

    out_path = LOG_DIR / "bc_stats.json"
    with open(out_path, "w") as f:
        json.dump(run_meta, f, indent=2)
    print(f"\nLogged to: {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
