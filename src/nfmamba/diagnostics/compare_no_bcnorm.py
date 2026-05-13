"""
compare_no_bcnorm.py — control experiment: BCNorm vs identity (no normalization).

Goal
----
Establish the *quantitative contribution* of BCNorm at initialization by running
two byte-identical models — one with BCNorm, one with `nn.Identity()` swapped in
for `B_norm`/`C_norm` — through the same input and comparing per-layer stats.

Why this is the right control
-----------------------------
The two models share every parameter (we deep-copy the state dict before the
swap), so the *raw* B and C projections are bit-identical between them. Any
difference in stats downstream is attributable to BCNorm and nothing else.

What to expect (theory)
-----------------------
1. **Magnitudes:**
   - BCNorm-post `post_rms ≈ 1.0` by construction (RMSNorm with weight=1 at init).
   - Identity-post `post_rms` = whatever the linear projection produces (the
     `pre_rms` from the BCNorm run). Compression ratio = `Id_rms / BC_rms`
     reveals how aggressively BCNorm is rescaling.
   - `post_max_abs` for BCNorm should be 3–5× post_std (Gaussian tail).
     Anything beyond that under Identity quantifies the heavy-tail problem.

2. **Geometry (cosine):**
   - At init with weight=1, RMSNorm is a uniform per-token positive scalar.
     Cosine is invariant under positive scalar rescaling, so
     `cos(BCNorm) ≈ cos(Identity)` at init. **This is a sanity check**, not a
     finding. Divergence here = bug.
   - During training the per-channel norm weight diverges from ones; that is
     where cosine fingerprints between variants will separate. Run this script
     on a trained checkpoint and the values WILL differ.

Run
---
    uv run python -m nfmamba.diagnostics.compare_no_bcnorm
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "mamba3-minimal") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "mamba3-minimal"))

from mamba3 import (  # noqa: E402
    Mamba3LMHeadModel,
    create_toy_model,
    get_device,
)

from nfmamba import install_bc_stabilizer  # noqa: E402
from .bc_cosine import BCCosineProbe  # noqa: E402
from .probe_bc import BCProbe  # noqa: E402
from nfmamba.utils.manifest import (  # noqa: E402
    gather_env,
    gather_git_state,
    lock_determinism,
    make_run_dir,
    write_manifest,
    write_summary,
)

# Constants — surfaced so they appear in the manifest verbatim.
EXPERIMENT = "compare_no_bcnorm"
DESCRIPTION = (
    "BCNorm vs nn.Identity control: byte-identical twin models, same input, "
    "compare per-layer B/C magnitude and cos(B,C) at random init."
)
SEED = 42
BATCH = 2
SEQLEN = 128
TOY_KW = {"d_model": 128, "n_layer": 4, "vocab_size": 256}
NUM_NOISE_BAND = 5e-3


def clone_model_with_shared_init(src: Mamba3LMHeadModel, device) -> Mamba3LMHeadModel:
    """Build a fresh model with the SAME architecture and copy all weights from src.

    We can't `deepcopy(src)` blindly because `Mamba3Config.__post_init__` is
    invoked through dataclass; instead we build a fresh model with the same
    config and load the source's state_dict. This keeps initialization
    byte-identical between the BCNorm and identity variants.
    """
    args = src.args
    twin = Mamba3LMHeadModel(args, device=device)
    twin.load_state_dict(copy.deepcopy(src.state_dict()))
    return twin


# ──────────────────────────────────────────────────────────────────────────────
# Stats container & comparison
# ──────────────────────────────────────────────────────────────────────────────

def _flatten_records(records: list[dict]) -> dict[tuple[int, str], dict]:
    """Re-key the BCProbe records by (layer_idx, tensor_name)."""
    return {(r["layer"], r["tensor"]): r for r in records}


def print_side_by_side(
    bc_records: list[dict],
    id_records: list[dict],
    bc_cos: dict[int, float],
    id_cos: dict[int, float],
    n_layers: int,
) -> None:
    """Two tables: magnitude comparison, then geometry comparison."""
    bc = _flatten_records(bc_records)
    idn = _flatten_records(id_records)

    # ── Table 1: post-stats comparison (BCNorm-post vs Identity-post) ──
    # For Identity the "post" output equals the "pre" — this IS the raw projection.
    header = (
        f"{'L':>2} {'T':>1} | "
        f"{'BC.post_std':>11} {'ID.post_std':>11} {'std_ratio':>9} | "
        f"{'BC.post_max':>11} {'ID.post_max':>11} {'max_ratio':>9} | "
        f"{'BC.post_rms':>11} {'ID.post_rms':>11}"
    )
    print("\n" + "=" * len(header))
    print("MAGNITUDE: BCNorm-post vs Identity-post (same raw projection)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for layer_idx in range(n_layers):
        for name in ("B", "C"):
            r_bc = bc[(layer_idx, name)]
            r_id = idn[(layer_idx, name)]
            std_ratio = r_id["post_std"] / max(r_bc["post_std"], 1e-12)
            max_ratio = r_id["post_max_abs"] / max(r_bc["post_max_abs"], 1e-12)
            print(
                f"{layer_idx:>2} {name:>1} | "
                f"{r_bc['post_std']:>11.4f} {r_id['post_std']:>11.4f} {std_ratio:>9.3f} | "
                f"{r_bc['post_max_abs']:>11.4f} {r_id['post_max_abs']:>11.4f} {max_ratio:>9.3f} | "
                f"{r_bc['post_rms_mean']:>11.4f} {r_id['post_rms_mean']:>11.4f}"
            )
    print("=" * len(header))

    # ── Table 2: cosine comparison ──
    cos_header = (
        f"{'L':>2} | {'BC cos(B,C)':>13} {'ID cos(B,C)':>13} {'abs_diff':>10}"
    )
    print("\n" + "=" * len(cos_header))
    print("GEOMETRY: per-layer mean cosine similarity between B and C")
    print("=" * len(cos_header))
    print(cos_header)
    print("-" * len(cos_header))
    max_abs_diff = 0.0
    for i in range(n_layers):
        diff = abs(bc_cos[i] - id_cos[i])
        max_abs_diff = max(max_abs_diff, diff)
        print(f"{i:>2} | {bc_cos[i]:>+13.6f} {id_cos[i]:>+13.6f} {diff:>10.2e}")
    print("=" * len(cos_header))
    print(
        f"\nmax |Δcos| across layers = {max_abs_diff:.2e}\n"
        "  (At init with RMSNorm weight=1, BCNorm is a uniform positive rescale\n"
        "   per token, so cosine is invariant. Expected: ≈ 0 within float scatter.)"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Determinism + environment capture ─────────────────────────────────────
    determinism = lock_determinism(SEED)
    git = gather_git_state(_REPO_ROOT)
    env = gather_env()
    device = get_device()
    print(f"Device: {device} | torch {env['torch']} | git {git['commit_short']}"
          + (" (dirty)" if git["dirty"] else ""))

    # ── Build BCNorm baseline ─────────────────────────────────────────────────
    bc_model = create_toy_model(**TOY_KW, device=device)
    bc_model.eval()
    args = bc_model.args
    bc_dim = args.d_state * args.mimo_rank if args.use_mimo else args.d_state
    print(
        f"Model: d_model={args.d_model}, n_layer={args.n_layer}, "
        f"d_state={args.d_state}, nheads={args.nheads}, headdim={args.headdim}, "
        f"bc_dim={bc_dim}"
    )

    # ── Build identity twin (same weights, BCNorm → Identity) ─────────────────
    id_model = clone_model_with_shared_init(bc_model, device=device)
    id_model.eval()
    install = install_bc_stabilizer(id_model, "identity")
    n_replaced = install.replaced
    print(f"Identity twin: replaced {n_replaced} norm modules with nn.Identity()")

    # ── Param sanity: every shared parameter is bit-identical ─────────────────
    bc_sd = bc_model.state_dict()
    id_sd = id_model.state_dict()
    shared = set(bc_sd) & set(id_sd)
    diffs = []
    for k in shared:
        if not torch.equal(bc_sd[k], id_sd[k]):
            diffs.append(k)
    if diffs:
        print(f"[FAIL] {len(diffs)} shared parameters differ: {diffs[:3]}...")
        raise AssertionError("Twin models are not parameter-identical.")
    print(f"[PASS] {len(shared)} shared parameters are bit-identical")

    # ── Same input for both ───────────────────────────────────────────────────
    # Re-seed immediately before sampling input_ids so the input is reproducible
    # independent of any RNG consumption above (model init advances the RNG).
    torch.manual_seed(SEED + 1)
    input_ids = torch.randint(0, args.vocab_size, (BATCH, SEQLEN), device=device)
    batch, seqlen = BATCH, SEQLEN

    # ── Run BCNorm model with both probes ─────────────────────────────────────
    bc_probe = BCProbe()
    bc_probe.attach(bc_model)
    bc_cos_probe = BCCosineProbe(bc_model)
    t0 = time.perf_counter()
    with torch.no_grad():
        bc_logits, _ = bc_model(input_ids)
    bc_ms = (time.perf_counter() - t0) * 1000
    bc_cos = bc_cos_probe.values()
    bc_probe.detach()
    bc_cos_probe.detach()
    print(f"\nBCNorm forward: {bc_ms:.1f} ms")

    # ── Run identity model with both probes ───────────────────────────────────
    id_probe = BCProbe()
    id_probe.attach(id_model)
    id_cos_probe = BCCosineProbe(id_model)
    t0 = time.perf_counter()
    with torch.no_grad():
        id_logits, _ = id_model(input_ids)
    id_ms = (time.perf_counter() - t0) * 1000
    id_cos = id_cos_probe.values()
    id_probe.detach()
    id_cos_probe.detach()
    print(f"Identity forward: {id_ms:.1f} ms")

    # ── Logit divergence ──────────────────────────────────────────────────────
    logit_diff = (bc_logits - id_logits).abs().max().item()
    print(
        f"\nMax |Δlogits| (BCNorm vs Identity) = {logit_diff:.4f}  "
        "(non-zero is expected — this is the architectural difference, not noise)"
    )

    # ── Side-by-side tables ───────────────────────────────────────────────────
    print_side_by_side(
        bc_probe.records, id_probe.records,
        bc_cos, id_cos,
        n_layers=args.n_layer,
    )

    # ── Headline numbers ──────────────────────────────────────────────────────
    bc = _flatten_records(bc_probe.records)
    idn = _flatten_records(id_probe.records)
    std_ratios = [
        idn[(i, t)]["post_std"] / max(bc[(i, t)]["post_std"], 1e-12)
        for i in range(args.n_layer) for t in ("B", "C")
    ]
    max_ratios = [
        idn[(i, t)]["post_max_abs"] / max(bc[(i, t)]["post_max_abs"], 1e-12)
        for i in range(args.n_layer) for t in ("B", "C")
    ]
    # std_ratio = Id/BC < 1 means BCNorm-post is LARGER than the raw projection,
    # i.e. BCNorm UPSCALED. Report the inverse so the direction is unambiguous.
    bc_over_id_std = [1.0 / r for r in std_ratios]
    bc_over_id_max = [1.0 / r for r in max_ratios]
    direction_std = "upscales" if sum(bc_over_id_std) / len(bc_over_id_std) > 1.0 else "downscales"
    print(
        f"\nHeadline: BCNorm {direction_std} B/C std by ×"
        f"{sum(bc_over_id_std)/len(bc_over_id_std):.2f} on average "
        f"(range ×{min(bc_over_id_std):.2f}–×{max(bc_over_id_std):.2f}); "
        f"max-abs by ×{sum(bc_over_id_max)/len(bc_over_id_max):.2f} "
        f"(range ×{min(bc_over_id_max):.2f}–×{max(bc_over_id_max):.2f})."
    )
    print(
        "  Interpretation: at random init the raw projections are SMALL "
        f"(std≈{1/(sum(bc_over_id_std)/len(bc_over_id_std)):.2f}); BCNorm normalizes them UP\n"
        "  to unit RMS. The 'tame heavy outliers' story applies at training time, not init —\n"
        "  re-run this script on a trained checkpoint to surface that regime."
    )

    cos_diffs = [abs(bc_cos[i] - id_cos[i]) for i in range(args.n_layer)]
    # Theory: RMSNorm@init with weight=1 is a per-token positive scalar; cosine
    # is exactly invariant. Observed Δcos comes from different fp32 reduction
    # orders in the two paths (norm → cos vs raw → cos), not from real geometry
    # change. NUM_NOISE_BAND (module-level) is a generous numerical-noise band;
    # anything beyond it warrants investigation.
    cos_invariant = max(cos_diffs) < NUM_NOISE_BAND
    print(
        f"\nCosine invariance under RMSNorm@init: "
        f"{'CONFIRMED (within fp32 noise)' if cos_invariant else 'VIOLATED'} "
        f"(max |Δcos| = {max(cos_diffs):.2e}, noise band = {NUM_NOISE_BAND:.0e})."
    )
    print(
        "  Theory: RMSNorm with weight=1 applies a uniform positive scalar per token,\n"
        "  which preserves cosine exactly. Observed drift is fp32 reduction-order noise.\n"
        "  Geometric divergence between BCNorm and Identity should appear only when\n"
        "  the per-channel norm weights train away from ones."
    )

    # ── Persist a self-describing run directory ───────────────────────────────
    base = _REPO_ROOT / "experiments" / "pilot_logs"
    run_dir = make_run_dir(EXPERIMENT, base=base, git=git)
    headline = {
        # BCNorm-post / Identity-post. >1 = BCNorm upscales, <1 = downscales.
        "mean_std_bc_over_id": sum(bc_over_id_std) / len(bc_over_id_std),
        "min_std_bc_over_id": min(bc_over_id_std),
        "max_std_bc_over_id": max(bc_over_id_std),
        "mean_max_abs_bc_over_id": sum(bc_over_id_max) / len(bc_over_id_max),
        "cosine_invariant_at_init": cos_invariant,
        "max_abs_cosine_delta": max(cos_diffs),
        "cosine_noise_band": NUM_NOISE_BAND,
    }
    stats = {
        "device": str(device),
        "input": {"batch": batch, "seqlen": seqlen, "dtype": str(input_ids.dtype)},
        "forward_ms": {"bcnorm": round(bc_ms, 2), "identity": round(id_ms, 2)},
        "logit_max_abs_diff": logit_diff,
        "bcnorm_stats": bc_probe.records,
        "identity_stats": id_probe.records,
        "bcnorm_cosine_per_layer": {str(k): v for k, v in bc_cos.items()},
        "identity_cosine_per_layer": {str(k): v for k, v in id_cos.items()},
        "headline": headline,
    }
    (run_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")

    # Save the exact input tensor for byte-replay (CPU tensor, ~1KB).
    torch.save(input_ids.detach().cpu(), run_dir / "inputs.pt")

    write_manifest(
        run_dir,
        experiment=EXPERIMENT,
        description=DESCRIPTION,
        config=args,                 # Mamba3Config dataclass
        seed=SEED,
        determinism=determinism,
        git=git,
        env=env,
        extra={
            "toy_model_kwargs": TOY_KW,
            "input_shape": [batch, seqlen],
            "input_seed": SEED + 1,
            "num_noise_band": NUM_NOISE_BAND,
            "param_count": sum(p.numel() for p in bc_model.parameters()),
            "bcnorm_modules_replaced": n_replaced,
        },
        script_path=Path(__file__),
    )

    write_summary(run_dir, [
        f"# {EXPERIMENT} — {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"- Device: `{device}`  ·  torch `{env['torch']}`  ·  "
        f"git `{git['commit_short']}`{' (dirty)' if git['dirty'] else ''}",
        f"- Seed: `{SEED}` (model), `{SEED+1}` (inputs)  ·  "
        f"input shape `({batch}, {seqlen})`",
        f"- Model: {TOY_KW}, params = {sum(p.numel() for p in bc_model.parameters()):,}",
        "",
        "## Headline",
        "",
        f"- BCNorm {direction_std} B/C std by **×"
        f"{headline['mean_std_bc_over_id']:.2f}** "
        f"(range ×{headline['min_std_bc_over_id']:.2f}–"
        f"×{headline['max_std_bc_over_id']:.2f}).",
        f"- BCNorm rescales B/C max-abs by ×"
        f"{headline['mean_max_abs_bc_over_id']:.2f}.",
        f"- Cosine invariance under RMSNorm@init: "
        f"**{'CONFIRMED' if cos_invariant else 'VIOLATED'}** "
        f"(max |Δcos| = {headline['max_abs_cosine_delta']:.2e}, "
        f"noise band = {NUM_NOISE_BAND:.0e}).",
        f"- Forward time: BCNorm {bc_ms:.1f} ms, Identity {id_ms:.1f} ms "
        f"(first-call CUDA warm-up; not a benchmark).",
        f"- Max |Δlogits| (BCNorm vs Identity) = {logit_diff:.4f}.",
        "",
        "## Files",
        "",
        "- `manifest.json` — env, git, seed, determinism, command",
        "- `config.json` — full Mamba3Config",
        "- `stats.json` — per-layer probe records and cosine values",
        "- `inputs.pt` — exact `input_ids` tensor for byte replay",
        "- `script.py` — frozen copy of the runner",
        "- `env.txt` — `pip freeze` of the active interpreter",
    ])

    rel = run_dir.relative_to(_REPO_ROOT)
    print(f"\nRun artifacts → {rel}")
    print(f"  · manifest.json   ({(run_dir / 'manifest.json').stat().st_size} B)")
    print(f"  · config.json     ({(run_dir / 'config.json').stat().st_size} B)")
    print(f"  · stats.json      ({(run_dir / 'stats.json').stat().st_size} B)")
    print(f"  · inputs.pt       ({(run_dir / 'inputs.pt').stat().st_size} B)")
    print(f"  · script.py + env.txt + summary.md")


if __name__ == "__main__":
    main()
