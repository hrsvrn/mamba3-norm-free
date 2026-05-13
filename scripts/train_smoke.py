#!/usr/bin/env python3
"""Smoke‑test each B/C stabiliser variant with a tiny LM training loop.

200 steps on random data — just enough to confirm that forward, backward,
and parameter updates complete without NaN / divergence.  Every run is
tracked by the manifest + JSONL logging infrastructure.

Usage::

    python scripts/train_smoke.py --stabilizer dyisru
    python scripts/train_smoke.py --stabilizer dypower --p 1.0
    python scripts/train_smoke.py --stabilizer derf --squash-before-bias
    python scripts/train_smoke.py --all  # smoke every registered variant
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "mamba3-minimal"))

from mamba3 import Mamba3Config, Mamba3LMHeadModel, get_device  # noqa: E402
from nfmamba import install_bc_stabilizer  # noqa: E402
from nfmamba.utils.manifest import (  # noqa: E402
    gather_env,
    gather_git_state,
    lock_determinism,
    make_run_dir,
    write_manifest,
)
from nfmamba.utils.train_logger import SmokeReport, StepLog, TrainLogger  # noqa: E402


# ── default smoke config ──────────────────────────────────────────────────────

SMOKE_CONFIG = {
    "d_model": 128,
    "n_layer": 4,
    "d_state": 64,
    "headdim": 32,
    "chunk_size": 32,
    "vocab_size": 512,
    "seqlen": 128,
    "batch_size": 2,
    "steps": 200,
    "lr": 1e-3,
    "weight_decay": 0.01,
}


# ── known stabilizers (kept in sync with nfmamba/modules/registry.py) ─────────

ALL_STABILIZERS: list[dict] = [
    {"name": "identity"},
    {"name": "bcnorm"},
    {"name": "dyt"},
    {"name": "derf"},
    {"name": "dyisru"},
    {"name": "dypower", "kwargs": {"p": 1.0}},
    {"name": "dypower", "kwargs": {"p": 2.0}},
    {"name": "dypower", "kwargs": {"p": 3.0}},
]


# ── helpers ────────────────────────────────────────────────────────────────────

def _create_model(device: torch.device) -> Mamba3LMHeadModel:
    args = Mamba3Config(
        d_model=SMOKE_CONFIG["d_model"],
        n_layer=SMOKE_CONFIG["n_layer"],
        d_state=SMOKE_CONFIG["d_state"],
        headdim=SMOKE_CONFIG["headdim"],
        chunk_size=SMOKE_CONFIG["chunk_size"],
        vocab_size=SMOKE_CONFIG["vocab_size"],
        use_mimo=False,
    )
    model = Mamba3LMHeadModel(args, device=device)
    for name, p in model.named_parameters():
        if "A_log" in name:
            nn.init.uniform_(p, -4, -1)
        elif "D" in name and p.dim() == 1:
            nn.init.ones_(p)
        elif "dt_bias" in name:
            nn.init.uniform_(p, 0.001, 0.1)
        elif "B_bias" in name or "C_bias" in name:
            pass  # already ones
        elif "mimo" in name:
            pass
        elif p.dim() >= 2:
            nn.init.normal_(p, std=0.02)
    return model


def _train_loop(
    model: Mamba3LMHeadModel,
    optim: AdamW,
    steps: int,
    device: torch.device,
    logger: TrainLogger,
) -> tuple[float, bool, bool]:
    seqlen = SMOKE_CONFIG["seqlen"]
    batch_size = SMOKE_CONFIG["batch_size"]
    vocab = SMOKE_CONFIG["vocab_size"]
    grad_ok = True
    final_loss = float("nan")
    initial_loss = None

    for step in range(steps):
        tokens = torch.randint(0, vocab, (batch_size, seqlen), device=device)
        logits, _ = model(tokens)
        # Labels are shifted right (next-token prediction)
        loss = nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            tokens[:, 1:].reshape(-1),
        )
        final_loss = loss.item()
        if initial_loss is None:
            initial_loss = final_loss

        optim.zero_grad()
        loss.backward()

        grad_norm = _compute_grad_norm(model)
        if not torch.isfinite(torch.tensor(grad_norm)):
            grad_ok = False

        optim.step()

        logger.log_step(
            StepLog(
                step=step,
                loss=final_loss,
                grad_norm=grad_norm,
                lr=optim.param_groups[0]["lr"],
                tokens=(step + 1) * batch_size * seqlen,
            )
        )

    loss_decreased = (
        initial_loss is not None
        and final_loss < initial_loss
    )
    return final_loss, loss_decreased, grad_ok


def _compute_grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().float().norm(2).item() ** 2
    return total ** 0.5


# ── single run ─────────────────────────────────────────────────────────────────

def run_one(
    *,
    stabilizer: str,
    seed: int = 42,
    squash_before_bias: bool = False,
    stabilize_b: bool = True,
    stabilize_c: bool = True,
    stabilizer_kwargs: dict | None = None,
    device: torch.device | None = None,
    run_dir: Path | None = None,
) -> SmokeReport:
    if device is None:
        device = get_device()
    if stabilizer_kwargs is None:
        stabilizer_kwargs = {}

    # ── reproducibility ──
    det = lock_determinism(seed)
    git = gather_git_state(ROOT)
    env = gather_env()

    # ── create model ──
    torch.manual_seed(seed)
    model = _create_model(device)
    model.train()

    # ── install stabilizer ──
    report = install_bc_stabilizer(
        model,
        stabilizer,
        stabilize_b=stabilize_b,
        stabilize_c=stabilize_c,
        squash_before_bias=squash_before_bias,
    )

    failures: list[str] = []

    # ── optimiser ──
    params = [p for p in model.parameters() if p.requires_grad]
    optim = AdamW(
        params,
        lr=SMOKE_CONFIG["lr"],
        weight_decay=SMOKE_CONFIG["weight_decay"],
    )

    # ── log ──
    if run_dir is None:
        run_dir = Path("/tmp/mamba3_smoke") / f"{stabilizer}_{seed}"
    logger = TrainLogger(run_dir)

    try:
        final_loss, loss_decreased, grad_ok = _train_loop(
            model, optim, SMOKE_CONFIG["steps"], device, logger
        )
    except Exception as exc:
        failures.append(str(exc))
        final_loss = float("nan")
        loss_decreased = False
        grad_ok = False
    finally:
        logger.close()

    # ── manifest (post-hoc — we still want env/git even on failure) ──
    write_manifest(
        run_dir,
        experiment=f"smoke_{stabilizer}",
        description=f"Smoke test for stabilizer={stabilizer}, "
        f"squash_before_bias={squash_before_bias}",
        config={
            **SMOKE_CONFIG,
            "stabilizer": stabilizer,
            "squash_before_bias": squash_before_bias,
            "stabilize_b": stabilize_b,
            "stabilize_c": stabilize_c,
            "install_report": {
                "name": report.name,
                "replaced": report.replaced,
                "stabilize_b": report.stabilize_b,
                "stabilize_c": report.stabilize_c,
                "squash_before_bias": report.squash_before_bias,
            },
        },
        seed=seed,
        determinism=det,
        git=git,
        env=env,
        script_path=Path(__file__),
    )

    return SmokeReport(
        stabilizer=stabilizer,
        squash_before_bias=squash_before_bias,
        stabilize_b=stabilize_b,
        stabilize_c=stabilize_c,
        final_loss=final_loss,
        loss_decreased=loss_decreased,
        grads_finite=grad_ok,
        seed=seed,
        steps=SMOKE_CONFIG["steps"],
        failures=failures,
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

def _label(d: dict) -> str:
    name = d["name"]
    kw = d.get("kwargs", {})
    if kw:
        return f"{name}({','.join(f'{k}={v}' for k, v in kw.items())})"
    return name


def main() -> None:
    p = argparse.ArgumentParser(description="Smoke-test BC stabilizer variants")
    p.add_argument("--stabilizer", "-s", default=None, help="Single stabilizer name")
    p.add_argument("--p", type=float, default=None, help="p value for DyPowerSign")
    p.add_argument("--all", action="store_true", help="Run all known stabilizers")
    p.add_argument("--squash-before-bias", action="store_true")
    p.add_argument("--stabilize-b", dest="stabilize_b", action="store_true", default=True)
    p.add_argument("--no-stabilize-b", dest="stabilize_b", action="store_false")
    p.add_argument("--stabilize-c", dest="stabilize_c", action="store_true", default=True)
    p.add_argument("--no-stabilize-c", dest="stabilize_c", action="store_false")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = get_device()

    if args.all:
        passed = 0
        for stab_entry in ALL_STABILIZERS:
            label = _label(stab_entry)
            kwargs = stab_entry.get("kwargs", {})
            result = run_one(
                stabilizer=stab_entry["name"],
                stabilizer_kwargs=kwargs,
                seed=args.seed,
                device=device,
            )
            status = "OK" if result.ok() else "FAIL"
            print(f"  {label:30s}  loss={result.final_loss:.4f}  {status}")
            if result.ok():
                passed += 1
        print(f"\n{passed}/{len(ALL_STABILIZERS)} passed")
    elif args.stabilizer is None:
        print("Need --stabilizer NAME or --all")
        sys.exit(1)
    else:
        kwargs = {}
        if args.p is not None:
            kwargs["p"] = args.p
        result = run_one(
            stabilizer=args.stabilizer,
            stabilizer_kwargs=kwargs,
            squash_before_bias=args.squash_before_bias,
            stabilize_b=args.stabilize_b,
            stabilize_c=args.stabilize_c,
            seed=args.seed,
            device=device,
        )
        status = "OK" if result.ok() else "FAIL"
        print(f"  {args.stabilizer:30s}  loss={result.final_loss:.4f}  {status}")
        for line in result.summary_lines():
            print(line)
        if not result.ok():
            sys.exit(1)


if __name__ == "__main__":
    main()
