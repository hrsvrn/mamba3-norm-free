#!/usr/bin/env python3
"""Micro-pretraining on Wikitext‑2 — 2000‑step bake‑off for BC stabilisers.

Each variant is trained for 2000 steps on packed Wikitext‑2 sequences with
GPT‑2 tokenization, repeated for 2 seeds.  Step‑level logs go to JSONL;
every run gets a self‑describing manifest (git SHA, env, determinism).

Usage::

    python scripts/train_micro.py --stabilizer dyisru
    python scripts/train_micro.py --stabilizer dypower_p1
    python scripts/train_micro.py --all  # 7 variants × 2 seeds = 14 runs
    python scripts/train_micro.py --all --squash-before-bias

The model is ~11M parameters — small enough to run on a single GPU in
a few minutes per seed but large enough to surface qualitative differences
between stabilisers (better than toy random‑token smoke).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from time import time
from typing import Iterator

from pathlib import Path

import numpy as np
import torch
from torch import nn, Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "mamba3-minimal"))

from mamba3 import Mamba3Config, Mamba3LMHeadModel, get_device  # noqa: E402
from nfmamba import install_bc_stabilizer  # noqa: E402
from nfmamba.utils.manifest import (  # noqa: E402
    gather_env,
    gather_git_state,
    lock_determinism,
    write_manifest,
    write_summary,
)
from nfmamba.utils.train_logger import SmokeReport, StepLog, TrainLogger  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

SEQUENCE_LENGTH = 256
BATCH_SIZE = 2
MICRO_STEPS = 2000
LOG_INTERVAL = 50
SEEDS = (42, 123)

# ──────────────────────────────────────────────────────────────────────────────
# Micro model (~7.7M params)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MicroModelConfig:
    d_model: int = 128
    n_layer: int = 4
    d_state: int = 64
    headdim: int = 32
    chunk_size: int = 32
    expand: int = 2
    vocab_size: int = 50257  # GPT‑2
    pad_vocab_size_multiple: int = 16
    lr: float = 3e-3
    weight_decay: float = 0.01


MICRO_CONFIG = MicroModelConfig()


# ──────────────────────────────────────────────────────────────────────────────
# Wikitext‑2 data (streaming, single‑pass)
# ──────────────────────────────────────────────────────────────────────────────

class WikitextDataset(IterableDataset):
    """Yield (input_ids, labels) pairs from Wikitext‑2.

    The underlying token buffer is built once and pinned in memory.
    Shuffling is NOT enabled by default (deterministic replay), but a
    ``shuffle_buffer`` can be set for training runs.
    """

    def __init__(
        self,
        seqlen: int = SEQUENCE_LENGTH,
        max_tokens: int | None = None,
        seed: int = 42,
    ):
        super().__init__()
        self.seqlen = seqlen
        self.max_tokens = max_tokens
        self.seed = seed
        self._buffer: np.ndarray | None = None

    def _build(self) -> np.ndarray:
        tok = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        # Load raw documents (lazy streaming)
        from datasets import load_dataset  # deferred import
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

        token_lists: list[list[int]] = []
        total = 0
        limit = self.max_tokens
        for row in ds:
            text = row["text"]
            if not (text and text.strip()):
                continue
            ids = tok.encode(text, add_special_tokens=False)
            ids.append(tok.eos_token_id)
            token_lists.append(ids)
            total += len(ids)
            if limit is not None and total >= limit:
                break

        # Pack into sequences
        if not token_lists:
            raise RuntimeError("Wikitext‑2 returned no documents")
        flat = np.concatenate([np.asarray(tl, dtype=np.int64) for tl in token_lists])
        usable = (flat.shape[0] - 1) // self.seqlen
        keep = usable * self.seqlen + 1
        return flat[:keep]

    @property
    def buffer(self) -> np.ndarray:
        if self._buffer is None:
            self._buffer = self._build()
        return self._buffer

    def num_sequences(self) -> int:
        return (self.buffer.shape[0] - 1) // self.seqlen

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        buf = self.buffer
        n = self.num_sequences()
        indices = list(range(n))
        rng = np.random.RandomState(self.seed)
        rng.shuffle(indices)
        for idx in indices:
            start = idx * self.seqlen
            end = start + self.seqlen + 1  # +1 for labels
            chunk = buf[start:end]
            inputs = torch.as_tensor(chunk[:-1], dtype=torch.long)
            labels = torch.as_tensor(chunk[1:], dtype=torch.long)
            yield inputs, labels


def make_loader(dataset: WikitextDataset, batch_size: int = BATCH_SIZE) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Model helpers
# ──────────────────────────────────────────────────────────────────────────────

def create_model(device: torch.device, cfg: MicroModelConfig) -> Mamba3LMHeadModel:
    args = Mamba3Config(
        d_model=cfg.d_model,
        n_layer=cfg.n_layer,
        d_state=cfg.d_state,
        headdim=cfg.headdim,
        chunk_size=cfg.chunk_size,
        vocab_size=cfg.vocab_size,
        pad_vocab_size_multiple=cfg.pad_vocab_size_multiple,
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


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _compute_grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.float().norm(2).item() ** 2
    return total ** 0.5


def train(
    *,
    model: Mamba3LMHeadModel,
    loader: DataLoader,
    optim: AdamW,
    steps: int,
    logger: TrainLogger,
) -> tuple[float, bool, bool]:
    grad_ok = True
    final_loss = float("nan")
    initial_loss = None
    running_loss = 0.0
    running_steps = 0

    model.train()
    it = iter(loader)
    t0 = time()

    for step in range(steps):
        try:
            inputs, labels = next(it)
        except StopIteration:
            it = iter(loader)
            inputs, labels = next(it)

        inputs = inputs.to(device=model.device)
        labels = labels.to(device=model.device)

        logits, _ = model(inputs)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
        )
        final_loss = loss.item()
        if initial_loss is None:
            initial_loss = final_loss
        running_loss += final_loss
        running_steps += 1

        optim.zero_grad()
        loss.backward()

        grad_norm = _compute_grad_norm(model)
        if not torch.isfinite(torch.tensor(grad_norm)):
            grad_ok = False
            break

        optim.step()

        # ── real-time streaming to console ──
        elapsed = time() - t0
        avg_loss = running_loss / running_steps
        tokens_per_sec = (step + 1) * BATCH_SIZE * SEQUENCE_LENGTH / elapsed if elapsed > 0 else 0
        sys.stdout.write(
            f"\r  step {step+1:>4d}/{steps} | "
            f"loss {final_loss:.4f} (avg {avg_loss:.4f}) | "
            f"grad {grad_norm:.1f} | "
            f"lr {optim.param_groups[0]['lr']:.2e} | "
            f"{elapsed:>5.1f}s | {tokens_per_sec:>6.0f} tok/s"
        )
        sys.stdout.flush()

        if step % LOG_INTERVAL == 0 or step == steps - 1:
            print()  # anchor logged steps with a newline
            logger.log_step(StepLog(
                step=step,
                loss=final_loss,
                grad_norm=grad_norm,
                lr=optim.param_groups[0]["lr"],
                tokens=(step + 1) * BATCH_SIZE * SEQUENCE_LENGTH,
            ))

    print()  # final newline after last live update
    loss_decreased = (
        initial_loss is not None
        and final_loss < initial_loss
    )
    return final_loss, loss_decreased, grad_ok


# ──────────────────────────────────────────────────────────────────────────────
# Single run
# ──────────────────────────────────────────────────────────────────────────────

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
    model_cfg: MicroModelConfig | None = None,
) -> SmokeReport:
    if device is None:
        device = get_device()
    if model_cfg is None:
        model_cfg = MICRO_CONFIG
    if stabilizer_kwargs is None:
        stabilizer_kwargs = {}

    det = lock_determinism(seed)
    git = gather_git_state(ROOT)
    env = gather_env()
    torch.manual_seed(seed)

    # ── data ──
    dataset = WikitextDataset(seqlen=SEQUENCE_LENGTH, seed=seed)
    loader = make_loader(dataset)

    # ── model ──
    model = create_model(device, model_cfg)

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
    optim = AdamW(params, lr=model_cfg.lr, weight_decay=model_cfg.weight_decay)

    # ── logger ──
    if run_dir is None:
        run_dir = Path("/tmp/mamba3_micro") / f"{stabilizer}_{seed}"
    logger = TrainLogger(run_dir)

    try:
        final_loss, loss_decreased, grad_ok = train(
            model=model,
            loader=loader,
            optim=optim,
            steps=MICRO_STEPS,
            logger=logger,
        )
    except Exception as exc:
        failures.append(str(exc))
        final_loss = float("nan")
        loss_decreased = False
        grad_ok = False
    finally:
        logger.close()

    # ── manifest ──
    write_manifest(
        run_dir,
        experiment=f"micro_{stabilizer}",
        description=(
            f"Micro‑pretraining: stabilizer={stabilizer}, "
            f"seed={seed}, squash_before_bias={squash_before_bias}, "
            f"steps={MICRO_STEPS}, seqlen={SEQUENCE_LENGTH}"
        ),
        config={
            "model": asdict(model_cfg),
            "stabilizer": stabilizer,
            "stabilizer_kwargs": stabilizer_kwargs,
            "squash_before_bias": squash_before_bias,
            "stabilize_b": stabilize_b,
            "stabilize_c": stabilize_c,
            "steps": MICRO_STEPS,
            "seqlen": SEQUENCE_LENGTH,
            "batch_size": BATCH_SIZE,
            "num_sequences": dataset.num_sequences(),
            "param_count": param_count(model),
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

    write_summary(run_dir, [
        f"# Micro‑pretraining — {stabilizer} (seed={seed})",
        f"- param count:     {param_count(model):,}",
        f"- sequences:       {dataset.num_sequences()}",
        f"- steps:           {MICRO_STEPS}",
        f"- final loss:      {final_loss:.4f}",
        f"- loss decreased:  {loss_decreased}",
        f"- gradients ok:    {grad_ok}",
        f"- bias‑first:      {squash_before_bias}",
        f"- failures:        {failures or 'none'}",
    ])

    return SmokeReport(
        stabilizer=stabilizer,
        squash_before_bias=squash_before_bias,
        stabilize_b=stabilize_b,
        stabilize_c=stabilize_c,
        final_loss=final_loss,
        loss_decreased=loss_decreased,
        grads_finite=grad_ok,
        seed=seed,
        steps=MICRO_STEPS,
        failures=failures,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Known variants
# ──────────────────────────────────────────────────────────────────────────────

STAB_VARIANTS: list[dict] = [
    {"name": "identity"},
    {"name": "bcnorm"},
    {"name": "dyt"},
    {"name": "derf"},
    {"name": "dyisru"},
    {"name": "dysoftsign"},
    {"name": "dypower_p1"},
]


def _label(d: dict) -> str:
    name = d["name"]
    kw = d.get("kwargs", {})
    if kw:
        return f"{name}({','.join(f'{k}={v}' for k, v in kw.items())})"
    return name


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Micro‑pretraining bake‑off on Wikitext‑2"
    )
    p.add_argument("--stabilizer", "-s", default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--squash-before-bias", action="store_true")
    p.add_argument("--seeds", action="store_true",
                   help=f"Run 2 seeds (default: single seed)")
    p.add_argument("--stabilize-b", dest="stab_b", action="store_true", default=True)
    p.add_argument("--no-stabilize-b", dest="stab_b", action="store_false")
    p.add_argument("--stabilize-c", dest="stab_c", action="store_true", default=True)
    p.add_argument("--no-stabilize-c", dest="stab_c", action="store_false")
    p.add_argument("--output", default="/tmp/mamba3_micro",
                   help="Base output directory")
    args = p.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"Model:  ~{param_count(create_model(device, MICRO_CONFIG)):,} params")
    print()

    seeds = SEEDS if args.seeds else [SEEDS[0]]

    results: list[tuple[str, SmokeReport]] = []

    if args.all:
        for stab_entry in STAB_VARIANTS:
            for seed in seeds:
                label = _label(stab_entry)
                kwargs = stab_entry.get("kwargs", {})
                run_dir = Path(args.output) / f"{stab_entry['name']}_{seed}"
                result = run_one(
                    stabilizer=stab_entry["name"],
                    stabilizer_kwargs=kwargs,
                    squash_before_bias=args.squash_before_bias,
                    stabilize_b=args.stab_b,
                    stabilize_c=args.stab_c,
                    seed=seed,
                    device=device,
                    run_dir=run_dir,
                )
                results.append((f"{label} (s={seed})", result))
    elif args.stabilizer:
        kwargs: dict = {}
        label = args.stabilizer
        for seed in seeds:
            run_dir = Path(args.output) / f"{args.stabilizer}_{seed}"
            result = run_one(
                stabilizer=args.stabilizer,
                stabilizer_kwargs=kwargs,
                squash_before_bias=args.squash_before_bias,
                stabilize_b=args.stab_b,
                stabilize_c=args.stab_c,
                seed=seed,
                device=device,
                run_dir=run_dir,
            )
            results.append((f"{label} (s={seed})", result))
    else:
        print("Need --stabilizer NAME or --all")
        sys.exit(1)

    # Summary table
    passed = 0
    col_w = max(len(lbl) for lbl, _ in results)
    print(f"\n{'variant':<{col_w}s}  {'seed':>4s}  {'loss':>8s}  {'status':>4s}")
    print("-" * (col_w + 4 + 8 + 6))
    for label, r in results:
        status = "OK" if r.ok() else "FAIL"
        if r.ok():
            passed += 1
        print(f"{label:<{col_w}s}  {r.seed:>4d}  {r.final_loss:>8.4f}  {status:>4s}")
    print(f"\n{passed}/{len(results)} passed")

    # Write aggregate summary
    out = Path(args.output) / "micro_summary.md"
    lines = [
        "# Micro‑pretraining Summary",
        "",
        f"Model: ~{param_count(create_model(device, MICRO_CONFIG)):,} params",
        f"Steps per run: {MICRO_STEPS}",
        f"Seeds: {SEEDS}",
        "",
        f"| variant | seed | loss | OK |",
        f"|---|---|---|---|",
    ]
    for label, r in results:
        lines.append(f"| {label} | {r.seed} | {r.final_loss:.4f} | {r.ok()} |")
    lines.append("")
    lines.append(f"**{passed}/{len(results)} passed**")
    out.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
