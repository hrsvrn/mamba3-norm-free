"""JSONL logging for training runs.

Exposes a lightweight ``TrainLogger`` that appends one line of JSON per step,
making step‑level data queryable without loading a pickle.  Designed to work
with the ``manifest.py`` run‑directory layout so every training log inherits
the same reproducibility metadata (git SHA, environment, determinism info).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StepLog:
    """One training step captured as a single JSONL line."""

    step: int
    loss: float
    grad_norm: float | None = None
    lr: float | None = None
    tokens: int = 0
    wall_time: float = 0.0  # seconds since epoch / script start

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        for fld in self.__dataclass_fields__:
            d[fld] = getattr(self, fld)
        return d


def _json_line(obj: dict[str, Any]) -> str:
    return json.dumps(obj, separators=(",", ":"))


class TrainLogger:
    """Append‑only JSONL log + optional checkpoint manifest.

    Usage::

        logger = TrainLogger(run_dir)
        for batch in loader:
            ...
            logger.log_step(StepLog(step=n, loss=loss.item(), ...))
            if n % ckpt_interval == 0:
                logger.log_checkpoint(n, f"step_{n:06d}.pt")
    """

    def __init__(self, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        self._step_path = run_dir / "train_log.jsonl"
        self._ckpt_path = run_dir / "checkpoints.json"
        self._file = self._step_path.open("a")
        self._step_count = 0

    def log_step(self, entry: StepLog) -> None:
        self._file.write(_json_line(entry.to_dict()) + "\n")
        self._step_count += 1

    def log_checkpoint(self, step: int, path: str) -> None:
        """Record a checkpoint for later retrieval."""
        record: dict[str, Any] = {"step": step, "path": path}
        entries: list[dict[str, Any]] = []
        if self._ckpt_path.exists():
            entries = json.loads(self._ckpt_path.read_text())
        entries.append(record)
        self._ckpt_path.write_text(json.dumps(entries, indent=2) + "\n")

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> TrainLogger:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    @property
    def step_count(self) -> int:
        return self._step_count


# ── convenience helpers ───────────────────────────────────────────────────────

@dataclass
class SmokeReport:
    """Summary returned by ``train_smoke.py`` after a short training run."""

    stabilizer: str
    squash_before_bias: bool
    stabilize_b: bool
    stabilize_c: bool
    final_loss: float
    loss_decreased: bool
    grads_finite: bool
    seed: int
    steps: int = 0
    failures: list[str] = field(default_factory=list)

    def ok(self) -> bool:
        return (
            math.isfinite(self.final_loss)
            and self.grads_finite
            and not self.failures
        )

    def summary_lines(self) -> list[str]:
        lines = [
            f"# Smoke report — {self.stabilizer}",
            f"- stabilizer:      {self.stabilizer}",
            f"- bias‑first:      {self.squash_before_bias}",
            f"- B replaced:      {self.stabilize_b}",
            f"- C replaced:      {self.stabilize_c}",
            f"- seed:            {self.seed}",
            f"- steps:           {self.steps}",
            f"- final loss:      {self.final_loss:.4f}",
            f"- loss decreased:  {self.loss_decreased}",
            f"- gradients ok:    {self.grads_finite}",
        ]
        if self.failures:
            lines.append("- failures:")
            for f in self.failures:
                lines.append(f"  * {f}")
        return lines
