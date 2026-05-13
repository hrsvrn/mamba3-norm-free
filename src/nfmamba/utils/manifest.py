"""
manifest.py — reproducibility plumbing for diagnostic runs.

Every diagnostic script in `nfmamba.diagnostics` is expected to use these helpers
to produce a self-describing, re-runnable artifact directory:

    experiments/pilot_logs/<experiment>/<timestamp>__<short_sha>[__dirty]/
        manifest.json     # env, git, command, determinism, seeds, config
        config.json       # full Mamba3Config (and any script-level overrides)
        env.txt           # `pip freeze` style snapshot for the active interp.
        inputs.pt         # exact input tensors (when small) for byte replay
        stats.json        # the probe outputs proper
        summary.md        # human-readable headline numbers
        script.py         # frozen copy of the calling script source

The manifest answers, for any future reader: *what was run, on what hardware,
against which commit, with what seed, and where can I re-run it?*

This is week-1 plumbing — when 180M ablations come online the same helper will
emit the same artifacts so trainer logs, eval logs, and diagnostic logs all
share one schema. (See skills.md § 4: failures are first-class deliverables.)
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch


# ──────────────────────────────────────────────────────────────────────────────
# Environment / git capture
# ──────────────────────────────────────────────────────────────────────────────

def _run(cmd: list[str], cwd: Path | None = None) -> str:
    try:
        out = subprocess.check_output(
            cmd, cwd=cwd, stderr=subprocess.DEVNULL, text=True
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def gather_git_state(repo_root: Path) -> dict[str, Any]:
    """Capture commit, branch, dirty flag, and a short-form summary."""
    sha = _run(["git", "rev-parse", "HEAD"], cwd=repo_root)
    short = sha[:8] if sha else "unknown"
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)
    status = _run(["git", "status", "--porcelain"], cwd=repo_root)
    dirty = bool(status)
    # Limit the diffstat we record to keep manifests compact but informative.
    diffstat = _run(["git", "diff", "--stat", "HEAD"], cwd=repo_root)
    return {
        "commit": sha or None,
        "commit_short": short,
        "branch": branch or None,
        "dirty": dirty,
        "porcelain_status": status,
        "diffstat": diffstat,
    }


def gather_env() -> dict[str, Any]:
    """Capture interpreter, torch, CUDA, hardware, and host identity."""
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "host": socket.gethostname(),
        "torch": torch.__version__,
        "torch_cuda_built": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
    }
    if torch.cuda.is_available():
        env["cuda_runtime"] = torch.version.cuda
        env["cudnn"] = torch.backends.cudnn.version()
        env["device_name"] = torch.cuda.get_device_name(0)
        env["device_capability"] = ".".join(
            str(x) for x in torch.cuda.get_device_capability(0)
        )
        env["device_count"] = torch.cuda.device_count()
    # Driver version (best effort).
    nvsmi = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    if nvsmi:
        env["nvidia_driver"] = nvsmi.splitlines()[0].strip()
    return env


def freeze_env(out_path: Path) -> None:
    """Write a `pip freeze`-style snapshot to env.txt. Best-effort.

    Tries (in order):
      1. `python -m pip freeze` — works in stdlib pip environments.
      2. `uv pip freeze` — works in uv-managed venvs (no embedded pip).
      3. `importlib.metadata` fallback — pure-Python, always works.
    """
    out = _run([sys.executable, "-m", "pip", "freeze"])
    if not out:
        out = _run(["uv", "pip", "freeze"])
    if not out:
        try:
            from importlib.metadata import distributions
            out = "\n".join(
                sorted(
                    f"{d.metadata['Name']}=={d.version}"
                    for d in distributions()
                    if d.metadata.get("Name")
                )
            )
        except Exception:  # noqa: BLE001
            out = "# could not enumerate installed packages"
    out_path.write_text(out + ("\n" if out and not out.endswith("\n") else ""))


# ──────────────────────────────────────────────────────────────────────────────
# Determinism
# ──────────────────────────────────────────────────────────────────────────────

def lock_determinism(seed: int) -> dict[str, Any]:
    """Install a reproducible RNG state and return the determinism manifest.

    Caveats:
      * `torch.use_deterministic_algorithms(True)` only takes effect when the
        operations downstream support it; we record success/failure.
      * `CUBLAS_WORKSPACE_CONFIG` must be set before any CUDA op runs to fully
        determinize cuBLAS GEMMs. We set it here; if the process has already
        used CUDA the env var is informational only.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    record: dict[str, Any] = {
        "seed": seed,
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        record["use_deterministic_algorithms"] = True
    except Exception as exc:  # noqa: BLE001
        record["use_deterministic_algorithms"] = f"failed: {exc}"

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        record["cudnn_deterministic"] = True
        record["cudnn_benchmark"] = False
    return record


# ──────────────────────────────────────────────────────────────────────────────
# Run-directory layout
# ──────────────────────────────────────────────────────────────────────────────

def make_run_dir(
    experiment: str,
    *,
    base: Path,
    git: dict[str, Any],
    timestamp: str | None = None,
) -> Path:
    """Create `<base>/<experiment>/<ts>__<sha>[__dirty]/` and return it.

    Naming includes the short SHA and a `__dirty` suffix when the working tree
    has uncommitted changes, so any artifact can be traced back to source state.
    """
    ts = timestamp or time.strftime("%Y-%m-%dT%H-%M-%S")
    suffix = git.get("commit_short", "nogit")
    name = f"{ts}__{suffix}"
    if git.get("dirty"):
        name += "__dirty"
    run_dir = base / experiment / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _config_to_dict(cfg: Any) -> dict[str, Any]:
    if isinstance(cfg, dict):
        return cfg
    if is_dataclass(cfg):
        return asdict(cfg)
    if hasattr(cfg, "__dict__"):
        return {k: v for k, v in cfg.__dict__.items() if not k.startswith("_")}
    raise TypeError(f"Cannot serialize config of type {type(cfg)}")


def write_manifest(
    run_dir: Path,
    *,
    experiment: str,
    description: str,
    config: Any,
    seed: int,
    determinism: dict[str, Any],
    git: dict[str, Any],
    env: dict[str, Any],
    extra: dict[str, Any] | None = None,
    script_path: Path | None = None,
) -> Path:
    """Write `manifest.json` (and copy the calling script for replay)."""
    manifest = {
        "experiment": experiment,
        "description": description,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": [sys.executable, *sys.argv],
        "cwd": str(Path.cwd()),
        "seed": seed,
        "determinism": determinism,
        "git": git,
        "env": env,
        "extra": extra or {},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (run_dir / "config.json").write_text(
        json.dumps(_config_to_dict(config), indent=2, default=str) + "\n"
    )
    freeze_env(run_dir / "env.txt")
    if script_path is not None and script_path.exists():
        # Frozen copy of the script source — defends against later edits.
        (run_dir / "script.py").write_text(script_path.read_text())
    return run_dir / "manifest.json"


def write_summary(run_dir: Path, lines: list[str]) -> Path:
    """Write a small human-readable summary.md with the headline numbers."""
    body = "\n".join(lines).rstrip() + "\n"
    out = run_dir / "summary.md"
    out.write_text(body)
    return out
