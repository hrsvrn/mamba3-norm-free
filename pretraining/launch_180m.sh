#!/usr/bin/env bash
# Launcher for the 180M Mamba-3 FineWebEdu 10B-token pretraining run.
# Default target: 1× RTX PRO 6000 Blackwell (96GB VRAM, 28 vCPU, 160GB RAM).
#
# Usage:
#   bash pretraining/launch_180m.sh                         # 1 GPU (default)
#   NPROC=N bash pretraining/launch_180m.sh                 # N-GPU DDP/FSDP (multi-GPU hosts only)
#   CONFIG=path/to/other.yaml bash pretraining/launch_180m.sh
#
# Hugging Face Hub upload requires:
#   HF_TOKEN=hf_xxxx                       # access token with write scope
#   HF_REPO_ID=<user>/mamba3-180m-fwe-10B  # optional; otherwise auto-derived from whoami + run name
#
# SLURM:
#   sbatch pretraining/launch_180m.sh
#
#SBATCH --job-name=mamba3-180m-finewebedu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=160G
#SBATCH --time=72:00:00
#SBATCH --output=runs/slurm-%j.out

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-pretraining/configs/mamba3_180m.yaml}"
NPROC="${NPROC:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"

export PYTHONPATH="${REPO_ROOT}/mamba-og:${PYTHONPATH:-}"

export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${REPO_ROOT}/.triton_cache}"
mkdir -p "${TRITON_CACHE_DIR}"

# Resolve the Python interpreter. Prefer the project's uv-managed venv (this
# repo uses pyproject.toml + uv.lock). Override with PYTHON=<path> if needed.
if [ -z "${PYTHON:-}" ]; then
    if [ -x "${REPO_ROOT}/.venv/bin/python" ]; then
        PYTHON="${REPO_ROOT}/.venv/bin/python"
    elif command -v uv >/dev/null 2>&1; then
        PYTHON="uv run --no-sync python"
    elif command -v python >/dev/null 2>&1; then
        PYTHON="python"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON="python3"
    else
        echo "ERROR: no python interpreter found (set PYTHON=... or run 'uv sync' first)" >&2
        exit 1
    fi
fi
# Same logic for torchrun: prefer venv binary, else `uv run torchrun`.
if [ -z "${TORCHRUN:-}" ]; then
    if [ -x "${REPO_ROOT}/.venv/bin/torchrun" ]; then
        TORCHRUN="${REPO_ROOT}/.venv/bin/torchrun"
    elif command -v uv >/dev/null 2>&1; then
        TORCHRUN="uv run --no-sync torchrun"
    else
        TORCHRUN="torchrun"
    fi
fi

# Llama-3.1 tokenizer: byte-identical across meta-llama / NousResearch / unsloth
# mirrors. The official meta-llama repos are gated; we warm the cache here via
# the same fallback chain `data._Llama3TokenizerAdapter` uses, so any auth or
# network failure surfaces before training starts (not 5 minutes in).
echo "==> warming Llama-3.1 tokenizer cache"
${PYTHON} - <<'PY'
import sys
sys.path.insert(0, "pretraining")
from data import _Llama3TokenizerAdapter
tok = _Llama3TokenizerAdapter()
print(f"    tokenizer OK ({tok.model_id}, eot={tok.eot_token})")
PY

echo "==> repo:    ${REPO_ROOT}"
echo "==> config:  ${CONFIG}"
echo "==> nproc:   ${NPROC}"
echo "==> python:  ${PYTHON}"
echo "==> HF repo: ${HF_REPO_ID:-<auto from whoami + run name>}"
${PYTHON} -c "import torch; print(f'==> torch:   {torch.__version__}  cuda={torch.cuda.is_available()}  n_gpu={torch.cuda.device_count()}')"

if [ "${NPROC}" = "1" ]; then
    exec ${PYTHON} pretraining/train_180m.py --config "${CONFIG}"
else
    exec ${TORCHRUN} \
        --standalone \
        --nproc_per_node="${NPROC}" \
        --master_port="${MASTER_PORT}" \
        pretraining/train_180m.py --config "${CONFIG}"
fi
