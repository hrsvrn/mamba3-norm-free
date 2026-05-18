#!/usr/bin/env bash
# Cluster launcher for the 180M Mamba-3 FineWebEdu 10B-token pretraining run.
#
# Usage:
#   bash pretraining/launch_180m.sh                         # 1 GPU
#   NPROC=8 bash pretraining/launch_180m.sh                 # 8-GPU DDP/FSDP
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
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
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

# Llama-3.1 tokenizer is gated -- download it ahead of time to fail fast if the
# token doesn't have access, rather than discovering it mid-run.
if [ -n "${HF_TOKEN:-}" ]; then
    echo "==> warming Llama-3.1 tokenizer cache"
    python - <<'PY'
import os
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B", token=os.environ.get("HF_TOKEN"))
print("    tokenizer OK")
PY
fi

echo "==> repo:    ${REPO_ROOT}"
echo "==> config:  ${CONFIG}"
echo "==> nproc:   ${NPROC}"
echo "==> python:  $(which python)"
echo "==> HF repo: ${HF_REPO_ID:-<auto from whoami + run name>}"
python -c "import torch; print(f'==> torch:   {torch.__version__}  cuda={torch.cuda.is_available()}  n_gpu={torch.cuda.device_count()}')"

if [ "${NPROC}" = "1" ]; then
    exec python pretraining/train_180m.py --config "${CONFIG}"
else
    exec torchrun \
        --standalone \
        --nproc_per_node="${NPROC}" \
        --master_port="${MASTER_PORT}" \
        pretraining/train_180m.py --config "${CONFIG}"
fi
