#!/usr/bin/env bash
# Cluster launcher for the 50M Mamba-3 FineWebEdu pretraining run.
#
# Usage:
#   bash pretraining/launch.sh                         # 1 GPU, default config
#   NPROC=8 bash pretraining/launch.sh                 # 8-GPU DDP
#   CONFIG=pretraining/configs/mamba3_50m.yaml bash pretraining/launch.sh
#
# SLURM:
#   sbatch pretraining/launch.sh
#
#SBATCH --job-name=mamba3-50m-finewebedu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=runs/slurm-%j.out

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-pretraining/configs/mamba3_50m.yaml}"
NPROC="${NPROC:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"

# Make mamba-og importable without an editable install.
export PYTHONPATH="${REPO_ROOT}/mamba-og:${PYTHONPATH:-}"

# Recommended for H100 BF16 throughput.
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
# Triton autotune cache — share across runs to avoid re-tuning every job.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${REPO_ROOT}/.triton_cache}"
mkdir -p "${TRITON_CACHE_DIR}"

echo "==> repo:    ${REPO_ROOT}"
echo "==> config:  ${CONFIG}"
echo "==> nproc:   ${NPROC}"
echo "==> python:  $(which python)"
python -c "import torch; print(f'==> torch:   {torch.__version__}  cuda={torch.cuda.is_available()}  n_gpu={torch.cuda.device_count()}')"

if [ "${NPROC}" = "1" ]; then
    exec python pretraining/train_50m.py --config "${CONFIG}"
else
    exec torchrun \
        --standalone \
        --nproc_per_node="${NPROC}" \
        --master_port="${MASTER_PORT}" \
        pretraining/train_50m.py --config "${CONFIG}"
fi
