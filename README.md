# mamba-norm-free

Ablation study replacing BCNorm in Mamba-3 with element-wise squashing functions (DyT, Derf, DyISRU, DySN) — targeting ICLR 2026.

## Motivation

BCNorm (QK-style cross-channel normalization on B/C projections) requires a global reduction that serializes memory-bound decode. Element-wise alternatives have no such bottleneck.

## Structure

```
mamba3-minimal/   # Pure PyTorch Mamba-3 reference implementation
theoretical-findings/  # BCNorm analysis and research scope notes
configs/          # Dataset manifest, experiment configs
src/              # Normalization modules and Triton kernels (WIP)
```

## Quickstart

```bash
# Reference implementation (no custom kernels required)
cd mamba3-minimal
pip install torch einops
python mamba3.py        # forward vs. step-by-step parity check
python tests/test_parity.py  # state-tracking: ~100% (vs Mamba-2's ~50%)
```

## Dependencies

```bash
uv sync  # Python 3.11
```
