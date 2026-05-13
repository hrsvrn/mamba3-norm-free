"""Deterministic dataset pipeline for the norm-free Mamba-3 ablations.

Submodules are imported on demand so that lightweight users (e.g. the
training loop, which only needs `dataloader`) do not pull in `transformers`
or `datasets`.
"""
