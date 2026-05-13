"""Identity stabilizer for no-BCNorm controls."""

from __future__ import annotations

from torch import nn


class IdentityStabilizer(nn.Identity):
    """Named identity module for experiment manifests and tests."""
