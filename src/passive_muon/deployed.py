"""Audited low-precision composition corresponding to KellerJordan/Muon.

This module is evidence about the deployed BF16 map, not a differentiable
theory oracle. Pairwise checks are meaningful; Jacobian certificates are not.
"""

from __future__ import annotations

from torch import Tensor

from passive_muon.normalizers import BF16FrobeniusPlusEpsNormalizer
from passive_muon.operator import NormalizedMap
from passive_muon.orthogonalizers import jordan_ns


def keller_jordan_map(matrix: Tensor, *, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Mirror the upstream cast, normalization, and Jordan iteration order."""

    return NormalizedMap(
        normalizer=BF16FrobeniusPlusEpsNormalizer(eps=eps),
        orthogonalizer=lambda value: jordan_ns(value, steps=steps),
    )(matrix)
