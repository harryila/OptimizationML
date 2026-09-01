"""Explicit composition of a normalizer, orthogonalizer, and linear repair."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from torch import Tensor

from passive_muon.normalizers import (
    ExactFrobeniusNormalizer,
    FixedScaleNormalizer,
    FlooredFrobeniusNormalizer,
    FrobeniusPlusEpsNormalizer,
)
from passive_muon.orthogonalizers import ORTHOGONALIZERS

MatrixMap = Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class NormalizedMap:
    """Compose named pieces without hiding data-dependent normalization."""

    normalizer: MatrixMap
    orthogonalizer: MatrixMap
    repair_rho: float = 0.0

    def __post_init__(self) -> None:
        if self.repair_rho < 0:
            raise ValueError("repair_rho must be nonnegative")

    def __call__(self, matrix: Tensor) -> Tensor:
        output = self.orthogonalizer(self.normalizer(matrix))
        if self.repair_rho:
            output = output + matrix.new_tensor(self.repair_rho) * matrix
        return output


def orthogonalize(
    matrix: Tensor,
    *,
    polynomial: str = "jordan",
    normalization: str = "current_frobenius",
    repair_rho: float = 0.0,
    steps: int | None = None,
    eps: float = 1e-7,
    floor: float | None = None,
    fixed_scale: float | None = None,
) -> Tensor:
    """Convenience interface specified by the research brief."""

    if polynomial not in ORTHOGONALIZERS:
        choices = ", ".join(sorted(ORTHOGONALIZERS))
        raise ValueError(f"unknown polynomial {polynomial!r}; choose from {choices}")

    if normalization == "current_frobenius":
        normalizer = ExactFrobeniusNormalizer()
    elif normalization == "current_frobenius_plus_eps":
        normalizer = FrobeniusPlusEpsNormalizer(eps=eps)
    elif normalization == "floored_frobenius":
        if floor is None:
            raise ValueError("floor must be provided for floored Frobenius normalization")
        normalizer = FlooredFrobeniusNormalizer(floor=floor)
    elif normalization == "fixed_scale":
        if fixed_scale is None:
            raise ValueError("fixed_scale must be provided for fixed normalization")
        normalizer = FixedScaleNormalizer(fixed_scale)
    else:
        raise ValueError(f"unknown normalization {normalization!r}")

    base = ORTHOGONALIZERS[polynomial]
    if polynomial == "exact_polar":
        orthogonalizer = base
    else:

        def orthogonalizer(value: Tensor) -> Tensor:
            if steps is None:
                return base(value)
            return base(value, steps=steps)

    return NormalizedMap(normalizer, orthogonalizer, repair_rho)(matrix)
