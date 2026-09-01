"""Public orthogonalizer implementations and registry."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from passive_muon.polynomials import (
    CANS_5X4,
    CLASSICAL_CUBIC,
    JORDAN_QUINTIC,
    POLAR_EXPRESS_5,
    TAYLOR_QUINTIC,
    QuinticCoefficients,
    StagedQuinticCoefficients,
    iterate_quintic,
    iterate_staged_quintic,
)


def polynomial_orthogonalizer(
    matrix: Tensor,
    *,
    coefficients: QuinticCoefficients,
    steps: int,
) -> Tensor:
    return iterate_quintic(matrix, coefficients, steps=steps)


def staged_polynomial_orthogonalizer(
    matrix: Tensor,
    *,
    coefficients: StagedQuinticCoefficients,
    steps: int | None = None,
) -> Tensor:
    """Apply a finite coefficient sequence without normalization or casting."""

    return iterate_staged_quintic(matrix, coefficients, steps=steps)


def jordan_ns(matrix: Tensor, *, steps: int = 5) -> Tensor:
    return polynomial_orthogonalizer(matrix, coefficients=JORDAN_QUINTIC, steps=steps)


def classical_ns(matrix: Tensor, *, steps: int = 5) -> Tensor:
    return polynomial_orthogonalizer(matrix, coefficients=CLASSICAL_CUBIC, steps=steps)


def taylor_ns5(matrix: Tensor, *, steps: int = 5) -> Tensor:
    return polynomial_orthogonalizer(matrix, coefficients=TAYLOR_QUINTIC, steps=steps)


def polar_express(matrix: Tensor, *, steps: int = 5) -> Tensor:
    """Pinned first-five Polar Express stages from the official repository.

    This is the smooth, normalization-free mathematical shadow.  The upstream
    BF16 cast and current-input Frobenius scaling remain separate policies.
    """

    return staged_polynomial_orthogonalizer(matrix, coefficients=POLAR_EXPRESS_5, steps=steps)


def cans_5x4(matrix: Tensor, *, steps: int = 4) -> Tensor:
    """Clean-room CANS degree-5, four-stage, ``delta=0.3`` baseline."""

    return staged_polynomial_orthogonalizer(matrix, coefficients=CANS_5X4, steps=steps)


def exact_polar(matrix: Tensor) -> Tensor:
    """Return the compact-SVD polar factor ``U @ Vh``.

    At rank-deficient inputs this is one valid subgradient selection, not a
    differentiable control map. Jacobian tests must use full-rank matrices with
    separated singular values.
    """

    if matrix.ndim != 2:
        raise ValueError(f"expected a 2D matrix, got shape {tuple(matrix.shape)}")
    u, _singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
    return u @ vh


ORTHOGONALIZERS: dict[str, Callable[..., Tensor]] = {
    "jordan": jordan_ns,
    "classical_ns": classical_ns,
    "taylor_ns5": taylor_ns5,
    "polar_express": polar_express,
    "cans_5x4": cans_5x4,
    "exact_polar": exact_polar,
}
