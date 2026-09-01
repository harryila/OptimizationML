"""Finite Newton--Schulz polynomial iterations.

No function in this module normalizes its input. This separation is essential
for attributing a monotonicity failure to normalization rather than to the
underlying scalar response.
"""

from __future__ import annotations

from torch import Tensor

from passive_muon.specs import (
    CANS_5X4,
    CLASSICAL_CUBIC,
    JORDAN_QUINTIC,
    POLAR_EXPRESS_5,
    TAYLOR_QUINTIC,
    QuinticCoefficients,
    StagedQuinticCoefficients,
    scalar_response_and_derivative_exact,
    staged_scalar_response_and_derivative_exact,
)


def _coefficient_tensors(
    coefficients: QuinticCoefficients, like: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    return tuple(like.new_tensor(float(value)) for value in coefficients.fractions())


def quintic_step(matrix: Tensor, coefficients: QuinticCoefficients) -> Tensor:
    """Apply one odd quintic spectral iteration without normalization."""

    if matrix.ndim != 2:
        raise ValueError(f"expected a 2D matrix, got shape {tuple(matrix.shape)}")
    a, b, c = _coefficient_tensors(coefficients, matrix)
    transposed = matrix.shape[0] > matrix.shape[1]
    x = matrix.mT if transposed else matrix
    gram = x @ x.mT
    correction = b * gram
    if coefficients.fractions()[2] != 0:
        correction = correction + c * (gram @ gram)
    out = a * x + correction @ x
    return out.mT if transposed else out


def iterate_quintic(
    matrix: Tensor,
    coefficients: QuinticCoefficients,
    *,
    steps: int,
) -> Tensor:
    """Apply a quintic iteration ``steps`` times without hidden scaling."""

    if steps < 0:
        raise ValueError("steps must be nonnegative")
    out = matrix
    for _ in range(steps):
        out = quintic_step(out, coefficients)
    return out


def iterate_staged_quintic(
    matrix: Tensor,
    coefficients: StagedQuinticCoefficients,
    *,
    steps: int | None = None,
) -> Tensor:
    """Apply an explicit prefix of a finite staged quintic sequence."""

    out = matrix
    for stage in coefficients.prefix(steps=steps):
        out = quintic_step(out, stage)
    return out


def scalar_step(value: Tensor, coefficients: QuinticCoefficients) -> Tensor:
    """Scalar counterpart of :func:`quintic_step`."""

    a, b, c = _coefficient_tensors(coefficients, value)
    return a * value + b * value**3 + c * value**5


def scalar_response(
    value: Tensor,
    coefficients: QuinticCoefficients,
    *,
    steps: int,
) -> Tensor:
    """Evaluate the composed singular-value response."""

    if steps < 0:
        raise ValueError("steps must be nonnegative")
    out = value
    for _ in range(steps):
        out = scalar_step(out, coefficients)
    return out


def staged_scalar_response(
    value: Tensor,
    coefficients: StagedQuinticCoefficients,
    *,
    steps: int | None = None,
) -> Tensor:
    """Scalar counterpart of :func:`iterate_staged_quintic`."""

    out = value
    for stage in coefficients.prefix(steps=steps):
        out = scalar_step(out, stage)
    return out


__all__ = [
    "CANS_5X4",
    "CLASSICAL_CUBIC",
    "JORDAN_QUINTIC",
    "POLAR_EXPRESS_5",
    "TAYLOR_QUINTIC",
    "QuinticCoefficients",
    "StagedQuinticCoefficients",
    "iterate_quintic",
    "iterate_staged_quintic",
    "quintic_step",
    "scalar_response",
    "scalar_response_and_derivative_exact",
    "scalar_step",
    "staged_scalar_response",
    "staged_scalar_response_and_derivative_exact",
]
