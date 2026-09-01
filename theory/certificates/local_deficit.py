"""Analytic diagonal-block deficit formula used by the theorem.

This module is dependency-free so the exact formula can be reviewed separately
from PyTorch autodiff.
"""

from __future__ import annotations

from decimal import Decimal


def two_coordinate_symmetric_jacobian(
    u1: Decimal,
    u2: Decimal,
    derivative1: Decimal,
    derivative2: Decimal,
    *,
    radius: Decimal = Decimal(1),
) -> tuple[tuple[Decimal, Decimal], tuple[Decimal, Decimal]]:
    """Return the 2x2 symmetric Jacobian on diagonal perturbations."""

    if radius <= 0:
        raise ValueError("radius must be positive")
    off_diagonal = -(derivative1 + derivative2) * u1 * u2 / (2 * radius)
    return (
        (derivative1 * u2**2 / radius, off_diagonal),
        (off_diagonal, derivative2 * u1**2 / radius),
    )
