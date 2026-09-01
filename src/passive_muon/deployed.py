"""Audited low-precision composition corresponding to KellerJordan/Muon.

This module is evidence about the deployed BF16 map, not a differentiable
theory oracle. Pairwise checks are meaningful; Jacobian certificates are not.
"""

from __future__ import annotations

from torch import Tensor

from passive_muon.specs import JORDAN_QUINTIC


def keller_jordan_map(matrix: Tensor, *, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Mirror pinned KellerJordan/Muon BF16 operations literally.

    The source expression ``b*A + c*A @ A`` is left-associative, so its
    second term is ``(c*A) @ A``.  That grouping and the one-time orientation
    are observable in BF16 and must not be replaced by an algebraically
    equivalent real-arithmetic polynomial helper.

    Audited upstream revision:
    ``f98f1cacc0263b04290753e32be8d498c1efc806``.
    """

    if matrix.ndim != 2:
        raise ValueError(f"expected a 2D matrix, got shape {tuple(matrix.shape)}")
    if not matrix.is_floating_point():
        raise TypeError("the deployed map requires a floating-point tensor")
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    if eps <= 0:
        raise ValueError("eps must be strictly positive")

    out = matrix.bfloat16()
    transposed = matrix.shape[-2] > matrix.shape[-1]
    if transposed:
        out = out.mT
    out = out / (out.norm(dim=(-2, -1), keepdim=True) + eps)
    a, b, c = (float(JORDAN_QUINTIC.a), float(JORDAN_QUINTIC.b), float(JORDAN_QUINTIC.c))
    for _ in range(steps):
        gram = out @ out.mT
        correction = b * gram + c * gram @ gram
        out = a * out + correction @ out
    if transposed:
        out = out.mT
    return out
