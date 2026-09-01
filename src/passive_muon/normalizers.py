"""Normalization policies, kept separate from every orthogonalizer."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch
from torch import Tensor


def _check_matrix(matrix: Tensor) -> None:
    if matrix.ndim != 2:
        raise ValueError(f"expected a 2D matrix, got shape {tuple(matrix.shape)}")
    if not matrix.is_floating_point():
        raise TypeError("normalization requires a floating-point tensor")


@dataclass(frozen=True)
class ExactFrobeniusNormalizer:
    """Return ``M / ||M||_F`` and reject the undefined zero input."""

    def __call__(self, matrix: Tensor) -> Tensor:
        _check_matrix(matrix)
        norm = torch.linalg.vector_norm(matrix)
        if bool(norm == 0):
            raise ValueError("exact Frobenius normalization is undefined at zero")
        return matrix / norm


@dataclass(frozen=True)
class FrobeniusPlusEpsNormalizer:
    """The deployed-style rule ``M / (||M||_F + eps)``."""

    eps: float = 1e-7

    def __post_init__(self) -> None:
        if self.eps <= 0:
            raise ValueError("eps must be strictly positive")

    def __call__(self, matrix: Tensor) -> Tensor:
        _check_matrix(matrix)
        norm = torch.linalg.vector_norm(matrix)
        eps = matrix.new_tensor(self.eps)
        return matrix / (norm + eps)


@dataclass(frozen=True)
class FlooredFrobeniusNormalizer:
    """Return ``M / max(floor, ||M||_F)`` for a fixed positive floor."""

    floor: float

    def __post_init__(self) -> None:
        if not isfinite(self.floor) or self.floor <= 0:
            raise ValueError("floor must be finite and strictly positive")

    def __call__(self, matrix: Tensor) -> Tensor:
        _check_matrix(matrix)
        norm = torch.linalg.vector_norm(matrix)
        floor = matrix.new_tensor(self.floor)
        if not bool(torch.isfinite(floor)) or bool(floor <= 0):
            raise ValueError("floor must remain finite and positive in the matrix dtype")
        denominator = torch.maximum(norm, floor)
        return matrix / denominator


@dataclass(frozen=True)
class BF16FrobeniusPlusEpsNormalizer:
    """Upstream deployment order: cast to BF16, then normalize with epsilon."""

    eps: float = 1e-7

    def __post_init__(self) -> None:
        if self.eps <= 0:
            raise ValueError("eps must be strictly positive")

    def __call__(self, matrix: Tensor) -> Tensor:
        _check_matrix(matrix)
        low_precision = matrix.to(torch.bfloat16)
        norm = torch.linalg.vector_norm(low_precision)
        return low_precision / (norm + low_precision.new_tensor(self.eps))


@dataclass(frozen=True)
class FixedScaleNormalizer:
    """Return ``M / scale`` for a scale independent of the current input."""

    scale: float

    def __post_init__(self) -> None:
        if self.scale <= 0:
            raise ValueError("scale must be strictly positive")

    def __call__(self, matrix: Tensor) -> Tensor:
        _check_matrix(matrix)
        return matrix / matrix.new_tensor(self.scale)


# Descriptive aliases used in the paper brief.
CurrentFrobeniusNormalizer = ExactFrobeniusNormalizer
