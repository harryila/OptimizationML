"""Incremental-passivity (monotonicity) deficit measurements.

These functions never attach an unqualified word ``global`` to a sampled or
pointwise quantity. A sampled maximum is a lower bound on any containing
domain's true deficit.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.func import jacrev

MatrixMap = Callable[[Tensor], Tensor]


def frobenius_inner(left: Tensor, right: Tensor) -> Tensor:
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {tuple(left.shape)} != {tuple(right.shape)}")
    return torch.sum(left * right)


def _check_pair(left: Tensor, right: Tensor) -> None:
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {tuple(left.shape)} != {tuple(right.shape)}")
    if left.ndim != 2:
        raise ValueError("passivity measurements expect two-dimensional matrices")
    if bool(torch.equal(left, right)):
        raise ValueError("the two inputs must be distinct")


def pairwise_gap(operator: MatrixMap, left: Tensor, right: Tensor) -> Tensor:
    """Return ``<T(left)-T(right), left-right>_F``."""

    _check_pair(left, right)
    delta_input = left - right
    delta_output = operator(left) - operator(right)
    return frobenius_inner(delta_output, delta_input)


def pairwise_ratio(operator: MatrixMap, left: Tensor, right: Tensor) -> Tensor:
    """Return the scale-aware incremental monotonicity quotient."""

    _check_pair(left, right)
    delta = left - right
    return pairwise_gap(operator, left, right) / frobenius_inner(delta, delta)


def pairwise_rho_required(operator: MatrixMap, left: Tensor, right: Tensor) -> Tensor:
    """Smallest constant linear shift that repairs this one pair."""

    ratio = pairwise_ratio(operator, left, right)
    return torch.clamp(-ratio, min=0)


def jacobian_matrix(operator: MatrixMap, matrix: Tensor) -> Tensor:
    """Materialize the Jacobian after vectorizing a small matrix.

    This is a diagnostic for small research fixtures, not a scalable training
    routine. The output must have the same shape as the input.
    """

    if matrix.ndim != 2:
        raise ValueError("Jacobian diagnostics expect a two-dimensional matrix")
    shape = matrix.shape
    flat_input = matrix.reshape(-1)

    def flattened(flat_matrix: Tensor) -> Tensor:
        output = operator(flat_matrix.reshape(shape))
        if output.shape != shape:
            raise ValueError(f"operator changed shape from {tuple(shape)} to {tuple(output.shape)}")
        return output.reshape(-1)

    return jacrev(flattened)(flat_input)


def symmetric_jacobian(operator: MatrixMap, matrix: Tensor) -> Tensor:
    jacobian = jacobian_matrix(operator, matrix)
    return (jacobian + jacobian.mT) / 2


@dataclass(frozen=True)
class LocalDeficit:
    """Pointwise lower-bound witness for any containing domain."""

    minimum_eigenvalue: Tensor
    rho_required: Tensor
    eigenvalues: Tensor
    symmetric_jacobian: Tensor


def local_deficit_at(operator: MatrixMap, matrix: Tensor) -> LocalDeficit:
    """Measure the smallest eigenvalue of the symmetric Jacobian at one input."""

    symmetric = symmetric_jacobian(operator, matrix)
    eigenvalues = torch.linalg.eigvalsh(symmetric)
    minimum = eigenvalues[0]
    return LocalDeficit(
        minimum_eigenvalue=minimum,
        rho_required=torch.clamp(-minimum, min=0),
        eigenvalues=eigenvalues,
        symmetric_jacobian=symmetric,
    )


@dataclass(frozen=True)
class SampledPairwiseDeficit:
    """Maximum required repair over a finite list of pairs."""

    rho_lower_bound: Tensor
    worst_pair_index: int
    ratios: Tensor


def sampled_pairwise_deficit(
    operator: MatrixMap,
    pairs: Iterable[tuple[Tensor, Tensor]],
) -> SampledPairwiseDeficit:
    """Return a sampled lower bound, never a domain-wide certificate."""

    measurements = [pairwise_ratio(operator, left, right) for left, right in pairs]
    if not measurements:
        raise ValueError("at least one pair is required")
    ratios = torch.stack(measurements)
    worst_index = int(torch.argmin(ratios))
    return SampledPairwiseDeficit(
        rho_lower_bound=torch.clamp(-ratios[worst_index], min=0),
        worst_pair_index=worst_index,
        ratios=ratios,
    )
