from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
import torch

from passive_muon.floored_certificate import (
    JORDAN_DERIVATIVE_LOWER,
    JORDAN_DERIVATIVE_UPPER,
    JORDAN_PAIR_DEFICIT_LOWER,
    certified_dimension_uniform_deficit,
    certified_repair_conductance,
    certify_jordan_derivative_range,
    exact_jordan_pair_deficit,
    projection_envelope_lower,
)
from passive_muon.normalizers import FlooredFrobeniusNormalizer
from passive_muon.orthogonalizers import jordan_ns

EXPECTED_TRACE_SHA256 = "701b9047e67c96d1f727642c693fb545f13c4981765310162a9a51b730f34d5e"
EXPECTED_UPPER = Fraction(41_528_474_059_081, 260_261_360_000)


@pytest.mark.parametrize("precision_bits", [160, 224])
def test_arb_derivative_certificate_replays_at_two_precisions(precision_bits: int) -> None:
    certificate = certify_jordan_derivative_range(precision_bits=precision_bits)
    assert certificate.derivative_lower == JORDAN_DERIVATIVE_LOWER
    assert certificate.derivative_upper == JORDAN_DERIVATIVE_UPPER
    assert certificate.configured_maximum_power == 48
    assert certificate.leaf_count == 25_370
    assert certificate.maximum_power == 32
    assert certificate.trace_sha256 == EXPECTED_TRACE_SHA256


def test_exact_pair_and_dimension_uniform_bound_form_a_tight_bracket() -> None:
    exact_pair = exact_jordan_pair_deficit()
    upper = certified_dimension_uniform_deficit()
    assert exact_pair > JORDAN_PAIR_DEFICIT_LOWER
    assert upper > JORDAN_PAIR_DEFICIT_LOWER
    assert upper == EXPECTED_UPPER
    relative_width = (upper - JORDAN_PAIR_DEFICIT_LOWER) / JORDAN_PAIR_DEFICIT_LOWER
    assert float(relative_width) < 1e-4


def test_certified_conductance_has_the_exact_inverse_floor_scaling() -> None:
    assert certified_repair_conductance(Fraction(1)) == EXPECTED_UPPER
    assert certified_repair_conductance(Fraction(5, 2)) == EXPECTED_UPPER * Fraction(2, 5)
    assert certified_repair_conductance(Fraction(2), mu=Fraction(1, 7)) == (
        EXPECTED_UPPER / 2 + Fraction(1, 7)
    )
    with pytest.raises(ValueError, match="floor must be positive"):
        certified_repair_conductance(Fraction(0))
    with pytest.raises(ValueError, match="mu must be nonnegative"):
        certified_repair_conductance(Fraction(1), mu=Fraction(-1))


def test_projection_envelope_bounds_random_full_symmetric_tangent_operators() -> None:
    """Numerical regression for the analytic lemma, not a sampled certificate."""

    slope_lower = float(JORDAN_DERIVATIVE_LOWER)
    slope_upper = float(JORDAN_DERIVATIVE_UPPER)
    envelope = float(projection_envelope_lower(JORDAN_DERIVATIVE_LOWER, JORDAN_DERIVATIVE_UPPER))
    generator = np.random.default_rng(20260901)

    for dimension in (2, 3, 7):
        for projection_rank in range(1, dimension):
            orthogonal, _ = np.linalg.qr(generator.standard_normal((dimension, dimension)))
            eigenvectors, _ = np.linalg.qr(generator.standard_normal((dimension, dimension)))
            eigenvalues = generator.uniform(slope_lower, slope_upper, size=dimension)
            operator = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            projection = orthogonal[:, :projection_rank] @ orthogonal[:, :projection_rank].T
            symmetric_product = (operator @ projection + projection @ operator) / 2
            assert np.linalg.eigvalsh(symmetric_product)[0] >= envelope - 1e-10


def _symmetric_jacobian_minimum(matrix: torch.Tensor, *, floor: float, rho: float) -> float:
    shape = matrix.shape

    def flattened(values: torch.Tensor) -> torch.Tensor:
        candidate = values.reshape(shape)
        normalized = FlooredFrobeniusNormalizer(floor)(candidate)
        return (jordan_ns(normalized, steps=5) + rho * candidate).reshape(-1)

    vector = matrix.detach().clone().requires_grad_(True).reshape(-1)
    jacobian = torch.autograd.functional.jacobian(flattened, vector)
    symmetric = (jacobian + jacobian.mT) / 2
    return float(torch.linalg.eigvalsh(symmetric)[0])


@pytest.mark.parametrize(
    "matrix",
    [
        torch.zeros((2, 3), dtype=torch.float64),
        torch.tensor([[0.3, 0.0, 0.0], [0.0, 0.3, 0.0]], dtype=torch.float64),
        torch.tensor([[0.6, 0.0], [0.0, 0.8], [0.0, 0.0]], dtype=torch.float64),
        torch.tensor([[1.2, 0.0, 0.0], [0.0, 1.6, 0.0]], dtype=torch.float64),
        torch.tensor([[0.2, -0.4, 0.1], [0.5, 0.3, -0.2]], dtype=torch.float64),
    ],
)
def test_full_rectangular_jacobians_respect_the_certified_repair(matrix: torch.Tensor) -> None:
    """Autodiff regression at zero, repeated, boundary, exterior, and dense inputs."""

    upper = float(certified_dimension_uniform_deficit())
    unrepaired_minimum = _symmetric_jacobian_minimum(matrix, floor=1.0, rho=0.0)
    repaired_minimum = _symmetric_jacobian_minimum(matrix, floor=1.0, rho=upper)
    assert unrepaired_minimum >= -upper - 2e-8
    assert repaired_minimum >= -2e-8


def test_floored_map_and_pairwise_deficit_have_inverse_floor_scaling() -> None:
    left = torch.tensor([[0.2, -0.1], [0.0, 0.3]], dtype=torch.float64)
    right = torch.tensor([[-0.1, 0.4], [0.2, 0.0]], dtype=torch.float64)

    def mapped(matrix: torch.Tensor, floor: float) -> torch.Tensor:
        return jordan_ns(FlooredFrobeniusNormalizer(floor)(matrix), steps=5)

    base_difference = left - right
    base_ratio = torch.sum((mapped(left, 1.0) - mapped(right, 1.0)) * base_difference) / torch.sum(
        base_difference**2
    )
    for floor in (0.25, 2.0, 7.0):
        torch.testing.assert_close(mapped(floor * left, floor), mapped(left, 1.0))
        scaled_difference = floor * base_difference
        scaled_ratio = torch.sum(
            (mapped(floor * left, floor) - mapped(floor * right, floor)) * scaled_difference
        ) / torch.sum(scaled_difference**2)
        torch.testing.assert_close(scaled_ratio, base_ratio / floor)
