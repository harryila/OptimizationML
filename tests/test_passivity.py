from __future__ import annotations

import pytest
import torch

from passive_muon.deployed import keller_jordan_map
from passive_muon.normalizers import ExactFrobeniusNormalizer, FixedScaleNormalizer
from passive_muon.operator import NormalizedMap
from passive_muon.orthogonalizers import classical_ns, jordan_ns
from passive_muon.passivity import (
    jacobian_matrix,
    local_deficit_at,
    pairwise_gap,
    pairwise_ratio,
    pairwise_rho_required,
)
from passive_muon.repairs import add_linear_repair


def _diagonal(values: tuple[float, float]) -> torch.Tensor:
    return torch.diag(torch.tensor(values, dtype=torch.float64))


def test_practical_jordan_pair_is_negative_only_with_current_normalization() -> None:
    left = _diagonal((3.0, 4.0))
    right = _diagonal((2.5, 3.0))
    current = NormalizedMap(ExactFrobeniusNormalizer(), lambda x: jordan_ns(x, steps=5))
    fixed = NormalizedMap(FixedScaleNormalizer(5.0), lambda x: jordan_ns(x, steps=5))

    assert pairwise_gap(current, left, right).item() == pytest.approx(-0.07025433715577033)
    assert pairwise_gap(fixed, left, right).item() == pytest.approx(0.3750465803803160)
    assert pairwise_rho_required(current, left, right).item() == pytest.approx(0.05620346972461626)


def test_deployed_map_matches_literal_pinned_upstream_bf16_order() -> None:
    matrix = _diagonal((3.0, 4.0))
    expected = matrix.bfloat16()
    expected = expected / (expected.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(5):
        gram = expected @ expected.mT
        correction = -4.7750 * gram + 2.0315 * gram @ gram
        expected = 3.4445 * expected + correction @ expected

    actual = keller_jordan_map(matrix)
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_classical_rational_fixture_attributes_failure_to_normalization() -> None:
    left = _diagonal((11 / 20, 57 / 80))
    right = _diagonal((13 / 20, 71 / 80))
    current = NormalizedMap(ExactFrobeniusNormalizer(), lambda x: classical_ns(x, steps=1))
    fixed = NormalizedMap(FixedScaleNormalizer(1.0), lambda x: classical_ns(x, steps=1))

    assert pairwise_gap(current, left, right).item() < -4.9e-4
    assert pairwise_gap(fixed, left, right).item() == pytest.approx(532639 / 20480000)


def test_local_jacobian_has_negative_mode_and_fixed_rho_shifts_it() -> None:
    matrix = _diagonal((0.6, 0.8))
    operator = NormalizedMap(ExactFrobeniusNormalizer(), lambda value: jordan_ns(value, steps=5))
    report = local_deficit_at(operator, matrix)
    assert report.minimum_eigenvalue.item() == pytest.approx(-0.05708517058992462)

    rho = report.rho_required.item() + 1e-10
    repaired = local_deficit_at(add_linear_repair(operator, rho), matrix)
    assert repaired.minimum_eigenvalue.item() >= -1e-11


def test_pairwise_ratio_scales_inversely_for_exact_normalization() -> None:
    left = _diagonal((3.0, 4.0))
    right = _diagonal((2.5, 3.0))
    operator = NormalizedMap(ExactFrobeniusNormalizer(), lambda value: jordan_ns(value, steps=5))
    base = pairwise_ratio(operator, left, right)
    scaled = pairwise_ratio(operator, left / 2, right / 2)
    torch.testing.assert_close(scaled, 2 * base)


def test_pairwise_required_rho_is_the_exact_threshold_for_that_pair() -> None:
    left = _diagonal((3.0, 4.0))
    right = _diagonal((2.5, 3.0))
    operator = NormalizedMap(ExactFrobeniusNormalizer(), lambda value: jordan_ns(value, steps=5))
    rho = pairwise_rho_required(operator, left, right).item()
    at_threshold = pairwise_gap(add_linear_repair(operator, rho), left, right)
    below = pairwise_gap(add_linear_repair(operator, rho - 1e-5), left, right)
    above = pairwise_gap(add_linear_repair(operator, rho + 1e-5), left, right)
    assert at_threshold.item() == pytest.approx(0.0, abs=1e-14)
    assert below.item() < 0 < above.item()


def test_jacrev_matches_central_directional_difference() -> None:
    matrix = torch.tensor([[0.6, 0.1], [-0.2, 0.75]], dtype=torch.float64)
    direction = torch.tensor([[0.2, -0.3], [0.4, 0.1]], dtype=torch.float64)
    operator = NormalizedMap(ExactFrobeniusNormalizer(), lambda value: classical_ns(value, steps=2))
    jacobian = jacobian_matrix(operator, matrix)
    autodiff = (jacobian @ direction.reshape(-1)).reshape_as(matrix)
    step = 1e-6
    difference = operator(matrix + step * direction) - operator(matrix - step * direction)
    finite_difference = difference / (2 * step)
    torch.testing.assert_close(autodiff, finite_difference, rtol=2e-7, atol=2e-8)


def test_linear_psd_control_has_zero_deficit() -> None:
    matrix = _diagonal((0.3, 0.7))

    def operator(value: torch.Tensor) -> torch.Tensor:
        return 2 * value

    report = local_deficit_at(operator, matrix)
    assert report.minimum_eigenvalue.item() == pytest.approx(2.0)
    assert report.rho_required.item() == 0
