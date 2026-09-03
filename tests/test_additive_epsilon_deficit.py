from __future__ import annotations

import sys
from fractions import Fraction
from itertools import pairwise

import pytest
import torch

from passive_muon.additive_epsilon_deficit import (
    DEPLOYED_EPSILON,
    PAIR_DEFICIT_STRICT_LOWER,
    PAIR_DIRECTION_HIGH,
    PAIR_DIRECTION_LOW,
    PAIR_NORMALIZED_RADIUS,
    PAIR_RAW_RADIUS,
    RADIUS_BANDS,
    certified_repair_conductance,
    certified_unit_epsilon_deficit_upper,
    certify_additive_epsilon_intervals,
    exact_rank_two_pair_deficit,
    exact_rank_two_pair_deficit_sha256,
    radius_band_bound,
    scale_unit_deficit,
    zero_input_derivative_gain,
)
from passive_muon.normalizers import FrobeniusPlusEpsNormalizer
from passive_muon.orthogonalizers import jordan_ns

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)

EXPECTED_UPPER = Fraction(6_602_082_433_275_499_863, 41_641_817_600_000_000)
EXPECTED_PAIR_SHA256 = "de076057ce58ad174a52a8e0e492f7efffe0488bbca19c03cd1ce0fee75a7336"
EXPECTED_GLOBAL_TRACE = "701b9047e67c96d1f727642c693fb545f13c4981765310162a9a51b730f34d5e"
EXPECTED_PREFIXES = (
    (Fraction(1, 200), 276, 19, "8bd62eb57bed686f163694b0a076b04c891a773482e794f05c29967bd90be3c7"),
    (Fraction(3, 500), 367, 23, "32a09cd669930a67d0b1be0079f359f0b56ceb16f785b60ac3a57a14af403a23"),
    (
        Fraction(63, 10_000),
        909,
        23,
        "bda19826bba8413a2a8b8b0e12911da78ca38e6cd089545f561e772206860b1a",
    ),
)


@pytest.mark.parametrize("precision_bits", [160, 224])
def test_arb_interval_certificate_replays_at_two_precisions(precision_bits: int) -> None:
    certificate = certify_additive_epsilon_intervals(precision_bits=precision_bits)
    assert certificate.global_range.leaf_count == 25_370
    assert certificate.global_range.maximum_power == 32
    assert certificate.global_range.trace_sha256 == EXPECTED_GLOBAL_TRACE
    assert len(certificate.prefix_ranges) == len(EXPECTED_PREFIXES)
    for actual, expected in zip(certificate.prefix_ranges, EXPECTED_PREFIXES, strict=True):
        endpoint, leaf_count, maximum_power, digest = expected
        assert actual.endpoint == endpoint
        assert actual.leaf_count == leaf_count
        assert actual.maximum_power == maximum_power
        assert actual.trace_sha256 == digest


def test_four_radius_bands_are_contiguous_and_give_locked_upper() -> None:
    assert RADIUS_BANDS[0].left == 0
    assert RADIUS_BANDS[-1].right == 1
    assert all(left.right == right.left for left, right in pairwise(RADIUS_BANDS))
    bounds = tuple(radius_band_bound(band) for band in RADIUS_BANDS)
    assert tuple(item.deficit_upper for item in bounds) == (
        Fraction(504, 5),
        Fraction(1_632_191_906_745_061_351, 10_531_913_600_000_000),
        Fraction(491_257_553_435_828_999, 3_099_060_000_000_000),
        EXPECTED_UPPER,
    )
    assert tuple(item.maximizing_radius for item in bounds) == (
        Fraction(0),
        Fraction(1, 200),
        Fraction(3, 500),
        Fraction(63, 10_000),
    )
    assert certified_unit_epsilon_deficit_upper() == EXPECTED_UPPER


def test_exact_rational_pair_is_a_strict_lower_witness() -> None:
    assert PAIR_DIRECTION_LOW**2 + PAIR_DIRECTION_HIGH**2 == 1
    assert PAIR_RAW_RADIUS == PAIR_NORMALIZED_RADIUS / (1 - PAIR_NORMALIZED_RADIUS)
    deficit = exact_rank_two_pair_deficit()
    assert deficit > PAIR_DEFICIT_STRICT_LOWER
    assert deficit < EXPECTED_UPPER
    assert exact_rank_two_pair_deficit_sha256() == EXPECTED_PAIR_SHA256
    assert len(str(deficit.numerator)) == 49_624
    assert len(str(deficit.denominator)) == 49_621


def test_exact_inverse_epsilon_scaling_and_repair() -> None:
    assert scale_unit_deficit(PAIR_DEFICIT_STRICT_LOWER, Fraction(1)) == (PAIR_DEFICIT_STRICT_LOWER)
    assert scale_unit_deficit(PAIR_DEFICIT_STRICT_LOWER, DEPLOYED_EPSILON) == 1_581_172_496
    assert certified_repair_conductance(Fraction(1)) == EXPECTED_UPPER
    assert certified_repair_conductance(DEPLOYED_EPSILON) == EXPECTED_UPPER * 10_000_000
    with pytest.raises(ValueError, match="epsilon must be positive"):
        certified_repair_conductance(Fraction(0))
    with pytest.raises(ValueError, match="epsilon must be positive"):
        scale_unit_deficit(Fraction(1), Fraction(-1))


def test_origin_derivative_is_positive_and_scales_exactly() -> None:
    expected = Fraction(15_516_041_187_205_853_449, 32_000_000_000_000_000)
    assert zero_input_derivative_gain(Fraction(1)) == expected
    assert zero_input_derivative_gain(DEPLOYED_EPSILON) == expected * 10_000_000
    with pytest.raises(ValueError, match="epsilon must be positive"):
        zero_input_derivative_gain(Fraction(0))


def _mapped(matrix: torch.Tensor, epsilon: float, rho: float = 0.0) -> torch.Tensor:
    normalized = FrobeniusPlusEpsNormalizer(eps=epsilon)(matrix)
    return jordan_ns(normalized, steps=5) + rho * matrix


def _minimum_symmetric_jacobian(matrix: torch.Tensor, epsilon: float, rho: float) -> float:
    shape = matrix.shape

    def flattened(vector: torch.Tensor) -> torch.Tensor:
        return _mapped(vector.reshape(shape), epsilon, rho).reshape(-1)

    vector = matrix.detach().clone().reshape(-1).requires_grad_(True)
    jacobian = torch.autograd.functional.jacobian(flattened, vector)
    symmetric = (jacobian + jacobian.mT) / 2
    return float(torch.linalg.eigvalsh(symmetric)[0])


@pytest.mark.parametrize(
    ("matrix", "epsilon"),
    [
        (torch.diag(torch.tensor([0.0064, 0.0064], dtype=torch.float64)), 1.0),
        (torch.tensor([[0.003, -0.002, 0.001], [0.004, 0.0, -0.003]], dtype=torch.float64), 1.0),
        (torch.tensor([[3e-8, 0.0], [0.0, 4e-8], [0.0, 0.0]], dtype=torch.float64), 1e-7),
        (torch.tensor([[0.3, -0.4], [0.2, 0.1]], dtype=torch.float64), 0.25),
    ],
)
def test_rectangular_autodiff_diagnostics_respect_global_upper(
    matrix: torch.Tensor, epsilon: float
) -> None:
    """Numerical falsification checks; the Arb/analytic proof is authoritative."""

    rho = float(EXPECTED_UPPER) / epsilon
    assert _minimum_symmetric_jacobian(matrix, epsilon, 0.0) >= -rho - 2e-5
    assert _minimum_symmetric_jacobian(matrix, epsilon, rho) >= -2e-5


def test_exact_pair_has_negative_gap_below_the_locked_lower_repair() -> None:
    raw_radius = float(PAIR_RAW_RADIUS)
    low = float(PAIR_DIRECTION_LOW)
    high = float(PAIR_DIRECTION_HIGH)
    left = raw_radius * torch.diag(torch.tensor([low, high], dtype=torch.float64))
    right = raw_radius * torch.diag(torch.tensor([high, low], dtype=torch.float64))
    difference = left - right
    output_difference = _mapped(left, 1.0) - _mapped(right, 1.0)
    quotient = torch.sum(output_difference * difference) / torch.sum(difference**2)
    assert -float(quotient) > float(PAIR_DEFICIT_STRICT_LOWER)

    under_repair = float(PAIR_DEFICIT_STRICT_LOWER)
    repaired_difference = output_difference + under_repair * difference
    assert float(torch.sum(repaired_difference * difference)) < 0


def test_identity_polynomial_is_a_monotone_negative_control() -> None:
    for radius, epsilon in (
        (Fraction(0), Fraction(1)),
        (Fraction(3, 7), Fraction(2, 5)),
        (Fraction(10_000), DEPLOYED_EPSILON),
    ):
        tangential = 1 / (radius + epsilon)
        radial = epsilon / (radius + epsilon) ** 2
        assert tangential > 0
        assert radial > 0
