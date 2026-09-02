from __future__ import annotations

import cmath
import math
import random
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from passive_muon.ema_nesterov_iqc import locked_ema_nesterov_certificate
from passive_muon.structure_aware_stability import (
    LOCKED_BETA,
    LOCKED_CENTER_GAIN,
    LOCKED_DEFICIT_UPPER,
    LOCKED_FLOOR,
    LOCKED_HESSIAN_LOWER,
    LOCKED_HESSIAN_UPPER,
    LOCKED_LEARNING_RATE,
    LOCKED_REPAIR_MARGIN,
    LOCKED_REPAIR_RHO,
    LOCKED_RESIDUAL_LIPSCHITZ,
    LOCKED_SKEW_NORM_UPPER,
    LOCKED_SYMMETRIC_GAIN_UPPER,
    LOCKED_TAU,
    audit_structure_aware_stability,
    bernstein_coefficients,
    frequency_polynomials,
    locked_structure_aware_certificate,
    locked_structure_aware_parameters,
    polynomial,
    polynomial_add,
    polynomial_compose_affine,
    polynomial_evaluate,
    polynomial_multiply,
    polynomial_scale,
    transfer_denominator,
    transfer_numerator,
)


def test_locked_center_and_residual_bounds_are_exact() -> None:
    assert Fraction(1) == LOCKED_FLOOR
    assert Fraction(1) == LOCKED_HESSIAN_LOWER
    assert Fraction(10) == LOCKED_HESSIAN_UPPER
    assert Fraction(19, 20) == LOCKED_BETA
    assert Fraction(648) == LOCKED_REPAIR_MARGIN
    assert Fraction(41_528_474_059_081, 260_261_360_000) == LOCKED_DEFICIT_UPPER
    assert Fraction(210_177_835_339_081, 260_261_360_000) == LOCKED_REPAIR_RHO
    assert Fraction(336_372_400_608_849, 260_261_360_000) == LOCKED_SYMMETRIC_GAIN_UPPER
    assert Fraction(505_021_761_888_849, 520_522_720_000) == LOCKED_CENTER_GAIN
    assert Fraction(6_444_259, 40_000) == LOCKED_SKEW_NORM_UPPER
    assert Fraction(251_582_619_905_461, 520_522_720_000) == LOCKED_RESIDUAL_LIPSCHITZ
    assert Fraction(1, 32_000) == LOCKED_LEARNING_RATE
    assert Fraction(99_999, 100_000) == LOCKED_TAU


def test_small_polynomial_helpers_replay_exact_identities() -> None:
    generator = random.Random(20260902)
    for _ in range(25):
        left = polynomial([generator.randint(-5, 5) for _ in range(4)])
        right = polynomial([generator.randint(-5, 5) for _ in range(3)])
        point = Fraction(generator.randint(-7, 7), generator.randint(1, 9))
        scale = Fraction(generator.randint(-4, 4), generator.randint(1, 7))
        assert polynomial_evaluate(polynomial_add(left, right), point) == (
            polynomial_evaluate(left, point) + polynomial_evaluate(right, point)
        )
        assert polynomial_evaluate(polynomial_multiply(left, right), point) == (
            polynomial_evaluate(left, point) * polynomial_evaluate(right, point)
        )
        assert polynomial_evaluate(polynomial_scale(scale, left), point) == (
            scale * polynomial_evaluate(left, point)
        )

    candidate = polynomial([3, -2, 5, 1])
    composed = polynomial_compose_affine(candidate, offset=Fraction(2, 7), scale=Fraction(5, 9))
    for point in (Fraction(0), Fraction(1, 3), Fraction(1)):
        assert polynomial_evaluate(composed, point) == polynomial_evaluate(
            candidate, Fraction(2, 7) + Fraction(5, 9) * point
        )


def test_bernstein_conversion_reconstructs_the_polynomial_exactly() -> None:
    power = polynomial([Fraction(-3, 5), Fraction(7, 3), -2, 4])
    lower = Fraction(2, 11)
    upper = Fraction(9, 7)
    coefficients = bernstein_coefficients(
        power,
        interval_lower=lower,
        interval_upper=upper,
        degree=5,
    )
    for u in (Fraction(0), Fraction(1, 7), Fraction(3, 5), Fraction(1)):
        bernstein_value = sum(
            (
                coefficient * math.comb(5, index) * u**index * (1 - u) ** (5 - index)
                for index, coefficient in enumerate(coefficients)
            ),
            Fraction(0),
        )
        assert bernstein_value == polynomial_evaluate(power, lower + (upper - lower) * u)

    with pytest.raises(ValueError, match="positive width"):
        bernstein_coefficients(power, interval_lower=1, interval_upper=1)
    with pytest.raises(ValueError, match="below the polynomial degree"):
        bernstein_coefficients(power, interval_lower=0, interval_upper=1, degree=2)


def test_locked_exact_bernstein_certificate_is_strict() -> None:
    audit = locked_structure_aware_certificate()
    assert audit.certified
    assert audit.jury_certified
    assert audit.frequency_certified
    assert [certificate.name for certificate in audit.jury_certificates] == [
        "D(+tau)",
        "D(-tau)",
        "tau^2-B",
    ]
    assert all(len(certificate.coefficients) == 2 for certificate in audit.jury_certificates)
    assert len(audit.q2_certificate.coefficients) == 2
    assert len(audit.vertex_certificate.coefficients) == 5
    assert audit.vertex_certificate.minimum_coefficient > Fraction(1, 100_000)

    lower = audit.parameters.scaled_curvature_lower
    upper = audit.parameters.scaled_curvature_upper
    assert lower == Fraction(1, 32_000)
    assert upper == Fraction(1, 3_200)
    assert audit.q2_certificate.coefficients[0] == polynomial_evaluate(
        audit.frequency_polynomials.q2, lower
    )
    assert audit.q2_certificate.coefficients[-1] == polynomial_evaluate(
        audit.frequency_polynomials.q2, upper
    )
    assert audit.vertex_certificate.coefficients[0] == polynomial_evaluate(
        audit.frequency_polynomials.vertex, lower
    )
    assert audit.vertex_certificate.coefficients[-1] == polynomial_evaluate(
        audit.frequency_polynomials.vertex, upper
    )


def test_nearby_larger_step_fails_the_exact_vertex_check() -> None:
    parameters = replace(
        locked_structure_aware_parameters(),
        learning_rate=Fraction(33, 1_000_000),
    )
    audit = audit_structure_aware_stability(parameters)
    assert audit.jury_certified
    assert audit.q2_certificate.strictly_positive
    assert audit.vertex_certificate.coefficients[-1] < 0
    assert not audit.frequency_certified
    assert not audit.certified


def test_frequency_polynomial_matches_direct_circle_evaluation() -> None:
    parameters = locked_structure_aware_parameters()
    frequency = frequency_polynomials(parameters)
    generator = random.Random(20260902)
    for _ in range(50):
        interpolation = generator.random()
        x = float(parameters.scaled_curvature_lower) + interpolation * float(
            parameters.scaled_curvature_upper - parameters.scaled_curvature_lower
        )
        omega = generator.uniform(-math.pi, math.pi)
        z = float(parameters.tau) * cmath.exp(1j * omega)
        numerator = transfer_numerator(z, scaled_curvature=Fraction(x), beta=parameters.beta)
        denominator = transfer_denominator(
            z,
            scaled_curvature=Fraction(x),
            beta=parameters.beta,
            center_gain=parameters.center_gain,
        )
        direct = (
            abs(denominator) ** 2 - float(parameters.residual_lipschitz) ** 2 * abs(numerator) ** 2
        )
        t = 1 - math.cos(omega)
        reduced = sum(
            float(polynomial_evaluate(coefficient, Fraction(x))) * t**power
            for power, coefficient in enumerate((frequency.q0, frequency.q1, frequency.q2))
        )
        assert reduced == pytest.approx(direct, rel=2e-11, abs=2e-14)
        assert direct > 0


def test_scaled_jury_conditions_match_direct_denominator_roots() -> None:
    audit = locked_structure_aware_certificate()
    parameters = audit.parameters
    generator = random.Random(20260902)
    for _ in range(50):
        interpolation = Fraction(generator.randrange(10_001), 10_000)
        x = parameters.scaled_curvature_lower + interpolation * (
            parameters.scaled_curvature_upper - parameters.scaled_curvature_lower
        )
        linear = polynomial_evaluate(audit.denominator_linear, x)
        constant = polynomial_evaluate(audit.denominator_constant, x)
        roots = np.roots([1.0, float(linear), float(constant)])
        assert np.max(np.abs(roots)) < float(parameters.tau)
        assert all(
            polynomial_evaluate(certificate.power_coefficients, x) > 0
            for certificate in audit.jury_certificates
        )


def test_locked_step_improves_the_p3_point_without_reaching_local_control() -> None:
    p3 = locked_ema_nesterov_certificate()
    improvement = LOCKED_LEARNING_RATE / p3.learning_rate
    assert improvement == Fraction(336_372_400_608_849, 2_082_090_880_000)
    assert 161 < float(improvement) < 162

    # This is the existing exact linked zero-linearization control, not a new
    # global upper certificate for the structure-aware model.
    local_threshold = Fraction(
        8_120_154_432_000_000_000_000_000,
        3_901_919_808_117_690_731_741_568_607,
    )
    remaining_gap = local_threshold / LOCKED_LEARNING_RATE
    assert 66 < float(remaining_gap) < 67
