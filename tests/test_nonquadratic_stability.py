from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import pytest

from passive_muon.nonquadratic_stability import (
    LOCKED_CENTERED_RESIDUAL_RADIUS,
    LOCKED_NONQUADRATIC_LEARNING_RATE,
    LOCKED_NONQUADRATIC_TAU,
    audit_nonquadratic_certificate,
    locked_nonquadratic_certificate,
    nonquadratic_iqc_matrices,
    nonquadratic_lmi_matrix,
)
from passive_muon.structure_aware_stability import (
    LOCKED_CENTER_GAIN,
    LOCKED_RESIDUAL_LIPSCHITZ,
)


def _quadratic_form(
    matrix: tuple[tuple[Fraction, ...], ...], vector: tuple[Fraction, ...]
) -> Fraction:
    return sum(
        (
            vector[row] * matrix[row][column] * vector[column]
            for row in range(len(vector))
            for column in range(len(vector))
        ),
        Fraction(0),
    )


def test_locked_nonquadratic_certificate_replays_strictly_exactly() -> None:
    certificate = locked_nonquadratic_certificate()
    audit = audit_nonquadratic_certificate(certificate)

    assert certificate.learning_rate == Fraction(1, 640_000)
    assert certificate.tau == Fraction(99_999, 100_000)
    assert certificate.condition_ratio == Fraction(1, 10)
    assert certificate.center_gain == LOCKED_CENTER_GAIN
    assert certificate.residual_lipschitz == LOCKED_RESIDUAL_LIPSCHITZ
    assert certificate.centered_residual_radius == LOCKED_CENTERED_RESIDUAL_RADIUS
    assert certificate.lambda_residual_lower == 0
    assert certificate.lambda_residual_upper == 0
    assert audit.certified
    assert audit.storage_positive_definite
    assert audit.lmi_negative_definite
    assert all(value > 0 for value in audit.storage_leading_minors)
    assert all(value > 0 for value in audit.negative_lmi_leading_minors)
    assert float(audit.negative_lmi_leading_minors[-1]) > 3.4e-13


def test_locked_storage_and_multipliers_are_the_committed_rationals() -> None:
    certificate = locked_nonquadratic_certificate()
    assert certificate.storage == (
        (Fraction(6_682_146_216, 10_000_000_000), Fraction(-1_788_501_566, 10_000_000_000)),
        (Fraction(-1_788_501_566, 10_000_000_000), Fraction(3_317_853_784, 10_000_000_000)),
    )
    assert certificate.lambda_gradient == Fraction(194_722_270, 10_000_000_000)
    assert certificate.lambda_residual_norm == Fraction(136_225_077, 10_000_000_000)
    assert Fraction(1, 20) * Fraction(1, 32_000) == LOCKED_NONQUADRATIC_LEARNING_RATE
    assert Fraction(99_999, 100_000) == LOCKED_NONQUADRATIC_TAU


def test_iqc_matrices_encode_the_claimed_scalar_inequalities() -> None:
    certificate = locked_nonquadratic_certificate()
    iqcs = nonquadratic_iqc_matrices(certificate)
    beta = certificate.beta
    sigma = certificate.centered_residual_ratio

    for w, gradient in (
        (Fraction(3, 2), Fraction(3, 20)),
        (Fraction(-5, 3), Fraction(-5, 3)),
        (Fraction(7, 5), Fraction(77, 100)),
    ):
        momentum = Fraction(-2, 7)
        q = beta**2 * momentum + (1 - beta**2) * gradient
        residual = sigma * q
        vector = (w, momentum, gradient, residual)
        expected_gradient = (gradient - certificate.condition_ratio * w) * (w - gradient)
        assert _quadratic_form(iqcs["gradient_interpolation"], vector) == expected_gradient
        assert _quadratic_form(iqcs["residual_lipschitz"], vector) == q**2 - residual**2
        assert _quadratic_form(iqcs["residual_lower_inner_product"], vector) == (
            sigma * q**2 + q * residual
        )
        assert _quadratic_form(iqcs["residual_upper_inner_product"], vector) == (
            sigma * q**2 - q * residual
        )


def test_lmi_is_symmetric_and_uses_tau_squared() -> None:
    certificate = locked_nonquadratic_certificate()
    lmi = nonquadratic_lmi_matrix(certificate)
    assert len(lmi) == 4
    assert all(len(row) == 4 for row in lmi)
    assert all(lmi[row][column] == lmi[column][row] for row in range(4) for column in range(4))

    slower_rate = replace(certificate, tau=Fraction(999_999, 1_000_000))
    slower_lmi = nonquadratic_lmi_matrix(slower_rate)
    expected_shift = slower_rate.tau**2 - certificate.tau**2
    for row in range(2):
        for column in range(2):
            assert slower_lmi[row][column] == (
                lmi[row][column] - expected_shift * certificate.storage[row][column]
            )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("strong_convexity", Fraction(0), "objective bounds"),
        ("beta", Fraction(1), "beta"),
        ("learning_rate", Fraction(0), "learning rate"),
        ("tau", Fraction(1), "tau"),
        ("lambda_gradient", Fraction(-1), "multipliers"),
    ],
)
def test_invalid_certificate_parameters_are_rejected(
    field: str, value: Fraction, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(locked_nonquadratic_certificate(), **{field: value})


def test_inexact_custom_certificate_scalars_are_rejected() -> None:
    with pytest.raises(TypeError, match=r"fractions\.Fraction"):
        replace(locked_nonquadratic_certificate(), learning_rate=1.0e-6)  # type: ignore[arg-type]
