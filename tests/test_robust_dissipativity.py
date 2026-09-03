from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import pytest

from passive_muon.pl_convergence import pl_convergence_lmi_matrix
from passive_muon.robust_dissipativity import (
    LOCKED_GRADIENT_NOISE_GAIN,
    LOCKED_IMPLEMENTATION_ERROR_GAIN,
    audit_robust_dissipativity,
    locked_robust_dissipativity_certificate,
    robust_dissipativity_lmi_matrix,
    robust_dissipativity_matrices,
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


def _nonconvex_interpolation_model(dx: Fraction, du: Fraction) -> Fraction:
    return (du**2 + 2 * dx * du - dx**2) / 4


def test_locked_robust_gains_and_physical_normalization_are_exact() -> None:
    certificate = locked_robust_dissipativity_certificate()
    base = certificate.pl_certificate

    assert Fraction(1, 2) == LOCKED_GRADIENT_NOISE_GAIN
    assert Fraction(1, 2_000_000) == LOCKED_IMPLEMENTATION_ERROR_GAIN
    assert certificate.gradient_noise_gain == Fraction(1, 2)
    assert certificate.implementation_error_gain == Fraction(1, 2_000_000)
    assert certificate.normalized_gradient_noise_penalty == Fraction(50)
    assert certificate.normalized_implementation_error_penalty == (
        base.center_gain**2 * base.smoothness**2 / 2_000_000
    )
    assert certificate.rate == Fraction(399_960_001, 400_000_000)
    assert base.learning_rate == Fraction(1, 32_000)
    one_minus_rate = 1 - certificate.rate
    assert one_minus_rate == Fraction(39_999, 400_000_000)
    assert certificate.gradient_noise_gain / one_minus_rate == Fraction(200_000_000, 39_999)
    assert certificate.implementation_error_gain / one_minus_rate == Fraction(200, 39_999)


def test_lift_reuses_one_noisy_gradient_in_ema_and_nesterov() -> None:
    certificate = locked_robust_dissipativity_certificate()
    base = certificate.pl_certificate
    matrices = robust_dissipativity_matrices(certificate)
    beta = base.beta
    alpha = base.dimensionless_step

    assert matrices.noisy_gradient_selector == (0, 1, 0, 0, 1, 0)
    assert matrices.transition[0] == (beta, 1 - beta, 0, 0, 1 - beta, 0)
    assert matrices.transition[1] == (0, 0, 0, 1, 0, 0)
    assert matrices.signal_selector == (
        beta**2,
        1 - beta**2,
        0,
        0,
        1 - beta**2,
        0,
    )
    assert matrices.step_selector == (
        -alpha * beta**2,
        -alpha * (1 - beta**2),
        -alpha * base.residual_ratio,
        0,
        -alpha * (1 - beta**2),
        -alpha,
    )


def test_robust_lmi_restricts_exactly_to_the_p6_lmi_without_inputs() -> None:
    certificate = locked_robust_dissipativity_certificate()
    robust_lmi = robust_dissipativity_lmi_matrix(certificate)
    p6_lmi = pl_convergence_lmi_matrix(certificate.pl_certificate)

    assert tuple(tuple(row[column] for column in range(4)) for row in robust_lmi[:4]) == p6_lmi


def test_disturbed_interpolation_matrices_have_the_directed_nonconvex_signs() -> None:
    certificate = locked_robust_dissipativity_certificate()
    matrices = robust_dissipativity_matrices(certificate)
    momentum = Fraction(-1, 5)
    gradient = Fraction(2, 7)
    residual = Fraction(1, 9)
    gradient_noise = Fraction(-1, 11)
    implementation_error = Fraction(1, 13)
    prefix = (momentum, gradient, residual, Fraction(0), gradient_noise, implementation_error)
    step = sum(
        (
            coefficient * value
            for coefficient, value in zip(matrices.step_selector, prefix, strict=True)
        ),
        Fraction(0),
    )
    curvature = Fraction(-1, 2)
    gradient_next = gradient + curvature * step
    vector = (momentum, gradient, residual, gradient_next, gradient_noise, implementation_error)
    function_current = Fraction(0)
    function_next = gradient * step + curvature * step**2 / 2
    points = {
        1: (Fraction(0), gradient, function_current),
        2: (step, gradient_next, function_next),
    }

    for edge, matrix in {
        (1, 2): matrices.interpolation_12,
        (2, 1): matrices.interpolation_21,
    }.items():
        index_i, index_j = edge
        x_i, u_i, function_i = points[index_i]
        x_j, u_j, function_j = points[index_j]
        dx = x_i - x_j
        du = u_i - u_j
        expected_quadratic = -u_j * dx - _nonconvex_interpolation_model(dx, du)
        assert _quadratic_form(matrix, vector) == expected_quadratic
        assert function_i - function_j + expected_quadratic >= 0


def test_locked_robust_lmi_replays_strictly_in_exact_arithmetic() -> None:
    audit = audit_robust_dissipativity()

    assert audit.certified
    assert audit.function_values_cancel
    assert audit.storage_leading_minors == (
        Fraction(14_487, 20_000),
        Fraction(650_021, 250_000_000),
    )
    assert len(audit.negative_lmi_leading_minors) == 6
    assert all(value > 0 for value in audit.negative_lmi_leading_minors)
    assert audit.negative_lmi_leading_minors[0] == Fraction(
        588366669441078114305592272723841308118612219507,
        35513159127688596684800000000000000000000000000000,
    )
    assert str(audit.negative_lmi_leading_minors[-1]) == (
        "251577500889200315419897798837151777987840652612652753050936"
        "77552588813246840551502751954845110066654533994497393/"
        "307906365046028293313654367610722058240000000000000000000000"
        "0000000000000000000000000000000000000000000000000000000000000"
    )
    assert all(
        audit.lmi[row][column] == audit.lmi[column][row] for row in range(6) for column in range(6)
    )


def test_invalid_robust_gains_are_rejected() -> None:
    certificate = locked_robust_dissipativity_certificate()

    with pytest.raises(ValueError, match="positive"):
        replace(certificate, gradient_noise_gain=Fraction(0))
    with pytest.raises(ValueError, match="positive"):
        replace(certificate, implementation_error_gain=Fraction(-1))
    with pytest.raises(TypeError, match=r"fractions\.Fraction"):
        replace(certificate, gradient_noise_gain=0.5)  # type: ignore[arg-type]
