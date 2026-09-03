from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import pytest

from passive_muon.pl_convergence import (
    LOCKED_PL_CONVERGENCE_LEARNING_RATE,
    LOCKED_PL_CONVERGENCE_TAU,
    audit_pl_convergence,
    locked_pl_convergence_certificate,
    pl_convergence_lmi_matrix,
    pl_convergence_matrices,
    pl_supply_function_coefficients,
)
from passive_muon.structure_aware_stability import LOCKED_LEARNING_RATE


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


def test_locked_pl_certificate_retains_the_full_p5_step() -> None:
    certificate = locked_pl_convergence_certificate()

    assert LOCKED_PL_CONVERGENCE_LEARNING_RATE == LOCKED_LEARNING_RATE
    assert certificate.learning_rate == Fraction(1, 32_000)
    assert Fraction(19_999, 20_000) == LOCKED_PL_CONVERGENCE_TAU
    assert certificate.storage == (
        (Fraction(72_435, 100_000), Fraction(-3_185, 100_000)),
        (Fraction(-3_185, 100_000), Fraction(499, 100_000)),
    )
    assert certificate.function_storage == Fraction(27_066, 100_000)
    assert certificate.interpolation_reverse_weight == Fraction(20_753, 100_000)
    assert certificate.lambda_residual_norm == Fraction(5_052, 100_000)
    assert (
        sum(certificate.storage[index][index] for index in range(2)) + certificate.function_storage
        == 1
    )


def test_pl_supply_weights_are_positive_and_cancel_values_exactly() -> None:
    certificate = locked_pl_convergence_certificate()

    assert certificate.lambda_interpolation_12 == Fraction(9_563_258_693_533, 20_000_000_000_000)
    assert certificate.lambda_interpolation_21 == Fraction(20_753, 100_000)
    assert certificate.lambda_pl_next == Fraction(541_306_467, 4_000_000_000_000)
    assert certificate.lambda_residual_norm == Fraction(1_263, 25_000)
    assert certificate.lambda_interpolation_12 > 0
    assert certificate.lambda_interpolation_21 > 0
    assert certificate.lambda_pl_next > 0
    assert pl_supply_function_coefficients(certificate) == (
        certificate.function_storage * certificate.tau**2,
        -certificate.function_storage,
    )


def test_lifted_pl_dynamics_use_the_physical_residual_scaling() -> None:
    certificate = locked_pl_convergence_certificate()
    matrices = pl_convergence_matrices(certificate)
    beta = certificate.beta
    alpha = certificate.dimensionless_step

    assert matrices.signal_selector == (beta**2, 1 - beta**2, 0, 0)
    assert matrices.step_selector == (
        -alpha * beta**2,
        -alpha * (1 - beta**2),
        -certificate.learning_rate * certificate.residual_lipschitz * certificate.smoothness,
        0,
    )
    assert matrices.transition[0] == (beta, 1 - beta, 0, 0)
    assert matrices.transition[1] == (0, 0, 0, 1)


def test_directed_matrices_match_nonconvex_smooth_interpolation_signs() -> None:
    certificate = locked_pl_convergence_certificate()
    matrices = pl_convergence_matrices(certificate)
    momentum = Fraction(-1, 5)
    gradient = Fraction(2, 7)
    residual = Fraction(1, 9)
    gradient_next = Fraction(-1, 8)
    vector = (momentum, gradient, residual, gradient_next)
    step = sum(
        (
            coefficient * value
            for coefficient, value in zip(matrices.step_selector, vector, strict=True)
        ),
        Fraction(0),
    )
    points = {
        1: (Fraction(0), gradient),
        2: (step, gradient_next),
    }

    for edge, matrix in {
        (1, 2): matrices.interpolation_12,
        (2, 1): matrices.interpolation_21,
    }.items():
        index_i, index_j = edge
        x_i, u_i = points[index_i]
        x_j, u_j = points[index_j]
        dx = x_i - x_j
        du = u_i - u_j
        expected = -u_j * dx - _nonconvex_interpolation_model(dx, du)
        assert _quadratic_form(matrix, vector) == expected


def test_directed_supplies_accept_negative_curvature_one_smooth_data() -> None:
    certificate = locked_pl_convergence_certificate()
    matrices = pl_convergence_matrices(certificate)
    momentum = Fraction(-1, 5)
    gradient = Fraction(2, 7)
    residual = Fraction(1, 9)
    vector_prefix = (momentum, gradient, residual)
    step = sum(
        (
            coefficient * value
            for coefficient, value in zip(matrices.step_selector[:3], vector_prefix, strict=True)
        ),
        Fraction(0),
    )
    curvature = Fraction(-1, 2)
    gradient_next = gradient + curvature * step
    vector = (*vector_prefix, gradient_next)
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
        _, _, function_i = points[index_i]
        _, _, function_j = points[index_j]
        supply = function_i - function_j + _quadratic_form(matrix, vector)
        assert supply >= 0


def test_locked_pl_lmi_replays_strictly_in_exact_arithmetic() -> None:
    certificate = locked_pl_convergence_certificate()
    audit = audit_pl_convergence(certificate)
    lmi = pl_convergence_lmi_matrix(certificate)

    assert audit.certified
    assert audit.function_values_cancel
    assert audit.storage_leading_minors == (
        Fraction(14_487, 20_000),
        Fraction(650_021, 250_000_000),
    )
    assert all(value > 0 for value in audit.negative_lmi_leading_minors)
    assert str(audit.negative_lmi_leading_minors[-1]) == (
        "106378237649828474982433094173183543070031301965649332843580933465570058399/"
        "2841052730215087734784000000000000000000000000000000000000000000000000000000000000000"
    )
    assert all(lmi[row][column] == lmi[column][row] for row in range(4) for column in range(4))


def test_invalid_pl_certificate_parameters_are_rejected() -> None:
    certificate = locked_pl_convergence_certificate()

    with pytest.raises(ValueError, match="PL constant"):
        replace(certificate, pl_constant=Fraction(11))
    with pytest.raises(ValueError, match="function storage"):
        replace(certificate, function_storage=Fraction(0))
    with pytest.raises(TypeError, match=r"fractions\.Fraction"):
        replace(certificate, learning_rate=1.0 / 32_000)  # type: ignore[arg-type]
