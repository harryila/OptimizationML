from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import pytest

from passive_muon.nonquadratic_convergence import (
    LOCKED_NONQUADRATIC_CONVERGENCE_LEARNING_RATE,
    LOCKED_NONQUADRATIC_CONVERGENCE_TAU,
    audit_nonquadratic_convergence,
    interpolation_function_coefficients,
    locked_nonquadratic_convergence_certificate,
    nonquadratic_convergence_lmi_matrix,
    nonquadratic_convergence_matrices,
)
from passive_muon.structure_aware_stability import LOCKED_LEARNING_RATE, LOCKED_TAU


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


def _interpolation_model(dx: Fraction, du: Fraction, condition_ratio: Fraction) -> Fraction:
    return (du**2 - 2 * condition_ratio * dx * du + condition_ratio * dx**2) / (
        2 * (1 - condition_ratio)
    )


def test_locked_certificate_uses_the_exact_full_p4_step() -> None:
    certificate = locked_nonquadratic_convergence_certificate()

    assert LOCKED_NONQUADRATIC_CONVERGENCE_LEARNING_RATE == LOCKED_LEARNING_RATE
    assert LOCKED_NONQUADRATIC_CONVERGENCE_TAU < LOCKED_TAU
    assert certificate.learning_rate == Fraction(1, 32_000)
    assert certificate.tau == Fraction(2_499, 2_500)
    assert certificate.storage == (
        (Fraction(495_723, 100_000_000), Fraction(-3_085_119, 100_000_000)),
        (Fraction(-3_085_119, 100_000_000), Fraction(72_422_647, 100_000_000)),
    )
    assert certificate.function_storage == Fraction(27_081_630, 100_000_000)
    assert certificate.interpolation_cycle_weight == Fraction(270_356, 100_000_000)
    assert certificate.interpolation_reverse_weight == Fraction(48_120, 100_000_000)
    assert certificate.lambda_residual_norm == Fraction(5_066_256, 100_000_000)
    assert (
        sum(certificate.storage[index][index] for index in range(2))
        + (certificate.function_storage)
        == 1
    )


def test_directed_weights_are_positive_and_cancel_function_values_exactly() -> None:
    certificate = locked_nonquadratic_convergence_certificate()

    assert certificate.interpolation_multipliers == {
        (0, 1): Fraction(67_589, 25_000_000),
        (1, 2): Fraction(17_111_528_143_163, 62_500_000_000_000),
        (2, 0): Fraction(155_434_393_163, 62_500_000_000_000),
        (2, 1): Fraction(1_203, 2_500_000),
    }
    assert all(value > 0 for value in certificate.interpolation_multipliers.values())
    assert interpolation_function_coefficients(certificate) == (
        certificate.function_storage * certificate.tau**2,
        -certificate.function_storage,
    )


def test_lifted_dynamics_have_the_physical_residual_scaling() -> None:
    certificate = locked_nonquadratic_convergence_certificate()
    matrices = nonquadratic_convergence_matrices(certificate)
    beta = certificate.beta
    alpha = certificate.learning_rate * certificate.center_gain * certificate.smoothness

    assert matrices.signal_selector == (
        0,
        beta**2,
        1 - beta**2,
        0,
        0,
    )
    assert matrices.transition[0] == (
        1,
        -alpha * beta**2,
        -alpha * (1 - beta**2),
        -certificate.learning_rate * certificate.residual_lipschitz * certificate.smoothness,
        0,
    )
    assert matrices.transition[1] == (0, beta, 1 - beta, 0, 0)


def test_interpolation_quadratics_match_exact_fixed_objective_inequalities() -> None:
    certificate = locked_nonquadratic_convergence_certificate()
    matrices = nonquadratic_convergence_matrices(certificate)
    curvature = Fraction(3, 5)
    w = Fraction(2, 3)
    momentum = Fraction(-1, 4)
    gradient = curvature * w
    residual = Fraction(0)
    signal = certificate.beta**2 * momentum + (1 - certificate.beta**2) * gradient
    w_next = (
        w
        - certificate.dimensionless_step * signal
        - certificate.dimensionless_step * certificate.residual_ratio * residual
    )
    gradient_next = curvature * w_next
    vector = (w, momentum, gradient, residual, gradient_next)
    points = {
        0: (Fraction(0), Fraction(0), Fraction(0)),
        1: (w, gradient, curvature * w**2 / 2),
        2: (w_next, gradient_next, curvature * w_next**2 / 2),
    }
    interpolation_matrices = {
        (0, 1): matrices.interpolation_01,
        (1, 2): matrices.interpolation_12,
        (2, 0): matrices.interpolation_20,
        (2, 1): matrices.interpolation_21,
    }

    for edge, matrix in interpolation_matrices.items():
        index_i, index_j = edge
        x_i, u_i, function_i = points[index_i]
        x_j, u_j, function_j = points[index_j]
        dx = x_i - x_j
        du = u_i - u_j
        expected_quadratic = -u_j * dx - _interpolation_model(dx, du, certificate.condition_ratio)
        assert _quadratic_form(matrix, vector) == expected_quadratic
        assert function_i - function_j + expected_quadratic >= 0


def test_locked_convergence_lmi_replays_strictly_in_exact_arithmetic() -> None:
    certificate = locked_nonquadratic_convergence_certificate()
    audit = audit_nonquadratic_convergence(certificate)
    lmi = nonquadratic_convergence_lmi_matrix(certificate)

    assert audit.certified
    assert audit.function_values_cancel
    assert audit.storage_leading_minors == (
        Fraction(495_723, 100_000_000),
        Fraction(1_319_180_629_731, 500_000_000_000_000),
    )
    assert all(value > 0 for value in audit.negative_lmi_leading_minors)
    assert str(audit.negative_lmi_leading_minors[-1]) == (
        "374354014686291560693271770756760938687166003695751463491130744858102650181826644319907/"
        "423349846931560000000000000000000000000000000000000000000000000000000000000000000000"
        "000000000000000000"
    )
    assert all(lmi[row][column] == lmi[column][row] for row in range(5) for column in range(5))


def test_invalid_interpolation_flow_is_rejected() -> None:
    certificate = locked_nonquadratic_convergence_certificate()
    with pytest.raises(ValueError, match="lambda_20"):
        replace(certificate, interpolation_cycle_weight=Fraction(0))

    with pytest.raises(TypeError, match=r"fractions\.Fraction"):
        replace(certificate, learning_rate=1.0 / 32_000)  # type: ignore[arg-type]
