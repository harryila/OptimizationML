from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from passive_muon.equivariant_resolvent_solver import (
    LOCKED_FP64_ABSOLUTE_RESIDUAL,
    EquivariantResolventSolverConfig,
    SolverStatus,
    _reduced_jacobian_diagonal_plus_rank_one,
    _sherman_morrison_solve,
    jordan_response_and_derivative_fp64,
    p13_operator_fp64,
    p13_singular_operator_fp64,
    radial_deficit_majorant_fp64,
    radial_integral_fp64,
    resolvent_graph_singular_values_fp64,
    shifted_p13_singular_operator_fp64,
    solve_resolvent_fp64,
    solve_resolvent_singular_values,
)
from passive_muon.radial_passivation_tradeoff import (
    SWITCH_RAW_RATIO,
    UNIT_DEFICIT_UPPER,
)


def _orthogonal(rows: int, columns: int, rng: np.random.Generator) -> np.ndarray:
    matrix, _ = np.linalg.qr(rng.normal(size=(rows, columns)))
    return matrix


def _direct_jordan_matrix(matrix: np.ndarray) -> np.ndarray:
    a = 6889.0 / 2000.0
    b = -191.0 / 40.0
    c = 4063.0 / 2000.0
    output = matrix.copy()
    for _ in range(5):
        transposed = output.shape[0] > output.shape[1]
        work = output.T if transposed else output
        gram = work @ work.T
        output = a * work + (b * gram + c * (gram @ gram)) @ work
        if transposed:
            output = output.T
    return output


def _direct_p13_matrix(matrix: np.ndarray, epsilon: float) -> np.ndarray:
    radius = np.linalg.norm(matrix, ord="fro")
    if radius == 0.0:
        return np.zeros_like(matrix)
    return (
        _direct_jordan_matrix(matrix / (radius + epsilon))
        + (radial_integral_fp64(radius / epsilon) / radius) * matrix
    )


def test_locked_config_records_the_p15_rule_and_fp64_budget() -> None:
    config = EquivariantResolventSolverConfig()

    assert config.epsilon == 1e-7
    assert config.resolvent_parameter == 1e-3
    assert config.shunt == 1_000.0
    assert config.p15_relative_tolerance == 1 / 250
    assert config.p15_absolute_tolerance == LOCKED_FP64_ABSOLUTE_RESIDUAL == 2**-40
    assert config.strict_relative_tolerance == 2**-40
    assert config.strict_absolute_tolerance == 2**-48
    assert config.maximum_iterations == 64
    assert config.maximum_backtracks == 64
    assert config.signal_norm_guard == pytest.approx(25.2335, rel=1e-6)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("epsilon", 0.0),
        ("resolvent_parameter", math.inf),
        ("shunt", -1.0),
        ("p15_relative_tolerance", -1.0),
        ("strict_relative_tolerance", 0.01),
        ("strict_absolute_tolerance", 1.0),
        ("maximum_iterations", -1),
        ("maximum_backtracks", -1),
        ("armijo_constant", 0.5),
        ("minimum_step", 2.0),
        ("negative_solution_tolerance", 0.0),
    ],
)
def test_invalid_solver_configuration_is_rejected(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        replace(EquivariantResolventSolverConfig(), **{field: value})


def test_five_stage_response_derivative_matches_centered_finite_difference() -> None:
    values = np.array([0.0, 0.01, 0.2, 0.6, 0.8, 1.0], dtype=np.float64)
    response, derivative = jordan_response_and_derivative_fp64(values)
    step = 2.0**-24
    forward, _ = jordan_response_and_derivative_fp64(values + step)
    backward, _ = jordan_response_and_derivative_fp64(values - step)

    assert response[0] == 0.0
    assert derivative[0] == pytest.approx((6889 / 2000) ** 5, rel=2e-15)
    np.testing.assert_allclose(derivative, (forward - backward) / (2 * step), rtol=3e-6)


def test_radial_formula_is_continuous_at_the_exact_switch() -> None:
    switch = float(SWITCH_RAW_RATIO)
    unit_upper = float(UNIT_DEFICIT_UPPER)

    assert radial_integral_fp64(switch) == pytest.approx(unit_upper * switch, rel=1e-15)
    assert radial_deficit_majorant_fp64(switch) == pytest.approx(unit_upper, rel=1e-15)
    assert radial_integral_fp64(math.nextafter(switch, math.inf)) == pytest.approx(
        unit_upper * switch,
        rel=2e-15,
    )
    assert radial_deficit_majorant_fp64(math.nextafter(switch, math.inf)) == pytest.approx(
        unit_upper,
        rel=2e-15,
    )


def test_p13_spectral_evaluation_matches_direct_matrix_polynomial() -> None:
    rng = np.random.default_rng(314159)
    for shape in ((2, 2), (3, 7), (7, 3), (6, 6)):
        matrix = rng.normal(size=shape).astype(np.float64)
        matrix *= 3.0 / np.linalg.norm(matrix, ord="fro")
        expected = _direct_p13_matrix(matrix, 1e-7)
        actual = p13_operator_fp64(matrix)
        np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=5e-13)


def test_reduced_jacobian_is_diagonal_plus_rank_one_and_matches_finite_difference() -> None:
    rng = np.random.default_rng(271828)
    config = EquivariantResolventSolverConfig()
    values = rng.uniform(0.01, 1.0, size=7).astype(np.float64)
    values *= 0.02 / np.linalg.norm(values)
    diagonal, left, right = _reduced_jacobian_diagonal_plus_rank_one(values, config)
    jacobian = np.diag(diagonal) + np.outer(left, right)
    direction = rng.normal(size=7).astype(np.float64)
    step = 2.0**-28
    forward = resolvent_graph_singular_values_fp64(values + step * direction, config=config)
    backward = resolvent_graph_singular_values_fp64(values - step * direction, config=config)

    np.testing.assert_allclose(
        jacobian @ direction,
        (forward - backward) / (2 * step),
        rtol=2e-7,
        atol=2e-7,
    )


def test_inner_linear_repair_and_jacobian_remain_finite_for_subnormal_input() -> None:
    config = EquivariantResolventSolverConfig()
    values = np.array([np.nextafter(0.0, 1.0)], dtype=np.float64)

    output = p13_singular_operator_fp64(values, epsilon=config.epsilon)
    diagonal, left, right = _reduced_jacobian_diagonal_plus_rank_one(values, config)

    assert np.isfinite(output).all()
    assert np.isfinite(diagonal).all()
    assert np.isfinite(left).all()
    assert np.isfinite(right).all()
    assert output[0] > 0.0


def test_signed_spectral_coordinates_are_odd_and_have_a_finite_jacobian() -> None:
    config = EquivariantResolventSolverConfig()
    values = np.array([-0.03, 0.01, -0.002], dtype=np.float64)

    positive = p13_singular_operator_fp64(values, epsilon=config.epsilon)
    negative = p13_singular_operator_fp64(-values, epsilon=config.epsilon)
    diagonal, left, right = _reduced_jacobian_diagonal_plus_rank_one(values, config)

    np.testing.assert_array_equal(negative, -positive)
    assert np.isfinite(diagonal).all()
    assert np.isfinite(left).all()
    assert np.isfinite(right).all()


def test_zero_input_is_exact_and_skips_svd() -> None:
    signal = np.zeros((3, 7), dtype=np.float64)
    result = solve_resolvent_fp64(signal)

    assert result.diagnostics.status is SolverStatus.ZERO_INPUT
    assert result.diagnostics.certified
    assert result.diagnostics.svd_count == 0
    assert result.diagnostics.reconstruction_matmuls == 0
    assert np.array_equal(result.approximate_solution, signal)
    assert np.array_equal(result.output, signal)
    assert np.array_equal(result.graph_residual, signal)


def test_unequal_singular_values_pass_strict_and_actual_p15_residual_checks() -> None:
    singular_values = np.array([3.0, 4.0], dtype=np.float64)
    result = solve_resolvent_singular_values(singular_values)
    config = EquivariantResolventSolverConfig()
    shifted = shifted_p13_singular_operator_fp64(
        result.solution_singular_values,
        epsilon=config.epsilon,
        shunt=config.shunt,
    )

    assert result.diagnostics.status is SolverStatus.STRICT_RESIDUAL
    assert result.diagnostics.certified
    assert result.diagnostics.iterations <= 16
    assert result.diagnostics.sherman_morrison_solves == result.diagnostics.iterations
    assert result.diagnostics.minimum_abs_sherman_morrison_denominator > 0.05
    assert result.diagnostics.residual_norm <= result.diagnostics.strict_target
    np.testing.assert_allclose(
        result.graph_residual,
        singular_values - result.solution_singular_values - config.resolvent_parameter * shifted,
        rtol=1e-13,
        atol=1e-15,
    )
    np.testing.assert_array_equal(
        result.output_singular_values,
        (singular_values - result.solution_singular_values) / config.resolvent_parameter,
    )
    assert result.output_singular_values[0] / 3.0 != result.output_singular_values[1] / 4.0


def test_rectangular_matrix_recomputes_actual_residual_and_required_output() -> None:
    rng = np.random.default_rng(12345)
    signal = rng.normal(size=(7, 3)).astype(np.float64)
    signal *= 5.0 / np.linalg.norm(signal, ord="fro")
    result = solve_resolvent_fp64(signal)
    config = EquivariantResolventSolverConfig()

    assert result.diagnostics.status is SolverStatus.STRICT_RESIDUAL
    assert result.diagnostics.certified
    assert result.diagnostics.svd_count == 2
    assert result.diagnostics.reconstruction_matmuls == 2
    assert result.diagnostics.residual_norm <= result.diagnostics.p15_threshold
    direct_shifted = (
        p13_operator_fp64(result.approximate_solution, epsilon=config.epsilon)
        + config.shunt * result.approximate_solution
    )
    np.testing.assert_array_equal(
        result.graph_residual,
        signal - result.approximate_solution - config.resolvent_parameter * direct_shifted,
    )
    np.testing.assert_array_equal(
        result.output,
        (signal - result.approximate_solution) / config.resolvent_parameter,
    )
    assert np.linalg.norm(result.graph_residual, ord="fro") == pytest.approx(
        result.diagnostics.residual_norm,
        rel=2e-15,
    )


def test_repeated_and_zero_singular_values_are_preserved() -> None:
    rng = np.random.default_rng(20260904)
    left = _orthogonal(6, 4, rng)
    right = _orthogonal(5, 4, rng)
    singular_values = np.array([4.0, 4.0, 0.0, 0.0], dtype=np.float64)
    signal = (left * singular_values) @ right.T
    result = solve_resolvent_fp64(signal)

    assert result.diagnostics.certified
    assert result.solution_singular_values[0] == pytest.approx(
        result.solution_singular_values[1],
        rel=2e-15,
    )
    assert abs(result.solution_singular_values[2]) < 1e-15
    assert abs(result.solution_singular_values[3]) < 1e-15
    assert np.linalg.matrix_rank(result.approximate_solution, tol=1e-12) == 2


def test_solver_is_numerically_biorthogonally_equivariant() -> None:
    rng = np.random.default_rng(424242)
    signal = rng.normal(size=(4, 3)).astype(np.float64)
    signal *= 2.0 / np.linalg.norm(signal, ord="fro")
    left = _orthogonal(4, 4, rng)
    right = _orthogonal(3, 3, rng)
    base = solve_resolvent_fp64(signal)
    transformed = solve_resolvent_fp64(left @ signal @ right.T)

    assert base.diagnostics.certified and transformed.diagnostics.certified
    np.testing.assert_allclose(
        transformed.approximate_solution,
        left @ base.approximate_solution @ right.T,
        rtol=5e-13,
        atol=5e-13,
    )
    np.testing.assert_allclose(
        transformed.output,
        left @ base.output @ right.T,
        rtol=5e-13,
        atol=5e-13,
    )


def test_switch_radius_rank_one_and_repeated_hard_cases_pass() -> None:
    config = replace(
        EquivariantResolventSolverConfig(),
        strict_relative_tolerance=1e-14,
        strict_absolute_tolerance=0.0,
    )
    solution_radius = config.epsilon * float(SWITCH_RAW_RATIO)
    for multiplier in (math.nextafter(1.0, 0.0), 1.0, math.nextafter(1.0, math.inf)):
        for rank in (1, 2, 16):
            solution = np.full(rank, solution_radius * multiplier / math.sqrt(rank))
            signal = resolvent_graph_singular_values_fp64(solution, config=config)
            result = solve_resolvent_singular_values(signal, config)
            assert result.diagnostics.certified
            assert result.diagnostics.iterations <= 16
            np.testing.assert_allclose(
                result.solution_singular_values,
                solution,
                rtol=3e-12,
                atol=1e-24,
            )


def test_seeded_rank_condition_and_scale_sweep_has_no_candidate_failures() -> None:
    rng = np.random.default_rng(7319)
    config = replace(
        EquivariantResolventSolverConfig(),
        strict_relative_tolerance=1e-12,
        strict_absolute_tolerance=1e-15,
    )
    maximum_iterations = 0
    maximum_backtracks = 0
    minimum_denominator = math.inf
    for index in range(600):
        rank = (1, 2, 4, 16, 64)[index % 5]
        norm = 10 ** rng.uniform(-16, math.log10(25.2335))
        family = index % 4
        if family == 0:
            raw = np.ones(rank)
        elif family == 1:
            raw = np.zeros(rank)
            raw[0] = 1.0
        elif family == 2:
            raw = 10 ** rng.uniform(-16, 0, size=rank)
        else:
            raw = rng.uniform(0.01, 1.0, size=rank)
        singular_values = np.asarray(raw / np.linalg.norm(raw) * norm, dtype=np.float64)
        result = solve_resolvent_singular_values(singular_values, config)
        assert result.diagnostics.certified
        maximum_iterations = max(maximum_iterations, result.diagnostics.iterations)
        maximum_backtracks = max(maximum_backtracks, result.diagnostics.backtracks)
        minimum_denominator = min(
            minimum_denominator,
            result.diagnostics.minimum_abs_sherman_morrison_denominator,
        )

    assert maximum_iterations <= 16
    assert maximum_backtracks >= 1
    assert maximum_backtracks <= 24
    assert minimum_denominator > 0.05


def test_p15_only_success_remains_distinct_from_graph_form_output() -> None:
    config = replace(EquivariantResolventSolverConfig(), maximum_iterations=0)
    singular_values = np.array([6e-5, 8e-5], dtype=np.float64)
    result = solve_resolvent_singular_values(singular_values, config)
    graph_form_output = shifted_p13_singular_operator_fp64(
        result.solution_singular_values,
        epsilon=config.epsilon,
        shunt=config.shunt,
    )

    assert result.diagnostics.status is SolverStatus.P15_RESIDUAL_ONLY
    assert result.diagnostics.certified
    assert result.diagnostics.residual_norm > result.diagnostics.strict_target
    assert np.linalg.norm(result.output_singular_values - graph_form_output) > 1e-5
    np.testing.assert_allclose(
        result.output_singular_values - graph_form_output,
        result.graph_residual / config.resolvent_parameter,
        rtol=2e-13,
        atol=2e-13,
    )


def test_maximum_iteration_and_unsafe_jacobian_failures_are_explicit() -> None:
    singular_values = np.array([0.6, 0.8], dtype=np.float64)
    no_iterations = solve_resolvent_singular_values(
        singular_values,
        replace(EquivariantResolventSolverConfig(), maximum_iterations=0),
    )
    unsafe = solve_resolvent_singular_values(
        singular_values,
        replace(
            EquivariantResolventSolverConfig(),
            minimum_sherman_morrison_denominator=2.0,
        ),
    )

    assert no_iterations.diagnostics.status is SolverStatus.MAXIMUM_ITERATIONS
    assert not no_iterations.diagnostics.certified
    assert no_iterations.output_singular_values is None
    assert unsafe.diagnostics.status is SolverStatus.UNSAFE_JACOBIAN
    assert not unsafe.diagnostics.certified
    assert unsafe.output_singular_values is None


def test_computed_negative_sherman_morrison_denominator_fails_closed() -> None:
    with pytest.raises(ArithmeticError, match="Sherman"):
        _sherman_morrison_solve(
            np.ones(1, dtype=np.float64),
            np.array([-2.0], dtype=np.float64),
            np.ones(1, dtype=np.float64),
            np.ones(1, dtype=np.float64),
            minimum_denominator=0.5,
        )


def test_invalid_inputs_and_signal_guard_are_rejected() -> None:
    with pytest.raises(TypeError):
        solve_resolvent_fp64(np.ones((2, 2), dtype=np.float32))
    with pytest.raises(ValueError):
        solve_resolvent_fp64(np.array([[math.nan]], dtype=np.float64))
    with pytest.raises(ValueError):
        solve_resolvent_singular_values(np.array([-1.0], dtype=np.float64))
    with pytest.raises(ValueError, match="P11"):
        solve_resolvent_fp64(np.array([[26.0]], dtype=np.float64))


def test_public_singular_operator_helpers_reject_nonfinite_ratios() -> None:
    with pytest.raises(ValueError):
        radial_integral_fp64(math.inf)
    with pytest.raises(ValueError):
        radial_deficit_majorant_fp64(-1.0)
    zeros = np.zeros(3, dtype=np.float64)
    assert np.array_equal(p13_singular_operator_fp64(zeros, epsilon=1e-7), zeros)
