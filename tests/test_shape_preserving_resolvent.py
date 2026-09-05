from __future__ import annotations

import math
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from passive_muon.equivariant_resolvent_solver import (
    EquivariantResolventSolverConfig,
    SolverStatus,
)
from passive_muon.shape_preserving_resolvent import (
    FULL_STEP_DESIGN,
    LOCKED_FULL_STEP_PL_CERTIFICATE,
    LOCKED_GATE_Q0,
    LOCKED_GATE_Q1,
    LOCKED_HIGH_FIDELITY_PL_CERTIFICATE,
    LOCKED_JORDAN_AFTER_RESOLVENT_GAIN_UPPER,
    PRIMARY_DESIGN,
    GateDesign,
    additive_epsilon_jordan_fp64,
    audit_shape_preserving_resolvent,
    evaluate_gated_singular_values_fp64,
    evaluate_shape_preserving_resolvent_fp64,
    jordan_after_resolvent_gain_upper,
    locked_pl_certificate,
    pointwise_gain_sector,
    quintic_gate_jet_exact,
    quintic_gate_weight_fp64,
    smootherstep_gate,
)
from passive_muon.shape_preserving_resolvent_certificate import (
    RAW_SHAPE_GAIN_UPPER,
)
from passive_muon.shape_preserving_resolvent_certificate import (
    pointwise_sector as independently_reconstructed_sector,
)


@pytest.mark.parametrize("design", [PRIMARY_DESIGN, FULL_STEP_DESIGN])
def test_locked_designs_replay_the_exact_pointwise_pl_certificate(design: GateDesign) -> None:
    audit = audit_shape_preserving_resolvent(design)
    sector = pointwise_gain_sector(design)
    independent = independently_reconstructed_sector(design)

    assert audit.certified
    assert audit.all_checks_pass
    assert audit.pl_audit.certified
    assert sector.lower == independent.lower_gain
    assert sector.upper == independent.upper_gain
    assert sector.center == independent.center
    assert sector.radius == independent.radius
    assert sector.jordan_after_resolvent_upper == RAW_SHAPE_GAIN_UPPER
    assert 0 < sector.lower < sector.upper
    assert audit.pl_certificate.learning_rate == design.learning_rate


def test_locked_design_constants_and_rates_are_exact() -> None:
    assert PRIMARY_DESIGN.cap == Fraction(3, 4)
    assert PRIMARY_DESIGN.passive_divisor == 4_096
    assert PRIMARY_DESIGN.learning_rate == Fraction(1, 128_000)
    assert FULL_STEP_DESIGN.cap == Fraction(1, 8)
    assert FULL_STEP_DESIGN.passive_divisor == 8_192
    assert FULL_STEP_DESIGN.learning_rate == Fraction(1, 32_000)
    assert locked_pl_certificate(PRIMARY_DESIGN) is LOCKED_HIGH_FIDELITY_PL_CERTIFICATE
    assert locked_pl_certificate(FULL_STEP_DESIGN) is LOCKED_FULL_STEP_PL_CERTIFICATE
    assert LOCKED_HIGH_FIDELITY_PL_CERTIFICATE.tau == Fraction(16_777_215, 16_777_216)
    assert LOCKED_FULL_STEP_PL_CERTIFICATE.tau == Fraction(16_777_209, 16_777_216)


def test_jordan_after_resolvent_gain_bound_reduces_to_the_locked_fraction() -> None:
    assert jordan_after_resolvent_gain_upper() == LOCKED_JORDAN_AFTER_RESOLVENT_GAIN_UPPER
    assert jordan_after_resolvent_gain_upper() == RAW_SHAPE_GAIN_UPPER
    assert 753 < jordan_after_resolvent_gain_upper() < 754


def test_quintic_gate_has_exact_c2_joins_and_locked_midpoint_slope() -> None:
    midpoint = (LOCKED_GATE_Q0 + LOCKED_GATE_Q1) / 2
    left = quintic_gate_jet_exact(LOCKED_GATE_Q0, PRIMARY_DESIGN)
    middle = quintic_gate_jet_exact(midpoint, PRIMARY_DESIGN)
    right = quintic_gate_jet_exact(LOCKED_GATE_Q1, PRIMARY_DESIGN)

    assert (left.value, left.first_derivative, left.second_derivative) == (0, 0, 0)
    assert (right.value, right.first_derivative, right.second_derivative) == (
        PRIMARY_DESIGN.cap,
        0,
        0,
    )
    assert middle.value == PRIMARY_DESIGN.cap / 2
    assert middle.first_derivative == PRIMARY_DESIGN.cap * Fraction(5, 2)
    assert middle.second_derivative == 0
    assert smootherstep_gate(Fraction(0), PRIMARY_DESIGN) == 0
    assert smootherstep_gate(Fraction(2), PRIMARY_DESIGN) == PRIMARY_DESIGN.cap


def test_fp64_gate_agrees_with_exact_gate_at_representable_points() -> None:
    for value in (Fraction(0), Fraction(1, 4), Fraction(5, 8), Fraction(1), Fraction(2)):
        exact = smootherstep_gate(value, FULL_STEP_DESIGN)
        computed = quintic_gate_weight_fp64(float(value), FULL_STEP_DESIGN)
        assert computed == pytest.approx(float(exact), rel=0.0, abs=3e-17)


def test_fp64_gate_clamps_only_roundoff_scale_overshoot_near_outer_join() -> None:
    squared_norm = math.nextafter(1.0, 0.0)
    coordinate = (squared_norm - 0.25) / 0.75
    unguarded = coordinate**3 * (10.0 + coordinate * (-15.0 + 6.0 * coordinate))

    assert unguarded > 1.0
    assert unguarded - 1.0 < 16.0 * math.ulp(1.0)
    assert quintic_gate_weight_fp64(squared_norm, PRIMARY_DESIGN) == float(PRIMARY_DESIGN.cap)


@pytest.mark.parametrize("design", [PRIMARY_DESIGN, FULL_STEP_DESIGN])
def test_guarded_singular_evaluator_uses_the_required_gated_output(
    design: GateDesign,
) -> None:
    signal = np.asarray([3.0, 4.0], dtype=np.float64)
    evaluated = evaluate_gated_singular_values_fp64(signal, design)
    expected = (1.0 - evaluated.gate_value) * evaluated.yosida_output / float(
        design.passive_divisor
    ) + evaluated.gate_value * evaluated.shape_output

    assert evaluated.solver_diagnostics.certified
    assert evaluated.solver_diagnostics.status is SolverStatus.STRICT_RESIDUAL
    assert evaluated.solver_diagnostics.residual_norm <= (
        evaluated.solver_diagnostics.p15_threshold
    )
    np.testing.assert_array_equal(evaluated.output, expected)
    assert evaluated.gate_value == float(design.cap)
    modal_gains = evaluated.output / signal
    assert modal_gains[0] != modal_gains[1]


@pytest.mark.parametrize("design", [PRIMARY_DESIGN, FULL_STEP_DESIGN])
def test_matrix_and_singular_paths_agree_on_the_canonical_diagonal(
    design: GateDesign,
) -> None:
    singular = evaluate_gated_singular_values_fp64(
        np.asarray([3.0, 4.0], dtype=np.float64),
        design,
    )
    matrix = evaluate_shape_preserving_resolvent_fp64(
        np.diag(np.asarray([3.0, 4.0], dtype=np.float64)),
        design,
    )

    np.testing.assert_allclose(np.diag(matrix.output), singular.output, rtol=3e-13, atol=5e-13)
    np.testing.assert_allclose(matrix.output - np.diag(np.diag(matrix.output)), 0.0, atol=1e-14)
    assert matrix.resolvent_result.diagnostics.certified
    assert matrix.gate_weight == singular.gate_value


def test_zero_input_is_exact_for_both_public_evaluators() -> None:
    singular = evaluate_gated_singular_values_fp64(
        np.zeros(3, dtype=np.float64),
        PRIMARY_DESIGN,
    )
    matrix = evaluate_shape_preserving_resolvent_fp64(
        np.zeros((3, 5), dtype=np.float64),
        PRIMARY_DESIGN,
    )

    assert singular.solver_diagnostics.status is SolverStatus.ZERO_INPUT
    assert matrix.resolvent_result.diagnostics.status is SolverStatus.ZERO_INPUT
    assert singular.gate_value == matrix.gate_weight == 0.0
    assert singular.squared_signal_norm == matrix.squared_signal_norm == 0.0
    assert np.array_equal(singular.output, np.zeros(3))
    assert np.array_equal(matrix.output, np.zeros((3, 5)))


def test_additive_epsilon_component_validates_the_fp64_matrix_contract() -> None:
    matrix = np.diag(np.asarray([0.6, 0.8], dtype=np.float64))
    output = additive_epsilon_jordan_fp64(matrix)

    assert output.shape == matrix.shape
    assert output.dtype == np.float64
    assert np.isfinite(output).all()
    with pytest.raises(TypeError, match=r"numpy\.float64"):
        additive_epsilon_jordan_fp64(matrix.astype(np.float32))
    with pytest.raises(ValueError, match="strictly positive"):
        additive_epsilon_jordan_fp64(matrix, epsilon=0.0)


def test_public_evaluators_reject_unlocked_designs_and_solver_architectures() -> None:
    signal = np.asarray([3.0, 4.0], dtype=np.float64)
    unlocked = GateDesign(
        name="unlocked",
        cap=Fraction(1, 2),
        passive_divisor=Fraction(4_096),
        learning_rate=Fraction(1, 64_000),
    )
    with pytest.raises(ValueError, match="no exact PL certificate"):
        evaluate_gated_singular_values_fp64(signal, unlocked)
    with pytest.raises(ValueError, match="epsilon"):
        evaluate_gated_singular_values_fp64(
            signal,
            PRIMARY_DESIGN,
            solver_config=replace(EquivariantResolventSolverConfig(), epsilon=2e-7),
        )
    with pytest.raises(ValueError, match="p15_relative_tolerance"):
        evaluate_gated_singular_values_fp64(
            signal,
            PRIMARY_DESIGN,
            solver_config=replace(
                EquivariantResolventSolverConfig(),
                p15_relative_tolerance=1.0 / 100.0,
            ),
        )
    with pytest.raises(ValueError, match="p15_absolute_tolerance"):
        evaluate_gated_singular_values_fp64(
            signal,
            PRIMARY_DESIGN,
            solver_config=replace(
                EquivariantResolventSolverConfig(),
                p15_absolute_tolerance=2.0**-32,
            ),
        )


@pytest.mark.parametrize(
    ("value", "exception", "match"),
    [
        ([3.0, 4.0], TypeError, "numpy.ndarray"),
        (np.asarray([3.0, 4.0], dtype=np.float32), TypeError, "numpy.float64"),
        (np.asarray([[3.0, 4.0]], dtype=np.float64), ValueError, "nonempty vector"),
        (np.asarray([], dtype=np.float64), ValueError, "nonempty vector"),
        (np.asarray([3.0, -4.0], dtype=np.float64), ValueError, "nonnegative"),
        (np.asarray([3.0, np.inf], dtype=np.float64), ValueError, "finite"),
    ],
)
def test_singular_evaluator_rejects_invalid_inputs(
    value: object,
    exception: type[Exception],
    match: str,
) -> None:
    with pytest.raises(exception, match=match):
        evaluate_gated_singular_values_fp64(value, PRIMARY_DESIGN)  # type: ignore[arg-type]


def test_gate_rejects_invalid_numerical_inputs() -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        quintic_gate_weight_fp64(-1.0, PRIMARY_DESIGN)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        quintic_gate_weight_fp64(float("inf"), PRIMARY_DESIGN)
    with pytest.raises(TypeError, match="Fraction or floating"):
        smootherstep_gate(1, PRIMARY_DESIGN)
