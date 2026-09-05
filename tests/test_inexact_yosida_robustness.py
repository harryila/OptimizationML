from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import pytest

from passive_muon.inexact_yosida_robustness import (
    FROZEN_ABSOLUTE_OUTPUT_PENALTY_THRESHOLD,
    LOCKED_ABSOLUTE_FORCING_GAIN,
    LOCKED_ABSOLUTE_OUTPUT_PENALTY,
    LOCKED_EFFECTIVE_RESIDUAL_LIPSCHITZ,
    LOCKED_FAILING_FROZEN_RADIUS,
    LOCKED_GRAPH_FORM_OUTPUT_ERROR_GAIN,
    LOCKED_PASSING_FROZEN_RADIUS,
    LOCKED_RESOLVENT_FORM_OUTPUT_ERROR_GAIN,
    LOCKED_SOLUTION_ERROR_GAIN,
    LOCKED_SOLVER_RELATIVE_RESIDUAL,
    InexactResolventBounds,
    InexactYosidaCertificate,
    audit_inexact_yosida_robustness,
    double_relative_residual_instability_control,
    exact_certificate_checks,
    frozen_storage_radius_control,
    inexact_yosida_lmi_matrix,
    inexact_yosida_matrices,
    locked_inexact_resolvent_bounds,
    locked_inexact_yosida_certificate,
    locked_inexact_yosida_pl_certificate,
    relative_tolerance_control,
    residual_error_estimate,
    scalar_residual_sharpness_control,
    unit_relative_residual_stall_control,
)
from passive_muon.yosida_stability import (
    LOCKED_RATE_SQUARED,
    locked_yosida_pl_certificate,
    locked_yosida_sector,
)


def test_locked_residual_to_solution_and_output_gains_are_exact() -> None:
    bounds = locked_inexact_resolvent_bounds()

    assert bounds.denominator == 2
    assert bounds.solution_error_gain == LOCKED_SOLUTION_ERROR_GAIN == Fraction(1, 2)
    assert bounds.resolvent_form_output_error_gain == LOCKED_RESOLVENT_FORM_OUTPUT_ERROR_GAIN == 500
    assert bounds.graph_form_output_error_gain == LOCKED_GRAPH_FORM_OUTPUT_ERROR_GAIN == 1_000

    estimate = residual_error_estimate(Fraction(7, 13))
    assert estimate.solution_error_bound == Fraction(7, 26)
    assert estimate.resolvent_form_output_error_bound == Fraction(3_500, 13)
    assert estimate.graph_form_output_error_bound == Fraction(7_000, 13)


def test_lower_endpoint_scalar_map_attains_both_locked_gains() -> None:
    witness = scalar_residual_sharpness_control()
    lam = Fraction(1, 1_000)
    mu = Fraction(1_000)

    assert witness.solution_error_ratio == Fraction(1, 2)
    assert witness.output_error_ratio == 500
    assert witness.residual_value == (
        witness.input_value - witness.approximate_solution - lam * mu * witness.approximate_solution
    )
    graph_output = mu * witness.approximate_solution
    assert witness.approximate_resolvent_output == graph_output + witness.residual_value / lam


def test_graph_output_has_a_distinct_and_larger_class_uniform_gain() -> None:
    # For scalar B=bI the graph-output residual gain is b/(1+lambda*b),
    # which approaches 1/lambda.  The resolvent-form gain is instead
    # 1/[lambda*(1+lambda*b)] and is maximized at b=mu.
    lam = Fraction(1, 1_000)
    large_slope = Fraction(1_000_000_000)
    graph_gain = large_slope / (1 + lam * large_slope)
    resolvent_form_gain = 1 / (lam * (1 + lam * large_slope))

    assert Fraction(999) < graph_gain < 1_000
    assert 0 < resolvent_form_gain < 1


def test_kappa_budget_enlarges_the_centered_radius_from_250_to_252() -> None:
    certificate = locked_inexact_yosida_certificate()

    assert (
        certificate.relative_residual_slope == LOCKED_SOLVER_RELATIVE_RESIDUAL == Fraction(1, 250)
    )
    assert certificate.sector.residual_lipschitz == 250
    assert certificate.output_error_gain * certificate.relative_residual_slope == Fraction(2)
    assert (
        certificate.effective_residual_lipschitz
        == certificate.pl_certificate.residual_lipschitz
        == LOCKED_EFFECTIVE_RESIDUAL_LIPSCHITZ
        == 252
    )


def test_locked_five_coordinate_dynamics_have_the_claimed_scaling() -> None:
    certificate = locked_inexact_yosida_certificate()
    matrices = inexact_yosida_matrices(certificate)

    assert certificate.pl_certificate.dimensionless_step == Fraction(15, 64)
    assert certificate.pl_certificate.residual_ratio == Fraction(42, 125)
    assert matrices.transition == (
        (
            Fraction(19, 20),
            Fraction(1, 20),
            Fraction(0),
            Fraction(0),
            Fraction(0),
        ),
        (
            Fraction(0),
            Fraction(0),
            Fraction(0),
            Fraction(1),
            Fraction(0),
        ),
    )
    assert matrices.signal_selector == (
        Fraction(361, 400),
        Fraction(39, 400),
        Fraction(0),
        Fraction(0),
        Fraction(0),
    )
    assert matrices.step_selector == (
        Fraction(-1_083, 5_120),
        Fraction(-117, 5_120),
        Fraction(-63, 800),
        Fraction(0),
        Fraction(-15, 64),
    )


def test_exact_five_by_five_lmi_and_forcing_gain_replay() -> None:
    certificate = locked_inexact_yosida_certificate()
    audit = audit_inexact_yosida_robustness()
    lmi = inexact_yosida_lmi_matrix(certificate)

    assert certificate.rate == LOCKED_RATE_SQUARED == Fraction(249_001, 250_000)
    assert (
        certificate.absolute_output_penalty
        == LOCKED_ABSOLUTE_OUTPUT_PENALTY
        == Fraction(1, 100_000)
    )
    assert certificate.normalized_absolute_output_penalty == Fraction(1_125, 2)
    assert certificate.absolute_forcing_gain == LOCKED_ABSOLUTE_FORCING_GAIN == Fraction(5, 2)
    assert certificate.ultimate_storage_gain == Fraction(625_000, 999)
    assert audit.storage_leading_minors == (
        Fraction(674_389, 1_000_000),
        Fraction(2_890_417_223, 1_000_000_000_000),
    )
    assert audit.negative_lmi_leading_minors[:2] == (
        Fraction(608_720_541_672_414_373, 26_214_400_000_000_000_000),
        Fraction(
            8_368_016_172_090_059_871_782_660_037,
            13_107_200_000_000_000_000_000_000_000_000,
        ),
    )
    assert audit.negative_lmi_leading_minors[-1] == Fraction(
        51_043_614_879_247_514_899_830_144_119_154_839_166_836_210_223,
        8_192_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000,
    )
    assert all(value > 0 for value in audit.negative_lmi_leading_minors)
    assert len(lmi) == 5 and all(len(row) == 5 for row in lmi)
    assert all(lmi[i][j] == lmi[j][i] for i in range(5) for j in range(5))
    assert audit.certified


def test_selected_absolute_penalty_is_strictly_above_exact_threshold() -> None:
    selected = locked_inexact_yosida_certificate()
    boundary = replace(
        selected,
        absolute_output_penalty=FROZEN_ABSOLUTE_OUTPUT_PENALTY_THRESHOLD,
    )
    boundary_audit = audit_inexact_yosida_robustness(boundary)

    assert FROZEN_ABSOLUTE_OUTPUT_PENALTY_THRESHOLD < LOCKED_ABSOLUTE_OUTPUT_PENALTY
    assert boundary_audit.negative_lmi_leading_minors[-1] == 0
    assert not boundary_audit.certified


def test_frozen_p14_storage_passes_at_252_and_is_rejected_at_253() -> None:
    passing = frozen_storage_radius_control(LOCKED_PASSING_FROZEN_RADIUS)
    failing = frozen_storage_radius_control(LOCKED_FAILING_FROZEN_RADIUS)

    assert passing.residual_radius == 252
    assert passing.certificate_passes
    assert all(value > 0 for value in passing.audit.negative_lmi_leading_minors)
    assert failing.residual_radius == 253
    assert failing.audit.negative_lmi_leading_minors[-1] < 0
    assert not failing.certificate_passes


def test_unchanged_p14_storage_and_multipliers_are_retained() -> None:
    p14 = locked_yosida_pl_certificate()
    p15 = locked_inexact_yosida_pl_certificate()

    assert p15 == replace(p14, residual_lipschitz=Fraction(252))
    assert p15.storage == p14.storage
    assert p15.function_storage == p14.function_storage
    assert p15.interpolation_reverse_weight == p14.interpolation_reverse_weight
    assert p15.lambda_residual_norm == p14.lambda_residual_norm
    assert p15.tau == p14.tau


def test_large_relative_residual_controls_stall_and_destabilize() -> None:
    stall = unit_relative_residual_stall_control()
    unstable = double_relative_residual_instability_control()

    assert stall.relative_tolerance == 1
    assert stall.residual_multiplier == -1
    assert stall.approximate_output_slope == 0
    assert stall.p_at_one == 0
    assert stall.stalls
    assert not stall.violates_schur_condition

    assert unstable.relative_tolerance == 2
    assert unstable.residual_multiplier == -2
    assert unstable.approximate_output_slope == -500
    assert unstable.p_at_one == Fraction(-1, 1_280)
    assert unstable.violates_schur_condition
    assert not unstable.stalls

    assert relative_tolerance_control(Fraction(1, 250)).approximate_output_slope == 498


def test_negative_output_control_reconstructs_scalar_characteristic() -> None:
    """Build the pinned scalar state matrix rather than trusting a stored Jury value."""

    control = double_relative_residual_instability_control()
    beta = Fraction(19, 20)
    one_minus_beta = 1 - beta
    eta = Fraction(1, 32_000)
    curvature = Fraction(1)
    output_slope = control.approximate_output_slope

    # State ordering is (W,m).  The signal is
    # beta**2*m + (1-beta**2)*curvature*W.
    state = (
        (
            1 - eta * output_slope * (1 - beta**2) * curvature,
            -eta * output_slope * beta**2,
        ),
        (one_minus_beta * curvature, beta),
    )
    identity_minus_state = (
        (1 - state[0][0], -state[0][1]),
        (-state[1][0], 1 - state[1][1]),
    )
    p_at_one = (
        identity_minus_state[0][0] * identity_minus_state[1][1]
        - identity_minus_state[0][1] * identity_minus_state[1][0]
    )

    assert p_at_one == control.p_at_one == Fraction(-1, 1_280)
    assert p_at_one == eta * one_minus_beta * curvature * output_slope


def test_exact_checks_all_pass() -> None:
    checks = exact_certificate_checks()

    assert checks
    assert all(checks.values())


def test_exact_type_and_consistency_guards() -> None:
    with pytest.raises(TypeError, match="residual_norm"):
        residual_error_estimate(0.1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="nonnegative"):
        residual_error_estimate(Fraction(-1))
    with pytest.raises(ValueError, match="must be positive"):
        InexactResolventBounds(Fraction(0), Fraction(1_000))
    with pytest.raises(ValueError, match="relative solve error"):
        InexactYosidaCertificate(
            sector=locked_yosida_sector(),
            pl_certificate=locked_yosida_pl_certificate(),
            relative_residual_slope=Fraction(1, 250),
            absolute_output_penalty=Fraction(1, 100_000),
        )
