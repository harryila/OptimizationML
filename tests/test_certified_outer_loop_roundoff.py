from __future__ import annotations

from fractions import Fraction

import pytest

from passive_muon.certified_outer_loop_composition_certificate import (
    AffinePortEnvelope,
)
from passive_muon.certified_outer_loop_roundoff import (
    BOUNDED_DECAY_PORT_BUDGET,
    LOCKED_EXTERNAL_GRADIENT_INTERCEPT,
    LOCKED_EXTERNAL_GRADIENT_SLOPE,
    LOCKED_MODEL_RECONSTRUCTION_INTERCEPT,
    LOCKED_MODEL_RECONSTRUCTION_SLOPE,
    LOCKED_OUTPUT_MAX_ABS,
    LOCKED_ROBUST_GRADIENT_INTERCEPT,
    LOCKED_ROBUST_GRADIENT_SLOPE,
    LOCKED_TOTAL_STEP_MAX_ABS,
    LOCKED_WEIGHT_DECAY_DISPLACEMENT,
    LOCKED_WEIGHT_DECAY_ROUNDED_STEP,
    ROBUST_GRADIENT_SOURCE_BUDGET,
    SQRT_ENCLOSURE_DENOMINATOR,
    ZERO_GRADIENT_SOURCE_BUDGET,
    DecayPortBudget,
    GradientSourceBudget,
    build_p21_roundoff_reduction,
    evaluate_p21_roundoff,
    exact_roundoff_checks,
    representative_roundoff_evaluations,
)
from passive_muon.scalable_sector_shield import LOCKED_REPRESENTATIVE_SHAPES
from passive_muon.sector_shielded_inexact_resolvent_certificate import (
    FASTER_RATE_POINT,
    MAXIMUM_STEP_POINT,
)


def test_all_exact_roundoff_checks_close() -> None:
    assert all(exact_roundoff_checks().values())


def test_locked_gradient_profiles_are_explicit_and_exact() -> None:
    assert ZERO_GRADIENT_SOURCE_BUDGET.slope == 0
    assert ZERO_GRADIENT_SOURCE_BUDGET.intercept == 0
    assert ROBUST_GRADIENT_SOURCE_BUDGET.slope == LOCKED_ROBUST_GRADIENT_SLOPE
    assert ROBUST_GRADIENT_SOURCE_BUDGET.intercept == LOCKED_ROBUST_GRADIENT_INTERCEPT
    assert Fraction(1, 4_096) == LOCKED_ROBUST_GRADIENT_SLOPE
    assert Fraction(1, 4_096) == LOCKED_ROBUST_GRADIENT_INTERCEPT
    assert Fraction(1, 8_192) == LOCKED_EXTERNAL_GRADIENT_SLOPE
    assert Fraction(1, 8_192) == LOCKED_EXTERNAL_GRADIENT_INTERCEPT
    assert Fraction(1, 81_920) == LOCKED_MODEL_RECONSTRUCTION_SLOPE
    assert Fraction(1, 81_920) == LOCKED_MODEL_RECONSTRUCTION_INTERCEPT
    assert (
        LOCKED_EXTERNAL_GRADIENT_SLOPE + 10 * LOCKED_MODEL_RECONSTRUCTION_SLOPE
        == LOCKED_ROBUST_GRADIENT_SLOPE
    )
    assert (
        LOCKED_EXTERNAL_GRADIENT_INTERCEPT + 10 * LOCKED_MODEL_RECONSTRUCTION_INTERCEPT
        == LOCKED_ROBUST_GRADIENT_INTERCEPT
    )
    assert SQRT_ENCLOSURE_DENOMINATOR == 2**80


@pytest.mark.parametrize("point", (FASTER_RATE_POINT, MAXIMUM_STEP_POINT))
def test_all_seven_shapes_close_stored_arithmetic(point) -> None:
    selected = [item for item in representative_roundoff_evaluations() if item.core.point == point]

    assert [item.reduction.shape for item in selected] == list(LOCKED_REPRESENTATIVE_SHAPES)
    assert all(item.certified for item in selected)
    assert all(item.reported_rate_upper < 1 for item in selected)
    assert all(item.reported_forcing_upper <= 1 - item.reported_rate_upper for item in selected)
    assert all(item.guard.shield_output_at_one <= LOCKED_OUTPUT_MAX_ABS for item in selected)
    assert all(item.guard.total_step_at_one <= LOCKED_TOTAL_STEP_MAX_ABS for item in selected)


def test_primary_and_secondary_reported_rates_round_outward() -> None:
    primary = evaluate_p21_roundoff((4_096, 14_336), FASTER_RATE_POINT)
    secondary = evaluate_p21_roundoff((4_096, 14_336), MAXIMUM_STEP_POINT)

    assert primary.reported_rate_upper == Fraction(1_098_368_546_995, 1_099_511_627_776)
    assert secondary.reported_rate_upper == Fraction(549_534_922_135, 549_755_813_888)
    assert primary.reported_rate_upper >= primary.absorption.absorbed_rate
    assert secondary.reported_rate_upper >= secondary.absorption.absorbed_rate
    assert primary.reported_rate_upper < 1
    assert secondary.reported_rate_upper < 1


def test_stored_signal_is_the_direct_sector_port() -> None:
    reduction = build_p21_roundoff_reduction(
        (4_096, 11_008),
        FASTER_RATE_POINT,
        gradient_budget=ROBUST_GRADIENT_SOURCE_BUDGET,
    )
    momentum, signal, output = reduction.normalized_ports

    assert isinstance(momentum, AffinePortEnvelope)
    assert isinstance(signal, AffinePortEnvelope)
    assert isinstance(output, AffinePortEnvelope)
    assert reduction.stored_signal.slope > 0
    assert reduction.shield_output.slope == Fraction(509, 512) * reduction.stored_signal.slope
    assert reduction.shield_output.intercept == (
        Fraction(509, 512) * reduction.stored_signal.intercept
    )
    assert output.slope > 0
    assert output.intercept > 0


def test_transposed_runtime_orientation_reuses_the_same_exact_shape_bound() -> None:
    forward = evaluate_p21_roundoff((768, 3_072), FASTER_RATE_POINT)
    reverse = evaluate_p21_roundoff((3_072, 768), FASTER_RATE_POINT)

    assert forward.reduction.shield_shape == reverse.reduction.shield_shape == (768, 3_072)
    assert not forward.reduction.transposed_for_shield
    assert reverse.reduction.transposed_for_shield
    assert forward.reduction.sqrt_entries_upper == reverse.reduction.sqrt_entries_upper
    assert forward.reported_rate_upper == reverse.reported_rate_upper
    assert forward.reported_forcing_upper == reverse.reported_forcing_upper
    assert reverse.certified


def test_zero_external_error_is_stronger_but_still_has_roundoff_neighborhood() -> None:
    zero = evaluate_p21_roundoff(
        (4_096, 14_336),
        FASTER_RATE_POINT,
        gradient_budget=ZERO_GRADIENT_SOURCE_BUDGET,
    )
    robust = evaluate_p21_roundoff((4_096, 14_336), FASTER_RATE_POINT)

    assert zero.certified and robust.certified
    assert zero.reported_rate_upper < robust.reported_rate_upper
    assert 0 < zero.reported_forcing_upper < robust.reported_forcing_upper
    assert zero.reported_objective_gap_upper < robust.reported_objective_gap_upper


@pytest.mark.parametrize("point", (FASTER_RATE_POINT, MAXIMUM_STEP_POINT))
def test_nonzero_decay_is_an_explicit_bounded_update_port(point) -> None:
    baseline = evaluate_p21_roundoff((4_096, 14_336), point)
    decay = evaluate_p21_roundoff(
        (4_096, 14_336),
        point,
        decay_budget=BOUNDED_DECAY_PORT_BUDGET,
    )

    assert Fraction(1, 131_072) == LOCKED_WEIGHT_DECAY_DISPLACEMENT
    assert Fraction(1, 131_072) == LOCKED_WEIGHT_DECAY_ROUNDED_STEP
    assert decay.reduction.decay_budget.logical_displacement == (LOCKED_WEIGHT_DECAY_DISPLACEMENT)
    assert decay.reduction.decay_budget.rounded_step == LOCKED_WEIGHT_DECAY_ROUNDED_STEP
    assert decay.certified
    assert decay.reported_rate_upper == baseline.reported_rate_upper
    assert decay.reported_forcing_upper > baseline.reported_forcing_upper
    assert decay.reported_objective_gap_upper < 1
    assert decay.guard.pending_after_decay_strict_upper < 3
    assert decay.guard.middle_candidate_strict_upper < 2**8
    assert all(decay.guard.lower_word_guard_checks.values())


def test_dead_zone_is_absorbed_as_a_positive_absolute_parameter_error() -> None:
    small = build_p21_roundoff_reduction((1, 1), FASTER_RATE_POINT)
    large = build_p21_roundoff_reduction((4_096, 14_336), FASTER_RATE_POINT)

    assert small.dead_zone_parameter_error == Fraction(1, 120 * 2**127)
    assert large.dead_zone_parameter_error > small.dead_zone_parameter_error
    assert large.dead_zone_parameter_error > 0


def test_invalid_budget_shape_point_and_decay_fail_closed() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        GradientSourceBudget(Fraction(-1), Fraction(0), "bad")
    with pytest.raises(ValueError, match="name"):
        GradientSourceBudget(Fraction(0), Fraction(0), "")
    with pytest.raises(ValueError, match="P20"):
        build_p21_roundoff_reduction((3, 7), FASTER_RATE_POINT)
    with pytest.raises(ValueError, match="nonnegative"):
        DecayPortBudget(Fraction(-1), Fraction(0), "bad")
    with pytest.raises(TypeError, match="DecayPortBudget"):
        build_p21_roundoff_reduction((2, 2), FASTER_RATE_POINT, decay_budget=Fraction(0))  # type: ignore[arg-type]
