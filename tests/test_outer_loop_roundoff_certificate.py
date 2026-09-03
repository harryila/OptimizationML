from __future__ import annotations

from fractions import Fraction

import pytest

from passive_muon.finite_precision_outer_loop import (
    CERTIFIED_STORAGE_PROFILE_OPERATOR_OUTPUT_MAX_ABS,
    EMA_GRADIENT_ENVELOPE_COEFFICIENT,
    EMA_MOMENTUM_ENVELOPE_COEFFICIENT,
    LEARNING_RATE_FP32_EXACT,
    LOCKED_MASTER_HIGH_MAX_ABS,
    LOCKED_MASTER_LOW_MAX_ABS,
    LOCKED_MASTER_MIDDLE_MAX_ABS,
    LOCKED_OPERATOR_OUTPUT_MAX_ABS,
    LOCKED_ROUNDED_STEP_MAX_ABS,
)
from passive_muon.outer_loop_roundoff_certificate import (
    AffineStorageEnvelope,
    audit_outer_loop_roundoff,
    build_outer_loop_roundoff_certificate,
    flat_direction_obstruction,
    harmonic_parameter_drift,
    locked_outer_loop_roundoff_certificate,
)


def test_locked_exact_rate_forcing_and_objective_neighborhood() -> None:
    certificate = locked_outer_loop_roundoff_certificate()

    assert certificate.rate == Fraction(549_700_907_325, 549_755_813_888) < 1
    assert certificate.constant_forcing == Fraction(2_162_331, 1_099_511_627_776)
    assert certificate.function_gap_ultimate == Fraction(399_957_341_889, 549_755_813_888) < 1
    assert certificate.gradient_young == 1
    assert certificate.operator_young == 837


def test_locked_port_envelopes_reconstruct_exactly() -> None:
    reduction = locked_outer_loop_roundoff_certificate().reduction

    assert reduction.sqrt_entries_upper == 6_715
    assert reduction.ema_momentum_coefficient == EMA_MOMENTUM_ENVELOPE_COEFFICIENT
    assert reduction.ema_gradient_coefficient == EMA_GRADIENT_ENVELOPE_COEFFICIENT
    assert reduction.runtime_learning_rate == LEARNING_RATE_FP32_EXACT
    expected = {
        "momentum_port": (
            Fraction(3_222_635, 1_099_511_627_776),
            Fraction(1, 1_099_511_627_776),
        ),
        "signal_port": (
            Fraction(522_283, 137_438_953_472),
            Fraction(1, 1_099_511_627_776),
        ),
        "actual_signal": (
            Fraction(27_744_481_000_049, 1_099_511_627_776),
            Fraction(1, 1_099_511_627_776),
        ),
        "operator_output": (
            Fraction(35_858_102_283_093_177, 1_099_511_627_776),
            Fraction(2_179_083_214_323, 1_099_511_627_776),
        ),
        "parameter_port": (
            Fraction(93_403, 549_755_813_888),
            Fraction(6_727, 1_099_511_627_776),
        ),
        "effective_gradient_error": (
            Fraction(16_113_175, 274_877_906_944),
            Fraction(5, 274_877_906_944),
        ),
        "effective_operator_error": (
            Fraction(9_288_413_563, 549_755_813_888),
            Fraction(2_179_298_479_615, 1_099_511_627_776),
        ),
    }
    for name, (slope, intercept) in expected.items():
        envelope = getattr(reduction, name)
        assert envelope == AffineStorageEnvelope(slope=slope, intercept=intercept)


def test_total_port_algebra_matches_p7_exactly() -> None:
    beta = Fraction(19, 20)
    a = 1 - beta
    momentum = Fraction(-7, 5)
    gradient = Fraction(13, 11)
    r_m = Fraction(17, 10_000)
    r_s = Fraction(-23, 20_000)

    momentum_next = beta * momentum + a * gradient + r_m
    actual_signal = beta * momentum_next + a * gradient + r_s
    effective_gradient = gradient + r_m / a
    p7_momentum_next = beta * momentum + a * effective_gradient
    nominal_signal = beta * p7_momentum_next + a * effective_gradient

    assert p7_momentum_next == momentum_next
    assert actual_signal - nominal_signal == r_s - r_m


def test_unit_storage_closes_every_finite_precision_guard() -> None:
    certificate = locked_outer_loop_roundoff_certificate()
    guard = certificate.guard

    assert certificate.constant_forcing <= guard.forcing_capacity
    assert guard.storage_radius == 1
    assert guard.signal_at_radius < guard.p9_signal_max_abs
    assert (
        guard.operator_output_at_radius
        < guard.certificate_operator_output_max_abs
        == CERTIFIED_STORAGE_PROFILE_OPERATOR_OUTPUT_MAX_ABS
        < guard.runtime_operator_output_max_abs
        == LOCKED_OPERATOR_OUTPUT_MAX_ABS
    )
    assert guard.rounded_step_at_radius < guard.rounded_step_max_abs
    assert guard.rounded_step_max_abs == LOCKED_ROUNDED_STEP_MAX_ABS == 2
    assert guard.master_high_max_abs == LOCKED_MASTER_HIGH_MAX_ABS == 2**30
    assert guard.master_middle_max_abs == LOCKED_MASTER_MIDDLE_MAX_ABS == 2**7
    assert guard.master_low_max_abs == LOCKED_MASTER_LOW_MAX_ABS == Fraction(1, 2**16)
    assert guard.ema_intermediates_are_finite
    assert all(guard.middle_low_invariant_checks.values())
    assert guard.high_word_guard_is_conditional


def test_exact_audit_accepts_and_records_noncoercive_high_word_scope() -> None:
    audit = audit_outer_loop_roundoff()

    assert audit.certified
    assert audit.certificate.certified
    assert all(audit.checks.values())
    assert audit.checks["high_guard_is_explicitly_conditional"]
    assert audit.checks["gradient_cast_and_ema_intermediates_are_finite"]


def test_flat_direction_witness_blocks_full_parameter_iss() -> None:
    obstruction = flat_direction_obstruction()
    displacement_8, energy_8 = harmonic_parameter_drift(8)
    displacement_64, energy_64 = harmonic_parameter_drift(64)

    assert obstruction.objective == "f(x,y)=x^2/2"
    assert obstruction.pl_constant == 1
    assert obstruction.smoothness_upper == 10
    assert obstruction.objective_gap == obstruction.gradient_norm == obstruction.momentum_norm == 0
    assert obstruction.bounded_error_causes_unbounded_parameter
    assert obstruction.square_summable_error_causes_parameter_drift
    assert displacement_64 > displacement_8
    assert energy_8 < energy_64 < 2


def test_invalid_certificate_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="positive Fraction"):
        build_outer_loop_roundoff_certificate(storage_radius=Fraction(0))
    with pytest.raises(ValueError, match="Young parameters"):
        build_outer_loop_roundoff_certificate(gradient_young=0)
    with pytest.raises(ValueError, match="horizon"):
        harmonic_parameter_drift(0)
    with pytest.raises(ValueError, match="does not pass the P9"):
        build_outer_loop_roundoff_certificate(shape=(4_608, 18_432))
