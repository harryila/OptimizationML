from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import pytest

from passive_muon.mixed_precision_certificate import (
    AFFINE_INTERCEPT_UPPER,
    AFFINE_SLOPE_UPPER,
    EXACT_A,
    EXACT_B,
    EXACT_C,
    EXACT_RHO,
    FP32_A,
    FP32_B,
    FP32_C,
    LOCAL_FP32_STAGE_ERROR_UPPER,
    LOCKED_STURM_CHAIN,
    P7_SIGNAL_KAPPA,
    P7_SIGNAL_STORAGE_GAIN,
    P8_FORCING,
    P8_FUNCTION_GAP_ULTIMATE,
    P8_RATE,
    P8_SAFE_FORCING_CAPACITY,
    P8_SAFE_STORAGE_RADIUS,
    REAL_ADAPTER_INTERCEPT_UPPER,
    REAL_ADAPTER_SLOPE_UPPER,
    audit_mixed_precision_certificate,
    locked_mixed_precision_certificate,
    rebuild_sturm_chain,
)
from passive_muon.specs import JORDAN_QUINTIC
from passive_muon.structure_aware_stability import LOCKED_FLOOR, LOCKED_REPAIR_RHO


def test_locked_fp32_coefficient_encodings_are_exact() -> None:
    assert (
        Fraction(6_889, 2_000),
        Fraction(-191, 40),
        Fraction(4_063, 2_000),
    ) == (EXACT_A, EXACT_B, EXACT_C)
    assert Fraction(-1, 32_768_000) == FP32_A - EXACT_A
    assert Fraction(-1, 10_485_760) == FP32_B - EXACT_B
    assert Fraction(53, 524_288_000) == FP32_C - EXACT_C


def test_primary_certificate_matches_the_imported_operator_specification() -> None:
    assert JORDAN_QUINTIC.fractions() == (EXACT_A, EXACT_B, EXACT_C)
    assert LOCKED_FLOOR == 1
    assert LOCKED_REPAIR_RHO == EXACT_RHO


def test_exact_sturm_reconstruction_certifies_the_scalar_range() -> None:
    audit = audit_mixed_precision_certificate()

    assert rebuild_sturm_chain(LOCKED_STURM_CHAIN[0]) == LOCKED_STURM_CHAIN
    assert audit.polynomial_discriminant == Fraction(-2_594_691, 500_000)
    assert audit.sturm_endpoint_signs == (
        (1, -1, -1, 1, -1, -1),
        (1, -1, -1, -1, 1, -1),
    )
    assert audit.sturm_variations == (3, 3)
    assert audit.sturm_endpoint_values[0][0] == Fraction(121, 100)
    assert audit.sturm_endpoint_values[1][0] == Fraction(12_657, 409_600)


def test_normalization_certificate_handles_the_subnormal_case_piecewise() -> None:
    audit = audit_mixed_precision_certificate()
    normalization = audit.normalization

    assert normalization["derived_norm_relative_error"] < Fraction(1, 2**19)
    assert normalization["normalized_output_error"] < Fraction(1, 500_000)
    assert normalization["initial_total_error"] < Fraction(1, 250)
    assert normalization["initial_frobenius_bound"] < Fraction(5, 4)
    assert normalization["initial_spectral_bound"] < Fraction(5, 4)


def test_separately_rounded_horner_stage_closes_the_exact_invariant() -> None:
    audit = audit_mixed_precision_certificate()
    recurrence = audit.recurrence

    assert recurrence["gamma4"] == Fraction(1, 4_194_303)
    assert recurrence["zmm"] == (8 * Fraction(1, 2**150) / (1 - 4 * Fraction(1, 2**24)))
    for name in (
        "epsilon_gram",
        "epsilon_c",
        "epsilon_t",
        "epsilon_d",
        "epsilon_e",
        "epsilon_y",
    ):
        assert recurrence[name] < LOCAL_FP32_STAGE_ERROR_UPPER
    assert recurrence["pre_bf16_frobenius"] == Fraction(239_587, 140_000)
    assert recurrence["pre_bf16_spectral"] == Fraction(24_201, 20_000)
    assert recurrence["next_frobenius"] < Fraction(7, 4)
    assert recurrence["next_spectral"] < Fraction(5, 4)


def test_affine_operator_error_bound_replays_exactly() -> None:
    repair = audit_mixed_precision_certificate().repair

    assert repair["complete_slope"] == Fraction(
        1_025_348_293_983_610_756_598_467,
        9_376_903_711_319_508_102_676_480_000,
    )
    assert repair["complete_slope"] < AFFINE_SLOPE_UPPER
    assert repair["complete_intercept"] < AFFINE_INTERCEPT_UPPER
    assert repair["real_adapter_slope"] == Fraction(
        29_321_583_773_060_684_572_869_055_679_171,
        157_318_338_976_009_032_452_353_483_079_680_000,
    )
    assert repair["real_adapter_slope"] < REAL_ADAPTER_SLOPE_UPPER
    assert repair["real_adapter_intercept"] < REAL_ADAPTER_INTERCEPT_UPPER


def test_p7_affine_absorption_constants_are_exact() -> None:
    closure = audit_mixed_precision_certificate().p7_closure

    assert Fraction(66_221_761, 10_400_336) == P7_SIGNAL_KAPPA
    assert closure["derived_signal_kappa"] == P7_SIGNAL_KAPPA
    assert P7_SIGNAL_STORAGE_GAIN == 100 * P7_SIGNAL_KAPPA
    assert closure["rate"] == P8_RATE
    assert closure["rate"] < 1
    assert closure["forcing"] == P8_FORCING
    assert closure["function_gap_ultimate"] == P8_FUNCTION_GAP_ULTIMATE
    assert closure["safe_storage_radius"] == P8_SAFE_STORAGE_RADIUS
    assert closure["safe_forcing_capacity"] == P8_SAFE_FORCING_CAPACITY
    assert closure["forcing"] <= closure["safe_forcing_capacity"]


def test_complete_locked_audit_passes_and_detects_parameter_drift() -> None:
    certificate = locked_mixed_precision_certificate()
    assert audit_mixed_precision_certificate(certificate).certified
    assert all(audit_mixed_precision_certificate(certificate).checks.values())

    changed = replace(certificate, young_theta=Fraction(5_125))
    changed_audit = audit_mixed_precision_certificate(changed)
    assert not changed_audit.certified
    assert not changed_audit.checks["selected_young_theta_is_locked"]


def test_invalid_locked_shape_and_stage_count_are_rejected() -> None:
    certificate = locked_mixed_precision_certificate()
    with pytest.raises(ValueError, match="shape"):
        replace(certificate, shape=(3, 3))
    with pytest.raises(ValueError, match="five"):
        replace(certificate, stages=4)
    with pytest.raises(TypeError, match=r"fractions\.Fraction"):
        replace(certificate, young_theta=5_124)  # type: ignore[arg-type]
