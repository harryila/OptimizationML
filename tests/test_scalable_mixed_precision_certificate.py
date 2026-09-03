from __future__ import annotations

from fractions import Fraction

import pytest

import passive_muon.scalable_mixed_precision_certificate as certificate_module
from passive_muon.scalable_mixed_precision import LOCKED_MAX_ENTRIES
from passive_muon.scalable_mixed_precision_certificate import (
    BF16_UNIT_ROUNDOFF,
    EXACT_A,
    EXACT_B,
    EXACT_C,
    EXACT_RHO,
    FP32_A,
    FP32_B,
    FP32_C,
    MatrixShape,
    audit_scalable_shape,
    compensated_boundary_rank_limit,
    ideal_two_term_boundary_rank_limit,
    one_term_boundary_rank_limit,
    polynomial_tube_audit,
    representative_audits,
    serial_norm_obstruction,
    two_term_bf16_bound,
)
from passive_muon.specs import JORDAN_QUINTIC
from passive_muon.structure_aware_stability import LOCKED_FLOOR, LOCKED_REPAIR_RHO


def test_exact_operator_data_matches_the_main_theorem_contract() -> None:
    assert JORDAN_QUINTIC.fractions() == (EXACT_A, EXACT_B, EXACT_C)
    assert (
        Fraction(6_889, 2_000),
        Fraction(-191, 40),
        Fraction(4_063, 2_000),
    ) == (EXACT_A, EXACT_B, EXACT_C)
    assert (
        Fraction(-1, 32_768_000),
        Fraction(-1, 10_485_760),
        Fraction(53, 524_288_000),
    ) == (FP32_A - EXACT_A, FP32_B - EXACT_B, FP32_C - EXACT_C)
    assert LOCKED_FLOOR == 1
    assert LOCKED_REPAIR_RHO == EXACT_RHO


def test_sterbenz_free_two_term_boundary_bound_reconstructs_exactly() -> None:
    bound = two_term_bf16_bound()

    assert bound.sterbenz_free
    assert bound.omega == Fraction(282_587_406_795_009, 2**64)
    assert bound.omega == (
        bound.residual_rounding_slope
        + bound.low_rounding_slope
        + bound.reconstruction_rounding_slope
    )
    assert bound.chi == (
        bound.residual_rounding_crumb
        + bound.low_rounding_crumb
        + bound.reconstruction_rounding_crumb
    )
    assert bound.omega < BF16_UNIT_ROUNDOFF**2 * Fraction(129, 128)
    assert bound.frobenius_error(MatrixShape(2, 3), Fraction(7, 5)) > 0


def test_scalar_tube_and_full_matrix_lipschitz_inputs_are_exact() -> None:
    audit = polynomial_tube_audit()

    assert audit.certified
    assert audit.factor_discriminant == Fraction(-2_594_691, 500_000)
    assert audit.factor_correction_at_endpoint == Fraction(-2_049, 1_280)
    assert audit.derivative_vertex == Fraction(2_865, 4_063)
    assert audit.derivative_at_vertex == Fraction(-6_525_559, 4_063_000)
    assert audit.derivative_at_endpoint == Fraction(3_000_459, 512_000)


def test_normalizer_and_all_representative_shape_replays_certify() -> None:
    audits = representative_audits()

    assert len(audits) == 7
    assert all(audit.normalizer.certified for audit in audits)
    assert all(audit.certified for audit in audits)
    assert {audit.shape.oriented for audit in audits} >= {
        (768, 3_072),
        (3_072, 12_288),
        (4_096, 4_096),
        (4_096, 11_008),
        (4_096, 14_336),
    }
    assert all(len(audit.stages) == 5 for audit in audits)
    assert all(
        stage.input_spectral < certificate_module.SPECTRAL_TUBE
        for audit in audits
        for stage in audit.stages
    )


def test_shape_parameterized_affine_bounds_and_p7_rates_are_exact() -> None:
    audits = representative_audits()
    rate = Fraction(137_425_214_491, 137_438_953_472)
    slope = Fraction(102_465_557, 549_755_813_888)

    assert all(audit.operator.real_slope == slope for audit in audits)
    assert all(audit.p7.rate == rate < 1 for audit in audits)
    for audit in audits:
        assert audit.operator.real_intercept > 0
        assert audit.p7.forcing > 0
        assert audit.p7.function_gap_ultimate > 0
        assert audit.p7.forcing <= audit.p7.safe_forcing_capacity

    largest = audit_scalable_shape((4_096, 11_008))
    assert largest.operator.real_intercept == Fraction(2_179_083_213_031, 1_099_511_627_776)
    assert largest.p7.function_gap_ultimate == Fraction(798_350_562_999, 1_099_511_627_776)
    assert largest.p7.function_gap_ultimate < 1


def test_every_propagated_stage_bound_uses_the_declared_exact_upper_grid() -> None:
    audit = audit_scalable_shape((4_096, 11_008))
    grid = certificate_module.UPPER_GRID_DENOMINATOR

    for stage in audit.stages:
        for name in (
            "gram_error",
            "scaled_gram_error",
            "shifted_gram_error",
            "product_error",
            "affine_error",
            "output_error_before_boundary",
            "pre_boundary_spectral",
            "pre_boundary_frobenius",
            "boundary_error",
            "next_spectral",
            "next_frobenius",
            "forward_error",
        ):
            assert (getattr(stage, name) * grid).denominator == 1


def test_rank_gates_separate_proof_obstruction_from_compensated_design() -> None:
    assert one_term_boundary_rank_limit() == 71
    assert compensated_boundary_rank_limit() == 4_656_751
    assert ideal_two_term_boundary_rank_limit() == 4_693_632
    assert compensated_boundary_rank_limit() > 60_000 * one_term_boundary_rank_limit()


def test_serial_all_ones_witness_and_recurrence_frontier_are_qualified() -> None:
    witness = serial_norm_obstruction((4_096, 11_008))
    assert witness.exact_square_sum == 43 * 2**20
    assert witness.computed_square_sum == 2**24
    assert witness.returned_singular_value_squared == Fraction(43, 16)
    assert witness.leaves_spectral_tube

    frontier = audit_scalable_shape((4_608, 18_432))
    failed = {name for name, passed in frontier.checks.items() if not passed}
    assert not frontier.certified
    assert failed == {"stage_5_input_in_spectral_tube"}


def test_certificate_shape_domain_rejects_invalid_or_unguarded_dimensions() -> None:
    with pytest.raises(ValueError, match="positive integers"):
        MatrixShape(0, 2)
    with pytest.raises(ValueError, match="positive integers"):
        MatrixShape(True, 2)
    with pytest.raises(ValueError, match=r"2\*\*52"):
        MatrixShape(1, LOCKED_MAX_ENTRIES + 1)
    with pytest.raises(ValueError, match="more than"):
        serial_norm_obstruction((4_096, 4_096))
