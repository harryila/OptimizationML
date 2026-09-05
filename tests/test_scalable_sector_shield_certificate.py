from __future__ import annotations

from fractions import Fraction

from passive_muon.scalable_mixed_precision_certificate import (
    FP32_HALF_MIN_SUBNORMAL as P9_FP32_HALF_MIN_SUBNORMAL,
)
from passive_muon.scalable_mixed_precision_certificate import (
    FP32_UNIT_ROUNDOFF as P9_FP32_UNIT_ROUNDOFF,
)
from passive_muon.scalable_mixed_precision_certificate import (
    REPRESENTATIVE_TRANSFORMER_SHAPES,
    MatrixShape,
)
from passive_muon.scalable_sector_shield import (
    LOCKED_SHAPE_CERTIFIED_RADIUS_HEX,
    LOCKED_SHAPE_INWARD_MARGIN_HEX,
)
from passive_muon.scalable_sector_shield_certificate import (
    BF16_MIN_SUBNORMAL,
    FP32_HALF_MIN_SUBNORMAL,
    FP32_MAX_SUBNORMAL,
    FP32_MIN_NORMAL,
    FP32_MIN_SUBNORMAL,
    FP32_UNIT_ROUNDOFF,
    LOCKED_ACCEPTANCE_BUFFER,
    LOCKED_ACCEPTANCE_COEFFICIENT,
    LOCKED_FASTER_RATE,
    LOCKED_FASTER_RATE_STEP,
    LOCKED_MAXIMUM_STEP,
    LOCKED_MAXIMUM_STEP_RATE,
    LOCKED_SECTOR_CENTER,
    LOCKED_SECTOR_LOWER,
    LOCKED_SECTOR_RADIUS,
    LOCKED_SECTOR_UPPER,
    RADIUS_GRID_DENOMINATOR,
    audit_scalable_sector_shield,
    diagnostic_sector_shield_audits,
    exact_contract_checks,
    representative_sector_shield_audits,
)


def test_p20_reuses_p9_rounding_constants_and_p19_sector_exactly() -> None:
    assert FP32_UNIT_ROUNDOFF == P9_FP32_UNIT_ROUNDOFF == Fraction(1, 2**24)
    assert FP32_HALF_MIN_SUBNORMAL == P9_FP32_HALF_MIN_SUBNORMAL == Fraction(1, 2**150)
    assert Fraction(125, 1_024) == LOCKED_SECTOR_LOWER
    assert Fraction(509, 512) == LOCKED_SECTOR_UPPER
    assert Fraction(1_143, 2_048) == LOCKED_SECTOR_CENTER
    assert Fraction(893, 2_048) == LOCKED_SECTOR_RADIUS
    assert LOCKED_SECTOR_CENTER == (LOCKED_SECTOR_LOWER + LOCKED_SECTOR_UPPER) / 2
    assert LOCKED_SECTOR_RADIUS == (LOCKED_SECTOR_UPPER - LOCKED_SECTOR_LOWER) / 2
    assert Fraction(1, 1_024) == LOCKED_ACCEPTANCE_BUFFER
    assert Fraction(891, 2_048) == LOCKED_ACCEPTANCE_COEFFICIENT


def test_all_seven_transformer_shapes_have_positive_exact_margins() -> None:
    audits = representative_sector_shield_audits()
    assert len(audits) == len(REPRESENTATIVE_TRANSFORMER_SHAPES) == 7
    assert tuple(audit.shape for audit in audits) == REPRESENTATIVE_TRANSFORMER_SHAPES
    assert all(audit.certified for audit in audits)
    assert all(audit.inward_margin > 0 for audit in audits)
    assert all(audit.certified_inward_radius < LOCKED_SECTOR_RADIUS for audit in audits)
    assert all(
        audit.certified_inward_radius * RADIUS_GRID_DENOMINATOR
        == int(audit.certified_inward_radius * RADIUS_GRID_DENOMINATOR)
        for audit in audits
    )


def test_two_largest_shape_margins_are_locked_exactly() -> None:
    audit_11008 = audit_scalable_sector_shield((4_096, 11_008))
    audit_14336 = audit_scalable_sector_shield((4_096, 14_336))
    assert audit_11008.inward_margin == Fraction(25_250_274_108_291, 144_115_188_075_855_872)
    assert audit_14336.inward_margin == Fraction(71_710_053_325_847, 1_152_921_504_606_846_976)
    assert audit_14336.inward_margin < audit_11008.inward_margin


def test_dimensionless_norm_error_and_original_unit_crumbs_are_separate() -> None:
    audit = audit_scalable_sector_shield((4_096, 14_336))
    norm = audit.norm
    h = norm.root_entries_upper
    # v=x/maxabs(x) has ||v||>=1, so this is dimensionless.
    assert norm.division_relative_error == FP32_UNIT_ROUNDOFF + h * FP32_HALF_MIN_SUBNORMAL
    # Dhat construction returns to original units; only here is tau divided by
    # the minimum normal signal anchor.
    assert audit.displacement_crumb == h * FP32_HALF_MIN_SUBNORMAL * (2 + FP32_UNIT_ROUNDOFF)
    assert audit.displacement_crumb_relative == audit.displacement_crumb / FP32_MIN_NORMAL
    assert FP32_MIN_NORMAL == FP32_HALF_MIN_SUBNORMAL / FP32_UNIT_ROUNDOFF


def test_norm_envelope_and_both_output_branches_fit_the_inward_disk() -> None:
    for audit in (*representative_sector_shield_audits(), *diagnostic_sector_shield_audits()):
        assert audit.norm.certified
        assert audit.accepted_candidate_radius <= audit.certified_inward_radius
        assert audit.rounded_half_radius <= audit.certified_inward_radius
        assert audit.certified_inward_radius + audit.inward_margin == LOCKED_SECTOR_RADIUS


def test_small_control_shapes_are_predeclared_but_not_headline_targets() -> None:
    diagnostics = diagnostic_sector_shield_audits()
    assert tuple(audit.shape for audit in diagnostics) == (
        MatrixShape(1, 1),
        MatrixShape(1, 2),
        MatrixShape(2, 2),
    )
    assert all(audit.certified for audit in diagnostics)
    assert all(audit.shape not in REPRESENTATIVE_TRANSFORMER_SHAPES for audit in diagnostics)


def test_runtime_hex_table_matches_every_exact_shape_audit() -> None:
    audits = (*diagnostic_sector_shield_audits(), *representative_sector_shield_audits())
    expected_shapes = {(audit.shape.rows, audit.shape.columns) for audit in audits}
    assert set(LOCKED_SHAPE_CERTIFIED_RADIUS_HEX) == expected_shapes
    assert set(LOCKED_SHAPE_INWARD_MARGIN_HEX) == expected_shapes
    for audit in audits:
        shape = (audit.shape.rows, audit.shape.columns)
        assert (
            float(audit.certified_inward_radius).hex() == LOCKED_SHAPE_CERTIFIED_RADIUS_HEX[shape]
        )
        assert float(audit.inward_margin).hex() == LOCKED_SHAPE_INWARD_MARGIN_HEX[shape]


def test_one_over_1024_buffer_is_necessary_for_largest_locked_shape() -> None:
    audit = audit_scalable_sector_shield((4_096, 14_336))
    # Undoing the acceptance buffer while leaving every other rounding term
    # unchanged makes the proved radius escape the P19 disk.
    unbuffered_radius = audit.accepted_candidate_radius + (
        LOCKED_ACCEPTANCE_BUFFER
        * audit.norm.upper_factor
        / audit.norm.lower_factor
        / (1 - FP32_UNIT_ROUNDOFF)
    )
    assert unbuffered_radius > LOCKED_SECTOR_RADIUS


def test_subnormal_guard_is_an_input_output_bound_not_an_objective_claim() -> None:
    audit = audit_scalable_sector_shield((4_096, 14_336))
    assert FP32_MAX_SUBNORMAL == FP32_MIN_NORMAL - FP32_MIN_SUBNORMAL
    assert audit.all_subnormal_input_radius == audit.norm.root_entries_upper * FP32_MAX_SUBNORMAL
    assert audit.zero_instead_of_half_output_error == audit.all_subnormal_input_radius / 2
    assert BF16_MIN_SUBNORMAL / 2 >= FP32_MIN_SUBNORMAL
    assert audit.checks["bf16_widened_minimum_can_be_halved_in_fp32"]


def test_both_p19_operating_point_rates_survive_sector_containment() -> None:
    assert Fraction(1, 83) == LOCKED_MAXIMUM_STEP
    assert LOCKED_MAXIMUM_STEP_RATE == Fraction(999_598_040_401, 1_000_000_000_000) < 1
    assert Fraction(1, 120) == LOCKED_FASTER_RATE_STEP
    assert LOCKED_FASTER_RATE == Fraction(624_350_169, 625_000_000) < 1
    assert LOCKED_FASTER_RATE < LOCKED_MAXIMUM_STEP_RATE


def test_all_cross_shape_contract_checks_pass() -> None:
    checks = exact_contract_checks()
    assert checks
    assert all(checks.values())
