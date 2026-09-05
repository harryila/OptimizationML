from __future__ import annotations

from fractions import Fraction

import pytest
from flint import arb, fmpq

from passive_muon.shape_preserving_resolvent_certificate import (
    BEST_SCALAR_DEPARTURE_GATE,
    CANONICAL_UPSTREAM_DEPARTURE_GUARD,
    FULL_STEP_CANONICAL_DEPARTURE_GUARD,
    FULL_STEP_CANONICAL_RETENTION_GUARD,
    FULL_STEP_DESIGN,
    GATE_INNER_SQUARED_RADIUS,
    GATE_OUTER_SQUARED_RADIUS,
    LOCKED_DESIGNS,
    PRIMARY_CANONICAL_DEPARTURE_GUARD,
    PRIMARY_CANONICAL_RETENTION_GUARD,
    PRIMARY_DESIGN,
    RAW_SHAPE_GAIN_UPPER,
    SWITCH_RAW_RATIO,
    UNSAFE_BAND_FORWARD_DERIVATIVE_GUARD,
    UNSAFE_BAND_RAW_SHAPE_DERIVATIVE_GUARD,
    UNSAFE_BAND_SOURCE_GUARD,
    UPSTREAM_SHAPING_RETENTION_GATE,
    GateDesign,
    certify_unsafe_derivative_band,
    evaluate_canonical_gated_fidelity,
    exact_certificate_checks,
    exact_unsafe_witness,
    interval_certificate_checks,
    pointwise_sector,
    quintic_smootherstep,
    smootherstep_gate,
    smootherstep_gate_derivatives,
    undersized_passive_region_control,
)


def _arb_fraction(value: Fraction) -> arb:
    return arb(fmpq(value.numerator, value.denominator))


def test_all_finite_exact_certificate_checks_pass() -> None:
    checks = exact_certificate_checks()
    assert checks
    assert all(checks.values())


def test_locked_pointwise_sectors_have_the_exact_frozen_values() -> None:
    primary = pointwise_sector(PRIMARY_DESIGN)
    full = pointwise_sector(FULL_STEP_DESIGN)

    assert primary.lower_gain == Fraction(125, 4_096)
    assert primary.center == Fraction(
        20_679_066_726_449_389_357_482_875,
        73_163_356_036_579_054_559_232,
    )
    assert primary.radius == Fraction(
        20_676_833_958_015_655_865_827_625,
        73_163_356_036_579_054_559_232,
    )
    assert full.lower_gain == Fraction(875, 16_384)
    assert full.center == Fraction(
        41_421_767_353_260_183_227_140_375,
        877_960_272_438_948_654_710_784,
    )
    assert full.radius == Fraction(
        41_374_879_216_151_779_902_380_125,
        877_960_272_438_948_654_710_784,
    )
    for sector in (primary, full):
        assert sector.upper_gain - sector.lower_gain == 2 * sector.radius
        assert sector.lower_gain < sector.center < sector.upper_gain
    assert 753 < RAW_SHAPE_GAIN_UPPER < 754


def test_general_pointwise_sector_keeps_the_pure_yosida_endpoint() -> None:
    design = GateDesign(
        "unit_divisor_control",
        Fraction(1, 2),
        Fraction(1),
        Fraction(1, 32_000),
    )
    sector = pointwise_sector(design)

    assert sector.upper_gain == 1_000


def test_exact_smootherstep_has_c2_joins_and_midpoint_maximum_slope() -> None:
    midpoint = Fraction(5, 8)
    assert quintic_smootherstep(Fraction(0)) == 0
    assert quintic_smootherstep(Fraction(1, 2)) == Fraction(1, 2)
    assert quintic_smootherstep(Fraction(1)) == 1
    assert smootherstep_gate_derivatives(GATE_INNER_SQUARED_RADIUS, Fraction(1)) == (
        0,
        0,
        0,
    )
    assert smootherstep_gate_derivatives(GATE_OUTER_SQUARED_RADIUS, Fraction(1)) == (
        1,
        0,
        0,
    )
    value, first, second = smootherstep_gate_derivatives(midpoint, Fraction(1))
    assert value == Fraction(1, 2)
    assert first == Fraction(5, 2)
    assert second == 0


@pytest.mark.parametrize("precision_bits", [160, 224])
def test_cross_precision_arb_checks_and_unsafe_band_close(precision_bits: int) -> None:
    checks = interval_certificate_checks(precision_bits=precision_bits)
    band = certify_unsafe_derivative_band(precision_bits=precision_bits)

    assert checks
    assert all(checks.values())
    assert band.is_strictly_unsafe
    assert band.source_value > _arb_fraction(UNSAFE_BAND_SOURCE_GUARD[0])
    assert band.source_value < _arb_fraction(UNSAFE_BAND_SOURCE_GUARD[1])
    assert band.forward_graph_derivative > _arb_fraction(UNSAFE_BAND_FORWARD_DERIVATIVE_GUARD[0])
    assert band.forward_graph_derivative < _arb_fraction(UNSAFE_BAND_FORWARD_DERIVATIVE_GUARD[1])
    assert band.raw_shape_derivative > _arb_fraction(UNSAFE_BAND_RAW_SHAPE_DERIVATIVE_GUARD[0])
    assert band.raw_shape_derivative < _arb_fraction(UNSAFE_BAND_RAW_SHAPE_DERIVATIVE_GUARD[1])


@pytest.mark.parametrize(
    ("design", "departure_guard", "retention_guard"),
    [
        (
            PRIMARY_DESIGN,
            PRIMARY_CANONICAL_DEPARTURE_GUARD,
            PRIMARY_CANONICAL_RETENTION_GUARD,
        ),
        (
            FULL_STEP_DESIGN,
            FULL_STEP_CANONICAL_DEPARTURE_GUARD,
            FULL_STEP_CANONICAL_RETENTION_GUARD,
        ),
    ],
)
def test_canonical_diag_3_4_passes_rigorous_fidelity_gates(
    design: GateDesign,
    departure_guard: tuple[Fraction, Fraction],
    retention_guard: tuple[Fraction, Fraction],
) -> None:
    for bits in (160, 224):
        evaluation = evaluate_canonical_gated_fidelity(design, precision_bits=bits)
        assert evaluation.graph_residual_norm > 0
        assert evaluation.gate_value == design.cap
        assert evaluation.modal_gains[0] != evaluation.modal_gains[1]
        assert evaluation.best_scalar_departure > _arb_fraction(departure_guard[0])
        assert evaluation.best_scalar_departure < _arb_fraction(departure_guard[1])
        assert evaluation.upstream_best_scalar_departure > _arb_fraction(
            CANONICAL_UPSTREAM_DEPARTURE_GUARD[0]
        )
        assert evaluation.upstream_best_scalar_departure < _arb_fraction(
            CANONICAL_UPSTREAM_DEPARTURE_GUARD[1]
        )
        assert evaluation.upstream_shaping_retention > _arb_fraction(retention_guard[0])
        assert evaluation.upstream_shaping_retention < _arb_fraction(retention_guard[1])
        assert evaluation.best_scalar_departure > _arb_fraction(BEST_SCALAR_DEPARTURE_GATE)
        assert evaluation.upstream_shaping_retention > _arb_fraction(
            UPSTREAM_SHAPING_RETENTION_GATE
        )
        assert evaluation.gate_decisions_are_definite
        assert evaluation.meaningful_fidelity_passes


def test_exact_unsafe_witness_and_undersized_gate_are_locked_negative_controls() -> None:
    witness = exact_unsafe_witness()
    control = undersized_passive_region_control()

    assert witness.raw_ratio == SWITCH_RAW_RATIO
    assert Fraction(19, 10_000) < witness.source_value < Fraction(1, 500)
    assert Fraction(-147_000) < witness.raw_shape_derivative < Fraction(-146_000)
    assert Fraction(999) < witness.yosida_derivative < Fraction(1_000)
    assert witness.response_sha256 == (
        "8a45fab1019bdec0355707f4784edd3a09ee75addb27ca63af8d955dcf8f94af"
    )
    assert witness.raw_shape_derivative_sha256 == (
        "a6a7929612a5275509eac2447ac6e75cc0dbdba3c1e235710ae60146ff327cdc"
    )
    assert control.witness_is_on_raw_shape_plateau
    assert Fraction(-19_000) < control.gated_derivative < Fraction(-18_000)
    assert control.fails_second_jury_condition
    assert control.second_jury_margin == (
        control.learning_rate * (1 - control.beta) * control.curvature * control.gated_derivative
    )


def test_exact_api_rejects_invalid_gate_and_precision_parameters() -> None:
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        quintic_smootherstep(Fraction(2))
    with pytest.raises(ValueError, match="nonnegative"):
        smootherstep_gate(Fraction(-1), Fraction(1, 2))
    with pytest.raises(ValueError, match="strictly ordered"):
        smootherstep_gate(
            Fraction(1),
            Fraction(1, 2),
            inner_squared_radius=Fraction(1),
            outer_squared_radius=Fraction(1),
        )
    with pytest.raises(ValueError, match="at least 96"):
        certify_unsafe_derivative_band(precision_bits=95)
    with pytest.raises(ValueError, match="one of the two locked"):
        evaluate_canonical_gated_fidelity(
            GateDesign("unlocked", Fraction(1, 2), Fraction(4_096), Fraction(1, 64_000))
        )
    with pytest.raises(ValueError, match="positive"):
        undersized_passive_region_control(curvature=Fraction(0))


def test_gate_design_rejects_invalid_exact_configs() -> None:
    for cap in (Fraction(0), Fraction(1), Fraction(2)):
        with pytest.raises(ValueError, match="strictly between"):
            GateDesign("invalid", cap, Fraction(1), Fraction(1))
    assert LOCKED_DESIGNS == (PRIMARY_DESIGN, FULL_STEP_DESIGN)
