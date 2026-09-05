from __future__ import annotations

from fractions import Fraction

import pytest
from flint import arb, fmpq

from passive_muon.sector_projected_resolvent_certificate import (
    BEST_SCALAR_DEPARTURE_GATE,
    CANONICAL_DEPARTURE_GUARD,
    CANONICAL_INPUT_AMPLITUDE_GUARD,
    CANONICAL_RETENTION_GUARD,
    CANONICAL_UPSTREAM_AMPLITUDE_GUARD,
    GENERIC_SECTOR_SCHUR_MARGIN,
    LOCKED_SECTOR_CENTER,
    LOCKED_SECTOR_LOWER,
    LOCKED_SECTOR_RADIUS,
    LOCKED_SECTOR_UPPER,
    P14_RATE_SQUARED,
    PARETO_CERTIFICATES,
    PRIMARY_CERTIFICATE_SPEC,
    SMALL_K_DEPARTURE_GUARD,
    SMALL_K_PROJECTION_GAIN,
    SMALL_K_RETENTION_GUARD,
    UPSTREAM_SHAPING_RETENTION_GATE,
    audit_all_pareto_certificates,
    evaluate_canonical_projected_fidelity,
    exact_certificate_checks,
    generic_sector_schur_obstruction,
    interval_certificate_checks,
    make_pl_certificate,
    unprojected_sector_control,
)


def _arb_fraction(value: Fraction) -> arb:
    return arb(fmpq(value.numerator, value.denominator))


def test_all_exact_p18_certificate_checks_pass() -> None:
    checks = exact_certificate_checks()
    assert checks
    assert all(checks.values())


def test_locked_sector_and_primary_rate_are_exact() -> None:
    assert Fraction(125, 1_024) == LOCKED_SECTOR_LOWER
    assert Fraction(509, 512) == LOCKED_SECTOR_UPPER
    assert Fraction(1_143, 2_048) == LOCKED_SECTOR_CENTER
    assert Fraction(893, 2_048) == LOCKED_SECTOR_RADIUS

    certificate = make_pl_certificate(PRIMARY_CERTIFICATE_SPEC)
    assert certificate.learning_rate == Fraction(1, 83)
    assert certificate.tau == Fraction(999_799, 1_000_000)
    assert certificate.tau**2 == Fraction(999_598_040_401, 1_000_000_000_000)
    assert certificate.tau**20 < P14_RATE_SQUARED
    assert certificate.learning_rate * LOCKED_SECTOR_LOWER > Fraction(1, 1_000)
    assert certificate.learning_rate * LOCKED_SECTOR_UPPER < Fraction(1, 80)


def test_all_six_exact_pareto_lmis_pass() -> None:
    audits = audit_all_pareto_certificates()
    assert len(audits) == len(PARETO_CERTIFICATES) == 6
    assert all(audit.certified for audit in audits)
    assert [audit.spec.learning_rate for audit in audits] == [
        Fraction(1, 75),
        Fraction(1, 83),
        Fraction(1, 90),
        Fraction(1, 95),
        Fraction(1, 120),
        Fraction(1, 150),
    ]


def test_primary_exact_sylvester_minors_are_frozen() -> None:
    primary = audit_all_pareto_certificates()[1].audit
    assert primary.storage_leading_minors == (
        Fraction(97, 125),
        Fraction(10_179, 250_000),
    )
    assert primary.negative_lmi_leading_minors == (
        Fraction(29_927_321_321_181_435_421_995_147, 924_625_928_192_000_000_000_000_000),
        Fraction(
            8_250_787_544_082_072_394_768_602_635_459_352_796_391,
            462_312_964_096_000_000_000_000_000_000_000_000_000_000,
        ),
        Fraction(
            1_391_032_153_976_146_286_203_283_830_581_420_859_889_968_429,
            288_945_602_560_000_000_000_000_000_000_000_000_000_000_000_000_000,
        ),
        Fraction(
            21_663_211_559_038_354_188_730_641_491_352_664_731_711_338_441_500_319,
            577_891_205_120_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000,
        ),
    )


def test_generic_sector_obstruction_is_exact_and_scoped() -> None:
    control = generic_sector_schur_obstruction()
    assert control.learning_rate == Fraction(1, 50)
    assert control.curvature == 10
    assert control.real_gain == LOCKED_SECTOR_CENTER
    assert control.skew_gain == LOCKED_SECTOR_RADIUS
    assert control.second_schur_margin == GENERIC_SECTOR_SCHUR_MARGIN < 0
    assert control.is_unstable


@pytest.mark.parametrize("precision_bits", [160, 224])
def test_cross_precision_canonical_fidelity_and_projection_close(precision_bits: int) -> None:
    checks = interval_certificate_checks(precision_bits=precision_bits)
    locked = evaluate_canonical_projected_fidelity(precision_bits=precision_bits)
    small = evaluate_canonical_projected_fidelity(
        SMALL_K_PROJECTION_GAIN,
        precision_bits=precision_bits,
    )

    assert checks
    assert all(checks.values())
    assert locked.projection_status == "inactive"
    assert locked.projection_scale == 1
    assert locked.best_scalar_departure > _arb_fraction(CANONICAL_DEPARTURE_GUARD[0])
    assert locked.best_scalar_departure < _arb_fraction(CANONICAL_DEPARTURE_GUARD[1])
    assert locked.upstream_shaping_retention > _arb_fraction(CANONICAL_RETENTION_GUARD[0])
    assert locked.upstream_shaping_retention < _arb_fraction(CANONICAL_RETENTION_GUARD[1])
    assert locked.output_to_upstream_amplitude > _arb_fraction(
        CANONICAL_UPSTREAM_AMPLITUDE_GUARD[0]
    )
    assert locked.output_to_upstream_amplitude < _arb_fraction(
        CANONICAL_UPSTREAM_AMPLITUDE_GUARD[1]
    )
    assert locked.output_to_input_amplitude > _arb_fraction(CANONICAL_INPUT_AMPLITUDE_GUARD[0])
    assert locked.output_to_input_amplitude < _arb_fraction(CANONICAL_INPUT_AMPLITUDE_GUARD[1])
    assert locked.best_scalar_departure > _arb_fraction(BEST_SCALAR_DEPARTURE_GATE)
    assert locked.upstream_shaping_retention > _arb_fraction(UPSTREAM_SHAPING_RETENTION_GATE)
    assert locked.decisions_are_definite
    assert locked.meaningful_fidelity_passes

    assert small.projection_status == "active"
    assert small.projection_scale < 1
    assert small.best_scalar_departure > _arb_fraction(SMALL_K_DEPARTURE_GUARD[0])
    assert small.best_scalar_departure < _arb_fraction(SMALL_K_DEPARTURE_GUARD[1])
    assert small.upstream_shaping_retention > _arb_fraction(SMALL_K_RETENTION_GUARD[0])
    assert small.upstream_shaping_retention < _arb_fraction(SMALL_K_RETENTION_GUARD[1])
    assert small.retention_fails
    assert not small.meaningful_fidelity_passes


def test_unprojected_huge_sector_is_a_diagnostic_not_a_structured_counterexample() -> None:
    control = unprojected_sector_control()
    assert control.lower == Fraction(125, 1_024)
    assert control.upper > 500
    assert control.effective_upper_step > Fraction(1, 80)


def test_interval_api_rejects_invalid_parameters() -> None:
    with pytest.raises(TypeError, match=r"fractions\.Fraction"):
        evaluate_canonical_projected_fidelity(1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="nonnegative"):
        evaluate_canonical_projected_fidelity(Fraction(-1))
    with pytest.raises(ValueError, match="at least 96"):
        evaluate_canonical_projected_fidelity(precision_bits=95)
    with pytest.raises(ValueError, match="frozen"):
        make_pl_certificate(
            type(PRIMARY_CERTIFICATE_SPEC)(
                "unfrozen",
                Fraction(1, 83),
                Fraction(999_799, 1_000_000),
                PRIMARY_CERTIFICATE_SPEC.storage,
                Fraction(1),
                Fraction(3_459, 500),
                Fraction(9, 250),
            )
        )
