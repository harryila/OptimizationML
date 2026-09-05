from __future__ import annotations

from fractions import Fraction

import pytest

from passive_muon.certified_outer_loop_composition_certificate import (
    PORT_VARIABLE_ORDER,
    PRIMARY_PORT_GAINS,
    SECONDARY_PORT_GAINS,
    AffinePortEnvelope,
    absorb_affine_ports,
    audit_stored_signal_port_certificate,
    build_stored_signal_port_certificate,
    centered_decay_corollary,
    exact_certificate_checks,
    zero_centered_decay_counterexample,
)
from passive_muon.sector_shielded_inexact_resolvent_certificate import (
    FASTER_RATE_POINT,
    MAXIMUM_STEP_POINT,
)


def test_p21_exact_checks_all_close() -> None:
    assert all(exact_certificate_checks().values())


@pytest.mark.parametrize(
    ("point", "gain", "rate"),
    (
        (FASTER_RATE_POINT, PRIMARY_PORT_GAINS, Fraction(624_350_169, 625_000_000)),
        (MAXIMUM_STEP_POINT, SECONDARY_PORT_GAINS, Fraction(999_598_040_401, 10**12)),
    ),
)
def test_stored_signal_certificates_are_exact_and_strict(point, gain, rate) -> None:
    audit = audit_stored_signal_port_certificate(point)
    certificate = audit.certificate

    assert len(PORT_VARIABLE_ORDER) == 7
    assert certificate.port_gains == gain
    assert certificate.rate == rate
    assert len(certificate.lmi) == 7
    assert all(len(row) == 7 for row in certificate.lmi)
    assert all(
        certificate.lmi[row][column] == certificate.lmi[column][row]
        for row in range(7)
        for column in range(7)
    )
    assert all(isinstance(value, Fraction) for row in certificate.lmi for value in row)
    assert audit.storage_positive_definite
    assert audit.lmi_negative_definite
    assert audit.zero_port_recovers_p18_exactly
    assert audit.function_values_cancel
    assert audit.certified
    assert len(audit.negative_lmi_leading_minors) == 7
    assert all(value > 0 for value in audit.negative_lmi_leading_minors)


def test_port_penalty_changes_only_the_three_port_diagonals() -> None:
    certificate = build_stored_signal_port_certificate(FASTER_RATE_POINT)
    for row in range(7):
        for column in range(7):
            expected_difference = (
                certificate.port_gains[row - 4] if row == column and row >= 4 else Fraction(0)
            )
            assert certificate.raw_lmi[row][column] - certificate.lmi[row][column] == (
                expected_difference
            )


def test_zero_affine_ports_recover_the_frozen_rate_without_forcing() -> None:
    certificate = build_stored_signal_port_certificate(FASTER_RATE_POINT)
    zero = AffinePortEnvelope(Fraction(0), Fraction(0))
    absorption = absorb_affine_ports(
        certificate,
        (zero, zero, zero),
        (Fraction(1), Fraction(1), Fraction(1)),
    )

    assert absorption.base_rate == certificate.rate
    assert absorption.absorbed_rate == certificate.rate
    assert absorption.constant_forcing == 0
    assert absorption.objective_gap_ultimate_bound == 0
    assert absorption.contractive
    assert absorption.forward_invariant


def test_affine_port_absorption_is_exact_young_arithmetic() -> None:
    certificate = build_stored_signal_port_certificate(FASTER_RATE_POINT)
    envelopes = (
        AffinePortEnvelope(Fraction(1, 10_000), Fraction(1, 1_000_000)),
        AffinePortEnvelope(Fraction(1, 20_000), Fraction(1, 2_000_000)),
        AffinePortEnvelope(Fraction(1, 40_000), Fraction(1, 4_000_000)),
    )
    young = (Fraction(1), Fraction(3), Fraction(7))
    absorption = absorb_affine_ports(certificate, envelopes, young)
    expected_rate = certificate.rate + sum(
        gain * (1 + theta) * envelope.slope**2
        for gain, envelope, theta in zip(certificate.port_gains, envelopes, young, strict=True)
    )
    expected_forcing = sum(
        gain * (1 + 1 / theta) * envelope.intercept**2
        for gain, envelope, theta in zip(certificate.port_gains, envelopes, young, strict=True)
    )

    assert absorption.absorbed_rate == expected_rate
    assert absorption.constant_forcing == expected_forcing
    assert absorption.objective_gap_ultimate_bound == (10 * expected_forcing / (1 - expected_rate))


def test_centered_decay_uses_exact_inverse_storage_gradient_bound() -> None:
    certificate = build_stored_signal_port_certificate(FASTER_RATE_POINT)
    result = centered_decay_corollary(
        certificate,
        weight_decay=Fraction(1, 1_000_000),
        strong_convexity=Fraction(1),
    )
    storage = certificate.pl_certificate.storage
    determinant = storage[0][0] * storage[1][1] - storage[0][1] ** 2
    expected_squared_slope = (
        (Fraction(1, 1_000_000) / certificate.pl_certificate.center_gain) ** 2
        * storage[0][0]
        / determinant
    )

    assert result.normalized_output_squared_slope == expected_squared_slope
    assert result.certified_rate == (
        certificate.rate + certificate.port_gains[2] * expected_squared_slope
    )
    assert result.contractive


def test_ordinary_zero_centered_decay_moves_a_nonzero_minimizer() -> None:
    control = zero_centered_decay_counterexample()

    assert control.next_iterate == Fraction(11_999, 12_000)
    assert control.displacement == Fraction(-1, 12_000)
    assert control.next_objective_gap == Fraction(1, 288_000_000)
    assert control.original_minimizer_is_not_an_equilibrium


def test_affine_and_decay_inputs_fail_closed_on_invalid_values() -> None:
    certificate = build_stored_signal_port_certificate(FASTER_RATE_POINT)
    zero = AffinePortEnvelope(Fraction(0), Fraction(0))
    with pytest.raises(ValueError, match="Young"):
        absorb_affine_ports(
            certificate,
            (zero, zero, zero),
            (Fraction(1), Fraction(0), Fraction(1)),
        )
    with pytest.raises(ValueError, match="nonnegative"):
        AffinePortEnvelope(Fraction(-1), Fraction(0))
    with pytest.raises(ValueError, match="positive"):
        centered_decay_corollary(
            certificate,
            weight_decay=Fraction(1, 100),
            strong_convexity=Fraction(0),
        )
