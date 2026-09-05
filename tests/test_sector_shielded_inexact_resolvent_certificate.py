from __future__ import annotations

from fractions import Fraction

import pytest

from passive_muon.sector_shielded_inexact_resolvent_certificate import (
    FASTER_RATE_POINT,
    LOCKED_SECTOR_CENTER,
    LOCKED_SECTOR_LOWER,
    LOCKED_SECTOR_RADIUS,
    LOCKED_SECTOR_UPPER,
    MAXIMUM_STEP_POINT,
    audit_operating_points,
    exact_certificate_checks,
    exact_shield_controls,
    make_pl_certificate,
    project_rational_vector,
    sector_supply,
    shield_ball_slack,
    squared_norm,
)


def test_all_exact_p19_certificate_checks_pass() -> None:
    checks = exact_certificate_checks()
    assert checks
    assert all(checks.values())


def test_locked_disk_is_exactly_the_p18_sector() -> None:
    assert Fraction(125, 1_024) == LOCKED_SECTOR_LOWER
    assert Fraction(509, 512) == LOCKED_SECTOR_UPPER
    assert Fraction(1_143, 2_048) == LOCKED_SECTOR_CENTER
    assert Fraction(893, 2_048) == LOCKED_SECTOR_RADIUS
    assert LOCKED_SECTOR_CENTER == (LOCKED_SECTOR_LOWER + LOCKED_SECTOR_UPPER) / 2
    assert LOCKED_SECTOR_RADIUS == (LOCKED_SECTOR_UPPER - LOCKED_SECTOR_LOWER) / 2


@pytest.mark.parametrize(
    ("source", "output"),
    [
        ((Fraction(0), Fraction(0)), (Fraction(3), Fraction(-7))),
        ((Fraction(3), Fraction(4)), (Fraction(0), Fraction(0))),
        ((Fraction(3), Fraction(4)), (Fraction(2), Fraction(-1))),
        ((Fraction(-5), Fraction(12)), (Fraction(7), Fraction(9))),
    ],
)
def test_disk_slack_and_sector_supply_are_identical(
    source: tuple[Fraction, ...],
    output: tuple[Fraction, ...],
) -> None:
    assert shield_ball_slack(source, output) == sector_supply(source, output)


def test_exact_controls_cover_zero_identity_and_corruptions() -> None:
    controls = {control.name: control for control in exact_shield_controls()}

    zero = controls["zero_input_singleton"]
    assert zero.projected == (0, 0)
    assert zero.projection_was_active
    assert zero.projected_supply == 0

    interior = controls["interior_identity"]
    assert interior.projected == interior.candidate
    assert not interior.projection_was_active
    assert interior.projected_supply > 0

    for name in ("radial_corruption", "tangential_corruption"):
        control = controls[name]
        assert control.candidate_supply < 0
        assert control.projection_was_active
        assert control.projected_supply == 0


def test_projection_is_nonexpansive_and_preserves_admissible_reference() -> None:
    source = (Fraction(3), Fraction(4))
    perpendicular = (Fraction(-4), Fraction(3))
    center = tuple(LOCKED_SECTOR_CENTER * value for value in source)
    candidate_one = tuple(
        center[index] + 2 * LOCKED_SECTOR_RADIUS * perpendicular[index] for index in range(2)
    )
    candidate_two = tuple(
        center[index] - LOCKED_SECTOR_RADIUS * perpendicular[index] / 2 for index in range(2)
    )
    projected_one = project_rational_vector(source, candidate_one)
    projected_two = project_rational_vector(source, candidate_two)

    before = tuple(x - y for x, y in zip(candidate_one, candidate_two, strict=True))
    after = tuple(x - y for x, y in zip(projected_one, projected_two, strict=True))
    assert squared_norm(after) <= squared_norm(before)
    assert projected_two == candidate_two


def test_both_p18_operating_points_replay_exactly() -> None:
    maximum, faster = audit_operating_points()
    assert maximum.certified
    assert faster.certified
    assert maximum.point.learning_rate == Fraction(1, 83)
    assert maximum.point.rate_squared == Fraction(999_598_040_401, 1_000_000_000_000)
    assert faster.point.learning_rate == Fraction(1, 120)
    assert faster.point.rate_squared == Fraction(624_350_169, 625_000_000)
    assert faster.point.rate_squared < maximum.point.rate_squared
    assert 1_724 < maximum.point.certified_lyapunov_rate_half_life < 1_725
    assert 666 < faster.point.certified_lyapunov_rate_half_life < 667


def test_both_exact_sylvester_replays_match_frozen_minors() -> None:
    maximum, faster = audit_operating_points()
    assert maximum.audit.storage_leading_minors == (
        Fraction(97, 125),
        Fraction(10_179, 250_000),
    )
    assert faster.audit.storage_leading_minors == (
        Fraction(599, 1_000),
        Fraction(260_163, 12_500_000),
    )
    assert len(maximum.audit.negative_lmi_leading_minors) == 4
    assert len(faster.audit.negative_lmi_leading_minors) == 4
    assert all(value > 0 for value in maximum.audit.negative_lmi_leading_minors)
    assert all(value > 0 for value in faster.audit.negative_lmi_leading_minors)


def test_exact_witness_helper_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        project_rational_vector((Fraction(1),), (Fraction(1), Fraction(2)))
    with pytest.raises(ValueError, match="rational norm"):
        project_rational_vector((Fraction(1), Fraction(1)), (Fraction(3), Fraction(4)))
    with pytest.raises(ValueError, match="frozen"):
        make_pl_certificate(
            type(MAXIMUM_STEP_POINT)(
                name="not_frozen",
                learning_rate=MAXIMUM_STEP_POINT.learning_rate,
                tau=MAXIMUM_STEP_POINT.tau,
                storage=MAXIMUM_STEP_POINT.storage,
                function_storage=MAXIMUM_STEP_POINT.function_storage,
                interpolation_reverse_weight=MAXIMUM_STEP_POINT.interpolation_reverse_weight,
                residual_multiplier=MAXIMUM_STEP_POINT.residual_multiplier,
            )
        )


def test_named_points_are_distinct_and_locked() -> None:
    assert MAXIMUM_STEP_POINT != FASTER_RATE_POINT
    assert make_pl_certificate(MAXIMUM_STEP_POINT).learning_rate == Fraction(1, 83)
    assert make_pl_certificate(FASTER_RATE_POINT).learning_rate == Fraction(1, 120)
