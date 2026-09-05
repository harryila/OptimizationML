"""Exact P19 sector-shield and smooth-PL certificate obligations.

For the P18 pointwise disk sector ``[m, M]``, write

``gamma=(m+M)/2`` and ``r=(M-m)/2``.  Given an input ``S`` and any finite
candidate ``C``, P19 exposes the Euclidean projection of ``C`` onto

``D_S = {U : ||U-gamma*S||_F <= r*||S||_F}``.

For ``S != 0`` this projection is

``gamma*S + min(1, r*||S||_F/||C-gamma*S||_F)*(C-gamma*S)``,

with the multiplier defined to be one when the second norm is zero.  For
``S=0``, the set is the singleton ``{0}`` and the projection is zero.  The
ball condition is exactly equivalent to the origin-centred pointwise sector
supply ``<U-mS, MS-U>_F >= 0``.  Consequently every finite candidate is
shielded into the same P18 supply, while every exact P18 output is unchanged.

Projection onto a nonempty closed convex set is nonexpansive in its projected
argument.  Here the set depends on ``S``, so the modular statement is
``||Pi_{D_S}(C1)-Pi_{D_S}(C2)|| <= ||C1-C2||`` for each *fixed* ``S``.  This
module does not claim joint nonexpansiveness in ``(S,C)`` or an incremental
sector for the shielded nonlinear map.

In particular, because the exact P18 output ``T18(S)`` belongs to ``D_S``,
``||Pi_{D_S}(C)-T18(S)|| <= ||C-T18(S)||``.  This is the exact modular
fidelity lemma.  A P15 graph residual does not by itself bound the left-hand
side for a nonlinear approximate-candidate construction: an intervening
bound from graph residual to candidate error is still required.

All authoritative arithmetic below uses :class:`fractions.Fraction`.  The
general projection theorem is algebraic; ``project_rational_vector`` is only
an exact finite-dimensional witness helper and therefore requires the two
norms appearing in the formula to be rational.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

from passive_muon.pl_convergence import (
    PlConvergenceAudit,
    PlConvergenceCertificate,
    audit_pl_convergence,
)

SECTOR_SHIELDED_INEXACT_RESOLVENT_SCHEMA_VERSION = (
    "passive-muon-sector-shielded-inexact-resolvent-certificate-v1"
)

LOCKED_SECTOR_LOWER = Fraction(125, 1_024)
LOCKED_SECTOR_UPPER = Fraction(509, 512)
LOCKED_SECTOR_CENTER = Fraction(1_143, 2_048)
LOCKED_SECTOR_RADIUS = Fraction(893, 2_048)

LOCKED_BETA = Fraction(19, 20)
LOCKED_PL_CONSTANT = Fraction(1)
LOCKED_SMOOTHNESS = Fraction(10)

RationalVector = tuple[Fraction, ...]


@dataclass(frozen=True)
class ShieldOperatingPoint:
    """Frozen exact P18 certificate data replayed by P19."""

    name: str
    learning_rate: Fraction
    tau: Fraction
    storage: tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]
    function_storage: Fraction
    interpolation_reverse_weight: Fraction
    residual_multiplier: Fraction

    @property
    def rate_squared(self) -> Fraction:
        return self.tau**2

    @property
    def certified_lyapunov_rate_half_life(self) -> float:
        """Return the half-life of the certified geometric upper bound."""

        return math.log(Fraction(1, 2)) / math.log(float(self.rate_squared))


MAXIMUM_STEP_POINT = ShieldOperatingPoint(
    name="maximum_step_eta_1_over_83",
    learning_rate=Fraction(1, 83),
    tau=Fraction(999_799, 1_000_000),
    storage=(
        (Fraction(97, 125), Fraction(-151, 500)),
        (Fraction(-151, 500), Fraction(17, 100)),
    ),
    function_storage=Fraction(1),
    interpolation_reverse_weight=Fraction(3_459, 500),
    residual_multiplier=Fraction(9, 250),
)

FASTER_RATE_POINT = ShieldOperatingPoint(
    name="faster_rate_eta_1_over_120",
    learning_rate=Fraction(1, 120),
    tau=Fraction(24_987, 25_000),
    storage=(
        (Fraction(599, 1_000), Fraction(-231, 625)),
        (Fraction(-231, 625), Fraction(657, 2_500)),
    ),
    function_storage=Fraction(1),
    interpolation_reverse_weight=Fraction(2_177, 200),
    residual_multiplier=Fraction(53, 2_500),
)

OPERATING_POINTS = (MAXIMUM_STEP_POINT, FASTER_RATE_POINT)


@dataclass(frozen=True)
class ShieldCertificateAudit:
    point: ShieldOperatingPoint
    certificate: PlConvergenceCertificate
    audit: PlConvergenceAudit

    @property
    def certified(self) -> bool:
        return self.audit.certified


@dataclass(frozen=True)
class ExactShieldControl:
    """One exact rational projection control."""

    name: str
    source: RationalVector
    candidate: RationalVector
    projected: RationalVector
    candidate_supply: Fraction
    projected_supply: Fraction
    candidate_ball_slack: Fraction
    projected_ball_slack: Fraction
    projection_was_active: bool


def _dot(left: RationalVector, right: RationalVector) -> Fraction:
    if len(left) != len(right):
        raise ValueError("vector dimensions do not agree")
    return sum((x * y for x, y in zip(left, right, strict=True)), Fraction(0))


def _add(left: RationalVector, right: RationalVector) -> RationalVector:
    if len(left) != len(right):
        raise ValueError("vector dimensions do not agree")
    return tuple(x + y for x, y in zip(left, right, strict=True))


def _subtract(left: RationalVector, right: RationalVector) -> RationalVector:
    if len(left) != len(right):
        raise ValueError("vector dimensions do not agree")
    return tuple(x - y for x, y in zip(left, right, strict=True))


def _scale(scale: Fraction, vector: RationalVector) -> RationalVector:
    return tuple(scale * value for value in vector)


def squared_norm(vector: RationalVector) -> Fraction:
    """Return the exact squared Euclidean/Frobenius norm."""

    return _dot(vector, vector)


def _fraction_sqrt(value: Fraction) -> Fraction:
    if value < 0:
        raise ValueError("a squared norm cannot be negative")
    numerator = math.isqrt(value.numerator)
    denominator = math.isqrt(value.denominator)
    if numerator**2 != value.numerator or denominator**2 != value.denominator:
        raise ValueError("exact witness helper requires a rational norm")
    return Fraction(numerator, denominator)


def sector_supply(source: RationalVector, output: RationalVector) -> Fraction:
    """Return ``<output-m*source, M*source-output>`` exactly."""

    return _dot(
        _subtract(output, _scale(LOCKED_SECTOR_LOWER, source)),
        _subtract(_scale(LOCKED_SECTOR_UPPER, source), output),
    )


def shield_ball_slack(source: RationalVector, output: RationalVector) -> Fraction:
    """Return ``r^2||source||^2-||output-gamma*source||^2``."""

    displacement = _subtract(output, _scale(LOCKED_SECTOR_CENTER, source))
    return LOCKED_SECTOR_RADIUS**2 * squared_norm(source) - squared_norm(displacement)


def project_rational_vector(source: RationalVector, candidate: RationalVector) -> RationalVector:
    """Evaluate the shield exactly when its two relevant norms are rational.

    This helper is used only for exact witness and corruption controls.  The
    theorem itself applies to every finite real matrix, including cases with
    irrational norms.
    """

    if len(source) != len(candidate):
        raise ValueError("vector dimensions do not agree")
    source_norm = _fraction_sqrt(squared_norm(source))
    if source_norm == 0:
        return tuple(Fraction(0) for _ in source)
    center = _scale(LOCKED_SECTOR_CENTER, source)
    displacement = _subtract(candidate, center)
    displacement_norm = _fraction_sqrt(squared_norm(displacement))
    radius = LOCKED_SECTOR_RADIUS * source_norm
    if displacement_norm == 0 or displacement_norm <= radius:
        return candidate
    return _add(center, _scale(radius / displacement_norm, displacement))


def make_pl_certificate(point: ShieldOperatingPoint) -> PlConvergenceCertificate:
    """Instantiate one of the two frozen P18 operating-point certificates."""

    if point not in OPERATING_POINTS:
        raise ValueError("point must be one of the frozen P19 operating points")
    return PlConvergenceCertificate(
        pl_constant=LOCKED_PL_CONSTANT,
        smoothness=LOCKED_SMOOTHNESS,
        beta=LOCKED_BETA,
        center_gain=LOCKED_SECTOR_CENTER,
        residual_lipschitz=LOCKED_SECTOR_RADIUS,
        learning_rate=point.learning_rate,
        tau=point.tau,
        storage=point.storage,
        function_storage=point.function_storage,
        interpolation_reverse_weight=point.interpolation_reverse_weight,
        lambda_residual_norm=point.residual_multiplier,
    )


def audit_operating_point(point: ShieldOperatingPoint) -> ShieldCertificateAudit:
    certificate = make_pl_certificate(point)
    return ShieldCertificateAudit(point, certificate, audit_pl_convergence(certificate))


def audit_operating_points() -> tuple[ShieldCertificateAudit, ShieldCertificateAudit]:
    return tuple(audit_operating_point(point) for point in OPERATING_POINTS)  # type: ignore[return-value]


def exact_shield_controls() -> tuple[ExactShieldControl, ...]:
    """Return exact zero, identity, radial, and tangential controls."""

    source = (Fraction(3), Fraction(4))
    perpendicular = (Fraction(-4), Fraction(3))
    center = _scale(LOCKED_SECTOR_CENTER, source)

    cases = (
        (
            "zero_input_singleton",
            (Fraction(0), Fraction(0)),
            (Fraction(7), Fraction(-11)),
        ),
        (
            "interior_identity",
            source,
            _add(center, _scale(LOCKED_SECTOR_RADIUS / 2, perpendicular)),
        ),
        (
            "radial_corruption",
            source,
            _scale(LOCKED_SECTOR_UPPER + 1, source),
        ),
        (
            "tangential_corruption",
            source,
            _add(center, _scale(2 * LOCKED_SECTOR_RADIUS, perpendicular)),
        ),
    )
    controls = []
    for name, selected_source, candidate in cases:
        projected = project_rational_vector(selected_source, candidate)
        controls.append(
            ExactShieldControl(
                name=name,
                source=selected_source,
                candidate=candidate,
                projected=projected,
                candidate_supply=sector_supply(selected_source, candidate),
                projected_supply=sector_supply(selected_source, projected),
                candidate_ball_slack=shield_ball_slack(selected_source, candidate),
                projected_ball_slack=shield_ball_slack(selected_source, projected),
                projection_was_active=projected != candidate,
            )
        )
    return tuple(controls)


def exact_certificate_checks() -> dict[str, bool]:
    """Replay all exact P19 geometry, controls, and smooth-PL certificates."""

    audits = audit_operating_points()
    controls = {control.name: control for control in exact_shield_controls()}
    source = (Fraction(3), Fraction(4))
    center = _scale(LOCKED_SECTOR_CENTER, source)
    exact_admissible_reference = controls["interior_identity"].candidate
    corrupted_candidate = controls["tangential_corruption"].candidate
    shielded_corrupted = controls["tangential_corruption"].projected
    second_candidate = _add(
        center,
        _scale(-LOCKED_SECTOR_RADIUS / 2, (Fraction(-4), Fraction(3))),
    )
    shielded_second = project_rational_vector(source, second_candidate)
    return {
        "sector_center_and_radius_are_exact": (
            (LOCKED_SECTOR_LOWER + LOCKED_SECTOR_UPPER) / 2 == LOCKED_SECTOR_CENTER
            and (LOCKED_SECTOR_UPPER - LOCKED_SECTOR_LOWER) / 2 == LOCKED_SECTOR_RADIUS
        ),
        "disk_slack_equals_sector_supply": all(
            control.candidate_supply == control.candidate_ball_slack
            and control.projected_supply == control.projected_ball_slack
            for control in controls.values()
        ),
        "zero_input_projects_every_candidate_to_zero": controls["zero_input_singleton"].projected
        == (0, 0),
        "interior_candidate_is_unchanged": (
            not controls["interior_identity"].projection_was_active
            and controls["interior_identity"].projected == controls["interior_identity"].candidate
            and controls["interior_identity"].projected_supply > 0
        ),
        "radial_corruption_is_repaired_to_boundary": (
            controls["radial_corruption"].candidate_supply < 0
            and controls["radial_corruption"].projection_was_active
            and controls["radial_corruption"].projected_supply == 0
        ),
        "tangential_corruption_is_repaired_to_boundary": (
            controls["tangential_corruption"].candidate_supply < 0
            and controls["tangential_corruption"].projection_was_active
            and controls["tangential_corruption"].projected_supply == 0
        ),
        "fixed_input_projection_control_is_nonexpansive": squared_norm(
            _subtract(shielded_corrupted, shielded_second)
        )
        <= squared_norm(_subtract(corrupted_candidate, second_candidate)),
        "shield_does_not_increase_error_to_admissible_exact_output": squared_norm(
            _subtract(shielded_corrupted, exact_admissible_reference)
        )
        <= squared_norm(_subtract(corrupted_candidate, exact_admissible_reference)),
        "shielded_finite_controls_all_lie_in_sector": all(
            control.projected_supply >= 0 for control in controls.values()
        ),
        "both_P18_operating_point_LMIs_replay_exactly": all(audit.certified for audit in audits),
        "maximum_step_rate_is_exact": (
            MAXIMUM_STEP_POINT.rate_squared == Fraction(999_598_040_401, 1_000_000_000_000)
        ),
        "faster_rate_is_exact": (
            FASTER_RATE_POINT.rate_squared == Fraction(624_350_169, 625_000_000)
        ),
        "eta_1_over_120_has_better_certified_rate": (
            FASTER_RATE_POINT.rate_squared < MAXIMUM_STEP_POINT.rate_squared
        ),
    }


__all__ = [
    "FASTER_RATE_POINT",
    "LOCKED_BETA",
    "LOCKED_PL_CONSTANT",
    "LOCKED_SECTOR_CENTER",
    "LOCKED_SECTOR_LOWER",
    "LOCKED_SECTOR_RADIUS",
    "LOCKED_SECTOR_UPPER",
    "LOCKED_SMOOTHNESS",
    "MAXIMUM_STEP_POINT",
    "OPERATING_POINTS",
    "SECTOR_SHIELDED_INEXACT_RESOLVENT_SCHEMA_VERSION",
    "ExactShieldControl",
    "ShieldCertificateAudit",
    "ShieldOperatingPoint",
    "audit_operating_point",
    "audit_operating_points",
    "exact_certificate_checks",
    "exact_shield_controls",
    "make_pl_certificate",
    "project_rational_vector",
    "sector_supply",
    "shield_ball_slack",
    "squared_norm",
]
