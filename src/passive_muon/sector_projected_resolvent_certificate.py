"""Exact and Arb certificate obligations for the P18 projected interface.

P18 replaces the raw shape branch of P17 by its radial projection into the
origin-centred disk sector ``[0,K]``.  At the locked values

``K=1, c=1024, h=3/4``

the complete exact-real interface lies in the dimension-uniform *pointwise*
sector ``[125/1024, 509/512]``.  That pointwise supply is sufficient for the
smooth-PL function-value certificate replayed below.  It is not an
incremental sector or an arbitrary-pair contraction statement.

This module deliberately does not import the P18 runtime implementation.  It
reconstructs the sector, six rational PL certificates, the generic-sector
Schur--Cohn obstruction, and an outward-rounded canonical fidelity proof from
the frozen P12--P17 constants.  The Arb calculation inflates the frozen P16
binary64 resolvent candidate by its recomputed graph residual; it is not an
IEEE-754 error certificate for the solver or projection implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

from flint import arb, ctx, fmpq

from passive_muon.pl_convergence import (
    PlConvergenceAudit,
    PlConvergenceCertificate,
    audit_pl_convergence,
)
from passive_muon.specs import JORDAN_QUINTIC

SECTOR_PROJECTED_RESOLVENT_SCHEMA_VERSION = "passive-muon-sector-projected-resolvent-certificate-v1"

# Locked exact-real P12--P17 operator data, repeated rather than imported from
# the runtime P18 module.
LOCKED_EPSILON = Fraction(1, 10_000_000)
LOCKED_LAMBDA = Fraction(1, 1_000)
LOCKED_MU = Fraction(1_000)
LOCKED_BETA = Fraction(19, 20)
LOCKED_PL_CONSTANT = Fraction(1)
LOCKED_SMOOTHNESS = Fraction(10)
LOCKED_P15_KAPPA = Fraction(1, 250)
LOCKED_PROJECTION_GAIN = Fraction(1)
LOCKED_PASSIVE_DIVISOR = Fraction(1_024)
LOCKED_GATE_CAP = Fraction(3, 4)
LOCKED_GATE_Q0 = Fraction(1, 4)
LOCKED_GATE_Q1 = Fraction(1)

YOSIDA_GAIN_LOWER = Fraction(500)
YOSIDA_GAIN_UPPER = Fraction(1_000)
LOCKED_SECTOR_LOWER = Fraction(125, 1_024)
LOCKED_SECTOR_UPPER = Fraction(509, 512)
LOCKED_SECTOR_CENTER = Fraction(1_143, 2_048)
LOCKED_SECTOR_RADIUS = Fraction(893, 2_048)

P14_RATE_SQUARED = Fraction(249_001, 250_000)
PRIMARY_EFFECTIVE_STEP_LOWER = Fraction(1, 1_000)
PRIMARY_EFFECTIVE_STEP_UPPER = Fraction(1, 80)

BEST_SCALAR_DEPARTURE_GATE = Fraction(1, 1_000)
UPSTREAM_SHAPING_RETENTION_GATE = Fraction(1, 10)
CANONICAL_INPUT = (Fraction(3), Fraction(4))
CANONICAL_APPROXIMATE_ROOT_HEX = (
    "0x1.705228c08d605p-1",
    "0x1.eb135495cac74p-1",
)
CANONICAL_DEPARTURE_GUARD = (Fraction(18_141, 500_000), Fraction(9_071, 250_000))
CANONICAL_RETENTION_GUARD = (Fraction(1_037, 2_000), Fraction(5_187, 10_000))
CANONICAL_UPSTREAM_AMPLITUDE_GUARD = (Fraction(289, 200), Fraction(1_447, 1_000))
CANONICAL_INPUT_AMPLITUDE_GUARD = (Fraction(77, 200), Fraction(193, 500))
SMALL_K_DEPARTURE_GUARD = (Fraction(543, 200_000), Fraction(2_717, 1_000_000))
SMALL_K_RETENTION_GUARD = (Fraction(97, 2_500), Fraction(389, 10_000))
SMALL_K_PROJECTION_GAIN = Fraction(1, 100)

JORDAN_DERIVATIVE_UPPER = Fraction(4_848_763, 10_000)
UNIT_DEFICIT_UPPER = Fraction(
    6_602_082_433_275_499_863,
    41_641_817_600_000_000,
)
TAIL_DERIVATIVE_LOWER = Fraction(-199_437, 1_250)
TAIL_PROJECTION_LOWER = Fraction(-41_528_474_059_081, 260_261_360_000)
SWITCH_RAW_RATIO = Fraction(63, 9_937)
RAW_SHAPE_GAIN_UPPER = Fraction(
    20_191_130_443_162_880_000_000,
    26_793_221_204_801_899_863,
)

GENERIC_SECTOR_FAILED_STEP = Fraction(1, 50)
GENERIC_SECTOR_SCHUR_MARGIN = Fraction(
    -3_768_360_579_178_620_269,
    1_759_218_604_441_600_000_000_000,
)


@dataclass(frozen=True)
class ParetoCertificateSpec:
    """Exact scalar data for one pointwise-sector smooth-PL certificate."""

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
    def half_life(self) -> float:
        return math.log(0.5) / math.log(float(self.rate_squared))


PARETO_CERTIFICATES = (
    ParetoCertificateSpec(
        "eta_1_over_75",
        Fraction(1, 75),
        Fraction(99_995, 100_000),
        ((Fraction(417, 500), Fraction(-113, 400)), (Fraction(-113, 400), Fraction(299, 2_000))),
        Fraction(1),
        Fraction(12_009, 2_000),
        Fraction(83, 2_000),
    ),
    ParetoCertificateSpec(
        "primary_eta_1_over_83",
        Fraction(1, 83),
        Fraction(999_799, 1_000_000),
        ((Fraction(97, 125), Fraction(-151, 500)), (Fraction(-151, 500), Fraction(17, 100))),
        Fraction(1),
        Fraction(3_459, 500),
        Fraction(9, 250),
    ),
    ParetoCertificateSpec(
        "eta_1_over_90",
        Fraction(1, 90),
        Fraction(3_999, 4_000),
        ((Fraction(183, 250), Fraction(-79, 250)), (Fraction(-79, 250), Fraction(47, 250))),
        Fraction(1),
        Fraction(767, 100),
        Fraction(4, 125),
    ),
    ParetoCertificateSpec(
        "eta_1_over_95",
        Fraction(1, 95),
        Fraction(9_997, 10_000),
        ((Fraction(88, 125), Fraction(-163, 500)), (Fraction(-163, 500), Fraction(201, 1_000))),
        Fraction(1),
        Fraction(4_107, 500),
        Fraction(3, 100),
    ),
    ParetoCertificateSpec(
        "pareto_eta_1_over_120",
        Fraction(1, 120),
        Fraction(24_987, 25_000),
        ((Fraction(599, 1_000), Fraction(-231, 625)), (Fraction(-231, 625), Fraction(657, 2_500))),
        Fraction(1),
        Fraction(2_177, 200),
        Fraction(53, 2_500),
    ),
    ParetoCertificateSpec(
        "eta_1_over_150",
        Fraction(1, 150),
        Fraction(4_997, 5_000),
        (
            (Fraction(5_133, 10_000), Fraction(-407, 1_000)),
            (Fraction(-407, 1_000), Fraction(663, 2_000)),
        ),
        Fraction(1),
        Fraction(27_489, 2_000),
        Fraction(153, 10_000),
    ),
)
PRIMARY_CERTIFICATE_SPEC = PARETO_CERTIFICATES[1]


@dataclass(frozen=True)
class SectorProjectedCertificateAudit:
    """One exact PL certificate and its finite rational audit."""

    spec: ParetoCertificateSpec
    certificate: PlConvergenceCertificate
    audit: PlConvergenceAudit

    @property
    def certified(self) -> bool:
        return self.audit.certified


@dataclass(frozen=True)
class GenericSectorSchurObstruction:
    """Exact Schur--Cohn failure for the complete abstract disk sector."""

    learning_rate: Fraction
    curvature: Fraction
    real_gain: Fraction
    skew_gain: Fraction
    trace: tuple[Fraction, Fraction]
    determinant: tuple[Fraction, Fraction]
    first_coefficient: tuple[Fraction, Fraction]
    zeroth_coefficient: tuple[Fraction, Fraction]
    second_schur_margin: Fraction

    @property
    def is_unstable(self) -> bool:
        return self.second_schur_margin < 0


@dataclass(frozen=True)
class CanonicalProjectedFidelityEnclosure:
    """Outward-rounded exact-output enclosure for canonical ``diag(3,4)``."""

    precision_bits: int
    projection_gain: Fraction
    graph_residual_norm: arb
    exact_root_error_norm_upper: arb
    exact_yosida_error_norm_upper: arb
    projection_scale: arb
    projection_status: str
    output_singular_values: tuple[arb, arb]
    best_scalar_departure: arb
    upstream_best_scalar_departure: arb
    upstream_shaping_retention: arb
    output_to_upstream_amplitude: arb
    output_to_input_amplitude: arb
    departure_passes: bool
    departure_fails: bool
    retention_passes: bool
    retention_fails: bool

    @property
    def decisions_are_definite(self) -> bool:
        return self.departure_passes != self.departure_fails and (
            self.retention_passes != self.retention_fails
        )

    @property
    def meaningful_fidelity_passes(self) -> bool:
        return self.departure_passes and self.retention_passes


@dataclass(frozen=True)
class UnprojectedSectorControl:
    """Exact huge-sector diagnostic for the unprojected P17 shape branch."""

    lower: Fraction
    upper: Fraction
    selected_step: Fraction
    effective_upper_step: Fraction


def make_pl_certificate(spec: ParetoCertificateSpec) -> PlConvergenceCertificate:
    """Instantiate one exact certificate from the frozen Pareto data."""

    if spec not in PARETO_CERTIFICATES:
        raise ValueError("spec must be one of the frozen P18 Pareto points")
    return PlConvergenceCertificate(
        pl_constant=LOCKED_PL_CONSTANT,
        smoothness=LOCKED_SMOOTHNESS,
        beta=LOCKED_BETA,
        center_gain=LOCKED_SECTOR_CENTER,
        residual_lipschitz=LOCKED_SECTOR_RADIUS,
        learning_rate=spec.learning_rate,
        tau=spec.tau,
        storage=spec.storage,
        function_storage=spec.function_storage,
        interpolation_reverse_weight=spec.interpolation_reverse_weight,
        lambda_residual_norm=spec.residual_multiplier,
    )


def audit_pareto_certificate(spec: ParetoCertificateSpec) -> SectorProjectedCertificateAudit:
    certificate = make_pl_certificate(spec)
    return SectorProjectedCertificateAudit(spec, certificate, audit_pl_convergence(certificate))


def audit_all_pareto_certificates() -> tuple[SectorProjectedCertificateAudit, ...]:
    return tuple(audit_pareto_certificate(spec) for spec in PARETO_CERTIFICATES)


def _complex_add(
    left: tuple[Fraction, Fraction], right: tuple[Fraction, Fraction]
) -> tuple[Fraction, Fraction]:
    return left[0] + right[0], left[1] + right[1]


def _complex_multiply(
    left: tuple[Fraction, Fraction], right: tuple[Fraction, Fraction]
) -> tuple[Fraction, Fraction]:
    return left[0] * right[0] - left[1] * right[1], left[0] * right[1] + left[1] * right[0]


def _complex_conjugate(value: tuple[Fraction, Fraction]) -> tuple[Fraction, Fraction]:
    return value[0], -value[1]


def _complex_norm_squared(value: tuple[Fraction, Fraction]) -> Fraction:
    return value[0] ** 2 + value[1] ** 2


def generic_sector_schur_obstruction() -> GenericSectorSchurObstruction:
    """Return the exact ``eta=1/50`` obstruction for the abstract sector.

    The admissible endpoint is ``T=gamma I+KQ`` on a two-dimensional block,
    where ``Q.T=-Q`` and ``Q.T@Q=I``.  It satisfies the complete centred disk
    supply exactly.  This proves that the *generic pointwise-sector model* at
    the predeclared step is unstable.  It is deliberately not presented as a
    realized Jacobian or trajectory of the structured P18 operator.
    """

    eta = GENERIC_SECTOR_FAILED_STEP
    beta = LOCKED_BETA
    curvature = LOCKED_SMOOTHNESS
    gain = (LOCKED_SECTOR_CENTER, LOCKED_SECTOR_RADIUS)
    trace = _complex_add(
        (1 + beta, Fraction(0)),
        (
            -eta * (1 - beta**2) * curvature * gain[0],
            -eta * (1 - beta**2) * curvature * gain[1],
        ),
    )
    determinant = _complex_add(
        (beta, Fraction(0)),
        (
            -eta * beta * (1 - beta) * curvature * gain[0],
            -eta * beta * (1 - beta) * curvature * gain[1],
        ),
    )
    first = (-trace[0], -trace[1])
    product = _complex_multiply(_complex_conjugate(first), determinant)
    difference = (first[0] - product[0], first[1] - product[1])
    margin = (1 - _complex_norm_squared(determinant)) ** 2 - _complex_norm_squared(difference)
    return GenericSectorSchurObstruction(
        learning_rate=eta,
        curvature=curvature,
        real_gain=gain[0],
        skew_gain=gain[1],
        trace=trace,
        determinant=determinant,
        first_coefficient=first,
        zeroth_coefficient=determinant,
        second_schur_margin=margin,
    )


def unprojected_sector_control() -> UnprojectedSectorControl:
    lower = LOCKED_SECTOR_LOWER
    upper = max(
        YOSIDA_GAIN_UPPER / LOCKED_PASSIVE_DIVISOR,
        (1 - LOCKED_GATE_CAP) * YOSIDA_GAIN_UPPER / LOCKED_PASSIVE_DIVISOR
        + LOCKED_GATE_CAP * RAW_SHAPE_GAIN_UPPER,
    )
    return UnprojectedSectorControl(lower, upper, GENERIC_SECTOR_FAILED_STEP, upper / 50)


def _arb_fraction(value: Fraction) -> arb:
    return arb(fmpq(value.numerator, value.denominator))


def _arb_jordan(value: arb) -> arb:
    a, b, c = (_arb_fraction(item) for item in JORDAN_QUINTIC.fractions())
    result = value
    for _ in range(5):
        square = result * result
        result *= a + square * (b + c * square)
    return result


def _arb_radial_integral(raw_ratio: arb) -> arb:
    switch = _arb_fraction(SWITCH_RAW_RATIO)
    if raw_ratio <= switch:
        return _arb_fraction(UNIT_DEFICIT_UPPER) * raw_ratio
    derivative = _arb_fraction(TAIL_DERIVATIVE_LOWER)
    projection = _arb_fraction(TAIL_PROJECTION_LOWER)
    return (
        _arb_fraction(UNIT_DEFICIT_UPPER) * switch
        - projection * ((1 + raw_ratio).log() - (1 + switch).log())
        + (derivative - projection) * (1 / (1 + raw_ratio) - 1 / (1 + switch))
    )


def _best_scalar_departure(output: tuple[arb, arb], source: tuple[arb, arb]) -> arb:
    determinant = abs(output[0] * source[1] - output[1] * source[0])
    return determinant / (
        (source[0] ** 2 + source[1] ** 2).sqrt() * (output[0] ** 2 + output[1] ** 2).sqrt()
    )


def canonical_approximate_root() -> tuple[Fraction, Fraction]:
    return tuple(
        Fraction.from_float(float.fromhex(value)) for value in CANONICAL_APPROXIMATE_ROOT_HEX
    )


def evaluate_canonical_projected_fidelity(
    projection_gain: Fraction = LOCKED_PROJECTION_GAIN,
    *,
    precision_bits: int = 192,
) -> CanonicalProjectedFidelityEnclosure:
    """Enclose the exact projected interface on canonical ``diag(3,4)``."""

    if not isinstance(projection_gain, Fraction):
        raise TypeError("projection_gain must be a fractions.Fraction")
    if projection_gain < 0:
        raise ValueError("projection_gain must be nonnegative")
    if precision_bits < 96:
        raise ValueError("precision_bits must be at least 96")

    with ctx.workprec(precision_bits):
        source = tuple(_arb_fraction(value) for value in CANONICAL_INPUT)
        approximate = tuple(_arb_fraction(value) for value in canonical_approximate_root())
        radius = (approximate[0] ** 2 + approximate[1] ** 2).sqrt()
        epsilon = _arb_fraction(LOCKED_EPSILON)
        normalized = tuple(value / (radius + epsilon) for value in approximate)
        jordan = tuple(_arb_jordan(value) for value in normalized)
        primitive = _arb_radial_integral(radius / epsilon)
        graph_output = tuple(
            response + (primitive / radius + _arb_fraction(LOCKED_MU)) * value
            for response, value in zip(jordan, approximate, strict=True)
        )
        residual = tuple(
            sigma - value - _arb_fraction(LOCKED_LAMBDA) * output
            for sigma, value, output in zip(source, approximate, graph_output, strict=True)
        )
        residual_norm = (residual[0] ** 2 + residual[1] ** 2).sqrt()
        root_error = residual_norm / 2
        yosida_error = 500 * residual_norm

        exact_root = tuple(value + arb(0, root_error) for value in approximate)
        exact_radius = (exact_root[0] ** 2 + exact_root[1] ** 2).sqrt()
        exact_shape = tuple(_arb_jordan(value / (exact_radius + epsilon)) for value in exact_root)
        approximate_yosida = tuple(
            (sigma - value) / _arb_fraction(LOCKED_LAMBDA)
            for sigma, value in zip(source, approximate, strict=True)
        )
        exact_yosida = tuple(value + arb(0, yosida_error) for value in approximate_yosida)

        inner = sum(value * sigma for value, sigma in zip(exact_shape, source, strict=True))
        shape_norm_squared = sum(value**2 for value in exact_shape)
        boundary = _arb_fraction(projection_gain) * inner
        inactive = boundary >= shape_norm_squared
        active = boundary < shape_norm_squared
        if inactive == active:
            raise ArithmeticError("canonical projection branch is not definite")
        if inactive:
            alpha = _arb_fraction(Fraction(1))
            status = "inactive"
        else:
            alpha = boundary / shape_norm_squared
            status = "active"
        projected = tuple(alpha * value for value in exact_shape)
        passive_weight = _arb_fraction((1 - LOCKED_GATE_CAP) / LOCKED_PASSIVE_DIVISOR)
        shape_weight = _arb_fraction(LOCKED_GATE_CAP)
        output = tuple(
            passive_weight * y_value + shape_weight * shape_value
            for y_value, shape_value in zip(exact_yosida, projected, strict=True)
        )

        upstream_normalized = tuple(
            value / (_arb_fraction(Fraction(5)) + epsilon) for value in source
        )
        upstream = tuple(_arb_jordan(value) for value in upstream_normalized)
        departure = _best_scalar_departure(output, source)
        upstream_departure = _best_scalar_departure(upstream, source)
        retention = departure / upstream_departure
        output_norm = (output[0] ** 2 + output[1] ** 2).sqrt()
        upstream_norm = (upstream[0] ** 2 + upstream[1] ** 2).sqrt()
        departure_gate = _arb_fraction(BEST_SCALAR_DEPARTURE_GATE)
        retention_gate = _arb_fraction(UPSTREAM_SHAPING_RETENTION_GATE)
        return CanonicalProjectedFidelityEnclosure(
            precision_bits=precision_bits,
            projection_gain=projection_gain,
            graph_residual_norm=residual_norm,
            exact_root_error_norm_upper=root_error,
            exact_yosida_error_norm_upper=yosida_error,
            projection_scale=alpha,
            projection_status=status,
            output_singular_values=output,
            best_scalar_departure=departure,
            upstream_best_scalar_departure=upstream_departure,
            upstream_shaping_retention=retention,
            output_to_upstream_amplitude=output_norm / upstream_norm,
            output_to_input_amplitude=output_norm / 5,
            departure_passes=departure >= departure_gate,
            departure_fails=departure < departure_gate,
            retention_passes=retention >= retention_gate,
            retention_fails=retention < retention_gate,
        )


def interval_certificate_checks(*, precision_bits: int = 192) -> dict[str, bool]:
    canonical = evaluate_canonical_projected_fidelity(precision_bits=precision_bits)
    small = evaluate_canonical_projected_fidelity(
        SMALL_K_PROJECTION_GAIN,
        precision_bits=precision_bits,
    )
    arb_fraction = _arb_fraction
    return {
        "canonical_projection_is_definitely_inactive": canonical.projection_status == "inactive",
        "canonical_departure_inside_guard": canonical.best_scalar_departure
        > arb_fraction(CANONICAL_DEPARTURE_GUARD[0])
        and canonical.best_scalar_departure < arb_fraction(CANONICAL_DEPARTURE_GUARD[1]),
        "canonical_retention_inside_guard": canonical.upstream_shaping_retention
        > arb_fraction(CANONICAL_RETENTION_GUARD[0])
        and canonical.upstream_shaping_retention < arb_fraction(CANONICAL_RETENTION_GUARD[1]),
        "canonical_upstream_amplitude_inside_guard": canonical.output_to_upstream_amplitude
        > arb_fraction(CANONICAL_UPSTREAM_AMPLITUDE_GUARD[0])
        and canonical.output_to_upstream_amplitude
        < arb_fraction(CANONICAL_UPSTREAM_AMPLITUDE_GUARD[1]),
        "canonical_input_amplitude_inside_guard": canonical.output_to_input_amplitude
        > arb_fraction(CANONICAL_INPUT_AMPLITUDE_GUARD[0])
        and canonical.output_to_input_amplitude < arb_fraction(CANONICAL_INPUT_AMPLITUDE_GUARD[1]),
        "canonical_fidelity_decisions_are_definite": canonical.decisions_are_definite,
        "canonical_fidelity_passes": canonical.meaningful_fidelity_passes,
        "small_k_projection_is_definitely_active": small.projection_status == "active",
        "small_k_departure_inside_guard": small.best_scalar_departure
        > arb_fraction(SMALL_K_DEPARTURE_GUARD[0])
        and small.best_scalar_departure < arb_fraction(SMALL_K_DEPARTURE_GUARD[1]),
        "small_k_retention_inside_guard": small.upstream_shaping_retention
        > arb_fraction(SMALL_K_RETENTION_GUARD[0])
        and small.upstream_shaping_retention < arb_fraction(SMALL_K_RETENTION_GUARD[1]),
        "small_k_fidelity_decision_is_definite": small.decisions_are_definite,
        "small_k_fails_retention_gate": small.retention_fails,
    }


def exact_certificate_checks() -> dict[str, bool]:
    audits = audit_all_pareto_certificates()
    primary = audits[1]
    obstruction = generic_sector_schur_obstruction()
    unprojected = unprojected_sector_control()
    factor_a, factor_b, factor_c = JORDAN_QUINTIC.fractions()
    return {
        "operator_is_locked_five_stage_jordan": JORDAN_QUINTIC.fractions()
        == (Fraction(6_889, 2_000), Fraction(-191, 40), Fraction(4_063, 2_000)),
        "quintic_stage_is_sign_preserving": factor_a > 0
        and factor_c > 0
        and factor_b**2 - 4 * factor_a * factor_c < 0,
        "locked_epsilon_lambda_mu_and_kappa_match": Fraction(1, 10_000_000) == LOCKED_EPSILON
        and Fraction(1, 1_000) == LOCKED_LAMBDA
        and LOCKED_MU == 1_000
        and Fraction(1, 250) == LOCKED_P15_KAPPA,
        "locked_projected_sector_is_exact": (
            Fraction(125, 1_024),
            Fraction(509, 512),
            Fraction(1_143, 2_048),
            Fraction(893, 2_048),
        )
        == (
            LOCKED_SECTOR_LOWER,
            LOCKED_SECTOR_UPPER,
            LOCKED_SECTOR_CENTER,
            LOCKED_SECTOR_RADIUS,
        ),
        "sector_endpoints_follow_projection_hull": LOCKED_SECTOR_LOWER
        == (1 - LOCKED_GATE_CAP) * YOSIDA_GAIN_LOWER / LOCKED_PASSIVE_DIVISOR
        and max(
            YOSIDA_GAIN_UPPER / LOCKED_PASSIVE_DIVISOR,
            (1 - LOCKED_GATE_CAP) * YOSIDA_GAIN_UPPER / LOCKED_PASSIVE_DIVISOR
            + LOCKED_GATE_CAP * LOCKED_PROJECTION_GAIN,
        )
        == LOCKED_SECTOR_UPPER,
        "all_six_pareto_certificates_pass": len(audits) == 6
        and all(audit.certified for audit in audits),
        "primary_rate_gate_is_exact": primary.spec.rate_squared**10 < P14_RATE_SQUARED,
        "primary_effective_step_guard_closes": primary.spec.learning_rate * LOCKED_SECTOR_LOWER
        > PRIMARY_EFFECTIVE_STEP_LOWER
        and primary.spec.learning_rate * LOCKED_SECTOR_UPPER < PRIMARY_EFFECTIVE_STEP_UPPER,
        "generic_sector_obstruction_matches_frozen_margin": obstruction.second_schur_margin
        == GENERIC_SECTOR_SCHUR_MARGIN,
        "generic_sector_obstruction_is_strict": obstruction.is_unstable,
        "unprojected_control_is_huge": unprojected.upper > 500
        and unprojected.effective_upper_step > PRIMARY_EFFECTIVE_STEP_UPPER,
        "small_k_control_sector_is_finite": SMALL_K_PROJECTION_GAIN > 0
        and max(
            YOSIDA_GAIN_UPPER / LOCKED_PASSIVE_DIVISOR,
            (1 - LOCKED_GATE_CAP) * YOSIDA_GAIN_UPPER / LOCKED_PASSIVE_DIVISOR
            + LOCKED_GATE_CAP * SMALL_K_PROJECTION_GAIN,
        )
        == Fraction(125, 128),
    }


__all__ = [
    "BEST_SCALAR_DEPARTURE_GATE",
    "CANONICAL_APPROXIMATE_ROOT_HEX",
    "CANONICAL_DEPARTURE_GUARD",
    "CANONICAL_INPUT",
    "CANONICAL_INPUT_AMPLITUDE_GUARD",
    "CANONICAL_RETENTION_GUARD",
    "CANONICAL_UPSTREAM_AMPLITUDE_GUARD",
    "GENERIC_SECTOR_FAILED_STEP",
    "GENERIC_SECTOR_SCHUR_MARGIN",
    "LOCKED_BETA",
    "LOCKED_EPSILON",
    "LOCKED_GATE_CAP",
    "LOCKED_GATE_Q0",
    "LOCKED_GATE_Q1",
    "LOCKED_LAMBDA",
    "LOCKED_MU",
    "LOCKED_P15_KAPPA",
    "LOCKED_PASSIVE_DIVISOR",
    "LOCKED_PROJECTION_GAIN",
    "LOCKED_SECTOR_CENTER",
    "LOCKED_SECTOR_LOWER",
    "LOCKED_SECTOR_RADIUS",
    "LOCKED_SECTOR_UPPER",
    "P14_RATE_SQUARED",
    "PARETO_CERTIFICATES",
    "PRIMARY_CERTIFICATE_SPEC",
    "PRIMARY_EFFECTIVE_STEP_LOWER",
    "PRIMARY_EFFECTIVE_STEP_UPPER",
    "RAW_SHAPE_GAIN_UPPER",
    "SECTOR_PROJECTED_RESOLVENT_SCHEMA_VERSION",
    "SMALL_K_DEPARTURE_GUARD",
    "SMALL_K_PROJECTION_GAIN",
    "SMALL_K_RETENTION_GUARD",
    "UPSTREAM_SHAPING_RETENTION_GATE",
    "CanonicalProjectedFidelityEnclosure",
    "GenericSectorSchurObstruction",
    "ParetoCertificateSpec",
    "SectorProjectedCertificateAudit",
    "UnprojectedSectorControl",
    "audit_all_pareto_certificates",
    "audit_pareto_certificate",
    "canonical_approximate_root",
    "evaluate_canonical_projected_fidelity",
    "exact_certificate_checks",
    "generic_sector_schur_obstruction",
    "interval_certificate_checks",
    "make_pl_certificate",
    "unprojected_sector_control",
]
