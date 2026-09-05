"""Exact algebra and interval checks for the P16 resolvent-solver study.

P16 separates three logically different statements:

* the P13 operator and its P14 resolvent are bi-orthogonally equivariant;
* the locked resolvent can be evaluated by a guarded singular-value solve;
* algebraic spectral shaping need not be *meaningful* Muon fidelity.

This module carries the exact constants and the two-mode shaping witness.  The
witness is constructed forward from ``u=diag(3/5,4/5)``.  Its input is
``s=u+lambda*B(u)``, so ``u`` is the unique exact resolvent value without any
numerical root assumption.  Although the radial primitive contains a
logarithm, the sign of the modal-gain difference is rational: the common
radial and shunt gains cancel.

Arb is used only for outward-rounded display and for the quantitative fidelity
decision.  The theorem remains an exact-real statement.  The FP64 solver and
its residual checks are separate, falsifiable implementation evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from flint import arb, ctx, fmpq

from passive_muon.additive_epsilon_deficit import DEPLOYED_EPSILON
from passive_muon.radial_passivation_tradeoff import (
    SWITCH_RAW_RATIO,
    TAIL_DERIVATIVE_LOWER,
    TAIL_PROJECTION_LOWER,
    UNIT_DEFICIT_UPPER,
    LogarithmicValue,
    evaluate_logarithmic_value,
    radial_integral_terms,
)
from passive_muon.specs import JORDAN_QUINTIC, scalar_response_and_derivative_exact
from passive_muon.yosida_stability import (
    LOCKED_BASE_STRONG_MONOTONICITY,
    LOCKED_RESOLVENT_PARAMETER,
)

EQUIVARIANT_RESOLVENT_SOLVER_SCHEMA_VERSION = (
    "passive-muon-equivariant-resolvent-solver-certificate-v1"
)

LOCKED_EPSILON = DEPLOYED_EPSILON
LOCKED_LAMBDA = LOCKED_RESOLVENT_PARAMETER
LOCKED_MU = LOCKED_BASE_STRONG_MONOTONICITY
LOCKED_RELATIVE_RESIDUAL = Fraction(1, 250)

# This quantitative rule is deliberately stricter than merely observing two
# unequal floating-point gains.  It was frozen after exploratory diagnostics
# and before the canonical P16 artifact.  Passing requires both a visible
# departure from the best scalar fit and retention of a non-negligible share
# of the upstream five-stage Jordan direction.
BEST_SCALAR_DEPARTURE_GATE = Fraction(1, 1_000)
UPSTREAM_SHAPING_RETENTION_GATE = Fraction(1, 10)

WITNESS_U = (Fraction(3, 5), Fraction(4, 5))
WITNESS_RADIUS = Fraction(1)

# Locked rational guards around the two-precision Arb evaluations.  These are
# intentionally much wider than the Arb balls and make the independent
# standard-library reconstruction straightforward.
LOCKED_BEST_SCALAR_DEPARTURE_GUARD = (
    Fraction(5_704_8, 10_000_000_000),
    Fraction(5_704_9, 10_000_000_000),
)
LOCKED_UPSTREAM_DEPARTURE_GUARD = (
    Fraction(69_988_4, 10_000_000),
    Fraction(69_988_5, 10_000_000),
)
LOCKED_RETENTION_GUARD = (
    Fraction(8_151, 100_000_000),
    Fraction(8_152, 100_000_000),
)

# The meaningful-fidelity decision uses the pre-existing normalization witness
# diag(3,4).  These are the exact binary64 words returned by the locked solver
# on the source platform before interval residual inflation.  They are data
# for an a posteriori enclosure, not assumed exact roots or portable solver
# outputs.
CANONICAL_INPUT = (Fraction(3), Fraction(4))
CANONICAL_APPROXIMATE_ROOT_HEX = (
    "0x1.705228c08d605p-1",
    "0x1.eb135495cac74p-1",
)
CANONICAL_DEPARTURE_GUARD = (
    Fraction(58_791, 10_000_000_000),
    Fraction(58_793, 10_000_000_000),
)
CANONICAL_UPSTREAM_DEPARTURE_GUARD = (
    Fraction(699_674, 10_000_000),
    Fraction(699_676, 10_000_000),
)
CANONICAL_RETENTION_GUARD = (
    Fraction(84_027, 1_000_000_000),
    Fraction(84_029, 1_000_000_000),
)

# The diagonal terms used by the Sherman--Morrison Newton solve stay positive
# in each P12/P13 normalized-radius band.  The last margin is the smallest.
JACOBIAN_DIAGONAL_MARGINS = (
    UNIT_DEFICIT_UPPER - Fraction(504, 5),
    UNIT_DEFICIT_UPPER - Fraction(77_877, 500) * Fraction(199, 200),
    UNIT_DEFICIT_UPPER - Fraction(6_379, 40) * Fraction(497, 500),
)


def _require_positive_fraction(value: Fraction, name: str) -> None:
    if not isinstance(value, Fraction):
        raise TypeError(f"{name} must be a fractions.Fraction")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _arb_fraction(value: Fraction) -> arb:
    return arb(fmpq(value.numerator, value.denominator))


def _add_logarithmic(left: LogarithmicValue, right: LogarithmicValue) -> LogarithmicValue:
    if left.logarithm_argument != right.logarithm_argument:
        raise ValueError("logarithmic arguments must agree")
    return LogarithmicValue(
        left.rational_part + right.rational_part,
        left.logarithm_coefficient + right.logarithm_coefficient,
        left.logarithm_argument,
    )


def _scale_logarithmic(scale: Fraction, value: LogarithmicValue) -> LogarithmicValue:
    return LogarithmicValue(
        scale * value.rational_part,
        scale * value.logarithm_coefficient,
        value.logarithm_argument,
    )


def _rational_logarithmic(value: Fraction, argument: Fraction) -> LogarithmicValue:
    return LogarithmicValue(value, Fraction(0), argument)


@dataclass(frozen=True)
class ForwardShapingWitness:
    """Exact rational/log representation of the locked two-mode witness."""

    epsilon: Fraction
    resolvent_parameter: Fraction
    shunt: Fraction
    resolvent_singular_values: tuple[Fraction, Fraction]
    normalized_singular_values: tuple[Fraction, Fraction]
    jordan_responses: tuple[Fraction, Fraction]
    radial_primitive: LogarithmicValue
    output_singular_values: tuple[LogarithmicValue, LogarithmicValue]
    input_singular_values: tuple[LogarithmicValue, LogarithmicValue]
    pre_resolvent_modal_gain_difference: Fraction

    @property
    def algebraically_distinct(self) -> bool:
        """Whether the second exact graph gain is strictly larger."""

        return self.pre_resolvent_modal_gain_difference > 0


@dataclass(frozen=True)
class ShapingEvaluation:
    """Outward-rounded locked and upstream fidelity diagnostics."""

    precision_bits: int
    input_singular_values: tuple[arb, arb]
    output_singular_values: tuple[arb, arb]
    modal_gains: tuple[arb, arb]
    modal_gain_gap: arb
    best_scalar_departure: arb
    upstream_best_scalar_departure: arb
    upstream_shaping_retention: arb
    absolute_gate_passes: bool
    retention_gate_passes: bool
    absolute_gate_fails: bool
    retention_gate_fails: bool

    @property
    def meaningful_fidelity_passes(self) -> bool:
        return self.absolute_gate_passes and self.retention_gate_passes

    @property
    def gate_decisions_are_definite(self) -> bool:
        """Whether neither interval overlaps its decision boundary."""

        return (self.absolute_gate_passes != self.absolute_gate_fails) and (
            self.retention_gate_passes != self.retention_gate_fails
        )


@dataclass(frozen=True)
class CanonicalShapingEnclosure:
    """Rigorous enclosure obtained from an FP64 candidate and its graph residual."""

    precision_bits: int
    approximate_root: tuple[Fraction, Fraction]
    graph_residual_norm: arb
    exact_output_error_norm_upper: arb
    exact_output_singular_values: tuple[arb, arb]
    modal_gains: tuple[arb, arb]
    modal_gain_gap: arb
    best_scalar_departure: arb
    upstream_best_scalar_departure: arb
    upstream_shaping_retention: arb
    absolute_gate_passes: bool
    retention_gate_passes: bool
    absolute_gate_fails: bool
    retention_gate_fails: bool

    @property
    def meaningful_fidelity_passes(self) -> bool:
        return self.absolute_gate_passes and self.retention_gate_passes

    @property
    def gate_decisions_are_definite(self) -> bool:
        """Whether neither interval overlaps its decision boundary."""

        return (self.absolute_gate_passes != self.absolute_gate_fails) and (
            self.retention_gate_passes != self.retention_gate_fails
        )


def forward_shaping_witness(
    *,
    epsilon: Fraction = LOCKED_EPSILON,
    resolvent_parameter: Fraction = LOCKED_LAMBDA,
    shunt: Fraction = LOCKED_MU,
) -> ForwardShapingWitness:
    """Construct ``s=u+lambda*B(u)`` from a rational unit-radius ``u``.

    The construction is exact up to the explicitly represented logarithm in
    P13's radial primitive.  Since ``||u||_F=1``, the normalized Jordan inputs
    and every Jordan response are exact rationals.
    """

    _require_positive_fraction(epsilon, "epsilon")
    _require_positive_fraction(resolvent_parameter, "resolvent_parameter")
    _require_positive_fraction(shunt, "shunt")
    if sum(value**2 for value in WITNESS_U) != WITNESS_RADIUS**2:
        raise AssertionError("the locked witness must have exact unit radius")

    normalized = tuple(value / (WITNESS_RADIUS + epsilon) for value in WITNESS_U)
    jordan = tuple(
        scalar_response_and_derivative_exact(value, JORDAN_QUINTIC, steps=5)[0]
        for value in normalized
    )
    primitive = radial_integral_terms(WITNESS_RADIUS / epsilon)
    argument = primitive.logarithm_argument

    outputs = tuple(
        _add_logarithmic(
            _rational_logarithmic(response + shunt * value, argument),
            _scale_logarithmic(value, primitive),
        )
        for value, response in zip(WITNESS_U, jordan, strict=True)
    )
    inputs = tuple(
        _add_logarithmic(
            _rational_logarithmic(value, argument),
            _scale_logarithmic(resolvent_parameter, output),
        )
        for value, output in zip(WITNESS_U, outputs, strict=True)
    )
    graph_gain_difference = jordan[1] / WITNESS_U[1] - jordan[0] / WITNESS_U[0]
    return ForwardShapingWitness(
        epsilon=epsilon,
        resolvent_parameter=resolvent_parameter,
        shunt=shunt,
        resolvent_singular_values=WITNESS_U,
        normalized_singular_values=normalized,
        jordan_responses=jordan,
        radial_primitive=primitive,
        output_singular_values=outputs,
        input_singular_values=inputs,
        pre_resolvent_modal_gain_difference=graph_gain_difference,
    )


def _arb_jordan(value: arb) -> arb:
    a, b, c = (_arb_fraction(item) for item in JORDAN_QUINTIC.fractions())
    result = value
    for _ in range(5):
        result = a * result + b * result**3 + c * result**5
    return result


def _arb_radial_integral(raw_ratio: arb) -> arb:
    """Evaluate P13's exact tail primitive directly on an Arb input."""

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
    """Return ``min_a ||output-a*source||_2 / ||output||_2`` in two modes."""

    determinant = abs(output[0] * source[1] - output[1] * source[0])
    source_norm = (source[0] ** 2 + source[1] ** 2).sqrt()
    output_norm = (output[0] ** 2 + output[1] ** 2).sqrt()
    return determinant / (source_norm * output_norm)


def evaluate_shaping_witness(
    witness: ForwardShapingWitness | None = None,
    *,
    precision_bits: int = 192,
) -> ShapingEvaluation:
    """Evaluate the locked witness with outward-rounded Arb arithmetic."""

    if precision_bits < 96:
        raise ValueError("precision_bits must be at least 96")
    selected = forward_shaping_witness() if witness is None else witness
    with ctx.workprec(precision_bits):
        outputs = tuple(
            evaluate_logarithmic_value(value, precision_bits=precision_bits)
            for value in selected.output_singular_values
        )
        inputs = tuple(
            evaluate_logarithmic_value(value, precision_bits=precision_bits)
            for value in selected.input_singular_values
        )
        gains = tuple(output / source for output, source in zip(outputs, inputs, strict=True))
        gap = gains[1] - gains[0]
        departure = _best_scalar_departure(outputs, inputs)
        input_norm = (inputs[0] ** 2 + inputs[1] ** 2).sqrt()
        epsilon_ball = _arb_fraction(selected.epsilon)
        upstream_inputs = tuple(value / (input_norm + epsilon_ball) for value in inputs)
        upstream_outputs = tuple(_arb_jordan(value) for value in upstream_inputs)
        upstream_departure = _best_scalar_departure(upstream_outputs, inputs)
        retention = departure / upstream_departure
        return ShapingEvaluation(
            precision_bits=precision_bits,
            input_singular_values=inputs,
            output_singular_values=outputs,
            modal_gains=gains,
            modal_gain_gap=gap,
            best_scalar_departure=departure,
            upstream_best_scalar_departure=upstream_departure,
            upstream_shaping_retention=retention,
            absolute_gate_passes=departure >= _arb_fraction(BEST_SCALAR_DEPARTURE_GATE),
            retention_gate_passes=retention >= _arb_fraction(UPSTREAM_SHAPING_RETENTION_GATE),
            absolute_gate_fails=departure < _arb_fraction(BEST_SCALAR_DEPARTURE_GATE),
            retention_gate_fails=retention < _arb_fraction(UPSTREAM_SHAPING_RETENTION_GATE),
        )


def canonical_approximate_root() -> tuple[Fraction, Fraction]:
    """Return the locked FP64 candidate as exact binary rational numbers."""

    return tuple(
        Fraction.from_float(float.fromhex(value)) for value in CANONICAL_APPROXIMATE_ROOT_HEX
    )


def evaluate_canonical_shaping_enclosure(
    *,
    precision_bits: int = 192,
) -> CanonicalShapingEnclosure:
    """Enclose the exact locked output on ``diag(3,4)``.

    The binary64 candidate is not trusted as a root.  Its graph residual is
    evaluated with Arb, and P15's exact ``500`` residual-to-output gain inflates
    the candidate output into a rigorous enclosure of the exact resolvent
    output.  The resulting disjoint gain intervals prove algebraic shaping on
    the pre-existing witness while the same enclosures reject both meaningful
    fidelity thresholds.
    """

    if precision_bits < 96:
        raise ValueError("precision_bits must be at least 96")
    root = canonical_approximate_root()
    with ctx.workprec(precision_bits):
        x = tuple(_arb_fraction(value) for value in root)
        source = tuple(_arb_fraction(value) for value in CANONICAL_INPUT)
        radius = (x[0] ** 2 + x[1] ** 2).sqrt()
        epsilon = _arb_fraction(LOCKED_EPSILON)
        normalized = tuple(value / (radius + epsilon) for value in x)
        jordan = tuple(_arb_jordan(value) for value in normalized)
        primitive = _arb_radial_integral(radius / epsilon)
        graph_output = tuple(
            response + (primitive / radius + _arb_fraction(LOCKED_MU)) * value
            for response, value in zip(jordan, x, strict=True)
        )
        residual = tuple(
            sigma - value - _arb_fraction(LOCKED_LAMBDA) * output
            for sigma, value, output in zip(source, x, graph_output, strict=True)
        )
        residual_norm = (residual[0] ** 2 + residual[1] ** 2).sqrt()
        output_error = _arb_fraction(Fraction(500)) * residual_norm
        approximate_output = tuple(
            (sigma - value) / _arb_fraction(LOCKED_LAMBDA)
            for sigma, value in zip(source, x, strict=True)
        )
        exact_output = tuple(value + arb(0, output_error) for value in approximate_output)
        gains = tuple(value / sigma for value, sigma in zip(exact_output, source, strict=True))
        gain_gap = gains[1] - gains[0]
        departure = _best_scalar_departure(exact_output, source)

        source_norm = _arb_fraction(Fraction(5))
        upstream_normalized = tuple(value / (source_norm + epsilon) for value in source)
        upstream_output = tuple(_arb_jordan(value) for value in upstream_normalized)
        upstream_departure = _best_scalar_departure(upstream_output, source)
        retention = departure / upstream_departure
        return CanonicalShapingEnclosure(
            precision_bits=precision_bits,
            approximate_root=root,
            graph_residual_norm=residual_norm,
            exact_output_error_norm_upper=output_error,
            exact_output_singular_values=exact_output,
            modal_gains=gains,
            modal_gain_gap=gain_gap,
            best_scalar_departure=departure,
            upstream_best_scalar_departure=upstream_departure,
            upstream_shaping_retention=retention,
            absolute_gate_passes=departure >= _arb_fraction(BEST_SCALAR_DEPARTURE_GATE),
            retention_gate_passes=retention >= _arb_fraction(UPSTREAM_SHAPING_RETENTION_GATE),
            absolute_gate_fails=departure < _arb_fraction(BEST_SCALAR_DEPARTURE_GATE),
            retention_gate_fails=retention < _arb_fraction(UPSTREAM_SHAPING_RETENTION_GATE),
        )


def exact_certificate_checks() -> dict[str, bool]:
    """Replay the finite rational identities behind the P16 reduction."""

    witness = forward_shaping_witness()
    a, b, c = JORDAN_QUINTIC.fractions()
    factor_discriminant = b**2 - 4 * a * c
    smallest_diagonal_margin = min(JACOBIAN_DIAGONAL_MARGINS)
    return {
        "witness_radius_is_one": sum(value**2 for value in WITNESS_U) == 1,
        "normalization_and_stages_are_locked": (
            witness.epsilon == Fraction(1, 10_000_000)
            and JORDAN_QUINTIC.fractions()
            == (Fraction(6_889, 2_000), Fraction(-191, 40), Fraction(4_063, 2_000))
        ),
        "five_stage_quintic_is_sign_preserving": factor_discriminant
        == Fraction(-2_594_691, 500_000),
        "witness_graph_gains_are_exactly_distinct": witness.algebraically_distinct,
        "jacobian_diagonal_band_margins_are_positive": all(
            value > 0 for value in JACOBIAN_DIAGONAL_MARGINS
        ),
        "smallest_locked_diagonal_margin_matches": smallest_diagonal_margin
        == Fraction(1_098_544_686_059_863, 41_641_817_600_000_000),
        "forward_map_is_two_strongly_monotone": 1 + LOCKED_LAMBDA * LOCKED_MU == 2,
        "p15_relative_residual_is_locked": Fraction(1, 250) == LOCKED_RELATIVE_RESIDUAL,
        "meaningful_fidelity_rule_is_nontrivial": (
            BEST_SCALAR_DEPARTURE_GATE > 0 and UPSTREAM_SHAPING_RETENTION_GATE > 0
        ),
        "forward_guard_is_strictly_below_absolute_gate": (
            LOCKED_BEST_SCALAR_DEPARTURE_GUARD[1] < BEST_SCALAR_DEPARTURE_GATE
        ),
        "forward_guard_is_strictly_below_retention_gate": (
            LOCKED_RETENTION_GUARD[1] < UPSTREAM_SHAPING_RETENTION_GATE
        ),
        "canonical_guard_is_strictly_below_absolute_gate": (
            CANONICAL_DEPARTURE_GUARD[1] < BEST_SCALAR_DEPARTURE_GATE
        ),
        "canonical_guard_is_strictly_below_retention_gate": (
            CANONICAL_RETENTION_GUARD[1] < UPSTREAM_SHAPING_RETENTION_GATE
        ),
    }


__all__ = [
    "BEST_SCALAR_DEPARTURE_GATE",
    "CANONICAL_APPROXIMATE_ROOT_HEX",
    "CANONICAL_DEPARTURE_GUARD",
    "CANONICAL_INPUT",
    "CANONICAL_RETENTION_GUARD",
    "CANONICAL_UPSTREAM_DEPARTURE_GUARD",
    "EQUIVARIANT_RESOLVENT_SOLVER_SCHEMA_VERSION",
    "JACOBIAN_DIAGONAL_MARGINS",
    "LOCKED_BEST_SCALAR_DEPARTURE_GUARD",
    "LOCKED_EPSILON",
    "LOCKED_LAMBDA",
    "LOCKED_MU",
    "LOCKED_RELATIVE_RESIDUAL",
    "LOCKED_RETENTION_GUARD",
    "LOCKED_UPSTREAM_DEPARTURE_GUARD",
    "UPSTREAM_SHAPING_RETENTION_GATE",
    "WITNESS_U",
    "CanonicalShapingEnclosure",
    "ForwardShapingWitness",
    "ShapingEvaluation",
    "canonical_approximate_root",
    "evaluate_canonical_shaping_enclosure",
    "evaluate_shaping_witness",
    "exact_certificate_checks",
    "forward_shaping_witness",
]
