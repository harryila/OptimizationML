"""Independent exact/Arb obligations for the P17 gated resolvent interface.

The exact-real interface certified here is

``T(S) = (1-theta(||S||_F**2))*Y(S)/c + theta(||S||_F**2)*E(J(S))``.

``J`` and ``Y=(I-J)/lambda`` are the locked P14/P16 resolvent and Yosida
maps, while ``E`` is the five-stage Jordan map with additive-epsilon
normalization.  This module deliberately does not import the P17 runtime
module.  It reconstructs the scalar graph, gate, pointwise sector, canonical
fidelity enclosure, and negative control from the frozen P12--P16 constants.

The sector proved here is pointwise, not incremental.  It is the supply used
by the function-value/PL LMI.  In particular, it does not claim that the raw
shape map has a positive derivative: the exact and Arb controls below prove
the opposite near the P13 switching radius.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from fractions import Fraction

from flint import arb, ctx, fmpq

from passive_muon.specs import JORDAN_QUINTIC, scalar_response_and_derivative_exact

if hasattr(sys, "set_int_max_str_digits"):
    # The exact five-stage rational witnesses have tens of thousands of
    # decimal digits; disabling only this conversion guard is safe here.
    sys.set_int_max_str_digits(0)

SHAPE_PRESERVING_RESOLVENT_SCHEMA_VERSION = "passive-muon-shape-preserving-resolvent-certificate-v1"

# P12--P16 exact-real graph constants, repeated here to keep this module an
# independent reconstruction of the P17 obligations.
LOCKED_EPSILON = Fraction(1, 10_000_000)
LOCKED_LAMBDA = Fraction(1, 1_000)
LOCKED_MU = Fraction(1_000)
LOCKED_BETA = Fraction(19, 20)

JORDAN_DERIVATIVE_UPPER = Fraction(4_848_763, 10_000)
SWITCH_NORMALIZED_RADIUS = Fraction(63, 10_000)
SWITCH_RAW_RATIO = Fraction(63, 9_937)
TAIL_DERIVATIVE_LOWER = Fraction(-199_437, 1_250)
TAIL_PROJECTION_LOWER = Fraction(-41_528_474_059_081, 260_261_360_000)
UNIT_DEFICIT_UPPER = Fraction(
    6_602_082_433_275_499_863,
    41_641_817_600_000_000,
)

YOSIDA_GAIN_LOWER = Fraction(500)
YOSIDA_GAIN_UPPER = Fraction(1_000)

# The exact dimension-free pointwise gain bound derived from the singular
# graph equation.  ``RAW_SHAPE_GAIN_UPPER_FORMULA`` is checked against the
# reduced fraction below by ``exact_certificate_checks``.
RAW_SHAPE_GAIN_UPPER = Fraction(
    20_191_130_443_162_880_000_000,
    26_793_221_204_801_899_863,
)

GATE_INNER_SQUARED_RADIUS = Fraction(1, 4)
GATE_OUTER_SQUARED_RADIUS = Fraction(1)

BEST_SCALAR_DEPARTURE_GATE = Fraction(1, 1_000)
UPSTREAM_SHAPING_RETENTION_GATE = Fraction(1, 10)

CANONICAL_INPUT = (Fraction(3), Fraction(4))
CANONICAL_APPROXIMATE_ROOT_HEX = (
    "0x1.705228c08d605p-1",
    "0x1.eb135495cac74p-1",
)
CANONICAL_UPSTREAM_DEPARTURE_GUARD = (
    Fraction(699_674, 10_000_000),
    Fraction(699_676, 10_000_000),
)
PRIMARY_CANONICAL_DEPARTURE_GUARD = (
    Fraction(567_757, 10_000_000),
    Fraction(567_759, 10_000_000),
)
PRIMARY_CANONICAL_RETENTION_GUARD = (
    Fraction(811_459, 1_000_000),
    Fraction(811_461, 1_000_000),
)
FULL_STEP_CANONICAL_DEPARTURE_GUARD = (
    Fraction(203_561, 10_000_000),
    Fraction(203_562, 10_000_000),
)
FULL_STEP_CANONICAL_RETENTION_GUARD = (
    Fraction(290_936, 1_000_000),
    Fraction(290_938, 1_000_000),
)

# This normalized-radius cell lies wholly in the P13 tail.  Direct Arb
# recurrence proves that the scalar raw-shape derivative is negative on the
# complete cell, rather than only at a sampled point.
UNSAFE_BAND_LEFT = SWITCH_NORMALIZED_RADIUS
UNSAFE_BAND_RIGHT = Fraction(6_301, 1_000_000)
UNSAFE_BAND_SOURCE_GUARD = (Fraction(1_977, 1_000_000), Fraction(1_980, 1_000_000))
UNSAFE_BAND_FORWARD_DERIVATIVE_GUARD = (Fraction(8_000), Fraction(13_000))
UNSAFE_BAND_RAW_SHAPE_DERIVATIVE_GUARD = (Fraction(-180_000), Fraction(-120_000))

# Negative control: the gate has already reached its raw-shape plateau at the
# exact unsafe witness.  These are norm radii; the public gate uses their
# squares.
UNDERSIZED_GATE_INNER_RADIUS = Fraction(1, 2_048)
UNDERSIZED_GATE_OUTER_RADIUS = Fraction(1, 1_024)


@dataclass(frozen=True)
class GateDesign:
    """Locked exact gate/interface design."""

    name: str
    cap: Fraction
    passive_divisor: Fraction
    learning_rate: Fraction

    def __post_init__(self) -> None:
        if not 0 < self.cap < 1:
            raise ValueError("gate cap must lie strictly between zero and one")
        if self.passive_divisor <= 0:
            raise ValueError("passive divisor must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning rate must be positive")


PRIMARY_DESIGN = GateDesign(
    name="high_fidelity_reduced_step",
    cap=Fraction(3, 4),
    passive_divisor=Fraction(4_096),
    learning_rate=Fraction(1, 128_000),
)
FULL_STEP_DESIGN = GateDesign(
    name="full_step_fidelity_floor",
    cap=Fraction(1, 8),
    passive_divisor=Fraction(8_192),
    learning_rate=Fraction(1, 32_000),
)
LOCKED_DESIGNS = (PRIMARY_DESIGN, FULL_STEP_DESIGN)


@dataclass(frozen=True)
class PointwiseSector:
    """Exact global pointwise sector for one locked gate design."""

    design: GateDesign
    lower_gain: Fraction
    upper_gain: Fraction
    center: Fraction
    radius: Fraction


@dataclass(frozen=True)
class UnsafeDerivativeBand:
    """Arb enclosure of a complete unsafe scalar raw-shape band."""

    precision_bits: int
    normalized_radius: arb
    resolvent_value: arb
    source_value: arb
    direct_shape_derivative: arb
    forward_graph_derivative: arb
    raw_shape_derivative: arb

    @property
    def is_strictly_unsafe(self) -> bool:
        return self.forward_graph_derivative > 0 and self.raw_shape_derivative < 0


@dataclass(frozen=True)
class ExactUnsafeWitness:
    """Exact rational scalar witness at ``t=63/10000``."""

    normalized_radius: Fraction
    raw_ratio: Fraction
    resolvent_value: Fraction
    jordan_response: Fraction
    jordan_derivative: Fraction
    source_value: Fraction
    direct_shape_derivative: Fraction
    forward_graph_derivative: Fraction
    raw_shape_derivative: Fraction
    yosida_derivative: Fraction
    response_sha256: str
    derivative_sha256: str
    source_sha256: str
    raw_shape_derivative_sha256: str


@dataclass(frozen=True)
class CanonicalGatedFidelityEnclosure:
    """P16-residual-inflated P17 output enclosure on ``diag(3,4)``."""

    precision_bits: int
    design: GateDesign
    approximate_root: tuple[Fraction, Fraction]
    graph_residual_norm: arb
    exact_root_error_norm_upper: arb
    exact_yosida_error_norm_upper: arb
    gate_value: Fraction
    output_singular_values: tuple[arb, arb]
    modal_gains: tuple[arb, arb]
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
        return (self.absolute_gate_passes != self.absolute_gate_fails) and (
            self.retention_gate_passes != self.retention_gate_fails
        )


@dataclass(frozen=True)
class UnderSizedPassiveRegionControl:
    """Exact frozen scalar Jury control for an under-sized passive region."""

    design: GateDesign
    witness_source: Fraction
    inner_radius: Fraction
    outer_radius: Fraction
    raw_shape_derivative: Fraction
    yosida_derivative: Fraction
    gated_derivative: Fraction
    curvature: Fraction
    beta: Fraction
    learning_rate: Fraction
    trace: Fraction
    determinant: Fraction
    second_jury_margin: Fraction

    @property
    def witness_is_on_raw_shape_plateau(self) -> bool:
        return self.witness_source > self.outer_radius

    @property
    def fails_second_jury_condition(self) -> bool:
        return self.second_jury_margin < 0


def _arb_fraction(value: Fraction) -> arb:
    return arb(fmpq(value.numerator, value.denominator))


def _arb_interval(left: Fraction, right: Fraction) -> arb:
    if left > right:
        raise ValueError("interval endpoints must be ordered")
    return _arb_fraction(left).union(_arb_fraction(right))


def _fraction_sha256(value: Fraction) -> str:
    payload = f"{value.numerator}/{value.denominator}".encode()
    return hashlib.sha256(payload).hexdigest()


def _arb_jordan(value: arb) -> arb:
    a, b, c = (_arb_fraction(item) for item in JORDAN_QUINTIC.fractions())
    result = value
    for _ in range(5):
        square = result * result
        result = result * (a + square * (b + c * square))
    return result


def _arb_jordan_derivative(value: arb) -> arb:
    a, b, c = (_arb_fraction(item) for item in JORDAN_QUINTIC.fractions())
    result = value
    derivative = arb(1)
    for _ in range(5):
        square = result * result
        derivative *= a + square * (3 * b + 5 * c * square)
        result *= a + square * (b + c * square)
    return derivative


def _arb_tail_deficit(raw_ratio: arb) -> arb:
    derivative = _arb_fraction(TAIL_DERIVATIVE_LOWER)
    projection = _arb_fraction(TAIL_PROJECTION_LOWER)
    return -(derivative + projection * raw_ratio) / (1 + raw_ratio) ** 2


def _arb_radial_integral(raw_ratio: arb) -> arb:
    """Evaluate P13's tail primitive without importing its implementation."""

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


def quintic_smootherstep(unit_coordinate: Fraction) -> Fraction:
    """Return ``6*t**5-15*t**4+10*t**3`` exactly on ``[0,1]``."""

    if not 0 <= unit_coordinate <= 1:
        raise ValueError("unit_coordinate must lie in [0,1]")
    value = unit_coordinate
    return 6 * value**5 - 15 * value**4 + 10 * value**3


def smootherstep_gate(
    squared_norm: Fraction,
    cap: Fraction,
    *,
    inner_squared_radius: Fraction = GATE_INNER_SQUARED_RADIUS,
    outer_squared_radius: Fraction = GATE_OUTER_SQUARED_RADIUS,
) -> Fraction:
    """Evaluate the locked C2 gate in exact rational arithmetic."""

    if squared_norm < 0:
        raise ValueError("squared_norm must be nonnegative")
    if not 0 <= cap <= 1:
        raise ValueError("cap must lie in [0,1]")
    if not 0 <= inner_squared_radius < outer_squared_radius:
        raise ValueError("gate radii must be nonnegative and strictly ordered")
    if squared_norm <= inner_squared_radius:
        return Fraction(0)
    if squared_norm >= outer_squared_radius:
        return cap
    coordinate = (squared_norm - inner_squared_radius) / (
        outer_squared_radius - inner_squared_radius
    )
    return cap * quintic_smootherstep(coordinate)


def smootherstep_gate_derivatives(
    squared_norm: Fraction,
    cap: Fraction,
    *,
    inner_squared_radius: Fraction = GATE_INNER_SQUARED_RADIUS,
    outer_squared_radius: Fraction = GATE_OUTER_SQUARED_RADIUS,
) -> tuple[Fraction, Fraction, Fraction]:
    """Return ``theta, d theta/dq, d2 theta/dq2`` exactly."""

    theta = smootherstep_gate(
        squared_norm,
        cap,
        inner_squared_radius=inner_squared_radius,
        outer_squared_radius=outer_squared_radius,
    )
    if squared_norm <= inner_squared_radius or squared_norm >= outer_squared_radius:
        return theta, Fraction(0), Fraction(0)
    width = outer_squared_radius - inner_squared_radius
    coordinate = (squared_norm - inner_squared_radius) / width
    first = cap * 30 * coordinate**2 * (1 - coordinate) ** 2 / width
    second = cap * 60 * coordinate * (1 - coordinate) * (1 - 2 * coordinate) / width**2
    return theta, first, second


def pointwise_sector(design: GateDesign) -> PointwiseSector:
    """Return the exact global pointwise residual sector for ``design``."""

    lower = (1 - design.cap) * YOSIDA_GAIN_LOWER / design.passive_divisor
    pure_yosida_upper = YOSIDA_GAIN_UPPER / design.passive_divisor
    shape_endpoint_upper = (
        1 - design.cap
    ) * YOSIDA_GAIN_UPPER / design.passive_divisor + design.cap * RAW_SHAPE_GAIN_UPPER
    upper = max(pure_yosida_upper, shape_endpoint_upper)
    return PointwiseSector(
        design=design,
        lower_gain=lower,
        upper_gain=upper,
        center=(lower + upper) / 2,
        radius=(upper - lower) / 2,
    )


def certify_unsafe_derivative_band(*, precision_bits: int = 192) -> UnsafeDerivativeBand:
    """Certify a complete tail band where ``d(E(J(s)))/ds < 0``.

    The interval variable is ``t=u/(epsilon+u)``.  Every operation uses Arb
    outward rounding, including the five-stage derivative recurrence and the
    P13 radial graph derivative.
    """

    if precision_bits < 96:
        raise ValueError("precision_bits must be at least 96")
    with ctx.workprec(precision_bits):
        normalized = _arb_interval(UNSAFE_BAND_LEFT, UNSAFE_BAND_RIGHT)
        epsilon = _arb_fraction(LOCKED_EPSILON)
        raw_ratio = normalized / (1 - normalized)
        resolvent_value = epsilon * raw_ratio
        jordan = _arb_jordan(normalized)
        jordan_derivative = _arb_jordan_derivative(normalized)
        direct_derivative = jordan_derivative * (1 - normalized) ** 2 / epsilon
        radial_derivative = _arb_tail_deficit(raw_ratio) / epsilon
        forward_derivative = 1 + _arb_fraction(LOCKED_LAMBDA) * (
            direct_derivative + radial_derivative + _arb_fraction(LOCKED_MU)
        )
        raw_shape_derivative = direct_derivative / forward_derivative
        primitive = _arb_radial_integral(raw_ratio)
        source = resolvent_value + _arb_fraction(LOCKED_LAMBDA) * (
            jordan + primitive + _arb_fraction(LOCKED_MU) * resolvent_value
        )
        return UnsafeDerivativeBand(
            precision_bits=precision_bits,
            normalized_radius=normalized,
            resolvent_value=resolvent_value,
            source_value=source,
            direct_shape_derivative=direct_derivative,
            forward_graph_derivative=forward_derivative,
            raw_shape_derivative=raw_shape_derivative,
        )


def exact_unsafe_witness() -> ExactUnsafeWitness:
    """Reconstruct the exact rational unsafe witness at the P13 switch."""

    normalized = SWITCH_NORMALIZED_RADIUS
    raw_ratio = normalized / (1 - normalized)
    resolvent_value = LOCKED_EPSILON * raw_ratio
    response, derivative = scalar_response_and_derivative_exact(
        normalized,
        JORDAN_QUINTIC,
        steps=5,
    )
    primitive = UNIT_DEFICIT_UPPER * raw_ratio
    source = resolvent_value + LOCKED_LAMBDA * (response + primitive + LOCKED_MU * resolvent_value)
    direct_derivative = derivative * (1 - normalized) ** 2 / LOCKED_EPSILON
    forward_derivative = 1 + LOCKED_LAMBDA * (
        direct_derivative + UNIT_DEFICIT_UPPER / LOCKED_EPSILON + LOCKED_MU
    )
    raw_shape_derivative = direct_derivative / forward_derivative
    yosida_derivative = (1 / LOCKED_LAMBDA) * (1 - 1 / forward_derivative)
    return ExactUnsafeWitness(
        normalized_radius=normalized,
        raw_ratio=raw_ratio,
        resolvent_value=resolvent_value,
        jordan_response=response,
        jordan_derivative=derivative,
        source_value=source,
        direct_shape_derivative=direct_derivative,
        forward_graph_derivative=forward_derivative,
        raw_shape_derivative=raw_shape_derivative,
        yosida_derivative=yosida_derivative,
        response_sha256=_fraction_sha256(response),
        derivative_sha256=_fraction_sha256(derivative),
        source_sha256=_fraction_sha256(source),
        raw_shape_derivative_sha256=_fraction_sha256(raw_shape_derivative),
    )


def _best_scalar_departure(output: tuple[arb, arb], source: tuple[arb, arb]) -> arb:
    determinant = abs(output[0] * source[1] - output[1] * source[0])
    source_norm = (source[0] ** 2 + source[1] ** 2).sqrt()
    output_norm = (output[0] ** 2 + output[1] ** 2).sqrt()
    return determinant / (source_norm * output_norm)


def canonical_approximate_root() -> tuple[Fraction, Fraction]:
    """Return the frozen P16 binary64 candidate as exact rationals."""

    return tuple(
        Fraction.from_float(float.fromhex(value)) for value in CANONICAL_APPROXIMATE_ROOT_HEX
    )


def evaluate_canonical_gated_fidelity(
    design: GateDesign,
    *,
    precision_bits: int = 192,
) -> CanonicalGatedFidelityEnclosure:
    """Enclose the exact gated output on the canonical ``diag(3,4)`` input.

    The P16 binary64 candidate is not trusted as an exact root.  Its graph
    residual is recomputed with Arb.  Strong monotonicity of ``I+lambda B``
    gives ``||u_hat-J(s)|| <= ||r||/2``; P15 gives
    ``||Y_hat-Y(s)|| <= 500||r||``.  Both inflations are applied before the
    raw-shape and fidelity calculations.
    """

    if precision_bits < 96:
        raise ValueError("precision_bits must be at least 96")
    if design not in LOCKED_DESIGNS:
        raise ValueError("design must be one of the two locked P17 designs")

    root = canonical_approximate_root()
    with ctx.workprec(precision_bits):
        approximate = tuple(_arb_fraction(value) for value in root)
        source = tuple(_arb_fraction(value) for value in CANONICAL_INPUT)
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
        root_error = _arb_fraction(Fraction(1, 2)) * residual_norm
        yosida_error = _arb_fraction(Fraction(500)) * residual_norm

        exact_root = tuple(value + arb(0, root_error) for value in approximate)
        exact_radius = (exact_root[0] ** 2 + exact_root[1] ** 2).sqrt()
        exact_normalized = tuple(value / (exact_radius + epsilon) for value in exact_root)
        exact_shape = tuple(_arb_jordan(value) for value in exact_normalized)
        approximate_yosida = tuple(
            (sigma - value) / _arb_fraction(LOCKED_LAMBDA)
            for sigma, value in zip(source, approximate, strict=True)
        )
        exact_yosida = tuple(value + arb(0, yosida_error) for value in approximate_yosida)

        squared_source_norm = sum(value**2 for value in CANONICAL_INPUT)
        gate = smootherstep_gate(squared_source_norm, design.cap)
        passive_weight = _arb_fraction((1 - gate) / design.passive_divisor)
        shape_weight = _arb_fraction(gate)
        output = tuple(
            passive_weight * y_value + shape_weight * shape_value
            for y_value, shape_value in zip(exact_yosida, exact_shape, strict=True)
        )
        gains = tuple(value / sigma for value, sigma in zip(output, source, strict=True))
        departure = _best_scalar_departure(output, source)

        source_norm = _arb_fraction(Fraction(5))
        upstream_normalized = tuple(value / (source_norm + epsilon) for value in source)
        upstream_output = tuple(_arb_jordan(value) for value in upstream_normalized)
        upstream_departure = _best_scalar_departure(upstream_output, source)
        retention = departure / upstream_departure
        absolute_threshold = _arb_fraction(BEST_SCALAR_DEPARTURE_GATE)
        retention_threshold = _arb_fraction(UPSTREAM_SHAPING_RETENTION_GATE)
        return CanonicalGatedFidelityEnclosure(
            precision_bits=precision_bits,
            design=design,
            approximate_root=root,
            graph_residual_norm=residual_norm,
            exact_root_error_norm_upper=root_error,
            exact_yosida_error_norm_upper=yosida_error,
            gate_value=gate,
            output_singular_values=output,
            modal_gains=gains,
            best_scalar_departure=departure,
            upstream_best_scalar_departure=upstream_departure,
            upstream_shaping_retention=retention,
            absolute_gate_passes=departure >= absolute_threshold,
            retention_gate_passes=retention >= retention_threshold,
            absolute_gate_fails=departure < absolute_threshold,
            retention_gate_fails=retention < retention_threshold,
        )


def interval_certificate_checks(*, precision_bits: int = 192) -> dict[str, bool]:
    """Replay the frozen Arb guards and threshold decisions."""

    band = certify_unsafe_derivative_band(precision_bits=precision_bits)
    source_lower, source_upper = UNSAFE_BAND_SOURCE_GUARD
    forward_lower, forward_upper = UNSAFE_BAND_FORWARD_DERIVATIVE_GUARD
    derivative_lower, derivative_upper = UNSAFE_BAND_RAW_SHAPE_DERIVATIVE_GUARD
    checks = {
        "unsafe_band_source_inside_guard": band.source_value > _arb_fraction(source_lower)
        and band.source_value < _arb_fraction(source_upper),
        "unsafe_band_forward_derivative_inside_guard": band.forward_graph_derivative
        > _arb_fraction(forward_lower)
        and band.forward_graph_derivative < _arb_fraction(forward_upper),
        "unsafe_band_raw_shape_derivative_inside_guard": band.raw_shape_derivative
        > _arb_fraction(derivative_lower)
        and band.raw_shape_derivative < _arb_fraction(derivative_upper),
        "unsafe_band_sign_decision_is_definite": band.is_strictly_unsafe,
    }
    guards = (
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
    )
    for design, departure_guard, retention_guard in guards:
        evaluation = evaluate_canonical_gated_fidelity(
            design,
            precision_bits=precision_bits,
        )
        prefix = design.name
        checks[f"{prefix}_departure_inside_guard"] = (
            evaluation.best_scalar_departure > _arb_fraction(departure_guard[0])
            and evaluation.best_scalar_departure < _arb_fraction(departure_guard[1])
        )
        checks[f"{prefix}_upstream_departure_inside_guard"] = (
            evaluation.upstream_best_scalar_departure
            > _arb_fraction(CANONICAL_UPSTREAM_DEPARTURE_GUARD[0])
            and evaluation.upstream_best_scalar_departure
            < _arb_fraction(CANONICAL_UPSTREAM_DEPARTURE_GUARD[1])
        )
        checks[f"{prefix}_retention_inside_guard"] = (
            evaluation.upstream_shaping_retention > _arb_fraction(retention_guard[0])
            and evaluation.upstream_shaping_retention < _arb_fraction(retention_guard[1])
        )
        checks[f"{prefix}_fidelity_decisions_are_definite"] = (
            evaluation.gate_decisions_are_definite and evaluation.meaningful_fidelity_passes
        )
    return checks


def undersized_passive_region_control(
    *,
    curvature: Fraction = Fraction(1),
) -> UnderSizedPassiveRegionControl:
    """Return an exact local Jury failure for an under-sized passive gate.

    This is a frozen local/incremental control at the nonzero scalar graph
    witness.  It is not a claim that the witness is an equilibrium of the
    unbiased PL optimizer.
    """

    if curvature <= 0:
        raise ValueError("curvature must be positive")
    witness = exact_unsafe_witness()
    design = FULL_STEP_DESIGN
    if witness.source_value <= UNDERSIZED_GATE_OUTER_RADIUS:
        raise AssertionError("unsafe witness is not on the raw-shape plateau")
    gated_derivative = (
        (1 - design.cap) * witness.yosida_derivative / design.passive_divisor
        + design.cap * witness.raw_shape_derivative
    )
    beta = LOCKED_BETA
    eta = design.learning_rate
    trace = 1 + beta - eta * gated_derivative * (1 - beta**2) * curvature
    determinant = beta - eta * gated_derivative * beta * (1 - beta) * curvature
    second_jury_margin = 1 - trace + determinant
    return UnderSizedPassiveRegionControl(
        design=design,
        witness_source=witness.source_value,
        inner_radius=UNDERSIZED_GATE_INNER_RADIUS,
        outer_radius=UNDERSIZED_GATE_OUTER_RADIUS,
        raw_shape_derivative=witness.raw_shape_derivative,
        yosida_derivative=witness.yosida_derivative,
        gated_derivative=gated_derivative,
        curvature=curvature,
        beta=beta,
        learning_rate=eta,
        trace=trace,
        determinant=determinant,
        second_jury_margin=second_jury_margin,
    )


def exact_certificate_checks() -> dict[str, bool]:
    """Replay the finite exact identities behind the P17 certificate."""

    raw_shape_formula = (JORDAN_DERIVATIVE_UPPER / LOCKED_EPSILON) / (
        1
        + LOCKED_LAMBDA
        * (LOCKED_MU + (UNIT_DEFICIT_UPPER + JORDAN_DERIVATIVE_UPPER) / LOCKED_EPSILON)
    )
    witness = exact_unsafe_witness()
    negative = undersized_passive_region_control()
    midpoint_q = Fraction(5, 8)
    midpoint_gate = smootherstep_gate_derivatives(midpoint_q, Fraction(1))[1]
    factor_a, factor_b, factor_c = JORDAN_QUINTIC.fractions()
    factor_discriminant = factor_b**2 - 4 * factor_a * factor_c
    primary_sector = pointwise_sector(PRIMARY_DESIGN)
    full_step_sector = pointwise_sector(FULL_STEP_DESIGN)
    return {
        "locked_polynomial_is_five_stage_jordan": JORDAN_QUINTIC.fractions()
        == (Fraction(6_889, 2_000), Fraction(-191, 40), Fraction(4_063, 2_000)),
        "quintic_stage_is_sign_preserving": factor_a > 0
        and factor_c > 0
        and factor_discriminant < 0,
        "raw_shape_gain_formula_matches_reduced_fraction": raw_shape_formula
        == RAW_SHAPE_GAIN_UPPER,
        "tail_majorant_dominates_switch_value": -TAIL_DERIVATIVE_LOWER > UNIT_DEFICIT_UPPER
        and -TAIL_PROJECTION_LOWER > UNIT_DEFICIT_UPPER,
        "smootherstep_joins_are_c2": smootherstep_gate_derivatives(
            GATE_INNER_SQUARED_RADIUS, Fraction(1)
        )
        == (Fraction(0), Fraction(0), Fraction(0))
        and smootherstep_gate_derivatives(GATE_OUTER_SQUARED_RADIUS, Fraction(1))
        == (Fraction(1), Fraction(0), Fraction(0)),
        "smootherstep_max_q_slope_is_exact": midpoint_gate == Fraction(5, 2),
        "gate_preserves_zero": all(
            smootherstep_gate(Fraction(0), design.cap) == 0 for design in LOCKED_DESIGNS
        ),
        "primary_sector_is_ordered": 0 < primary_sector.lower_gain < primary_sector.upper_gain,
        "primary_sector_matches_locked_fractions": primary_sector.lower_gain == Fraction(125, 4_096)
        and primary_sector.center
        == Fraction(
            20_679_066_726_449_389_357_482_875,
            73_163_356_036_579_054_559_232,
        )
        and primary_sector.radius
        == Fraction(
            20_676_833_958_015_655_865_827_625,
            73_163_356_036_579_054_559_232,
        ),
        "full_step_sector_is_ordered": 0
        < full_step_sector.lower_gain
        < full_step_sector.upper_gain,
        "full_step_sector_matches_locked_fractions": full_step_sector.lower_gain
        == Fraction(875, 16_384)
        and full_step_sector.center
        == Fraction(
            41_421_767_353_260_183_227_140_375,
            877_960_272_438_948_654_710_784,
        )
        and full_step_sector.radius
        == Fraction(
            41_374_879_216_151_779_902_380_125,
            877_960_272_438_948_654_710_784,
        ),
        "unsafe_witness_is_at_locked_switch": witness.raw_ratio == SWITCH_RAW_RATIO,
        "unsafe_witness_source_is_locked_near_two_millith": Fraction(19, 10_000)
        < witness.source_value
        < Fraction(1, 500),
        "unsafe_witness_jordan_slope_is_negative": Fraction(-160)
        < witness.jordan_derivative
        < Fraction(-159),
        "unsafe_raw_shape_derivative_is_large_and_negative": Fraction(-147_000)
        < witness.raw_shape_derivative
        < Fraction(-146_000),
        "unsafe_yosida_derivative_remains_passive": Fraction(999)
        < witness.yosida_derivative
        < Fraction(1_000),
        "undersized_gate_is_on_raw_shape_plateau": negative.witness_is_on_raw_shape_plateau,
        "undersized_gate_derivative_is_negative": Fraction(-19_000)
        < negative.gated_derivative
        < Fraction(-18_000),
        "undersized_gate_fails_second_jury_condition": negative.fails_second_jury_condition,
        "jury_identity_matches_direct_formula": negative.second_jury_margin
        == negative.learning_rate
        * (1 - negative.beta)
        * negative.curvature
        * negative.gated_derivative,
        "fidelity_thresholds_are_frozen": Fraction(1, 1_000) == BEST_SCALAR_DEPARTURE_GATE
        and Fraction(1, 10) == UPSTREAM_SHAPING_RETENTION_GATE,
        "primary_fidelity_guards_imply_pass": PRIMARY_CANONICAL_DEPARTURE_GUARD[0]
        > BEST_SCALAR_DEPARTURE_GATE
        and PRIMARY_CANONICAL_RETENTION_GUARD[0] > UPSTREAM_SHAPING_RETENTION_GATE,
        "full_step_fidelity_guards_imply_pass": FULL_STEP_CANONICAL_DEPARTURE_GUARD[0]
        > BEST_SCALAR_DEPARTURE_GATE
        and FULL_STEP_CANONICAL_RETENTION_GUARD[0] > UPSTREAM_SHAPING_RETENTION_GATE,
        "unsafe_guards_imply_signs": UNSAFE_BAND_FORWARD_DERIVATIVE_GUARD[0] > 0
        and UNSAFE_BAND_RAW_SHAPE_DERIVATIVE_GUARD[1] < 0,
    }


__all__ = [
    "BEST_SCALAR_DEPARTURE_GATE",
    "CANONICAL_APPROXIMATE_ROOT_HEX",
    "CANONICAL_INPUT",
    "CANONICAL_UPSTREAM_DEPARTURE_GUARD",
    "FULL_STEP_CANONICAL_DEPARTURE_GUARD",
    "FULL_STEP_CANONICAL_RETENTION_GUARD",
    "FULL_STEP_DESIGN",
    "GATE_INNER_SQUARED_RADIUS",
    "GATE_OUTER_SQUARED_RADIUS",
    "LOCKED_DESIGNS",
    "LOCKED_EPSILON",
    "LOCKED_LAMBDA",
    "LOCKED_MU",
    "PRIMARY_CANONICAL_DEPARTURE_GUARD",
    "PRIMARY_CANONICAL_RETENTION_GUARD",
    "PRIMARY_DESIGN",
    "RAW_SHAPE_GAIN_UPPER",
    "SHAPE_PRESERVING_RESOLVENT_SCHEMA_VERSION",
    "SWITCH_NORMALIZED_RADIUS",
    "SWITCH_RAW_RATIO",
    "UNDERSIZED_GATE_INNER_RADIUS",
    "UNDERSIZED_GATE_OUTER_RADIUS",
    "UNSAFE_BAND_FORWARD_DERIVATIVE_GUARD",
    "UNSAFE_BAND_LEFT",
    "UNSAFE_BAND_RAW_SHAPE_DERIVATIVE_GUARD",
    "UNSAFE_BAND_RIGHT",
    "UNSAFE_BAND_SOURCE_GUARD",
    "UPSTREAM_SHAPING_RETENTION_GATE",
    "CanonicalGatedFidelityEnclosure",
    "ExactUnsafeWitness",
    "GateDesign",
    "PointwiseSector",
    "UnderSizedPassiveRegionControl",
    "UnsafeDerivativeBand",
    "canonical_approximate_root",
    "certify_unsafe_derivative_band",
    "evaluate_canonical_gated_fidelity",
    "exact_certificate_checks",
    "exact_unsafe_witness",
    "interval_certificate_checks",
    "pointwise_sector",
    "quintic_smootherstep",
    "smootherstep_gate",
    "smootherstep_gate_derivatives",
    "undersized_passive_region_control",
]
