"""Rigorous full-matrix bounds for additive-epsilon normalization.

The map certified here is the real-arithmetic surrogate

``E_h,eps(M) = H_h(M / (||M||_F + eps))``.

It is deliberately distinct from the discontinuous BF16 implementation.  The
only interval obligations are scalar five-stage Jordan derivative bounds and
an algebraic finite-pair witness.  The lift to every finite rectangular matrix
shape is analytic and uses exact rational arithmetic.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from fractions import Fraction

from flint import arb, ctx

from passive_muon.floored_certificate import (
    JORDAN_DERIVATIVE_LOWER,
    JORDAN_DERIVATIVE_UPPER,
    ScalarRangeCertificate,
    _arb_rational,
    _jordan_derivative_ball,
    certify_jordan_derivative_range,
    fraction_sha256,
    projection_envelope_lower,
)
from passive_muon.specs import (
    JORDAN_QUINTIC,
    scalar_response_and_derivative_exact,
)

ADDITIVE_EPSILON_SCHEMA_VERSION = "passive-muon-additive-epsilon-deficit-certificate-v1"


# Each row covers normalization radii t in [left, right].  Its scalar lower
# bound is certified on the larger prefix [0, right], so it bounds every
# rectangular spectral derivative mode at that t.
@dataclass(frozen=True)
class RadiusBand:
    left: Fraction
    right: Fraction
    derivative_lower: Fraction

    def __post_init__(self) -> None:
        if not 0 <= self.left < self.right <= 1:
            raise ValueError("radius bands must lie in [0,1] and be ordered")


RADIUS_BANDS = (
    RadiusBand(Fraction(0), Fraction(1, 200), Fraction(-504, 5)),
    RadiusBand(Fraction(1, 200), Fraction(3, 500), Fraction(-77_877, 500)),
    RadiusBand(Fraction(3, 500), Fraction(63, 10_000), Fraction(-6_379, 40)),
    RadiusBand(Fraction(63, 10_000), Fraction(1), JORDAN_DERIVATIVE_LOWER),
)

# A fully rational symmetric 2 x 2 finite pair.  The Pythagorean direction
# makes both raw matrices have exact norm ``PAIR_RAW_RADIUS``; swapping the
# diagonal entries therefore keeps the additive denominator fixed.
PAIR_NORMALIZED_RADIUS = Fraction(8_974_467, 1_000_000_000)
PAIR_RAW_RADIUS = PAIR_NORMALIZED_RADIUS / (1 - PAIR_NORMALIZED_RADIUS)
PAIR_DIRECTION_LOW = Fraction(803_760, 1_136_689)
PAIR_DIRECTION_HIGH = Fraction(803_761, 1_136_689)
PAIR_DEFICIT_STRICT_LOWER = Fraction(98_823_281, 625_000)

DEPLOYED_EPSILON = Fraction(1, 10_000_000)


@dataclass(frozen=True, order=True)
class ScaledDyadicCell:
    """Cell ``endpoint * [left/2**power, (left+1)/2**power]``."""

    left: int
    power: int

    def __post_init__(self) -> None:
        if self.power < 0:
            raise ValueError("power must be nonnegative")
        if not 0 <= self.left < 1 << self.power:
            raise ValueError("scaled dyadic cell lies outside its prefix")


@dataclass(frozen=True)
class PrefixRangeCertificate:
    """Adaptive Arb proof that ``h'(s) > lower`` on ``[0, endpoint]``."""

    precision_bits: int
    endpoint: Fraction
    derivative_lower: Fraction
    initial_power: int
    configured_maximum_power: int
    maximum_power: int
    leaves: tuple[ScaledDyadicCell, ...]
    trace_sha256: str

    @property
    def leaf_count(self) -> int:
        return len(self.leaves)


@dataclass(frozen=True)
class AdditiveEpsilonIntervalPass:
    """One precision pass for all scalar interval obligations."""

    precision_bits: int
    global_range: ScalarRangeCertificate
    prefix_ranges: tuple[PrefixRangeCertificate, ...]


@dataclass(frozen=True)
class RadiusBandBound:
    """Exact algebraic result for one normalized-radius band."""

    band: RadiusBand
    projection_lower: Fraction
    maximizing_radius: Fraction
    deficit_upper: Fraction


def _scaled_cell_ball(cell: ScaledDyadicCell, endpoint: Fraction) -> arb:
    denominator = 1 << cell.power
    lower = endpoint * Fraction(cell.left, denominator)
    upper = endpoint * Fraction(cell.left + 1, denominator)
    return _arb_rational(lower).union(_arb_rational(upper))


def _ordered_scaled_leaves(
    leaves: list[ScaledDyadicCell],
) -> tuple[ScaledDyadicCell, ...]:
    maximum_power = max(cell.power for cell in leaves)
    ordered = tuple(sorted(leaves, key=lambda cell: cell.left << (maximum_power - cell.power)))
    expected_left = 0
    for cell in ordered:
        scaled_left = cell.left << (maximum_power - cell.power)
        scaled_width = 1 << (maximum_power - cell.power)
        if scaled_left != expected_left:
            raise AssertionError("adaptive cells do not form a contiguous prefix cover")
        expected_left += scaled_width
    if expected_left != 1 << maximum_power:
        raise AssertionError("adaptive cells do not cover the complete prefix")
    return ordered


def _prefix_trace_digest(
    endpoint: Fraction,
    derivative_lower: Fraction,
    leaves: tuple[ScaledDyadicCell, ...],
) -> str:
    header = f"{endpoint.numerator}/{endpoint.denominator};"
    header += f"{derivative_lower.numerator}/{derivative_lower.denominator}\n"
    cells = "".join(f"{cell.left}/{cell.power}\n" for cell in leaves)
    return hashlib.sha256((header + cells).encode()).hexdigest()


def certify_prefix_derivative_lower(
    endpoint: Fraction,
    derivative_lower: Fraction,
    *,
    precision_bits: int = 160,
    initial_power: int = 8,
    maximum_power: int = 48,
) -> PrefixRangeCertificate:
    """Prove ``h'(s) > derivative_lower`` on ``0 <= s <= endpoint``.

    The recurrence is evaluated directly with outward-rounded Arb balls; the
    degree-3125 composed polynomial is never expanded.
    """

    if not 0 < endpoint <= 1:
        raise ValueError("endpoint must lie in (0,1]")
    if precision_bits < 80:
        raise ValueError("precision_bits must be at least 80")
    if initial_power < 0:
        raise ValueError("initial_power must be nonnegative")
    if maximum_power < initial_power:
        raise ValueError("maximum_power must be at least initial_power")

    leaves: list[ScaledDyadicCell] = []
    pending = [ScaledDyadicCell(left, initial_power) for left in range(1 << initial_power)]
    with ctx.workprec(precision_bits):
        coefficients = tuple(_arb_rational(value) for value in JORDAN_QUINTIC.fractions())
        lower_ball = _arb_rational(derivative_lower)
        while pending:
            cell = pending.pop()
            derivative = _jordan_derivative_ball(
                _scaled_cell_ball(cell, endpoint), coefficients=coefficients
            )
            if derivative > lower_ball:
                leaves.append(cell)
                continue
            if cell.power >= maximum_power:
                raise RuntimeError(
                    "could not certify prefix derivative lower bound on "
                    f"{endpoint} at cell {cell.left}/2^{cell.power}"
                )
            child_power = cell.power + 1
            pending.append(ScaledDyadicCell(2 * cell.left + 1, child_power))
            pending.append(ScaledDyadicCell(2 * cell.left, child_power))

    ordered = _ordered_scaled_leaves(leaves)
    return PrefixRangeCertificate(
        precision_bits=precision_bits,
        endpoint=endpoint,
        derivative_lower=derivative_lower,
        initial_power=initial_power,
        configured_maximum_power=maximum_power,
        maximum_power=max(cell.power for cell in ordered),
        leaves=ordered,
        trace_sha256=_prefix_trace_digest(endpoint, derivative_lower, ordered),
    )


def exact_rank_two_pair_deficit() -> Fraction:
    """Return the exact pairwise deficit of the rational rank-two witness."""

    if PAIR_DIRECTION_LOW**2 + PAIR_DIRECTION_HIGH**2 != 1:
        raise AssertionError("locked witness direction is not exactly unit length")
    low_response, _ = scalar_response_and_derivative_exact(
        PAIR_NORMALIZED_RADIUS * PAIR_DIRECTION_LOW,
        JORDAN_QUINTIC,
        steps=5,
    )
    high_response, _ = scalar_response_and_derivative_exact(
        PAIR_NORMALIZED_RADIUS * PAIR_DIRECTION_HIGH,
        JORDAN_QUINTIC,
        steps=5,
    )
    raw_difference = PAIR_RAW_RADIUS * (PAIR_DIRECTION_HIGH - PAIR_DIRECTION_LOW)
    return -(high_response - low_response) / raw_difference


def exact_rank_two_pair_deficit_sha256() -> str:
    """Hash the roughly 50,000-digit exact witness quotient."""

    return fraction_sha256(exact_rank_two_pair_deficit())


def certify_additive_epsilon_intervals(*, precision_bits: int = 160) -> AdditiveEpsilonIntervalPass:
    """Replay every Arb obligation at one working precision."""

    global_range = certify_jordan_derivative_range(precision_bits=precision_bits)
    prefix_ranges = tuple(
        certify_prefix_derivative_lower(
            band.right,
            band.derivative_lower,
            precision_bits=precision_bits,
        )
        for band in RADIUS_BANDS[:-1]
    )
    return AdditiveEpsilonIntervalPass(
        precision_bits=precision_bits,
        global_range=global_range,
        prefix_ranges=prefix_ranges,
    )


def _negative_jacobian_envelope(
    t: Fraction,
    derivative_lower: Fraction,
    projection_lower: Fraction,
) -> Fraction:
    """Return the deficit envelope at normalized radius ``t``."""

    return -(1 - t) * ((1 - t) * derivative_lower + t * projection_lower)


def radius_band_bound(band: RadiusBand) -> RadiusBandBound:
    """Exactly maximize one band's quadratic Jacobian-deficit envelope."""

    projection_lower = projection_envelope_lower(band.derivative_lower, JORDAN_DERIVATIVE_UPPER)
    candidates = [
        (
            _negative_jacobian_envelope(band.left, band.derivative_lower, projection_lower),
            band.left,
        ),
        (
            _negative_jacobian_envelope(band.right, band.derivative_lower, projection_lower),
            band.right,
        ),
    ]
    if band.derivative_lower != projection_lower:
        stationary = (2 * band.derivative_lower - projection_lower) / (
            2 * (band.derivative_lower - projection_lower)
        )
        if band.left <= stationary <= band.right:
            candidates.append(
                (
                    _negative_jacobian_envelope(
                        stationary, band.derivative_lower, projection_lower
                    ),
                    stationary,
                )
            )
    deficit_upper, maximizing_radius = max(candidates)
    if deficit_upper < 0:
        deficit_upper = Fraction(0)
    return RadiusBandBound(
        band=band,
        projection_lower=projection_lower,
        maximizing_radius=maximizing_radius,
        deficit_upper=deficit_upper,
    )


def radius_band_deficit_upper(band: RadiusBand) -> Fraction:
    """Return one band's exact deficit upper bound."""

    return radius_band_bound(band).deficit_upper


def certified_unit_epsilon_deficit_upper() -> Fraction:
    """Return the dimension-uniform upper certificate for ``delta(E_h,1)``."""

    return max(radius_band_deficit_upper(band) for band in RADIUS_BANDS)


def certified_repair_conductance(epsilon: Fraction) -> Fraction:
    """Return the sufficient constant repair ``bar_delta_1 / epsilon``."""

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    return certified_unit_epsilon_deficit_upper() / epsilon


def scale_unit_deficit(value: Fraction, epsilon: Fraction) -> Fraction:
    """Scale a unit-epsilon deficit bound or witness by the exact ``1/eps`` law."""

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    return value / epsilon


def zero_input_derivative_gain(epsilon: Fraction) -> Fraction:
    """Return the positive Frechet derivative gain at ``M=0``."""

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    a, _, _ = JORDAN_QUINTIC.fractions()
    return a**5 / epsilon
