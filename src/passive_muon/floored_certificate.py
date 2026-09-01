"""Rigorous scalar certificate and analytic full-matrix repair bound.

The only numerical proof obligation is a one-dimensional enclosure of the
derivative of the five-step Jordan scalar map.  Arb ball arithmetic verifies
that enclosure on an adaptive dyadic cover of ``[0, 1]``.  The matrix bound is
then exact rational arithmetic.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from fractions import Fraction

from flint import arb, ctx, fmpq

from passive_muon.specs import JORDAN_QUINTIC, scalar_response_and_derivative_exact

if hasattr(sys, "set_int_max_str_digits"):
    # The exact five-stage finite-pair ratio has about 39,000 decimal digits.
    sys.set_int_max_str_digits(0)


JORDAN_DERIVATIVE_LOWER = Fraction(-1_595_496, 10_000)
JORDAN_DERIVATIVE_UPPER = Fraction(4_848_763, 10_000)

JORDAN_PAIR_LEFT = Fraction(3_174_788_063, 500_000_000_000)
JORDAN_PAIR_RIGHT = Fraction(3_174_789_063, 500_000_000_000)
JORDAN_PAIR_DEFICIT_LOWER = Fraction(31_909_905_157, 200_000_000)


@dataclass(frozen=True, order=True)
class DyadicCell:
    """Closed interval ``[left/2**power, (left+1)/2**power]``."""

    left: int
    power: int

    def __post_init__(self) -> None:
        if self.power < 0:
            raise ValueError("power must be nonnegative")
        if not 0 <= self.left < 1 << self.power:
            raise ValueError("dyadic cell lies outside [0, 1]")


@dataclass(frozen=True)
class ScalarRangeCertificate:
    """Summary of a replayable adaptive Arb proof cover."""

    precision_bits: int
    initial_power: int
    configured_maximum_power: int
    maximum_power: int
    derivative_lower: Fraction
    derivative_upper: Fraction
    leaves: tuple[DyadicCell, ...]
    trace_sha256: str

    @property
    def leaf_count(self) -> int:
        return len(self.leaves)


def _arb_rational(value: Fraction) -> arb:
    return arb(fmpq(value.numerator, value.denominator))


def _dyadic_ball(cell: DyadicCell) -> arb:
    denominator = 1 << cell.power
    lower = arb(fmpq(cell.left, denominator))
    upper = arb(fmpq(cell.left + 1, denominator))
    return lower.union(upper)


def _jordan_derivative_ball(
    value: arb,
    *,
    coefficients: tuple[arb, arb, arb],
    steps: int = 5,
) -> arb:
    """Enclose the composed derivative without expanding the polynomial."""

    a, b, c = coefficients
    derivative = arb(1)
    for _ in range(steps):
        square = value * value
        slope = a + square * (3 * b + 5 * c * square)
        derivative = derivative * slope
        value = value * (a + square * (b + c * square))
    return derivative


def _ordered_leaves(leaves: list[DyadicCell]) -> tuple[DyadicCell, ...]:
    maximum_power = max(cell.power for cell in leaves)
    ordered = tuple(sorted(leaves, key=lambda cell: cell.left << (maximum_power - cell.power)))
    expected_left = 0
    for cell in ordered:
        scaled_left = cell.left << (maximum_power - cell.power)
        scaled_width = 1 << (maximum_power - cell.power)
        if scaled_left != expected_left:
            raise AssertionError("adaptive cells do not form a contiguous cover")
        expected_left += scaled_width
    if expected_left != 1 << maximum_power:
        raise AssertionError("adaptive cells do not cover all of [0, 1]")
    return ordered


def _trace_digest(leaves: tuple[DyadicCell, ...]) -> str:
    canonical = "".join(f"{cell.left}/{cell.power}\n" for cell in leaves).encode()
    return hashlib.sha256(canonical).hexdigest()


def certify_jordan_derivative_range(
    *,
    precision_bits: int = 160,
    initial_power: int = 12,
    maximum_power: int = 48,
    derivative_lower: Fraction = JORDAN_DERIVATIVE_LOWER,
    derivative_upper: Fraction = JORDAN_DERIVATIVE_UPPER,
) -> ScalarRangeCertificate:
    """Certify ``derivative_lower < h'(s) < derivative_upper`` on ``[0, 1]``.

    Every accepted dyadic leaf is checked by outward-rounded Arb arithmetic.
    Cells whose derivative ball overlaps either requested bound are bisected.
    The returned cover is deterministic and its digest can be locked in a
    result manifest.
    """

    if precision_bits < 80:
        raise ValueError("precision_bits must be at least 80")
    if initial_power < 0:
        raise ValueError("initial_power must be nonnegative")
    if maximum_power < initial_power:
        raise ValueError("maximum_power must be at least initial_power")
    if derivative_lower >= derivative_upper:
        raise ValueError("derivative bounds must be ordered")

    leaves: list[DyadicCell] = []
    pending = [DyadicCell(left, initial_power) for left in range(1 << initial_power)]
    with ctx.workprec(precision_bits):
        coefficients = tuple(_arb_rational(value) for value in JORDAN_QUINTIC.fractions())
        lower_ball = _arb_rational(derivative_lower)
        upper_ball = _arb_rational(derivative_upper)
        while pending:
            cell = pending.pop()
            derivative = _jordan_derivative_ball(_dyadic_ball(cell), coefficients=coefficients)
            if derivative > lower_ball and derivative < upper_ball:
                leaves.append(cell)
                continue
            if cell.power >= maximum_power:
                raise RuntimeError(
                    "could not certify requested derivative range on "
                    f"[{cell.left}/2^{cell.power}, {cell.left + 1}/2^{cell.power}]"
                )
            child_power = cell.power + 1
            pending.append(DyadicCell(2 * cell.left + 1, child_power))
            pending.append(DyadicCell(2 * cell.left, child_power))

    ordered = _ordered_leaves(leaves)
    return ScalarRangeCertificate(
        precision_bits=precision_bits,
        initial_power=initial_power,
        configured_maximum_power=maximum_power,
        maximum_power=max(cell.power for cell in ordered),
        derivative_lower=derivative_lower,
        derivative_upper=derivative_upper,
        leaves=ordered,
        trace_sha256=_trace_digest(ordered),
    )


def projection_envelope_lower(slope_lower: Fraction, slope_upper: Fraction) -> Fraction:
    """Sharp uniform lower bound for ``Sym(AQ)`` from ``a I <= A <= b I``.

    Here ``Q`` is any orthogonal projection and ``Sym(AQ)=(AQ+QA)/2``.
    The result is exact rational arithmetic.
    """

    if slope_lower > slope_upper:
        raise ValueError("slope bounds must be ordered")
    if slope_lower + slope_upper > 0 and slope_upper + 3 * slope_lower >= 0:
        return -((slope_upper - slope_lower) ** 2) / (8 * (slope_upper + slope_lower))
    return min(slope_lower, Fraction(0))


def certified_dimension_uniform_deficit() -> Fraction:
    """Return the certified full-matrix deficit upper bound for floor one."""

    envelope = projection_envelope_lower(JORDAN_DERIVATIVE_LOWER, JORDAN_DERIVATIVE_UPPER)
    return max(Fraction(0), -envelope)


def certified_repair_conductance(floor: Fraction, *, mu: Fraction = Fraction(0)) -> Fraction:
    """Return the sufficient conductance ``bar_delta_1 / floor + mu``."""

    if floor <= 0:
        raise ValueError("floor must be positive")
    if mu < 0:
        raise ValueError("mu must be nonnegative")
    return certified_dimension_uniform_deficit() / floor + mu


def exact_jordan_pair_deficit() -> Fraction:
    """Return an exact finite-pair lower witness inside the unit floor."""

    left_response, _ = scalar_response_and_derivative_exact(
        JORDAN_PAIR_LEFT, JORDAN_QUINTIC, steps=5
    )
    right_response, _ = scalar_response_and_derivative_exact(
        JORDAN_PAIR_RIGHT, JORDAN_QUINTIC, steps=5
    )
    return -(right_response - left_response) / (JORDAN_PAIR_RIGHT - JORDAN_PAIR_LEFT)


def fraction_sha256(value: Fraction) -> str:
    """Hash a canonical exact fraction without serializing it into manifests."""

    return hashlib.sha256(f"{value.numerator}/{value.denominator}".encode()).hexdigest()
