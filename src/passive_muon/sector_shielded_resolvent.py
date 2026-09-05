"""Exact and fail-closed FP64 projection onto the locked P18 sector disk.

For the P18 pointwise sector ``[m, M]``, put

``gamma = (m + M) / 2`` and ``radius = (M - m) / 2``.

Then the sector inequality is exactly the moving-ball condition

``||U - gamma*S||_F <= radius*||S||_F``.

This module exposes the metric projection onto that ball.  The exact helper
uses algebraic SymPy expressions when the radial projection introduces a
square root.  The FP64 helper is deliberately a reference implementation:
it uses scaled norms, projects to an inward-rounded radius, and then verifies
the *stored binary64 output* against the original rational disk with exact
dyadic arithmetic.  A successful return is therefore a certificate, not a
floating-point tolerance check.

The exact check is intentionally expensive.  P19 values correctness and a
fail-closed interface; a scalable implementation certificate belongs to P20.
In particular, a sufficiently small nonzero subnormal signal need not admit
any binary64 point in this strict sector.  Such a call raises rather than
emitting an uncertified update.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Final

import numpy as np
from numpy.typing import NDArray
from sympy import Expr, Rational, simplify, sqrt

from passive_muon.equivariant_resolvent_solver import EquivariantResolventSolverConfig
from passive_muon.sector_projected_resolvent import (
    LOCKED_POINTWISE_SECTOR,
    SectorProjectedMatrixEvaluation,
    SectorProjectedSingularValueEvaluation,
    evaluate_sector_projected_resolvent_fp64,
    evaluate_sector_projected_singular_values_fp64,
)

FloatArray = NDArray[np.float64]

LOCKED_SHIELD_CENTER: Final = LOCKED_POINTWISE_SECTOR.center
LOCKED_SHIELD_RADIUS: Final = LOCKED_POINTWISE_SECTOR.radius
LOCKED_FP64_INWARD_ULPS: Final = 32
LOCKED_FP64_FALLBACK_GAIN: Final = Fraction(1, 2)


def _require_fraction(value: Fraction, name: str) -> None:
    if not isinstance(value, Fraction):
        raise TypeError(f"{name} must be a fractions.Fraction")


def _require_fraction_vector(values: Sequence[Fraction], name: str) -> tuple[Fraction, ...]:
    selected = tuple(values)
    if not selected:
        raise ValueError(f"{name} must be nonempty")
    for value in selected:
        _require_fraction(value, f"{name} entry")
    return selected


def _sympy_rational(value: Fraction) -> Rational:
    return Rational(value.numerator, value.denominator)


@dataclass(frozen=True)
class ExactSectorShieldProjection:
    """Exact algebraic metric projection onto one locked moving disk."""

    output: tuple[Expr, ...]
    scale: Expr
    candidate_inside: bool
    active: bool
    signal_squared_norm: Fraction
    candidate_displacement_squared_norm: Fraction
    disk_squared_radius: Fraction
    sector_margin: Expr

    @property
    def certified(self) -> bool:
        """Whether exact symbolic simplification proves disk membership."""

        nonnegative = self.sector_margin.is_nonnegative
        return nonnegative is True


@dataclass(frozen=True)
class FP64SectorShieldDiagnostics:
    """Diagnostics for a successfully certified binary64 shield call."""

    candidate_inside: bool
    active: bool
    fail_closed: bool
    reason: str | None
    used_interior_fallback: bool
    inward_ulps: int
    inward_radius: float
    signal_scale: float
    signal_scaled_norm: float
    displacement_scale: float
    displacement_scaled_norm: float
    exact_sector_margin: Fraction

    @property
    def certified(self) -> bool:
        """A constructed result is certified precisely by its exact margin."""

        return self.exact_sector_margin >= 0


@dataclass(frozen=True)
class FP64SectorShieldResult:
    """A finite binary64 update that passed the mandatory exact postcheck."""

    output: FloatArray
    diagnostics: FP64SectorShieldDiagnostics


@dataclass(frozen=True)
class SectorShieldedSingularValueEvaluation:
    """P18 singular-coordinate evaluation followed by the P19 shield."""

    p18_evaluation: SectorProjectedSingularValueEvaluation
    output: FloatArray
    shield_diagnostics: FP64SectorShieldDiagnostics


@dataclass(frozen=True)
class SectorShieldedMatrixEvaluation:
    """P18 full-matrix evaluation followed by the P19 shield."""

    p18_evaluation: SectorProjectedMatrixEvaluation
    output: FloatArray
    shield_diagnostics: FP64SectorShieldDiagnostics


class SectorShieldFailure(FloatingPointError):
    """Raised when the shield cannot emit a certified binary64 update."""


def project_onto_sector_disk_exact(
    candidate: Sequence[Fraction],
    signal: Sequence[Fraction],
    *,
    center: Fraction = LOCKED_SHIELD_CENTER,
    radius: Fraction = LOCKED_SHIELD_RADIUS,
) -> ExactSectorShieldProjection:
    """Project a rational candidate onto ``D_S`` with exact algebraic output.

    The projection is total at ``S=0``: the disk is the singleton ``{0}``.
    Outside the disk the common radial scale contains an exact square root,
    so output coordinates are returned as SymPy expressions rather than
    prematurely rounded rationals.
    """

    selected_candidate = _require_fraction_vector(candidate, "candidate")
    selected_signal = _require_fraction_vector(signal, "signal")
    _require_fraction(center, "center")
    _require_fraction(radius, "radius")
    if len(selected_candidate) != len(selected_signal):
        raise ValueError("candidate and signal must have equal lengths")
    if radius < 0 or center < radius:
        raise ValueError("disk parameters must satisfy center >= radius >= 0")

    centered = tuple(
        value - center * source
        for value, source in zip(selected_candidate, selected_signal, strict=True)
    )
    signal_squared_norm = sum(value * value for value in selected_signal)
    displacement_squared_norm = sum(value * value for value in centered)
    disk_squared_radius = radius * radius * signal_squared_norm
    candidate_inside = displacement_squared_norm <= disk_squared_radius

    if candidate_inside:
        scale: Expr = Rational(1)
    elif displacement_squared_norm == 0:
        # This branch can only occur for an invalid negative radius, already
        # rejected above, but retaining it makes the formula visibly total.
        scale = Rational(0)
    else:
        scale = sqrt(
            _sympy_rational(disk_squared_radius) / _sympy_rational(displacement_squared_norm)
        )

    exact_center = tuple(_sympy_rational(center * source) for source in selected_signal)
    exact_displacement = tuple(_sympy_rational(value) for value in centered)
    output = tuple(
        simplify(base + scale * delta)
        for base, delta in zip(exact_center, exact_displacement, strict=True)
    )
    symbolic_displacement_squared = sum(
        simplify(value - base) ** 2 for value, base in zip(output, exact_center, strict=True)
    )
    margin = simplify(_sympy_rational(disk_squared_radius) - symbolic_displacement_squared)
    result = ExactSectorShieldProjection(
        output=output,
        scale=scale,
        candidate_inside=candidate_inside,
        active=not candidate_inside,
        signal_squared_norm=signal_squared_norm,
        candidate_displacement_squared_norm=displacement_squared_norm,
        disk_squared_radius=disk_squared_radius,
        sector_margin=margin,
    )
    if not result.certified:
        raise ArithmeticError("exact sector-shield projection did not certify")
    return result


def _require_fp64_pair(
    candidate: FloatArray,
    signal: FloatArray,
    *,
    require_finite_candidate: bool,
) -> None:
    for value, name in ((candidate, "candidate"), (signal, "signal")):
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{name} must be a numpy.ndarray")
        if value.dtype != np.float64:
            raise TypeError(f"{name} must have dtype numpy.float64")
        if value.size == 0:
            raise ValueError(f"{name} must be nonempty")
    if candidate.shape != signal.shape:
        raise ValueError("candidate and signal must have equal shapes")
    if not bool(np.isfinite(signal).all()):
        raise SectorShieldFailure("nonfinite signal rejected without emitting an update")
    if require_finite_candidate and not bool(np.isfinite(candidate).all()):
        raise SectorShieldFailure("candidate must contain only finite entries")


def _balanced_norm_parts(values: FloatArray) -> tuple[float, float]:
    """Return ``(scale, ||values/scale||_F)`` without risky squaring."""

    maximum = float(np.max(np.abs(values)))
    if maximum == 0.0:
        return 0.0, 0.0
    scaled = np.ascontiguousarray(values / maximum, dtype=np.float64)
    # Every summand is in [0,1].  fsum reduces accumulation error while the
    # maximum scaling prevents overflow and avoids premature underflow.
    squared = math.fsum(float(value) * float(value) for value in scaled.ravel())
    return maximum, math.sqrt(squared)


def _fraction_vector_from_fp64(values: FloatArray) -> tuple[Fraction, ...]:
    return tuple(Fraction.from_float(float(value)) for value in values.ravel())


def fp64_sector_disk_margin_exact(
    output: FloatArray,
    signal: FloatArray,
    *,
    center: Fraction = LOCKED_SHIELD_CENTER,
    radius: Fraction = LOCKED_SHIELD_RADIUS,
) -> Fraction:
    """Check stored binary64 arrays against ``D_S`` with exact arithmetic."""

    _require_fp64_pair(output, signal, require_finite_candidate=True)
    _require_fraction(center, "center")
    _require_fraction(radius, "radius")
    if radius < 0 or center < radius:
        raise ValueError("disk parameters must satisfy center >= radius >= 0")
    exact_output = _fraction_vector_from_fp64(output)
    exact_signal = _fraction_vector_from_fp64(signal)
    displacement_squared = sum(
        (value - center * source) ** 2
        for value, source in zip(exact_output, exact_signal, strict=True)
    )
    signal_squared = sum(value * value for value in exact_signal)
    return radius * radius * signal_squared - displacement_squared


def fp64_output_in_exact_sector(output: FloatArray, signal: FloatArray) -> bool:
    """Return whether the stored binary64 output lies in the locked P18 disk."""

    return fp64_sector_disk_margin_exact(output, signal) >= 0


def _move_float_inward(value: float, ulps: int) -> float:
    selected = value
    for _ in range(ulps):
        selected = math.nextafter(selected, 0.0)
    return selected


def _safe_interior_fallback(signal: FloatArray) -> tuple[FloatArray, Fraction]:
    """Return a dyadic half-gain update, or fail if it is unrepresentable."""

    output = np.ascontiguousarray(0.5 * signal, dtype=np.float64)
    if not bool(np.isfinite(output).all()):
        raise SectorShieldFailure("interior fallback overflowed")
    margin = fp64_sector_disk_margin_exact(output, signal)
    if margin < 0:
        # For example, half of the least positive subnormal rounds to zero;
        # neither zero nor the original subnormal lies in this strict sector.
        raise SectorShieldFailure(
            "no certified dyadic interior update was representable for this signal"
        )
    return output, margin


def shield_sector_candidate_fp64(
    signal: FloatArray,
    candidate: FloatArray,
    *,
    inward_ulps: int = LOCKED_FP64_INWARD_ULPS,
) -> FP64SectorShieldResult:
    """Project ``candidate`` into the locked P18 disk and certify the result.

    The first membership test and the final postcondition use exact rational
    arithmetic on the stored binary64 values.  A candidate already inside is
    returned bit-for-bit unchanged.  Outside candidates use a balanced,
    common-scale radial projection to an inward-rounded radius.  If that
    construction does not pass the exact postcheck, the function falls back
    to the dyadic interior point ``S/2``.  Nonfinite candidates use the same
    fallback and are marked fail-closed; nonfinite signals are rejected.
    """

    _require_fp64_pair(candidate, signal, require_finite_candidate=False)
    if not isinstance(inward_ulps, int):
        raise TypeError("inward_ulps must be an integer")
    if inward_ulps < 1:
        raise ValueError("inward_ulps must be positive")

    inward_radius = _move_float_inward(float(LOCKED_SHIELD_RADIUS), inward_ulps)
    candidate_finite = bool(np.isfinite(candidate).all())
    signal_scale, signal_scaled_norm = _balanced_norm_parts(signal)

    if signal_scale == 0.0:
        output = np.zeros_like(signal)
        candidate_inside = candidate_finite and bool(np.equal(candidate, 0.0).all())
        return FP64SectorShieldResult(
            output=output,
            diagnostics=FP64SectorShieldDiagnostics(
                candidate_inside=candidate_inside,
                active=not candidate_inside,
                fail_closed=not candidate_finite,
                reason=None if candidate_finite else "nonfinite_candidate_zero_signal",
                used_interior_fallback=not candidate_finite,
                inward_ulps=inward_ulps,
                inward_radius=inward_radius,
                signal_scale=0.0,
                signal_scaled_norm=0.0,
                displacement_scale=0.0,
                displacement_scaled_norm=0.0,
                exact_sector_margin=Fraction(0),
            ),
        )

    if not candidate_finite:
        output, margin = _safe_interior_fallback(signal)
        return FP64SectorShieldResult(
            output=output,
            diagnostics=FP64SectorShieldDiagnostics(
                candidate_inside=False,
                active=True,
                fail_closed=True,
                reason="nonfinite_candidate_interior_fallback",
                used_interior_fallback=True,
                inward_ulps=inward_ulps,
                inward_radius=inward_radius,
                signal_scale=signal_scale,
                signal_scaled_norm=signal_scaled_norm,
                displacement_scale=math.nan,
                displacement_scaled_norm=math.nan,
                exact_sector_margin=margin,
            ),
        )

    candidate_margin = fp64_sector_disk_margin_exact(candidate, signal)
    if candidate_margin >= 0:
        return FP64SectorShieldResult(
            output=np.ascontiguousarray(candidate.copy(), dtype=np.float64),
            diagnostics=FP64SectorShieldDiagnostics(
                candidate_inside=True,
                active=False,
                fail_closed=False,
                reason=None,
                used_interior_fallback=False,
                inward_ulps=inward_ulps,
                inward_radius=inward_radius,
                signal_scale=signal_scale,
                signal_scaled_norm=signal_scaled_norm,
                displacement_scale=0.0,
                displacement_scaled_norm=0.0,
                exact_sector_margin=candidate_margin,
            ),
        )

    common_scale = max(signal_scale, float(np.max(np.abs(candidate))))
    scaled_signal = np.ascontiguousarray(signal / common_scale, dtype=np.float64)
    scaled_candidate = np.ascontiguousarray(candidate / common_scale, dtype=np.float64)
    gamma = float(LOCKED_SHIELD_CENTER)
    scaled_center = np.ascontiguousarray(gamma * scaled_signal, dtype=np.float64)
    scaled_displacement = np.ascontiguousarray(
        scaled_candidate - scaled_center,
        dtype=np.float64,
    )
    displacement_scale, displacement_scaled_norm = _balanced_norm_parts(scaled_displacement)
    scaled_signal_scale, scaled_signal_unit_norm = _balanced_norm_parts(scaled_signal)
    if displacement_scale == 0.0:
        # Exact arithmetic already proved the candidate was outside, so this
        # can only be a floating-point collapse.  Use the safe interior point.
        output, margin = _safe_interior_fallback(signal)
        return FP64SectorShieldResult(
            output=output,
            diagnostics=FP64SectorShieldDiagnostics(
                candidate_inside=False,
                active=True,
                fail_closed=True,
                reason="scaled_displacement_collapsed_to_zero",
                used_interior_fallback=True,
                inward_ulps=inward_ulps,
                inward_radius=inward_radius,
                signal_scale=signal_scale,
                signal_scaled_norm=signal_scaled_norm,
                displacement_scale=0.0,
                displacement_scaled_norm=0.0,
                exact_sector_margin=margin,
            ),
        )

    radial_scale = (
        inward_radius
        * (scaled_signal_scale / displacement_scale)
        * (scaled_signal_unit_norm / displacement_scaled_norm)
    )
    radial_scale = min(math.nextafter(1.0, 0.0), max(0.0, radial_scale))
    scaled_output = np.ascontiguousarray(
        scaled_center + radial_scale * scaled_displacement,
        dtype=np.float64,
    )
    output = np.ascontiguousarray(scaled_output * common_scale, dtype=np.float64)

    used_fallback = False
    fail_closed = False
    reason: str | None = None
    if bool(np.isfinite(output).all()):
        margin = fp64_sector_disk_margin_exact(output, signal)
    else:
        margin = Fraction(-1)
    if margin < 0:
        output, margin = _safe_interior_fallback(signal)
        used_fallback = True
        fail_closed = True
        reason = "inward_projection_failed_exact_postcheck"

    return FP64SectorShieldResult(
        output=output,
        diagnostics=FP64SectorShieldDiagnostics(
            candidate_inside=False,
            active=True,
            fail_closed=fail_closed,
            reason=reason,
            used_interior_fallback=used_fallback,
            inward_ulps=inward_ulps,
            inward_radius=inward_radius,
            signal_scale=signal_scale,
            signal_scaled_norm=signal_scaled_norm,
            displacement_scale=displacement_scale,
            displacement_scaled_norm=displacement_scaled_norm,
            exact_sector_margin=margin,
        ),
    )


def evaluate_sector_shielded_singular_values_fp64(
    signal_values: FloatArray,
    *,
    solver_config: EquivariantResolventSolverConfig | None = None,
    inward_ulps: int = LOCKED_FP64_INWARD_ULPS,
) -> SectorShieldedSingularValueEvaluation:
    """Run the guarded P18 singular evaluator and apply the P19 shield."""

    base = evaluate_sector_projected_singular_values_fp64(
        signal_values,
        solver_config=solver_config,
    )
    shielded = shield_sector_candidate_fp64(
        signal_values,
        base.output,
        inward_ulps=inward_ulps,
    )
    return SectorShieldedSingularValueEvaluation(
        p18_evaluation=base,
        output=shielded.output,
        shield_diagnostics=shielded.diagnostics,
    )


def evaluate_sector_shielded_resolvent_fp64(
    signal: FloatArray,
    *,
    solver_config: EquivariantResolventSolverConfig | None = None,
    inward_ulps: int = LOCKED_FP64_INWARD_ULPS,
) -> SectorShieldedMatrixEvaluation:
    """Run the guarded P18 matrix evaluator and apply the P19 shield."""

    base = evaluate_sector_projected_resolvent_fp64(
        signal,
        solver_config=solver_config,
    )
    shielded = shield_sector_candidate_fp64(
        signal,
        base.output,
        inward_ulps=inward_ulps,
    )
    return SectorShieldedMatrixEvaluation(
        p18_evaluation=base,
        output=shielded.output,
        shield_diagnostics=shielded.diagnostics,
    )


__all__ = [
    "LOCKED_FP64_FALLBACK_GAIN",
    "LOCKED_FP64_INWARD_ULPS",
    "LOCKED_SHIELD_CENTER",
    "LOCKED_SHIELD_RADIUS",
    "ExactSectorShieldProjection",
    "FP64SectorShieldDiagnostics",
    "FP64SectorShieldResult",
    "SectorShieldFailure",
    "SectorShieldedMatrixEvaluation",
    "SectorShieldedSingularValueEvaluation",
    "evaluate_sector_shielded_resolvent_fp64",
    "evaluate_sector_shielded_singular_values_fp64",
    "fp64_output_in_exact_sector",
    "fp64_sector_disk_margin_exact",
    "project_onto_sector_disk_exact",
    "shield_sector_candidate_fp64",
]
