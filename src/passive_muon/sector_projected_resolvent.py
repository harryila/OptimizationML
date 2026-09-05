"""Origin-sector projection for a shape-preserving resolvent interface.

Let ``J`` and ``Y`` be the P16 resolvent and Yosida maps, and put

``X(S) = E_h,epsilon(J(S))``.

P17 exposed ``X`` directly on the shape branch.  This module instead defines

``alpha_K = min(1, K*<X,S>/||X||**2)``, ``Z = alpha_K*X``

when ``X`` is nonzero, with ``Z=0`` when ``X=0``, and then

``T = (1-theta)*Y/c + theta*Z``.

The P16 singular-vector theorem and positivity of the five-stage Jordan map
give ``<X(S),S> >= 0``.  Exact algebra then gives

``||Z||**2 <= K*<Z,S>``,

so ``Z`` lies in the origin-centred pointwise disk sector ``[0,K]``.  If
``Y/c`` lies in ``[a,b]=[500/c,1000/c]`` and ``0<=theta<=h``, the uniform
pointwise sector of ``T`` is

``[(1-h)*a, max(b, (1-h)*b+h*K)]``.

This is a dimension-uniform pointwise statement.  It is not an incremental
sector: derivatives of both the radial gate and ``alpha_K`` require separate
analysis.  The exact helpers below operate on :class:`fractions.Fraction`;
the FP64 helpers are implementation diagnostics, not rounding certificates.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Final

import numpy as np
from numpy.typing import NDArray

from passive_muon.equivariant_resolvent_solver import (
    EquivariantResolventSolverConfig,
    MatrixSolveResult,
    SolverDiagnostics,
)
from passive_muon.shape_preserving_resolvent import (
    PRIMARY_DESIGN,
    evaluate_gated_singular_values_fp64,
    evaluate_shape_preserving_resolvent_fp64,
)
from passive_muon.yosida_stability import (
    LOCKED_YOSIDA_LIPSCHITZ,
    LOCKED_YOSIDA_STRONG_MONOTONICITY,
)

FloatArray = NDArray[np.float64]

LOCKED_PROJECTION_GAIN: Final = Fraction(1)
LOCKED_PASSIVE_DIVISOR: Final = Fraction(1_024)
LOCKED_GATE_CAP: Final = Fraction(3, 4)


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


@dataclass(frozen=True)
class SectorProjectedDesign:
    """Exact parameters of an origin-sector-projected gated interface."""

    projection_gain: Fraction
    passive_divisor: Fraction
    gate_cap: Fraction

    def __post_init__(self) -> None:
        _require_fraction(self.projection_gain, "projection_gain")
        _require_fraction(self.passive_divisor, "passive_divisor")
        _require_fraction(self.gate_cap, "gate_cap")
        if self.projection_gain < 0:
            raise ValueError("projection_gain must be nonnegative")
        if self.passive_divisor <= 0:
            raise ValueError("passive_divisor must be positive")
        if not 0 <= self.gate_cap <= 1:
            raise ValueError("gate_cap must lie in [0,1]")


@dataclass(frozen=True)
class ProjectedPointwiseSector:
    """Exact origin-centred pointwise sector enclosing the gated blend."""

    lower: Fraction
    upper: Fraction
    center: Fraction
    radius: Fraction

    def __post_init__(self) -> None:
        for name in ("lower", "upper", "center", "radius"):
            _require_fraction(getattr(self, name), name)
        if not 0 <= self.lower <= self.upper:
            raise ValueError("pointwise sector must satisfy 0 <= lower <= upper")
        if self.center != (self.lower + self.upper) / 2:
            raise ValueError("center must be the sector midpoint")
        if self.radius != (self.upper - self.lower) / 2:
            raise ValueError("radius must be half the sector width")

    @property
    def condition_ratio(self) -> Fraction | None:
        """Return ``upper/lower``, or ``None`` when the lower gain is zero."""

        return None if self.lower == 0 else self.upper / self.lower


@dataclass(frozen=True)
class ExactSectorProjection:
    """Exact projection data for one finite-dimensional vectorization."""

    scale: Fraction
    output: tuple[Fraction, ...]
    shape_signal_inner_product: Fraction
    shape_squared_norm: Fraction
    projected_signal_inner_product: Fraction
    projected_squared_norm: Fraction
    sector_margin: Fraction

    @property
    def certified(self) -> bool:
        """Whether the exact ``[0,K]`` sector margin is nonnegative."""

        return 0 <= self.scale <= 1 and self.sector_margin >= 0


@dataclass(frozen=True)
class ExactProjectedBlend:
    """Exact projected shape branch and gated output for one vectorization."""

    output: tuple[Fraction, ...]
    projection: ExactSectorProjection
    gate_weight: Fraction
    global_sector: ProjectedPointwiseSector
    sector_margin: Fraction

    @property
    def certified(self) -> bool:
        return self.projection.certified and self.sector_margin >= 0


@dataclass(frozen=True)
class SectorProjectedSingularValueEvaluation:
    """Computed-residual-checked FP64 evaluation in singular coordinates."""

    solution_singular_values: FloatArray
    yosida_output: FloatArray
    shape_output: FloatArray
    projection_alpha: float
    shape_signal_inner_product: float
    shape_squared_norm: float
    projection_active: bool
    projected_shape_output: FloatArray
    output: FloatArray
    gate_value: float
    squared_signal_norm: float
    solver_diagnostics: SolverDiagnostics


@dataclass(frozen=True)
class SectorProjectedMatrixEvaluation:
    """Computed-residual-checked FP64 full-matrix evaluation."""

    yosida_output: FloatArray
    shape_output: FloatArray
    projection_alpha: float
    shape_signal_inner_product: float
    shape_squared_norm: float
    projection_active: bool
    projected_shape_output: FloatArray
    output: FloatArray
    gate_value: float
    squared_signal_norm: float
    resolvent_result: MatrixSolveResult


LOCKED_SECTOR_PROJECTED_DESIGN: Final = SectorProjectedDesign(
    projection_gain=LOCKED_PROJECTION_GAIN,
    passive_divisor=LOCKED_PASSIVE_DIVISOR,
    gate_cap=LOCKED_GATE_CAP,
)


def sector_disk_margin_exact(
    output: Sequence[Fraction],
    signal: Sequence[Fraction],
    lower: Fraction,
    upper: Fraction,
) -> Fraction:
    """Return ``<output-lower*signal, upper*signal-output>`` exactly."""

    selected_output = _require_fraction_vector(output, "output")
    selected_signal = _require_fraction_vector(signal, "signal")
    _require_fraction(lower, "lower")
    _require_fraction(upper, "upper")
    if len(selected_output) != len(selected_signal):
        raise ValueError("output and signal must have equal lengths")
    if not 0 <= lower <= upper:
        raise ValueError("sector endpoints must satisfy 0 <= lower <= upper")
    return sum(
        (value - lower * source) * (upper * source - value)
        for value, source in zip(selected_output, selected_signal, strict=True)
    )


def sector_projection_scale_exact(
    shape_signal_inner_product: Fraction,
    shape_squared_norm: Fraction,
    projection_gain: Fraction = LOCKED_PROJECTION_GAIN,
) -> Fraction:
    """Return ``alpha_K`` exactly, with the zero-output case defined as zero.

    ``shape_signal_inner_product`` must be nonnegative.  For the locked map
    this follows analytically from its common singular vectors and
    nonnegative mode values.  A zero squared norm is required to have zero
    inner product; the returned scale is then zero so the definition is total
    at ``X=0``.
    """

    _require_fraction(shape_signal_inner_product, "shape_signal_inner_product")
    _require_fraction(shape_squared_norm, "shape_squared_norm")
    _require_fraction(projection_gain, "projection_gain")
    if shape_signal_inner_product < 0:
        raise ValueError("shape_signal_inner_product must be nonnegative")
    if shape_squared_norm < 0:
        raise ValueError("shape_squared_norm must be nonnegative")
    if projection_gain < 0:
        raise ValueError("projection_gain must be nonnegative")
    if shape_squared_norm == 0:
        if shape_signal_inner_product != 0:
            raise ValueError("zero shape norm requires zero shape-signal inner product")
        return Fraction(0)
    return min(
        Fraction(1),
        projection_gain * shape_signal_inner_product / shape_squared_norm,
    )


def project_onto_origin_sector_exact(
    shape_output: Sequence[Fraction],
    signal: Sequence[Fraction],
    projection_gain: Fraction = LOCKED_PROJECTION_GAIN,
) -> ExactSectorProjection:
    """Project ``shape_output`` into the pointwise disk sector ``[0,K]``.

    The vectors may be any finite vectorizations, so the helper covers every
    finite matrix shape.  It verifies the Gram-data consistency needed by the
    algebra rather than treating arbitrary scalar summaries as inner
    products of real vectors.
    """

    selected_shape = _require_fraction_vector(shape_output, "shape_output")
    selected_signal = _require_fraction_vector(signal, "signal")
    _require_fraction(projection_gain, "projection_gain")
    if len(selected_shape) != len(selected_signal):
        raise ValueError("shape_output and signal must have equal lengths")
    if projection_gain < 0:
        raise ValueError("projection_gain must be nonnegative")

    inner = sum(
        value * source for value, source in zip(selected_shape, selected_signal, strict=True)
    )
    shape_norm_squared = sum(value * value for value in selected_shape)
    signal_norm_squared = sum(source * source for source in selected_signal)
    if inner < 0:
        raise ValueError("shape_output must have nonnegative inner product with signal")
    if inner * inner > shape_norm_squared * signal_norm_squared:
        raise ValueError("inconsistent exact Gram data")

    scale = sector_projection_scale_exact(inner, shape_norm_squared, projection_gain)
    projected = tuple(scale * value for value in selected_shape)
    projected_inner = scale * inner
    projected_norm_squared = scale * scale * shape_norm_squared
    margin = projection_gain * projected_inner - projected_norm_squared
    if margin < 0:
        raise ArithmeticError("exact sector projection failed its defining inequality")
    return ExactSectorProjection(
        scale=scale,
        output=projected,
        shape_signal_inner_product=inner,
        shape_squared_norm=shape_norm_squared,
        projected_signal_inner_product=projected_inner,
        projected_squared_norm=projected_norm_squared,
        sector_margin=margin,
    )


def projected_blend_pointwise_sector(
    design: SectorProjectedDesign = LOCKED_SECTOR_PROJECTED_DESIGN,
    *,
    yosida_lower: Fraction = LOCKED_YOSIDA_STRONG_MONOTONICITY,
    yosida_upper: Fraction = LOCKED_YOSIDA_LIPSCHITZ,
) -> ProjectedPointwiseSector:
    """Return the uniform exact pointwise sector of the projected blend.

    For a fixed gate value ``theta``, convexity of origin-centred sector disks
    gives endpoints

    ``[(1-theta)*a, (1-theta)*b+theta*K]``,

    where ``a=yosida_lower/c`` and ``b=yosida_upper/c``.  Taking the interval
    hull over ``0<=theta<=gate_cap`` gives the returned endpoints.
    """

    if not isinstance(design, SectorProjectedDesign):
        raise TypeError("design must be a SectorProjectedDesign")
    _require_fraction(yosida_lower, "yosida_lower")
    _require_fraction(yosida_upper, "yosida_upper")
    if not 0 <= yosida_lower <= yosida_upper:
        raise ValueError("Yosida gains must satisfy 0 <= lower <= upper")

    passive_lower = yosida_lower / design.passive_divisor
    passive_upper = yosida_upper / design.passive_divisor
    lower = (1 - design.gate_cap) * passive_lower
    capped_upper = (1 - design.gate_cap) * passive_upper + design.gate_cap * design.projection_gain
    upper = max(passive_upper, capped_upper)
    return ProjectedPointwiseSector(
        lower=lower,
        upper=upper,
        center=(lower + upper) / 2,
        radius=(upper - lower) / 2,
    )


LOCKED_POINTWISE_SECTOR: Final = projected_blend_pointwise_sector()


def blend_sector_projected_exact(
    signal: Sequence[Fraction],
    yosida_output: Sequence[Fraction],
    shape_output: Sequence[Fraction],
    gate_weight: Fraction,
    design: SectorProjectedDesign = LOCKED_SECTOR_PROJECTED_DESIGN,
    *,
    yosida_lower: Fraction = LOCKED_YOSIDA_STRONG_MONOTONICITY,
    yosida_upper: Fraction = LOCKED_YOSIDA_LIPSCHITZ,
) -> ExactProjectedBlend:
    """Apply the projected blend and verify its global pointwise sector.

    ``yosida_output`` is the unscaled ``Y(S)``.  The helper checks its locked
    ``[yosida_lower,yosida_upper]`` pointwise supply before forming ``Y/c``.
    """

    selected_signal = _require_fraction_vector(signal, "signal")
    selected_yosida = _require_fraction_vector(yosida_output, "yosida_output")
    selected_shape = _require_fraction_vector(shape_output, "shape_output")
    _require_fraction(gate_weight, "gate_weight")
    if not isinstance(design, SectorProjectedDesign):
        raise TypeError("design must be a SectorProjectedDesign")
    if not (len(selected_signal) == len(selected_yosida) == len(selected_shape)):
        raise ValueError("signal and branch outputs must have equal lengths")
    if not 0 <= gate_weight <= design.gate_cap:
        raise ValueError("gate_weight must lie between zero and the design cap")
    if (
        sector_disk_margin_exact(
            selected_yosida,
            selected_signal,
            yosida_lower,
            yosida_upper,
        )
        < 0
    ):
        raise ValueError("yosida_output does not satisfy the supplied pointwise sector")

    projection = project_onto_origin_sector_exact(
        selected_shape,
        selected_signal,
        design.projection_gain,
    )
    passive_weight = 1 - gate_weight
    output = tuple(
        passive_weight * value / design.passive_divisor + gate_weight * projected
        for value, projected in zip(
            selected_yosida,
            projection.output,
            strict=True,
        )
    )
    sector = projected_blend_pointwise_sector(
        design,
        yosida_lower=yosida_lower,
        yosida_upper=yosida_upper,
    )
    margin = sector_disk_margin_exact(output, selected_signal, sector.lower, sector.upper)
    if margin < 0:
        raise ArithmeticError("exact projected blend escaped its global pointwise sector")
    return ExactProjectedBlend(
        output=output,
        projection=projection,
        gate_weight=gate_weight,
        global_sector=sector,
        sector_margin=margin,
    )


def _require_fp64_pair(shape_output: FloatArray, signal: FloatArray) -> None:
    for value, name in ((shape_output, "shape_output"), (signal, "signal")):
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{name} must be a numpy.ndarray")
        if value.dtype != np.float64:
            raise TypeError(f"{name} must have dtype numpy.float64")
        if value.size == 0:
            raise ValueError(f"{name} must be nonempty")
        if not bool(np.isfinite(value).all()):
            raise ValueError(f"{name} must contain only finite entries")
    if shape_output.shape != signal.shape:
        raise ValueError("shape_output and signal must have equal shapes")


def project_onto_origin_sector_fp64(
    shape_output: FloatArray,
    signal: FloatArray,
    projection_gain: float = 1.0,
) -> tuple[float, FloatArray]:
    """Return the diagnostic FP64 scale and projected shape output."""

    _require_fp64_pair(shape_output, signal)
    if not math.isfinite(projection_gain) or projection_gain < 0.0:
        raise ValueError("projection_gain must be finite and nonnegative")
    inner = float(np.vdot(shape_output, signal).real)
    shape_norm_squared = float(np.vdot(shape_output, shape_output).real)
    if not math.isfinite(inner) or not math.isfinite(shape_norm_squared):
        raise FloatingPointError("nonfinite FP64 projection summary")
    # The exact P18 branch has a nonnegative inner product analytically.  A
    # tiny negative value can nevertheless appear after FP64 reductions.  Its
    # positive part is the fail-safe projection scale: it returns the zero
    # shape branch, which remains in [0,K], instead of crashing the optimizer.
    # This diagnostic helper makes no fidelity claim for an arbitrary
    # genuinely anti-aligned branch.
    inner = max(0.0, inner)
    if shape_norm_squared == 0.0:
        return 0.0, np.zeros_like(shape_output)
    scale = min(1.0, projection_gain * inner / shape_norm_squared)
    if not math.isfinite(scale) or scale < 0.0:
        raise FloatingPointError("invalid FP64 sector-projection scale")
    return scale, np.ascontiguousarray(scale * shape_output, dtype=np.float64)


def blend_sector_projected_fp64(
    signal: FloatArray,
    yosida_output: FloatArray,
    shape_output: FloatArray,
    gate_weight: float,
    *,
    projection_gain: float = 1.0,
    passive_divisor: float = 1_024.0,
) -> tuple[float, FloatArray, FloatArray]:
    """Return ``(alpha, Z, T)`` for precomputed FP64 branch outputs."""

    _require_fp64_pair(shape_output, signal)
    _require_fp64_pair(yosida_output, signal)
    if not math.isfinite(gate_weight) or not 0.0 <= gate_weight <= 1.0:
        raise ValueError("gate_weight must lie in [0,1]")
    if not math.isfinite(passive_divisor) or passive_divisor <= 0.0:
        raise ValueError("passive_divisor must be finite and positive")
    scale, projected = project_onto_origin_sector_fp64(
        shape_output,
        signal,
        projection_gain,
    )
    output = (1.0 - gate_weight) * yosida_output / passive_divisor + gate_weight * projected
    if not bool(np.isfinite(output).all()):
        raise FloatingPointError("nonfinite FP64 projected-blend output")
    return scale, projected, np.ascontiguousarray(output, dtype=np.float64)


def _projection_diagnostics_fp64(
    shape_output: FloatArray,
    signal: FloatArray,
    projection_alpha: float,
) -> tuple[float, float, bool]:
    inner = float(np.vdot(shape_output, signal).real)
    squared_norm = float(np.vdot(shape_output, shape_output).real)
    if not math.isfinite(inner) or not math.isfinite(squared_norm):
        raise FloatingPointError("nonfinite FP64 projection diagnostics")
    active = squared_norm > 0.0 and projection_alpha < 1.0
    return inner, squared_norm, active


def evaluate_sector_projected_singular_values_fp64(
    signal_values: FloatArray,
    *,
    solver_config: EquivariantResolventSolverConfig | None = None,
) -> SectorProjectedSingularValueEvaluation:
    """Evaluate the locked P18 interface in P16 singular coordinates.

    The P17 wrapper supplies ``X=E_h,epsilon(J(S))``, the raw Yosida output,
    the identical ``3/4`` smootherstep gate, and a P16 solution that has
    passed the actual computed P15 residual postcondition.  P18 changes only
    the shape branch by applying the origin-sector projection and changes the
    passive divisor to ``1024``.
    """

    base = evaluate_gated_singular_values_fp64(
        signal_values,
        PRIMARY_DESIGN,
        solver_config=solver_config,
    )
    if not base.solver_diagnostics.certified:
        raise RuntimeError("P16 singular solve did not preserve the P15 postcondition")
    alpha, projected, output = blend_sector_projected_fp64(
        signal_values,
        base.yosida_output,
        base.shape_output,
        base.gate_value,
        projection_gain=float(LOCKED_PROJECTION_GAIN),
        passive_divisor=float(LOCKED_PASSIVE_DIVISOR),
    )
    inner, squared_norm, active = _projection_diagnostics_fp64(
        base.shape_output,
        signal_values,
        alpha,
    )
    return SectorProjectedSingularValueEvaluation(
        solution_singular_values=base.solution_singular_values,
        yosida_output=base.yosida_output,
        shape_output=base.shape_output,
        projection_alpha=alpha,
        shape_signal_inner_product=inner,
        shape_squared_norm=squared_norm,
        projection_active=active,
        projected_shape_output=projected,
        output=output,
        gate_value=base.gate_value,
        squared_signal_norm=base.squared_signal_norm,
        solver_diagnostics=base.solver_diagnostics,
    )


def evaluate_sector_projected_resolvent_fp64(
    signal: FloatArray,
    *,
    solver_config: EquivariantResolventSolverConfig | None = None,
) -> SectorProjectedMatrixEvaluation:
    """Evaluate the locked P18 full-matrix interface with P16 postchecking.

    This is a guarded FP64 reference path, not a floating-point error theorem.
    The exact sector theorem concerns the mathematical resolvent and exact
    branch outputs.
    """

    base = evaluate_shape_preserving_resolvent_fp64(
        signal,
        PRIMARY_DESIGN,
        solver_config=solver_config,
    )
    solved = base.resolvent_result
    if not solved.diagnostics.certified or solved.output is None:
        raise RuntimeError("P16 matrix solve did not preserve the P15 postcondition")
    alpha, projected, output = blend_sector_projected_fp64(
        signal,
        solved.output,
        base.jordan_output,
        base.gate_weight,
        projection_gain=float(LOCKED_PROJECTION_GAIN),
        passive_divisor=float(LOCKED_PASSIVE_DIVISOR),
    )
    inner, squared_norm, active = _projection_diagnostics_fp64(
        base.jordan_output,
        signal,
        alpha,
    )
    return SectorProjectedMatrixEvaluation(
        yosida_output=np.ascontiguousarray(solved.output, dtype=np.float64),
        shape_output=base.jordan_output,
        projection_alpha=alpha,
        shape_signal_inner_product=inner,
        shape_squared_norm=squared_norm,
        projection_active=active,
        projected_shape_output=projected,
        output=output,
        gate_value=base.gate_weight,
        squared_signal_norm=base.squared_signal_norm,
        resolvent_result=solved,
    )


__all__ = [
    "LOCKED_GATE_CAP",
    "LOCKED_PASSIVE_DIVISOR",
    "LOCKED_POINTWISE_SECTOR",
    "LOCKED_PROJECTION_GAIN",
    "LOCKED_SECTOR_PROJECTED_DESIGN",
    "ExactProjectedBlend",
    "ExactSectorProjection",
    "ProjectedPointwiseSector",
    "SectorProjectedDesign",
    "SectorProjectedMatrixEvaluation",
    "SectorProjectedSingularValueEvaluation",
    "blend_sector_projected_exact",
    "blend_sector_projected_fp64",
    "evaluate_sector_projected_resolvent_fp64",
    "evaluate_sector_projected_singular_values_fp64",
    "project_onto_origin_sector_exact",
    "project_onto_origin_sector_fp64",
    "projected_blend_pointwise_sector",
    "sector_disk_margin_exact",
    "sector_projection_scale_exact",
]
