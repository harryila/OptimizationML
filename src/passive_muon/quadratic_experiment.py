"""Controlled diagonal matrix-quadratic learning-rate sweeps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from passive_muon.specs import PolynomialCoefficients
from passive_muon.spectral_analysis import scalar_response_and_derivative

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class QuadraticEnsemble:
    """Two-mode SPD matrix quadratics represented in a shared SVD basis."""

    curvatures: FloatArray
    initial_coordinates: FloatArray
    condition_numbers: FloatArray

    @property
    def size(self) -> int:
        return int(self.curvatures.shape[0])


def build_quadratic_ensemble(
    *,
    condition_numbers: tuple[float, ...],
    seeds_per_condition: int,
    seed: int,
) -> QuadraticEnsemble:
    """Create deterministic 2x2 diagonal matrix quadratics with unit initial gradient."""

    if any(condition < 1 for condition in condition_numbers):
        raise ValueError("condition numbers must be at least one")
    if seeds_per_condition <= 0:
        raise ValueError("seeds_per_condition must be positive")
    generator = np.random.default_rng(seed)
    curvatures: list[tuple[float, float]] = []
    coordinates: list[tuple[float, float]] = []
    conditions: list[float] = []
    for condition in condition_numbers:
        for _ in range(seeds_per_condition):
            eigenvalues = np.array([1 / condition, 1.0], dtype=np.float64)
            if bool(generator.integers(0, 2)):
                eigenvalues = eigenvalues[::-1].copy()
            angle = generator.uniform(0.08, np.pi / 2 - 0.08)
            signs = generator.choice(np.array([-1.0, 1.0]), size=2)
            initial_gradient = signs * np.array([np.cos(angle), np.sin(angle)])
            initial_point = initial_gradient / eigenvalues
            curvatures.append(tuple(eigenvalues))
            coordinates.append(tuple(initial_point))
            conditions.append(condition)
    return QuadraticEnsemble(
        curvatures=np.asarray(curvatures, dtype=np.float64),
        initial_coordinates=np.asarray(coordinates, dtype=np.float64),
        condition_numbers=np.asarray(conditions, dtype=np.float64),
    )


@dataclass(frozen=True)
class SweepOutcome:
    final_ratios: FloatArray
    maximum_ratios: FloatArray
    tail_maximum_ratios: FloatArray
    first_target_iterations: NDArray[np.int64]
    success: NDArray[np.bool_]
    bounded: NDArray[np.bool_]
    minimum_positive_gradient_norm: float
    maximum_gradient_norm: float


def run_quadratic_sweep(
    ensemble: QuadraticEnsemble,
    learning_rates: FloatArray,
    coefficients: PolynomialCoefficients,
    *,
    polynomial_steps: int,
    iterations: int,
    normalization: str,
    normalizer_scale: float,
    repair_rho: float,
    success_ratio: float,
    divergence_ratio: float,
    output_scale: float = 1.0,
    hold_iterations: int = 1,
) -> SweepOutcome:
    """Run all problems and learning rates in one vectorized simulation."""

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if normalizer_scale <= 0:
        raise ValueError("normalizer_scale must be positive")
    if repair_rho < 0:
        raise ValueError("repair_rho must be nonnegative")
    if output_scale <= 0:
        raise ValueError("output_scale must be positive")
    if hold_iterations <= 0 or hold_iterations > iterations:
        raise ValueError("hold_iterations must lie between one and iterations")
    if normalization not in {"current_plus_eps", "fixed_scale"}:
        raise ValueError(f"unknown normalization {normalization!r}")

    rates = np.asarray(learning_rates, dtype=np.float64)
    if rates.ndim != 1 or np.any(rates <= 0):
        raise ValueError("learning rates must be a positive one-dimensional array")
    curvatures = ensemble.curvatures[:, None, :]
    coordinates = np.broadcast_to(
        ensemble.initial_coordinates[:, None, :],
        (ensemble.size, rates.size, 2),
    ).copy()
    initial_objective = 0.5 * np.sum(curvatures * coordinates**2, axis=-1)
    maximum_ratio = np.ones_like(initial_objective)
    tail_maximum_ratio = np.zeros_like(initial_objective)
    first_target_iteration = np.full(initial_objective.shape, -1, dtype=np.int64)
    minimum_positive_gradient_norm = np.inf
    maximum_gradient_norm = 0.0

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for iteration in range(iterations):
            gradient = curvatures * coordinates
            gradient_norms = np.linalg.norm(gradient, axis=-1)
            finite_positive_norms = gradient_norms[
                np.isfinite(gradient_norms) & (gradient_norms > 0)
            ]
            if finite_positive_norms.size:
                minimum_positive_gradient_norm = min(
                    minimum_positive_gradient_norm,
                    float(np.min(finite_positive_norms)),
                )
                maximum_gradient_norm = max(
                    maximum_gradient_norm,
                    float(np.max(finite_positive_norms)),
                )
            if normalization == "current_plus_eps":
                norms = gradient_norms[..., None]
                polynomial_input = gradient / (norms + normalizer_scale)
            else:
                polynomial_input = gradient / normalizer_scale
            update, _ = scalar_response_and_derivative(
                polynomial_input, coefficients, steps=polynomial_steps
            )
            update = output_scale * (update + repair_rho * gradient)
            coordinates = coordinates - rates[None, :, None] * update
            objective = 0.5 * np.sum(curvatures * coordinates**2, axis=-1)
            ratio = objective / initial_objective
            maximum_ratio = np.maximum(maximum_ratio, ratio)
            newly_reached = (
                (first_target_iteration < 0) & np.isfinite(ratio) & (ratio <= success_ratio)
            )
            first_target_iteration[newly_reached] = iteration + 1
            if iteration >= iterations - hold_iterations:
                tail_maximum_ratio = np.maximum(tail_maximum_ratio, ratio)
            invalid = ~np.isfinite(ratio) | (ratio > divergence_ratio * 10)
            coordinates[invalid] = np.nan

    gradient = curvatures * coordinates
    final_objective = 0.5 * np.sum(coordinates * gradient, axis=-1)
    final_ratio = final_objective / initial_objective
    finite = np.isfinite(final_ratio) & np.isfinite(maximum_ratio)
    bounded = finite & (maximum_ratio <= divergence_ratio)
    success = bounded & (tail_maximum_ratio <= success_ratio)
    final_ratio = np.where(finite, final_ratio, np.inf)
    maximum_ratio = np.where(finite, maximum_ratio, np.inf)
    tail_maximum_ratio = np.where(finite, tail_maximum_ratio, np.inf)
    return SweepOutcome(
        final_ratio,
        maximum_ratio,
        tail_maximum_ratio,
        first_target_iteration,
        success,
        bounded,
        float(minimum_positive_gradient_norm),
        maximum_gradient_norm,
    )


@dataclass(frozen=True)
class IntervalSummary:
    minimum: float | None
    maximum: float | None
    log10_width: float
    point_count: int


@dataclass(frozen=True)
class GridBandSummary:
    """Finite-grid target-band endpoints and adjacent failure brackets."""

    component_count: int
    lower_pass: float | None
    upper_pass: float | None
    lower_fail_below: float | None
    upper_fail_above: float | None
    left_censored: bool
    right_censored: bool
    log10_width: float
    point_count: int


@dataclass(frozen=True)
class PrefixBoundarySummary:
    """Upper bracket of the true prefix beginning at the smallest grid value."""

    upper_pass: float | None
    upper_fail: float | None
    right_censored: bool
    point_count: int


def widest_true_interval(mask: NDArray[np.bool_], values: FloatArray) -> IntervalSummary:
    """Return the widest contiguous true interval on an ordered positive grid."""

    if mask.ndim != 1 or values.ndim != 1 or mask.size != values.size:
        raise ValueError("mask and values must be one-dimensional and aligned")
    best_start = best_stop = None
    start = None
    for index, active in enumerate(np.append(mask, False)):
        if active and start is None:
            start = index
        elif not active and start is not None:
            stop = index - 1
            if best_start is None or np.log(values[stop] / values[start]) > np.log(
                values[best_stop] / values[best_start]
            ):
                best_start, best_stop = start, stop
            start = None
    if best_start is None or best_stop is None:
        return IntervalSummary(None, None, 0.0, 0)
    minimum = float(values[best_start])
    maximum = float(values[best_stop])
    return IntervalSummary(
        minimum=minimum,
        maximum=maximum,
        log10_width=float(np.log10(maximum / minimum)),
        point_count=best_stop - best_start + 1,
    )


def summarize_grid_band(mask: NDArray[np.bool_], values: FloatArray) -> GridBandSummary:
    """Summarize the widest true component without bridging disjoint islands."""

    interval = widest_true_interval(mask, values)
    padded = np.concatenate(([False], mask, [False]))
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    if interval.minimum is None or interval.maximum is None:
        return GridBandSummary(0, None, None, None, None, False, False, 0.0, 0)
    start = int(np.flatnonzero(values == interval.minimum)[0])
    stop = int(np.flatnonzero(values == interval.maximum)[0])
    return GridBandSummary(
        component_count=int(starts.size),
        lower_pass=interval.minimum,
        upper_pass=interval.maximum,
        lower_fail_below=float(values[start - 1]) if start > 0 else None,
        upper_fail_above=float(values[stop + 1]) if stop + 1 < values.size else None,
        left_censored=start == 0,
        right_censored=stop + 1 == values.size,
        log10_width=interval.log10_width,
        point_count=interval.point_count,
    )


def summarize_true_prefix(mask: NDArray[np.bool_], values: FloatArray) -> PrefixBoundarySummary:
    """Summarize the accepted prefix that starts at the smallest grid value."""

    if mask.ndim != 1 or values.ndim != 1 or mask.size != values.size:
        raise ValueError("mask and values must be one-dimensional and aligned")
    if not bool(mask[0]):
        return PrefixBoundarySummary(None, float(values[0]), False, 0)
    failures = np.flatnonzero(~mask)
    if failures.size == 0:
        return PrefixBoundarySummary(float(values[-1]), None, True, values.size)
    failure = int(failures[0])
    return PrefixBoundarySummary(
        upper_pass=float(values[failure - 1]),
        upper_fail=float(values[failure]),
        right_censored=False,
        point_count=failure,
    )
