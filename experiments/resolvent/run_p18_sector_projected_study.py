#!/usr/bin/env python3
"""Deterministic P18 sector-projected useful-rate diagnostics.

The exact P18 theorem is the dimension-independent pointwise sector proved in
``passive_muon.sector_projected_resolvent``.  This script supplies numerical
falsification, amplitude, and Muon-fidelity evidence for the locked
``K=1``, ``c=1024``, ``theta<=3/4`` interface at ``eta=1/83``.

Every declared evaluation calls the guarded P16 solver and fails closed unless
the *computed* graph residual satisfies the P15 stopping rule.  The broad grid,
the operating annulus, and the Transformer spectra are sampled FP64 evidence;
their extrema are not global theorems or IEEE-754 rounding certificates.
Transformer cases use complete ``min(m,n)`` singular-value vectors without
allocating dense matrices or claiming accelerator timings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from passive_muon.equivariant_resolvent_solver import (
    EquivariantResolventSolverConfig,
    SolverDiagnostics,
    jordan_response_and_derivative_fp64,
    resolvent_graph_singular_values_fp64,
)
from passive_muon.radial_passivation_tradeoff import P11_SIGNAL_NORM_BOUND
from passive_muon.sector_projected_resolvent import (
    LOCKED_GATE_CAP,
    LOCKED_PASSIVE_DIVISOR,
    LOCKED_POINTWISE_SECTOR,
    LOCKED_PROJECTION_GAIN,
    SectorProjectedDesign,
    blend_sector_projected_fp64,
    evaluate_sector_projected_resolvent_fp64,
    evaluate_sector_projected_singular_values_fp64,
    projected_blend_pointwise_sector,
)
from passive_muon.shape_preserving_resolvent import (
    FULL_STEP_DESIGN,
    LOCKED_JORDAN_AFTER_RESOLVENT_GAIN_UPPER,
    PRIMARY_DESIGN,
    quintic_gate_weight_fp64,
)
from passive_muon.upstream_momentum import (
    PINNED_KELLER_JORDAN_MUON_REVISION,
    PINNED_MUON_PY_SHA256,
)

FloatArray = NDArray[np.float64]

SCHEMA_VERSION: Final = "passive-muon-p18-sector-projected-study-v1"
SEED: Final = 20_260_905
STAGES: Final = 5
P11_SIGNAL_GUARD: Final = float(P11_SIGNAL_NORM_BOUND)

SELECTED_LEARNING_RATE: Final = Fraction(1, 83)
FASTER_RATE_FRONTIER_LEARNING_RATE: Final = Fraction(1, 120)
UNDERSIZED_PROJECTION_GAIN: Final = Fraction(1, 100)

# The scale-sensitive gates complement P16's scale-invariant fidelity gates.
GLOBAL_INPUT_AMPLITUDE_LOWER: Final = Fraction(1, 10)
GLOBAL_INPUT_AMPLITUDE_UPPER: Final = Fraction(1)
ANNULUS_UPSTREAM_AMPLITUDE_LOWER: Final = Fraction(1, 4)
ANNULUS_UPSTREAM_AMPLITUDE_UPPER: Final = Fraction(8)
EFFECTIVE_UPDATE_LOWER: Final = Fraction(1, 1_000)
EFFECTIVE_UPDATE_UPPER: Final = Fraction(1, 80)

# Frozen without retuning from the P16/P17 studies.
MEANINGFUL_SCALAR_RESIDUAL_THRESHOLD: Final = 1.0e-3
MEANINGFUL_SHAPING_RETENTION_THRESHOLD: Final = 1.0e-1
INFORMATIVE_UPSTREAM_SHAPING_THRESHOLD: Final = 1.0e-2

OPERATING_ANNULUS_INNER_RADIUS: Final = Fraction(3, 4)
OPERATING_ANNULUS_OUTER_RADIUS: Final = Fraction(25)
OPERATING_ANNULUS_STRICT_RELATIVE_TOLERANCE: Final = 2.0**-44
OPERATING_ANNULUS_STRICT_ABSOLUTE_TOLERANCE: Final = 2.0**-50
RANK_ACCUMULATION_MODE_RATIO: Final = 41.0 / 10_000.0
REALISTIC_TRANSFORMER_SHAPES: Final = (
    (768, 768),
    (768, 3_072),
    (3_072, 12_288),
    (4_096, 11_008),
)

VARIANT_LOCKED: Final = "locked_K_1"
VARIANT_UNDERSIZED: Final = "undersized_K_1_over_100"
VARIANT_UNPROJECTED: Final = "unprojected_raw_shape"
VARIANTS: Final = (VARIANT_LOCKED, VARIANT_UNDERSIZED, VARIANT_UNPROJECTED)

ROOT = Path(__file__).resolve().parents[2]
CANONICAL_TARGET = ROOT / "results" / "summaries" / "p18_sector_projected_study.json"
SOURCE_PATHS = (
    Path("experiments/resolvent/run_p18_sector_projected_study.py"),
    Path("src/passive_muon/additive_epsilon_deficit.py"),
    Path("src/passive_muon/equivariant_resolvent_solver.py"),
    Path("src/passive_muon/radial_passivation_tradeoff.py"),
    Path("src/passive_muon/sector_projected_resolvent.py"),
    Path("src/passive_muon/shape_preserving_resolvent.py"),
    Path("src/passive_muon/shape_preserving_resolvent_certificate.py"),
    Path("src/passive_muon/specs.py"),
    Path("src/passive_muon/upstream_momentum.py"),
    Path("src/passive_muon/yosida_stability.py"),
    Path("pyproject.toml"),
    Path("uv.lock"),
)


@dataclass(frozen=True)
class P18StudyConfig:
    """Complete deterministic configuration of the P18 numerical study."""

    seed: int = SEED
    spectrum_grid_points: int = 17
    operating_annulus_grid_points: int = 33
    include_realistic_spectrum_cases: bool = True
    solver: EquivariantResolventSolverConfig = field(
        default_factory=EquivariantResolventSolverConfig
    )

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for name in ("spectrum_grid_points", "operating_annulus_grid_points"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 5:
                raise ValueError(f"{name} must be an integer at least five")
        if not isinstance(self.include_realistic_spectrum_cases, bool):
            raise TypeError("include_realistic_spectrum_cases must be Boolean")
        if not isinstance(self.solver, EquivariantResolventSolverConfig):
            raise TypeError("solver must be an EquivariantResolventSolverConfig")


def _norm(values: FloatArray) -> float:
    return float(np.linalg.norm(np.asarray(values, dtype=np.float64).ravel()))


def _unit(values: FloatArray) -> FloatArray:
    norm = _norm(values)
    if norm == 0.0:
        raise ValueError("cannot normalize a zero template")
    return np.asarray(values / norm, dtype=np.float64)


def _fraction(value: Fraction) -> dict[str, object]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "fraction": str(value),
        "decimal": float(value),
    }


def _sha256_float64(values: FloatArray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes(order="C")).hexdigest()


def _best_scalar_record(signal: FloatArray, output: FloatArray) -> dict[str, float | None]:
    source = np.asarray(signal, dtype=np.float64).ravel()
    target = np.asarray(output, dtype=np.float64).ravel()
    source_square = float(np.dot(source, source))
    output_norm = _norm(target)
    if source_square == 0.0 or output_norm == 0.0:
        return {
            "coefficient": None,
            "residual_frobenius": 0.0 if output_norm == 0.0 else output_norm,
            "relative_residual": 0.0 if output_norm == 0.0 else 1.0,
        }
    coefficient = float(np.dot(source, target)) / source_square
    residual = _norm(target - coefficient * source)
    return {
        "coefficient": coefficient,
        "residual_frobenius": residual,
        "relative_residual": residual / output_norm,
    }


def _directional_sine(left: FloatArray, right: FloatArray) -> float | None:
    denominator = _norm(left) * _norm(right)
    if denominator == 0.0:
        return None
    cosine = (
        float(
            np.dot(
                np.asarray(left, dtype=np.float64).ravel(),
                np.asarray(right, dtype=np.float64).ravel(),
            )
        )
        / denominator
    )
    cosine = max(-1.0, min(1.0, cosine))
    return math.sqrt(max(0.0, 1.0 - cosine * cosine))


def _modal_gain_summary(signal: FloatArray, output: FloatArray) -> dict[str, object]:
    source = np.asarray(signal, dtype=np.float64)
    gains = np.divide(
        output,
        source,
        out=np.full_like(output, np.nan),
        where=source > 0.0,
    )
    finite = gains[np.isfinite(gains)]
    if finite.size == 0:
        return {"positive_mode_count": 0, "values": []}
    record: dict[str, object] = {
        "positive_mode_count": int(finite.size),
        "minimum": float(np.min(finite)),
        "median": float(np.median(finite)),
        "maximum": float(np.max(finite)),
        "spread": float(np.max(finite) - np.min(finite)),
    }
    if finite.size <= 8:
        record["values"] = [float(value) for value in finite]
    else:
        record["quantiles_05_50_95"] = [
            float(value) for value in np.quantile(finite, [0.05, 0.5, 0.95])
        ]
    return record


def _upstream_jordan_singular_values(
    signal_values: FloatArray,
    epsilon: float,
) -> FloatArray:
    signal_norm = _norm(signal_values)
    if signal_norm == 0.0:
        return np.zeros_like(signal_values)
    normalized = np.asarray(signal_values / (signal_norm + epsilon), dtype=np.float64)
    response, _derivative = jordan_response_and_derivative_fp64(normalized)
    return response


def _diagnostics_record(diagnostics: SolverDiagnostics) -> dict[str, object]:
    ratio = (
        diagnostics.residual_norm / diagnostics.p15_threshold
        if diagnostics.p15_threshold > 0.0
        else 0.0
    )
    return {
        "status": diagnostics.status.value,
        "certified": diagnostics.certified,
        "iterations": diagnostics.iterations,
        "backtracks": diagnostics.backtracks,
        "actual_residual_norm": diagnostics.residual_norm,
        "p15_threshold": diagnostics.p15_threshold,
        "residual_to_p15_threshold_ratio": ratio,
        "operator_evaluations": diagnostics.operator_evaluations,
        "jacobian_evaluations": diagnostics.jacobian_evaluations,
        "sherman_morrison_solves": diagnostics.sherman_morrison_solves,
        "svd_count": diagnostics.svd_count,
        "reconstruction_matmuls": diagnostics.reconstruction_matmuls,
    }


class _CallLedger:
    """Collect actual P15 residual evidence and fail closed."""

    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    def add(self, label: str, dimension: int, diagnostics: SolverDiagnostics) -> None:
        if not diagnostics.certified:
            raise RuntimeError(f"declared solver call {label!r} failed closed")
        if diagnostics.residual_norm > diagnostics.p15_threshold:
            raise RuntimeError(f"declared solver call {label!r} violated the P15 postcondition")
        self.records.append(
            {"label": label, "dimension": dimension, **_diagnostics_record(diagnostics)}
        )

    def summary(self) -> dict[str, object]:
        if not self.records:
            raise RuntimeError("study executed no solver calls")
        iterations = [int(record["iterations"]) for record in self.records]
        ratios = [float(record["residual_to_p15_threshold_ratio"]) for record in self.records]
        return {
            "call_count": len(self.records),
            "all_calls_computed_residual_certified": True,
            "status_counts": dict(
                sorted(Counter(str(record["status"]) for record in self.records).items())
            ),
            "iteration_histogram": {
                str(key): value for key, value in sorted(Counter(iterations).items())
            },
            "minimum_iterations": min(iterations),
            "maximum_iterations": max(iterations),
            "maximum_backtracks": max(int(record["backtracks"]) for record in self.records),
            "worst_residual_to_p15_threshold_ratio": max(ratios),
            "maximum_actual_residual": max(
                float(record["actual_residual_norm"]) for record in self.records
            ),
            "total_operator_evaluations": sum(
                int(record["operator_evaluations"]) for record in self.records
            ),
            "total_sherman_morrison_solves": sum(
                int(record["sherman_morrison_solves"]) for record in self.records
            ),
            "scalar_coordinate_iteration_work_proxy": sum(
                int(record["dimension"]) * int(record["iterations"]) for record in self.records
            ),
        }


def _reference_outputs(
    signal: FloatArray,
    yosida: FloatArray,
    shape: FloatArray,
    primary_gate: float,
) -> tuple[FloatArray, FloatArray]:
    primary = (1.0 - primary_gate) * yosida / float(
        PRIMARY_DESIGN.passive_divisor
    ) + primary_gate * shape
    full_gate = quintic_gate_weight_fp64(
        float(np.vdot(signal, signal).real),
        FULL_STEP_DESIGN,
    )
    full = (1.0 - full_gate) * yosida / float(FULL_STEP_DESIGN.passive_divisor) + full_gate * shape
    return (
        np.ascontiguousarray(primary, dtype=np.float64),
        np.ascontiguousarray(full, dtype=np.float64),
    )


def _variant_outputs(
    signal: FloatArray,
    yosida: FloatArray,
    shape: FloatArray,
    gate_weight: float,
    locked_output: FloatArray,
) -> tuple[dict[str, FloatArray], float]:
    small_alpha, _small_projected, small_output = blend_sector_projected_fp64(
        signal,
        yosida,
        shape,
        gate_weight,
        projection_gain=float(UNDERSIZED_PROJECTION_GAIN),
        passive_divisor=float(LOCKED_PASSIVE_DIVISOR),
    )
    unprojected = (1.0 - gate_weight) * yosida / float(LOCKED_PASSIVE_DIVISOR) + gate_weight * shape
    return (
        {
            VARIANT_LOCKED: np.ascontiguousarray(locked_output, dtype=np.float64),
            VARIANT_UNDERSIZED: small_output,
            VARIANT_UNPROJECTED: np.ascontiguousarray(unprojected, dtype=np.float64),
        },
        small_alpha,
    )


def _metric_record(
    signal: FloatArray,
    output: FloatArray,
    upstream: FloatArray,
    primary_reference: FloatArray,
    full_step_reference: FloatArray,
) -> dict[str, object]:
    signal_norm = _norm(signal)
    output_norm = _norm(output)
    upstream_norm = _norm(upstream)
    primary_norm = _norm(primary_reference)
    full_norm = _norm(full_step_reference)
    best = _best_scalar_record(signal, output)
    upstream_best = _best_scalar_record(signal, upstream)
    upstream_departure = float(upstream_best["relative_residual"])
    departure = float(best["relative_residual"])
    retention = departure / upstream_departure if upstream_departure > 0.0 else None
    input_amplitude = output_norm / signal_norm if signal_norm > 0.0 else 0.0
    upstream_amplitude = output_norm / upstream_norm if upstream_norm > 0.0 else None
    effective_gain = float(SELECTED_LEARNING_RATE) * input_amplitude
    informative = upstream_departure >= INFORMATIVE_UPSTREAM_SHAPING_THRESHOLD
    fidelity_passes = (
        informative
        and departure >= MEANINGFUL_SCALAR_RESIDUAL_THRESHOLD
        and retention is not None
        and retention >= MEANINGFUL_SHAPING_RETENTION_THRESHOLD
    )
    return {
        "input_frobenius": signal_norm,
        "output_frobenius": output_norm,
        "upstream_output_frobenius": upstream_norm,
        "output_to_input_amplitude": input_amplitude,
        "output_to_upstream_amplitude": upstream_amplitude,
        "selected_effective_update_per_input": effective_gain,
        "selected_effective_update_frobenius": float(SELECTED_LEARNING_RATE) * output_norm,
        "effective_update_ratio_to_p17_primary": (
            float(SELECTED_LEARNING_RATE)
            * output_norm
            / (float(PRIMARY_DESIGN.learning_rate) * primary_norm)
            if primary_norm > 0.0
            else None
        ),
        "effective_update_ratio_to_p17_full_step": (
            float(SELECTED_LEARNING_RATE)
            * output_norm
            / (float(FULL_STEP_DESIGN.learning_rate) * full_norm)
            if full_norm > 0.0
            else None
        ),
        "best_scalar_multiple": best,
        "best_scalar_departure": departure,
        "upstream_best_scalar_multiple": upstream_best,
        "shaping_retention_fraction": retention,
        "modal_gains": _modal_gain_summary(signal, output),
        "directional_sine_to_upstream": _directional_sine(output, upstream),
        "informative_upstream_shaping": informative,
        "meaningful_fidelity_gate_passes": fidelity_passes,
        "sampled_global_input_amplitude_gate_passes": (
            float(GLOBAL_INPUT_AMPLITUDE_LOWER)
            <= input_amplitude
            <= float(GLOBAL_INPUT_AMPLITUDE_UPPER)
        ),
        "annulus_upstream_amplitude_gate_passes": (
            upstream_amplitude is not None
            and float(ANNULUS_UPSTREAM_AMPLITUDE_LOWER)
            <= upstream_amplitude
            <= float(ANNULUS_UPSTREAM_AMPLITUDE_UPPER)
        ),
        "selected_effective_update_gate_passes": (
            float(EFFECTIVE_UPDATE_LOWER) <= effective_gain <= float(EFFECTIVE_UPDATE_UPPER)
        ),
    }


def _evaluate_singular_case(
    signal: FloatArray,
    *,
    label: str,
    solver: EquivariantResolventSolverConfig,
    ledger: _CallLedger,
) -> dict[str, object]:
    selected = np.ascontiguousarray(signal, dtype=np.float64)
    evaluated = evaluate_sector_projected_singular_values_fp64(
        selected,
        solver_config=solver,
    )
    ledger.add(label, selected.size, evaluated.solver_diagnostics)
    upstream = _upstream_jordan_singular_values(selected, solver.epsilon)
    primary, full = _reference_outputs(
        selected,
        evaluated.yosida_output,
        evaluated.shape_output,
        evaluated.gate_value,
    )
    outputs, small_alpha = _variant_outputs(
        selected,
        evaluated.yosida_output,
        evaluated.shape_output,
        evaluated.gate_value,
        evaluated.output,
    )
    return {
        "gate_weight": evaluated.gate_value,
        "projection_alpha": evaluated.projection_alpha,
        "projection_active": evaluated.projection_active,
        "undersized_projection_alpha": small_alpha,
        "diagnostics": _diagnostics_record(evaluated.solver_diagnostics),
        "variants": {
            name: _metric_record(selected, output, upstream, primary, full)
            for name, output in outputs.items()
        },
    }


def _canonical_case(
    solver: EquivariantResolventSolverConfig,
    ledger: _CallLedger,
) -> dict[str, object]:
    signal = np.diag(np.asarray([3.0, 4.0], dtype=np.float64))
    evaluated = evaluate_sector_projected_resolvent_fp64(signal, solver_config=solver)
    diagnostics = evaluated.resolvent_result.diagnostics
    ledger.add("canonical_diag_3_4", 2, diagnostics)
    upstream = np.diag(
        _upstream_jordan_singular_values(np.asarray([3.0, 4.0], dtype=np.float64), solver.epsilon)
    )
    primary, full = _reference_outputs(
        signal,
        evaluated.yosida_output,
        evaluated.shape_output,
        evaluated.gate_value,
    )
    outputs, small_alpha = _variant_outputs(
        signal,
        evaluated.yosida_output,
        evaluated.shape_output,
        evaluated.gate_value,
        evaluated.output,
    )
    return {
        "input": "diag(3,4)",
        "scope": "full 2x2 SVD/reconstruction FP64 path; not an interval root certificate",
        "input_sha256_float64_bytes": _sha256_float64(signal),
        "approximate_solution_singular_values": [
            float(value) for value in evaluated.resolvent_result.solution_singular_values
        ],
        "gate_weight": evaluated.gate_value,
        "projection_alpha": evaluated.projection_alpha,
        "projection_active": evaluated.projection_active,
        "undersized_projection_alpha": small_alpha,
        "diagnostics": _diagnostics_record(diagnostics),
        "variants": {
            name: _metric_record(signal, output, upstream, primary, full)
            for name, output in outputs.items()
        },
    }


def _declared_grid(points: int) -> tuple[FloatArray, FloatArray]:
    radii = np.unique(
        np.concatenate(
            (
                np.geomspace(1.0e-12, P11_SIGNAL_GUARD, points, dtype=np.float64),
                np.asarray([0.475, 1.0, 1.5, 1.67, 2.075, 3.0], dtype=np.float64),
            )
        )
    )
    ratios = np.unique(
        np.concatenate(
            (
                np.geomspace(1.0e-6, 1.0, points, dtype=np.float64),
                np.asarray([1_000.0 / 249_999.0, RANK_ACCUMULATION_MODE_RATIO]),
            )
        )
    )
    return radii, ratios


def _operating_annulus_grid(points: int) -> tuple[FloatArray, FloatArray]:
    lower = float(OPERATING_ANNULUS_INNER_RADIUS)
    upper = float(OPERATING_ANNULUS_OUTER_RADIUS)
    radii = np.unique(
        np.concatenate(
            (
                np.geomspace(lower, upper, points, dtype=np.float64),
                np.linspace(lower, upper, points, dtype=np.float64),
            )
        )
    )
    ratios = np.unique(
        np.concatenate(
            (
                np.geomspace(1.0e-6, 1.0, points, dtype=np.float64),
                np.asarray(
                    [
                        1_000.0 / 249_999.0,
                        RANK_ACCUMULATION_MODE_RATIO,
                        0.01,
                        0.03,
                        0.1,
                        0.3,
                        0.5,
                        0.7,
                        0.9,
                    ],
                    dtype=np.float64,
                ),
            )
        )
    )
    return radii, ratios


def _empty_state() -> dict[str, object]:
    return {
        "evaluated": 0,
        "informative": 0,
        "fidelity_passes": 0,
        "sampled_global_input_amplitude_passes": 0,
        "annulus_upstream_amplitude_passes": 0,
        "selected_effective_update_passes": 0,
        "metrics": [],
    }


def _observe(
    state: dict[str, object],
    metrics: dict[str, object],
    witness: dict[str, float],
) -> None:
    state["evaluated"] = int(state["evaluated"]) + 1
    if bool(metrics["informative_upstream_shaping"]):
        state["informative"] = int(state["informative"]) + 1
        if bool(metrics["meaningful_fidelity_gate_passes"]):
            state["fidelity_passes"] = int(state["fidelity_passes"]) + 1
    if bool(metrics["sampled_global_input_amplitude_gate_passes"]):
        state["sampled_global_input_amplitude_passes"] = (
            int(state["sampled_global_input_amplitude_passes"]) + 1
        )
    if bool(metrics["annulus_upstream_amplitude_gate_passes"]):
        state["annulus_upstream_amplitude_passes"] = (
            int(state["annulus_upstream_amplitude_passes"]) + 1
        )
    if bool(metrics["selected_effective_update_gate_passes"]):
        state["selected_effective_update_passes"] = (
            int(state["selected_effective_update_passes"]) + 1
        )
    observations = state["metrics"]
    assert isinstance(observations, list)
    observations.append({"witness": witness, "metrics": metrics})


def _extreme(
    observations: list[dict[str, object]],
    key: str,
    *,
    maximize: bool,
    informative_only: bool = False,
) -> dict[str, object] | None:
    eligible = [
        record
        for record in observations
        if record["metrics"][key] is not None  # type: ignore[index]
        and (
            not informative_only or bool(record["metrics"]["informative_upstream_shaping"])  # type: ignore[index]
        )
    ]
    if not eligible:
        return None
    selector = max if maximize else min
    return selector(eligible, key=lambda record: float(record["metrics"][key]))  # type: ignore[index]


def _summarize_state(state: dict[str, object]) -> dict[str, object]:
    observations = state.pop("metrics")
    assert isinstance(observations, list)
    evaluated = int(state["evaluated"])
    informative = int(state["informative"])
    return {
        **state,
        "all_informative_fidelity_pass": (
            int(state["fidelity_passes"]) == informative if informative else None
        ),
        "all_sampled_global_input_amplitudes_pass": (
            int(state["sampled_global_input_amplitude_passes"]) == evaluated
        ),
        "all_annulus_upstream_amplitudes_pass": (
            int(state["annulus_upstream_amplitude_passes"]) == evaluated
        ),
        "all_selected_effective_updates_pass": (
            int(state["selected_effective_update_passes"]) == evaluated
        ),
        "minimum_output_to_input_amplitude": _extreme(
            observations, "output_to_input_amplitude", maximize=False
        ),
        "maximum_output_to_input_amplitude": _extreme(
            observations, "output_to_input_amplitude", maximize=True
        ),
        "minimum_output_to_upstream_amplitude": _extreme(
            observations, "output_to_upstream_amplitude", maximize=False
        ),
        "maximum_output_to_upstream_amplitude": _extreme(
            observations, "output_to_upstream_amplitude", maximize=True
        ),
        "minimum_selected_effective_update_per_input": _extreme(
            observations, "selected_effective_update_per_input", maximize=False
        ),
        "maximum_selected_effective_update_per_input": _extreme(
            observations, "selected_effective_update_per_input", maximize=True
        ),
        "minimum_informative_best_scalar_departure": _extreme(
            observations,
            "best_scalar_departure",
            maximize=False,
            informative_only=True,
        ),
        "minimum_informative_shaping_retention": _extreme(
            observations,
            "shaping_retention_fraction",
            maximize=False,
            informative_only=True,
        ),
    }


def _run_grid(
    config: P18StudyConfig,
    ledger: _CallLedger,
    *,
    operating_annulus: bool,
) -> dict[str, object]:
    if operating_annulus:
        radii, ratios = _operating_annulus_grid(config.operating_annulus_grid_points)
        solver = replace(
            config.solver,
            strict_relative_tolerance=min(
                config.solver.strict_relative_tolerance,
                OPERATING_ANNULUS_STRICT_RELATIVE_TOLERANCE,
            ),
            strict_absolute_tolerance=min(
                config.solver.strict_absolute_tolerance,
                OPERATING_ANNULUS_STRICT_ABSOLUTE_TOLERANCE,
            ),
        )
    else:
        radii, ratios = _declared_grid(config.spectrum_grid_points)
        solver = config.solver

    states = {name: _empty_state() for name in VARIANTS}
    groups = {name: _empty_state() for name in ("gate_off", "transition", "plateau")}
    skipped_over_guard = 0
    for radius in radii:
        for ratio in ratios:
            if operating_annulus:
                signal = float(radius) * _unit(np.asarray([ratio, 1.0], dtype=np.float64))
            else:
                solution = float(radius) * _unit(np.asarray([ratio, 1.0], dtype=np.float64))
                signal = resolvent_graph_singular_values_fp64(solution, config=solver)
                if _norm(signal) > solver.signal_norm_guard:
                    skipped_over_guard += 1
                    continue
            record = _evaluate_singular_case(
                signal,
                label=(
                    f"annulus:r={radius:.17g}:a={ratio:.17g}"
                    if operating_annulus
                    else f"broad:r={radius:.17g}:a={ratio:.17g}"
                ),
                solver=solver,
                ledger=ledger,
            )
            witness = {
                "input_frobenius": _norm(signal),
                "grid_radius": float(radius),
                "mode_ratio": float(ratio),
                "gate_weight": float(record["gate_weight"]),
                "projection_alpha": float(record["projection_alpha"]),
            }
            variants = record["variants"]
            assert isinstance(variants, dict)
            for name in VARIANTS:
                _observe(states[name], variants[name], witness)
            gate = float(record["gate_weight"])
            if gate == 0.0:
                group = "gate_off"
            elif gate == float(LOCKED_GATE_CAP):
                group = "plateau"
            else:
                group = "transition"
            _observe(groups[group], variants[VARIANT_LOCKED], witness)

    summaries = {name: _summarize_state(state) for name, state in states.items()}
    group_summaries = {name: _summarize_state(state) for name, state in groups.items()}
    result: dict[str, object] = {
        "status": "sampled FP64 diagnostic; extrema are not global certificates",
        "radius_count": int(radii.size),
        "mode_ratio_count": int(ratios.size),
        "candidate_count": int(radii.size * ratios.size),
        "skipped_over_p11_guard": skipped_over_guard,
        "variants": summaries,
        "locked_variant_by_gate_region": group_summaries,
    }
    if operating_annulus:
        locked = summaries[VARIANT_LOCKED]
        small = summaries[VARIANT_UNDERSIZED]
        unprojected = summaries[VARIANT_UNPROJECTED]
        acceptance = {
            "all_locked_informative_points_pass_frozen_p16_fidelity": locked[
                "all_informative_fidelity_pass"
            ],
            "all_locked_points_pass_upstream_amplitude_window": locked[
                "all_annulus_upstream_amplitudes_pass"
            ],
            "all_locked_points_pass_effective_update_window": locked[
                "all_selected_effective_updates_pass"
            ],
            "undersized_K_has_a_fidelity_failure": (
                int(small["fidelity_passes"]) < int(small["informative"])
            ),
            "undersized_K_has_an_upstream_amplitude_failure": (
                int(small["annulus_upstream_amplitude_passes"]) < int(small["evaluated"])
            ),
            "unprojected_has_an_effective_update_failure": (
                int(unprojected["selected_effective_update_passes"]) < int(unprojected["evaluated"])
            ),
        }
        required_acceptance = (
            "all_locked_informative_points_pass_frozen_p16_fidelity",
            "all_locked_points_pass_upstream_amplitude_window",
            "all_locked_points_pass_effective_update_window",
            "undersized_K_has_a_fidelity_failure",
            "unprojected_has_an_effective_update_failure",
        )
        failed = [name for name in required_acceptance if not bool(acceptance[name])]
        if failed:
            raise RuntimeError(f"P18 operating-annulus acceptance failed: {failed}")
        result.update(
            {
                "scope": (
                    "post-P17 sampled operating annulus; all-point statements below concern "
                    "only this declared grid, not all matrices in the annulus"
                ),
                "input_radius_interval": {
                    "lower": _fraction(OPERATING_ANNULUS_INNER_RADIUS),
                    "upper": _fraction(OPERATING_ANNULUS_OUTER_RADIUS),
                },
                "solver_policy": {
                    "strict_relative_tolerance": solver.strict_relative_tolerance,
                    "strict_absolute_tolerance": solver.strict_absolute_tolerance,
                    "p15_relative_tolerance": solver.p15_relative_tolerance,
                    "p15_absolute_tolerance": solver.p15_absolute_tolerance,
                },
                "acceptance": acceptance,
                "required_acceptance_checks": list(required_acceptance),
                "all_required_acceptance_checks_pass": True,
            }
        )
    else:
        result["parameterization"] = (
            "u=r*(a,1)/sqrt(1+a^2), then s=u+lambda*B(u); each retained s is "
            "independently resolved by the fail-closed P16 solver"
        )
        result["origin_fidelity_disclosure"] = (
            "the passive gate-off region intentionally need not retain upstream Jordan "
            "shaping; broad-grid fidelity failures are expected and fully counted"
        )
    return result


def _realistic_spectrum(shape: tuple[int, int], index: int) -> tuple[str, FloatArray]:
    rank = min(shape)
    if index == 0:
        return "unit_flat_repeated", np.full(rank, 1.0 / math.sqrt(rank), dtype=np.float64)
    if index == 1:
        values = np.geomspace(1.0, 1.0e-4, rank, dtype=np.float64)
        return "half_guard_log_condition_1e4", 0.5 * P11_SIGNAL_GUARD * _unit(values)
    if index == 2:
        values = np.concatenate(
            (
                np.ones(rank // 2, dtype=np.float64),
                np.full(rank - rank // 2, 0.01, dtype=np.float64),
            )
        )
        return "guard_two_repeated_blocks", 0.999 * P11_SIGNAL_GUARD * _unit(values)
    values = np.geomspace(1.0, 1.0e-6, rank, dtype=np.float64)
    return "guard_log_condition_1e6", 0.999 * P11_SIGNAL_GUARD * _unit(values)


def _run_realistic_spectra(
    config: P18StudyConfig,
    ledger: _CallLedger,
) -> list[dict[str, object]]:
    if not config.include_realistic_spectrum_cases:
        return []
    records: list[dict[str, object]] = []
    for index, shape in enumerate(REALISTIC_TRANSFORMER_SHAPES):
        label, signal = _realistic_spectrum(shape, index)
        evaluated = _evaluate_singular_case(
            signal,
            label=f"transformer:{shape[0]}x{shape[1]}:{label}",
            solver=config.solver,
            ledger=ledger,
        )
        records.append(
            {
                "label": label,
                "declared_matrix_shape": list(shape),
                "singular_value_count": int(signal.size),
                "representation": (
                    "complete min(m,n) singular spectrum; no dense matrix, SVD, "
                    "accelerator timing, or memory-traffic benchmark"
                ),
                "input_frobenius": _norm(signal),
                "input_sha256_float64_bytes": _sha256_float64(signal),
                **evaluated,
            }
        )
    return records


def _exact_gate_records() -> dict[str, object]:
    sector = LOCKED_POINTWISE_SECTOR
    expected = {
        "lower": Fraction(125, 1_024),
        "upper": Fraction(509, 512),
        "center": Fraction(1_143, 2_048),
        "radius": Fraction(893, 2_048),
    }
    if any(getattr(sector, name) != value for name, value in expected.items()):
        raise RuntimeError("locked P18 pointwise sector differs from the frozen study design")
    effective_lower = SELECTED_LEARNING_RATE * sector.lower
    effective_upper = SELECTED_LEARNING_RATE * sector.upper
    candidate_passes = (
        sector.lower >= GLOBAL_INPUT_AMPLITUDE_LOWER
        and sector.upper <= GLOBAL_INPUT_AMPLITUDE_UPPER
        and effective_lower >= EFFECTIVE_UPDATE_LOWER
        and effective_upper <= EFFECTIVE_UPDATE_UPPER
    )
    if not candidate_passes:
        raise RuntimeError("locked exact P18 amplitude/effective-update gate failed")

    small_design = SectorProjectedDesign(
        projection_gain=UNDERSIZED_PROJECTION_GAIN,
        passive_divisor=LOCKED_PASSIVE_DIVISOR,
        gate_cap=LOCKED_GATE_CAP,
    )
    small_sector = projected_blend_pointwise_sector(small_design)
    passive_upper = Fraction(1_000) / LOCKED_PASSIVE_DIVISOR
    unprojected_upper = max(
        passive_upper,
        (1 - LOCKED_GATE_CAP) * passive_upper
        + LOCKED_GATE_CAP * LOCKED_JORDAN_AFTER_RESOLVENT_GAIN_UPPER,
    )
    unprojected_lower = (1 - LOCKED_GATE_CAP) * Fraction(500) / LOCKED_PASSIVE_DIVISOR
    return {
        "claim": (
            "dimension-uniform exact-real origin-centred pointwise sector for every "
            "finite rectangular matrix shape; not an incremental or floating-point sector"
        ),
        "locked_design": {
            "projection_gain": _fraction(LOCKED_PROJECTION_GAIN),
            "passive_divisor": _fraction(LOCKED_PASSIVE_DIVISOR),
            "gate_cap": _fraction(LOCKED_GATE_CAP),
            "pointwise_sector": {
                "lower": _fraction(sector.lower),
                "upper": _fraction(sector.upper),
                "center": _fraction(sector.center),
                "radius": _fraction(sector.radius),
                "condition_ratio": _fraction(sector.condition_ratio),
            },
            "selected_learning_rate": _fraction(SELECTED_LEARNING_RATE),
            "selected_effective_update_sector": {
                "lower": _fraction(effective_lower),
                "upper": _fraction(effective_upper),
            },
            "global_amplitude_gate_passes": candidate_passes,
        },
        "undersized_projection_control": {
            "projection_gain": _fraction(UNDERSIZED_PROJECTION_GAIN),
            "pointwise_sector": {
                "lower": _fraction(small_sector.lower),
                "upper": _fraction(small_sector.upper),
            },
            "interpretation": (
                "its global scale bound is benign, but the sampled fidelity/amplitude "
                "control tests whether excessive projection erases Muon shaping"
            ),
        },
        "unprojected_control": {
            "pointwise_sector": {
                "lower": _fraction(unprojected_lower),
                "upper": _fraction(unprojected_upper),
            },
            "selected_effective_upper": _fraction(SELECTED_LEARNING_RATE * unprojected_upper),
            "global_amplitude_gate_passes": (
                unprojected_lower >= GLOBAL_INPUT_AMPLITUDE_LOWER
                and unprojected_upper <= GLOBAL_INPUT_AMPLITUDE_UPPER
            ),
            "selected_effective_update_gate_passes": (
                SELECTED_LEARNING_RATE * unprojected_lower >= EFFECTIVE_UPDATE_LOWER
                and SELECTED_LEARNING_RATE * unprojected_upper <= EFFECTIVE_UPDATE_UPPER
            ),
            "interpretation": (
                "removing the ray projection recovers P17's huge raw-shape sector and "
                "fails both the global amplitude and selected-step upper gates"
            ),
        },
    }


def _git_state() -> dict[str, object]:
    def run(*arguments: str) -> str | None:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    status = run("status", "--porcelain")
    return {
        "sha": run("rev-parse", "HEAD") or "unavailable",
        "branch": run("branch", "--show-current") or "unavailable",
        "dirty": bool(status) if status is not None else None,
    }


def _source_snapshot() -> dict[str, str]:
    return {
        path.as_posix(): hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in SOURCE_PATHS
    }


def _provenance(config: P18StudyConfig) -> dict[str, object]:
    return {
        "git": _git_state(),
        "seed": config.seed,
        "hardware": {
            "device": "cpu",
            "machine": platform.machine(),
            "processor": platform.processor() or "unreported",
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
        },
        "software": {
            "python": sys.version,
            "numpy": str(np.__version__),
            "blas_lapack": np.__config__.CONFIG,
        },
        "source_snapshot": _source_snapshot(),
    }


def run_study(config: P18StudyConfig | None = None) -> dict[str, object]:
    """Run exact gate checks and all declared deterministic diagnostics."""

    selected = P18StudyConfig() if config is None else config
    exact = _exact_gate_records()
    ledger = _CallLedger()
    canonical = _canonical_case(selected.solver, ledger)
    broad = _run_grid(selected, ledger, operating_annulus=False)
    annulus = _run_grid(selected, ledger, operating_annulus=True)
    transformer = _run_realistic_spectra(selected, ledger)
    return {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": {
            "global_exact": (
                "the locked origin-sector projection gives a pointwise amplitude sector "
                "for every finite rectangular matrix shape in exact real arithmetic"
            ),
            "sampled_evidence": (
                "canonical, broad-grid, annulus, and Transformer fidelity/amplitude "
                "results are deterministic FP64 diagnostics, not global certificates"
            ),
            "residual_scope": (
                "every solver call checks the actual computed P15 graph residual; no claim "
                "is made that FP64 evaluation error is enclosed by directed rounding"
            ),
            "incremental_warning": (
                "the pointwise sector is sufficient for the separate P6-style trajectory "
                "LMI but is not an incremental Jacobian sector or arbitrary-pair contraction"
            ),
            "effective_update_comparison": (
                "P18 uses eta=1/83, versus 1/128000 for the P17 primary design and "
                "1/32000 for the P17/P14 full step; the reported effective-update ratios "
                "therefore expose the intentionally much larger step scale and are not "
                "optimizer-parity claims"
            ),
        },
        "config": {
            **asdict(selected),
            "solver": asdict(selected.solver),
            "normalization": "S/(||S||_F+epsilon)",
            "epsilon": selected.solver.epsilon,
            "jordan_coefficients": ["6889/2000", "-191/40", "4063/2000"],
            "jordan_iteration_count": STAGES,
            "selected_learning_rate": _fraction(SELECTED_LEARNING_RATE),
            "faster_rate_frontier_learning_rate": _fraction(FASTER_RATE_FRONTIER_LEARNING_RATE),
            "selected_learning_rate_ratio_to_p17_primary": _fraction(
                SELECTED_LEARNING_RATE / PRIMARY_DESIGN.learning_rate
            ),
            "selected_learning_rate_ratio_to_p17_full_step": _fraction(
                SELECTED_LEARNING_RATE / FULL_STEP_DESIGN.learning_rate
            ),
            "p15_stopping_rule": "||r||_F <= ||s||_F/250 + rbar_fp64",
            "frozen_p16_fidelity_thresholds": {
                "best_scalar_departure": MEANINGFUL_SCALAR_RESIDUAL_THRESHOLD,
                "upstream_shaping_retention": MEANINGFUL_SHAPING_RETENTION_THRESHOLD,
                "informative_upstream_departure": INFORMATIVE_UPSTREAM_SHAPING_THRESHOLD,
            },
            "scale_sensitive_gates": {
                "global_output_to_input": {
                    "lower": _fraction(GLOBAL_INPUT_AMPLITUDE_LOWER),
                    "upper": _fraction(GLOBAL_INPUT_AMPLITUDE_UPPER),
                },
                "annulus_output_to_upstream": {
                    "lower": _fraction(ANNULUS_UPSTREAM_AMPLITUDE_LOWER),
                    "upper": _fraction(ANNULUS_UPSTREAM_AMPLITUDE_UPPER),
                },
                "selected_effective_update_per_input": {
                    "lower": _fraction(EFFECTIVE_UPDATE_LOWER),
                    "upper": _fraction(EFFECTIVE_UPDATE_UPPER),
                },
            },
            "realistic_transformer_shapes": [list(shape) for shape in REALISTIC_TRANSFORMER_SHAPES],
        },
        "upstream_formula_provenance": {
            "repository": "https://github.com/KellerJordan/Muon",
            "revision": PINNED_KELLER_JORDAN_MUON_REVISION,
            "audited_file": "muon.py",
            "audited_file_sha256": PINNED_MUON_PY_SHA256,
            "normalization": "X/(X.norm(dim=(-2,-1),keepdim=True)+1e-7)",
            "comparison_scope": (
                "FP64 five-stage spectral counterpart before aspect scaling and BF16 casts"
            ),
        },
        "exact_global_amplitude_and_effective_update_gates": exact,
        "canonical_diag_3_4": canonical,
        "declared_broad_two_mode_grid": broad,
        "declared_operating_annulus_grid": annulus,
        "realistic_complete_spectrum_cases": transformer,
        "solver_summary": ledger.summary(),
        "experiment_provenance": _provenance(selected),
        "canonical_target": str(CANONICAL_TARGET.relative_to(ROOT)),
    }


def canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--spectrum-grid-points", type=int, default=17)
    parser.add_argument("--operating-annulus-grid-points", type=int, default=33)
    parser.add_argument("--skip-realistic-spectrum-cases", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    config = P18StudyConfig(
        seed=arguments.seed,
        spectrum_grid_points=arguments.spectrum_grid_points,
        operating_annulus_grid_points=arguments.operating_annulus_grid_points,
        include_realistic_spectrum_cases=not arguments.skip_realistic_spectrum_cases,
    )
    payload = run_study(config)
    rendered = canonical_json(payload)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
