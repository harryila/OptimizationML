#!/usr/bin/env python3
"""Deterministic P17 shape-preserving resolvent diagnostics.

The exact P17 theorem is the dimension-independent pointwise gain sector in
``passive_muon.shape_preserving_resolvent``.  This script supplies numerical
falsification and fidelity evidence for its two locked C2-gated designs.  It
does not turn a sampled spectrum grid into a global theorem and it does not
certify IEEE-754 rounding.

Every declared evaluation calls the guarded P16 solver and fails closed unless
the *computed* graph residual satisfies the P15 stopping rule.  Transformer
cases use complete ``min(m,n)`` singular-value vectors, but do not allocate a
dense matrix, execute an SVD, or benchmark an accelerator.
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
    radial_deficit_majorant_fp64,
    resolvent_graph_singular_values_fp64,
    solve_resolvent_singular_values,
)
from passive_muon.floored_certificate import JORDAN_DERIVATIVE_UPPER
from passive_muon.radial_passivation_tradeoff import P11_SIGNAL_NORM_BOUND
from passive_muon.shape_preserving_resolvent import (
    FULL_STEP_DESIGN,
    LOCKED_GATE_Q0,
    LOCKED_GATE_Q1,
    PRIMARY_DESIGN,
    GateDesign,
    audit_shape_preserving_resolvent,
    evaluate_gated_singular_values_fp64,
    evaluate_shape_preserving_resolvent_fp64,
    pointwise_gain_sector,
    quintic_gate_weight_fp64,
)
from passive_muon.shape_preserving_resolvent_certificate import (
    UNDERSIZED_GATE_INNER_RADIUS,
    UNDERSIZED_GATE_OUTER_RADIUS,
    UNSAFE_BAND_LEFT,
)
from passive_muon.upstream_momentum import (
    PINNED_KELLER_JORDAN_MUON_REVISION,
    PINNED_MUON_PY_SHA256,
)

FloatArray = NDArray[np.float64]

SCHEMA_VERSION: Final = "passive-muon-p17-shape-preserving-study-v1"
SEED: Final = 20_260_905
STAGES: Final = 5
P11_SIGNAL_GUARD: Final = float(P11_SIGNAL_NORM_BOUND)

# These are copied, not retuned, from the frozen P16 fidelity study.
MEANINGFUL_SCALAR_RESIDUAL_THRESHOLD: Final = 1.0e-3
MEANINGFUL_SHAPING_RETENTION_THRESHOLD: Final = 1.0e-1
INFORMATIVE_UPSTREAM_SHAPING_THRESHOLD: Final = 1.0e-2

DESIGNS: Final = (PRIMARY_DESIGN, FULL_STEP_DESIGN)
REALISTIC_TRANSFORMER_SHAPES: Final = (
    (768, 768),
    (768, 3_072),
    (3_072, 12_288),
    (4_096, 11_008),
)
RANK_ACCUMULATION_RANKS: Final = (1, 2, 16, 64, 256, 768, 4_096, 11_008)
RANK_ACCUMULATION_MODE_RATIO: Final = 41.0 / 10_000.0

# This post-exploratory operating annulus is deliberately separate from the
# broad grid.  The broad grid established that no uniform fidelity claim is
# defensible near the origin, where the locked gate is off.  These exact
# rational endpoints are now frozen for the narrower sampled diagnostic.
OPERATING_ANNULUS_INNER_RADIUS: Final = Fraction(3, 4)
OPERATING_ANNULUS_OUTER_RADIUS: Final = Fraction(25)
OPERATING_ANNULUS_STRICT_RELATIVE_TOLERANCE: Final = 2.0**-44
OPERATING_ANNULUS_STRICT_ABSOLUTE_TOLERANCE: Final = 2.0**-50

ROOT = Path(__file__).resolve().parents[2]
CANONICAL_TARGET = ROOT / "results" / "summaries" / "p17_shape_preserving_study.json"
SOURCE_PATHS = (
    Path("experiments/resolvent/run_p17_shape_preserving_study.py"),
    Path("experiments/resolvent/run_p16_solver_study.py"),
    Path("src/passive_muon/additive_epsilon_deficit.py"),
    Path("src/passive_muon/equivariant_resolvent_solver.py"),
    Path("src/passive_muon/floored_certificate.py"),
    Path("src/passive_muon/momentum_iqc.py"),
    Path("src/passive_muon/pl_convergence.py"),
    Path("src/passive_muon/radial_passivation_tradeoff.py"),
    Path("src/passive_muon/shape_preserving_resolvent.py"),
    Path("src/passive_muon/shape_preserving_resolvent_certificate.py"),
    Path("src/passive_muon/specs.py"),
    Path("src/passive_muon/structure_aware_stability.py"),
    Path("src/passive_muon/upstream_momentum.py"),
    Path("src/passive_muon/yosida_stability.py"),
    Path("pyproject.toml"),
    Path("uv.lock"),
)


@dataclass(frozen=True)
class P17StudyConfig:
    """Complete deterministic configuration of the focused numerical study."""

    seed: int = SEED
    spectrum_grid_points: int = 17
    operating_annulus_grid_points: int = 33
    rank_radius_points: int = 13
    include_realistic_spectrum_cases: bool = True
    solver: EquivariantResolventSolverConfig = field(
        default_factory=EquivariantResolventSolverConfig
    )

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for name in (
            "spectrum_grid_points",
            "operating_annulus_grid_points",
            "rank_radius_points",
        ):
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
    x = np.asarray(signal, dtype=np.float64).ravel()
    y = np.asarray(output, dtype=np.float64).ravel()
    square = float(np.dot(x, x))
    output_norm = _norm(y)
    if square == 0.0 or output_norm == 0.0:
        return {
            "coefficient": None,
            "residual_frobenius": 0.0 if output_norm == 0.0 else output_norm,
            "relative_residual": 0.0 if output_norm == 0.0 else 1.0,
        }
    coefficient = float(np.dot(x, y)) / square
    residual_norm = _norm(y - coefficient * x)
    return {
        "coefficient": coefficient,
        "residual_frobenius": residual_norm,
        "relative_residual": residual_norm / output_norm,
    }


def _directional_sine(left: FloatArray, right: FloatArray) -> float | None:
    x = np.asarray(left, dtype=np.float64).ravel()
    y = np.asarray(right, dtype=np.float64).ravel()
    denominator = _norm(x) * _norm(y)
    if denominator == 0.0:
        return None
    cosine = float(np.dot(x, y)) / denominator
    cosine = max(-1.0, min(1.0, cosine))
    return math.sqrt(max(0.0, 1.0 - cosine * cosine))


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


def _modal_gain_summary(signal: FloatArray, output: FloatArray) -> dict[str, object]:
    selected = np.asarray(signal, dtype=np.float64)
    gains = np.divide(
        output,
        selected,
        out=np.full_like(output, np.nan),
        where=selected > 0.0,
    )
    finite = gains[np.isfinite(gains)]
    if finite.size == 0:
        return {"positive_mode_count": 0, "values": []}
    summary: dict[str, object] = {
        "positive_mode_count": int(finite.size),
        "minimum": float(np.min(finite)),
        "median": float(np.median(finite)),
        "maximum": float(np.max(finite)),
        "spread": float(np.max(finite) - np.min(finite)),
    }
    if finite.size <= 8:
        summary["values"] = [float(value) for value in finite]
    else:
        summary["quantiles_05_50_95"] = [
            float(value) for value in np.quantile(finite, [0.05, 0.5, 0.95])
        ]
    return summary


def _computed_residual_error_allowance(
    diagnostics: SolverDiagnostics,
    gate_weight: float,
    design: GateDesign,
    solver: EquivariantResolventSolverConfig,
) -> dict[str, float]:
    """Propagate the computed graph residual through conservative exact bounds.

    The inequalities ``||u_hat-J(s)|| <= ||r||/2`` and
    ``||Y_hat-Y|| <= 500||r||`` are exact-real consequences of the P14 graph.
    ``Lip(E)<=b/epsilon`` supplies a deliberately coarse bound for the shape
    component.  The recorded ``r`` itself is still computed in FP64, so this
    is an error allowance, not a directed-rounding enclosure.
    """

    residual = diagnostics.residual_norm
    solution = 0.5 * residual
    yosida = 500.0 * residual
    jordan = float(JORDAN_DERIVATIVE_UPPER) * solution / solver.epsilon
    blended = (1.0 - gate_weight) * yosida / float(design.passive_divisor) + gate_weight * jordan
    return {
        "computed_graph_residual": residual,
        "solution_error_allowance": solution,
        "yosida_output_error_allowance": yosida,
        "jordan_output_error_allowance": jordan,
        "blended_output_error_allowance": blended,
    }


def _fidelity_record(
    signal: FloatArray,
    output: FloatArray,
    upstream: FloatArray,
    *,
    output_error_allowance: float,
) -> dict[str, object]:
    signal_norm = _norm(signal)
    output_norm = _norm(output)
    best = _best_scalar_record(signal, output)
    upstream_best = _best_scalar_record(signal, upstream)
    distance = float(best["residual_frobenius"])
    lower_denominator = output_norm + output_error_allowance
    upper_denominator = max(output_norm - output_error_allowance, 0.0)
    lower = (
        max(0.0, distance - output_error_allowance) / lower_denominator
        if lower_denominator > 0.0
        else 0.0
    )
    upper = (
        min(1.0, (distance + output_error_allowance) / upper_denominator)
        if upper_denominator > 0.0
        else 1.0
    )
    upstream_departure = float(upstream_best["relative_residual"])
    retention = (
        float(best["relative_residual"]) / upstream_departure if upstream_departure > 0.0 else None
    )
    retention_interval = (
        [lower / upstream_departure, upper / upstream_departure]
        if upstream_departure > 0.0
        else None
    )
    amplitude = output_norm / signal_norm if signal_norm > 0.0 else 0.0
    amplitude_interval = (
        [
            max(0.0, output_norm - output_error_allowance) / signal_norm,
            (output_norm + output_error_allowance) / signal_norm,
        ]
        if signal_norm > 0.0
        else [0.0, 0.0]
    )
    absolute_gate = float(best["relative_residual"]) >= (MEANINGFUL_SCALAR_RESIDUAL_THRESHOLD)
    retention_gate = retention is not None and retention >= MEANINGFUL_SHAPING_RETENTION_THRESHOLD
    return {
        "input_frobenius": signal_norm,
        "output_frobenius": output_norm,
        "output_to_input_amplitude": amplitude,
        "computed_residual_amplitude_interval": amplitude_interval,
        "best_scalar_multiple": best,
        "computed_residual_best_scalar_departure_interval": [lower, upper],
        "upstream_best_scalar_multiple": upstream_best,
        "shaping_retention_fraction": retention,
        "computed_residual_shaping_retention_interval": retention_interval,
        "modal_gains": _modal_gain_summary(signal, output),
        "directional_sine_to_upstream": _directional_sine(output, upstream),
        "absolute_shaping_gate_passes": absolute_gate,
        "upstream_retention_gate_passes": retention_gate,
        "meaningful_fidelity_gate_passes": absolute_gate and retention_gate,
    }


def _diagnostics_record(diagnostics: SolverDiagnostics) -> dict[str, object]:
    return {
        "status": diagnostics.status.value,
        "certified": diagnostics.certified,
        "iterations": diagnostics.iterations,
        "backtracks": diagnostics.backtracks,
        "actual_residual_norm": diagnostics.residual_norm,
        "p15_threshold": diagnostics.p15_threshold,
        "residual_to_p15_threshold_ratio": (
            diagnostics.residual_norm / diagnostics.p15_threshold
            if diagnostics.p15_threshold > 0.0
            else 0.0
        ),
        "operator_evaluations": diagnostics.operator_evaluations,
        "jacobian_evaluations": diagnostics.jacobian_evaluations,
        "sherman_morrison_solves": diagnostics.sherman_morrison_solves,
        "svd_count": diagnostics.svd_count,
        "reconstruction_matmuls": diagnostics.reconstruction_matmuls,
    }


class _CallLedger:
    """Collect solver evidence and enforce the fail-closed study policy."""

    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    def add(self, label: str, dimension: int, diagnostics: SolverDiagnostics) -> None:
        if not diagnostics.certified:
            raise RuntimeError(f"declared solver call {label!r} failed closed")
        if diagnostics.residual_norm > diagnostics.p15_threshold:
            raise RuntimeError(f"declared solver call {label!r} violated its P15 postcondition")
        self.records.append(
            {
                "label": label,
                "dimension": dimension,
                **_diagnostics_record(diagnostics),
            }
        )

    def summary(self) -> dict[str, object]:
        if not self.records:
            raise RuntimeError("study executed no solver calls")
        iterations = [int(record["iterations"]) for record in self.records]
        backtracks = [int(record["backtracks"]) for record in self.records]
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
            "mean_iterations": float(np.mean(iterations)),
            "maximum_backtracks": max(backtracks),
            "worst_residual_to_p15_threshold_ratio": max(ratios),
            "maximum_actual_residual": max(
                float(record["actual_residual_norm"]) for record in self.records
            ),
            "total_operator_evaluations": sum(
                int(record["operator_evaluations"]) for record in self.records
            ),
            "total_jacobian_evaluations": sum(
                int(record["jacobian_evaluations"]) for record in self.records
            ),
            "total_sherman_morrison_solves": sum(
                int(record["sherman_morrison_solves"]) for record in self.records
            ),
            "scalar_coordinate_iteration_work_proxy": sum(
                int(record["dimension"]) * int(record["iterations"]) for record in self.records
            ),
        }


def _design_record(
    signal: FloatArray,
    output: FloatArray,
    upstream: FloatArray,
    gate_weight: float,
    design: GateDesign,
    diagnostics: SolverDiagnostics,
    solver: EquivariantResolventSolverConfig,
) -> dict[str, object]:
    allowance = _computed_residual_error_allowance(
        diagnostics,
        gate_weight,
        design,
        solver,
    )
    fidelity = _fidelity_record(
        signal,
        output,
        upstream,
        output_error_allowance=allowance["blended_output_error_allowance"],
    )
    return {
        "design": design.name,
        "gate_weight": gate_weight,
        "diagnostics": _diagnostics_record(diagnostics),
        "computed_residual_error_allowance": allowance,
        "fidelity": fidelity,
    }


def _evaluate_spectrum_designs(
    signal_values: FloatArray,
    *,
    label: str,
    solver: EquivariantResolventSolverConfig,
    ledger: _CallLedger,
) -> dict[str, dict[str, object]]:
    upstream = _upstream_jordan_singular_values(signal_values, solver.epsilon)
    records: dict[str, dict[str, object]] = {}
    for design in DESIGNS:
        evaluated = evaluate_gated_singular_values_fp64(
            np.ascontiguousarray(signal_values, dtype=np.float64),
            design,
            solver_config=solver,
        )
        ledger.add(
            f"{label}:{design.name}",
            signal_values.size,
            evaluated.solver_diagnostics,
        )
        records[design.name] = _design_record(
            signal_values,
            evaluated.output,
            upstream,
            evaluated.gate_value,
            design,
            evaluated.solver_diagnostics,
            solver,
        )
    return records


def _canonical_cases(
    solver: EquivariantResolventSolverConfig,
    ledger: _CallLedger,
) -> dict[str, object]:
    signal = np.diag(np.asarray([3.0, 4.0], dtype=np.float64))
    upstream_values = _upstream_jordan_singular_values(
        np.asarray([3.0, 4.0], dtype=np.float64),
        solver.epsilon,
    )
    upstream = np.diag(upstream_values)
    designs: dict[str, object] = {}
    for design in DESIGNS:
        evaluated = evaluate_shape_preserving_resolvent_fp64(
            signal,
            design,
            solver_config=solver,
        )
        solved = evaluated.resolvent_result
        ledger.add(f"canonical_diag_3_4:{design.name}", 2, solved.diagnostics)
        allowance = _computed_residual_error_allowance(
            solved.diagnostics,
            evaluated.gate_weight,
            design,
            solver,
        )
        designs[design.name] = {
            "gate_weight": evaluated.gate_weight,
            "squared_signal_norm": evaluated.squared_signal_norm,
            "input_sha256_float64_bytes": _sha256_float64(signal),
            "approximate_solution_singular_values": [
                float(value) for value in solved.solution_singular_values
            ],
            "diagnostics": _diagnostics_record(solved.diagnostics),
            "computed_residual_error_allowance": allowance,
            "fidelity": _fidelity_record(
                signal,
                evaluated.output,
                upstream,
                output_error_allowance=allowance["blended_output_error_allowance"],
            ),
        }
    return {
        "input": "diag(3,4)",
        "scope": (
            "full 2x2 SVD/reconstruction path; FP64 evidence with a conservative "
            "computed-residual allowance, not an interval-arithmetic root certificate"
        ),
        "designs": designs,
    }


def _declared_grid(points: int) -> tuple[FloatArray, FloatArray]:
    radii = np.geomspace(1.0e-12, P11_SIGNAL_GUARD, points, dtype=np.float64)
    radii = np.unique(
        np.concatenate(
            (
                radii,
                np.asarray([0.475, 1.0, 1.5, 1.67, 2.075, 3.0], dtype=np.float64),
            )
        )
    )
    ratios = np.geomspace(1.0e-6, 1.0, points, dtype=np.float64)
    ratios = np.unique(
        np.concatenate(
            (
                ratios,
                np.asarray([1_000.0 / 249_999.0, RANK_ACCUMULATION_MODE_RATIO]),
            )
        )
    )
    return radii, ratios


def _extreme(
    current: dict[str, object] | None,
    candidate: dict[str, object],
    key: str,
    *,
    maximize: bool,
) -> dict[str, object]:
    if current is None:
        return candidate
    left = float(current["metrics"][key])  # type: ignore[index]
    right = float(candidate["metrics"][key])  # type: ignore[index]
    if (right > left) == maximize:
        return candidate
    return current


def _run_declared_spectrum_grid(
    config: P17StudyConfig,
    ledger: _CallLedger,
) -> dict[str, object]:
    radii, ratios = _declared_grid(config.spectrum_grid_points)
    states: dict[str, dict[str, object]] = {
        design.name: {
            "evaluated": 0,
            "informative": 0,
            "gate_passes": 0,
            "min_departure": None,
            "max_departure": None,
            "min_retention": None,
            "max_amplitude": None,
        }
        for design in DESIGNS
    }
    skipped_over_guard = 0
    for radius in radii:
        for ratio in ratios:
            direction = _unit(np.asarray([ratio, 1.0], dtype=np.float64))
            constructed_solution = radius * direction
            signal = resolvent_graph_singular_values_fp64(
                constructed_solution,
                config=config.solver,
            )
            if _norm(signal) > config.solver.signal_norm_guard:
                skipped_over_guard += 1
                continue
            label = f"grid:r={radius:.17g}:a={ratio:.17g}"
            records = _evaluate_spectrum_designs(
                signal,
                label=label,
                solver=config.solver,
                ledger=ledger,
            )
            for design in DESIGNS:
                state = states[design.name]
                record = records[design.name]
                fidelity = record["fidelity"]
                assert isinstance(fidelity, dict)
                departure = float(fidelity["best_scalar_multiple"]["relative_residual"])
                upstream_departure = float(
                    fidelity["upstream_best_scalar_multiple"]["relative_residual"]
                )
                retention = fidelity["shaping_retention_fraction"]
                metrics = {
                    "best_scalar_departure": departure,
                    "upstream_best_scalar_departure": upstream_departure,
                    "shaping_retention_fraction": retention,
                    "output_to_input_amplitude": fidelity["output_to_input_amplitude"],
                    "gate_weight": record["gate_weight"],
                    "residual_to_p15_threshold_ratio": record["diagnostics"][
                        "residual_to_p15_threshold_ratio"
                    ],
                }
                candidate = {
                    "constructed_solution_radius": float(radius),
                    "constructed_solution_mode_ratio": float(ratio),
                    "input_frobenius": _norm(signal),
                    "metrics": metrics,
                }
                state["evaluated"] = int(state["evaluated"]) + 1
                state["max_amplitude"] = _extreme(
                    state["max_amplitude"], candidate, "output_to_input_amplitude", maximize=True
                )
                if upstream_departure < INFORMATIVE_UPSTREAM_SHAPING_THRESHOLD:
                    continue
                state["informative"] = int(state["informative"]) + 1
                if bool(fidelity["meaningful_fidelity_gate_passes"]):
                    state["gate_passes"] = int(state["gate_passes"]) + 1
                state["min_departure"] = _extreme(
                    state["min_departure"], candidate, "best_scalar_departure", maximize=False
                )
                state["max_departure"] = _extreme(
                    state["max_departure"], candidate, "best_scalar_departure", maximize=True
                )
                if retention is not None:
                    state["min_retention"] = _extreme(
                        state["min_retention"],
                        candidate,
                        "shaping_retention_fraction",
                        maximize=False,
                    )

    for state in states.values():
        informative = int(state["informative"])
        state["frozen_gate_pass_fraction_on_informative_points"] = (
            int(state["gate_passes"]) / informative if informative else None
        )
    return {
        "status": "sampled diagnostic; extrema are not global certificates",
        "parameterization": (
            "u=r*(a,1)/sqrt(1+a^2), followed by s=u+lambda*B(u); every retained "
            "s is then independently resolved by the fail-closed P16 solver"
        ),
        "radius_count": int(radii.size),
        "mode_ratio_count": int(ratios.size),
        "candidate_count": int(radii.size * ratios.size),
        "skipped_over_p11_guard": skipped_over_guard,
        "informative_upstream_shaping_threshold": INFORMATIVE_UPSTREAM_SHAPING_THRESHOLD,
        "frozen_p16_absolute_threshold": MEANINGFUL_SCALAR_RESIDUAL_THRESHOLD,
        "frozen_p16_retention_threshold": MEANINGFUL_SHAPING_RETENTION_THRESHOLD,
        "designs": states,
    }


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


def _run_operating_annulus_grid(
    config: P17StudyConfig,
    ledger: _CallLedger,
) -> dict[str, object]:
    """Check the frozen P16 gates on a separately declared input annulus.

    Unlike :func:`_run_declared_spectrum_grid`, this grid directly fixes the
    input Frobenius radius.  It is intentionally a sampled, post-exploratory
    operating-domain statement for the primary design, not a global fidelity
    theorem.  A future numerical regression fails the study rather than
    silently weakening the frozen gate.
    """

    radii, ratios = _operating_annulus_grid(config.operating_annulus_grid_points)
    diagnostic_solver = replace(
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
    informative = 0
    point_gate_passes = 0
    allowance_gate_passes = 0
    minimum_departure: dict[str, object] | None = None
    minimum_retention: dict[str, object] | None = None
    minimum_allowance_departure: dict[str, object] | None = None
    minimum_allowance_retention: dict[str, object] | None = None
    maximum_amplitude: dict[str, object] | None = None
    minimum_gate_weight = math.inf
    maximum_gate_weight = -math.inf

    for radius in radii:
        for ratio in ratios:
            signal = radius * _unit(np.asarray([ratio, 1.0], dtype=np.float64))
            evaluated = evaluate_gated_singular_values_fp64(
                signal,
                PRIMARY_DESIGN,
                solver_config=diagnostic_solver,
            )
            ledger.add(
                f"operating_annulus:r={radius:.17g}:a={ratio:.17g}",
                2,
                evaluated.solver_diagnostics,
            )
            minimum_gate_weight = min(minimum_gate_weight, evaluated.gate_value)
            maximum_gate_weight = max(maximum_gate_weight, evaluated.gate_value)
            upstream = _upstream_jordan_singular_values(signal, config.solver.epsilon)
            allowance = _computed_residual_error_allowance(
                evaluated.solver_diagnostics,
                evaluated.gate_value,
                PRIMARY_DESIGN,
                diagnostic_solver,
            )
            fidelity = _fidelity_record(
                signal,
                evaluated.output,
                upstream,
                output_error_allowance=allowance["blended_output_error_allowance"],
            )
            upstream_departure = float(
                fidelity["upstream_best_scalar_multiple"]["relative_residual"]
            )
            departure = float(fidelity["best_scalar_multiple"]["relative_residual"])
            retention = fidelity["shaping_retention_fraction"]
            departure_interval = fidelity["computed_residual_best_scalar_departure_interval"]
            retention_interval = fidelity["computed_residual_shaping_retention_interval"]
            assert isinstance(departure_interval, list)
            candidate = {
                "input_frobenius": float(radius),
                "mode_ratio": float(ratio),
                "upstream_best_scalar_departure": upstream_departure,
                "best_scalar_departure": departure,
                "shaping_retention_fraction": retention,
                "computed_residual_departure_lower": float(departure_interval[0]),
                "computed_residual_retention_lower": (
                    float(retention_interval[0]) if isinstance(retention_interval, list) else None
                ),
                "output_to_input_amplitude": fidelity["output_to_input_amplitude"],
                "gate_weight": evaluated.gate_value,
                "residual_to_p15_threshold_ratio": (
                    evaluated.solver_diagnostics.residual_norm
                    / evaluated.solver_diagnostics.p15_threshold
                ),
            }
            if maximum_amplitude is None or float(candidate["output_to_input_amplitude"]) > float(
                maximum_amplitude["output_to_input_amplitude"]
            ):
                maximum_amplitude = candidate
            if upstream_departure < INFORMATIVE_UPSTREAM_SHAPING_THRESHOLD:
                continue
            informative += 1
            if bool(fidelity["meaningful_fidelity_gate_passes"]):
                point_gate_passes += 1
            allowance_retention_lower = candidate["computed_residual_retention_lower"]
            allowance_passes = (
                float(candidate["computed_residual_departure_lower"])
                >= MEANINGFUL_SCALAR_RESIDUAL_THRESHOLD
                and allowance_retention_lower is not None
                and float(allowance_retention_lower) >= MEANINGFUL_SHAPING_RETENTION_THRESHOLD
            )
            if allowance_passes:
                allowance_gate_passes += 1
            if minimum_departure is None or departure < float(
                minimum_departure["best_scalar_departure"]
            ):
                minimum_departure = candidate
            if minimum_retention is None or float(retention) < float(
                minimum_retention["shaping_retention_fraction"]
            ):
                minimum_retention = candidate
            if minimum_allowance_departure is None or float(
                candidate["computed_residual_departure_lower"]
            ) < float(minimum_allowance_departure["computed_residual_departure_lower"]):
                minimum_allowance_departure = candidate
            if minimum_allowance_retention is None or float(
                candidate["computed_residual_retention_lower"]
            ) < float(minimum_allowance_retention["computed_residual_retention_lower"]):
                minimum_allowance_retention = candidate

    if informative == 0:
        raise RuntimeError("operating-annulus grid contained no informative fidelity point")
    if point_gate_passes != informative:
        raise RuntimeError("a primary-design operating-annulus point failed a frozen P16 gate")
    if allowance_gate_passes != informative:
        raise RuntimeError(
            "a primary-design operating-annulus point failed after the computed-residual allowance"
        )
    return {
        "status": (
            "all informative sampled primary-design points pass both frozen P16 gates, "
            "including the conservative computed-residual allowance"
        ),
        "scope": (
            "post-exploratory sampled operating-annulus diagnostic; not a global theorem, "
            "not an interval-arithmetic floating-point certificate, and not evidence near zero"
        ),
        "selection_disclosure": (
            "the annulus was locked only after the broad grid exposed the expected gate-off "
            "fidelity failure near the origin"
        ),
        "numerical_solver_policy": {
            "operator_and_p15_postcondition_unchanged": True,
            "purpose": (
                "a stricter internal Newton target makes the conservative graph-residual "
                "fidelity allowance informative; it does not alter the P15 acceptance rule"
            ),
            "strict_relative_tolerance": diagnostic_solver.strict_relative_tolerance,
            "strict_absolute_tolerance": diagnostic_solver.strict_absolute_tolerance,
            "p15_relative_tolerance": diagnostic_solver.p15_relative_tolerance,
            "p15_absolute_tolerance": diagnostic_solver.p15_absolute_tolerance,
        },
        "design": PRIMARY_DESIGN.name,
        "input_radius_interval": {
            "lower": _fraction(OPERATING_ANNULUS_INNER_RADIUS),
            "upper": _fraction(OPERATING_ANNULUS_OUTER_RADIUS),
        },
        "radius_count": int(radii.size),
        "mode_ratio_count": int(ratios.size),
        "candidate_count": int(radii.size * ratios.size),
        "informative_point_count": informative,
        "informative_upstream_shaping_threshold": INFORMATIVE_UPSTREAM_SHAPING_THRESHOLD,
        "frozen_p16_absolute_threshold": MEANINGFUL_SCALAR_RESIDUAL_THRESHOLD,
        "frozen_p16_retention_threshold": MEANINGFUL_SHAPING_RETENTION_THRESHOLD,
        "point_estimate_gate_pass_count": point_gate_passes,
        "computed_residual_allowance_gate_pass_count": allowance_gate_passes,
        "all_informative_point_estimates_pass": point_gate_passes == informative,
        "all_informative_computed_residual_allowances_pass": (allowance_gate_passes == informative),
        "gate_weight_range": [minimum_gate_weight, maximum_gate_weight],
        "minimum_best_scalar_departure": minimum_departure,
        "minimum_shaping_retention": minimum_retention,
        "minimum_computed_residual_departure_lower": minimum_allowance_departure,
        "minimum_computed_residual_retention_lower": minimum_allowance_retention,
        "maximum_output_to_input_amplitude": maximum_amplitude,
    }


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
    config: P17StudyConfig,
    ledger: _CallLedger,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    if not config.include_realistic_spectrum_cases:
        return records
    for index, shape in enumerate(REALISTIC_TRANSFORMER_SHAPES):
        label, signal = _realistic_spectrum(shape, index)
        designs = _evaluate_spectrum_designs(
            signal,
            label=f"realistic:{shape[0]}x{shape[1]}:{label}",
            solver=config.solver,
            ledger=ledger,
        )
        records.append(
            {
                "label": label,
                "declared_matrix_shape": list(shape),
                "singular_value_count": int(signal.size),
                "representation": (
                    "complete min(m,n) singular spectrum; no dense matrix, SVD, or "
                    "accelerator timing"
                ),
                "input_frobenius": _norm(signal),
                "input_sha256_float64_bytes": _sha256_float64(signal),
                "input_spectrum_summary": {
                    "minimum": float(np.min(signal)),
                    "median": float(np.median(signal)),
                    "maximum": float(np.max(signal)),
                    "positive_condition_number": float(np.max(signal) / np.min(signal)),
                },
                "designs": designs,
                "cost_comparison": {
                    "hypothetical_five_stage_jordan_dense_gemm_count": 15,
                    "executed_solver_cost": "scalar diagonal-plus-rank-one Newton system",
                    "important_caveat": (
                        "an actual dense deployment also pays for an SVD and reconstruction; "
                        "this spectrum-only replay is not a runtime benchmark"
                    ),
                },
            }
        )
    return records


def _rank_radii(points: int) -> FloatArray:
    return np.unique(
        np.concatenate(
            (
                np.geomspace(1.0e-12, 3.0, points, dtype=np.float64),
                np.asarray([1.0e-7, 1.0e-5, 0.001, 0.1, 0.475, 1.0, 1.5, 3.0]),
            )
        )
    )


def _run_rank_accumulation(
    config: P17StudyConfig,
    ledger: _CallLedger,
) -> dict[str, object]:
    radii = _rank_radii(config.rank_radius_points)
    by_rank: list[dict[str, object]] = []
    for rank in RANK_ACCUMULATION_RANKS:
        direction = np.full(rank, RANK_ACCUMULATION_MODE_RATIO, dtype=np.float64)
        direction[0] = 1.0
        direction = _unit(direction)
        maxima: dict[str, dict[str, object] | None] = {design.name: None for design in DESIGNS}
        evaluated = 0
        for radius in radii:
            constructed_solution = radius * direction
            signal = resolvent_graph_singular_values_fp64(
                constructed_solution,
                config=config.solver,
            )
            if _norm(signal) > config.solver.signal_norm_guard:
                continue
            records = _evaluate_spectrum_designs(
                signal,
                label=f"rank_accumulation:k={rank}:r={radius:.17g}",
                solver=config.solver,
                ledger=ledger,
            )
            evaluated += 1
            for design in DESIGNS:
                record = records[design.name]
                fidelity = record["fidelity"]
                assert isinstance(fidelity, dict)
                candidate = {
                    "constructed_solution_radius": float(radius),
                    "input_frobenius": _norm(signal),
                    "gate_weight": record["gate_weight"],
                    "output_to_input_amplitude": fidelity["output_to_input_amplitude"],
                    "computed_residual_amplitude_interval": fidelity[
                        "computed_residual_amplitude_interval"
                    ],
                    "best_scalar_departure": fidelity["best_scalar_multiple"]["relative_residual"],
                    "modal_gain_summary": fidelity["modal_gains"],
                    "residual_to_p15_threshold_ratio": record["diagnostics"][
                        "residual_to_p15_threshold_ratio"
                    ],
                }
                current = maxima[design.name]
                if current is None or float(candidate["output_to_input_amplitude"]) > float(
                    current["output_to_input_amplitude"]
                ):
                    maxima[design.name] = candidate
        if evaluated == 0 or any(value is None for value in maxima.values()):
            raise RuntimeError(f"rank-accumulation grid retained no case at rank {rank}")
        by_rank.append(
            {
                "rank": rank,
                "mode_ratio": RANK_ACCUMULATION_MODE_RATIO,
                "evaluated_radius_count": evaluated,
                "maximum_sampled_amplitude_by_design": maxima,
            }
        )
    return {
        "status": (
            "sampled adversarial diagnostic; growth across ranks is not a lower-bound "
            "theorem and the exact pointwise sector remains authoritative"
        ),
        "construction": (
            "one leading solution singular mode and k-1 modes at ratio 41/10000, "
            "normalized to each sampled solution radius and mapped forward through the graph"
        ),
        "radius_count": int(radii.size),
        "radius_range": [float(np.min(radii)), float(np.max(radii))],
        "by_rank": by_rank,
    }


def _unsafe_scalar_derivative(
    normalized_value: float,
    *,
    gate_weight: float,
    divisor: float,
    solver: EquivariantResolventSolverConfig,
) -> dict[str, float]:
    epsilon = solver.epsilon
    root = epsilon * normalized_value / (1.0 - normalized_value)
    _response, derivative = jordan_response_and_derivative_fp64(
        np.asarray([normalized_value], dtype=np.float64)
    )
    jordan_derivative = float(derivative[0]) * (1.0 - normalized_value) ** 2 / epsilon
    repair_derivative = radial_deficit_majorant_fp64(root / epsilon) / epsilon
    shifted_derivative = jordan_derivative + repair_derivative + solver.shunt
    resolvent_derivative = 1.0 / (1.0 + solver.resolvent_parameter * shifted_derivative)
    yosida_derivative = shifted_derivative * resolvent_derivative
    gated_derivative = (
        1.0 - gate_weight
    ) * yosida_derivative / divisor + gate_weight * jordan_derivative * resolvent_derivative
    return {
        "normalized_root": normalized_value,
        "root": root,
        "jordan_derivative_at_root": jordan_derivative,
        "radial_repair_derivative_at_root": repair_derivative,
        "shifted_graph_derivative": shifted_derivative,
        "resolvent_derivative": resolvent_derivative,
        "yosida_derivative": yosida_derivative,
        "gated_interface_derivative": gated_derivative,
    }


def _negative_controls(
    config: P17StudyConfig,
    ledger: _CallLedger,
) -> dict[str, object]:
    undersized_q0 = UNDERSIZED_GATE_INNER_RADIUS**2
    undersized_q1 = UNDERSIZED_GATE_OUTER_RADIUS**2
    undersized_cap = Fraction(1, 8)
    normalized = float(UNSAFE_BAND_LEFT)
    exact_root = config.solver.epsilon * normalized / (1.0 - normalized)
    constructed_solution = np.asarray([exact_root], dtype=np.float64)
    signal = resolvent_graph_singular_values_fp64(
        constructed_solution,
        config=config.solver,
    )
    solved = solve_resolvent_singular_values(signal, config.solver)
    if not solved.diagnostics.certified or solved.output_singular_values is None:
        raise RuntimeError("undersized-gate scalar witness failed closed")
    ledger.add("negative_control:undersized_gate_scalar", 1, solved.diagnostics)
    input_square = _norm(signal) ** 2
    if input_square <= float(undersized_q0):
        undersized_weight = 0.0
    elif input_square >= float(undersized_q1):
        undersized_weight = float(undersized_cap)
    else:
        coordinate = (input_square - float(undersized_q0)) / float(undersized_q1 - undersized_q0)
        undersized_weight = (
            float(undersized_cap) * coordinate**3 * (10.0 + coordinate * (-15.0 + 6.0 * coordinate))
        )
    locked_weight = quintic_gate_weight_fp64(input_square, FULL_STEP_DESIGN)
    undersized = _unsafe_scalar_derivative(
        normalized,
        gate_weight=undersized_weight,
        divisor=float(FULL_STEP_DESIGN.passive_divisor),
        solver=config.solver,
    )
    locked = _unsafe_scalar_derivative(
        normalized,
        gate_weight=locked_weight,
        divisor=float(FULL_STEP_DESIGN.passive_divisor),
        solver=config.solver,
    )
    return {
        "undersized_passive_region": {
            "status": (
                "negative derivative; this rejects an incremental/passivity claim for the "
                "undersized gate, not the locked pointwise PL theorem"
            ),
            "gate_q0": _fraction(undersized_q0),
            "gate_q1": _fraction(undersized_q1),
            "gate_ceiling": _fraction(undersized_cap),
            "constructed_input": float(signal[0]),
            "constructed_input_squared": float(signal[0] ** 2),
            "claimed_exact_input_bracket": [19.0 / 10_000.0, 1.0 / 500.0],
            "on_gate_plateau": undersized_weight == float(undersized_cap),
            "gate_weight": undersized_weight,
            "derivative_diagnostic": undersized,
            "solver_diagnostics": _diagnostics_record(solved.diagnostics),
        },
        "locked_full_step_gate_at_same_scalar_witness": {
            "gate_weight": locked_weight,
            "gate_off_below_q0": locked_weight == 0.0,
            "derivative_diagnostic": locked,
            "interpretation": (
                "the locked gate suppresses the unsafe shape component in this tiny-signal "
                "region; this scalar diagnostic alone is not a full-matrix derivative theorem"
            ),
        },
        "rank_one_fidelity_control": {
            "interpretation": (
                "one positive singular mode is necessarily collinear with its input and "
                "cannot witness nontrivial spectral shaping"
            ),
            "best_scalar_departure": 0.0,
        },
    }


def _exact_design_certificates() -> dict[str, object]:
    records: dict[str, object] = {}
    for design in DESIGNS:
        audit = audit_shape_preserving_resolvent(design)
        sector = pointwise_gain_sector(design)
        records[design.name] = {
            "certified": audit.certified,
            "claim": (
                "dimension-uniform full-matrix origin-centred pointwise sector; not an "
                "incremental Jacobian sector"
            ),
            "passive_divisor": _fraction(design.passive_divisor),
            "gate": {
                "q0": _fraction(LOCKED_GATE_Q0),
                "q1": _fraction(LOCKED_GATE_Q1),
                "ceiling": _fraction(design.cap),
            },
            "learning_rate": _fraction(design.learning_rate),
            "contraction_tau": _fraction(audit.pl_certificate.tau),
            "pointwise_sector": {
                "lower": _fraction(sector.lower),
                "upper": _fraction(sector.upper),
                "center": _fraction(sector.center),
                "radius": _fraction(sector.radius),
                "condition_ratio": _fraction(sector.condition_ratio),
            },
            "exact_checks": {name: value for name, value in audit.checks},
            "pl_lmi_certified": audit.pl_audit.certified,
        }
    return records


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


def _provenance(config: P17StudyConfig) -> dict[str, object]:
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


def run_study(config: P17StudyConfig | None = None) -> dict[str, object]:
    """Run the locked exact audits and all declared numerical diagnostics."""

    selected = P17StudyConfig() if config is None else config
    ledger = _CallLedger()
    exact = _exact_design_certificates()
    if not all(bool(record["certified"]) for record in exact.values()):  # type: ignore[union-attr]
        raise RuntimeError("a locked P17 exact design audit failed")

    canonical = _canonical_cases(selected.solver, ledger)
    grid = _run_declared_spectrum_grid(selected, ledger)
    operating_annulus = _run_operating_annulus_grid(selected, ledger)
    realistic = _run_realistic_spectra(selected, ledger)
    rank_accumulation = _run_rank_accumulation(selected, ledger)
    negative = _negative_controls(selected, ledger)

    canonical_designs = canonical["designs"]
    assert isinstance(canonical_designs, dict)
    canonical_passes = {
        name: bool(record["fidelity"]["meaningful_fidelity_gate_passes"])
        for name, record in canonical_designs.items()
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": {
            "exact_theorem": (
                "the separately replayed pointwise sectors cover every finite rectangular "
                "matrix shape and imply the locked global smooth-PL trajectory results"
            ),
            "numerical_study": (
                "deterministic FP64 falsification/fidelity evidence; sampled extrema are not "
                "global certificates and computed residuals are not directed-rounding bounds"
            ),
            "incremental_warning": (
                "the radial gate derivative adds a rank-one term, so this is not incremental "
                "passivity or an incremental sector"
            ),
            "large_shape_caveat": (
                "large-shape cases use complete singular spectra but omit dense SVD, "
                "reconstruction, accelerator timing, and memory traffic"
            ),
        },
        "config": {
            **asdict(selected),
            "solver": asdict(selected.solver),
            "normalization": "S/(||S||_F+epsilon)",
            "epsilon": selected.solver.epsilon,
            "jordan_coefficients": ["6889/2000", "-191/40", "4063/2000"],
            "jordan_iteration_count": STAGES,
            "p15_stopping_rule": "||r||_F <= ||s||_F/250 + rbar_fp64",
            "deployed_yosida_output": "(s-u_hat)/lambda",
            "frozen_p16_thresholds": {
                "best_scalar_departure": MEANINGFUL_SCALAR_RESIDUAL_THRESHOLD,
                "upstream_shaping_retention": MEANINGFUL_SHAPING_RETENTION_THRESHOLD,
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
                "FP64 five-stage spectral counterpart before aspect scaling and without BF16 casts"
            ),
        },
        "exact_design_certificates": exact,
        "canonical_diag_3_4": canonical,
        "canonical_fidelity_gate": {
            "passes_by_design": canonical_passes,
            "all_locked_designs_pass": all(canonical_passes.values()),
            "thresholds_unchanged_from_p16": True,
        },
        "declared_dense_two_mode_grid": grid,
        "declared_primary_operating_annulus_grid": operating_annulus,
        "realistic_complete_spectrum_cases": realistic,
        "rank_accumulation_diagnostic": rank_accumulation,
        "negative_controls": negative,
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
    parser.add_argument("--rank-radius-points", type=int, default=13)
    parser.add_argument("--skip-realistic-spectrum-cases", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    config = P17StudyConfig(
        seed=arguments.seed,
        spectrum_grid_points=arguments.spectrum_grid_points,
        operating_annulus_grid_points=arguments.operating_annulus_grid_points,
        rank_radius_points=arguments.rank_radius_points,
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
