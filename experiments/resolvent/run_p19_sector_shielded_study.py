#!/usr/bin/env python3
"""Deterministic P19 sector-shield and inexact-resolvent diagnostics.

P19 wraps any finite approximate operator candidate in the certified P18 disk

``D_S = {U : ||U-(1143/2048)S||_F <= (893/2048)||S||_F}``.

The exact shield theorem and the exact smooth--PL certificate are separate
from this script.  Here we exercise the inward-margin binary64 implementation,
show that it is normally inactive on the guarded P16/P18 reference solver,
replay the frozen P18 fidelity gates, and deliberately corrupt candidates to
exercise containment.  Every P16 call records its computed P15 graph residual
as *fidelity* evidence; solver accuracy is not needed for shielded stability.

The annulus extrema are deterministic sampled diagnostics, not global claims.
No result below is an IEEE-754 proof for a full optimizer implementation.
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
)
from passive_muon.floored_certificate import JORDAN_DERIVATIVE_UPPER
from passive_muon.sector_projected_resolvent import (
    evaluate_sector_projected_resolvent_fp64,
    evaluate_sector_projected_singular_values_fp64,
)
from passive_muon.sector_projected_resolvent_certificate import (
    PARETO_CERTIFICATES,
    audit_pareto_certificate,
)
from passive_muon.sector_shielded_resolvent import (
    LOCKED_SHIELD_CENTER,
    LOCKED_SHIELD_RADIUS,
    SectorShieldFailure,
    shield_sector_candidate_fp64,
)
from passive_muon.upstream_momentum import (
    PINNED_KELLER_JORDAN_MUON_REVISION,
    PINNED_MUON_PY_SHA256,
)

FloatArray = NDArray[np.float64]

SCHEMA_VERSION: Final = "passive-muon-p19-sector-shielded-study-v1"
SEED: Final = 20_260_905
STAGES: Final = 5

MAXIMUM_STEP: Final = Fraction(1, 83)
FASTER_RATE_STEP: Final = Fraction(1, 120)
OPERATING_POINTS: Final = (MAXIMUM_STEP, FASTER_RATE_STEP)

# Frozen P16/P18 fidelity and scale-sensitive gates; P19 does not retune them.
MEANINGFUL_SCALAR_RESIDUAL_THRESHOLD: Final = 1.0e-3
MEANINGFUL_SHAPING_RETENTION_THRESHOLD: Final = 1.0e-1
INFORMATIVE_UPSTREAM_SHAPING_THRESHOLD: Final = 1.0e-2
ANNULUS_UPSTREAM_AMPLITUDE_LOWER: Final = Fraction(1, 4)
ANNULUS_UPSTREAM_AMPLITUDE_UPPER: Final = Fraction(8)
EFFECTIVE_UPDATE_LOWER: Final = Fraction(1, 1_000)
EFFECTIVE_UPDATE_UPPER: Final = Fraction(1, 80)

OPERATING_ANNULUS_INNER_RADIUS: Final = Fraction(3, 4)
OPERATING_ANNULUS_OUTER_RADIUS: Final = Fraction(25)
OPERATING_ANNULUS_STRICT_RELATIVE_TOLERANCE: Final = 2.0**-44
OPERATING_ANNULUS_STRICT_ABSOLUTE_TOLERANCE: Final = 2.0**-50
RANK_ACCUMULATION_MODE_RATIO: Final = 41.0 / 10_000.0

ROOT = Path(__file__).resolve().parents[2]
CANONICAL_TARGET = ROOT / "results" / "summaries" / "p19_sector_shielded_study.json"
SOURCE_PATHS = (
    Path("experiments/resolvent/run_p19_sector_shielded_study.py"),
    Path("src/passive_muon/equivariant_resolvent_solver.py"),
    Path("src/passive_muon/sector_projected_resolvent.py"),
    Path("src/passive_muon/sector_projected_resolvent_certificate.py"),
    Path("src/passive_muon/sector_shielded_resolvent.py"),
    Path("src/passive_muon/shape_preserving_resolvent.py"),
    Path("src/passive_muon/upstream_momentum.py"),
    Path("pyproject.toml"),
    Path("uv.lock"),
)


@dataclass(frozen=True)
class P19StudyConfig:
    """Complete deterministic configuration of the P19 numerical study."""

    seed: int = SEED
    operating_annulus_grid_points: int = 33
    inward_ulps: int = 32
    solver: EquivariantResolventSolverConfig = field(
        default_factory=EquivariantResolventSolverConfig
    )

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if (
            not isinstance(self.operating_annulus_grid_points, int)
            or isinstance(self.operating_annulus_grid_points, bool)
            or self.operating_annulus_grid_points < 5
        ):
            raise ValueError("operating_annulus_grid_points must be an integer at least five")
        if (
            not isinstance(self.inward_ulps, int)
            or isinstance(self.inward_ulps, bool)
            or self.inward_ulps < 1
        ):
            raise ValueError("inward_ulps must be a positive integer")
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


def _best_scalar_departure(signal: FloatArray, output: FloatArray) -> float:
    source = np.asarray(signal, dtype=np.float64).ravel()
    target = np.asarray(output, dtype=np.float64).ravel()
    square = float(np.dot(source, source))
    output_norm = _norm(target)
    if square == 0.0 or output_norm == 0.0:
        return 0.0 if output_norm == 0.0 else 1.0
    coefficient = float(np.dot(source, target)) / square
    return _norm(target - coefficient * source) / output_norm


def _upstream_jordan_singular_values(signal: FloatArray, epsilon: float) -> FloatArray:
    signal_norm = _norm(signal)
    if signal_norm == 0.0:
        return np.zeros_like(signal)
    normalized = np.asarray(signal / (signal_norm + epsilon), dtype=np.float64)
    response, _derivative = jordan_response_and_derivative_fp64(normalized)
    return response


def _fidelity_record(
    signal: FloatArray,
    output: FloatArray,
    upstream: FloatArray,
) -> dict[str, object]:
    signal_norm = _norm(signal)
    output_norm = _norm(output)
    upstream_norm = _norm(upstream)
    departure = _best_scalar_departure(signal, output)
    upstream_departure = _best_scalar_departure(signal, upstream)
    retention = departure / upstream_departure if upstream_departure > 0.0 else None
    input_amplitude = output_norm / signal_norm if signal_norm > 0.0 else 0.0
    upstream_amplitude = output_norm / upstream_norm if upstream_norm > 0.0 else None
    informative = upstream_departure >= INFORMATIVE_UPSTREAM_SHAPING_THRESHOLD
    fidelity_passes = (
        informative
        and departure >= MEANINGFUL_SCALAR_RESIDUAL_THRESHOLD
        and retention is not None
        and retention >= MEANINGFUL_SHAPING_RETENTION_THRESHOLD
    )
    effective = {
        str(step): {
            "value": float(step) * input_amplitude,
            "gate_passes": (
                float(EFFECTIVE_UPDATE_LOWER)
                <= float(step) * input_amplitude
                <= float(EFFECTIVE_UPDATE_UPPER)
            ),
        }
        for step in OPERATING_POINTS
    }
    return {
        "input_frobenius": signal_norm,
        "output_frobenius": output_norm,
        "upstream_output_frobenius": upstream_norm,
        "output_to_input_amplitude": input_amplitude,
        "output_to_upstream_amplitude": upstream_amplitude,
        "best_scalar_departure": departure,
        "upstream_best_scalar_departure": upstream_departure,
        "shaping_retention_fraction": retention,
        "informative_upstream_shaping": informative,
        "meaningful_fidelity_gate_passes": fidelity_passes,
        "annulus_upstream_amplitude_gate_passes": (
            upstream_amplitude is not None
            and float(ANNULUS_UPSTREAM_AMPLITUDE_LOWER)
            <= upstream_amplitude
            <= float(ANNULUS_UPSTREAM_AMPLITUDE_UPPER)
        ),
        "effective_updates_per_input": effective,
    }


def _diagnostics_record(diagnostics: SolverDiagnostics) -> dict[str, object]:
    threshold = diagnostics.p15_threshold
    return {
        "status": diagnostics.status.value,
        "certified": diagnostics.certified,
        "iterations": diagnostics.iterations,
        "backtracks": diagnostics.backtracks,
        "actual_graph_residual_norm": diagnostics.residual_norm,
        "p15_threshold": threshold,
        "residual_to_p15_threshold_ratio": (
            diagnostics.residual_norm / threshold if threshold > 0.0 else 0.0
        ),
    }


def _computed_residual_error_allowance(
    diagnostics: SolverDiagnostics,
    gate_weight: float,
    solver: EquivariantResolventSolverConfig,
) -> dict[str, object]:
    """Record what the computed P15 graph residual does and does not bound.

    In exact real arithmetic the graph residual gives ``||u_hat-J(s)||<=r/2``
    and ``||Y_hat-Y||<=500r``.  The global Jordan derivative bound then gives
    a deliberately loose pre-projection shape allowance.  The P18 ray
    projection is not a Euclidean metric projection and may be nonsmooth, so
    no final P18-candidate allowance is inferred from these quantities alone.
    The final P19 Euclidean shield *is* nonexpansive for fixed ``S`` and hence
    preserves any independently established candidate-error bound.
    """

    residual = diagnostics.residual_norm
    solution = 0.5 * residual
    yosida = 500.0 * residual
    raw_shape = float(JORDAN_DERIVATIVE_UPPER) * solution / solver.epsilon
    unprojected_blend = (1.0 - gate_weight) * yosida / 1_024.0 + gate_weight * raw_shape
    return {
        "computed_graph_residual": residual,
        "exact_real_solution_error_allowance": solution,
        "exact_real_yosida_output_error_allowance": yosida,
        "coarse_raw_jordan_output_error_allowance": raw_shape,
        "coarse_unprojected_blend_error_allowance": unprojected_blend,
        "final_ray_projected_p18_candidate_error_allowance": None,
        "why_final_allowance_is_not_claimed": (
            "the P18 ray projection is not the Euclidean disk projection and the loose "
            "P15 threshold alone does not bound its nonlinear scale change"
        ),
        "final_shield_contract": (
            "for fixed S, the exact Euclidean shield is nonexpansive in its candidate; "
            "it preserves but does not create an independently proved P18 error bound"
        ),
    }


class _SolverLedger:
    """Record computed graph residuals as fidelity evidence, never as safety."""

    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    def add(
        self,
        label: str,
        diagnostics: SolverDiagnostics,
        gate_weight: float,
        solver: EquivariantResolventSolverConfig,
    ) -> None:
        if not diagnostics.certified or diagnostics.residual_norm > diagnostics.p15_threshold:
            raise RuntimeError(f"guarded reference solve failed for {label!r}")
        self.records.append(
            {
                "label": label,
                **_diagnostics_record(diagnostics),
                "error_allowance": _computed_residual_error_allowance(
                    diagnostics,
                    gate_weight,
                    solver,
                ),
            }
        )

    def summary(self) -> dict[str, object]:
        if not self.records:
            raise RuntimeError("study executed no guarded solver calls")
        ratios = [float(record["residual_to_p15_threshold_ratio"]) for record in self.records]
        iterations = [int(record["iterations"]) for record in self.records]
        return {
            "role": (
                "computed P15 graph residuals quantify solver/root/Y fidelity ingredients; "
                "they are not sufficient alone for final nonlinear P18 fidelity, while the "
                "sector shield independently supplies the P19 stability interface"
            ),
            "call_count": len(self.records),
            "all_calls_computed_residual_certified": True,
            "status_counts": dict(
                sorted(Counter(str(record["status"]) for record in self.records).items())
            ),
            "worst_residual_to_p15_threshold_ratio": max(ratios),
            "maximum_actual_graph_residual": max(
                float(record["actual_graph_residual_norm"]) for record in self.records
            ),
            "minimum_iterations": min(iterations),
            "maximum_iterations": max(iterations),
            "maximum_backtracks": max(int(record["backtracks"]) for record in self.records),
            "maximum_exact_real_solution_error_allowance": max(
                float(record["error_allowance"]["exact_real_solution_error_allowance"])
                for record in self.records
            ),
            "maximum_exact_real_yosida_output_error_allowance": max(
                float(record["error_allowance"]["exact_real_yosida_output_error_allowance"])
                for record in self.records
            ),
            "maximum_coarse_raw_jordan_output_error_allowance": max(
                float(record["error_allowance"]["coarse_raw_jordan_output_error_allowance"])
                for record in self.records
            ),
            "final_p18_candidate_error_allowance_from_residual": None,
            "limitation": (
                "the nonlinear P18 ray projection prevents promoting these loose branch "
                "allowances into a final-candidate or rigorous annulus-fidelity enclosure"
            ),
        }


def _shield_record(signal: FloatArray, candidate: FloatArray, inward_ulps: int) -> object:
    return shield_sector_candidate_fp64(
        np.ascontiguousarray(signal, dtype=np.float64),
        np.ascontiguousarray(candidate, dtype=np.float64),
        inward_ulps=inward_ulps,
    )


def _shield_diagnostics_record(result: object) -> dict[str, object]:
    diagnostics = result.diagnostics  # type: ignore[attr-defined]

    def finite_or_none(value: float) -> float | None:
        return value if math.isfinite(value) else None

    return {
        "active": diagnostics.active,
        "fail_closed": diagnostics.fail_closed,
        "reason": diagnostics.reason,
        "candidate_inside": diagnostics.candidate_inside,
        "used_interior_fallback": diagnostics.used_interior_fallback,
        "inward_radius": diagnostics.inward_radius,
        "signal_scale": finite_or_none(diagnostics.signal_scale),
        "signal_scaled_norm": finite_or_none(diagnostics.signal_scaled_norm),
        "displacement_scale": finite_or_none(diagnostics.displacement_scale),
        "displacement_scaled_norm": finite_or_none(diagnostics.displacement_scaled_norm),
        "exact_sector_margin": str(diagnostics.exact_sector_margin),
        "exact_sector_certified": diagnostics.exact_sector_margin >= 0,
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


def _canonical_case(config: P19StudyConfig, ledger: _SolverLedger) -> dict[str, object]:
    signal = np.diag(np.asarray([3.0, 4.0], dtype=np.float64))
    evaluated = evaluate_sector_projected_resolvent_fp64(signal, solver_config=config.solver)
    diagnostics = evaluated.resolvent_result.diagnostics
    ledger.add("canonical_diag_3_4", diagnostics, evaluated.gate_value, config.solver)
    shielded = _shield_record(signal, evaluated.output, config.inward_ulps)
    upstream_values = _upstream_jordan_singular_values(
        np.asarray([3.0, 4.0], dtype=np.float64),
        config.solver.epsilon,
    )
    upstream = np.diag(upstream_values)
    output = np.asarray(shielded.output, dtype=np.float64)
    difference = _norm(output - evaluated.output)
    return {
        "input": "diag(3,4)",
        "input_sha256_float64_bytes": _sha256_float64(signal),
        "solver_diagnostics": _diagnostics_record(diagnostics),
        "computed_residual_error_allowance": _computed_residual_error_allowance(
            diagnostics,
            evaluated.gate_value,
            config.solver,
        ),
        "shield_diagnostics": _shield_diagnostics_record(shielded),
        "shield_was_identity_bitwise": bool(np.array_equal(output, evaluated.output)),
        "shield_change_frobenius": difference,
        "unshielded_metrics": _fidelity_record(signal, evaluated.output, upstream),
        "shielded_metrics": _fidelity_record(signal, output, upstream),
    }


def _run_annulus(config: P19StudyConfig, ledger: _SolverLedger) -> dict[str, object]:
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
    evaluated_count = 0
    informative_count = 0
    fidelity_pass_count = 0
    upstream_amplitude_pass_count = 0
    effective_pass_counts = {str(step): 0 for step in OPERATING_POINTS}
    shield_active_count = 0
    shield_identity_count = 0
    worst_change = 0.0
    minimum_departure = math.inf
    minimum_retention = math.inf
    minimum_upstream_amplitude = math.inf
    maximum_upstream_amplitude = 0.0
    minimum_input_amplitude = math.inf
    maximum_input_amplitude = 0.0

    for radius in radii:
        for ratio in ratios:
            signal = float(radius) * _unit(np.asarray([ratio, 1.0], dtype=np.float64))
            label = f"annulus:r={radius:.17g}:a={ratio:.17g}"
            evaluated = evaluate_sector_projected_singular_values_fp64(
                signal,
                solver_config=solver,
            )
            ledger.add(label, evaluated.solver_diagnostics, evaluated.gate_value, solver)
            shielded = _shield_record(signal, evaluated.output, config.inward_ulps)
            output = np.asarray(shielded.output, dtype=np.float64)
            shield_diagnostics = shielded.diagnostics
            if shield_diagnostics.fail_closed:
                raise RuntimeError(f"finite guarded P18 candidate failed closed at {label}")
            if shield_diagnostics.exact_sector_margin < 0:
                raise RuntimeError(f"FP64 shield escaped exact float sector at {label}")
            upstream = _upstream_jordan_singular_values(signal, solver.epsilon)
            metrics = _fidelity_record(signal, output, upstream)
            evaluated_count += 1
            shield_active_count += int(shield_diagnostics.active)
            identical = bool(np.array_equal(output, evaluated.output))
            shield_identity_count += int(identical)
            worst_change = max(worst_change, _norm(output - evaluated.output))
            if bool(metrics["informative_upstream_shaping"]):
                informative_count += 1
                fidelity_pass_count += int(bool(metrics["meaningful_fidelity_gate_passes"]))
                minimum_departure = min(minimum_departure, float(metrics["best_scalar_departure"]))
                retention = metrics["shaping_retention_fraction"]
                assert retention is not None
                minimum_retention = min(minimum_retention, float(retention))
            upstream_amplitude_pass_count += int(
                bool(metrics["annulus_upstream_amplitude_gate_passes"])
            )
            upstream_amplitude = metrics["output_to_upstream_amplitude"]
            assert upstream_amplitude is not None
            minimum_upstream_amplitude = min(
                minimum_upstream_amplitude,
                float(upstream_amplitude),
            )
            maximum_upstream_amplitude = max(
                maximum_upstream_amplitude,
                float(upstream_amplitude),
            )
            input_amplitude = float(metrics["output_to_input_amplitude"])
            minimum_input_amplitude = min(minimum_input_amplitude, input_amplitude)
            maximum_input_amplitude = max(maximum_input_amplitude, input_amplitude)
            effective = metrics["effective_updates_per_input"]
            assert isinstance(effective, dict)
            for step in OPERATING_POINTS:
                effective_pass_counts[str(step)] += int(bool(effective[str(step)]["gate_passes"]))

    if shield_active_count != 0 or shield_identity_count != evaluated_count:
        raise RuntimeError("P19 shield unexpectedly modified a guarded P18 annulus candidate")
    if fidelity_pass_count != informative_count:
        raise RuntimeError("P19 shield failed a frozen P16/P18 annulus fidelity gate")
    if upstream_amplitude_pass_count != evaluated_count:
        raise RuntimeError("P19 shield failed the frozen annulus amplitude gate")
    if any(count != evaluated_count for count in effective_pass_counts.values()):
        raise RuntimeError("P19 shield failed an operating-point effective-update gate")
    return {
        "scope": (
            "post-P17 sampled operating annulus; every all-point statement below concerns "
            "only this frozen grid, not every matrix in the annulus"
        ),
        "input_radius_interval": {
            "lower": _fraction(OPERATING_ANNULUS_INNER_RADIUS),
            "upper": _fraction(OPERATING_ANNULUS_OUTER_RADIUS),
        },
        "radius_count": int(radii.size),
        "mode_ratio_count": int(ratios.size),
        "evaluated": evaluated_count,
        "informative": informative_count,
        "fidelity_passes": fidelity_pass_count,
        "all_informative_fidelity_pass": True,
        "upstream_amplitude_passes": upstream_amplitude_pass_count,
        "all_upstream_amplitudes_pass": True,
        "effective_update_passes": effective_pass_counts,
        "all_effective_update_gates_pass": True,
        "shield_active_count": shield_active_count,
        "shield_inactive_count": evaluated_count,
        "shield_bitwise_identity_count": shield_identity_count,
        "shield_normally_inactive": True,
        "worst_shield_change_frobenius": worst_change,
        "minimum_informative_best_scalar_departure": minimum_departure,
        "minimum_informative_shaping_retention": minimum_retention,
        "minimum_output_to_upstream_amplitude": minimum_upstream_amplitude,
        "maximum_output_to_upstream_amplitude": maximum_upstream_amplitude,
        "minimum_output_to_input_amplitude": minimum_input_amplitude,
        "maximum_output_to_input_amplitude": maximum_input_amplitude,
    }


def _corrupted_candidates(
    config: P19StudyConfig,
    ledger: _SolverLedger,
) -> dict[str, object]:
    signal = np.asarray([3.0, 4.0], dtype=np.float64)
    reference_evaluation = evaluate_sector_projected_singular_values_fp64(
        signal,
        solver_config=config.solver,
    )
    ledger.add(
        "corruption_reference_diag_3_4",
        reference_evaluation.solver_diagnostics,
        reference_evaluation.gate_value,
        config.solver,
    )
    reference = np.asarray(reference_evaluation.output, dtype=np.float64)
    reference_shielded = _shield_record(signal, reference, config.inward_ulps)
    if reference_shielded.diagnostics.active:
        raise RuntimeError("guarded P18 corruption reference unexpectedly activated the shield")
    signal_norm = _norm(signal)
    center = float(LOCKED_SHIELD_CENTER) * signal
    radius = float(LOCKED_SHIELD_RADIUS) * signal_norm
    directions = {
        "premature_zero": np.zeros_like(signal),
        "outward_centerline": center + 4.0 * radius * _unit(signal),
        "anti_aligned": -3.0 * signal,
        "large_skew": center + 7.0 * radius * _unit(np.asarray([-4.0, 3.0])),
    }
    records: dict[str, object] = {}
    for name, candidate in directions.items():
        unshielded_margin = radius * radius - _norm(candidate - center) ** 2
        if not unshielded_margin < 0.0:
            raise RuntimeError(f"corrupted control {name!r} unexpectedly lies in the disk")
        shielded = _shield_record(signal, candidate, config.inward_ulps)
        if shielded.diagnostics.exact_sector_margin < 0:
            raise RuntimeError(f"shielded corrupted control {name!r} escaped the sector")
        candidate_error = _norm(candidate - reference)
        shielded_error = _norm(np.asarray(shielded.output) - reference)
        if shielded_error > candidate_error:
            raise RuntimeError(f"shield expanded the declared corruption {name!r}")
        records[name] = {
            "candidate": [float(value) for value in candidate],
            "unshielded_nominal_disk_margin_fp64": unshielded_margin,
            "unshielded_violates_sector": True,
            "shielded_output": [float(value) for value in shielded.output],
            "shield_diagnostics": _shield_diagnostics_record(shielded),
            "candidate_distance_to_guarded_p18_reference": candidate_error,
            "shielded_distance_to_guarded_p18_reference": shielded_error,
            "observed_distance_nonexpansion": True,
        }

    nonfinite_records: dict[str, object] = {}
    for name, candidate in {
        "nan_candidate": np.asarray([math.nan, 1.0], dtype=np.float64),
        "positive_infinity_candidate": np.asarray([math.inf, -1.0], dtype=np.float64),
        "negative_infinity_candidate": np.asarray([-math.inf, 2.0], dtype=np.float64),
    }.items():
        shielded = _shield_record(signal, candidate, config.inward_ulps)
        if not shielded.diagnostics.fail_closed:
            raise RuntimeError(f"nonfinite control {name!r} did not fail closed")
        if not bool(np.isfinite(shielded.output).all()):
            raise RuntimeError(f"nonfinite control {name!r} leaked a nonfinite output")
        if shielded.diagnostics.exact_sector_margin < 0:
            raise RuntimeError(f"nonfinite control {name!r} escaped the sector")
        nonfinite_records[name] = {
            "shielded_output": [float(value) for value in shielded.output],
            "shield_diagnostics": _shield_diagnostics_record(shielded),
        }

    zero_signal = np.zeros(2, dtype=np.float64)
    zero_signal_records: dict[str, object] = {}
    for name, candidate in {
        "zero_candidate": np.zeros(2, dtype=np.float64),
        "finite_nonzero_candidate": np.asarray([7.0, -9.0], dtype=np.float64),
        "nonfinite_candidate": np.asarray([math.nan, math.inf], dtype=np.float64),
    }.items():
        shielded = _shield_record(zero_signal, candidate, config.inward_ulps)
        if not bool(np.equal(shielded.output, 0.0).all()):
            raise RuntimeError(f"zero-signal control {name!r} did not return zero")
        zero_signal_records[name] = {
            "shielded_output": [float(value) for value in shielded.output],
            "shield_diagnostics": _shield_diagnostics_record(shielded),
        }

    rejected_signals: dict[str, str] = {}
    for name, bad_signal in {
        "nan_signal": np.asarray([math.nan, 1.0], dtype=np.float64),
        "infinite_signal": np.asarray([math.inf, -1.0], dtype=np.float64),
    }.items():
        try:
            _shield_record(bad_signal, np.zeros(2, dtype=np.float64), config.inward_ulps)
        except SectorShieldFailure as error:
            rejected_signals[name] = str(error)
        else:
            raise RuntimeError(f"nonfinite signal control {name!r} was not rejected")
    return {
        "signal": [3.0, 4.0],
        "guarded_p18_reference": [float(value) for value in reference],
        "exact_nonexpansivity_contract": (
            "for fixed S, Euclidean projection onto D_S is one-Lipschitz in the candidate; "
            "because exact P18 lies in D_S, ||Pi(C)-T18|| <= ||C-T18||"
        ),
        "finite_corruptions": records,
        "all_finite_unshielded_candidates_violate": True,
        "all_finite_shielded_outputs_certified": True,
        "all_declared_finite_corruptions_reduce_distance_to_p18_reference": True,
        "nonfinite_candidate_controls": nonfinite_records,
        "all_nonfinite_candidates_fail_closed": True,
        "zero_signal_controls": zero_signal_records,
        "all_zero_signal_outputs_are_zero": True,
        "nonfinite_signal_rejections": rejected_signals,
        "all_nonfinite_signals_rejected": True,
        "nonfinite_signal_policy": "a nonfinite S has no well-defined D_S and is rejected",
    }


def _operating_point_records() -> list[dict[str, object]]:
    selected = {
        Fraction(1, 83): "maximum_step",
        Fraction(1, 120): "faster_certified_rate",
    }
    by_step = {spec.learning_rate: spec for spec in PARETO_CERTIFICATES}
    records: list[dict[str, object]] = []
    for step in OPERATING_POINTS:
        spec = by_step[step]
        audit = audit_pareto_certificate(spec)
        if not audit.certified:
            raise RuntimeError(f"frozen P18 certificate failed exact replay at eta={step}")
        records.append(
            {
                "role": selected[step],
                "learning_rate": _fraction(step),
                "rate_squared": _fraction(spec.rate_squared),
                "certified_lyapunov_rate_half_life": spec.half_life,
                "exact_certificate_replay_passes": True,
                "shield_interpretation": (
                    "every exposed output remains in the identical P18 pointwise sector, "
                    "so the unchanged P18 smooth-PL certificate applies"
                ),
            }
        )
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


def _provenance(config: P19StudyConfig) -> dict[str, object]:
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


def run_study(config: P19StudyConfig | None = None) -> dict[str, object]:
    """Run both exact replays and all declared deterministic diagnostics."""

    selected = P19StudyConfig() if config is None else config
    ledger = _SolverLedger()
    canonical = _canonical_case(selected, ledger)
    annulus = _run_annulus(selected, ledger)
    corruptions = _corrupted_candidates(selected, ledger)
    return {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": {
            "global_exact": (
                "the exact Euclidean projection places every finite candidate in the "
                "dimension-independent P18 disk sector; this script is diagnostic evidence"
            ),
            "stability": (
                "shielded safety needs no graph-residual accuracy; the two exact P18 "
                "smooth-PL certificates replay because the exposed sector is unchanged"
            ),
            "fidelity": (
                "computed P15 residuals and the frozen sampled annulus measure fidelity to "
                "P18, not basic stability or a global fidelity theorem"
            ),
            "nonexpansivity": (
                "for each fixed finite S, exact projection onto D_S is nonexpansive in C; "
                "since exact P18 is in D_S, shielding preserves any independently proved "
                "candidate-to-P18 error bound, but the P15 residual alone does not provide one"
            ),
            "floating_point": (
                "the inward-margin FP64 shield supplies an exact-float containment check; "
                "this is not yet a complete FP32/BF16 optimizer certificate"
            ),
        },
        "config": {
            **asdict(selected),
            "solver": asdict(selected.solver),
            "normalization": "S/(||S||_F+epsilon)",
            "epsilon": selected.solver.epsilon,
            "jordan_coefficients": ["6889/2000", "-191/40", "4063/2000"],
            "jordan_iteration_count": STAGES,
            "shield_center": _fraction(LOCKED_SHIELD_CENTER),
            "shield_radius": _fraction(LOCKED_SHIELD_RADIUS),
            "operating_points": [_fraction(step) for step in OPERATING_POINTS],
            "frozen_fidelity_thresholds": {
                "best_scalar_departure": MEANINGFUL_SCALAR_RESIDUAL_THRESHOLD,
                "upstream_shaping_retention": MEANINGFUL_SHAPING_RETENTION_THRESHOLD,
                "informative_upstream_departure": INFORMATIVE_UPSTREAM_SHAPING_THRESHOLD,
            },
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
        "operating_point_replays": _operating_point_records(),
        "canonical_diag_3_4": canonical,
        "declared_operating_annulus_grid": annulus,
        "corrupted_candidate_controls": corruptions,
        "solver_graph_residual_fidelity_summary": ledger.summary(),
        "experiment_provenance": _provenance(selected),
        "canonical_target": str(CANONICAL_TARGET.relative_to(ROOT)),
    }


def canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--operating-annulus-grid-points", type=int, default=33)
    parser.add_argument("--inward-ulps", type=int, default=32)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    config = P19StudyConfig(
        seed=arguments.seed,
        operating_annulus_grid_points=arguments.operating_annulus_grid_points,
        inward_ulps=arguments.inward_ulps,
    )
    payload = run_study(config)
    rendered = canonical_json(payload)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
