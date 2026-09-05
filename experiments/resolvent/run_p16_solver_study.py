#!/usr/bin/env python3
"""Deterministic P16 reference-solver and spectral-fidelity study.

This is numerical evidence, not a proof of the P16 equivariance theorem and
not a floating-point error certificate.  Every successful solver call is
nevertheless required to pass the *computed* P15 graph-residual test.  The
study deliberately reports near-scalar outcomes as well as nonzero modal
gain differences; a nonzero difference by itself is not called meaningful
Muon fidelity.

The dense cases exercise the complete SVD/reconstruction path.  Large
Transformer shapes are represented by their complete singular-value vectors
without allocating dense matrices.  Those cases test the reduced solver and
its iteration counts, but are not end-to-end runtime benchmarks.
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
    MatrixSolveResult,
    SingularValueSolveResult,
    SolverDiagnostics,
    jordan_response_and_derivative_fp64,
    p13_operator_fp64,
    radial_integral_fp64,
    shifted_p13_singular_operator_fp64,
    solve_resolvent_fp64,
    solve_resolvent_singular_values,
)
from passive_muon.pl_convergence import audit_pl_convergence
from passive_muon.radial_passivation_tradeoff import P11_SIGNAL_NORM_BOUND
from passive_muon.upstream_momentum import (
    PINNED_KELLER_JORDAN_MUON_REVISION,
    PINNED_MUON_PY_SHA256,
)
from passive_muon.yosida_stability import locked_yosida_pl_certificate

FloatArray = NDArray[np.float64]

SCHEMA_VERSION: Final = "passive-muon-p16-equivariant-resolvent-solver-study-v1"
SEED: Final = 20_260_904
STAGES: Final = 5
EPSILON: Final = 1.0e-7
P11_SIGNAL_GUARD: Final = float(P11_SIGNAL_NORM_BOUND)
MEANINGFUL_SCALAR_RESIDUAL_THRESHOLD: Final = 1.0e-3
MEANINGFUL_SHAPING_RETENTION_THRESHOLD: Final = 1.0e-1

# Full matrices are intentionally modest so the canonical study remains a
# portable CPU replay.  The large shape declarations below use complete
# k=min(m,n) singular spectra and do not allocate m-by-n arrays.
DEFAULT_DENSE_SHAPES: Final = ((2, 2), (3, 5), (5, 3), (4, 7), (8, 16), (16, 8))
REALISTIC_TRANSFORMER_SHAPES: Final = (
    (768, 768),
    (768, 3_072),
    (3_072, 12_288),
    (4_096, 11_008),
)

# These points were chosen before the canonical run.  The exact PL replay in
# this script tests only the *frozen P14 storage and multipliers*; rejection is
# not an instability theorem and does not rule out a newly optimized storage.
FRONTIER_CANDIDATES: Final = (
    ("lambda_1_over_800_mu_500", Fraction(1, 800), Fraction(500)),
    ("lambda_1_over_1000_mu_750", Fraction(1, 1_000), Fraction(750)),
    ("lambda_3_over_4000_mu_1000", Fraction(3, 4_000), Fraction(1_000)),
    ("lambda_1_over_2000_mu_1500", Fraction(1, 2_000), Fraction(1_500)),
    ("lambda_3_over_10000_mu_3000", Fraction(3, 10_000), Fraction(3_000)),
    ("locked_lambda_1_over_1000_mu_1000", Fraction(1, 1_000), Fraction(1_000)),
)

ROOT = Path(__file__).resolve().parents[2]
CANONICAL_TARGET = ROOT / "results" / "summaries" / "p16_solver_study.json"
SOURCE_PATHS = (
    Path("experiments/resolvent/run_p16_solver_study.py"),
    Path("src/passive_muon/additive_epsilon_deficit.py"),
    Path("src/passive_muon/equivariant_resolvent_solver.py"),
    Path("src/passive_muon/momentum_iqc.py"),
    Path("src/passive_muon/pl_convergence.py"),
    Path("src/passive_muon/radial_passivation_tradeoff.py"),
    Path("src/passive_muon/inexact_yosida_robustness.py"),
    Path("src/passive_muon/specs.py"),
    Path("src/passive_muon/structure_aware_stability.py"),
    Path("src/passive_muon/upstream_momentum.py"),
    Path("src/passive_muon/yosida_stability.py"),
    Path("tests/test_p16_solver_study.py"),
    Path("third_party/UPSTREAM_COMMITS.md"),
    Path("pyproject.toml"),
    Path("uv.lock"),
)


def _validate_shape(shape: tuple[int, int]) -> None:
    if len(shape) != 2 or any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in shape
    ):
        raise ValueError("matrix shapes must contain two positive integers")


@dataclass(frozen=True)
class SolverStudyConfig:
    """Complete deterministic configuration of the numerical study."""

    seed: int = SEED
    dense_shapes: tuple[tuple[int, int], ...] = DEFAULT_DENSE_SHAPES
    include_realistic_spectrum_cases: bool = True
    frontier_grid_points: int = 49
    solver: EquivariantResolventSolverConfig = field(
        default_factory=EquivariantResolventSolverConfig
    )

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if len(self.dense_shapes) != 6:
            raise ValueError("exactly six dense shapes are required")
        for shape in self.dense_shapes:
            _validate_shape(shape)
        if len(set(self.dense_shapes)) != len(self.dense_shapes):
            raise ValueError("dense shapes must be unique")
        if not isinstance(self.include_realistic_spectrum_cases, bool):
            raise TypeError("include_realistic_spectrum_cases must be Boolean")
        if (
            not isinstance(self.frontier_grid_points, int)
            or isinstance(self.frontier_grid_points, bool)
            or self.frontier_grid_points < 5
        ):
            raise ValueError("frontier_grid_points must be an integer at least five")
        if not isinstance(self.solver, EquivariantResolventSolverConfig):
            raise TypeError("solver must be an EquivariantResolventSolverConfig")


def _norm(values: FloatArray) -> float:
    return float(np.linalg.norm(np.asarray(values, dtype=np.float64).ravel()))


def _unit(values: FloatArray) -> FloatArray:
    norm = _norm(values)
    if norm == 0.0:
        raise ValueError("cannot normalize a zero template")
    return np.asarray(values / norm, dtype=np.float64)


def _directional_sine(left: FloatArray, right: FloatArray) -> float | None:
    """Return the sine of the acute Frobenius angle, or ``None`` at zero."""

    left_flat = np.asarray(left, dtype=np.float64).ravel()
    right_flat = np.asarray(right, dtype=np.float64).ravel()
    denominator = _norm(left_flat) * _norm(right_flat)
    if denominator == 0.0:
        return None
    cosine = float(np.dot(left_flat, right_flat)) / denominator
    cosine = max(-1.0, min(1.0, cosine))
    return math.sqrt(max(0.0, 1.0 - cosine * cosine))


def _best_scalar_record(signal: FloatArray, output: FloatArray) -> dict[str, float | None]:
    signal_flat = np.asarray(signal, dtype=np.float64).ravel()
    output_flat = np.asarray(output, dtype=np.float64).ravel()
    signal_square = float(np.dot(signal_flat, signal_flat))
    output_norm = _norm(output_flat)
    if signal_square == 0.0 or output_norm == 0.0:
        return {
            "coefficient": None,
            "residual_frobenius": 0.0 if output_norm == 0.0 else output_norm,
            "relative_residual": 0.0 if output_norm == 0.0 else 1.0,
            "directional_sine": _directional_sine(signal_flat, output_flat),
        }
    coefficient = float(np.dot(signal_flat, output_flat)) / signal_square
    residual = output_flat - coefficient * signal_flat
    residual_norm = _norm(residual)
    return {
        "coefficient": coefficient,
        "residual_frobenius": residual_norm,
        "relative_residual": residual_norm / output_norm,
        "directional_sine": _directional_sine(signal_flat, output_flat),
    }


def _relative_error(reference: FloatArray, value: FloatArray) -> float:
    denominator = _norm(reference)
    difference = _norm(np.asarray(value, dtype=np.float64) - reference)
    if denominator == 0.0:
        return 0.0 if difference == 0.0 else math.inf
    return difference / denominator


def upstream_jordan_singular_values(signal_values: FloatArray, epsilon: float) -> FloatArray:
    """Return the pre-aspect, five-stage current-plus-epsilon Jordan map."""

    norm = _norm(signal_values)
    if norm == 0.0:
        return np.zeros_like(signal_values)
    normalized = np.asarray(signal_values / (norm + epsilon), dtype=np.float64)
    response, _derivative = jordan_response_and_derivative_fp64(normalized)
    return response


def upstream_jordan_fp64(signal: FloatArray, epsilon: float = EPSILON) -> FloatArray:
    """Numerical full-matrix upstream comparator before aspect scaling."""

    matrix = np.asarray(signal, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("signal must be a matrix")
    if _norm(matrix) == 0.0:
        return np.zeros_like(matrix)
    left, singular_values, right_transpose = np.linalg.svd(matrix, full_matrices=False)
    output_values = upstream_jordan_singular_values(singular_values, epsilon)
    return np.asarray((left * output_values) @ right_transpose, dtype=np.float64)


def _singular_condition(values: FloatArray) -> float | None:
    selected = np.asarray(values, dtype=np.float64)
    if selected.size < 2 or bool(np.any(selected <= 0.0)):
        return None
    return float(np.max(selected) / np.min(selected))


def _fidelity_record(
    signal: FloatArray,
    output: FloatArray,
    upstream: FloatArray,
    *,
    signal_singular_values: FloatArray | None = None,
    output_singular_values: FloatArray | None = None,
    upstream_singular_values: FloatArray | None = None,
) -> dict[str, object]:
    best = _best_scalar_record(signal, output)
    upstream_best = _best_scalar_record(signal, upstream)
    output_norm = _norm(output)
    scalar_errors = {
        str(gain): _relative_error(output, gain * np.asarray(signal, dtype=np.float64))
        for gain in (500, 750, 1_000)
    }
    if signal_singular_values is None:
        signal_singular_values = np.linalg.svd(signal, compute_uv=False)
    if output_singular_values is None:
        output_singular_values = np.linalg.svd(output, compute_uv=False)
    if upstream_singular_values is None:
        upstream_singular_values = np.linalg.svd(upstream, compute_uv=False)
    signal_condition = _singular_condition(signal_singular_values)
    output_condition = _singular_condition(output_singular_values)
    upstream_condition = _singular_condition(upstream_singular_values)
    modal_gains: list[float | None] = [
        float(output_value / input_value) if input_value > 0.0 else None
        for input_value, output_value in zip(
            signal_singular_values,
            output_singular_values,
            strict=True,
        )
    ]
    finite_modal_gains = [value for value in modal_gains if value is not None]
    modal_gain_spread = (
        max(finite_modal_gains) - min(finite_modal_gains) if len(finite_modal_gains) >= 2 else None
    )
    best_departure = float(best["relative_residual"])
    upstream_departure = float(upstream_best["relative_residual"])
    shaping_retention = best_departure / upstream_departure if upstream_departure > 0.0 else None
    absolute_gate_passes = best_departure >= MEANINGFUL_SCALAR_RESIDUAL_THRESHOLD
    retention_gate_passes = (
        shaping_retention is not None
        and shaping_retention >= MEANINGFUL_SHAPING_RETENTION_THRESHOLD
    )
    return {
        "input_frobenius": _norm(signal),
        "output_frobenius": output_norm,
        "upstream_jordan_frobenius": _norm(upstream),
        "best_scalar_multiple": best,
        "upstream_best_scalar_multiple": upstream_best,
        "shaping_retention_fraction": shaping_retention,
        "modal_gains": modal_gains,
        "modal_gain_spread": modal_gain_spread,
        "relative_error_from_fixed_scalar_multiples": scalar_errors,
        "directional_sine_to_upstream_jordan": _directional_sine(output, upstream),
        "input_condition_number": signal_condition,
        "output_condition_number": output_condition,
        "upstream_jordan_condition_number": upstream_condition,
        "output_to_input_condition_ratio": (
            output_condition / signal_condition
            if output_condition is not None and signal_condition is not None
            else None
        ),
        "upstream_to_input_condition_ratio": (
            upstream_condition / signal_condition
            if upstream_condition is not None and signal_condition is not None
            else None
        ),
        "near_scalar_at_declared_threshold": (
            best_departure < MEANINGFUL_SCALAR_RESIDUAL_THRESHOLD
        ),
        "absolute_shaping_gate_passes": absolute_gate_passes,
        "upstream_retention_gate_passes": retention_gate_passes,
        "meaningful_fidelity_gate_passes": absolute_gate_passes and retention_gate_passes,
    }


def _diagnostics_record(diagnostics: SolverDiagnostics) -> dict[str, object]:
    return {
        "status": diagnostics.status.value,
        "certified": diagnostics.certified,
        "iterations": diagnostics.iterations,
        "backtracks": diagnostics.backtracks,
        "initial_residual_norm": diagnostics.initial_residual_norm,
        "reduced_residual_norm": diagnostics.reduced_residual_norm,
        "actual_residual_norm": diagnostics.residual_norm,
        "p15_threshold": diagnostics.p15_threshold,
        "strict_target": diagnostics.strict_target,
        "operator_evaluations": diagnostics.operator_evaluations,
        "jacobian_evaluations": diagnostics.jacobian_evaluations,
        "sherman_morrison_solves": diagnostics.sherman_morrison_solves,
        "svd_count": diagnostics.svd_count,
        "reconstruction_matmuls": diagnostics.reconstruction_matmuls,
        "minimum_abs_sherman_morrison_denominator": (
            diagnostics.minimum_abs_sherman_morrison_denominator
        ),
    }


def _cost_proxy(shape: tuple[int, int], diagnostics: SolverDiagnostics) -> dict[str, object]:
    short = min(shape)
    long = max(shape)
    # One Jordan stage forms XX^T, forms its square, then left-multiplies X.
    jordan_flops = STAGES * (4 * long * short**2 + 2 * short**3)
    reconstruction_flops = diagnostics.reconstruction_matmuls * 2 * long * short**2
    return {
        "five_stage_jordan_dense_gemm_count": 3 * STAGES,
        "five_stage_jordan_leading_flop_proxy": jordan_flops,
        "solver_svd_count": diagnostics.svd_count,
        "solver_reconstruction_matmul_count": diagnostics.reconstruction_matmuls,
        "solver_reconstruction_leading_flop_proxy": reconstruction_flops,
        "solver_scalar_newton_iterations": diagnostics.iterations,
        "solver_sherman_morrison_vector_solves": diagnostics.sherman_morrison_solves,
        "important_caveat": (
            "the SVD cost is implementation-dependent and excluded from the reconstruction "
            "flop proxy; the solve SVD plus the literal graph-check SVD and reconstruction "
            "are not claimed faster than 15 GEMMs"
        ),
    }


def _orthonormal_columns(rows: int, columns: int, rng: np.random.Generator) -> FloatArray:
    raw = rng.standard_normal((rows, columns), dtype=np.float64)
    q, r = np.linalg.qr(raw, mode="reduced")
    signs = np.where(np.diag(r) < 0.0, -1.0, 1.0)
    return np.asarray(q * signs, dtype=np.float64)


def _matrix_from_spectrum(
    shape: tuple[int, int], values: FloatArray, rng: np.random.Generator
) -> FloatArray:
    rows, columns = shape
    rank = min(shape)
    if values.shape != (rank,):
        raise ValueError("spectrum length must equal min(shape)")
    left = _orthonormal_columns(rows, rank, rng)
    right = _orthonormal_columns(columns, rank, rng)
    return np.asarray((left * values) @ right.T, dtype=np.float64)


def constructed_fidelity_witness(
    config: EquivariantResolventSolverConfig | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return ``(s,u,Y)`` for the locked unequal-mode graph witness.

    This matches the theorem artifact's exact ``u=(3/5,4/5)`` forward
    construction.  The theorem artifact uses exact rational/Arb arithmetic;
    this routine is only its FP64 study copy.
    """

    selected = EquivariantResolventSolverConfig() if config is None else config
    approximate_solution = np.asarray([3.0 / 5.0, 4.0 / 5.0], dtype=np.float64)
    output = shifted_p13_singular_operator_fp64(
        approximate_solution,
        epsilon=selected.epsilon,
        shunt=selected.shunt,
    )
    signal = approximate_solution + selected.resolvent_parameter * output
    return signal, approximate_solution, output


def constructed_modal_gap_stress_case(
    config: EquivariantResolventSolverConfig | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return an exploratory ill-conditioned case with a larger gain gap."""

    selected = EquivariantResolventSolverConfig() if config is None else config
    direction = np.asarray([1_000.0 / 250_001.0, 249_999.0 / 250_001.0])
    approximate_solution = 1.5 * direction
    output = shifted_p13_singular_operator_fp64(
        approximate_solution,
        epsilon=selected.epsilon,
        shunt=selected.shunt,
    )
    signal = approximate_solution + selected.resolvent_parameter * output
    return signal, approximate_solution, output


def build_dense_cases(config: SolverStudyConfig) -> tuple[tuple[str, FloatArray], ...]:
    """Build deterministic zero, boundary, repeated, rank-deficient, and guard cases."""

    rng = np.random.default_rng(config.seed)
    shapes = config.dense_shapes
    cases: list[tuple[str, FloatArray]] = []

    cases.append(("zero", np.zeros(shapes[0], dtype=np.float64)))

    near_rank = min(shapes[1])
    near_values = np.zeros(near_rank, dtype=np.float64)
    near_values[0] = 2.0**-40
    cases.append(("near_zero_rank_one", _matrix_from_spectrum(shapes[1], near_values, rng)))

    rank = min(shapes[2])
    repeated = np.zeros(rank, dtype=np.float64)
    repeated[: min(3, rank)] = EPSILON / math.sqrt(min(3, rank))
    cases.append(("epsilon_scale_repeated", _matrix_from_spectrum(shapes[2], repeated, rng)))

    rank = min(shapes[3])
    repeated_unit = np.full(rank, 1.0 / math.sqrt(rank), dtype=np.float64)
    cases.append(("unit_repeated", _matrix_from_spectrum(shapes[3], repeated_unit, rng)))

    rank = min(shapes[4])
    dense_values = np.geomspace(1.0, 1.0e-3, rank, dtype=np.float64)
    dense_values = _unit(dense_values)
    cases.append(("unit_log_spectrum", _matrix_from_spectrum(shapes[4], dense_values, rng)))

    rank = min(shapes[5])
    guard_values = np.geomspace(1.0, 1.0e-4, rank, dtype=np.float64)
    guard_values = 0.999 * P11_SIGNAL_GUARD * _unit(guard_values)
    cases.append(
        ("near_p11_guard_log_spectrum", _matrix_from_spectrum(shapes[5], guard_values, rng))
    )

    witness_signal, _witness_solution, _witness_output = constructed_fidelity_witness(config.solver)
    cases.append(("exact_forward_shaping_witness", np.diag(witness_signal)))
    cases.append(("balanced_canonical_input_diag_3_4", np.diag([3.0, 4.0])))
    stress_signal, _stress_solution, _stress_output = constructed_modal_gap_stress_case(
        config.solver
    )
    cases.append(("exploratory_ill_conditioned_modal_gap_stress", np.diag(stress_signal)))
    return tuple(cases)


def _matrix_case_record(
    label: str,
    signal: FloatArray,
    solver_config: EquivariantResolventSolverConfig,
) -> dict[str, object]:
    result: MatrixSolveResult = solve_resolvent_fp64(signal, solver_config)
    if not result.diagnostics.certified or result.output is None:
        raise RuntimeError(f"declared dense case {label!r} failed closed")
    upstream = upstream_jordan_fp64(signal, solver_config.epsilon)
    shifted = (
        p13_operator_fp64(result.approximate_solution, epsilon=solver_config.epsilon)
        + solver_config.shunt * result.approximate_solution
    )
    output_formula = (
        np.asarray(signal, dtype=np.float64) - result.approximate_solution
    ) / solver_config.resolvent_parameter
    output_singular_values = np.linalg.svd(result.output, compute_uv=False)
    upstream_singular_values = np.linalg.svd(upstream, compute_uv=False)
    return {
        "label": label,
        "shape": list(signal.shape),
        "input_sha256_float64_bytes": hashlib.sha256(
            np.ascontiguousarray(signal).tobytes(order="C")
        ).hexdigest(),
        "diagnostics": _diagnostics_record(result.diagnostics),
        "residual_to_threshold_ratio": (
            result.diagnostics.residual_norm / result.diagnostics.p15_threshold
            if result.diagnostics.p15_threshold > 0.0
            else 0.0
        ),
        "resolvent_output_formula_error": _norm(result.output - output_formula),
        "graph_form_B_output_error": _norm(result.output - shifted),
        "fidelity": _fidelity_record(
            signal,
            result.output,
            upstream,
            signal_singular_values=result.input_singular_values,
            output_singular_values=output_singular_values,
            upstream_singular_values=upstream_singular_values,
        ),
        "cost_proxy": _cost_proxy(signal.shape, result.diagnostics),
    }


def _realistic_spectrum(shape: tuple[int, int], index: int) -> tuple[str, FloatArray]:
    rank = min(shape)
    if index == 0:
        values = np.full(rank, 1.0 / math.sqrt(rank), dtype=np.float64)
        return "unit_flat_repeated", values
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


def _spectrum_case_record(
    shape: tuple[int, int],
    label: str,
    signal_values: FloatArray,
    solver_config: EquivariantResolventSolverConfig,
) -> dict[str, object]:
    result: SingularValueSolveResult = solve_resolvent_singular_values(signal_values, solver_config)
    if not result.diagnostics.certified or result.output_singular_values is None:
        raise RuntimeError(f"declared spectrum case {label!r} failed closed")
    upstream = upstream_jordan_singular_values(signal_values, solver_config.epsilon)
    return {
        "label": label,
        "declared_matrix_shape": list(shape),
        "representation": (
            "complete k=min(m,n) singular-value vector; no dense matrix or SVD allocated"
        ),
        "singular_value_count": int(signal_values.size),
        "input_sha256_float64_bytes": hashlib.sha256(
            np.ascontiguousarray(signal_values).tobytes(order="C")
        ).hexdigest(),
        "diagnostics": _diagnostics_record(result.diagnostics),
        "residual_to_threshold_ratio": (
            result.diagnostics.residual_norm / result.diagnostics.p15_threshold
            if result.diagnostics.p15_threshold > 0.0
            else 0.0
        ),
        "fidelity": _fidelity_record(
            signal_values,
            result.output_singular_values,
            upstream,
            signal_singular_values=signal_values,
            output_singular_values=result.output_singular_values,
            upstream_singular_values=upstream,
        ),
        "cost_proxy": {
            "solver_scalar_newton_iterations": result.diagnostics.iterations,
            "solver_sherman_morrison_vector_solves": (result.diagnostics.sherman_morrison_solves),
            "dense_cost_not_executed": True,
            "hypothetical_five_stage_jordan_dense_gemm_count": 15,
            "important_caveat": "this is not an end-to-end dense runtime measurement",
        },
    }


def _two_mode_metrics(
    solution: FloatArray,
    *,
    epsilon: float,
    resolvent_parameter: float,
    shunt: float,
) -> dict[str, float] | None:
    output = shifted_p13_singular_operator_fp64(
        solution,
        epsilon=epsilon,
        shunt=shunt,
    )
    signal = solution + resolvent_parameter * output
    signal_norm = _norm(signal)
    if signal_norm == 0.0 or signal_norm > P11_SIGNAL_GUARD:
        return None
    output_norm = _norm(output)
    cross = abs(float(signal[0] * output[1] - signal[1] * output[0]))
    scalar_residual = cross / (signal_norm * output_norm)
    upstream = upstream_jordan_singular_values(signal, epsilon)
    upstream_norm = _norm(upstream)
    upstream_cross = abs(float(signal[0] * upstream[1] - signal[1] * upstream[0]))
    upstream_shaping = upstream_cross / (signal_norm * upstream_norm)
    signal_condition = float(max(signal) / min(signal))
    output_condition = float(max(output) / min(output))
    return {
        "solution_radius": _norm(solution),
        "solution_mode_ratio": float(min(solution) / max(solution)),
        "input_frobenius": signal_norm,
        "input_condition_number": signal_condition,
        "output_condition_number": output_condition,
        "output_to_input_condition_ratio": output_condition / signal_condition,
        "condition_reduction_fraction": 1.0 - output_condition / signal_condition,
        "best_scalar_relative_residual": scalar_residual,
        "upstream_jordan_shaping": upstream_shaping,
        "shaping_retention_fraction": (
            scalar_residual / upstream_shaping if upstream_shaping > 0.0 else 0.0
        ),
        "modal_gain_low": float(output[0] / signal[0]),
        "modal_gain_high": float(output[1] / signal[1]),
    }


def _frontier_grid(grid_points: int) -> tuple[FloatArray, FloatArray]:
    radii = np.geomspace(1.0e-12, P11_SIGNAL_GUARD, grid_points, dtype=np.float64)
    radii = np.unique(
        np.concatenate((radii, np.asarray([0.475, 1.0, 1.5, 1.67, 2.075, 3.0], dtype=np.float64)))
    )
    ratios = np.geomspace(1.0e-6, 1.0, grid_points, dtype=np.float64)
    ratios = np.unique(
        np.concatenate((ratios, np.asarray([1_000.0 / 249_999.0, 0.0041], dtype=np.float64)))
    )
    return radii, ratios


def _fraction_string(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _frozen_p14_replay(lam: Fraction, shunt: Fraction) -> dict[str, object]:
    lower = shunt / (1 + lam * shunt)
    upper = 1 / lam
    center = (lower + upper) / 2
    radius = (upper - lower) / 2
    certificate = replace(
        locked_yosida_pl_certificate(),
        center_gain=center,
        residual_lipschitz=radius,
    )
    audit = audit_pl_convergence(certificate)
    return {
        "scope": (
            "exact replay with frozen P14 storage/multipliers; failure is only rejection "
            "of that sufficient certificate"
        ),
        "certified": audit.certified,
        "sector_lower": _fraction_string(lower),
        "sector_upper": _fraction_string(upper),
        "sector_center": _fraction_string(center),
        "sector_radius": _fraction_string(radius),
        "negative_lmi_leading_minors": [
            _fraction_string(value) for value in audit.negative_lmi_leading_minors
        ],
    }


def run_stability_fidelity_frontier(grid_points: int) -> dict[str, object]:
    """Run the declared two-mode sampled frontier plus exact frozen-LMI replays."""

    radii, ratios = _frontier_grid(grid_points)
    candidate_records: list[dict[str, object]] = []
    for name, lam_exact, shunt_exact in FRONTIER_CANDIDATES:
        lam = float(lam_exact)
        shunt = float(shunt_exact)
        maximum_scalar: dict[str, float] | None = None
        maximum_condition_reduction: dict[str, float] | None = None
        maximum_retention: dict[str, float] | None = None
        evaluated = 0
        for radius in radii:
            for ratio in ratios:
                direction = np.asarray([ratio, 1.0], dtype=np.float64)
                direction = _unit(direction)
                metrics = _two_mode_metrics(
                    radius * direction,
                    epsilon=EPSILON,
                    resolvent_parameter=lam,
                    shunt=shunt,
                )
                if metrics is None:
                    continue
                evaluated += 1
                if (
                    maximum_scalar is None
                    or metrics["best_scalar_relative_residual"]
                    > maximum_scalar["best_scalar_relative_residual"]
                ):
                    maximum_scalar = metrics
                if (
                    maximum_condition_reduction is None
                    or metrics["condition_reduction_fraction"]
                    > maximum_condition_reduction["condition_reduction_fraction"]
                ):
                    maximum_condition_reduction = metrics
                if metrics["upstream_jordan_shaping"] >= 0.01 and (
                    maximum_retention is None
                    or metrics["shaping_retention_fraction"]
                    > maximum_retention["shaping_retention_fraction"]
                ):
                    maximum_retention = metrics
        if maximum_scalar is None or maximum_condition_reduction is None:
            raise RuntimeError("frontier grid contained no input inside the P11 guard")
        candidate_records.append(
            {
                "name": name,
                "resolvent_parameter": _fraction_string(lam_exact),
                "shunt": _fraction_string(shunt_exact),
                "evaluated_point_count": evaluated,
                "maximum_best_scalar_relative_residual": maximum_scalar,
                "maximum_condition_reduction": maximum_condition_reduction,
                "maximum_shaping_retention_with_upstream_shaping_at_least_0.01": (
                    maximum_retention
                ),
                "frozen_p14_certificate": _frozen_p14_replay(lam_exact, shunt_exact),
                "near_scalar_on_declared_grid": (
                    maximum_scalar["best_scalar_relative_residual"]
                    < MEANINGFUL_SCALAR_RESIDUAL_THRESHOLD
                ),
                "retains_declared_fraction_of_upstream_shaping": (
                    maximum_retention is not None
                    and maximum_retention["shaping_retention_fraction"]
                    >= MEANINGFUL_SHAPING_RETENTION_THRESHOLD
                ),
            }
        )
    return {
        "status": "sampled diagnostic; not a global fidelity or stability certificate",
        "grid_parameterization": (
            "solutions u=r*(a,1)/sqrt(1+a^2); inputs constructed as s=u+lambda*B(u)"
        ),
        "radius_count": int(radii.size),
        "mode_ratio_count": int(ratios.size),
        "radius_range": [float(np.min(radii)), float(np.max(radii))],
        "mode_ratio_range": [float(np.min(ratios)), float(np.max(ratios))],
        "p11_input_guard": P11_SIGNAL_GUARD,
        "declared_near_scalar_threshold": MEANINGFUL_SCALAR_RESIDUAL_THRESHOLD,
        "declared_shaping_retention_threshold": MEANINGFUL_SHAPING_RETENTION_THRESHOLD,
        "candidates": candidate_records,
    }


def _failure_controls(config: SolverStudyConfig) -> dict[str, object]:
    probe = np.asarray([0.5, 1.0], dtype=np.float64)
    no_iterations = solve_resolvent_singular_values(
        probe,
        replace(config.solver, maximum_iterations=0),
    )

    def rejected(values: FloatArray) -> dict[str, object]:
        try:
            solve_resolvent_singular_values(values, config.solver)
        except (TypeError, ValueError, FloatingPointError) as error:
            return {"rejected": True, "exception": type(error).__name__, "message": str(error)}
        return {"rejected": False, "exception": None, "message": None}

    graph_candidate = np.zeros_like(probe)
    residual = probe.copy()
    required_output = (probe - graph_candidate) / config.solver.resolvent_parameter
    wrong_output = shifted_p13_singular_operator_fp64(
        graph_candidate,
        epsilon=config.solver.epsilon,
        shunt=config.solver.shunt,
    )

    witness_signal, witness_solution, _output = constructed_fidelity_witness(config.solver)
    radial_only_gain = config.solver.shunt + radial_integral_fp64(
        _norm(witness_solution) / config.solver.epsilon
    ) / _norm(witness_solution)
    radial_only_output = radial_only_gain * witness_solution
    radial_only_signal = witness_solution + config.solver.resolvent_parameter * radial_only_output
    radial_only_scalar = _best_scalar_record(radial_only_signal, radial_only_output)

    return {
        "zero_iteration_cap": {
            "status": no_iterations.diagnostics.status.value,
            "certified": no_iterations.diagnostics.certified,
            "residual_norm": no_iterations.diagnostics.residual_norm,
            "p15_threshold": no_iterations.diagnostics.p15_threshold,
        },
        "over_guard_input": rejected(
            np.asarray([config.solver.signal_norm_guard * (1.0 + 1.0e-6)], dtype=np.float64)
        ),
        "nonfinite_input": rejected(np.asarray([math.nan], dtype=np.float64)),
        "wrong_B_of_candidate_output": {
            "candidate": "u_hat=0 at s=(1/2,1)",
            "graph_residual_frobenius": _norm(residual),
            "required_output_minus_B_candidate_frobenius": _norm(required_output - wrong_output),
            "identity": "(s-u_hat)/lambda-B(u_hat)=r/lambda",
            "passes_p15": _norm(residual)
            <= config.solver.p15_relative_tolerance * _norm(probe)
            + config.solver.p15_absolute_tolerance,
        },
        "radial_only_no_jordan_control": {
            "best_scalar_relative_residual": radial_only_scalar["relative_residual"],
            "interpretation": (
                "removing the Jordan term leaves an exactly radial graph and cannot be a "
                "two-mode fidelity witness"
            ),
            "locked_witness_input_frobenius": _norm(witness_signal),
        },
    }


def _summary(
    dense_cases: list[dict[str, object]], spectrum_cases: list[dict[str, object]]
) -> dict[str, object]:
    cases = dense_cases + spectrum_cases
    diagnostics = [case["diagnostics"] for case in cases]
    iterations = [int(item["iterations"]) for item in diagnostics]  # type: ignore[index]
    backtracks = [int(item["backtracks"]) for item in diagnostics]  # type: ignore[index]
    residual_ratios = [float(case["residual_to_threshold_ratio"]) for case in cases]
    status_counts = Counter(str(item["status"]) for item in diagnostics)  # type: ignore[index]
    return {
        "case_count": len(cases),
        "dense_case_count": len(dense_cases),
        "spectrum_only_case_count": len(spectrum_cases),
        "certified_call_count": sum(bool(item["certified"]) for item in diagnostics),  # type: ignore[index]
        "failed_call_count": sum(not bool(item["certified"]) for item in diagnostics),  # type: ignore[index]
        "status_counts": dict(sorted(status_counts.items())),
        "iteration_histogram": {
            str(key): value for key, value in sorted(Counter(iterations).items())
        },
        "minimum_iterations": min(iterations),
        "maximum_iterations": max(iterations),
        "mean_iterations": float(np.mean(iterations)),
        "maximum_backtracks": max(backtracks),
        "worst_residual_to_p15_threshold_ratio": max(residual_ratios),
        "maximum_actual_residual": max(float(item["actual_residual_norm"]) for item in diagnostics),  # type: ignore[index]
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


def _provenance(config: SolverStudyConfig) -> dict[str, object]:
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


def run_study(config: SolverStudyConfig | None = None) -> dict[str, object]:
    """Run all dense, shape-level, failure-control, and frontier cases."""

    selected = SolverStudyConfig() if config is None else config
    dense_cases = [
        _matrix_case_record(label, signal, selected.solver)
        for label, signal in build_dense_cases(selected)
    ]
    spectrum_cases: list[dict[str, object]] = []
    if selected.include_realistic_spectrum_cases:
        for index, shape in enumerate(REALISTIC_TRANSFORMER_SHAPES):
            label, values = _realistic_spectrum(shape, index)
            spectrum_cases.append(_spectrum_case_record(shape, label, values, selected.solver))

    frontier = run_stability_fidelity_frontier(selected.frontier_grid_points)
    canonical = next(
        case for case in dense_cases if case["label"] == "balanced_canonical_input_diag_3_4"
    )
    canonical_fidelity = canonical["fidelity"]
    assert isinstance(canonical_fidelity, dict)
    frontier_candidates = frontier["candidates"]
    assert isinstance(frontier_candidates, list)
    joint_frontier_passes = [
        candidate["name"]
        for candidate in frontier_candidates
        if candidate["frozen_p14_certificate"]["certified"]
        and not candidate["near_scalar_on_declared_grid"]
        and candidate["retains_declared_fraction_of_upstream_shaping"]
    ]
    meaningful_passes = bool(canonical_fidelity["meaningful_fidelity_gate_passes"])

    payload = {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": {
            "status": "deterministic FP64 diagnostic; not a theorem or rounding certificate",
            "solver_scope": (
                "the returned output is (s-u_hat)/lambda and every declared successful call "
                "passes the computed P15 residual rule"
            ),
            "upstream_comparator": (
                "locked rational Jordan coefficients represented and evaluated in FP64, five "
                "stages, current Frobenius plus epsilon normalization, before aspect scaling "
                "and without BF16 casts"
            ),
            "large_shape_caveat": (
                "Transformer cases execute the complete reduced singular-value system but "
                "do not allocate dense matrices, execute an SVD, or measure accelerator time"
            ),
            "fidelity_caveat": (
                "sampled noncollinearity is evidence, not a global extremum certificate; "
                "the exact/Arb unequal-mode witness is a separate theorem artifact"
            ),
        },
        "config": {
            **asdict(selected),
            "dense_shapes": [list(shape) for shape in selected.dense_shapes],
            "solver": asdict(selected.solver),
            "realistic_transformer_shapes": [list(shape) for shape in REALISTIC_TRANSFORMER_SHAPES],
            "epsilon": selected.solver.epsilon,
            "normalization": "s/(||s||_F+epsilon)",
            "jordan_coefficients": ["6889/2000", "-191/40", "4063/2000"],
            "jordan_iteration_count": STAGES,
            "p15_stopping_rule": "||r||_F <= ||s||_F/250 + rbar_fp64",
            "deployed_output": "(s-u_hat)/lambda",
        },
        "upstream_formula_provenance": {
            "repository": "https://github.com/KellerJordan/Muon",
            "revision": PINNED_KELLER_JORDAN_MUON_REVISION,
            "audited_file": "muon.py",
            "audited_file_sha256": PINNED_MUON_PY_SHA256,
            "normalization": "X/(X.norm(dim=(-2,-1),keepdim=True)+1e-7)",
            "comparison_scope": "FP64 five-stage spectral counterpart only",
        },
        "summary": _summary(dense_cases, spectrum_cases),
        "fidelity_gate": {
            "status": (
                "passes_meaningful_fidelity_gate"
                if meaningful_passes
                else "distinct_but_effectively_scalar"
            ),
            "canonical_input": "diag(3,4)",
            "separate_exact_artifact": (
                "the P16 exact/Arb certificate proves unequal modal gains; this study "
                "reports the independently solved FP64 diagnostics"
            ),
            "best_scalar_departure": canonical_fidelity["best_scalar_multiple"][
                "relative_residual"
            ],
            "best_scalar_departure_threshold": MEANINGFUL_SCALAR_RESIDUAL_THRESHOLD,
            "upstream_best_scalar_departure": canonical_fidelity["upstream_best_scalar_multiple"][
                "relative_residual"
            ],
            "upstream_shaping_retention": canonical_fidelity["shaping_retention_fraction"],
            "upstream_shaping_retention_threshold": (MEANINGFUL_SHAPING_RETENTION_THRESHOLD),
            "modal_gains": canonical_fidelity["modal_gains"],
            "modal_gain_spread": canonical_fidelity["modal_gain_spread"],
            "joint_stability_fidelity_frontier_passes": joint_frontier_passes,
            "interpretation": (
                "no declared frontier point passes both the frozen P14 certificate and "
                "the frozen pre-certificate meaningful-fidelity gates"
                if not joint_frontier_passes
                else "at least one declared frontier point passes both gates"
            ),
        },
        "dense_cases": dense_cases,
        "realistic_spectrum_cases": spectrum_cases,
        "failure_controls": _failure_controls(selected),
        "stability_fidelity_frontier": frontier,
        "experiment_provenance": _provenance(selected),
        "canonical_target": str(CANONICAL_TARGET.relative_to(ROOT)),
    }
    return payload


def canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _parse_shape(value: str) -> tuple[int, int]:
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("shape must use ROWSxCOLUMNS")
    try:
        shape = (int(parts[0]), int(parts[1]))
        _validate_shape(shape)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return shape


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--dense-shape",
        dest="dense_shapes",
        action="append",
        type=_parse_shape,
        help=("dense ROWSxCOLUMNS shape; repeat exactly six times to replace the six defaults"),
    )
    parser.add_argument("--frontier-grid-points", type=int, default=49)
    parser.add_argument("--skip-realistic-spectrum-cases", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    dense_shapes = tuple(arguments.dense_shapes) if arguments.dense_shapes else DEFAULT_DENSE_SHAPES
    if len(dense_shapes) != 6:
        raise SystemExit("the study needs exactly six dense shapes")
    config = SolverStudyConfig(
        seed=arguments.seed,
        dense_shapes=dense_shapes,
        include_realistic_spectrum_cases=not arguments.skip_realistic_spectrum_cases,
        frontier_grid_points=arguments.frontier_grid_points,
    )
    payload = run_study(config)
    rendered = canonical_json(payload)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
