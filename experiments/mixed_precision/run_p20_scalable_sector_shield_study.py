#!/usr/bin/env python3
"""Deterministic P20 scalable-sector-shield diagnostics.

The P20 theorem is an exact, shape-parameterized rounding certificate.  This
script is deliberately separate evidence: it exercises the locked CPU
FP32/BF16 reference implementation, checks the P18 and pinned-upstream
candidate families, and records adversarial and near-zero behavior.

Only the canonical ``2 x 2`` upstream comparison executes the literal
clean-room BF16 matrix operation graph.  Transformer-size cases use complete
packed singular spectra and a diagonal-coordinate counterpart of that graph;
they allocate no dense matrix and do not establish backend bit parity.  All
annulus, spectrum, fidelity, activation, and distortion results are sampled
diagnostics, not global claims.  Global sector safety comes only from the
separate P20 arithmetic certificate.
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
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from enum import Enum
from fractions import Fraction
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from passive_muon.deployed import keller_jordan_map
from passive_muon.equivariant_resolvent_solver import (
    EquivariantResolventSolverConfig,
    jordan_response_and_derivative_fp64,
)
from passive_muon.radial_passivation_tradeoff import P11_SIGNAL_NORM_BOUND
from passive_muon.scalable_mixed_precision_certificate import (
    REPRESENTATIVE_TRANSFORMER_SHAPES,
)
from passive_muon.scalable_sector_shield import (
    ScalableSectorShieldConfig,
    locked_scalable_shield_backend_self_check,
    scalable_sector_shield_manifest,
    shield_reduced_spectrum_mixed_precision,
    shield_sector_candidate_mixed_precision,
)
from passive_muon.scalable_sector_shield_certificate import (
    representative_sector_shield_audits,
)
from passive_muon.sector_projected_resolvent import (
    evaluate_sector_projected_resolvent_fp64,
    evaluate_sector_projected_singular_values_fp64,
)
from passive_muon.sector_shielded_resolvent import fp64_sector_disk_margin_exact
from passive_muon.upstream_momentum import (
    PINNED_KELLER_JORDAN_MUON_REVISION,
    PINNED_MUON_PY_SHA256,
)

FloatArray = NDArray[np.float64]

SCHEMA_VERSION: Final = "passive-muon-p20-scalable-sector-shield-study-v1"
SEED: Final = 20_260_906
STAGES: Final = 5
EPSILON: Final = 1.0e-7
LOCKED_DEFAULT_DECISION_DIGEST: Final = (
    "998ef020d1b642cf923a0789dfdd489b2b949a2aed0b3948cff67b88584a5c9c"
)
P11_SIGNAL_GUARD: Final = float(P11_SIGNAL_NORM_BOUND)

MEANINGFUL_SCALAR_RESIDUAL_THRESHOLD: Final = 1.0e-3
MEANINGFUL_SHAPING_RETENTION_THRESHOLD: Final = 1.0e-1
INFORMATIVE_UPSTREAM_SHAPING_THRESHOLD: Final = 1.0e-2
OPERATING_ANNULUS_INNER_RADIUS: Final = Fraction(3, 4)
OPERATING_ANNULUS_OUTER_RADIUS: Final = Fraction(25)
OPERATING_ANNULUS_STRICT_RELATIVE_TOLERANCE: Final = 2.0**-44
OPERATING_ANNULUS_STRICT_ABSOLUTE_TOLERANCE: Final = 2.0**-50
RANK_ACCUMULATION_MODE_RATIO: Final = 41.0 / 10_000.0

CANDIDATE_P18: Final = "guarded_p18"
CANDIDATE_UPSTREAM: Final = "pinned_upstream"
CANDIDATE_FAMILIES: Final = (CANDIDATE_P18, CANDIDATE_UPSTREAM)
DENSE_SMOKE_SHAPE: Final = (768, 768)

ROOT = Path(__file__).resolve().parents[2]
CANONICAL_TARGET = ROOT / "results" / "summaries" / "p20_scalable_sector_shield_study.json"
SOURCE_PATHS = (
    Path("experiments/mixed_precision/run_p20_scalable_sector_shield_study.py"),
    Path("src/passive_muon/deployed.py"),
    Path("src/passive_muon/equivariant_resolvent_solver.py"),
    Path("src/passive_muon/scalable_mixed_precision.py"),
    Path("src/passive_muon/scalable_mixed_precision_certificate.py"),
    Path("src/passive_muon/scalable_sector_shield.py"),
    Path("src/passive_muon/scalable_sector_shield_certificate.py"),
    Path("src/passive_muon/sector_projected_resolvent.py"),
    Path("src/passive_muon/sector_shielded_resolvent.py"),
    Path("src/passive_muon/upstream_momentum.py"),
    Path("pyproject.toml"),
    Path("uv.lock"),
)


@dataclass(frozen=True)
class P20StudyConfig:
    """Complete deterministic configuration of the P20 study."""

    seed: int = SEED
    operating_annulus_grid_points: int = 33
    include_transformer_spectra: bool = True
    torch_threads: int = 1
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
        if not isinstance(self.include_transformer_spectra, bool):
            raise TypeError("include_transformer_spectra must be Boolean")
        if (
            not isinstance(self.torch_threads, int)
            or isinstance(self.torch_threads, bool)
            or self.torch_threads <= 0
        ):
            raise ValueError("torch_threads must be a positive integer")
        if not isinstance(self.solver, EquivariantResolventSolverConfig):
            raise TypeError("solver must be an EquivariantResolventSolverConfig")


def _jsonable(value: object) -> object:
    if isinstance(value, Fraction):
        return {
            "numerator": value.numerator,
            "denominator": value.denominator,
            "fraction": str(value),
            "decimal": float(value),
        }
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _norm(values: Tensor | FloatArray) -> float:
    if isinstance(values, Tensor):
        return float(torch.linalg.vector_norm(values.to(torch.float64)))
    array = np.asarray(values, dtype=np.float64)
    maximum = float(np.max(np.abs(array)))
    if maximum == 0.0:
        return 0.0
    scaled = array / maximum
    return maximum * math.sqrt(math.fsum(float(item) ** 2 for item in scaled.ravel()))


def _unit(values: FloatArray) -> FloatArray:
    norm = _norm(values)
    if norm == 0.0:
        raise ValueError("cannot normalize a zero diagnostic template")
    return np.ascontiguousarray(values / norm, dtype=np.float64)


def _tensor_sha256(values: Tensor) -> str:
    host = values.detach().cpu().contiguous()
    storage = host.view(torch.uint16) if host.dtype == torch.bfloat16 else host
    return hashlib.sha256(storage.numpy().tobytes(order="C")).hexdigest()


def _cosine(left: Tensor, right: Tensor) -> float | None:
    left64 = left.to(torch.float64).reshape(-1)
    right64 = right.to(torch.float64).reshape(-1)
    denominator = _norm(left64) * _norm(right64)
    if denominator == 0.0:
        return None
    value = float(torch.dot(left64, right64)) / denominator
    return max(-1.0, min(1.0, value))


def _best_scalar_departure(signal: FloatArray, output: FloatArray) -> float:
    source = np.asarray(signal, dtype=np.float64).ravel()
    target = np.asarray(output, dtype=np.float64).ravel()
    square = float(np.dot(source, source))
    output_norm = _norm(target)
    if square == 0.0 or output_norm == 0.0:
        return 0.0 if output_norm == 0.0 else 1.0
    coefficient = float(np.dot(source, target)) / square
    return _norm(target - coefficient * source) / output_norm


def _upstream_jordan_fp64(signal: FloatArray) -> FloatArray:
    radius = _norm(signal)
    if radius == 0.0:
        return np.zeros_like(signal)
    normalized = np.ascontiguousarray(signal / (radius + EPSILON), dtype=np.float64)
    response, _derivative = jordan_response_and_derivative_fp64(normalized)
    return response


def _fidelity_record(
    signal: FloatArray,
    output: FloatArray,
    upstream: FloatArray,
) -> dict[str, object]:
    departure = _best_scalar_departure(signal, output)
    upstream_departure = _best_scalar_departure(signal, upstream)
    retention = departure / upstream_departure if upstream_departure > 0.0 else None
    informative = upstream_departure >= INFORMATIVE_UPSTREAM_SHAPING_THRESHOLD
    return {
        "best_scalar_departure": departure,
        "upstream_best_scalar_departure": upstream_departure,
        "shaping_retention_fraction": retention,
        "informative_upstream_shaping": informative,
        "meaningful_fidelity_gate_passes": (
            informative
            and departure >= MEANINGFUL_SCALAR_RESIDUAL_THRESHOLD
            and retention is not None
            and retention >= MEANINGFUL_SHAPING_RETENTION_THRESHOLD
        ),
        "output_to_input_amplitude": _norm(output) / max(_norm(signal), sys.float_info.min),
        "output_to_upstream_amplitude": _norm(output) / max(_norm(upstream), sys.float_info.min),
    }


def _pinned_diagonal_coordinate_candidate_bf16(signal: Tensor) -> Tensor:
    """Evaluate the pinned public operation order on packed diagonal modes.

    Products are scalar because a mathematical diagonal matrix has no modal
    cross terms.  The packed norm does not reproduce a dense backend's
    reduction tree, so this helper is a formula diagnostic rather than a
    literal large-matrix backend execution.
    """

    if signal.ndim != 1 or signal.dtype != torch.float32 or signal.device.type != "cpu":
        raise ValueError("reduced signal must be one CPU float32 vector")
    if not bool(torch.isfinite(signal).all()):
        raise ValueError("reduced signal must be finite")
    state = signal.to(torch.bfloat16)
    denominator = torch.linalg.vector_norm(state) + EPSILON
    state = state / denominator
    a, b, c = 3.4445, -4.775, 2.0315
    for _ in range(STAGES):
        gram = state * state
        correction = b * gram + (c * gram) * gram
        state = a * state + correction * state
    return state.contiguous()


def _shield_metrics(signal: Tensor, candidate: Tensor, result: object) -> dict[str, object]:
    output = result.output
    diagnostics = result.diagnostics
    candidate_finite = bool(torch.isfinite(candidate).all())
    candidate_norm = _norm(candidate) if candidate_finite else None
    change = (
        _norm(output.to(torch.float64) - candidate.to(torch.float64)) if candidate_finite else None
    )
    signal64_full = signal.detach().cpu().to(torch.float64).reshape(-1)
    output64_full = output.detach().cpu().to(torch.float64).reshape(-1)
    support = torch.logical_or(signal64_full != 0.0, output64_full != 0.0)
    if bool(support.any()):
        signal64 = signal64_full[support].numpy()
        output64 = output64_full[support].numpy()
    else:
        signal64 = np.zeros(1, dtype=np.float64)
        output64 = np.zeros(1, dtype=np.float64)
    exact_margin = fp64_sector_disk_margin_exact(output64, signal64)
    return {
        "candidate_sha256": _tensor_sha256(candidate),
        "output_sha256": _tensor_sha256(output),
        "candidate_finite": candidate_finite,
        "candidate_frobenius": candidate_norm,
        "output_frobenius": _norm(output),
        "change_frobenius": change,
        "relative_change": (
            change / max(candidate_norm, sys.float_info.min)
            if change is not None and candidate_norm is not None
            else None
        ),
        "candidate_output_cosine": _cosine(candidate, output) if candidate_finite else None,
        "bitwise_identity": bool(torch.equal(candidate, output)),
        "offline_exact_p19_disk_margin_for_stored_values": str(exact_margin),
        "offline_exact_p19_disk_check_passes": exact_margin >= 0,
        "shield_diagnostics": _jsonable(diagnostics),
    }


def _dense_reference_smoke() -> dict[str, object]:
    """Exercise the full 768x768 reduction tree without a dense upstream matmul."""

    config = ScalableSectorShieldConfig(matrix_shape=DENSE_SMOKE_SHAPE)
    signal = torch.zeros(DENSE_SMOKE_SHAPE, dtype=torch.float32)
    signal[0, 0] = 3.0
    signal[1, 1] = 4.0
    candidate_fp32 = (0.5 * signal).contiguous()
    candidate_bf16 = candidate_fp32.to(torch.bfloat16).contiguous()
    fp32 = shield_sector_candidate_mixed_precision(signal, candidate_fp32, config)
    bf16 = shield_sector_candidate_mixed_precision(
        signal.to(torch.bfloat16).contiguous(), candidate_bf16, config
    )
    return {
        "scope": (
            "full configured 768x768 P20 reduction path on a sparse deterministic matrix; "
            "this is an arithmetic smoke test, not a Transformer workload benchmark"
        ),
        "fp32_inputs": _shield_metrics(signal, candidate_fp32, fp32),
        "bf16_inputs": _shield_metrics(signal.to(torch.bfloat16), candidate_bf16, bf16),
    }


def _solver_config_for_annulus(config: P20StudyConfig) -> EquivariantResolventSolverConfig:
    return replace(
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


def _canonical_case(config: P20StudyConfig) -> dict[str, object]:
    signal_values = torch.tensor([3.0, 4.0], dtype=torch.float32)
    signal = torch.diag(signal_values)
    shield_config = ScalableSectorShieldConfig(matrix_shape=(2, 2))
    p18_evaluation = evaluate_sector_projected_resolvent_fp64(
        signal.to(torch.float64).numpy(), solver_config=config.solver
    )
    p18_candidate = torch.from_numpy(p18_evaluation.output).to(torch.float32).contiguous()
    literal_bf16 = keller_jordan_map(signal, steps=STAGES, eps=EPSILON)
    upstream_candidate = literal_bf16.contiguous()
    reduced_upstream = _pinned_diagonal_coordinate_candidate_bf16(
        torch.tensor([3.0, 4.0], dtype=torch.float32)
    )
    if not torch.equal(torch.diagonal(literal_bf16), reduced_upstream):
        raise RuntimeError("packed diagonal operation order failed canonical literal parity")
    records: dict[str, object] = {}
    for name, candidate in (
        (CANDIDATE_P18, p18_candidate),
        (CANDIDATE_UPSTREAM, upstream_candidate),
    ):
        result = shield_sector_candidate_mixed_precision(signal, candidate, shield_config)
        records[name] = _shield_metrics(signal, candidate, result)

    p18_output = records[CANDIDATE_P18]
    assert isinstance(p18_output, dict)
    shielded_p18 = shield_sector_candidate_mixed_precision(
        signal, p18_candidate, shield_config
    ).output
    upstream_fp64 = np.diag(_upstream_jordan_fp64(np.asarray([3.0, 4.0], dtype=np.float64)))
    return {
        "input": "diag(3,4)",
        "execution_scope": (
            "literal clean-room pinned BF16 2x2 matrix operation graph followed by the full "
            "2x2 P20 shield under its frozen generated margin"
        ),
        "input_sha256": _tensor_sha256(signal),
        "literal_upstream_bf16_sha256": _tensor_sha256(literal_bf16),
        "reduced_helper_matches_literal_diagonal_bits": True,
        "p18_solver_residual_passes": p18_evaluation.resolvent_result.diagnostics.certified,
        "candidate_records": records,
        "shielded_p18_fidelity": _fidelity_record(
            signal.to(torch.float64).numpy(),
            shielded_p18.to(torch.float64).numpy(),
            upstream_fp64,
        ),
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


def _empty_family_state() -> dict[str, object]:
    return {
        "evaluated": 0,
        "active": 0,
        "clipped": 0,
        "bitwise_identity": 0,
        "fallback": 0,
        "fail_closed": 0,
        "exact_disk_passes": 0,
        "minimum_cosine": 1.0,
        "maximum_relative_change": 0.0,
        "informative": 0,
        "fidelity_passes": 0,
    }


def _observe_family(
    state: dict[str, object],
    metrics: dict[str, object],
    fidelity: dict[str, object] | None,
) -> None:
    diagnostics = metrics["shield_diagnostics"]
    assert isinstance(diagnostics, dict)
    state["evaluated"] = int(state["evaluated"]) + 1
    state["active"] = int(state["active"]) + int(bool(diagnostics.get("active", False)))
    state["clipped"] = int(state["clipped"]) + int(
        bool(diagnostics.get("candidate_clipped", False))
    )
    state["bitwise_identity"] = int(state["bitwise_identity"]) + int(
        bool(metrics["bitwise_identity"])
    )
    state["fallback"] = int(state["fallback"]) + int(
        bool(diagnostics.get("fallback", diagnostics.get("used_fallback", False)))
    )
    state["fail_closed"] = int(state["fail_closed"]) + int(
        bool(diagnostics.get("fail_closed", False))
    )
    state["exact_disk_passes"] = int(state["exact_disk_passes"]) + int(
        bool(metrics["offline_exact_p19_disk_check_passes"])
    )
    cosine = metrics["candidate_output_cosine"]
    if cosine is not None:
        state["minimum_cosine"] = min(float(state["minimum_cosine"]), float(cosine))
    state["maximum_relative_change"] = max(
        float(state["maximum_relative_change"]), float(metrics["relative_change"])
    )
    if fidelity is not None and bool(fidelity["informative_upstream_shaping"]):
        state["informative"] = int(state["informative"]) + 1
        state["fidelity_passes"] = int(state["fidelity_passes"]) + int(
            bool(fidelity["meaningful_fidelity_gate_passes"])
        )


def _run_annulus(config: P20StudyConfig) -> dict[str, object]:
    radii, ratios = _operating_annulus_grid(config.operating_annulus_grid_points)
    solver = _solver_config_for_annulus(config)
    shield_config = ScalableSectorShieldConfig(matrix_shape=(2, 2))
    states = {name: _empty_family_state() for name in CANDIDATE_FAMILIES}
    worst_p18_residual_ratio = 0.0
    for radius in radii:
        for ratio in ratios:
            signal_values = float(radius) * _unit(np.asarray([ratio, 1.0], dtype=np.float64))
            signal = torch.diag(torch.from_numpy(signal_values).to(torch.float32)).contiguous()
            p18_evaluation = evaluate_sector_projected_singular_values_fp64(
                signal_values, solver_config=solver
            )
            diagnostics = p18_evaluation.solver_diagnostics
            residual_ratio = (
                diagnostics.residual_norm / diagnostics.p15_threshold
                if diagnostics.p15_threshold > 0.0
                else 0.0
            )
            worst_p18_residual_ratio = max(worst_p18_residual_ratio, residual_ratio)
            candidates = {
                CANDIDATE_P18: torch.from_numpy(p18_evaluation.output)
                .to(torch.float32)
                .contiguous(),
                CANDIDATE_UPSTREAM: keller_jordan_map(
                    signal, steps=STAGES, eps=EPSILON
                ).contiguous(),
            }
            candidates[CANDIDATE_P18] = torch.diag(candidates[CANDIDATE_P18]).contiguous()
            upstream_fp64 = _upstream_jordan_fp64(signal_values)
            for family, candidate in candidates.items():
                result = shield_sector_candidate_mixed_precision(
                    signal, candidate.contiguous(), shield_config
                )
                metrics = _shield_metrics(signal, candidate, result)
                fidelity = None
                if family == CANDIDATE_P18:
                    fidelity = _fidelity_record(
                        signal_values,
                        torch.diagonal(result.output).to(torch.float64).numpy(),
                        upstream_fp64,
                    )
                _observe_family(states[family], metrics, fidelity)

    for state in states.values():
        evaluated = int(state["evaluated"])
        state["inactive"] = evaluated - int(state["active"])
        state["all_outputs_pass_offline_exact_disk_check"] = (
            int(state["exact_disk_passes"]) == evaluated
        )
        informative = int(state["informative"])
        state["all_informative_fidelity_pass"] = (
            int(state["fidelity_passes"]) == informative if informative else None
        )
        state["active_accounting_closes"] = int(state["active"]) == (
            int(state["clipped"]) + int(state["fallback"])
        )
    p18_state = states[CANDIDATE_P18]
    p18_evaluated = int(p18_state["evaluated"])
    p18_inactive = int(p18_state["inactive"])
    return {
        "scope": (
            "frozen post-P17 two-mode operating-annulus grid on literal 2x2 CPU paths; "
            "sampled activation and fidelity evidence, not a global theorem"
        ),
        "radius_count": int(radii.size),
        "mode_ratio_count": int(ratios.size),
        "case_count": int(radii.size * ratios.size),
        "candidate_evaluation_count": int(2 * radii.size * ratios.size),
        "candidate_families": states,
        "guarded_p18_inactivity_interpretation": {
            "inactive_count": p18_inactive,
            "active_clip_count": int(p18_state["clipped"]),
            "half_fallback_count": int(p18_state["fallback"]),
            "inactive_fraction": p18_inactive / p18_evaluated,
            "normally_inactive": p18_inactive > int(p18_state["active"]),
            "qualification": (
                "inactivity is reported by explicit sampled counts; the narrower static "
                "screen can clip P18 outputs near its sector boundary and no entire-annulus "
                "inactivity claim is made"
            ),
        },
        "worst_p18_residual_to_threshold_ratio": worst_p18_residual_ratio,
    }


def _shape_spectra(shape: tuple[int, int]) -> tuple[tuple[str, FloatArray], ...]:
    rank = min(shape)
    flat = np.full(rank, 1.0 / math.sqrt(rank), dtype=np.float64)
    logarithmic = 0.5 * P11_SIGNAL_GUARD * _unit(np.geomspace(1.0, 1.0e-4, rank, dtype=np.float64))
    blocks = np.concatenate(
        (
            np.ones(rank // 2, dtype=np.float64),
            np.full(rank - rank // 2, 0.01, dtype=np.float64),
        )
    )
    blocks = 0.999 * P11_SIGNAL_GUARD * _unit(blocks)
    return (
        ("unit_flat_repeated", flat),
        ("half_guard_log_condition_1e4", logarithmic),
        ("guard_two_repeated_blocks", blocks),
    )


def _run_transformer_spectra(config: P20StudyConfig) -> dict[str, object]:
    shapes = tuple((item.rows, item.columns) for item in REPRESENTATIVE_TRANSFORMER_SHAPES)
    if not config.include_transformer_spectra:
        return {
            "scope": "disabled by configuration",
            "declared_shapes": [list(shape) for shape in shapes],
            "cases": [],
            "summary": {"shape_count": 0, "case_count": 0},
        }
    cases: list[dict[str, object]] = []
    for shape in shapes:
        shield_config = ScalableSectorShieldConfig(matrix_shape=shape)
        for label, values in _shape_spectra(shape):
            signal = torch.from_numpy(values).to(torch.float32).contiguous()
            p18 = evaluate_sector_projected_singular_values_fp64(
                values, solver_config=config.solver
            )
            candidates = {
                CANDIDATE_P18: torch.from_numpy(p18.output).to(torch.float32).contiguous(),
                CANDIDATE_UPSTREAM: _pinned_diagonal_coordinate_candidate_bf16(signal),
            }
            candidate_records: dict[str, object] = {}
            for family, candidate in candidates.items():
                result = shield_reduced_spectrum_mixed_precision(signal, candidate, shield_config)
                metrics = _shield_metrics(signal, candidate, result)
                candidate_records[family] = metrics
            cases.append(
                {
                    "shape": list(shape),
                    "label": label,
                    "case_role": (
                        "boundary_stress"
                        if label == "unit_flat_repeated"
                        else "operating_spectrum_diagnostic"
                    ),
                    "rank": int(signal.numel()),
                    "input_frobenius": _norm(signal),
                    "input_sha256": _tensor_sha256(signal),
                    "p18_solver_residual_passes": p18.solver_diagnostics.certified,
                    "candidate_records": candidate_records,
                }
            )

    summary: dict[str, object] = {
        "shape_count": len(shapes),
        "case_count": len(cases),
        "candidate_evaluation_count": len(cases) * len(CANDIDATE_FAMILIES),
        "all_p18_solver_residuals_pass": all(
            bool(case["p18_solver_residual_passes"]) for case in cases
        ),
    }
    for family in CANDIDATE_FAMILIES:
        records = [case["candidate_records"][family] for case in cases]  # type: ignore[index]
        diagnostics = [record["shield_diagnostics"] for record in records]
        summary[family] = {
            "evaluated": len(records),
            "active": sum(bool(item.get("active", False)) for item in diagnostics),
            "clipped": sum(bool(item.get("candidate_clipped", False)) for item in diagnostics),
            "bitwise_identity": sum(bool(item["bitwise_identity"]) for item in records),
            "fallback": sum(
                bool(item.get("fallback", item.get("used_fallback", False))) for item in diagnostics
            ),
            "all_outputs_pass_offline_exact_disk_check": all(
                bool(item["offline_exact_p19_disk_check_passes"]) for item in records
            ),
            "minimum_candidate_output_cosine": min(
                float(item["candidate_output_cosine"])
                for item in records
                if item["candidate_output_cosine"] is not None
            ),
            "maximum_relative_change": max(float(item["relative_change"]) for item in records),
        }
    p18_operating = [
        case["candidate_records"][CANDIDATE_P18]  # type: ignore[index]
        for case in cases
        if case["case_role"] == "operating_spectrum_diagnostic"
    ]
    p18_stress = [
        case["candidate_records"][CANDIDATE_P18]  # type: ignore[index]
        for case in cases
        if case["case_role"] == "boundary_stress"
    ]
    summary["guarded_p18_inactivity_interpretation"] = {
        "operating_spectrum_case_count": len(p18_operating),
        "operating_spectrum_inactive_count": sum(
            not bool(item["shield_diagnostics"]["active"]) for item in p18_operating
        ),
        "all_operating_spectrum_cases_inactive": all(
            not bool(item["shield_diagnostics"]["active"]) for item in p18_operating
        ),
        "flat_boundary_stress_case_count": len(p18_stress),
        "flat_boundary_stress_active_count": sum(
            bool(item["shield_diagnostics"]["active"]) for item in p18_stress
        ),
        "qualification": (
            "the narrower static P20 screen intentionally activates on the repeated-flat "
            "P18 boundary stress; normal inactivity is claimed only for the declared "
            "operating spectra, while annulus activation is reported separately by counts"
        ),
    }
    audits = representative_sector_shield_audits()
    return {
        "scope": (
            "complete packed singular spectra with declared P9 shapes and shape-specific "
            "P20 margins; no dense allocation, dense reduction-tree parity, SVD, GPU timing, "
            "or claim of literal large-matrix backend output"
        ),
        "upstream_scope": (
            "pinned-formula BF16 diagonal-coordinate diagnostic; these large-shape cases "
            "do not execute the literal clean-room matrix graph"
        ),
        "declared_shapes": [list(shape) for shape in shapes],
        "exact_shape_margin_references": [
            {
                "shape": [audit.shape.rows, audit.shape.columns],
                "certified_inward_radius": _jsonable(audit.certified_inward_radius),
                "inward_margin": _jsonable(audit.inward_margin),
                "certified": audit.certified,
            }
            for audit in audits
        ],
        "spectrum_families": [label for label, _values in _shape_spectra(shapes[0])],
        "cases": cases,
        "summary": summary,
    }


def _one_ulp_outward_candidate(signal: Tensor) -> Tensor:
    upper = signal.new_tensor(509.0 / 512.0)
    candidate = (upper * signal).contiguous()
    flat = candidate.reshape(-1)
    source = signal.reshape(-1)
    index = int(torch.argmax(torch.abs(source)).item())
    direction = math.inf if float(source[index]) >= 0.0 else -math.inf
    flat[index] = torch.nextafter(flat[index], flat.new_tensor(direction))
    return candidate


def _try_control(
    name: str,
    signal: Tensor,
    candidate: Tensor,
    config: ScalableSectorShieldConfig,
) -> dict[str, object]:
    candidate_margin: Fraction | None = None
    if bool(torch.isfinite(signal).all()) and bool(torch.isfinite(candidate).all()):
        candidate_margin = fp64_sector_disk_margin_exact(
            candidate.to(torch.float64).numpy(), signal.to(torch.float64).numpy()
        )
    try:
        result = shield_sector_candidate_mixed_precision(signal, candidate, config)
    except (FloatingPointError, ValueError) as error:
        return {
            "name": name,
            "returned": False,
            "exception_type": type(error).__name__,
            "exception": str(error),
        }
    return {
        "name": name,
        "returned": True,
        "candidate_offline_exact_p19_disk_margin": (
            str(candidate_margin) if candidate_margin is not None else None
        ),
        "candidate_was_inside_original_disk": (
            candidate_margin >= 0 if candidate_margin is not None else None
        ),
        **_shield_metrics(signal, candidate, result),
    }


def _try_reduced_control(
    name: str,
    signal: Tensor,
    candidate: Tensor,
    config: ScalableSectorShieldConfig,
) -> dict[str, object]:
    candidate_margin: Fraction | None = None
    if bool(torch.isfinite(signal).all()) and bool(torch.isfinite(candidate).all()):
        candidate_margin = fp64_sector_disk_margin_exact(
            candidate.to(torch.float64).numpy(), signal.to(torch.float64).numpy()
        )
    try:
        result = shield_reduced_spectrum_mixed_precision(signal, candidate, config)
    except (FloatingPointError, ValueError) as error:
        return {
            "name": name,
            "returned": False,
            "exception_type": type(error).__name__,
            "exception": str(error),
        }
    return {
        "name": name,
        "returned": True,
        "candidate_offline_exact_p19_disk_margin": (
            str(candidate_margin) if candidate_margin is not None else None
        ),
        "candidate_was_inside_original_disk": (
            candidate_margin >= 0 if candidate_margin is not None else None
        ),
        **_shield_metrics(signal, candidate, result),
    }


def _adversarial_controls() -> dict[str, object]:
    signal = torch.diag(torch.tensor([3.0, 4.0], dtype=torch.float32))
    config = ScalableSectorShieldConfig(matrix_shape=(2, 2))
    skew = torch.tensor([[0.0, -4.0], [3.0, 0.0]], dtype=torch.float32)
    candidates = {
        "zero": torch.zeros_like(signal),
        "outward": 8.0 * signal,
        "anti_aligned": -3.0 * signal,
        "orthogonal_corruption": 5.0 * skew,
        "one_ulp_outward": _one_ulp_outward_candidate(signal),
        "nan_candidate": torch.full_like(signal, math.nan),
        "infinite_candidate": torch.full_like(signal, math.inf),
    }
    records = {
        name: _try_control(name, signal, candidate.contiguous(), config)
        for name, candidate in candidates.items()
    }

    zero_signal = torch.zeros((2, 2), dtype=torch.float32)
    zero_records = {
        "zero_candidate": _try_control(
            "zero_candidate", zero_signal, torch.zeros_like(zero_signal), config
        ),
        "finite_nonzero_candidate": _try_control(
            "finite_nonzero_candidate", zero_signal, torch.ones_like(zero_signal), config
        ),
        "nonfinite_candidate": _try_control(
            "nonfinite_candidate", zero_signal, torch.full_like(zero_signal, math.nan), config
        ),
    }
    rejected_signals: dict[str, dict[str, object]] = {}
    for name, bad_signal in {
        "nan_signal": torch.tensor([[math.nan, 0.0], [0.0, 1.0]], dtype=torch.float32),
        "infinite_signal": torch.tensor([[math.inf, 0.0], [0.0, -1.0]], dtype=torch.float32),
    }.items():
        record = _try_control(name, bad_signal, torch.zeros_like(bad_signal), config)
        if bool(record["returned"]):
            raise RuntimeError(f"nonfinite signal control {name!r} emitted an update")
        rejected_signals[name] = record
    return {
        "scope": (
            "small CPU controls; offline exact P19 checks validate returned stored values "
            "but are not part of the P20 runtime"
        ),
        "signal": signal.tolist(),
        "finite_and_nonfinite_candidate_controls": records,
        "zero_signal_controls": zero_records,
        "nonfinite_signal_rejections": rejected_signals,
    }


def _fp32_from_bits(bits: int) -> Tensor:
    signed = bits if bits < 2**31 else bits - 2**32
    return torch.tensor([signed], dtype=torch.int32).view(torch.float32)[0]


def _near_zero_controls() -> dict[str, object]:
    config = ScalableSectorShieldConfig(matrix_shape=(2, 2))
    values = {
        "minimum_normal": _fp32_from_bits(0x0080_0000),
        "even_subnormal_exact_halving": _fp32_from_bits(0x0000_0002),
        "maximum_subnormal": _fp32_from_bits(0x007F_FFFF),
        "minimum_subnormal": _fp32_from_bits(0x0000_0001),
    }
    records: dict[str, object] = {}
    for name, value in values.items():
        signal = torch.zeros((2, 2), dtype=torch.float32)
        signal[0, 0] = value
        records[name] = _try_control(name, signal, 7.0 * signal, config)
        input_bits = int(value.view(torch.int32).item()) & 0xFFFF_FFFF
        records[name]["input_bits_hex"] = f"0x{input_bits:08x}"
    return {
        "ftz_policy": (
            "the locked proof backend requires gradual underflow; a backend that flushes "
            "any probe to zero fails the arithmetic contract before emitting updates"
        ),
        "finite_precision_neighborhood": (
            "signals failing the theorem's exact-halving/normal guard belong to the explicit "
            "near-zero finite-precision neighborhood; they do not inherit exact sector output"
        ),
        "cases": records,
    }


def _git_state() -> dict[str, object]:
    def run(*arguments: str) -> str | None:
        completed = subprocess.run(
            ["git", *arguments], cwd=ROOT, capture_output=True, text=True, check=False
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


def _provenance(config: P20StudyConfig) -> dict[str, object]:
    return {
        "git": _git_state(),
        "seed": config.seed,
        "hardware": {
            "device": "cpu",
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor() or "unreported",
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
        },
        "software": {
            "python": sys.version,
            "numpy": str(np.__version__),
            "torch": str(torch.__version__),
            "torch_config": torch.__config__.show(),
        },
        "runtime": {
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        },
        "source_snapshot": _source_snapshot(),
    }


def _decision_digest(payload: dict[str, object]) -> str:
    annulus = payload["operating_annulus"]
    transformer = payload["transformer_spectrum_diagnostics"]
    controls = payload["adversarial_controls"]
    assert isinstance(annulus, dict)
    assert isinstance(transformer, dict)
    assert isinstance(controls, dict)
    annulus_decisions = {
        family: {
            key: record[key]
            for key in (
                "evaluated",
                "active",
                "inactive",
                "bitwise_identity",
                "fallback",
                "fail_closed",
                "exact_disk_passes",
                "informative",
                "fidelity_passes",
            )
        }
        for family, record in annulus["candidate_families"].items()
    }
    transformer_summary = transformer["summary"]
    transformer_decisions = {
        "shape_count": transformer_summary["shape_count"],
        "case_count": transformer_summary["case_count"],
        "all_p18_solver_residuals_pass": transformer_summary.get("all_p18_solver_residuals_pass"),
        **{
            family: {
                key: transformer_summary[family][key]
                for key in (
                    "evaluated",
                    "active",
                    "bitwise_identity",
                    "fallback",
                    "all_outputs_pass_offline_exact_disk_check",
                )
            }
            for family in CANDIDATE_FAMILIES
            if family in transformer_summary
        },
    }
    adversarial_decisions = {
        name: {
            "returned": record["returned"],
            "exception_type": record.get("exception_type"),
            "active": (
                record.get("shield_diagnostics", {}).get("active") if record["returned"] else None
            ),
            "used_fallback": (
                record.get("shield_diagnostics", {}).get("used_fallback")
                if record["returned"]
                else None
            ),
        }
        for name, record in controls["finite_and_nonfinite_candidate_controls"].items()
    }
    stable = {
        "annulus": annulus_decisions,
        "transformer": transformer_decisions,
        "adversarial": adversarial_decisions,
    }
    rendered = json.dumps(stable, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def run_study(config: P20StudyConfig | None = None) -> dict[str, object]:
    """Run all declared P20 candidate and shield diagnostics."""

    selected = P20StudyConfig() if config is None else config
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(selected.torch_threads)
    torch.use_deterministic_algorithms(True)
    try:
        payload: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "claim_scope": {
                "proof_boundary": (
                    "global stored-output containment belongs to the independent P20 exact "
                    "rounding certificate; this file contains deterministic falsification"
                ),
                "candidate_independence": (
                    "P19 safety permits any finite, inaccurate, corrupted, or history-dependent "
                    "candidate; candidate comparisons below concern fidelity and activation"
                ),
                "literal_upstream_boundary": (
                    "canonical and annulus 2x2 cases run the literal pinned BF16 matrix graph; "
                    "large shapes use a reduced diagonal-coordinate formula diagnostic without "
                    "backend parity"
                ),
                "sampled_boundary": (
                    "annulus and Transformer spectra are finite sampled diagnostics, not global "
                    "fidelity, activation, or neural-training claims"
                ),
            },
            "config": {
                **asdict(selected),
                "solver": asdict(selected.solver),
                "candidate_families": list(CANDIDATE_FAMILIES),
                "normalization": "X/(||X||_F+1e-7)",
                "epsilon": EPSILON,
                "jordan_coefficients": ["6889/2000", "-191/40", "4063/2000"],
                "jordan_iteration_count": STAGES,
                "declared_transformer_shapes": [
                    [shape.rows, shape.columns] for shape in REPRESENTATIVE_TRANSFORMER_SHAPES
                ],
            },
            "locked_runtime_contract": {
                "representative_manifest": scalable_sector_shield_manifest(
                    ScalableSectorShieldConfig(matrix_shape=(4_096, 11_008))
                ),
                "backend_self_check": locked_scalable_shield_backend_self_check(),
            },
            "upstream_formula_provenance": {
                "repository": "https://github.com/KellerJordan/Muon",
                "revision": PINNED_KELLER_JORDAN_MUON_REVISION,
                "audited_file": "muon.py",
                "audited_file_sha256": PINNED_MUON_PY_SHA256,
                "normalization": "X/(X.norm(dim=(-2,-1),keepdim=True)+1e-7)",
                "candidate_scope": (
                    "five-stage BF16 Jordan orthogonalizer before aspect scaling, weight decay, "
                    "distributed semantics, momentum, or parameter update"
                ),
            },
            "canonical_diag_3_4": _canonical_case(selected),
            "dense_reference_smoke": _dense_reference_smoke(),
            "operating_annulus": _run_annulus(selected),
            "transformer_spectrum_diagnostics": _run_transformer_spectra(selected),
            "adversarial_controls": _adversarial_controls(),
            "near_zero_and_ftz_controls": _near_zero_controls(),
            "cross_platform_replay": {
                "strategy": (
                    "run the locked arithmetic self-check and this study on macOS arm64 and "
                    "Ubuntu x86_64; compare decision_digest and acceptance booleans, archive raw "
                    "bit hashes separately, and deny certificate inheritance if a backend probe "
                    "or decision differs"
                ),
                "runtime_big_integer_check": False,
                "offline_small_control_exact_check": True,
                "cross_platform_status": (
                    "strategy frozen; each reported execution records one platform and does not "
                    "by itself prove a second-platform match"
                ),
            },
            "canonical_target": str(CANONICAL_TARGET.relative_to(ROOT)),
        }
        payload["decision_digest"] = _decision_digest(payload)
        default_decision_grid = selected == P20StudyConfig()
        payload["cross_platform_replay"]["locked_default_decision_digest"] = (
            LOCKED_DEFAULT_DECISION_DIGEST
        )
        payload["cross_platform_replay"]["this_run_uses_locked_default_grid"] = (
            default_decision_grid
        )
        payload["cross_platform_replay"]["locked_default_digest_matches"] = (
            payload["decision_digest"] == LOCKED_DEFAULT_DECISION_DIGEST
            if default_decision_grid
            else None
        )
        if default_decision_grid and payload["decision_digest"] != LOCKED_DEFAULT_DECISION_DIGEST:
            raise RuntimeError("P20 default discrete decision digest changed")
        payload["experiment_provenance"] = _provenance(selected)
        return payload
    finally:
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_deterministic)


def canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--operating-annulus-grid-points", type=int, default=33)
    parser.add_argument("--skip-transformer-spectra", action="store_true")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    config = P20StudyConfig(
        seed=arguments.seed,
        operating_annulus_grid_points=arguments.operating_annulus_grid_points,
        include_transformer_spectra=not arguments.skip_transformer_spectra,
        torch_threads=arguments.torch_threads,
    )
    payload = run_study(config)
    rendered = canonical_json(payload)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
