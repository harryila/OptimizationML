#!/usr/bin/env python3
"""CPU diagnostic for P9 mixed-precision stage-boundary policies.

This is a reproducible numerical comparison, not a theorem replay.  In
particular, every polynomial matrix product below uses PyTorch's optimized
native FP32 ``@`` implementation.  Its reduction order, contraction, and
library implementation are not the balanced proof kernel certified by P9.

The diagnostic compares three otherwise matched boundary policies:

* one BF16 cast only after the fifth stage;
* one-term BF16 storage at the normalized input and every stage; and
* compensated high/low BF16 storage at those same boundaries.

The float64 target is also numerical, not exact arithmetic.  The exact
serial-normalizer obstruction is recorded symbolically without allocating its
large all-ones witness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from torch import Tensor

from passive_muon.scalable_mixed_precision import serial_normalizer_obstruction
from passive_muon.scalable_mixed_precision_certificate import (
    EXACT_A,
    EXACT_B,
    EXACT_C,
    EXACT_RHO,
    FP32_A,
    FP32_B,
    FP32_C,
    FP32_RHO,
)

SCHEMA_VERSION: Final = "passive-muon-scalable-mixed-precision-diagnostic-v1"
SEED: Final = 20_260_902
STAGES: Final = 5
POLICIES: Final = (
    "single_final_bf16",
    "repeated_one_term_bf16",
    "compensated_high_low_bf16",
)
DEFAULT_SHAPES: Final = ((2, 2), (8, 16), (32, 64), (64, 128))
REALISTIC_SHAPES: Final = (
    (768, 768),
    (768, 3_072),
    (3_072, 12_288),
    (4_096, 11_008),
)
OBSTRUCTION_SHAPE: Final = (4_096, 11_008)

ROOT = Path(__file__).resolve().parents[2]
CANONICAL_TARGET = ROOT / "results" / "summaries" / "scalable_mixed_precision_diagnostic.json"
SOURCE_PATHS = (
    Path("experiments/mixed_precision/run_scalable_mixed_precision_diagnostic.py"),
    Path("src/passive_muon/scalable_mixed_precision.py"),
    Path("src/passive_muon/scalable_mixed_precision_certificate.py"),
    Path("tests/test_scalable_mixed_precision_diagnostic.py"),
    Path("pyproject.toml"),
    Path("uv.lock"),
)


def _validate_shape(shape: tuple[int, int]) -> None:
    if len(shape) != 2 or any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in shape
    ):
        raise ValueError("each shape must contain two positive integers")


@dataclass(frozen=True)
class DiagnosticConfig:
    """Complete deterministic diagnostic configuration."""

    seed: int = SEED
    shapes: tuple[tuple[int, int], ...] = DEFAULT_SHAPES
    include_realistic_shapes: bool = False
    torch_threads: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not self.shapes:
            raise ValueError("at least one executable shape is required")
        for shape in self.shapes:
            _validate_shape(shape)
        if len(set(self.shapes)) != len(self.shapes):
            raise ValueError("executable shapes must be unique")
        if not isinstance(self.include_realistic_shapes, bool):
            raise TypeError("include_realistic_shapes must be Boolean")
        if (
            not isinstance(self.torch_threads, int)
            or isinstance(self.torch_threads, bool)
            or self.torch_threads <= 0
        ):
            raise ValueError("torch_threads must be a positive integer")

    @property
    def resolved_shapes(self) -> tuple[tuple[int, int], ...]:
        shapes = list(self.shapes)
        if self.include_realistic_shapes:
            shapes.extend(shape for shape in REALISTIC_SHAPES if shape not in shapes)
        return tuple(shapes)


def _stable_normalize(matrix: Tensor) -> Tensor:
    """Scale-free unit max-floor normalization in the matrix dtype."""

    one = matrix.new_tensor(1.0)
    sigma = torch.maximum(one, torch.amax(torch.abs(matrix)))
    scaled = matrix / sigma
    scaled_norm = torch.linalg.vector_norm(scaled)
    denominator = torch.maximum(one / sigma, scaled_norm)
    return scaled / denominator


def _coefficients(dtype: torch.dtype) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    device = torch.device("cpu")
    if dtype == torch.float32:
        values = (FP32_A, FP32_B, FP32_C, FP32_RHO)
    elif dtype == torch.float64:
        values = (EXACT_A, EXACT_B, EXACT_C, EXACT_RHO)
    else:  # pragma: no cover - callers lock both choices
        raise ValueError("coefficient dtype must be float32 or float64")
    return tuple(torch.tensor(float(value), dtype=dtype, device=device) for value in values)  # type: ignore[return-value]


def _native_stage(state: Tensor, identity: Tensor, a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    """Evaluate one thick Horner stage with optimized native matmul."""

    gram = state @ state.mT
    shifted = c * gram + b * identity
    product = shifted @ gram
    affine = product + a * identity
    return affine @ state


def _one_term_boundary(value: Tensor) -> Tensor:
    return value.to(torch.bfloat16).to(torch.float32)


def _compensated_boundary(value: Tensor) -> Tensor:
    high = value.to(torch.bfloat16).to(torch.float32)
    residual = value - high
    low = residual.to(torch.bfloat16).to(torch.float32)
    return high + low


def evaluate_policy(signal: Tensor, policy: str) -> tuple[Tensor, Tensor]:
    """Return the FP32 polynomial and repaired output for one policy."""

    if policy not in POLICIES:
        raise ValueError(f"unknown boundary policy: {policy!r}")
    if signal.ndim != 2 or signal.dtype != torch.float32 or signal.device.type != "cpu":
        raise ValueError("signal must be one CPU FP32 matrix")
    if not bool(torch.isfinite(signal).all()):
        raise ValueError("signal must be finite")

    normalized = _stable_normalize(signal)
    transposed = signal.shape[0] > signal.shape[1]
    state = normalized.mT.contiguous() if transposed else normalized
    if policy == "repeated_one_term_bf16":
        state = _one_term_boundary(state)
    elif policy == "compensated_high_low_bf16":
        state = _compensated_boundary(state)

    a, b, c, rho = _coefficients(torch.float32)
    identity = torch.eye(state.shape[0], dtype=torch.float32, device="cpu")
    for stage in range(1, STAGES + 1):
        state = _native_stage(state, identity, a, b, c)
        if policy == "repeated_one_term_bf16":
            state = _one_term_boundary(state)
        elif policy == "compensated_high_low_bf16":
            state = _compensated_boundary(state)
        elif policy == "single_final_bf16" and stage == STAGES:
            state = _one_term_boundary(state)

    polynomial = state.mT.contiguous() if transposed else state
    repaired = polynomial + rho * signal
    return polynomial, repaired


def float64_target(signal: Tensor) -> tuple[Tensor, Tensor]:
    """Evaluate the exact-real formula in float64 numerical arithmetic."""

    if signal.ndim != 2 or signal.dtype != torch.float32 or signal.device.type != "cpu":
        raise ValueError("target input must be one CPU FP32 matrix")
    source = signal.to(torch.float64)
    state = _stable_normalize(source)
    transposed = source.shape[0] > source.shape[1]
    state = state.mT.contiguous() if transposed else state
    a, b, c, rho = _coefficients(torch.float64)
    identity = torch.eye(state.shape[0], dtype=torch.float64, device="cpu")
    for _ in range(STAGES):
        state = _native_stage(state, identity, a, b, c)
    polynomial = state.mT.contiguous() if transposed else state
    return polynomial, polynomial + rho * source


def _unit_frobenius(matrix: Tensor) -> Tensor:
    norm = torch.linalg.vector_norm(matrix)
    if float(norm) == 0.0:
        raise ValueError("cannot normalize a zero diagnostic template")
    return matrix / norm


def build_cases(
    shape: tuple[int, int], *, seed: int, shape_index: int
) -> tuple[tuple[str, Tensor], ...]:
    """Build four deterministic inputs for one shape."""

    _validate_shape(shape)
    rows, columns = shape
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 1_000_003 * shape_index)

    dense = _unit_frobenius(torch.randn((rows, columns), dtype=torch.float32, generator=generator))
    left = torch.randn(rows, dtype=torch.float32, generator=generator)
    right = torch.randn(columns, dtype=torch.float32, generator=generator)
    rank_one = 64.0 * _unit_frobenius(torch.outer(left, right))

    flat_spectrum = torch.zeros((rows, columns), dtype=torch.float32)
    diagonal = min(rows, columns)
    indices = torch.arange(diagonal)
    flat_spectrum[indices, indices] = float(1 / math.sqrt(diagonal))

    row_sign = torch.where(torch.arange(rows) % 2 == 0, 1.0, -1.0).to(torch.float32)
    column_sign = torch.where(torch.arange(columns) % 2 == 0, 1.0, -1.0).to(torch.float32)
    alternating = 0.5 * _unit_frobenius(torch.outer(row_sign, column_sign))
    return (
        ("seeded_dense_unit_frobenius", dense),
        ("seeded_rank_one_scale_64", rank_one),
        ("flat_spectrum_floor_boundary", flat_spectrum),
        ("alternating_rank_one_inside_floor", alternating),
    )


def _tensor_sha256(matrix: Tensor) -> str:
    contiguous = matrix.detach().contiguous().numpy()
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def _norm64(matrix: Tensor) -> float:
    return float(torch.linalg.vector_norm(matrix.to(torch.float64)))


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


def _provenance(config: DiagnosticConfig) -> dict[str, object]:
    return {
        "git": _git_state(),
        "seed": config.seed,
        "hardware": {
            "device": "cpu",
            "machine": platform.machine(),
            "processor": platform.processor() or "unreported",
            "platform": platform.platform(),
            "logical_cpu_count": __import__("os").cpu_count(),
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


def _case_record(
    *,
    shape: tuple[int, int],
    label: str,
    signal: Tensor,
    target_polynomial: Tensor,
    target_operator: Tensor,
) -> dict[str, object]:
    policy_records: dict[str, dict[str, object]] = {}
    target_operator_norm = _norm64(target_operator)
    for policy in POLICIES:
        polynomial, operator = evaluate_policy(signal, policy)
        polynomial_error = _norm64(polynomial.to(torch.float64) - target_polynomial)
        operator_error = _norm64(operator.to(torch.float64) - target_operator)
        policy_records[policy] = {
            "finite": bool(torch.isfinite(polynomial).all() and torch.isfinite(operator).all()),
            "polynomial_error_frobenius": polynomial_error,
            "operator_error_frobenius": operator_error,
            "operator_relative_error": operator_error
            / max(target_operator_norm, sys.float_info.min),
            "polynomial_frobenius": _norm64(polynomial),
            "operator_frobenius": _norm64(operator),
        }
    repeated_error = float(policy_records["repeated_one_term_bf16"]["operator_error_frobenius"])
    compensated_error = float(
        policy_records["compensated_high_low_bf16"]["operator_error_frobenius"]
    )
    return {
        "shape": list(shape),
        "label": label,
        "input": {
            "sha256_float32_bytes": _tensor_sha256(signal),
            "max_abs": float(torch.amax(torch.abs(signal))),
            "frobenius": _norm64(signal),
        },
        "float64_target": {
            "polynomial_frobenius": _norm64(target_polynomial),
            "operator_frobenius": target_operator_norm,
        },
        "policies": policy_records,
        "compensated_error_no_larger_than_repeated_one_term": compensated_error <= repeated_error,
    }


def _summary(cases: list[dict[str, object]]) -> dict[str, object]:
    policy_summary: dict[str, dict[str, object]] = {}
    for policy in POLICIES:
        records = [case["policies"][policy] for case in cases]  # type: ignore[index]
        errors = [float(record["operator_error_frobenius"]) for record in records]
        relatives = [float(record["operator_relative_error"]) for record in records]
        policy_summary[policy] = {
            "finite_count": sum(bool(record["finite"]) for record in records),
            "nonfinite_count": sum(not bool(record["finite"]) for record in records),
            "maximum_operator_error_frobenius": max(errors, default=0.0),
            "mean_operator_error_frobenius": sum(errors) / len(errors) if errors else 0.0,
            "maximum_operator_relative_error": max(relatives, default=0.0),
        }
    family_counts = Counter(str(case["label"]) for case in cases)
    compensated_wins = sum(
        bool(case["compensated_error_no_larger_than_repeated_one_term"]) for case in cases
    )
    return {
        "case_count": len(cases),
        "policy_evaluation_count": len(cases) * len(POLICIES),
        "shape_count": len({tuple(case["shape"]) for case in cases}),  # type: ignore[arg-type]
        "label_counts": dict(sorted(family_counts.items())),
        "policy_summary": policy_summary,
        "compensated_no_larger_than_repeated_count": compensated_wins,
        "compensated_no_larger_than_repeated_fraction": compensated_wins / len(cases)
        if cases
        else 0.0,
    }


def run_diagnostic(config: DiagnosticConfig) -> dict[str, object]:
    """Run the complete deterministic CPU diagnostic."""

    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(config.torch_threads)
    torch.use_deterministic_algorithms(True)
    try:
        cases: list[dict[str, object]] = []
        for shape_index, shape in enumerate(config.resolved_shapes):
            for label, signal in build_cases(shape, seed=config.seed, shape_index=shape_index):
                target_polynomial, target_operator = float64_target(signal)
                cases.append(
                    _case_record(
                        shape=shape,
                        label=label,
                        signal=signal,
                        target_polynomial=target_polynomial,
                        target_operator=target_operator,
                    )
                )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "claim_scope": {
                "status": "sampled CPU diagnostic; not a certificate",
                "native_matmul_caveat": (
                    "PyTorch optimized FP32 matmul is not the fixed balanced reduction in the "
                    "P9 theorem kernel; these observations do not inherit its proof"
                ),
                "target_caveat": "the float64 comparator is numerical and is not exact arithmetic",
                "policy_interpretation": (
                    "boundary policies are controlled comparisons with identical normalization, "
                    "stage formula, coefficients, repair, inputs, and native FP32 matmul"
                ),
            },
            "config": {
                **asdict(config),
                "shapes": [list(shape) for shape in config.shapes],
                "resolved_shapes": [list(shape) for shape in config.resolved_shapes],
                "policies": list(POLICIES),
                "stages": STAGES,
                "normalization": "max-floor at 1; scale-free native FP32 reduction; no epsilon",
                "epsilon": "none",
                "exact_coefficients": {
                    "a": str(EXACT_A),
                    "b": str(EXACT_B),
                    "c": str(EXACT_C),
                    "rho": str(EXACT_RHO),
                },
                "fp32_coefficients": {
                    "a": str(FP32_A),
                    "b": str(FP32_B),
                    "c": str(FP32_C),
                    "rho": str(FP32_RHO),
                },
                "stage_formula": "G=X@X.T; T=c32*G+b32*I; D=T@G; E=D+a32*I; Y=E@X",
                "input_cases_per_shape": 4,
            },
            "serial_normalizer_obstruction": {
                **serial_normalizer_obstruction(OBSTRUCTION_SHAPE),
                "allocated": False,
                "qualification": (
                    "exact symbolic arithmetic; the all-ones witness was not allocated"
                ),
            },
            "summary": _summary(cases),
            "cases": cases,
            "experiment_provenance": _provenance(config),
            "canonical_target": str(CANONICAL_TARGET.relative_to(ROOT)),
        }
        return payload
    finally:
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_deterministic)


def canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _parse_shape(value: str) -> tuple[int, int]:
    normalized = value.lower()
    parts = normalized.split("x")
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
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--shape",
        dest="shapes",
        action="append",
        type=_parse_shape,
        help="executable ROWSxCOLUMNS shape; repeat to replace the modest defaults",
    )
    parser.add_argument(
        "--include-realistic-shapes",
        action="store_true",
        help="also run expensive 768-4096 Transformer shapes (explicit opt-in)",
    )
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    config = DiagnosticConfig(
        seed=arguments.seed,
        shapes=tuple(arguments.shapes) if arguments.shapes else DEFAULT_SHAPES,
        include_realistic_shapes=arguments.include_realistic_shapes,
        torch_threads=arguments.torch_threads,
    )
    payload = run_diagnostic(config)
    rendered = canonical_json(payload)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
