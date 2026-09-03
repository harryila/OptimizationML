#!/usr/bin/env python3
"""Sampled falsification study for the locked P8 mixed-precision operator.

The float64 comparator evaluates the exact-real *formula* more accurately than
the locked FP32/BF16 implementation; it is not exact arithmetic and is not a
proof.  The global affine error guarantee comes only from the rational P8
certificate.  This experiment searches for counterexamples on a deterministic
adversarial set and a reproducible seeded random grid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import torch

from passive_muon.mixed_precision import (
    LOCKED_MATRIX_SHAPE,
    LOCKED_SAFE_MAX_ABS_POWER,
    MixedPrecisionConfig,
    mixed_precision_manifest,
    mixed_precision_repaired_operator,
)
from passive_muon.mixed_precision_certificate import (
    AFFINE_INTERCEPT_UPPER,
    AFFINE_SLOPE_UPPER,
    EXACT_A,
    EXACT_B,
    EXACT_C,
    EXACT_RHO,
)

SCHEMA_VERSION = "passive-muon-mixed-precision-falsification-v1"
ALL_REAL_ADAPTER_AFFINE_SLOPE_UPPER = Fraction(1, 5_000)
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "results" / "summaries" / "mixed_precision_falsification.json"
SOURCE_PATHS = (
    Path("experiments/mixed_precision/run_mixed_precision_falsification.py"),
    Path("src/passive_muon/mixed_precision.py"),
    Path("src/passive_muon/mixed_precision_certificate.py"),
    Path("src/passive_muon/specs.py"),
    Path("src/passive_muon/structure_aware_stability.py"),
    Path("tests/test_mixed_precision.py"),
    Path("tests/test_mixed_precision_experiment.py"),
)


@dataclass(frozen=True)
class FalsificationConfig:
    """Complete deterministic grid configuration."""

    seed: int = 2_026_090_2
    random_cases_per_family: int = 12
    random_exponent_min: int = -120
    random_exponent_max: int = 116
    matrix_shape: tuple[int, int] = LOCKED_MATRIX_SHAPE
    safe_max_abs_power: int = LOCKED_SAFE_MAX_ABS_POWER

    def __post_init__(self) -> None:
        if self.matrix_shape != LOCKED_MATRIX_SHAPE:
            raise ValueError("the executable falsification study is locked to shape (2, 2)")
        if self.safe_max_abs_power != LOCKED_SAFE_MAX_ABS_POWER:
            raise ValueError("safe_max_abs_power must match the locked implementation")
        if self.random_cases_per_family <= 0:
            raise ValueError("random_cases_per_family must be positive")
        if self.random_exponent_min < -120:
            raise ValueError("random_exponent_min must be at least -120")
        if self.random_exponent_max > self.safe_max_abs_power:
            raise ValueError("random_exponent_max exceeds the certified safe range")
        if self.random_exponent_min > self.random_exponent_max:
            raise ValueError("random exponent interval is empty")


def _serial_mm64(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    output = np.empty((2, 2), dtype=np.float64)
    for row in range(2):
        for column in range(2):
            first = np.float64(left[row, 0] * right[0, column])
            second = np.float64(left[row, 1] * right[1, column])
            output[row, column] = np.float64(first + second)
    return output


def _stable_norm64(matrix: np.ndarray) -> float:
    maximum = float(np.max(np.abs(matrix)))
    if maximum == 0.0:
        return 0.0
    scaled = matrix / maximum
    total = 0.0
    for value in scaled.reshape(-1):
        total += float(value) * float(value)
    return maximum * math.sqrt(total)


def float64_exact_real_formula(signal_fp32: np.ndarray) -> np.ndarray:
    """Evaluate the exact-real target formula in float64 (not exact arithmetic)."""

    signal = np.asarray(signal_fp32, dtype=np.float32).astype(np.float64)
    norm = _stable_norm64(signal)
    state = signal / max(1.0, norm)
    identity = np.eye(2, dtype=np.float64)
    a = float(EXACT_A)
    b = float(EXACT_B)
    c = float(EXACT_C)
    for _ in range(5):
        gram = _serial_mm64(state, state.T)
        t_matrix = b * identity + c * gram
        d_matrix = _serial_mm64(t_matrix, gram)
        e_matrix = a * identity + d_matrix
        state = _serial_mm64(e_matrix, state)
    return state + float(EXACT_RHO) * signal


def _adversarial_grid() -> list[tuple[str, str, np.ndarray]]:
    f32 = np.float32
    minimum_subnormal = np.array([1], dtype=np.uint32).view(np.float32)[0]
    just_above_half = np.nextafter(f32(0.5), f32(np.inf), dtype=np.float32)
    just_below_one = np.nextafter(f32(1.0), f32(0.0), dtype=np.float32)
    just_above_one = np.nextafter(f32(1.0), f32(np.inf), dtype=np.float32)
    cases: list[tuple[str, str, np.ndarray]] = [
        ("zero", "zero", np.zeros((2, 2), dtype=np.float32)),
        (
            "subnormal_signed",
            "subnormal",
            np.array(
                [[minimum_subnormal, -minimum_subnormal], [f32(0.0), minimum_subnormal]],
                dtype=np.float32,
            ),
        ),
        (
            "half_threshold_dense",
            "floor_threshold",
            np.array([[0.5, -0.5], [0.5, -0.5]], dtype=np.float32),
        ),
        (
            "just_above_half",
            "floor_threshold",
            np.diag(np.array([just_above_half, f32(0.0)], dtype=np.float32)),
        ),
        ("rank_one_floor", "floor_boundary", np.diag(np.array([1.0, 0.0], np.float32))),
        (
            "rank_one_dense_floor",
            "floor_boundary",
            np.full((2, 2), f32(0.5), dtype=np.float32),
        ),
        (
            "three_four_unit_direction",
            "floor_boundary",
            np.diag(np.array([0.6, 0.8], dtype=np.float32)),
        ),
        (
            "just_below_floor_rank_one",
            "floor_boundary",
            np.diag(np.array([just_below_one, 0.0], dtype=np.float32)),
        ),
        (
            "just_above_floor_rank_one",
            "floor_boundary",
            np.diag(np.array([just_above_one, 0.0], dtype=np.float32)),
        ),
    ]

    rank_one_base = np.array([[0.5, -1.0], [-0.25, 0.5]], dtype=np.float32)
    for scale in (f32(0.25), f32(1.0), f32(64.0), f32(2.0**60), f32(2.0**116)):
        cases.append((f"rank_one_scale_{float(scale).hex()}", "rank_one", rank_one_base * scale))

    identity = np.eye(2, dtype=np.float32)
    exchange = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    for scale in (f32(1.0), f32(64.0), f32(2.0**60), f32(2.0**116)):
        cases.append(
            (f"repeated_identity_{float(scale).hex()}", "repeated_singular", identity * scale)
        )
        cases.append(
            (f"repeated_exchange_{float(scale).hex()}", "repeated_singular", exchange * scale)
        )

    base = np.array([[3.0, -4.0], [0.5, 2.0]], dtype=np.float32)
    row_swapped = base[[1, 0], :].copy()
    column_swapped = base[:, [1, 0]].copy()
    column_signed = base * np.array([-1.0, 1.0], dtype=np.float32)
    variants = (
        ("base", base),
        ("negative", -base),
        ("transpose", base.T.copy()),
        ("row_swap", row_swapped),
        ("column_swap", column_swapped),
        ("column_sign", column_signed),
        ("row_swap_column_sign", row_swapped * np.array([-1.0, 1.0], np.float32)),
        ("negative_transpose", -base.T.copy()),
    )
    for name, matrix in variants:
        cases.append((f"signed_permutation_{name}", "signs_permutations", matrix.copy()))

    moderate_base = np.array([[1.0, -0.375], [0.625, 0.125]], dtype=np.float32)
    for scale in (f32(2.0**-20), f32(1.0), f32(1.0e6), f32(2.0**100)):
        cases.append((f"dense_scale_{float(scale).hex()}", "moderate_large", moderate_base * scale))
    return cases


def _random_grid(config: FalsificationConfig) -> list[tuple[str, str, np.ndarray]]:
    rng = np.random.default_rng(config.seed)
    cases: list[tuple[str, str, np.ndarray]] = []
    count = config.random_cases_per_family

    for index in range(count):
        mantissas = rng.integers(-1024, 1025, size=(2, 2), dtype=np.int32).astype(np.float32)
        mantissas /= np.float32(1024.0)
        exponent = int(rng.integers(config.random_exponent_min, config.random_exponent_max + 1))
        matrix = np.ldexp(mantissas, exponent).astype(np.float32)
        cases.append((f"seeded_dense_log_{index:03d}", "seeded_dense_log", matrix))

    boundary_templates = (
        np.diag(np.array([0.6, 0.8], dtype=np.float32)),
        np.full((2, 2), np.float32(0.5), dtype=np.float32),
        np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float32),
        np.array([[0.8, 0.0], [0.0, -0.6]], dtype=np.float32),
    )
    multipliers = np.array(
        [
            np.nextafter(np.float32(1.0), np.float32(0.0), dtype=np.float32),
            np.float32(1.0),
            np.nextafter(np.float32(1.0), np.float32(np.inf), dtype=np.float32),
        ],
        dtype=np.float32,
    )
    for index in range(count):
        template = boundary_templates[int(rng.integers(0, len(boundary_templates)))].copy()
        if bool(rng.integers(0, 2)):
            template = template.T.copy()
        signs = np.where(rng.integers(0, 2, size=(2, 2)), 1.0, -1.0).astype(np.float32)
        matrix = (template * signs * multipliers[int(rng.integers(0, 3))]).astype(np.float32)
        cases.append((f"seeded_near_floor_{index:03d}", "seeded_near_floor", matrix))

    for index in range(count):
        left = rng.integers(-8, 9, size=2, dtype=np.int32)
        right = rng.integers(-8, 9, size=2, dtype=np.int32)
        if not np.any(left):
            left[0] = 1
        if not np.any(right):
            right[0] = 1
        outer = np.outer(left, right).astype(np.float32)
        outer /= np.max(np.abs(outer))
        exponent = int(rng.integers(config.random_exponent_min, config.random_exponent_max + 1))
        matrix = np.ldexp(outer, exponent).astype(np.float32)
        cases.append((f"seeded_rank_one_{index:03d}", "seeded_rank_one", matrix))

    signed_permutations = (
        np.eye(2, dtype=np.float32),
        -np.eye(2, dtype=np.float32),
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float32),
    )
    for index in range(count):
        basis = signed_permutations[int(rng.integers(0, len(signed_permutations)))]
        exponent = int(rng.integers(config.random_exponent_min, config.random_exponent_max + 1))
        matrix = np.ldexp(basis, exponent).astype(np.float32)
        cases.append((f"seeded_repeated_singular_{index:03d}", "seeded_repeated_singular", matrix))
    return cases


def build_grid(config: FalsificationConfig) -> list[tuple[str, str, np.ndarray]]:
    """Return the complete ordered adversarial plus seeded grid."""

    return _adversarial_grid() + _random_grid(config)


def _input_record(matrix: np.ndarray) -> dict[str, object]:
    contiguous = np.ascontiguousarray(matrix, dtype=np.float32)
    bits = contiguous.view(np.uint32)
    return {
        "values": [[float(value) for value in row] for row in contiguous],
        "float32_bits_hex_row_major": [f"0x{int(value):08x}" for value in bits.reshape(-1)],
    }


def _grid_sha256(grid: list[tuple[str, str, np.ndarray]]) -> str:
    payload = [
        {
            "label": label,
            "family": family,
            "bits": _input_record(matrix)["float32_bits_hex_row_major"],
        }
        for label, family, matrix in grid
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _git_state() -> dict[str, object]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "sha": revision.stdout.strip() if revision.returncode == 0 else "unavailable",
        "branch": branch.stdout.strip() if branch.returncode == 0 else "unavailable",
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def _source_snapshot() -> dict[str, str]:
    return {
        str(path): hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in SOURCE_PATHS
        if (ROOT / path).is_file()
    }


def run_study(config: FalsificationConfig) -> dict[str, Any]:
    """Run the sampled study and return a JSON-serializable payload."""

    grid = build_grid(config)
    implementation_config = MixedPrecisionConfig()
    cases: list[dict[str, Any]] = []
    nonfinite_count = 0
    candidate_violation_count = 0
    for index, (label, family, matrix) in enumerate(grid):
        contiguous = np.ascontiguousarray(matrix, dtype=np.float32)
        signal = torch.from_numpy(contiguous.copy())
        with torch.no_grad():
            observed = mixed_precision_repaired_operator(signal, implementation_config)
        observed64 = observed.detach().cpu().numpy().astype(np.float64)
        reference64 = float64_exact_real_formula(contiguous)
        difference = observed64 - reference64
        error = _stable_norm64(difference)
        signal_norm = _stable_norm64(contiguous.astype(np.float64))
        affine_bound = float(AFFINE_SLOPE_UPPER) * signal_norm + float(AFFINE_INTERCEPT_UPPER)
        ratio = error / affine_bound
        adapter_affine_bound = float(ALL_REAL_ADAPTER_AFFINE_SLOPE_UPPER) * signal_norm + float(
            AFFINE_INTERCEPT_UPPER
        )
        adapter_ratio = error / adapter_affine_bound
        finite = bool(
            np.all(np.isfinite(observed64))
            and np.all(np.isfinite(reference64))
            and math.isfinite(error)
            and math.isfinite(ratio)
        )
        candidate_violation = (not finite) or error > affine_bound
        nonfinite_count += int(not finite)
        candidate_violation_count += int(candidate_violation)
        cases.append(
            {
                "index": index,
                "label": label,
                "family": family,
                "input": _input_record(contiguous),
                "input_frobenius_norm_float64": signal_norm,
                "observed_error_frobenius": error,
                "certified_affine_bound_float64": affine_bound,
                "affine_bound_ratio": ratio,
                "all_real_adapter_affine_bound_float64": adapter_affine_bound,
                "all_real_adapter_affine_bound_ratio": adapter_ratio,
                "finite": finite,
                "candidate_violation": candidate_violation,
            }
        )

    maximum_error_case = max(cases, key=lambda case: case["observed_error_frobenius"])
    maximum_ratio_case = max(cases, key=lambda case: case["affine_bound_ratio"])
    maximum_adapter_ratio_case = max(
        cases, key=lambda case: case["all_real_adapter_affine_bound_ratio"]
    )
    family_counts: dict[str, int] = {}
    for case in cases:
        family = str(case["family"])
        family_counts[family] = family_counts.get(family, 0) + 1

    return {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": {
            "kind": "sampled executable falsification only",
            "qualification": (
                "Zero candidates on this finite grid do not prove the global affine bound; "
                "the exact rational P8 certificate is authoritative."
            ),
            "reference_qualification": (
                "The comparator is a float64 evaluation of the exact-real target formula, "
                "not exact arithmetic and not a proof oracle."
            ),
            "intercept_note": (
                "The certified 347/100 = 3.47 intercept is intentionally conservative; "
                "observed tightness is not claimed."
            ),
            "adapter_note": (
                "Primary ratios use the sharper theorem for binary32 inputs. The all-real "
                "adapter envelope accounts for rounding an arbitrary real input to binary32; "
                "its looser bound is evaluated here on the same binary32 sample grid."
            ),
        },
        "config": {
            **asdict(config),
            "matrix_shape": list(config.matrix_shape),
            "random_families": [
                "seeded_dense_log",
                "seeded_near_floor",
                "seeded_rank_one",
                "seeded_repeated_singular",
            ],
            "random_generator": "numpy.random.Generator(PCG64)",
            "random_integer_mantissa_grid": "signed integers divided by 1024",
            "adversarial_grid_is_explicit_in_cases": True,
        },
        "operator": {
            "implementation_manifest": mixed_precision_manifest(implementation_config),
            "target": (
                "five exact-coefficient Jordan stages after max(1, Frobenius norm) floor, "
                "plus exact-rho linear repair"
            ),
            "target_evaluation_dtype": "float64",
            "target_evaluation_is_exact": False,
            "affine_slope_exact": str(AFFINE_SLOPE_UPPER),
            "affine_intercept_exact": str(AFFINE_INTERCEPT_UPPER),
            "all_real_adapter_affine_slope_exact": str(ALL_REAL_ADAPTER_AFFINE_SLOPE_UPPER),
            "all_real_adapter_affine_intercept_exact": str(AFFINE_INTERCEPT_UPPER),
        },
        "grid": {
            "ordered_case_count": len(grid),
            "grid_sha256_from_labels_families_and_input_bits": _grid_sha256(grid),
            "family_counts": dict(sorted(family_counts.items())),
            "coverage": [
                "zero and signed float32 subnormals",
                "normalizer half-threshold and Frobenius-floor boundary",
                "rank-one matrices",
                "repeated singular values",
                "sign, transpose, row, and column permutations",
                "moderate and large scales through the inclusive 2^116 limit",
                "seeded dyadic dense, near-floor, rank-one, and repeated-singular cases",
            ],
        },
        "summary": {
            "case_count": len(cases),
            "candidate_violation_count": candidate_violation_count,
            "nonfinite_count": nonfinite_count,
            "maximum_observed_error_frobenius": maximum_error_case["observed_error_frobenius"],
            "maximum_observed_error_case": maximum_error_case["label"],
            "maximum_affine_bound_ratio": maximum_ratio_case["affine_bound_ratio"],
            "maximum_affine_bound_ratio_case": maximum_ratio_case["label"],
            "maximum_all_real_adapter_affine_bound_ratio": maximum_adapter_ratio_case[
                "all_real_adapter_affine_bound_ratio"
            ],
            "maximum_all_real_adapter_affine_bound_ratio_case": maximum_adapter_ratio_case["label"],
        },
        "cases": cases,
        "experiment_provenance": {
            "seed": config.seed,
            "git": _git_state(),
            "hardware": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "torch_device": "cpu",
                "torch_threads": torch.get_num_threads(),
            },
            "software": {
                "python": sys.version,
                "numpy": np.__version__,
                "torch": str(torch.__version__),
            },
            "source_snapshot": _source_snapshot(),
        },
    }


def canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=FalsificationConfig.seed)
    parser.add_argument(
        "--random-cases-per-family",
        type=int,
        default=FalsificationConfig.random_cases_per_family,
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    config = FalsificationConfig(
        seed=arguments.seed,
        random_cases_per_family=arguments.random_cases_per_family,
    )
    payload = run_study(config)
    rendered = canonical_json(payload)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered)
    sys.stdout.write(rendered)


if __name__ == "__main__":
    main()
