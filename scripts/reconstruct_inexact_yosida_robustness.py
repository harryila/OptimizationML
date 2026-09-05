#!/usr/bin/env python3
"""Independently reconstruct the exact P15 inexact-Yosida certificate.

This script uses only the Python standard library.  It rebuilds the residual
conversion, stopping-rule decomposition, normalized five-coordinate dynamics,
directed smooth interpolation supplies, exact LMI, tolerance controls, and
Sylvester minors before reading an optional canonical artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANONICAL = ROOT / "results/summaries/inexact_yosida_robustness_certificate.json"
SCHEMA_VERSION = "passive-muon-inexact-yosida-robustness-certificate-v1"
RECONSTRUCTION_SCHEMA_VERSION = "passive-muon-p15-independent-reconstruction-v1"

EPSILON = Fraction(1, 10_000_000)
RESOLVENT_PARAMETER = Fraction(1, 1_000)
BASE_STRONG_MONOTONICITY = Fraction(1_000)
FORWARD_STRONG_MONOTONICITY = Fraction(2)
OUTPUT_ERROR_GAIN = Fraction(500)
RELATIVE_RESIDUAL = Fraction(1, 250)
BASE_RESIDUAL_RADIUS = Fraction(250)
EFFECTIVE_RESIDUAL_RADIUS = Fraction(252)
ABSOLUTE_OUTPUT_PENALTY = Fraction(1, 100_000)
ABSOLUTE_FORCING_GAIN = Fraction(5, 2)

PL_CONSTANT = Fraction(1)
SMOOTHNESS = Fraction(10)
BETA = Fraction(19, 20)
LEARNING_RATE = Fraction(1, 32_000)
CENTER_GAIN = Fraction(750)
TAU = Fraction(499, 500)
RATE = Fraction(249_001, 250_000)

STORAGE = (
    (Fraction(674_389, 1_000_000), Fraction(-73_827, 1_000_000)),
    (Fraction(-73_827, 1_000_000), Fraction(12_368, 1_000_000)),
)
FUNCTION_STORAGE = Fraction(313_243, 1_000_000)
INTERPOLATION_REVERSE = Fraction(622_414, 1_000_000)
RESIDUAL_MULTIPLIER = Fraction(27_530, 1_000_000)

REJECTED_RELATIVE_RESIDUAL = Fraction(3, 500)
REJECTED_EFFECTIVE_RADIUS = Fraction(253)
INSUFFICIENT_ABSOLUTE_OUTPUT_PENALTY = Fraction(1, 1_000_000)
CRITICAL_ABSOLUTE_OUTPUT_PENALTY = Fraction(
    7_167_838_551_819_102_967_459_883_001_403_922_668_449_707_353,
    1_283_935_131_617_993_795_632_989_901_464_334_924_254_261_960_000_000,
)

P14_ARTIFACT_PATH = "results/summaries/yosida_stability_certificate.json"
P14_ARTIFACT_SHA256 = "8f31416eb8e278bc8a3765f72b528f5f538e33f28d5baedc5888c5d849447c62"
P14_SOURCE_COMMIT = "ea18aa856e3af6d025e325a6db27a681f09c0e7a"
P14_ARTIFACT_COMMIT = "e323a4d1e73050fb6b6d8bcd09c515ea30de3553"
P14_CHECKPOINT_TAG = "p14-yosida-stability-checkpoint"
P14_CHECKPOINT_TAG_OBJECT = "986fade33e4bd90bed091798d2415d1e0ab608a2"

EXPECTED_SOURCE_PATHS = (
    ".github/workflows/ci.yml",
    ".github/workflows/p15-inexact-yosida-robustness.yml",
    "scripts/certify_inexact_yosida_robustness.py",
    "scripts/reconstruct_inexact_yosida_robustness.py",
    "src/passive_muon/inexact_yosida_robustness.py",
    "src/passive_muon/yosida_stability.py",
    "src/passive_muon/pl_convergence.py",
    "src/passive_muon/momentum_iqc.py",
    "src/passive_muon/additive_epsilon_deficit.py",
    "src/passive_muon/radial_passivation_tradeoff.py",
    "src/passive_muon/specs.py",
    "src/passive_muon/structure_aware_stability.py",
    "src/passive_muon/upstream_momentum.py",
    "tests/test_inexact_yosida_robustness.py",
    "tests/test_inexact_yosida_robustness_cli.py",
    "tests/test_inexact_yosida_robustness_reconstruction.py",
    "tests/test_inexact_yosida_robustness_result_manifest.py",
    "tests/test_current_research_index.py",
    "theory/inexact_yosida_robustness_certificate.md",
    "theory/audits/P15_INEXACT_YOSIDA_ROBUSTNESS_HUMAN_PROOF_AUDIT.md",
    "results/summaries/P15_INEXACT_YOSIDA_ROBUSTNESS_RESULTS.md",
    "third_party/UPSTREAM_COMMITS.md",
    "pyproject.toml",
    "uv.lock",
)

Vector = tuple[Fraction, ...]
Matrix = tuple[tuple[Fraction, ...], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--require-canonical", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _add(*matrices: Matrix) -> Matrix:
    return tuple(
        tuple(
            sum((matrix[row][column] for matrix in matrices), Fraction(0))
            for column in range(len(matrices[0][0]))
        )
        for row in range(len(matrices[0]))
    )


def _scale(scale: Fraction, matrix: Matrix) -> Matrix:
    return tuple(tuple(scale * value for value in row) for row in matrix)


def _outer(left: Vector, right: Vector) -> Matrix:
    return tuple(tuple(a * b for b in right) for a in left)


def _symmetric_outer(left: Vector, right: Vector) -> Matrix:
    return _scale(Fraction(1, 2), _add(_outer(left, right), _outer(right, left)))


def _vector_add(*vectors: Vector) -> Vector:
    return tuple(
        sum((vector[index] for vector in vectors), Fraction(0)) for index in range(len(vectors[0]))
    )


def _vector_scale(scale: Fraction, vector: Vector) -> Vector:
    return tuple(scale * value for value in vector)


def _transpose(matrix: Matrix) -> Matrix:
    return tuple(tuple(row[index] for row in matrix) for index in range(len(matrix[0])))


def _multiply(left: Matrix, right: Matrix) -> Matrix:
    return tuple(
        tuple(
            sum(
                (left[row][inner] * right[inner][column] for inner in range(len(right))),
                Fraction(0),
            )
            for column in range(len(right[0]))
        )
        for row in range(len(left))
    )


def _gram(rows: tuple[Vector, Vector], matrix: Matrix) -> Matrix:
    return _multiply(_transpose(rows), _multiply(matrix, rows))


def _directed_interpolation(
    input_i: Vector,
    gradient_i: Vector,
    input_j: Vector,
    gradient_j: Vector,
) -> Matrix:
    dx = _vector_add(input_i, _vector_scale(Fraction(-1), input_j))
    du = _vector_add(gradient_i, _vector_scale(Fraction(-1), gradient_j))
    model = _scale(
        Fraction(1, 4),
        _add(
            _outer(du, du),
            _scale(Fraction(2), _symmetric_outer(dx, du)),
            _scale(Fraction(-1), _outer(dx, dx)),
        ),
    )
    return _add(
        _scale(Fraction(-1), _symmetric_outer(gradient_j, dx)),
        _scale(Fraction(-1), model),
    )


def _determinant(matrix: Matrix) -> Fraction:
    work = [list(row) for row in matrix]
    determinant = Fraction(1)
    for pivot_index in range(len(work)):
        pivot_row = next(
            (row for row in range(pivot_index, len(work)) if work[row][pivot_index]),
            None,
        )
        if pivot_row is None:
            return Fraction(0)
        if pivot_row != pivot_index:
            work[pivot_index], work[pivot_row] = work[pivot_row], work[pivot_index]
            determinant = -determinant
        pivot = work[pivot_index][pivot_index]
        determinant *= pivot
        for row in range(pivot_index + 1, len(work)):
            factor = work[row][pivot_index] / pivot
            for column in range(pivot_index + 1, len(work)):
                work[row][column] -= factor * work[pivot_index][column]
    return determinant


def _leading_minors(matrix: Matrix) -> tuple[Fraction, ...]:
    return tuple(
        _determinant(
            tuple(tuple(matrix[row][column] for column in range(size)) for row in range(size))
        )
        for size in range(1, len(matrix) + 1)
    )


def _lmi_for(
    *,
    residual_radius: Fraction,
    absolute_output_penalty: Fraction | None,
) -> dict[str, object]:
    dimension = 5 if absolute_output_penalty is not None else 4
    basis = tuple(
        tuple(Fraction(row == column) for row in range(dimension)) for column in range(dimension)
    )
    momentum, gradient, residual, gradient_next = basis[:4]
    absolute_error = basis[4] if dimension == 5 else (Fraction(0),) * dimension
    null = (Fraction(0),) * dimension

    alpha = LEARNING_RATE * CENTER_GAIN * SMOOTHNESS
    residual_ratio = residual_radius / CENTER_GAIN
    condition_ratio = PL_CONSTANT / SMOOTHNESS
    signal = _vector_add(
        _vector_scale(BETA**2, momentum),
        _vector_scale(1 - BETA**2, gradient),
    )
    step = _vector_add(
        _vector_scale(-alpha, signal),
        _vector_scale(-alpha * residual_ratio, residual),
        _vector_scale(-alpha, absolute_error),
    )
    momentum_next = _vector_add(
        _vector_scale(BETA, momentum),
        _vector_scale(1 - BETA, gradient),
    )
    transition = (momentum_next, gradient_next)
    selection = (momentum, gradient)
    interpolation_12 = _directed_interpolation(null, gradient, step, gradient_next)
    interpolation_21 = _directed_interpolation(step, gradient_next, null, gradient)
    residual_supply = _add(
        _outer(signal, signal),
        _scale(Fraction(-1), _outer(residual, residual)),
    )
    absolute_error_norm = _outer(absolute_error, absolute_error)

    lambda_21 = INTERPOLATION_REVERSE
    lambda_12 = lambda_21 + FUNCTION_STORAGE * RATE
    lambda_pl = FUNCTION_STORAGE * (1 - RATE) / (2 * condition_ratio)
    terms = [
        _gram(transition, STORAGE),
        _scale(-RATE, _gram(selection, STORAGE)),
        _scale(lambda_12, interpolation_12),
        _scale(lambda_21, interpolation_21),
        _scale(lambda_pl, _outer(gradient_next, gradient_next)),
        _scale(RESIDUAL_MULTIPLIER, residual_supply),
    ]
    normalized_absolute_penalty = None
    if absolute_output_penalty is not None:
        normalized_absolute_penalty = absolute_output_penalty * (CENTER_GAIN * SMOOTHNESS) ** 2
        terms.append(_scale(-normalized_absolute_penalty, absolute_error_norm))
    lmi = _add(*terms)
    negative_minors = _leading_minors(_scale(Fraction(-1), lmi))
    return {
        "dimensionless_step": alpha,
        "residual_ratio": residual_ratio,
        "condition_ratio": condition_ratio,
        "transition": transition,
        "selection": selection,
        "signal": signal,
        "step": step,
        "interpolation_12": interpolation_12,
        "interpolation_21": interpolation_21,
        "pl_next": _outer(gradient_next, gradient_next),
        "residual_supply": residual_supply,
        "absolute_error_norm": absolute_error_norm,
        "normalized_absolute_penalty": normalized_absolute_penalty,
        "lambda_12": lambda_12,
        "lambda_21": lambda_21,
        "lambda_pl": lambda_pl,
        "lambda_residual": RESIDUAL_MULTIPLIER,
        "lmi": lmi,
        "negative_minors": negative_minors,
        "function_coefficients": (
            lambda_12 - lambda_21,
            -lambda_12 + lambda_21 - 2 * condition_ratio * lambda_pl,
        ),
        "expected_function_coefficients": (FUNCTION_STORAGE * RATE, -FUNCTION_STORAGE),
    }


def _fraction_strings(values: tuple[Fraction, ...]) -> list[str]:
    return [str(value) for value in values]


def _matrix_strings(matrix: Matrix) -> list[list[str]]:
    return [[str(value) for value in row] for row in matrix]


def _vector_strings(vector: Vector) -> list[str]:
    return [str(value) for value in vector]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_output(*arguments: str) -> str | None:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _git_blob_sha256(commit: str, path: str) -> str:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"could not read frozen source {commit}:{path}")
    return hashlib.sha256(completed.stdout).hexdigest()


def _source_snapshot_for(canonical: dict[str, object]) -> dict[str, str]:
    expected = canonical["proof_replay_provenance"]["source_snapshot"]
    if set(expected) != set(EXPECTED_SOURCE_PATHS):
        raise RuntimeError("source snapshot path set does not match the P15 lock")
    git = canonical["git"]
    if git["dirty"] is False:
        return {path: _git_blob_sha256(git["sha"], path) for path in expected}
    return {path: _sha256(ROOT / path) for path in expected}


def _reconstruct() -> dict[str, object]:
    selected = _lmi_for(
        residual_radius=EFFECTIVE_RESIDUAL_RADIUS,
        absolute_output_penalty=ABSOLUTE_OUTPUT_PENALTY,
    )
    radius_rejection = _lmi_for(
        residual_radius=REJECTED_EFFECTIVE_RADIUS,
        absolute_output_penalty=None,
    )
    insufficient_penalty = _lmi_for(
        residual_radius=EFFECTIVE_RESIDUAL_RADIUS,
        absolute_output_penalty=INSUFFICIENT_ABSOLUTE_OUTPUT_PENALTY,
    )
    critical_penalty = _lmi_for(
        residual_radius=EFFECTIVE_RESIDUAL_RADIUS,
        absolute_output_penalty=CRITICAL_ABSOLUTE_OUTPUT_PENALTY,
    )
    storage_minors = _leading_minors(STORAGE)
    ultimate_storage_gain = ABSOLUTE_FORCING_GAIN / (1 - RATE)
    ultimate_objective_gain = SMOOTHNESS * ultimate_storage_gain / FUNCTION_STORAGE

    # Abstract stopping-rule controls use A=0, hence B=1000*I and J=s/2.
    # With r=-kappa*s, u_hat=(1+kappa)*s/2 and
    # Y_hat=500*(1-kappa)*s.
    abstract_boundary_kappa = Fraction(1)
    abstract_boundary_slope = 500 * (1 - abstract_boundary_kappa)
    abstract_failure_kappa = Fraction(2)
    abstract_failure_slope = 500 * (1 - abstract_failure_kappa)
    abstract_failure_p_at_one = LEARNING_RATE * abstract_failure_slope * (1 - BETA)

    checks = {
        "forward_map_is_two_strong": (
            1 + RESOLVENT_PARAMETER * BASE_STRONG_MONOTONICITY == FORWARD_STRONG_MONOTONICITY
        ),
        "equation_to_output_gain_is_500": (
            1 / (RESOLVENT_PARAMETER * FORWARD_STRONG_MONOTONICITY) == OUTPUT_ERROR_GAIN
        ),
        "relative_output_error_is_two": OUTPUT_ERROR_GAIN * RELATIVE_RESIDUAL == 2,
        "effective_radius_is_252": (
            BASE_RESIDUAL_RADIUS + OUTPUT_ERROR_GAIN * RELATIVE_RESIDUAL
            == EFFECTIVE_RESIDUAL_RADIUS
        ),
        "absolute_forcing_gain_is_five_halves": (
            ABSOLUTE_OUTPUT_PENALTY * OUTPUT_ERROR_GAIN**2 == ABSOLUTE_FORCING_GAIN
        ),
        "rate_is_strict": RATE == TAU**2 < 1,
        "storage_is_positive_definite": all(value > 0 for value in storage_minors),
        "selected_five_by_five_lmi_is_strict": all(
            value > 0 for value in selected["negative_minors"]
        ),
        "function_values_cancel": (
            selected["function_coefficients"] == selected["expected_function_coefficients"]
        ),
        "frozen_radius_253_certificate_is_rejected": (radius_rejection["negative_minors"][-1] < 0),
        "selected_penalty_exceeds_exact_boundary": (
            ABSOLUTE_OUTPUT_PENALTY > CRITICAL_ABSOLUTE_OUTPUT_PENALTY
        ),
        "insufficient_penalty_is_below_boundary": (
            INSUFFICIENT_ABSOLUTE_OUTPUT_PENALTY < CRITICAL_ABSOLUTE_OUTPUT_PENALTY
        ),
        "critical_penalty_has_zero_determinant": critical_penalty["negative_minors"][-1] == 0,
        "insufficient_penalty_fails": insufficient_penalty["negative_minors"][-1] < 0,
        "abstract_kappa_one_stalls": abstract_boundary_slope == 0,
        "abstract_kappa_two_is_ascent": (
            abstract_failure_slope == -500 and abstract_failure_p_at_one == Fraction(-1, 1280)
        ),
        "p14_artifact_hash_matches": (_sha256(ROOT / P14_ARTIFACT_PATH) == P14_ARTIFACT_SHA256),
        "p14_checkpoint_tag_matches": (
            _git_output("rev-parse", P14_CHECKPOINT_TAG) == P14_CHECKPOINT_TAG_OBJECT
            and _git_output("rev-parse", f"{P14_CHECKPOINT_TAG}^{{}}") == P14_ARTIFACT_COMMIT
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"independent P15 reconstruction failed: {failed}")

    return {
        "checks": checks,
        "parameters": {
            "epsilon": str(EPSILON),
            "lambda": str(RESOLVENT_PARAMETER),
            "mu": str(BASE_STRONG_MONOTONICITY),
            "beta": str(BETA),
            "eta": str(LEARNING_RATE),
            "smoothness": str(SMOOTHNESS),
            "pl_constant": str(PL_CONSTANT),
            "tau": str(TAU),
            "q": str(RATE),
        },
        "residual_envelope": {
            "forward_strong_monotonicity": str(FORWARD_STRONG_MONOTONICITY),
            "equation_to_output_gain": str(OUTPUT_ERROR_GAIN),
            "relative_residual": str(RELATIVE_RESIDUAL),
            "base_centered_radius": str(BASE_RESIDUAL_RADIUS),
            "relative_output_radius": str(OUTPUT_ERROR_GAIN * RELATIVE_RESIDUAL),
            "effective_centered_radius": str(EFFECTIVE_RESIDUAL_RADIUS),
            "absolute_output_from_rbar": str(OUTPUT_ERROR_GAIN),
        },
        "certificate": {
            "storage": _matrix_strings(STORAGE),
            "function_storage": str(FUNCTION_STORAGE),
            "interpolation_reverse": str(INTERPOLATION_REVERSE),
            "lambda_12": str(selected["lambda_12"]),
            "lambda_21": str(selected["lambda_21"]),
            "lambda_pl_next": str(selected["lambda_pl"]),
            "lambda_residual": str(selected["lambda_residual"]),
            "absolute_output_penalty": str(ABSOLUTE_OUTPUT_PENALTY),
            "normalized_absolute_output_penalty": str(selected["normalized_absolute_penalty"]),
            "absolute_forcing_gain": str(ABSOLUTE_FORCING_GAIN),
            "ultimate_storage_gain": str(ultimate_storage_gain),
            "ultimate_objective_gain": str(ultimate_objective_gain),
            "transition": _matrix_strings(selected["transition"]),
            "selection": _matrix_strings(selected["selection"]),
            "signal": _vector_strings(selected["signal"]),
            "step": _vector_strings(selected["step"]),
            "interpolation_12": _matrix_strings(selected["interpolation_12"]),
            "interpolation_21": _matrix_strings(selected["interpolation_21"]),
            "pl_next": _matrix_strings(selected["pl_next"]),
            "residual_supply": _matrix_strings(selected["residual_supply"]),
            "absolute_error_norm": _matrix_strings(selected["absolute_error_norm"]),
            "lmi": _matrix_strings(selected["lmi"]),
            "storage_minors": _fraction_strings(storage_minors),
            "negative_lmi_minors": _fraction_strings(selected["negative_minors"]),
            "function_coefficients": _fraction_strings(selected["function_coefficients"]),
            "expected_function_coefficients": _fraction_strings(
                selected["expected_function_coefficients"]
            ),
        },
        "controls": {
            "rejected_relative_residual": str(REJECTED_RELATIVE_RESIDUAL),
            "rejected_effective_radius": str(REJECTED_EFFECTIVE_RADIUS),
            "rejected_radius_negative_minors": _fraction_strings(
                radius_rejection["negative_minors"]
            ),
            "critical_absolute_output_penalty": str(CRITICAL_ABSOLUTE_OUTPUT_PENALTY),
            "insufficient_absolute_output_penalty": str(INSUFFICIENT_ABSOLUTE_OUTPUT_PENALTY),
            "insufficient_penalty_negative_minors": _fraction_strings(
                insufficient_penalty["negative_minors"]
            ),
            "critical_penalty_determinant": str(critical_penalty["negative_minors"][-1]),
            "abstract_stall_kappa": str(abstract_boundary_kappa),
            "abstract_stall_yosida_slope": str(abstract_boundary_slope),
            "abstract_ascent_kappa": str(abstract_failure_kappa),
            "abstract_ascent_yosida_slope": str(abstract_failure_slope),
            "abstract_ascent_p_at_one": str(abstract_failure_p_at_one),
        },
        "prior": {
            "artifact_path": P14_ARTIFACT_PATH,
            "artifact_sha256": P14_ARTIFACT_SHA256,
            "source_commit": P14_SOURCE_COMMIT,
            "artifact_commit": P14_ARTIFACT_COMMIT,
            "checkpoint_tag": P14_CHECKPOINT_TAG,
            "checkpoint_tag_object": P14_CHECKPOINT_TAG_OBJECT,
        },
    }


def _compare_canonical(
    canonical_path: Path,
    reconstruction: dict[str, object],
) -> dict[str, object]:
    if not canonical_path.exists():
        return {"status": "not_found", "comparisons": {}, "all_exact_fields_match": False}
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    comparisons = {
        "schema": canonical["schema_version"] == SCHEMA_VERSION,
        "parameters": canonical["reconstruction_fields"]["parameters"]
        == reconstruction["parameters"],
        "residual_envelope": canonical["reconstruction_fields"]["residual_envelope"]
        == reconstruction["residual_envelope"],
        "certificate": canonical["reconstruction_fields"]["certificate"]
        == reconstruction["certificate"],
        "controls": canonical["reconstruction_fields"]["controls"] == reconstruction["controls"],
        "prior": canonical["prior_p14"] == reconstruction["prior"],
        "source_snapshot": canonical["proof_replay_provenance"]["source_snapshot"]
        == _source_snapshot_for(canonical),
        "all_primary_checks": all(canonical["audit"]["checks"].values()),
        "scope": (
            canonical["claim_scope"]["arithmetic_model"] == "exact real arithmetic"
            and canonical["claim_scope"]["matrix_domain"]
            == "R^(m x n) for every fixed finite positive m,n"
            and canonical["claim_scope"]["objective_class"]
            == "differentiable globally 10-smooth, global PL constant 1"
        ),
    }
    return {
        "status": "matched" if all(comparisons.values()) else "mismatch",
        "comparisons": comparisons,
        "all_exact_fields_match": all(comparisons.values()),
    }


def main() -> int:
    args = parse_args()
    reconstruction = _reconstruct()
    comparison = _compare_canonical(args.canonical, reconstruction)
    if args.require_canonical and comparison["status"] != "matched":
        raise SystemExit("canonical P15 certificate is missing or mismatched")
    payload = {
        "schema_version": RECONSTRUCTION_SCHEMA_VERSION,
        "implementation_scope": {
            "project_package_imported": False,
            "numerical_library_imported": False,
            "canonical_read_order": "only after complete independent reconstruction",
        },
        "reconstruction": reconstruction,
        "canonical_comparison": comparison,
        "all_internal_exact_checks_passed": all(reconstruction["checks"].values()),
        "all_exact_checks_passed": (
            all(reconstruction["checks"].values()) and comparison["status"] == "matched"
        ),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
