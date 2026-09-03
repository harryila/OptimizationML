#!/usr/bin/env python3
"""Independently reconstruct the exact P14 Yosida stability certificate.

Only the Python standard library is imported.  The Yosida sector, normalized
dynamics, directed smooth interpolation LMI, Sylvester minors, and scalar
boundary controls are rebuilt from literal rational constants before the
canonical JSON is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANONICAL = ROOT / "results/summaries/yosida_stability_certificate.json"
SCHEMA_VERSION = "passive-muon-yosida-stability-certificate-v1"
RECONSTRUCTION_SCHEMA_VERSION = "passive-muon-p14-independent-reconstruction-v1"

EPSILON = Fraction(1, 10_000_000)
LAMBDA = Fraction(1, 1_000)
MU = Fraction(1_000)
BETA = Fraction(19, 20)
ETA = Fraction(1, 32_000)
SMOOTHNESS = Fraction(10)
PL_CONSTANT = Fraction(1)
TAU = Fraction(499, 500)
D14 = Fraction(0)

P13_UNIT_ORIGIN_SLOPE = Fraction(
    66_983_030_848_166_374_889_967_883,
    104_104_544_000_000_000_000_000,
)
P13_DIRECT_JURY_MARGIN = Fraction(
    -1_942_248_049_655_000_871_809_068_607,
    66_626_908_160_000_000_000_000,
)
BAD_LAMBDA = Fraction(1, 100_000)

STORAGE = (
    (Fraction(674_389, 1_000_000), Fraction(-73_827, 1_000_000)),
    (Fraction(-73_827, 1_000_000), Fraction(12_368, 1_000_000)),
)
FUNCTION_STORAGE = Fraction(313_243, 1_000_000)
INTERPOLATION_REVERSE = Fraction(622_414, 1_000_000)
RESIDUAL_MULTIPLIER = Fraction(27_530, 1_000_000)

P13_ARTIFACT_PATH = "results/summaries/radial_passivation_tradeoff_certificate.json"
P13_ARTIFACT_SHA256 = "3541bf2478178bc7487d23dbad36b6b7cb05bfb00c82e64ca7c5386b054caed9"
P13_SOURCE_COMMIT = "ef566f7d04a5c2ef1a69def69724e84c67702008"
P13_ARTIFACT_COMMIT = "31710a9a47ecd134fecfc6fea93deba27b5f40b8"
P13_CHECKPOINT_TAG = "p13-radial-passivation-checkpoint"
P13_CHECKPOINT_TAG_OBJECT = "9d42f5d18bef47de26dded7809214dfdc685c5dd"
JORDAN_COEFFICIENTS = {"a": "6889/2000", "b": "-191/40", "c": "4063/2000"}
PINNED_UPSTREAM_REVISION = "f98f1cacc0263b04290753e32be8d498c1efc806"
PINNED_UPSTREAM_FILE_SHA256 = "2479665a90124f62e4df557816665851ca317e42fcfda2af1da02c1f44ab5f3d"

EXPECTED_SOURCE_PATHS = (
    ".github/workflows/ci.yml",
    ".github/workflows/p14-yosida-stability.yml",
    "scripts/certify_yosida_stability.py",
    "scripts/reconstruct_yosida_stability.py",
    "src/passive_muon/additive_epsilon_deficit.py",
    "src/passive_muon/momentum_iqc.py",
    "src/passive_muon/yosida_stability.py",
    "src/passive_muon/pl_convergence.py",
    "src/passive_muon/radial_passivation_tradeoff.py",
    "src/passive_muon/specs.py",
    "src/passive_muon/structure_aware_stability.py",
    "src/passive_muon/upstream_momentum.py",
    "tests/test_yosida_stability.py",
    "tests/test_yosida_stability_cli.py",
    "tests/test_yosida_stability_reconstruction.py",
    "tests/test_yosida_stability_result_manifest.py",
    "tests/test_current_research_index.py",
    "theory/yosida_stability_certificate.md",
    "theory/audits/P14_YOSIDA_STABILITY_HUMAN_PROOF_AUDIT.md",
    "results/summaries/P14_YOSIDA_STABILITY_RESULTS.md",
    "third_party/UPSTREAM_COMMITS.md",
    "pyproject.toml",
    "uv.lock",
)

RationalVector = tuple[Fraction, ...]
RationalMatrix = tuple[tuple[Fraction, ...], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--require-canonical", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _decimal(value: Fraction, *, digits: int = 50) -> str:
    with localcontext() as context:
        context.prec = digits
        return str(Decimal(value.numerator) / Decimal(value.denominator))


def _transpose(matrix: RationalMatrix) -> RationalMatrix:
    return tuple(tuple(row[index] for row in matrix) for index in range(len(matrix[0])))


def _multiply(left: RationalMatrix, right: RationalMatrix) -> RationalMatrix:
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


def _add(*matrices: RationalMatrix) -> RationalMatrix:
    rows = len(matrices[0])
    columns = len(matrices[0][0])
    return tuple(
        tuple(
            sum((matrix[row][column] for matrix in matrices), Fraction(0))
            for column in range(columns)
        )
        for row in range(rows)
    )


def _scale(scale: Fraction, matrix: RationalMatrix) -> RationalMatrix:
    return tuple(tuple(scale * value for value in row) for row in matrix)


def _outer(left: RationalVector, right: RationalVector) -> RationalMatrix:
    return tuple(tuple(left_value * right_value for right_value in right) for left_value in left)


def _symmetric_outer(left: RationalVector, right: RationalVector) -> RationalMatrix:
    return _scale(Fraction(1, 2), _add(_outer(left, right), _outer(right, left)))


def _vector_add(*vectors: RationalVector) -> RationalVector:
    return tuple(
        sum((vector[index] for vector in vectors), Fraction(0)) for index in range(len(vectors[0]))
    )


def _vector_scale(scale: Fraction, vector: RationalVector) -> RationalVector:
    return tuple(scale * value for value in vector)


def _determinant(matrix: RationalMatrix) -> Fraction:
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


def _leading_minors(matrix: RationalMatrix) -> tuple[Fraction, ...]:
    return tuple(
        _determinant(
            tuple(tuple(matrix[row][column] for column in range(size)) for row in range(size))
        )
        for size in range(1, len(matrix) + 1)
    )


def _nonconvex_interpolation(
    input_i: RationalVector,
    gradient_i: RationalVector,
    input_j: RationalVector,
    gradient_j: RationalVector,
) -> RationalMatrix:
    """Quadratic part of exact directed (-1,1)-smooth interpolation."""

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


def _gram(rows: tuple[RationalVector, RationalVector], matrix: RationalMatrix) -> RationalMatrix:
    return _multiply(_transpose(rows), _multiply(matrix, rows))


def _exact_lmi() -> dict[str, object]:
    dimension = 4
    basis = tuple(
        tuple(Fraction(int(row == column)) for row in range(dimension))
        for column in range(dimension)
    )
    momentum, gradient, residual, gradient_next = basis
    null = (Fraction(0),) * dimension

    yosida_lower = MU / (1 + LAMBDA * MU)
    yosida_upper = 1 / LAMBDA
    center = (yosida_lower + yosida_upper) / 2
    radius = (yosida_upper - yosida_lower) / 2
    alpha = ETA * center * SMOOTHNESS
    residual_ratio = radius / center
    condition_ratio = PL_CONSTANT / SMOOTHNESS

    signal = _vector_add(
        _vector_scale(BETA**2, momentum),
        _vector_scale(1 - BETA**2, gradient),
    )
    step = _vector_add(
        _vector_scale(-alpha, signal),
        _vector_scale(-alpha * residual_ratio, residual),
    )
    momentum_next = _vector_add(
        _vector_scale(BETA, momentum),
        _vector_scale(1 - BETA, gradient),
    )
    transition = (momentum_next, gradient_next)
    selection = (momentum, gradient)

    interpolation_12 = _nonconvex_interpolation(null, gradient, step, gradient_next)
    interpolation_21 = _nonconvex_interpolation(step, gradient_next, null, gradient)
    residual_supply = _add(
        _outer(signal, signal),
        _scale(Fraction(-1), _outer(residual, residual)),
    )
    pl_next = _outer(gradient_next, gradient_next)

    lambda_21 = INTERPOLATION_REVERSE
    lambda_12 = lambda_21 + FUNCTION_STORAGE * TAU**2
    lambda_pl = FUNCTION_STORAGE * (1 - TAU**2) / (2 * condition_ratio)
    lmi = _add(
        _gram(transition, STORAGE),
        _scale(-(TAU**2), _gram(selection, STORAGE)),
        _scale(lambda_12, interpolation_12),
        _scale(lambda_21, interpolation_21),
        _scale(lambda_pl, pl_next),
        _scale(RESIDUAL_MULTIPLIER, residual_supply),
    )
    negative_lmi = _scale(Fraction(-1), lmi)
    storage_minors = _leading_minors(STORAGE)
    negative_minors = _leading_minors(negative_lmi)
    function_coefficients = (
        lambda_12 - lambda_21,
        -lambda_12 + lambda_21 - 2 * condition_ratio * lambda_pl,
    )
    expected_function_coefficients = (
        FUNCTION_STORAGE * TAU**2,
        -FUNCTION_STORAGE,
    )
    return {
        "yosida_lower": yosida_lower,
        "yosida_upper": yosida_upper,
        "center": center,
        "residual_radius": radius,
        "dimensionless_step": alpha,
        "residual_ratio": residual_ratio,
        "condition_ratio": condition_ratio,
        "lambda_12": lambda_12,
        "lambda_21": lambda_21,
        "lambda_pl_next": lambda_pl,
        "lambda_residual": RESIDUAL_MULTIPLIER,
        "transition": transition,
        "selection": selection,
        "signal_selector": signal,
        "step_selector": step,
        "interpolation_12": interpolation_12,
        "interpolation_21": interpolation_21,
        "pl_next": pl_next,
        "residual_norm": residual_supply,
        "lmi": lmi,
        "storage_minors": storage_minors,
        "negative_lmi_minors": negative_minors,
        "function_coefficients": function_coefficients,
        "expected_function_coefficients": expected_function_coefficients,
    }


def _jury_margin(operator_slope: Fraction, curvature: Fraction) -> Fraction:
    theta = ETA * curvature * operator_slope
    return 2 * (1 + BETA) - theta * (1 - BETA) * (1 + 2 * BETA)


def _controls() -> dict[str, Fraction]:
    base_origin_slope = P13_UNIT_ORIGIN_SLOPE / EPSILON + MU

    def yosida_slope(parameter: Fraction) -> Fraction:
        return base_origin_slope / (1 + parameter * base_origin_slope)

    selected_slope = yosida_slope(LAMBDA)
    bad_slope = yosida_slope(BAD_LAMBDA)
    critical_theta = 2 * (1 + BETA) / ((1 - BETA) * (1 + 2 * BETA))
    critical_lambda = ETA * SMOOTHNESS / critical_theta - 1 / base_origin_slope
    selected_sector_endpoint_margin = _jury_margin(1 / LAMBDA, SMOOTHNESS)
    bad_sector_endpoint_margin = _jury_margin(1 / BAD_LAMBDA, SMOOTHNESS)
    return {
        "base_origin_slope": base_origin_slope,
        "critical_theta": critical_theta,
        "critical_lambda_curvature_10": critical_lambda,
        "selected_lambda": LAMBDA,
        "selected_actual_origin_slope": selected_slope,
        "selected_actual_jury_margin_curvature_10": _jury_margin(selected_slope, SMOOTHNESS),
        "selected_sector_endpoint_jury_margin_curvature_10": selected_sector_endpoint_margin,
        "bad_lambda": BAD_LAMBDA,
        "bad_actual_origin_slope": bad_slope,
        "bad_actual_jury_margin_curvature_10": _jury_margin(bad_slope, SMOOTHNESS),
        "bad_sector_endpoint_jury_margin_curvature_10": bad_sector_endpoint_margin,
        "shifted_lambda_zero_actual_origin_slope": base_origin_slope,
        "shifted_lambda_zero_actual_jury_margin_curvature_10": _jury_margin(
            base_origin_slope, SMOOTHNESS
        ),
        "p13_direct_jury_margin_curvature_1": P13_DIRECT_JURY_MARGIN,
    }


def _sharpness() -> dict[str, object]:
    lower = MU / (1 + LAMBDA * MU)
    upper = 1 / LAMBDA
    skew_gain = Fraction(1_000)
    denominator = (1 + LAMBDA * MU) ** 2 + (LAMBDA * skew_gain) ** 2
    skew_real = (MU * (1 + LAMBDA * MU) + LAMBDA * skew_gain**2) / denominator
    skew_imaginary = skew_gain / denominator
    center = Fraction(750)
    radius = Fraction(250)
    centered_residual = (skew_real - center) ** 2 + skew_imaginary**2 - radius**2
    sector_residual = (skew_real - lower) * (upper - skew_real) - skew_imaginary**2
    sequence = []
    for base_slope in (Fraction(1_000), Fraction(1_000_000), Fraction(1_000_000_000)):
        yosida_slope = base_slope / (1 + LAMBDA * base_slope)
        sequence.append(
            {
                "base_A_slope": str(base_slope - MU),
                "shifted_B_slope": str(base_slope),
                "yosida_slope": str(yosida_slope),
                "gap_to_upper": str(upper - yosida_slope),
            }
        )
    gaps = tuple(Fraction(row["gap_to_upper"]) for row in sequence)
    return {
        "lower_base_A_slope": "0",
        "lower_yosida_slope": str(lower),
        "skew_real": str(skew_real),
        "skew_imaginary": str(skew_imaginary),
        "skew_centered_residual": str(centered_residual),
        "skew_sector_residual": str(sector_residual),
        "upper_endpoint": str(upper),
        "upper_approach_sequence": sequence,
        "upper_gaps_strictly_decrease": all(left > right > 0 for left, right in pairwise(gaps)),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _git_output(*arguments: str) -> str | None:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _source_snapshot_for(canonical: dict[str, object]) -> dict[str, str]:
    expected = canonical["proof_replay_provenance"]["source_snapshot"]
    if set(expected) != set(EXPECTED_SOURCE_PATHS):
        raise RuntimeError("source snapshot path set does not match the P14 lock")
    git = canonical["git"]
    if git["dirty"] is False:
        return {path: _git_blob_sha256(git["sha"], path) for path in expected}
    return {path: _sha256(ROOT / path) for path in expected}


def _fraction_strings(values: tuple[Fraction, ...]) -> list[str]:
    return [str(value) for value in values]


def _reconstruct() -> dict[str, object]:
    lmi = _exact_lmi()
    controls = _controls()
    sharpness = _sharpness()
    sector_factor = 1 / (LAMBDA * (1 + LAMBDA * MU))
    checks = {
        "selected_parameters_positive": LAMBDA > 0 and MU > 0,
        "yosida_sector_is_500_to_1000": (
            lmi["yosida_lower"] == 500 and lmi["yosida_upper"] == 1_000
        ),
        "residual_center_is_750_radius_250": (
            lmi["center"] == 750 and lmi["residual_radius"] == 250
        ),
        "sector_surplus_factor_is_500": sector_factor == 500,
        "dimensionless_step_is_15_over_64": lmi["dimensionless_step"] == Fraction(15, 64),
        "residual_ratio_is_one_third": lmi["residual_ratio"] == Fraction(1, 3),
        "rate_is_249001_over_250000": Fraction(249_001, 250_000) == TAU**2,
        "storage_is_positive_definite": all(value > 0 for value in lmi["storage_minors"]),
        "lmi_is_strictly_negative_definite": all(value > 0 for value in lmi["negative_lmi_minors"]),
        "function_value_flow_cancels": (
            lmi["function_coefficients"] == lmi["expected_function_coefficients"]
        ),
        "all_multipliers_are_nonnegative": all(
            lmi[key] >= 0 for key in ("lambda_12", "lambda_21", "lambda_pl_next", "lambda_residual")
        ),
        "deterministic_additive_term_is_zero": D14 == 0,
        "selected_actual_point_is_locally_schur": (
            controls["selected_actual_jury_margin_curvature_10"] > 0
        ),
        "selected_sector_endpoint_is_locally_schur": (
            controls["selected_sector_endpoint_jury_margin_curvature_10"] > 0
        ),
        "finite_bad_lambda_is_locally_unstable": (
            controls["bad_actual_jury_margin_curvature_10"] < 0
        ),
        "finite_bad_sector_endpoint_is_unstable": (
            controls["bad_sector_endpoint_jury_margin_curvature_10"] < 0
        ),
        "bad_and_selected_straddle_exact_threshold": (
            BAD_LAMBDA < controls["critical_lambda_curvature_10"] < LAMBDA
        ),
        "p13_direct_control_remains_unstable": (controls["p13_direct_jury_margin_curvature_1"] < 0),
        "shifted_lambda_zero_control_is_unstable": (
            controls["shifted_lambda_zero_actual_jury_margin_curvature_10"] < 0
        ),
        "p13_artifact_hash_matches": (_sha256(ROOT / P13_ARTIFACT_PATH) == P13_ARTIFACT_SHA256),
        "p13_checkpoint_tag_matches": (
            _git_output("rev-parse", P13_CHECKPOINT_TAG) == P13_CHECKPOINT_TAG_OBJECT
            and _git_output("rev-parse", f"{P13_CHECKPOINT_TAG}^{{}}") == P13_ARTIFACT_COMMIT
        ),
        "abstract_lower_endpoint_is_attained": sharpness["lower_yosida_slope"] == "500",
        "abstract_skew_boundary_is_exact": (
            sharpness["skew_real"] == "600"
            and sharpness["skew_imaginary"] == "200"
            and sharpness["skew_centered_residual"] == "0"
            and sharpness["skew_sector_residual"] == "0"
        ),
        "abstract_upper_endpoint_is_approached": sharpness["upper_gaps_strictly_decrease"],
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"independent P14 reconstruction failed: {failed}")

    return {
        "checks": checks,
        "parameters": {
            "epsilon": str(EPSILON),
            "lambda": str(LAMBDA),
            "mu": str(MU),
            "beta": str(BETA),
            "eta": str(ETA),
            "smoothness": str(SMOOTHNESS),
            "pl_constant": str(PL_CONSTANT),
            "tau": str(TAU),
            "q": str(TAU**2),
            "D14": str(D14),
        },
        "sector": {
            "lower": str(lmi["yosida_lower"]),
            "upper": str(lmi["yosida_upper"]),
            "center": str(lmi["center"]),
            "residual_radius": str(lmi["residual_radius"]),
            "surplus_factor": str(sector_factor),
            "dimensionless_step": str(lmi["dimensionless_step"]),
            "residual_ratio": str(lmi["residual_ratio"]),
        },
        "certificate": {
            "storage": [[str(value) for value in row] for row in STORAGE],
            "transition": [[str(value) for value in row] for row in lmi["transition"]],
            "selection": [[str(value) for value in row] for row in lmi["selection"]],
            "signal_selector": [str(value) for value in lmi["signal_selector"]],
            "step_selector": [str(value) for value in lmi["step_selector"]],
            "interpolation_12": [[str(value) for value in row] for row in lmi["interpolation_12"]],
            "interpolation_21": [[str(value) for value in row] for row in lmi["interpolation_21"]],
            "pl_next": [[str(value) for value in row] for row in lmi["pl_next"]],
            "residual_norm": [[str(value) for value in row] for row in lmi["residual_norm"]],
            "lmi_matrix": [[str(value) for value in row] for row in lmi["lmi"]],
            "function_storage": str(FUNCTION_STORAGE),
            "interpolation_reverse": str(INTERPOLATION_REVERSE),
            "lambda_12": str(lmi["lambda_12"]),
            "lambda_21": str(lmi["lambda_21"]),
            "lambda_pl_next": str(lmi["lambda_pl_next"]),
            "lambda_residual": str(lmi["lambda_residual"]),
            "storage_leading_minors": _fraction_strings(lmi["storage_minors"]),
            "negative_lmi_leading_minors": _fraction_strings(lmi["negative_lmi_minors"]),
            "function_coefficients": _fraction_strings(lmi["function_coefficients"]),
            "expected_function_coefficients": _fraction_strings(
                lmi["expected_function_coefficients"]
            ),
        },
        "controls": {key: str(value) for key, value in controls.items()},
        "sharpness": sharpness,
        "prior": {
            "artifact_path": P13_ARTIFACT_PATH,
            "artifact_sha256": P13_ARTIFACT_SHA256,
            "source_commit": P13_SOURCE_COMMIT,
            "artifact_commit": P13_ARTIFACT_COMMIT,
            "checkpoint_tag": P13_CHECKPOINT_TAG,
            "checkpoint_tag_object": P13_CHECKPOINT_TAG_OBJECT,
        },
    }


def _compare_canonical(
    canonical_path: Path, reconstruction: dict[str, object]
) -> dict[str, object]:
    if not canonical_path.exists():
        return {"status": "not_found", "comparisons": {}, "all_exact_fields_match": False}
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    comparisons = {
        "schema": canonical["schema_version"] == SCHEMA_VERSION,
        "parameters": {
            key: canonical["parameters"][key]["exact"]
            for key in ("epsilon", "lambda", "mu", "beta", "eta", "smoothness", "pl_constant")
        }
        | {
            "tau": canonical["exact_pl_certificate"]["tau"]["exact"],
            "q": canonical["exact_pl_certificate"]["q"]["exact"],
            "D14": canonical["exact_pl_certificate"]["D14"]["exact"],
        }
        == reconstruction["parameters"],
        "sector": {
            "lower": canonical["yosida_sector"]["lower"]["exact"],
            "upper": canonical["yosida_sector"]["upper"]["exact"],
            "center": canonical["yosida_sector"]["center"]["exact"],
            "residual_radius": canonical["yosida_sector"]["residual_radius"]["exact"],
            "surplus_factor": canonical["yosida_sector"]["surplus_factor"]["exact"],
            "dimensionless_step": canonical["normalized_dynamics"]["dimensionless_step"]["exact"],
            "residual_ratio": canonical["normalized_dynamics"]["residual_ratio"]["exact"],
        }
        == reconstruction["sector"],
        "certificate": canonical["exact_pl_certificate"]["reconstruction_fields"]
        == reconstruction["certificate"],
        "controls": canonical["boundary_controls"]["reconstruction_fields"]
        == reconstruction["controls"],
        "sharpness": {
            "lower_base_A_slope": canonical["abstract_sector_sharpness"]["lower_endpoint"][
                "base_A_slope"
            ]["exact"],
            "lower_yosida_slope": canonical["abstract_sector_sharpness"]["lower_endpoint"][
                "yosida_slope"
            ]["exact"],
            "skew_real": canonical["abstract_sector_sharpness"]["skew_boundary"]["real_part"][
                "exact"
            ],
            "skew_imaginary": canonical["abstract_sector_sharpness"]["skew_boundary"][
                "imaginary_part"
            ]["exact"],
            "skew_centered_residual": canonical["abstract_sector_sharpness"]["skew_boundary"][
                "centered_circle_residual"
            ]["exact"],
            "skew_sector_residual": canonical["abstract_sector_sharpness"]["skew_boundary"][
                "sector_residual"
            ]["exact"],
            "upper_endpoint": canonical["abstract_sector_sharpness"]["upper_endpoint_approach"][
                "upper_endpoint"
            ]["exact"],
            "upper_approach_sequence": [
                {
                    key: row[key]["exact"]
                    for key in (
                        "base_A_slope",
                        "shifted_B_slope",
                        "yosida_slope",
                        "gap_to_upper",
                    )
                }
                for row in canonical["abstract_sector_sharpness"]["upper_endpoint_approach"][
                    "sequence"
                ]
            ],
            "upper_gaps_strictly_decrease": canonical["abstract_sector_sharpness"][
                "upper_endpoint_approach"
            ]["gaps_strictly_decrease_to_zero"],
        }
        == reconstruction["sharpness"],
        "prior": canonical["prior_p13"] == reconstruction["prior"],
        "source_snapshot": canonical["proof_replay_provenance"]["source_snapshot"]
        == _source_snapshot_for(canonical),
        "all_primary_checks": all(canonical["audit"]["checks"].values()),
        "scope": (
            canonical["claim_scope"]["arithmetic_model"] == "exact real arithmetic"
            and canonical["claim_scope"]["matrix_domain"]
            == "R^(m x n) for every fixed finite positive m,n"
            and canonical["claim_scope"]["objective_class"]
            == "differentiable globally 10-smooth with finite infimum, global PL constant 1"
        ),
        "operator_provenance": (
            canonical["operator"]["domain"] == "R^(m x n) for every fixed finite positive m,n"
            and canonical["operator"]["epsilon_domain"]
            == "every real epsilon>0; 1e-7 is pinned only for controls"
            and canonical["operator"]["additive_normalization"]
            == "M/(||M||_F+epsilon); no max floor"
            and canonical["operator"]["coefficients_exact"] == JORDAN_COEFFICIENTS
            and canonical["operator"]["newton_schulz_stages"] == 5
            and canonical["operator"]["upstream_formula_provenance"]["revision"]
            == PINNED_UPSTREAM_REVISION
            and canonical["operator"]["upstream_formula_provenance"]["audited_file_sha256"]
            == PINNED_UPSTREAM_FILE_SHA256
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
        raise SystemExit("canonical P14 certificate is missing or mismatched")
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
