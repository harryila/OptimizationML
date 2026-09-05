#!/usr/bin/env python3
"""Independently reconstruct the exact P18 projected-sector certificate.

Only the Python standard library is imported.  The locked pointwise sector,
six smooth-PL LMIs, the useful-rate comparisons, and the generic-sector
complex-skew obstruction are rebuilt with :class:`fractions.Fraction` before
an optional canonical JSON artifact is read.

The complex-skew control concerns the abstract origin-centred sector class.
It is not a claim that the nonlinear P18 operator realizes the witness.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANONICAL = ROOT / "results/summaries/sector_projected_useful_rate_certificate.json"
SCHEMA_VERSION = "passive-muon-sector-projected-resolvent-certificate-v1"
RECONSTRUCTION_SCHEMA_VERSION = "passive-muon-p18-independent-reconstruction-v1"

BETA = Fraction(19, 20)
SMOOTHNESS = Fraction(10)
PL_CONSTANT = Fraction(1)

EPSILON = Fraction(1, 10_000_000)
JORDAN_COEFFICIENTS = (
    Fraction(6_889, 2_000),
    Fraction(-191, 40),
    Fraction(4_063, 2_000),
)
JORDAN_STEPS = 5
RESOLVENT_LAMBDA = Fraction(1, 1_000)
PASSIVE_SHUNT = Fraction(1_000)

PROJECTION_GAIN = Fraction(1)
PASSIVE_DIVISOR = Fraction(1_024)
GATE_CAP = Fraction(3, 4)
GATE_Q0 = Fraction(1, 4)
GATE_Q1 = Fraction(1)
SMOOTHERSTEP_COEFFICIENTS_DESCENDING = (
    Fraction(6),
    Fraction(-15),
    Fraction(10),
    Fraction(0),
    Fraction(0),
    Fraction(0),
)
YOSIDA_LOWER = Fraction(500)
YOSIDA_UPPER = Fraction(1_000)

LOCKED_SECTOR_LOWER = Fraction(125, 1_024)
LOCKED_SECTOR_UPPER = Fraction(509, 512)
LOCKED_SECTOR_CENTER = Fraction(1_143, 2_048)
LOCKED_SECTOR_RADIUS = Fraction(893, 2_048)

P14_RATE_SQUARED = Fraction(249_001, 250_000)
USEFUL_EFFECTIVE_STEP_LOWER = Fraction(1, 1_000)
USEFUL_EFFECTIVE_STEP_UPPER = Fraction(1, 80)

RAW_SHAPE_GAIN_UPPER = Fraction(
    20_191_130_443_162_880_000_000,
    26_793_221_204_801_899_863,
)
SMALL_PROJECTION_GAIN = Fraction(1, 100)

OBSTRUCTION_ETA = Fraction(1, 50)
OBSTRUCTION_CURVATURE = Fraction(10)
OBSTRUCTION_REAL = Fraction(1_143, 2_048)
OBSTRUCTION_IMAGINARY = Fraction(893, 2_048)
LOCKED_SCHUR_COHN_MARGIN = Fraction(
    -3_768_360_579_178_620_269,
    1_759_218_604_441_600_000_000_000,
)

RationalVector = tuple[Fraction, ...]
RationalMatrix = tuple[tuple[Fraction, ...], ...]
ComplexFraction = tuple[Fraction, Fraction]

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)


@dataclass(frozen=True)
class CertificateDesign:
    """Literal rational data for one exact smooth-PL certificate."""

    name: str
    learning_rate: Fraction
    tau: Fraction
    storage: RationalMatrix
    function_storage: Fraction
    interpolation_reverse_weight: Fraction
    residual_multiplier: Fraction


DESIGNS = (
    CertificateDesign(
        name="eta_1_over_75",
        learning_rate=Fraction(1, 75),
        tau=Fraction(99_995, 100_000),
        storage=(
            (Fraction(417, 500), Fraction(-113, 400)),
            (Fraction(-113, 400), Fraction(299, 2_000)),
        ),
        function_storage=Fraction(1),
        interpolation_reverse_weight=Fraction(12_009, 2_000),
        residual_multiplier=Fraction(83, 2_000),
    ),
    CertificateDesign(
        name="primary_eta_1_over_83",
        learning_rate=Fraction(1, 83),
        tau=Fraction(999_799, 1_000_000),
        storage=(
            (Fraction(97, 125), Fraction(-151, 500)),
            (Fraction(-151, 500), Fraction(17, 100)),
        ),
        function_storage=Fraction(1),
        interpolation_reverse_weight=Fraction(3_459, 500),
        residual_multiplier=Fraction(9, 250),
    ),
    CertificateDesign(
        name="eta_1_over_90",
        learning_rate=Fraction(1, 90),
        tau=Fraction(3_999, 4_000),
        storage=(
            (Fraction(183, 250), Fraction(-79, 250)),
            (Fraction(-79, 250), Fraction(47, 250)),
        ),
        function_storage=Fraction(1),
        interpolation_reverse_weight=Fraction(767, 100),
        residual_multiplier=Fraction(4, 125),
    ),
    CertificateDesign(
        name="eta_1_over_95",
        learning_rate=Fraction(1, 95),
        tau=Fraction(9_997, 10_000),
        storage=(
            (Fraction(88, 125), Fraction(-163, 500)),
            (Fraction(-163, 500), Fraction(201, 1_000)),
        ),
        function_storage=Fraction(1),
        interpolation_reverse_weight=Fraction(4_107, 500),
        residual_multiplier=Fraction(3, 100),
    ),
    CertificateDesign(
        name="pareto_eta_1_over_120",
        learning_rate=Fraction(1, 120),
        tau=Fraction(24_987, 25_000),
        storage=(
            (Fraction(599, 1_000), Fraction(-231, 625)),
            (Fraction(-231, 625), Fraction(657, 2_500)),
        ),
        function_storage=Fraction(1),
        interpolation_reverse_weight=Fraction(2_177, 200),
        residual_multiplier=Fraction(53, 2_500),
    ),
    CertificateDesign(
        name="eta_1_over_150",
        learning_rate=Fraction(1, 150),
        tau=Fraction(4_997, 5_000),
        storage=(
            (Fraction(5_133, 10_000), Fraction(-407, 1_000)),
            (Fraction(-407, 1_000), Fraction(663, 2_000)),
        ),
        function_storage=Fraction(1),
        interpolation_reverse_weight=Fraction(27_489, 2_000),
        residual_multiplier=Fraction(153, 10_000),
    ),
)

PRIMARY_DESIGN = DESIGNS[1]
PARETO_DESIGN = DESIGNS[4]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--require-canonical", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


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
    return tuple(
        tuple(
            sum((matrix[row][column] for matrix in matrices), Fraction(0))
            for column in range(len(matrices[0][0]))
        )
        for row in range(len(matrices[0]))
    )


def _scale(scale: Fraction, matrix: RationalMatrix) -> RationalMatrix:
    return tuple(tuple(scale * value for value in row) for row in matrix)


def _outer(left: RationalVector, right: RationalVector) -> RationalMatrix:
    return tuple(tuple(x * y for y in right) for x in left)


def _symmetric_outer(left: RationalVector, right: RationalVector) -> RationalMatrix:
    return _scale(Fraction(1, 2), _add(_outer(left, right), _outer(right, left)))


def _vector_add(*vectors: RationalVector) -> RationalVector:
    return tuple(
        sum((vector[index] for vector in vectors), Fraction(0)) for index in range(len(vectors[0]))
    )


def _vector_scale(scale: Fraction, vector: RationalVector) -> RationalVector:
    return tuple(scale * value for value in vector)


def _gram(rows: tuple[RationalVector, RationalVector], matrix: RationalMatrix) -> RationalMatrix:
    return _multiply(_transpose(rows), _multiply(matrix, rows))


def _determinant(matrix: RationalMatrix) -> Fraction:
    work = [list(row) for row in matrix]
    result = Fraction(1)
    for pivot_index in range(len(work)):
        pivot_row = next(
            (row for row in range(pivot_index, len(work)) if work[row][pivot_index]),
            None,
        )
        if pivot_row is None:
            return Fraction(0)
        if pivot_row != pivot_index:
            work[pivot_index], work[pivot_row] = work[pivot_row], work[pivot_index]
            result = -result
        pivot = work[pivot_index][pivot_index]
        result *= pivot
        for row in range(pivot_index + 1, len(work)):
            factor = work[row][pivot_index] / pivot
            for column in range(pivot_index + 1, len(work)):
                work[row][column] -= factor * work[pivot_index][column]
    return result


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
    """Return the quadratic part of directed (-1,1)-smooth interpolation."""

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


def _locked_sector() -> tuple[Fraction, Fraction, Fraction, Fraction]:
    passive_lower = YOSIDA_LOWER / PASSIVE_DIVISOR
    passive_upper = YOSIDA_UPPER / PASSIVE_DIVISOR
    lower = (1 - GATE_CAP) * passive_lower
    upper = max(
        passive_upper,
        (1 - GATE_CAP) * passive_upper + GATE_CAP * PROJECTION_GAIN,
    )
    return lower, upper, (lower + upper) / 2, (upper - lower) / 2


def _exact_lmi(design: CertificateDesign) -> dict[str, object]:
    dimension = 4
    basis = tuple(
        tuple(Fraction(int(row == column)) for row in range(dimension))
        for column in range(dimension)
    )
    momentum, gradient, residual, gradient_next = basis
    null = (Fraction(0),) * dimension
    lower, upper, center, radius = _locked_sector()
    dimensionless_step = design.learning_rate * center * SMOOTHNESS
    residual_ratio = radius / center
    condition_ratio = PL_CONSTANT / SMOOTHNESS

    signal = _vector_add(
        _vector_scale(BETA**2, momentum),
        _vector_scale(1 - BETA**2, gradient),
    )
    step = _vector_add(
        _vector_scale(-dimensionless_step, signal),
        _vector_scale(-dimensionless_step * residual_ratio, residual),
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

    lambda_21 = design.interpolation_reverse_weight
    lambda_12 = lambda_21 + design.function_storage * design.tau**2
    lambda_pl = design.function_storage * (1 - design.tau**2) / (2 * condition_ratio)
    lmi = _add(
        _gram(transition, design.storage),
        _scale(-(design.tau**2), _gram(selection, design.storage)),
        _scale(lambda_12, interpolation_12),
        _scale(lambda_21, interpolation_21),
        _scale(lambda_pl, pl_next),
        _scale(design.residual_multiplier, residual_supply),
    )
    function_coefficients = (
        lambda_12 - lambda_21,
        -lambda_12 + lambda_21 - 2 * condition_ratio * lambda_pl,
    )
    return {
        "lower": lower,
        "upper": upper,
        "center": center,
        "radius": radius,
        "dimensionless_step": dimensionless_step,
        "residual_ratio": residual_ratio,
        "storage_minors": _leading_minors(design.storage),
        "negative_lmi_minors": _leading_minors(_scale(Fraction(-1), lmi)),
        "function_coefficients": function_coefficients,
        "expected_function_coefficients": (
            design.function_storage * design.tau**2,
            -design.function_storage,
        ),
        "multipliers": (
            lambda_12,
            lambda_21,
            lambda_pl,
            design.residual_multiplier,
        ),
    }


def _complex_add(left: ComplexFraction, right: ComplexFraction) -> ComplexFraction:
    return left[0] + right[0], left[1] + right[1]


def _complex_scale(scale: Fraction, value: ComplexFraction) -> ComplexFraction:
    return scale * value[0], scale * value[1]


def _complex_multiply(left: ComplexFraction, right: ComplexFraction) -> ComplexFraction:
    return (
        left[0] * right[0] - left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def _complex_conjugate(value: ComplexFraction) -> ComplexFraction:
    return value[0], -value[1]


def _complex_squared_norm(value: ComplexFraction) -> Fraction:
    return value[0] ** 2 + value[1] ** 2


def _complex_skew_obstruction() -> dict[str, object]:
    """Rebuild the quadratic Schur-Cohn failure for the generic sector class."""

    gain = (OBSTRUCTION_REAL, OBSTRUCTION_IMAGINARY)
    trace = _complex_add(
        (1 + BETA, Fraction(0)),
        _complex_scale(
            -OBSTRUCTION_ETA * (1 - BETA**2) * OBSTRUCTION_CURVATURE,
            gain,
        ),
    )
    determinant = _complex_add(
        (BETA, Fraction(0)),
        _complex_scale(
            -OBSTRUCTION_ETA * BETA * (1 - BETA) * OBSTRUCTION_CURVATURE,
            gain,
        ),
    )
    coefficient_one = (-trace[0], -trace[1])
    coefficient_zero = determinant
    schur_vector = _complex_add(
        coefficient_one,
        _complex_scale(
            Fraction(-1),
            _complex_multiply(_complex_conjugate(coefficient_one), coefficient_zero),
        ),
    )
    determinant_margin = 1 - _complex_squared_norm(coefficient_zero)
    schur_cohn_margin = determinant_margin**2 - _complex_squared_norm(schur_vector)
    sector_boundary_residual = (OBSTRUCTION_REAL - LOCKED_SECTOR_LOWER) * (
        LOCKED_SECTOR_UPPER - OBSTRUCTION_REAL
    ) - OBSTRUCTION_IMAGINARY**2
    return {
        "gain": gain,
        "trace": trace,
        "determinant": determinant,
        "coefficient_one": coefficient_one,
        "coefficient_zero": coefficient_zero,
        "schur_vector": schur_vector,
        "determinant_margin": determinant_margin,
        "schur_cohn_margin": schur_cohn_margin,
        "sector_boundary_residual": sector_boundary_residual,
    }


def _strings(values: tuple[Fraction, ...]) -> list[str]:
    return [str(value) for value in values]


def _matrix_strings(matrix: RationalMatrix) -> list[list[str]]:
    return [[str(value) for value in row] for row in matrix]


def _complex_strings(value: ComplexFraction) -> dict[str, str]:
    return {"real": str(value[0]), "imaginary": str(value[1])}


def _reconstruction_fields() -> tuple[dict[str, object], dict[str, bool]]:
    lower, upper, center, radius = _locked_sector()
    lmis = {design.name: _exact_lmi(design) for design in DESIGNS}
    obstruction = _complex_skew_obstruction()
    polynomial_factor_discriminant = (
        JORDAN_COEFFICIENTS[1] ** 2 - 4 * JORDAN_COEFFICIENTS[0] * JORDAN_COEFFICIENTS[2]
    )

    certificate_fields: dict[str, object] = {}
    for design in DESIGNS:
        lmi = lmis[design.name]
        certificate_fields[design.name] = {
            "learning_rate": str(design.learning_rate),
            "tau": str(design.tau),
            "rate_squared": str(design.tau**2),
            "storage": _matrix_strings(design.storage),
            "function_storage": str(design.function_storage),
            "interpolation_reverse_weight": str(design.interpolation_reverse_weight),
            "residual_multiplier": str(design.residual_multiplier),
            "dimensionless_step": str(lmi["dimensionless_step"]),
            "residual_ratio": str(lmi["residual_ratio"]),
            "storage_leading_minors": _strings(lmi["storage_minors"]),
            "negative_lmi_leading_minors": _strings(lmi["negative_lmi_minors"]),
            "multipliers": _strings(lmi["multipliers"]),
        }

    passive_lower = YOSIDA_LOWER / PASSIVE_DIVISOR
    passive_upper = YOSIDA_UPPER / PASSIVE_DIVISOR
    blended_upper = (1 - GATE_CAP) * passive_upper + GATE_CAP * PROJECTION_GAIN
    primary_lower_step = PRIMARY_DESIGN.learning_rate * lower
    primary_upper_step = PRIMARY_DESIGN.learning_rate * upper

    unprojected_blended_upper = (1 - GATE_CAP) * passive_upper + GATE_CAP * RAW_SHAPE_GAIN_UPPER
    unprojected_upper = max(passive_upper, unprojected_blended_upper)
    unprojected_effective_step = OBSTRUCTION_ETA * unprojected_upper

    small_projection_blended_upper = (
        1 - GATE_CAP
    ) * passive_upper + GATE_CAP * SMALL_PROJECTION_GAIN
    small_projection_upper = max(passive_upper, small_projection_blended_upper)

    fields = {
        "operator_specification": {
            "domain": "R^(m x n) for arbitrary positive finite m and n",
            "arithmetic": "exact real",
            "resolvent": "exact",
            "normalization": "U/(||U||_F+epsilon)",
            "epsilon": str(EPSILON),
            "polynomial_coefficients": {
                "a": str(JORDAN_COEFFICIENTS[0]),
                "b": str(JORDAN_COEFFICIENTS[1]),
                "c": str(JORDAN_COEFFICIENTS[2]),
            },
            "polynomial_iteration_count": JORDAN_STEPS,
            "resolvent_lambda": str(RESOLVENT_LAMBDA),
            "passive_shunt": str(PASSIVE_SHUNT),
            "sector_scope": "origin-centred pointwise, not incremental",
        },
        "locked_design": {
            "projection_gain": str(PROJECTION_GAIN),
            "passive_divisor": str(PASSIVE_DIVISOR),
            "gate_cap": str(GATE_CAP),
            "yosida_sector": {
                "lower": str(YOSIDA_LOWER),
                "upper": str(YOSIDA_UPPER),
            },
            "passive_branch_sector": {
                "lower": str(passive_lower),
                "upper": str(passive_upper),
            },
            "blended_upper_at_gate_cap": str(blended_upper),
            "projected_pointwise_sector": {
                "lower": str(lower),
                "upper": str(upper),
                "center": str(center),
                "radius": str(radius),
            },
        },
        "exact_certificates": certificate_fields,
        "useful_rate": {
            "primary_design": PRIMARY_DESIGN.name,
            "pareto_design": PARETO_DESIGN.name,
            "p14_rate_squared": str(P14_RATE_SQUARED),
            "primary_rate_to_tenth_power": str(PRIMARY_DESIGN.tau**20),
            "effective_step_window": {
                "lower": str(USEFUL_EFFECTIVE_STEP_LOWER),
                "upper": str(USEFUL_EFFECTIVE_STEP_UPPER),
            },
            "primary_effective_sector": {
                "lower": str(primary_lower_step),
                "upper": str(primary_upper_step),
            },
        },
        "complex_skew_obstruction": {
            "scope": (
                "admissible constant two-dimensional map in the generic "
                "origin-centred sector; not an actual-P18 realizability claim"
            ),
            "learning_rate": str(OBSTRUCTION_ETA),
            "curvature": str(OBSTRUCTION_CURVATURE),
            "gain": _complex_strings(obstruction["gain"]),
            "sector_boundary_residual": str(obstruction["sector_boundary_residual"]),
            "trace": _complex_strings(obstruction["trace"]),
            "determinant": _complex_strings(obstruction["determinant"]),
            "characteristic_coefficient_one": _complex_strings(obstruction["coefficient_one"]),
            "characteristic_coefficient_zero": _complex_strings(obstruction["coefficient_zero"]),
            "schur_vector": _complex_strings(obstruction["schur_vector"]),
            "determinant_disk_margin": str(obstruction["determinant_margin"]),
            "schur_cohn_margin": str(obstruction["schur_cohn_margin"]),
        },
        "controls": {
            "unprojected_shape_branch": {
                "raw_shape_gain_upper": str(RAW_SHAPE_GAIN_UPPER),
                "uniform_upper": str(unprojected_upper),
                "effective_upper_at_eta_1_over_50": str(unprojected_effective_step),
                "scope": "huge generic-sector diagnostic, not an instability proof",
            },
            "projection_gain_1_over_100": {
                "projection_gain": str(SMALL_PROJECTION_GAIN),
                "blended_upper_at_gate_cap": str(small_projection_blended_upper),
                "uniform_upper": str(small_projection_upper),
                "scope": "fidelity-only negative control, not a sector failure",
            },
        },
    }

    checks = {
        "quintic_stage_factor_is_strictly_positive": (
            JORDAN_COEFFICIENTS[0] > 0
            and JORDAN_COEFFICIENTS[2] > 0
            and polynomial_factor_discriminant < 0
        ),
        "yosida_sector_follows_locked_resolvent": (
            YOSIDA_LOWER == PASSIVE_SHUNT / (1 + RESOLVENT_LAMBDA * PASSIVE_SHUNT)
            and YOSIDA_UPPER == 1 / RESOLVENT_LAMBDA
        ),
        "locked_sector_matches_claimed_fractions": (
            lower == LOCKED_SECTOR_LOWER
            and upper == LOCKED_SECTOR_UPPER
            and center == LOCKED_SECTOR_CENTER
            and radius == LOCKED_SECTOR_RADIUS
        ),
        "projected_blend_upper_is_active": (
            blended_upper == LOCKED_SECTOR_UPPER and blended_upper > passive_upper
        ),
        "all_storage_matrices_are_positive_definite": all(
            all(value > 0 for value in lmis[design.name]["storage_minors"]) for design in DESIGNS
        ),
        "exact_frontier_contains_six_certificates": len(DESIGNS) == 6,
        "all_six_PL_LMIs_are_strictly_negative_definite": all(
            all(value > 0 for value in lmis[design.name]["negative_lmi_minors"])
            for design in DESIGNS
        ),
        "all_function_value_flows_cancel": all(
            lmis[design.name]["function_coefficients"]
            == lmis[design.name]["expected_function_coefficients"]
            for design in DESIGNS
        ),
        "all_LMI_multipliers_are_nonnegative": all(
            all(value >= 0 for value in lmis[design.name]["multipliers"]) for design in DESIGNS
        ),
        "primary_tenth_power_beats_P14_rate": (PRIMARY_DESIGN.tau**20 < P14_RATE_SQUARED),
        "primary_effective_sector_is_in_useful_window": (
            primary_lower_step >= USEFUL_EFFECTIVE_STEP_LOWER
            and primary_upper_step <= USEFUL_EFFECTIVE_STEP_UPPER
        ),
        "locked_amplitude_sector_is_in_declared_window": (lower > Fraction(1, 10) and upper < 1),
        "complex_skew_gain_is_on_sector_boundary": (
            obstruction["sector_boundary_residual"] == 0 and obstruction["gain"] == (center, radius)
        ),
        "complex_skew_determinant_is_inside_unit_disk": (obstruction["determinant_margin"] > 0),
        "complex_skew_margin_matches_lock": (
            obstruction["schur_cohn_margin"] == LOCKED_SCHUR_COHN_MARGIN
        ),
        "complex_skew_control_fails_Schur_Cohn": (obstruction["schur_cohn_margin"] < 0),
        "unprojected_upper_exceeds_one": unprojected_upper > 1,
        "unprojected_effective_upper_exceeds_useful_window": (
            unprojected_effective_step > USEFUL_EFFECTIVE_STEP_UPPER
        ),
        "small_projection_gain_leaves_passive_upper_active": (
            small_projection_upper == passive_upper
            and small_projection_blended_upper < passive_upper
        ),
    }
    return fields, checks


def _reconstruct() -> dict[str, object]:
    fields, checks = _reconstruction_fields()
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"independent P18 reconstruction failed: {failed}")
    return {"checks": checks, "reconstruction_fields": fields}


def _compare_canonical(
    canonical_path: Path,
    reconstruction: dict[str, object],
) -> dict[str, object]:
    if not canonical_path.exists():
        return {
            "status": "not_found",
            "comparisons": {},
            "all_exact_fields_match": False,
        }
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    comparisons = {
        "schema": canonical.get("schema_version") == SCHEMA_VERSION,
        "reconstruction_fields": canonical.get("reconstruction_fields")
        == reconstruction["reconstruction_fields"],
        "all_primary_checks": isinstance(canonical.get("audit", {}).get("checks"), dict)
        and all(canonical["audit"]["checks"].values()),
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
        raise SystemExit("canonical P18 certificate is missing or mismatched")
    payload = {
        "schema_version": RECONSTRUCTION_SCHEMA_VERSION,
        "claim_scope": {
            "matrix_domain": "R^(m x n) for every fixed finite positive m,n",
            "arithmetic": "exact real with an exact resolvent",
            "optimizer_beta": str(BETA),
            "objective_smoothness": str(SMOOTHNESS),
            "objective_PL_constant": str(PL_CONSTANT),
            "gate_squared_radius_interval": [str(GATE_Q0), str(GATE_Q1)],
            "gate_smootherstep_coefficients_descending": _strings(
                SMOOTHERSTEP_COEFFICIENTS_DESCENDING
            ),
            "sector_kind": "origin-centred pointwise, not incremental",
            "generic_obstruction": (
                "an abstract sector-class negative control, not realizability by "
                "the structured P18 operator"
            ),
        },
        "implementation_scope": {
            "project_package_imported": False,
            "numerical_library_imported": False,
            "canonical_read_order": "only after complete independent reconstruction",
        },
        "reconstruction": reconstruction,
        "canonical_comparison": comparison,
        "all_internal_exact_checks_passed": all(reconstruction["checks"].values()),
        "all_exact_checks_passed": all(reconstruction["checks"].values())
        and comparison["status"] == "matched",
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
