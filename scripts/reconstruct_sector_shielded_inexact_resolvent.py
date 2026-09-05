#!/usr/bin/env python3
"""Independently reconstruct the exact P19 sector-shield certificate.

This script uses only the Python standard library.  It rebuilds the shield
geometry, rational corruption controls, and both 4-by-4 smooth-PL LMIs before
optionally reading a generated canonical artifact.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANONICAL = ROOT / "results/summaries/sector_shielded_inexact_resolvent_certificate.json"
SCHEMA_VERSION = "passive-muon-sector-shielded-inexact-resolvent-certificate-v1"
RECONSTRUCTION_SCHEMA_VERSION = "passive-muon-p19-independent-reconstruction-v1"

BETA = Fraction(19, 20)
SMOOTHNESS = Fraction(10)
PL_CONSTANT = Fraction(1)
LOWER = Fraction(125, 1_024)
UPPER = Fraction(509, 512)
CENTER = Fraction(1_143, 2_048)
RADIUS = Fraction(893, 2_048)

RationalVector = tuple[Fraction, ...]
RationalMatrix = tuple[tuple[Fraction, ...], ...]

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)


@dataclass(frozen=True)
class Design:
    name: str
    learning_rate: Fraction
    tau: Fraction
    storage: RationalMatrix
    function_storage: Fraction
    interpolation_reverse_weight: Fraction
    residual_multiplier: Fraction


DESIGNS = (
    Design(
        name="maximum_step_eta_1_over_83",
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
    Design(
        name="faster_rate_eta_1_over_120",
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
)


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


def _matrix_add(*matrices: RationalMatrix) -> RationalMatrix:
    return tuple(
        tuple(
            sum((matrix[row][column] for matrix in matrices), Fraction(0))
            for column in range(len(matrices[0][0]))
        )
        for row in range(len(matrices[0]))
    )


def _matrix_scale(scale: Fraction, matrix: RationalMatrix) -> RationalMatrix:
    return tuple(tuple(scale * value for value in row) for row in matrix)


def _outer(left: RationalVector, right: RationalVector) -> RationalMatrix:
    return tuple(tuple(x * y for y in right) for x in left)


def _symmetric_outer(left: RationalVector, right: RationalVector) -> RationalMatrix:
    return _matrix_scale(
        Fraction(1, 2),
        _matrix_add(_outer(left, right), _outer(right, left)),
    )


def _vector_add(*vectors: RationalVector) -> RationalVector:
    return tuple(
        sum((vector[index] for vector in vectors), Fraction(0)) for index in range(len(vectors[0]))
    )


def _vector_scale(scale: Fraction, vector: RationalVector) -> RationalVector:
    return tuple(scale * value for value in vector)


def _vector_subtract(left: RationalVector, right: RationalVector) -> RationalVector:
    return tuple(x - y for x, y in zip(left, right, strict=True))


def _dot(left: RationalVector, right: RationalVector) -> Fraction:
    return sum((x * y for x, y in zip(left, right, strict=True)), Fraction(0))


def _squared_norm(vector: RationalVector) -> Fraction:
    return _dot(vector, vector)


def _fraction_sqrt(value: Fraction) -> Fraction:
    numerator = math.isqrt(value.numerator)
    denominator = math.isqrt(value.denominator)
    if numerator**2 != value.numerator or denominator**2 != value.denominator:
        raise AssertionError("locked exact control unexpectedly has an irrational norm")
    return Fraction(numerator, denominator)


def _project(source: RationalVector, candidate: RationalVector) -> RationalVector:
    source_norm = _fraction_sqrt(_squared_norm(source))
    if source_norm == 0:
        return tuple(Fraction(0) for _ in source)
    center = _vector_scale(CENTER, source)
    displacement = _vector_subtract(candidate, center)
    displacement_norm = _fraction_sqrt(_squared_norm(displacement))
    radius = RADIUS * source_norm
    if displacement_norm == 0 or displacement_norm <= radius:
        return candidate
    return _vector_add(center, _vector_scale(radius / displacement_norm, displacement))


def _supply(source: RationalVector, output: RationalVector) -> Fraction:
    return _dot(
        _vector_subtract(output, _vector_scale(LOWER, source)),
        _vector_subtract(_vector_scale(UPPER, source), output),
    )


def _ball_slack(source: RationalVector, output: RationalVector) -> Fraction:
    displacement = _vector_subtract(output, _vector_scale(CENTER, source))
    return RADIUS**2 * _squared_norm(source) - _squared_norm(displacement)


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


def _gram(rows: RationalMatrix, matrix: RationalMatrix) -> RationalMatrix:
    return _multiply(_transpose(rows), _multiply(matrix, rows))


def _nonconvex_interpolation(
    input_i: RationalVector,
    gradient_i: RationalVector,
    input_j: RationalVector,
    gradient_j: RationalVector,
) -> RationalMatrix:
    dx = _vector_add(input_i, _vector_scale(Fraction(-1), input_j))
    du = _vector_add(gradient_i, _vector_scale(Fraction(-1), gradient_j))
    model = _matrix_scale(
        Fraction(1, 4),
        _matrix_add(
            _outer(du, du),
            _matrix_scale(Fraction(2), _symmetric_outer(dx, du)),
            _matrix_scale(Fraction(-1), _outer(dx, dx)),
        ),
    )
    return _matrix_add(
        _matrix_scale(Fraction(-1), _symmetric_outer(gradient_j, dx)),
        _matrix_scale(Fraction(-1), model),
    )


def _exact_lmi(design: Design) -> dict[str, object]:
    dimension = 4
    basis = tuple(
        tuple(Fraction(int(row == column)) for row in range(dimension))
        for column in range(dimension)
    )
    momentum, gradient, residual, gradient_next = basis
    null = (Fraction(0),) * dimension
    dimensionless_step = design.learning_rate * CENTER * SMOOTHNESS
    residual_ratio = RADIUS / CENTER
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
    residual_supply = _matrix_add(
        _outer(signal, signal),
        _matrix_scale(Fraction(-1), _outer(residual, residual)),
    )
    pl_next = _outer(gradient_next, gradient_next)

    lambda_21 = design.interpolation_reverse_weight
    lambda_12 = lambda_21 + design.function_storage * design.tau**2
    lambda_pl = design.function_storage * (1 - design.tau**2) / (2 * condition_ratio)
    lmi = _matrix_add(
        _gram(transition, design.storage),
        _matrix_scale(-(design.tau**2), _gram(selection, design.storage)),
        _matrix_scale(lambda_12, interpolation_12),
        _matrix_scale(lambda_21, interpolation_21),
        _matrix_scale(lambda_pl, pl_next),
        _matrix_scale(design.residual_multiplier, residual_supply),
    )
    return {
        "dimensionless_step": dimensionless_step,
        "residual_ratio": residual_ratio,
        "storage_minors": _leading_minors(design.storage),
        "negative_lmi_minors": _leading_minors(_matrix_scale(Fraction(-1), lmi)),
        "function_coefficients": (
            lambda_12 - lambda_21,
            -lambda_12 + lambda_21 - 2 * condition_ratio * lambda_pl,
        ),
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


def _matrix_strings(matrix: RationalMatrix) -> list[list[str]]:
    return [[str(value) for value in row] for row in matrix]


def _vector_strings(vector: RationalVector) -> list[str]:
    return [str(value) for value in vector]


def _control_fields() -> dict[str, dict[str, object]]:
    source = (Fraction(3), Fraction(4))
    perpendicular = (Fraction(-4), Fraction(3))
    center = _vector_scale(CENTER, source)
    cases = (
        ("zero_input_singleton", (Fraction(0), Fraction(0)), (Fraction(7), Fraction(-11))),
        (
            "interior_identity",
            source,
            _vector_add(center, _vector_scale(RADIUS / 2, perpendicular)),
        ),
        ("radial_corruption", source, _vector_scale(UPPER + 1, source)),
        (
            "tangential_corruption",
            source,
            _vector_add(center, _vector_scale(2 * RADIUS, perpendicular)),
        ),
    )
    result: dict[str, dict[str, object]] = {}
    for name, selected_source, candidate in cases:
        projected = _project(selected_source, candidate)
        result[name] = {
            "source": _vector_strings(selected_source),
            "candidate": _vector_strings(candidate),
            "projected": _vector_strings(projected),
            "candidate_supply": str(_supply(selected_source, candidate)),
            "projected_supply": str(_supply(selected_source, projected)),
            "candidate_ball_slack": str(_ball_slack(selected_source, candidate)),
            "projected_ball_slack": str(_ball_slack(selected_source, projected)),
            "projection_was_active": projected != candidate,
        }
    return result


def _pl_fields(design: Design, lmi: dict[str, object]) -> dict[str, object]:
    return {
        "learning_rate": str(design.learning_rate),
        "tau": str(design.tau),
        "rate_squared": str(design.tau**2),
        "storage": _matrix_strings(design.storage),
        "function_storage": str(design.function_storage),
        "interpolation_reverse_weight": str(design.interpolation_reverse_weight),
        "residual_multiplier": str(design.residual_multiplier),
        "dimensionless_step": str(lmi["dimensionless_step"]),
        "residual_ratio": str(lmi["residual_ratio"]),
        "storage_leading_minors": [str(value) for value in lmi["storage_minors"]],
        "negative_lmi_leading_minors": [str(value) for value in lmi["negative_lmi_minors"]],
        "multipliers": [str(value) for value in lmi["multipliers"]],
    }


def _reconstruction_fields() -> tuple[dict[str, object], dict[str, bool]]:
    lmis = {design.name: _exact_lmi(design) for design in DESIGNS}
    controls = _control_fields()
    source = (Fraction(3), Fraction(4))
    exact_reference = tuple(Fraction(value) for value in controls["interior_identity"]["candidate"])
    corrupted = tuple(Fraction(value) for value in controls["tangential_corruption"]["candidate"])
    shielded = tuple(Fraction(value) for value in controls["tangential_corruption"]["projected"])
    fields = {
        "shield_geometry": {
            "matrix_domain": "R^(m x n) for arbitrary positive finite m and n",
            "sector_kind": "origin-centred pointwise, not incremental",
            "lower": str(LOWER),
            "upper": str(UPPER),
            "center": str(CENTER),
            "radius": str(RADIUS),
            "set": "D_S={U: ||U-gamma*S||_F <= radius*||S||_F}",
            "projection_nonzero": ("gamma*S+min(1,radius*||S||_F/||C-gamma*S||_F)*(C-gamma*S)"),
            "zero_displacement_rule": "the radial multiplier is 1 when C=gamma*S",
            "zero_input_rule": "Pi_{D_0}(C)=0",
            "sector_equivalence": ("<U-lower*S,upper*S-U>_F=radius^2*||S||_F^2-||U-center*S||_F^2"),
            "fixed_input_nonexpansiveness": ("||Pi_{D_S}(C1)-Pi_{D_S}(C2)||_F<=||C1-C2||_F"),
            "exact_output_fidelity_lemma": (
                "||Pi_{D_S}(C)-T18(S)||_F<=||C-T18(S)||_F for T18(S) in D_S"
            ),
            "fidelity_caveat": (
                "a P15 graph residual requires an intervening candidate-error bound"
            ),
        },
        "operating_points": {
            design.name: _pl_fields(design, lmis[design.name]) for design in DESIGNS
        },
        "exact_controls": controls,
    }
    checks = {
        "sector_center_is_exact": CENTER == (LOWER + UPPER) / 2,
        "sector_radius_is_exact": RADIUS == (UPPER - LOWER) / 2,
        "all_control_sector_supplies_equal_ball_slacks": all(
            Fraction(control["candidate_supply"]) == Fraction(control["candidate_ball_slack"])
            and Fraction(control["projected_supply"]) == Fraction(control["projected_ball_slack"])
            for control in controls.values()
        ),
        "zero_input_is_singleton_zero": controls["zero_input_singleton"]["projected"] == ["0", "0"],
        "interior_is_identity": (
            controls["interior_identity"]["projection_was_active"] is False
            and controls["interior_identity"]["candidate"]
            == controls["interior_identity"]["projected"]
        ),
        "radial_corruption_is_repaired": (
            Fraction(controls["radial_corruption"]["candidate_supply"]) < 0
            and Fraction(controls["radial_corruption"]["projected_supply"]) == 0
        ),
        "tangential_corruption_is_repaired": (
            Fraction(controls["tangential_corruption"]["candidate_supply"]) < 0
            and Fraction(controls["tangential_corruption"]["projected_supply"]) == 0
        ),
        "all_projected_controls_are_safe": all(
            Fraction(control["projected_supply"]) >= 0 for control in controls.values()
        ),
        "shield_does_not_increase_error_to_admissible_reference": (
            _squared_norm(_vector_subtract(shielded, exact_reference))
            <= _squared_norm(_vector_subtract(corrupted, exact_reference))
        ),
        "both_storage_matrices_are_positive_definite": all(
            all(value > 0 for value in lmis[design.name]["storage_minors"]) for design in DESIGNS
        ),
        "both_PL_LMIs_are_strictly_negative_definite": all(
            all(value > 0 for value in lmis[design.name]["negative_lmi_minors"])
            for design in DESIGNS
        ),
        "both_function_value_flows_cancel": all(
            lmis[design.name]["function_coefficients"]
            == lmis[design.name]["expected_function_coefficients"]
            for design in DESIGNS
        ),
        "both_multiplier_sets_are_nonnegative": all(
            all(value >= 0 for value in lmis[design.name]["multipliers"]) for design in DESIGNS
        ),
        "maximum_step_rate_is_frozen": (
            DESIGNS[0].tau ** 2 == Fraction(999_598_040_401, 1_000_000_000_000)
        ),
        "faster_rate_is_frozen": (DESIGNS[1].tau ** 2 == Fraction(624_350_169, 625_000_000)),
        "eta_1_over_120_rate_is_better": DESIGNS[1].tau ** 2 < DESIGNS[0].tau ** 2,
        "source_control_is_nonzero": _squared_norm(source) > 0,
    }
    return fields, checks


def _compare_canonical(
    canonical_path: Path,
    fields: dict[str, object],
) -> dict[str, object]:
    if not canonical_path.exists():
        return {"status": "not_found", "comparisons": {}, "all_exact_fields_match": False}
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    comparisons = {
        "schema": canonical.get("schema_version") == SCHEMA_VERSION,
        "reconstruction_fields": canonical.get("reconstruction_fields") == fields,
        "all_generator_checks": isinstance(canonical.get("audit", {}).get("checks"), dict)
        and all(canonical["audit"]["checks"].values()),
    }
    return {
        "status": "matched" if all(comparisons.values()) else "mismatch",
        "comparisons": comparisons,
        "all_exact_fields_match": all(comparisons.values()),
    }


def main() -> int:
    arguments = parse_args()
    fields, checks = _reconstruction_fields()
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"independent P19 reconstruction failed: {failed}")
    comparison = _compare_canonical(arguments.canonical, fields)
    if arguments.require_canonical and comparison["status"] != "matched":
        raise SystemExit("canonical P19 certificate is missing or mismatched")
    payload = {
        "schema_version": RECONSTRUCTION_SCHEMA_VERSION,
        "claim_scope": {
            "matrix_domain": "R^(m x n) for every fixed finite positive m,n",
            "arithmetic": "exact real",
            "sector_kind": "origin-centred pointwise, not incremental",
            "candidate_scope": "every finite candidate; rational vectors used only as controls",
        },
        "implementation_scope": {
            "project_package_imported": False,
            "numerical_library_imported": False,
            "canonical_read_order": "only after complete independent reconstruction",
        },
        "reconstruction": {"checks": checks, "reconstruction_fields": fields},
        "canonical_comparison": comparison,
        "all_internal_exact_checks_passed": all(checks.values()),
        "all_exact_checks_passed": all(checks.values()) and comparison["status"] == "matched",
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
