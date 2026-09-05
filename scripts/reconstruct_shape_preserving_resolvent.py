#!/usr/bin/env python3
"""Independently reconstruct the exact P17 gated-resolvent certificate.

Only the Python standard library is imported.  Both pointwise sectors, both
smooth-PL LMIs, the C2 gate identities, and the unsafe scalar control are
rebuilt with :class:`fractions.Fraction` before an optional canonical JSON
artifact is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANONICAL = ROOT / "results/summaries/shape_preserving_resolvent_certificate.json"
SCHEMA_VERSION = "passive-muon-shape-preserving-resolvent-certificate-v1"
RECONSTRUCTION_SCHEMA_VERSION = "passive-muon-p17-independent-reconstruction-v1"

EPSILON = Fraction(1, 10_000_000)
LAMBDA = Fraction(1, 1_000)
MU = Fraction(1_000)
BETA = Fraction(19, 20)
SMOOTHNESS = Fraction(10)
PL_CONSTANT = Fraction(1)

COEFFICIENTS = (Fraction(6_889, 2_000), Fraction(-191, 40), Fraction(4_063, 2_000))
STEPS = 5
DERIVATIVE_UPPER = Fraction(4_848_763, 10_000)
UNIT_DEFICIT_UPPER = Fraction(
    6_602_082_433_275_499_863,
    41_641_817_600_000_000,
)
TAIL_DERIVATIVE_LOWER = Fraction(-199_437, 1_250)
TAIL_PROJECTION_LOWER = Fraction(-41_528_474_059_081, 260_261_360_000)

YOSIDA_LOWER = Fraction(500)
YOSIDA_UPPER = Fraction(1_000)
RAW_SHAPE_GAIN_UPPER = Fraction(
    20_191_130_443_162_880_000_000,
    26_793_221_204_801_899_863,
)

GATE_Q0 = Fraction(1, 4)
GATE_Q1 = Fraction(1)
SMOOTHERSTEP_COEFFICIENTS = (Fraction(10), Fraction(-15), Fraction(6))
SMOOTHERSTEP_COEFFICIENTS_DESCENDING = (
    Fraction(6),
    Fraction(-15),
    Fraction(10),
    Fraction(0),
    Fraction(0),
    Fraction(0),
)

UNSAFE_T = Fraction(63, 10_000)
UNSAFE_Z = Fraction(63, 9_937)
SOURCE_GUARD = (Fraction(19, 10_000), Fraction(1, 500))
RAW_SHAPE_DERIVATIVE_GUARD = (Fraction(-147_000), Fraction(-146_000))
YOSIDA_DERIVATIVE_GUARD = (Fraction(999), Fraction(1_000))
UNDERSIZED_GATE_RADII = (Fraction(1, 2_048), Fraction(1, 1_024))
GATED_DERIVATIVE_GUARD = (Fraction(-19_000), Fraction(-18_000))

RESPONSE_SHA256 = "8a45fab1019bdec0355707f4784edd3a09ee75addb27ca63af8d955dcf8f94af"
DERIVATIVE_SHA256 = "edbcd595994188e9710363d6de797276a24fb399a332959fdda4bf610fba56a9"
SOURCE_SHA256 = "9b1d2dcd92ea077f8ed7f7bc4b2bfacd9f3018a47b45db6aed180edb7f181579"
RAW_SHAPE_DERIVATIVE_SHA256 = "a6a7929612a5275509eac2447ac6e75cc0dbdba3c1e235710ae60146ff327cdc"
GATED_DERIVATIVE_SHA256 = "59fe66b1985bcf2b0f1f2981197d74978c321bbbee93228aa6465b1b3c17c341"
SECOND_JURY_MARGIN_SHA256 = "4a9270640c0e8769496d53651eec732fa3c78cee7bf0732b1a773e0fbbdc54b4"

RationalVector = tuple[Fraction, ...]
RationalMatrix = tuple[tuple[Fraction, ...], ...]

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)


@dataclass(frozen=True)
class Design:
    name: str
    cap: Fraction
    passive_divisor: Fraction
    learning_rate: Fraction
    tau: Fraction
    storage: RationalMatrix
    function_storage: Fraction
    interpolation_reverse_weight: Fraction
    residual_multiplier: Fraction


FULL_STEP = Design(
    name="full_step_fidelity_floor",
    cap=Fraction(1, 8),
    passive_divisor=Fraction(8_192),
    learning_rate=Fraction(1, 32_000),
    tau=Fraction(16_777_209, 16_777_216),
    storage=(
        (Fraction(13_151, 100_000), Fraction(-6_559, 50_000)),
        (Fraction(-6_559, 50_000), Fraction(6_547, 50_000)),
    ),
    function_storage=Fraction(1),
    interpolation_reverse_weight=Fraction(24_999, 5_000),
    residual_multiplier=Fraction(737, 100_000),
)
HIGH_FIDELITY = Design(
    name="high_fidelity_reduced_step",
    cap=Fraction(3, 4),
    passive_divisor=Fraction(4_096),
    learning_rate=Fraction(1, 128_000),
    tau=Fraction(16_777_215, 16_777_216),
    storage=(
        (Fraction(1_792, 7_757), Fraction(-2_143, 9_279)),
        (Fraction(-2_143, 9_279), Fraction(2_168, 9_389)),
    ),
    function_storage=Fraction(1),
    interpolation_reverse_weight=Fraction(1_831, 200),
    residual_multiplier=Fraction(110, 9_963),
)
DESIGNS = (FULL_STEP, HIGH_FIDELITY)


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


def _pointwise_sector(design: Design) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    lower = (1 - design.cap) * YOSIDA_LOWER / design.passive_divisor
    pure_yosida_upper = YOSIDA_UPPER / design.passive_divisor
    endpoint_upper = (
        1 - design.cap
    ) * YOSIDA_UPPER / design.passive_divisor + design.cap * RAW_SHAPE_GAIN_UPPER
    upper = max(pure_yosida_upper, endpoint_upper)
    return lower, upper, (lower + upper) / 2, (upper - lower) / 2


def _exact_lmi(design: Design) -> dict[str, object]:
    dimension = 4
    basis = tuple(
        tuple(Fraction(int(row == column)) for row in range(dimension))
        for column in range(dimension)
    )
    momentum, gradient, residual, gradient_next = basis
    null = (Fraction(0),) * dimension
    lower, upper, center, radius = _pointwise_sector(design)
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


def _jordan_fraction(value: Fraction) -> tuple[Fraction, Fraction]:
    a, b, c = COEFFICIENTS
    response = value
    derivative = Fraction(1)
    for _ in range(STEPS):
        square = response**2
        derivative *= a + 3 * b * square + 5 * c * square**2
        response *= a + b * square + c * square**2
    return response, derivative


def _fraction_sha256(value: Fraction) -> str:
    return hashlib.sha256(f"{value.numerator}/{value.denominator}".encode()).hexdigest()


def _unsafe_control() -> dict[str, Fraction | str]:
    u = EPSILON * UNSAFE_Z
    response, derivative = _jordan_fraction(UNSAFE_T)
    primitive = UNIT_DEFICIT_UPPER * UNSAFE_Z
    source = u + LAMBDA * (response + primitive + MU * u)
    direct_derivative = derivative * (1 - UNSAFE_T) ** 2 / EPSILON
    forward_derivative = 1 + LAMBDA * (direct_derivative + UNIT_DEFICIT_UPPER / EPSILON + MU)
    raw_shape_derivative = direct_derivative / forward_derivative
    yosida_derivative = (1 / LAMBDA) * (1 - 1 / forward_derivative)
    gated_derivative = (
        1 - FULL_STEP.cap
    ) * yosida_derivative / FULL_STEP.passive_divisor + FULL_STEP.cap * raw_shape_derivative
    second_jury_margin = FULL_STEP.learning_rate * (1 - BETA) * gated_derivative
    return {
        "u": u,
        "response": response,
        "derivative": derivative,
        "source": source,
        "direct_derivative": direct_derivative,
        "forward_derivative": forward_derivative,
        "raw_shape_derivative": raw_shape_derivative,
        "yosida_derivative": yosida_derivative,
        "gated_derivative": gated_derivative,
        "second_jury_margin": second_jury_margin,
        "response_sha256": _fraction_sha256(response),
        "derivative_sha256": _fraction_sha256(derivative),
        "source_sha256": _fraction_sha256(source),
        "raw_shape_derivative_sha256": _fraction_sha256(raw_shape_derivative),
        "gated_derivative_sha256": _fraction_sha256(gated_derivative),
        "second_jury_margin_sha256": _fraction_sha256(second_jury_margin),
    }


def _strings(values: tuple[Fraction, ...]) -> list[str]:
    return [str(value) for value in values]


def _matrix_strings(matrix: RationalMatrix) -> list[list[str]]:
    return [[str(value) for value in row] for row in matrix]


def _smootherstep(value: Fraction) -> Fraction:
    return value**3 * (
        SMOOTHERSTEP_COEFFICIENTS[0]
        + SMOOTHERSTEP_COEFFICIENTS[1] * value
        + SMOOTHERSTEP_COEFFICIENTS[2] * value**2
    )


def _smootherstep_first(value: Fraction) -> Fraction:
    return 30 * value**2 * (1 - value) ** 2


def _smootherstep_second(value: Fraction) -> Fraction:
    return 60 * value * (1 - value) * (1 - 2 * value)


def _reconstruction_fields() -> tuple[dict[str, object], dict[str, bool]]:
    raw_formula = (DERIVATIVE_UPPER / EPSILON) / (
        1 + LAMBDA * (MU + (UNIT_DEFICIT_UPPER + DERIVATIVE_UPPER) / EPSILON)
    )
    join_jets = [
        [
            str(value)
            for value in (_smootherstep(0), _smootherstep_first(0), _smootherstep_second(0))
        ],
        [
            str(value)
            for value in (_smootherstep(1), _smootherstep_first(1), _smootherstep_second(1))
        ],
    ]
    maximum_q_slope = _smootherstep_first(Fraction(1, 2)) / (GATE_Q1 - GATE_Q0)

    design_fields: dict[str, object] = {}
    lmis: dict[str, dict[str, object]] = {}
    for design in DESIGNS:
        lmi = _exact_lmi(design)
        lmis[design.name] = lmi
        design_fields[design.name] = {
            "cap": str(design.cap),
            "passive_divisor": str(design.passive_divisor),
            "learning_rate": str(design.learning_rate),
            "tau": str(design.tau),
            "sector": {
                "lower": str(lmi["lower"]),
                "upper": str(lmi["upper"]),
                "center": str(lmi["center"]),
                "radius": str(lmi["radius"]),
            },
            "pl": {
                "storage": _matrix_strings(design.storage),
                "function_storage": str(design.function_storage),
                "interpolation_reverse_weight": str(design.interpolation_reverse_weight),
                "residual_multiplier": str(design.residual_multiplier),
                "storage_leading_minors": _strings(lmi["storage_minors"]),
                "negative_lmi_leading_minors": _strings(lmi["negative_lmi_minors"]),
                "rate_squared": str(design.tau**2),
            },
        }

    unsafe = _unsafe_control()
    fields = {
        "raw_shape_gain_upper": str(raw_formula),
        "gate": {
            "q0": str(GATE_Q0),
            "q1": str(GATE_Q1),
            "smootherstep_coefficients_descending": _strings(SMOOTHERSTEP_COEFFICIENTS_DESCENDING),
            "join_jets": join_jets,
            "max_q_slope_at_unit_cap": str(maximum_q_slope),
        },
        "designs": design_fields,
        "unsafe": {
            "t": str(UNSAFE_T),
            "z": str(UNSAFE_Z),
            "u": str(unsafe["u"]),
            "response_sha256": unsafe["response_sha256"],
            "derivative_sha256": unsafe["derivative_sha256"],
            "source_sha256": unsafe["source_sha256"],
            "raw_shape_derivative_sha256": unsafe["raw_shape_derivative_sha256"],
            "source_guard": _strings(SOURCE_GUARD),
            "raw_shape_derivative_guard": _strings(RAW_SHAPE_DERIVATIVE_GUARD),
            "yosida_derivative_guard": _strings(YOSIDA_DERIVATIVE_GUARD),
            "undersized_gate_radii": _strings(UNDERSIZED_GATE_RADII),
            "gated_derivative_sha256": unsafe["gated_derivative_sha256"],
            "gated_derivative_guard": _strings(GATED_DERIVATIVE_GUARD),
            "second_jury_margin_sha256": unsafe["second_jury_margin_sha256"],
        },
    }
    checks = {
        "raw_shape_gain_formula_matches_lock": raw_formula == RAW_SHAPE_GAIN_UPPER,
        "tail_endpoints_dominate_unit_deficit": -TAIL_DERIVATIVE_LOWER > UNIT_DEFICIT_UPPER
        and -TAIL_PROJECTION_LOWER > UNIT_DEFICIT_UPPER,
        "left_gate_join_is_C2": join_jets[0] == ["0", "0", "0"],
        "right_gate_join_is_C2": join_jets[1] == ["1", "0", "0"],
        "maximum_q_slope_is_five_halves": maximum_q_slope == Fraction(5, 2),
        "all_storage_matrices_are_positive_definite": all(
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
        "all_LMI_multipliers_are_nonnegative": all(
            all(value >= 0 for value in lmis[design.name]["multipliers"]) for design in DESIGNS
        ),
        "unsafe_response_hash_matches": unsafe["response_sha256"] == RESPONSE_SHA256,
        "unsafe_derivative_hash_matches": unsafe["derivative_sha256"] == DERIVATIVE_SHA256,
        "unsafe_source_hash_matches": unsafe["source_sha256"] == SOURCE_SHA256,
        "unsafe_raw_shape_derivative_hash_matches": unsafe["raw_shape_derivative_sha256"]
        == RAW_SHAPE_DERIVATIVE_SHA256,
        "unsafe_source_guard_closes": SOURCE_GUARD[0] < unsafe["source"] < SOURCE_GUARD[1],
        "unsafe_raw_shape_derivative_guard_closes": RAW_SHAPE_DERIVATIVE_GUARD[0]
        < unsafe["raw_shape_derivative"]
        < RAW_SHAPE_DERIVATIVE_GUARD[1],
        "unsafe_yosida_derivative_guard_closes": YOSIDA_DERIVATIVE_GUARD[0]
        < unsafe["yosida_derivative"]
        < YOSIDA_DERIVATIVE_GUARD[1],
        "undersized_gate_is_on_raw_shape_plateau": unsafe["source"] > UNDERSIZED_GATE_RADII[1],
        "unsafe_gated_derivative_hash_matches": unsafe["gated_derivative_sha256"]
        == GATED_DERIVATIVE_SHA256,
        "unsafe_gated_derivative_guard_closes": GATED_DERIVATIVE_GUARD[0]
        < unsafe["gated_derivative"]
        < GATED_DERIVATIVE_GUARD[1],
        "second_jury_margin_hash_matches": unsafe["second_jury_margin_sha256"]
        == SECOND_JURY_MARGIN_SHA256,
        "second_jury_margin_is_negative": unsafe["second_jury_margin"] < 0,
    }
    return fields, checks


def _reconstruct() -> dict[str, object]:
    fields, checks = _reconstruction_fields()
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"independent P17 reconstruction failed: {failed}")
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
        raise SystemExit("canonical P17 certificate is missing or mismatched")
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
