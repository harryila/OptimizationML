#!/usr/bin/env python3
"""Independently reconstruct the exact P5 full-step convergence LMI.

This replay intentionally imports no project package or numerical algebra
library.  It starts from the published rational theorem data, constructs every
quadratic form from the stated dynamics and interpolation inequality, and uses
a local recursive determinant routine for the Sylvester checks.  The committed
canonical manifest is read only after that reconstruction has finished.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import TypeAlias

Vector: TypeAlias = tuple[Fraction, ...]
Matrix: TypeAlias = tuple[tuple[Fraction, ...], ...]

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANONICAL = ROOT / "results/summaries/nonquadratic_convergence_certificate.json"


def _zero_matrix(size: int) -> Matrix:
    return tuple(tuple(Fraction(0) for _ in range(size)) for _ in range(size))


def _add(*matrices: Matrix) -> Matrix:
    size = len(matrices[0])
    return tuple(
        tuple(
            sum((matrix[row][column] for matrix in matrices), Fraction(0)) for column in range(size)
        )
        for row in range(size)
    )


def _scale(scale: Fraction, matrix: Matrix) -> Matrix:
    return tuple(tuple(scale * value for value in row) for row in matrix)


def _outer(left: Vector, right: Vector) -> Matrix:
    return tuple(tuple(left_value * right_value for right_value in right) for left_value in left)


def _symmetric_product(left: Vector, right: Vector) -> Matrix:
    return _scale(Fraction(1, 2), _add(_outer(left, right), _outer(right, left)))


def _vector_add(*vectors: Vector) -> Vector:
    return tuple(
        sum((vector[index] for vector in vectors), Fraction(0)) for index in range(len(vectors[0]))
    )


def _vector_scale(scale: Fraction, vector: Vector) -> Vector:
    return tuple(scale * value for value in vector)


def _quadratic_storage(rows: tuple[Vector, Vector], storage: Matrix) -> Matrix:
    return _add(
        _scale(storage[0][0], _outer(rows[0], rows[0])),
        _scale(2 * storage[0][1], _symmetric_product(rows[0], rows[1])),
        _scale(storage[1][1], _outer(rows[1], rows[1])),
    )


def _interpolation_quadratic(
    *,
    input_i: Vector,
    gradient_i: Vector,
    input_j: Vector,
    gradient_j: Vector,
    condition_ratio: Fraction,
) -> Matrix:
    """Quadratic part of the exact normalized F_(k,1) inequality."""

    dx = _vector_add(input_i, _vector_scale(-1, input_j))
    du = _vector_add(gradient_i, _vector_scale(-1, gradient_j))
    phi = _scale(
        Fraction(1, 2) / (1 - condition_ratio),
        _add(
            _outer(du, du),
            _scale(-2 * condition_ratio, _symmetric_product(dx, du)),
            _scale(condition_ratio, _outer(dx, dx)),
        ),
    )
    return _add(_scale(-1, _symmetric_product(gradient_j, dx)), _scale(-1, phi))


def _determinant(matrix: Matrix) -> Fraction:
    """Compute an exact determinant by local cofactor expansion."""

    size = len(matrix)
    if size == 0:
        return Fraction(1)
    if any(len(row) != size for row in matrix):
        raise ValueError("determinant requires a square matrix")
    if size == 1:
        return matrix[0][0]
    result = Fraction(0)
    for column, value in enumerate(matrix[0]):
        if value == 0:
            continue
        minor = tuple(
            tuple(row[index] for index in range(size) if index != column) for row in matrix[1:]
        )
        result += (-1 if column % 2 else 1) * value * _determinant(minor)
    return result


def _leading_minors(matrix: Matrix) -> tuple[Fraction, ...]:
    return tuple(
        _determinant(
            tuple(tuple(matrix[row][column] for column in range(size)) for row in range(size))
        )
        for size in range(1, len(matrix) + 1)
    )


def reconstruct() -> dict[str, object]:
    """Build the certificate independently from the published rationals."""

    dimension = 5
    basis = tuple(
        tuple(Fraction(row == column) for row in range(dimension)) for column in range(dimension)
    )
    w, momentum, gradient, residual, gradient_next = basis
    zero = (Fraction(0),) * dimension

    ell = Fraction(1)
    smoothness = Fraction(10)
    beta = Fraction(19, 20)
    learning_rate = Fraction(1, 32_000)
    center_gain = Fraction(505_021_761_888_849, 520_522_720_000)
    residual_lipschitz = Fraction(251_582_619_905_461, 520_522_720_000)
    tau = Fraction(2_499, 2_500)
    storage = (
        (Fraction(495_723, 100_000_000), Fraction(-3_085_119, 100_000_000)),
        (Fraction(-3_085_119, 100_000_000), Fraction(72_422_647, 100_000_000)),
    )
    function_storage = Fraction(2_708_163, 10_000_000)
    edge_weights = {
        (0, 1): Fraction(67_589, 25_000_000),
        (1, 2): Fraction(17_111_528_143_163, 62_500_000_000_000),
        (2, 0): Fraction(155_434_393_163, 62_500_000_000_000),
        (2, 1): Fraction(1_203, 2_500_000),
    }
    residual_weight = Fraction(316_641, 6_250_000)

    alpha = learning_rate * center_gain * smoothness
    signal = _vector_add(
        _vector_scale(beta**2, momentum),
        _vector_scale(1 - beta**2, gradient),
    )
    w_next = _vector_add(
        w,
        _vector_scale(-alpha, signal),
        _vector_scale(-learning_rate * residual_lipschitz * smoothness, residual),
    )
    momentum_next = _vector_add(
        _vector_scale(beta, momentum),
        _vector_scale(1 - beta, gradient),
    )

    current_storage = _quadratic_storage((w, momentum), storage)
    next_storage = _quadratic_storage((w_next, momentum_next), storage)
    residual_supply = _add(_outer(signal, signal), _scale(-1, _outer(residual, residual)))

    points = {
        0: (zero, zero),
        1: (w, gradient),
        2: (w_next, gradient_next),
    }
    interpolation_supplies: dict[tuple[int, int], Matrix] = {}
    for edge in edge_weights:
        index_i, index_j = edge
        interpolation_supplies[edge] = _interpolation_quadratic(
            input_i=points[index_i][0],
            gradient_i=points[index_i][1],
            input_j=points[index_j][0],
            gradient_j=points[index_j][1],
            condition_ratio=ell / smoothness,
        )

    lmi_terms = [next_storage, _scale(-(tau**2), current_storage)]
    lmi_terms.extend(
        _scale(edge_weights[edge], interpolation_supplies[edge]) for edge in edge_weights
    )
    lmi_terms.append(_scale(residual_weight, residual_supply))
    lmi = _add(*lmi_terms)

    function_coefficients = (
        -edge_weights[(0, 1)] + edge_weights[(1, 2)] - edge_weights[(2, 1)],
        -edge_weights[(1, 2)] + edge_weights[(2, 0)] + edge_weights[(2, 1)],
    )
    expected_function_coefficients = (function_storage * tau**2, -function_storage)
    storage_minors = _leading_minors(storage)
    negative_lmi_minors = _leading_minors(_scale(-1, lmi))

    return {
        "lmi_matrix": lmi,
        "storage_leading_minors": storage_minors,
        "negative_lmi_leading_minors": negative_lmi_minors,
        "function_coefficients": function_coefficients,
        "expected_function_coefficients": expected_function_coefficients,
        "storage_positive_definite": all(value > 0 for value in storage_minors),
        "lmi_negative_definite": all(value > 0 for value in negative_lmi_minors),
        "function_values_cancel": function_coefficients == expected_function_coefficients,
    }


def _parse_fraction_matrix(values: list[list[str]]) -> Matrix:
    return tuple(tuple(Fraction(value) for value in row) for row in values)


def _parse_fraction_items(values: list[dict[str, str]]) -> tuple[Fraction, ...]:
    return tuple(Fraction(value["exact"]) for value in values)


def compare_with_canonical(reconstruction: dict[str, object], path: Path) -> dict[str, object]:
    """Read the canonical result after reconstruction and compare exact fields."""

    canonical_bytes = path.read_bytes()
    canonical = json.loads(canonical_bytes)
    locked = canonical["locked_exact_certificate"]
    interpolation = canonical["objective_interpolation"]["weighted_function_coefficients"]
    comparisons = {
        "lmi_matrix": reconstruction["lmi_matrix"] == _parse_fraction_matrix(locked["lmi_matrix"]),
        "storage_leading_minors": reconstruction["storage_leading_minors"]
        == _parse_fraction_items(locked["storage_leading_principal_minors"]),
        "negative_lmi_leading_minors": reconstruction["negative_lmi_leading_minors"]
        == _parse_fraction_items(locked["negative_lmi_leading_principal_minors"]),
        "function_coefficients": reconstruction["function_coefficients"]
        == tuple(Fraction(value["exact"]) for value in interpolation["actual_current_next"]),
        "expected_function_coefficients": reconstruction["expected_function_coefficients"]
        == tuple(
            Fraction(value["exact"]) for value in interpolation["expected_to_cancel_lyapunov_gap"]
        ),
    }
    return {
        "path": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
        "sha256": hashlib.sha256(canonical_bytes).hexdigest(),
        "comparisons": comparisons,
        "all_exact_fields_match": all(comparisons.values()),
    }


def _fraction_strings(values: object) -> object:
    if isinstance(values, Fraction):
        return str(values)
    if isinstance(values, tuple):
        return [_fraction_strings(value) for value in values]
    if isinstance(values, dict):
        return {key: _fraction_strings(value) for key, value in values.items()}
    return values


def build_payload(canonical_path: Path = DEFAULT_CANONICAL) -> dict[str, object]:
    # Keep this ordering explicit: no canonical artifact is read until the
    # independent construction and exact definiteness checks are complete.
    reconstruction = reconstruct()
    canonical_comparison = compare_with_canonical(reconstruction, canonical_path)
    certified = (
        bool(reconstruction["storage_positive_definite"])
        and bool(reconstruction["lmi_negative_definite"])
        and bool(reconstruction["function_values_cancel"])
        and bool(canonical_comparison["all_exact_fields_match"])
    )
    return {
        "schema_version": "passive-muon-p5-independent-reconstruction-v1",
        "implementation_scope": {
            "arithmetic": "stdlib fractions.Fraction",
            "project_package_imported": False,
            "numerical_algebra_library_imported": False,
            "determinant_method": "local recursive cofactor expansion",
            "canonical_read_order": "only after independent matrix reconstruction",
            "qualification": "independent code-path reconstruction; not a human proof audit",
        },
        "reconstruction": _fraction_strings(reconstruction),
        "canonical_comparison": canonical_comparison,
        "all_exact_checks_passed": certified,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_payload(args.canonical)
    if not payload["all_exact_checks_passed"]:
        raise AssertionError("independent P5 reconstruction failed")
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
