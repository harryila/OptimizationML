#!/usr/bin/env python3
"""Independently reconstruct the exact P6 smooth-PL convergence LMI.

This replay imports no project package or numerical algebra library.  It
starts from the published rational theorem parameters, builds the two directed
nonconvex interpolation supplies and the complete 4-by-4 dissipation LMI, and
checks definiteness with a local exact determinant implementation.  The
committed canonical artifact is read only after the independent construction
and its internal exact checks have finished.
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
DEFAULT_CANONICAL = ROOT / "results/summaries/pl_convergence_certificate.json"

# The canonical JSON was generated at SOURCE_COMMIT.  Later documentation and
# integrity work culminated in CHECKPOINT_COMMIT, tagged p6-checkpoint.  Keeping
# these roles separate avoids presenting the generation SHA as the final P6 SHA.
SOURCE_COMMIT = "a8f650f6c60dcbc5d2f83647fd367348f4c67548"
CHECKPOINT_COMMIT = "ef88d8f5b26148af0ec1ca70b506048938bf9bef"
CHECKPOINT_TAG = "p6-checkpoint"


def _add(*matrices: Matrix) -> Matrix:
    """Add equally shaped matrices over the rationals."""

    shape = (len(matrices[0]), len(matrices[0][0]))
    if any((len(matrix), len(matrix[0])) != shape for matrix in matrices):
        raise ValueError("matrix dimensions do not agree")
    return tuple(
        tuple(
            sum((matrix[row][column] for matrix in matrices), Fraction(0))
            for column in range(shape[1])
        )
        for row in range(shape[0])
    )


def _scale(scale: Fraction, matrix: Matrix) -> Matrix:
    return tuple(tuple(scale * value for value in row) for row in matrix)


def _outer(left: Vector, right: Vector) -> Matrix:
    return tuple(tuple(left_value * right_value for right_value in right) for left_value in left)


def _product(left: Vector, right: Vector) -> Matrix:
    """Return the symmetric matrix representing ``left(x) * right(x)``."""

    return _scale(Fraction(1, 2), _add(_outer(left, right), _outer(right, left)))


def _square(vector: Vector) -> Matrix:
    return _outer(vector, vector)


def _vector_add(*vectors: Vector) -> Vector:
    if any(len(vector) != len(vectors[0]) for vector in vectors):
        raise ValueError("vector dimensions do not agree")
    return tuple(
        sum((vector[index] for vector in vectors), Fraction(0)) for index in range(len(vectors[0]))
    )


def _vector_scale(scale: Fraction, vector: Vector) -> Vector:
    return tuple(scale * value for value in vector)


def _storage_quadratic(coordinates: tuple[Vector, Vector], storage: Matrix) -> Matrix:
    """Expand ``coordinates.T @ storage @ coordinates`` as a quadratic form."""

    first, second = coordinates
    return _add(
        _scale(storage[0][0], _square(first)),
        _scale(2 * storage[0][1], _product(first, second)),
        _scale(storage[1][1], _square(second)),
    )


def _directed_nonconvex_interpolation(
    *,
    input_i: Vector,
    gradient_i: Vector,
    input_j: Vector,
    gradient_j: Vector,
) -> Matrix:
    """Build the quadratic part of the directed ``(-1, 1)`` supply.

    With ``dx=x_i-x_j`` and ``du=u_i-u_j``, the complete supply is

    ``F_i-F_j-u_j*dx-(du**2+2*dx*du-dx**2)/4 >= 0``.

    The construction is scalar-coordinate algebra.  Tensoring the resulting
    matrix with an identity gives the Frobenius-space inequality in every
    finite matrix dimension.
    """

    dx = _vector_add(input_i, _vector_scale(Fraction(-1), input_j))
    du = _vector_add(gradient_i, _vector_scale(Fraction(-1), gradient_j))
    smooth_model = _scale(
        Fraction(1, 4),
        _add(
            _square(du),
            _scale(Fraction(2), _product(dx, du)),
            _scale(Fraction(-1), _square(dx)),
        ),
    )
    return _add(
        _scale(Fraction(-1), _product(gradient_j, dx)),
        _scale(Fraction(-1), smooth_model),
    )


def _determinant(matrix: Matrix) -> Fraction:
    """Compute an exact determinant using fraction-preserving Bareiss steps."""

    size = len(matrix)
    if any(len(row) != size for row in matrix):
        raise ValueError("determinant requires a square matrix")
    if size == 0:
        return Fraction(1)
    work = [list(row) for row in matrix]
    sign = Fraction(1)
    previous_pivot = Fraction(1)
    for pivot_index in range(size - 1):
        if work[pivot_index][pivot_index] == 0:
            swap_index = next(
                (row for row in range(pivot_index + 1, size) if work[row][pivot_index] != 0),
                None,
            )
            if swap_index is None:
                return Fraction(0)
            work[pivot_index], work[swap_index] = work[swap_index], work[pivot_index]
            sign *= -1
        pivot = work[pivot_index][pivot_index]
        for row in range(pivot_index + 1, size):
            for column in range(pivot_index + 1, size):
                work[row][column] = (
                    work[row][column] * pivot - work[row][pivot_index] * work[pivot_index][column]
                ) / previous_pivot
        previous_pivot = pivot
    return sign * work[-1][-1]


def _leading_principal_minors(matrix: Matrix) -> tuple[Fraction, ...]:
    return tuple(
        _determinant(
            tuple(tuple(matrix[row][column] for column in range(size)) for row in range(size))
        )
        for size in range(1, len(matrix) + 1)
    )


def _is_symmetric(matrix: Matrix) -> bool:
    return len(matrix) == len(matrix[0]) and all(
        matrix[row][column] == matrix[column][row]
        for row in range(len(matrix))
        for column in range(len(matrix))
    )


def reconstruct() -> dict[str, object]:
    """Build and audit P6 solely from the published exact theorem parameters."""

    dimension = 4
    basis = tuple(
        tuple(Fraction(row == column) for row in range(dimension)) for column in range(dimension)
    )
    momentum, gradient, residual, gradient_next = basis
    zero = (Fraction(0),) * dimension

    # Objective, optimizer, and operator-decomposition parameters.
    pl_constant = Fraction(1)
    smoothness = Fraction(10)
    condition_ratio = pl_constant / smoothness
    beta = Fraction(19, 20)
    learning_rate = Fraction(1, 32_000)
    center_gain = Fraction(505_021_761_888_849, 520_522_720_000)
    residual_lipschitz = Fraction(251_582_619_905_461, 520_522_720_000)
    dimensionless_step = learning_rate * center_gain * smoothness
    residual_ratio = residual_lipschitz / center_gain

    # Storage and nonnegative supply multipliers.
    tau = Fraction(19_999, 20_000)
    storage = (
        (Fraction(72_435, 100_000), Fraction(-3_185, 100_000)),
        (Fraction(-3_185, 100_000), Fraction(499, 100_000)),
    )
    function_storage = Fraction(27_066, 100_000)
    reverse_interpolation_weight = Fraction(20_753, 100_000)
    interpolation_12_weight = reverse_interpolation_weight + function_storage * tau**2
    interpolation_21_weight = reverse_interpolation_weight
    pl_next_weight = function_storage * (1 - tau**2) / (2 * condition_ratio)
    residual_weight = Fraction(5_052, 100_000)

    # chi=(m/L, grad(f)/L, E(s)/(K_E L), grad(f_next)/L).
    signal = _vector_add(
        _vector_scale(beta**2, momentum),
        _vector_scale(1 - beta**2, gradient),
    )
    step = _vector_add(
        _vector_scale(-dimensionless_step, signal),
        _vector_scale(-learning_rate * residual_lipschitz * smoothness, residual),
    )
    momentum_next = _vector_add(
        _vector_scale(beta, momentum),
        _vector_scale(1 - beta, gradient),
    )

    transition = (momentum_next, gradient_next)
    selection = (momentum, gradient)
    current_storage = _storage_quadratic(selection, storage)
    next_storage = _storage_quadratic(transition, storage)

    interpolation_12 = _directed_nonconvex_interpolation(
        input_i=zero,
        gradient_i=gradient,
        input_j=step,
        gradient_j=gradient_next,
    )
    interpolation_21 = _directed_nonconvex_interpolation(
        input_i=step,
        gradient_i=gradient_next,
        input_j=zero,
        gradient_j=gradient,
    )
    pl_next = _square(gradient_next)
    residual_supply = _add(_square(signal), _scale(Fraction(-1), _square(residual)))

    lmi = _add(
        next_storage,
        _scale(-(tau**2), current_storage),
        _scale(interpolation_12_weight, interpolation_12),
        _scale(interpolation_21_weight, interpolation_21),
        _scale(pl_next_weight, pl_next),
        _scale(residual_weight, residual_supply),
    )

    function_coefficients = (
        interpolation_12_weight - interpolation_21_weight,
        -interpolation_12_weight + interpolation_21_weight - 2 * condition_ratio * pl_next_weight,
    )
    expected_function_coefficients = (function_storage * tau**2, -function_storage)
    storage_minors = _leading_principal_minors(storage)
    negative_lmi_minors = _leading_principal_minors(_scale(Fraction(-1), lmi))
    checks = {
        "storage_symmetric": _is_symmetric(storage),
        "lmi_symmetric": _is_symmetric(lmi),
        "storage_positive_definite": all(value > 0 for value in storage_minors),
        "lmi_negative_definite": all(value > 0 for value in negative_lmi_minors),
        "function_values_cancel": function_coefficients == expected_function_coefficients,
        "multipliers_nonnegative": all(
            value >= 0
            for value in (
                interpolation_12_weight,
                interpolation_21_weight,
                pl_next_weight,
                residual_weight,
            )
        ),
    }

    return {
        "parameters": {
            "pl_constant": pl_constant,
            "smoothness": smoothness,
            "condition_ratio": condition_ratio,
            "beta": beta,
            "learning_rate": learning_rate,
            "center_gain": center_gain,
            "residual_lipschitz": residual_lipschitz,
            "residual_ratio": residual_ratio,
            "dimensionless_step": dimensionless_step,
            "tau": tau,
            "function_value_rate": tau**2,
            "storage": storage,
            "function_storage": function_storage,
            "storage_normalization": storage[0][0] + storage[1][1] + function_storage,
            "multipliers": {
                "interpolation_current_to_next": interpolation_12_weight,
                "interpolation_next_to_current": interpolation_21_weight,
                "pl_at_next": pl_next_weight,
                "residual_lipschitz": residual_weight,
            },
        },
        "dynamics": {
            "transition_matrix": transition,
            "state_selection_matrix": selection,
            "signal_selector": signal,
            "step_selector": step,
        },
        "supplies": {
            "interpolation_current_to_next": interpolation_12,
            "interpolation_next_to_current": interpolation_21,
            "pl_at_next": pl_next,
            "residual_lipschitz": residual_supply,
        },
        "lmi_matrix": lmi,
        "function_coefficients": function_coefficients,
        "expected_function_coefficients": expected_function_coefficients,
        "storage_leading_principal_minors": storage_minors,
        "negative_lmi_leading_principal_minors": negative_lmi_minors,
        "checks": checks,
        "all_internal_exact_checks_passed": all(checks.values()),
    }


def _fraction(payload: dict[str, str]) -> Fraction:
    return Fraction(payload["exact"])


def _fraction_items(payload: list[dict[str, str]]) -> tuple[Fraction, ...]:
    return tuple(_fraction(item) for item in payload)


def _fraction_matrix(payload: list[list[str]]) -> Matrix:
    return tuple(tuple(Fraction(value) for value in row) for row in payload)


def compare_with_canonical(reconstruction: dict[str, object], path: Path) -> dict[str, object]:
    """Read the canonical artifact and compare every reconstructed exact field."""

    canonical_bytes = path.read_bytes()
    canonical = json.loads(canonical_bytes)
    objective = canonical["objective_class"]
    decomposition = canonical["operator_decomposition"]
    locked = canonical["locked_exact_certificate"]
    parameters = reconstruction["parameters"]
    dynamics = reconstruction["dynamics"]
    supplies = reconstruction["supplies"]
    multipliers = parameters["multipliers"]
    canonical_multipliers = locked["multipliers"]
    interpolation = objective["weighted_function_coefficients"]

    comparisons = {
        "generation_source_commit": canonical["git"]["sha"] == SOURCE_COMMIT,
        "pl_constant": parameters["pl_constant"] == _fraction(objective["pl_constant_ell_PL"]),
        "smoothness": parameters["smoothness"] == _fraction(objective["smoothness_L"]),
        "condition_ratio": parameters["condition_ratio"]
        == _fraction(objective["normalized_pl_constant_k"]),
        "center_gain": parameters["center_gain"] == _fraction(decomposition["center_gain_gamma"]),
        "residual_lipschitz": parameters["residual_lipschitz"]
        == _fraction(decomposition["residual_lipschitz_K_E"]),
        "residual_ratio": parameters["residual_ratio"]
        == _fraction(decomposition["residual_ratio_K_E_over_gamma"]),
        "beta": parameters["beta"] == _fraction(locked["beta"]),
        "learning_rate": parameters["learning_rate"] == _fraction(locked["learning_rate_eta"]),
        "tau": parameters["tau"] == _fraction(locked["exponential_norm_rate_tau"]),
        "function_value_rate": parameters["function_value_rate"]
        == _fraction(locked["function_value_rate_q"]),
        "dimensionless_step": parameters["dimensionless_step"]
        == _fraction(locked["dimensionless_step_eta_gamma_L"]),
        "storage": parameters["storage"] == _fraction_matrix(locked["storage_P"]),
        "function_storage": parameters["function_storage"]
        == _fraction(locked["function_storage_c_F"]),
        "storage_normalization": parameters["storage_normalization"]
        == _fraction(locked["storage_normalization_trace_P_plus_c_F"]),
        "interpolation_current_to_next_weight": multipliers["interpolation_current_to_next"]
        == _fraction(canonical_multipliers["interpolation_current_to_next"]),
        "interpolation_next_to_current_weight": multipliers["interpolation_next_to_current"]
        == _fraction(canonical_multipliers["interpolation_next_to_current"]),
        "pl_at_next_weight": multipliers["pl_at_next"]
        == _fraction(canonical_multipliers["pl_at_next"]),
        "residual_lipschitz_weight": multipliers["residual_lipschitz"]
        == _fraction(canonical_multipliers["residual_lipschitz"]),
        "transition_matrix": dynamics["transition_matrix"]
        == _fraction_matrix(locked["transition_matrix"]),
        "state_selection_matrix": dynamics["state_selection_matrix"]
        == _fraction_matrix(locked["state_selection_matrix"]),
        "signal_selector": dynamics["signal_selector"]
        == tuple(Fraction(value) for value in locked["signal_selector"]),
        "step_selector": dynamics["step_selector"]
        == tuple(Fraction(value) for value in locked["step_selector"]),
        "interpolation_current_to_next_matrix": supplies["interpolation_current_to_next"]
        == _fraction_matrix(locked["interpolation_quadratic_matrices"]["current_to_next"]),
        "interpolation_next_to_current_matrix": supplies["interpolation_next_to_current"]
        == _fraction_matrix(locked["interpolation_quadratic_matrices"]["next_to_current"]),
        "pl_at_next_matrix": supplies["pl_at_next"]
        == _fraction_matrix(locked["pl_next_quadratic_matrix"]),
        "residual_lipschitz_matrix": supplies["residual_lipschitz"]
        == _fraction_matrix(locked["residual_iqc_matrix"]),
        "lmi_matrix": reconstruction["lmi_matrix"] == _fraction_matrix(locked["lmi_matrix"]),
        "function_coefficients": reconstruction["function_coefficients"]
        == _fraction_items(interpolation["actual_current_next"]),
        "expected_function_coefficients": reconstruction["expected_function_coefficients"]
        == _fraction_items(interpolation["expected_to_cancel_storage_gap"]),
        "storage_leading_principal_minors": reconstruction["storage_leading_principal_minors"]
        == _fraction_items(locked["storage_leading_principal_minors"]),
        "negative_lmi_leading_principal_minors": reconstruction[
            "negative_lmi_leading_principal_minors"
        ]
        == _fraction_items(locked["negative_lmi_leading_principal_minors"]),
        "canonical_exact_flags": all(
            locked[key]
            for key in (
                "storage_positive_definite",
                "lmi_negative_definite",
                "function_values_cancel",
                "all_exact_checks_passed",
            )
        ),
    }
    return {
        "path": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
        "sha256": hashlib.sha256(canonical_bytes).hexdigest(),
        "canonical_generation_source_commit": canonical["git"]["sha"],
        "comparisons": comparisons,
        "all_exact_fields_match": all(comparisons.values()),
    }


def _fraction_strings(value: object) -> object:
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, tuple):
        return [_fraction_strings(item) for item in value]
    if isinstance(value, dict):
        return {key: _fraction_strings(item) for key, item in value.items()}
    return value


def build_payload(canonical_path: Path = DEFAULT_CANONICAL) -> dict[str, object]:
    # This ordering is part of the audit: no canonical bytes are read until a
    # complete, internally checked certificate exists independently in memory.
    reconstruction = reconstruct()
    if not reconstruction["all_internal_exact_checks_passed"]:
        raise AssertionError("independent P6 construction failed its internal exact checks")
    canonical_comparison = compare_with_canonical(reconstruction, canonical_path)
    return {
        "schema_version": "passive-muon-p6-independent-reconstruction-v1",
        "revision_roles": {
            "certificate_generation_source_commit": SOURCE_COMMIT,
            "final_checkpoint_commit": CHECKPOINT_COMMIT,
            "final_checkpoint_tag": CHECKPOINT_TAG,
            "distinction": (
                "a8f650f6 is the certificate generation/source commit; "
                "ef88d8f5, tagged p6-checkpoint, is the final P6 checkpoint"
            ),
        },
        "implementation_scope": {
            "arithmetic": "stdlib fractions.Fraction",
            "project_package_imported": False,
            "numerical_algebra_library_imported": False,
            "determinant_method": "local exact Bareiss elimination",
            "canonical_read_order": "only after complete independent construction and audit",
            "qualification": "independent code-path reconstruction; not a human proof audit",
        },
        "reconstruction": _fraction_strings(reconstruction),
        "canonical_comparison": canonical_comparison,
        "all_exact_checks_passed": bool(canonical_comparison["all_exact_fields_match"]),
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
        raise AssertionError("independent P6 reconstruction disagrees with the canonical artifact")
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
