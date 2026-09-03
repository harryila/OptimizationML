#!/usr/bin/env python3
"""Independently reconstruct the exact P7 robust-dissipativity LMI.

This replay imports neither :mod:`passive_muon` nor a numerical-algebra
library.  Starting only from the published rational parameters, it rebuilds
the disturbed EMA/Nesterov dynamics, both directed smooth-nonconvex
interpolation supplies, the P6 value storage and multipliers, and the complete
6-by-6 dissipation LMI.  Positivity and negativity are checked with a local
exact Bareiss determinant implementation and Sylvester's criterion.

The canonical JSON, when present, is read only *after* the complete independent
construction has passed its internal checks.  This ordering makes the script a
genuine second code path rather than a parser for the primary certificate.
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
DEFAULT_CANONICAL = ROOT / "results/summaries/robust_dissipativity_certificate.json"

# Exact values independently locked for the P7 candidate.  In particular, the
# six minors are checked before any canonical artifact is opened.
EXPECTED_STORAGE_MINORS = (
    Fraction(14_487, 20_000),
    Fraction(650_021, 250_000_000),
)
EXPECTED_NEGATIVE_LMI_MINORS = (
    Fraction(
        588366669441078114305592272723841308118612219507,
        35513159127688596684800000000000000000000000000000,
    ),
    Fraction(
        111653153906865859147084187785008643627781514215908602092179,
        1420526365107543867392000000000000000000000000000000000000000000,
    ),
    Fraction(
        30245282351805947095058111344751251587087360962215007639796547873,
        71026318255377193369600000000000000000000000000000000000000000000000000,
    ),
    Fraction(
        106378237649828474982433094173183543070031301965649332843580933465570058399,
        2841052730215087734784000000000000000000000000000000000000000000000000000000000000000,
    ),
    Fraction(
        358878445547025338702743793976131456516389191266001498332974313243385014147631510993,
        568210546043017546956800000000000000000000000000000000000000000000000000000000000000000000000,
    ),
    Fraction(
        25157750088920031541989779883715177798784065261265275305093677552588813246840551502751954845110066654533994497393,
        3079063650460282933136543676107220582400000000000000000000000000000000000000000000000000000000000000000000000000000000000,
    ),
)


def _add(*matrices: Matrix) -> Matrix:
    """Add equally shaped matrices over the rationals."""

    rows = len(matrices[0])
    columns = len(matrices[0][0])
    if any(len(matrix) != rows for matrix in matrices) or any(
        len(row) != columns for matrix in matrices for row in matrix
    ):
        raise ValueError("matrix dimensions do not agree")
    return tuple(
        tuple(
            sum((matrix[row][column] for matrix in matrices), Fraction(0))
            for column in range(columns)
        )
        for row in range(rows)
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
    """Expand ``coordinates.T @ storage @ coordinates``."""

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
    """Build the quadratic part of directed normalized 1-smooth interpolation.

    With ``dx=x_i-x_j`` and ``du=u_i-u_j``, the complete supply is

    ``F_i-F_j-u_j*dx-(du**2+2*dx*du-dx**2)/4 >= 0``.

    The scalar-coordinate matrix tensors with an identity, yielding the same
    inequality on every finite-dimensional Frobenius space.
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


def _determinant_bareiss(matrix: Matrix) -> Fraction:
    """Compute an exact determinant by local Bareiss elimination."""

    size = len(matrix)
    if any(len(row) != size for row in matrix):
        raise ValueError("determinant requires a square matrix")
    if size == 0:
        return Fraction(1)
    if size == 1:
        return matrix[0][0]

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
            work[row][pivot_index] = Fraction(0)
        previous_pivot = pivot
    return sign * work[-1][-1]


def _leading_principal_minors(matrix: Matrix) -> tuple[Fraction, ...]:
    return tuple(
        _determinant_bareiss(
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
    """Build and audit P7 solely from its stated exact rational data."""

    dimension = 6
    basis = tuple(
        tuple(Fraction(row == column) for row in range(dimension)) for column in range(dimension)
    )
    momentum, gradient, residual, gradient_next, gradient_noise, implementation_error = basis
    zero = (Fraction(0),) * dimension

    # Objective, optimizer, and centered repaired-operator parameters retained
    # exactly from P6.
    pl_constant = Fraction(1)
    smoothness = Fraction(10)
    condition_ratio = pl_constant / smoothness
    beta = Fraction(19, 20)
    learning_rate = Fraction(1, 32_000)
    center_gain = Fraction(505_021_761_888_849, 520_522_720_000)
    residual_lipschitz = Fraction(251_582_619_905_461, 520_522_720_000)
    dimensionless_step = learning_rate * center_gain * smoothness
    residual_ratio = residual_lipschitz / center_gain

    tau = Fraction(19_999, 20_000)
    rate = tau**2
    storage = (
        (Fraction(72_435, 100_000), Fraction(-3_185, 100_000)),
        (Fraction(-3_185, 100_000), Fraction(499, 100_000)),
    )
    function_storage = Fraction(27_066, 100_000)
    reverse_interpolation_weight = Fraction(20_753, 100_000)
    interpolation_12_weight = reverse_interpolation_weight + function_storage * rate
    interpolation_21_weight = reverse_interpolation_weight
    pl_next_weight = function_storage * (1 - rate) / (2 * condition_ratio)
    residual_weight = Fraction(5_052, 100_000)

    # The physical input gains are stated in xi/e coordinates.  The LMI uses
    # w=xi/L and h=e/(gamma L), hence these two exact conversions.
    gradient_noise_gain = Fraction(1, 2)
    implementation_error_gain = Fraction(1, 2_000_000)
    normalized_gradient_noise_penalty = gradient_noise_gain * smoothness**2
    normalized_implementation_error_penalty = (
        implementation_error_gain * (center_gain * smoothness) ** 2
    )

    # chi=(z,u,v,u_next,w,h), with z=m/L, u=grad(f)/L,
    # v=E(s)/(K_E L), w=xi/L, and h=e/(gamma L).
    noisy_gradient = _vector_add(gradient, gradient_noise)
    signal = _vector_add(
        _vector_scale(beta**2, momentum),
        _vector_scale(1 - beta**2, noisy_gradient),
    )
    step = _vector_add(
        _vector_scale(-dimensionless_step, signal),
        _vector_scale(-dimensionless_step * residual_ratio, residual),
        _vector_scale(-dimensionless_step, implementation_error),
    )
    momentum_next = _vector_add(
        _vector_scale(beta, momentum),
        _vector_scale(1 - beta, noisy_gradient),
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
    residual_supply = _add(
        _square(signal),
        _scale(Fraction(-1), _square(residual)),
    )
    gradient_noise_norm = _square(gradient_noise)
    implementation_error_norm = _square(implementation_error)

    lmi = _add(
        next_storage,
        _scale(-rate, current_storage),
        _scale(interpolation_12_weight, interpolation_12),
        _scale(interpolation_21_weight, interpolation_21),
        _scale(pl_next_weight, pl_next),
        _scale(residual_weight, residual_supply),
        _scale(-normalized_gradient_noise_penalty, gradient_noise_norm),
        _scale(-normalized_implementation_error_penalty, implementation_error_norm),
    )

    function_coefficients = (
        interpolation_12_weight - interpolation_21_weight,
        -interpolation_12_weight + interpolation_21_weight - 2 * condition_ratio * pl_next_weight,
    )
    expected_function_coefficients = (function_storage * rate, -function_storage)
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
        "physical_gains_positive": gradient_noise_gain > 0 and implementation_error_gain > 0,
        "normalized_penalties_exact": normalized_gradient_noise_penalty == 50
        and normalized_implementation_error_penalty
        == implementation_error_gain * center_gain**2 * smoothness**2,
        "storage_minors_match_locked_values": storage_minors == EXPECTED_STORAGE_MINORS,
        "six_lmi_minors_match_locked_values": negative_lmi_minors == EXPECTED_NEGATIVE_LMI_MINORS,
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
            "rate": rate,
            "storage": storage,
            "function_storage": function_storage,
            "multipliers": {
                "interpolation_current_to_next": interpolation_12_weight,
                "interpolation_next_to_current": interpolation_21_weight,
                "pl_at_next": pl_next_weight,
                "residual_lipschitz": residual_weight,
            },
            "physical_gains": {
                "gradient_noise": gradient_noise_gain,
                "implementation_error": implementation_error_gain,
            },
            "normalized_penalties": {
                "gradient_noise": normalized_gradient_noise_penalty,
                "implementation_error": normalized_implementation_error_penalty,
            },
        },
        "dynamics": {
            "transition_matrix": transition,
            "state_selection_matrix": selection,
            "noisy_gradient_selector": noisy_gradient,
            "signal_selector": signal,
            "step_selector": step,
        },
        "supplies": {
            "interpolation_current_to_next": interpolation_12,
            "interpolation_next_to_current": interpolation_21,
            "pl_at_next": pl_next,
            "residual_lipschitz": residual_supply,
            "gradient_noise_norm": gradient_noise_norm,
            "implementation_error_norm": implementation_error_norm,
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


def _fraction_vector(payload: list[str]) -> Vector:
    return tuple(Fraction(value) for value in payload)


def compare_with_canonical(reconstruction: dict[str, object], path: Path) -> dict[str, object]:
    """Compare every reconstructed exact field with a canonical P7 artifact."""

    canonical_bytes = path.read_bytes()
    canonical = json.loads(canonical_bytes)
    locked = canonical["locked_exact_certificate"]
    parameters = reconstruction["parameters"]
    dynamics = reconstruction["dynamics"]
    supplies = reconstruction["supplies"]
    multipliers = parameters["multipliers"]
    physical_gains = parameters["physical_gains"]
    normalized_penalties = parameters["normalized_penalties"]
    canonical_multipliers = locked["multipliers"]
    canonical_function_coefficients = locked["function_coefficients"]

    comparisons = {
        "schema_version": canonical["schema_version"]
        == "passive-muon-robust-dissipativity-certificate-v1",
        "pl_constant": parameters["pl_constant"] == _fraction(locked["pl_constant_ell_PL"]),
        "smoothness": parameters["smoothness"] == _fraction(locked["smoothness_L"]),
        "center_gain": parameters["center_gain"] == _fraction(locked["center_gain_gamma"]),
        "residual_lipschitz": parameters["residual_lipschitz"]
        == _fraction(locked["residual_lipschitz_K_E"]),
        "beta": parameters["beta"] == _fraction(locked["beta"]),
        "learning_rate": parameters["learning_rate"] == _fraction(locked["learning_rate_eta"]),
        "rate": parameters["rate"] == _fraction(locked["rate_q"]),
        "gradient_noise_gain": physical_gains["gradient_noise"]
        == _fraction(locked["physical_gains"]["gamma_g"]),
        "implementation_error_gain": physical_gains["implementation_error"]
        == _fraction(locked["physical_gains"]["gamma_R"]),
        "gradient_noise_penalty": normalized_penalties["gradient_noise"]
        == _fraction(locked["normalized_penalties"]["w_squared"]),
        "implementation_error_penalty": normalized_penalties["implementation_error"]
        == _fraction(locked["normalized_penalties"]["h_squared"]),
        "storage": parameters["storage"] == _fraction_matrix(locked["storage_P"]),
        "function_storage": parameters["function_storage"]
        == _fraction(locked["function_storage_c_F"]),
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
        "noisy_gradient_selector": dynamics["noisy_gradient_selector"]
        == _fraction_vector(locked["noisy_gradient_selector"]),
        "signal_selector": dynamics["signal_selector"]
        == _fraction_vector(locked["signal_selector"]),
        "step_selector": dynamics["step_selector"] == _fraction_vector(locked["step_selector"]),
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
        == _fraction_items(canonical_function_coefficients["actual"]),
        "expected_function_coefficients": reconstruction["expected_function_coefficients"]
        == _fraction_items(canonical_function_coefficients["expected"]),
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
                "all_exact_checks_passed",
            )
        )
        and canonical_function_coefficients["cancel_exactly"],
    }
    return {
        "status": "matched" if all(comparisons.values()) else "mismatch",
        "path": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
        "sha256": hashlib.sha256(canonical_bytes).hexdigest(),
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


def build_payload(
    canonical_path: Path = DEFAULT_CANONICAL,
    *,
    require_canonical: bool = False,
) -> dict[str, object]:
    # This order is authoritative: the canonical bytes cannot influence the
    # independent algebra or any internal exact check.
    reconstruction = reconstruct()
    if not reconstruction["all_internal_exact_checks_passed"]:
        raise AssertionError("independent P7 construction failed its internal exact checks")

    if canonical_path.exists():
        canonical_comparison = compare_with_canonical(reconstruction, canonical_path)
    elif require_canonical:
        raise FileNotFoundError(f"canonical P7 artifact not found: {canonical_path}")
    else:
        canonical_comparison = {
            "status": "not_found",
            "path": str(
                canonical_path.relative_to(ROOT)
                if canonical_path.is_relative_to(ROOT)
                else canonical_path
            ),
            "comparisons": {},
            "all_exact_fields_match": None,
        }

    canonical_ok = canonical_comparison["all_exact_fields_match"]
    return {
        "schema_version": "passive-muon-p7-independent-reconstruction-v1",
        "implementation_scope": {
            "arithmetic": "stdlib fractions.Fraction",
            "project_package_imported": False,
            "numerical_algebra_library_imported": False,
            "determinant_method": "local exact Bareiss elimination",
            "definiteness_method": "Sylvester leading-principal-minor criterion",
            "canonical_read_order": "only after complete independent construction and audit",
            "qualification": "independent code-path reconstruction; not a human proof audit",
        },
        "reconstruction": _fraction_strings(reconstruction),
        "canonical_comparison": canonical_comparison,
        "all_internal_exact_checks_passed": True,
        "all_exact_checks_passed": canonical_ok is not False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--require-canonical", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_payload(args.canonical, require_canonical=args.require_canonical)
    if not payload["all_exact_checks_passed"]:
        raise AssertionError("independent P7 reconstruction disagrees with the canonical artifact")
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
