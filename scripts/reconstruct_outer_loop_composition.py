#!/usr/bin/env python3
"""Independently reconstruct the exact P21 outer-loop certificate.

This replay imports neither :mod:`passive_muon` nor a numerical library.  It
rebuilds the complete stored-signal 7-by-7 LMIs, their exact Sylvester minors,
and all seven-shape finite-precision envelopes with only integer arithmetic,
``fractions.Fraction``, and ``math.isqrt``.
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from math import isqrt
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANONICAL = ROOT / "results/summaries/certified_outer_loop_composition_certificate.json"
SCHEMA_VERSION = "passive-muon-p21-independent-reconstruction-v1"
CANONICAL_SCHEMA_VERSION = "passive-muon-certified-outer-loop-composition-artifact-v1"

VARIABLE_ORDER = (
    "normalized_momentum",
    "normalized_gradient",
    "normalized_sector_residual",
    "normalized_next_gradient",
    "normalized_momentum_roundoff",
    "normalized_signal_roundoff",
    "normalized_output_equivalent_error",
)

BETA = Fraction(19, 20)
PL_CONSTANT = Fraction(1)
SMOOTHNESS = Fraction(10)
SECTOR_LOWER = Fraction(125, 1_024)
SECTOR_UPPER = Fraction(509, 512)
SECTOR_CENTER = Fraction(1_143, 2_048)
SECTOR_RADIUS = Fraction(893, 2_048)

U32 = Fraction(1, 2**24)
TAU32 = Fraction(1, 2**150)
BETA32 = Fraction(15_938_355, 16_777_216)
ONE_MINUS_BETA32 = Fraction(13_421_773, 268_435_456)
BETA32_BITS = 0x3F73_3333
ONE_MINUS_BETA32_BITS = 0x3D4C_CCCD
PRIMARY_ETA32 = Fraction(8_947_849, 1_073_741_824)
PRIMARY_ETA32_BITS = 0x3C08_8889
MAXIMUM_ETA32 = Fraction(1_617_081, 134_217_728)
MAXIMUM_ETA32_BITS = 0x3C45_65C8

MASTER_HIGH_MAX = Fraction(2**30)
MASTER_MIDDLE_MAX = Fraction(2**7)
MASTER_LOW_MAX = Fraction(1, 2**16)
OUTPUT_MAX = Fraction(64)
TOTAL_STEP_MAX = Fraction(1)
SQRT_GRID = 2**80
REPORT_GRID = 2**40
STORAGE_RADIUS = Fraction(1)
YOUNG_PARAMETERS = (Fraction(1), Fraction(1), Fraction(1))
ROBUST_GRADIENT_SLOPE = Fraction(1, 4_096)
ROBUST_GRADIENT_INTERCEPT = Fraction(1, 4_096)
EXTERNAL_GRADIENT_SLOPE = Fraction(1, 8_192)
EXTERNAL_GRADIENT_INTERCEPT = Fraction(1, 8_192)
MODEL_RECONSTRUCTION_SLOPE = Fraction(1, 81_920)
MODEL_RECONSTRUCTION_INTERCEPT = Fraction(1, 81_920)
BOUNDED_DECAY_DISPLACEMENT = Fraction(1, 131_072)
BOUNDED_DECAY_ROUNDED_STEP = Fraction(1, 131_072)

SHAPES = (
    (768, 768),
    (768, 3_072),
    (768, 50_257),
    (3_072, 12_288),
    (4_096, 4_096),
    (4_096, 11_008),
    (4_096, 14_336),
)

POINTS = (
    {
        "name": "faster_rate_eta_1_over_120",
        "eta": Fraction(1, 120),
        "eta32": PRIMARY_ETA32,
        "tau": Fraction(24_987, 25_000),
        "storage": (
            (Fraction(599, 1_000), Fraction(-231, 625)),
            (Fraction(-231, 625), Fraction(657, 2_500)),
        ),
        "function_storage": Fraction(1),
        "reverse_weight": Fraction(2_177, 200),
        "residual_multiplier": Fraction(53, 2_500),
        "port_gains": (Fraction(16_384), Fraction(1_024), Fraction(512)),
    },
    {
        "name": "maximum_step_eta_1_over_83",
        "eta": Fraction(1, 83),
        "eta32": MAXIMUM_ETA32,
        "tau": Fraction(999_799, 1_000_000),
        "storage": (
            (Fraction(97, 125), Fraction(-151, 500)),
            (Fraction(-151, 500), Fraction(17, 100)),
        ),
        "function_storage": Fraction(1),
        "reverse_weight": Fraction(3_459, 500),
        "residual_multiplier": Fraction(9, 250),
        "port_gains": (Fraction(32_768), Fraction(2_048), Fraction(512)),
    },
)

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)

Vector = tuple[Fraction, ...]
Matrix = tuple[tuple[Fraction, ...], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--require-canonical", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _transpose(matrix: Matrix) -> Matrix:
    return tuple(tuple(row[index] for row in matrix) for index in range(len(matrix[0])))


def _multiply(left: Matrix, right: Matrix) -> Matrix:
    if len(left[0]) != len(right):
        raise AssertionError("matrix dimensions do not align")
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


def _add_matrix(*matrices: Matrix) -> Matrix:
    rows, columns = len(matrices[0]), len(matrices[0][0])
    return tuple(
        tuple(
            sum((matrix[row][column] for matrix in matrices), Fraction(0))
            for column in range(columns)
        )
        for row in range(rows)
    )


def _scale_matrix(scale: Fraction, matrix: Matrix) -> Matrix:
    return tuple(tuple(scale * value for value in row) for row in matrix)


def _outer(left: Vector, right: Vector) -> Matrix:
    return tuple(tuple(x * y for y in right) for x in left)


def _symmetric_outer(left: Vector, right: Vector) -> Matrix:
    return _scale_matrix(Fraction(1, 2), _add_matrix(_outer(left, right), _outer(right, left)))


def _add_vector(*vectors: Vector) -> Vector:
    return tuple(
        sum((vector[index] for vector in vectors), Fraction(0)) for index in range(len(vectors[0]))
    )


def _scale_vector(scale: Fraction, vector: Vector) -> Vector:
    return tuple(scale * value for value in vector)


def _interpolation(
    input_i: Vector, gradient_i: Vector, input_j: Vector, gradient_j: Vector
) -> Matrix:
    dx = _add_vector(input_i, _scale_vector(Fraction(-1), input_j))
    du = _add_vector(gradient_i, _scale_vector(Fraction(-1), gradient_j))
    model = _scale_matrix(
        Fraction(1, 4),
        _add_matrix(
            _outer(du, du),
            _scale_matrix(Fraction(2), _symmetric_outer(dx, du)),
            _scale_matrix(Fraction(-1), _outer(dx, dx)),
        ),
    )
    return _add_matrix(
        _scale_matrix(Fraction(-1), _symmetric_outer(gradient_j, dx)),
        _scale_matrix(Fraction(-1), model),
    )


def _determinant(matrix: Matrix) -> Fraction:
    """Exact determinant by fraction-preserving Gaussian elimination."""

    work = [list(row) for row in matrix]
    size = len(work)
    sign = 1
    result = Fraction(1)
    for column in range(size):
        pivot = next((row for row in range(column, size) if work[row][column]), None)
        if pivot is None:
            return Fraction(0)
        if pivot != column:
            work[column], work[pivot] = work[pivot], work[column]
            sign *= -1
        pivot_value = work[column][column]
        result *= pivot_value
        for row in range(column + 1, size):
            ratio = work[row][column] / pivot_value
            for index in range(column + 1, size):
                work[row][index] -= ratio * work[column][index]
    return sign * result


def _leading_minors(matrix: Matrix) -> tuple[Fraction, ...]:
    return tuple(
        _determinant(tuple(tuple(row[:size]) for row in matrix[:size]))
        for size in range(1, len(matrix) + 1)
    )


def _raw_lmi(point: dict[str, Any]) -> Matrix:
    dimension = 7
    basis = tuple(
        tuple(Fraction(row == column) for row in range(dimension)) for column in range(dimension)
    )
    momentum, gradient, residual, gradient_next, momentum_error, signal_error, output_error = basis
    alpha = point["eta"] * SECTOR_CENTER * SMOOTHNESS
    residual_ratio = SECTOR_RADIUS / SECTOR_CENTER
    stored_signal = _add_vector(
        _scale_vector(BETA**2, momentum),
        _scale_vector(1 - BETA**2, gradient),
        _scale_vector(BETA, momentum_error),
        signal_error,
    )
    step = _add_vector(
        _scale_vector(-alpha, stored_signal),
        _scale_vector(-alpha * residual_ratio, residual),
        _scale_vector(-alpha, output_error),
    )
    momentum_next = _add_vector(
        _scale_vector(BETA, momentum),
        _scale_vector(1 - BETA, gradient),
        momentum_error,
    )
    transition: Matrix = (momentum_next, gradient_next)
    selection: Matrix = (momentum, gradient)
    null = (Fraction(0),) * dimension
    interpolation_12 = _interpolation(null, gradient, step, gradient_next)
    interpolation_21 = _interpolation(step, gradient_next, null, gradient)
    residual_norm = _add_matrix(
        _outer(stored_signal, stored_signal),
        _scale_matrix(Fraction(-1), _outer(residual, residual)),
    )
    rate = point["tau"] ** 2
    forward_weight = point["reverse_weight"] + rate
    reverse_weight = point["reverse_weight"]
    pl_next = (1 - rate) / (2 * (PL_CONSTANT / SMOOTHNESS))
    return _add_matrix(
        _multiply(_transpose(transition), _multiply(point["storage"], transition)),
        _scale_matrix(
            -rate,
            _multiply(_transpose(selection), _multiply(point["storage"], selection)),
        ),
        _scale_matrix(forward_weight, interpolation_12),
        _scale_matrix(reverse_weight, interpolation_21),
        _scale_matrix(pl_next, _outer(gradient_next, gradient_next)),
        _scale_matrix(point["residual_multiplier"], residual_norm),
    )


def _frozen_p18_lmi(point: dict[str, Any]) -> Matrix:
    """Independently rebuild the original four-variable P18 PL LMI."""

    dimension = 4
    basis = tuple(
        tuple(Fraction(row == column) for row in range(dimension)) for column in range(dimension)
    )
    momentum, gradient, residual, gradient_next = basis
    alpha = point["eta"] * SECTOR_CENTER * SMOOTHNESS
    stored_signal = _add_vector(
        _scale_vector(BETA**2, momentum),
        _scale_vector(1 - BETA**2, gradient),
    )
    step = _add_vector(
        _scale_vector(-alpha, stored_signal),
        _scale_vector(-alpha * SECTOR_RADIUS / SECTOR_CENTER, residual),
    )
    momentum_next = _add_vector(
        _scale_vector(BETA, momentum),
        _scale_vector(1 - BETA, gradient),
    )
    transition: Matrix = (momentum_next, gradient_next)
    selection: Matrix = (momentum, gradient)
    null = (Fraction(0),) * dimension
    interpolation_12 = _interpolation(null, gradient, step, gradient_next)
    interpolation_21 = _interpolation(step, gradient_next, null, gradient)
    residual_norm = _add_matrix(
        _outer(stored_signal, stored_signal),
        _scale_matrix(Fraction(-1), _outer(residual, residual)),
    )
    rate = point["tau"] ** 2
    return _add_matrix(
        _multiply(_transpose(transition), _multiply(point["storage"], transition)),
        _scale_matrix(
            -rate,
            _multiply(_transpose(selection), _multiply(point["storage"], selection)),
        ),
        _scale_matrix(point["reverse_weight"] + rate, interpolation_12),
        _scale_matrix(point["reverse_weight"], interpolation_21),
        _scale_matrix(
            (1 - rate) / (2 * (PL_CONSTANT / SMOOTHNESS)),
            _outer(gradient_next, gradient_next),
        ),
        _scale_matrix(point["residual_multiplier"], residual_norm),
    )


def _matrix_strings(matrix: Matrix) -> list[list[str]]:
    return [[str(value) for value in row] for row in matrix]


def _core_fields(point: dict[str, Any]) -> dict[str, object]:
    raw = _raw_lmi(point)
    gains = point["port_gains"]
    lmi = tuple(
        tuple(
            value - (gains[row - 4] if row == column and row >= 4 else 0)
            for column, value in enumerate(values)
        )
        for row, values in enumerate(raw)
    )
    zero = tuple(tuple(row[column] for column in range(4)) for row in raw[:4])
    frozen = _frozen_p18_lmi(point)
    negative = _scale_matrix(Fraction(-1), lmi)
    storage_minors = _leading_minors(point["storage"])
    lmi_minors = _leading_minors(negative)
    rate = point["tau"] ** 2
    forward_weight = point["reverse_weight"] + rate
    pl_next = (1 - rate) / (2 * (PL_CONSTANT / SMOOTHNESS))
    function_coefficients = (
        forward_weight - point["reverse_weight"],
        -forward_weight + point["reverse_weight"] - 2 * (PL_CONSTANT / SMOOTHNESS) * pl_next,
    )
    expected = (rate, Fraction(-1))
    checks = {
        "storage_positive_definite": all(value > 0 for value in storage_minors),
        "lmi_negative_definite": all(value > 0 for value in lmi_minors),
        "zero_port_recovers_p18_exactly": zero == frozen,
        "function_values_cancel": function_coefficients == expected,
    }
    return {
        "name": point["name"],
        "learning_rate": str(point["eta"]),
        "tau": str(point["tau"]),
        "rate": str(rate),
        "pl_constant": str(PL_CONSTANT),
        "smoothness": str(SMOOTHNESS),
        "beta": str(BETA),
        "sector_lower": str(SECTOR_LOWER),
        "sector_upper": str(SECTOR_UPPER),
        "center_gain": str(SECTOR_CENTER),
        "residual_lipschitz": str(SECTOR_RADIUS),
        "dimensionless_step": str(point["eta"] * SECTOR_CENTER * SMOOTHNESS),
        "residual_ratio": str(SECTOR_RADIUS / SECTOR_CENTER),
        "storage": _matrix_strings(point["storage"]),
        "function_storage": str(point["function_storage"]),
        "interpolation_reverse_weight": str(point["reverse_weight"]),
        "lambda_interpolation_12": str(forward_weight),
        "lambda_interpolation_21": str(point["reverse_weight"]),
        "lambda_pl_next": str(pl_next),
        "lambda_residual_norm": str(point["residual_multiplier"]),
        "port_gains": [str(value) for value in gains],
        "raw_lmi": _matrix_strings(raw),
        "penalized_lmi": _matrix_strings(lmi),
        "storage_leading_minors": [str(value) for value in storage_minors],
        "negative_lmi_leading_minors": [str(value) for value in lmi_minors],
        "zero_port_lmi": _matrix_strings(zero),
        "frozen_p18_lmi": _matrix_strings(frozen),
        "function_coefficients": [str(value) for value in function_coefficients],
        "expected_function_coefficients": [str(value) for value in expected],
        "checks": checks,
        "certified": all(checks.values()),
    }


def _ceil_sqrt(value: int) -> int:
    root = isqrt(value)
    return root if root * root == value else root + 1


def _sqrt_upper(value: Fraction) -> Fraction:
    if value == 0:
        return Fraction(0)
    scaled = value.numerator * SQRT_GRID**2
    numerator = isqrt(scaled // value.denominator)
    lower = Fraction(numerator, SQRT_GRID)
    return lower if lower * lower == value else Fraction(numerator + 1, SQRT_GRID)


def _round_up(value: Fraction) -> Fraction:
    numerator, remainder = divmod(value.numerator * REPORT_GRID, value.denominator)
    return Fraction(numerator + bool(remainder), REPORT_GRID)


def _dual_slope(a: Fraction, b: Fraction, storage: Matrix) -> Fraction:
    p00, p01 = storage[0]
    _, p11 = storage[1]
    determinant = p00 * p11 - p01**2
    return _sqrt_upper((p11 * a**2 - 2 * p01 * a * b + p00 * b**2) / determinant)


def _envelope(slope: Fraction, intercept: Fraction) -> dict[str, Fraction]:
    return {"slope": slope, "intercept": intercept}


def _envelope_strings(envelope: dict[str, Fraction]) -> dict[str, str]:
    return {key: str(value) for key, value in envelope.items()}


def _evaluation(
    point: dict[str, Any],
    shape: tuple[int, int],
    *,
    profile: str,
    gradient_name: str,
    gradient_slope: Fraction,
    gradient_intercept: Fraction,
    decay_name: str,
    decay_logical_displacement: Fraction,
    decay_rounded_step: Fraction,
) -> dict[str, object]:
    root = _ceil_sqrt(shape[0] * shape[1])
    eta = point["eta"]
    eta32 = point["eta32"]
    storage = point["storage"]
    one_minus_beta = 1 - BETA
    two_operation_factor = 2 * U32 + U32**2
    c_beta = abs(BETA32 - BETA) + two_operation_factor * BETA32
    c_gradient = abs(ONE_MINUS_BETA32 - one_minus_beta) + two_operation_factor * ONE_MINUS_BETA32
    b_ema = root * (3 + 2 * U32) * TAU32
    cast_crumb = root * TAU32
    represented_beta = (1 + U32) ** 2 * BETA32
    represented_gradient = (1 + U32) ** 2 * ONE_MINUS_BETA32

    momentum_state = _dual_slope(
        c_beta,
        c_gradient * (1 + U32) + one_minus_beta * U32,
        storage,
    )
    momentum_zeta = (c_gradient + one_minus_beta) * (1 + U32)
    momentum = _envelope(
        SMOOTHNESS * momentum_state + momentum_zeta * gradient_slope,
        b_ema + (c_gradient + one_minus_beta) * cast_crumb + momentum_zeta * gradient_intercept,
    )

    signal_state = _dual_slope(
        c_beta * represented_beta,
        c_beta * represented_gradient * (1 + U32) + c_gradient * (1 + U32) + one_minus_beta * U32,
        storage,
    )
    signal_zeta = (c_beta * represented_gradient + c_gradient + one_minus_beta) * (1 + U32)
    signal_roundoff = _envelope(
        SMOOTHNESS * signal_state + signal_zeta * gradient_slope,
        c_beta * (b_ema + represented_gradient * cast_crumb)
        + b_ema
        + (c_gradient + one_minus_beta) * cast_crumb
        + signal_zeta * gradient_intercept,
    )

    signal_momentum = represented_beta**2
    signal_gradient = represented_gradient * (1 + represented_beta)
    stored_state = _dual_slope(
        signal_momentum,
        signal_gradient * (1 + U32),
        storage,
    )
    stored_zeta = signal_gradient * (1 + U32)
    stored_signal = _envelope(
        SMOOTHNESS * stored_state + stored_zeta * gradient_slope,
        (1 + represented_beta) * b_ema
        + signal_gradient * cast_crumb
        + stored_zeta * gradient_intercept,
    )
    shield_output = _envelope(
        SECTOR_UPPER * stored_signal["slope"],
        SECTOR_UPPER * stored_signal["intercept"],
    )
    master_coefficient = abs(eta32 - eta) + two_operation_factor * eta32
    master = _envelope(
        master_coefficient * shield_output["slope"],
        master_coefficient * shield_output["intercept"]
        + U32 * root * MASTER_LOW_MAX
        + root * (2 + U32) * TAU32,
    )
    dead_zone = eta * root * Fraction(1, 2**127)
    alpha = eta * SECTOR_CENTER * SMOOTHNESS
    output_equivalent = _envelope(
        master["slope"] / alpha,
        (master["intercept"] + dead_zone + decay_logical_displacement) / alpha,
    )
    normalized = (
        _envelope(momentum["slope"] / SMOOTHNESS, momentum["intercept"] / SMOOTHNESS),
        _envelope(
            signal_roundoff["slope"] / SMOOTHNESS,
            signal_roundoff["intercept"] / SMOOTHNESS,
        ),
        output_equivalent,
    )
    base_rate = point["tau"] ** 2
    absorbed_rate = base_rate + sum(
        gain * 2 * envelope["slope"] ** 2
        for gain, envelope in zip(point["port_gains"], normalized, strict=True)
    )
    forcing = sum(
        gain * 2 * envelope["intercept"] ** 2
        for gain, envelope in zip(point["port_gains"], normalized, strict=True)
    )
    objective = SMOOTHNESS * forcing / (1 - absorbed_rate)
    reported_rate = _round_up(absorbed_rate)
    reported_forcing = _round_up(forcing)
    reported_objective = _round_up(SMOOTHNESS * reported_forcing / (1 - reported_rate))

    signal_at_one = stored_signal["slope"] + stored_signal["intercept"]
    output_at_one = shield_output["slope"] + shield_output["intercept"]
    rounded_operator_step = (1 + U32) * eta32 * output_at_one + root * TAU32
    total_step = rounded_operator_step + decay_rounded_step
    pending_operator = (1 + U32) * (MASTER_LOW_MAX + rounded_operator_step) + TAU32
    pending_decay = (
        pending_operator
        if decay_rounded_step == 0
        else (1 + U32) * (pending_operator + decay_rounded_step) + TAU32
    )
    middle_candidate = (1 + U32) * (MASTER_MIDDLE_MAX + pending_decay) + TAU32
    lower_checks = {
        "pending_after_operator_below_three": pending_operator < 3,
        "pending_after_decay_below_three": pending_decay < 3,
        "middle_candidate_below_2^8": middle_candidate < 2**8,
        "new_low_within_2^-16_guard": Fraction(1, 2**17) <= MASTER_LOW_MAX,
        "new_middle_within_2^7_guard": Fraction(2**6) <= MASTER_MIDDLE_MAX,
        "high_two_sum_cannot_overflow": MASTER_HIGH_MAX + middle_candidate < 2**31,
    }
    guard_checks = {
        "stored_signal_is_finite_under_unit_storage": signal_at_one < 2**120,
        "shield_output_fits_runtime_guard": output_at_one <= OUTPUT_MAX,
        "combined_step_fits_runtime_guard": total_step <= TOTAL_STEP_MAX,
        "p21_two_subtraction_lower_word_invariants_close": all(lower_checks.values()),
        "decay_budget_is_an_explicit_conditional_premise": bool(decay_name),
        "high_word_guard_remains_explicitly_conditional": True,
    }
    checks = {
        "core_stored_signal_lmi_is_strict": all(
            value > 0
            for value in _leading_minors(
                _scale_matrix(
                    Fraction(-1),
                    tuple(
                        tuple(
                            value
                            - (point["port_gains"][row - 4] if row == column and row >= 4 else 0)
                            for column, value in enumerate(values)
                        )
                        for row, values in enumerate(_raw_lmi(point))
                    ),
                )
            )
        ),
        "reported_rate_rounding_is_upward": reported_rate >= absorbed_rate,
        "reported_forcing_rounding_is_upward": reported_forcing >= forcing,
        "absorbed_rate_is_strictly_contractive": reported_rate < 1,
        "unit_storage_ball_is_forward_invariant": reported_forcing <= 1 - reported_rate,
        "objective_neighborhood_is_finite": reported_objective >= 0,
        "all_runtime_guards_close": all(guard_checks.values()),
        "shape_or_transpose_is_P20_certified": shape in SHAPES,
        "gradient_reconstruction_budget_is_explicit": bool(gradient_name),
    }
    return {
        "profile": profile,
        "point": point["name"],
        "shape": list(shape),
        "shield_shape": list(shape),
        "transposed_for_shield": False,
        "gradient_budget": {
            "name": gradient_name,
            "slope": str(gradient_slope),
            "intercept": str(gradient_intercept),
        },
        "decay_budget": {
            "name": decay_name,
            "logical_displacement": str(decay_logical_displacement),
            "rounded_step": str(decay_rounded_step),
        },
        "sqrt_entries_upper": root,
        "eta_fp32": str(eta32),
        "eta_representation_error": str(eta32 - eta),
        "physical_envelopes": {
            "momentum_roundoff": _envelope_strings(momentum),
            "signal_roundoff": _envelope_strings(signal_roundoff),
            "stored_signal": _envelope_strings(stored_signal),
            "shield_output": _envelope_strings(shield_output),
            "master_roundoff": _envelope_strings(master),
        },
        "dead_zone_parameter_error": str(dead_zone),
        "normalized_ports": {
            name: _envelope_strings(envelope)
            for name, envelope in zip(
                ("momentum_roundoff", "signal_roundoff", "output_equivalent_error"),
                normalized,
                strict=True,
            )
        },
        "absorption": {
            "base_rate": str(base_rate),
            "absorbed_rate": str(absorbed_rate),
            "constant_forcing": str(forcing),
            "storage_radius": str(STORAGE_RADIUS),
            "objective_gap_ultimate_bound": str(objective),
            "contractive": absorbed_rate < 1,
            "forward_invariant": absorbed_rate < 1 and forcing <= 1 - absorbed_rate,
        },
        "reported": {
            "rate_upper": str(reported_rate),
            "forcing_upper": str(reported_forcing),
            "objective_gap_upper": str(reported_objective),
        },
        "guard": {
            "stored_signal_at_one": str(signal_at_one),
            "shield_output_at_one": str(output_at_one),
            "rounded_operator_step_at_one": str(rounded_operator_step),
            "total_step_at_one": str(total_step),
            "output_max_abs": str(OUTPUT_MAX),
            "total_step_max_abs": str(TOTAL_STEP_MAX),
            "master_high_max_abs": str(MASTER_HIGH_MAX),
            "master_middle_max_abs": str(MASTER_MIDDLE_MAX),
            "master_low_max_abs": str(MASTER_LOW_MAX),
            "pending_after_operator_strict_upper": str(pending_operator),
            "pending_after_decay_strict_upper": str(pending_decay),
            "middle_candidate_strict_upper": str(middle_candidate),
            "high_word_guard_is_conditional": True,
            "lower_word_guard_checks": lower_checks,
            "checks": guard_checks,
        },
        "checks": checks,
        "certified": all(checks.values()),
    }


def _all_evaluations() -> list[dict[str, object]]:
    profiles = (
        (
            "zero_external_error",
            "zero",
            Fraction(0),
            Fraction(0),
            "disabled",
            Fraction(0),
            Fraction(0),
        ),
        (
            "joint_gradient_error",
            "joint_external_gradient_and_reconstruction_1_over_4096",
            ROBUST_GRADIENT_SLOPE,
            ROBUST_GRADIENT_INTERCEPT,
            "disabled",
            Fraction(0),
            Fraction(0),
        ),
        (
            "joint_gradient_plus_bounded_decay",
            "joint_external_gradient_and_reconstruction_1_over_4096",
            ROBUST_GRADIENT_SLOPE,
            ROBUST_GRADIENT_INTERCEPT,
            "conditional_1_over_131072",
            BOUNDED_DECAY_DISPLACEMENT,
            BOUNDED_DECAY_ROUNDED_STEP,
        ),
    )
    return [
        _evaluation(
            point,
            shape,
            profile=profile,
            gradient_name=name,
            gradient_slope=slope,
            gradient_intercept=intercept,
            decay_name=decay_name,
            decay_logical_displacement=decay_displacement,
            decay_rounded_step=decay_rounded_step,
        )
        for (
            profile,
            name,
            slope,
            intercept,
            decay_name,
            decay_displacement,
            decay_rounded_step,
        ) in profiles
        for point in POINTS
        for shape in SHAPES
    ]


def reconstruction_fields() -> dict[str, object]:
    primary = POINTS[0]
    storage = primary["storage"]
    determinant = storage[0][0] * storage[1][1] - storage[0][1] ** 2
    centered_slope_square = (
        (Fraction(1, 1_000_000) / SECTOR_CENTER) ** 2 * storage[0][0] / determinant
    )
    centered_rate = primary["tau"] ** 2 + primary["port_gains"][2] * centered_slope_square
    next_iterate = Fraction(11_999, 12_000)
    displacement = Fraction(-1, 12_000)
    next_gap = Fraction(1, 288_000_000)
    return {
        "variable_order": list(VARIABLE_ORDER),
        "stored_graph": {
            "normalized_momentum": "z=m/L",
            "normalized_gradient": "u=grad f(W)/L",
            "momentum": "z_next=beta*z+(1-beta)*u+a_m",
            "stored_signal": "p=beta^2*z+(1-beta^2)*u+beta*a_m+a_s",
            "sector_output": "U/L=gamma*p+K_T*v; ||v||_F<=||p||_F",
            "parameter_step": "Delta W=-eta*gamma*L*(p+(K_T/gamma)*v+h)",
            "no_incremental_comparison": True,
        },
        "operator_contract": {
            "objective_domain": (
                "differentiable globally 10-smooth objectives satisfying the global PL "
                "inequality with constant 1; nonconvex objectives and nonunique minimizers "
                "are included"
            ),
            "matrix_domain": (
                "the seven P20 representative Transformer shapes and their nonduplicate "
                "transposes; transposition into P20 orientation is a Frobenius isometry"
            ),
            "candidate_normalization": "M/(||M||_F+epsilon)",
            "epsilon": "1/10000000",
            "polynomial": "q(x)=a*x+b*x^3+c*x^5",
            "polynomial_coefficients": {
                "a": "6889/2000",
                "b": "-191/40",
                "c": "4063/2000",
            },
            "newton_schulz_stages": 5,
            "upstream_revision": "f98f1cacc0263b04290753e32be8d498c1efc806",
            "upstream_muon_sha256": (
                "2479665a90124f62e4df557816665851ca317e42fcfda2af1da02c1f44ab5f3d"
            ),
            "aspect_scaling_placement": "after BF16 candidate and before P20 shield",
            "shield_sector": {
                "lower": str(SECTOR_LOWER),
                "upper": str(SECTOR_UPPER),
                "center": str(SECTOR_CENTER),
                "radius": str(SECTOR_RADIUS),
            },
            "all_subnormal_wrapper": (
                "return zero; charge the distance to the conceptual stored-signal/2 point "
                "as an absolute parameter-update port"
            ),
        },
        "arithmetic_contract": {
            "runtime_schema": "passive-muon-certified-outer-loop-composition-v1",
            "certificate_schema": "passive-muon-certified-outer-loop-composition-certificate-v1",
            "roundoff_schema": "passive-muon-certified-outer-loop-roundoff-v1",
            "backend": "torch-cpu-eager proof-reference composition",
            "shield_backend": (
                "torch-cpu-eager-ieee-rne-balanced-fp32-screen-radial-clip-half-fallback-v1"
            ),
            "chain": (
                "stored FP32 gradient -> FP32 EMA/Nesterov with reused bg -> five-stage "
                "BF16 candidate -> aspect scale -> P20 FP32 shield -> compensated three-word "
                "FP32 master update"
            ),
            "rounding": "IEEE-754 round-to-nearest ties-to-even at every named operation",
            "gradual_underflow_required": True,
            "ftz_daz_allowed": False,
            "fma_allowed": False,
            "reassociation_allowed": False,
            "candidate_input_dtypes": ["binary16-bfloat", "binary32"],
            "shield_output_dtype": "binary32",
            "norm_reduction_block_entries": 2**20,
            "fp32_unit_roundoff": str(U32),
            "fp32_half_min_subnormal": str(TAU32),
            "beta": str(BETA),
            "beta_fp32": str(BETA32),
            "beta_fp32_bits": f"0x{BETA32_BITS:08x}",
            "one_minus_beta_fp32": str(ONE_MINUS_BETA32),
            "one_minus_beta_fp32_bits": f"0x{ONE_MINUS_BETA32_BITS:08x}",
            "primary_eta_fp32": str(PRIMARY_ETA32),
            "primary_eta_fp32_bits": f"0x{PRIMARY_ETA32_BITS:08x}",
            "maximum_eta_fp32": str(MAXIMUM_ETA32),
            "maximum_eta_fp32_bits": f"0x{MAXIMUM_ETA32_BITS:08x}",
            "operator_output_max_abs": str(OUTPUT_MAX),
            "total_step_max_abs": str(TOTAL_STEP_MAX),
            "master_high_max_abs": str(MASTER_HIGH_MAX),
            "master_middle_max_abs": str(MASTER_MIDDLE_MAX),
            "master_low_max_abs": str(MASTER_LOW_MAX),
        },
        "roundoff_constants": {
            "sqrt_enclosure_denominator": SQRT_GRID,
            "report_grid_denominator": REPORT_GRID,
            "storage_radius": str(STORAGE_RADIUS),
            "young_parameters": [str(value) for value in YOUNG_PARAMETERS],
            "robust_gradient_slope": str(ROBUST_GRADIENT_SLOPE),
            "robust_gradient_intercept": str(ROBUST_GRADIENT_INTERCEPT),
            "external_gradient_slope": str(EXTERNAL_GRADIENT_SLOPE),
            "external_gradient_intercept": str(EXTERNAL_GRADIENT_INTERCEPT),
            "model_reconstruction_slope": str(MODEL_RECONSTRUCTION_SLOPE),
            "model_reconstruction_intercept": str(MODEL_RECONSTRUCTION_INTERCEPT),
            "gradient_source_composition": (
                "zeta=external_gradient_error+gradient_reconstruction_error; "
                "global L=10 maps ||H-W||<=a_W*sqrt(V)+b_W into "
                "10*a_W*sqrt(V)+10*b_W"
            ),
            "bounded_decay_displacement": str(BOUNDED_DECAY_DISPLACEMENT),
            "bounded_decay_rounded_step": str(BOUNDED_DECAY_ROUNDED_STEP),
            "output_guard": str(OUTPUT_MAX),
            "total_step_guard": str(TOTAL_STEP_MAX),
        },
        "core_certificates": [_core_fields(point) for point in POINTS],
        "shape_profiles": _all_evaluations(),
        "weight_decay_scope": {
            "main_pl_theorem_weight_decay": "zero",
            "bounded_port": {
                "name": "conditional_1_over_131072",
                "logical_displacement": str(BOUNDED_DECAY_DISPLACEMENT),
                "rounded_step": str(BOUNDED_DECAY_ROUNDED_STEP),
                "status": "conditional runtime premise; not inferred from wd or master range",
            },
            "centered_strong_convexity_corollary": {
                "weight_decay": "1/1000000",
                "strong_convexity": "1",
                "normalized_output_squared_slope": str(centered_slope_square),
                "certified_rate": str(centered_rate),
                "contractive": centered_rate < 1,
            },
            "ordinary_decay_nonzero_minimizer_control": {
                "learning_rate": "1/120",
                "weight_decay": "1/100",
                "minimizer": "1",
                "next_iterate": str(next_iterate),
                "displacement": str(displacement),
                "next_objective_gap": str(next_gap),
                "original_minimizer_is_not_an_equilibrium": displacement != 0 and next_gap > 0,
            },
        },
    }


def _internal_checks(fields: dict[str, object]) -> dict[str, bool]:
    cores = fields["core_certificates"]
    profiles = fields["shape_profiles"]
    return {
        "two_complete_core_certificates": isinstance(cores, list)
        and len(cores) == 2
        and all(
            len(core["raw_lmi"]) == 7
            and all(len(row) == 7 for row in core["raw_lmi"])
            and len(core["negative_lmi_leading_minors"]) == 7
            for core in cores
        ),
        "both_exact_lmis_are_strict": isinstance(cores, list)
        and all(core["certified"] for core in cores),
        "zero_ports_recover_p18": isinstance(cores, list)
        and all(core["zero_port_lmi"] == core["frozen_p18_lmi"] for core in cores),
        "all_profiles_present": isinstance(profiles, list)
        and len(profiles) == 3 * 2 * 7
        and {profile["profile"] for profile in profiles}
        == {
            "zero_external_error",
            "joint_gradient_error",
            "joint_gradient_plus_bounded_decay",
        },
        "all_shape_profiles_certified": isinstance(profiles, list)
        and all(profile["certified"] for profile in profiles),
        "both_large_shapes_present": isinstance(profiles, list)
        and [4_096, 11_008] in [profile["shape"] for profile in profiles]
        and [4_096, 14_336] in [profile["shape"] for profile in profiles],
        "both_reported_rates_are_strict": isinstance(profiles, list)
        and all(Fraction(profile["reported"]["rate_upper"]) < 1 for profile in profiles),
        "bounded_decay_is_conditional_port": (
            fields["weight_decay_scope"]["main_pl_theorem_weight_decay"] == "zero"
        ),
    }


def _canonical_comparison(
    canonical: Path, fields: dict[str, object], *, required: bool
) -> dict[str, object]:
    if not canonical.is_file():
        if required:
            raise FileNotFoundError(f"canonical P21 artifact not found: {canonical}")
        return {"status": "not_found", "path": str(canonical), "comparisons": {}}
    payload = json.loads(canonical.read_text(encoding="utf-8"))
    comparisons = {
        "schema_version": payload.get("schema_version") == CANONICAL_SCHEMA_VERSION,
        "reconstruction_fields": payload.get("reconstruction_fields") == fields,
        "all_exact_checks_passed": payload.get("all_exact_checks_passed") is True,
    }
    if not all(comparisons.values()):
        raise AssertionError("canonical P21 artifact has missing or mismatched exact fields")
    return {"status": "matched", "path": str(canonical), "comparisons": comparisons}


def main() -> None:
    args = parse_args()
    fields = reconstruction_fields()
    checks = _internal_checks(fields)
    comparison = _canonical_comparison(args.canonical, fields, required=args.require_canonical)
    result = {
        "schema_version": SCHEMA_VERSION,
        "implementation_scope": {
            "project_package_imported": False,
            "numerical_library_imported": False,
            "arithmetic": "fractions.Fraction, integer isqrt, and exact elimination",
        },
        "reconstruction": {"fields": fields, "checks": checks},
        "canonical_comparison": comparison,
        "all_internal_exact_checks_passed": all(checks.values()),
        "all_exact_checks_passed": all(checks.values())
        and comparison["status"] in {"not_found", "matched"},
    }
    if not result["all_exact_checks_passed"]:
        raise SystemExit("independent P21 reconstruction failed")
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
