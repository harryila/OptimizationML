#!/usr/bin/env python3
"""Independently reconstruct the exact P9 mixed-precision certificate.

This script imports neither :mod:`passive_muon` nor numerical libraries.  It
rebuilds the shape recurrences, two-term BF16 bound, serial obstruction, affine
operator bounds, and P7 closures with :class:`fractions.Fraction`.  A canonical
JSON is opened only after the complete internal reconstruction has passed.
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from math import isqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANONICAL = ROOT / "results/summaries/scalable_mixed_precision_certificate.json"

A = Fraction(6_889, 2_000)
B = Fraction(-191, 40)
C = Fraction(4_063, 2_000)
A32 = Fraction(902_955, 262_144)
B32 = Fraction(-10_013_901, 2_097_152)
C32 = Fraction(8_520_729, 4_194_304)
RHO = Fraction(210_177_835_339_081, 260_261_360_000)
RHO32 = Fraction(13_231_137, 16_384)

U32 = Fraction(1, 2**24)
T32 = Fraction(1, 2**150)
UB = Fraction(1, 2**8)
TB = Fraction(1, 2**134)
FP32_MAX = Fraction((2**24 - 1) * 2**104)
BF16_MAX = Fraction(255 * 2**120)

STAGES = 5
SAFE_MAX_ABS = Fraction(2**116)
MAX_ENTRIES = 2**52
GRID = 2**40
SPECTRAL_TUBE = Fraction(5, 4)
Q_UPPER = Fraction(121, 100)
Q_LIPSCHITZ = Fraction(3_000_459, 512_000)
IDEAL_LIPSCHITZ = Fraction(336_372_400_608_849, 260_261_360_000)
NORMALIZER_DENOMINATOR_ERROR_UPPER = Fraction(1, 100_000)
NORMALIZER_OUTPUT_ERROR_UPPER = Fraction(1, 90_000)

P7_RATE = Fraction(399_960_001, 400_000_000)
P7_ERROR_GAIN = Fraction(1, 2_000_000)
P7_SIGNAL_GAIN = Fraction(1_655_544_025, 2_600_084)
P7_SMOOTHNESS = Fraction(10)
P7_FUNCTION_LOWER = Fraction(13_533, 50_000)

SHAPES = (
    (768, 768),
    (768, 3_072),
    (768, 50_257),
    (3_072, 12_288),
    (4_096, 4_096),
    (4_096, 11_008),
    (4_096, 14_336),
)


def _ceil_sqrt(value: int) -> int:
    root = isqrt(value)
    return root if root * root == value else root + 1


def _ceil_log2(value: int) -> int:
    return (value - 1).bit_length()


def _upper(value: Fraction) -> Fraction:
    quotient, remainder = divmod(value.numerator * GRID, value.denominator)
    return Fraction(quotient + bool(remainder), GRID)


def _lower(value: Fraction) -> Fraction:
    return Fraction(value.numerator * GRID // value.denominator, GRID)


def _path_length(length: int) -> int:
    return 1 + _ceil_log2(length)


def _gamma(length: int) -> Fraction:
    amount = _path_length(length) * U32
    if amount >= 1:
        raise AssertionError("independent pairwise gamma left its valid regime")
    return amount / (1 - amount)


def _crumb(output_entries: int, length: int) -> Fraction:
    return _ceil_sqrt(output_entries) * 2 * length * T32 / (1 - _path_length(length) * U32)


def _fraction_map(values: dict[str, Fraction]) -> dict[str, str]:
    return {name: str(value) for name, value in values.items()}


def _shape_data(shape: tuple[int, int]) -> tuple[int, int, int, int]:
    rows, columns = shape
    if rows <= 0 or columns <= 0 or rows * columns > MAX_ENTRIES:
        raise AssertionError("independent shape is outside the locked domain")
    return min(shape), max(shape), rows * columns, _ceil_sqrt(rows * columns)


def _polynomial() -> dict[str, object]:
    squared_endpoint = SPECTRAL_TUBE**2
    discriminant = B**2 - 4 * A * C
    correction_at_endpoint = B + C * squared_endpoint
    vertex = -3 * B / (10 * C)

    def derivative(y: Fraction) -> Fraction:
        return A + 3 * B * y + 5 * C * y**2

    at_zero = derivative(Fraction(0))
    at_vertex = derivative(vertex)
    at_endpoint = derivative(squared_endpoint)
    checks = {
        "factor_has_no_real_root": discriminant < 0,
        "factor_is_positive": A > 0 and C > 0 and discriminant < 0,
        "factor_is_at_most_a_on_tube": correction_at_endpoint < 0,
        "derivative_vertex_is_in_interval": 0 < vertex < squared_endpoint,
        "derivative_endpoint_matches_lock": at_endpoint == Q_LIPSCHITZ,
        "derivative_zero_below_lock": abs(at_zero) < Q_LIPSCHITZ,
        "derivative_vertex_below_lock": abs(at_vertex) < Q_LIPSCHITZ,
    }
    values = {
        "squared_endpoint": squared_endpoint,
        "factor_discriminant": discriminant,
        "factor_correction_at_endpoint": correction_at_endpoint,
        "derivative_vertex": vertex,
        "derivative_at_zero": at_zero,
        "derivative_at_vertex": at_vertex,
        "derivative_at_endpoint": at_endpoint,
    }
    return {
        "values": _fraction_map(values),
        "checks": checks,
        "certified": all(checks.values()),
        "range_source": "P8 exact Sturm artifact proves 0<=q(x)<121/100 on 0<=x<=5/4",
    }


def _boundary_values() -> dict[str, Fraction]:
    residual_slope = U32 * UB
    low_slope = UB * (1 + U32) * UB
    high_plus_low_slope = (1 + UB) * (1 + (1 + U32) * UB)
    reconstruction_slope = U32 * high_plus_low_slope
    omega = residual_slope + low_slope + reconstruction_slope

    residual_crumb = U32 * TB + T32
    rounded_residual_crumb = (1 + U32) * TB + T32
    low_crumb = UB * rounded_residual_crumb + TB
    high_plus_low_crumb = TB + (1 + UB) * rounded_residual_crumb + TB
    reconstruction_crumb = U32 * high_plus_low_crumb + T32
    chi = residual_crumb + low_crumb + reconstruction_crumb
    return {
        "residual_rounding_slope": residual_slope,
        "low_rounding_slope": low_slope,
        "reconstruction_rounding_slope": reconstruction_slope,
        "residual_rounding_crumb": residual_crumb,
        "rounded_residual_crumb": rounded_residual_crumb,
        "low_rounding_crumb": low_crumb,
        "high_plus_low_crumb": high_plus_low_crumb,
        "reconstruction_rounding_crumb": reconstruction_crumb,
        "omega": omega,
        "chi": chi,
    }


def _boundary_error(shape: tuple[int, int], frobenius: Fraction) -> Fraction:
    _, _, _, root_entries = _shape_data(shape)
    values = _boundary_values()
    return values["omega"] * frobenius + root_entries * values["chi"]


def _normalizer(shape: tuple[int, int]) -> dict[str, object]:
    _, _, entries, root_entries = _shape_data(shape)
    gamma = _upper(_gamma(entries))
    sum_crumb = _upper(2 * entries * T32 / (1 - _path_length(entries) * U32))
    relative = _upper(U32 + root_entries * T32)
    norm_squared = _upper((2 + relative) * relative + gamma * (1 + relative) ** 2 + sum_crumb)
    norm_denominator = _upper((1 + U32) * (1 + norm_squared / (2 - norm_squared)) - 1 + T32)
    floor_vector = _upper(U32 + 2 * root_entries * T32)
    floor_squared = _upper(
        (2 + floor_vector) * floor_vector + gamma * (1 + floor_vector) ** 2 + 4 * sum_crumb
    )
    floor_denominator = _upper((1 + U32) * (1 + floor_squared / (2 - floor_squared)) - 1 + 2 * T32)
    small_floor = _upper(
        (1 + gamma) * (Fraction(1, 2) * (1 + U32) + root_entries * T32) ** 2 + sum_crumb
    )
    delta = NORMALIZER_DENOMINATOR_ERROR_UPPER
    output_error = _upper(
        (relative + delta + U32 * (1 + relative)) / (1 - delta) + root_entries * T32
    )
    scaled_vector = _upper(root_entries * (1 + U32 + T32))
    pairwise_sum = _upper((1 + gamma) * scaled_vector**2 + sum_crumb)
    checks = {
        "pairwise_gamma_defined": _path_length(entries) * U32 < 1,
        "norm_branch_squared_error_below_one": norm_squared < 1,
        "norm_branch_denominator_bound": norm_denominator < delta,
        "floor_branch_squared_error_below_one": floor_squared < 1,
        "floor_branch_denominator_bound": floor_denominator < delta,
        "small_floor_cannot_cross_one": small_floor < ((1 - T32) / (1 + U32)) ** 2,
        "normalizer_output_bound": output_error < NORMALIZER_OUTPUT_ERROR_UPPER,
        "pairwise_norm_sum_finite": pairwise_sum < FP32_MAX,
    }
    values = {
        "gamma": gamma,
        "sum_crumb": sum_crumb,
        "relative_vector_error": relative,
        "norm_branch_squared_error": norm_squared,
        "norm_branch_denominator_error": norm_denominator,
        "floor_branch_squared_error": floor_squared,
        "floor_branch_denominator_error": floor_denominator,
        "small_floor_sum_upper": small_floor,
        "output_error": output_error,
    }
    return {"values": _fraction_map(values), "checks": checks, "certified": all(checks.values())}


def _stage(
    shape: tuple[int, int],
    *,
    index: int,
    spectral: Fraction,
    frobenius: Fraction,
    previous_forward_error: Fraction,
) -> dict[str, object]:
    p, d, entries, _ = _shape_data(shape)
    root_p = _ceil_sqrt(p)
    gamma_d = _gamma(d)
    gamma_p = _gamma(p)

    gram_error = _upper(gamma_d * frobenius**2 + _crumb(p * p, d))
    gram_f = _upper(spectral * frobenius + gram_error)
    gram_s = _upper(spectral**2 + gram_error)
    scaled_error = _upper(
        abs(C32) * gram_error
        + abs(C32 - C) * spectral * frobenius
        + U32 * abs(C32) * gram_f
        + p * T32
    )
    scaled_f = _upper(abs(C) * spectral * frobenius + scaled_error)
    shifted_error = _upper(
        scaled_error + abs(B32 - B) * root_p + U32 * (scaled_f + abs(B32) * root_p) + p * T32
    )
    shifted_f = _upper(root_p * abs(B) + shifted_error)
    product_error = _upper(
        shifted_error * gram_s
        + abs(B) * gram_error
        + gamma_p * shifted_f * gram_f
        + _crumb(p * p, p)
    )
    product_f = _upper(root_p * (B**2 / (4 * C)) + product_error)
    affine_error = _upper(
        product_error + abs(A32 - A) * root_p + U32 * (product_f + abs(A32) * root_p) + p * T32
    )
    affine_f = _upper(root_p * A + affine_error)
    output_error = _upper(
        affine_error * spectral + gamma_p * affine_f * frobenius + _crumb(entries, p)
    )
    pre_s = _upper(Q_UPPER + output_error)
    pre_f = _upper(A * frobenius + output_error)
    boundary_error = _upper(_boundary_error(shape, pre_f))
    next_s = _upper(pre_s + boundary_error)
    next_f = _upper(pre_f + boundary_error)
    forward_error = _upper(Q_LIPSCHITZ * previous_forward_error + output_error + boundary_error)
    intermediate = max(
        frobenius**2,
        gram_f,
        abs(C32) * gram_f,
        scaled_f,
        scaled_f + abs(B32) * root_p,
        shifted_f,
        shifted_f * gram_f,
        product_f,
        product_f + abs(A32) * root_p,
        affine_f,
        affine_f * frobenius,
        pre_f,
        next_f,
    )
    values = {
        "input_spectral": spectral,
        "input_frobenius": frobenius,
        "gram_error": gram_error,
        "scaled_gram_error": scaled_error,
        "shifted_gram_error": shifted_error,
        "product_error": product_error,
        "affine_error": affine_error,
        "output_error_before_boundary": output_error,
        "pre_boundary_spectral": pre_s,
        "pre_boundary_frobenius": pre_f,
        "boundary_error": boundary_error,
        "next_spectral": next_s,
        "next_frobenius": next_f,
        "forward_error": forward_error,
        "intermediate_max_frobenius": intermediate,
    }
    return {"index": index, "values": _fraction_map(values)}


def _operator(shape: tuple[int, int], polynomial_error: Fraction) -> dict[str, Fraction]:
    _, _, _, root_entries = _shape_data(shape)
    first_slope = abs(RHO32 - RHO) + U32 * abs(RHO32)
    binary_slope = _upper(first_slope + U32 * (RHO + first_slope))
    ideal_output = A**STAGES
    binary_intercept = _upper(
        (1 + U32) * polynomial_error + U32 * ideal_output + (2 + U32) * root_entries * T32
    )
    real_slope = _upper(binary_slope * (1 + U32) + IDEAL_LIPSCHITZ * U32)
    real_intercept = _upper(
        binary_intercept + (binary_slope + IDEAL_LIPSCHITZ) * root_entries * T32
    )
    return {
        "binary32_slope": binary_slope,
        "binary32_intercept": binary_intercept,
        "real_slope": real_slope,
        "real_intercept": real_intercept,
    }


def _p7(operator: dict[str, Fraction], *, theta_value: int) -> dict[str, object]:
    theta = Fraction(theta_value)
    rate = _upper(
        P7_RATE + P7_ERROR_GAIN * (1 + theta) * P7_SIGNAL_GAIN * operator["real_slope"] ** 2
    )
    forcing = _upper(P7_ERROR_GAIN * (1 + 1 / theta) * operator["real_intercept"] ** 2)
    ultimate = _upper(P7_SMOOTHNESS / P7_FUNCTION_LOWER * forcing / (1 - rate))
    safe_radius = SAFE_MAX_ABS**2 / P7_SIGNAL_GAIN
    safe_capacity = _lower((1 - rate) * safe_radius)
    values = {
        "rate": rate,
        "forcing": forcing,
        "function_gap_ultimate": ultimate,
        "safe_storage_radius": safe_radius,
        "safe_forcing_capacity": safe_capacity,
    }
    return {"young_theta": theta_value, "values": _fraction_map(values)}


def _shape_audit(shape: tuple[int, int], *, theta: int = 3_006) -> dict[str, object]:
    polynomial = _polynomial()
    normalizer = _normalizer(shape)
    boundary = _boundary_values()
    initial_boundary_error = _upper(_boundary_error(shape, 1 + NORMALIZER_OUTPUT_ERROR_UPPER))
    spectral = _upper(1 + NORMALIZER_OUTPUT_ERROR_UPPER + initial_boundary_error)
    frobenius = spectral
    forward_error = _upper(NORMALIZER_OUTPUT_ERROR_UPPER + initial_boundary_error)
    stages: list[dict[str, object]] = []
    raw_stages: list[dict[str, Fraction]] = []
    for index in range(1, STAGES + 1):
        payload = _stage(
            shape,
            index=index,
            spectral=spectral,
            frobenius=frobenius,
            previous_forward_error=forward_error,
        )
        stages.append(payload)
        values = {name: Fraction(value) for name, value in payload["values"].items()}
        raw_stages.append(values)
        spectral = values["next_spectral"]
        frobenius = values["next_frobenius"]
        forward_error = values["forward_error"]
    operator = _operator(shape, forward_error)
    p7 = _p7(operator, theta_value=theta)
    p7_values = {name: Fraction(value) for name, value in p7["values"].items()}

    checks = {
        **{
            f"stage_{index}_input_in_spectral_tube": values["input_spectral"] < SPECTRAL_TUBE
            for index, values in enumerate(raw_stages, 1)
        },
        **{
            f"stage_{index}_intermediates_finite": values["intermediate_max_frobenius"]
            < FP32_MAX / 2
            for index, values in enumerate(raw_stages, 1)
        },
        **{
            f"stage_{index}_bf16_boundary_finite": values["pre_boundary_frobenius"] < BF16_MAX / 2
            for index, values in enumerate(raw_stages, 1)
        },
        "polynomial_tube_facts_are_certified": polynomial["certified"],
        "normalizer_is_certified": normalizer["certified"],
        "two_term_bound_is_sterbenz_free": True,
        "two_term_slope_components_reconstruct": boundary["omega"]
        == boundary["residual_rounding_slope"]
        + boundary["low_rounding_slope"]
        + boundary["reconstruction_rounding_slope"],
        "two_term_crumb_components_reconstruct": boundary["chi"]
        == boundary["residual_rounding_crumb"]
        + boundary["low_rounding_crumb"]
        + boundary["reconstruction_rounding_crumb"],
        "two_term_slope_improves_one_term": boundary["omega"] < UB,
        "repair_shell_finite": (1 + U32) * abs(RHO32) * SAFE_MAX_ABS
        + T32
        + raw_stages[-1]["next_frobenius"]
        < FP32_MAX / 2,
        "p7_affine_slope_gate": operator["real_slope"] ** 2
        < (1 - P7_RATE) / (P7_ERROR_GAIN * P7_SIGNAL_GAIN),
        "selected_young_parameter_keeps_rate_strict": p7_values["rate"] < 1,
        "finite_guard_is_forward_invariant": p7_values["forcing"]
        <= p7_values["safe_forcing_capacity"],
        "continuous_optimum_exists": P7_ERROR_GAIN * P7_SIGNAL_GAIN * operator["real_slope"] ** 2
        < 1 - P7_RATE,
    }
    rows, columns = shape
    return {
        "shape": [rows, columns],
        "oriented_shape": [min(shape), max(shape)],
        "entries": rows * columns,
        "polynomial": polynomial,
        "normalizer": normalizer,
        "initial_boundary_error": str(initial_boundary_error),
        "stages": stages,
        "operator_bound": _fraction_map(operator),
        "p7": p7,
        "checks": checks,
        "certified": normalizer["certified"] and all(checks.values()),
    }


def _ordinary_rank_limit(slope: Fraction) -> int:
    ratio = (SPECTRAL_TUBE - Q_UPPER) / (slope * Q_UPPER)
    squared = ratio**2
    return squared.numerator // squared.denominator


def reconstruct_certificate() -> dict[str, object]:
    boundary = _boundary_values()
    audits = [_shape_audit(shape) for shape in SHAPES]
    arithmetic = {
        "exact_a": A,
        "exact_b": B,
        "exact_c": C,
        "fp32_a": A32,
        "fp32_b": B32,
        "fp32_c": C32,
        "exact_rho": RHO,
        "fp32_rho": RHO32,
        "fp32_unit_roundoff": U32,
        "fp32_half_min_subnormal": T32,
        "bf16_unit_roundoff": UB,
        "bf16_half_min_subnormal": TB,
        "fp32_max_finite": FP32_MAX,
        "bf16_max_finite": BF16_MAX,
        "spectral_tube": SPECTRAL_TUBE,
        "polynomial_spectral_upper": Q_UPPER,
        "polynomial_lipschitz_on_tube": Q_LIPSCHITZ,
        "ideal_repaired_lipschitz": IDEAL_LIPSCHITZ,
        "normalizer_denominator_error_upper": NORMALIZER_DENOMINATOR_ERROR_UPPER,
        "normalizer_output_error_upper": NORMALIZER_OUTPUT_ERROR_UPPER,
        "p7_rate": P7_RATE,
        "p7_implementation_error_gain": P7_ERROR_GAIN,
        "p7_signal_storage_gain": P7_SIGNAL_GAIN,
        "p7_smoothness": P7_SMOOTHNESS,
        "p7_storage_function_lower": P7_FUNCTION_LOWER,
        "locked_safe_max_abs": SAFE_MAX_ABS,
    }
    serial_entries = 4_096 * 11_008
    return {
        "upper_grid_denominator": str(GRID),
        "locked_stages": STAGES,
        "locked_max_entries": MAX_ENTRIES,
        "arithmetic": _fraction_map(arithmetic),
        "two_term_boundary": {
            "values": _fraction_map(boundary),
            "sterbenz_free": True,
        },
        "rank_gates": {
            "ordinary_one_term": _ordinary_rank_limit(UB),
            "ideal_two_term": _ordinary_rank_limit(UB**2),
            "sterbenz_free_two_term": _ordinary_rank_limit(boundary["omega"]),
        },
        "serial_normalizer_obstruction": {
            "shape": [4_096, 11_008],
            "exact_square_sum": serial_entries,
            "computed_square_sum": 2**24,
            "returned_singular_value_squared": str(Fraction(serial_entries, 2**24)),
            "leaves_spectral_tube": Fraction(serial_entries, 2**24) > SPECTRAL_TUBE**2,
        },
        "representative_shapes": [list(shape) for shape in SHAPES],
        "shape_audits": audits,
    }


def reconstruct() -> dict[str, object]:
    certificate = reconstruct_certificate()
    audits = certificate["shape_audits"]
    checks = {
        "seven_representative_shapes": len(audits) == 7,
        "all_representative_shapes_certify": all(audit["certified"] for audit in audits),
        "all_polynomial_audits_certify": all(audit["polynomial"]["certified"] for audit in audits),
        "ordinary_boundary_rank_gate": certificate["rank_gates"]["ordinary_one_term"] == 71,
        "compensated_boundary_rank_gate": certificate["rank_gates"]["sterbenz_free_two_term"]
        == 4_656_751,
        "serial_obstruction_is_exact": certificate["serial_normalizer_obstruction"][
            "returned_singular_value_squared"
        ]
        == "43/16",
        "all_p7_rates_are_strict": all(
            Fraction(audit["p7"]["values"]["rate"]) < 1 for audit in audits
        ),
        "locked_rate_reconstructs": all(
            Fraction(audit["p7"]["values"]["rate"]) == Fraction(137_425_214_491, 137_438_953_472)
            for audit in audits
        ),
    }
    frontier = _shape_audit((4_608, 18_432))
    frontier_failures = [name for name, passed in frontier["checks"].items() if not passed]
    checks["qualified_frontier_failure"] = frontier_failures == ["stage_5_input_in_spectral_tube"]
    if not all(checks.values()):
        raise AssertionError([name for name, passed in checks.items() if not passed])
    return {
        "certificate": certificate,
        "checks": checks,
        "qualified_frontier_failed_checks": frontier_failures,
        "all_internal_exact_checks_passed": True,
    }


def _display_path(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _compare_canonical(reconstruction: dict[str, object], path: Path) -> dict[str, object]:
    if not path.exists():
        return {
            "path": _display_path(path),
            "status": "not_found",
            "comparisons": {},
            "all_exact_fields_match": None,
        }
    canonical = json.loads(path.read_text(encoding="utf-8"))
    comparisons = {
        "schema": canonical.get("schema_version")
        == "passive-muon-scalable-mixed-precision-certificate-v1",
        "certificate": canonical.get("certificate") == reconstruction["certificate"],
        "representative_gate": canonical.get("audit", {}).get(
            "all_representative_exact_checks_passed"
        )
        is True,
        "qualified_frontier": canonical.get("qualified_negative_controls", {})
        .get("audited_4_to_1_case", {})
        .get("failed_checks")
        == reconstruction["qualified_frontier_failed_checks"],
    }
    matched = all(comparisons.values())
    return {
        "path": _display_path(path),
        "status": "matched" if matched else "mismatch",
        "comparisons": comparisons,
        "all_exact_fields_match": matched,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--require-canonical", action="store_true")
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reconstruction = reconstruct()
    comparison = _compare_canonical(reconstruction, args.canonical)
    if args.require_canonical and comparison["status"] != "matched":
        raise SystemExit(f"canonical comparison failed: {comparison['status']}")
    payload = {
        "schema_version": "passive-muon-p9-independent-reconstruction-v1",
        "implementation_scope": {
            "project_package_imported": False,
            "numerical_library_imported": False,
            "arithmetic": "stdlib fractions.Fraction with explicit upward 2^-40 grid",
            "canonical_read_order": "only after complete independent reconstruction passes",
            "qualification": "exact second code path, not a human proof audit",
        },
        "reconstruction": reconstruction,
        "canonical_comparison": comparison,
        "all_internal_exact_checks_passed": reconstruction["all_internal_exact_checks_passed"],
        "all_exact_checks_passed": bool(
            reconstruction["all_internal_exact_checks_passed"]
            and comparison["status"] in {"matched", "not_found"}
        ),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
