#!/usr/bin/env python3
"""Independently reconstruct the exact P20 scalable-shield certificate.

This script imports neither :mod:`passive_muon` nor numerical libraries.  It
rebuilds every shape-dependent norm envelope and inward margin with only the
Python standard library, then optionally compares the exact reconstruction
fields with a generated canonical artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from math import isqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANONICAL = ROOT / "results/summaries/scalable_sector_shield_certificate.json"
SCHEMA_VERSION = "passive-muon-p20-independent-reconstruction-v1"

LOWER = Fraction(125, 1_024)
UPPER = Fraction(509, 512)
CENTER = Fraction(1_143, 2_048)
RADIUS = Fraction(893, 2_048)
ACCEPTANCE_BUFFER = Fraction(1, 1_024)
ACCEPTANCE_COEFFICIENT = Fraction(891, 2_048)

U32 = Fraction(1, 2**24)
TAU32 = Fraction(1, 2**150)
MIN_SUBNORMAL32 = Fraction(1, 2**149)
MIN_NORMAL32 = Fraction(1, 2**126)
MAX_SUBNORMAL32 = MIN_NORMAL32 - MIN_SUBNORMAL32
MIN_SUBNORMAL_BF16 = Fraction(1, 2**133)

SQRT_GRID = 2**80
RADIUS_GRID = 2**60
BLOCK_ENTRIES = 2**20
MAX_ENTRIES = 2**52

MAXIMUM_STEP = Fraction(1, 83)
MAXIMUM_RATE = Fraction(999_598_040_401, 1_000_000_000_000)
FASTER_STEP = Fraction(1, 120)
FASTER_RATE = Fraction(624_350_169, 625_000_000)

SHAPES = (
    (768, 768),
    (768, 3_072),
    (768, 50_257),
    (3_072, 12_288),
    (4_096, 4_096),
    (4_096, 11_008),
    (4_096, 14_336),
)
DIAGNOSTIC_SHAPES = ((1, 1), (1, 2), (2, 2))

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--require-canonical", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _ceil_sqrt(value: int) -> int:
    root = isqrt(value)
    return root if root * root == value else root + 1


def _ceil_log2(value: int) -> int:
    return (value - 1).bit_length()


def _sqrt_bounds(value: Fraction) -> tuple[Fraction, Fraction]:
    scaled = value.numerator * SQRT_GRID**2
    numerator = isqrt(scaled // value.denominator)
    lower = Fraction(numerator, SQRT_GRID)
    upper = lower if lower * lower == value else Fraction(numerator + 1, SQRT_GRID)
    if not lower * lower <= value <= upper * upper:
        raise AssertionError("independent sqrt enclosure failed")
    return lower, upper


def _up_radius(value: Fraction) -> Fraction:
    quotient, remainder = divmod(value.numerator * RADIUS_GRID, value.denominator)
    return Fraction(quotient + bool(remainder), RADIUS_GRID)


def _shape_fields(shape: tuple[int, int]) -> dict[str, object]:
    rows, columns = shape
    entries = rows * columns
    if rows <= 0 or columns <= 0 or entries > MAX_ENTRIES:
        raise AssertionError("independent shape lies outside the P9 domain")
    h = _ceil_sqrt(entries)
    path = 1 + _ceil_log2(entries)
    path_error = path * U32
    gamma = path_error / (1 - path_error)
    sum_crumb = 2 * entries * TAU32 / (1 - path_error)
    normalized_error = U32 + h * TAU32
    square_lower = (1 - gamma) * (1 - normalized_error) ** 2 - sum_crumb
    square_upper = (1 + gamma) * (1 + normalized_error) ** 2 + sum_crumb
    sqrt_lower, _ = _sqrt_bounds(square_lower)
    _, sqrt_upper = _sqrt_bounds(square_upper)
    ell = (1 - U32) * sqrt_lower - TAU32
    upper_factor = (1 + U32) * sqrt_upper + TAU32

    displacement_slope = CENTER * U32 * (1 + U32)
    displacement_crumb = h * TAU32 * (2 + U32)
    relative_crumb = displacement_crumb / MIN_NORMAL32
    accepted_radius = (
        ACCEPTANCE_COEFFICIENT * upper_factor / ell + displacement_slope + relative_crumb
    ) / (1 - U32)
    fallback_radius = abs(CENTER - Fraction(1, 2)) + h * TAU32 / MIN_NORMAL32
    raw_radius = max(accepted_radius, fallback_radius)
    certified_radius = _up_radius(raw_radius)
    delta = RADIUS - certified_radius
    dead_zone = h * MAX_SUBNORMAL32
    zero_half_error = dead_zone / 2

    norm_certified = (
        0 < ell <= 1 <= upper_factor
        and sqrt_lower**2 <= square_lower
        and sqrt_upper**2 >= square_upper
        and path * U32 < 1
    )
    checks = {
        "shape_is_within_p9_entry_domain": entries <= MAX_ENTRIES,
        "sector_center_reconstructs": CENTER == (LOWER + UPPER) / 2,
        "sector_radius_reconstructs": RADIUS == (UPPER - LOWER) / 2,
        "acceptance_coefficient_is_exact_binary32": (
            ACCEPTANCE_COEFFICIENT.denominator <= 2**24
            and ACCEPTANCE_COEFFICIENT.denominator.bit_count() == 1
        ),
        "balanced_norm_envelope_certified": norm_certified,
        "blocked_tree_has_global_path_depth": path == 1 + _ceil_log2(entries),
        "normal_anchor_converts_crumbs": MIN_NORMAL32 == TAU32 / U32,
        "accepted_candidate_has_strict_inward_margin": accepted_radius < RADIUS,
        "rounded_half_has_strict_inward_margin": fallback_radius < RADIUS,
        "outward_radius_rounding_is_sound": raw_radius <= certified_radius,
        "certified_radius_is_inside_original_disk": certified_radius < RADIUS,
        "inward_margin_is_positive": delta > 0,
        "returned_output_is_in_p19_disk": certified_radius <= RADIUS,
        "normal_fallback_rounding_is_absorbed": fallback_radius <= certified_radius,
        "accepted_candidate_rounding_is_absorbed": accepted_radius <= certified_radius,
        "bf16_widened_minimum_can_be_halved_in_fp32": (MIN_SUBNORMAL_BF16 / 2 >= MIN_SUBNORMAL32),
        "dead_zone_bound_is_finite": zero_half_error > 0,
    }
    return {
        "shape": [rows, columns],
        "entries": entries,
        "root_entries_upper": h,
        "balanced_norm": {
            "pairwise_path_length": path,
            "reduction_block_entries": BLOCK_ENTRIES,
            "reduction_block_count": (entries + BLOCK_ENTRIES - 1) // BLOCK_ENTRIES,
            "pairwise_gamma": str(gamma),
            "pairwise_crumb": str(sum_crumb),
            "division_relative_error": str(normalized_error),
            "squared_norm_lower": str(square_lower),
            "squared_norm_upper": str(square_upper),
            "sqrt_lower": str(sqrt_lower),
            "sqrt_upper": str(sqrt_upper),
            "lower_factor": str(ell),
            "upper_factor": str(upper_factor),
        },
        "margin": {
            "normal_anchor": str(MIN_NORMAL32),
            "acceptance_coefficient": str(ACCEPTANCE_COEFFICIENT),
            "displacement_signal_slope": str(displacement_slope),
            "displacement_crumb": str(displacement_crumb),
            "displacement_crumb_relative": str(relative_crumb),
            "accepted_candidate_radius": str(accepted_radius),
            "rounded_half_radius": str(fallback_radius),
            "certified_inward_radius_raw": str(raw_radius),
            "certified_inward_radius": str(certified_radius),
            "inward_margin": str(delta),
        },
        "subnormal_guard": {
            "all_subnormal_input_radius": str(dead_zone),
            "zero_instead_of_half_output_error": str(zero_half_error),
        },
        "checks": checks,
        "certified": all(checks.values()),
    }


def reconstruction_fields() -> dict[str, object]:
    return {
        "sector": {
            "lower": str(LOWER),
            "upper": str(UPPER),
            "center": str(CENTER),
            "radius": str(RADIUS),
            "acceptance_buffer": str(ACCEPTANCE_BUFFER),
            "acceptance_coefficient": str(ACCEPTANCE_COEFFICIENT),
        },
        "arithmetic_constants": {
            "fp32_unit_roundoff": str(U32),
            "fp32_half_min_subnormal": str(TAU32),
            "fp32_min_subnormal": str(MIN_SUBNORMAL32),
            "fp32_min_normal": str(MIN_NORMAL32),
            "fp32_max_subnormal": str(MAX_SUBNORMAL32),
            "bf16_min_subnormal": str(MIN_SUBNORMAL_BF16),
            "sqrt_enclosure_denominator": SQRT_GRID,
            "radius_grid_denominator": RADIUS_GRID,
            "reduction_block_entries": BLOCK_ENTRIES,
        },
        "formulae": {
            "norm_vector": "v=x/maxabs(x); z=fl32(v)",
            "norm_reduction": (
                "qhat=balanced_pairwise_fl32(sum_i fl32(z_i*z_i)); "
                "n32=fl32(sqrt(qhat)); Nhat=float64(maxabs(x))*float64(n32)"
            ),
            "candidate_displacement": "Dhat=fl32(C-fl32((1143/2048)*S))",
            "acceptance": ("Nhat(Dhat)<=nextafter(fl64((891/2048)*Nhat(S)),-infinity)"),
            "accepted_radius": (
                "((891/2048)*U/ell+(1143/2048)*u*(1+u)+h*tau*(2+u)/sigma_min)/(1-u)"
            ),
            "fallback_radius": "abs(1143/2048-1/2)+h*tau/sigma_min",
            "delta": "893/2048-ceil_2^-60(max(accepted_radius,fallback_radius))",
        },
        "shapes": [_shape_fields(shape) for shape in SHAPES],
        "diagnostic_shapes": [_shape_fields(shape) for shape in DIAGNOSTIC_SHAPES],
        "operating_points": {
            "maximum_step": {
                "learning_rate": str(MAXIMUM_STEP),
                "rate": str(MAXIMUM_RATE),
            },
            "faster_rate": {
                "learning_rate": str(FASTER_STEP),
                "rate": str(FASTER_RATE),
            },
        },
    }


def _internal_checks(fields: dict[str, object]) -> dict[str, bool]:
    shapes = fields["shapes"]
    diagnostics = fields["diagnostic_shapes"]
    return {
        "sector_geometry": CENTER == (LOWER + UPPER) / 2 and RADIUS == (UPPER - LOWER) / 2,
        "seven_shapes": isinstance(shapes, list) and len(shapes) == 7,
        "all_shapes_certified": isinstance(shapes, list)
        and all(bool(shape["certified"]) for shape in shapes),
        "three_diagnostic_shapes_certified": isinstance(diagnostics, list)
        and [shape["shape"] for shape in diagnostics] == [[1, 1], [1, 2], [2, 2]]
        and all(bool(shape["certified"]) for shape in diagnostics),
        "large_shapes_present": isinstance(shapes, list)
        and [4_096, 11_008] in [shape["shape"] for shape in shapes]
        and [4_096, 14_336] in [shape["shape"] for shape in shapes],
        "all_margins_positive": isinstance(shapes, list)
        and all(Fraction(shape["margin"]["inward_margin"]) > 0 for shape in shapes),
        "both_rates_strict": MAXIMUM_RATE < 1 and FASTER_RATE < 1,
        "faster_rate_is_better": FASTER_RATE < MAXIMUM_RATE,
    }


def _canonical_comparison(
    canonical: Path, fields: dict[str, object], *, required: bool
) -> dict[str, object]:
    if not canonical.is_file():
        if required:
            raise FileNotFoundError(f"canonical P20 artifact not found: {canonical}")
        return {"status": "not_found", "path": str(canonical), "comparisons": {}}
    payload = json.loads(canonical.read_text(encoding="utf-8"))
    comparisons = {
        "schema_version": (
            payload.get("schema_version") == "passive-muon-scalable-sector-shield-v1"
        ),
        "reconstruction_fields": payload.get("reconstruction_fields") == fields,
        "all_exact_checks_passed": payload.get("all_exact_checks_passed") is True,
    }
    if not all(comparisons.values()):
        raise AssertionError("canonical P20 artifact has missing or mismatched exact fields")
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
            "arithmetic": "fractions.Fraction and integer isqrt",
        },
        "reconstruction": {"fields": fields, "checks": checks},
        "canonical_comparison": comparison,
        "all_internal_exact_checks_passed": all(checks.values()),
        "all_exact_checks_passed": all(checks.values())
        and comparison["status"] in {"not_found", "matched"},
    }
    if not result["all_exact_checks_passed"]:
        raise SystemExit("independent P20 reconstruction failed")
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
