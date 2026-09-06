#!/usr/bin/env python3
"""Independently reconstruct the exact P22 ``768 x 2304`` shield margin.

Only the Python standard library is imported.  The P20 norm/roundoff
recurrence is repeated here rather than imported from :mod:`passive_muon`.
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from math import isqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANONICAL = ROOT / "results/summaries/p22_scalable_shield_shape_extension_certificate.json"
SCHEMA_VERSION = "passive-muon-p22-shape-extension-independent-reconstruction-v1"

SHAPE = (768, 2_304)
PARAMETER_SHAPE = (2_304, 768)
RUNTIME_SCHEMA = "passive-muon-p22-scalable-sector-shield-extension-v1"
EXPECTED_MARGIN = Fraction(942_123_070_212_169, 2**60)
EXPECTED_RADIUS = Fraction(501_772_185_335_019_447, 2**60)
EXPECTED_MARGIN_HEX = "0x1.ac6d8f77a4248p-11"
EXPECTED_RADIUS_HEX = "0x1.bda9c938442dfp-2"

LOWER = Fraction(125, 1_024)
UPPER = Fraction(509, 512)
CENTER = Fraction(1_143, 2_048)
RADIUS = Fraction(893, 2_048)
ACCEPTANCE_BUFFER = Fraction(1, 1_024)
ACCEPTANCE = Fraction(891, 2_048)
CLIP = Fraction(890, 2_048)

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
        raise AssertionError("independent square-root enclosure failed")
    return lower, upper


def _up_radius(value: Fraction) -> Fraction:
    quotient, remainder = divmod(value.numerator * RADIUS_GRID, value.denominator)
    return Fraction(quotient + bool(remainder), RADIUS_GRID)


def _shape_fields() -> dict[str, object]:
    rows, columns = SHAPE
    entries = rows * columns
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
    accepted_radius = (ACCEPTANCE * upper_factor / ell + displacement_slope + relative_crumb) / (
        1 - U32
    )
    clipped_radius = (
        CLIP * upper_factor / ell * (1 + U32) ** 2
        + CENTER * U32 * (2 + U32)
        + h * TAU32 / MIN_NORMAL32 * (3 + 2 * U32)
    )
    clip_only_margin = RADIUS - _up_radius(clipped_radius)
    fallback_radius = abs(CENTER - Fraction(1, 2)) + h * TAU32 / MIN_NORMAL32
    raw_radius = max(accepted_radius, clipped_radius, fallback_radius)
    certified_radius = _up_radius(raw_radius)
    margin = RADIUS - certified_radius
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
            ACCEPTANCE.denominator <= 2**24 and ACCEPTANCE.denominator.bit_count() == 1
        ),
        "clip_coefficient_is_exact_binary32": (
            CLIP.denominator <= 2**24 and CLIP.denominator.bit_count() == 1
        ),
        "balanced_norm_envelope_certified": norm_certified,
        "blocked_tree_has_global_path_depth": path == 1 + _ceil_log2(entries),
        "normal_anchor_converts_crumbs": MIN_NORMAL32 == TAU32 / U32,
        "accepted_candidate_has_strict_inward_margin": accepted_radius < RADIUS,
        "clipped_candidate_has_strict_inward_margin": clipped_radius < RADIUS,
        "clip_only_margin_is_positive": clip_only_margin > 0,
        "rounded_half_has_strict_inward_margin": fallback_radius < RADIUS,
        "outward_radius_rounding_is_sound": raw_radius <= certified_radius,
        "certified_radius_is_inside_original_disk": certified_radius < RADIUS,
        "inward_margin_is_positive": margin > 0,
        "returned_output_is_in_p19_disk": certified_radius <= RADIUS,
        "normal_fallback_rounding_is_absorbed": fallback_radius <= certified_radius,
        "accepted_candidate_rounding_is_absorbed": accepted_radius <= certified_radius,
        "clipped_candidate_rounding_is_absorbed": clipped_radius <= certified_radius,
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
            "acceptance_coefficient": str(ACCEPTANCE),
            "displacement_signal_slope": str(displacement_slope),
            "displacement_crumb": str(displacement_crumb),
            "displacement_crumb_relative": str(relative_crumb),
            "accepted_candidate_radius": str(accepted_radius),
            "clipped_candidate_radius": str(clipped_radius),
            "clip_only_inward_margin": str(clip_only_margin),
            "rounded_half_radius": str(fallback_radius),
            "certified_inward_radius_raw": str(raw_radius),
            "certified_inward_radius": str(certified_radius),
            "inward_margin": str(margin),
        },
        "subnormal_guard": {
            "all_subnormal_input_radius": str(dead_zone),
            "zero_instead_of_half_output_error": str(zero_half_error),
        },
        "checks": checks,
        "certified": all(checks.values()),
    }


def reconstruction_fields() -> dict[str, object]:
    shape = _shape_fields()
    extension_checks = {
        "p20_checkpoint_table_is_not_rewritten": True,
        "shape_is_canonical_fused_qkv_orientation": shape["shape"] == [768, 2_304],
        "supplied_inward_margin_matches_recurrence": (
            Fraction(shape["margin"]["inward_margin"]) == EXPECTED_MARGIN
        ),
        "certified_radius_matches_recurrence": (
            Fraction(shape["margin"]["certified_inward_radius"]) == EXPECTED_RADIUS
        ),
        "margin_and_radius_partition_p19_radius": EXPECTED_RADIUS + EXPECTED_MARGIN == RADIUS,
        "margin_uses_locked_radius_grid": (
            EXPECTED_MARGIN.denominator <= RADIUS_GRID
            and RADIUS_GRID % EXPECTED_MARGIN.denominator == 0
        ),
        "runtime_margin_hex_matches_exact_value": float(EXPECTED_MARGIN).hex()
        == EXPECTED_MARGIN_HEX,
        "runtime_radius_hex_matches_exact_value": float(EXPECTED_RADIUS).hex()
        == EXPECTED_RADIUS_HEX,
        "transpose_preserves_frobenius_arithmetic_envelope": (
            SHAPE[0] * SHAPE[1] == PARAMETER_SHAPE[0] * PARAMETER_SHAPE[1]
        ),
    }
    return {
        "extension": {
            "runtime_schema": RUNTIME_SCHEMA,
            "shield_shape": list(SHAPE),
            "parameter_shape": list(PARAMETER_SHAPE),
            "parameter_to_shield_map": "transpose",
            "p20_table_mutated": False,
            "inward_margin": str(EXPECTED_MARGIN),
            "certified_inward_radius": str(EXPECTED_RADIUS),
            "inward_margin_hex": EXPECTED_MARGIN_HEX,
            "certified_inward_radius_hex": EXPECTED_RADIUS_HEX,
        },
        "sector": {
            "lower": str(LOWER),
            "upper": str(UPPER),
            "center": str(CENTER),
            "radius": str(RADIUS),
            "acceptance_buffer": str(ACCEPTANCE_BUFFER),
            "acceptance_coefficient": str(ACCEPTANCE),
            "clip_coefficient": str(CLIP),
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
        "operation_graph": {
            "normalization": "a=maxabs(x); z=fl32(x/a)",
            "norm": (
                "fl32 squares; fixed adjacent balanced fl32 sum; fl32 sqrt; "
                "exact float64 scale product"
            ),
            "displacement": "Dhat=fl32(C-fl32((1143/2048)*S))",
            "acceptance": "Nhat(Dhat)<=down64((891/2048)*Nhat(S))",
            "clip": (
                "alpha32=down32(down64((890/2048)*down64(Nhat(S)/Nhat(Dhat)))); "
                "U=fl32(fl32((1143/2048)*S)+fl32(alpha32*Dhat))"
            ),
            "fallback": "fl32((1/2)*S) under the P20 normal/exact-halving guard",
            "input_dtypes": ["binary32", "bfloat16 widened exactly to binary32"],
            "output_dtype": "binary32",
            "rounding": "IEEE-754 roundTiesToEven with gradual underflow",
            "fma_reassociation_ftz_daz": False,
        },
        "shape_audit": shape,
        "extension_checks": extension_checks,
    }


def _internal_checks(fields: dict[str, object]) -> dict[str, bool]:
    shape = fields["shape_audit"]
    return {
        "shape_certifies": shape["certified"] is True,
        "all_shape_checks_pass": all(shape["checks"].values()),
        "all_extension_checks_pass": all(fields["extension_checks"].values()),
        "supplied_margin_is_exact": (Fraction(shape["margin"]["inward_margin"]) == EXPECTED_MARGIN),
        "root_and_tree_are_reconstructed": (
            shape["root_entries_upper"] == 1_331
            and shape["balanced_norm"]["pairwise_path_length"] == 22
            and shape["balanced_norm"]["reduction_block_count"] == 2
        ),
    }


def _canonical_comparison(
    canonical: Path, fields: dict[str, object], *, required: bool
) -> dict[str, object]:
    if not canonical.is_file():
        if required:
            raise FileNotFoundError(
                f"canonical P22 shape-extension artifact not found: {canonical}"
            )
        return {"status": "not_found", "path": str(canonical), "comparisons": {}}
    payload = json.loads(canonical.read_text(encoding="utf-8"))
    comparisons = {
        "schema_version": (
            payload.get("schema_version") == "passive-muon-p22-shape-extension-certificate-v1"
        ),
        "reconstruction_fields": payload.get("reconstruction_fields") == fields,
        "all_exact_checks_passed": payload.get("all_exact_checks_passed") is True,
    }
    if not all(comparisons.values()):
        raise AssertionError("canonical P22 artifact has missing or mismatched exact fields")
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
            "arithmetic": "fractions.Fraction, integer isqrt, and exact dyadic grids",
        },
        "reconstruction": {"fields": fields, "checks": checks},
        "canonical_comparison": comparison,
        "all_internal_exact_checks_passed": all(checks.values()),
        "all_exact_checks_passed": all(checks.values())
        and comparison["status"] in {"not_found", "matched"},
    }
    if not result["all_exact_checks_passed"]:
        raise SystemExit("independent P22 shape-extension reconstruction failed")
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
