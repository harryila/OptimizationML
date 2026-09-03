#!/usr/bin/env python3
"""Replay the exact P9 shape-parameterized mixed-precision certificate."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Any

from passive_muon.scalable_mixed_precision_certificate import (
    BF16_HALF_MIN_SUBNORMAL,
    BF16_MAX_FINITE,
    BF16_UNIT_ROUNDOFF,
    EXACT_A,
    EXACT_B,
    EXACT_C,
    EXACT_RHO,
    FP32_A,
    FP32_B,
    FP32_C,
    FP32_HALF_MIN_SUBNORMAL,
    FP32_MAX_FINITE,
    FP32_RHO,
    FP32_UNIT_ROUNDOFF,
    IDEAL_REPAIRED_LIPSCHITZ,
    LOCKED_MAX_ENTRIES,
    LOCKED_SAFE_MAX_ABS,
    LOCKED_STAGES,
    NORMALIZER_DENOMINATOR_ERROR_UPPER,
    NORMALIZER_OUTPUT_ERROR_UPPER,
    P7_IMPLEMENTATION_ERROR_GAIN,
    P7_RATE,
    P7_SIGNAL_STORAGE_GAIN,
    P7_SMOOTHNESS,
    P7_STORAGE_FUNCTION_LOWER,
    POLYNOMIAL_LIPSCHITZ_ON_TUBE,
    POLYNOMIAL_SPECTRAL_UPPER,
    REPRESENTATIVE_TRANSFORMER_SHAPES,
    SPECTRAL_TUBE,
    UPPER_GRID_DENOMINATOR,
    ScalableMixedPrecisionAudit,
    audit_scalable_shape,
    compensated_boundary_rank_limit,
    ideal_two_term_boundary_rank_limit,
    one_term_boundary_rank_limit,
    representative_audits,
    serial_norm_obstruction,
    two_term_bf16_bound,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = (
    "scripts/certify_scalable_mixed_precision.py",
    "scripts/reconstruct_scalable_mixed_precision.py",
    "src/passive_muon/scalable_mixed_precision.py",
    "src/passive_muon/scalable_mixed_precision_certificate.py",
    "src/passive_muon/mixed_precision_certificate.py",
    "src/passive_muon/floored_certificate.py",
    "src/passive_muon/pl_convergence.py",
    "src/passive_muon/specs.py",
    "src/passive_muon/structure_aware_stability.py",
    "src/passive_muon/robust_dissipativity.py",
    "tests/test_scalable_mixed_precision.py",
    "tests/test_scalable_mixed_precision_certificate.py",
    "tests/test_scalable_mixed_precision_cli.py",
    "tests/test_scalable_mixed_precision_reconstruction.py",
    "theory/robust_dissipativity_certificate.md",
    "theory/mixed_precision_certificate.md",
    "theory/scalable_mixed_precision_certificate.md",
    "theory/audits/P9_HUMAN_PROOF_AUDIT.md",
    ".github/workflows/p9-scalable-mixed-precision.yml",
    "pyproject.toml",
    "uv.lock",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="optional JSON output path; JSON is always printed to stdout",
    )
    return parser.parse_args()


def _decimal(value: Fraction, *, digits: int = 50) -> str:
    with localcontext() as context:
        context.prec = digits
        return str(Decimal(value.numerator) / Decimal(value.denominator))


def _fraction_payload(value: Fraction) -> dict[str, str]:
    return {"exact": str(value), "decimal": _decimal(value)}


def _exact_fraction_map(values: dict[str, Fraction]) -> dict[str, str]:
    return {name: str(value) for name, value in values.items()}


def _sha256(relative_path: str) -> str:
    return hashlib.sha256((REPOSITORY_ROOT / relative_path).read_bytes()).hexdigest()


def _source_snapshot() -> dict[str, str]:
    return {path: _sha256(path) for path in SOURCE_PATHS}


def _git_output(*arguments: str) -> str | None:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _git_state() -> dict[str, object]:
    status = _git_output("status", "--porcelain")
    return {
        "sha": _git_output("rev-parse", "HEAD") or "unavailable",
        "branch": _git_output("branch", "--show-current") or "unavailable",
        "dirty": bool(status) if status is not None else None,
    }


def _normalizer_payload(audit: ScalableMixedPrecisionAudit) -> dict[str, object]:
    normalizer = audit.normalizer
    values = {
        "gamma": normalizer.gamma,
        "sum_crumb": normalizer.sum_crumb,
        "relative_vector_error": normalizer.relative_vector_error,
        "norm_branch_squared_error": normalizer.norm_branch_squared_error,
        "norm_branch_denominator_error": normalizer.norm_branch_denominator_error,
        "floor_branch_squared_error": normalizer.floor_branch_squared_error,
        "floor_branch_denominator_error": normalizer.floor_branch_denominator_error,
        "small_floor_sum_upper": normalizer.small_floor_sum_upper,
        "output_error": normalizer.output_error,
    }
    return {
        "values": _exact_fraction_map(values),
        "checks": normalizer.checks,
        "certified": normalizer.certified,
    }


def _polynomial_payload(audit: ScalableMixedPrecisionAudit) -> dict[str, object]:
    polynomial = audit.polynomial
    values = {
        "squared_endpoint": polynomial.squared_endpoint,
        "factor_discriminant": polynomial.factor_discriminant,
        "factor_correction_at_endpoint": polynomial.factor_correction_at_endpoint,
        "derivative_vertex": polynomial.derivative_vertex,
        "derivative_at_zero": polynomial.derivative_at_zero,
        "derivative_at_vertex": polynomial.derivative_at_vertex,
        "derivative_at_endpoint": polynomial.derivative_at_endpoint,
    }
    return {
        "values": _exact_fraction_map(values),
        "checks": polynomial.checks,
        "certified": polynomial.certified,
        "range_source": ("P8 exact Sturm artifact proves 0<=q(x)<121/100 on 0<=x<=5/4"),
    }


def _stage_payload(audit: ScalableMixedPrecisionAudit) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for stage in audit.stages:
        values = {
            "input_spectral": stage.input_spectral,
            "input_frobenius": stage.input_frobenius,
            "gram_error": stage.gram_error,
            "scaled_gram_error": stage.scaled_gram_error,
            "shifted_gram_error": stage.shifted_gram_error,
            "product_error": stage.product_error,
            "affine_error": stage.affine_error,
            "output_error_before_boundary": stage.output_error_before_boundary,
            "pre_boundary_spectral": stage.pre_boundary_spectral,
            "pre_boundary_frobenius": stage.pre_boundary_frobenius,
            "boundary_error": stage.boundary_error,
            "next_spectral": stage.next_spectral,
            "next_frobenius": stage.next_frobenius,
            "forward_error": stage.forward_error,
            "intermediate_max_frobenius": stage.intermediate_max_frobenius,
        }
        records.append({"index": stage.index, "values": _exact_fraction_map(values)})
    return records


def _shape_payload(audit: ScalableMixedPrecisionAudit) -> dict[str, object]:
    operator = {
        "binary32_slope": audit.operator.binary32_slope,
        "binary32_intercept": audit.operator.binary32_intercept,
        "real_slope": audit.operator.real_slope,
        "real_intercept": audit.operator.real_intercept,
    }
    p7 = {
        "rate": audit.p7.rate,
        "forcing": audit.p7.forcing,
        "function_gap_ultimate": audit.p7.function_gap_ultimate,
        "safe_storage_radius": audit.p7.safe_storage_radius,
        "safe_forcing_capacity": audit.p7.safe_forcing_capacity,
    }
    return {
        "shape": [audit.shape.rows, audit.shape.columns],
        "oriented_shape": list(audit.shape.oriented),
        "entries": audit.shape.entries,
        "polynomial": _polynomial_payload(audit),
        "normalizer": _normalizer_payload(audit),
        "initial_boundary_error": str(audit.initial_boundary_error),
        "stages": _stage_payload(audit),
        "operator_bound": _exact_fraction_map(operator),
        "p7": {
            "young_theta": audit.p7.young_theta,
            "values": _exact_fraction_map(p7),
        },
        "checks": audit.checks,
        "certified": audit.certified,
    }


def _boundary_payload() -> dict[str, object]:
    boundary = two_term_bf16_bound()
    values = {
        "residual_rounding_slope": boundary.residual_rounding_slope,
        "low_rounding_slope": boundary.low_rounding_slope,
        "reconstruction_rounding_slope": boundary.reconstruction_rounding_slope,
        "residual_rounding_crumb": boundary.residual_rounding_crumb,
        "rounded_residual_crumb": boundary.rounded_residual_crumb,
        "low_rounding_crumb": boundary.low_rounding_crumb,
        "high_plus_low_crumb": boundary.high_plus_low_crumb,
        "reconstruction_rounding_crumb": boundary.reconstruction_rounding_crumb,
        "omega": boundary.omega,
        "chi": boundary.chi,
    }
    return {"values": _exact_fraction_map(values), "sterbenz_free": boundary.sterbenz_free}


def _certificate_payload(audits: tuple[ScalableMixedPrecisionAudit, ...]) -> dict[str, object]:
    witness = serial_norm_obstruction((4_096, 11_008))
    arithmetic = {
        "exact_a": EXACT_A,
        "exact_b": EXACT_B,
        "exact_c": EXACT_C,
        "fp32_a": FP32_A,
        "fp32_b": FP32_B,
        "fp32_c": FP32_C,
        "exact_rho": EXACT_RHO,
        "fp32_rho": FP32_RHO,
        "fp32_unit_roundoff": FP32_UNIT_ROUNDOFF,
        "fp32_half_min_subnormal": FP32_HALF_MIN_SUBNORMAL,
        "bf16_unit_roundoff": BF16_UNIT_ROUNDOFF,
        "bf16_half_min_subnormal": BF16_HALF_MIN_SUBNORMAL,
        "fp32_max_finite": FP32_MAX_FINITE,
        "bf16_max_finite": BF16_MAX_FINITE,
        "spectral_tube": SPECTRAL_TUBE,
        "polynomial_spectral_upper": POLYNOMIAL_SPECTRAL_UPPER,
        "polynomial_lipschitz_on_tube": POLYNOMIAL_LIPSCHITZ_ON_TUBE,
        "ideal_repaired_lipschitz": IDEAL_REPAIRED_LIPSCHITZ,
        "normalizer_denominator_error_upper": NORMALIZER_DENOMINATOR_ERROR_UPPER,
        "normalizer_output_error_upper": NORMALIZER_OUTPUT_ERROR_UPPER,
        "p7_rate": P7_RATE,
        "p7_implementation_error_gain": P7_IMPLEMENTATION_ERROR_GAIN,
        "p7_signal_storage_gain": P7_SIGNAL_STORAGE_GAIN,
        "p7_smoothness": P7_SMOOTHNESS,
        "p7_storage_function_lower": P7_STORAGE_FUNCTION_LOWER,
        "locked_safe_max_abs": LOCKED_SAFE_MAX_ABS,
    }
    return {
        "upper_grid_denominator": str(UPPER_GRID_DENOMINATOR),
        "locked_stages": LOCKED_STAGES,
        "locked_max_entries": LOCKED_MAX_ENTRIES,
        "arithmetic": _exact_fraction_map(arithmetic),
        "two_term_boundary": _boundary_payload(),
        "rank_gates": {
            "ordinary_one_term": one_term_boundary_rank_limit(),
            "ideal_two_term": ideal_two_term_boundary_rank_limit(),
            "sterbenz_free_two_term": compensated_boundary_rank_limit(),
        },
        "serial_normalizer_obstruction": {
            "shape": [witness.shape.rows, witness.shape.columns],
            "exact_square_sum": witness.exact_square_sum,
            "computed_square_sum": witness.computed_square_sum,
            "returned_singular_value_squared": str(witness.returned_singular_value_squared),
            "leaves_spectral_tube": witness.leaves_spectral_tube,
        },
        "representative_shapes": [
            [shape.rows, shape.columns] for shape in REPRESENTATIVE_TRANSFORMER_SHAPES
        ],
        "shape_audits": [_shape_payload(audit) for audit in audits],
    }


def build_payload() -> dict[str, Any]:
    audits = representative_audits()
    if not all(audit.certified for audit in audits):
        failures = {
            audit.shape.oriented: [name for name, passed in audit.checks.items() if not passed]
            for audit in audits
            if not audit.certified
        }
        raise AssertionError(f"exact P9 replay failed: {failures}")

    frontier = audit_scalable_shape((4_608, 18_432))
    frontier_failures = [name for name, passed in frontier.checks.items() if not passed]
    if frontier_failures != ["stage_5_input_in_spectral_tube"]:
        raise AssertionError(f"unexpected qualified frontier result: {frontier_failures}")

    certificate = _certificate_payload(audits)
    return {
        "schema_version": "passive-muon-scalable-mixed-precision-certificate-v1",
        "claim_scope": {
            "evidence_kind": "exact rational shape-parameterized forward-error certificate",
            "guarantee": (
                "for every listed shape, ||R_hat(s)-R(s)||_F <= A_rc||s||_F+B_rc "
                "with the exact shape-row coefficients, and the resulting P7 rate is <1"
            ),
            "matrix_domain": (
                "finite contiguous CPU FP32 matrices at seven fixed Transformer shapes, "
                "max_abs<=2^116 and entries<=2^52; an all-real RN32 boundary adapter is "
                "included in each operator row"
            ),
            "normalization": "exact target s/max(1,||s||_F), no additive epsilon",
            "polynomial": (
                "five Jordan quintic stages with exact coefficients 6889/2000, -191/40, "
                "4063/2000 and the P7 deficit-derived linear repair"
            ),
            "qualified_frontier": (
                "4608x18432 is one audited 4:1 out-of-envelope point failing only the "
                "stage-five input tube; it is not a global largest-rank theorem"
            ),
            "not_claimed": [
                "literal upstream Muon or its current-plus-epsilon normalizer",
                "a dimension-free certificate for every matrix shape",
                "a native BLAS, GPU, tensor-core, or production-throughput implementation",
                "state compression; two BF16 buffers use the same nominal storage as FP32",
                "end-to-end FP32 EMA/Nesterov or parameter-update convergence",
                "aspect-ratio scaling, weight decay, or stochastic-gradient coverage",
                "minimality or tightness of the affine intercepts and P7 neighborhoods",
            ],
        },
        "implementation_contract": {
            "orientation": "transpose once to p=min(rows,columns), d=max(rows,columns)",
            "normalizer": (
                "sigma=max(1,maxabs(s)); z=fl32(s/sigma); balanced FP32 norm; "
                "fl32(z/max(fl32(1/sigma),fl32(sqrt(sum(z*z)))))"
            ),
            "stage_dag": [
                "G=balanced_fp32(X X^T)",
                "T=fl32(c32*G+b32*I)",
                "D=balanced_fp32(T G)",
                "E=fl32(D+a32*I)",
                "Y=balanced_fp32(E X)",
            ],
            "boundary": (
                "high=RN_bf16(Y); residual=fl32(Y-fp32(high)); "
                "low=RN_bf16(residual); X_next=fl32(fp32(high)+fp32(low))"
            ),
            "rounding": "IEEE-754 round-to-nearest, ties-to-even; gradual underflow",
            "fma_and_reassociation": "excluded",
            "upper_rationalization": (
                "every named propagated upper bound rounds upward to denominator 2^40"
            ),
        },
        "certificate": certificate,
        "display_summary": [
            {
                "shape": item["shape"],
                "A_rc": _fraction_payload(Fraction(item["operator_bound"]["real_slope"])),
                "B_rc": _fraction_payload(Fraction(item["operator_bound"]["real_intercept"])),
                "q_rc": _fraction_payload(Fraction(item["p7"]["values"]["rate"])),
                "function_gap_ultimate": _fraction_payload(
                    Fraction(item["p7"]["values"]["function_gap_ultimate"])
                ),
            }
            for item in certificate["shape_audits"]
        ],
        "qualified_negative_controls": {
            "one_term_boundary": (
                "the generic one-term BF16 spectral-tube proof supports rank at most 71; "
                "this is a proof obstruction, not an instability theorem"
            ),
            "serial_normalizer": (
                "the 4096x11008 all-ones serial FP32 square sum sticks at 2^24 and "
                "returns squared singular value 43/16, outside the 5/4 tube"
            ),
            "audited_4_to_1_case": {
                "shape": [4_608, 18_432],
                "failed_checks": frontier_failures,
                "interpretation": "specific recurrence boundary, not a global rank frontier",
            },
        },
        "audit": {
            "representative_shape_count": len(audits),
            "all_representative_exact_checks_passed": all(audit.certified for audit in audits),
            "all_shape_checks": {
                f"{audit.shape.rows}x{audit.shape.columns}": audit.checks for audit in audits
            },
        },
        "proof_replay_provenance": {
            "arithmetic": "fractions.Fraction exact arithmetic with explicit upward 2^-40 grid",
            "numeric_solver_role": "none",
            "source_snapshot": _source_snapshot(),
            "software": {"python": sys.version},
            "hardware": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
            },
        },
        "git": _git_state(),
    }


def main() -> None:
    args = parse_args()
    payload = build_payload()
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
