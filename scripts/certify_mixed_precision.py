#!/usr/bin/env python3
"""Replay the exact P8 mixed-precision operator-error certificate."""

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

from passive_muon.mixed_precision_certificate import (
    AFFINE_INTERCEPT_UPPER,
    AFFINE_SLOPE_UPPER,
    BF16_HALF_MIN_SUBNORMAL,
    BF16_UNIT_ROUNDOFF,
    EXACT_A,
    EXACT_B,
    EXACT_C,
    EXACT_RHO,
    FP32_A,
    FP32_B,
    FP32_C,
    FP32_HALF_MIN_SUBNORMAL,
    FP32_RHO,
    FP32_UNIT_ROUNDOFF,
    FROBENIUS_INVARIANT,
    IDEAL_REPAIRED_LIPSCHITZ,
    LOCAL_FP32_STAGE_ERROR_UPPER,
    LOCKED_SAFE_MAX_ABS,
    LOCKED_SHAPE,
    LOCKED_STAGES,
    NORMALIZATION_OUTPUT_ERROR_UPPER,
    P8_FORCING,
    P8_FUNCTION_GAP_ULTIMATE,
    P8_RATE,
    P8_SAFE_FORCING_CAPACITY,
    P8_SAFE_STORAGE_RADIUS,
    POLYNOMIAL_ERROR_UPPER,
    POLYNOMIAL_UPPER,
    REAL_ADAPTER_INTERCEPT_UPPER,
    REAL_ADAPTER_SLOPE_UPPER,
    SPECTRAL_INVARIANT,
    SQRT_TWO_UPPER,
    audit_mixed_precision_certificate,
    locked_mixed_precision_certificate,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = (
    "scripts/certify_mixed_precision.py",
    "scripts/reconstruct_mixed_precision.py",
    "src/passive_muon/mixed_precision.py",
    "src/passive_muon/mixed_precision_certificate.py",
    "src/passive_muon/specs.py",
    "src/passive_muon/structure_aware_stability.py",
    "src/passive_muon/floored_certificate.py",
    "src/passive_muon/robust_dissipativity.py",
    "src/passive_muon/pl_convergence.py",
    "tests/test_mixed_precision_certificate.py",
    "tests/test_mixed_precision_cli.py",
    "tests/test_mixed_precision_reconstruction.py",
    "tests/test_mixed_precision.py",
    "theory/robust_dissipativity_certificate.md",
    "theory/mixed_precision_certificate.md",
    "theory/audits/P7_HUMAN_PROOF_AUDIT.md",
    ".github/workflows/p8-mixed-precision.yml",
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


def _fraction_map(values: dict[str, Fraction]) -> dict[str, dict[str, str]]:
    return {name: _fraction_payload(value) for name, value in values.items()}


def _polynomial_payload(
    chain: tuple[tuple[Fraction, ...], ...],
) -> list[list[str]]:
    return [[str(value) for value in polynomial] for polynomial in chain]


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


def build_payload() -> dict[str, Any]:
    certificate = locked_mixed_precision_certificate()
    audit = audit_mixed_precision_certificate(certificate)
    if not audit.certified:
        failed = [name for name, passed in audit.checks.items() if not passed]
        raise AssertionError(f"exact P8 replay failed: {failed}")

    return {
        "schema_version": "passive-muon-mixed-precision-certificate-v1",
        "claim_scope": {
            "evidence_kind": "exact rational shape-locked forward-error certificate",
            "guarantee": (
                "||R_hat(s)-R(s)||_F <= (11/100000)||s||_F+347/100 for every "
                "locked 2x2 FP32 input with max_abs<=2^116"
            ),
            "p7_consequence": (
                "exact-real outer-loop storage dissipation for the real-boundary adapter "
                "R_hat_real(s)=kernel(RN32(s)), with zero gradient noise, a strict modified "
                "rate, and constant forcing"
            ),
            "matrix_domain": (
                "primary bound: finite FP32 2x2 inputs; adapter bound: finite real 2x2 inputs; "
                "both require max_abs<=2^116"
            ),
            "overflow_scope": (
                "the explicit max_abs<=2^116 guard and stage invariant exclude overflow; NaN, "
                "infinity, FTZ, and DAZ execution are excluded"
            ),
            "not_claimed": [
                "upstream Muon or its additive-epsilon/current-normalized operator",
                "the older all-intermediate-BF16 evaluation DAG",
                "arbitrary matrix shape or dimension-independent mixed-precision error",
                "an accelerator-native BF16 matmul backend",
                "end-to-end FP32 EMA/Nesterov or FP32 parameter-update convergence",
                "exact convergence rather than an ultimate objective neighborhood",
                "tightness or minimality of the 347/100 intercept",
            ],
        },
        "implementation_contract": {
            "shape": list(LOCKED_SHAPE),
            "maximum_input_absolute_value": _fraction_payload(LOCKED_SAFE_MAX_ABS),
            "stage_count": LOCKED_STAGES,
            "rounding": "IEEE-754 round-to-nearest, ties-to-even",
            "subnormals": "gradual underflow; FTZ and DAZ disabled",
            "stage_storage": "BF16 normalized input plus five BF16 stage outputs",
            "stage_compute": "FP32 fixed-order Horner evaluation",
            "matrix_product": (
                "row-major 2x2 product with each length-2 dot evaluated as two separately "
                "rounded FP32 multiplies followed by one separately rounded FP32 addition"
            ),
            "fma_policy": "FMA contraction is not assumed and is excluded from this contract",
            "stage_dag": [
                "A=FP32(X@X^T)",
                "C=FP32(c32*A); T=FP32(b32*I+C)",
                "D=FP32(T@A)",
                "E=FP32(a32*I+D)",
                "Y=FP32(E@X)",
                "X_next=BF16_RNE(Y)",
            ],
            "normalizer": (
                "scaled serial FP32 Frobenius max-floor; for max_abs<=1/2 return s exactly, "
                "otherwise use the certified active-branch norm routine"
            ),
            "normalization_rule_target": "N(s)=s/max(1,||s||_F); no additive epsilon",
            "repair_and_return": "rho32*s and polynomial-plus-repair addition in FP32",
            "real_boundary_adapter": (
                "R_hat_real(s)=kernel(RN32(s)), with entrywise FP32 RNE before the guarded kernel"
            ),
            "accepted_execution": "every intermediate and returned value must be finite",
        },
        "formats": {
            "fp32_unit_roundoff": _fraction_payload(FP32_UNIT_ROUNDOFF),
            "fp32_half_min_subnormal": _fraction_payload(FP32_HALF_MIN_SUBNORMAL),
            "bf16_unit_roundoff": _fraction_payload(BF16_UNIT_ROUNDOFF),
            "bf16_half_min_subnormal": _fraction_payload(BF16_HALF_MIN_SUBNORMAL),
            "sqrt_two_strict_upper": _fraction_payload(SQRT_TWO_UPPER),
        },
        "operator": {
            "ideal_formula": "R(s)=H_(q composed 5)(s/max(1,||s||_F))+rho*s",
            "epsilon": "none; exact max-floor target",
            "polynomial_formula": "q(x)=a*x+b*x^3+c*x^5",
            "polynomial_coefficients": {
                "exact": {
                    "a": str(EXACT_A),
                    "b": str(EXACT_B),
                    "c": str(EXACT_C),
                },
                "fp32": {
                    "a": {"exact": str(FP32_A), "bits_hex": "0x405c72b0"},
                    "b": {"exact": str(FP32_B), "bits_hex": "0xc098cccd"},
                    "c": {"exact": str(FP32_C), "bits_hex": "0x40020419"},
                },
            },
            "repair": {
                "rho_exact": _fraction_payload(EXACT_RHO),
                "rho_fp32": {
                    **_fraction_payload(FP32_RHO),
                    "bits_hex": "0x4449e421",
                },
            },
        },
        "polynomial_range_certificate": {
            "nonnegative_factor_discriminant": _fraction_payload(audit.polynomial_discriminant),
            "claim": "0<=q(s)<121/100 for every 0<=s<=5/4",
            "upper_polynomial": "g(s)=121/100-q(s)",
            "sturm_chain_coefficients_ascending": _polynomial_payload(audit.sturm_chain),
            "endpoint_values": [
                [str(value) for value in values] for values in audit.sturm_endpoint_values
            ],
            "endpoint_signs": [list(signs) for signs in audit.sturm_endpoint_signs],
            "endpoint_variations": list(audit.sturm_variations),
            "same_variation_proves_no_roots": audit.sturm_variations[0]
            == audit.sturm_variations[1],
            "polynomial_upper": _fraction_payload(POLYNOMIAL_UPPER),
        },
        "normalization_certificate": {
            "branch_qualification": (
                "relative norm-error analysis is used only for max_abs>1/2; below that threshold "
                "the exact 2x2 Frobenius norm is at most one and the input is returned"
            ),
            "values": _fraction_map(audit.normalization),
            "normalized_output_error_upper": _fraction_payload(NORMALIZATION_OUTPUT_ERROR_UPPER),
        },
        "stage_invariant_certificate": {
            "spectral_invariant": _fraction_payload(SPECTRAL_INVARIANT),
            "frobenius_invariant": _fraction_payload(FROBENIUS_INVARIANT),
            "local_fp32_horner_error_upper": _fraction_payload(LOCAL_FP32_STAGE_ERROR_UPPER),
            "important_qualification": (
                "the 1/20000 bound is pre-BF16 FP32-Horner error; boundary rounding is recorded "
                "separately and is not included in that number"
            ),
            "one_stage_recurrence": _fraction_map(audit.recurrence),
            "induction": (
                "the initial cast enters the invariant and the identical one-stage bound closes "
                "it strictly, so it holds after each of exactly five stages"
            ),
        },
        "operator_error_certificate": {
            "polynomial_error_upper": _fraction_payload(POLYNOMIAL_ERROR_UPPER),
            "repair_rounding": _fraction_map(audit.repair),
            "binary32_input_bound": {
                "formula": "||kernel(s)-R(s)||_F <= A32*||s||_F+B32",
                "A32": _fraction_payload(AFFINE_SLOPE_UPPER),
                "B32": _fraction_payload(AFFINE_INTERCEPT_UPPER),
            },
            "real_boundary_adapter_bound": {
                "formula": "||kernel(RN32(s))-R(s)||_F <= A_real*||s||_F+B_real",
                "cast_error": "||RN32(s)-s||_F<=u32*||s||_F+2*t32",
                "ideal_repaired_map_lipschitz": _fraction_payload(IDEAL_REPAIRED_LIPSCHITZ),
                "A_real": _fraction_payload(REAL_ADAPTER_SLOPE_UPPER),
                "B_real": _fraction_payload(REAL_ADAPTER_INTERCEPT_UPPER),
            },
        },
        "p7_closure": {
            "derivation": (
                "for the real-boundary adapter, ||s||_F^2<=C_s V and "
                "(A_real||s||+B_real)^2<=(1+theta)A_real^2||s||^2+"
                "(1+1/theta)B_real^2"
            ),
            "values": _fraction_map(audit.p7_closure),
            "modified_rate_q8": _fraction_payload(P8_RATE),
            "constant_forcing": _fraction_payload(P8_FORCING),
            "zero_gradient_noise_function_gap_limsup": _fraction_payload(P8_FUNCTION_GAP_ULTIMATE),
            "safe_range_invariant": {
                "initial_storage_upper": _fraction_payload(P8_SAFE_STORAGE_RADIUS),
                "forcing_capacity": _fraction_payload(P8_SAFE_FORCING_CAPACITY),
                "signal_selector": "||s_(t+1)||_F^2<=C_s*V_t",
                "induction": (
                    "when gradient noise is zero, V_0<=H and B_V<=(1-q8)H imply "
                    "V_t<=H and max_abs(s_(t+1))<=2^116 for every t"
                ),
            },
            "outer_loop_scope": (
                "P7 exact-real EMA/Nesterov and parameter shell with the real-boundary adapter "
                "instantiating only the additive post-operator error port and with gradient noise "
                "set to zero; the direct operator bound requires max_abs<=2^116, while V_0<=H "
                "is a sufficient condition that preserves this guard for every future signal"
            ),
        },
        "audit": {
            "checks": audit.checks,
            "all_exact_checks_passed": audit.certified,
        },
        "negative_control": {
            "older_dag_status": (
                "the all-intermediate-BF16 DAG is not certified; cancellation-aware normwise "
                "bounds leave the useful spectral tube after its first stage"
            ),
            "interpretation": (
                "failure of that upper-bound strategy is not an executable instability proof"
            ),
        },
        "proof_replay_provenance": {
            "arithmetic": "fractions.Fraction exact rational arithmetic",
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
