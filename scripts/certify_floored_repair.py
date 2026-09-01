#!/usr/bin/env python3
"""Generate the rigorous five-step Jordan floored-normalizer certificate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Any

import flint

from passive_muon.floored_certificate import (
    JORDAN_DERIVATIVE_LOWER,
    JORDAN_DERIVATIVE_UPPER,
    JORDAN_PAIR_DEFICIT_LOWER,
    JORDAN_PAIR_LEFT,
    JORDAN_PAIR_RIGHT,
    certified_dimension_uniform_deficit,
    certify_jordan_derivative_range,
    exact_jordan_pair_deficit,
    fraction_sha256,
)
from passive_muon.specs import JORDAN_QUINTIC, zero_slope_gain_exact

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = (
    ".github/workflows/ci.yml",
    "scripts/certify_floored_repair.py",
    "src/passive_muon/floored_certificate.py",
    "src/passive_muon/normalizers.py",
    "src/passive_muon/operator.py",
    "src/passive_muon/orthogonalizers.py",
    "src/passive_muon/polynomials.py",
    "src/passive_muon/specs.py",
    "theory/floored_normalizer_certificate.md",
    "theory/claims.md",
    "tests/test_cli.py",
    "tests/test_floored_certificate.py",
    "tests/test_normalizers.py",
    "third_party/UPSTREAM_COMMITS.md",
    "pyproject.toml",
    "uv.lock",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--precision-bits",
        type=int,
        action="append",
        default=None,
        help="Arb precision; repeat for a cross-precision replay (default: 160 and 224)",
    )
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


def _run_git(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _source_snapshot() -> dict[str, str]:
    return {
        path: hashlib.sha256((REPOSITORY_ROOT / path).read_bytes()).hexdigest()
        for path in SOURCE_PATHS
    }


def build_payload(*, precisions: tuple[int, ...]) -> dict[str, Any]:
    if len(precisions) < 2:
        raise ValueError("at least two Arb precisions are required for the locked certificate")
    if len(set(precisions)) != len(precisions):
        raise ValueError("Arb precisions must be distinct")

    passes: list[dict[str, Any]] = []
    for precision in precisions:
        start = time.perf_counter()
        certificate = certify_jordan_derivative_range(precision_bits=precision)
        passes.append(
            {
                "precision_bits": precision,
                "initial_dyadic_power": certificate.initial_power,
                "configured_maximum_dyadic_power": certificate.configured_maximum_power,
                "maximum_dyadic_power": certificate.maximum_power,
                "leaf_count": certificate.leaf_count,
                "ordered_leaf_trace_sha256": certificate.trace_sha256,
                "elapsed_seconds": time.perf_counter() - start,
            }
        )

    trace_digests = {item["ordered_leaf_trace_sha256"] for item in passes}
    if len(trace_digests) != 1:
        raise AssertionError("precision passes produced different proof covers")

    pair_deficit = exact_jordan_pair_deficit()
    if pair_deficit <= JORDAN_PAIR_DEFICIT_LOWER:
        raise AssertionError("exact finite-pair lower certificate failed")
    upper = certified_dimension_uniform_deficit()
    if upper <= JORDAN_PAIR_DEFICIT_LOWER:
        raise AssertionError("deficit bracket is not ordered")

    zero_gain = zero_slope_gain_exact(JORDAN_QUINTIC, steps=5)
    relative_width = (upper - JORDAN_PAIR_DEFICIT_LOWER) / JORDAN_PAIR_DEFICIT_LOWER
    revision = _run_git("rev-parse", "HEAD")
    status = _run_git("status", "--porcelain")

    return {
        "schema_version": "passive-muon-floored-certificate-v1",
        "claim_scope": {
            "evidence_kind": (
                "rigorous_dimension_uniform_full_matrix_upper_certificate_"
                "and_exact_finite_pair_lower_witness"
            ),
            "matrix_domain": "R^(m x n) for every fixed finite positive m,n",
            "dimension_uniform": True,
            "arithmetic_model": "real arithmetic",
            "not_claimed": [
                "exact minimal conductance for each fixed matrix shape",
                "stability of momentum Muon training",
                "a BF16 backend theorem",
            ],
        },
        "operator": {
            "formula": "F_h,c(M)=H_h(M/max(c,||M||_F))",
            "certified_floor_c": "1",
            "normalization_rule": "fixed_frobenius_floor",
            "epsilon": "0",
            "orthogonalizer": JORDAN_QUINTIC.name,
            "scalar_composition": "h=q composed with itself 5 times",
            "stage_recurrence": ("x_next=a*x+b*x^3+c*x^5; d_next=(a+3*b*x^2+5*c*x^4)*d"),
            "coefficients_exact": {
                "a": "6889/2000",
                "b": "-191/40",
                "c": "4063/2000",
            },
            "steps": 5,
            "upstream": JORDAN_QUINTIC.source,
        },
        "deficit_definition": {
            "formula": ("sup_(A!=B) [-<F(A)-F(B),A-B>_F/||A-B||_F^2]_+"),
            "authority_at_floor_boundary": "global pairwise inequality",
        },
        "scalar_interval_certificate": {
            "method": (
                "adaptive dyadic cover of [0,1] with outward-rounded Arb balls; "
                "five-stage recurrence evaluated without polynomial expansion"
            ),
            "derivative_lower_exact": str(JORDAN_DERIVATIVE_LOWER),
            "derivative_upper_exact": str(JORDAN_DERIVATIVE_UPPER),
            "derivative_lower_decimal": _decimal(JORDAN_DERIVATIVE_LOWER),
            "derivative_upper_decimal": _decimal(JORDAN_DERIVATIVE_UPPER),
            "strict_enclosure": "lower < h'(s) < upper for every s in [0,1]",
            "passes": passes,
        },
        "full_matrix_reduction": {
            "tangent_modes": [
                "h'(sigma_i)",
                "(h(sigma_i)-h(sigma_j))/(sigma_i-sigma_j)",
                "(h(sigma_i)+h(sigma_j))/(sigma_i+sigma_j)",
                "h(sigma_i)/sigma_i for rectangular null-side modes",
            ],
            "mode_bound": "every mode is a secant average of h' on [-1,1]",
            "exterior_symmetric_jacobian": "(A*Q+Q*A)/(2*||M||_F)",
            "projection_envelope_formula": "-(b-a)^2/(8*(a+b))",
            "boundary_method": (
                "line integration for the pairwise inequality; Clarke hull "
                "{I-theta*u tensor u: 0<=theta<=1} as a supplementary check"
            ),
        },
        "exact_finite_pair_lower_witness": {
            "embedding": "rank-one scalar mode inside the unit Frobenius floor",
            "left_input_exact": str(JORDAN_PAIR_LEFT),
            "right_input_exact": str(JORDAN_PAIR_RIGHT),
            "left_input_decimal": _decimal(JORDAN_PAIR_LEFT),
            "right_input_decimal": _decimal(JORDAN_PAIR_RIGHT),
            "deficit_formula": "-(h(right)-h(left))/(right-left)",
            "deficit_exact_sha256": fraction_sha256(pair_deficit),
            "certified_deficit_strict_lower_exact": str(JORDAN_PAIR_DEFICIT_LOWER),
            "certified_deficit_strict_lower_decimal": _decimal(JORDAN_PAIR_DEFICIT_LOWER),
            "high_precision_decimal": _decimal(pair_deficit),
        },
        "dimension_uniform_upper_certificate": {
            "exact": str(upper),
            "decimal": _decimal(upper),
            "repair_for_general_floor": "rho=bar_delta_1/c",
            "monotonicity_guarantee": ("F_h,c(M)+(bar_delta_1/c)M is globally monotone"),
        },
        "certified_bracket_at_c_1": {
            "strict_lower_exact": str(JORDAN_PAIR_DEFICIT_LOWER),
            "upper_exact": str(upper),
            "strict_lower_decimal": _decimal(JORDAN_PAIR_DEFICIT_LOWER),
            "upper_decimal": _decimal(upper),
            "absolute_width_decimal": _decimal(upper - JORDAN_PAIR_DEFICIT_LOWER),
            "relative_width": _decimal(relative_width),
            "relative_width_percent": _decimal(100 * relative_width),
            "upper_over_zero_input_gain": _decimal(upper / zero_gain),
            "zero_input_gain_exact": str(zero_gain),
            "zero_input_gain_decimal": _decimal(zero_gain),
        },
        "scaling_and_stability": {
            "exact_global_scaling": "delta(F_h,c)=delta(F_h,1)/c",
            "strong_repair": "rho=bar_delta_1/c+mu, mu>0",
            "system": "dot(M)=-(F_h,c(M)+rho*M-b)",
            "guarantee": (
                "unique equilibrium and ||M(t)-M_star||_F <= exp(-mu*t)||M(0)-M_star||_F"
            ),
            "scope": "simplified continuous-time integrator feedback only",
        },
        "software": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "python_flint": flint.__version__,
            "flint": flint.__FLINT_VERSION__,
        },
        "hardware": {
            "platform": platform.platform(),
            "operating_system": platform.system(),
            "operating_system_release": platform.release(),
            "machine_architecture": platform.machine(),
            "processor": platform.processor() or None,
            "logical_cpu_count": os.cpu_count(),
            "byte_order": sys.byteorder,
        },
        "git": {
            "sha": revision.stdout.strip() if revision.returncode == 0 else None,
            "dirty": bool(status.stdout.strip()),
        },
        "experiment_provenance": {
            "seed": None,
            "randomness": "none; deterministic exact and interval arithmetic",
            "command": (
                "uv run --locked python scripts/certify_floored_repair.py "
                "--output results/summaries/floored_repair_certificate.json"
            ),
            "source_snapshot": _source_snapshot(),
        },
    }


def main() -> int:
    args = parse_args()
    precisions = tuple(args.precision_bits or (160, 224))
    payload = build_payload(precisions=precisions)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
