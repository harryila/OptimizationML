#!/usr/bin/env python3
"""Generate the rigorous additive-epsilon full-matrix deficit certificate."""

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

from passive_muon.additive_epsilon_deficit import (
    ADDITIVE_EPSILON_SCHEMA_VERSION,
    DEPLOYED_EPSILON,
    PAIR_DEFICIT_STRICT_LOWER,
    PAIR_DIRECTION_HIGH,
    PAIR_DIRECTION_LOW,
    PAIR_NORMALIZED_RADIUS,
    PAIR_RAW_RADIUS,
    RADIUS_BANDS,
    certified_repair_conductance,
    certified_unit_epsilon_deficit_upper,
    certify_additive_epsilon_intervals,
    exact_rank_two_pair_deficit,
    exact_rank_two_pair_deficit_sha256,
    radius_band_bound,
    scale_unit_deficit,
    zero_input_derivative_gain,
)
from passive_muon.floored_certificate import (
    JORDAN_DERIVATIVE_UPPER,
)
from passive_muon.floored_certificate import (
    certified_dimension_uniform_deficit as floored_deficit_upper,
)
from passive_muon.specs import JORDAN_QUINTIC
from passive_muon.upstream_momentum import (
    PINNED_KELLER_JORDAN_MUON_REVISION,
    PINNED_MUON_PY_SHA256,
)

if hasattr(sys, "set_int_max_str_digits"):
    # The exact rational pair quotient has roughly 50,000 decimal digits.
    sys.set_int_max_str_digits(0)

ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = "results/summaries/additive_epsilon_deficit_certificate.json"
SOURCE_PATHS = (
    ".github/workflows/p12-additive-epsilon-deficit.yml",
    "README.md",
    "TASKS.md",
    "experiments/README.md",
    "results/README.md",
    "results/summaries/RESULTS.md",
    "scripts/certify_additive_epsilon_deficit.py",
    "scripts/reconstruct_additive_epsilon_deficit.py",
    "src/passive_muon/additive_epsilon_deficit.py",
    "src/passive_muon/floored_certificate.py",
    "src/passive_muon/specs.py",
    "src/passive_muon/upstream_momentum.py",
    "tests/test_additive_epsilon_deficit.py",
    "tests/test_additive_epsilon_deficit_cli.py",
    "tests/test_additive_epsilon_deficit_reconstruction.py",
    "tests/test_additive_epsilon_deficit_result_manifest.py",
    "tests/test_current_research_index.py",
    "theory/additive_epsilon_deficit_certificate.md",
    "theory/audits/P12_ADDITIVE_EPSILON_HUMAN_PROOF_AUDIT.md",
    "theory/claims.md",
    "results/summaries/P12_ADDITIVE_EPSILON_RESULTS.md",
    "third_party/UPSTREAM_COMMITS.md",
    "pyproject.toml",
    "uv.lock",
)

PRIOR_ARTIFACT_SHA256 = {
    "results/summaries/floored_repair_certificate.json": (
        "244e9abd35f86451d8cd5d7f48b59abe961313ca5b633fc4a6365d396997ade9"
    ),
    "results/summaries/deficit_audit.json": (
        "cc5ab80928a66fb12f9a68448ef0438a393959f88fd455a66db81381d077018a"
    ),
    "results/summaries/bf16_witness.json": (
        "b5b1c0cba2326b3c07b06c3dc85dceb4e10a31ab57318d5b348934c4aef0694c"
    ),
    "results/summaries/implementation_margin_certificate.json": (
        "ad74050d66f711d2bb8e3e37f80fc29f4d6162419a2f5224931e1ca7fd2e536f"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--precision-bits",
        type=int,
        action="append",
        default=None,
        help="Arb precision; repeat for cross-precision replay (default: 160 and 224)",
    )
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    return parser.parse_args()


def _decimal(value: Fraction, *, digits: int = 60) -> str:
    with localcontext() as context:
        context.prec = digits
        return str(Decimal(value.numerator) / Decimal(value.denominator))


def _fraction(value: Fraction) -> dict[str, str]:
    return {"exact": str(value), "decimal": _decimal(value)}


def _sha256(path: str) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def _source_snapshot() -> dict[str, str]:
    return {path: _sha256(path) for path in SOURCE_PATHS}


def _git_output(*arguments: str) -> str | None:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
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


def _verify_prior_artifacts() -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for path, expected in PRIOR_ARTIFACT_SHA256.items():
        observed = _sha256(path)
        if observed != expected:
            raise AssertionError(f"prior artifact hash mismatch: {path}")
        records[path] = {"sha256": observed, "matched": True}
    return records


def _interval_pass(precision_bits: int) -> dict[str, Any]:
    started = time.perf_counter()
    certificate = certify_additive_epsilon_intervals(precision_bits=precision_bits)
    return {
        "precision_bits": precision_bits,
        "elapsed_seconds": time.perf_counter() - started,
        "global_derivative_cover": {
            "initial_dyadic_power": certificate.global_range.initial_power,
            "configured_maximum_dyadic_power": (certificate.global_range.configured_maximum_power),
            "leaf_count": certificate.global_range.leaf_count,
            "maximum_dyadic_power": certificate.global_range.maximum_power,
            "ordered_leaf_trace_sha256": certificate.global_range.trace_sha256,
        },
        "prefix_derivative_covers": [
            {
                "endpoint_exact": str(item.endpoint),
                "derivative_strict_lower_exact": str(item.derivative_lower),
                "initial_dyadic_power": item.initial_power,
                "configured_maximum_dyadic_power": item.configured_maximum_power,
                "leaf_count": item.leaf_count,
                "maximum_dyadic_power": item.maximum_power,
                "ordered_leaf_trace_sha256": item.trace_sha256,
            }
            for item in certificate.prefix_ranges
        ],
    }


def build_payload(*, precisions: tuple[int, ...]) -> dict[str, Any]:
    if len(precisions) < 2 or len(set(precisions)) != len(precisions):
        raise ValueError("at least two distinct Arb precisions are required")

    passes = [_interval_pass(precision) for precision in precisions]
    cover_signatures = {
        json.dumps(
            {
                "global": item["global_derivative_cover"],
                "prefix": item["prefix_derivative_covers"],
            },
            sort_keys=True,
        )
        for item in passes
    }
    if len(cover_signatures) != 1:
        raise AssertionError("Arb precision passes produced different proof covers")

    pair_deficit = exact_rank_two_pair_deficit()
    upper = certified_unit_epsilon_deficit_upper()
    if not PAIR_DEFICIT_STRICT_LOWER < pair_deficit <= upper:
        raise AssertionError("certified unit-epsilon bracket is not ordered")

    bands = [radius_band_bound(band) for band in RADIUS_BANDS]
    active = max(bands, key=lambda item: item.deficit_upper)
    floor_upper = floored_deficit_upper()
    deployed_upper = scale_unit_deficit(upper, DEPLOYED_EPSILON)
    deployed_lower = scale_unit_deficit(PAIR_DEFICIT_STRICT_LOWER, DEPLOYED_EPSILON)
    zero_gain_unit = zero_input_derivative_gain(Fraction(1))
    relative_width = (upper - PAIR_DEFICIT_STRICT_LOWER) / PAIR_DEFICIT_STRICT_LOWER

    return {
        "schema_version": ADDITIVE_EPSILON_SCHEMA_VERSION,
        "claim_scope": {
            "evidence_kind": (
                "dimension_uniform_full_rectangular_global_upper_certificate_"
                "and_exact_rational_finite_pair_lower_witness"
            ),
            "matrix_domain": "R^(m x n) for every fixed finite positive m,n",
            "dimension_uniform_upper": True,
            "lower_witness_shape": [2, 2],
            "arithmetic_model": "exact real arithmetic",
            "not_claimed": [
                "the exact deficit or minimal repair for any fixed shape",
                "continuity or monotonicity of the deployed BF16 map",
                "stability of unrepaired or literal upstream Muon",
            ],
        },
        "operator": {
            "formula": "E_h,eps(M)=H_h(M/(||M||_F+eps))",
            "normalization_rule": "additive Frobenius epsilon",
            "epsilon_domain": "every real eps>0",
            "orthogonalizer": JORDAN_QUINTIC.name,
            "scalar_composition": "h=q composed with itself exactly 5 times",
            "coefficients_exact": {
                "a": "6889/2000",
                "b": "-191/40",
                "c": "4063/2000",
            },
            "steps": 5,
            "source": JORDAN_QUINTIC.source,
        },
        "deficit_definition": {
            "formula": "sup_(A!=B)[-<E(A)-E(B),A-B>_F/||A-B||_F^2]_+",
            "authority": "global pairwise deficit",
            "exact_scaling": "delta(E_h,eps)=delta(E_h,1)/eps",
            "scaling_identity": "E_h,eps(eps*X)=E_h,1(X)",
            "restricted_domain_caveat": "a restricted domain must scale with eps",
        },
        "normalizer_derivative": {
            "normalized_radius": "t=||M||_F/(eps+||M||_F)",
            "nonzero_input": "DN_eps=(1-t)/eps*(I-t*P_M)",
            "origin": "DN_eps(0)=I/eps; the whole quotient is Frechet differentiable",
            "outer_factor": "(1-t)/eps",
        },
        "full_rectangular_reduction": {
            "spectral_derivative_modes": [
                "h'(sigma_i)",
                "(h(sigma_i)-h(sigma_j))/(sigma_i-sigma_j)",
                "(h(sigma_i)+h(sigma_j))/(sigma_i+sigma_j)",
                "h(sigma_i)/sigma_i on rectangular null-side modes",
            ],
            "radial_location": "P_M acts only in the diagonal singular-value block",
            "mode_envelope": (
                "every mode is a secant average of even h' on [-t,t], hence lies "
                "between the certified prefix lower bound and global upper bound"
            ),
            "decomposition": "I-tP=(1-t)I+tQ with Q=I-P",
            "projection_lemma": "Sym(AQ)>=Gamma(a,b)I",
            "band_bound": ("Sym(DE)>=((1-t)/eps)*((1-t)*a_band+t*Gamma(a_band,b))*I"),
            "pairwise_promotion": (
                "E is globally C1 and Lipschitz; integrate the symmetric Jacobian "
                "bound along line segments"
            ),
        },
        "arb_interval_certificate": {
            "method": (
                "outward-rounded Arb evaluation of the five-stage value/derivative "
                "recurrence on adaptive scaled-dyadic covers"
            ),
            "global_derivative_strict_upper": _fraction(JORDAN_DERIVATIVE_UPPER),
            "cross_precision_cover_match": True,
            "passes": passes,
        },
        "radius_band_envelope": [
            {
                "left": _fraction(item.band.left),
                "right": _fraction(item.band.right),
                "prefix_derivative_strict_lower": _fraction(item.band.derivative_lower),
                "projection_lower": _fraction(item.projection_lower),
                "maximizing_radius": _fraction(item.maximizing_radius),
                "deficit_upper": _fraction(item.deficit_upper),
                "active": item == active,
            }
            for item in bands
        ],
        "exact_finite_pair_lower_witness": {
            "shape": [2, 2],
            "construction": ("M_left=r*diag(u_low,u_high), M_right=r*diag(u_high,u_low)"),
            "normalized_radius": _fraction(PAIR_NORMALIZED_RADIUS),
            "raw_radius_at_eps_1": _fraction(PAIR_RAW_RADIUS),
            "unit_direction_low": _fraction(PAIR_DIRECTION_LOW),
            "unit_direction_high": _fraction(PAIR_DIRECTION_HIGH),
            "unit_direction_check": str(PAIR_DIRECTION_LOW**2 + PAIR_DIRECTION_HIGH**2),
            "exact_deficit_sha256": exact_rank_two_pair_deficit_sha256(),
            "exact_numerator_decimal_digits": len(str(pair_deficit.numerator)),
            "exact_denominator_decimal_digits": len(str(pair_deficit.denominator)),
            "high_precision_decimal": _decimal(pair_deficit),
            "strict_lower": _fraction(PAIR_DEFICIT_STRICT_LOWER),
            "embedding": "zero-pad into every shape with min(m,n)>=2",
        },
        "certified_bracket_at_epsilon_1": {
            "strict_lower": _fraction(PAIR_DEFICIT_STRICT_LOWER),
            "upper": _fraction(upper),
            "absolute_width": _fraction(upper - PAIR_DEFICIT_STRICT_LOWER),
            "relative_width": _fraction(relative_width),
            "relative_width_percent": _fraction(100 * relative_width),
        },
        "repair": {
            "sufficient_formula": "rho_eps=bar_delta_1/eps",
            "minimality_qualification": (
                "the sufficient upper endpoint is not claimed minimal; in a shape "
                "containing the pair, every global repair must strictly exceed the "
                "locked strict lower endpoint"
            ),
            "epsilon_1": _fraction(upper),
            "deployed_epsilon": _fraction(DEPLOYED_EPSILON),
            "deployed_sufficient": _fraction(deployed_upper),
            "deployed_necessary_strict_lower": _fraction(deployed_lower),
            "max_floor_c_1_sufficient": _fraction(floor_upper),
            "deployed_sufficient_over_floor_c_1": _fraction(deployed_upper / floor_upper),
        },
        "usefulness_at_deployed_epsilon": {
            "classification": "catastrophically large as a constant global repair",
            "unit_input_repair_component_norm": _fraction(deployed_upper),
            "base_operator_global_norm_upper": _fraction(JORDAN_DERIVATIVE_UPPER),
            "necessary_repair_over_base_upper": _fraction(deployed_lower / JORDAN_DERIVATIVE_UPPER),
            "sufficient_repair_over_base_upper": _fraction(
                deployed_upper / JORDAN_DERIVATIVE_UPPER
            ),
            "reason": (
                "for ||M||_F=1, ||H_h(N_eps(M))||_F<="
                "sup|h'|*||N_eps(M)||_F<484.8763, while any global constant "
                "repair exceeds 1.581e9 in norm"
            ),
        },
        "epsilon_evaluations": [
            {
                "epsilon": _fraction(epsilon),
                "strict_lower": _fraction(scale_unit_deficit(PAIR_DEFICIT_STRICT_LOWER, epsilon)),
                "upper_and_sufficient_repair": _fraction(certified_repair_conductance(epsilon)),
                "zero_input_derivative_gain": _fraction(zero_input_derivative_gain(epsilon)),
            }
            for epsilon in (
                Fraction(1),
                Fraction(1, 10),
                Fraction(1, 1_000),
                DEPLOYED_EPSILON,
            )
        ],
        "negative_controls_and_boundaries": {
            "affine_identity_control": (
                "for h(s)=s, N_eps is monotone: tangential eigenvalue 1/(r+eps) "
                "and radial eigenvalue eps/(r+eps)^2"
            ),
            "origin_control": _fraction(zero_gain_unit),
            "infinite_radius_limit": "t approaches 1 and the certified derivative bound tends to 0",
            "under_repair_control": (
                "any rho<=strict finite-pair lower endpoint fails monotonicity on the locked pair"
            ),
        },
        "upstream_formula_provenance": {
            "repository": "https://github.com/KellerJordan/Muon",
            "revision": PINNED_KELLER_JORDAN_MUON_REVISION,
            "audited_file": "muon.py",
            "audited_file_sha256": PINNED_MUON_PY_SHA256,
            "formula": "X/(X.norm(dim=(-2,-1),keepdim=True)+1e-7)",
            "epsilon_placement": "added to the BF16 Frobenius norm before division",
            "operation_order": (
                "cast input to BF16; orient once; norm; add Python-float epsilon; "
                "divide; run five BF16 Jordan stages; orient back"
            ),
            "coefficient_order": "A=X@X.T; B=b*A+(c*A)@A; X=a*X+B@X",
            "provenance_file": "third_party/UPSTREAM_COMMITS.md",
        },
        "real_arithmetic_vs_bf16": {
            "theorem": "continuous exact-real surrogate only",
            "deployed_map": "backend-specific and discontinuous because of BF16 casts",
            "separation": (
                "the upstream pin supports formula provenance; it does not transfer this "
                "Jacobian or repair theorem to the BF16 implementation"
            ),
            "recorded_backend_witness_sha256": PRIOR_ARTIFACT_SHA256[
                "results/summaries/bf16_witness.json"
            ],
            "recorded_epsilon_absorption": (
                "the pinned CPU BF16 witness records epsilon being absorbed in denominator rounding"
            ),
        },
        "prior_artifacts": _verify_prior_artifacts(),
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
        "git": _git_state(),
        "proof_replay_provenance": {
            "seed": None,
            "randomness": "none; exact rational and outward-rounded interval arithmetic",
            "command": (
                "uv run --locked python scripts/certify_additive_epsilon_deficit.py "
                "--output results/summaries/additive_epsilon_deficit_certificate.json"
            ),
            "source_snapshot": _source_snapshot(),
        },
        "audit": {
            "all_interval_checks_passed": True,
            "all_exact_checks_passed": True,
            "human_proof_audit": "pending; packet committed but unsigned",
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
