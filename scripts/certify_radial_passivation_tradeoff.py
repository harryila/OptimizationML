#!/usr/bin/env python3
"""Generate the exact/Arb P13 radial-passivation tradeoff certificate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Any

import flint
from flint import arb, ctx, fmpq

from passive_muon.additive_epsilon_deficit import DEPLOYED_EPSILON
from passive_muon.radial_passivation_tradeoff import (
    LOCKED_BETA,
    LOCKED_LEARNING_RATE,
    P11_SIGNAL_NORM_BOUND,
    RADIAL_PASSIVATION_SCHEMA_VERSION,
    SWITCH_NORMALIZED_RADIUS,
    SWITCH_RAW_RATIO,
    TAIL_DERIVATIVE_LOWER,
    TAIL_PROJECTION_LOWER,
    UNIT_DEFICIT_UPPER,
    constant_repair_magnitude,
    corrected_origin_slope,
    ema_nesterov_explicit_step_control,
    exact_certificate_checks,
    radial_integral_terms,
    repair_lipschitz_constant,
    repair_magnitude_ball,
    universal_lipschitz_strict_lower,
)
from passive_muon.specs import JORDAN_QUINTIC
from passive_muon.upstream_momentum import (
    PINNED_KELLER_JORDAN_MUON_REVISION,
    PINNED_MUON_PY_SHA256,
)

ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = "results/summaries/radial_passivation_tradeoff_certificate.json"

SOURCE_PATHS = (
    ".github/workflows/ci.yml",
    ".github/workflows/p12-additive-epsilon-deficit.yml",
    ".github/workflows/p13-radial-passivation-tradeoff.yml",
    "scripts/certify_radial_passivation_tradeoff.py",
    "scripts/reconstruct_radial_passivation_tradeoff.py",
    "scripts/reconstruct_additive_epsilon_deficit.py",
    "scripts/reconstruct_outer_loop_roundoff.py",
    "src/passive_muon/radial_passivation_tradeoff.py",
    "src/passive_muon/additive_epsilon_deficit.py",
    "src/passive_muon/floored_certificate.py",
    "src/passive_muon/specs.py",
    "src/passive_muon/upstream_momentum.py",
    "tests/test_radial_passivation_tradeoff.py",
    "tests/test_radial_passivation_tradeoff_cli.py",
    "tests/test_radial_passivation_tradeoff_reconstruction.py",
    "tests/test_radial_passivation_tradeoff_result_manifest.py",
    "tests/test_additive_epsilon_deficit_result_manifest.py",
    "tests/test_current_research_index.py",
    "tests/test_outer_loop_roundoff_cli.py",
    "tests/test_result_manifests.py",
    "theory/radial_passivation_tradeoff.md",
    "theory/audits/P13_RADIAL_PASSIVATION_HUMAN_PROOF_AUDIT.md",
    "results/summaries/P13_RADIAL_PASSIVATION_RESULTS.md",
    "third_party/UPSTREAM_COMMITS.md",
    "pyproject.toml",
    "uv.lock",
)

PRIOR_ARTIFACT_SHA256 = {
    "results/summaries/additive_epsilon_deficit_certificate.json": (
        "e764a60723543fa0342d0c178886fe57a91b5f4511739cd69d8f2e07dc21df2c"
    ),
    "results/summaries/implementation_margin_certificate.json": (
        "ad74050d66f711d2bb8e3e37f80fc29f4d6162419a2f5224931e1ca7fd2e536f"
    ),
}
P12_PAIR_EXACT_DEFICIT_SHA256 = "de076057ce58ad174a52a8e0e492f7efffe0488bbca19c03cd1ce0fee75a7336"

MAGNITUDE_GUARDS = {
    Fraction(1): (
        Fraction(2_571_857_826_470_212_145, 10**15),
        Fraction(2_571_857_826_470_212_146, 10**15),
    ),
    P11_SIGNAL_NORM_BOUND: (
        Fraction(3_086_959_580_254_301_425, 10**15),
        Fraction(3_086_959_580_254_301_426, 10**15),
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


def _arb_fraction(value: Fraction) -> arb:
    return arb(fmpq(value.numerator, value.denominator))


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


def _arb_pass(precision_bits: int) -> dict[str, Any]:
    if precision_bits < 80:
        raise ValueError("precision_bits must be at least 80")
    evaluations = []
    with ctx.workprec(precision_bits):
        for raw_radius, (guard_lower, guard_upper) in MAGNITUDE_GUARDS.items():
            value = repair_magnitude_ball(
                raw_radius,
                DEPLOYED_EPSILON,
                precision_bits=precision_bits,
            )
            passed = value > _arb_fraction(guard_lower) and value < _arb_fraction(guard_upper)
            if not passed:
                raise AssertionError(
                    f"Arb magnitude at radius {raw_radius} escaped its rational guard"
                )
            evaluations.append(
                {
                    "raw_radius_exact": str(raw_radius),
                    "arb_enclosure": str(value),
                    "strictly_inside_locked_guard": True,
                }
            )
    return {"precision_bits": precision_bits, "evaluations": evaluations}


def _magnitude_evaluation(raw_radius: Fraction) -> dict[str, Any]:
    guard_lower, guard_upper = MAGNITUDE_GUARDS[raw_radius]
    z = raw_radius / DEPLOYED_EPSILON
    terms = radial_integral_terms(z)
    constant = constant_repair_magnitude(raw_radius, DEPLOYED_EPSILON)
    return {
        "raw_radius": _fraction(raw_radius),
        "z": _fraction(z),
        "radial_output_guard": {
            "lower_exact": str(guard_lower),
            "upper_exact": str(guard_upper),
            "lower_decimal": _decimal(guard_lower),
            "upper_decimal": _decimal(guard_upper),
        },
        "closed_form_terms": {
            "rational_part": _fraction(terms.rational_part),
            "logarithm_coefficient": _fraction(terms.logarithm_coefficient),
            "logarithm_argument": _fraction(terms.logarithm_argument),
        },
        "independent_fraction_enclosure": {
            "method": "96-term exact Fraction atanh series after binary range reduction",
            "width_strict_upper_exact": "1/1000000000000000000000000000000",
        },
        "constant_repair_output": _fraction(constant),
        "constant_over_radial_strict_lower": _fraction(constant / guard_upper),
        "radial_over_polynomial_global_gain_strict_upper": _fraction(
            guard_upper / Fraction(4_848_763, 10_000)
        ),
    }


def _explicit_step_payload() -> dict[str, Any]:
    control = ema_nesterov_explicit_step_control()
    unit_origin = corrected_origin_slope(Fraction(1))
    critical_theta = control.schur_gain_threshold
    full_eta_critical = critical_theta * DEPLOYED_EPSILON / unit_origin
    repair_eta_critical = critical_theta * DEPLOYED_EPSILON / UNIT_DEFICIT_UPPER
    repair_theta = LOCKED_LEARNING_RATE * UNIT_DEFICIT_UPPER / DEPLOYED_EPSILON
    repair_jury = 2 * (1 + LOCKED_BETA) - repair_theta * (1 - LOCKED_BETA) * (1 + 2 * LOCKED_BETA)
    w0 = Fraction(1, 389_025_000)
    signal = (1 - LOCKED_BETA**2) * w0
    one_step_ratio = LOCKED_LEARNING_RATE * UNIT_DEFICIT_UPPER / (399 * w0)
    exact = {
        "beta": LOCKED_BETA,
        "eta": LOCKED_LEARNING_RATE,
        "epsilon": DEPLOYED_EPSILON,
        "origin_polynomial_gain_unit": JORDAN_QUINTIC.fractions()[0] ** 5,
        "combined_origin_gain_unit": unit_origin,
        "critical_theta": critical_theta,
        "repair_only_eta_critical": repair_eta_critical,
        "full_repaired_eta_critical": full_eta_critical,
        "repair_only_jury_margin": repair_jury,
        "full_repaired_jury_margin": control.second_jury_margin,
        "one_step_w0": w0,
        "one_step_signal": signal,
        "one_step_normalized_signal": signal / (signal + DEPLOYED_EPSILON),
        "one_step_repair_ratio_lower": one_step_ratio,
        "one_step_objective_growth_strict_lower": (one_step_ratio - 1) ** 2,
    }
    return {key: _fraction(value) for key, value in exact.items()}


def build_payload(*, precisions: tuple[int, ...]) -> dict[str, Any]:
    if len(precisions) < 2 or len(set(precisions)) != len(precisions):
        raise ValueError("at least two distinct Arb precisions are required")
    exact_checks = exact_certificate_checks()
    arb_passes = [_arb_pass(precision) for precision in precisions]
    if not all(exact_checks.values()):
        failed = [name for name, passed in exact_checks.items() if not passed]
        raise AssertionError(f"exact P13 checks failed: {failed}")

    relative_gap = (
        UNIT_DEFICIT_UPPER - universal_lipschitz_strict_lower(Fraction(1))
    ) / universal_lipschitz_strict_lower(Fraction(1))
    tail_derivative_constant = 2 * TAIL_DERIVATIVE_LOWER - TAIL_PROJECTION_LOWER
    explicit_step = _explicit_step_payload()
    analytic_checks = {
        **exact_checks,
        "tail_derivative_numerator_negative_for_all_z_nonnegative": (
            tail_derivative_constant < 0 and TAIL_PROJECTION_LOWER < 0
        ),
        "repair_only_explicit_step_jury_margin_negative": (
            Fraction(explicit_step["repair_only_jury_margin"]["exact"]) < 0
        ),
        "full_repaired_explicit_step_jury_margin_negative": (
            Fraction(explicit_step["full_repaired_jury_margin"]["exact"]) < 0
        ),
        "all_arb_magnitude_guards_pass": all(
            row["strictly_inside_locked_guard"]
            for replay in arb_passes
            for row in replay["evaluations"]
        ),
    }

    return {
        "schema_version": RADIAL_PASSIVATION_SCHEMA_VERSION,
        "canonical_result_path": RESULT_PATH,
        "claim_scope": {
            "evidence_kind": (
                "exact full-matrix radial passivation theorem, two-precision Arb magnitude "
                "evaluation, and exact rank-two universal stiffness lower bound"
            ),
            "matrix_domain": "R^(m x n) for every fixed finite positive m,n",
            "lower_tradeoff_shape_condition": "min(m,n)>=2",
            "arithmetic_model": "exact real arithmetic",
            "normalization": "current Frobenius norm plus additive epsilon; no max floor",
            "epsilon_domain": "every real epsilon>0",
            "guarantee": "E_h,epsilon+G_epsilon is globally pairwise monotone",
            "not_claimed": [
                "strong monotonicity",
                "stability of an explicit optimizer at the prior 1/32000 step",
                "a theorem for the discontinuous BF16 implementation",
                "a bounded radial correction or removal of 1/epsilon stiffness",
                "minimal Lipschitz constant in a fixed matrix shape",
                "propagation of this repair through P7--P11 or neural-network convergence",
            ],
        },
        "operator": {
            "base_formula": "E_h,eps(M)=H_h(M/(||M||_F+eps))",
            "repair_formula": "G_eps(0)=0; G_eps(M)=p(||M||_F/eps)M/||M||_F",
            "orthogonalizer": JORDAN_QUINTIC.name,
            "scalar_composition": "h=q composed with itself exactly 5 times",
            "coefficients_exact": {
                "a": "6889/2000",
                "b": "-191/40",
                "c": "4063/2000",
            },
            "steps": 5,
            "domain": "every fixed finite real rectangular matrix space",
        },
        "radial_majorant": {
            "unit_deficit_upper": _fraction(UNIT_DEFICIT_UPPER),
            "final_band_derivative_lower": _fraction(TAIL_DERIVATIVE_LOWER),
            "final_band_projection_lower": _fraction(TAIL_PROJECTION_LOWER),
            "pair_strict_lower": _fraction(universal_lipschitz_strict_lower(Fraction(1))),
            "z0": _fraction(SWITCH_RAW_RATIO),
            "t0": _fraction(SWITCH_NORMALIZED_RADIUS),
            "formula": {
                "inner": "d_hat(z)=U for 0<=z<=z0",
                "tail": "d_hat(z)=-(a+Gamma*z)/(1+z)^2 for z>=z0",
                "integral": "p(z)=integral_0^z d_hat(u)du",
            },
            "closed_form": {
                "tail_majorant": "-(a+Gamma*z)/(1+z)^2",
                "tail_integral": ("U*z0-Gamma*log((1+z)/(1+z0))+(a-Gamma)*(1/(1+z)-1/(1+z0))"),
                "join_value": str(UNIT_DEFICIT_UPPER),
                "tail_derivative_constant": str(tail_derivative_constant),
            },
            "exact_identities": {
                "t0_equals_z0_over_one_plus_z0": True,
                "tail_value_at_z0_equals_U": True,
                "a_minus_Gamma": _fraction(TAIL_DERIVATIVE_LOWER - TAIL_PROJECTION_LOWER),
                "minus_Gamma": _fraction(-TAIL_PROJECTION_LOWER),
            },
            "shape": "positive, continuous, nonincreasing; tail is strictly decreasing",
        },
        "full_matrix_monotonicity": {
            "p12_pointwise_bound": (
                "Sym DE_h,eps(M) >= -d_hat(||M||_F/eps)/eps * I on the complete "
                "Frobenius tangent space"
            ),
            "repair_derivative": (
                "DG=(d_hat(z)/eps)P_M+(p(z)/(eps*z))(I-P_M), with the continuous "
                "origin value U/eps*I"
            ),
            "average_inequality": "d_hat(z)<=p(z)/z<=U",
            "jacobian_sum": "Sym D(E_h,eps+G_eps)>=0 everywhere",
            "pairwise_promotion": (
                "global C1 regularity and line-segment integration prove the authoritative "
                "pairwise monotonicity inequality"
            ),
            "dimension_uniform": True,
            "not_a_diagonal_certificate": True,
        },
        "stiffness_tradeoff": {
            "constructed_lipschitz": {
                "formula": "Lip(G_eps)=U/eps",
                "unit_constant": _fraction(repair_lipschitz_constant(Fraction(1))),
                "equality_reason": "the upper derivative bound is attained on 0<=z<=z0",
            },
            "universal_lower_bound": {
                "formula": "Lip(C_eps)>pair_strict_lower/eps",
                "unit_strict_lower": _fraction(universal_lipschitz_strict_lower(Fraction(1))),
                "correction_scope": "every globally Lipschitz correction, not only radial maps",
                "proof": "locked P12 pair, monotonicity, then Cauchy--Schwarz",
                "shape_condition": "min(m,n)>=2",
                "exact_pair_deficit_sha256": P12_PAIR_EXACT_DEFICIT_SHA256,
            },
            "relative_gap": _fraction(relative_gap),
            "relative_gap_percent": _fraction(100 * relative_gap),
            "interpretation": (
                "output magnitude becomes logarithmic, but differential stiffness remains "
                "order 1/epsilon and the construction is near the universal pair lower bound"
            ),
        },
        "arb_magnitude_certificate": {
            "method": "outward-rounded Arb evaluation of the exact rational/logarithmic form",
            "cross_precision_guard_match": True,
            "passes": arb_passes,
        },
        "magnitude_evaluations": [
            _magnitude_evaluation(raw_radius) for raw_radius in MAGNITUDE_GUARDS
        ],
        "p11_signal_benchmark": {
            **_fraction(P11_SIGNAL_NORM_BOUND),
            "meaning": "exact P11 V<=1 signal guard, not the rounded display 25.2335",
        },
        "scaling_and_asymptotics": {
            "repair_scaling": "G_eps(eps*X)=G_1(X)",
            "fixed_raw_epsilon_limit": "p(||M||/eps)=-Gamma*log(||M||/eps)+O(1)",
            "large_z_derivative": "d_hat(z)->0 and p(z)/z->0",
            "output_growth": "logarithmic and unbounded",
        },
        "explicit_step_negative_control": explicit_step,
        "explicit_step_interpretation": {
            "loop": (
                "scalar curvature-one quadratic with beta=19/20 and the pinned EMA state "
                "then Nesterov signal ordering"
            ),
            "jury_condition": (
                "theta=eta*kappa*(h'(0)+U)/eps must be below 2(1+beta)/((1-beta)(1+2beta))"
            ),
            "repair_only_control": (
                "the same failed Jury check with U alone isolates stiffness of G_eps"
            ),
            "one_step_control": (
                "m0=0,w0=1/389025000 gives s1=eps/399 inside the linear repair core; "
                "the repair alone multiplies the step by more than 4800 times w0"
            ),
            "polynomial_sign_check": (
                "the Jordan quintic preserves positive scalar sign because the quadratic "
                "factor has negative discriminant -2594691/500000 and positive leading term"
            ),
            "conclusion": (
                "the prior eta=1/32000 is locally unstable; this is an exact scalar "
                "negative control, not a global training claim"
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
        },
        "real_arithmetic_vs_bf16": {
            "theorem": "continuous exact-real surrogate only",
            "deployed_map": "backend-specific and discontinuous because of BF16 casts",
            "qualification": (
                "the upstream pin supports formula provenance only; P13 does not certify "
                "literal upstream BF16, its rounded coefficients, or a BF16 radial repair"
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
            "randomness": "none; exact rational and outward-rounded Arb arithmetic",
            "command": (
                "uv run --locked python scripts/certify_radial_passivation_tradeoff.py "
                "--output results/summaries/radial_passivation_tradeoff_certificate.json"
            ),
            "source_snapshot": _source_snapshot(),
        },
        "audit": {
            "checks": analytic_checks,
            "all_exact_and_interval_checks_passed": all(analytic_checks.values()),
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
