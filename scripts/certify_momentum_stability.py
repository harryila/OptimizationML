#!/usr/bin/env python3
"""Replay the exact momentum IQC certificate and matched CPU examples."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from passive_muon.floored_certificate import JORDAN_DERIVATIVE_UPPER
from passive_muon.momentum_experiment import (
    MomentumTrajectory,
    period_residual,
    run_rank_one_momentum_trajectory,
)
from passive_muon.momentum_iqc import (
    audit_momentum_certificate,
    closed_form_factor_matrix,
    closed_form_sector_certificate,
    closed_form_sector_lmi_matrix,
    fraction_matrix_to_strings,
    locked_momentum_certificate,
    repaired_zero_input_gain,
    scalar_momentum_jury_margin,
    sector_alpha_limit,
)
from passive_muon.specs import JORDAN_QUINTIC, zero_slope_gain_exact

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = (
    ".github/workflows/ci.yml",
    "README.md",
    "experiments/README.md",
    "results/README.md",
    "results/summaries/RESULTS.md",
    "scripts/certify_momentum_stability.py",
    "src/passive_muon/floored_certificate.py",
    "src/passive_muon/momentum_experiment.py",
    "src/passive_muon/momentum_iqc.py",
    "src/passive_muon/normalizers.py",
    "src/passive_muon/operator.py",
    "src/passive_muon/orthogonalizers.py",
    "src/passive_muon/polynomials.py",
    "src/passive_muon/specs.py",
    "theory/claims.md",
    "theory/momentum_iqc_certificate.md",
    "tests/test_cli.py",
    "tests/test_momentum_iqc.py",
    "tests/test_result_manifests.py",
    "third_party/UPSTREAM_COMMITS.md",
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


def _trajectory_payload(trajectory: MomentumTrajectory) -> dict[str, Any]:
    payload = asdict(trajectory)
    payload["tail"] = [
        {"position": position, "momentum": momentum} for position, momentum in trajectory.tail
    ]
    return payload


def _complex_sector_radius(*, alpha: float, beta: float, nu: float) -> float:
    slope = complex(nu, np.sqrt(1 - nu**2))
    roots = np.roots([1.0, -(1 + beta - alpha * slope), beta])
    return float(np.max(np.abs(roots)))


def build_payload() -> dict[str, Any]:
    certificate = locked_momentum_certificate()
    audit = audit_momentum_certificate(certificate)
    if not audit.certified:
        raise AssertionError("exact rational momentum LMI certificate failed")

    alpha_limit = sector_alpha_limit(certificate.beta, certificate.normalized_strongness)
    if certificate.dimensionless_step >= alpha_limit:
        raise AssertionError("locked point does not lie inside the analytic sector region")
    closed_form = closed_form_sector_certificate(
        beta=certificate.beta,
        normalized_strongness=certificate.normalized_strongness,
        dimensionless_step=certificate.dimensionless_step,
    )
    closed_form_lmi = closed_form_sector_lmi_matrix(closed_form)
    closed_form_factor = closed_form_factor_matrix(closed_form)
    if closed_form_lmi != closed_form_factor:
        raise AssertionError("closed-form LMI factorization failed in exact arithmetic")

    stable_eta = certificate.learning_rate
    unstable_eta = Fraction(1, 2_000)
    divergent_eta = Fraction(1, 500)
    zero_gain = repaired_zero_input_gain(
        floor=certificate.floor,
        repair_rho=certificate.repair_rho,
    )
    stable_jury_margin = scalar_momentum_jury_margin(
        learning_rate=stable_eta,
        beta=certificate.beta,
        curvature=certificate.hessian_upper,
        repaired_gain=zero_gain,
    )
    unstable_jury_margin = scalar_momentum_jury_margin(
        learning_rate=unstable_eta,
        beta=certificate.beta,
        curvature=certificate.hessian_upper,
        repaired_gain=zero_gain,
    )
    if stable_jury_margin <= 0 or unstable_jury_margin >= 0:
        raise AssertionError("matched local Jury controls have the wrong signs")
    actual_local_learning_rate_threshold = (
        2 * (1 + certificate.beta) / (certificate.hessian_upper * zero_gain)
    )

    common_trajectory = {
        "beta": float(certificate.beta),
        "curvature": float(certificate.hessian_upper),
        "floor": float(certificate.floor),
        "repair_rho": float(certificate.repair_rho),
        "initial_position": 1.0,
        "initial_momentum": 0.0,
    }
    stable = run_rank_one_momentum_trajectory(
        learning_rate=float(stable_eta),
        iterations=100_000,
        **common_trajectory,
    )
    unstable_cycle = run_rank_one_momentum_trajectory(
        learning_rate=float(unstable_eta),
        iterations=20_000,
        tail_length=12,
        **common_trajectory,
    )
    divergent = run_rank_one_momentum_trajectory(
        learning_rate=float(divergent_eta),
        iterations=1_000,
        **common_trajectory,
    )
    unstable_period_four_residual = period_residual(unstable_cycle, period=4)
    if abs(stable.final_position) >= 1e-30:
        raise AssertionError("stable trajectory did not reach its locked diagnostic threshold")
    if unstable_period_four_residual >= 1e-14 or abs(unstable_cycle.final_position) <= 1e-4:
        raise AssertionError("outside-region nonconvergent trajectory changed")
    if not divergent.exceeded_divergence_threshold:
        raise AssertionError("outside-region divergent trajectory changed")

    outside_alpha = 2 * alpha_limit
    generic_outside_radius = _complex_sector_radius(
        alpha=float(outside_alpha),
        beta=float(certificate.beta),
        nu=float(certificate.normalized_strongness),
    )
    if generic_outside_radius <= 1:
        raise AssertionError("generic sharpness witness did not become Schur unstable")

    revision = _run_git("rev-parse", "HEAD")
    status = _run_git("status", "--porcelain")
    return {
        "schema_version": "passive-muon-momentum-iqc-certificate-v1",
        "claim_scope": {
            "evidence_kind": (
                "exact_dimension_independent_iqc_theorem_exact_rational_lmi_replay_"
                "and_matched_float64_rank_one_trajectories"
            ),
            "quadratic_domain": (
                "every fixed finite-dimensional real quadratic with ell*I <= H <= L*I"
            ),
            "operator_domain": "R^(m x n) for every fixed finite positive m,n",
            "guarantee": (
                "global incremental exponential stability of the specified "
                "real-arithmetic momentum loop"
            ),
            "not_claimed": [
                "stability for nonquadratic or stochastic objectives",
                "a discrete BF16 theorem",
                "neural-network training convergence",
                "necessity of the sector boundary for the actual floored-Jordan loop",
                "a practically large certified learning rate",
            ],
        },
        "operator": {
            "formula": "R_rho,c(M)=H_h(M/max(c,||M||_F))+rho*M",
            "normalization_rule": "fixed_frobenius_floor",
            "floor_c_exact": str(certificate.floor),
            "additive_epsilon": "not_applicable; the denominator uses a max floor",
            "arithmetic_model": "real arithmetic",
            "orthogonalizer": JORDAN_QUINTIC.name,
            "scalar_composition": "h=q composed with itself 5 times",
            "stage_recurrence": "x_next=a*x+(b*x^2+c*x^4)*x",
            "coefficients_exact": {
                "a": "6889/2000",
                "b": "-191/40",
                "c": "4063/2000",
            },
            "steps": 5,
            "upstream": JORDAN_QUINTIC.source,
            "deficit_upper_at_c_1": _fraction_payload(certificate.deficit_upper),
            "derivative_absolute_upper": _fraction_payload(JORDAN_DERIVATIVE_UPPER),
            "repair_margin_mu": _fraction_payload(certificate.repair_margin),
            "repair_rho": _fraction_payload(certificate.repair_rho),
            "global_lipschitz_bound_K": _fraction_payload(certificate.lipschitz_bound),
        },
        "quadratic_and_loop": {
            "objective": "f(W)=0.5*<W-W_star,H(W-W_star)>",
            "hessian_lower_ell": str(certificate.hessian_lower),
            "hessian_upper_L": str(certificate.hessian_upper),
            "momentum_beta": str(certificate.beta),
            "update_order": [
                "m_(t+1)=beta*m_t+H*(W_t-W_star)",
                "W_(t+1)=W_t-eta*R_rho,c(m_(t+1))",
            ],
            "equilibrium": "(W_star,0)",
        },
        "analytic_sector_theorem": {
            "state_transform": [
                "y=H^(1/2)*Delta_W",
                "z=H^(-1/2)*Delta_m",
            ],
            "transformed_strong_monotonicity": "mu*ell",
            "transformed_lipschitz_bound": "K*L",
            "normalized_strongness_formula": "nu=mu*ell/(K*L)",
            "normalized_strongness": _fraction_payload(certificate.normalized_strongness),
            "dimensionless_step_formula": "alpha=eta*K*L",
            "strict_region": (
                "0<=beta<1 and 0<alpha<2*(1-beta)^2*(1+beta)*nu/((1+beta)^2-4*beta*nu^2)"
            ),
            "alpha_strict_supremum": _fraction_payload(alpha_limit),
            "eta_strict_supremum": _fraction_payload(certificate.certified_learning_rate_supremum),
            "sharpness_scope": (
                "exact for the reduced (nu-strong, one-Lipschitz) sector class; "
                "potentially conservative for linked H and floored-Jordan R"
            ),
            "iqcs": [
                "<Delta_v,Delta_u>-nu*||Delta_v||^2 >= 0",
                "||Delta_v||^2-||Delta_u||^2 >= 0",
            ],
            "invalid_iqc_explicitly_not_used": (
                "the gradient/cocoercive sector product for a symmetric slope"
            ),
            "closed_form_strict_factorization": {
                "formula": "M=-epsilon*I-gamma*n*n^T",
                "epsilon_exact": str(closed_form.epsilon),
                "gamma_exact": str(closed_form.factor_gamma),
                "factor_vector_exact": [str(value) for value in closed_form.factor_vector],
                "storage_exact": fraction_matrix_to_strings(closed_form.storage),
                "lambda_strong_exact": str(closed_form.lambda_strong),
                "lambda_lipschitz_exact": str(closed_form.lambda_lipschitz),
                "identity_replayed_exactly": True,
            },
        },
        "locked_rate_certificate": {
            "learning_rate_eta": _fraction_payload(stable_eta),
            "dimensionless_step_alpha": _fraction_payload(certificate.dimensionless_step),
            "alpha_over_strict_supremum": _decimal(certificate.dimensionless_step / alpha_limit),
            "conditioning_scale_gamma": str(certificate.conditioning_scale),
            "rate_squared": _fraction_payload(certificate.rate_squared),
            "rate": str(np.sqrt(float(certificate.rate_squared))),
            "storage_exact": fraction_matrix_to_strings(certificate.storage),
            "lambda_strong_exact": str(certificate.lambda_strong),
            "lambda_lipschitz_exact": str(certificate.lambda_lipschitz),
            "storage_leading_minors_exact": [str(value) for value in audit.storage_leading_minors],
            "negative_lmi_leading_minors_exact": [
                str(value) for value in audit.negative_lmi_leading_minors
            ],
            "lmi_exact": fraction_matrix_to_strings(audit.lmi),
            "sylvester_positive_storage": audit.storage_positive_definite,
            "sylvester_negative_lmi": audit.lmi_negative_definite,
            "solver_role": (
                "a floating-point SDP proposed P and multipliers; this manifest "
                "replays the rounded certificate using exact rational arithmetic"
            ),
        },
        "sector_boundary_witness": {
            "map": "U(v)=nu*v+sqrt(1-nu^2)*J*v in R^2, J^T=-J, J^T*J=I",
            "properties": "nu-strongly monotone and exactly one-Lipschitz",
            "characteristic": "z^2-(1+beta-alpha*(nu+i*sqrt(1-nu^2)))*z+beta",
            "boundary_claim": "unit-circle root at alpha=alpha_strict_supremum",
            "outside_alpha_exact": str(outside_alpha),
            "outside_alpha_factor": "2",
            "outside_float64_spectral_radius": generic_outside_radius,
            "scope": "sharpness of the reduced sector description only",
        },
        "actual_floored_jordan_local_instability": {
            "invariant_mode": "curvature-L rank-one scalar mode",
            "zero_input_polynomial_gain": _fraction_payload(
                zero_slope_gain_exact(JORDAN_QUINTIC, steps=5)
            ),
            "repaired_zero_input_gain": _fraction_payload(zero_gain),
            "jury_condition": "eta*L*repaired_zero_input_gain < 2*(1+beta)",
            "certified_point_jury_margin": _fraction_payload(stable_jury_margin),
            "outside_learning_rate": _fraction_payload(unstable_eta),
            "outside_jury_margin": _fraction_payload(unstable_jury_margin),
            "outside_locally_unstable": unstable_jury_margin < 0,
            "exact_local_learning_rate_threshold": _fraction_payload(
                actual_local_learning_rate_threshold
            ),
            "local_threshold_over_global_sector_supremum": _decimal(
                actual_local_learning_rate_threshold / certificate.certified_learning_rate_supremum
            ),
            "interpretation": (
                "the actual zero-linearization threshold is much larger than the "
                "dimension-uniform sector guarantee; this does not weaken the "
                "sufficiency theorem but makes it practically conservative"
            ),
        },
        "matched_rank_one_float64_examples": {
            "interpretation": (
                "diagnostic trajectories of the actual floored five-step Jordan map; "
                "the exact LMI and Jury signs, not sampling, carry the theorem claims"
            ),
            "shared_configuration": {
                "quadratic_hessian": "diag(1,10)",
                "invariant_initial_mode": "curvature-10 eigenvector",
                "beta": float(certificate.beta),
                "floor": float(certificate.floor),
                "repair_rho": float(certificate.repair_rho),
                "initial_position": 1.0,
                "initial_momentum": 0.0,
                "orthogonalizer": JORDAN_QUINTIC.name,
                "steps": 5,
                "additive_epsilon": "not_applicable; max-floor normalization",
                "arithmetic": (
                    "Python IEEE-754 binary64 scalar recurrence; NumPy is used "
                    "only for the separate spectral-radius diagnostic"
                ),
                "device": "CPU",
            },
            "certified_stable": {
                "learning_rate_exact": str(stable_eta),
                **_trajectory_payload(stable),
            },
            "outside_nonconvergent": {
                "learning_rate_exact": str(unstable_eta),
                "period_four_tail_residual": unstable_period_four_residual,
                **_trajectory_payload(unstable_cycle),
            },
            "far_outside_divergent": {
                "learning_rate_exact": str(divergent_eta),
                **_trajectory_payload(divergent),
            },
        },
        "software": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "numpy": np.__version__,
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
            "branch": "p3",
            "frozen_fallback_commit": "518cc9384a7a478f3c5956532fd26a6937f70d5f",
        },
        "experiment_provenance": {
            "seed": None,
            "randomness": "none; exact rational arithmetic and deterministic float64 recurrences",
            "command": (
                "uv run --locked python scripts/certify_momentum_stability.py "
                "--output results/summaries/momentum_iqc_certificate.json"
            ),
            "source_snapshot": _source_snapshot(),
        },
    }


def main() -> int:
    args = parse_args()
    payload = build_payload()
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
