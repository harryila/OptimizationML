#!/usr/bin/env python3
"""Replay the exact EMA/Nesterov IQC certificate and matched CPU controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from dataclasses import asdict
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Any

import mpmath as mp
import numpy as np
import scipy
import torch
from scipy.optimize import brentq

from passive_muon.ema_nesterov_experiment import (
    EmaNesterovTrajectory,
    ema_nesterov_period_residual,
    run_rank_one_ema_nesterov_trajectory,
)
from passive_muon.ema_nesterov_iqc import (
    audit_ema_nesterov_certificate,
    ema_nesterov_jury_margin,
    ema_nesterov_local_learning_rate_threshold,
    locked_ema_nesterov_certificate,
    skew_boundary_polynomial,
)
from passive_muon.floored_certificate import JORDAN_DERIVATIVE_UPPER
from passive_muon.momentum_iqc import fraction_matrix_to_strings, repaired_zero_input_gain
from passive_muon.specs import JORDAN_QUINTIC, zero_slope_gain_exact
from passive_muon.upstream_momentum import (
    PINNED_DEFAULT_BETA,
    PINNED_KELLER_JORDAN_MUON_REVISION,
    PINNED_MUON_PY_SHA256,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = (
    ".github/workflows/ci.yml",
    "README.md",
    "TASKS.md",
    "experiments/README.md",
    "pyproject.toml",
    "results/README.md",
    "results/summaries/RESULTS.md",
    "scripts/certify_ema_nesterov_stability.py",
    "src/passive_muon/ema_nesterov_experiment.py",
    "src/passive_muon/ema_nesterov_iqc.py",
    "src/passive_muon/floored_certificate.py",
    "src/passive_muon/momentum_experiment.py",
    "src/passive_muon/momentum_iqc.py",
    "src/passive_muon/normalizers.py",
    "src/passive_muon/operator.py",
    "src/passive_muon/orthogonalizers.py",
    "src/passive_muon/polynomials.py",
    "src/passive_muon/specs.py",
    "src/passive_muon/upstream_momentum.py",
    "tests/test_cli.py",
    "tests/test_ema_nesterov_iqc.py",
    "tests/test_result_manifests.py",
    "tests/test_upstream_momentum.py",
    "theory/claims.md",
    "theory/ema_nesterov_iqc_certificate.md",
    "third_party/UPSTREAM_COMMITS.md",
    "uv.lock",
)
FROZEN_FALLBACK_COMMIT = "518cc9384a7a478f3c5956532fd26a6937f70d5f"


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


def _trajectory_payload(trajectory: EmaNesterovTrajectory) -> dict[str, Any]:
    payload = asdict(trajectory)
    payload["tail"] = [
        {"position": position, "momentum": momentum, "signal": signal}
        for position, momentum, signal in trajectory.tail
    ]
    return payload


def _skew_boundary_alpha(*, beta: float, nu: float) -> float:
    def polynomial(alpha: float) -> float:
        return (
            beta**2 * (1 - beta) ** 2 * (2 * beta + 1) * alpha**3
            - 2 * (1 - beta) * beta * (3 * beta**2 - 1) * nu * alpha**2
            + ((1 + beta) * (2 * beta**2 - beta + 1) + 4 * beta * (beta**2 - beta - 1) * nu**2)
            * alpha
            - 2 * (1 - beta**2) * nu
        )

    return float(brentq(polynomial, 0.0, 1.0, xtol=5e-15, rtol=1e-14))


def _numerically_optimize_repair_margin() -> dict[str, Any]:
    locked = locked_ema_nesterov_certificate()
    with mp.workdps(60):
        beta = mp.mpf(locked.beta.numerator) / locked.beta.denominator
        condition_ratio = (
            mp.mpf(locked.hessian_lower.numerator)
            / locked.hessian_lower.denominator
            / (mp.mpf(locked.hessian_upper.numerator) / locked.hessian_upper.denominator)
        )
        hessian_upper = mp.mpf(locked.hessian_upper.numerator) / locked.hessian_upper.denominator
        base_gain_fraction = locked.deficit_upper + JORDAN_DERIVATIVE_UPPER
        base_gain = mp.mpf(base_gain_fraction.numerator) / base_gain_fraction.denominator

        def equations(repair_margin: mp.mpf, alpha: mp.mpf) -> tuple[mp.mpf, mp.mpf]:
            lipschitz = base_gain + repair_margin
            nu = condition_ratio * repair_margin / lipschitz
            c3 = beta**2 * (1 - beta) ** 2 * (2 * beta + 1)
            c2 = -2 * (1 - beta) * beta * (3 * beta**2 - 1) * nu
            c1 = (1 + beta) * (2 * beta**2 - beta + 1) + 4 * beta * (beta**2 - beta - 1) * nu**2
            c0 = -2 * (1 - beta**2) * nu
            polynomial = c3 * alpha**3 + c2 * alpha**2 + c1 * alpha + c0
            derivative_alpha = 3 * c3 * alpha**2 + 2 * c2 * alpha + c1
            derivative_nu = (
                -2 * (1 - beta) * beta * (3 * beta**2 - 1) * alpha**2
                + 8 * beta * (beta**2 - beta - 1) * nu * alpha
                - 2 * (1 - beta**2)
            )
            derivative_nu_margin = condition_ratio * base_gain / lipschitz**2
            stationarity = (
                -derivative_nu * derivative_nu_margin * lipschitz - derivative_alpha * alpha
            )
            return polynomial, stationarity

        margin_mp, alpha_mp = mp.findroot(
            equations,
            (mp.mpf(648), mp.mpf("0.0027104")),
            tol=mp.mpf("1e-50"),
        )
        lipschitz_mp = base_gain + margin_mp
        nu_mp = condition_ratio * margin_mp / lipschitz_mp
        eta_mp = alpha_mp / (hessian_upper * lipschitz_mp)
        margin = float(margin_mp)
        nu = float(nu_mp)
        alpha = float(alpha_mp)
        eta = float(eta_mp)
    scan_margins = np.geomspace(1e-4, 1e7, 1_101)
    scan_etas = []
    base_gain_float = float(locked.deficit_upper + JORDAN_DERIVATIVE_UPPER)
    condition_ratio_float = float(locked.hessian_lower / locked.hessian_upper)
    beta_float = float(locked.beta)
    hessian_upper_float = float(locked.hessian_upper)
    for scan_margin in scan_margins:
        scan_lipschitz = base_gain_float + float(scan_margin)
        scan_nu = condition_ratio_float * float(scan_margin) / scan_lipschitz
        scan_alpha = _skew_boundary_alpha(beta=beta_float, nu=scan_nu)
        scan_etas.append(scan_alpha / (hessian_upper_float * scan_lipschitz))
    scan_best_index = int(np.argmax(scan_etas))
    if eta < scan_etas[scan_best_index]:
        raise AssertionError("stationary repair-margin design lost to the broad scan")
    return {
        "repair_margin": margin,
        "normalized_strongness": nu,
        "alpha_skew_boundary": alpha,
        "eta_skew_boundary": eta,
        "working_precision_decimal_digits": 60,
        "corroborating_log_scan": {
            "lower_repair_margin": float(scan_margins[0]),
            "upper_repair_margin": float(scan_margins[-1]),
            "point_count": len(scan_margins),
            "best_grid_repair_margin": float(scan_margins[scan_best_index]),
            "best_grid_eta_skew_boundary": float(scan_etas[scan_best_index]),
        },
    }


def _local_spectral_radius(
    *, learning_rate: float, beta: float, curvature: float, repaired_gain: float
) -> float:
    one_minus_beta = 1 - beta
    scaled_step = learning_rate * curvature * repaired_gain
    matrix = np.array(
        [
            [
                1 - scaled_step * one_minus_beta * (1 + beta),
                -learning_rate * repaired_gain * beta**2,
            ],
            [one_minus_beta * curvature, beta],
        ],
        dtype=np.float64,
    )
    return float(np.max(np.abs(np.linalg.eigvals(matrix))))


def build_payload() -> dict[str, Any]:
    certificate = locked_ema_nesterov_certificate()
    audit = audit_ema_nesterov_certificate(certificate)
    if not audit.certified:
        raise AssertionError("exact rational EMA/Nesterov LMI certificate failed")
    if float(certificate.beta) != PINNED_DEFAULT_BETA:
        raise AssertionError("locked beta does not match the pinned upstream default")

    design_optimum = _numerically_optimize_repair_margin()
    locked_alpha_boundary = _skew_boundary_alpha(
        beta=float(certificate.beta),
        nu=float(certificate.normalized_strongness),
    )
    locked_eta_boundary = locked_alpha_boundary / (
        float(certificate.lipschitz_bound * certificate.hessian_upper)
    )
    rational_boundary_lower = Fraction(2_710_354, 1_000_000_000)
    rational_boundary_upper = Fraction(2_710_355, 1_000_000_000)
    lower_sign = skew_boundary_polynomial(
        alpha=rational_boundary_lower,
        beta=certificate.beta,
        normalized_strongness=certificate.normalized_strongness,
    )
    upper_sign = skew_boundary_polynomial(
        alpha=rational_boundary_upper,
        beta=certificate.beta,
        normalized_strongness=certificate.normalized_strongness,
    )
    if not lower_sign < 0 < upper_sign:
        raise AssertionError("exact rational skew-boundary sign bracket failed")

    zero_gain = repaired_zero_input_gain(
        floor=certificate.floor,
        repair_rho=certificate.repair_rho,
    )
    local_threshold = ema_nesterov_local_learning_rate_threshold(
        beta=certificate.beta,
        curvature=certificate.hessian_upper,
        repaired_gain=zero_gain,
    )
    outside_cycle_eta = Fraction(1, 400)
    divergent_eta = Fraction(1, 200)
    certified_jury_margin = ema_nesterov_jury_margin(
        learning_rate=certificate.learning_rate,
        beta=certificate.beta,
        curvature=certificate.hessian_upper,
        repaired_gain=zero_gain,
    )
    outside_jury_margin = ema_nesterov_jury_margin(
        learning_rate=outside_cycle_eta,
        beta=certificate.beta,
        curvature=certificate.hessian_upper,
        repaired_gain=zero_gain,
    )
    if certified_jury_margin <= 0 or outside_jury_margin >= 0:
        raise AssertionError("matched exact local Jury controls have the wrong signs")

    common_trajectory = {
        "beta": float(certificate.beta),
        "curvature": float(certificate.hessian_upper),
        "floor": float(certificate.floor),
        "repair_rho": float(certificate.repair_rho),
        "initial_position": 1.0,
        "initial_momentum": 0.0,
    }
    stable = run_rank_one_ema_nesterov_trajectory(
        learning_rate=float(certificate.learning_rate),
        iterations=100_000,
        **common_trajectory,
    )
    outside_cycle = run_rank_one_ema_nesterov_trajectory(
        learning_rate=float(outside_cycle_eta),
        iterations=20_000,
        tail_length=12,
        **common_trajectory,
    )
    divergent = run_rank_one_ema_nesterov_trajectory(
        learning_rate=float(divergent_eta),
        iterations=1_000,
        **common_trajectory,
    )
    period_two_residual = ema_nesterov_period_residual(outside_cycle, period=2)
    if abs(stable.final_position) >= 1e-100:
        raise AssertionError("certified trajectory did not reach its locked diagnostic threshold")
    if period_two_residual >= 1e-13 or abs(outside_cycle.final_position) <= 1e-4:
        raise AssertionError("outside-region period-two diagnostic changed")
    if not divergent.exceeded_divergence_threshold:
        raise AssertionError("outside-region divergence diagnostic changed")

    optimized_repaired_gain = (
        float(certificate.deficit_upper)
        + design_optimum["repair_margin"]
        + float(zero_slope_gain_exact(JORDAN_QUINTIC, steps=5))
    )
    optimized_local_threshold = (
        2
        * (1 + float(certificate.beta))
        / (
            (1 - float(certificate.beta))
            * (1 + 2 * float(certificate.beta))
            * float(certificate.hessian_upper)
            * optimized_repaired_gain
        )
    )
    optimized_gap = optimized_local_threshold / design_optimum["eta_skew_boundary"]
    locked_gap = float(local_threshold / certificate.learning_rate)
    if optimized_gap <= 1_000:
        raise AssertionError("hard appendix-only decision rule unexpectedly did not trigger")

    revision = _run_git("rev-parse", "HEAD")
    branch = _run_git("branch", "--show-current")
    status = _run_git("status", "--porcelain")
    return {
        "schema_version": "passive-muon-ema-nesterov-iqc-certificate-v1",
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
                "global incremental exponential stability for the specified real-arithmetic "
                "EMA/Nesterov loop using the repaired floored operator"
            ),
            "upstream_match": (
                "matches pinned EMA/Nesterov state-and-signal ordering after replacing the "
                "orthogonalizer by repaired floored R and omitting decoupled weight decay"
            ),
            "paper_position": "appendix_or_proof_of_principle_supporting_result",
            "not_claimed": [
                "stability of the complete pinned practical Muon implementation",
                "stability for nonquadratic or stochastic objectives",
                "a discrete BF16 theorem",
                "neural-network training convergence",
                "sufficiency of the numerical skew-response boundary for the nonlinear sector",
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
        "pinned_ema_nesterov_ordering": {
            "upstream_repository": "https://github.com/KellerJordan/Muon",
            "upstream_revision": PINNED_KELLER_JORDAN_MUON_REVISION,
            "audited_file": "muon.py",
            "audited_source_sha256": PINNED_MUON_PY_SHA256,
            "default_beta": str(certificate.beta),
            "default_nesterov": True,
            "update_order": [
                "g_t=H*(W_t-W_star)",
                "m_(t+1)=beta*m_t+(1-beta)*g_t",
                "s_(t+1)=beta*m_(t+1)+(1-beta)*g_t",
                "W_(t+1)=W_t-eta*R_rho,c(s_(t+1))",
            ],
            "local_automated_lerp_parity_regression": True,
            "independent_upstream_parity_audit_complete": False,
            "omitted_upstream_features": [
                "BF16 cast and backend accumulation behavior",
                "additive 1e-7 exact-current normalizer",
                "upstream transpose and aspect-ratio scaling",
                "decoupled weight decay",
                "distributed optimizer and training system",
            ],
        },
        "conditioned_iqc_theorem": {
            "state_transform": [
                "y=H^(1/2)*Delta_W",
                "z=H^(-1/2)*Delta_m",
            ],
            "nesterov_input": "p=(1-beta^2)*y+beta^2*z",
            "transition": [
                "y_next=y-alpha*u",
                "z_next=(1-beta)*y+beta*z",
            ],
            "normalized_output": "u=H^(1/2)*Delta_R/(K*L)",
            "normalized_strongness_formula": "nu=mu*ell/(K*L)",
            "normalized_strongness": _fraction_payload(certificate.normalized_strongness),
            "dimensionless_step_formula": "alpha=eta*K*L",
            "separate_incremental_iqcs": [
                "<p,u>-nu*||p||^2 >= 0",
                "||p||^2-||u||^2 >= 0",
            ],
            "lmi": ("T^T*P*T-tau^2*E^T*P*E+lambda_mu*Q_mu+lambda_L*Q_L < 0"),
            "dimension_independent": True,
            "full_matrix_tangent_space": True,
        },
        "repair_margin_design_search": {
            "status": "numerical_design_search_not_part_of_exact_proof",
            "method": (
                "60-digit stationarity solve for the first complex-skew Schur boundary, "
                "corroborated on a 1,101-point log scan over [1e-4,1e7]; exact rational "
                "LMI below is authoritative"
            ),
            "beta": float(certificate.beta),
            "condition_ratio_ell_over_L": float(
                certificate.hessian_lower / certificate.hessian_upper
            ),
            "numerical_stationary_design": design_optimum,
            "rational_lock_mu": int(certificate.repair_margin),
            "locked_mu_alpha_skew_boundary": locked_alpha_boundary,
            "locked_mu_eta_skew_boundary": locked_eta_boundary,
            "rational_lock_relative_eta_loss": (
                design_optimum["eta_skew_boundary"] - locked_eta_boundary
            )
            / design_optimum["eta_skew_boundary"],
            "exact_locked_mu_skew_boundary_sign_bracket": {
                "lower_alpha": _fraction_payload(rational_boundary_lower),
                "lower_polynomial_sign": "negative" if lower_sign < 0 else "nonnegative",
                "upper_alpha": _fraction_payload(rational_boundary_upper),
                "upper_polynomial_sign": "positive" if upper_sign > 0 else "nonpositive",
            },
            "qualification": (
                "the skew response is an admissible necessary control for sector-only "
                "theorems; no symbolic sufficiency claim is made up to this boundary"
            ),
        },
        "locked_exact_rate_certificate": {
            "c": str(certificate.floor),
            "ell": str(certificate.hessian_lower),
            "L": str(certificate.hessian_upper),
            "beta": str(certificate.beta),
            "one_minus_beta": str(1 - certificate.beta),
            "repair_margin_mu": _fraction_payload(certificate.repair_margin),
            "repair_rho": _fraction_payload(certificate.repair_rho),
            "lipschitz_bound_K": _fraction_payload(certificate.lipschitz_bound),
            "normalized_strongness_nu": _fraction_payload(certificate.normalized_strongness),
            "dimensionless_step_alpha": _fraction_payload(certificate.dimensionless_step),
            "learning_rate_eta": _fraction_payload(certificate.learning_rate),
            "rate_squared": _fraction_payload(certificate.rate_squared),
            "storage_P": fraction_matrix_to_strings(certificate.storage),
            "lambda_strong": _fraction_payload(certificate.lambda_strong),
            "lambda_lipschitz": _fraction_payload(certificate.lambda_lipschitz),
            "lmi": fraction_matrix_to_strings(audit.lmi),
            "storage_leading_minors": [
                _fraction_payload(value) for value in audit.storage_leading_minors
            ],
            "negative_lmi_leading_minors": [
                _fraction_payload(value) for value in audit.negative_lmi_leading_minors
            ],
            "sylvester_positive_storage": audit.storage_positive_definite,
            "sylvester_negative_lmi": audit.lmi_negative_definite,
            "solver_role": "floating-point SDP used for discovery only; replay is rational",
        },
        "actual_floored_jordan_local_control": {
            "zero_slope_gain_h_prime_0": _fraction_payload(
                zero_slope_gain_exact(JORDAN_QUINTIC, steps=5)
            ),
            "repaired_zero_input_gain": _fraction_payload(zero_gain),
            "local_threshold_formula": ("2*(1+beta)/((1-beta)*(1+2*beta)*L*(rho+h_prime_0/c))"),
            "local_learning_rate_threshold": _fraction_payload(local_threshold),
            "certified_point_jury_margin": _fraction_payload(certified_jury_margin),
            "outside_point_eta": _fraction_payload(outside_cycle_eta),
            "outside_point_jury_margin": _fraction_payload(outside_jury_margin),
            "outside_locally_unstable": outside_jury_margin < 0,
            "optimized_skew_boundary_gap_factor": optimized_gap,
            "optimized_design_local_threshold": optimized_local_threshold,
            "locked_exact_certificate_gap_factor": locked_gap,
            "hard_decision_rule_triggered": optimized_gap > 1_000,
            "result_classification": "appendix_or_proof_of_principle",
        },
        "matched_rank_one_float64_examples": {
            "shared_configuration": {
                "objective_mode": "curvature-10 rank-one invariant subspace",
                "initial_position": 1.0,
                "initial_momentum": 0.0,
                "beta": float(certificate.beta),
                "floor": float(certificate.floor),
                "repair_rho": float(certificate.repair_rho),
                "operator": "actual repository five-stage floored Jordan scalar response",
                "controlled_variable": "learning_rate_only",
                "dtype": "float64",
            },
            "certified_stable": {
                **_trajectory_payload(stable),
                "learning_rate_exact": str(certificate.learning_rate),
                "learning_rate": float(certificate.learning_rate),
            },
            "outside_period_two": {
                **_trajectory_payload(outside_cycle),
                "learning_rate_exact": str(outside_cycle_eta),
                "learning_rate": float(outside_cycle_eta),
                "period_two_tail_residual": period_two_residual,
                "local_linear_spectral_radius": _local_spectral_radius(
                    learning_rate=float(outside_cycle_eta),
                    beta=float(certificate.beta),
                    curvature=float(certificate.hessian_upper),
                    repaired_gain=float(zero_gain),
                ),
            },
            "far_outside_divergent": {
                **_trajectory_payload(divergent),
                "learning_rate_exact": str(divergent_eta),
                "learning_rate": float(divergent_eta),
                "local_linear_spectral_radius": _local_spectral_radius(
                    learning_rate=float(divergent_eta),
                    beta=float(certificate.beta),
                    curvature=float(certificate.hessian_upper),
                    repaired_gain=float(zero_gain),
                ),
            },
        },
        "review_status": {
            "independent_human_review_C7_C8_complete": False,
            "independent_pinned_upstream_parity_audit_complete": False,
            "automated_local_two_lerp_regression_present": True,
        },
        "prior_art": [
            {
                "citation": "Lessard, Recht, and Packard (2016)",
                "url": "https://arxiv.org/abs/1408.3595",
                "role": "foundational optimization IQC framework",
            },
            {
                "citation": "Zhang, Bao, Lessard, and Grosse, JMLR 22(103), 2021",
                "url": "https://jmlr.org/papers/v22/20-1068.html",
                "role": (
                    "prior IQC analysis of finite-memory algorithms with strongly "
                    "monotone Lipschitz, potentially nonconservative operators"
                ),
            },
        ],
        "experiment_provenance": {
            "seed": None,
            "determinism": "exact rational replay plus deterministic CPU float64 diagnostics",
            "source_snapshot": _source_snapshot(),
        },
        "software": {
            "python": platform.python_version(),
            "mpmath": mp.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pytorch": torch.__version__,
        },
        "hardware": {
            "platform": platform.platform(),
            "machine_architecture": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
        "git": {
            "sha": revision.stdout.strip() if revision.returncode == 0 else "unknown",
            "branch": branch.stdout.strip() if branch.returncode == 0 else "unknown",
            "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
            "frozen_fallback_commit": FROZEN_FALLBACK_COMMIT,
        },
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
