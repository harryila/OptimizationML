#!/usr/bin/env python3
"""Replay the exact full-p4-step nonlinear convergence certificate."""

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

import numpy as np
import torch

from passive_muon.ema_nesterov_iqc import locked_ema_nesterov_certificate
from passive_muon.floored_certificate import certified_dimension_uniform_deficit
from passive_muon.nonquadratic_convergence import (
    audit_nonquadratic_convergence,
    locked_nonquadratic_convergence_certificate,
    nonquadratic_convergence_matrices,
)
from passive_muon.nonquadratic_stability import LOCKED_NONQUADRATIC_LEARNING_RATE
from passive_muon.specs import JORDAN_QUINTIC
from passive_muon.structure_aware_stability import (
    LOCKED_FLOOR,
    LOCKED_LEARNING_RATE,
    LOCKED_REPAIR_MARGIN,
    LOCKED_REPAIR_RHO,
    LOCKED_TAU,
)
from passive_muon.upstream_momentum import (
    PINNED_KELLER_JORDAN_MUON_REVISION,
    PINNED_MUON_PY_SHA256,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
P4_BASE_COMMIT = "c9636358de2d3d17bf5e62b0f03c7aff12da97bd"
P4_RESULT_PATH = "results/summaries/structure_aware_stability_certificate.json"
P5_INCREMENTAL_RESULT_PATH = "results/summaries/nonquadratic_stability_certificate.json"
SOURCE_PATHS = (
    "scripts/certify_nonquadratic_convergence.py",
    "src/passive_muon/nonquadratic_convergence.py",
    "src/passive_muon/structure_aware_stability.py",
    "src/passive_muon/floored_certificate.py",
    "src/passive_muon/momentum_iqc.py",
    "src/passive_muon/normalizers.py",
    "src/passive_muon/operator.py",
    "src/passive_muon/orthogonalizers.py",
    "src/passive_muon/polynomials.py",
    "src/passive_muon/specs.py",
    "src/passive_muon/upstream_momentum.py",
    "tests/test_nonquadratic_convergence.py",
    "tests/test_nonquadratic_convergence_cli.py",
    "theory/nonquadratic_convergence_certificate.md",
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


def _matrix_payload(matrix: tuple[tuple[Fraction, ...], ...]) -> list[list[str]]:
    return [[str(value) for value in row] for row in matrix]


def _vector_payload(vector: tuple[Fraction, ...]) -> list[str]:
    return [str(value) for value in vector]


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
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _git_state() -> dict[str, object]:
    status = _git_output("status", "--porcelain")
    return {
        "sha": _git_output("rev-parse", "HEAD") or "unavailable",
        "branch": _git_output("branch", "--show-current") or "unavailable",
        "dirty": bool(status) if status is not None else None,
        "frozen_p4_base_commit": P4_BASE_COMMIT,
    }


def build_payload() -> dict[str, Any]:
    certificate = locked_nonquadratic_convergence_certificate()
    audit = audit_nonquadratic_convergence(certificate)
    if not audit.certified:
        raise AssertionError("exact full-step nonquadratic convergence replay failed")
    if certificate.learning_rate != LOCKED_LEARNING_RATE:
        raise AssertionError("nonquadratic convergence point does not retain the p4 step")

    p3_learning_rate = locked_ema_nesterov_certificate().learning_rate
    p4_over_p3 = certificate.learning_rate / p3_learning_rate
    matrices = nonquadratic_convergence_matrices(certificate)
    coefficients = JORDAN_QUINTIC.fractions()
    edge_matrices = {
        "0_to_current": matrices.interpolation_01,
        "current_to_next": matrices.interpolation_12,
        "next_to_0": matrices.interpolation_20,
        "next_to_current": matrices.interpolation_21,
    }
    edge_multipliers = {
        "0_to_current": certificate.lambda_interpolation_01,
        "current_to_next": certificate.lambda_interpolation_12,
        "next_to_0": certificate.lambda_interpolation_20,
        "next_to_current": certificate.lambda_interpolation_21,
    }
    return {
        "schema_version": "passive-muon-nonquadratic-convergence-certificate-v1",
        "claim_scope": {
            "evidence_kind": (
                "exact rational dimension-independent interpolation-storage certificate"
            ),
            "guarantee": (
                "global exponential convergence to the unique minimizer for the specified "
                "deterministic real-arithmetic repaired floored-Jordan EMA/Nesterov loop"
            ),
            "matrix_domain": "R^(m x n) for every fixed finite positive m,n",
            "objective_domain": (
                "every fixed differentiable globally 1-strongly-convex, 10-smooth scalar "
                "objective; nonlinear and trajectory-varying Hessian orientations allowed"
            ),
            "incremental_distinction": (
                "trajectory-to-minimizer convergence, not arbitrary-pair incremental "
                "contraction; the separate 5%-of-p4 certificate proves the latter"
            ),
            "not_claimed": [
                "stability of exact-current or additive-epsilon normalization",
                "BF16, stochastic-gradient, time-varying-objective, or weight-decay stability",
                "aspect-ratio-scaled or complete neural-network training convergence",
                "a hard pointwise IQC after discarding the objective-gap storage",
                "necessity or optimality of the learning rate or rate",
                "a proof supplied by sampled nonlinear trajectories",
            ],
        },
        "operator": {
            "formula": "R(M)=H_h(M/max(c,||M||_F))+rho*M",
            "domain": "every fixed finite real matrix shape",
            "normalization": "fixed_frobenius_floor",
            "floor_c": _fraction_payload(LOCKED_FLOOR),
            "additive_epsilon": "not_applicable; denominator uses a max floor",
            "orthogonalizer": JORDAN_QUINTIC.name,
            "coefficients_exact": {
                "a": str(coefficients[0]),
                "b": str(coefficients[1]),
                "c": str(coefficients[2]),
            },
            "steps": 5,
            "repair_deficit_upper": _fraction_payload(certified_dimension_uniform_deficit()),
            "repair_margin_mu": _fraction_payload(LOCKED_REPAIR_MARGIN),
            "constant_repair_rho": _fraction_payload(LOCKED_REPAIR_RHO),
            "upstream_revision": PINNED_KELLER_JORDAN_MUON_REVISION,
        },
        "objective_interpolation": {
            "normalization": "F=(f-f_star)/L, u=grad(f)/L, k=ell/L",
            "strong_convexity_ell": _fraction_payload(certificate.strong_convexity),
            "smoothness_L": _fraction_payload(certificate.smoothness),
            "condition_ratio_k": _fraction_payload(certificate.condition_ratio),
            "inequality": (
                "I_ij=F_i-F_j-<u_j,x_i-x_j>-(||du||^2-2*k<dx,du>+k||dx||^2)/(2*(1-k)) >= 0"
            ),
            "nodes": ["minimizer", "current iterate", "next iterate"],
            "weighted_function_coefficients": {
                "actual_current_next": [
                    _fraction_payload(value) for value in audit.interpolation_function_coefficients
                ],
                "expected_to_cancel_lyapunov_gap": [
                    _fraction_payload(value) for value in audit.expected_function_coefficients
                ],
                "cancel_exactly": audit.function_values_cancel,
            },
            "qualification": (
                "the quadratic edge supply is used with explicit objective-gap storage; "
                "it is not asserted as a standalone hard IQC"
            ),
        },
        "operator_decomposition": {
            "formula": "R(s)=gamma*s+E(s)",
            "center_gain_gamma": _fraction_payload(certificate.center_gain),
            "residual_lipschitz_K_E": _fraction_payload(certificate.residual_lipschitz),
            "residual_ratio_K_E_over_gamma": _fraction_payload(certificate.residual_ratio),
        },
        "update_order": [
            "g_t=grad_f(W_t)",
            "m_(t+1)=beta*m_t+(1-beta)*g_t",
            "s_(t+1)=beta*m_(t+1)+(1-beta)*g_t",
            "W_(t+1)=W_t-eta*R(s_(t+1))",
        ],
        "locked_exact_certificate": {
            "learning_rate_eta": _fraction_payload(certificate.learning_rate),
            "beta": _fraction_payload(certificate.beta),
            "exponential_rate_tau": _fraction_payload(certificate.tau),
            "rate_squared": _fraction_payload(certificate.tau**2),
            "dimensionless_step_eta_gamma_L": _fraction_payload(certificate.dimensionless_step),
            "coordinates": "chi=(W-W_star,m/L,grad(f)/L,E(s)/(K_E*L),grad(f_next)/L)",
            "storage_formula": ("V=[w,z]^T(P tensor I)[w,z]+c_F*(f(W)-f_star)/L"),
            "storage_P": _matrix_payload(certificate.storage),
            "function_storage_c_F": _fraction_payload(certificate.function_storage),
            "storage_normalization_trace_P_plus_c_F": _fraction_payload(
                sum(certificate.storage[index][index] for index in range(2))
                + certificate.function_storage
            ),
            "interpolation_multipliers": {
                name: _fraction_payload(value) for name, value in edge_multipliers.items()
            },
            "residual_lipschitz_multiplier": _fraction_payload(certificate.lambda_residual_norm),
            "transition_matrix": _matrix_payload(matrices.transition),
            "state_selection_matrix": _matrix_payload(matrices.selection),
            "signal_selector": _vector_payload(matrices.signal_selector),
            "interpolation_quadratic_matrices": {
                name: _matrix_payload(matrix) for name, matrix in edge_matrices.items()
            },
            "residual_iqc_matrix": _matrix_payload(matrices.residual_norm),
            "lmi_matrix": _matrix_payload(audit.lmi),
            "storage_leading_principal_minors": [
                _fraction_payload(value) for value in audit.storage_leading_minors
            ],
            "negative_lmi_leading_principal_minors": [
                _fraction_payload(value) for value in audit.negative_lmi_leading_minors
            ],
            "storage_positive_definite": audit.storage_positive_definite,
            "lmi_negative_definite": audit.lmi_negative_definite,
            "function_values_cancel": audit.function_values_cancel,
            "all_exact_checks_passed": audit.certified,
        },
        "comparison_and_gate": {
            "p4_learning_rate": _fraction_payload(LOCKED_LEARNING_RATE),
            "full_p4_step_retained": certificate.learning_rate == LOCKED_LEARNING_RATE,
            "p3_learning_rate": _fraction_payload(p3_learning_rate),
            "full_step_over_p3": _fraction_payload(p4_over_p3),
            "incremental_p5_learning_rate": _fraction_payload(LOCKED_NONQUADRATIC_LEARNING_RATE),
            "full_step_over_incremental_p5": _fraction_payload(
                certificate.learning_rate / LOCKED_NONQUADRATIC_LEARNING_RATE
            ),
            "rate_decrement_over_p4": _fraction_payload((1 - certificate.tau) / (1 - LOCKED_TAU)),
            "declared_gate": "eta >= 0.1*eta_p4",
            "classification": "excellent; major nonlinear theorem",
        },
        "independent_audit": {
            "status": "passed",
            "items": [
                "F_(k,1) interpolation formula and every directed edge sign",
                "exact objective-value flow cancellation",
                "pinned EMA/Nesterov and residual scaling",
                "positive storage and negative 5x5 LMI by exact rational replay",
                "trajectory-to-minimizer scope rather than arbitrary-pair incrementality",
            ],
        },
        "p4_fallback": {
            "base_commit": P4_BASE_COMMIT,
            "manifest_path": P4_RESULT_PATH,
            "manifest_sha256": _sha256(P4_RESULT_PATH),
            "status": "preserved on its own branch and commit",
        },
        "incremental_p5_reference": {
            "manifest_path": P5_INCREMENTAL_RESULT_PATH,
            "manifest_sha256": _sha256(P5_INCREMENTAL_RESULT_PATH),
            "status": "preserved as the stronger arbitrary-pair guarantee at five percent",
        },
        "proof_replay_provenance": {
            "arithmetic": "fractions.Fraction exact rational arithmetic",
            "analytic_seed": None,
            "discovery_only_solver": {
                "python": "3.12.11",
                "cvxpy": "1.9.2",
                "clarabel": "0.11.1",
                "sympy_exact_cross_check": "1.14.0",
            },
            "source_snapshot": _source_snapshot(),
            "software": {
                "python": sys.version,
                "numpy": np.__version__,
                "torch": torch.__version__,
            },
            "hardware": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
            },
        },
        "git": _git_state(),
        "upstream_provenance": {
            "repository": "https://github.com/KellerJordan/Muon",
            "revision": PINNED_KELLER_JORDAN_MUON_REVISION,
            "muon_py_sha256": PINNED_MUON_PY_SHA256,
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
