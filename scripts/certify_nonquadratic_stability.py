#!/usr/bin/env python3
"""Replay and serialize the exact p5 nonquadratic stability certificate."""

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
from passive_muon.nonquadratic_stability import (
    LOCKED_NONQUADRATIC_LEARNING_RATE,
    audit_nonquadratic_certificate,
    locked_nonquadratic_certificate,
    nonquadratic_iqc_matrices,
)
from passive_muon.specs import JORDAN_QUINTIC
from passive_muon.structure_aware_stability import (
    LOCKED_FLOOR,
    LOCKED_LEARNING_RATE,
    LOCKED_REPAIR_MARGIN,
    LOCKED_REPAIR_RHO,
)
from passive_muon.upstream_momentum import (
    PINNED_KELLER_JORDAN_MUON_REVISION,
    PINNED_MUON_PY_SHA256,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
P4_BASE_COMMIT = "c9636358de2d3d17bf5e62b0f03c7aff12da97bd"
P4_RESULT_PATH = "results/summaries/structure_aware_stability_certificate.json"
SOURCE_PATHS = (
    "scripts/certify_nonquadratic_stability.py",
    "src/passive_muon/nonquadratic_stability.py",
    "src/passive_muon/structure_aware_stability.py",
    "src/passive_muon/floored_certificate.py",
    "src/passive_muon/momentum_iqc.py",
    "src/passive_muon/normalizers.py",
    "src/passive_muon/operator.py",
    "src/passive_muon/orthogonalizers.py",
    "src/passive_muon/polynomials.py",
    "src/passive_muon/specs.py",
    "src/passive_muon/upstream_momentum.py",
    "tests/test_nonquadratic_certificate_cli.py",
    "tests/test_nonquadratic_stability.py",
    "theory/nonquadratic_stability_certificate.md",
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
    certificate = locked_nonquadratic_certificate()
    audit = audit_nonquadratic_certificate(certificate)
    if not audit.certified:
        raise AssertionError("exact p5 nonlinear LMI certificate failed")

    p3_learning_rate = locked_ema_nesterov_certificate().learning_rate
    p5_over_p3 = LOCKED_NONQUADRATIC_LEARNING_RATE / p3_learning_rate
    p5_over_p4 = LOCKED_NONQUADRATIC_LEARNING_RATE / LOCKED_LEARNING_RATE
    if p5_over_p4 != Fraction(1, 20):
        raise AssertionError("locked p5 point is not exactly five percent of p4")

    coefficients = JORDAN_QUINTIC.fractions()
    iqcs = nonquadratic_iqc_matrices(certificate)
    return {
        "schema_version": "passive-muon-nonquadratic-stability-certificate-v1",
        "claim_scope": {
            "evidence_kind": "exact_rational_dimension_independent_static_iqc_certificate",
            "guarantee": (
                "global incremental exponential stability for the specified deterministic "
                "real-arithmetic repaired floored-Jordan EMA/Nesterov loop"
            ),
            "matrix_domain": "R^(m x n) for every fixed finite positive m,n",
            "objective_domain": (
                "every fixed differentiable globally 1-strongly-convex, 10-smooth scalar "
                "objective; nonlinear and trajectory-varying local Hessian orientations allowed"
            ),
            "not_claimed": [
                "stability of exact-current or additive-epsilon normalization",
                "BF16, stochastic-gradient, weight-decay, or aspect-ratio-scaled stability",
                "neural-network training convergence",
                "necessity or optimality of the certified learning rate",
                "infeasibility of richer dynamic or cyclic IQCs at larger steps",
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
        "objective_class": {
            "strong_convexity_ell": _fraction_payload(certificate.strong_convexity),
            "smoothness_L": _fraction_payload(certificate.smoothness),
            "potential_required": True,
            "pointwise_pairwise_iqc": "<dg-ell*dW,L*dW-dg> >= 0",
            "fixed_hessian_or_common_eigenbasis_required": False,
        },
        "operator_decomposition": {
            "formula": "R(s)=gamma*s+E(s)",
            "center_gain_gamma": _fraction_payload(certificate.center_gain),
            "residual_lipschitz_K_E": _fraction_payload(certificate.residual_lipschitz),
            "centered_residual_radius_d": _fraction_payload(certificate.centered_residual_radius),
            "residual_ratio_K_E_over_gamma": _fraction_payload(certificate.residual_ratio),
            "centered_ratio_d_over_K_E": _fraction_payload(certificate.centered_residual_ratio),
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
            "coordinates": "chi=(dW,dm/L,dg/L,de/(K_E*L))",
            "storage_P": _matrix_payload(certificate.storage),
            "multipliers": {
                "gradient_interpolation": _fraction_payload(certificate.lambda_gradient),
                "residual_lipschitz": _fraction_payload(certificate.lambda_residual_norm),
                "centered_residual_lower": _fraction_payload(certificate.lambda_residual_lower),
                "centered_residual_upper": _fraction_payload(certificate.lambda_residual_upper),
                "centered_constraint_status": (
                    "valid independent IQCs; inactive at the locked point after numerical search"
                ),
            },
            "iqc_matrices": {name: _matrix_payload(matrix) for name, matrix in iqcs.items()},
            "lmi_matrix": _matrix_payload(audit.lmi),
            "storage_leading_principal_minors": [
                _fraction_payload(value) for value in audit.storage_leading_minors
            ],
            "negative_lmi_leading_principal_minors": [
                _fraction_payload(value) for value in audit.negative_lmi_leading_minors
            ],
            "storage_positive_definite": audit.storage_positive_definite,
            "lmi_negative_definite": audit.lmi_negative_definite,
            "all_exact_checks_passed": audit.certified,
        },
        "comparison_and_gate": {
            "p4_learning_rate": _fraction_payload(LOCKED_LEARNING_RATE),
            "p5_over_p4": _fraction_payload(p5_over_p4),
            "p3_learning_rate": _fraction_payload(p3_learning_rate),
            "p5_over_p3": _fraction_payload(p5_over_p3),
            "declared_band": "0.01*eta_p4 <= eta_p5 < 0.1*eta_p4",
            "classification": "strong nonlinear extension",
        },
        "discovery_only_diagnostics": {
            "authoritative_for_theorem": False,
            "solver_environment": {
                "python": "3.12.11",
                "cvxpy": "1.9.2",
                "clarabel": "0.11.1",
                "independent_sign_check": "SCS via CVXPY",
            },
            "static_iqc_rate_one_boundary_eta_approx": 1.69390537084e-6,
            "boundary_over_p4_approx": 0.05420497187,
            "locked_point_fraction_of_numerical_boundary_approx": 0.9224246,
            "centered_iqc_effect": (
                "less than 1e-11 relative boundary change; numerical multipliers near zero"
            ),
            "qualification": (
                "solver diagnostics selected the rational interior point; exact Sylvester "
                "signs carry the theorem and no general larger-step impossibility is claimed"
            ),
        },
        "p4_fallback": {
            "base_commit": P4_BASE_COMMIT,
            "manifest_path": P4_RESULT_PATH,
            "manifest_sha256": _sha256(P4_RESULT_PATH),
            "status": "preserved; p5 is a separate branch",
        },
        "proof_replay_provenance": {
            "arithmetic": "fractions.Fraction exact rational arithmetic",
            "analytic_seed": None,
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
