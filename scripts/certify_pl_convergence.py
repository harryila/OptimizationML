#!/usr/bin/env python3
"""Replay the exact full-p5-step smooth-PL convergence certificate."""

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

from passive_muon.floored_certificate import certified_dimension_uniform_deficit
from passive_muon.nonquadratic_convergence import (
    LOCKED_NONQUADRATIC_CONVERGENCE_LEARNING_RATE,
    LOCKED_NONQUADRATIC_CONVERGENCE_TAU,
)
from passive_muon.pl_convergence import (
    audit_pl_convergence,
    locked_pl_convergence_certificate,
    pl_convergence_matrices,
)
from passive_muon.specs import JORDAN_QUINTIC
from passive_muon.structure_aware_stability import (
    LOCKED_FLOOR,
    LOCKED_REPAIR_MARGIN,
    LOCKED_REPAIR_RHO,
)
from passive_muon.upstream_momentum import (
    PINNED_KELLER_JORDAN_MUON_REVISION,
    PINNED_MUON_PY_SHA256,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
P5_CHECKPOINT_TAG = "p5-checkpoint"
P5_CHECKPOINT_TAG_OBJECT = "ad54ad81034a4e87bdf043d12afc0fa8f054cf8e"
P5_CHECKPOINT_COMMIT = "a549fb4c206335ef9ec264524e0f581216b250d4"
P5_FULL_STEP_RESULT_PATH = "results/summaries/nonquadratic_convergence_certificate.json"
P5_INCREMENTAL_RESULT_PATH = "results/summaries/nonquadratic_stability_certificate.json"
SOURCE_PATHS = (
    "scripts/certify_pl_convergence.py",
    "src/passive_muon/pl_convergence.py",
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
    "tests/test_pl_convergence.py",
    "tests/test_pl_convergence_cli.py",
    "theory/pl_convergence_certificate.md",
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
        "frozen_p5_checkpoint_tag": P5_CHECKPOINT_TAG,
        "frozen_p5_checkpoint_commit": P5_CHECKPOINT_COMMIT,
    }


def build_payload() -> dict[str, Any]:
    certificate = locked_pl_convergence_certificate()
    audit = audit_pl_convergence(certificate)
    if not audit.certified:
        raise AssertionError("exact smooth-PL convergence replay failed")
    if certificate.learning_rate != LOCKED_NONQUADRATIC_CONVERGENCE_LEARNING_RATE:
        raise AssertionError("smooth-PL point does not retain the full p5 step")

    matrices = pl_convergence_matrices(certificate)
    coefficients = JORDAN_QUINTIC.fractions()
    return {
        "schema_version": "passive-muon-pl-convergence-certificate-v1",
        "claim_scope": {
            "evidence_kind": "exact rational dimension-independent value-storage certificate",
            "guarantee": (
                "global linear function-value convergence and momentum decay for the specified "
                "deterministic exact-real repaired floored-Jordan EMA/Nesterov loop"
            ),
            "matrix_domain": "R^(m x n) for every fixed finite positive m,n",
            "objective_domain": (
                "every fixed differentiable globally 10-smooth scalar objective with finite "
                "infimum satisfying the global PL inequality with constant 1; nonconvex and "
                "nonunique-minimizer objectives are included"
            ),
            "pl_convention": "0.5*||grad f(W)||_F^2 >= ell_PL*(f(W)-f_star)",
            "headline_conclusion": ("f(W_t)-f_star <= C*q^t and m_t -> 0 for q=tau^2<1"),
            "iterate_corollary": (
                "geometrically summable updates imply convergence to some trajectory-dependent "
                "global minimizer, not to a unique or preselected minimizer"
            ),
            "incremental_distinction": (
                "not arbitrary-pair incremental contraction; PL is pointwise and permits a "
                "non-singleton minimizer set"
            ),
            "not_claimed": [
                "convexity, strong convexity, or a unique minimizer",
                "arbitrary-pair incremental contraction",
                "a theorem under only a local PL inequality",
                "stochastic-gradient, time-varying-objective, or BF16 stability",
                "weight-decay, aspect-ratio-scaled, or complete neural-network convergence",
                "stability of exact-current or additive-epsilon normalization",
                "an unrepaired upstream Muon guarantee",
                "necessity or optimality of the learning rate or rate",
                "a proof supplied by numerical SDP status or sampled trajectories",
            ],
        },
        "operator": {
            "formula": "R(M)=H_h(M/max(c,||M||_F))+rho*M",
            "domain": "every fixed finite real matrix shape",
            "normalization": "exact fixed Frobenius max floor",
            "floor_c": _fraction_payload(LOCKED_FLOOR),
            "additive_epsilon": "none; denominator is max(c,||M||_F)",
            "orthogonalizer": JORDAN_QUINTIC.name,
            "coefficients_exact": {
                "a": str(coefficients[0]),
                "b": str(coefficients[1]),
                "c": str(coefficients[2]),
            },
            "newton_schulz_steps": 5,
            "repair_deficit_upper": _fraction_payload(certified_dimension_uniform_deficit()),
            "repair_margin_mu": _fraction_payload(LOCKED_REPAIR_MARGIN),
            "constant_repair_rho": _fraction_payload(LOCKED_REPAIR_RHO),
            "upstream_revision": PINNED_KELLER_JORDAN_MUON_REVISION,
        },
        "objective_class": {
            "smoothness_L": _fraction_payload(certificate.smoothness),
            "pl_constant_ell_PL": _fraction_payload(certificate.pl_constant),
            "normalized_pl_constant_k": _fraction_payload(certificate.condition_ratio),
            "finite_infimum_assumption": "f_star=inf_W f(W)>-infinity",
            "normalization": "F=(f-f_star)/L, u=grad(f)/L, z=m/L",
            "smooth_nonconvex_interpolation": (
                "I_ij=F_i-F_j-<u_j,dx>-(||du||^2+2<dx,du>-||dx||^2)/4 >= 0"
            ),
            "interpolation_curvature_class": "(-1,1); negative curvature is allowed",
            "interpolation_nodes": ["current iterate", "next iterate"],
            "pl_supply": "||u_next||^2-2*k*F_next >= 0",
            "weighted_function_coefficients": {
                "actual_current_next": [
                    _fraction_payload(value) for value in audit.supply_function_coefficients
                ],
                "expected_to_cancel_storage_gap": [
                    _fraction_payload(value) for value in audit.expected_function_coefficients
                ],
                "cancel_exactly": audit.function_values_cancel,
            },
        },
        "operator_decomposition": {
            "formula": "R(s)=gamma*s+E(s)",
            "center_gain_gamma": _fraction_payload(certificate.center_gain),
            "residual_lipschitz_K_E": _fraction_payload(certificate.residual_lipschitz),
            "residual_ratio_K_E_over_gamma": _fraction_payload(certificate.residual_ratio),
            "normalized_residual_supply": "||p||_F^2-||v||_F^2 >= 0",
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
            "exponential_norm_rate_tau": _fraction_payload(certificate.tau),
            "function_value_rate_q": _fraction_payload(certificate.tau**2),
            "dimensionless_step_eta_gamma_L": _fraction_payload(certificate.dimensionless_step),
            "coordinates": "chi=(m/L,grad(f)/L,E(s)/(K_E*L),grad(f_next)/L)",
            "storage_formula": "V=[z,u]^T(P tensor I)[z,u]+c_F*(f(W)-f_star)/L",
            "storage_P": _matrix_payload(certificate.storage),
            "function_storage_c_F": _fraction_payload(certificate.function_storage),
            "storage_normalization_trace_P_plus_c_F": _fraction_payload(
                sum(certificate.storage[index][index] for index in range(2))
                + certificate.function_storage
            ),
            "multipliers": {
                "interpolation_current_to_next": _fraction_payload(
                    certificate.lambda_interpolation_12
                ),
                "interpolation_next_to_current": _fraction_payload(
                    certificate.lambda_interpolation_21
                ),
                "pl_at_next": _fraction_payload(certificate.lambda_pl_next),
                "residual_lipschitz": _fraction_payload(certificate.lambda_residual_norm),
            },
            "transition_matrix": _matrix_payload(matrices.transition),
            "state_selection_matrix": _matrix_payload(matrices.selection),
            "signal_selector": _vector_payload(matrices.signal_selector),
            "step_selector": _vector_payload(matrices.step_selector),
            "interpolation_quadratic_matrices": {
                "current_to_next": _matrix_payload(matrices.interpolation_12),
                "next_to_current": _matrix_payload(matrices.interpolation_21),
            },
            "pl_next_quadratic_matrix": _matrix_payload(matrices.pl_next),
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
        "consequence": {
            "function_value_bound": "f(W_t)-f_star <= (L*V_0/c_F)*q^t",
            "momentum": "m_t -> 0 geometrically",
            "gradient": "grad(f(W_t)) -> 0 geometrically",
            "iterate": (
                "W_t converges to some trajectory-dependent global minimizer because its "
                "increments are geometrically summable"
            ),
            "uniqueness": "not assumed and not concluded",
        },
        "comparison_and_gate": {
            "p5_learning_rate": _fraction_payload(LOCKED_NONQUADRATIC_CONVERGENCE_LEARNING_RATE),
            "full_p5_step_retained": (
                certificate.learning_rate == LOCKED_NONQUADRATIC_CONVERGENCE_LEARNING_RATE
            ),
            "p5_norm_rate_tau": _fraction_payload(LOCKED_NONQUADRATIC_CONVERGENCE_TAU),
            "p6_rate_decrement_as_fraction_of_p5": _fraction_payload(
                (1 - certificate.tau) / (1 - LOCKED_NONQUADRATIC_CONVERGENCE_TAU)
            ),
            "declared_gate": "full p5 step retained",
            "classification": "major p6 result",
        },
        "independent_audit": {
            "status": "passed",
            "review_type": (
                "automated exact-algebra cross-check in an independent agent session; not a "
                "human proof audit and not a separately committed reconstruction"
            ),
            "review_date": "2026-09-02",
            "human_proof_audit": "pending",
            "items": [
                "nonconvex (-1,1) directed interpolation formula and signs",
                "normalized PL convention and next-point supply sign",
                "pinned EMA/Nesterov and physical residual scaling",
                "exact objective-value flow cancellation",
                "positive storage and negative 4x4 LMI by exact rational reconstruction",
                "function-value rather than arbitrary-pair incremental scope",
            ],
        },
        "p5_checkpoint": {
            "tag": P5_CHECKPOINT_TAG,
            "annotated_tag_git_oid": P5_CHECKPOINT_TAG_OBJECT,
            "peeled_commit": P5_CHECKPOINT_COMMIT,
            "full_step_manifest_path": P5_FULL_STEP_RESULT_PATH,
            "full_step_manifest_sha256": _sha256(P5_FULL_STEP_RESULT_PATH),
            "incremental_manifest_path": P5_INCREMENTAL_RESULT_PATH,
            "incremental_manifest_sha256": _sha256(P5_INCREMENTAL_RESULT_PATH),
            "status": "frozen and unchanged; p6 is additive on a separate branch",
        },
        "proof_replay_provenance": {
            "arithmetic": "fractions.Fraction exact rational arithmetic",
            "analytic_seed": None,
            "discovery_only_solver": {
                "role": (
                    "used only to discover a candidate; solver feasibility and floating-point "
                    "eigenvalues are nonauthoritative"
                ),
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
