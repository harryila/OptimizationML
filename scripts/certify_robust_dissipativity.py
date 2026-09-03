#!/usr/bin/env python3
"""Replay the exact full-step robust dissipativity certificate."""

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

from passive_muon.floored_certificate import certified_dimension_uniform_deficit
from passive_muon.robust_dissipativity import (
    audit_robust_dissipativity,
    locked_robust_dissipativity_certificate,
    robust_dissipativity_matrices,
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
P6_SOURCE_COMMIT = "a8f650f6c60dcbc5d2f83647fd367348f4c67548"
P6_CHECKPOINT_COMMIT = "ef88d8f5b26148af0ec1ca70b506048938bf9bef"
P6_CHECKPOINT_TAG = "p6-checkpoint"
P6_CHECKPOINT_TAG_OBJECT = "678e4ead01732c2a0d8db786147a063f0ae119ec"
P6_RESULT_PATH = "results/summaries/pl_convergence_certificate.json"
SOURCE_PATHS = (
    "scripts/certify_robust_dissipativity.py",
    "scripts/reconstruct_robust_dissipativity.py",
    "src/passive_muon/robust_dissipativity.py",
    "src/passive_muon/pl_convergence.py",
    "src/passive_muon/structure_aware_stability.py",
    "src/passive_muon/floored_certificate.py",
    "src/passive_muon/momentum_iqc.py",
    "src/passive_muon/normalizers.py",
    "src/passive_muon/operator.py",
    "src/passive_muon/orthogonalizers.py",
    "src/passive_muon/polynomials.py",
    "src/passive_muon/specs.py",
    "src/passive_muon/upstream_momentum.py",
    "tests/test_robust_dissipativity.py",
    "tests/test_robust_dissipativity_cli.py",
    "tests/test_robust_dissipativity_reconstruction.py",
    "theory/robust_dissipativity_certificate.md",
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
    }


def build_payload() -> dict[str, Any]:
    certificate = locked_robust_dissipativity_certificate()
    base = certificate.pl_certificate
    audit = audit_robust_dissipativity(certificate)
    if not audit.certified:
        raise AssertionError("exact robust dissipativity replay failed")

    matrices = robust_dissipativity_matrices(certificate)
    coefficients = JORDAN_QUINTIC.fractions()
    one_minus_rate = 1 - certificate.rate
    return {
        "schema_version": "passive-muon-robust-dissipativity-certificate-v1",
        "claim_scope": {
            "evidence_kind": (
                "exact rational dimension-independent pathwise dissipation certificate"
            ),
            "guarantee": (
                "V_(t+1) <= q*V_t + gamma_g*||xi_t||_F^2 + "
                "gamma_R*||e_t||_F^2 for every disturbance realization"
            ),
            "iss_scope": (
                "global storage/output ISS for objective gap, momentum, and true gradient; "
                "not full-state ISS in W when the minimizer set is non-singleton"
            ),
            "matrix_domain": "R^(m x n) for every fixed finite positive m,n",
            "objective_domain": (
                "every fixed differentiable globally 10-smooth scalar objective with finite "
                "infimum satisfying the global PL inequality with constant 1; nonconvex and "
                "nonunique-minimizer objectives are included"
            ),
            "initialization": "arbitrary finite same-shaped W_0 and m_0",
            "arithmetic": "exact real arithmetic in the optimizer and Fraction proof replay",
            "not_claimed": [
                "stability of the unrepaired exact-current-normalized Muon operator",
                "a convergence theorem for upstream BF16 or stochastic neural-network training",
                "that arbitrary BF16 roundoff obeys the assumed output-error bound",
                "independence, zero mean, or a distribution for the pathwise inequality",
                "iterate convergence under merely square-summable disturbances",
                "a unique minimizer or arbitrary-pair incremental contraction",
                "necessity or optimality of either disturbance gain",
            ],
        },
        "operator": {
            "formula": "R(M)=H_h(M/max(c,||M||_F))+rho*M",
            "domain": "every fixed finite real matrix shape",
            "normalization_rule": "M/max(c,||M||_F), exact fixed Frobenius max floor",
            "floor_c": _fraction_payload(LOCKED_FLOOR),
            "epsilon": "none; this is a max-floor normalizer, not additive epsilon",
            "orthogonalizer": JORDAN_QUINTIC.name,
            "polynomial_formula": "h(x)=a*x+b*x^3+c*x^5",
            "polynomial_coefficients_exact": {
                "a": str(coefficients[0]),
                "b": str(coefficients[1]),
                "c": str(coefficients[2]),
            },
            "newton_schulz_iteration_count": 5,
            "repair_deficit_upper": _fraction_payload(certified_dimension_uniform_deficit()),
            "repair_margin_mu": _fraction_payload(LOCKED_REPAIR_MARGIN),
            "constant_repair_rho": _fraction_payload(LOCKED_REPAIR_RHO),
            "upstream_revision": PINNED_KELLER_JORDAN_MUON_REVISION,
        },
        "disturbed_update": {
            "gradient_measurement": "g_hat_t=grad_f(W_t)+xi_t",
            "momentum": "m_(t+1)=beta*m_t+(1-beta)*g_hat_t",
            "nesterov_signal": "s_(t+1)=beta*m_(t+1)+(1-beta)*g_hat_t",
            "parameter_update": "W_(t+1)=W_t-eta*(R(s_(t+1))+e_t)",
            "reuse_rule": (
                "the identical g_hat_t, hence the identical xi_t, is reused in both the EMA "
                "and Nesterov occurrences"
            ),
            "implementation_error_location": "additive at the output of R, before scaling by eta",
            "disturbance_assumption_for_pathwise_result": (
                "xi_t and e_t are arbitrary finite same-shape real matrices"
            ),
        },
        "normalization": {
            "z": "m/L",
            "u": "grad(f(W))/L (true gradient, not g_hat/L)",
            "v": "E(s)/(K_E*L)",
            "u_next": "grad(f(W_next))/L",
            "w": "xi/L",
            "h": "e/(gamma*L)",
            "lifted_coordinates": "chi=(z,u,v,u_next,w,h)",
        },
        "locked_exact_certificate": {
            "learning_rate_eta": _fraction_payload(base.learning_rate),
            "beta": _fraction_payload(base.beta),
            "smoothness_L": _fraction_payload(base.smoothness),
            "pl_constant_ell_PL": _fraction_payload(base.pl_constant),
            "center_gain_gamma": _fraction_payload(base.center_gain),
            "residual_lipschitz_K_E": _fraction_payload(base.residual_lipschitz),
            "rate_q": _fraction_payload(certificate.rate),
            "one_minus_q": _fraction_payload(one_minus_rate),
            "physical_gains": {
                "gamma_g": _fraction_payload(certificate.gradient_noise_gain),
                "gamma_R": _fraction_payload(certificate.implementation_error_gain),
            },
            "normalized_penalties": {
                "w_squared": _fraction_payload(certificate.normalized_gradient_noise_penalty),
                "h_squared": _fraction_payload(certificate.normalized_implementation_error_penalty),
                "conversion": (
                    "gamma_g*L^2*||w||^2=gamma_g*||xi||^2 and "
                    "gamma_R*gamma^2*L^2*||h||^2=gamma_R*||e||^2"
                ),
            },
            "storage_formula": ("V=[z,u]^T(P tensor I)[z,u]+c_F*(f(W)-f_star)/L"),
            "storage_P": _matrix_payload(base.storage),
            "function_storage_c_F": _fraction_payload(base.function_storage),
            "multipliers": {
                "interpolation_current_to_next": _fraction_payload(base.lambda_interpolation_12),
                "interpolation_next_to_current": _fraction_payload(base.lambda_interpolation_21),
                "pl_at_next": _fraction_payload(base.lambda_pl_next),
                "residual_lipschitz": _fraction_payload(base.lambda_residual_norm),
            },
            "transition_matrix": _matrix_payload(matrices.transition),
            "state_selection_matrix": _matrix_payload(matrices.selection),
            "noisy_gradient_selector": _vector_payload(matrices.noisy_gradient_selector),
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
            "function_coefficients": {
                "actual": [
                    _fraction_payload(value) for value in audit.supply_function_coefficients
                ],
                "expected": [
                    _fraction_payload(value) for value in audit.expected_function_coefficients
                ],
                "cancel_exactly": audit.function_values_cancel,
            },
            "storage_positive_definite": audit.storage_positive_definite,
            "lmi_negative_definite": audit.lmi_negative_definite,
            "all_exact_checks_passed": audit.certified,
        },
        "consequences": {
            "bounded_deterministic_inputs": (
                "if ||xi_t||^2<=B_g and ||e_t||^2<=B_R, then "
                "V_t<=q^t*V_0+(gamma_g*B_g+gamma_R*B_R)*(1-q^t)/(1-q)"
            ),
            "ultimate_storage_bound": ("limsup V_t <= (gamma_g*B_g+gamma_R*B_R)/(1-q)"),
            "ultimate_storage_coefficients_for_norm_bounds": {
                "gradient_noise_X_squared": _fraction_payload(
                    certificate.gradient_noise_gain / one_minus_rate
                ),
                "implementation_error_E_squared": _fraction_payload(
                    certificate.implementation_error_gain / one_minus_rate
                ),
            },
            "square_summable_inputs": (
                "sum_t(||xi_t||^2+||e_t||^2)<infinity implies V_t->0, hence "
                "f(W_t)-f_star->0, m_t->0, and grad(f(W_t))->0"
            ),
            "iterate_qualification": (
                "square-summable disturbances alone do not imply convergence of W_t; absolute "
                "summability of xi_t and e_t, together with the certified decay, is sufficient "
                "for summable parameter increments and iterate convergence"
            ),
            "bounded_second_moment_stochastic": (
                "taking expectations gives E[V_(t+1)]<=q*E[V_t]+"
                "gamma_g*E||xi_t||^2+gamma_R*E||e_t||^2; bounded second moments give "
                "an expected ultimate neighborhood"
            ),
            "stochastic_assumptions": (
                "(W_0,m_0) is F_0-measurable with E[V_0]<infinity; the current state is "
                "F_t-measurable; xi_t and e_t are F_(t+1)-measurable; their conditional "
                "second moments given F_t obey the displayed bounds"
            ),
            "unbiasedness_note": (
                "conditional unbiasedness may be imposed for a standard stochastic-gradient "
                "interpretation and then the gradient second moment may be called a variance "
                "bound, but unbiasedness is not required by the stronger pathwise inequality"
            ),
        },
        "p6_provenance": {
            "source_generation_commit": P6_SOURCE_COMMIT,
            "final_checkpoint_commit": P6_CHECKPOINT_COMMIT,
            "checkpoint_tag": P6_CHECKPOINT_TAG,
            "annotated_tag_git_oid": P6_CHECKPOINT_TAG_OBJECT,
            "result_manifest_path": P6_RESULT_PATH,
            "result_manifest_sha256": _sha256(P6_RESULT_PATH),
            "relationship": (
                "P7 retains the exact P6 storage, supplies, rate, and full learning rate and "
                "adds two disturbance coordinates plus exact squared-input penalties"
            ),
        },
        "proof_replay_provenance": {
            "arithmetic": "fractions.Fraction exact rational arithmetic",
            "numeric_solver_role": "none in replay; any discovery solve is nonauthoritative",
            "source_snapshot": _source_snapshot(),
            "software": {"python": sys.version},
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
