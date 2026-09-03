#!/usr/bin/env python3
"""Generate the exact P11 two-port implementation-margin certificate."""

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

import torch

from passive_muon.implementation_margin_certificate import (
    IMPLEMENTATION_MARGIN_SCHEMA_VERSION,
    LOCKED_BUDGET_GRID_DENOMINATOR,
    P10_DIAGNOSTIC_CHECKPOINT_COMMIT,
    P10_EXACT_ARTIFACT_COMMIT,
    P10_THEOREM_SOURCE_COMMIT,
    FrontierRow,
    GridBoundary,
    ImplementationMarginBudget,
    ImplementationMarginEvaluation,
    audit_implementation_margin,
)
from passive_muon.specs import JORDAN_QUINTIC
from passive_muon.structure_aware_stability import LOCKED_FLOOR, LOCKED_REPAIR_RHO
from passive_muon.upstream_momentum import (
    PINNED_KELLER_JORDAN_MUON_REVISION,
    PINNED_MUON_PY_SHA256,
)

ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = "results/summaries/implementation_margin_certificate.json"
SOURCE_PATHS = (
    "scripts/certify_implementation_margin.py",
    "scripts/reconstruct_implementation_margin.py",
    "src/passive_muon/implementation_margin_certificate.py",
    "src/passive_muon/outer_loop_roundoff_certificate.py",
    "src/passive_muon/finite_precision_outer_loop.py",
    "src/passive_muon/scalable_mixed_precision_certificate.py",
    "src/passive_muon/robust_dissipativity.py",
    "src/passive_muon/specs.py",
    "src/passive_muon/structure_aware_stability.py",
    "src/passive_muon/upstream_momentum.py",
    "tests/test_implementation_margin_certificate.py",
    "tests/test_implementation_margin_cli.py",
    "tests/test_implementation_margin_reconstruction.py",
    "tests/test_result_manifests.py",
    "theory/implementation_margin_certificate.md",
    "theory/audits/P11_HUMAN_PROOF_AUDIT.md",
    "results/summaries/P11_RESULTS.md",
    ".github/workflows/p11-implementation-margin.yml",
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


def _fraction(value: Fraction) -> dict[str, str]:
    return {"exact": str(value), "decimal": _decimal(value)}


def _budget(budget: ImplementationMarginBudget) -> dict[str, dict[str, str]]:
    return {name: _fraction(value) for name, value in budget.__dict__.items()}


def _evaluation(evaluation: ImplementationMarginEvaluation) -> dict[str, Any]:
    reduction = evaluation.reduction
    guard = evaluation.guard
    function_gap = evaluation.function_gap_ultimate
    return {
        "budget": _budget(reduction.budget),
        "q11": _fraction(evaluation.rate),
        "D11": _fraction(evaluation.constant_forcing),
        "one_minus_q11_minus_D11": _fraction(1 - evaluation.rate - evaluation.constant_forcing),
        "function_gap_ultimate": None if function_gap is None else _fraction(function_gap),
        "augmented_envelopes": {
            name: {
                "slope": _fraction(getattr(reduction, name).slope),
                "intercept": _fraction(getattr(reduction, name).intercept),
            }
            for name in (
                "momentum_port",
                "signal_port",
                "actual_signal",
                "p9_reference_output",
                "deployed_output",
                "parameter_port",
                "effective_gradient_error",
                "effective_operator_error",
            )
        },
        "guards_at_V_le_1": {
            "signal_frobenius_bound": _fraction(guard.signal_at_one),
            "signal_max_abs_limit": _fraction(guard.signal_max_abs),
            "p9_reference_output_frobenius_bound": _fraction(guard.p9_reference_output_at_one),
            "deployed_output_frobenius_bound": _fraction(guard.deployed_output_at_one),
            "certificate_output_max_abs": _fraction(guard.certificate_output_max_abs),
            "runtime_output_max_abs": _fraction(guard.runtime_output_max_abs),
            "rounded_step_entrywise_bound": _fraction(guard.rounded_step_at_one),
            "rounded_step_max_abs": _fraction(guard.rounded_step_max_abs),
            "ema_intermediate_frobenius_bounds": {
                name: _fraction(value)
                for name, value in guard.ema_intermediate_frobenius_bounds.items()
            },
            "ema_intermediates_are_finite": guard.ema_intermediates_are_finite,
            "middle_low_invariant_checks": guard.middle_low_invariant_checks,
            "high_word_guard": "conditional |high|<=2^30 premise, rechecked per call",
        },
        "checks": evaluation.checks,
        "certified": evaluation.certified,
    }


def _ticks(value: Fraction) -> int:
    ticks = value * LOCKED_BUDGET_GRID_DENOMINATOR
    if ticks.denominator != 1:
        raise AssertionError("reported boundary value is not on the locked grid")
    return ticks.numerator


def _boundary(boundary: GridBoundary) -> dict[str, Any]:
    return {
        "axis": boundary.varied_axis,
        "fixed_budget": _budget(boundary.fixed_budget),
        "largest_certified_grid_value": _fraction(boundary.accepted_value),
        "largest_certified_grid_ticks": _ticks(boundary.accepted_value),
        "adjacent_rejected_grid_value": _fraction(boundary.rejected_value),
        "adjacent_rejected_grid_ticks": _ticks(boundary.rejected_value),
        "grid_step": _fraction(boundary.grid_step),
        "accepted_q11": _fraction(boundary.accepted_rate),
        "accepted_D11": _fraction(boundary.accepted_forcing),
        "accepted_invariance_slack": _fraction(boundary.accepted_invariance_slack),
        "rejected_invariance_slack": _fraction(boundary.rejected_invariance_slack),
        "active_constraint": boundary.active_constraint,
        "adjacent_control_failed_checks": list(boundary.rejected_checks),
        "negative_control_interpretation": (
            "rejection of this sufficient certificate, not a dynamical-instability witness"
        ),
    }


def _frontier_row(row: FrontierRow) -> dict[str, Any]:
    return {
        "gradient_axis": row.gradient_axis,
        "operator_axis": row.operator_axis,
        "gradient_value": _fraction(row.gradient_value),
        "gradient_grid_ticks": _ticks(row.gradient_value),
        "adjacent_rejected_gradient_value": _fraction(row.rejected_gradient_value),
        "adjacent_rejected_gradient_grid_ticks": _ticks(row.rejected_gradient_value),
        "largest_certified_operator_value": _fraction(row.operator_value),
        "operator_grid_ticks": _ticks(row.operator_value),
        "adjacent_rejected_operator_value": _fraction(row.rejected_operator_value),
        "adjacent_rejected_operator_grid_ticks": _ticks(row.rejected_operator_value),
        "q11": _fraction(row.rate),
        "D11": _fraction(row.forcing),
        "invariance_slack": _fraction(row.invariance_slack),
        "deployed_output_at_V_le_1": _fraction(row.deployed_output_at_one),
        "adjacent_gradient_control_invariance_slack": _fraction(
            row.rejected_gradient_invariance_slack
        ),
        "adjacent_operator_control_invariance_slack": _fraction(
            row.rejected_operator_invariance_slack
        ),
        "adjacent_gradient_control_failed_checks": list(row.rejected_gradient_checks),
        "adjacent_control_failed_checks": list(row.rejected_checks),
    }


def _sha256(relative_path: str) -> str:
    return hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()


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


def build_payload() -> dict[str, Any]:
    """Return the self-contained exact P11 result payload."""

    audit = audit_implementation_margin()
    if not audit.certified:
        failures = [name for name, passed in audit.checks.items() if not passed]
        raise AssertionError(f"exact P11 replay failed: {failures}")
    p10 = audit.p10
    coefficients = JORDAN_QUINTIC.fractions()
    sensitivity = audit.sensitivities
    return {
        "schema_version": IMPLEMENTATION_MARGIN_SCHEMA_VERSION,
        "claim_scope": {
            "evidence_kind": (
                "exact rational composition of the frozen P10 certificate on a 2^-40 "
                "acceptance-budget grid"
            ),
            "guarantee": (
                "for every external error realization satisfying the displayed affine "
                "Frobenius budgets, V_(t+1)<=q11*V_t+D11; q11<1 and D11<=1-q11 "
                "preserve V<=1 while all displayed guards hold"
            ),
            "objective_domain": (
                "differentiable globally 10-smooth objectives satisfying the global PL "
                "inequality with constant 1; nonconvex objectives and nonunique minimizers "
                "are included"
            ),
            "matrix_domain": "the locked 4096x11008 P10 proof-reference shell",
            "normalization": "exact target s/max(1,||s||_F), no additive epsilon",
            "budget_maximum_scope": (
                "largest accepted nonnegative multiple of 2^-40 on each reported axis or "
                "frontier ray; not an unrestricted-rational maximum"
            ),
            "not_claimed": [
                "measurement of a production model, gradient implementation, or CUDA kernel",
                "literal upstream Muon, current-plus-epsilon normalization, or an unrepaired map",
                "actual instability immediately outside a certified budget",
                "an unconditional invariant for the high master word",
                "model-forward parity when only one master word is consumed",
                "full-parameter ISS, a unique minimizer, or arbitrary-pair contraction",
                "tightness of P7 gains, P10 envelopes, or P11 margins",
            ],
        },
        "external_port_contract": {
            "pre_cast_gradient": "y_t=grad f(W_t)+zeta_t; g_hat_t=C32(y_t)",
            "gradient_budget": "||zeta_t||_F<=a_g*sqrt(V_t)+b_g",
            "p9_reference_output": "U9_t=Rhat_P9(s_(t+1))",
            "deployed_output": (
                "Udep_t is a finite contiguous FP32 tensor and nu_t=Udep_t-U9_t over the reals"
            ),
            "operator_budget": "||nu_t||_F<=a_R*sqrt(V_t)+b_R",
            "no_executed_nu_addition": True,
            "weight_error_conversion": (
                "zeta has gradient units; under 10-smoothness, a reconstructed-weight error "
                "dW contributes at most 10*||dW||_F before any other gradient error"
            ),
        },
        "port_reduction": {
            "momentum_residual_definition": ("r_m=epsilon_m+(1-beta)*(C32(g+zeta)-(g+zeta))"),
            "signal_residual_definition": ("r_s=epsilon_s+(1-beta)*(C32(g+zeta)-(g+zeta))"),
            "effective_gradient_error": "xi_eff=zeta+r_m/(1-beta)",
            "exact_signal_cancellation": (
                "implemented signal minus P7 nominal signal equals r_s-r_m; the explicit "
                "zeta term cancels"
            ),
            "effective_operator_error": ("e_eff=R(s)-R(s0)+(U9-R(s))+nu-r_W/eta"),
            "rounding_rule": (
                "add exact incremental sensitivities to frozen safe P10 envelopes, then round "
                "each augmented slope/intercept and q11,D11 upward once on 2^-40"
            ),
            "total_square_rule": (
                "retain the full augmented Xi and E squares, including base/external and "
                "gradient/operator cross terms"
            ),
        },
        "locked_model": {
            "shape": [p10.reduction.shape.rows, p10.reduction.shape.columns],
            "beta": _fraction(p10.beta),
            "learning_rate_eta": _fraction(p10.learning_rate),
            "floor_c": _fraction(LOCKED_FLOOR),
            "epsilon": "none; exact max-floor normalization",
            "repair_rho": _fraction(LOCKED_REPAIR_RHO),
            "orthogonalizer": JORDAN_QUINTIC.name,
            "polynomial_coefficients_exact": {
                "a": str(coefficients[0]),
                "b": str(coefficients[1]),
                "c": str(coefficients[2]),
            },
            "newton_schulz_iteration_count": 5,
            "gradient_young": p10.gradient_young,
            "operator_young": p10.operator_young,
            "upstream_revision": PINNED_KELLER_JORDAN_MUON_REVISION,
            "upstream_muon_sha256": PINNED_MUON_PY_SHA256,
        },
        "p10_provenance_roles": {
            "theorem_source_commit": P10_THEOREM_SOURCE_COMMIT,
            "exact_artifact_recorded_commit": P10_EXACT_ARTIFACT_COMMIT,
            "diagnostic_head_checkpoint_commit": P10_DIAGNOSTIC_CHECKPOINT_COMMIT,
            "human_audit_status": "pending; no external sign-off is claimed",
        },
        "exact_external_sensitivities": {
            name: _fraction(value) for name, value in sensitivity.__dict__.items()
        },
        "zero_external_error_replay": _evaluation(audit.zero_error),
        "one_axis_grid_maxima": {
            axis: _boundary(boundary) for axis, boundary in audit.axis_boundaries.items()
        },
        "pareto_frontiers": {
            "interpretation": (
                "two exact two-dimensional slices of the four-dimensional acceptance set; "
                "each row is coordinatewise maximal and each adjacent gradient/operator "
                "control is one 2^-40 tick outside"
            ),
            "slope_slice_b_g_eq_b_R_eq_0": [_frontier_row(row) for row in audit.slope_frontier],
            "intercept_slice_a_g_eq_a_R_eq_0": [
                _frontier_row(row) for row in audit.intercept_frontier
            ],
        },
        "jointly_nonzero_subunit_profile": _evaluation(audit.jointly_nonzero),
        "audit": {
            "checks": audit.checks,
            "all_exact_checks_passed": audit.certified,
        },
        "proof_replay_provenance": {
            "seed": None,
            "arithmetic": "fractions.Fraction exact rational arithmetic",
            "budget_grid_denominator": LOCKED_BUDGET_GRID_DENOMINATOR,
            "python": sys.version,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "source_snapshot": {path: _sha256(path) for path in SOURCE_PATHS},
        },
        "git": _git_state(),
        "canonical_result_path": RESULT_PATH,
    }


def main() -> None:
    args = parse_args()
    payload = build_payload()
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
