#!/usr/bin/env python3
"""Replay the exact P10 finite-precision outer-loop certificate."""

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

from passive_muon.finite_precision_outer_loop import (
    BETA_FP32_BITS,
    FINITE_PRECISION_OUTER_LOOP_SCHEMA_VERSION,
    LEARNING_RATE_FP32_BITS,
    LOCKED_OUTER_BACKEND,
    ONE_MINUS_BETA_FP32_BITS,
    FinitePrecisionOuterLoopConfig,
    fp32_master_stalling_witness,
    outer_residual_envelope,
)
from passive_muon.outer_loop_roundoff_certificate import (
    AffineStorageEnvelope,
    audit_outer_loop_roundoff,
)
from passive_muon.specs import JORDAN_QUINTIC
from passive_muon.structure_aware_stability import LOCKED_FLOOR, LOCKED_REPAIR_RHO
from passive_muon.upstream_momentum import (
    PINNED_KELLER_JORDAN_MUON_REVISION,
    PINNED_MUON_PY_SHA256,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = "results/summaries/outer_loop_roundoff_certificate.json"
SOURCE_PATHS = (
    "scripts/certify_outer_loop_roundoff.py",
    "scripts/reconstruct_outer_loop_roundoff.py",
    "src/passive_muon/outer_loop_roundoff_certificate.py",
    "src/passive_muon/finite_precision_outer_loop.py",
    "src/passive_muon/scalable_mixed_precision.py",
    "src/passive_muon/scalable_mixed_precision_certificate.py",
    "src/passive_muon/mixed_precision_certificate.py",
    "src/passive_muon/floored_certificate.py",
    "src/passive_muon/robust_dissipativity.py",
    "src/passive_muon/pl_convergence.py",
    "src/passive_muon/specs.py",
    "src/passive_muon/structure_aware_stability.py",
    "src/passive_muon/upstream_momentum.py",
    "tests/test_outer_loop_roundoff_certificate.py",
    "tests/test_outer_loop_roundoff_cli.py",
    "tests/test_outer_loop_roundoff_reconstruction.py",
    "tests/test_finite_precision_outer_loop.py",
    "tests/test_finite_precision_outer_loop_diagnostic.py",
    "tests/test_result_manifests.py",
    "theory/finite_precision_outer_loop_certificate.md",
    "theory/audits/P10_HUMAN_PROOF_AUDIT.md",
    "results/summaries/P10_RESULTS.md",
    ".github/workflows/p10-finite-precision.yml",
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


def _envelope_payload(envelope: AffineStorageEnvelope) -> dict[str, dict[str, str]]:
    return {
        "slope": _fraction_payload(envelope.slope),
        "intercept": _fraction_payload(envelope.intercept),
    }


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
    return completed.stdout.strip() if completed.returncode == 0 else None


def _git_state() -> dict[str, object]:
    status = _git_output("status", "--porcelain")
    return {
        "sha": _git_output("rev-parse", "HEAD") or "unavailable",
        "branch": _git_output("branch", "--show-current") or "unavailable",
        "dirty": bool(status) if status is not None else None,
    }


def build_payload() -> dict[str, Any]:
    """Build the exact, self-contained P10 result payload."""

    audit = audit_outer_loop_roundoff()
    if not audit.certified:
        failures = [name for name, passed in audit.checks.items() if not passed]
        raise AssertionError(f"exact P10 replay failed: {failures}")
    certificate = audit.certificate
    reduction = certificate.reduction
    raw_outer = outer_residual_envelope(
        FinitePrecisionOuterLoopConfig((reduction.shape.rows, reduction.shape.columns))
    )
    guard = certificate.guard
    obstruction = audit.obstruction
    coefficients = JORDAN_QUINTIC.fractions()
    stalling_witness = fp32_master_stalling_witness()
    if not stalling_witness["certified"]:
        raise AssertionError("the actual-P9 FP32 stalling witness did not replay")

    envelopes = {
        "momentum_port_r_m": reduction.momentum_port,
        "signal_port_r_s": reduction.signal_port,
        "actual_fp32_signal": reduction.actual_signal,
        "p9_operator_output": reduction.operator_output,
        "master_update_port_r_W": reduction.parameter_port,
        "effective_p7_gradient_error": reduction.effective_gradient_error,
        "effective_p7_operator_error": reduction.effective_operator_error,
    }
    return {
        "schema_version": "passive-muon-outer-loop-roundoff-certificate-v1",
        "claim_scope": {
            "evidence_kind": (
                "exact rational compositional P7/P9/P10 certificate with upward rounding"
            ),
            "guarantee": (
                "at shape 4096x11008, V_(t+1)<=q10*V_t+D10; if V_0<=1 and the "
                "conditional high-word guard is rechecked, V_t<=1 and the objective-gap "
                "limsup is bounded by the displayed value below one"
            ),
            "objective_domain": (
                "differentiable globally 10-smooth objectives satisfying the global PL "
                "inequality with constant 1; nonconvex and nonunique minimizers are included"
            ),
            "gradient_boundary": (
                "the exact objective gradient at logical W=high+middle+low is rounded once "
                "entrywise to FP32 before the locked EMA/Nesterov graph"
            ),
            "matrix_domain": (
                "one fixed 4096x11008 CPU FP32 shell and P9 operator instance under every "
                "displayed finite-range guard"
            ),
            "normalization": "exact target s/max(1,||s||_F), no additive epsilon",
            "not_claimed": [
                "literal upstream Muon, current-plus-epsilon normalization, or an unrepaired map",
                "native BLAS, GPU, tensor-core, fused, or compiler-reassociated arithmetic",
                (
                    "rounding incurred while computing the real objective gradient before "
                    "its final cast"
                ),
                "model-forward parity when a model consumes only the high master word",
                "an unconditional invariant for the high master word",
                "full-parameter ISS or iterate convergence on nonunique PL minimizer sets",
                "a unique minimizer, arbitrary-pair contraction, or tightness of the gains",
            ],
        },
        "outer_update": {
            "gradient_boundary": "g_hat=C32(grad f(W)); W=high+middle+low over the reals",
            "raw_shell_ports": (
                "r_tilde_m and r_tilde_s are defined relative to represented g_hat"
            ),
            "total_theorem_ports": (
                "r_m=r_tilde_m+(1-beta)*(g_hat-grad f(W)); r_s=r_tilde_s+(1-beta)*(g_hat-grad f(W))"
            ),
            "momentum": "m_next=beta*m+(1-beta)*grad f(W)+r_m",
            "nesterov_signal": "s_next=beta*m_next+(1-beta)*grad f(W)+r_s",
            "master_update": "W_next=W-eta*Rhat(s_next)+r_W",
            "effective_gradient_error": "xi_eff=r_m/(1-beta)",
            "nominal_p7_signal": ("s0=beta^2*m+(1-beta^2)*grad f(W)+(1+beta)*r_m"),
            "signal_difference": "s_next-s0=r_s-r_m",
            "effective_operator_error": ("e_eff=R(s_next)-R(s0)+(Rhat(s_next)-R(s_next))-r_W/eta"),
        },
        "operator": {
            "formula": "R(M)=H_(q composed 5 times)(M/max(1,||M||_F))+rho*M",
            "domain": "fixed finite real 4096x11008 matrices under the P9/P10 guards",
            "normalization_rule": "M/max(c,||M||_F), exact fixed Frobenius max floor",
            "floor_c": _fraction_payload(LOCKED_FLOOR),
            "epsilon": "none; this is a max-floor normalizer, not additive epsilon",
            "orthogonalizer": JORDAN_QUINTIC.name,
            "polynomial_formula": "q(x)=a*x+b*x^3+c*x^5",
            "polynomial_coefficients_exact": {
                "a": str(coefficients[0]),
                "b": str(coefficients[1]),
                "c": str(coefficients[2]),
            },
            "newton_schulz_iteration_count": 5,
            "constant_repair_rho": _fraction_payload(LOCKED_REPAIR_RHO),
            "upstream_revision": PINNED_KELLER_JORDAN_MUON_REVISION,
            "upstream_muon_sha256": PINNED_MUON_PY_SHA256,
        },
        "arithmetic_contract": {
            "outer_schema": FINITE_PRECISION_OUTER_LOOP_SCHEMA_VERSION,
            "outer_backend": LOCKED_OUTER_BACKEND,
            "rounding": "IEEE-754 roundTiesToEven at every named FP32 operation",
            "gradual_underflow_required": True,
            "ftz_daz_allowed": False,
            "fma_allowed": False,
            "reassociation_allowed": False,
            "seed": "none; deterministic exact certificate and executable witness",
            "raw_port_formulas": {
                "r_tilde_m": ("||r_tilde_m||_F<=C_beta||m||_F+C_g||g_hat||_F+b_ema"),
                "r_tilde_s": ("||r_tilde_s||_F<=C_beta||m_next||_F+C_g||g_hat||_F+b_ema"),
                "gradient_boundary": ("||g_hat-grad f(W)||_F<=u||grad f(W)||_F+ceil(sqrt(n))*tau"),
                "r_W": ("||r_W||_F<=C_eta||operator_output||_F+u||low||_F+b_master"),
            },
        },
        "locked_runtime_scalars": {
            "beta_source": _fraction_payload(certificate.beta),
            "beta_fp32": _fraction_payload(reduction.runtime_beta),
            "beta_fp32_bits_hex": f"0x{BETA_FP32_BITS:08x}",
            "one_minus_beta_source": _fraction_payload(1 - certificate.beta),
            "one_minus_beta_fp32": _fraction_payload(reduction.runtime_one_minus_beta),
            "one_minus_beta_fp32_bits_hex": f"0x{ONE_MINUS_BETA_FP32_BITS:08x}",
            "learning_rate_source": _fraction_payload(certificate.learning_rate),
            "learning_rate_fp32": _fraction_payload(reduction.runtime_learning_rate),
            "learning_rate_fp32_bits_hex": f"0x{LEARNING_RATE_FP32_BITS:08x}",
        },
        "port_reduction": {
            "shape": [reduction.shape.rows, reduction.shape.columns],
            "sqrt_entries_upper": reduction.sqrt_entries_upper,
            "raw_ema_momentum_coefficient": _fraction_payload(reduction.ema_momentum_coefficient),
            "raw_ema_gradient_coefficient": _fraction_payload(reduction.ema_gradient_coefficient),
            "raw_ema_crumb": _fraction_payload(reduction.ema_crumb),
            "raw_master_output_coefficient": _fraction_payload(raw_outer.master_output_coefficient),
            "raw_master_low_word_coefficient": _fraction_payload(
                raw_outer.master_low_word_coefficient
            ),
            "raw_master_crumb": _fraction_payload(raw_outer.master_absolute_crumb),
            "raw_master_absolute_part_under_low_guard": _fraction_payload(
                raw_outer.master_absolute_part_under_low_guard
            ),
            "raw_master_absolute_bound_under_output_and_low_guards": _fraction_payload(
                raw_outer.master_absolute_bound_under_output_and_low_guards
            ),
            "raw_master_absolute_bound_under_certificate_guards": _fraction_payload(
                raw_outer.master_absolute_bound_under_certificate_guards
            ),
            "represented_momentum_coefficient": _fraction_payload(
                reduction.represented_momentum_coefficient
            ),
            "represented_gradient_coefficient": _fraction_payload(
                reduction.represented_gradient_coefficient
            ),
            "p9_binary32_operator_error": {
                "slope": _fraction_payload(reduction.p9_binary32_slope),
                "intercept": _fraction_payload(reduction.p9_binary32_intercept),
            },
            "ideal_repaired_lipschitz": _fraction_payload(reduction.repaired_lipschitz),
            "affine_storage_envelopes": {
                name: _envelope_payload(envelope) for name, envelope in envelopes.items()
            },
            "form": "every envelope is norm<=slope*sqrt(V)+intercept",
            "upper_grid_denominator": str(2**40),
        },
        "locked_certificate": {
            "p7_rate": _fraction_payload(certificate.p7_rate),
            "p7_gradient_gain": _fraction_payload(certificate.p7_gradient_gain),
            "p7_operator_gain": _fraction_payload(certificate.p7_operator_gain),
            "gradient_young": certificate.gradient_young,
            "operator_young": certificate.operator_young,
            "rate_q10": _fraction_payload(certificate.rate),
            "constant_forcing_D10": _fraction_payload(certificate.constant_forcing),
            "one_minus_q10": _fraction_payload(1 - certificate.rate),
            "function_gap_ultimate": _fraction_payload(certificate.function_gap_ultimate),
            "storage_inequality": "V_next<=q10*V+D10",
        },
        "guard_closure": {
            "storage_radius": _fraction_payload(guard.storage_radius),
            "signal_at_radius": _fraction_payload(guard.signal_at_radius),
            "p9_signal_max_abs": _fraction_payload(guard.p9_signal_max_abs),
            "operator_output_at_radius": _fraction_payload(guard.operator_output_at_radius),
            "certificate_operator_output_max_abs": _fraction_payload(
                guard.certificate_operator_output_max_abs
            ),
            "runtime_operator_output_max_abs": _fraction_payload(
                guard.runtime_operator_output_max_abs
            ),
            "rounded_step_at_radius": _fraction_payload(guard.rounded_step_at_radius),
            "rounded_step_quantity": "entrywise maxabs",
            "rounded_step_max_abs": _fraction_payload(guard.rounded_step_max_abs),
            "master_word_max_abs": {
                "high": _fraction_payload(guard.master_high_max_abs),
                "middle": _fraction_payload(guard.master_middle_max_abs),
                "low": _fraction_payload(guard.master_low_max_abs),
            },
            "signal_capacity_radius": _fraction_payload(guard.signal_capacity_radius),
            "operator_capacity_radius": _fraction_payload(guard.operator_capacity_radius),
            "forcing_capacity_at_locked_radius": _fraction_payload(guard.forcing_capacity),
            "ema_intermediate_frobenius_bounds": {
                name: _fraction_payload(value)
                for name, value in guard.ema_intermediate_frobenius_bounds.items()
            },
            "fp32_max_finite": _fraction_payload(guard.fp32_max_finite),
            "ema_intermediates_are_finite": guard.ema_intermediates_are_finite,
            "middle_low_invariant_checks": guard.middle_low_invariant_checks,
            "high_word_guard": (
                "conditional and rechecked after every update; not implied by PL storage"
            ),
        },
        "flat_direction_obstruction": {
            "objective": obstruction.objective,
            "smoothness_upper": str(obstruction.smoothness_upper),
            "pl_constant": str(obstruction.pl_constant),
            "witness": (
                "at zero state, any sufficiently small fixed r_W,t=(0,epsilon) gives "
                "linear flat-direction drift; r_W,t=(0,epsilon/(t+1)) is square "
                "summable but gives harmonic drift for every epsilon>0"
            ),
            "objective_gap": str(obstruction.objective_gap),
            "gradient_norm": str(obstruction.gradient_norm),
            "momentum_norm": str(obstruction.momentum_norm),
            "bounded_error_causes_unbounded_parameter": (
                obstruction.bounded_error_causes_unbounded_parameter
            ),
            "square_summable_error_causes_parameter_drift": (
                obstruction.square_summable_error_causes_parameter_drift
            ),
        },
        "actual_p9_fp32_stalling_witness": stalling_witness,
        "audit": {
            "checks": audit.checks,
            "all_exact_checks_passed": audit.certified,
            "p7_exact_lmi_replayed": audit.p7_audit.certified,
            "p9_shape_certificate_replayed": audit.p9_audit.certified,
        },
        "proof_replay_provenance": {
            "arithmetic": (
                "fractions.Fraction exact arithmetic, integer square-root ceilings, and "
                "explicit upward/downward 2^-40 rationalization"
            ),
            "numeric_solver_role": "none",
            "source_snapshot": _source_snapshot(),
            "software": {"python": sys.version, "torch": str(torch.__version__)},
            "hardware": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
            },
        },
        "git": _git_state(),
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
