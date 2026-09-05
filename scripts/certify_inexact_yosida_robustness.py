#!/usr/bin/env python3
"""Generate the exact P15 inexact-Yosida robustness certificate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Any

from passive_muon.inexact_yosida_robustness import (
    FROZEN_ABSOLUTE_OUTPUT_PENALTY_THRESHOLD,
    LOCKED_ABSOLUTE_FORCING_GAIN,
    LOCKED_FAILING_FROZEN_RADIUS,
    LOCKED_PASSING_FROZEN_RADIUS,
    audit_inexact_yosida_robustness,
    double_relative_residual_instability_control,
    exact_certificate_checks,
    frozen_storage_radius_control,
    inexact_yosida_matrices,
    locked_inexact_resolvent_bounds,
    locked_inexact_yosida_certificate,
    scalar_residual_sharpness_control,
    unit_relative_residual_stall_control,
)
from passive_muon.specs import JORDAN_QUINTIC
from passive_muon.upstream_momentum import (
    PINNED_KELLER_JORDAN_MUON_REVISION,
    PINNED_MUON_PY_SHA256,
)

ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = "results/summaries/inexact_yosida_robustness_certificate.json"
SCHEMA_VERSION = "passive-muon-inexact-yosida-robustness-certificate-v1"

EPSILON = Fraction(1, 10_000_000)
INSUFFICIENT_ABSOLUTE_OUTPUT_PENALTY = Fraction(1, 1_000_000)
REJECTED_RELATIVE_RESIDUAL = Fraction(3, 500)

P14_ARTIFACT_PATH = "results/summaries/yosida_stability_certificate.json"
P14_ARTIFACT_SHA256 = "8f31416eb8e278bc8a3765f72b528f5f538e33f28d5baedc5888c5d849447c62"
P14_SOURCE_COMMIT = "ea18aa856e3af6d025e325a6db27a681f09c0e7a"
P14_ARTIFACT_COMMIT = "e323a4d1e73050fb6b6d8bcd09c515ea30de3553"
P14_CHECKPOINT_TAG = "p14-yosida-stability-checkpoint"
P14_CHECKPOINT_TAG_OBJECT = "986fade33e4bd90bed091798d2415d1e0ab608a2"

SOURCE_PATHS = (
    ".github/workflows/ci.yml",
    ".github/workflows/p15-inexact-yosida-robustness.yml",
    "scripts/certify_inexact_yosida_robustness.py",
    "scripts/reconstruct_inexact_yosida_robustness.py",
    "src/passive_muon/inexact_yosida_robustness.py",
    "src/passive_muon/yosida_stability.py",
    "src/passive_muon/pl_convergence.py",
    "src/passive_muon/momentum_iqc.py",
    "src/passive_muon/additive_epsilon_deficit.py",
    "src/passive_muon/radial_passivation_tradeoff.py",
    "src/passive_muon/specs.py",
    "src/passive_muon/structure_aware_stability.py",
    "src/passive_muon/upstream_momentum.py",
    "tests/test_inexact_yosida_robustness.py",
    "tests/test_inexact_yosida_robustness_cli.py",
    "tests/test_inexact_yosida_robustness_reconstruction.py",
    "tests/test_inexact_yosida_robustness_result_manifest.py",
    "tests/test_current_research_index.py",
    "theory/inexact_yosida_robustness_certificate.md",
    "theory/audits/P15_INEXACT_YOSIDA_ROBUSTNESS_HUMAN_PROOF_AUDIT.md",
    "results/summaries/P15_INEXACT_YOSIDA_ROBUSTNESS_RESULTS.md",
    "third_party/UPSTREAM_COMMITS.md",
    "pyproject.toml",
    "uv.lock",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _decimal(value: Fraction, *, digits: int = 60) -> str:
    with localcontext() as context:
        context.prec = digits
        return str(Decimal(value.numerator) / Decimal(value.denominator))


def _fraction(value: Fraction) -> dict[str, str]:
    return {"exact": str(value), "decimal": _decimal(value)}


def _matrix(matrix: tuple[tuple[Fraction, ...], ...]) -> list[list[str]]:
    return [[str(value) for value in row] for row in matrix]


def _vector(vector: tuple[Fraction, ...]) -> list[str]:
    return [str(value) for value in vector]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_output(*arguments: str) -> str | None:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _git_blob_sha256(commit: str, path: str) -> str:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"could not read frozen source {commit}:{path}")
    return hashlib.sha256(completed.stdout).hexdigest()


def _git_state() -> dict[str, object]:
    status = _git_output("status", "--porcelain")
    return {
        "sha": _git_output("rev-parse", "HEAD") or "unavailable",
        "branch": _git_output("branch", "--show-current") or "unavailable",
        "dirty": bool(status) if status is not None else None,
    }


def _source_snapshot() -> dict[str, str]:
    return {path: _sha256(ROOT / path) for path in SOURCE_PATHS}


def _prior_p14() -> tuple[dict[str, str], dict[str, bool]]:
    path = ROOT / P14_ARTIFACT_PATH
    observed_hash = _sha256(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    record = {
        "artifact_path": P14_ARTIFACT_PATH,
        "artifact_sha256": observed_hash,
        "source_commit": payload["git"]["sha"],
        "artifact_commit": P14_ARTIFACT_COMMIT,
        "checkpoint_tag": P14_CHECKPOINT_TAG,
        "checkpoint_tag_object": P14_CHECKPOINT_TAG_OBJECT,
    }
    checks = {
        "p14_artifact_hash_matches": observed_hash == P14_ARTIFACT_SHA256,
        "p14_historical_artifact_blob_matches": (
            _git_blob_sha256(P14_ARTIFACT_COMMIT, P14_ARTIFACT_PATH) == P14_ARTIFACT_SHA256
        ),
        "p14_source_commit_matches": payload["git"]["sha"] == P14_SOURCE_COMMIT,
        "p14_source_was_clean": payload["git"]["dirty"] is False,
        "p14_zero_error_rate_matches": (
            payload["exact_pl_certificate"]["q"]["exact"] == "249001/250000"
            and payload["exact_pl_certificate"]["D14"]["exact"] == "0"
        ),
        "p14_checkpoint_tag_matches": (
            _git_output("rev-parse", P14_CHECKPOINT_TAG) == P14_CHECKPOINT_TAG_OBJECT
            and _git_output("rev-parse", f"{P14_CHECKPOINT_TAG}^{{}}") == P14_ARTIFACT_COMMIT
        ),
    }
    return record, checks


def _reconstruction_fields() -> dict[str, object]:
    certificate = locked_inexact_yosida_certificate()
    base = certificate.pl_certificate
    audit = audit_inexact_yosida_robustness(certificate)
    matrices = inexact_yosida_matrices(certificate)
    bounds = locked_inexact_resolvent_bounds()
    radius_rejection = frozen_storage_radius_control(LOCKED_FAILING_FROZEN_RADIUS)
    insufficient = replace(
        certificate,
        absolute_output_penalty=INSUFFICIENT_ABSOLUTE_OUTPUT_PENALTY,
    )
    insufficient_audit = audit_inexact_yosida_robustness(insufficient)
    critical = replace(
        certificate,
        absolute_output_penalty=FROZEN_ABSOLUTE_OUTPUT_PENALTY_THRESHOLD,
    )
    critical_audit = audit_inexact_yosida_robustness(critical)
    stall = unit_relative_residual_stall_control()
    ascent = double_relative_residual_instability_control()
    ultimate_objective_gain = (
        base.smoothness * certificate.ultimate_storage_gain / base.function_storage
    )
    return {
        "parameters": {
            "epsilon": str(EPSILON),
            "lambda": str(bounds.resolvent_parameter),
            "mu": str(bounds.base_strong_monotonicity),
            "beta": str(base.beta),
            "eta": str(base.learning_rate),
            "smoothness": str(base.smoothness),
            "pl_constant": str(base.pl_constant),
            "tau": str(base.tau),
            "q": str(certificate.rate),
        },
        "residual_envelope": {
            "forward_strong_monotonicity": str(bounds.denominator),
            "equation_to_output_gain": str(bounds.resolvent_form_output_error_gain),
            "relative_residual": str(certificate.relative_residual_slope),
            "base_centered_radius": str(certificate.sector.residual_lipschitz),
            "relative_output_radius": str(
                certificate.output_error_gain * certificate.relative_residual_slope
            ),
            "effective_centered_radius": str(certificate.effective_residual_lipschitz),
            "absolute_output_from_rbar": str(certificate.output_error_gain),
        },
        "certificate": {
            "storage": _matrix(base.storage),
            "function_storage": str(base.function_storage),
            "interpolation_reverse": str(base.interpolation_reverse_weight),
            "lambda_12": str(base.lambda_interpolation_12),
            "lambda_21": str(base.lambda_interpolation_21),
            "lambda_pl_next": str(base.lambda_pl_next),
            "lambda_residual": str(base.lambda_residual_norm),
            "absolute_output_penalty": str(certificate.absolute_output_penalty),
            "normalized_absolute_output_penalty": str(
                certificate.normalized_absolute_output_penalty
            ),
            "absolute_forcing_gain": str(certificate.absolute_forcing_gain),
            "ultimate_storage_gain": str(certificate.ultimate_storage_gain),
            "ultimate_objective_gain": str(ultimate_objective_gain),
            "transition": _matrix(matrices.transition),
            "selection": _matrix(matrices.selection),
            "signal": _vector(matrices.signal_selector),
            "step": _vector(matrices.step_selector),
            "interpolation_12": _matrix(matrices.interpolation_12),
            "interpolation_21": _matrix(matrices.interpolation_21),
            "pl_next": _matrix(matrices.pl_next),
            "residual_supply": _matrix(matrices.residual_norm),
            "absolute_error_norm": _matrix(matrices.absolute_output_error_norm),
            "lmi": _matrix(audit.lmi),
            "storage_minors": [str(value) for value in audit.storage_leading_minors],
            "negative_lmi_minors": [str(value) for value in audit.negative_lmi_leading_minors],
            "function_coefficients": [str(value) for value in audit.supply_function_coefficients],
            "expected_function_coefficients": [
                str(value) for value in audit.expected_function_coefficients
            ],
        },
        "controls": {
            "rejected_relative_residual": str(REJECTED_RELATIVE_RESIDUAL),
            "rejected_effective_radius": str(LOCKED_FAILING_FROZEN_RADIUS),
            "rejected_radius_negative_minors": [
                str(value) for value in radius_rejection.audit.negative_lmi_leading_minors
            ],
            "critical_absolute_output_penalty": str(FROZEN_ABSOLUTE_OUTPUT_PENALTY_THRESHOLD),
            "insufficient_absolute_output_penalty": str(INSUFFICIENT_ABSOLUTE_OUTPUT_PENALTY),
            "insufficient_penalty_negative_minors": [
                str(value) for value in insufficient_audit.negative_lmi_leading_minors
            ],
            "critical_penalty_determinant": str(critical_audit.negative_lmi_leading_minors[-1]),
            "abstract_stall_kappa": str(stall.relative_tolerance),
            "abstract_stall_yosida_slope": str(stall.approximate_output_slope),
            "abstract_ascent_kappa": str(ascent.relative_tolerance),
            "abstract_ascent_yosida_slope": str(ascent.approximate_output_slope),
            "abstract_ascent_p_at_one": str(ascent.p_at_one),
        },
    }


def build_payload() -> dict[str, Any]:
    certificate = locked_inexact_yosida_certificate()
    base = certificate.pl_certificate
    bounds = locked_inexact_resolvent_bounds()
    audit = audit_inexact_yosida_robustness(certificate)
    core_checks = exact_certificate_checks()
    matrices = inexact_yosida_matrices(certificate)
    witness = scalar_residual_sharpness_control()
    radius_pass = frozen_storage_radius_control(LOCKED_PASSING_FROZEN_RADIUS)
    radius_fail = frozen_storage_radius_control(LOCKED_FAILING_FROZEN_RADIUS)
    insufficient = replace(
        certificate,
        absolute_output_penalty=INSUFFICIENT_ABSOLUTE_OUTPUT_PENALTY,
    )
    insufficient_audit = audit_inexact_yosida_robustness(insufficient)
    critical = replace(
        certificate,
        absolute_output_penalty=FROZEN_ABSOLUTE_OUTPUT_PENALTY_THRESHOLD,
    )
    critical_audit = audit_inexact_yosida_robustness(critical)
    stall = unit_relative_residual_stall_control()
    ascent = double_relative_residual_instability_control()
    prior_p14, prior_checks = _prior_p14()
    reconstruction = _reconstruction_fields()

    checks = {
        **prior_checks,
        **{f"core_{name}": value for name, value in core_checks.items()},
        "five_by_five_lmi_is_strict": audit.certified,
        "full_p14_step_and_rate_retained": (
            base.learning_rate == Fraction(1, 32_000)
            and certificate.rate == Fraction(249_001, 250_000)
        ),
        "absolute_forcing_gain_is_five_halves": (
            certificate.absolute_forcing_gain == LOCKED_ABSOLUTE_FORCING_GAIN
        ),
        "residual_gain_is_sharp_over_abstract_class": (
            witness.solution_error_ratio == bounds.solution_error_gain
            and witness.output_error_ratio == bounds.resolvent_form_output_error_gain
        ),
        "radius_252_passes": radius_pass.certificate_passes,
        "radius_253_rejects_frozen_certificate": not radius_fail.certificate_passes,
        "selected_absolute_penalty_passes": (
            certificate.absolute_output_penalty > FROZEN_ABSOLUTE_OUTPUT_PENALTY_THRESHOLD
        ),
        "critical_absolute_penalty_is_semidefinite_boundary": (
            critical_audit.negative_lmi_leading_minors[-1] == 0
        ),
        "insufficient_absolute_penalty_fails": (
            insufficient_audit.negative_lmi_leading_minors[-1] < 0
        ),
        "unit_relative_residual_stalls": stall.stalls and stall.p_at_one == 0,
        "double_relative_residual_is_unstable": (
            ascent.approximate_output_slope == -500 and ascent.p_at_one == Fraction(-1, 1_280)
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"P15 exact checks failed: {failed}")

    coefficients = JORDAN_QUINTIC.fractions()
    ultimate_objective_gain = (
        base.smoothness * certificate.ultimate_storage_gain / base.function_storage
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "canonical_result_path": RESULT_PATH,
        "claim_scope": {
            "evidence_kind": (
                "exact residual conversion and dimension-independent five-by-five "
                "smooth-PL dissipation certificate"
            ),
            "matrix_domain": "R^(m x n) for every fixed finite positive m,n",
            "objective_class": "differentiable globally 10-smooth, global PL constant 1",
            "arithmetic_model": "exact real arithmetic",
            "guarantee": (
                "V_(t+1)<=(249001/250000)*V_t+(5/2)*rbar^2 whenever every "
                "approximate solve passes the locked equation-residual stopping rule"
            ),
            "not_claimed": [
                "a concrete iterative solver, iteration count, or complexity guarantee",
                "that an unverified approximate solution obeys the residual envelope",
                "BF16, FP32, stochastic-gradient, or literal upstream Muon convergence",
                "that radius 253 is impossible for every storage or multiplier choice",
                "a unique minimizer or arbitrary-pair trajectory contraction",
            ],
        },
        "operator": {
            "base": "A_epsilon=E_h,epsilon+G_epsilon from P13",
            "domain": "R^(m x n) for every fixed finite positive m,n",
            "epsilon_domain": "every real epsilon>0; 1e-7 is pinned for provenance",
            "normalization": "M/(||M||_F+epsilon); no max floor",
            "coefficients_exact": {
                name: str(value) for name, value in zip(("a", "b", "c"), coefficients, strict=True)
            },
            "newton_schulz_stages": 5,
            "shift": "B=A_epsilon+1000*I",
            "forward_map": "F=I+(1/1000)*B",
            "upstream_formula_provenance": {
                "repository": "https://github.com/KellerJordan/Muon",
                "revision": PINNED_KELLER_JORDAN_MUON_REVISION,
                "audited_file": "muon.py",
                "audited_file_sha256": PINNED_MUON_PY_SHA256,
                "formula": "X/(X.norm(dim=(-2,-1),keepdim=True)+1e-7)",
                "qualification": "formula provenance only; P15 is an exact-real design",
            },
        },
        "inexact_resolvent": {
            "equation_residual": "r=s-u_hat-lambda*B(u_hat)",
            "approximate_output": "Y_hat=(s-u_hat)/lambda",
            "graph_output_not_covered": (
                "B(u_hat)=Y_hat-r/lambda is a different output convention with "
                "class-uniform residual gain 1000"
            ),
            "forward_strong_monotonicity": _fraction(bounds.denominator),
            "solution_error_gain": _fraction(bounds.solution_error_gain),
            "output_error_gain": _fraction(bounds.resolvent_form_output_error_gain),
            "sharpness_witness": {
                "abstract_base": "A=0, hence B=1000*I",
                "input": _fraction(witness.input_value),
                "residual": _fraction(witness.residual_value),
                "solution_error_ratio": _fraction(witness.solution_error_ratio),
                "output_error_ratio": _fraction(witness.output_error_ratio),
                "qualification": (
                    "sharp over the abstract monotone base class; not asserted to be attained "
                    "by the locked P13 map"
                ),
            },
        },
        "stopping_rule": {
            "rule": "||r||_F<=kappa*||s||_F+rbar at every oracle call",
            "relative_kappa": _fraction(certificate.relative_residual_slope),
            "absolute_floor": "rbar>=0 is arbitrary and time-uniform",
            "relative_output_radius": _fraction(
                certificate.output_error_gain * certificate.relative_residual_slope
            ),
            "p14_centered_radius": _fraction(certificate.sector.residual_lipschitz),
            "effective_centered_radius": _fraction(certificate.effective_residual_lipschitz),
            "absolute_output_bound": "||e_abs||_F<=500*rbar",
            "split": (
                "write Y_hat-Y=d_rel+e_abs with ||d_rel||<=2||s|| and "
                "||e_abs||<=500*rbar; such a collinear split exists whenever the summed "
                "residual envelope holds"
            ),
        },
        "exact_robust_certificate": {
            "tau": _fraction(base.tau),
            "q": _fraction(certificate.rate),
            "absolute_output_penalty": _fraction(certificate.absolute_output_penalty),
            "normalized_absolute_output_penalty": _fraction(
                certificate.normalized_absolute_output_penalty
            ),
            "C15": _fraction(certificate.absolute_forcing_gain),
            "inequality": "V_next<=(249001/250000)*V+(5/2)*rbar^2",
            "storage": _matrix(base.storage),
            "function_storage": _fraction(base.function_storage),
            "multipliers": {
                "interpolation_12": _fraction(base.lambda_interpolation_12),
                "interpolation_21": _fraction(base.lambda_interpolation_21),
                "pl_next": _fraction(base.lambda_pl_next),
                "residual": _fraction(base.lambda_residual_norm),
            },
            "coordinates": "chi=(m/L,grad(f)/L,v,grad(f_next)/L,e_abs/(gamma*L))",
            "transition_matrix": _matrix(matrices.transition),
            "selection_matrix": _matrix(matrices.selection),
            "signal_selector": _vector(matrices.signal_selector),
            "step_selector": _vector(matrices.step_selector),
            "interpolation_matrices": {
                "current_to_next": _matrix(matrices.interpolation_12),
                "next_to_current": _matrix(matrices.interpolation_21),
            },
            "pl_next_matrix": _matrix(matrices.pl_next),
            "residual_iqc_matrix": _matrix(matrices.residual_norm),
            "absolute_error_norm_matrix": _matrix(matrices.absolute_output_error_norm),
            "lmi_matrix": _matrix(audit.lmi),
            "storage_leading_minors": [_fraction(value) for value in audit.storage_leading_minors],
            "negative_lmi_leading_minors": [
                _fraction(value) for value in audit.negative_lmi_leading_minors
            ],
            "function_coefficients": [
                _fraction(value) for value in audit.supply_function_coefficients
            ],
            "expected_function_coefficients": [
                _fraction(value) for value in audit.expected_function_coefficients
            ],
            "certified": audit.certified,
        },
        "consequences": {
            "zero_floor": "rbar=0 recovers geometric convergence at the exact P14 rate",
            "ultimate_storage": {
                "bound": "limsup V_t<=(625000/999)*rbar^2",
                "coefficient": _fraction(certificate.ultimate_storage_gain),
            },
            "ultimate_objective": {
                "bound": ("limsup(f(W_t)-f_star)<=(6250000000000/312929757)*rbar^2"),
                "coefficient": _fraction(ultimate_objective_gain),
            },
            "iterate_qualification": (
                "a persistent positive residual floor gives an objective/storage neighborhood, "
                "not parameter convergence; rbar=0 gives the P14 convergence conclusion"
            ),
        },
        "controls": {
            "frozen_radius": {
                "passing_radius": _fraction(LOCKED_PASSING_FROZEN_RADIUS),
                "passing": radius_pass.certificate_passes,
                "rejected_relative_kappa": _fraction(REJECTED_RELATIVE_RESIDUAL),
                "rejected_radius": _fraction(LOCKED_FAILING_FROZEN_RADIUS),
                "rejected_fourth_minor": _fraction(
                    radius_fail.audit.negative_lmi_leading_minors[-1]
                ),
                "qualification": (
                    "radius 253 rejects only the unchanged P14 storage and multipliers; "
                    "it is not an instability or impossibility theorem"
                ),
            },
            "absolute_output_penalty": {
                "critical": _fraction(FROZEN_ABSOLUTE_OUTPUT_PENALTY_THRESHOLD),
                "selected": _fraction(certificate.absolute_output_penalty),
                "selected_fifth_minor": _fraction(audit.negative_lmi_leading_minors[-1]),
                "insufficient": _fraction(INSUFFICIENT_ABSOLUTE_OUTPUT_PENALTY),
                "insufficient_fifth_minor": _fraction(
                    insufficient_audit.negative_lmi_leading_minors[-1]
                ),
                "critical_fifth_minor": _fraction(critical_audit.negative_lmi_leading_minors[-1]),
                "qualification": "exact boundary for this frozen five-by-five certificate",
            },
            "abstract_stopping_boundary": {
                "scope": "uniform abstract monotone-base class, not the locked P13 map",
                "kappa_one": {
                    "residual": "r=-s",
                    "approximate_output_slope": _fraction(stall.approximate_output_slope),
                    "p_at_one": _fraction(stall.p_at_one),
                    "conclusion": "the optimizer stalls and no strict uniform rate is possible",
                },
                "kappa_two": {
                    "residual": "r=-2s",
                    "approximate_output_slope": _fraction(ascent.approximate_output_slope),
                    "p_at_one": _fraction(ascent.p_at_one),
                    "conclusion": "negative operator gain violates the scalar Schur condition",
                },
            },
        },
        "prior_p14": prior_p14,
        "implementation_boundary": {
            "solver": "not specified",
            "residual_verification": "required at every iteration",
            "output_convention": "resolvent form (s-u_hat)/lambda only",
            "finite_precision": "not modeled",
            "bf16": "not certified",
            "upstream": "not literal upstream Muon",
        },
        "reconstruction_fields": reconstruction,
        "software": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
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
            "randomness": "none; exact rational arithmetic",
            "command": (
                "uv run --locked python scripts/certify_inexact_yosida_robustness.py "
                "--output results/summaries/inexact_yosida_robustness_certificate.json"
            ),
            "independent_command": (
                "uv run --locked python scripts/reconstruct_inexact_yosida_robustness.py "
                "--require-canonical"
            ),
            "source_snapshot": _source_snapshot(),
        },
        "audit": {
            "checks": checks,
            "all_exact_checks_passed": all(checks.values()),
            "human_proof_audit": (
                "pending and unsigned: "
                "theory/audits/P15_INEXACT_YOSIDA_ROBUSTNESS_HUMAN_PROOF_AUDIT.md"
            ),
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
