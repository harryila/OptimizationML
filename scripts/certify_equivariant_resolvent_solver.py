#!/usr/bin/env python3
"""Generate the exact/Arb P16 equivariant-resolvent certificate."""

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

from passive_muon.equivariant_resolvent_certificate import (
    BEST_SCALAR_DEPARTURE_GATE,
    CANONICAL_APPROXIMATE_ROOT_HEX,
    CANONICAL_DEPARTURE_GUARD,
    CANONICAL_INPUT,
    CANONICAL_RETENTION_GUARD,
    CANONICAL_UPSTREAM_DEPARTURE_GUARD,
    EQUIVARIANT_RESOLVENT_SOLVER_SCHEMA_VERSION,
    JACOBIAN_DIAGONAL_MARGINS,
    LOCKED_BEST_SCALAR_DEPARTURE_GUARD,
    LOCKED_EPSILON,
    LOCKED_LAMBDA,
    LOCKED_MU,
    LOCKED_RETENTION_GUARD,
    LOCKED_UPSTREAM_DEPARTURE_GUARD,
    UPSTREAM_SHAPING_RETENTION_GATE,
    evaluate_canonical_shaping_enclosure,
    evaluate_shaping_witness,
    exact_certificate_checks,
    forward_shaping_witness,
)
from passive_muon.specs import JORDAN_QUINTIC
from passive_muon.upstream_momentum import (
    PINNED_KELLER_JORDAN_MUON_REVISION,
    PINNED_MUON_PY_SHA256,
)

ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = "results/summaries/equivariant_resolvent_solver_certificate.json"

P15_ARTIFACT_PATH = "results/summaries/inexact_yosida_robustness_certificate.json"
P15_ARTIFACT_SHA256 = "6b0c1da2c30172cd37988f0488e400fe6ecd2872300926191619c27fea647255"
P15_SOURCE_COMMIT = "287079ba0a22b4b79aef3a5f9d0ede27dfe4269a"
P15_ARTIFACT_COMMIT = "8ae64e8e11fdd0b16bd19523a2baf78400dd743c"
P15_CHECKPOINT_TAG = "p15-inexact-yosida-robustness-checkpoint"
P15_CHECKPOINT_TAG_OBJECT = "24263cb3a20cd6008895c6da66ad25d5a673b0fb"

SOURCE_PATHS = (
    ".github/workflows/ci.yml",
    ".github/workflows/p16-equivariant-resolvent-solver.yml",
    "experiments/resolvent/run_p16_solver_study.py",
    "scripts/certify_equivariant_resolvent_solver.py",
    "scripts/reconstruct_equivariant_resolvent_solver.py",
    "src/passive_muon/equivariant_resolvent_certificate.py",
    "src/passive_muon/equivariant_resolvent_solver.py",
    "src/passive_muon/inexact_yosida_robustness.py",
    "src/passive_muon/pl_convergence.py",
    "src/passive_muon/momentum_iqc.py",
    "src/passive_muon/additive_epsilon_deficit.py",
    "src/passive_muon/radial_passivation_tradeoff.py",
    "src/passive_muon/specs.py",
    "src/passive_muon/structure_aware_stability.py",
    "src/passive_muon/upstream_momentum.py",
    "src/passive_muon/yosida_stability.py",
    "tests/test_equivariant_resolvent_certificate.py",
    "tests/test_equivariant_resolvent_solver.py",
    "tests/test_p16_solver_study.py",
    "tests/test_equivariant_resolvent_solver_cli.py",
    "tests/test_equivariant_resolvent_solver_reconstruction.py",
    "tests/test_equivariant_resolvent_solver_result_manifest.py",
    "tests/test_current_research_index.py",
    "theory/equivariant_resolvent_solver.md",
    "theory/audits/P16_EQUIVARIANT_RESOLVENT_SOLVER_HUMAN_PROOF_AUDIT.md",
    "results/summaries/P16_EQUIVARIANT_RESOLVENT_SOLVER_RESULTS.md",
    "third_party/UPSTREAM_COMMITS.md",
    "pyproject.toml",
    "uv.lock",
)

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--precision-bits",
        type=int,
        action="append",
        default=None,
        help="Arb precision; repeat for cross-precision replay (default: 160 and 224)",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _decimal(value: Fraction, *, digits: int = 60) -> str:
    with localcontext() as context:
        context.prec = digits
        return str(Decimal(value.numerator) / Decimal(value.denominator))


def _fraction(value: Fraction) -> dict[str, str]:
    return {"exact": str(value), "decimal": _decimal(value)}


def _logarithmic(value: Any) -> dict[str, dict[str, str]]:
    return {
        "rational_part": _fraction(value.rational_part),
        "logarithm_coefficient": _fraction(value.logarithm_coefficient),
        "logarithm_argument": _fraction(value.logarithm_argument),
    }


def _arb_fraction(value: Fraction) -> arb:
    return arb(fmpq(value.numerator, value.denominator))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_output(*arguments: str) -> str | None:
    completed = subprocess.run(
        ["git", *arguments], cwd=ROOT, check=False, capture_output=True, text=True
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _git_blob_sha256(commit: str, path: str) -> str:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{path}"], cwd=ROOT, check=False, capture_output=True
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


def _prior_p15() -> tuple[dict[str, str], dict[str, bool]]:
    path = ROOT / P15_ARTIFACT_PATH
    observed = _sha256(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    record = {
        "artifact_path": P15_ARTIFACT_PATH,
        "artifact_sha256": observed,
        "source_commit": payload["git"]["sha"],
        "artifact_commit": P15_ARTIFACT_COMMIT,
        "checkpoint_tag": P15_CHECKPOINT_TAG,
        "checkpoint_tag_object": P15_CHECKPOINT_TAG_OBJECT,
    }
    checks = {
        "p15_artifact_hash_matches": observed == P15_ARTIFACT_SHA256,
        "p15_historical_artifact_blob_matches": (
            _git_blob_sha256(P15_ARTIFACT_COMMIT, P15_ARTIFACT_PATH) == P15_ARTIFACT_SHA256
        ),
        "p15_source_commit_matches": payload["git"]["sha"] == P15_SOURCE_COMMIT,
        "p15_source_was_clean": payload["git"]["dirty"] is False,
        "p15_checkpoint_tag_matches": (
            _git_output("rev-parse", P15_CHECKPOINT_TAG) == P15_CHECKPOINT_TAG_OBJECT
            and _git_output("rev-parse", f"{P15_CHECKPOINT_TAG}^{{}}") == P15_ARTIFACT_COMMIT
        ),
        "p15_stopping_rule_matches": (
            payload["stopping_rule"]["relative_kappa"]["exact"] == "1/250"
            and payload["inexact_resolvent"]["approximate_output"] == "Y_hat=(s-u_hat)/lambda"
        ),
    }
    return record, checks


def _guard_payload(bounds: tuple[Fraction, Fraction]) -> dict[str, str]:
    return {"strict_lower": str(bounds[0]), "strict_upper": str(bounds[1])}


def _arb_pass(precision_bits: int) -> dict[str, Any]:
    evaluation = evaluate_shaping_witness(precision_bits=precision_bits)
    departure_lower, departure_upper = LOCKED_BEST_SCALAR_DEPARTURE_GUARD
    upstream_lower, upstream_upper = LOCKED_UPSTREAM_DEPARTURE_GUARD
    retention_lower, retention_upper = LOCKED_RETENTION_GUARD
    checks = {
        "modal_gain_gap_positive": evaluation.modal_gain_gap > 0,
        "departure_inside_guard": (
            evaluation.best_scalar_departure > _arb_fraction(departure_lower)
            and evaluation.best_scalar_departure < _arb_fraction(departure_upper)
        ),
        "upstream_departure_inside_guard": (
            evaluation.upstream_best_scalar_departure > _arb_fraction(upstream_lower)
            and evaluation.upstream_best_scalar_departure < _arb_fraction(upstream_upper)
        ),
        "retention_inside_guard": (
            evaluation.upstream_shaping_retention > _arb_fraction(retention_lower)
            and evaluation.upstream_shaping_retention < _arb_fraction(retention_upper)
        ),
        "absolute_fidelity_gate_fails": evaluation.absolute_gate_fails,
        "relative_fidelity_gate_fails": evaluation.retention_gate_fails,
        "gate_decisions_are_definite": evaluation.gate_decisions_are_definite,
        "departure_guard_upper_is_below_gate": (departure_upper < BEST_SCALAR_DEPARTURE_GATE),
        "retention_guard_upper_is_below_gate": (retention_upper < UPSTREAM_SHAPING_RETENTION_GATE),
    }
    return {
        "precision_bits": precision_bits,
        "input_singular_values": [str(value) for value in evaluation.input_singular_values],
        "output_singular_values": [str(value) for value in evaluation.output_singular_values],
        "modal_gains": [str(value) for value in evaluation.modal_gains],
        "modal_gain_gap": str(evaluation.modal_gain_gap),
        "best_scalar_departure": str(evaluation.best_scalar_departure),
        "upstream_best_scalar_departure": str(evaluation.upstream_best_scalar_departure),
        "upstream_shaping_retention": str(evaluation.upstream_shaping_retention),
        "checks": checks,
        "all_checks_pass": all(checks.values()),
    }


def _canonical_arb_pass(precision_bits: int) -> dict[str, Any]:
    evaluation = evaluate_canonical_shaping_enclosure(precision_bits=precision_bits)
    departure_lower, departure_upper = CANONICAL_DEPARTURE_GUARD
    upstream_lower, upstream_upper = CANONICAL_UPSTREAM_DEPARTURE_GUARD
    retention_lower, retention_upper = CANONICAL_RETENTION_GUARD
    checks = {
        "p15_output_error_enclosure_is_small": (
            evaluation.exact_output_error_norm_upper < _arb_fraction(Fraction(1, 10**10))
        ),
        "modal_gain_intervals_are_disjoint": evaluation.modal_gain_gap > 0,
        "departure_inside_guard": (
            evaluation.best_scalar_departure > _arb_fraction(departure_lower)
            and evaluation.best_scalar_departure < _arb_fraction(departure_upper)
        ),
        "upstream_departure_inside_guard": (
            evaluation.upstream_best_scalar_departure > _arb_fraction(upstream_lower)
            and evaluation.upstream_best_scalar_departure < _arb_fraction(upstream_upper)
        ),
        "retention_inside_guard": (
            evaluation.upstream_shaping_retention > _arb_fraction(retention_lower)
            and evaluation.upstream_shaping_retention < _arb_fraction(retention_upper)
        ),
        "absolute_fidelity_gate_fails": evaluation.absolute_gate_fails,
        "relative_fidelity_gate_fails": evaluation.retention_gate_fails,
        "gate_decisions_are_definite": evaluation.gate_decisions_are_definite,
        "departure_guard_upper_is_below_gate": (departure_upper < BEST_SCALAR_DEPARTURE_GATE),
        "retention_guard_upper_is_below_gate": (retention_upper < UPSTREAM_SHAPING_RETENTION_GATE),
    }
    return {
        "precision_bits": precision_bits,
        "approximate_root_exact_binary": [str(value) for value in evaluation.approximate_root],
        "graph_residual_norm": str(evaluation.graph_residual_norm),
        "exact_output_error_norm_upper": str(evaluation.exact_output_error_norm_upper),
        "exact_output_singular_values": [
            str(value) for value in evaluation.exact_output_singular_values
        ],
        "modal_gains": [str(value) for value in evaluation.modal_gains],
        "modal_gain_gap": str(evaluation.modal_gain_gap),
        "best_scalar_departure": str(evaluation.best_scalar_departure),
        "upstream_best_scalar_departure": str(evaluation.upstream_best_scalar_departure),
        "upstream_shaping_retention": str(evaluation.upstream_shaping_retention),
        "checks": checks,
        "all_checks_pass": all(checks.values()),
    }


def build_payload(*, precisions: tuple[int, ...]) -> dict[str, Any]:
    if len(precisions) < 2 or len(set(precisions)) != len(precisions):
        raise ValueError("at least two distinct Arb precisions are required")
    exact_checks = exact_certificate_checks()
    witness = forward_shaping_witness()
    arb_passes = [_arb_pass(precision) for precision in precisions]
    canonical_arb_passes = [_canonical_arb_pass(precision) for precision in precisions]
    prior_p15, prior_checks = _prior_p15()
    checks = {
        **{f"exact_{name}": value for name, value in exact_checks.items()},
        **prior_checks,
        "all_cross_precision_witness_guards_pass": all(
            replay["all_checks_pass"] for replay in arb_passes
        ),
        "all_canonical_cross_precision_guards_pass": all(
            replay["all_checks_pass"] for replay in canonical_arb_passes
        ),
        "locked_point_fails_declared_meaningful_fidelity_gate": all(
            replay["checks"]["absolute_fidelity_gate_fails"]
            and replay["checks"]["relative_fidelity_gate_fails"]
            for replay in canonical_arb_passes
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"P16 certificate checks failed: {failed}")

    return {
        "schema_version": EQUIVARIANT_RESOLVENT_SOLVER_SCHEMA_VERSION,
        "canonical_result_path": RESULT_PATH,
        "claim_scope": {
            "matrix_domain": "R^(m x n) for every fixed finite positive m,n",
            "arithmetic_model_theorem": "exact real arithmetic",
            "implementation_model": "safeguarded FP64 checked-success reference",
            "normalization": "current Frobenius norm plus additive epsilon; no max floor",
            "epsilon": _fraction(LOCKED_EPSILON),
            "polynomial": "five composed Jordan quintics",
            "coefficients_exact": [str(value) for value in JORDAN_QUINTIC.fractions()],
            "guarantee": (
                "equivariant singular-value reduction and reliable exact-arithmetic guarded "
                "termination; each declared FP64 success independently passes the computed "
                "FP64 form of P15's residual test"
            ),
            "classification": (
                "solver/reduction success but meaningful-Muon-fidelity gate failure at the "
                "locked P14 point"
            ),
            "not_claimed": [
                "a uniform FP64 rounding-error or iteration-count theorem",
                "meaningful Muon-like shaping for the locked P14 parameters",
                "global exclusion of every alternative lambda,mu pair",
                "a BF16, GPU, accelerator-throughput, or literal upstream-Muon solver",
                "neural-network convergence",
            ],
        },
        "operator": {
            "base": "A_eps=E_h,eps+G_eps",
            "shifted": "B_eps,mu=A_eps+mu*I",
            "forward": "F(u)=u+lambda*B_eps,mu(u)",
            "resolvent": "J=F^(-1)",
            "deployed_output": "Y_hat=(s-u_hat)/lambda=1000*(s-u_hat)",
            "forbidden_approximate_output": "B(u_hat)",
            "lambda": _fraction(LOCKED_LAMBDA),
            "mu": _fraction(LOCKED_MU),
        },
        "equivariance_and_reduction": {
            "equivariance": "J(Q*S*R^T)=Q*J(S)*R^T for orthogonal Q,R",
            "singular_vector_statement": (
                "the unique resolvent preserves singular subspaces; repeated positive blocks "
                "have equal solved values and zero input modes map to zero"
            ),
            "scalar_system": (
                "Phi_i=(1+lambda*(mu+p(r/eps)/r))*x_i+lambda*h(x_i/(r+eps))-sigma_i=0"
            ),
            "jacobian": "D Phi=diag(D_i)+a*v^T with v=x/r",
            "diagonal": ("D_i=1+lambda*(mu+p(r/eps)/r)+lambda*h'(x_i/(r+eps))/(r+eps)"),
            "rank_one_vector": ("a_i=lambda*x_i*((p/r)' - h'(x_i/(r+eps))/(r+eps)^2)"),
            "sherman_morrison": ("delta=D^-1*b-D^-1*a*(v^T*D^-1*b)/(1+v^T*D^-1*a)"),
            "band_diagonal_margins": [_fraction(value) for value in JACOBIAN_DIAGONAL_MARGINS],
            "forward_strong_monotonicity": _fraction(1 + LOCKED_LAMBDA * LOCKED_MU),
            "branches": [
                "r=0 uses the exact scalar derivative",
                "0<r/eps<=63/9937 uses p/r=U/eps and derivative zero",
                "the join is C1; only the second derivative has a bounded jump",
                "signed Newton iterates avoid an unjustified positivity-constrained proof",
            ],
        },
        "guarded_termination_theorem": {
            "coordinate_model": "signed diagonal coordinates in fixed singular bases",
            "merit": "psi(x)=||Phi(x)||_2^2/2",
            "descent_identity": "grad(psi)^T*(-D Phi^-1 Phi)=-||Phi||_2^2",
            "argument": (
                "two-strong monotonicity bounds every merit level set; the C1,1 Jacobian and "
                "its inverse are bounded there, so Armijo backtracking converges globally"
            ),
            "finite_termination": (
                "for s!=0 the P15 relative threshold is positive, hence convergence reaches "
                "it in finitely many exact-real iterations; s=0 returns exactly"
            ),
            "fp64_contract": (
                "nonfinite values or unsafe diagonal/SM denominators reject; success is emitted "
                "only after reevaluating B on the stored candidate and recomputing the literal "
                "full-matrix P15 residual"
            ),
            "relative_residual": _fraction(Fraction(1, 250)),
        },
        "exact_forward_shaping_witness": {
            "construction": "u=diag(3/5,4/5), s=u+lambda*B(u), Y=B(u)",
            "resolvent_singular_values": [
                str(value) for value in witness.resolvent_singular_values
            ],
            "normalized_singular_values": [
                str(value) for value in witness.normalized_singular_values
            ],
            "radial_primitive": _logarithmic(witness.radial_primitive),
            "input_singular_values": [
                _logarithmic(value) for value in witness.input_singular_values
            ],
            "output_singular_values": [
                _logarithmic(value) for value in witness.output_singular_values
            ],
            "pre_resolvent_modal_gain_difference_exact": str(
                witness.pre_resolvent_modal_gain_difference
            ),
            "exact_sign_argument": (
                "the common shunt and radial gains cancel; k2-k1=h(w2)/u2-h(w1)/u1>0, "
                "and k/(1+lambda*k) is strictly increasing"
            ),
            "conclusion": "the locked map is algebraically non-scalar on this 2x2 input",
            "arb_cross_precision": arb_passes,
        },
        "meaningful_fidelity_decision": {
            "metric": "chi(T;s)=min_a||T-a*s||_F/||T||_F",
            "absolute_gate": _fraction(BEST_SCALAR_DEPARTURE_GATE),
            "retention_metric": "chi(Y;s)/chi(upstream_five_stage_Jordan;s)",
            "retention_gate": _fraction(UPSTREAM_SHAPING_RETENTION_GATE),
            "adoption": (
                "frozen after exploratory probes and before canonical artifact generation; "
                "not represented as a preregistered threshold"
            ),
            "canonical_input": [str(value) for value in CANONICAL_INPUT],
            "approximate_root_hex": list(CANONICAL_APPROXIMATE_ROOT_HEX),
            "enclosure_method": (
                "evaluate the binary64 candidate graph residual with Arb, then inflate its "
                "resolvent-form output by P15's exact gain 500"
            ),
            "locked_departure_guard": _guard_payload(CANONICAL_DEPARTURE_GUARD),
            "upstream_departure_guard": _guard_payload(CANONICAL_UPSTREAM_DEPARTURE_GUARD),
            "retention_guard": _guard_payload(CANONICAL_RETENTION_GUARD),
            "arb_cross_precision": canonical_arb_passes,
            "algebraic_noncollapse_passes": True,
            "meaningful_fidelity_passes": False,
            "interpretation": (
                "distinct modal gains are necessary but not sufficient; at the locked point "
                "the departure is about 5.879e-6 and retains about 8.403e-5 of the upstream "
                "direction, so the method is effectively a scalar preconditioner here"
            ),
        },
        "required_frontier": {
            "status": "completed as a deterministic diagnostic in the companion solver study",
            "logical_scope": (
                "finite sampled lambda,mu/input grid; failure to find a point is not a global "
                "impossibility theorem"
            ),
        },
        "prior_p15": prior_p15,
        "upstream_formula_provenance": {
            "repository": "https://github.com/KellerJordan/Muon",
            "revision": PINNED_KELLER_JORDAN_MUON_REVISION,
            "audited_file": "muon.py",
            "audited_file_sha256": PINNED_MUON_PY_SHA256,
            "normalization": "X/(X.norm(dim=(-2,-1),keepdim=True)+1e-7)",
            "comparison_scope": "exact-real five-stage spectral counterpart only",
        },
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
                "uv run --locked python scripts/certify_equivariant_resolvent_solver.py "
                "--output results/summaries/equivariant_resolvent_solver_certificate.json"
            ),
            "source_snapshot": _source_snapshot(),
        },
        "audit": {
            "checks": checks,
            "all_exact_and_interval_checks_passed": all(checks.values()),
            "human_proof_audit": "pending; packet committed but unsigned",
        },
    }


def main() -> int:
    args = parse_args()
    precisions = tuple(args.precision_bits or (160, 224))
    with ctx.workprec(max(precisions)):
        payload = build_payload(precisions=precisions)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
