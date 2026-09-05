#!/usr/bin/env python3
"""Generate the exact/Arb P17 shape-preserving resolvent certificate."""

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

from passive_muon.pl_convergence import PlConvergenceCertificate
from passive_muon.shape_preserving_resolvent import (
    LOCKED_FULL_STEP_TAU,
    LOCKED_HIGH_FIDELITY_TAU,
    audit_shape_preserving_resolvent,
)
from passive_muon.shape_preserving_resolvent_certificate import (
    BEST_SCALAR_DEPARTURE_GATE,
    CANONICAL_UPSTREAM_DEPARTURE_GUARD,
    FULL_STEP_CANONICAL_DEPARTURE_GUARD,
    FULL_STEP_CANONICAL_RETENTION_GUARD,
    FULL_STEP_DESIGN,
    GATE_INNER_SQUARED_RADIUS,
    GATE_OUTER_SQUARED_RADIUS,
    LOCKED_DESIGNS,
    LOCKED_EPSILON,
    LOCKED_LAMBDA,
    LOCKED_MU,
    PRIMARY_CANONICAL_DEPARTURE_GUARD,
    PRIMARY_CANONICAL_RETENTION_GUARD,
    PRIMARY_DESIGN,
    RAW_SHAPE_GAIN_UPPER,
    SHAPE_PRESERVING_RESOLVENT_SCHEMA_VERSION,
    UNDERSIZED_GATE_INNER_RADIUS,
    UNDERSIZED_GATE_OUTER_RADIUS,
    UNSAFE_BAND_FORWARD_DERIVATIVE_GUARD,
    UNSAFE_BAND_LEFT,
    UNSAFE_BAND_RAW_SHAPE_DERIVATIVE_GUARD,
    UNSAFE_BAND_RIGHT,
    UNSAFE_BAND_SOURCE_GUARD,
    UPSTREAM_SHAPING_RETENTION_GATE,
    certify_unsafe_derivative_band,
    evaluate_canonical_gated_fidelity,
    exact_certificate_checks,
    exact_unsafe_witness,
    interval_certificate_checks,
    pointwise_sector,
    undersized_passive_region_control,
)
from passive_muon.specs import JORDAN_QUINTIC
from passive_muon.upstream_momentum import (
    PINNED_KELLER_JORDAN_MUON_REVISION,
    PINNED_MUON_PY_SHA256,
)

ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = "results/summaries/shape_preserving_resolvent_certificate.json"
SCHEMA_VERSION = SHAPE_PRESERVING_RESOLVENT_SCHEMA_VERSION

P16_ARTIFACT_PATH = "results/summaries/equivariant_resolvent_solver_certificate.json"
P16_ARTIFACT_SHA256 = "1ca2a9179c050a024cd0ba1e07db2aacb0d17ea48aa1920ae54d266c8e67906d"
P16_SOURCE_COMMIT = "886dab0b5688939047d1facfa7839338ff1fabc8"
P16_ARTIFACT_COMMIT = "31ccdcdc32f5bf510321bca2c610074760fb1345"
P16_CHECKPOINT_TAG = "p16-equivariant-resolvent-solver-diagnostic"
P16_CHECKPOINT_TAG_OBJECT = "23d02a204e3e3ccf5066f24e285f5f6a4fcbfb54"

SOURCE_PATHS = (
    ".github/workflows/ci.yml",
    ".github/workflows/p17-shape-preserving-resolvent.yml",
    "experiments/resolvent/run_p17_shape_preserving_study.py",
    "scripts/certify_shape_preserving_resolvent.py",
    "scripts/reconstruct_shape_preserving_resolvent.py",
    "src/passive_muon/additive_epsilon_deficit.py",
    "src/passive_muon/equivariant_resolvent_solver.py",
    "src/passive_muon/floored_certificate.py",
    "src/passive_muon/momentum_iqc.py",
    "src/passive_muon/pl_convergence.py",
    "src/passive_muon/radial_passivation_tradeoff.py",
    "src/passive_muon/shape_preserving_resolvent.py",
    "src/passive_muon/shape_preserving_resolvent_certificate.py",
    "src/passive_muon/specs.py",
    "src/passive_muon/structure_aware_stability.py",
    "src/passive_muon/upstream_momentum.py",
    "src/passive_muon/yosida_stability.py",
    "tests/test_shape_preserving_resolvent.py",
    "tests/test_shape_preserving_resolvent_certificate.py",
    "tests/test_shape_preserving_resolvent_cli.py",
    "tests/test_shape_preserving_resolvent_reconstruction.py",
    "tests/test_shape_preserving_resolvent_result_manifest.py",
    "tests/test_p17_shape_preserving_study.py",
    "tests/test_current_research_index.py",
    "theory/shape_preserving_resolvent.md",
    "theory/audits/P17_SHAPE_PRESERVING_RESOLVENT_HUMAN_PROOF_AUDIT.md",
    "results/summaries/P17_SHAPE_PRESERVING_RESOLVENT_RESULTS.md",
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


def _guard(bounds: tuple[Fraction, Fraction]) -> dict[str, str]:
    return {"strict_lower": str(bounds[0]), "strict_upper": str(bounds[1])}


def _fraction_sha256(value: Fraction) -> str:
    return hashlib.sha256(f"{value.numerator}/{value.denominator}".encode()).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_output(*arguments: str) -> str | None:
    completed = subprocess.run(
        ["git", *arguments], cwd=ROOT, check=False, capture_output=True, text=True
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
    missing = [path for path in SOURCE_PATHS if not (ROOT / path).is_file()]
    if missing:
        raise RuntimeError(f"P17 source snapshot is incomplete: {missing}")
    return {path: _sha256(ROOT / path) for path in SOURCE_PATHS}


def _prior_p16() -> tuple[dict[str, str], dict[str, bool]]:
    path = ROOT / P16_ARTIFACT_PATH
    observed = _sha256(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    record = {
        "artifact_path": P16_ARTIFACT_PATH,
        "artifact_sha256": observed,
        "source_commit": payload["git"]["sha"],
        "artifact_commit": P16_ARTIFACT_COMMIT,
        "checkpoint_tag": P16_CHECKPOINT_TAG,
        "checkpoint_tag_object": P16_CHECKPOINT_TAG_OBJECT,
    }
    checks = {
        "p16_artifact_hash_matches": observed == P16_ARTIFACT_SHA256,
        "p16_historical_artifact_blob_matches": (
            _git_blob_sha256(P16_ARTIFACT_COMMIT, P16_ARTIFACT_PATH) == P16_ARTIFACT_SHA256
        ),
        "p16_source_commit_matches": payload["git"]["sha"] == P16_SOURCE_COMMIT,
        "p16_source_was_clean": payload["git"]["dirty"] is False,
        "p16_checkpoint_tag_matches": (
            _git_output("rev-parse", P16_CHECKPOINT_TAG) == P16_CHECKPOINT_TAG_OBJECT
            and _git_output("rev-parse", f"{P16_CHECKPOINT_TAG}^{{}}") == P16_ARTIFACT_COMMIT
        ),
    }
    return record, checks


def _pl_fields(certificate: PlConvergenceCertificate, audit: Any) -> dict[str, object]:
    return {
        "storage": _matrix(certificate.storage),
        "function_storage": str(certificate.function_storage),
        "interpolation_reverse_weight": str(certificate.interpolation_reverse_weight),
        "residual_multiplier": str(certificate.lambda_residual_norm),
        "storage_leading_minors": [str(value) for value in audit.storage_leading_minors],
        "negative_lmi_leading_minors": [str(value) for value in audit.negative_lmi_leading_minors],
        "rate_squared": str(certificate.tau**2),
    }


def reconstruction_fields() -> dict[str, object]:
    """Return the exact fields rebuilt by the standard-library script."""

    unsafe = exact_unsafe_witness()
    negative = undersized_passive_region_control()
    design_fields: dict[str, object] = {}
    for design in LOCKED_DESIGNS:
        runtime = audit_shape_preserving_resolvent(design)
        sector = pointwise_sector(design)
        tau = LOCKED_HIGH_FIDELITY_TAU if design == PRIMARY_DESIGN else LOCKED_FULL_STEP_TAU
        design_fields[design.name] = {
            "cap": str(design.cap),
            "passive_divisor": str(design.passive_divisor),
            "learning_rate": str(design.learning_rate),
            "tau": str(tau),
            "sector": {
                "lower": str(sector.lower_gain),
                "upper": str(sector.upper_gain),
                "center": str(sector.center),
                "radius": str(sector.radius),
            },
            "pl": _pl_fields(runtime.pl_certificate, runtime.pl_audit),
        }
    return {
        "raw_shape_gain_upper": str(RAW_SHAPE_GAIN_UPPER),
        "gate": {
            "q0": str(GATE_INNER_SQUARED_RADIUS),
            "q1": str(GATE_OUTER_SQUARED_RADIUS),
            "smootherstep_coefficients_descending": ["6", "-15", "10", "0", "0", "0"],
            "join_jets": [["0", "0", "0"], ["1", "0", "0"]],
            "max_q_slope_at_unit_cap": "5/2",
        },
        "designs": design_fields,
        "unsafe": {
            "t": str(unsafe.normalized_radius),
            "z": str(unsafe.raw_ratio),
            "u": str(unsafe.resolvent_value),
            "response_sha256": unsafe.response_sha256,
            "derivative_sha256": unsafe.derivative_sha256,
            "source_sha256": unsafe.source_sha256,
            "raw_shape_derivative_sha256": unsafe.raw_shape_derivative_sha256,
            "source_guard": ["19/10000", "1/500"],
            "raw_shape_derivative_guard": ["-147000", "-146000"],
            "yosida_derivative_guard": ["999", "1000"],
            "undersized_gate_radii": [
                str(UNDERSIZED_GATE_INNER_RADIUS),
                str(UNDERSIZED_GATE_OUTER_RADIUS),
            ],
            "gated_derivative_sha256": _fraction_sha256(negative.gated_derivative),
            "gated_derivative_guard": ["-19000", "-18000"],
            "second_jury_margin_sha256": _fraction_sha256(negative.second_jury_margin),
        },
    }


def _interval_pass(precision_bits: int) -> dict[str, object]:
    checks = interval_certificate_checks(precision_bits=precision_bits)
    band = certify_unsafe_derivative_band(precision_bits=precision_bits)
    fidelity: dict[str, object] = {}
    for design in LOCKED_DESIGNS:
        evaluation = evaluate_canonical_gated_fidelity(design, precision_bits=precision_bits)
        fidelity[design.name] = {
            "gate_value": str(evaluation.gate_value),
            "graph_residual_norm": str(evaluation.graph_residual_norm),
            "root_error_norm_upper": str(evaluation.exact_root_error_norm_upper),
            "yosida_error_norm_upper": str(evaluation.exact_yosida_error_norm_upper),
            "output_singular_values": [str(value) for value in evaluation.output_singular_values],
            "modal_gains": [str(value) for value in evaluation.modal_gains],
            "best_scalar_departure": str(evaluation.best_scalar_departure),
            "upstream_best_scalar_departure": str(evaluation.upstream_best_scalar_departure),
            "upstream_shaping_retention": str(evaluation.upstream_shaping_retention),
            "meaningful_fidelity_passes": evaluation.meaningful_fidelity_passes,
            "gate_decisions_are_definite": evaluation.gate_decisions_are_definite,
        }
    return {
        "precision_bits": precision_bits,
        "unsafe_band": {
            "normalized_radius": str(band.normalized_radius),
            "source_value": str(band.source_value),
            "forward_graph_derivative": str(band.forward_graph_derivative),
            "raw_shape_derivative": str(band.raw_shape_derivative),
            "strictly_unsafe": band.is_strictly_unsafe,
        },
        "canonical_fidelity": fidelity,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
    }


def build_payload() -> dict[str, object]:
    prior, prior_checks = _prior_p16()
    exact_checks = exact_certificate_checks()
    runtime_audits = {
        design.name: audit_shape_preserving_resolvent(design) for design in LOCKED_DESIGNS
    }
    interval_passes = [_interval_pass(bits) for bits in (160, 224)]
    all_checks = {
        **{f"exact_{name}": value for name, value in exact_checks.items()},
        **prior_checks,
        **{f"runtime_{name}_audit": audit.certified for name, audit in runtime_audits.items()},
        "all_cross_precision_interval_checks_pass": all(
            bool(item["all_checks_pass"]) for item in interval_passes
        ),
        "both_canonical_designs_pass_frozen_fidelity_gates": all(
            all(
                record["meaningful_fidelity_passes"] and record["gate_decisions_are_definite"]
                for record in item["canonical_fidelity"].values()
            )
            for item in interval_passes
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": {
            "classification": (
                "global exact-real smooth-PL convergence with a meaningful-shaping "
                "gated interface; pointwise, not incremental, sector"
            ),
            "matrix_domain": "R^(m x n) for every fixed finite positive m,n",
            "normalization": "current Frobenius norm plus additive epsilon; no max floor",
            "epsilon": _fraction(LOCKED_EPSILON),
            "polynomial": "five composed Jordan quintics",
            "coefficients_exact": [str(value) for value in JORDAN_QUINTIC.fractions()],
            "arithmetic_model_theorem": "exact real arithmetic and exact resolvent",
            "not_claimed": [
                "an incremental sector or arbitrary-pair contraction theorem",
                "propagation of P16 approximate-solver error through the shape channel",
                "an FP32, BF16, accelerator, or literal upstream-Muon theorem",
                "weight decay, aspect scaling, stochastic gradients, or neural training",
                "a global fidelity extremum inferred from the declared sampled grid",
            ],
        },
        "operator": {
            "epsilon": _fraction(LOCKED_EPSILON),
            "lambda": _fraction(LOCKED_LAMBDA),
            "mu": _fraction(LOCKED_MU),
            "base": "A_epsilon=E_h,epsilon+G_epsilon",
            "resolvent": "J=(I+lambda*(A_epsilon+mu*I))^(-1)",
            "yosida": "Y=(I-J)/lambda",
            "shape_channel": "X=E_h,epsilon composed with J",
            "interface": "T=(1-theta(||S||_F^2))*Y/c+theta(||S||_F^2)*X",
        },
        "gate": {
            "regularity": "C2 quintic smootherstep with exact zero first and second join jets",
            "inner_squared_radius": _fraction(GATE_INNER_SQUARED_RADIUS),
            "outer_squared_radius": _fraction(GATE_OUTER_SQUARED_RADIUS),
            "transition": "psi(t)=6*t^5-15*t^4+10*t^3",
        },
        "pointwise_gain_bound": {
            "jordan_derivative_upper": _fraction(Fraction(4_848_763, 10_000)),
            "radial_majorant_switch_value": _fraction(
                Fraction(6_602_082_433_275_499_863, 41_641_817_600_000_000)
            ),
            "raw_shape_gain_upper": _fraction(RAW_SHAPE_GAIN_UPPER),
            "dimension_uniform": True,
            "supply": "||T(S)-gamma*S||_F <= K*||S||_F",
            "scope": "origin-centered pointwise modal sector; not an incremental slope bound",
        },
        "designs": {
            name: {
                "config": {
                    "cap": _fraction(audit.config.cap),
                    "passive_divisor": _fraction(audit.config.passive_divisor),
                    "learning_rate": _fraction(audit.config.learning_rate),
                    "tau": _fraction(audit.pl_certificate.tau),
                    "rate_squared": _fraction(audit.pl_certificate.tau**2),
                    "beta": _fraction(audit.pl_certificate.beta),
                    "smoothness": _fraction(audit.pl_certificate.smoothness),
                    "pl_constant": _fraction(audit.pl_certificate.pl_constant),
                },
                "sector": {
                    "lower": _fraction(audit.sector.lower),
                    "upper": _fraction(audit.sector.upper),
                    "center": _fraction(audit.sector.center),
                    "radius": _fraction(audit.sector.radius),
                    "condition_ratio": _fraction(audit.sector.condition_ratio),
                },
                "pl_certificate": {
                    **_pl_fields(audit.pl_certificate, audit.pl_audit),
                    "lmi": _matrix(audit.pl_audit.lmi),
                    "function_flow_observed": [
                        str(value) for value in audit.pl_audit.supply_function_coefficients
                    ],
                    "function_flow_expected": [
                        str(value) for value in audit.pl_audit.expected_function_coefficients
                    ],
                    "certified": audit.certified,
                },
            }
            for name, audit in runtime_audits.items()
        },
        "unsafe_derivative": {
            "interval_passes": interval_passes,
            "band": {
                "normalized_radius": [str(UNSAFE_BAND_LEFT), str(UNSAFE_BAND_RIGHT)],
                "source_guard": _guard(UNSAFE_BAND_SOURCE_GUARD),
                "forward_graph_derivative_guard": _guard(UNSAFE_BAND_FORWARD_DERIVATIVE_GUARD),
                "raw_shape_derivative_guard": _guard(UNSAFE_BAND_RAW_SHAPE_DERIVATIVE_GUARD),
            },
            "undersized_passive_region": {
                "inner_radius": _fraction(UNDERSIZED_GATE_INNER_RADIUS),
                "outer_radius": _fraction(UNDERSIZED_GATE_OUTER_RADIUS),
                "classification": (
                    "exact frozen local/incremental Jury failure at a non-equilibrium signal"
                ),
            },
        },
        "fidelity_gate": {
            "best_scalar_departure_threshold": _fraction(BEST_SCALAR_DEPARTURE_GATE),
            "upstream_shaping_retention_threshold": _fraction(UPSTREAM_SHAPING_RETENTION_GATE),
            "canonical_input": "diag(3,4)",
            "canonical_upstream_departure_guard": _guard(CANONICAL_UPSTREAM_DEPARTURE_GUARD),
            "design_guards": {
                PRIMARY_DESIGN.name: {
                    "departure": _guard(PRIMARY_CANONICAL_DEPARTURE_GUARD),
                    "retention": _guard(PRIMARY_CANONICAL_RETENTION_GUARD),
                },
                FULL_STEP_DESIGN.name: {
                    "departure": _guard(FULL_STEP_CANONICAL_DEPARTURE_GUARD),
                    "retention": _guard(FULL_STEP_CANONICAL_RETENTION_GUARD),
                },
            },
            "classification": "both locked points rigorously pass on the canonical input",
        },
        "reconstruction_fields": reconstruction_fields(),
        "prior_p16": prior,
        "upstream_formula_provenance": {
            "repository": "https://github.com/KellerJordan/Muon",
            "revision": PINNED_KELLER_JORDAN_MUON_REVISION,
            "audited_file": "muon.py",
            "audited_file_sha256": PINNED_MUON_PY_SHA256,
            "normalization": "X/(X.norm(dim=(-2,-1),keepdim=True)+1e-7)",
            "comparison_scope": "exact-real five-stage spectral counterpart only",
        },
        "proof_replay_provenance": {
            "methods": [
                "exact rational algebra and Sylvester tests",
                "adaptive-free direct Arb recurrence on a locked complete unsafe band",
                "cross-precision Arb canonical fidelity with graph-residual inflation",
                "standard-library-only independent exact reconstruction",
            ],
            "arb_precision_bits": [160, 224],
            "source_snapshot": _source_snapshot(),
        },
        "audit": {
            "checks": all_checks,
            "all_exact_and_interval_checks_passed": all(all_checks.values()),
            "human_proof_audit": "pending; packet committed but unsigned",
        },
        "git": _git_state(),
        "hardware": {
            "machine": platform.machine(),
            "processor": platform.processor() or "unreported",
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
        },
        "software": {"python": sys.version},
        "canonical_result_path": RESULT_PATH,
    }


def main() -> None:
    arguments = parse_args()
    payload = build_payload()
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
