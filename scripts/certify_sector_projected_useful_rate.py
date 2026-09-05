#!/usr/bin/env python3
"""Generate the exact/Arb P18 sector-projected useful-rate certificate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Any

from passive_muon.sector_projected_resolvent_certificate import (
    BEST_SCALAR_DEPARTURE_GATE,
    CANONICAL_DEPARTURE_GUARD,
    CANONICAL_INPUT,
    CANONICAL_INPUT_AMPLITUDE_GUARD,
    CANONICAL_RETENTION_GUARD,
    CANONICAL_UPSTREAM_AMPLITUDE_GUARD,
    GENERIC_SECTOR_FAILED_STEP,
    LOCKED_EPSILON,
    LOCKED_GATE_CAP,
    LOCKED_GATE_Q0,
    LOCKED_GATE_Q1,
    LOCKED_LAMBDA,
    LOCKED_MU,
    LOCKED_P15_KAPPA,
    LOCKED_PASSIVE_DIVISOR,
    LOCKED_PROJECTION_GAIN,
    LOCKED_SECTOR_CENTER,
    LOCKED_SECTOR_LOWER,
    LOCKED_SECTOR_RADIUS,
    LOCKED_SECTOR_UPPER,
    P14_RATE_SQUARED,
    PRIMARY_EFFECTIVE_STEP_LOWER,
    PRIMARY_EFFECTIVE_STEP_UPPER,
    RAW_SHAPE_GAIN_UPPER,
    SECTOR_PROJECTED_RESOLVENT_SCHEMA_VERSION,
    SMALL_K_DEPARTURE_GUARD,
    SMALL_K_PROJECTION_GAIN,
    SMALL_K_RETENTION_GUARD,
    UPSTREAM_SHAPING_RETENTION_GATE,
    audit_all_pareto_certificates,
    evaluate_canonical_projected_fidelity,
    exact_certificate_checks,
    generic_sector_schur_obstruction,
    interval_certificate_checks,
    unprojected_sector_control,
)
from passive_muon.specs import JORDAN_QUINTIC
from passive_muon.upstream_momentum import (
    PINNED_KELLER_JORDAN_MUON_REVISION,
    PINNED_MUON_PY_SHA256,
)

ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = "results/summaries/sector_projected_useful_rate_certificate.json"
SCHEMA_VERSION = SECTOR_PROJECTED_RESOLVENT_SCHEMA_VERSION

P17_ARTIFACT_PATH = "results/summaries/shape_preserving_resolvent_certificate.json"
P17_ARTIFACT_SHA256 = "6ceea44127493cc3cb811826879413f87363a21457670e0e8be687aaef980455"
P17_SOURCE_COMMIT = "d5f0f89cf9c1bf3fdff5cea131389725c12bbe6b"
P17_ARTIFACT_COMMIT = "1f7024b5ad562601b4da3d3f25ba0dc5758a7c39"
P17_CHECKPOINT_TAG = "p17-shape-preserving-resolvent-checkpoint"
P17_CHECKPOINT_TAG_OBJECT = "3a3c860f62fc0feb573435ea0670bfb6045ca943"

SOURCE_PATHS = (
    "scripts/certify_sector_projected_useful_rate.py",
    "scripts/reconstruct_sector_projected_useful_rate.py",
    "src/passive_muon/momentum_iqc.py",
    "src/passive_muon/pl_convergence.py",
    "src/passive_muon/sector_projected_resolvent.py",
    "src/passive_muon/sector_projected_resolvent_certificate.py",
    "src/passive_muon/specs.py",
    "src/passive_muon/structure_aware_stability.py",
    "src/passive_muon/upstream_momentum.py",
    "tests/test_sector_projected_resolvent.py",
    "tests/test_sector_projected_resolvent_certificate.py",
    "tests/test_sector_projected_resolvent_cli.py",
    "tests/test_sector_projected_resolvent_reconstruction.py",
    "results/summaries/shape_preserving_resolvent_certificate.json",
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
        raise RuntimeError(f"P18 source snapshot is incomplete: {missing}")
    return {path: _sha256(ROOT / path) for path in SOURCE_PATHS}


def _prior_p17() -> tuple[dict[str, str], dict[str, bool]]:
    path = ROOT / P17_ARTIFACT_PATH
    observed = _sha256(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    record = {
        "artifact_path": P17_ARTIFACT_PATH,
        "artifact_sha256": observed,
        "source_commit": payload["git"]["sha"],
        "artifact_commit": P17_ARTIFACT_COMMIT,
        "checkpoint_tag": P17_CHECKPOINT_TAG,
        "checkpoint_tag_object": P17_CHECKPOINT_TAG_OBJECT,
    }
    checks = {
        "p17_artifact_hash_matches": observed == P17_ARTIFACT_SHA256,
        "p17_historical_artifact_blob_matches": (
            _git_blob_sha256(P17_ARTIFACT_COMMIT, P17_ARTIFACT_PATH) == P17_ARTIFACT_SHA256
        ),
        "p17_source_commit_matches": payload["git"]["sha"] == P17_SOURCE_COMMIT,
        "p17_source_was_clean": payload["git"]["dirty"] is False,
        "p17_checkpoint_tag_matches": (
            _git_output("rev-parse", P17_CHECKPOINT_TAG) == P17_CHECKPOINT_TAG_OBJECT
            and _git_output("rev-parse", f"{P17_CHECKPOINT_TAG}^{{}}") == P17_ARTIFACT_COMMIT
        ),
    }
    return record, checks


def _pl_fields(audit: Any) -> dict[str, object]:
    certificate = audit.certificate
    replay = audit.audit
    return {
        "learning_rate": str(certificate.learning_rate),
        "tau": str(certificate.tau),
        "rate_squared": str(certificate.tau**2),
        "storage": _matrix(certificate.storage),
        "function_storage": str(certificate.function_storage),
        "interpolation_reverse_weight": str(certificate.interpolation_reverse_weight),
        "residual_multiplier": str(certificate.lambda_residual_norm),
        "dimensionless_step": str(certificate.dimensionless_step),
        "residual_ratio": str(certificate.residual_ratio),
        "storage_leading_minors": [str(value) for value in replay.storage_leading_minors],
        "negative_lmi_leading_minors": [str(value) for value in replay.negative_lmi_leading_minors],
        "multipliers": [
            str(certificate.lambda_interpolation_12),
            str(certificate.lambda_interpolation_21),
            str(certificate.lambda_pl_next),
            str(certificate.lambda_residual_norm),
        ],
    }


def _complex(value: tuple[Fraction, Fraction]) -> dict[str, str]:
    return {"real": str(value[0]), "imaginary": str(value[1])}


def reconstruction_fields() -> dict[str, object]:
    """Return exactly the fields rebuilt without project imports."""

    audits = audit_all_pareto_certificates()
    primary = audits[1]
    obstruction = generic_sector_schur_obstruction()
    unprojected = unprojected_sector_control()
    passive_upper = Fraction(1_000, 1_024)
    blended_upper = Fraction(1, 4) * passive_upper + Fraction(3, 4)
    small_blended_upper = Fraction(1, 4) * passive_upper + Fraction(3, 400)
    return {
        "operator_specification": {
            "domain": "R^(m x n) for arbitrary positive finite m and n",
            "arithmetic": "exact real",
            "resolvent": "exact",
            "normalization": "U/(||U||_F+epsilon)",
            "epsilon": str(LOCKED_EPSILON),
            "polynomial_coefficients": {
                "a": str(JORDAN_QUINTIC.fractions()[0]),
                "b": str(JORDAN_QUINTIC.fractions()[1]),
                "c": str(JORDAN_QUINTIC.fractions()[2]),
            },
            "polynomial_iteration_count": 5,
            "resolvent_lambda": str(LOCKED_LAMBDA),
            "passive_shunt": str(LOCKED_MU),
            "sector_scope": "origin-centred pointwise, not incremental",
        },
        "locked_design": {
            "projection_gain": str(LOCKED_PROJECTION_GAIN),
            "passive_divisor": str(LOCKED_PASSIVE_DIVISOR),
            "gate_cap": str(LOCKED_GATE_CAP),
            "yosida_sector": {"lower": "500", "upper": "1000"},
            "passive_branch_sector": {"lower": "125/256", "upper": "125/128"},
            "blended_upper_at_gate_cap": str(blended_upper),
            "projected_pointwise_sector": {
                "lower": str(LOCKED_SECTOR_LOWER),
                "upper": str(LOCKED_SECTOR_UPPER),
                "center": str(LOCKED_SECTOR_CENTER),
                "radius": str(LOCKED_SECTOR_RADIUS),
            },
        },
        "exact_certificates": {audit.spec.name: _pl_fields(audit) for audit in audits},
        "useful_rate": {
            "primary_design": primary.spec.name,
            "pareto_design": audits[4].spec.name,
            "p14_rate_squared": str(P14_RATE_SQUARED),
            "primary_rate_to_tenth_power": str(primary.spec.tau**20),
            "effective_step_window": {
                "lower": str(PRIMARY_EFFECTIVE_STEP_LOWER),
                "upper": str(PRIMARY_EFFECTIVE_STEP_UPPER),
            },
            "primary_effective_sector": {
                "lower": str(primary.spec.learning_rate * LOCKED_SECTOR_LOWER),
                "upper": str(primary.spec.learning_rate * LOCKED_SECTOR_UPPER),
            },
        },
        "complex_skew_obstruction": {
            "scope": (
                "admissible constant two-dimensional map in the generic origin-centred "
                "sector; not an actual-P18 realizability claim"
            ),
            "learning_rate": str(obstruction.learning_rate),
            "curvature": str(obstruction.curvature),
            "gain": _complex((obstruction.real_gain, obstruction.skew_gain)),
            "sector_boundary_residual": "0",
            "trace": _complex(obstruction.trace),
            "determinant": _complex(obstruction.determinant),
            "characteristic_coefficient_one": _complex(obstruction.first_coefficient),
            "characteristic_coefficient_zero": _complex(obstruction.zeroth_coefficient),
            "schur_vector": _complex(
                (
                    obstruction.first_coefficient[0]
                    - (
                        obstruction.first_coefficient[0] * obstruction.zeroth_coefficient[0]
                        + obstruction.first_coefficient[1] * obstruction.zeroth_coefficient[1]
                    ),
                    obstruction.first_coefficient[1]
                    - (
                        obstruction.first_coefficient[0] * obstruction.zeroth_coefficient[1]
                        - obstruction.first_coefficient[1] * obstruction.zeroth_coefficient[0]
                    ),
                )
            ),
            "determinant_disk_margin": str(
                1 - obstruction.zeroth_coefficient[0] ** 2 - obstruction.zeroth_coefficient[1] ** 2
            ),
            "schur_cohn_margin": str(obstruction.second_schur_margin),
        },
        "controls": {
            "unprojected_shape_branch": {
                "raw_shape_gain_upper": str(RAW_SHAPE_GAIN_UPPER),
                "uniform_upper": str(unprojected.upper),
                "effective_upper_at_eta_1_over_50": str(unprojected.effective_upper_step),
                "scope": "huge generic-sector diagnostic, not an instability proof",
            },
            "projection_gain_1_over_100": {
                "projection_gain": str(SMALL_K_PROJECTION_GAIN),
                "blended_upper_at_gate_cap": str(small_blended_upper),
                "uniform_upper": str(passive_upper),
                "scope": "fidelity-only negative control, not a sector failure",
            },
        },
    }


def _interval_pass(precision_bits: int) -> dict[str, object]:
    checks = interval_certificate_checks(precision_bits=precision_bits)
    records: dict[str, object] = {}
    for name, gain in (
        ("locked_K_1", Fraction(1)),
        ("fidelity_control_K_1_over_100", Fraction(1, 100)),
    ):
        evaluated = evaluate_canonical_projected_fidelity(gain, precision_bits=precision_bits)
        records[name] = {
            "projection_gain": str(gain),
            "projection_status": evaluated.projection_status,
            "projection_scale": str(evaluated.projection_scale),
            "graph_residual_norm": str(evaluated.graph_residual_norm),
            "root_error_norm_upper": str(evaluated.exact_root_error_norm_upper),
            "yosida_error_norm_upper": str(evaluated.exact_yosida_error_norm_upper),
            "output_singular_values": [str(value) for value in evaluated.output_singular_values],
            "best_scalar_departure": str(evaluated.best_scalar_departure),
            "upstream_best_scalar_departure": str(evaluated.upstream_best_scalar_departure),
            "upstream_shaping_retention": str(evaluated.upstream_shaping_retention),
            "output_to_upstream_amplitude": str(evaluated.output_to_upstream_amplitude),
            "output_to_input_amplitude": str(evaluated.output_to_input_amplitude),
            "decisions_are_definite": evaluated.decisions_are_definite,
            "meaningful_fidelity_passes": evaluated.meaningful_fidelity_passes,
        }
    return {
        "precision_bits": precision_bits,
        "canonical": records,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
    }


def build_payload() -> dict[str, object]:
    prior, prior_checks = _prior_p17()
    exact_checks = exact_certificate_checks()
    audits = audit_all_pareto_certificates()
    interval_passes = [_interval_pass(bits) for bits in (160, 224)]
    primary = audits[1]
    all_checks = {
        **{f"exact_{name}": value for name, value in exact_checks.items()},
        **prior_checks,
        "all_pl_replays_pass": all(audit.certified for audit in audits),
        "all_cross_precision_interval_checks_pass": all(
            bool(record["all_checks_pass"]) for record in interval_passes
        ),
        "locked_canonical_fidelity_passes": all(
            bool(record["canonical"]["locked_K_1"]["meaningful_fidelity_passes"])
            for record in interval_passes
        ),
        "small_k_control_fails_fidelity": all(
            not bool(
                record["canonical"]["fidelity_control_K_1_over_100"]["meaningful_fidelity_passes"]
            )
            for record in interval_passes
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": {
            "classification": (
                "global exact-real smooth-PL convergence from a dimension-uniform "
                "origin-centred pointwise sector"
            ),
            "matrix_domain": "R^(m x n) for every fixed finite positive m,n",
            "normalization": "current Frobenius norm plus additive epsilon; no max floor",
            "epsilon": _fraction(LOCKED_EPSILON),
            "polynomial": "five composed Jordan quintics",
            "coefficients_exact": [str(value) for value in JORDAN_QUINTIC.fractions()],
            "arithmetic_model_theorem": "exact real arithmetic and exact resolvent",
            "not_claimed": [
                "an incremental sector or arbitrary-pair contraction theorem",
                "realizability of the generic eta=1/50 complex-skew obstruction by P18",
                "propagation of P16 graph residual through the nonlinear sector projection",
                "an FP64 rounding, BF16, accelerator, or literal upstream-Muon theorem",
                "weight decay, aspect scaling, stochastic gradients, or neural training",
            ],
        },
        "operator": {
            "epsilon": _fraction(LOCKED_EPSILON),
            "lambda": _fraction(LOCKED_LAMBDA),
            "mu": _fraction(LOCKED_MU),
            "p15_relative_residual_tolerance": _fraction(LOCKED_P15_KAPPA),
            "projection_gain": _fraction(LOCKED_PROJECTION_GAIN),
            "passive_divisor": _fraction(LOCKED_PASSIVE_DIVISOR),
            "gate_cap": _fraction(LOCKED_GATE_CAP),
            "gate_squared_radius_interval": [str(LOCKED_GATE_Q0), str(LOCKED_GATE_Q1)],
            "definition": ("Z=alpha*X, alpha=min(1,K*<X,S>/||X||^2); T=(1-theta)*Y/c+theta*Z"),
        },
        "pointwise_sector": {
            "lower": _fraction(LOCKED_SECTOR_LOWER),
            "upper": _fraction(LOCKED_SECTOR_UPPER),
            "center": _fraction(LOCKED_SECTOR_CENTER),
            "radius": _fraction(LOCKED_SECTOR_RADIUS),
            "supply": "||T(S)-gamma*S||_F <= K_E*||S||_F",
            "scope": "pointwise at the origin, not incremental",
        },
        "pareto_certificates": {
            audit.spec.name: {
                **_pl_fields(audit),
                "half_life_steps": audit.spec.half_life,
                "certified": audit.certified,
            }
            for audit in audits
        },
        "primary_acceptance": {
            "name": primary.spec.name,
            "learning_rate": _fraction(primary.spec.learning_rate),
            "rate_squared": _fraction(primary.spec.rate_squared),
            "half_life_steps": primary.spec.half_life,
            "p14_half_life_steps": math.log(0.5) / math.log(float(P14_RATE_SQUARED)),
            "half_life_ratio_to_p14": primary.spec.half_life
            / (math.log(0.5) / math.log(float(P14_RATE_SQUARED))),
            "exact_rate_gate": "q_primary^10 < q_P14",
            "exact_rate_gate_passes": primary.spec.rate_squared**10 < P14_RATE_SQUARED,
            "effective_update_guard": {
                "required": [str(PRIMARY_EFFECTIVE_STEP_LOWER), str(PRIMARY_EFFECTIVE_STEP_UPPER)],
                "attained": [
                    str(primary.spec.learning_rate * LOCKED_SECTOR_LOWER),
                    str(primary.spec.learning_rate * LOCKED_SECTOR_UPPER),
                ],
            },
        },
        "generic_sector_obstruction": {
            "learning_rate": _fraction(GENERIC_SECTOR_FAILED_STEP),
            "schur_cohn_margin": _fraction(generic_sector_schur_obstruction().second_schur_margin),
            "classification": (
                "strict instability of an admissible constant complex-skew endpoint in the "
                "generic sector class; not a structured-P18 counterexample"
            ),
        },
        "fidelity": {
            "canonical_input": "diag(3,4)",
            "canonical_input_exact": [str(value) for value in CANONICAL_INPUT],
            "thresholds": {
                "best_scalar_departure": str(BEST_SCALAR_DEPARTURE_GATE),
                "upstream_shaping_retention": str(UPSTREAM_SHAPING_RETENTION_GATE),
            },
            "locked_guards": {
                "departure": _guard(CANONICAL_DEPARTURE_GUARD),
                "retention": _guard(CANONICAL_RETENTION_GUARD),
                "output_to_upstream_amplitude": _guard(CANONICAL_UPSTREAM_AMPLITUDE_GUARD),
                "output_to_input_amplitude": _guard(CANONICAL_INPUT_AMPLITUDE_GUARD),
            },
            "small_k_negative_control_guards": {
                "departure": _guard(SMALL_K_DEPARTURE_GUARD),
                "retention": _guard(SMALL_K_RETENTION_GUARD),
            },
            "interval_passes": interval_passes,
        },
        "controls": reconstruction_fields()["controls"],
        "reconstruction_fields": reconstruction_fields(),
        "prior_p17": prior,
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
                "exact rational pointwise-sector algebra and PL Sylvester tests",
                "exact Schur-Cohn arithmetic for a generic-sector negative control",
                "cross-precision outward-rounded Arb canonical fidelity with residual inflation",
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
