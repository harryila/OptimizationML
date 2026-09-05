#!/usr/bin/env python3
"""Generate the exact P19 sector-shield certificate."""

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

from passive_muon.sector_shielded_inexact_resolvent_certificate import (
    FASTER_RATE_POINT,
    LOCKED_SECTOR_CENTER,
    LOCKED_SECTOR_LOWER,
    LOCKED_SECTOR_RADIUS,
    LOCKED_SECTOR_UPPER,
    MAXIMUM_STEP_POINT,
    SECTOR_SHIELDED_INEXACT_RESOLVENT_SCHEMA_VERSION,
    audit_operating_points,
    exact_certificate_checks,
    exact_shield_controls,
)

ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = "results/summaries/sector_shielded_inexact_resolvent_certificate.json"
SCHEMA_VERSION = SECTOR_SHIELDED_INEXACT_RESOLVENT_SCHEMA_VERSION

P18_ARTIFACT_PATH = "results/summaries/sector_projected_useful_rate_certificate.json"
P18_ARTIFACT_SHA256 = "42172726212f76f98b12b322ecac50ffe9fe1601bf03621f7dbb85578cdbcce6"
P18_SOURCE_COMMIT = "77e81b2462455cd96c5dfbb74b5a67a4cea5a475"
P18_ARTIFACT_COMMIT = "5e8b2f15081d887b2a7bccd029dc890f12cbfe9e"
P18_CHECKPOINT_TAG = "p18-sector-projected-useful-rate-checkpoint"
P18_CHECKPOINT_TAG_OBJECT = "147a4f10283e2afb2efcf018e7c318bacd377b60"

SOURCE_PATHS = (
    "scripts/certify_sector_shielded_inexact_resolvent.py",
    "scripts/reconstruct_sector_shielded_inexact_resolvent.py",
    "src/passive_muon/momentum_iqc.py",
    "src/passive_muon/pl_convergence.py",
    "src/passive_muon/sector_shielded_inexact_resolvent_certificate.py",
    "src/passive_muon/sector_shielded_resolvent.py",
    "experiments/resolvent/run_p19_sector_shielded_study.py",
    "tests/test_sector_shielded_inexact_resolvent_certificate.py",
    "tests/test_sector_shielded_inexact_resolvent_cli.py",
    "tests/test_sector_shielded_inexact_resolvent_reconstruction.py",
    "tests/test_sector_shielded_resolvent.py",
    "tests/test_p19_sector_shielded_study.py",
    P18_ARTIFACT_PATH,
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
    missing = [path for path in SOURCE_PATHS if not (ROOT / path).is_file()]
    if missing:
        raise RuntimeError(f"P19 exact source snapshot is incomplete: {missing}")
    return {path: _sha256(ROOT / path) for path in SOURCE_PATHS}


def _prior_p18() -> tuple[dict[str, str], dict[str, bool]]:
    path = ROOT / P18_ARTIFACT_PATH
    observed = _sha256(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    record = {
        "artifact_path": P18_ARTIFACT_PATH,
        "artifact_sha256": observed,
        "source_commit": payload["git"]["sha"],
        "artifact_commit": P18_ARTIFACT_COMMIT,
        "checkpoint_tag": P18_CHECKPOINT_TAG,
        "checkpoint_tag_object": P18_CHECKPOINT_TAG_OBJECT,
    }
    checks = {
        "p18_artifact_hash_matches": observed == P18_ARTIFACT_SHA256,
        "p18_historical_artifact_blob_matches": (
            _git_blob_sha256(P18_ARTIFACT_COMMIT, P18_ARTIFACT_PATH) == P18_ARTIFACT_SHA256
        ),
        "p18_source_commit_matches": payload["git"]["sha"] == P18_SOURCE_COMMIT,
        "p18_source_was_clean": payload["git"]["dirty"] is False,
        "p18_exact_certificate_checks_pass": isinstance(
            payload.get("audit", {}).get("checks"), dict
        )
        and all(payload["audit"]["checks"].values()),
        "p18_certified_disk_matches_P19_shield": (
            payload["pointwise_sector"]["lower"]["exact"] == str(LOCKED_SECTOR_LOWER)
            and payload["pointwise_sector"]["upper"]["exact"] == str(LOCKED_SECTOR_UPPER)
            and payload["pointwise_sector"]["center"]["exact"] == str(LOCKED_SECTOR_CENTER)
            and payload["pointwise_sector"]["radius"]["exact"] == str(LOCKED_SECTOR_RADIUS)
        ),
        "p18_checkpoint_tag_matches": (
            _git_output("rev-parse", P18_CHECKPOINT_TAG) == P18_CHECKPOINT_TAG_OBJECT
            and _git_output("rev-parse", f"{P18_CHECKPOINT_TAG}^{{}}") == P18_ARTIFACT_COMMIT
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


def reconstruction_fields() -> dict[str, object]:
    """Return the exact fields rebuilt by the standard-library script."""

    audits = audit_operating_points()
    controls = exact_shield_controls()
    return {
        "shield_geometry": {
            "matrix_domain": "R^(m x n) for arbitrary positive finite m and n",
            "sector_kind": "origin-centred pointwise, not incremental",
            "lower": str(LOCKED_SECTOR_LOWER),
            "upper": str(LOCKED_SECTOR_UPPER),
            "center": str(LOCKED_SECTOR_CENTER),
            "radius": str(LOCKED_SECTOR_RADIUS),
            "set": "D_S={U: ||U-gamma*S||_F <= radius*||S||_F}",
            "projection_nonzero": ("gamma*S+min(1,radius*||S||_F/||C-gamma*S||_F)*(C-gamma*S)"),
            "zero_displacement_rule": "the radial multiplier is 1 when C=gamma*S",
            "zero_input_rule": "Pi_{D_0}(C)=0",
            "sector_equivalence": ("<U-lower*S,upper*S-U>_F=radius^2*||S||_F^2-||U-center*S||_F^2"),
            "fixed_input_nonexpansiveness": ("||Pi_{D_S}(C1)-Pi_{D_S}(C2)||_F<=||C1-C2||_F"),
            "exact_output_fidelity_lemma": (
                "||Pi_{D_S}(C)-T18(S)||_F<=||C-T18(S)||_F for T18(S) in D_S"
            ),
            "fidelity_caveat": (
                "a P15 graph residual requires an intervening candidate-error bound"
            ),
        },
        "operating_points": {audit.point.name: _pl_fields(audit) for audit in audits},
        "exact_controls": {
            control.name: {
                "source": _vector(control.source),
                "candidate": _vector(control.candidate),
                "projected": _vector(control.projected),
                "candidate_supply": str(control.candidate_supply),
                "projected_supply": str(control.projected_supply),
                "candidate_ball_slack": str(control.candidate_ball_slack),
                "projected_ball_slack": str(control.projected_ball_slack),
                "projection_was_active": control.projection_was_active,
            }
            for control in controls
        },
    }


def build_payload() -> dict[str, object]:
    prior, prior_checks = _prior_p18()
    audits = audit_operating_points()
    exact_checks = exact_certificate_checks()
    maximum, faster = audits
    all_checks = {
        **{f"exact_{name}": value for name, value in exact_checks.items()},
        **prior_checks,
        "both_operating_points_are_strictly_certified": maximum.certified and faster.certified,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": {
            "classification": (
                "dimension-uniform exact-real pointwise-sector safety for arbitrary finite "
                "approximate candidates, with exact replay of the P18 smooth-PL rates"
            ),
            "matrix_domain": "R^(m x n) for every fixed finite positive m,n",
            "arithmetic_model_theorem": "exact real arithmetic",
            "candidate_scope": "every finite approximate or corrupted candidate C(S)",
            "not_claimed": [
                "an incremental sector or arbitrary-pair contraction theorem",
                "joint nonexpansiveness in input and candidate",
                "that the P15 graph residual alone bounds nonlinear P18 candidate error",
                "an IEEE-754, BF16, accelerator, or literal upstream-Muon theorem",
            ],
        },
        "sector_shield": {
            "lower": _fraction(LOCKED_SECTOR_LOWER),
            "upper": _fraction(LOCKED_SECTOR_UPPER),
            "center": _fraction(LOCKED_SECTOR_CENTER),
            "radius": _fraction(LOCKED_SECTOR_RADIUS),
            "formula": reconstruction_fields()["shield_geometry"],
            "exact_P18_identity": (
                "P18 globally certifies T18(S) in D_S for every finite matrix shape; "
                "therefore Pi_{D_S}(T18(S))=T18(S)"
            ),
            "finite_candidate_safety": (
                "Pi_{D_S}(C(S)) lies in D_S for every finite C(S), including S=0"
            ),
            "fp64_runtime_interface": {
                "successful_return": (
                    "the stored binary64 arrays pass an exact Fraction postcheck against "
                    "the original P18 disk"
                ),
                "projection_postcheck_failure": "replace by an exact-checked S/2 fallback",
                "nonfinite_candidate": "replace by the certified S/2 fallback",
                "nonfinite_source": "reject without returning an update",
                "unrepresentable_nonzero_subnormal": ("fail closed without returning an update"),
            },
        },
        "operating_points": {
            audit.point.name: {
                **_pl_fields(audit),
                "certified_lyapunov_rate_half_life_steps": (
                    audit.point.certified_lyapunov_rate_half_life
                ),
                "certified": audit.certified,
            }
            for audit in audits
        },
        "acceptance": {
            "maximum_step": {
                "name": maximum.point.name,
                "learning_rate": _fraction(MAXIMUM_STEP_POINT.learning_rate),
                "rate_squared": _fraction(MAXIMUM_STEP_POINT.rate_squared),
                "certified_lyapunov_rate_half_life_steps": (
                    MAXIMUM_STEP_POINT.certified_lyapunov_rate_half_life
                ),
            },
            "faster_rate": {
                "name": faster.point.name,
                "learning_rate": _fraction(FASTER_RATE_POINT.learning_rate),
                "rate_squared": _fraction(FASTER_RATE_POINT.rate_squared),
                "certified_lyapunov_rate_half_life_steps": (
                    FASTER_RATE_POINT.certified_lyapunov_rate_half_life
                ),
            },
            "faster_point_has_strictly_better_rate": (
                FASTER_RATE_POINT.rate_squared < MAXIMUM_STEP_POINT.rate_squared
            ),
        },
        "exact_corruption_controls": reconstruction_fields()["exact_controls"],
        "reconstruction_fields": reconstruction_fields(),
        "prior_p18": prior,
        "proof_replay_provenance": {
            "methods": [
                "exact rational disk-sector algebra",
                "exact rational zero, identity, radial, and tangential controls",
                "exact rational 4-by-4 smooth-PL LMI Sylvester replay",
                "standard-library-only independent reconstruction",
            ],
            "source_snapshot": _source_snapshot(),
        },
        "audit": {
            "checks": all_checks,
            "all_exact_checks_passed": all(all_checks.values()),
            "human_proof_audit": "pending; packet external review is not automated here",
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
