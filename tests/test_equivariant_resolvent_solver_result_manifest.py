from __future__ import annotations

import hashlib
import json
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CERTIFICATE = ROOT / "results/summaries/equivariant_resolvent_solver_certificate.json"
STUDY = ROOT / "results/summaries/p16_solver_study.json"


def _load(path: Path, label: str) -> dict[str, object]:
    if not path.exists():
        pytest.skip(f"committed P16 {label} has not landed yet")
    return json.loads(path.read_text(encoding="utf-8"))


def test_p16_certificate_records_clean_source_commit_and_honest_scope() -> None:
    payload = _load(CERTIFICATE, "certificate")
    assert payload["git"]["branch"] == "p16-equivariant-resolvent-solver"
    assert payload["git"]["dirty"] is False
    assert len(payload["git"]["sha"]) == 40
    int(payload["git"]["sha"], 16)
    scope = payload["claim_scope"]
    assert scope["implementation_model"] == "safeguarded FP64 checked-success reference"
    assert "fidelity gate failure" in scope["classification"]
    assert "rounding-error" in scope["not_claimed"][0]


def test_p16_certificate_records_exact_reduction_and_fidelity_failure() -> None:
    payload = _load(CERTIFICATE, "certificate")
    reduction = payload["equivariance_and_reduction"]
    assert Fraction(reduction["forward_strong_monotonicity"]["exact"]) == 2
    assert all(Fraction(row["exact"]) > 0 for row in reduction["band_diagonal_margins"])
    fidelity = payload["meaningful_fidelity_decision"]
    assert Fraction(fidelity["absolute_gate"]["exact"]) == Fraction(1, 1_000)
    assert Fraction(fidelity["retention_gate"]["exact"]) == Fraction(1, 10)
    assert fidelity["algebraic_noncollapse_passes"]
    assert not fidelity["meaningful_fidelity_passes"]
    assert all(row["all_checks_pass"] for row in fidelity["arb_cross_precision"])


def test_p16_study_records_solver_success_and_failed_joint_gate() -> None:
    payload = _load(STUDY, "solver study")
    assert payload["experiment_provenance"]["git"]["branch"] == ("p16-equivariant-resolvent-solver")
    assert payload["experiment_provenance"]["git"]["dirty"] is False
    assert payload["summary"]["certified_call_count"] == 13
    assert payload["summary"]["failed_call_count"] == 0
    assert payload["summary"]["worst_residual_to_p15_threshold_ratio"] < 1.0
    candidates = payload["stability_fidelity_frontier"]["candidates"]
    assert len(candidates) == 6
    assert any(row["frozen_p14_certificate"]["certified"] for row in candidates)
    assert not any(
        row["frozen_p14_certificate"]["certified"]
        and not row["near_scalar_on_declared_grid"]
        and row["retains_declared_fraction_of_upstream_shaping"]
        for row in candidates
    )


@pytest.mark.parametrize("path,label", [(CERTIFICATE, "certificate"), (STUDY, "solver study")])
def test_p16_manifest_source_snapshot_matches_recorded_commit(path: Path, label: str) -> None:
    payload = _load(path, label)
    if path == CERTIFICATE:
        commit = payload["git"]["sha"]
        snapshot = payload["proof_replay_provenance"]["source_snapshot"]
    else:
        commit = payload["experiment_provenance"]["git"]["sha"]
        snapshot = payload["experiment_provenance"]["source_snapshot"]
    for relative_path, expected in snapshot.items():
        completed = subprocess.run(
            ["git", "show", f"{commit}:{relative_path}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        assert hashlib.sha256(completed.stdout).hexdigest() == expected


def test_p16_human_audit_remains_explicitly_pending() -> None:
    payload = _load(CERTIFICATE, "certificate")
    assert payload["audit"]["all_exact_and_interval_checks_passed"] is True
    assert payload["audit"]["human_proof_audit"].startswith("pending")
