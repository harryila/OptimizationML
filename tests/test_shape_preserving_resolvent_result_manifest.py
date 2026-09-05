from __future__ import annotations

import hashlib
import json
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CERTIFICATE = ROOT / "results" / "summaries" / "shape_preserving_resolvent_certificate.json"
STUDY = ROOT / "results" / "summaries" / "p17_shape_preserving_study.json"


def _load(path: Path, label: str) -> dict[str, object]:
    if not path.exists():
        pytest.skip(f"committed P17 {label} has not landed yet")
    return json.loads(path.read_text(encoding="utf-8"))


def test_p17_certificate_records_clean_source_commit_and_honest_scope() -> None:
    payload = _load(CERTIFICATE, "certificate")
    assert payload["git"]["branch"] == "p17-shape-preserving-resolvent"
    assert payload["git"]["dirty"] is False
    assert len(payload["git"]["sha"]) == 40
    int(payload["git"]["sha"], 16)
    scope = payload["claim_scope"]
    assert scope["arithmetic_model_theorem"] == "exact real arithmetic and exact resolvent"
    assert "pointwise, not incremental" in scope["classification"]
    assert "incremental sector" in scope["not_claimed"][0]
    assert "BF16" in scope["not_claimed"][2]


def test_p17_certificate_records_both_exact_stability_fidelity_points() -> None:
    payload = _load(CERTIFICATE, "certificate")
    assert payload["audit"]["all_exact_and_interval_checks_passed"] is True
    assert all(payload["audit"]["checks"].values())
    assert Fraction(payload["pointwise_gain_bound"]["raw_shape_gain_upper"]["exact"]) == Fraction(
        20_191_130_443_162_880_000_000,
        26_793_221_204_801_899_863,
    )
    designs = payload["designs"]
    assert set(designs) == {"high_fidelity_reduced_step", "full_step_fidelity_floor"}
    assert all(record["pl_certificate"]["certified"] for record in designs.values())
    fidelity = payload["fidelity_gate"]
    assert Fraction(fidelity["best_scalar_departure_threshold"]["exact"]) == Fraction(1, 1_000)
    assert Fraction(fidelity["upstream_shaping_retention_threshold"]["exact"]) == Fraction(1, 10)
    interval_passes = payload["unsafe_derivative"]["interval_passes"]
    assert [row["precision_bits"] for row in interval_passes] == [160, 224]
    assert all(row["all_checks_pass"] for row in interval_passes)


def test_p17_study_records_computed_residual_success_and_negative_control() -> None:
    payload = _load(STUDY, "numerical study")
    assert payload["experiment_provenance"]["git"]["branch"] == ("p17-shape-preserving-resolvent")
    assert payload["experiment_provenance"]["git"]["dirty"] is False
    assert payload["canonical_fidelity_gate"]["all_locked_designs_pass"]
    assert payload["canonical_fidelity_gate"]["thresholds_unchanged_from_p16"]
    assert all(row["certified"] for row in payload["exact_design_certificates"].values())
    solver = payload["solver_summary"]
    assert solver["call_count"] > 0
    assert solver["all_calls_computed_residual_certified"]
    assert solver["worst_residual_to_p15_threshold_ratio"] < 1.0
    controls = payload["negative_controls"]
    assert controls["undersized_passive_region"]["on_gate_plateau"]
    assert (
        controls["undersized_passive_region"]["derivative_diagnostic"]["gated_interface_derivative"]
        < 0
    )
    assert controls["locked_full_step_gate_at_same_scalar_witness"]["gate_off_below_q0"]


@pytest.mark.parametrize("path,label", [(CERTIFICATE, "certificate"), (STUDY, "study")])
def test_p17_manifest_source_snapshot_matches_recorded_commit(path: Path, label: str) -> None:
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


def test_p17_human_audit_remains_explicitly_pending() -> None:
    payload = _load(CERTIFICATE, "certificate")
    assert payload["audit"]["human_proof_audit"].startswith("pending")
