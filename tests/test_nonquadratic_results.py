from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CERTIFICATE_PATH = ROOT / "results/summaries/nonquadratic_stability_certificate.json"
PROBE_PATH = ROOT / "results/summaries/nonquadratic_falsification.json"
CODE_COMMIT = "f366af4345328fe66ffb269a4fe13ffe6d55d583"
P4_MANIFEST_SHA256 = "0d5f91232f3c4f0b4766260fea55fc540a4effb7a076f72147f2ac38ddd81cbc"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_source_snapshot(snapshot: dict[str, str]) -> None:
    assert snapshot
    for relative_path, expected_digest in snapshot.items():
        actual_digest = hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
        assert actual_digest == expected_digest


def test_nonquadratic_result_replays_exact_common_storage_certificate() -> None:
    payload = _load(CERTIFICATE_PATH)
    assert payload["schema_version"] == "passive-muon-nonquadratic-stability-certificate-v1"
    assert payload["git"] == {
        "branch": "p5-nonquadratic-stability",
        "dirty": False,
        "frozen_p4_base_commit": "c9636358de2d3d17bf5e62b0f03c7aff12da97bd",
        "sha": CODE_COMMIT,
    }

    locked = payload["locked_exact_certificate"]
    assert Fraction(locked["learning_rate_eta"]["exact"]) == Fraction(1, 640_000)
    assert Fraction(locked["exponential_rate_tau"]["exact"]) == Fraction(99_999, 100_000)
    assert locked["all_exact_checks_passed"]
    assert locked["storage_positive_definite"]
    assert locked["lmi_negative_definite"]
    assert all(Fraction(item["exact"]) > 0 for item in locked["storage_leading_principal_minors"])
    assert all(
        Fraction(item["exact"]) > 0 for item in locked["negative_lmi_leading_principal_minors"]
    )


def test_nonquadratic_result_records_scope_gate_and_p4_fallback() -> None:
    payload = _load(CERTIFICATE_PATH)
    scope = payload["claim_scope"]
    assert "fixed differentiable" in scope["objective_domain"]
    assert "trajectory-varying local Hessian orientations" in scope["objective_domain"]
    assert "R^(m x n)" in scope["matrix_domain"]
    assert any("BF16" in item for item in scope["not_claimed"])
    assert any("richer dynamic" in item for item in scope["not_claimed"])

    gate = payload["comparison_and_gate"]
    assert Fraction(gate["p5_over_p4"]["exact"]) == Fraction(1, 20)
    assert Fraction(gate["p5_over_p3"]["exact"]) > 8
    assert gate["classification"] == "strong nonlinear extension"
    assert payload["discovery_only_diagnostics"]["authoritative_for_theorem"] is False

    fallback = payload["p4_fallback"]
    assert fallback["base_commit"] == "c9636358de2d3d17bf5e62b0f03c7aff12da97bd"
    assert fallback["manifest_sha256"] == P4_MANIFEST_SHA256
    assert hashlib.sha256((ROOT / fallback["manifest_path"]).read_bytes()).hexdigest() == (
        P4_MANIFEST_SHA256
    )


def test_nonquadratic_probe_records_changing_orientations_and_no_candidate() -> None:
    payload = _load(PROBE_PATH)
    assert payload["schema_version"] == "passive-muon-nonquadratic-falsification-v1"
    assert "sampled passes do not certify" in payload["claim_scope"]
    assert payload["experiment_provenance"]["git"] == {
        "branch": "p5-nonquadratic-stability",
        "dirty": False,
        "sha": CODE_COMMIT,
    }
    assert payload["p4_reference"]["manifest_sha256"] == P4_MANIFEST_SHA256

    summary = payload["summary"]
    assert summary["trial_count"] == 36
    assert summary["orientation_changing_case_count"] == 36
    assert summary["maximum_orientation_commutator"] > 1e-3
    assert summary["sampled_secant_eigenvalue_minimum"] >= 1.0 - 1e-12
    assert summary["sampled_secant_eigenvalue_maximum"] <= 10.0 + 1e-12
    assert summary["candidate_instability_count"] == 0
    assert summary["storage_rate_violating_case_count"] == 0
    assert summary["divergence_count"] == 0
    assert summary["nonfinite_count"] == 0
    assert summary["maximum_normalized_storage_rate_excess"] < 0


def test_nonquadratic_result_source_snapshots_match_the_code_commit() -> None:
    certificate = _load(CERTIFICATE_PATH)
    probe = _load(PROBE_PATH)
    _assert_source_snapshot(certificate["proof_replay_provenance"]["source_snapshot"])
    _assert_source_snapshot(probe["experiment_provenance"]["source_snapshot"])
