from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CERTIFICATE_PATH = ROOT / "results/summaries/nonquadratic_convergence_certificate.json"
PROBE_PATH = ROOT / "results/summaries/nonquadratic_convergence_falsification.json"
CODE_COMMIT = "7db9896c55bd131369e5e314a3e0175a9cf18c85"
P4_BASE_COMMIT = "c9636358de2d3d17bf5e62b0f03c7aff12da97bd"
P4_MANIFEST_SHA256 = "0d5f91232f3c4f0b4766260fea55fc540a4effb7a076f72147f2ac38ddd81cbc"
INCREMENTAL_P5_MANIFEST_SHA256 = "d9dac9bbd762586e4b84b921d5a015811d915b207d7898fcd1eedf76f83c1d9f"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_source_snapshot(snapshot: dict[str, str]) -> None:
    assert snapshot
    for relative_path, expected_digest in snapshot.items():
        actual_digest = hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
        assert actual_digest == expected_digest


def test_full_step_result_replays_exact_interpolation_storage_certificate() -> None:
    payload = _load(CERTIFICATE_PATH)

    assert payload["schema_version"] == ("passive-muon-nonquadratic-convergence-certificate-v1")
    assert payload["git"] == {
        "branch": "p5-nonquadratic-stability",
        "dirty": False,
        "frozen_p4_base_commit": P4_BASE_COMMIT,
        "sha": CODE_COMMIT,
    }

    locked = payload["locked_exact_certificate"]
    assert Fraction(locked["learning_rate_eta"]["exact"]) == Fraction(1, 32_000)
    assert Fraction(locked["exponential_rate_tau"]["exact"]) == Fraction(2_499, 2_500)
    assert Fraction(locked["storage_normalization_trace_P_plus_c_F"]["exact"]) == 1
    assert locked["storage_positive_definite"]
    assert locked["lmi_negative_definite"]
    assert locked["function_values_cancel"]
    assert locked["all_exact_checks_passed"]
    assert all(Fraction(item["exact"]) > 0 for item in locked["storage_leading_principal_minors"])
    assert all(
        Fraction(item["exact"]) > 0 for item in locked["negative_lmi_leading_principal_minors"]
    )


def test_full_step_result_records_scope_gate_and_preserved_checkpoints() -> None:
    payload = _load(CERTIFICATE_PATH)
    scope = payload["claim_scope"]
    assert "unique minimizer" in scope["guarantee"]
    assert "fixed differentiable" in scope["objective_domain"]
    assert "trajectory-varying Hessian orientations" in scope["objective_domain"]
    assert "trajectory-to-minimizer" in scope["incremental_distinction"]
    assert "not arbitrary-pair incremental" in scope["incremental_distinction"]
    assert "R^(m x n)" in scope["matrix_domain"]
    assert any("BF16" in item for item in scope["not_claimed"])
    assert any("sampled nonlinear trajectories" in item for item in scope["not_claimed"])

    gate = payload["comparison_and_gate"]
    assert gate["full_p4_step_retained"]
    assert Fraction(gate["full_step_over_incremental_p5"]["exact"]) == 20
    assert Fraction(gate["full_step_over_p3"]["exact"]) > 160
    assert Fraction(gate["rate_decrement_over_p4"]["exact"]) == 40
    assert gate["classification"] == "excellent; major nonlinear theorem"

    p4 = payload["p4_fallback"]
    assert p4["base_commit"] == P4_BASE_COMMIT
    assert p4["manifest_sha256"] == P4_MANIFEST_SHA256
    incremental = payload["incremental_p5_reference"]
    assert incremental["manifest_sha256"] == INCREMENTAL_P5_MANIFEST_SHA256
    for reference, digest in (
        (p4, P4_MANIFEST_SHA256),
        (incremental, INCREMENTAL_P5_MANIFEST_SHA256),
    ):
        assert (
            hashlib.sha256((ROOT / reference["manifest_path"]).read_bytes()).hexdigest() == digest
        )


def test_full_step_probe_records_changing_orientations_and_no_candidate() -> None:
    payload = _load(PROBE_PATH)
    assert payload["schema_version"] == ("passive-muon-nonquadratic-convergence-falsification-v1")
    assert "sampled passes do not prove" in payload["claim_scope"]["qualification"]
    assert payload["experiment_provenance"]["git"] == {
        "branch": "p5-nonquadratic-stability",
        "dirty": False,
        "sha": CODE_COMMIT,
    }

    summary = payload["summary"]
    assert summary["trial_count"] == 36
    assert summary["orientation_changing_case_count"] == 36
    assert summary["maximum_orientation_commutator"] > 0.2
    assert summary["sampled_hessian_eigenvalue_minimum"] >= 1.0 - 3e-13
    assert summary["sampled_hessian_eigenvalue_maximum"] <= 10.0 + 3e-13
    assert summary["candidate_violation_count"] == 0
    assert summary["lyapunov_rate_violating_case_count"] == 0
    assert summary["divergence_count"] == 0
    assert summary["nonfinite_count"] == 0
    assert summary["maximum_lyapunov_ratio"] < float(Fraction(2_499, 2_500) ** 2)
    assert summary["maximum_normalized_lyapunov_rate_excess"] < 0

    references = payload["certificate_references"]
    assert references["p4"]["base_commit"] == P4_BASE_COMMIT
    assert references["p4"]["manifest_sha256"] == P4_MANIFEST_SHA256
    assert references["incremental_p5"]["manifest_sha256"] == (INCREMENTAL_P5_MANIFEST_SHA256)


def test_full_step_result_source_snapshots_match_the_code_commit() -> None:
    certificate = _load(CERTIFICATE_PATH)
    probe = _load(PROBE_PATH)
    _assert_source_snapshot(certificate["proof_replay_provenance"]["source_snapshot"])
    _assert_source_snapshot(probe["experiment_provenance"]["source_snapshot"])
