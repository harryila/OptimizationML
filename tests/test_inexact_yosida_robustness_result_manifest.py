from __future__ import annotations

import hashlib
import json
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/summaries/inexact_yosida_robustness_certificate.json"


def _load() -> dict[str, object]:
    if not RESULT.exists():
        pytest.skip("committed P15 inexact-Yosida certificate has not landed yet")
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_p15_manifest_records_clean_source_commit_and_scope() -> None:
    payload = _load()
    assert payload["schema_version"] == ("passive-muon-inexact-yosida-robustness-certificate-v1")
    assert payload["git"]["branch"] == "p15-inexact-yosida-robustness"
    assert payload["git"]["dirty"] is False
    assert len(payload["git"]["sha"]) == 40
    int(payload["git"]["sha"], 16)
    assert payload["claim_scope"]["matrix_domain"] == (
        "R^(m x n) for every fixed finite positive m,n"
    )
    assert payload["claim_scope"]["arithmetic_model"] == "exact real arithmetic"
    assert "stopping rule" in payload["claim_scope"]["guarantee"]


def test_p15_manifest_records_exact_residual_conversion_and_output_convention() -> None:
    payload = _load()
    residual = payload["inexact_resolvent"]
    assert residual["equation_residual"] == "r=s-u_hat-lambda*B(u_hat)"
    assert residual["approximate_output"] == "Y_hat=(s-u_hat)/lambda"
    assert Fraction(residual["forward_strong_monotonicity"]["exact"]) == 2
    assert Fraction(residual["solution_error_gain"]["exact"]) == Fraction(1, 2)
    assert Fraction(residual["output_error_gain"]["exact"]) == 500
    assert "different output convention" in residual["graph_output_not_covered"]
    witness = residual["sharpness_witness"]
    assert Fraction(witness["solution_error_ratio"]["exact"]) == Fraction(1, 2)
    assert Fraction(witness["output_error_ratio"]["exact"]) == 500


def test_p15_manifest_records_full_exact_lmi_and_nonvacuous_bound() -> None:
    payload = _load()
    certificate = payload["exact_robust_certificate"]
    assert Fraction(certificate["q"]["exact"]) == Fraction(249_001, 250_000)
    assert Fraction(certificate["C15"]["exact"]) == Fraction(5, 2)
    assert Fraction(certificate["normalized_absolute_output_penalty"]["exact"]) == Fraction(
        1_125, 2
    )
    assert certificate["certified"]
    assert len(certificate["lmi_matrix"]) == 5
    assert all(len(row) == 5 for row in certificate["lmi_matrix"])
    assert all(Fraction(row["exact"]) > 0 for row in certificate["negative_lmi_leading_minors"])
    assert Fraction(
        payload["consequences"]["ultimate_storage"]["coefficient"]["exact"]
    ) == Fraction(625_000, 999)
    assert Fraction(
        payload["consequences"]["ultimate_objective"]["coefficient"]["exact"]
    ) == Fraction(6_250_000_000_000, 312_929_757)


def test_p15_manifest_records_qualified_certificate_and_true_negative_controls() -> None:
    payload = _load()
    controls = payload["controls"]
    frozen = controls["frozen_radius"]
    assert frozen["passing"]
    assert Fraction(frozen["passing_radius"]["exact"]) == 252
    assert Fraction(frozen["rejected_radius"]["exact"]) == 253
    assert Fraction(frozen["rejected_fourth_minor"]["exact"]) < 0
    assert "not an instability" in frozen["qualification"]

    penalty = controls["absolute_output_penalty"]
    assert (
        Fraction(penalty["insufficient"]["exact"])
        < Fraction(penalty["critical"]["exact"])
        < Fraction(penalty["selected"]["exact"])
    )
    assert Fraction(penalty["critical_fifth_minor"]["exact"]) == 0
    assert Fraction(penalty["insufficient_fifth_minor"]["exact"]) < 0
    assert Fraction(penalty["selected_fifth_minor"]["exact"]) > 0

    stopping = controls["abstract_stopping_boundary"]
    assert Fraction(stopping["kappa_one"]["approximate_output_slope"]["exact"]) == 0
    assert Fraction(stopping["kappa_one"]["p_at_one"]["exact"]) == 0
    assert Fraction(stopping["kappa_two"]["approximate_output_slope"]["exact"]) == -500
    assert Fraction(stopping["kappa_two"]["p_at_one"]["exact"]) == Fraction(-1, 1_280)


def test_p15_manifest_source_snapshot_and_p14_lock_match_history() -> None:
    payload = _load()
    snapshot = payload["proof_replay_provenance"]["source_snapshot"]
    commit = payload["git"]["sha"]
    for path, expected in snapshot.items():
        completed = subprocess.run(
            ["git", "show", f"{commit}:{path}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        assert hashlib.sha256(completed.stdout).hexdigest() == expected

    prior = payload["prior_p14"]
    assert prior["artifact_sha256"] == (
        "8f31416eb8e278bc8a3765f72b528f5f538e33f28d5baedc5888c5d849447c62"
    )
    assert prior["source_commit"] == "ea18aa856e3af6d025e325a6db27a681f09c0e7a"
    assert prior["artifact_commit"] == "e323a4d1e73050fb6b6d8bcd09c515ea30de3553"
    assert prior["checkpoint_tag"] == "p14-yosida-stability-checkpoint"
    assert prior["checkpoint_tag_object"] == "986fade33e4bd90bed091798d2415d1e0ab608a2"


def test_p15_manifest_is_exact_real_and_human_audit_is_pending() -> None:
    payload = _load()
    boundary = payload["implementation_boundary"]
    assert boundary["solver"] == "not specified"
    assert boundary["residual_verification"] == "required at every iteration"
    assert boundary["output_convention"] == "resolvent form (s-u_hat)/lambda only"
    assert boundary["finite_precision"] == "not modeled"
    assert boundary["bf16"] == "not certified"
    assert payload["audit"]["all_exact_checks_passed"] is True
    assert payload["audit"]["human_proof_audit"].startswith("pending and unsigned")
