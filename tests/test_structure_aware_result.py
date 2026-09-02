from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "results" / "summaries" / "structure_aware_stability_certificate.json"


def _payload() -> dict:
    return json.loads(RESULT_PATH.read_text(encoding="utf-8"))


def test_structure_aware_result_locks_exact_certificate_and_scope() -> None:
    payload = _payload()
    assert payload["schema_version"] == ("passive-muon-structure-aware-stability-certificate-v1")
    assert payload["claim_scope"]["matrix_domain"] == (
        "R^(m x n) for every fixed finite positive m,n"
    )
    assert "I <= H <= 10*I" in payload["claim_scope"]["quadratic_domain"]
    assert payload["operator"]["normalization"] == "fixed_frobenius_floor"
    assert payload["operator"]["additive_epsilon"].startswith("not_applicable")
    assert payload["operator"]["steps"] == 5

    locked = payload["locked_exact_certificate"]
    assert Fraction(locked["learning_rate_eta"]["exact"]) == Fraction(1, 32_000)
    assert Fraction(locked["exponential_rate_tau"]["exact"]) == Fraction(99_999, 100_000)
    assert locked["all_exact_checks_passed"]
    assert all(item["strictly_positive"] for item in locked["scaled_jury"])
    assert locked["frequency_gap"]["q2_bernstein_certificate"]["strictly_positive"]
    assert locked["frequency_gap"]["negative_discriminant_bernstein_certificate"][
        "strictly_positive"
    ]


def test_structure_aware_result_passes_gate_and_records_sampled_nonproof() -> None:
    payload = _payload()
    comparison = payload["comparison_to_p3_and_linked_control"]
    assert comparison["predeclared_at_least_100x_gate_passed"]
    assert comparison["within_100x_of_linked_local_threshold"]
    assert 161 < float(comparison["p4_over_p3_learning_rate"]["decimal"]) < 162
    assert 66 < float(comparison["linked_local_threshold_over_p4"]["decimal"]) < 67

    probe = payload["sampled_full_matrix_adversarial_probe"]
    assert "sampling diagnostic only" in probe["claim_scope"]
    assert probe["experiment_provenance"]["seed"] == 2_026_090_2
    assert probe["summary"]["trial_count"] == 48
    assert probe["summary"]["divergence_count"] == 0
    assert probe["summary"]["nonfinite_count"] == 0


def test_structure_aware_result_was_generated_cleanly_from_hashed_sources() -> None:
    payload = _payload()
    git = payload["git"]
    assert git["branch"] == "p4-structure-aware-stability"
    assert git["dirty"] is False
    assert git["sha"] == "5a505d47d5b737f39e4c8f5c93419072e0cc31f5"

    snapshot = payload["experiment_provenance"]["source_snapshot"]
    assert snapshot
    for relative_path, expected_digest in snapshot.items():
        actual_digest = hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
        assert actual_digest == expected_digest
