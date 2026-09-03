from __future__ import annotations

import hashlib
import json
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/summaries/yosida_stability_certificate.json"


def _load() -> dict[str, object]:
    if not RESULT.exists():
        pytest.skip("committed P14 Yosida-stability certificate has not landed yet")
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_p14_manifest_records_clean_source_commit_and_exact_scope() -> None:
    payload = _load()
    assert payload["schema_version"] == "passive-muon-yosida-stability-certificate-v1"
    assert payload["git"]["branch"] == "p14-yosida-stability"
    assert payload["git"]["dirty"] is False
    assert len(payload["git"]["sha"]) == 40
    int(payload["git"]["sha"], 16)

    scope = payload["claim_scope"]
    assert scope["matrix_domain"] == "R^(m x n) for every fixed finite positive m,n"
    assert scope["arithmetic_model"] == "exact real arithmetic"
    assert "finite infimum" in scope["objective_class"]
    assert "some global minimizer" in scope["guarantee"]
    assert any("finite-tolerance resolvent" in item for item in scope["not_claimed"])


def test_p14_manifest_records_operator_formula_and_upstream_provenance() -> None:
    payload = _load()
    operator = payload["operator"]
    assert operator["domain"] == "R^(m x n) for every fixed finite positive m,n"
    assert operator["epsilon_domain"] == "every real epsilon>0; 1e-7 is pinned only for controls"
    assert operator["additive_normalization"] == "M/(||M||_F+epsilon); no max floor"
    assert operator["coefficients_exact"] == {
        "a": "6889/2000",
        "b": "-191/40",
        "c": "4063/2000",
    }
    assert operator["newton_schulz_stages"] == 5
    upstream = operator["upstream_formula_provenance"]
    assert upstream["revision"] == "f98f1cacc0263b04290753e32be8d498c1efc806"
    assert upstream["audited_file_sha256"] == (
        "2479665a90124f62e4df557816665851ca317e42fcfda2af1da02c1f44ab5f3d"
    )
    assert "formula provenance only" in upstream["qualification"]


def test_p14_manifest_records_exact_sector_lmi_and_zero_additive_term() -> None:
    payload = _load()
    sector = payload["yosida_sector"]
    assert Fraction(sector["lower"]["exact"]) == 500
    assert Fraction(sector["upper"]["exact"]) == 1_000
    assert Fraction(sector["center"]["exact"]) == 750
    assert Fraction(sector["residual_radius"]["exact"]) == 250

    certificate = payload["exact_pl_certificate"]
    assert Fraction(certificate["tau"]["exact"]) == Fraction(499, 500)
    assert Fraction(certificate["q"]["exact"]) == Fraction(249_001, 250_000)
    assert Fraction(certificate["D14"]["exact"]) == 0
    assert certificate["certified"]
    lmi = certificate["reconstruction_fields"]["lmi_matrix"]
    assert len(lmi) == 4 and all(len(row) == 4 for row in lmi)
    assert all(lmi[row][column] == lmi[column][row] for row in range(4) for column in range(4))
    assert len(certificate["reconstruction_fields"]["transition"]) == 2
    assert len(certificate["reconstruction_fields"]["step_selector"]) == 4
    assert certificate["function_coefficients"] == certificate["expected_function_coefficients"]
    assert all(Fraction(row["exact"]) > 0 for row in certificate["storage_leading_minors"])
    assert all(Fraction(row["exact"]) > 0 for row in certificate["negative_lmi_leading_minors"])


def test_p14_manifest_records_exact_positive_negative_and_sharpness_controls() -> None:
    payload = _load()
    controls = payload["boundary_controls"]
    selected = controls["selected_point"]
    bad = controls["finite_insufficient_regularization"]
    assert Fraction(selected["actual_curvature_10_jury_margin"]["exact"]) == Fraction(
        165_247_188_769_511_634_053_550_767_361,
        42_869_153_068_208_111_929_579_445_120,
    )
    assert Fraction(bad["actual_curvature_10_jury_margin"]["exact"]) == Fraction(
        -6_764_637_554_766_138_263_886_756_183,
        10_717_453_168_649_723_982_394_861_280,
    )
    assert Fraction(selected["sector_endpoint_curvature_10_jury_margin"]["exact"]) == Fraction(
        2_467, 640
    )
    assert Fraction(bad["sector_endpoint_curvature_10_jury_margin"]["exact"]) == Fraction(-101, 160)
    assert Fraction(controls["shifted_lambda_zero"]["actual_curvature_10_jury_margin"]["exact"]) < 0
    assert (
        Fraction(controls["p13_direct_lambda_zero_baseline"]["jury_margin_curvature_1"]["exact"])
        < 0
    )

    sharpness = payload["abstract_sector_sharpness"]
    assert sharpness["lower_endpoint"]["attains_lower_endpoint"]
    assert Fraction(sharpness["skew_boundary"]["centered_circle_residual"]["exact"]) == 0
    assert Fraction(sharpness["skew_boundary"]["sector_residual"]["exact"]) == 0
    gaps = [
        Fraction(row["gap_to_upper"]["exact"])
        for row in sharpness["upper_endpoint_approach"]["sequence"]
    ]
    assert gaps[0] > gaps[1] > gaps[2] > 0


def test_p14_manifest_source_snapshot_and_p13_lock_match_history() -> None:
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

    prior = payload["prior_p13"]
    assert prior["source_commit"] == "ef566f7d04a5c2ef1a69def69724e84c67702008"
    assert prior["artifact_commit"] == "31710a9a47ecd134fecfc6fea93deba27b5f40b8"
    assert prior["checkpoint_tag"] == "p13-radial-passivation-checkpoint"
    assert prior["checkpoint_tag_object"] == "9d42f5d18bef47de26dded7809214dfdc685c5dd"
    assert (
        prior["artifact_sha256"]
        == hashlib.sha256((ROOT / prior["artifact_path"]).read_bytes()).hexdigest()
    )


def test_p14_manifest_is_exact_real_and_human_audit_is_pending() -> None:
    payload = _load()
    boundary = payload["implementation_boundary"]
    assert boundary["resolvent_evaluation"] == "mathematical exact oracle"
    assert "D14=0" in boundary["solve_error"]
    assert boundary["finite_precision"] == "not modeled"
    assert boundary["bf16"] == "not certified"
    assert payload["audit"]["all_exact_checks_passed"] is True
    assert payload["audit"]["human_proof_audit"].startswith("pending and unsigned")
