from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_MARGIN_PATH = ROOT / "results/summaries/implementation_margin_certificate.json"


def _load() -> dict:
    return json.loads(IMPLEMENTATION_MARGIN_PATH.read_text(encoding="utf-8"))


def test_p11_manifest_records_exact_margins_frontiers_and_p10_provenance() -> None:
    payload = _load()
    assert payload["schema_version"] == "passive-muon-implementation-margin-certificate-v1"
    assert payload["git"]["branch"] == "p11-certified-implementation-margin"
    assert payload["git"]["dirty"] is False
    assert payload["locked_model"]["shape"] == [4_096, 11_008]
    assert payload["locked_model"]["epsilon"].startswith("none")
    assert payload["locked_model"]["newton_schulz_iteration_count"] == 5
    assert payload["locked_model"]["polynomial_coefficients_exact"] == {
        "a": "6889/2000",
        "b": "-191/40",
        "c": "4063/2000",
    }

    zero = payload["zero_external_error_replay"]
    assert Fraction(zero["q11"]["exact"]) == Fraction(549_700_907_325, 549_755_813_888)
    assert Fraction(zero["D11"]["exact"]) == Fraction(2_162_331, 1_099_511_627_776)
    expected_ticks = {
        "gradient_slope": 10_815_225_547,
        "operator_slope": 513_245_498_810,
        "gradient_intercept": 10_879_487_718,
        "operator_intercept": 13_351_103_462_525,
    }
    maxima = payload["one_axis_grid_maxima"]
    for axis, ticks in expected_ticks.items():
        assert maxima[axis]["largest_certified_grid_ticks"] == ticks
        assert maxima[axis]["adjacent_rejected_grid_ticks"] == ticks + 1

    joint = payload["jointly_nonzero_subunit_profile"]
    assert Fraction(joint["function_gap_ultimate"]["exact"]) < 1
    assert joint["certified"] is True
    provenance = payload["p10_provenance_roles"]
    assert provenance["theorem_source_commit"].startswith("2d62b566")
    assert provenance["exact_artifact_recorded_commit"].startswith("6e7ea000")
    assert provenance["diagnostic_head_checkpoint_commit"].startswith("3246972d")
    assert provenance["human_audit_status"].startswith("pending")
    assert payload["audit"]["all_exact_checks_passed"] is True


def test_p11_source_snapshot_matches_its_clean_source_commit() -> None:
    payload = _load()
    snapshot = payload["proof_replay_provenance"]["source_snapshot"]
    assert snapshot
    for relative_path, expected_digest in snapshot.items():
        source = ROOT / relative_path
        assert source.is_file(), relative_path
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected_digest
