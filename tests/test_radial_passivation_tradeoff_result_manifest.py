from __future__ import annotations

import hashlib
import json
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/summaries/radial_passivation_tradeoff_certificate.json"


def _load() -> dict[str, object]:
    if not RESULT.exists():
        pytest.skip("committed radial-passivation certificate has not landed yet")
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_p13_manifest_records_clean_source_commit_and_full_scope() -> None:
    payload = _load()
    assert payload["schema_version"] == ("passive-muon-radial-passivation-tradeoff-certificate-v1")
    assert payload["git"]["branch"] == "p13-radial-passivation-tradeoff"
    assert payload["git"]["dirty"] is False
    assert len(payload["git"]["sha"]) == 40
    int(payload["git"]["sha"], 16)
    assert payload["claim_scope"]["matrix_domain"] == (
        "R^(m x n) for every fixed finite positive m,n"
    )
    assert payload["claim_scope"]["lower_tradeoff_shape_condition"] == "min(m,n)>=2"


def test_p13_manifest_records_exact_monotonicity_and_stiffness_tradeoff() -> None:
    payload = _load()
    majorant = payload["radial_majorant"]
    upper = Fraction(majorant["unit_deficit_upper"]["exact"])
    lower = Fraction(majorant["pair_strict_lower"]["exact"])
    assert upper == Fraction(6_602_082_433_275_499_863, 41_641_817_600_000_000)
    assert lower == Fraction(98_823_281, 625_000)
    assert Fraction(majorant["z0"]["exact"]) == Fraction(63, 9_937)
    assert payload["full_matrix_monotonicity"]["dimension_uniform"] is True
    assert payload["full_matrix_monotonicity"]["not_a_diagonal_certificate"] is True
    tradeoff = payload["stiffness_tradeoff"]
    assert Fraction(tradeoff["constructed_lipschitz"]["unit_constant"]["exact"]) == upper
    assert Fraction(tradeoff["universal_lower_bound"]["unit_strict_lower"]["exact"]) == lower
    assert Fraction(tradeoff["relative_gap_percent"]["exact"]) < Fraction(28, 100)


def test_p13_manifest_records_arb_magnitudes_and_explicit_negative_controls() -> None:
    payload = _load()
    assert [row["precision_bits"] for row in payload["arb_magnitude_certificate"]["passes"]] == [
        160,
        224,
    ]
    assert len(payload["magnitude_evaluations"]) == 2
    for row in payload["magnitude_evaluations"]:
        lower = Fraction(row["radial_output_guard"]["lower_exact"])
        upper = Fraction(row["radial_output_guard"]["upper_exact"])
        constant = Fraction(row["constant_repair_output"]["exact"])
        assert 0 < lower < upper < constant
    control = payload["explicit_step_negative_control"]
    assert Fraction(control["critical_theta"]["exact"]) == Fraction(780, 29)
    assert Fraction(control["repair_only_jury_margin"]["exact"]) < 0
    assert Fraction(control["full_repaired_jury_margin"]["exact"]) < 0
    assert Fraction(control["one_step_signal"]["exact"]) == Fraction(1, 3_990_000_000)
    assert Fraction(control["one_step_normalized_signal"]["exact"]) == Fraction(1, 400)
    assert Fraction(control["one_step_repair_ratio_lower"]["exact"]) > 4_800
    assert Fraction(control["one_step_objective_growth_strict_lower"]["exact"]) > 23_325_554


def test_p13_manifest_source_snapshot_and_prior_artifacts_match() -> None:
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
    for path, record in payload["prior_artifacts"].items():
        assert record["matched"] is True
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == record["sha256"]


def test_p13_manifest_is_strictly_exact_real_and_human_audit_is_pending() -> None:
    payload = _load()
    assert payload["real_arithmetic_vs_bf16"]["theorem"] == ("continuous exact-real surrogate only")
    assert "discontinuous" in payload["real_arithmetic_vs_bf16"]["deployed_map"]
    assert payload["upstream_formula_provenance"]["revision"] == (
        "f98f1cacc0263b04290753e32be8d498c1efc806"
    )
    assert payload["audit"]["all_exact_and_interval_checks_passed"] is True
    assert payload["audit"]["human_proof_audit"].startswith("pending")
