from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path


def _run_generator(root: Path, *arguments: str) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, str(root / "scripts/certify_implementation_margin.py"), *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_p11_cli_records_exact_ports_boundaries_frontiers_and_scope() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = _run_generator(root)

    assert payload["schema_version"] == "passive-muon-implementation-margin-certificate-v1"
    scope = payload["claim_scope"]
    assert "4096x11008" in scope["matrix_domain"]
    assert "2^-40" in scope["budget_maximum_scope"]
    assert any("actual instability" in item for item in scope["not_claimed"])
    ports = payload["external_port_contract"]
    assert ports["gradient_budget"] == "||zeta_t||_F<=a_g*sqrt(V_t)+b_g"
    assert ports["operator_budget"] == "||nu_t||_F<=a_R*sqrt(V_t)+b_R"
    assert ports["no_executed_nu_addition"] is True
    assert "gradient units" in ports["weight_error_conversion"]

    model = payload["locked_model"]
    assert model["shape"] == [4_096, 11_008]
    assert model["learning_rate_eta"]["exact"] == "1/32000"
    assert model["floor_c"]["exact"] == "1"
    assert model["epsilon"].startswith("none")
    assert model["newton_schulz_iteration_count"] == 5
    assert model["polynomial_coefficients_exact"] == {
        "a": "6889/2000",
        "b": "-191/40",
        "c": "4063/2000",
    }

    provenance = payload["p10_provenance_roles"]
    assert provenance == {
        "theorem_source_commit": "2d62b566e5a34895470d4eeb4b76789add826cbe",
        "exact_artifact_recorded_commit": "6e7ea000692a269cbd3766146eb7423733d18694",
        "diagnostic_head_checkpoint_commit": "3246972d44fc6e04c15cf4e205615acbb879df36",
        "human_audit_status": "pending; no external sign-off is claimed",
    }

    zero = payload["zero_external_error_replay"]
    assert Fraction(zero["q11"]["exact"]) == Fraction(549_700_907_325, 549_755_813_888)
    assert Fraction(zero["D11"]["exact"]) == Fraction(2_162_331, 1_099_511_627_776)
    assert Fraction(zero["function_gap_ultimate"]["exact"]) == Fraction(
        399_957_341_889, 549_755_813_888
    )

    expected_ticks = {
        "gradient_slope": 10_815_225_547,
        "operator_slope": 513_245_498_810,
        "gradient_intercept": 10_879_487_718,
        "operator_intercept": 13_351_103_462_525,
    }
    maxima = payload["one_axis_grid_maxima"]
    assert set(maxima) == set(expected_ticks)
    for axis, ticks in expected_ticks.items():
        item = maxima[axis]
        assert item["largest_certified_grid_ticks"] == ticks
        assert item["adjacent_rejected_grid_ticks"] == ticks + 1
        assert item["accepted_invariance_slack"]["exact"] == "0"
        assert item["rejected_invariance_slack"]["exact"] == "-1/1099511627776"
        assert item["adjacent_control_failed_checks"] == ["unit_storage_is_forward_invariant"]

    frontiers = payload["pareto_frontiers"]
    assert len(frontiers["slope_slice_b_g_eq_b_R_eq_0"]) == 9
    assert len(frontiers["intercept_slice_a_g_eq_a_R_eq_0"]) == 9
    for row in (
        *frontiers["slope_slice_b_g_eq_b_R_eq_0"],
        *frontiers["intercept_slice_a_g_eq_a_R_eq_0"],
    ):
        assert row["adjacent_rejected_gradient_grid_ticks"] == row["gradient_grid_ticks"] + 1
        assert row["adjacent_rejected_operator_grid_ticks"] == row["operator_grid_ticks"] + 1
        assert row["invariance_slack"]["exact"] == "0"
        assert row["adjacent_gradient_control_invariance_slack"]["exact"] == ("-1/1099511627776")
        assert row["adjacent_operator_control_invariance_slack"]["exact"] == ("-1/1099511627776")
        assert row["adjacent_gradient_control_failed_checks"] == [
            "unit_storage_is_forward_invariant"
        ]
        assert row["adjacent_control_failed_checks"] == ["unit_storage_is_forward_invariant"]

    joint = payload["jointly_nonzero_subunit_profile"]
    assert all(Fraction(value["exact"]) > 0 for value in joint["budget"].values())
    assert Fraction(joint["q11"]["exact"]) == Fraction(274_850_515_349, 274_877_906_944)
    assert Fraction(joint["D11"]["exact"]) == Fraction(1_254_603, 549_755_813_888)
    assert Fraction(joint["function_gap_ultimate"]["exact"]) < 1
    assert joint["certified"]
    assert payload["audit"]["all_exact_checks_passed"]

    snapshot = payload["proof_replay_provenance"]["source_snapshot"]
    assert "src/passive_muon/implementation_margin_certificate.py" in snapshot
    assert "scripts/reconstruct_implementation_margin.py" in snapshot
    assert "theory/audits/P11_HUMAN_PROOF_AUDIT.md" in snapshot
    assert all(len(digest) == 64 for digest in snapshot.values())


def test_p11_cli_output_file_matches_stdout(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "p11.json"
    payload = _run_generator(root, "--output", str(output))

    assert output.is_file()
    assert json.loads(output.read_text(encoding="utf-8")) == payload


def test_checked_p11_source_snapshot_is_current_and_clean() -> None:
    root = Path(__file__).resolve().parents[1]
    canonical_path = root / "results/summaries/implementation_margin_certificate.json"
    if not canonical_path.is_file():
        return

    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    git = canonical["git"]
    assert len(git["sha"]) == 40
    assert git["branch"] == "p11-certified-implementation-margin"
    assert git["dirty"] is False
    snapshot = canonical["proof_replay_provenance"]["source_snapshot"]
    assert snapshot
    for relative_path, expected_digest in snapshot.items():
        source = root / relative_path
        assert source.is_file(), relative_path
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected_digest
