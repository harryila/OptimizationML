from __future__ import annotations

import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path


def test_p18_generator_writes_complete_scoped_payload(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "p18.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/certify_sector_projected_useful_rate.py"),
            "--output",
            str(output),
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    stdout = json.loads(completed.stdout)
    written = json.loads(output.read_text(encoding="utf-8"))

    assert stdout == written
    assert written["schema_version"] == ("passive-muon-sector-projected-resolvent-certificate-v1")
    assert written["audit"]["all_exact_and_interval_checks_passed"]
    assert all(written["audit"]["checks"].values())

    scope = written["claim_scope"]
    assert scope["matrix_domain"] == "R^(m x n) for every fixed finite positive m,n"
    assert scope["arithmetic_model_theorem"] == "exact real arithmetic and exact resolvent"
    assert "pointwise" in scope["classification"]
    assert "incremental" in scope["not_claimed"][0]
    assert Fraction(scope["epsilon"]["exact"]) == Fraction(1, 10_000_000)
    assert scope["coefficients_exact"] == ["6889/2000", "-191/40", "4063/2000"]

    sector = written["pointwise_sector"]
    assert Fraction(sector["lower"]["exact"]) == Fraction(125, 1_024)
    assert Fraction(sector["upper"]["exact"]) == Fraction(509, 512)
    assert Fraction(sector["center"]["exact"]) == Fraction(1_143, 2_048)
    assert Fraction(sector["radius"]["exact"]) == Fraction(893, 2_048)

    certificates = written["pareto_certificates"]
    assert len(certificates) == 6
    assert all(record["certified"] for record in certificates.values())
    primary = written["primary_acceptance"]
    assert primary["name"] == "primary_eta_1_over_83"
    assert Fraction(primary["learning_rate"]["exact"]) == Fraction(1, 83)
    assert primary["exact_rate_gate_passes"]
    assert primary["half_life_ratio_to_p14"] < 10

    obstruction = written["generic_sector_obstruction"]
    assert Fraction(obstruction["learning_rate"]["exact"]) == Fraction(1, 50)
    assert Fraction(obstruction["schur_cohn_margin"]["exact"]) < 0
    assert "not a structured-P18 counterexample" in obstruction["classification"]

    assert all(record["all_checks_pass"] for record in written["fidelity"]["interval_passes"])
    assert written["prior_p17"]["artifact_sha256"] == (
        "6ceea44127493cc3cb811826879413f87363a21457670e0e8be687aaef980455"
    )
    snapshot = written["proof_replay_provenance"]["source_snapshot"]
    assert "src/passive_muon/sector_projected_resolvent.py" in snapshot
    assert "scripts/reconstruct_sector_projected_useful_rate.py" in snapshot
    assert all(len(value) == 64 for value in snapshot.values())
