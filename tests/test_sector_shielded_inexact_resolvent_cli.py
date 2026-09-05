from __future__ import annotations

import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path


def test_p19_generator_writes_complete_exact_payload(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "p19.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/certify_sector_shielded_inexact_resolvent.py"),
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
    assert written["schema_version"] == (
        "passive-muon-sector-shielded-inexact-resolvent-certificate-v1"
    )
    assert written["audit"]["all_exact_checks_passed"]
    assert all(written["audit"]["checks"].values())

    scope = written["claim_scope"]
    assert scope["matrix_domain"] == "R^(m x n) for every fixed finite positive m,n"
    assert scope["candidate_scope"] == "every finite approximate or corrupted candidate C(S)"
    assert "incremental" in scope["not_claimed"][0]
    assert "graph residual alone" in scope["not_claimed"][2]

    shield = written["sector_shield"]
    assert Fraction(shield["lower"]["exact"]) == Fraction(125, 1_024)
    assert Fraction(shield["upper"]["exact"]) == Fraction(509, 512)
    assert Fraction(shield["center"]["exact"]) == Fraction(1_143, 2_048)
    assert Fraction(shield["radius"]["exact"]) == Fraction(893, 2_048)
    assert "globally certifies" in shield["exact_P18_identity"]
    assert "exact Fraction postcheck" in shield["fp64_runtime_interface"]["successful_return"]
    assert shield["fp64_runtime_interface"]["nonfinite_source"].startswith("reject")

    maximum = written["operating_points"]["maximum_step_eta_1_over_83"]
    faster = written["operating_points"]["faster_rate_eta_1_over_120"]
    assert maximum["certified"] and faster["certified"]
    assert Fraction(maximum["learning_rate"]) == Fraction(1, 83)
    assert Fraction(faster["learning_rate"]) == Fraction(1, 120)
    assert 1_724 < maximum["certified_lyapunov_rate_half_life_steps"] < 1_725
    assert 666 < faster["certified_lyapunov_rate_half_life_steps"] < 667

    controls = written["exact_corruption_controls"]
    assert Fraction(controls["radial_corruption"]["candidate_supply"]) < 0
    assert Fraction(controls["radial_corruption"]["projected_supply"]) == 0
    assert Fraction(controls["tangential_corruption"]["candidate_supply"]) < 0
    assert Fraction(controls["tangential_corruption"]["projected_supply"]) == 0

    assert written["prior_p18"]["artifact_sha256"] == (
        "42172726212f76f98b12b322ecac50ffe9fe1601bf03621f7dbb85578cdbcce6"
    )
    snapshot = written["proof_replay_provenance"]["source_snapshot"]
    assert "src/passive_muon/sector_shielded_resolvent.py" in snapshot
    assert "tests/test_sector_shielded_resolvent.py" in snapshot
    assert "scripts/reconstruct_sector_shielded_inexact_resolvent.py" in snapshot
    assert all(len(value) == 64 for value in snapshot.values())
