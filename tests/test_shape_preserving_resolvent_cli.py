from __future__ import annotations

import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path


def test_p17_generator_writes_complete_scoped_payload(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "p17.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/certify_shape_preserving_resolvent.py"),
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
    assert written["schema_version"] == ("passive-muon-shape-preserving-resolvent-certificate-v1")
    assert written["audit"]["all_exact_and_interval_checks_passed"]
    assert all(written["audit"]["checks"].values())
    scope = written["claim_scope"]
    assert scope["matrix_domain"] == "R^(m x n) for every fixed finite positive m,n"
    assert scope["arithmetic_model_theorem"] == "exact real arithmetic and exact resolvent"
    assert "pointwise, not incremental" in scope["classification"]
    assert Fraction(scope["epsilon"]["exact"]) == Fraction(1, 10_000_000)
    assert scope["coefficients_exact"] == ["6889/2000", "-191/40", "4063/2000"]

    operator = written["operator"]
    assert Fraction(operator["lambda"]["exact"]) == Fraction(1, 1_000)
    assert Fraction(operator["mu"]["exact"]) == 1_000
    assert operator["interface"].startswith("T=(1-theta")

    assert Fraction(written["pointwise_gain_bound"]["raw_shape_gain_upper"]["exact"]) == Fraction(
        20_191_130_443_162_880_000_000,
        26_793_221_204_801_899_863,
    )
    designs = written["designs"]
    assert set(designs) == {"high_fidelity_reduced_step", "full_step_fidelity_floor"}
    assert Fraction(designs["high_fidelity_reduced_step"]["config"]["learning_rate"]["exact"]) == (
        Fraction(1, 128_000)
    )
    assert Fraction(designs["full_step_fidelity_floor"]["config"]["learning_rate"]["exact"]) == (
        Fraction(1, 32_000)
    )
    assert all(record["pl_certificate"]["certified"] for record in designs.values())

    fidelity = written["fidelity_gate"]
    assert Fraction(fidelity["best_scalar_departure_threshold"]["exact"]) == Fraction(1, 1_000)
    assert Fraction(fidelity["upstream_shaping_retention_threshold"]["exact"]) == Fraction(1, 10)
    assert "both locked points" in fidelity["classification"]
    assert all(row["all_checks_pass"] for row in written["unsafe_derivative"]["interval_passes"])

    snapshot = written["proof_replay_provenance"]["source_snapshot"]
    assert "src/passive_muon/shape_preserving_resolvent.py" in snapshot
    assert "tests/test_shape_preserving_resolvent_certificate.py" in snapshot
    assert "theory/shape_preserving_resolvent.md" in snapshot
    assert all(len(value) == 64 for value in snapshot.values())
