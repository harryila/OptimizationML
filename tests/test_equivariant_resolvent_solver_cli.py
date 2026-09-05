from __future__ import annotations

import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path


def test_p16_generator_writes_complete_scoped_payload(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "p16.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/certify_equivariant_resolvent_solver.py"),
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
    assert written["schema_version"] == ("passive-muon-equivariant-resolvent-solver-certificate-v1")
    assert written["audit"]["all_exact_and_interval_checks_passed"]
    assert all(written["audit"]["checks"].values())
    assert written["claim_scope"]["matrix_domain"] == (
        "R^(m x n) for every fixed finite positive m,n"
    )
    assert written["claim_scope"]["arithmetic_model_theorem"] == "exact real arithmetic"
    assert "fidelity gate failure" in written["claim_scope"]["classification"]

    operator = written["operator"]
    assert Fraction(operator["lambda"]["exact"]) == Fraction(1, 1_000)
    assert Fraction(operator["mu"]["exact"]) == 1_000
    assert operator["deployed_output"] == "Y_hat=(s-u_hat)/lambda=1000*(s-u_hat)"

    reduction = written["equivariance_and_reduction"]
    assert "Q*J(S)*R^T" in reduction["equivariance"]
    assert "diag(D_i)+a*v^T" in reduction["jacobian"]
    assert all(Fraction(row["exact"]) > 0 for row in reduction["band_diagonal_margins"])
    assert Fraction(reduction["forward_strong_monotonicity"]["exact"]) == 2

    witness = written["exact_forward_shaping_witness"]
    assert witness["resolvent_singular_values"] == ["3/5", "4/5"]
    assert Fraction(witness["pre_resolvent_modal_gain_difference_exact"]) > 0
    assert all(row["all_checks_pass"] for row in witness["arb_cross_precision"])

    fidelity = written["meaningful_fidelity_decision"]
    assert Fraction(fidelity["absolute_gate"]["exact"]) == Fraction(1, 1_000)
    assert Fraction(fidelity["retention_gate"]["exact"]) == Fraction(1, 10)
    assert fidelity["algebraic_noncollapse_passes"]
    assert not fidelity["meaningful_fidelity_passes"]
    assert all(row["all_checks_pass"] for row in fidelity["arb_cross_precision"])

    snapshot = written["proof_replay_provenance"]["source_snapshot"]
    assert "src/passive_muon/equivariant_resolvent_solver.py" in snapshot
    assert "theory/equivariant_resolvent_solver.md" in snapshot
    assert all(len(value) == 64 for value in snapshot.values())


def test_p16_generator_requires_two_distinct_arb_precisions(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/certify_equivariant_resolvent_solver.py"),
            "--precision-bits",
            "160",
            "--output",
            str(tmp_path / "invalid.json"),
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "at least two distinct Arb precisions" in completed.stderr
