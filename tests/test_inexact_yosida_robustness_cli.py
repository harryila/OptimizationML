from __future__ import annotations

import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path


def test_p15_generator_writes_complete_exact_payload(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "p15.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/certify_inexact_yosida_robustness.py"),
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
    assert written["schema_version"] == ("passive-muon-inexact-yosida-robustness-certificate-v1")
    assert written["audit"]["all_exact_checks_passed"]
    assert all(written["audit"]["checks"].values())
    assert written["claim_scope"]["matrix_domain"] == (
        "R^(m x n) for every fixed finite positive m,n"
    )
    assert written["claim_scope"]["arithmetic_model"] == "exact real arithmetic"

    residual = written["inexact_resolvent"]
    assert residual["equation_residual"] == "r=s-u_hat-lambda*B(u_hat)"
    assert residual["approximate_output"] == "Y_hat=(s-u_hat)/lambda"
    assert Fraction(residual["solution_error_gain"]["exact"]) == Fraction(1, 2)
    assert Fraction(residual["output_error_gain"]["exact"]) == 500
    assert "1000" in residual["graph_output_not_covered"]

    stopping = written["stopping_rule"]
    assert stopping["rule"] == "||r||_F<=kappa*||s||_F+rbar at every oracle call"
    assert Fraction(stopping["relative_kappa"]["exact"]) == Fraction(1, 250)
    assert Fraction(stopping["relative_output_radius"]["exact"]) == 2
    assert Fraction(stopping["p14_centered_radius"]["exact"]) == 250
    assert Fraction(stopping["effective_centered_radius"]["exact"]) == 252

    certificate = written["exact_robust_certificate"]
    assert Fraction(certificate["q"]["exact"]) == Fraction(249_001, 250_000)
    assert Fraction(certificate["C15"]["exact"]) == Fraction(5, 2)
    assert Fraction(certificate["absolute_output_penalty"]["exact"]) == Fraction(1, 100_000)
    assert certificate["certified"]
    assert len(certificate["lmi_matrix"]) == 5
    assert all(len(row) == 5 for row in certificate["lmi_matrix"])
    assert all(Fraction(row["exact"]) > 0 for row in certificate["negative_lmi_leading_minors"])

    controls = written["controls"]
    assert controls["frozen_radius"]["passing"]
    assert Fraction(controls["frozen_radius"]["rejected_fourth_minor"]["exact"]) < 0
    penalty = controls["absolute_output_penalty"]
    assert (
        Fraction(penalty["insufficient"]["exact"])
        < Fraction(penalty["critical"]["exact"])
        < Fraction(penalty["selected"]["exact"])
    )
    assert Fraction(penalty["critical_fifth_minor"]["exact"]) == 0
    assert Fraction(penalty["insufficient_fifth_minor"]["exact"]) < 0
    assert Fraction(penalty["selected_fifth_minor"]["exact"]) > 0
    assert Fraction(
        controls["abstract_stopping_boundary"]["kappa_two"]["p_at_one"]["exact"]
    ) == Fraction(-1, 1_280)

    snapshot = written["proof_replay_provenance"]["source_snapshot"]
    assert "src/passive_muon/inexact_yosida_robustness.py" in snapshot
    assert "theory/inexact_yosida_robustness_certificate.md" in snapshot
    assert all(len(value) == 64 for value in snapshot.values())
