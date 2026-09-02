from __future__ import annotations

import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path


def test_nonquadratic_convergence_cli_replays_full_step_and_scope() -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "scripts/certify_nonquadratic_convergence.py")],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["schema_version"] == ("passive-muon-nonquadratic-convergence-certificate-v1")
    scope = payload["claim_scope"]
    assert "unique minimizer" in scope["guarantee"]
    assert "trajectory-to-minimizer" in scope["incremental_distinction"]
    assert "trajectory-varying Hessian orientations" in scope["objective_domain"]
    assert "R^(m x n)" in scope["matrix_domain"]

    locked = payload["locked_exact_certificate"]
    assert Fraction(locked["learning_rate_eta"]["exact"]) == Fraction(1, 32_000)
    assert Fraction(locked["exponential_rate_tau"]["exact"]) == Fraction(2_499, 2_500)
    assert Fraction(locked["storage_normalization_trace_P_plus_c_F"]["exact"]) == 1
    assert locked["storage_positive_definite"]
    assert locked["lmi_negative_definite"]
    assert locked["function_values_cancel"]
    assert locked["all_exact_checks_passed"]
    assert all(
        Fraction(item["exact"]) > 0 for item in locked["negative_lmi_leading_principal_minors"]
    )

    gate = payload["comparison_and_gate"]
    assert gate["full_p4_step_retained"]
    assert Fraction(gate["full_step_over_incremental_p5"]["exact"]) == 20
    assert Fraction(gate["full_step_over_p3"]["exact"]) > 160
    assert Fraction(gate["rate_decrement_over_p4"]["exact"]) == 40
    assert gate["classification"] == "excellent; major nonlinear theorem"
    independent_audit = payload["independent_audit"]
    assert independent_audit["status"] == "passed"
    assert "not a human proof audit" in independent_audit["review_type"]
    assert independent_audit["human_proof_audit"] == "pending"
    assert (
        "not authoritative" in payload["proof_replay_provenance"]["discovery_only_solver"]["role"]
    )
