from __future__ import annotations

import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path


def test_nonquadratic_certificate_cli_replays_exact_lmi_and_scope() -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "scripts/certify_nonquadratic_stability.py")],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["schema_version"] == "passive-muon-nonquadratic-stability-certificate-v1"
    scope = payload["claim_scope"]
    assert "every fixed differentiable" in scope["objective_domain"]
    assert "trajectory-varying local Hessian orientations" in scope["objective_domain"]
    assert scope["matrix_domain"] == "R^(m x n) for every fixed finite positive m,n"

    locked = payload["locked_exact_certificate"]
    assert Fraction(locked["learning_rate_eta"]["exact"]) == Fraction(1, 640_000)
    assert Fraction(locked["exponential_rate_tau"]["exact"]) == Fraction(99_999, 100_000)
    assert locked["storage_positive_definite"]
    assert locked["lmi_negative_definite"]
    assert locked["all_exact_checks_passed"]
    assert all(
        Fraction(item["exact"]) > 0 for item in locked["negative_lmi_leading_principal_minors"]
    )

    gate = payload["comparison_and_gate"]
    assert Fraction(gate["p5_over_p4"]["exact"]) == Fraction(1, 20)
    assert Fraction(gate["p5_over_p3"]["exact"]) > 8
    assert gate["classification"] == "strong nonlinear extension"
    assert payload["p4_fallback"]["status"] == "preserved; p5 is a separate branch"
    assert payload["discovery_only_diagnostics"]["authoritative_for_theorem"] is False
