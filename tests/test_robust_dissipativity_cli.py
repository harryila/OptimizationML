from __future__ import annotations

import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path


def test_robust_dissipativity_cli_replays_scope_dynamics_and_provenance() -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "scripts/certify_robust_dissipativity.py")],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["schema_version"] == "passive-muon-robust-dissipativity-certificate-v1"
    scope = payload["claim_scope"]
    assert "for every disturbance realization" in scope["guarantee"]
    assert "R^(m x n)" in scope["matrix_domain"]
    assert "nonconvex" in scope["objective_domain"]
    assert scope["initialization"] == "arbitrary finite same-shaped W_0 and m_0"
    assert "exact real arithmetic" in scope["arithmetic"]
    assert "storage/output ISS" in scope["iss_scope"]
    assert "not full-state ISS in W" in scope["iss_scope"]
    assert any("BF16" in item for item in scope["not_claimed"])

    operator = payload["operator"]
    assert operator["normalization_rule"].startswith("M/max(c,||M||_F)")
    assert Fraction(operator["floor_c"]["exact"]) == 1
    assert operator["epsilon"].startswith("none")
    assert operator["polynomial_coefficients_exact"] == {
        "a": "6889/2000",
        "b": "-191/40",
        "c": "4063/2000",
    }
    assert operator["newton_schulz_iteration_count"] == 5
    assert Fraction(operator["constant_repair_rho"]["exact"]) > 0

    update = payload["disturbed_update"]
    assert "grad_f(W_t)+xi_t" in update["gradient_measurement"]
    assert "g_hat_t" in update["momentum"]
    assert "g_hat_t" in update["nesterov_signal"]
    assert "R(s_(t+1))+e_t" in update["parameter_update"]
    assert "identical xi_t" in update["reuse_rule"]

    locked = payload["locked_exact_certificate"]
    assert Fraction(locked["learning_rate_eta"]["exact"]) == Fraction(1, 32_000)
    assert Fraction(locked["rate_q"]["exact"]) == Fraction(399_960_001, 400_000_000)
    assert Fraction(locked["physical_gains"]["gamma_g"]["exact"]) == Fraction(1, 2)
    assert Fraction(locked["physical_gains"]["gamma_R"]["exact"]) == Fraction(1, 2_000_000)
    assert Fraction(locked["normalized_penalties"]["w_squared"]["exact"]) == 50
    assert len(locked["lmi_matrix"]) == 6
    assert all(len(row) == 6 for row in locked["lmi_matrix"])
    assert len(locked["negative_lmi_leading_principal_minors"]) == 6
    assert all(
        Fraction(item["exact"]) > 0 for item in locked["negative_lmi_leading_principal_minors"]
    )
    assert locked["function_coefficients"]["cancel_exactly"]
    assert locked["storage_positive_definite"]
    assert locked["lmi_negative_definite"]
    assert locked["all_exact_checks_passed"]

    consequences = payload["consequences"]
    assert "limsup" in consequences["ultimate_storage_bound"]
    ultimate = consequences["ultimate_storage_coefficients_for_norm_bounds"]
    assert Fraction(ultimate["gradient_noise_X_squared"]["exact"]) == Fraction(200_000_000, 39_999)
    assert Fraction(ultimate["implementation_error_E_squared"]["exact"]) == Fraction(200, 39_999)
    assert "V_t->0" in consequences["square_summable_inputs"]
    assert "do not imply convergence of W_t" in consequences["iterate_qualification"]
    assert "absolute summability" in consequences["iterate_qualification"]
    assert "expected ultimate neighborhood" in consequences["bounded_second_moment_stochastic"]
    assert "E[V_0]<infinity" in consequences["stochastic_assumptions"]
    assert "F_(t+1)-measurable" in consequences["stochastic_assumptions"]
    assert "variance bound" in consequences["unbiasedness_note"]
    assert "not required" in consequences["unbiasedness_note"]

    p6 = payload["p6_provenance"]
    assert p6["source_generation_commit"].startswith("a8f650f6")
    assert p6["final_checkpoint_commit"].startswith("ef88d8f5")
    assert p6["checkpoint_tag"] == "p6-checkpoint"
    assert len(p6["annotated_tag_git_oid"]) == 40
    assert len(p6["result_manifest_sha256"]) == 64

    snapshot = payload["proof_replay_provenance"]["source_snapshot"]
    assert "src/passive_muon/robust_dissipativity.py" in snapshot
    assert "tests/test_robust_dissipativity.py" in snapshot
    assert all(len(value) == 64 for value in snapshot.values())
