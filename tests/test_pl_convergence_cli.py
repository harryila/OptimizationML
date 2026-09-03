from __future__ import annotations

import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path


def test_pl_convergence_cli_replays_full_step_scope_and_provenance() -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "scripts/certify_pl_convergence.py")],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["schema_version"] == "passive-muon-pl-convergence-certificate-v1"
    scope = payload["claim_scope"]
    assert "function-value convergence and momentum decay" in scope["guarantee"]
    assert "nonconvex" in scope["objective_domain"]
    assert "finite infimum" in scope["objective_domain"]
    assert "not arbitrary-pair incremental contraction" in scope["incremental_distinction"]
    assert "not to a unique" in scope["iterate_corollary"]
    assert "R^(m x n)" in scope["matrix_domain"]

    operator = payload["operator"]
    assert operator["normalization"] == "exact fixed Frobenius max floor"
    assert Fraction(operator["floor_c"]["exact"]) == 1
    assert operator["additive_epsilon"].startswith("none")
    assert operator["coefficients_exact"] == {
        "a": "6889/2000",
        "b": "-191/40",
        "c": "4063/2000",
    }
    assert operator["newton_schulz_steps"] == 5
    assert Fraction(operator["constant_repair_rho"]["exact"]) > 0

    objective = payload["objective_class"]
    assert Fraction(objective["smoothness_L"]["exact"]) == 10
    assert Fraction(objective["pl_constant_ell_PL"]["exact"]) == 1
    assert Fraction(objective["normalized_pl_constant_k"]["exact"]) == Fraction(1, 10)
    assert "inf_W" in objective["finite_infimum_assumption"]
    assert "negative curvature is allowed" in objective["interpolation_curvature_class"]
    assert objective["weighted_function_coefficients"]["cancel_exactly"]

    locked = payload["locked_exact_certificate"]
    assert Fraction(locked["learning_rate_eta"]["exact"]) == Fraction(1, 32_000)
    assert Fraction(locked["exponential_norm_rate_tau"]["exact"]) == Fraction(19_999, 20_000)
    assert Fraction(locked["function_value_rate_q"]["exact"]) == Fraction(399_960_001, 400_000_000)
    assert Fraction(locked["storage_normalization_trace_P_plus_c_F"]["exact"]) == 1
    assert locked["storage_positive_definite"]
    assert locked["lmi_negative_definite"]
    assert locked["function_values_cancel"]
    assert locked["all_exact_checks_passed"]
    assert all(
        Fraction(item["exact"]) > 0 for item in locked["negative_lmi_leading_principal_minors"]
    )

    consequence = payload["consequence"]
    assert "q^t" in consequence["function_value_bound"]
    assert consequence["uniqueness"] == "not assumed and not concluded"
    gate = payload["comparison_and_gate"]
    assert gate["full_p5_step_retained"]
    assert gate["classification"] == "major p6 result"
    assert Fraction(gate["p6_rate_decrement_as_fraction_of_p5"]["exact"]) == Fraction(1, 8)

    checkpoint = payload["p5_checkpoint"]
    assert checkpoint["tag"] == "p5-checkpoint"
    assert checkpoint["peeled_commit"] == "a549fb4c206335ef9ec264524e0f581216b250d4"
    assert len(checkpoint["annotated_tag_git_oid"]) == 40
    assert len(checkpoint["full_step_manifest_sha256"]) == 64
    assert len(checkpoint["incremental_manifest_sha256"]) == 64

    audit = payload["independent_audit"]
    assert audit["status"] == "passed"
    assert "automated" in audit["review_type"]
    assert "not a human proof audit" in audit["review_type"]
    assert audit["human_proof_audit"] == "pending"
    discovery = payload["proof_replay_provenance"]["discovery_only_solver"]
    assert "nonauthoritative" in discovery["role"]
    snapshot = payload["proof_replay_provenance"]["source_snapshot"]
    assert "src/passive_muon/pl_convergence.py" in snapshot
    assert "theory/pl_convergence_certificate.md" in snapshot
    assert all(len(value) == 64 for value in snapshot.values())
