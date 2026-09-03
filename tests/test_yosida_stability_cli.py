from __future__ import annotations

import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path


def test_p14_generator_writes_complete_exact_payload(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "p14.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/certify_yosida_stability.py"),
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
    assert written["schema_version"] == "passive-muon-yosida-stability-certificate-v1"
    assert written["audit"]["all_exact_checks_passed"]
    assert all(written["audit"]["checks"].values())

    scope = written["claim_scope"]
    assert scope["matrix_domain"] == "R^(m x n) for every fixed finite positive m,n"
    assert "global PL constant 1" in scope["objective_class"]
    assert "finite infimum" in scope["objective_class"]
    assert scope["arithmetic_model"] == "exact real arithmetic"
    assert "some global minimizer" in scope["guarantee"]

    operator = written["operator"]
    assert operator["epsilon_domain"] == "every real epsilon>0; 1e-7 is pinned only for controls"
    assert operator["additive_normalization"] == "M/(||M||_F+epsilon); no max floor"
    assert operator["coefficients_exact"] == {
        "a": "6889/2000",
        "b": "-191/40",
        "c": "4063/2000",
    }
    assert operator["newton_schulz_stages"] == 5
    assert operator["upstream_formula_provenance"]["revision"] == (
        "f98f1cacc0263b04290753e32be8d498c1efc806"
    )

    sector = written["yosida_sector"]
    assert Fraction(sector["lower"]["exact"]) == 500
    assert Fraction(sector["upper"]["exact"]) == 1_000
    assert Fraction(sector["center"]["exact"]) == 750
    assert Fraction(sector["residual_radius"]["exact"]) == 250
    assert "<dJ,dy>-mu*||dJ||^2" in sector["joint_incremental_iqc"]

    certificate = written["exact_pl_certificate"]
    assert Fraction(certificate["tau"]["exact"]) == Fraction(499, 500)
    assert Fraction(certificate["q"]["exact"]) == Fraction(249_001, 250_000)
    assert Fraction(certificate["D14"]["exact"]) == 0
    assert certificate["certified"]
    assert len(certificate["reconstruction_fields"]["lmi_matrix"]) == 4
    assert all(len(row) == 4 for row in certificate["reconstruction_fields"]["lmi_matrix"])
    assert len(certificate["reconstruction_fields"]["transition"]) == 2
    assert len(certificate["reconstruction_fields"]["signal_selector"]) == 4
    assert all(Fraction(row["exact"]) > 0 for row in certificate["storage_leading_minors"])
    assert all(Fraction(row["exact"]) > 0 for row in certificate["negative_lmi_leading_minors"])

    controls = written["boundary_controls"]
    assert Fraction(controls["selected_point"]["actual_curvature_10_jury_margin"]["exact"]) > 0
    assert (
        Fraction(
            controls["finite_insufficient_regularization"]["actual_curvature_10_jury_margin"][
                "exact"
            ]
        )
        < 0
    )
    assert Fraction(controls["shifted_lambda_zero"]["actual_curvature_10_jury_margin"]["exact"]) < 0
    assert (
        Fraction(controls["p13_direct_lambda_zero_baseline"]["jury_margin_curvature_1"]["exact"])
        < 0
    )
    assert written["core_boundary_cross_checks"]["all_match_literal_controls"]

    sharpness = written["abstract_sector_sharpness"]
    assert sharpness["lower_endpoint"]["attains_lower_endpoint"]
    assert Fraction(sharpness["skew_boundary"]["real_part"]["exact"]) == 600
    assert Fraction(sharpness["skew_boundary"]["imaginary_part"]["exact"]) == 200
    assert Fraction(sharpness["skew_boundary"]["sector_residual"]["exact"]) == 0
    assert sharpness["upper_endpoint_approach"]["gaps_strictly_decrease_to_zero"]

    prior = written["prior_p13"]
    assert prior["artifact_sha256"] == (
        "3541bf2478178bc7487d23dbad36b6b7cb05bfb00c82e64ca7c5386b054caed9"
    )
    snapshot = written["proof_replay_provenance"]["source_snapshot"]
    assert "src/passive_muon/yosida_stability.py" in snapshot
    assert "src/passive_muon/additive_epsilon_deficit.py" in snapshot
    assert "src/passive_muon/momentum_iqc.py" in snapshot
    assert "src/passive_muon/structure_aware_stability.py" in snapshot
    assert "theory/audits/P14_YOSIDA_STABILITY_HUMAN_PROOF_AUDIT.md" in snapshot
    assert all(len(value) == 64 for value in snapshot.values())
