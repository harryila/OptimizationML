from __future__ import annotations

import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path


def test_structure_aware_cli_replays_exact_certificate_and_gate() -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "certify_structure_aware_stability.py"),
            "--probe-trials-per-case",
            "1",
            "--probe-iterations",
            "5",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["schema_version"] == ("passive-muon-structure-aware-stability-certificate-v1")
    locked = payload["locked_exact_certificate"]
    assert Fraction(locked["learning_rate_eta"]["exact"]) == Fraction(1, 32_000)
    assert Fraction(locked["exponential_rate_tau"]["exact"]) == Fraction(99_999, 100_000)
    assert locked["all_exact_checks_passed"]
    assert all(item["strictly_positive"] for item in locked["scaled_jury"])
    assert locked["frequency_gap"]["q2_bernstein_certificate"]["strictly_positive"]
    assert locked["frequency_gap"]["negative_discriminant_bernstein_certificate"][
        "strictly_positive"
    ]

    comparison = payload["comparison_to_p3_and_linked_control"]
    assert comparison["predeclared_at_least_100x_gate_passed"]
    assert comparison["within_100x_of_linked_local_threshold"]
    assert float(comparison["p4_over_p3_learning_rate"]["decimal"]) > 161

    probe = payload["sampled_full_matrix_adversarial_probe"]
    assert "sampling diagnostic only" in probe["claim_scope"]
    assert probe["summary"]["divergence_count"] == 0
    assert probe["summary"]["nonfinite_count"] == 0
