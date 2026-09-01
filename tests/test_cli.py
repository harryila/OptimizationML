from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_counterexample_cli_emits_schema_versioned_json() -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "scripts" / "find_counterexample.py")],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["schema_version"] == "passive-muon-witness-v2"
    assert payload["local_exact_certificate"]["certified_indefinite"] is True
    assert (
        payload["fixed_scale_local_exact_certificate"][
            "certified_full_2x2_local_monotonicity_at_witness"
        ]
        is True
    )


def test_counterexample_cli_rejects_noncanonical_step_count() -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "find_counterexample.py"),
            "--steps",
            "4",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "invalid choice: '4'" in completed.stderr


def test_floored_certificate_cli_replays_two_precisions() -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "scripts" / "certify_floored_repair.py")],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["schema_version"] == "passive-muon-floored-certificate-v1"
    passes = payload["scalar_interval_certificate"]["passes"]
    assert [item["precision_bits"] for item in passes] == [160, 224]
    assert len({item["ordered_leaf_trace_sha256"] for item in passes}) == 1


def test_momentum_certificate_cli_replays_exact_lmis_and_matched_examples() -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "scripts" / "certify_momentum_stability.py")],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["schema_version"] == "passive-muon-momentum-iqc-certificate-v1"
    assert payload["analytic_sector_theorem"]["closed_form_strict_factorization"][
        "identity_replayed_exactly"
    ]
    assert payload["locked_rate_certificate"]["sylvester_positive_storage"]
    assert payload["locked_rate_certificate"]["sylvester_negative_lmi"]
    assert payload["actual_floored_jordan_local_instability"]["outside_locally_unstable"]
    examples = payload["matched_rank_one_float64_examples"]
    assert not examples["certified_stable"]["exceeded_divergence_threshold"]
    assert examples["far_outside_divergent"]["exceeded_divergence_threshold"]


def test_ema_nesterov_cli_replays_exact_lmi_and_hard_decision_rule() -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "scripts" / "certify_ema_nesterov_stability.py")],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["schema_version"] == "passive-muon-ema-nesterov-iqc-certificate-v1"
    locked = payload["locked_exact_rate_certificate"]
    assert locked["sylvester_positive_storage"]
    assert locked["sylvester_negative_lmi"]
    local = payload["actual_floored_jordan_local_control"]
    assert local["outside_locally_unstable"]
    assert local["hard_decision_rule_triggered"]
    assert local["result_classification"] == "appendix_or_proof_of_principle"
    examples = payload["matched_rank_one_float64_examples"]
    assert not examples["certified_stable"]["exceeded_divergence_threshold"]
    assert examples["far_outside_divergent"]["exceeded_divergence_threshold"]
