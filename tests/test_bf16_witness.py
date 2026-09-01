from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest
import torch

from passive_muon.bf16_witness import (
    BF16_WITNESS_SCHEMA_VERSION,
    KELLER_JORDAN_REVISION,
    canonical_bf16_witness,
)


def test_bf16_witness_schema_records_full_operator_configuration() -> None:
    payload = canonical_bf16_witness()
    configuration = payload["configuration"]
    orthogonalizer = configuration["orthogonalizer"]

    assert payload["schema_version"] == BF16_WITNESS_SCHEMA_VERSION
    assert payload["claim_scope"]["evidence_kind"] == "backend_specific_executable_pairwise_witness"
    assert len(payload["claim_scope"]["not_claimed"]) >= 4
    assert configuration["pair_restriction"] == "positive_diagonal"
    assert configuration["normalization"]["epsilon"] == pytest.approx(1e-7)
    assert configuration["normalization"]["matches_pinned_upstream_epsilon"] is True
    assert orthogonalizer["steps"] == 5
    assert orthogonalizer["coefficients"]["a"]["source_decimal"] == "3.4445"
    assert orthogonalizer["coefficients"]["b"]["source_decimal"] == "-4.7750"
    assert orthogonalizer["coefficients"]["c"]["source_decimal"] == "2.0315"
    assert orthogonalizer["recurrence"] == "A=X@X.T; B=b*A+(c*A)@A; X=a*X+B@X"
    assert configuration["upstream"]["revision"] == KELLER_JORDAN_REVISION
    assert any("left-associatively" in step for step in configuration["operation_order"])


def test_bf16_witness_output_is_internally_consistent_on_current_backend() -> None:
    payload = canonical_bf16_witness()
    pair = payload["canonical_pair"]
    measurement = pair["measurement"]
    exact_gap = Fraction(measurement["gap_exact_fraction_from_individually_recorded_values"])
    exact_distance = Fraction(measurement["distance_squared_exact_fraction"])
    exact_ratio = Fraction(measurement["ratio_exact_fraction_from_individually_recorded_values"])
    runtime_gap = Fraction.from_float(measurement["gap_decimal"])

    assert pair["left_output"]["dtype"] == "torch.bfloat16"
    assert pair["right_output"]["dtype"] == "torch.bfloat16"
    assert pair["left_execution_trace"]["matches_public_and_independent_literal_pinned_order"]
    assert pair["right_execution_trace"]["matches_public_and_independent_literal_pinned_order"]
    assert len(pair["left_execution_trace"]["quintic_stages"]) == 5
    assert len(pair["right_execution_trace"]["quintic_stages"]) == 5
    assert exact_ratio == exact_gap / exact_distance
    assert measurement["runtime_gap_matches_exact_recorded_value_arithmetic"] == (
        runtime_gap == exact_gap
    )
    assert measurement["violates_incremental_monotonicity"] == (measurement["gap_decimal"] < 0)
    expected_kind = (
        "negative_pair_witness"
        if measurement["violates_incremental_monotonicity"]
        else "nonnegative_pair_evaluation"
    )
    assert payload["claim_scope"]["observed_result_kind"] == expected_kind


def test_bf16_witness_characterizes_only_observable_rounding() -> None:
    payload = canonical_bf16_witness()
    behavior = payload["observable_bf16_behavior"]
    normalization = behavior["input_casts_and_normalization"]
    probe = behavior["matmul_accumulation_probe"]

    for side in ("left", "right"):
        assert normalization[side]["after"]["dtype"] == "torch.bfloat16"
        assert normalization[side]["norm"]["dtype"] == "torch.bfloat16"
        assert normalization[side]["epsilon_bf16_cast_reference"]["dtype"] == "torch.bfloat16"
        assert normalization[side]["python_float_epsilon"]["decimal"] == "1e-07"
        assert isinstance(normalization[side]["epsilon_absorbed"], bool)

    assert probe["public_matmul_result"]["dtype"] == "torch.bfloat16"
    assert isinstance(probe["public_matmul_matches_exact_sum_then_bf16"], bool)
    assert isinstance(probe["public_matmul_matches_sequential_public_bf16_ops"], bool)
    assert "do not reveal" in probe["interpretation_limit"]


def test_bf16_witness_rejects_noncanonical_or_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        canonical_bf16_witness(eps=0)
    with pytest.raises(ValueError, match="fixed at five"):
        canonical_bf16_witness(steps=4)


def test_bf16_witness_cli_records_run_provenance_and_source_hashes() -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "scripts" / "record_bf16_witness.py")],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    snapshots = payload["experiment_provenance"]["source_snapshot"]

    assert payload["schema_version"] == BF16_WITNESS_SCHEMA_VERSION
    assert payload["software"]["pytorch"] == torch.__version__
    assert payload["hardware"]["machine_architecture"]
    assert payload["hardware"]["execution_device"]["device_type"] == "cpu"
    assert payload["git"]["sha"]
    assert payload["experiment_provenance"]["seed"] is None
    assert payload["experiment_provenance"]["randomness"].startswith("none;")
    assert (
        snapshots["scripts/record_bf16_witness.py"]
        == hashlib.sha256((root / "scripts" / "record_bf16_witness.py").read_bytes()).hexdigest()
    )
    assert (
        snapshots["src/passive_muon/bf16_witness.py"]
        == hashlib.sha256(
            (root / "src" / "passive_muon" / "bf16_witness.py").read_bytes()
        ).hexdigest()
    )
