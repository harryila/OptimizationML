from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from passive_muon.deployed import keller_jordan_map
from passive_muon.p21_shadow_trace import (
    CAPTURE_STEPS,
    EARLY_CAPTURE_STEPS,
    LATE_CAPTURE_STEPS,
    MIDDLE_CAPTURE_STEPS,
    P21_SHADOW_TRACE_SCHEMA_VERSION,
    ShadowTraceProtocolError,
    frozen_protocol_sha256,
    load_frozen_protocol,
    nearest_rank,
    observe_aspect_scaled_candidate,
    phase_for_step,
    summarize_shadow_trace,
    upstream_aspect_factor,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "training" / "run_p21_synthetic_shadow_trace.py"
SPEC = importlib.util.spec_from_file_location("p21_synthetic_shadow_trace", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("failed to load the P21 synthetic shadow-trace runner")
EXPERIMENT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EXPERIMENT
SPEC.loader.exec_module(EXPERIMENT)

run_synthetic_shadow_trace = EXPERIMENT.run_synthetic_shadow_trace


@pytest.fixture(scope="module")
def synthetic_payload() -> dict[str, object]:
    return run_synthetic_shadow_trace()


def _observe(signal: torch.Tensor, candidate: torch.Tensor, *, step: int = 0) -> dict[str, object]:
    return observe_aspect_scaled_candidate(
        signal,
        candidate,
        optimizer_step=step,
        tokens_seen=(step + 1) * 4_096,
        seed=20_260_907,
        parameter_name="synthetic.layers.0.projection.weight",
        layer=0,
        role="projection",
        learning_rate=0.02,
        beta=0.95,
        weight_decay=0.0,
    )


def test_protocol_freezes_schedule_scope_metrics_and_gate_values() -> None:
    protocol = load_frozen_protocol()
    assert protocol["schema_version"] == P21_SHADOW_TRACE_SCHEMA_VERSION
    schedule = protocol["capture_schedule"]
    assert schedule["total_optimizer_steps"] == 256
    assert schedule["early"] == list(EARLY_CAPTURE_STEPS)
    assert schedule["middle"] == list(MIDDLE_CAPTURE_STEPS)
    assert schedule["late"] == list(LATE_CAPTURE_STEPS)
    assert schedule["steps"] == list(CAPTURE_STEPS)
    assert len(CAPTURE_STEPS) == 24
    assert len(frozen_protocol_sha256()) == 64

    gates = protocol["frozen_gates"]
    assert gates["p16_p18_reused"]["informative_candidate_best_scalar_departure"] == (">= 1/100")
    assert gates["mild_intervention"]["maximum_overall_activation_rate"] == "1/4"
    assert gates["mild_intervention"]["relative_correction_p95_nearest_rank"] == "1/4"
    assert protocol["claim_scope"]["synthetic_runner"].startswith(
        "Deterministic CPU infrastructure"
    )
    assert "no CUDA or MPS" in protocol["external_execution_blockers"][3]


def test_phase_mapping_and_nearest_rank_semantics_are_exact() -> None:
    assert phase_for_step(0) == "early"
    assert phase_for_step(127) == "middle"
    assert phase_for_step(255) == "late"
    with pytest.raises(ShadowTraceProtocolError, match="not in the frozen"):
        phase_for_step(8)

    values = [5.0, 1.0, 4.0, 3.0, 2.0]
    assert nearest_rank(values, 5, 100) == 1.0
    assert nearest_rank(values, 50, 100) == 3.0
    assert nearest_rank(values, 95, 100) == 5.0
    with pytest.raises(ShadowTraceProtocolError, match="at least one"):
        nearest_rank([], 1, 2)
    with pytest.raises(ShadowTraceProtocolError, match="finite"):
        nearest_rank([1.0, float("nan")], 1, 2)


def test_observer_applies_aspect_before_transposed_shield_without_mutating_inputs() -> None:
    signal = torch.tensor([[1.2], [1.6]], dtype=torch.float32)
    raw_candidate = keller_jordan_map(signal).contiguous()
    signal_before = signal.clone()
    candidate_before = raw_candidate.clone()

    record = _observe(signal, raw_candidate)

    assert torch.equal(signal, signal_before)
    assert torch.equal(raw_candidate, candidate_before)
    assert record["parameter"]["original_shape"] == [2, 1]
    assert record["parameter"]["shield_shape"] == [1, 2]
    assert record["parameter"]["transposed_for_shield"] is True
    assert record["optimizer"]["aspect_applied_before_shield"] is True
    assert record["optimizer"]["upstream_aspect_factor"] == pytest.approx(2.0**0.5)
    expected = raw_candidate.clone()
    expected.mul_(2.0**0.5)
    assert record["metrics"]["aspect_candidate_norm"] == pytest.approx(
        float(torch.linalg.vector_norm(expected.double()))
    )
    assert record["storage"]["aspect_candidate_dtype"] == "torch.bfloat16"
    assert record["storage"]["output_dtype"] == "torch.float32"
    assert record["shield"]["sector_certified_by_successful_p20_call"] is True


def test_upstream_aspect_factor_and_unsupported_shape_fail_closed() -> None:
    assert upstream_aspect_factor((768, 3_072)) == 1.0
    assert upstream_aspect_factor((3_072, 768)) == 2.0
    signal = torch.ones((3, 4), dtype=torch.float32)
    candidate = torch.ones((3, 4), dtype=torch.bfloat16)
    with pytest.raises(ShadowTraceProtocolError, match="neither orientation"):
        _observe(signal, candidate)


def test_zero_denominators_are_classified_and_not_replaced_by_numeric_zero() -> None:
    signal = torch.zeros((2, 2), dtype=torch.float32)
    candidate = torch.zeros((2, 2), dtype=torch.bfloat16)
    record = _observe(signal, candidate)
    metrics = record["metrics"]
    assert metrics["relative_correction"] is None
    assert metrics["relative_correction_status"] == "both_zero"
    assert metrics["candidate_output_cosine"] is None
    assert metrics["candidate_output_cosine_status"] == "both_zero"
    assert metrics["output_to_signal_amplitude"] is None
    assert metrics["output_to_signal_status"] == "both_zero"
    assert record["per_observation_gates"]["frozen_output_to_signal"] is False

    summary = summarize_shadow_trace(
        [record],
        evidence_kind="synthetic_cpu_diagnostic",
        provenance={},
        intended_parameter_numel={"synthetic.layers.0.projection.weight": 4},
    )
    assert summary["diagnostic_quantiles"]["relative_correction"] is None
    assert summary["diagnostic_quantiles"]["candidate_output_cosine"] is None
    assert summary["diagnostic_quantiles"]["output_to_candidate_amplitude"] is None
    assert summary["diagnostic_sample_counts"]["relative_correction"] == 0
    assert summary["all_predeclared_checks_pass"] is False


def test_unrepresentable_all_subnormal_signal_records_zero_disturbance_policy() -> None:
    least_subnormal = torch.tensor([1], dtype=torch.int32).view(torch.float32)[0]
    signal = torch.zeros((2, 2), dtype=torch.float32)
    signal[0, 0] = least_subnormal
    candidate = torch.zeros((2, 2), dtype=torch.bfloat16)
    record = _observe(signal.contiguous(), candidate)
    assert record["shield"]["action"] == "dead_zone_zero"
    assert record["shield"]["dead_zone_zero"] is True
    assert record["shield"]["sector_certified_by_successful_p20_call"] is False
    assert record["storage"]["output_dtype"] == "torch.float32"
    assert record["metrics"]["output_norm"] == 0.0


def test_nonfinite_signal_is_a_hard_trace_failure() -> None:
    signal = torch.eye(2, dtype=torch.float32)
    signal[0, 0] = float("nan")
    candidate = torch.eye(2, dtype=torch.bfloat16)
    with pytest.raises(ShadowTraceProtocolError, match="hard trace-gate"):
        _observe(signal, candidate)


def test_nonfinite_candidate_falls_back_safely_but_is_visible_to_trace_gate() -> None:
    signal = torch.eye(2, dtype=torch.float32)
    candidate = torch.eye(2, dtype=torch.bfloat16)
    candidate[0, 0] = float("nan")
    record = _observe(signal, candidate)
    assert record["storage"]["aspect_candidate_finite"] is False
    assert record["shield"]["action"] == "half_fallback"
    assert record["shield"]["sector_certified_by_successful_p20_call"] is True
    assert record["metrics"]["candidate_output_cosine"] is None
    assert record["metrics"]["candidate_output_cosine_status"] == "nonfinite_candidate"


def test_synthetic_runner_is_complete_deterministic_and_explicitly_not_real_gradient(
    synthetic_payload: dict[str, object],
) -> None:
    assert synthetic_payload["schema_version"] == P21_SHADOW_TRACE_SCHEMA_VERSION
    assert synthetic_payload["evidence_kind"] == "synthetic_cpu_diagnostic"
    assert synthetic_payload["real_gradient_evidence"] is False
    assert "no model, dataset, backward pass" in synthetic_payload["claim_scope"]
    assert synthetic_payload["real_trace_status"]["blocked"] is True
    assert synthetic_payload["synthetic_fixture"]["observation_count"] == 144
    assert synthetic_payload["coverage"]["schedule_complete"] is True
    assert synthetic_payload["coverage"]["observed_steps"] == list(CAPTURE_STEPS)
    assert synthetic_payload["coverage"]["inventory_complete_every_capture"] is True
    assert synthetic_payload["coverage"]["parameter_count_coverage"] == 1.0
    assert synthetic_payload["coverage"]["parameter_numel_coverage"] == 1.0
    assert synthetic_payload["shield_actions"]["active"] == 18
    assert synthetic_payload["shield_actions"]["activation_rate"] == pytest.approx(0.125)
    assert synthetic_payload["shield_actions"]["dead_zone_zero"] == 0
    assert synthetic_payload["shield_actions"]["nonfinite_candidate_events"] == 0
    assert all(synthetic_payload["hard_gate_checks"].values())
    assert all(synthetic_payload["frozen_fidelity_gate_checks"].values())
    assert all(synthetic_payload["mild_intervention_gate_checks"].values())
    assert synthetic_payload["all_predeclared_checks_pass"] is True

    for cell in synthetic_payload["phase_role_cells"].values():
        assert cell["count"] == 24
        assert cell["minimum_count_reached"] is True
        assert cell["activation_rate"] == pytest.approx(0.125)
        assert cell["passes_if_evaluated"] is True


def test_summary_fails_coverage_when_an_intended_parameter_is_omitted() -> None:
    observations = []
    for step in CAPTURE_STEPS:
        signal = torch.tensor([[0.6, 0.0], [0.0, 0.8]], dtype=torch.float32)
        observations.append(_observe(signal, keller_jordan_map(signal), step=step))
    summary = summarize_shadow_trace(
        observations,
        evidence_kind="real_gradient_shadow_trace",
        provenance={"fixture": "coverage-negative-control"},
        intended_parameter_numel={
            "synthetic.layers.0.projection.weight": 4,
            "missing.parameter.weight": 4,
        },
    )
    assert summary["coverage"]["parameter_count_coverage"] == 0.5
    assert summary["coverage"]["parameter_numel_coverage"] == 0.5
    assert summary["hard_gate_checks"]["intended_muon_inventory_complete_every_capture"] is False
    assert summary["all_predeclared_checks_pass"] is False


def test_synthetic_runner_repeats_decisions_for_same_seed(
    synthetic_payload: dict[str, object],
) -> None:
    repeated = run_synthetic_shadow_trace()
    for key in (
        "coverage",
        "shield_actions",
        "diagnostic_quantiles",
        "hard_gate_checks",
        "frozen_fidelity_gate_checks",
        "mild_intervention_gate_checks",
        "all_predeclared_checks_pass",
    ):
        assert repeated[key] == synthetic_payload[key]


def test_synthetic_cli_writes_machine_readable_diagnostic(tmp_path: Path) -> None:
    output = tmp_path / "p21-synthetic.json"
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(output)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    written = json.loads(output.read_text(encoding="utf-8"))
    printed = json.loads(completed.stdout)
    assert written == printed
    assert written["real_gradient_evidence"] is False
    assert written["real_trace_status"]["blocked"] is True
    assert written["all_predeclared_checks_pass"] is True
