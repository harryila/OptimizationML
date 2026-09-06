from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "training" / "p22_nanogpt_shadow_trace.py"
SPEC = importlib.util.spec_from_file_location("p22_nanogpt_shadow_trace", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("failed to load the P22 real-gradient shadow-trace harness")
P22 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = P22
SPEC.loader.exec_module(P22)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _run_manifest(mode: str) -> dict[str, object]:
    steps = []
    for step in range(P22.EXPECTED_OPTIMIZER_STEPS):
        checkpoint = step in P22.STATE_CHECKPOINT_STEPS
        record: dict[str, object] = {
            "step": step,
            "tokens_seen": (step + 1) * 128,
            "data_token_offset": step * 128,
            "batch_sha256": _digest(f"batch:{step}"),
            "loss_tensor_sha256": _digest(f"loss:{step}"),
            "model_state_sha256": _digest(f"model:{step}") if checkpoint else None,
            "optimizer_state_sha256": _digest(f"optimizer:{step}") if checkpoint else None,
            "state_checkpoint_present": checkpoint,
            "rng_before_step_sha256": _digest(f"rng-before:{step}"),
            "rng_after_step_sha256": _digest(f"rng-after:{step}"),
        }
        if mode == "trace_on" and step in P22.EXPECTED_CAPTURE_STEPS:
            record["observer_rng_unchanged"] = True
            record["capture"] = {
                "parameter_count": 48,
                "actual_pre_aspect_candidate": True,
                "actual_post_aspect_candidate": True,
                "p20_shadow_only": True,
                "record_sha256": _digest(f"capture:{step}"),
            }
        steps.append(record)
    return {
        "schema_version": P22.P22_RUN_MANIFEST_SCHEMA,
        "evidence_kind": "real_gradient_shadow_trace",
        "trace_mode": mode,
        "protocol_sha256": P22.protocol_sha256(),
        "run_identity_sha256": _digest("frozen-run-identity"),
        "initial_state_sha256": _digest("initial-state"),
        "accelerator_backend": "mps",
        "actual_accelerator_candidates": True,
        "shadow_update_applied": False,
        "capture_steps": list(P22.EXPECTED_CAPTURE_STEPS) if mode == "trace_on" else [],
        "steps": steps,
        "final_state": {
            "model_state_sha256": _digest("final-model"),
            "optimizer_state_sha256": _digest("final-optimizer"),
            "rng_state_sha256": _digest("final-rng"),
        },
    }


def test_protocol_pins_sources_model_data_optimizer_and_feasible_schedule() -> None:
    protocol = P22.load_protocol()
    assert protocol["schema_version"] == P22.P22_PROTOCOL_SCHEMA
    assert protocol["status"] == "frozen_protocol_no_run"
    assert protocol["claim_boundary"]["real_run_present"] is False
    assert protocol["upstream"]["trainer"]["revision"] == P22.NANOGPT_REVISION
    assert protocol["upstream"]["muon"]["revision"] == P22.MUON_REVISION
    assert protocol["data"]["dataset_revision"] == ("9bb295ddab0e05d785b879661af7260fed5140fc")
    assert protocol["data"]["tokenizer_package"] == "tiktoken==0.14.0"
    assert protocol["model"]["vocab_size"] == 50_257
    assert protocol["model"]["block_size"] == 1_024
    assert protocol["run"]["sequence_length"] == 128
    assert protocol["run"]["tokens_per_optimizer_step"] == 128
    assert protocol["run"]["total_training_tokens"] == 32_768
    assert protocol["run"]["capture_steps"] == list(P22.EXPECTED_CAPTURE_STEPS)
    assert protocol["optimizer"]["muon_learning_rate"] == "1/120"
    assert protocol["optimizer"]["muon_weight_decay"] == 0
    assert protocol["optimizer"]["normalization_epsilon"] == "1e-7"
    assert protocol["optimizer"]["jordan_stages"] == 5
    assert len(P22.protocol_sha256()) == 64


def test_gpt2_inventory_includes_every_layer_and_p22_qkv_extension() -> None:
    inventory = P22.expected_gpt2_small_muon_inventory()
    assert len(inventory) == 48
    assert len({item["name"] for item in inventory}) == 48
    qkv = [item for item in inventory if ".attn.c_attn.weight" in item["name"]]
    assert len(qkv) == 12
    assert all(item["stored_shape"] == [2_304, 768] for item in qkv)
    assert all(item["shield_shape"] == [768, 2_304] for item in qkv)
    assert all(item["transposed_for_shield"] is True for item in qkv)
    coverage = P22.inventory_coverage()
    assert coverage["complete"] is True
    assert coverage["covered_parameter_count"] == 48
    assert coverage["numel_fraction"] == (
        f"{coverage['intended_numel']}/{coverage['intended_numel']}"
    )


def test_unmaterialized_fineweb_template_fails_closed() -> None:
    result = P22.validate_fineweb_manifest(P22.data_template_path())
    assert result["ready"] is False
    assert "FineWeb manifest is not marked materialized" in result["blockers"]
    assert any("path is not materialized" in blocker for blocker in result["blockers"])


def test_fineweb_manifest_rejects_truncated_server_rows(tmp_path: Path) -> None:
    source = tmp_path / "rows.json"
    source.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "row_idx": 0,
                        "row": {"text": "complete-looking prefix"},
                        "truncated_cells": ["text"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    template = json.loads(P22.data_template_path().read_text(encoding="utf-8"))
    template["status"] = "materialized"
    template["dataset"]["source_shards"] = [
        {
            "source_kind": "captured datasets-server response",
            "path": source.name,
            "byte_count": source.stat().st_size,
            "sha256": P22.sha256_file(source),
        }
    ]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(template), encoding="utf-8")
    result = P22.validate_fineweb_manifest(manifest)
    assert result["ready"] is False
    assert any("truncated or malformed" in blocker for blocker in result["blockers"])


def test_state_hash_is_exact_dtype_shape_and_value_sensitive() -> None:
    state = {
        "weight": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        "step": 3,
        "nested": ("x", np.array([4, 5], dtype=np.uint32)),
    }
    repeated = {
        "nested": ("x", np.array([4, 5], dtype=np.uint32)),
        "step": 3,
        "weight": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
    }
    assert P22.canonical_state_sha256(state) == P22.canonical_state_sha256(repeated)
    changed_dtype = dict(state)
    changed_dtype["weight"] = state["weight"].to(torch.float64)
    assert P22.canonical_state_sha256(state) != P22.canonical_state_sha256(changed_dtype)
    changed_value = dict(state)
    changed_value["weight"] = torch.tensor([[1.0, 2.5]], dtype=torch.float32)
    assert P22.canonical_state_sha256(state) != P22.canonical_state_sha256(changed_value)


def test_rng_neutral_observer_accepts_read_only_and_restores_mutation() -> None:
    random.seed(1337)
    np.random.seed(1337)
    torch.manual_seed(1337)
    result, digest = P22.run_rng_neutral_observer(lambda: "observed")
    assert result == "observed"
    assert digest == P22.canonical_state_sha256(P22.capture_rng_state())

    before = P22.capture_rng_state()

    def mutate_rng() -> None:
        random.random()
        np.random.random()
        torch.rand(1)

    with pytest.raises(P22.P22ProtocolError, match="changed training RNG"):
        P22.run_rng_neutral_observer(mutate_rng)
    assert P22.canonical_state_sha256(P22.capture_rng_state()) == P22.canonical_state_sha256(before)


def test_missing_real_prerequisites_return_blocked_report(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        P22,
        "validate_accelerator_runtime",
        lambda backend: {
            "ready": True,
            "blockers": [],
            "details": {"selected_backend": backend, "evidence_scope": "test-only"},
        },
    )
    report = P22.build_preflight_report(
        nanogpt_root=None,
        muon_source=None,
        fineweb_manifest=None,
        instrumentation_patch=None,
        accelerator_backend="mps",
    )
    assert report["status"] == "blocked"
    assert report["real_run_executed"] is False
    assert report["shape_inventory"]["complete"] is True
    assert any("NanoGPT checkout was not supplied" in blocker for blocker in report["blockers"])
    assert "not a gradient trace" in report["claim_boundary"]


def test_noninterference_comparison_is_bitwise_and_fail_closed(tmp_path: Path) -> None:
    off_path = tmp_path / "trace-off.json"
    on_path = tmp_path / "trace-on.json"
    off_path.write_text(json.dumps(_run_manifest("trace_off")), encoding="utf-8")
    on_payload = _run_manifest("trace_on")
    on_path.write_text(json.dumps(on_payload), encoding="utf-8")

    passing = P22.compare_noninterference_manifests(off_path, on_path)
    assert passing["passes"] is True
    assert passing["comparison"] == "bitwise_exact_no_tolerances"
    assert passing["checked_step_count"] == 256
    assert passing["mismatch_count"] == 0

    on_payload["steps"][127]["loss_tensor_sha256"] = _digest("one-bit-different-in-principle")
    on_path.write_text(json.dumps(on_payload), encoding="utf-8")
    failing = P22.compare_noninterference_manifests(off_path, on_path)
    assert failing["passes"] is False
    assert failing["mismatch_count"] == 1
    assert failing["mismatches"] == [{"scope": "step", "step": 127, "field": "loss_tensor_sha256"}]


def test_trace_off_repeatability_is_a_separate_exact_gate(tmp_path: Path) -> None:
    off_a_path = tmp_path / "trace-off-a.json"
    off_b_path = tmp_path / "trace-off-b.json"
    off_a_path.write_text(json.dumps(_run_manifest("trace_off")), encoding="utf-8")
    off_b = _run_manifest("trace_off")
    off_b_path.write_text(json.dumps(off_b), encoding="utf-8")

    passing = P22.compare_repeatability_manifests(off_a_path, off_b_path)
    assert passing["passes"] is True
    assert passing["mismatch_count"] == 0
    assert passing["state_checkpoint_steps"] == sorted(P22.STATE_CHECKPOINT_STEPS)

    off_b["steps"][64]["rng_after_step_sha256"] = _digest("different-baseline-rng")
    off_b_path.write_text(json.dumps(off_b), encoding="utf-8")
    failing = P22.compare_repeatability_manifests(off_a_path, off_b_path)
    assert failing["passes"] is False
    assert failing["mismatches"] == [
        {"scope": "step", "step": 64, "field": "rng_after_step_sha256"}
    ]


def test_noninterference_rejects_synthetic_or_incomplete_capture(tmp_path: Path) -> None:
    off_path = tmp_path / "trace-off.json"
    on_path = tmp_path / "trace-on.json"
    off_path.write_text(json.dumps(_run_manifest("trace_off")), encoding="utf-8")
    on_payload = _run_manifest("trace_on")
    on_payload["steps"][0]["capture"]["actual_post_aspect_candidate"] = False
    on_path.write_text(json.dumps(on_payload), encoding="utf-8")
    with pytest.raises(P22.P22ProtocolError, match="capture contract mismatch"):
        P22.compare_noninterference_manifests(off_path, on_path)

    synthetic = _run_manifest("trace_on")
    synthetic["evidence_kind"] = "synthetic_cpu_diagnostic"
    on_path.write_text(json.dumps(synthetic), encoding="utf-8")
    with pytest.raises(P22.P22ProtocolError, match="not labeled real-gradient"):
        P22.compare_noninterference_manifests(off_path, on_path)


def test_run_manifest_rejects_wrong_schedule_and_missing_final_digest(tmp_path: Path) -> None:
    off_path = tmp_path / "trace-off.json"
    on_path = tmp_path / "trace-on.json"
    off_path.write_text(json.dumps(_run_manifest("trace_off")), encoding="utf-8")
    on_payload = _run_manifest("trace_on")
    on_payload["steps"][9]["data_token_offset"] = 0
    on_path.write_text(json.dumps(on_payload), encoding="utf-8")
    with pytest.raises(P22.P22ProtocolError, match="data offset changed"):
        P22.compare_noninterference_manifests(off_path, on_path)

    on_payload = _run_manifest("trace_on")
    del on_payload["final_state"]["rng_state_sha256"]
    on_path.write_text(json.dumps(on_payload), encoding="utf-8")
    with pytest.raises(P22.P22ProtocolError, match="invalid final rng_state_sha256"):
        P22.compare_noninterference_manifests(off_path, on_path)
