from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

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

RUNNER_SCRIPT = ROOT / "experiments" / "training" / "run_p22_real_gradient_shadow_trace.py"
RUNNER_SPEC = importlib.util.spec_from_file_location(
    "run_p22_real_gradient_shadow_trace_test", RUNNER_SCRIPT
)
if RUNNER_SPEC is None or RUNNER_SPEC.loader is None:
    raise RuntimeError("failed to load the P22 real-gradient runner")
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
sys.modules[RUNNER_SPEC.name] = RUNNER
RUNNER_SPEC.loader.exec_module(RUNNER)


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
                "actual_post_aspect_cuda_candidate": False,
                "p20_shadow_only": True,
                "record_sha256": _digest(f"capture:{step}"),
            }
            record["observation_count"] = 48
        else:
            record["observer_rng_unchanged"] = None
            record["capture"] = None
            record["observation_count"] = 0
        steps.append(record)
    return {
        "schema_version": P22.P22_RUN_MANIFEST_SCHEMA,
        "evidence_kind": "real_gradient_shadow_trace",
        "trace_mode": mode,
        "protocol_sha256": P22.protocol_sha256(),
        "run_identity_sha256": _digest("frozen-run-identity"),
        "initial_state_sha256": _digest("initial-state"),
        "accelerator_backend": "mps",
        "candidate_execution": {
            "backend": "mps",
            "actual_accelerator_candidate_computed_and_applied": True,
            "shield_shadow_only": True,
        },
        "candidate_observation": (
            {
                "status": "observed_actual_post_aspect_mps",
                "capture_step_count": len(P22.EXPECTED_CAPTURE_STEPS),
                "observation_count": P22.EXPECTED_CANDIDATE_OBSERVATIONS,
            }
            if mode == "trace_on"
            else {
                "status": "not_observed",
                "capture_step_count": 0,
                "observation_count": 0,
            }
        ),
        "shadow_update_applied": False,
        "capture_steps": list(P22.EXPECTED_CAPTURE_STEPS) if mode == "trace_on" else [],
        "steps": steps,
        "model_optimizer_binding": {
            "verified": True,
            "verification": "exact in-process Python object identity after accelerator move",
            "selected_device_type": "mps",
            "stored_parameter_devices": ["mps:0"],
            "stored_parameter_dtypes": ["torch.float32"],
            "model_unique_parameter_count": 75,
            "optimizer_parameter_count": 75,
            "muon_parameter_count": 48,
            "auxiliary_parameter_count": 27,
            "complete_unique_identity_coverage": True,
            "no_duplicate_optimizer_parameters": True,
            "muon_metadata_exact": True,
            "tied_parameter_aliases": [["lm_head.weight", "transformer.wte.weight"]],
            "model_config": {
                "block_size": 1024,
                "vocab_size": 50_257,
                "n_layer": 12,
                "n_head": 12,
                "n_embd": 768,
                "dropout": 0.0,
                "bias": False,
            },
            "optimizer_groups": [
                {
                    "group_index": 0,
                    "use_muon": False,
                    "hyperparameters": {
                        "learning_rate": 3.0 / 5_000.0,
                        "betas": [0.9, 0.95],
                        "epsilon": 1.0e-10,
                        "weight_decay": 0.0,
                    },
                    "parameter_names": [f"aux.{index}" for index in range(27)],
                },
                {
                    "group_index": 1,
                    "use_muon": True,
                    "hyperparameters": {
                        "learning_rate": 1.0 / 120.0,
                        "momentum": 19.0 / 20.0,
                        "weight_decay": 0.0,
                    },
                    "parameter_names": [f"muon.{index}" for index in range(48)],
                },
            ],
        },
        "raw_trace": (
            {
                "path": "p22-raw-trace.json",
                "sha256": _digest("raw-trace"),
                "observation_count": P22.EXPECTED_CANDIDATE_OBSERVATIONS,
            }
            if mode == "trace_on"
            else None
        ),
        "final_state": {
            "model_state_sha256": _digest("final-model"),
            "optimizer_state_sha256": _digest("final-optimizer"),
            "rng_state_sha256": _digest("final-rng"),
        },
    }


class _TinyTiedModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = torch.nn.Module()
        self.transformer.wte = torch.nn.Embedding(2, 2)
        self.lm_head = torch.nn.Linear(2, 2, bias=False)
        self.lm_head.weight = self.transformer.wte.weight
        self.hidden = torch.nn.Parameter(torch.ones((2, 2), dtype=torch.float32))
        self.scale = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.config = SimpleNamespace(
            block_size=RUNNER.MODEL_BLOCK_SIZE,
            vocab_size=RUNNER.VOCAB_SIZE,
            n_layer=RUNNER.MODEL_N_LAYER,
            n_head=RUNNER.MODEL_N_HEAD,
            n_embd=RUNNER.MODEL_N_EMBD,
            dropout=RUNNER.MODEL_DROPOUT,
            bias=RUNNER.MODEL_BIAS,
        )


def _tiny_binding_fixture() -> tuple[
    _TinyTiedModel,
    SimpleNamespace,
    dict[int, object],
]:
    model = _TinyTiedModel()
    optimizer = SimpleNamespace(
        param_groups=[
            {
                "params": [model.transformer.wte.weight, model.scale],
                "use_muon": False,
                "lr": RUNNER.AUXILIARY_LEARNING_RATE,
                "betas": RUNNER.AUXILIARY_BETAS,
                "eps": RUNNER.AUXILIARY_EPSILON,
                "weight_decay": RUNNER.AUXILIARY_WEIGHT_DECAY,
            },
            {
                "params": [model.hidden],
                "use_muon": True,
                "lr": RUNNER.MUON_LEARNING_RATE,
                "momentum": RUNNER.MUON_MOMENTUM,
                "weight_decay": RUNNER.MUON_WEIGHT_DECAY,
            },
        ]
    )
    metadata = {
        id(model.hidden): RUNNER.MuonParameterMetadata(
            name="hidden",
            layer=0,
            role="test_hidden",
        )
    }
    return model, optimizer, metadata


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


def test_model_optimizer_binding_audit_uses_exact_post_move_objects() -> None:
    model, optimizer, metadata = _tiny_binding_fixture()
    audit = RUNNER._audit_model_optimizer_binding(
        model=model,
        optimizer=optimizer,
        parameter_metadata=metadata,
        backend="cpu",
        expected_muon_parameter_count=1,
    )

    assert audit["verified"] is True
    assert audit["complete_unique_identity_coverage"] is True
    assert audit["no_duplicate_optimizer_parameters"] is True
    assert audit["model_unique_parameter_count"] == 3
    assert audit["optimizer_parameter_count"] == 3
    assert audit["muon_parameter_count"] == 1
    assert audit["auxiliary_parameter_count"] == 2
    assert audit["stored_parameter_devices"] == ["cpu"]
    assert audit["stored_parameter_dtypes"] == ["torch.float32"]
    assert audit["tied_parameter_aliases"] == [["lm_head.weight", "transformer.wte.weight"]]
    inventory = audit["parameter_inventory"]
    assert [record["name"] for record in inventory] == [
        "hidden",
        "scale",
        "transformer.wte.weight",
    ]
    hidden = inventory[0]
    assert hidden["shape"] == [2, 2]
    assert hidden["numel"] == 4
    assert hidden["optimizer_group_index"] == 1
    assert hidden["use_muon"] is True
    assert hidden["device"] == "cpu"
    assert hidden["dtype"] == "torch.float32"


def test_model_optimizer_binding_audit_rejects_stale_parameter_objects() -> None:
    model, optimizer, metadata = _tiny_binding_fixture()
    optimizer.param_groups[1]["params"] = [torch.nn.Parameter(model.hidden.detach().clone())]

    with pytest.raises(RUNNER.P22RunError, match="exactly cover"):
        RUNNER._audit_model_optimizer_binding(
            model=model,
            optimizer=optimizer,
            parameter_metadata=metadata,
            backend="cpu",
            expected_muon_parameter_count=1,
        )


def test_model_optimizer_binding_audit_rejects_duplicate_parameter_objects() -> None:
    model, optimizer, metadata = _tiny_binding_fixture()
    optimizer.param_groups[0]["params"].append(model.scale)

    with pytest.raises(RUNNER.P22RunError, match="more than once"):
        RUNNER._audit_model_optimizer_binding(
            model=model,
            optimizer=optimizer,
            parameter_metadata=metadata,
            backend="cpu",
            expected_muon_parameter_count=1,
        )


def test_model_optimizer_binding_audit_rejects_wrong_device() -> None:
    model, optimizer, metadata = _tiny_binding_fixture()

    with pytest.raises(RUNNER.P22RunError, match="selected accelerator"):
        RUNNER._audit_model_optimizer_binding(
            model=model,
            optimizer=optimizer,
            parameter_metadata=metadata,
            backend="cuda",
            expected_muon_parameter_count=1,
        )


def test_model_optimizer_binding_audit_rejects_model_config_drift() -> None:
    model, optimizer, metadata = _tiny_binding_fixture()
    model.config.n_head = 8

    with pytest.raises(RUNNER.P22RunError, match="model configuration"):
        RUNNER._audit_model_optimizer_binding(
            model=model,
            optimizer=optimizer,
            parameter_metadata=metadata,
            backend="cpu",
            expected_muon_parameter_count=1,
        )


def test_model_optimizer_binding_audit_rejects_optimizer_hyperparameter_drift() -> None:
    model, optimizer, metadata = _tiny_binding_fixture()
    optimizer.param_groups[1]["momentum"] = 0.9

    with pytest.raises(RUNNER.P22RunError, match="optimizer hyperparameters"):
        RUNNER._audit_model_optimizer_binding(
            model=model,
            optimizer=optimizer,
            parameter_metadata=metadata,
            backend="cpu",
            expected_muon_parameter_count=1,
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


def test_run_manifest_distinguishes_candidate_execution_from_observation(
    tmp_path: Path,
) -> None:
    off_path = tmp_path / "trace-off.json"
    on_path = tmp_path / "trace-on.json"
    off = _run_manifest("trace_off")
    on = _run_manifest("trace_on")
    off_path.write_text(json.dumps(off), encoding="utf-8")
    on_path.write_text(json.dumps(on), encoding="utf-8")

    loaded_off = P22._load_run_manifest(off_path, "trace_off")
    loaded_on = P22._load_run_manifest(on_path, "trace_on")
    assert loaded_off["candidate_execution"] == loaded_on["candidate_execution"]
    assert loaded_off["candidate_observation"] == {
        "status": "not_observed",
        "capture_step_count": 0,
        "observation_count": 0,
    }
    assert loaded_on["candidate_observation"] == {
        "status": "observed_actual_post_aspect_mps",
        "capture_step_count": 24,
        "observation_count": 1_152,
    }
    assert sum(record["observation_count"] for record in loaded_off["steps"]) == 0
    assert sum(record["observation_count"] for record in loaded_on["steps"]) == 1_152


def test_run_manifest_rejects_inconsistent_candidate_observation_counts(tmp_path: Path) -> None:
    path = tmp_path / "trace.json"
    off = _run_manifest("trace_off")
    off["steps"][0]["observation_count"] = 1
    path.write_text(json.dumps(off), encoding="utf-8")
    with pytest.raises(P22.P22ProtocolError, match="has a candidate observation"):
        P22._load_run_manifest(path, "trace_off")

    on = _run_manifest("trace_on")
    on["candidate_observation"]["observation_count"] = 1_151
    path.write_text(json.dumps(on), encoding="utf-8")
    with pytest.raises(P22.P22ProtocolError, match="candidate-observation contract"):
        P22._load_run_manifest(path, "trace_on")


def test_run_manifest_requires_verified_model_optimizer_binding(tmp_path: Path) -> None:
    path = tmp_path / "trace.json"
    manifest = _run_manifest("trace_off")
    manifest["model_optimizer_binding"]["complete_unique_identity_coverage"] = False
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(P22.P22ProtocolError, match="complete_unique_identity_coverage"):
        P22._load_run_manifest(path, "trace_off")


def test_historical_v1_manifest_remains_replayable(tmp_path: Path) -> None:
    path = tmp_path / "legacy-trace-off.json"
    legacy = _run_manifest("trace_off")
    legacy["schema_version"] = P22.P22_LEGACY_RUN_MANIFEST_SCHEMA
    legacy["actual_accelerator_candidates"] = True
    del legacy["candidate_execution"]
    del legacy["candidate_observation"]
    path.write_text(json.dumps(legacy), encoding="utf-8")

    loaded = P22._load_run_manifest(path, "trace_off")
    assert loaded["schema_version"] == P22.P22_LEGACY_RUN_MANIFEST_SCHEMA


def test_trace_off_runner_rejects_raw_candidate_output_before_preflight(tmp_path: Path) -> None:
    placeholder = tmp_path / "placeholder"
    with pytest.raises(RUNNER.P22RunError, match="trace_off forbids"):
        RUNNER.run_trace(
            trace_mode="trace_off",
            nanogpt_root=placeholder,
            muon_source=placeholder,
            data_manifest=placeholder,
            instrumentation_patch=placeholder,
            backend="mps",
            output=tmp_path / "off.json",
            raw_trace_output=tmp_path / "raw.json",
        )


def test_runner_lifecycle_callback_brackets_state_collection_and_writes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []

    class _Model:
        def train(self) -> None:
            events.append("model.train")

    def build_model(*args: object, **kwargs: object) -> tuple[object, ...]:
        del args, kwargs
        events.append("model_optimizer_observer")
        return (
            _Model(),
            object(),
            object(),
            {},
            {
                "verified": True,
                "parameter_inventory": [],
            },
        )

    def state_hashes(*args: object, **kwargs: object) -> dict[str, str]:
        del args, kwargs
        events.append("state_hashes")
        return {
            "model_state_sha256": _digest("model"),
            "optimizer_state_sha256": _digest("optimizer"),
        }

    monkeypatch.setattr(RUNNER, "EXPECTED_OPTIMIZER_STEPS", 0)
    monkeypatch.setattr(RUNNER, "build_preflight_report", lambda **kwargs: {"status": "ready"})
    monkeypatch.setattr(RUNNER, "_set_determinism", lambda backend: None)
    monkeypatch.setattr(
        RUNNER,
        "_load_train_tokens",
        lambda path: (object(), {"manifest_sha256": _digest("data")}),
    )
    monkeypatch.setattr(RUNNER, "_model_and_optimizer", build_model)
    monkeypatch.setattr(RUNNER, "_state_hashes", state_hashes)
    monkeypatch.setattr(RUNNER, "capture_rng_state", lambda backend: {"backend": backend})
    monkeypatch.setattr(RUNNER, "canonical_state_sha256", lambda value: _digest(repr(value)))
    monkeypatch.setattr(RUNNER, "_source_snapshot", lambda **kwargs: {})
    monkeypatch.setattr(RUNNER, "_run_identity", lambda *args: _digest("run"))
    monkeypatch.setattr(RUNNER, "_runtime_provenance", lambda backend: {"backend": backend})
    monkeypatch.setattr(
        RUNNER,
        "write_json_atomic",
        lambda path, payload: events.append("write"),
    )

    phases: list[str] = []

    def lifecycle(phase: str) -> None:
        phases.append(phase)
        events.append(phase)

    RUNNER.run_trace(
        trace_mode="trace_off",
        nanogpt_root=tmp_path,
        muon_source=tmp_path,
        data_manifest=tmp_path,
        instrumentation_patch=RUNNER.SCRIPT_DIR / "p22_observed_muon.py",
        backend="mps",
        output=tmp_path / "off.json",
        raw_trace_output=None,
        _lifecycle_callback=lifecycle,
    )

    assert phases == ["initialized", "completed"]
    assert events == [
        "model_optimizer_observer",
        "initialized",
        "model.train",
        "state_hashes",
        "state_hashes",
        "completed",
        "write",
    ]


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
