from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/training/run_p23_deterministic_cuda_shadow_trace.py"
SPEC = importlib.util.spec_from_file_location("run_p23_cuda_acquisition_test", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("failed to load the P23 acquisition runner")
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _minimal_manifest(role: str, label: str = "run") -> dict[str, object]:
    empty_snapshot = {
        "schema_version": RUNNER.P23.P23_LOADED_FILE_SNAPSHOT_SCHEMA,
        "module_count": 0,
        "mapped_file_count": 0,
        "modules": {},
        "mapped_files": {},
        "content_sha256": RUNNER.P23.canonical_json_sha256({"modules": {}, "mapped_files": {}}),
    }
    return {
        "run_identity_sha256": _digest(label),
        "execution_identity": {
            "source_modules": {},
            "loaded_file_closure": RUNNER.P23.build_loaded_file_closure(
                empty_snapshot,
                empty_snapshot,
            ),
        },
        "acquisition_role": role,
        "process_instance": {
            "schema_version": RUNNER.P23_PROCESS_INSTANCE_SCHEMA,
            "nonce": _digest(f"nonce:{label}"),
            "pid": 1,
            "created_time_ns": 1,
        },
    }


def _empty_loaded_file_verification() -> dict[str, object]:
    return {
        "passes": True,
        "comparison": "initialized_membership_recollected_all_retained_bytes_rehashed",
        "initialized_module_count": 0,
        "completed_module_count": 0,
        "initialized_mapped_file_count": 0,
        "completed_mapped_file_count": 0,
        "unique_file_count": 0,
    }


def _passing_report(
    off_a_path: Path,
    off_b_path: Path,
    off_a: dict[str, object],
    off_b: dict[str, object],
) -> dict[str, object]:
    return {
        "schema_version": RUNNER.P23.P23_REPEATABILITY_SCHEMA,
        "passes": True,
        "comparison": "bitwise_exact_no_tolerances",
        "mismatch_count": 0,
        "mismatches": [],
        "binding_schema": RUNNER.P23_GATE_REPORT_BINDING_SCHEMA,
        "inputs": {
            "trace_off_a": RUNNER._artifact_record(off_a_path, off_a),
            "trace_off_b": RUNNER._artifact_record(off_b_path, off_b),
        },
        "identity_verification": {
            "trace_off_a": {
                "passes": True,
                "comparison": "independently_recomputed_exact_identity",
                "run_identity_sha256": off_a["run_identity_sha256"],
                "source_module_count": 0,
                "loaded_file_verification": _empty_loaded_file_verification(),
            },
            "trace_off_b": {
                "passes": True,
                "comparison": "independently_recomputed_exact_identity",
                "run_identity_sha256": off_b["run_identity_sha256"],
                "source_module_count": 0,
                "loaded_file_verification": _empty_loaded_file_verification(),
            },
        },
    }


def _context(tmp_path: Path) -> dict[str, object]:
    inputs = _common_paths(tmp_path)
    data_validation = {
        "ready": True,
        "manifest_sha256": _digest("data"),
        "artifacts": [],
    }
    return {
        "addendum_path": tmp_path / "addendum.json",
        "runtime_lock": {"runtime_lock_sha256": _digest("lock")},
        "repository": {"head": "a" * 40},
        "static_contract": {"schema_version": RUNNER.P23_STATIC_CONTRACT_SCHEMA},
        "runtime": {"backend": "cuda"},
        "data_validation": data_validation,
        "inputs": inputs,
        "scopes": {
            "repository": ROOT,
            "nanogpt": tmp_path / "nanoGPT",
            "muon": tmp_path / "muon.py",
        },
    }


def _common_paths(tmp_path: Path) -> dict[str, Path]:
    inputs = tmp_path / "inputs"
    return {
        "nanogpt_root": inputs / "nanoGPT",
        "muon_source": inputs / "muon" / "muon.py",
        "data_manifest": inputs / "fineweb" / "data.json",
        "instrumentation_patch": inputs / "observer.py",
        "runtime_lock_path": inputs / "runtime-lock.json",
        "host_attestation_path": inputs / "host-attestation.json",
        "addendum_path": inputs / "addendum.json",
    }


def _valid_observation(item: dict[str, object], step: int) -> dict[str, object]:
    name = str(item["name"])
    match = RUNNER.P22_RUNNER.MUON_NAME.fullmatch(name)
    assert match is not None
    signal_norm = 1.0
    candidate_norm = 1.0
    output_norm = 0.5
    correction_norm = 0.5
    candidate_departure = 0.1
    output_departure = 0.05
    output_to_signal = output_norm / signal_norm
    output_to_candidate = output_norm / candidate_norm
    relative_correction = correction_norm / candidate_norm
    return {
        "schema_version": RUNNER.P22_OBSERVER.P22_SHADOW_OBSERVATION_SCHEMA_VERSION,
        "evidence_kind": "real_gradient_shadow_observation",
        "optimizer_step": step,
        "tokens_seen": (step + 1) * RUNNER.P23.EXPECTED_SEQUENCE_LENGTH,
        "phase": RUNNER.P21_TRACE.phase_for_step(step),
        "seed": RUNNER.P22_RUNNER.SEED,
        "parameter": {
            "name": name,
            "layer": int(match.group("layer")),
            "role": RUNNER.P22_RUNNER.ROLE_NAMES[match.group("role")],
            "original_shape": item["stored_shape"],
            "shield_shape": item["shield_shape"],
            "transposed_for_shield": item["transposed_for_shield"],
            "entry_count": item["numel"],
        },
        "optimizer": {
            "beta": RUNNER.P22_RUNNER.MUON_MOMENTUM,
            "learning_rate": RUNNER.P22_RUNNER.MUON_LEARNING_RATE,
            "weight_decay": 0.0,
            "upstream_aspect_factor": RUNNER.P21_TRACE.upstream_aspect_factor(
                tuple(item["stored_shape"])
            ),
            "candidate_captured_after_accelerator_aspect": True,
            "aspect_applied_by_observer": False,
        },
        "storage": {
            "signal_dtype": "torch.float32",
            "raw_candidate_dtype": None,
            "aspect_candidate_dtype": "torch.bfloat16",
            "output_dtype": "torch.float32",
            "source_device": "cuda",
            "device": "cpu",
            "raw_candidate_finite": None,
            "aspect_candidate_finite": True,
            "signal_sha256": _digest(f"signal:{name}:{step}"),
            "raw_candidate_sha256": None,
            "aspect_candidate_sha256": _digest(f"candidate:{name}:{step}"),
            "output_sha256": _digest(f"output:{name}:{step}"),
        },
        "shield": {
            "action": "radial_clip",
            "active": True,
            "sector_certified_by_successful_p20_call": True,
            "sector_certificate_scope": "P20 plus exact P22 768x2304 extension",
            "signal_guard": "normal_anchored",
            "reason": "test fixture",
            "dead_zone_zero": False,
            "dead_zone_update_disturbance_upper": None,
        },
        "metrics": {
            "signal_norm": signal_norm,
            "raw_candidate_norm": None,
            "aspect_candidate_norm": candidate_norm,
            "output_norm": output_norm,
            "correction_norm": correction_norm,
            "relative_correction": relative_correction,
            "relative_correction_status": "finite_denominator",
            "candidate_output_cosine": 1.0,
            "candidate_output_cosine_status": "finite_denominator",
            "output_to_signal_amplitude": output_to_signal,
            "output_to_signal_status": "finite_denominator",
            "output_to_candidate_amplitude": output_to_candidate,
            "output_to_candidate_status": "finite_denominator",
            "candidate_best_scalar_departure": candidate_departure,
            "output_best_scalar_departure": output_departure,
            "shaping_retention": output_departure / candidate_departure,
            "informative_candidate": True,
            "in_frozen_operating_annulus": True,
            "primary_eta_1_over_120_effective_gain": (
                float(RUNNER.P21_TRACE.PRIMARY_LEARNING_RATE) * output_to_signal
            ),
            "secondary_eta_1_over_83_effective_gain": (
                float(RUNNER.P21_TRACE.SECONDARY_LEARNING_RATE) * output_to_signal
            ),
            "observed_training_step_norm": (RUNNER.P22_RUNNER.MUON_LEARNING_RATE * output_norm),
            "primary_theorem_step_norm": (
                float(RUNNER.P21_TRACE.PRIMARY_LEARNING_RATE) * output_norm
            ),
        },
        "per_observation_gates": {
            "successful_p20_sector_call": True,
            "frozen_fidelity": True,
            "frozen_annulus_output_to_candidate": True,
            "frozen_output_to_signal": True,
            "frozen_primary_effective_gain": True,
        },
        "observer_rng_sha256": _digest(f"rng:{name}:{step}"),
    }


def _valid_raw_bundle(tmp_path: Path) -> tuple[dict[str, object], dict[str, object], Path]:
    inventory = RUNNER.P22_CORE.expected_gpt2_small_muon_inventory()
    observations: list[dict[str, object]] = []
    steps: list[dict[str, object]] = [
        {"step": step, "capture": None} for step in range(RUNNER.P23.EXPECTED_OPTIMIZER_STEPS)
    ]
    for step in RUNNER.P23.EXPECTED_CAPTURE_STEPS:
        chunk = [_valid_observation(item, step) for item in inventory]
        observations.extend(chunk)
        steps[step]["capture"] = {"record_sha256": RUNNER.P22_CORE.canonical_state_sha256(chunk)}
    raw = {
        "schema_version": "passive-muon-p22-raw-real-gradient-trace-v1",
        "evidence_kind": "real_gradient_shadow_trace",
        "protocol_sha256": RUNNER.P22_CORE.protocol_sha256(),
        "run_identity_sha256": _digest("p22-run"),
        "capture_steps": list(RUNNER.P23.EXPECTED_CAPTURE_STEPS),
        "observation_count": RUNNER.P23.EXPECTED_OBSERVATION_COUNT,
        "observations": observations,
    }
    raw_path = tmp_path / "raw.json"
    _write(raw_path, raw)
    manifest = {
        "protocol_sha256": RUNNER.P22_CORE.protocol_sha256(),
        "p22_acquisition": {"run_identity_sha256": _digest("p22-run")},
        "raw_trace": {
            "path": str(raw_path),
            "sha256": RUNNER.P23.sha256_file(raw_path),
            "observation_count": RUNNER.P23.EXPECTED_OBSERVATION_COUNT,
        },
        "shape_inventory": {
            "parameters": {str(item["name"]): int(item["numel"]) for item in inventory}
        },
        "steps": steps,
    }
    return raw, manifest, raw_path


def _aggregate_validation_fixture(
    tmp_path: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    raw, trace_on, _raw_path = _valid_raw_bundle(tmp_path)
    trace_on.update(
        {
            "source_snapshot": {"source": _digest("source")},
            "runtime": {"backend": "cuda"},
            "data": {"manifest_sha256": _digest("data")},
        }
    )
    repeatability_record = {"kind": "repeatability"}
    noninterference_record = {"kind": "noninterference"}
    trace_on_record = {
        "kind": "trace_on",
        "path": str(tmp_path / "trace-on.json"),
        "sha256": _digest("trace-on-bytes"),
    }
    off_inputs = {
        "trace_off_a": {"kind": "off-a"},
        "trace_off_b": {"kind": "off-b"},
    }
    repeatability = {"inputs": off_inputs}
    noninterference = {
        "inputs": {**off_inputs, "trace_on": trace_on_record},
        "repeatability_gate": repeatability,
    }
    p23 = {
        "schema_version": RUNNER.P23_AGGREGATE_SCHEMA,
        "cuda_only": True,
        "repeatability_report": repeatability_record,
        "noninterference_report": noninterference_record,
        "trace_on_manifest": trace_on_record,
        "repeatability_report_replayed_exactly": True,
        "noninterference_report_replayed_exactly": True,
        "repeatability_passes": True,
        "noninterference_passes": True,
        "metric_reconstruction_scope": RUNNER.METRIC_RECONSTRUCTION_SCOPE,
    }
    raw_entry = trace_on["raw_trace"]
    provenance = {
        "p22_protocol_sha256": RUNNER.P22_CORE.protocol_sha256(),
        "run_manifest_path": str(Path(trace_on_record["path"]).resolve()),
        "run_manifest_sha256": trace_on_record["sha256"],
        "raw_trace_path": str(Path(raw_entry["path"]).resolve()),
        "raw_trace_sha256": raw_entry["sha256"],
        "source_snapshot": trace_on["source_snapshot"],
        "runtime": trace_on["runtime"],
        "data_manifest_sha256": trace_on["data"]["manifest_sha256"],
        "post_aspect_candidate_captured_from_accelerator": True,
        "aspect_recomputed_by_cpu_observer": False,
    }
    intended = {
        str(item["name"]): int(item["numel"])
        for item in RUNNER.P22_CORE.expected_gpt2_small_muon_inventory()
    }
    payload = RUNNER.P21_TRACE.summarize_shadow_trace(
        raw["observations"],
        evidence_kind="real_gradient_shadow_trace",
        provenance=provenance,
        intended_parameter_numel=intended,
    )
    payload["p22_extension"] = {
        "schema_version": "passive-muon-p22-real-gradient-summary-v1",
        "post_aspect_candidate_captured_from_accelerator": True,
        "shape_768x2304_separately_certified": True,
        "early_middle_late_are_step_windows_not_mature_training_phases": True,
    }
    payload["p23"] = p23
    linked = {
        "repeatability": repeatability,
        "noninterference": noninterference,
        "trace_on": trace_on,
    }
    return payload, raw, linked


def test_prepare_context_configures_before_cuda_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    lock = {"runtime_lock_sha256": _digest("lock")}
    attestation = {
        "schema_version": RUNNER.P23.P23_HOST_ATTESTATION_SCHEMA,
        "host_attestation_sha256": _digest("attestation"),
    }
    runtime = {"backend": "cuda"}

    monkeypatch.setattr(
        RUNNER.P23,
        "load_and_validate_addendum",
        lambda *args, **kwargs: events.append("addendum"),
    )
    monkeypatch.setattr(
        RUNNER,
        "_validate_frozen_fidelity_gate_constants",
        lambda **kwargs: events.append("fidelity_gates"),
    )
    monkeypatch.setattr(
        RUNNER,
        "_validate_frozen_p22_acquisition_constants",
        lambda **kwargs: events.append(
            "p22_static_contract" if "static_contract" in kwargs else "p22_constants"
        ),
    )
    monkeypatch.setattr(
        RUNNER.P23,
        "load_runtime_lock",
        lambda *args, **kwargs: (events.append("runtime_lock"), lock)[1],
    )
    monkeypatch.setattr(
        RUNNER.P23,
        "load_and_validate_host_attestation",
        lambda *args, **kwargs: (events.append("host_attestation"), attestation)[1],
    )
    monkeypatch.setattr(
        RUNNER.P23,
        "collect_git_provenance",
        lambda *args, **kwargs: (events.append("git"), {})[1],
    )
    monkeypatch.setattr(
        RUNNER,
        "_validate_runtime_lock_repository_binding",
        lambda *args: events.append("runtime_lock_git"),
    )
    monkeypatch.setattr(
        RUNNER,
        "_validate_host_attestation_repository_binding",
        lambda *args: events.append("host_attestation_git"),
    )
    monkeypatch.setattr(
        RUNNER,
        "build_static_contract",
        lambda **kwargs: (events.append("static"), {})[1],
    )
    monkeypatch.setattr(
        RUNNER.P22_CORE,
        "validate_fineweb_manifest",
        lambda path: {"ready": True, "manifest_sha256": _digest("data"), "artifacts": []},
    )
    monkeypatch.setattr(
        RUNNER.P23,
        "configure_cuda_determinism",
        lambda: events.append("configure"),
    )
    monkeypatch.setattr(
        RUNNER.P23,
        "collect_cuda_runtime",
        lambda **kwargs: (events.append("cuda_runtime"), runtime)[1],
    )
    monkeypatch.setattr(
        RUNNER.P23,
        "validate_runtime_against_lock",
        lambda *args: events.append("validate_runtime"),
    )

    paths = {name: Path(name) for name in _common_paths(Path("root"))}
    RUNNER._prepare_context(**paths)

    assert events == [
        "addendum",
        "fidelity_gates",
        "p22_constants",
        "runtime_lock",
        "host_attestation",
        "git",
        "runtime_lock_git",
        "host_attestation_git",
        "static",
        "p22_static_contract",
        "configure",
        "cuda_runtime",
        "validate_runtime",
    ]


def test_live_fidelity_constants_match_hash_frozen_protocol() -> None:
    RUNNER._validate_frozen_fidelity_gate_constants()


def test_frozen_fidelity_gate_table_tamper_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = RUNNER.P21_TRACE.load_frozen_protocol()
    protocol["frozen_gates"]["mild_intervention"][  # type: ignore[index]
        "candidate_output_cosine_minimum"
    ] = "3/4"
    monkeypatch.setattr(RUNNER.P21_TRACE, "load_frozen_protocol", lambda: protocol)

    with pytest.raises(RUNNER.P23AcquisitionError, match="fidelity-gate table changed"):
        RUNNER._validate_frozen_fidelity_gate_constants()


@pytest.mark.parametrize(
    "constant_name",
    [
        "PRIMARY_LEARNING_RATE",
        "SECONDARY_LEARNING_RATE",
        "INFORMATIVE_CANDIDATE_DEPARTURE",
        "MINIMUM_OUTPUT_DEPARTURE",
        "MINIMUM_SHAPING_RETENTION",
        "ANNULUS_SIGNAL_NORM_LOWER",
        "ANNULUS_SIGNAL_NORM_UPPER",
        "ANNULUS_OUTPUT_TO_CANDIDATE_LOWER",
        "ANNULUS_OUTPUT_TO_CANDIDATE_UPPER",
        "OUTPUT_TO_SIGNAL_LOWER",
        "OUTPUT_TO_SIGNAL_UPPER",
        "PRIMARY_EFFECTIVE_GAIN_LOWER",
        "PRIMARY_EFFECTIVE_GAIN_UPPER",
        "MAXIMUM_OVERALL_ACTIVATION_RATE",
        "MAXIMUM_CELL_ACTIVATION_RATE",
        "MINIMUM_CELL_COUNT",
        "MAXIMUM_MEDIAN_RELATIVE_CORRECTION",
        "MAXIMUM_P95_RELATIVE_CORRECTION",
        "MAXIMUM_RELATIVE_CORRECTION",
        "MINIMUM_P05_COSINE",
        "MINIMUM_COSINE",
        "MINIMUM_P05_OUTPUT_TO_CANDIDATE",
        "MAXIMUM_P95_OUTPUT_TO_CANDIDATE",
    ],
)
def test_each_live_fidelity_constant_tamper_is_rejected(
    monkeypatch: pytest.MonkeyPatch, constant_name: str
) -> None:
    monkeypatch.setattr(RUNNER.P21_TRACE, constant_name, object())

    with pytest.raises(RUNNER.P23AcquisitionError, match=constant_name):
        RUNNER._validate_frozen_fidelity_gate_constants()


def test_live_p22_acquisition_constants_match_frozen_protocol() -> None:
    RUNNER._validate_frozen_p22_acquisition_constants()


def test_frozen_p22_protocol_semantic_tamper_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = RUNNER.P22_CORE.load_protocol()
    protocol["optimizer"]["aspect_formula"] = "sqrt(rows/columns)"  # type: ignore[index]
    monkeypatch.setattr(RUNNER.P22_CORE, "load_protocol", lambda: protocol)

    with pytest.raises(
        RUNNER.P23AcquisitionError,
        match=r"optimizer\.aspect_formula changed",
    ):
        RUNNER._validate_frozen_p22_acquisition_constants()


@pytest.mark.parametrize(
    ("owner_name", "constant_name"),
    [
        ("P22_RUNNER", "SEED"),
        ("P22_RUNNER", "SEQUENCE_LENGTH"),
        ("P22_RUNNER", "MICRO_BATCH_SIZE"),
        ("P22_RUNNER", "GRADIENT_ACCUMULATION_STEPS"),
        ("P22_RUNNER", "MODEL_BLOCK_SIZE"),
        ("P22_RUNNER", "VOCAB_SIZE"),
        ("P22_RUNNER", "MODEL_N_LAYER"),
        ("P22_RUNNER", "MODEL_N_HEAD"),
        ("P22_RUNNER", "MODEL_N_EMBD"),
        ("P22_RUNNER", "MODEL_DROPOUT"),
        ("P22_RUNNER", "MODEL_BIAS"),
        ("P22_RUNNER", "MUON_LEARNING_RATE"),
        ("P22_RUNNER", "MUON_MOMENTUM"),
        ("P22_RUNNER", "MUON_WEIGHT_DECAY"),
        ("P22_RUNNER", "AUXILIARY_LEARNING_RATE"),
        ("P22_RUNNER", "AUXILIARY_BETAS"),
        ("P22_RUNNER", "AUXILIARY_EPSILON"),
        ("P22_RUNNER", "AUXILIARY_WEIGHT_DECAY"),
        ("P22_RUNNER", "CAPTURE_STEPS"),
        ("P22_RUNNER", "EXPECTED_OPTIMIZER_STEPS"),
        ("P22_RUNNER", "EXPECTED_CANDIDATE_OBSERVATIONS"),
        ("P22_RUNNER", "EXPECTED_TRAINING_TOKENS_CONSUMED"),
        ("P22_RUNNER", "STATE_CHECKPOINT_STEPS"),
        ("P22_RUNNER", "P22_RUN_MANIFEST_SCHEMA"),
        ("P22_RUNNER", "ROLE_NAMES"),
        ("P22_RUNNER", "EXPECTED_TIED_ALIAS_INVENTORY"),
        ("P22_CORE", "P22_PROTOCOL_SCHEMA"),
        ("P22_CORE", "P22_RUN_MANIFEST_SCHEMA"),
        ("P22_CORE", "NANOGPT_REVISION"),
        ("P22_CORE", "NANOGPT_TREE"),
        ("P22_CORE", "NANOGPT_TRAIN_BLOB"),
        ("P22_CORE", "NANOGPT_MODEL_BLOB"),
        ("P22_CORE", "MUON_REVISION"),
        ("P22_CORE", "MUON_SOURCE_SHA256"),
        ("P22_CORE", "EXPECTED_OPTIMIZER_STEPS"),
        ("P22_CORE", "EXPECTED_CAPTURE_STEPS"),
        ("P22_CORE", "EXPECTED_CANDIDATE_OBSERVATIONS"),
        ("P22_CORE", "EXPECTED_MODEL_PARAMETER_COUNT"),
        ("P22_CORE", "EXPECTED_MUON_PARAMETER_COUNT"),
        ("P22_CORE", "EXPECTED_AUXILIARY_PARAMETER_COUNT"),
        ("P22_CORE", "EXPECTED_TIED_PARAMETER_ALIASES"),
        ("P22_CORE", "EXPECTED_TRAINING_TOKENS_CONSUMED"),
        ("P22_CORE", "EXPECTED_SEQUENCE_LENGTH"),
        ("P22_CORE", "EXPECTED_TRAIN_POOL_TOKENS"),
        ("P22_CORE", "EXPECTED_VALIDATION_TOKENS"),
        ("P22_CORE", "EXPECTED_TRAIN_BYTES"),
        ("P22_CORE", "EXPECTED_VALIDATION_BYTES"),
        ("P22_CORE", "PROTOCOL_PATH"),
        ("P22_CORE", "DATA_TEMPLATE_PATH"),
        ("P22_CORE", "NONINTERFERENCE_STEP_FIELDS"),
        ("P22_CORE", "NONINTERFERENCE_STATE_FIELDS"),
        ("P22_CORE", "STATE_CHECKPOINT_STEPS"),
        ("P22_CORE", "NONINTERFERENCE_FINAL_FIELDS"),
        ("P22_OBSERVER", "P22_SHADOW_OBSERVATION_SCHEMA_VERSION"),
    ],
)
def test_each_live_p22_acquisition_constant_tamper_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    owner_name: str,
    constant_name: str,
) -> None:
    owner = getattr(RUNNER, owner_name)
    monkeypatch.setattr(owner, constant_name, object())

    with pytest.raises(RUNNER.P23AcquisitionError, match=constant_name):
        RUNNER._validate_frozen_p22_acquisition_constants()


def test_live_p22_muon_parameter_rule_tamper_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(RUNNER.P22_RUNNER, "MUON_NAME", object())

    with pytest.raises(RUNNER.P23AcquisitionError, match="MUON_NAME"):
        RUNNER._validate_frozen_p22_acquisition_constants()


def test_p23_static_contract_semantic_tamper_is_rejected() -> None:
    with pytest.raises(
        RUNNER.P23AcquisitionError,
        match="static contract field schema_version",
    ):
        RUNNER._validate_frozen_p22_acquisition_constants(
            static_contract={"schema_version": "tampered"}
        )


def test_source_collector_uses_git_tracked_repository_allowlist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    def collect(scopes: object, **kwargs: object) -> dict[str, dict[str, object]]:
        captured["scopes"] = scopes
        captured.update(kwargs)
        return {"runner": {"scope": "repository"}}

    monkeypatch.setattr(RUNNER.P23, "collect_runtime_module_hashes", collect)
    context = {
        "scopes": {
            "repository": ROOT,
            "nanogpt": tmp_path / "nanoGPT",
            "muon": tmp_path / "muon.py",
        },
        "repository": {
            "tracked_files": {
                "tracked.py": {"kind": "file"},
                "symlink.py": {"kind": "symlink"},
            }
        },
    }
    RUNNER._collect_source_modules(context)
    assert captured["scope_file_allowlists"] == {"repository": {"tracked.py"}}


def test_static_contract_is_role_and_output_independent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        RUNNER.P22_CORE,
        "validate_nanogpt_checkout",
        lambda path: {
            "ready": True,
            "details": {
                "revision": "revision",
                "tree": "tree",
                "train_py_blob": "train",
                "model_py_blob": "model",
                "origin": "origin",
            },
        },
    )
    monkeypatch.setattr(
        RUNNER.P22_CORE,
        "validate_muon_source",
        lambda path: {"ready": True, "revision": "muon", "sha256": _digest("muon")},
    )
    monkeypatch.setattr(
        RUNNER.P22_CORE,
        "validate_fineweb_manifest",
        lambda path: {
            "ready": True,
            "manifest_sha256": _digest("data"),
            "artifacts": [{"sha256": _digest("tokens"), "byte_count": 8}],
        },
    )
    monkeypatch.setattr(
        RUNNER.P22_CORE,
        "validate_instrumentation_patch",
        lambda path: {"ready": True, "sha256": _digest("observer"), "byte_count": 10},
    )
    monkeypatch.setattr(RUNNER.P22_CORE, "protocol_sha256", lambda: _digest("protocol"))

    contract = RUNNER.build_static_contract(
        nanogpt_root=Path("nanoGPT"),
        muon_source=Path("muon.py"),
        data_manifest=Path("data.json"),
        instrumentation_patch=Path("observer.py"),
        runtime_lock_path=ROOT / "runtime-lock.json",
        runtime_lock={"runtime_lock_sha256": _digest("lock")},
        host_attestation_path=ROOT / "host-attestation.json",
        host_attestation={
            "schema_version": RUNNER.P23.P23_HOST_ATTESTATION_SCHEMA,
            "host_attestation_sha256": _digest("attestation"),
        },
    )

    assert contract["backend"] == "cuda"
    assert contract["candidate_observation_count"] == 1_152
    assert contract["host_attestation"]["sha256"] == _digest("attestation")
    assert "acquisition_role" not in contract
    assert "trace_mode" not in contract
    assert "output" not in contract
    assert set(contract["identity_excludes"]) >= {
        "acquisition_role",
        "trace_mode",
        "output_path",
        "elapsed_seconds",
    }


def test_trace_on_without_complete_gate_is_barred_before_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        RUNNER,
        "_prepare_context",
        lambda **kwargs: pytest.fail("CUDA/runtime preparation must not run"),
    )

    with pytest.raises(RUNNER.P23AcquisitionError, match="trace_on requires"):
        RUNNER.acquire_run(
            role="trace_on",
            **_common_paths(tmp_path),
            output=tmp_path / "trace-on.json",
        )


def test_process_instance_is_fixed_for_interpreter_lifetime() -> None:
    first = RUNNER._new_process_instance()
    second = RUNNER._new_process_instance()
    assert first == second
    assert first is not second


def test_trace_on_cryptographic_gate_rejects_nonzero_mismatches_before_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    off_a_path = tmp_path / "off-a.json"
    off_b_path = tmp_path / "off-b.json"
    report_path = tmp_path / "repeatability.json"
    off_a = _minimal_manifest("trace_off_a", "off-a")
    off_b = _minimal_manifest("trace_off_b", "off-b")
    _write(off_a_path, off_a)
    _write(off_b_path, off_b)
    report = _passing_report(off_a_path, off_b_path, off_a, off_b)
    report["mismatch_count"] = 1
    report["mismatches"] = [{"scope": "step"}]
    _write(report_path, report)
    monkeypatch.setattr(RUNNER.P23, "validate_run_manifest", lambda *args: None)
    monkeypatch.setattr(
        RUNNER.P22_CORE,
        "_validate_model_optimizer_binding",
        lambda *args: None,
    )
    monkeypatch.setattr(
        RUNNER,
        "_prepare_context",
        lambda **kwargs: pytest.fail("CUDA/runtime preparation must not run"),
    )

    with pytest.raises(RUNNER.P23AcquisitionError, match="zero mismatches"):
        RUNNER.acquire_run(
            role="trace_on",
            **_common_paths(tmp_path),
            output=tmp_path / "trace-on.json",
            raw_trace_output=tmp_path / "raw.json",
            trace_off_a_path=off_a_path,
            trace_off_b_path=off_b_path,
            repeatability_report_path=report_path,
        )


@pytest.mark.parametrize(
    ("role_b", "reuse_nonce", "message"),
    [
        ("trace_off_a", False, "acquisition_role='trace_off_b'"),
        ("trace_off_b", True, "reuse one process-instance nonce"),
    ],
)
def test_trace_on_rejects_wrong_roles_and_copied_process_instances_before_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    role_b: str,
    reuse_nonce: bool,
    message: str,
) -> None:
    off_a_path = tmp_path / "off-a.json"
    off_b_path = tmp_path / "off-b.json"
    report_path = tmp_path / "repeatability.json"
    off_a = _minimal_manifest("trace_off_a", "off-a")
    off_b = _minimal_manifest(role_b, "off-b")
    if reuse_nonce:
        off_b["process_instance"]["nonce"] = off_a["process_instance"]["nonce"]
    _write(off_a_path, off_a)
    _write(off_b_path, off_b)
    _write(report_path, _passing_report(off_a_path, off_b_path, off_a, off_b))
    monkeypatch.setattr(RUNNER.P23, "validate_run_manifest", lambda *args: None)
    monkeypatch.setattr(
        RUNNER.P22_CORE,
        "_validate_model_optimizer_binding",
        lambda *args: None,
    )
    monkeypatch.setattr(
        RUNNER,
        "_prepare_context",
        lambda **kwargs: pytest.fail("CUDA/runtime preparation must not run"),
    )

    with pytest.raises(RUNNER.P23AcquisitionError, match=message):
        RUNNER.acquire_run(
            role="trace_on",
            **_common_paths(tmp_path),
            output=tmp_path / "trace-on.json",
            raw_trace_output=tmp_path / "raw.json",
            trace_off_a_path=off_a_path,
            trace_off_b_path=off_b_path,
            repeatability_report_path=report_path,
        )


def test_enrichment_exposes_exact_cuda_capture_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        RUNNER.P22_CORE,
        "_validate_model_optimizer_binding",
        lambda *args: None,
    )
    monkeypatch.setattr(RUNNER.P23, "execution_identity_sha256", lambda value: _digest("identity"))
    binding = RUNNER._expected_binding_audit()
    shape_inventory = RUNNER._expected_shape_inventory()
    source_snapshot = {"source": _digest("source")}
    data = {"manifest_sha256": _digest("data")}
    payload = {
        "schema_version": RUNNER.P22_CORE.P22_RUN_MANIFEST_SCHEMA,
        "trace_mode": "trace_on",
        "accelerator_backend": "cuda",
        "run_identity_sha256": RUNNER.P22_RUNNER._run_identity(
            source_snapshot,
            str(data["manifest_sha256"]),
            "cuda",
        ),
        "runtime": {"backend": "cuda-p22"},
        "source_snapshot": source_snapshot,
        "data": data,
        "model_optimizer_binding": binding,
        "shape_inventory": shape_inventory,
        "steps": [
            {
                "capture": {
                    "parameter_count": 48,
                    "actual_post_aspect_candidate": True,
                    "actual_post_aspect_cuda_candidate": False,
                }
            },
            {"capture": None},
        ],
    }

    result = RUNNER._enrich_p22_payload(
        payload,
        role="trace_on",
        execution_identity={
            "schema_version": "identity",
            "static_contract": {
                "model_optimizer_binding": binding,
                "shape_inventory": shape_inventory,
            },
        },
        runtime={"backend": "cuda"},
        process_instance={"schema_version": RUNNER.P23_PROCESS_INSTANCE_SCHEMA},
    )

    assert result["schema_version"] == RUNNER.P23.P23_RUN_MANIFEST_SCHEMA
    assert result["acquisition_role"] == "trace_on"
    assert result["candidate_execution"] == {
        "backend": "cuda",
        "actual_accelerator_candidate_computed_and_applied": True,
        "shield_shadow_only": True,
    }
    assert result["candidate_observation"] == RUNNER.P23.TRACE_ON_OBSERVATION
    assert result["p22_acquisition"] == RUNNER._p22_identity_binding(payload)
    assert "manifest_canonical_sha256" not in result["p22_acquisition"]
    assert result["metric_reconstruction_scope"] == RUNNER.METRIC_RECONSTRUCTION_SCOPE
    assert result["steps"][0]["capture"]["actual_post_aspect_cuda_candidate"] is True
    assert payload["steps"][0]["capture"]["actual_post_aspect_cuda_candidate"] is False


def test_expected_identity_records_exact_parameter_shapes() -> None:
    binding = RUNNER._expected_binding_audit()
    inventory = binding["parameter_inventory"]
    assert isinstance(inventory, list)
    assert len(inventory) == 75
    qkv = next(
        record for record in inventory if record["name"] == "transformer.h.0.attn.c_attn.weight"
    )
    assert qkv == {
        "name": "transformer.h.0.attn.c_attn.weight",
        "aliases": ["transformer.h.0.attn.c_attn.weight"],
        "optimizer_group_index": 1,
        "use_muon": True,
        "device": "cuda:0",
        "dtype": "torch.float32",
        "shape": [2304, 768],
        "numel": 2304 * 768,
    }
    shapes = RUNNER._expected_shape_inventory()["stored_shapes"]
    assert shapes["transformer.h.0.attn.c_attn.weight"] == [2304, 768]


def test_enrichment_rejects_binding_outside_execution_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        RUNNER.P22_CORE,
        "_validate_model_optimizer_binding",
        lambda *args: None,
    )
    payload = {
        "schema_version": RUNNER.P22_CORE.P22_RUN_MANIFEST_SCHEMA,
        "trace_mode": "trace_off",
        "accelerator_backend": "cuda",
        "model_optimizer_binding": {"verified": True},
        "shape_inventory": RUNNER._expected_shape_inventory(),
    }
    identity = {
        "static_contract": {
            "model_optimizer_binding": {"verified": False},
            "shape_inventory": RUNNER._expected_shape_inventory(),
        }
    }

    with pytest.raises(RUNNER.P23AcquisitionError, match="binding differs"):
        RUNNER._enrich_p22_payload(
            payload,
            role="trace_off_a",
            execution_identity=identity,
            runtime={"backend": "cuda"},
            process_instance={},
        )


def test_trace_off_acquisition_locks_sources_and_enriches_after_p22(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[str] = []
    context = _context(tmp_path)
    source_map = {
        "runner": {
            "scope": "repository",
            "logical_path": "repository:runner.py",
            "sha256": _digest("runner"),
        }
    }
    monkeypatch.setattr(
        RUNNER,
        "_prepare_context",
        lambda **kwargs: (events.append("prepare"), context)[1],
    )
    monkeypatch.setattr(
        RUNNER,
        "_collect_source_modules",
        lambda value: (events.append("sources"), source_map)[1],
    )
    monkeypatch.setattr(
        RUNNER,
        "_collect_loaded_files",
        lambda value: (events.append("loaded_files"), {"snapshot": True})[1],
    )
    monkeypatch.setattr(
        RUNNER.P23,
        "build_loaded_file_closure",
        lambda initial, final: (events.append("loaded_file_lock"), {"closure": True})[1],
    )
    monkeypatch.setattr(
        RUNNER,
        "_assert_repository_unchanged",
        lambda value: events.append("git_lock"),
    )
    monkeypatch.setattr(
        RUNNER,
        "_assert_data_unchanged",
        lambda value: events.append("data_lock"),
    )

    def run_p22(**kwargs: object) -> dict[str, object]:
        events.append("p22")
        callback = kwargs["_lifecycle_callback"]
        callback("initialized")
        events.append("training")
        callback("completed")
        return {"p22": True}

    monkeypatch.setattr(RUNNER.P22_RUNNER, "run_trace", run_p22)
    monkeypatch.setattr(
        RUNNER.P22_CORE,
        "_load_run_manifest",
        lambda *args: {"p22": True},
    )
    monkeypatch.setattr(
        RUNNER.P23,
        "build_execution_identity",
        lambda **kwargs: (events.append("identity"), {"identity": True})[1],
    )
    monkeypatch.setattr(
        RUNNER,
        "_enrich_p22_payload",
        lambda *args, **kwargs: (events.append("enrich"), {"manifest": True})[1],
    )
    monkeypatch.setattr(
        RUNNER.P23,
        "validate_run_manifest",
        lambda *args: events.append("validate"),
    )
    monkeypatch.setattr(
        RUNNER,
        "_verify_manifest_identity",
        lambda *args, **kwargs: events.append("verify"),
    )
    monkeypatch.setattr(
        RUNNER.P22_CORE,
        "write_json_atomic",
        lambda *args: events.append("write"),
    )

    result = RUNNER.acquire_run(
        role="trace_off_a",
        **_common_paths(tmp_path),
        output=tmp_path / "off-a.json",
    )

    assert result == {"manifest": True}
    assert events == [
        "prepare",
        "p22",
        "sources",
        "loaded_files",
        "training",
        "sources",
        "loaded_files",
        "loaded_file_lock",
        "git_lock",
        "data_lock",
        "identity",
        "enrich",
        "validate",
        "verify",
        "write",
    ]


def test_trace_on_replays_live_identity_before_first_training_step(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[str] = []
    off_a_path = tmp_path / "off-a.json"
    off_b_path = tmp_path / "off-b.json"
    report_path = tmp_path / "repeatability.json"
    off_a = _minimal_manifest("trace_off_a", "off-a")
    off_b = _minimal_manifest("trace_off_b", "off-b")
    _write(off_a_path, off_a)
    _write(off_b_path, off_b)
    _write(report_path, _passing_report(off_a_path, off_b_path, off_a, off_b))
    monkeypatch.setattr(
        RUNNER.P23,
        "validate_run_manifest",
        lambda *args: events.append("manifest_check"),
    )
    monkeypatch.setattr(
        RUNNER.P22_CORE,
        "_validate_model_optimizer_binding",
        lambda *args: events.append("binding_check"),
    )
    monkeypatch.setattr(
        RUNNER,
        "_prepare_context",
        lambda **kwargs: (events.append("prepare"), _context(tmp_path))[1],
    )
    monkeypatch.setattr(RUNNER, "_collect_source_modules", lambda value: {"module": {}})
    monkeypatch.setattr(RUNNER, "_collect_loaded_files", lambda value: {"snapshot": True})
    monkeypatch.setattr(
        RUNNER.P23,
        "build_loaded_file_closure",
        lambda initial, final: {"closure": True},
    )
    monkeypatch.setattr(RUNNER, "_assert_repository_unchanged", lambda value: None)
    monkeypatch.setattr(RUNNER, "_assert_data_unchanged", lambda value: None)

    def replay(**kwargs: object) -> dict[str, object]:
        events.append("live_repeatability_replay")
        return json.loads(report_path.read_text(encoding="utf-8"))

    monkeypatch.setattr(RUNNER, "_repeatability_report", replay)

    def run_p22(**kwargs: object) -> dict[str, object]:
        callback = kwargs["_lifecycle_callback"]
        callback("initialized")
        events.append("first_training_step")
        callback("completed")
        raw_path = kwargs["raw_trace_output"]
        assert isinstance(raw_path, Path)
        _write(raw_path, {"observations": []})
        return {"p22": True}

    monkeypatch.setattr(RUNNER.P22_RUNNER, "run_trace", run_p22)
    monkeypatch.setattr(RUNNER.P22_CORE, "_load_run_manifest", lambda *args: {"p22": True})
    monkeypatch.setattr(RUNNER.P23, "build_execution_identity", lambda **kwargs: {})
    monkeypatch.setattr(RUNNER, "_enrich_p22_payload", lambda *args, **kwargs: {})
    monkeypatch.setattr(RUNNER, "_verify_manifest_identity", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        RUNNER,
        "_validate_raw_trace_binding",
        lambda *args, **kwargs: {"observations": []},
    )

    RUNNER.acquire_run(
        role="trace_on",
        **_common_paths(tmp_path),
        output=tmp_path / "trace-on.json",
        raw_trace_output=tmp_path / "raw.json",
        trace_off_a_path=off_a_path,
        trace_off_b_path=off_b_path,
        repeatability_report_path=report_path,
    )

    assert events.index("manifest_check") < events.index("prepare")
    assert events.index("live_repeatability_replay") < events.index("first_training_step")


def test_aggregate_refuses_failed_gate_before_p22_aggregation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _common_paths(tmp_path)
    off_a_path = tmp_path / "off-a.json"
    off_b_path = tmp_path / "off-b.json"
    trace_on_path = tmp_path / "trace-on.json"
    repeatability_path = tmp_path / "repeatability.json"
    noninterference_path = tmp_path / "noninterference.json"
    for path, label in (
        (off_a_path, "off-a"),
        (off_b_path, "off-b"),
        (trace_on_path, "trace-on"),
    ):
        role = "trace_on" if label == "trace-on" else f"trace_off_{label[-1]}"
        _write(path, _minimal_manifest(role, label))
    _write(
        repeatability_path,
        {
            "schema_version": RUNNER.P23.P23_REPEATABILITY_SCHEMA,
            "passes": False,
            "mismatch_count": 1,
            "mismatches": [{}],
        },
    )
    _write(noninterference_path, {})
    monkeypatch.setattr(RUNNER, "_prepare_context", lambda **kwargs: _context(tmp_path))
    monkeypatch.setattr(RUNNER, "_prime_verifier_initialization", lambda *args: None)
    monkeypatch.setattr(RUNNER, "_collect_source_modules", lambda value: {})
    monkeypatch.setattr(
        RUNNER.P22_RUNNER,
        "aggregate_trace",
        lambda **kwargs: pytest.fail("P22 aggregate must remain barred"),
    )

    with pytest.raises(RUNNER.P23AcquisitionError, match="not a passing"):
        RUNNER.aggregate(
            trace_off_a_path=off_a_path,
            trace_off_b_path=off_b_path,
            trace_on_path=trace_on_path,
            repeatability_report_path=repeatability_path,
            noninterference_report_path=noninterference_path,
            **paths,
            output=tmp_path / "aggregate.json",
        )


def test_runtime_lock_must_be_repository_tracked(tmp_path: Path) -> None:
    with pytest.raises(RUNNER.P23AcquisitionError, match="inside"):
        RUNNER._validate_runtime_lock_repository_binding(
            tmp_path / "runtime-lock.json",
            {"runtime_lock_sha256": _digest("lock")},
            {"tracked_files": {}},
        )
    with pytest.raises(RUNNER.P23AcquisitionError, match="not tracked"):
        RUNNER._validate_runtime_lock_repository_binding(
            ROOT / "untracked-runtime-lock.json",
            {"runtime_lock_sha256": _digest("lock")},
            {"tracked_files": {}},
        )
    with pytest.raises(RUNNER.P23AcquisitionError, match=r"host attestation.*not tracked"):
        RUNNER._validate_host_attestation_repository_binding(
            ROOT / "untracked-host-attestation.json",
            {"host_attestation_sha256": _digest("attestation")},
            {"tracked_files": {}},
        )


def test_raw_trace_capture_digest_is_deeply_bound(tmp_path: Path) -> None:
    raw, manifest, raw_path = _valid_raw_bundle(tmp_path)
    RUNNER._validate_raw_trace_binding(manifest)

    raw["observations"][0]["parameter"]["name"] = "corrupted"
    _write(raw_path, raw)
    manifest["raw_trace"]["sha256"] = RUNNER.P23.sha256_file(raw_path)
    with pytest.raises(RUNNER.P23AcquisitionError, match="undeclared parameter"):
        RUNNER._validate_raw_trace_binding(manifest)


@pytest.mark.parametrize(
    ("section", "field", "replacement", "message"),
    [
        ("parameter", "original_shape", [1, 1], "metadata changed"),
        ("optimizer", "beta", 0.9, "optimizer contract changed"),
        ("storage", "signal_dtype", "torch.bfloat16", "storage contract changed"),
        ("storage", "aspect_candidate_finite", False, "aborts and discards"),
        (
            "metrics",
            "primary_eta_1_over_120_effective_gain",
            0.0,
            "effective-step relation changed",
        ),
        ("per_observation_gates", "frozen_fidelity", False, "gate relations changed"),
    ],
)
def test_raw_semantic_tamper_fails_even_after_hashes_are_rebound(
    tmp_path: Path,
    section: str,
    field: str,
    replacement: object,
    message: str,
) -> None:
    raw, manifest, raw_path = _valid_raw_bundle(tmp_path)
    observation = raw["observations"][0]
    observation[section][field] = replacement
    first_chunk = raw["observations"][: RUNNER.P23.EXPECTED_PARAMETER_COUNT]
    manifest["steps"][0]["capture"]["record_sha256"] = RUNNER.P22_CORE.canonical_state_sha256(
        first_chunk
    )
    _write(raw_path, raw)
    manifest["raw_trace"]["sha256"] = RUNNER.P23.sha256_file(raw_path)

    with pytest.raises(RUNNER.P23AcquisitionError, match=message):
        RUNNER._validate_raw_trace_binding(manifest)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("protocol_sha256", "tampered"),
        ("evidence_kind", "synthetic_cpu_diagnostic"),
        ("accelerator_backend", "mps"),
        ("model_optimizer_binding", {"verified": False}),
        ("shape_inventory", {"parameter_count": 0}),
        ("data", {"ready": False}),
        ("runtime", {"backend": "mps"}),
        ("metric_reconstruction_scope", "overclaim"),
    ],
)
def test_static_binding_rejects_top_level_tampering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    protocol = _digest("protocol")
    sources = {"source": _digest("source")}
    data = {
        "ready": True,
        "manifest_sha256": _digest("data"),
        "artifacts": [],
    }
    runtime = {"backend": "cuda"}
    p22_runtime = {"backend": "cuda-p22"}
    static = {
        "model_optimizer_binding": {"verified": True},
        "shape_inventory": {"parameter_count": 48},
    }
    paths = _common_paths(tmp_path)
    context = {
        "static_contract": static,
        "runtime": runtime,
        "data_validation": data,
        "inputs": paths,
    }
    manifest = {
        "execution_identity": {"static_contract": static},
        "protocol_sha256": protocol,
        "evidence_kind": "real_gradient_shadow_trace",
        "accelerator_backend": "cuda",
        "model_optimizer_binding": static["model_optimizer_binding"],
        "shape_inventory": static["shape_inventory"],
        "data": data,
        "runtime": runtime,
        "metric_reconstruction_scope": RUNNER.METRIC_RECONSTRUCTION_SCOPE,
        "source_snapshot": sources,
        "p22_runtime_provenance": p22_runtime,
    }
    monkeypatch.setattr(RUNNER.P22_CORE, "protocol_sha256", lambda: protocol)
    monkeypatch.setattr(RUNNER.P22_RUNNER, "_source_snapshot", lambda **kwargs: sources)
    monkeypatch.setattr(RUNNER.P22_RUNNER, "_runtime_provenance", lambda backend: p22_runtime)
    manifest["p22_acquisition"] = RUNNER._p22_identity_binding(manifest)
    monkeypatch.setattr(RUNNER, "_validate_reconstructed_p22_manifest", lambda *args: None)
    RUNNER._validate_manifest_static_binding(
        manifest,
        expected_mode="trace_off",
        context=context,
    )

    tampered = json.loads(json.dumps(manifest))
    tampered[field] = replacement
    with pytest.raises(RUNNER.P23AcquisitionError):
        RUNNER._validate_manifest_static_binding(
            tampered,
            expected_mode="trace_off",
            context=context,
        )


def test_exact_gate_replay_rejects_passing_but_different_report() -> None:
    supplied = {"passes": True, "mismatches": [], "marker": "stored"}
    replayed = {"passes": True, "mismatches": [], "marker": "fresh"}
    with pytest.raises(RUNNER.P23AcquisitionError, match="differs from fresh exact replay"):
        RUNNER._require_exact_gate_replay(supplied, replayed, gate="trace_on")


def test_gate_report_validator_rechecks_linked_artifact_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    off_a_path = tmp_path / "off-a.json"
    off_b_path = tmp_path / "off-b.json"
    off_a = _minimal_manifest("trace_off_a", "off-a")
    off_b = _minimal_manifest("trace_off_b", "off-b")
    _write(off_a_path, off_a)
    _write(off_b_path, off_b)
    report = _passing_report(off_a_path, off_b_path, off_a, off_b)
    monkeypatch.setattr(RUNNER.P23, "validate_run_manifest", lambda *args: None)
    monkeypatch.setattr(
        RUNNER.P23,
        "compare_repeatability_manifests",
        lambda *args: {
            "schema_version": RUNNER.P23.P23_REPEATABILITY_SCHEMA,
            "passes": True,
            "comparison": "bitwise_exact_no_tolerances",
            "mismatch_count": 0,
            "mismatches": [],
        },
    )
    RUNNER._validate_gate_report_payload(
        report,
        expected_schema=RUNNER.P23.P23_REPEATABILITY_SCHEMA,
    )

    off_a["tampered"] = True
    _write(off_a_path, off_a)
    with pytest.raises(RUNNER.P23AcquisitionError, match="retained bytes"):
        RUNNER._validate_gate_report_payload(
            report,
            expected_schema=RUNNER.P23.P23_REPEATABILITY_SCHEMA,
        )


def test_gate_report_validator_rejects_tampered_loaded_file_verification(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    off_a_path = tmp_path / "off-a.json"
    off_b_path = tmp_path / "off-b.json"
    off_a = _minimal_manifest("trace_off_a", "off-a")
    off_b = _minimal_manifest("trace_off_b", "off-b")
    _write(off_a_path, off_a)
    _write(off_b_path, off_b)
    report = _passing_report(off_a_path, off_b_path, off_a, off_b)
    comparison = {
        "schema_version": RUNNER.P23.P23_REPEATABILITY_SCHEMA,
        "passes": True,
        "comparison": "bitwise_exact_no_tolerances",
        "mismatch_count": 0,
        "mismatches": [],
    }
    monkeypatch.setattr(RUNNER.P23, "validate_run_manifest", lambda *args: None)
    monkeypatch.setattr(
        RUNNER.P23,
        "compare_repeatability_manifests",
        lambda *args: comparison,
    )

    verification = report["identity_verification"]
    assert isinstance(verification, dict)
    off_a_verification = verification["trace_off_a"]
    assert isinstance(off_a_verification, dict)
    loaded = off_a_verification["loaded_file_verification"]
    assert isinstance(loaded, dict)
    loaded["unique_file_count"] = 1
    with pytest.raises(RUNNER.P23AcquisitionError, match="identity-verification"):
        RUNNER._validate_gate_report_payload(
            report,
            expected_schema=RUNNER.P23.P23_REPEATABILITY_SCHEMA,
        )


def test_failed_repeatability_report_is_exactly_validated_for_sanitization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    off_a_path = tmp_path / "off-a.json"
    off_b_path = tmp_path / "off-b.json"
    off_a = _minimal_manifest("trace_off_a", "off-a")
    off_b = _minimal_manifest("trace_off_b", "off-b")
    _write(off_a_path, off_a)
    _write(off_b_path, off_b)
    mismatch = {"scope": "step", "step": 0, "field": "loss_tensor_sha256"}
    comparison = {
        "schema_version": RUNNER.P23.P23_REPEATABILITY_SCHEMA,
        "passes": False,
        "comparison": "bitwise_exact_no_tolerances",
        "mismatch_count": 1,
        "mismatches": [mismatch],
    }
    report = {
        **comparison,
        "binding_schema": RUNNER.P23_GATE_REPORT_BINDING_SCHEMA,
        "inputs": {
            "trace_off_a": RUNNER._artifact_record(off_a_path, off_a),
            "trace_off_b": RUNNER._artifact_record(off_b_path, off_b),
        },
        "identity_verification": {
            role: {
                "passes": True,
                "comparison": "independently_recomputed_exact_identity",
                "run_identity_sha256": manifest["run_identity_sha256"],
                "source_module_count": 0,
                "loaded_file_verification": _empty_loaded_file_verification(),
            }
            for role, manifest in (("trace_off_a", off_a), ("trace_off_b", off_b))
        },
    }
    monkeypatch.setattr(RUNNER.P23, "validate_run_manifest", lambda *args: None)
    monkeypatch.setattr(
        RUNNER.P23,
        "compare_repeatability_manifests",
        lambda *args: comparison,
    )

    RUNNER._validate_gate_report_payload(
        report,
        expected_schema=RUNNER.P23.P23_REPEATABILITY_SCHEMA,
    )
    report["mismatches"] = []
    with pytest.raises(RUNNER.P23AcquisitionError, match="linked-manifest replay"):
        RUNNER._validate_gate_report_payload(
            report,
            expected_schema=RUNNER.P23.P23_REPEATABILITY_SCHEMA,
        )


def test_blocked_noninterference_report_preserves_failed_repeatability(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    off_a_path = tmp_path / "off-a.json"
    off_b_path = tmp_path / "off-b.json"
    trace_on_path = tmp_path / "trace-on.json"
    off_a = _minimal_manifest("trace_off_a", "off-a")
    off_b = _minimal_manifest("trace_off_b", "off-b")
    trace_on = _minimal_manifest("trace_on", "trace-on")
    for path, manifest in (
        (off_a_path, off_a),
        (off_b_path, off_b),
        (trace_on_path, trace_on),
    ):
        _write(path, manifest)
    mismatch = {"scope": "initial", "field": "model_state_sha256"}
    comparison = {
        "schema_version": RUNNER.P23.P23_REPEATABILITY_SCHEMA,
        "passes": False,
        "comparison": "bitwise_exact_no_tolerances",
        "mismatch_count": 1,
        "mismatches": [mismatch],
    }
    inputs = {
        "trace_off_a": RUNNER._artifact_record(off_a_path, off_a),
        "trace_off_b": RUNNER._artifact_record(off_b_path, off_b),
    }
    repeatability = {
        **comparison,
        "binding_schema": RUNNER.P23_GATE_REPORT_BINDING_SCHEMA,
        "inputs": inputs,
        "identity_verification": {
            role: {
                "passes": True,
                "comparison": "independently_recomputed_exact_identity",
                "run_identity_sha256": manifest["run_identity_sha256"],
                "source_module_count": 0,
                "loaded_file_verification": _empty_loaded_file_verification(),
            }
            for role, manifest in (("trace_off_a", off_a), ("trace_off_b", off_b))
        },
    }
    blocked = {
        "schema_version": RUNNER.P23.P23_NONINTERFERENCE_SCHEMA,
        "passes": False,
        "comparison": "blocked_by_failed_exact_repeatability",
        "mismatch_count": 1,
        "mismatches": [mismatch],
        "binding_schema": RUNNER.P23_GATE_REPORT_BINDING_SCHEMA,
        "inputs": {
            **inputs,
            "trace_on": RUNNER._artifact_record(trace_on_path, trace_on),
        },
        "repeatability_gate": repeatability,
    }
    monkeypatch.setattr(RUNNER.P23, "validate_run_manifest", lambda *args: None)
    monkeypatch.setattr(
        RUNNER.P23,
        "compare_repeatability_manifests",
        lambda *args: comparison,
    )

    RUNNER._validate_gate_report_payload(
        blocked,
        expected_schema=RUNNER.P23.P23_NONINTERFERENCE_SCHEMA,
    )


def test_aggregate_observations_must_equal_hash_bound_raw_trace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload, raw, linked = _aggregate_validation_fixture(tmp_path)

    def linked_artifact(
        record: object, *, require_identity: bool
    ) -> tuple[dict[str, object], dict[str, object]]:
        assert isinstance(record, dict)
        return record, linked[str(record["kind"])]

    monkeypatch.setattr(RUNNER, "_validate_artifact_record", linked_artifact)
    monkeypatch.setattr(RUNNER, "_validate_gate_report_payload", lambda *args, **kwargs: None)
    monkeypatch.setattr(RUNNER.P23, "validate_run_manifest", lambda *args: None)
    monkeypatch.setattr(RUNNER, "_validate_process_instance", lambda *args: "nonce")
    monkeypatch.setattr(RUNNER, "_validate_raw_trace_binding", lambda *args: raw)

    RUNNER._validate_aggregate_payload(payload)
    tampered_observations = json.loads(json.dumps(raw["observations"]))
    tampered_observations[0]["storage"]["signal_sha256"] = _digest("self-consistent-tamper")
    intended = {
        str(item["name"]): int(item["numel"])
        for item in RUNNER.P22_CORE.expected_gpt2_small_muon_inventory()
    }
    tampered = RUNNER.P21_TRACE.summarize_shadow_trace(
        tampered_observations,
        evidence_kind="real_gradient_shadow_trace",
        provenance=payload["provenance"],
        intended_parameter_numel=intended,
    )
    tampered["p22_extension"] = payload["p22_extension"]
    tampered["p23"] = payload["p23"]
    with pytest.raises(RUNNER.P23AcquisitionError, match="hash-bound raw trace"):
        RUNNER._validate_aggregate_payload(tampered)


def test_aggregate_rejects_cross_artifact_gate_splicing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload, raw, linked = _aggregate_validation_fixture(tmp_path)
    linked["noninterference"] = json.loads(json.dumps(linked["noninterference"]))
    linked["noninterference"]["repeatability_gate"] = {"inputs": {}}

    def linked_artifact(
        record: object, *, require_identity: bool
    ) -> tuple[dict[str, object], dict[str, object]]:
        assert isinstance(record, dict)
        return record, linked[str(record["kind"])]

    monkeypatch.setattr(RUNNER, "_validate_artifact_record", linked_artifact)
    monkeypatch.setattr(RUNNER, "_validate_gate_report_payload", lambda *args, **kwargs: None)
    monkeypatch.setattr(RUNNER.P23, "validate_run_manifest", lambda *args: None)
    monkeypatch.setattr(RUNNER, "_validate_process_instance", lambda *args: "nonce")
    monkeypatch.setattr(RUNNER, "_validate_raw_trace_binding", lambda *args: raw)

    with pytest.raises(RUNNER.P23AcquisitionError, match="splices inconsistent"):
        RUNNER._validate_aggregate_payload(payload)


def test_fineweb_is_revalidated_after_training(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    context = _context(tmp_path)
    monkeypatch.setattr(
        RUNNER.P22_CORE,
        "validate_fineweb_manifest",
        lambda path: {
            "ready": True,
            "manifest_sha256": _digest("changed"),
            "artifacts": [],
        },
    )
    with pytest.raises(RUNNER.P23AcquisitionError, match="FineWeb provenance changed"):
        RUNNER._assert_data_unchanged(context)


def test_native_outputs_reject_source_data_existing_and_symlink_paths(tmp_path: Path) -> None:
    dependencies = tmp_path / "dependencies"
    nanogpt = dependencies / "nanoGPT"
    muon = dependencies / "muon" / "muon.py"
    data_manifest = dependencies / "fineweb" / "manifest.json"
    evidence = tmp_path / "evidence"
    nanogpt.mkdir(parents=True)
    muon.parent.mkdir(parents=True)
    data_manifest.parent.mkdir(parents=True)
    evidence.mkdir()
    _write(data_manifest, {"train": {"path": "train.bin"}})
    inputs = {"Muon": muon, "FineWeb manifest": data_manifest}
    RUNNER._assert_native_output_paths(
        outputs={"manifest": evidence / "run.json"},
        inputs=inputs,
        nanogpt_root=nanogpt,
        data_manifest=data_manifest,
    )
    for forbidden in (
        nanogpt / "run.json",
        muon.parent / "run.json",
        data_manifest.parent / "run.json",
    ):
        with pytest.raises(RUNNER.P23AcquisitionError, match="protected"):
            RUNNER._assert_native_output_paths(
                outputs={"manifest": forbidden},
                inputs=inputs,
                nanogpt_root=nanogpt,
                data_manifest=data_manifest,
            )
    existing = evidence / "existing.json"
    _write(existing, {})
    with pytest.raises(RUNNER.P23AcquisitionError, match="already exists"):
        RUNNER._assert_native_output_paths(
            outputs={"manifest": existing},
            inputs=inputs,
            nanogpt_root=nanogpt,
            data_manifest=data_manifest,
        )
    alias = evidence / "alias.json"
    alias.symlink_to(evidence / "missing.json")
    with pytest.raises(RUNNER.P23AcquisitionError, match="already exists"):
        RUNNER._assert_native_output_paths(
            outputs={"manifest": alias},
            inputs=inputs,
            nanogpt_root=nanogpt,
            data_manifest=data_manifest,
        )


def test_sanitized_output_requires_external_nonoverlapping_native_root(tmp_path: Path) -> None:
    roots = {
        "repository": tmp_path / "repository",
        "preprocessor_alias": tmp_path / "preprocessor.py",
        "nanogpt": tmp_path / "nanogpt",
        "muon": tmp_path / "muon",
        "data": tmp_path / "data",
        "python_environment": tmp_path / "p23-venv",
        "native": tmp_path / "native",
    }
    roots["native"].mkdir()
    native_input = roots["native"] / "run.json"
    _write(native_input, {})
    RUNNER._assert_sanitize_output_path(
        manifest_path=native_input,
        output=roots["native"] / "sanitized-run.json",
        path_roots=roots,
    )
    with pytest.raises(RUNNER.P23AcquisitionError, match="outside the repository"):
        RUNNER._assert_sanitize_output_path(
            manifest_path=native_input,
            output=ROOT / "sanitized-run.json",
            path_roots=roots,
        )
    overlapping = dict(roots)
    overlapping["data"] = roots["native"] / "data"
    with pytest.raises(RUNNER.P23AcquisitionError, match="non-overlapping"):
        RUNNER._assert_sanitize_output_path(
            manifest_path=native_input,
            output=roots["native"] / "sanitized-run.json",
            path_roots=overlapping,
        )


def test_sanitized_wrapper_binds_exact_native_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    roots = {
        "repository": tmp_path / "repository",
        "preprocessor_alias": tmp_path / "preprocessor.py",
        "nanogpt": tmp_path / "nanogpt",
        "muon": tmp_path / "muon",
        "data": tmp_path / "data",
        "python_environment": tmp_path / "p23-venv",
        "native": tmp_path / "native",
    }
    roots["native"].mkdir()
    native = roots["native"] / "artifact.json"
    output = roots["native"] / "sanitized-artifact.json"
    _write(native, {"schema_version": "test"})
    expected_bytes = native.read_bytes()
    monkeypatch.setattr(RUNNER, "_validate_sanitizable_artifact", lambda payload: None)
    monkeypatch.setattr(
        RUNNER.P23,
        "sanitize_complete_manifest",
        lambda payload, **kwargs: {"schema_version": "sanitized", "manifest": payload},
    )

    result = RUNNER.sanitize(manifest_path=native, output=output, path_roots=roots)

    assert result["native_artifact_sha256"] == hashlib.sha256(expected_bytes).hexdigest()
    assert result["native_artifact_byte_count"] == len(expected_bytes)


@pytest.mark.parametrize(
    ("kind", "payload"),
    [
        (
            "trace_off_a",
            {
                "schema_version": RUNNER.P23.P23_RUN_MANIFEST_SCHEMA,
                "trace_mode": "trace_off",
                "acquisition_role": "trace_off_a",
                "metric_reconstruction_scope": RUNNER.METRIC_RECONSTRUCTION_SCOPE,
                "p22_acquisition": {"binding": True},
            },
        ),
        (
            "trace_off_b",
            {
                "schema_version": RUNNER.P23.P23_RUN_MANIFEST_SCHEMA,
                "trace_mode": "trace_off",
                "acquisition_role": "trace_off_b",
                "metric_reconstruction_scope": RUNNER.METRIC_RECONSTRUCTION_SCOPE,
                "p22_acquisition": {"binding": True},
            },
        ),
        (
            "trace_on",
            {
                "schema_version": RUNNER.P23.P23_RUN_MANIFEST_SCHEMA,
                "trace_mode": "trace_on",
                "acquisition_role": "trace_on",
                "metric_reconstruction_scope": RUNNER.METRIC_RECONSTRUCTION_SCOPE,
                "p22_acquisition": {"binding": True},
            },
        ),
        ("raw", {"schema_version": "passive-muon-p22-raw-real-gradient-trace-v1"}),
        ("repeatability", {"schema_version": RUNNER.P23.P23_REPEATABILITY_SCHEMA}),
        ("noninterference", {"schema_version": RUNNER.P23.P23_NONINTERFERENCE_SCHEMA}),
        (
            "aggregate",
            {
                "schema_version": RUNNER.P21_TRACE.P21_SHADOW_TRACE_SCHEMA_VERSION,
                "p23": {},
            },
        ),
    ],
)
def test_sanitizer_dispatches_all_seven_native_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    payload: dict[str, object],
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        RUNNER.P23,
        "validate_run_manifest",
        lambda *args: calls.append("run"),
    )
    monkeypatch.setattr(RUNNER, "_validate_process_instance", lambda *args: "nonce")
    monkeypatch.setattr(RUNNER, "_p22_identity_binding", lambda value: {"binding": True})
    monkeypatch.setattr(RUNNER, "_validate_reconstructed_p22_manifest", lambda *args: None)
    monkeypatch.setattr(RUNNER, "_validate_raw_trace_binding", lambda *args: calls.append("raw"))
    monkeypatch.setattr(RUNNER, "_validate_raw_trace_payload", lambda *args: calls.append("raw"))
    monkeypatch.setattr(
        RUNNER,
        "_validate_gate_report_payload",
        lambda *args, **kwargs: calls.append("gate"),
    )
    monkeypatch.setattr(
        RUNNER,
        "_validate_aggregate_payload",
        lambda *args: calls.append("aggregate"),
    )

    RUNNER._validate_sanitizable_artifact(payload)

    expected = (
        "run"
        if kind.startswith("trace_off")
        else "raw"
        if kind in {"trace_on", "raw"}
        else "gate"
        if kind in {"repeatability", "noninterference"}
        else "aggregate"
    )
    assert expected in calls


def test_cli_has_no_addendum_override() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'add_argument("--addendum"' not in source


def test_pre_torch_bootstrap_requires_image_interpreter_and_clean_import_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    nanogpt = tmp_path / "nanoGPT"
    muon = tmp_path / "muon"
    nanogpt.mkdir()
    muon.mkdir()
    environment = {
        "PATH": RUNNER._PINNED_EXECUTABLE_PATH,
        "CUDA_VISIBLE_DEVICES": "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "NVIDIA_VISIBLE_DEVICES": "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "VIRTUAL_ENV": "/opt/p23-venv",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "P23_REPO": str(tmp_path),
        "P23_NANOGPT": str(nanogpt),
        "P23_MUON_ROOT": str(muon),
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONHASHSEED": "1337",
        "NVIDIA_TF32_OVERRIDE": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    for name in (
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONOPTIMIZE",
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "LD_AUDIT",
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(RUNNER, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(
        RUNNER,
        "_preimport_python_flags",
        lambda: dict(RUNNER._FROZEN_PREIMPORT_PYTHON_FLAGS),
    )
    monkeypatch.setattr(RUNNER.sys, "executable", "/usr/bin/python")
    with pytest.raises(RuntimeError, match="/opt/p23-venv/bin/python"):
        RUNNER._assert_pre_torch_process_contract()

    monkeypatch.setattr(RUNNER.sys, "executable", "/opt/p23-venv/bin/python")
    RUNNER._assert_pre_torch_process_contract()

    monkeypatch.setenv("PATH", "/workspace/evidence/p23:/usr/bin")
    with pytest.raises(RuntimeError, match="process-start environment mismatch"):
        RUNNER._assert_pre_torch_process_contract()
    monkeypatch.setenv("PATH", RUNNER._PINNED_EXECUTABLE_PATH)

    monkeypatch.setattr(RUNNER.sys, "version_info", (3, 11, 13))
    with pytest.raises(RuntimeError, match=r"requires Python 3\.12"):
        RUNNER._assert_pre_torch_process_contract()
    monkeypatch.setattr(RUNNER.sys, "version_info", (3, 12, 14))
    RUNNER._assert_pre_torch_process_contract()

    monkeypatch.setenv("PYTHONPATH", "/shadow")
    with pytest.raises(RuntimeError, match="process-start environment mismatch"):
        RUNNER._assert_pre_torch_process_contract()

    monkeypatch.delenv("PYTHONPATH")
    monkeypatch.setenv("LD_PRELOAD", "/shadow.so")
    with pytest.raises(RuntimeError, match="process-start environment mismatch"):
        RUNNER._assert_pre_torch_process_contract()

    monkeypatch.delenv("LD_PRELOAD")
    monkeypatch.setattr(
        RUNNER,
        "_preimport_python_flags",
        lambda: {**RUNNER._FROZEN_PREIMPORT_PYTHON_FLAGS, "optimize": 1},
    )
    with pytest.raises(RuntimeError, match="interpreter flags mismatch"):
        RUNNER._assert_pre_torch_process_contract()

    monkeypatch.setattr(
        RUNNER,
        "_preimport_python_flags",
        lambda: dict(RUNNER._FROZEN_PREIMPORT_PYTHON_FLAGS),
    )
    monkeypatch.setenv("PYTHONOPTIMIZE", "1")
    with pytest.raises(RuntimeError, match="process-start environment mismatch"):
        RUNNER._assert_pre_torch_process_contract()

    monkeypatch.delenv("PYTHONOPTIMIZE")
    (tmp_path / ".venv").mkdir()
    with pytest.raises(RuntimeError, match=r"repository-local \.venv"):
        RUNNER._assert_pre_torch_process_contract()


def test_pre_torch_bootstrap_rejects_controlled_root_bytecode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repository = tmp_path / "repository"
    nanogpt = tmp_path / "nanoGPT"
    muon = tmp_path / "muon"
    for root in (repository, nanogpt, muon):
        root.mkdir()
    cache = nanogpt / "__pycache__"
    cache.mkdir()
    (cache / "model.cpython-312.pyc").write_bytes(b"stale")
    environment = {
        "PATH": RUNNER._PINNED_EXECUTABLE_PATH,
        "CUDA_VISIBLE_DEVICES": "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "NVIDIA_VISIBLE_DEVICES": "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "VIRTUAL_ENV": "/opt/p23-venv",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONHASHSEED": "1337",
        "NVIDIA_TF32_OVERRIDE": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "P23_REPO": str(repository),
        "P23_NANOGPT": str(nanogpt),
        "P23_MUON_ROOT": str(muon),
    }
    for name in (
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONOPTIMIZE",
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "LD_AUDIT",
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(RUNNER, "REPOSITORY_ROOT", repository)
    monkeypatch.setattr(RUNNER.sys, "executable", "/opt/p23-venv/bin/python")
    monkeypatch.setattr(
        RUNNER,
        "_preimport_python_flags",
        lambda: dict(RUNNER._FROZEN_PREIMPORT_PYTHON_FLAGS),
    )

    with pytest.raises(RuntimeError, match="forbidden bytecode"):
        RUNNER._assert_pre_torch_process_contract()


def test_path_root_parser_is_explicit_and_unique(tmp_path: Path) -> None:
    labels = (
        "repository",
        "preprocessor_alias",
        "nanogpt",
        "muon",
        "data",
        "python_environment",
        "native",
    )
    values = [f"{label}={tmp_path / label}" for label in labels]
    roots = RUNNER._path_roots(values)
    assert roots == {label: (tmp_path / label).resolve() for label in labels}
    with pytest.raises(RUNNER.P23AcquisitionError, match="unique"):
        RUNNER._path_roots([*values, f"native={tmp_path}"])
    with pytest.raises(RUNNER.P23AcquisitionError, match="absolute"):
        RUNNER._path_roots([*values[:-1], "native=relative"])
    with pytest.raises(RUNNER.P23AcquisitionError, match="exactly"):
        RUNNER._path_roots(values[:-1])
