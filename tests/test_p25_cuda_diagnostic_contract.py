from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import TypeAlias

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "experiments/training/p25_cuda_diagnostic_contract.json"
RECONSTRUCTOR = ROOT / "scripts/reconstruct_p25_cuda_diagnostic_contract.py"
FINALIZER = ROOT / "scripts/finalize_p25_cuda_diagnostic_contract.py"
SANITIZER = ROOT / "scripts/sanitize_p25_executable_origin_diagnostic.py"
HOST_ORCHESTRATOR = ROOT / "scripts/run_p25_cuda_attempt.sh"
P24_STATIC_COMMIT = "9003130ca31085f555ff144216a599442cbdf3ca"
FROZEN_P24_EXECUTION_PATHS = (
    "experiments/training/p24_cuda_executable_origin_contract.json",
    "experiments/training/run_p24_executable_origin_diagnostic.py",
    "scripts/sanitize_p24_executable_origin_diagnostic.py",
    "experiments/training/prepare_p24_immutable_remote_module.py",
    "experiments/training/p24_runtime.Dockerfile",
    "experiments/training/run_p23_deterministic_cuda_shadow_trace.py",
    "src/passive_muon/p21_shadow_trace.py",
)
PathPart: TypeAlias = str | int


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _leaf_paths(value: object, prefix: tuple[PathPart, ...] = ()) -> Iterator[tuple[PathPart, ...]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _leaf_paths(item, (*prefix, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _leaf_paths(item, (*prefix, index))
    else:
        yield prefix


def _at_path(payload: object, path: tuple[PathPart, ...]) -> object:
    current = payload
    for part in path:
        current = current[part]  # type: ignore[index]
    return current


def _replace_at_path(payload: object, path: tuple[PathPart, ...], value: object) -> None:
    parent = _at_path(payload, path[:-1])
    parent[path[-1]] = value  # type: ignore[index]


def _mutate(value: object) -> object:
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, str):
        return f"{value}__mutated"
    if value is None:
        return "unexpected_non_null"
    raise AssertionError(type(value).__name__)


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_p25_contract_reconstructs_only_a_pre_execution_claim() -> None:
    result = _load_module("p25_reconstruct", RECONSTRUCTOR).reconstruct(CONTRACT)
    assert result["internally_consistent"] is True
    assert len(result["checks"]) == 23
    assert all(result["checks"].values())
    assert "Pre-execution" in result["scope"]
    assert "no P25 diagnostic" in result["scope"]
    assert "certified" not in result


def test_p25_freezes_parent_failure_without_a_favorable_rerun() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    parent = contract["parent_p24_terminal_facts"]

    assert parent["p24_rerun_allowed"] is False
    assert parent["old_p23_container_diagnostic_may_be_reexecuted"] is False
    assert parent["native_artifact"]["sha256"] == (
        "ce44c53c9f274cef3eda7b3adea755ce6b1313773c8dfe00fc9f194b74fb2bd2"
    )
    assert parent["native_artifact"]["byte_count"] == 3_760_729
    assert parent["checks"] == {
        "total": 36,
        "true": 35,
        "false": 1,
        "sole_false_check": "torch_determinism_matches_lock",
    }
    assert len(parent["torch_determinism_mismatches"]) == 6
    assert parent["sanitizer_failure"]["sanitized_artifact_created"] is False


def test_p25_is_one_new_diagnostic_then_unchanged_exact_acquisition() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    order = contract["ordered_execution"]
    failure = contract["failure_policy"]
    gates = contract["frozen_p23_gates"]

    assert contract["corrected_diagnostic_contract"]["attempt_count"] == 1
    assert failure["p25_corrected_diagnostic_attempts_allowed"] == 1
    assert failure["no_favorable_reruns"] is True
    assert order.index("run exactly one corrected P25 remediated executable-origin diagnostic") < (
        order.index(
            "trace_off_a using the unchanged P23 runner with a fresh required failure-output path"
        )
    )
    assert order.index("verify_repeatability using exact equality only") < order.index(
        "trace_on only if exact repeatability reports zero mismatches"
    )
    assert order.index("verify_noninterference using exact equality only") < order.index(
        "aggregate_fidelity only if exact noninterference reports zero mismatches"
    )
    assert gates["allclose_allowed"] is False
    assert gates["numeric_tolerance"] is None
    assert gates["repeatability_required_mismatches"] == 0
    assert gates["noninterference_required_mismatches"] == 0
    assert gates["trace_on_observations"] == 1152


def test_p25_replacement_runtime_is_fresh_exact_and_no_overwrite() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    runtime = contract["replacement_runtime"]
    mounts = runtime["mount_contract"]
    failure = contract["failure_policy"]

    assert runtime["attempt_root_must_be_fresh_and_empty"] is True
    assert runtime["historical_p23_and_p24_evidence_read_only"] is True
    assert runtime["new_oci_digest_required"] is True
    assert runtime["new_p25_named_runtime_lock_required"] is True
    assert runtime["new_p25_named_host_attestation_required"] is True
    assert runtime["root_filesystem_read_only"] is True
    assert runtime["network_mode"] == "none"
    assert runtime["gpu_contract"] == {
        "uuid": "GPU-ee4c9bf9-42f5-7cdd-66d2-f638c2db549d",
        "name": "NVIDIA A100-SXM4-40GB",
        "compute_capability": [8, 0],
        "mig_enabled": False,
        "device_count": 1,
        "bf16_required": True,
        "cpu_mps_mig_or_ordinal_substitution_allowed": False,
    }
    assert mounts["bind_mount_count"] == 10
    assert mounts["tmpfs_count"] == 1
    assert runtime["running_inspection_and_mountinfo_updated_in_place_after_launch"] is True
    assert runtime["rename_replacement_of_bound_host_inspection_files_forbidden"] is True
    assert runtime["diagnostic_parent_must_equal_preregistration_head"] is True
    assert runtime["diagnostic_exact_name_status"] == [
        "A\texperiments/training/p25_cuda_runtime_lock.json",
        "A\texperiments/training/p25_host_attestation.json",
    ]
    assert runtime["bridge_parent_must_equal_diagnostic_head"] is True
    assert runtime["bridge_exact_name_status"] == (
        "A\tresults/summaries/p25_executable_origin_diagnostic.sanitized.json"
    )
    assert runtime["bridge_reconstructs_contract_and_rehashes_all_frozen_sources_before_gradients"]
    assert runtime[
        "bridge_rehashes_retained_native_against_strictly_parsed_wrapper_before_gradients"
    ]
    assert failure["all_native_and_sanitized_outputs_must_be_fresh_external_and_nonaliasing"]
    assert failure["overwrite_or_truncation_of_existing_evidence"] == "forbidden"


def test_p25_configures_determinism_before_trigger_without_cuda_init() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    diagnostic = contract["corrected_diagnostic_contract"]
    trigger = diagnostic["ordered_trigger"]

    assert trigger.index("import Torch") < trigger.index(
        "configure the complete locked deterministic Torch state"
    )
    assert trigger.index("configure the complete locked deterministic Torch state") < trigger.index(
        "verify the complete locked deterministic Torch state exactly"
    )
    assert trigger.index(
        "verify the complete locked deterministic Torch state exactly"
    ) < trigger.index("construct one CPU FP32 Parameter")
    assert trigger.index("construct one CPU FP32 Parameter") < trigger.index(
        "construct torch.optim.SGD"
    )
    assert diagnostic["cuda_uninitialized_before_and_after_determinism_configuration"] is True


def test_p25_redacts_keys_and_values_and_rejects_collisions() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    sanitizer = contract["corrected_sanitizer_contract"]
    historical = contract["historical_p24_ingestion"]

    assert sanitizer["mapping_keys_and_values_use_one_exact_root_transform"] is True
    assert sanitizer["post_redaction_mapping_key_collision_rejection"] is True
    assert sanitizer["strict_duplicate_json_key_rejection"] is True
    assert sanitizer["total_replacement_count_required"] is True
    assert sanitizer["complete_post_redaction_absolute_path_scan_required"] is True
    assert sanitizer["passing_manifest_top_level_schema_closed"] is True
    assert sanitizer["passing_manifest_exact_check_inventory_count"] == 40
    assert sanitizer["passing_manifest_requires_every_exact_check_true"] is True
    assert sanitizer["independent_contract_source_and_artifact_binding_reconstruction"] is True
    assert (
        sanitizer["independent_runtime_lock_host_attestation_and_backlink_reconstruction"] is True
    )
    assert (
        sanitizer["independent_locked_python_torch_and_live_cuda_gpu_state_reconstruction"] is True
    )
    assert sanitizer["independent_pinned_torch_source_reconstruction"] is True
    assert sanitizer["independent_process_event_and_module_snapshot_reconstruction"] is True
    assert sanitizer["independent_immutable_generated_module_closure_reconstruction"] is True
    assert sanitizer["failed_or_error_manifest_cannot_authorize_acquisition"] is True
    assert historical["exact_two_absolute_tmp_key_locations_required"] is True
    assert historical["output_retains_p24_failed_disposition"] is True
    assert historical["output_does_not_authorize_image_build_or_acquisition"] is True
    assert historical["attempt_count"] == 1
    assert historical["offline_container_image"] == (
        "localhost:5000/p23-runtime@"
        "sha256:44ef23717780b1cbf112b183e7988b1319ddfed6b1d224efaa33e1e6d96de4c1"
    )
    assert historical["offline_container_network_mode"] == "none"
    assert historical["offline_container_root_filesystem_read_only"] is True
    assert historical["offline_container_gpu_devices"] == 0
    assert historical["fresh_output_root_pattern"] == (
        "/secure/p25/historical-YYYYMMDD-NN/evidence"
    )
    assert historical["historical_p23_p24_evidence_directory_mounted_writable"] is False
    assert historical["retained_p24_native_overlay_mounted_read_only"] is True
    assert historical["preregistration_parent_must_equal_source_freeze_bootstrap"] is True
    assert historical["preregistration_exact_name_status"] == (
        "A\tresults/summaries/p25_historical_p24_native_outcome.json"
    )
    assert historical["historical_wrapper_must_be_absent_at_bootstrap"] is True


def test_p25_contract_is_accepted_by_the_credential_and_nonfinite_scan() -> None:
    sanitizer = _load_module("p25_contract_sanitizer_compatibility", SANITIZER)
    contract = sanitizer.load_json_strict(CONTRACT.read_bytes())
    sanitizer.reject_credentials_and_nonfinite(contract)


def test_p25_acquisition_authenticates_the_exact_bridge_before_gradients() -> None:
    script = HOST_ORCHESTRATOR.read_text(encoding="utf-8")

    assert "require_commit P25_DIAGNOSTIC_HEAD" in script
    assert "diagnostic commit must be one direct child of the preregistration commit" in script
    assert "diagnostic commit must add only the reviewed P25 lock and attestation" in script
    assert 'rev-parse "$P25_BRIDGE_HEAD^"' in script
    assert "bridge delta must add only the reviewed sanitized diagnostic wrapper" in script
    assert "p25-bridge-contract-reconstruction" in script
    assert "retained native SHA-256 differs from wrapper" in script
    assert "sanitizer.validate_native_semantics(native_payload)" in script
    assert "P25 frozen source differs" in script
    assert script.index("p25-diagnostic-bridge-verification") < script.index("p23-trace-off-a")


def test_p25_finalizer_is_noop_after_freeze_and_refuses_source_drift(tmp_path: Path) -> None:
    finalizer = _load_module("p25_finalize", FINALIZER)
    frozen = json.loads(CONTRACT.read_text(encoding="utf-8"))
    temporary = tmp_path / "contract.json"
    _write(temporary, frozen)

    result = finalizer.finalize(temporary)
    assert result["changed"] is False
    assert result["ready_for_reviewed_static_commit"] is True

    drifted = copy.deepcopy(frozen)
    drifted["execution_sources"]["corrected_executable_origin_diagnostic"]["sha256"] = "0" * 64
    _write(temporary, drifted)
    with pytest.raises(finalizer.FinalizationError, match="drifted"):
        finalizer.finalize(temporary)


def test_p25_finalizer_fills_only_declared_pending_records(tmp_path: Path) -> None:
    finalizer = _load_module("p25_finalize_pending", FINALIZER)
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    allowed = payload["source_hash_finalization"]["allowed_records"]
    for name in allowed:
        payload["execution_sources"][name]["sha256"] = "PENDING_SHA256"
    temporary = tmp_path / "contract.json"
    _write(temporary, payload)

    result = finalizer.finalize(temporary)
    finalized = json.loads(temporary.read_text(encoding="utf-8"))
    assert result["changed"] is True
    assert result["filled_records"] == sorted(allowed)
    assert result["ready_for_reviewed_static_commit"] is True
    assert all(len(finalized["execution_sources"][name]["sha256"]) == 64 for name in allowed)


def test_p25_preserves_every_frozen_p24_execution_byte() -> None:
    result = subprocess.run(
        ["git", "diff", "--exit-code", P24_STATIC_COMMIT, "--", *FROZEN_P24_EXECUTION_PATHS],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_every_contract_leaf_mutation_fails_reconstruction(tmp_path: Path) -> None:
    module = _load_module("p25_reconstruct_mutations", RECONSTRUCTOR)
    canonical = json.loads(CONTRACT.read_text(encoding="utf-8"))
    mutated_path = tmp_path / "mutated.json"

    for leaf_path in _leaf_paths(canonical):
        mutated = copy.deepcopy(canonical)
        _replace_at_path(mutated, leaf_path, _mutate(_at_path(mutated, leaf_path)))
        _write(mutated_path, mutated)
        assert module.reconstruct(mutated_path)["internally_consistent"] is False, leaf_path


def test_duplicate_contract_key_fails_closed(tmp_path: Path) -> None:
    module = _load_module("p25_reconstruct_duplicate", RECONSTRUCTOR)
    text = CONTRACT.read_text(encoding="utf-8")
    duplicate = text.replace(
        '  "status": "frozen_pre_remediation_build_and_pre_acquisition",',
        '  "status": "frozen_pre_remediation_build_and_pre_acquisition",\n'
        '  "status": "frozen_pre_remediation_build_and_pre_acquisition",',
        1,
    )
    path = tmp_path / "duplicate.json"
    path.write_text(duplicate, encoding="utf-8")
    assert module.reconstruct(path)["internally_consistent"] is False
