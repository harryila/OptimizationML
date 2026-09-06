from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/ingest_p24_native_outcome_for_p25.py"


def _module():
    specification = importlib.util.spec_from_file_location("p25_p24_ingestion_test", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _historical_fixture(module: object) -> dict[str, object]:
    locked = {
        "deterministic_algorithms": True,
        "deterministic_debug_mode": "error",
        "cpu_threads": 1,
        "interop_threads": 1,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "float32_matmul_precision": "highest",
        "tf32": False,
        "fp16_reduced_precision_reduction": False,
        "bf16_reduced_precision_reduction": False,
    }
    live = {
        "deterministic_algorithms": False,
        "deterministic_debug_mode": 0,
        "cpu_threads": 1,
        "interop_threads": 30,
        "cudnn_benchmark": False,
        "cudnn_deterministic": False,
        "float32_matmul_precision": "highest",
        "tf32": False,
        "fp16_reduced_precision_reduction": True,
        "bf16_reduced_precision_reduction": True,
    }
    generated = {
        "path": "/tmp/generated/_remote_module_non_scriptable.py",
        "byte_count": module.P24_GENERATED_MODULE_BYTE_COUNT,
        "sha256": module.P24_GENERATED_MODULE_SHA256,
        "effective_mount": {
            "mount_id": 10,
            "mount_point": "/tmp",
            "filesystem_type": "tmpfs",
            "mount_options": ["rw", "noexec"],
            "super_options": ["rw"],
            "writable": True,
        },
    }
    records = {"_remote_module_non_scriptable": generated}
    artifacts = {
        name: {
            "path": f"/workspace/OptimizationML/{record['path']}",
            "byte_count": record["byte_count"],
            "sha256": record["sha256"],
        }
        for name, record in module._P24_ARTIFACTS.items()
    }
    # The native record uses direct, not repository-relative, remote paths.
    artifacts["contract"]["path"] = (
        "/workspace/OptimizationML/experiments/training/p24_cuda_executable_origin_contract.json"
    )
    artifacts["diagnostic_source"]["path"] = (
        "/workspace/OptimizationML/experiments/training/run_p24_executable_origin_diagnostic.py"
    )
    artifacts["host_attestation"]["path"] = (
        "/workspace/OptimizationML/experiments/training/p23_host_attestation.json"
    )
    artifacts["runtime_lock"]["path"] = (
        "/workspace/OptimizationML/experiments/training/p23_cuda_runtime_lock.json"
    )
    payload = {
        "artifacts": artifacts,
        "checks": copy.deepcopy(module._EXPECTED_CHECKS),
        "claim_boundary": "minimal diagnostic only",
        "contract_binding": {
            "expected_sha256": artifacts["contract"]["sha256"],
            "content": {"schema_version": "fixture"},
            "source_records": {},
        },
        "file_backed_module_closure": {
            "count": 1,
            "canonical_sha256": module._canonical_sha256(records),
            "records": records,
        },
        "host_attestation_binding": {
            "expected_sha256": artifacts["host_attestation"]["sha256"],
            "content": {"container": {"tmpfs_contract": {"mounts": {"/tmp": "rw,noexec"}}}},
        },
        "live_runtime": live,
        "mode": "baseline",
        "passes": False,
        "pinned_sources": {},
        "process": {},
        "repository": {},
        "runtime_lock_binding": {
            "expected_sha256": artifacts["runtime_lock"]["sha256"],
            "content": {
                "determinism": locked,
                "container": {"tmpfs_contract": {"mounts": {"/tmp": "rw,noexec"}}},
            },
        },
        "schema_version": module.P24_NATIVE_SCHEMA,
        "scope": "minimal diagnostic",
        "seed": None,
        "seed_semantics": "not applicable: the diagnostic performs no random operation",
        "status": "fails",
        "trigger": {
            "generated_module": generated,
            "new_modules_after_optimizer": ["_remote_module_non_scriptable"],
            "new_modules_after_parameter": [],
            "new_modules_after_torch_import": [],
        },
    }
    assert set(payload) == module._P24_SUCCESS_FIELDS
    return payload


def _roots(tmp_path: Path) -> dict[str, Path]:
    module = _module()
    roots = {name: tmp_path / name for name in module.SANITIZER.REQUIRED_LOGICAL_ROOTS}
    for path in roots.values():
        path.mkdir()
    return roots


def test_historical_constants_pin_the_retained_external_artifact() -> None:
    module = _module()
    assert module.P24_NATIVE_SHA256 == (
        "ce44c53c9f274cef3eda7b3adea755ce6b1313773c8dfe00fc9f194b74fb2bd2"
    )
    assert module.P24_NATIVE_BYTE_COUNT == 3_760_729
    assert module.P24_ORIGINAL_SANITIZER_EXIT == 2
    assert module.P24_ORIGINAL_SANITIZER_STDERR_SHA256 == (
        "3939a4478d2bd761c810732b01d9f3af9dd31b09c31efe376eece88ac279034f"
    )


def test_fixture_reconstructs_all_checks_six_mismatches_module_and_tmp_keys() -> None:
    module = _module()
    validation = module.validate_historical_payload(_historical_fixture(module), repository=None)
    assert validation["check_count"] == 36
    assert validation["true_check_count"] == 35
    assert validation["false_checks"] == ["torch_determinism_matches_lock"]
    assert validation["reconstructed_passes"] is False
    assert validation["determinism"]["mismatch_count"] == 6
    assert {item["field"] for item in validation["determinism"]["mismatches"]} == set(
        module._EXPECTED_MISMATCH_VALUES
    )
    assert validation["generated_module"] == {
        "sha256": module.P24_GENERATED_MODULE_SHA256,
        "byte_count": 2355,
        "path_class": "temporary-root descendant",
        "effective_mount_point": "temporary-root",
        "filesystem_type": "tmpfs",
        "writable": True,
        "loaded_only_after_optimizer": True,
    }
    assert validation["absolute_tmp_mapping_key_count"] == 2
    assert tuple(validation["absolute_tmp_mapping_key_json_pointers"]) == (
        "json-pointer:host_attestation_binding/content/container/tmpfs_contract/mounts/~1tmp",
        "json-pointer:runtime_lock_binding/content/container/tmpfs_contract/mounts/~1tmp",
    )


def test_ingestion_preserves_p24_failure_and_distinguishes_p25_success(
    tmp_path: Path,
) -> None:
    module = _module()
    roots = _roots(tmp_path)
    payload = _historical_fixture(module)
    raw = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    native = roots["native"] / "p24.native.json"
    output = roots["native"] / "p25.wrapper.json"
    native.write_bytes(raw)
    result = module.ingest_historical(
        native=native,
        output=output,
        path_roots=roots,
        expected_native_sha256=hashlib.sha256(raw).hexdigest(),
        expected_native_byte_count=len(raw),
        repository=None,
    )
    assert result["original_p24_terminal_outcome"]["status"] == "fails"
    assert result["original_p24_terminal_outcome"]["passes"] is False
    assert result["original_p24_sanitization"]["exit_code"] == 2
    assert result["original_p24_sanitization"]["output_created"] is False
    assert result["p25_historical_validation"]["passes"] is True
    redaction = result["p25_corrected_redaction"]
    assert redaction["passes"] is True
    assert redaction["key_replacement_count"] == 2
    assert redaction["total_replacement_count"] == (
        redaction["key_replacement_count"] + redaction["value_replacement_count"]
    )
    rendered = json.dumps(result, sort_keys=True)
    assert '"/tmp"' not in rendered
    assert "/workspace/OptimizationML" not in rendered
    module.SANITIZER._assert_no_absolute_paths(result)
    assert json.loads(output.read_text(encoding="utf-8")) == result
    with pytest.raises(module.SANITIZER.SanitizationError, match="fresh path"):
        module.ingest_historical(
            native=native,
            output=output,
            path_roots=roots,
            expected_native_sha256=hashlib.sha256(raw).hexdigest(),
            expected_native_byte_count=len(raw),
            repository=None,
        )


def test_ingestion_rejects_byte_binding_and_semantic_tampering(tmp_path: Path) -> None:
    module = _module()
    roots = _roots(tmp_path)
    payload = _historical_fixture(module)
    raw = json.dumps(payload, sort_keys=True).encode()
    native = roots["native"] / "p24.native.json"
    native.write_bytes(raw)
    with pytest.raises(module.IngestionError, match="exact SHA/size"):
        module.ingest_historical(
            native=native,
            output=roots["native"] / "bad-binding.json",
            path_roots=roots,
            expected_native_sha256="0" * 64,
            expected_native_byte_count=len(raw),
            repository=None,
        )

    tampered = copy.deepcopy(payload)
    tampered["checks"]["generated_module_bytes_exact"] = False
    with pytest.raises(module.IngestionError, match="36-check reconstruction"):
        module.validate_historical_payload(tampered, repository=None)


def test_actual_external_artifact_is_optional_but_exact_when_present() -> None:
    """This is not a dependency: it becomes an extra local authentication check."""

    module = _module()
    native = Path("/private/tmp/p24-baseline-executable-origin.native.json")
    if not native.is_file():
        return
    raw = native.read_bytes()
    assert len(raw) == module.P24_NATIVE_BYTE_COUNT
    assert hashlib.sha256(raw).hexdigest() == module.P24_NATIVE_SHA256
    payload = module.SANITIZER.load_json_strict(raw)
    validation = module.validate_historical_payload(payload)
    assert validation["false_checks"] == ["torch_determinism_matches_lock"]
