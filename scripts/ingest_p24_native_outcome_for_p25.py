#!/usr/bin/env python3
"""Authenticate, reconstruct, and safely wrap the retained P24 native result.

P24's first native baseline diagnostic is immutable evidence.  Its original
sanitizer failed because two absolute ``/tmp`` paths occurred as mapping keys.
This P25 ingester does not alter or replace those bytes: it authenticates the
exact external artifact, reconstructs its terminal result, and applies P25's
correct key-and-value redaction into a new no-overwrite wrapper.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Final

ROOT = Path(__file__).resolve().parents[1]
SANITIZER_PATH = ROOT / "scripts/sanitize_p25_executable_origin_diagnostic.py"

SCHEMA: Final = "passive-muon-p25-historical-p24-native-outcome-v1"
P24_NATIVE_SCHEMA: Final = "passive-muon-p24-executable-origin-diagnostic-v1"
P24_NATIVE_SHA256: Final = "ce44c53c9f274cef3eda7b3adea755ce6b1313773c8dfe00fc9f194b74fb2bd2"
P24_NATIVE_BYTE_COUNT: Final = 3_760_729
P24_GENERATED_MODULE_SHA256: Final = (
    "8205b16956fb264841ecd8644784a0d157f87df79b17c16825dc1163433ce5d8"
)
P24_GENERATED_MODULE_BYTE_COUNT: Final = 2_355
P24_ORIGINAL_SANITIZER_EXIT: Final = 2
P24_ORIGINAL_SANITIZER_STDOUT_SHA256: Final = hashlib.sha256(b"").hexdigest()
P24_ORIGINAL_SANITIZER_STDERR: Final = (
    "P24 diagnostic sanitization blocked: redacted diagnostic retains a declared native path\n"
)
P24_ORIGINAL_SANITIZER_STDERR_SHA256: Final = hashlib.sha256(
    P24_ORIGINAL_SANITIZER_STDERR.encode()
).hexdigest()

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_P24_SUCCESS_FIELDS: Final = {
    "artifacts",
    "checks",
    "claim_boundary",
    "contract_binding",
    "file_backed_module_closure",
    "host_attestation_binding",
    "live_runtime",
    "mode",
    "passes",
    "pinned_sources",
    "process",
    "repository",
    "runtime_lock_binding",
    "schema_version",
    "scope",
    "seed",
    "seed_semantics",
    "status",
    "trigger",
}
_P24_CHECK_NAMES: Final = {
    "bound_artifacts_unchanged_after_trigger",
    "compiled_cuda_matches_lock",
    "container_hostname_matches_lock",
    "contract_schema_and_status_exact",
    "contract_sha256_matches_expected",
    "contract_source_records_exact",
    "contract_sources_unchanged_after_trigger",
    "cuda_runtime_matches_lock",
    "cudnn_matches_lock",
    "generated_module_bytes_exact",
    "generated_module_loaded_only_after_optimizer",
    "host_attestation_schema_and_status_exact",
    "host_attestation_sha256_matches_expected",
    "instantiator_source_exact_for_mode",
    "live_mountinfo_matches_lock",
    "mode_declared",
    "mode_origin_matches",
    "process_environment_matches_lock_exactly",
    "python_runtime_matches_lock",
    "remote_module_source_exact",
    "repository_clean",
    "repository_head_matches_expected",
    "repository_tree_matches_expected",
    "repository_unchanged_after_trigger",
    "rootfs_declared_and_observed_read_only",
    "runtime_lock_and_attestation_container_exact",
    "runtime_lock_and_attestation_gpu_exact",
    "runtime_lock_host_attestation_backlink_exact",
    "runtime_lock_p23_addendum_matches_contract_and_source",
    "runtime_lock_schema_and_status_exact",
    "runtime_lock_sha256_matches_expected",
    "single_live_gpu_matches_lock",
    "template_source_exact",
    "torch_determinism_matches_lock",
    "torch_git_version_matches_lock",
    "torch_version_matches_lock",
}
_EXPECTED_CHECKS: Final = {
    name: name != "torch_determinism_matches_lock" for name in _P24_CHECK_NAMES
}
_DETERMINISM_FIELDS: Final = (
    "deterministic_algorithms",
    "deterministic_debug_mode",
    "cpu_threads",
    "interop_threads",
    "cudnn_benchmark",
    "cudnn_deterministic",
    "float32_matmul_precision",
    "tf32",
    "fp16_reduced_precision_reduction",
    "bf16_reduced_precision_reduction",
)
_EXPECTED_MISMATCH_VALUES: Final = {
    "deterministic_algorithms": (False, True),
    "deterministic_debug_mode": (0, 2),
    "interop_threads": (30, 1),
    "cudnn_deterministic": (False, True),
    "fp16_reduced_precision_reduction": (True, False),
    "bf16_reduced_precision_reduction": (True, False),
}
_EXPECTED_MATCH_VALUES: Final = {
    "cpu_threads": (1, 1),
    "cudnn_benchmark": (False, False),
    "float32_matmul_precision": ("highest", "highest"),
    "tf32": (False, False),
}
_P24_ARTIFACTS: Final = {
    "contract": {
        "path": "experiments/training/p24_cuda_executable_origin_contract.json",
        "byte_count": 14_843,
        "sha256": "bdb2d3aad7d70d0ed5c6dddcd03642c9384d8b67d6f5ca507987d28e122dd446",
    },
    "diagnostic_source": {
        "path": "experiments/training/run_p24_executable_origin_diagnostic.py",
        "byte_count": 39_433,
        "sha256": "5e3fe58a394ad98ab8f2d03f5992cabeb417110b7d2b05e86fbbf61b5f241afb",
    },
    "host_attestation": {
        "path": "experiments/training/p23_host_attestation.json",
        "byte_count": 4_124,
        "sha256": "3f67ce327ab13b1a177be5ae6a767cc49a624e7c7b432d5e63fbb933e8d1b3a1",
    },
    "runtime_lock": {
        "path": "experiments/training/p23_cuda_runtime_lock.json",
        "byte_count": 6_599,
        "sha256": "54de9d5d2e8d497e06c45e7672988ffa2948239c78a3e5164dc7d266ddae6a2d",
    },
}
_TMP_KEY_POINTERS: Final = (
    "/host_attestation_binding/content/container/tmpfs_contract/mounts/~1tmp",
    "/runtime_lock_binding/content/container/tmpfs_contract/mounts/~1tmp",
)


class IngestionError(RuntimeError):
    """Raised when the historical artifact cannot be admitted into P25."""


def _load_sanitizer():
    specification = importlib.util.spec_from_file_location("p25_origin_sanitizer", SANITIZER_PATH)
    if specification is None or specification.loader is None:
        raise IngestionError("cannot load the P25 sanitizer")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


SANITIZER = _load_sanitizer()


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise IngestionError(f"{label} must be a mapping")
    return value


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _json_pointer_escape(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _tmp_mapping_key_pointers(value: object, pointer: str = "") -> list[str]:
    result: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise IngestionError("historical JSON mapping key is not a string")
            child = f"{pointer}/{_json_pointer_escape(key)}"
            if "/tmp" in key:
                result.append(child)
            result.extend(_tmp_mapping_key_pointers(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            result.extend(_tmp_mapping_key_pointers(item, f"{pointer}/{index}"))
    return result


def _expected_determinism_value(field: str, locked: Mapping[str, object]) -> object:
    if field == "deterministic_debug_mode":
        if locked.get(field) != "error":
            raise IngestionError("P24 locked deterministic debug mode differs")
        return 2
    return locked.get(field)


def _validate_repository_artifacts(payload: Mapping[str, object], repository: Path | None) -> None:
    artifacts = _mapping(payload.get("artifacts"), "P24 artifact inventory")
    if set(artifacts) != set(_P24_ARTIFACTS):
        raise IngestionError("P24 artifact inventory differs")
    for name, expected in _P24_ARTIFACTS.items():
        record = _mapping(artifacts.get(name), f"P24 artifact {name}")
        if (
            record.get("byte_count") != expected["byte_count"]
            or record.get("sha256") != expected["sha256"]
        ):
            raise IngestionError(f"P24 artifact binding differs: {name}")
        if repository is None:
            continue
        local_path = repository / str(expected["path"])
        raw = local_path.read_bytes()
        if (
            len(raw) != expected["byte_count"]
            or hashlib.sha256(raw).hexdigest() != expected["sha256"]
        ):
            raise IngestionError(f"local frozen P24 artifact differs: {name}")


def validate_historical_payload(
    payload: object,
    *,
    repository: Path | None = ROOT,
) -> dict[str, object]:
    """Reconstruct the exact semantic facts used by the P24 terminal result."""

    native = _mapping(payload, "P24 native diagnostic")
    if set(native) != _P24_SUCCESS_FIELDS:
        raise IngestionError("P24 top-level field set differs")
    if native.get("schema_version") != P24_NATIVE_SCHEMA:
        raise IngestionError("P24 native schema differs")
    if native.get("mode") != "baseline" or native.get("status") != "fails":
        raise IngestionError("P24 mode or terminal status differs")
    if native.get("passes") is not False:
        raise IngestionError("P24 aggregate pass flag differs")
    if native.get("seed") is not None or native.get("seed_semantics") != (
        "not applicable: the diagnostic performs no random operation"
    ):
        raise IngestionError("P24 seed semantics differ")

    checks = _mapping(native.get("checks"), "P24 check map")
    if set(checks) != _P24_CHECK_NAMES or dict(checks) != _EXPECTED_CHECKS:
        raise IngestionError("P24 36-check reconstruction differs")
    reconstructed_passes = all(value is True for value in checks.values())
    if reconstructed_passes is not False or (native.get("status") == "passes") is not False:
        raise IngestionError("P24 check aggregate and terminal status disagree")

    runtime = _mapping(native.get("runtime_lock_binding"), "runtime-lock binding")
    runtime_content = _mapping(runtime.get("content"), "runtime-lock content")
    locked = _mapping(runtime_content.get("determinism"), "locked determinism")
    live = _mapping(native.get("live_runtime"), "live runtime")
    mismatch_records: list[dict[str, object]] = []
    match_records: list[dict[str, object]] = []
    for field in _DETERMINISM_FIELDS:
        expected = _expected_determinism_value(field, locked)
        observed = live.get(field)
        record = {"field": field, "observed": observed, "locked": expected}
        (match_records if observed == expected else mismatch_records).append(record)
    mismatch_values = {
        str(record["field"]): (record["observed"], record["locked"]) for record in mismatch_records
    }
    match_values = {
        str(record["field"]): (record["observed"], record["locked"]) for record in match_records
    }
    if mismatch_values != _EXPECTED_MISMATCH_VALUES or match_values != _EXPECTED_MATCH_VALUES:
        raise IngestionError("P24 exact Torch runtime-state comparison differs")
    if checks["torch_determinism_matches_lock"] is not (not mismatch_records):
        raise IngestionError("P24 determinism check does not follow the reconstructed mismatches")

    trigger = _mapping(native.get("trigger"), "P24 trigger")
    generated = _mapping(trigger.get("generated_module"), "P24 generated module")
    mount = _mapping(generated.get("effective_mount"), "P24 generated-module mount")
    generated_path = generated.get("path")
    if (
        generated.get("sha256") != P24_GENERATED_MODULE_SHA256
        or generated.get("byte_count") != P24_GENERATED_MODULE_BYTE_COUNT
        or not isinstance(generated_path, str)
        or not generated_path.startswith("/tmp/")
        or mount.get("mount_point") != "/tmp"
        or mount.get("filesystem_type") != "tmpfs"
        or mount.get("writable") is not True
    ):
        raise IngestionError("P24 generated-module reproduction differs")
    closure = _mapping(native.get("file_backed_module_closure"), "P24 module closure")
    records = _mapping(closure.get("records"), "P24 module-closure records")
    if (
        closure.get("count") != len(records)
        or closure.get("canonical_sha256") != _canonical_sha256(records)
        or records.get("_remote_module_non_scriptable") != generated
    ):
        raise IngestionError("P24 generated module and final closure differ")
    if "_remote_module_non_scriptable" not in trigger.get("new_modules_after_optimizer", []):
        raise IngestionError("P24 generated module was not observed after optimizer construction")
    if "_remote_module_non_scriptable" in trigger.get("new_modules_after_torch_import", []):
        raise IngestionError("P24 generated module was already loaded after Torch import")
    if "_remote_module_non_scriptable" in trigger.get("new_modules_after_parameter", []):
        raise IngestionError("P24 generated module was already loaded after Parameter construction")

    tmp_key_pointers = sorted(_tmp_mapping_key_pointers(native))
    if tuple(tmp_key_pointers) != _TMP_KEY_POINTERS:
        raise IngestionError("P24 absolute-/tmp mapping-key inventory differs")

    resolved_repository = repository.resolve(strict=True) if repository is not None else None
    _validate_repository_artifacts(native, resolved_repository)

    artifacts = _mapping(native.get("artifacts"), "P24 artifact inventory")
    for name, binding_name in (
        ("contract", "contract_binding"),
        ("runtime_lock", "runtime_lock_binding"),
        ("host_attestation", "host_attestation_binding"),
    ):
        binding = _mapping(native.get(binding_name), f"P24 {binding_name}")
        record = _mapping(artifacts.get(name), f"P24 artifact {name}")
        if binding.get("expected_sha256") != record.get("sha256"):
            raise IngestionError(f"P24 {binding_name} expected digest differs")

    return {
        "check_count": len(checks),
        "true_check_count": sum(value is True for value in checks.values()),
        "false_checks": sorted(name for name, value in checks.items() if value is False),
        "reconstructed_checks": dict(sorted(checks.items())),
        "reconstructed_passes": reconstructed_passes,
        "determinism": {
            "compared_field_count": len(_DETERMINISM_FIELDS),
            "match_count": len(match_records),
            "mismatch_count": len(mismatch_records),
            "matches": match_records,
            "mismatches": mismatch_records,
        },
        "generated_module": {
            "sha256": generated["sha256"],
            "byte_count": generated["byte_count"],
            "path_class": "temporary-root descendant",
            "effective_mount_point": "temporary-root",
            "filesystem_type": mount["filesystem_type"],
            "writable": mount["writable"],
            "loaded_only_after_optimizer": True,
        },
        "absolute_tmp_mapping_key_count": len(tmp_key_pointers),
        "absolute_tmp_mapping_key_json_pointers": [
            f"json-pointer:{pointer.removeprefix('/')}" for pointer in tmp_key_pointers
        ],
    }


def ingest_historical(
    *,
    native: Path,
    output: Path,
    path_roots: Mapping[str, str | Path],
    expected_native_sha256: str = P24_NATIVE_SHA256,
    expected_native_byte_count: int = P24_NATIVE_BYTE_COUNT,
    repository: Path | None = ROOT,
) -> dict[str, object]:
    """Authenticate and publish a corrected, semantically explicit wrapper."""

    if _SHA256.fullmatch(expected_native_sha256) is None or expected_native_byte_count <= 0:
        raise IngestionError("expected historical artifact binding is invalid")
    roots = SANITIZER.validate_path_roots(path_roots)
    native_resolved = native.resolve(strict=True)
    output_resolved = output.parent.resolve(strict=True) / output.name
    if not SANITIZER._is_within(native_resolved, roots["native"]) or not SANITIZER._is_within(
        output_resolved, roots["native"]
    ):
        raise IngestionError("historical native and wrapper must remain in the evidence root")
    if native_resolved == output_resolved:
        raise IngestionError("historical native and wrapper paths must be distinct")
    raw, observed = SANITIZER._stable_read(native)
    if observed != {
        "byte_count": expected_native_byte_count,
        "sha256": expected_native_sha256,
    }:
        raise IngestionError("retained P24 native artifact fails its exact SHA/size binding")
    payload = SANITIZER.load_json_strict(raw)
    validation = validate_historical_payload(payload, repository=repository)
    assert isinstance(payload, Mapping)
    redacted, counts = SANITIZER.redact_payload(payload, roots)
    result = {
        "schema_version": SCHEMA,
        "source_artifact": {
            "sha256": expected_native_sha256,
            "byte_count": expected_native_byte_count,
            "schema_version": P24_NATIVE_SCHEMA,
            "retention": "external native bytes retained unchanged",
        },
        "original_p24_terminal_outcome": {
            "mode": "baseline",
            "status": "fails",
            "passes": False,
            "diagnostic_exit_code": 1,
            "sole_false_check": "torch_determinism_matches_lock",
        },
        "original_p24_sanitization": {
            "status": "blocked",
            "exit_code": P24_ORIGINAL_SANITIZER_EXIT,
            "output_created": False,
            "stdout_sha256": P24_ORIGINAL_SANITIZER_STDOUT_SHA256,
            "stderr": P24_ORIGINAL_SANITIZER_STDERR.rstrip("\n"),
            "stderr_sha256": P24_ORIGINAL_SANITIZER_STDERR_SHA256,
            "classification": "P24 value-only redaction retained two absolute mapping keys",
        },
        "p25_historical_validation": {
            "status": "passes",
            "passes": True,
            **validation,
        },
        "p25_corrected_redaction": {
            "status": "passes",
            "passes": True,
            "sanitizer_schema_version": SANITIZER.SANITIZED_SCHEMA,
            "logical_root_labels": sorted(roots),
            **counts,
            "manifest": redacted,
        },
        "claim_boundary": (
            "P25 authenticates and safely redacts the retained terminal P24 diagnostic. It "
            "does not change P24's failed disposition, rerun P24, build an image, observe a "
            "gradient, or establish CUDA acquisition/fidelity/training evidence."
        ),
    }
    SANITIZER.atomic_write_json(output_resolved, result)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", required=True, type=Path)
    parser.add_argument("--path-root", action="append", default=[])
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    try:
        result = ingest_historical(
            native=arguments.native,
            output=arguments.output,
            path_roots=SANITIZER.parse_path_roots(arguments.path_root),
        )
    except (OSError, TypeError, ValueError, SANITIZER.SanitizationError, IngestionError) as error:
        print(f"P25 historical P24 ingestion blocked: {error}", file=sys.stderr)
        return 2
    output = arguments.output.resolve()
    print(
        json.dumps(
            {
                "output": str(output),
                "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "source_artifact_sha256": result["source_artifact"]["sha256"],
                "p24_status": result["original_p24_terminal_outcome"]["status"],
                "p25_validation_status": result["p25_historical_validation"]["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
