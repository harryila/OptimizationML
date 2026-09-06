#!/usr/bin/env python3
"""Strictly validate and redact one retained native P25 diagnostic.

This module deliberately has no dependency outside the Python standard
library.  In particular, it can run next to (rather than inside) the locked
CUDA process.  Mapping keys and string values pass through the same path
redactor; this matters for mount maps whose keys are absolute paths.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Final

NATIVE_SCHEMA: Final = "passive-muon-p25-executable-origin-diagnostic-v1"
SANITIZED_SCHEMA: Final = "passive-muon-p25-sanitized-executable-origin-diagnostic-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ROOT_LABEL = re.compile(r"^[a-z][a-z0-9_]*$")
_EMBEDDED_ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z0-9_./-])/(?P<path>[^\s,:;'\"(){}\[\]]+)")
REQUIRED_LOGICAL_ROOTS: Final = {
    "data",
    "muon",
    "nanogpt",
    "native",
    "preprocessor_alias",
    "python_environment",
    "python_standard_library",
    "repository",
    "temporary",
}
_CREDENTIAL_KEY_PARTS: Final = {
    "access_token",
    "api_key",
    "authorization",
    "credential",
    "credentials",
    "password",
    "private_key",
    "secret",
    "token",
    "token_id",
    "token_secret",
}
_PASS_TOP_LEVEL_FIELDS: Final = {
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
_PASS_CHECK_NAMES: Final = {
    "bound_artifacts_unchanged_after_trigger",
    "compiled_cuda_matches_lock",
    "container_hostname_matches_lock",
    "contract_schema_and_status_exact",
    "contract_sha256_matches_expected",
    "contract_source_records_exact",
    "contract_sources_unchanged_after_trigger",
    "cuda_runtime_matches_lock",
    "cudnn_matches_lock",
    "generated_module_absent_until_optimizer",
    "generated_module_byte_count_exact",
    "generated_module_bytes_exact",
    "host_attestation_schema_and_status_exact",
    "host_attestation_sha256_matches_expected",
    "instantiator_source_exact_for_mode",
    "live_mountinfo_matches_lock",
    "mode_declared",
    "process_environment_matches_lock_exactly",
    "python_interpreter_flags_match_lock_and_frozen_p23",
    "python_runtime_matches_lock",
    "remote_module_source_exact",
    "remediated_origin_matches",
    "repository_clean",
    "repository_head_matches_expected",
    "repository_tree_matches_expected",
    "repository_unchanged_after_trigger",
    "rootfs_declared_and_observed_read_only",
    "runtime_lock_and_attestation_container_exact",
    "runtime_lock_and_attestation_gpu_exact",
    "runtime_lock_declares_exact_p23_torch_determinism",
    "runtime_lock_host_attestation_backlink_exact",
    "runtime_lock_p23_addendum_matches_contract_and_source",
    "runtime_lock_schema_and_status_exact",
    "runtime_lock_sha256_matches_expected",
    "single_live_gpu_matches_lock",
    "template_source_exact",
    "torch_determinism_configured_before_cuda",
    "torch_determinism_preserved_through_cpu_trigger",
    "torch_git_version_matches_lock",
    "torch_version_matches_lock",
}
_EXPECTED_EVENT_ORDER: Final = [
    "preflight_validated",
    "torch_imported",
    "torch_determinism_configured",
    "cpu_fp32_parameter_constructed",
    "sgd_optimizer_constructed",
]
_EXPECTED_SNAPSHOT_PRESENCE: Final = {
    "before_torch_import": False,
    "after_torch_import": False,
    "after_determinism_configuration": False,
    "after_cpu_parameter": False,
    "after_sgd_optimizer": True,
}
_GENERATED_PATH: Final = (
    "/opt/p23-venv/lib/python3.12/site-packages/torch/"
    "_p24_generated_remote_modules/_remote_module_non_scriptable.py"
)
_GENERATED_SHA256: Final = "8205b16956fb264841ecd8644784a0d157f87df79b17c16825dc1163433ce5d8"
_GENERATED_BYTE_COUNT: Final = 2355
_RUNTIME_LOCK_SCHEMA: Final = "passive-muon-p23-cuda-runtime-lock-v3"
_HOST_ATTESTATION_SCHEMA: Final = "passive-muon-p23-host-attestation-v3"
_CONTRACT_SCHEMA: Final = "passive-muon-p25-cuda-diagnostic-correction-contract-v1"
_CONTRACT_STATUS: Final = "frozen_pre_remediation_build_and_pre_acquisition"
_RUNTIME_LOCK_KEYS: Final = {
    "container",
    "determinism",
    "gpu",
    "loader_environment",
    "p23_addendum_sha256",
    "schema_version",
    "software",
    "status",
}
_HOST_ATTESTATION_KEYS: Final = {
    "container",
    "evidence",
    "gpu",
    "schema_version",
    "status",
}
_LOCKED_DETERMINISM_KEYS: Final = {
    "bf16_reduced_precision_reduction",
    "cpu_threads",
    "cublas_workspace_config",
    "cuda_visible_devices",
    "cudnn_benchmark",
    "cudnn_deterministic",
    "deterministic_algorithms",
    "deterministic_debug_mode",
    "float32_matmul_precision",
    "fp16_reduced_precision_reduction",
    "interop_threads",
    "nvidia_tf32_override",
    "nvidia_visible_devices",
    "pythonhashseed",
    "sdpa_backend",
    "tf32",
}
_FROZEN_PYTHON_FLAGS: Final = {
    "bytes_warning": 0,
    "debug": 0,
    "dev_mode": False,
    "dont_write_bytecode": 1,
    "hash_randomization": 1,
    "ignore_environment": 0,
    "inspect": 0,
    "int_max_str_digits": 4300,
    "interactive": 0,
    "isolated": 0,
    "no_site": 0,
    "no_user_site": 1,
    "optimize": 0,
    "quiet": 0,
    "safe_path": False,
    "utf8_mode": 0,
    "verbose": 0,
    "warn_default_encoding": 0,
}
_EXPECTED_TORCH_DETERMINISM: Final = {
    "allow_bf16_reduced_precision_reduction": False,
    "allow_fp16_reduced_precision_reduction": False,
    "allow_tf32_cudnn": False,
    "allow_tf32_matmul": False,
    "cuda_initialized": False,
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
    "cudnn_sdpa": False,
    "deterministic_algorithms": True,
    "deterministic_debug_mode": 2,
    "flash_sdpa": False,
    "float32_matmul_precision": "highest",
    "math_sdpa": True,
    "memory_efficient_sdpa": False,
    "torch_num_interop_threads": 1,
    "torch_num_threads": 1,
}
_PINNED_SOURCE_SHA256: Final = {
    "instantiator": "1ecc8ae1ac4a517511914fff6e01c78ed3ad828bbe8a07b02079c206ee56bceb",
    "remote_module": "f9bb2f5c5438791581d399e38a27606e123bdbeb3c6cb53683318a06060439c1",
    "template": "0ff1856bbd031b5298d46c06c0502abc20bd804f42c1949ed4127e8c773660cc",
}
_PINNED_SOURCE_SUFFIX: Final = {
    "instantiator": "lib/python3.12/site-packages/torch/distributed/nn/jit/instantiator.py",
    "remote_module": "lib/python3.12/site-packages/torch/distributed/nn/api/remote_module.py",
    "template": (
        "lib/python3.12/site-packages/torch/distributed/nn/jit/templates/remote_module_template.py"
    ),
}


class SanitizationError(RuntimeError):
    """Raised when validation or redaction must fail closed."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise SanitizationError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(token: str) -> object:
    raise SanitizationError(f"nonfinite JSON constant is forbidden: {token}")


def load_json_strict(raw: bytes) -> object:
    """Decode UTF-8 JSON while rejecting duplicate keys and nonfinite values."""

    try:
        return json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_constant,
        )
    except UnicodeDecodeError as error:
        raise SanitizationError("native diagnostic is not UTF-8") from error
    except json.JSONDecodeError as error:
        raise SanitizationError("native diagnostic is not valid JSON") from error


def _normalized_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", key.casefold()).strip("_")


def reject_credentials_and_nonfinite(value: object) -> None:
    """Reject credential-like keys and nonfinite values at every depth."""

    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            if not isinstance(raw_key, str):
                raise SanitizationError("JSON mapping keys must be strings")
            normalized = _normalized_key(raw_key)
            credential_like = normalized in _CREDENTIAL_KEY_PARTS or any(
                normalized.startswith(f"{part}_") or normalized.endswith(f"_{part}")
                for part in _CREDENTIAL_KEY_PARTS
            )
            # ``token`` is also the name of a Python standard-library module.
            # Admit only the exact file-backed-module record shape used by the
            # import-closure inventory; a scalar/object credential remains
            # forbidden.
            module_record_exception = (
                normalized == "token"
                and isinstance(item, Mapping)
                and {"path", "byte_count", "sha256", "effective_mount"} <= set(item)
            )
            if credential_like and not module_record_exception:
                raise SanitizationError(f"credential-like field is forbidden: {raw_key}")
            reject_credentials_and_nonfinite(item)
        return
    if isinstance(value, list):
        for item in value:
            reject_credentials_and_nonfinite(item)
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise SanitizationError("nonfinite numeric value is forbidden")


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SanitizationError(f"{label} must be a mapping")
    return value


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise SanitizationError(f"{label} must be a lowercase SHA-256")
    return value


def _require_binding(value: object, label: str) -> Mapping[str, object]:
    binding = _mapping(value, label)
    if set(binding) != {"byte_count", "path", "sha256"}:
        raise SanitizationError(f"{label} schema differs")
    if type(binding.get("byte_count")) is not int or binding.get("byte_count", 0) <= 0:
        raise SanitizationError(f"{label} byte count differs")
    if not isinstance(binding.get("path"), str) or not Path(str(binding["path"])).is_absolute():
        raise SanitizationError(f"{label} path differs")
    _require_sha256(binding.get("sha256"), f"{label} SHA-256")
    return binding


def _locked_determinism_is_exact(value: object) -> bool:
    lock = _mapping(value, "runtime-lock determinism")
    visible = lock.get("cuda_visible_devices")
    return (
        set(lock) == _LOCKED_DETERMINISM_KEYS
        and isinstance(visible, str)
        and visible.startswith("GPU-")
        and lock.get("nvidia_visible_devices") == visible
        and lock.get("cublas_workspace_config") == ":4096:8"
        and lock.get("nvidia_tf32_override") == "0"
        and lock.get("pythonhashseed") == "1337"
        and lock.get("deterministic_algorithms") is True
        and lock.get("deterministic_debug_mode") == "error"
        and lock.get("sdpa_backend") == "math"
        and lock.get("cudnn_deterministic") is True
        and lock.get("cudnn_benchmark") is False
        and lock.get("tf32") is False
        and lock.get("float32_matmul_precision") == "highest"
        and lock.get("fp16_reduced_precision_reduction") is False
        and lock.get("bf16_reduced_precision_reduction") is False
        and type(lock.get("cpu_threads")) is int
        and lock.get("cpu_threads") == 1
        and type(lock.get("interop_threads")) is int
        and lock.get("interop_threads") == 1
    )


def _expected_environment(runtime_lock: Mapping[str, object]) -> dict[str, str | None]:
    determinism = _mapping(runtime_lock.get("determinism"), "runtime-lock determinism")
    loader = _mapping(runtime_lock.get("loader_environment"), "runtime-lock loader environment")
    expected: dict[str, str | None] = {
        name: item if isinstance(item, str) else None for name, item in loader.items()
    }
    expected.update(
        {
            "CUBLAS_WORKSPACE_CONFIG": str(determinism.get("cublas_workspace_config")),
            "CUDA_VISIBLE_DEVICES": str(determinism.get("cuda_visible_devices")),
            "MKL_NUM_THREADS": str(determinism.get("cpu_threads")),
            "NVIDIA_TF32_OVERRIDE": str(determinism.get("nvidia_tf32_override")),
            "NVIDIA_VISIBLE_DEVICES": str(determinism.get("nvidia_visible_devices")),
            "OMP_NUM_THREADS": str(determinism.get("cpu_threads")),
            "PYTHONHASHSEED": str(determinism.get("pythonhashseed")),
            "PYTHONHOME": None,
            "PYTHONPATH": None,
            "PYTHONNOUSERSITE": "1",
            "VIRTUAL_ENV": "/opt/p23-venv",
        }
    )
    expected_names = {
        "CUBLAS_WORKSPACE_CONFIG",
        "CUDA_VISIBLE_DEVICES",
        "LD_AUDIT",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "MKL_NUM_THREADS",
        "NVIDIA_TF32_OVERRIDE",
        "NVIDIA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "PATH",
        "PYTHONHASHSEED",
        "PYTHONHOME",
        "PYTHONNOUSERSITE",
        "PYTHONOPTIMIZE",
        "PYTHONPATH",
        "PYTHONDONTWRITEBYTECODE",
        "VIRTUAL_ENV",
    }
    if set(expected) != expected_names:
        raise SanitizationError("runtime-lock loader environment schema differs")
    return expected


def _validate_contract_sources(
    contract: Mapping[str, object], source_records_value: object
) -> None:
    source_records = _mapping(source_records_value, "contract source records")
    expected: dict[str, Mapping[str, object]] = {}
    for group_name in ("unchanged_authorities", "execution_sources"):
        group = _mapping(contract.get(group_name), f"contract {group_name}")
        for name, value in group.items():
            expected[f"{group_name}.{name}"] = _mapping(value, f"contract {group_name}.{name}")
    remediation = _mapping(contract.get("remediation_design"), "contract remediation design")
    for name in ("build_patch", "image_recipe"):
        expected[f"remediation_sources.{name}"] = _mapping(
            remediation.get(name), f"contract remediation {name}"
        )
    if set(source_records) != set(expected):
        raise SanitizationError("contract source-record inventory differs")
    for name, authority in expected.items():
        record = _mapping(source_records.get(name), f"contract source record {name}")
        relative = authority.get("path")
        digest = authority.get("sha256")
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or _SHA256.fullmatch(str(digest)) is None
            or set(record) != {"byte_count", "logical_path", "path", "sha256"}
            or record.get("logical_path") != relative
            or record.get("sha256") != digest
            or type(record.get("byte_count")) is not int
            or record.get("byte_count", 0) <= 0
            or not isinstance(record.get("path"), str)
            or not Path(str(record["path"])).is_absolute()
        ):
            raise SanitizationError(f"contract source record differs: {name}")


def _validate_runtime_semantics(payload: Mapping[str, object]) -> None:
    artifacts = _mapping(payload.get("artifacts"), "passing diagnostic artifacts")
    if set(artifacts) != {"contract", "diagnostic_source", "host_attestation", "runtime_lock"}:
        raise SanitizationError("passing diagnostic artifact inventory differs")
    artifact_bindings = {
        name: _require_binding(value, f"artifact {name}") for name, value in artifacts.items()
    }
    artifact_payloads: dict[str, bytes] = {}
    for name, binding in artifact_bindings.items():
        raw, observed = _stable_read(Path(str(binding["path"])))
        if observed != {
            "byte_count": binding.get("byte_count"),
            "sha256": binding.get("sha256"),
        }:
            raise SanitizationError(f"artifact {name} differs from its native binding")
        artifact_payloads[name] = raw

    contract_binding = _mapping(payload.get("contract_binding"), "contract binding")
    runtime_binding = _mapping(payload.get("runtime_lock_binding"), "runtime-lock binding")
    attestation_binding = _mapping(
        payload.get("host_attestation_binding"), "host-attestation binding"
    )
    for artifact_name, binding in (
        ("contract", contract_binding),
        ("runtime_lock", runtime_binding),
        ("host_attestation", attestation_binding),
    ):
        if binding.get("expected_sha256") != artifact_bindings[artifact_name].get("sha256"):
            raise SanitizationError(f"passing diagnostic {artifact_name} hash binding differs")

    contract = _mapping(contract_binding.get("content"), "contract content")
    runtime_lock = _mapping(runtime_binding.get("content"), "runtime-lock content")
    attestation = _mapping(attestation_binding.get("content"), "host-attestation content")
    for artifact_name, content in (
        ("contract", contract),
        ("runtime_lock", runtime_lock),
        ("host_attestation", attestation),
    ):
        # Re-read the exact bound authority bytes, parse them independently,
        # and compare their semantic content.  This remains valid even when a
        # containing native JSON record sorted the embedded mapping keys.
        observed_content = load_json_strict(artifact_payloads[artifact_name])
        if not isinstance(observed_content, Mapping) or dict(observed_content) != dict(content):
            raise SanitizationError(
                f"embedded {artifact_name} content differs from its bound artifact bytes"
            )
    if (
        contract.get("schema_version") != _CONTRACT_SCHEMA
        or contract.get("status") != _CONTRACT_STATUS
    ):
        raise SanitizationError("passing diagnostic contract content differs")
    _validate_contract_sources(contract, contract_binding.get("source_records"))
    diagnostic_authority = _mapping(
        _mapping(contract.get("execution_sources"), "contract execution sources").get(
            "corrected_executable_origin_diagnostic"
        ),
        "diagnostic source authority",
    )
    if artifact_bindings["diagnostic_source"].get("sha256") != diagnostic_authority.get("sha256"):
        raise SanitizationError("diagnostic source artifact differs from contract")

    if (
        set(runtime_lock) != _RUNTIME_LOCK_KEYS
        or runtime_lock.get("schema_version") != _RUNTIME_LOCK_SCHEMA
        or runtime_lock.get("status") != "pinned_for_acquisition"
        or set(attestation) != _HOST_ATTESTATION_KEYS
        or attestation.get("schema_version") != _HOST_ATTESTATION_SCHEMA
        or attestation.get("status") != "procedurally_host_attested"
    ):
        raise SanitizationError("runtime-lock or host-attestation schema differs")
    if not _locked_determinism_is_exact(runtime_lock.get("determinism")):
        raise SanitizationError("runtime-lock deterministic state differs")
    container = _mapping(runtime_lock.get("container"), "runtime-lock container")
    attested_container = _mapping(attestation.get("container"), "attested container")
    gpu = _mapping(runtime_lock.get("gpu"), "runtime-lock GPU")
    attested_gpu = _mapping(attestation.get("gpu"), "attested GPU")
    container_without_backlink = {
        key: value for key, value in container.items() if key != "host_attestation_sha256"
    }
    if (
        container_without_backlink != dict(attested_container)
        or dict(gpu) != dict(attested_gpu)
        or container.get("host_attestation_sha256")
        != artifact_bindings["host_attestation"].get("sha256")
        or container.get("rootfs_read_only") is not True
        or container.get("network_mode") != "none"
    ):
        raise SanitizationError("runtime-lock and host-attestation binding differs")
    gpu_uuid = gpu.get("uuid")
    if (
        not isinstance(gpu_uuid, str)
        or not gpu_uuid.startswith("GPU-")
        or not isinstance(gpu.get("name"), str)
        or not gpu.get("name")
    ):
        raise SanitizationError("runtime-lock GPU identity differs")
    addendum = _mapping(
        _mapping(contract.get("unchanged_authorities"), "contract authorities").get("p23_addendum"),
        "P23 addendum authority",
    )
    if runtime_lock.get("p23_addendum_sha256") != addendum.get("sha256"):
        raise SanitizationError("runtime-lock P23 addendum binding differs")

    software = _mapping(runtime_lock.get("software"), "runtime-lock software")
    python_flags = _mapping(software.get("python_flags"), "runtime-lock Python flags")
    if dict(python_flags) != _FROZEN_PYTHON_FLAGS:
        raise SanitizationError("runtime-lock Python flags differ")
    live = _mapping(payload.get("live_runtime"), "live runtime")
    expected_live = {
        "hostname": container.get("hostname"),
        "platform": software.get("platform"),
        "python": software.get("python"),
        "python_executable": software.get("python_executable"),
        "python_flags": dict(python_flags),
        "torch_version": software.get("torch_version"),
        "torch_git_version": software.get("torch_git_version"),
        "torch_cuda_compiled_version": software.get("torch_cuda_compiled_version"),
        "cuda_runtime_version": software.get("cuda_runtime_version"),
        "cudnn_version": software.get("cudnn_version"),
        "cuda_available": True,
        "cuda_device_count": 1,
        "cuda_device_name": gpu.get("name"),
        "cuda_device_uuid": gpu_uuid,
        "environment": _expected_environment(runtime_lock),
        "mountinfo_sha256": container.get("mountinfo_sha256"),
        "cuda_initialized_after_import": False,
        "cuda_initialized_after_configuration": False,
        "configured_determinism": _EXPECTED_TORCH_DETERMINISM,
        "determinism_after_cpu_parameter": _EXPECTED_TORCH_DETERMINISM,
        "determinism_after_sgd_optimizer": _EXPECTED_TORCH_DETERMINISM,
    }
    if set(live) != set(expected_live) | {"root_effective_mount"}:
        raise SanitizationError("live-runtime schema differs")
    if any(live.get(name) != expected for name, expected in expected_live.items()):
        raise SanitizationError("live runtime does not reconstruct from the frozen lock")
    root_mount = _mapping(live.get("root_effective_mount"), "live root mount")
    if root_mount.get("mount_point") != "/" or root_mount.get("writable") is not False:
        raise SanitizationError("live root mount is not read-only")

    pinned = _mapping(payload.get("pinned_sources"), "pinned Torch sources")
    if set(pinned) != set(_PINNED_SOURCE_SHA256):
        raise SanitizationError("pinned Torch-source inventory differs")
    for name, expected_digest in _PINNED_SOURCE_SHA256.items():
        binding = _require_binding(pinned.get(name), f"pinned Torch source {name}")
        if binding.get("sha256") != expected_digest or not str(binding.get("path")).endswith(
            _PINNED_SOURCE_SUFFIX[name]
        ):
            raise SanitizationError(f"pinned Torch source differs: {name}")


def validate_native_semantics(payload: Mapping[str, object]) -> None:
    """Prevent a malformed pass record from authorizing later CUDA work."""

    status = payload.get("status")
    passes = payload.get("passes")
    if status not in {"passes", "fails", "error"}:
        raise SanitizationError("native diagnostic status differs")
    if not isinstance(passes, bool) or (status == "passes") is not passes:
        raise SanitizationError("native diagnostic status and passes flag disagree")
    if not passes:
        return

    if set(payload) != _PASS_TOP_LEVEL_FIELDS:
        raise SanitizationError("passing diagnostic top-level schema differs")
    if payload.get("mode") != "remediated":
        raise SanitizationError("passing diagnostic mode is not remediated")
    if payload.get("seed") is not None or payload.get("seed_semantics") != (
        "not applicable: the diagnostic performs no random operation"
    ):
        raise SanitizationError("passing diagnostic seed semantics differ")

    checks = _mapping(payload.get("checks"), "passing diagnostic checks")
    if set(checks) != _PASS_CHECK_NAMES or not all(value is True for value in checks.values()):
        raise SanitizationError("passing diagnostic exact check inventory differs")

    repository = _mapping(payload.get("repository"), "passing diagnostic repository")
    observed_repository = _mapping(repository.get("observed"), "observed repository")
    if (
        observed_repository.get("dirty") is not False
        or repository.get("expected_head") != observed_repository.get("head")
        or repository.get("expected_tree") != observed_repository.get("tree")
    ):
        raise SanitizationError("passing diagnostic repository binding differs")

    _validate_runtime_semantics(payload)

    process = _mapping(payload.get("process"), "passing diagnostic process")
    if (
        set(process) != {"argv", "completed_time_ns", "nonce", "pid", "started_time_ns"}
        or type(process.get("pid")) is not int
        or process.get("pid", 0) <= 0
        or type(process.get("started_time_ns")) is not int
        or type(process.get("completed_time_ns")) is not int
        or process.get("completed_time_ns", 0) < process.get("started_time_ns", 0)
        or _SHA256.fullmatch(str(process.get("nonce"))) is None
        or not isinstance(process.get("argv"), list)
        or not process.get("argv")
    ):
        raise SanitizationError("passing diagnostic process record differs")

    trigger = _mapping(payload.get("trigger"), "passing diagnostic trigger")
    if trigger.get("event_order") != _EXPECTED_EVENT_ORDER:
        raise SanitizationError("passing diagnostic event order differs")
    snapshots = _mapping(trigger.get("module_snapshots"), "module snapshots")
    if set(snapshots) != set(_EXPECTED_SNAPSHOT_PRESENCE):
        raise SanitizationError("passing diagnostic snapshot inventory differs")
    for name, expected_present in _EXPECTED_SNAPSHOT_PRESENCE.items():
        snapshot = _mapping(snapshots.get(name), f"module snapshot {name}")
        if (
            set(snapshot)
            != {"canonical_sha256", "file_backed_module_count", "generated_module_present"}
            or snapshot.get("generated_module_present") is not expected_present
            or type(snapshot.get("file_backed_module_count")) is not int
            or snapshot.get("file_backed_module_count", 0) <= 0
            or _SHA256.fullmatch(str(snapshot.get("canonical_sha256"))) is None
        ):
            raise SanitizationError("passing diagnostic module-presence order differs")

    generated = _mapping(trigger.get("generated_module"), "generated module")
    mount = _mapping(generated.get("effective_mount"), "generated-module mount")
    if (
        generated.get("path") != _GENERATED_PATH
        or generated.get("sha256") != _GENERATED_SHA256
        or generated.get("byte_count") != _GENERATED_BYTE_COUNT
        or mount.get("writable") is not False
        or trigger.get("temporary_generated_module_names") != []
    ):
        raise SanitizationError("passing diagnostic immutable generated-module origin differs")
    closure = _mapping(payload.get("file_backed_module_closure"), "module closure")
    records = _mapping(closure.get("records"), "module-closure records")
    canonical_records = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    if (
        closure.get("count") != len(records)
        or closure.get("canonical_sha256") != canonical_records
        or records.get("_remote_module_non_scriptable") != generated
    ):
        raise SanitizationError("passing diagnostic generated module and closure differ")


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def validate_path_roots(raw_roots: Mapping[str, str | Path]) -> dict[str, Path]:
    """Resolve a nonoverlapping logical-root map used by the redactor."""

    if set(raw_roots) != REQUIRED_LOGICAL_ROOTS:
        raise SanitizationError("path roots differ from the required nine-label inventory")
    roots: dict[str, Path] = {}
    for name, raw_path in raw_roots.items():
        if not isinstance(name, str) or _ROOT_LABEL.fullmatch(name) is None:
            raise SanitizationError(f"invalid logical root label: {name!r}")
        path = Path(raw_path)
        if not path.is_absolute() or path == Path("/"):
            raise SanitizationError(f"logical root {name!r} must be a non-root absolute path")
        try:
            roots[name] = path.resolve(strict=True)
        except OSError as error:
            raise SanitizationError(f"logical root {name!r} does not exist") from error
    if len(set(roots.values())) != len(roots):
        raise SanitizationError("logical roots must resolve to distinct paths")
    names = sorted(roots)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            left = roots[left_name]
            right = roots[right_name]
            if _is_within(left, right) or _is_within(right, left):
                raise SanitizationError("logical roots must be pairwise nonoverlapping")
    return roots


class _PathRedactor:
    """Apply one path rule to both JSON keys and JSON string values."""

    def __init__(self, roots: Mapping[str, Path]) -> None:
        self.roots = sorted(roots.items(), key=lambda item: len(str(item[1])), reverse=True)
        self.key_replacement_count = 0
        self.value_replacement_count = 0

    @staticmethod
    def _external_token(raw_path: str) -> str:
        digest = hashlib.sha256(raw_path.encode("utf-8", errors="surrogatepass")).hexdigest()
        return f"external_absolute_path_sha256:{digest}"

    def _sanitize_text(self, value: str) -> tuple[str, int]:
        # A colon-delimited PATH is a list of paths, not one absolute path.
        # Route it through the embedded-path pass so every component is
        # redacted independently.
        if Path(value).is_absolute() and os.pathsep not in value:
            lexical = Path(value).resolve(strict=False)
            for name, root in self.roots:
                try:
                    relative = lexical.relative_to(root)
                except ValueError:
                    continue
                suffix = relative.as_posix()
                return (f"{name}:{suffix}" if suffix != "." else f"{name}:"), 1
            return self._external_token(value), 1

        sanitized = value
        replacement_count = 0
        for name, root in self.roots:
            raw_root = str(root)
            occurrences = sanitized.count(raw_root)
            if occurrences:
                sanitized = sanitized.replace(raw_root, f"{name}:")
                sanitized = sanitized.replace(f"{name}:/", f"{name}:")
                replacement_count += occurrences

        def redact_external_path(match: re.Match[str]) -> str:
            nonlocal replacement_count
            replacement_count += 1
            return self._external_token(match.group(0))

        sanitized = _EMBEDDED_ABSOLUTE_PATH.sub(redact_external_path, sanitized)
        return sanitized, replacement_count

    def sanitize(self, value: object, *, mapping_key: bool = False) -> object:
        if isinstance(value, Mapping):
            result: dict[str, object] = {}
            original_by_sanitized: dict[str, str] = {}
            for raw_key, item in value.items():
                if not isinstance(raw_key, str):
                    raise SanitizationError("JSON mapping keys must be strings")
                sanitized_key, count = self._sanitize_text(raw_key)
                self.key_replacement_count += count
                if sanitized_key in result:
                    prior = original_by_sanitized[sanitized_key]
                    raise SanitizationError(
                        f"path redaction creates a mapping-key collision: {prior!r} and {raw_key!r}"
                    )
                original_by_sanitized[sanitized_key] = raw_key
                result[sanitized_key] = self.sanitize(item)
            return result
        if isinstance(value, list):
            return [self.sanitize(item) for item in value]
        if isinstance(value, str):
            sanitized, count = self._sanitize_text(value)
            if mapping_key:
                self.key_replacement_count += count
            else:
                self.value_replacement_count += count
            return sanitized
        return value


def _assert_no_absolute_paths(value: object) -> None:
    """Scan all sanitized keys and values for any undeclared absolute path."""

    def inspect_text(text: str) -> None:
        if Path(text).is_absolute() or _EMBEDDED_ABSOLUTE_PATH.search(text) is not None:
            raise SanitizationError("redacted diagnostic retains an absolute path")

    def inspect(item: object) -> None:
        if isinstance(item, Mapping):
            for key, nested in item.items():
                if not isinstance(key, str):  # pragma: no cover - JSON invariant
                    raise SanitizationError("redacted mapping key is not a string")
                inspect_text(key)
                inspect(nested)
        elif isinstance(item, list):
            for nested in item:
                inspect(nested)
        elif isinstance(item, str):
            inspect_text(item)

    inspect(value)


def redact_payload(
    payload: Mapping[str, object], roots: Mapping[str, Path]
) -> tuple[dict[str, object], dict[str, int]]:
    """Redact a validated mapping and return separate key/value counts."""

    reject_credentials_and_nonfinite(payload)
    redactor = _PathRedactor(roots)
    redacted = redactor.sanitize(payload)
    if not isinstance(redacted, dict):  # pragma: no cover - guarded by the input type
        raise AssertionError("mapping redaction did not return a mapping")
    _assert_no_absolute_paths(redacted)
    key_count = redactor.key_replacement_count
    value_count = redactor.value_replacement_count
    return redacted, {
        "key_replacement_count": key_count,
        "value_replacement_count": value_count,
        "total_replacement_count": key_count + value_count,
    }


def atomic_write_json(output: Path, payload: object) -> None:
    """Publish canonical JSON atomically while refusing every overwrite."""

    parent = output.parent.resolve(strict=True)
    destination = parent / output.name
    if destination.exists() or destination.is_symlink():
        raise SanitizationError("output must be a fresh path")
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=parent, prefix=f".{output.name}.", delete=False
        ) as handle:
            temporary_name = handle.name
            os.chmod(temporary_name, 0o600)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, destination, follow_symlinks=False)
        except FileExistsError as error:
            raise SanitizationError("output must be a fresh path") from error
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_name is not None:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary_name)


def _stable_read(path: Path) -> tuple[bytes, dict[str, object]]:
    if path.is_symlink():
        raise SanitizationError("native diagnostic must not be a symbolic link")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise SanitizationError("native diagnostic must be a regular file")
    before = resolved.stat()
    raw = resolved.read_bytes()
    after = resolved.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise SanitizationError("native diagnostic changed while it was read")
    return raw, {
        "byte_count": len(raw),
        "sha256": sha256_bytes(raw),
    }


def sanitize_native(
    *,
    native: Path,
    expected_native_sha256: str,
    output: Path,
    path_roots: Mapping[str, str | Path],
    expected_native_schema: str = NATIVE_SCHEMA,
) -> dict[str, object]:
    """Byte-bind, redact, and publish a P25 diagnostic."""

    if _SHA256.fullmatch(expected_native_sha256) is None:
        raise SanitizationError("expected native SHA-256 is invalid")
    roots = validate_path_roots(path_roots)
    native_resolved = native.resolve(strict=True)
    output_resolved = output.parent.resolve(strict=True) / output.name
    if not _is_within(native_resolved, roots["native"]) or not _is_within(
        output_resolved, roots["native"]
    ):
        raise SanitizationError("native and sanitized artifacts must remain in the evidence root")
    if native_resolved == output_resolved:
        raise SanitizationError("native and sanitized paths must be distinct")
    raw, binding = _stable_read(native)
    if binding["sha256"] != expected_native_sha256:
        raise SanitizationError("native diagnostic differs from its expected SHA-256")
    payload = load_json_strict(raw)
    if not isinstance(payload, Mapping):
        raise SanitizationError("native diagnostic must be a JSON object")
    if payload.get("schema_version") != expected_native_schema:
        raise SanitizationError("native diagnostic schema differs")
    validate_native_semantics(payload)
    status = payload["status"]
    passes = payload["passes"]
    redacted, counts = redact_payload(payload, roots)
    result = {
        "schema_version": SANITIZED_SCHEMA,
        "native_artifact": binding,
        "native_schema_version": expected_native_schema,
        "mode": payload.get("mode"),
        "status": status,
        "passes": passes,
        "logical_root_labels": sorted(roots),
        "redaction": counts,
        "manifest": redacted,
        "claim_boundary": (
            "Offline byte-bound redaction of one P25 executable-origin diagnostic; not a "
            "gradient acquisition, fidelity result, or training result."
        ),
    }
    atomic_write_json(output_resolved, result)
    return result


def parse_path_roots(values: list[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path or name in roots:
            raise SanitizationError("path roots must be unique NAME=ABSOLUTE_PATH entries")
        roots[name] = Path(raw_path)
    return roots


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", required=True, type=Path)
    parser.add_argument("--expected-native-sha256", required=True)
    parser.add_argument("--path-root", action="append", default=[])
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    try:
        result = sanitize_native(
            native=arguments.native,
            expected_native_sha256=arguments.expected_native_sha256,
            output=arguments.output,
            path_roots=parse_path_roots(arguments.path_root),
        )
    except (OSError, TypeError, ValueError, SanitizationError) as error:
        print(f"P25 diagnostic sanitization blocked: {error}", file=sys.stderr)
        return 2
    output = arguments.output.resolve()
    print(
        json.dumps(
            {
                "output": str(output),
                "output_sha256": sha256_bytes(output.read_bytes()),
                "native_artifact_sha256": result["native_artifact"]["sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
