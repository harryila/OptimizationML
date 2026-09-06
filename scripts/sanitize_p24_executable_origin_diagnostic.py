#!/usr/bin/env python3
"""Validate and redact one retained native P24 executable-origin diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTIC_SOURCE = ROOT / "experiments/training/run_p24_executable_origin_diagnostic.py"
SANITIZED_SCHEMA = "passive-muon-p24-sanitized-executable-origin-diagnostic-v1"
EXPECTED_ROOT_LABELS = {
    "repository",
    "nanogpt",
    "muon",
    "data",
    "preprocessor_alias",
    "python_environment",
    "python_standard_library",
    "temporary",
    "native",
}
SAFE_PATH_VALUE = "/opt/p23-venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
SUCCESS_FIELDS = {
    "schema_version",
    "mode",
    "status",
    "scope",
    "seed",
    "seed_semantics",
    "process",
    "repository",
    "artifacts",
    "contract_binding",
    "runtime_lock_binding",
    "host_attestation_binding",
    "live_runtime",
    "pinned_sources",
    "trigger",
    "file_backed_module_closure",
    "checks",
    "claim_boundary",
    "passes",
}
ERROR_FIELDS = SUCCESS_FIELDS | {"error"}
FORBIDDEN_CREDENTIAL_KEYS = {
    "access_token",
    "api_key",
    "authorization",
    "credential",
    "password",
    "secret",
    "token_secret",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_OBJECT = re.compile(r"^[0-9a-f]{40}$")
LIVE_RUNTIME_FIELDS = {
    "hostname",
    "platform",
    "python",
    "python_executable",
    "torch_version",
    "torch_git_version",
    "torch_cuda_compiled_version",
    "cuda_runtime_version",
    "cudnn_version",
    "cuda_available",
    "cuda_device_count",
    "cuda_device_name",
    "cuda_device_uuid",
    "environment",
    "mountinfo_sha256",
    "root_effective_mount",
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
}
COMPLETED_CHECK_NAMES = {
    "mode_declared",
    "contract_schema_and_status_exact",
    "contract_sha256_matches_expected",
    "runtime_lock_schema_and_status_exact",
    "runtime_lock_sha256_matches_expected",
    "runtime_lock_p23_addendum_matches_contract_and_source",
    "host_attestation_schema_and_status_exact",
    "host_attestation_sha256_matches_expected",
    "runtime_lock_host_attestation_backlink_exact",
    "runtime_lock_and_attestation_container_exact",
    "runtime_lock_and_attestation_gpu_exact",
    "repository_clean",
    "repository_head_matches_expected",
    "repository_tree_matches_expected",
    "contract_source_records_exact",
    "container_hostname_matches_lock",
    "rootfs_declared_and_observed_read_only",
    "live_mountinfo_matches_lock",
    "process_environment_matches_lock_exactly",
    "python_runtime_matches_lock",
    "repository_unchanged_after_trigger",
    "contract_sources_unchanged_after_trigger",
    "bound_artifacts_unchanged_after_trigger",
    "torch_version_matches_lock",
    "torch_git_version_matches_lock",
    "compiled_cuda_matches_lock",
    "cuda_runtime_matches_lock",
    "cudnn_matches_lock",
    "single_live_gpu_matches_lock",
    "torch_determinism_matches_lock",
    "remote_module_source_exact",
    "template_source_exact",
    "instantiator_source_exact_for_mode",
    "generated_module_loaded_only_after_optimizer",
    "generated_module_bytes_exact",
    "mode_origin_matches",
}
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_EMBEDDED_ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z0-9_./-])/(?P<path>[^\s,:;'\"(){}\[\]]+)")


class SanitizationError(RuntimeError):
    """Raised when native diagnostic validation or redaction fails closed."""


def _load_diagnostic_module():
    specification = importlib.util.spec_from_file_location(
        "p24_executable_origin_diagnostic_for_sanitizer", DIAGNOSTIC_SOURCE
    )
    if specification is None or specification.loader is None:
        raise SanitizationError("cannot load the pinned P24 diagnostic implementation")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


DIAGNOSTIC = _load_diagnostic_module()


def _mapping(value: object, fields: set[str], label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise SanitizationError(f"{label} field set changed")
    return value


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise SanitizationError(f"{label} is not a SHA-256 digest")
    return value


def _positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise SanitizationError(f"{label} is not a positive integer")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SanitizationError(f"{label} is not a nonnegative integer")
    return value


def _reject_credentials(value: object) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower()
            if normalized in FORBIDDEN_CREDENTIAL_KEYS:
                raise SanitizationError(f"credential-like field is forbidden: {key}")
            _reject_credentials(item)
    elif isinstance(value, list):
        for item in value:
            _reject_credentials(item)


def _validate_artifact_record(
    value: object,
    label: str,
    *,
    rehash: bool = False,
    retained_bytes: dict[str, bytes] | None = None,
) -> Mapping[str, object]:
    record = _mapping(value, {"path", "byte_count", "sha256"}, label)
    if not isinstance(record.get("path"), str) or not Path(str(record["path"])).is_absolute():
        raise SanitizationError(f"{label}.path is not absolute")
    _nonnegative_int(record.get("byte_count"), f"{label}.byte_count")
    _sha256(record.get("sha256"), f"{label}.sha256")
    if rehash:
        try:
            observed, raw = DIAGNOSTIC._stable_file_binding(Path(str(record["path"])), label)
        except (OSError, DIAGNOSTIC.DiagnosticError) as error:
            raise SanitizationError(f"{label} cannot be rehashed") from error
        if observed != dict(record):
            raise SanitizationError(f"{label} changed after the native diagnostic")
        if retained_bytes is not None:
            retained_bytes[label] = raw
    return record


def _validate_best_effort_artifact(value: object, label: str) -> None:
    if not isinstance(value, Mapping):
        raise SanitizationError(f"{label} is not a mapping")
    expected = {"path", "exists", "is_file"}
    is_file = value.get("is_file")
    if is_file is True:
        expected |= {"byte_count", "sha256"}
    if set(value) != expected:
        raise SanitizationError(f"{label} field set changed")
    path = value.get("path")
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise SanitizationError(f"{label}.path is not absolute")
    if not isinstance(value.get("exists"), bool) or not isinstance(is_file, bool):
        raise SanitizationError(f"{label} existence fields are not Boolean")
    if is_file and not value.get("exists"):
        raise SanitizationError(f"{label} cannot be a file without existing")
    if is_file:
        _nonnegative_int(value.get("byte_count"), f"{label}.byte_count")
        _sha256(value.get("sha256"), f"{label}.sha256")
    resolved = Path(str(path)).resolve()
    if resolved.exists() is not value.get("exists") or resolved.is_file() is not is_file:
        raise SanitizationError(f"{label} existence changed after the native diagnostic")
    if is_file:
        try:
            observed, _ = DIAGNOSTIC._stable_file_binding(resolved, label)
        except (OSError, DIAGNOSTIC.DiagnosticError) as error:
            raise SanitizationError(f"{label} cannot be rehashed") from error
        expected_record = {
            "path": str(resolved),
            "exists": True,
            "is_file": True,
            "byte_count": observed["byte_count"],
            "sha256": observed["sha256"],
        }
        if dict(value) != expected_record:
            raise SanitizationError(f"{label} changed after the native diagnostic")


def _validate_repository(value: object) -> Mapping[str, object]:
    binding = _mapping(value, {"expected_head", "expected_tree", "observed"}, "repository")
    observed = _mapping(
        binding.get("observed"),
        {"path", "head", "tree", "dirty", "porcelain_v1_z_sha256"},
        "repository.observed",
    )
    if not isinstance(observed.get("path"), str) or not Path(str(observed["path"])).is_absolute():
        raise SanitizationError("repository path is not absolute")
    for field in ("expected_head", "expected_tree"):
        value = binding.get(field)
        if not isinstance(value, str) or _GIT_OBJECT.fullmatch(value) is None:
            raise SanitizationError(f"repository.{field} is invalid")
    for field in ("head", "tree"):
        current = observed.get(field)
        if not isinstance(current, str) or _GIT_OBJECT.fullmatch(current) is None:
            raise SanitizationError(f"repository observed {field} is invalid")
    _sha256(observed.get("porcelain_v1_z_sha256"), "repository porcelain digest")
    if not isinstance(observed.get("dirty"), bool):
        raise SanitizationError("repository dirty flag is not Boolean")
    return binding


def _validate_binding(
    value: object,
    *,
    artifact: Mapping[str, object],
    label: str,
    content_schema: str,
) -> Mapping[str, object]:
    fields = {"expected_sha256", "content"}
    if label == "contract":
        fields.add("source_records")
    binding = _mapping(value, fields, f"{label}_binding")
    _sha256(binding.get("expected_sha256"), f"{label} expected digest")
    content = binding.get("content")
    if not isinstance(content, Mapping) or content.get("schema_version") != content_schema:
        raise SanitizationError(f"{label} retained content schema differs")
    return binding


def _json_bytes_equal_content(payload: bytes, content: object) -> bool:
    try:
        decoded = json.loads(
            payload,
            parse_constant=lambda token: (_ for _ in ()).throw(
                SanitizationError(f"nonfinite JSON constant is forbidden: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, SanitizationError):
        return False
    return decoded == content


def _validate_mount(value: object, label: str) -> Mapping[str, object]:
    mount = _mapping(
        value,
        {
            "mount_id",
            "mount_point",
            "filesystem_type",
            "mount_options",
            "super_options",
            "writable",
        },
        label,
    )
    if not isinstance(mount.get("mount_id"), int) or isinstance(mount.get("mount_id"), bool):
        raise SanitizationError(f"{label}.mount_id is not an integer")
    if not all(isinstance(mount.get(field), str) for field in ("mount_point", "filesystem_type")):
        raise SanitizationError(f"{label} string fields differ")
    if not all(
        isinstance(mount.get(field), list)
        and mount[field]
        and all(isinstance(item, str) and item for item in mount[field])
        for field in ("mount_options", "super_options")
    ):
        raise SanitizationError(f"{label} option fields differ")
    if not isinstance(mount.get("writable"), bool):
        raise SanitizationError(f"{label}.writable is not Boolean")
    options = mount["mount_options"]
    has_rw = "rw" in options
    has_ro = "ro" in options
    if has_rw == has_ro or mount.get("writable") is not has_rw:
        raise SanitizationError(f"{label}.writable contradicts mount options")
    return mount


def _validate_module_record(value: object, label: str) -> Mapping[str, object]:
    record = _mapping(
        value,
        {"path", "byte_count", "sha256", "effective_mount"},
        label,
    )
    _validate_artifact_record(
        {key: record[key] for key in ("path", "byte_count", "sha256")},
        label,
    )
    _validate_mount(record.get("effective_mount"), f"{label}.effective_mount")
    return record


def _validate_success(payload: Mapping[str, object]) -> None:
    """Validate and independently reconstruct one completed diagnostic.

    A completed ``fails`` record is evidence, not an invalid artifact.  The
    structural relations below therefore reject malformed evidence while the
    exact reconstructed check map determines whether the retained outcome is a
    pass or a fail.
    """

    process = _mapping(
        payload.get("process"),
        {"pid", "nonce", "started_time_ns", "completed_time_ns", "argv"},
        "process",
    )
    _positive_int(process.get("pid"), "process.pid")
    _sha256(process.get("nonce"), "process.nonce")
    started = _positive_int(process.get("started_time_ns"), "process.started_time_ns")
    completed = _positive_int(process.get("completed_time_ns"), "process.completed_time_ns")
    if (
        completed < started
        or not isinstance(process.get("argv"), list)
        or not process["argv"]
        or not all(isinstance(item, str) for item in process["argv"])
    ):
        raise SanitizationError("process interval or argv is invalid")
    repository = _validate_repository(payload.get("repository"))
    repository_observed = repository["observed"]
    assert isinstance(repository_observed, Mapping)

    artifacts = _mapping(
        payload.get("artifacts"),
        {"diagnostic_source", "contract", "runtime_lock", "host_attestation"},
        "artifacts",
    )
    artifact_bytes: dict[str, bytes] = {}
    artifact_records = {
        name: _validate_artifact_record(
            record,
            f"artifacts.{name}",
            rehash=True,
            retained_bytes=artifact_bytes,
        )
        for name, record in artifacts.items()
    }
    contract = _validate_binding(
        payload.get("contract_binding"),
        artifact=artifact_records["contract"],
        label="contract",
        content_schema=DIAGNOSTIC.CONTRACT_SCHEMA,
    )
    runtime = _validate_binding(
        payload.get("runtime_lock_binding"),
        artifact=artifact_records["runtime_lock"],
        label="runtime_lock",
        content_schema=DIAGNOSTIC.RUNTIME_LOCK_SCHEMA,
    )
    attestation = _validate_binding(
        payload.get("host_attestation_binding"),
        artifact=artifact_records["host_attestation"],
        label="host_attestation",
        content_schema=DIAGNOSTIC.HOST_ATTESTATION_SCHEMA,
    )
    contract_content = contract["content"]
    runtime_content = runtime["content"]
    attestation_content = attestation["content"]
    assert isinstance(contract_content, Mapping)
    assert isinstance(runtime_content, Mapping)
    assert isinstance(attestation_content, Mapping)
    if contract_content.get("status") != "frozen_pre_remediation_build_and_pre_acquisition":
        raise SanitizationError("contract status differs")
    source_records = contract.get("source_records")
    if not isinstance(source_records, Mapping) or not source_records:
        raise SanitizationError("contract source records are absent")
    for name, raw_record in source_records.items():
        record = _mapping(
            raw_record,
            {"logical_path", "path", "byte_count", "sha256"},
            f"contract source {name}",
        )
        if not isinstance(record.get("logical_path"), str):
            raise SanitizationError(f"contract source {name} has no logical path")
        _validate_artifact_record(
            {key: record[key] for key in ("path", "byte_count", "sha256")},
            f"contract source {name}",
            rehash=True,
        )
    execution_sources = contract_content.get("execution_sources")
    if not isinstance(execution_sources, Mapping):
        raise SanitizationError("contract content has no execution sources")
    diagnostic_source = execution_sources.get("native_executable_origin_diagnostic")
    if not isinstance(diagnostic_source, Mapping) or diagnostic_source.get(
        "sha256"
    ) != artifact_records["diagnostic_source"].get("sha256"):
        raise SanitizationError("diagnostic source does not match contract")

    authorities = contract_content.get("unchanged_authorities")
    remediation = contract_content.get("remediation_design")
    if not isinstance(authorities, Mapping) or not isinstance(remediation, Mapping):
        raise SanitizationError("contract authorities or remediation design are absent")
    remediation_entries = {
        "build_patch": remediation.get("build_patch"),
        "image_recipe": remediation.get("image_recipe"),
    }
    expected_source_records: dict[str, Mapping[str, object]] = {}
    for group_name, entries in (
        ("unchanged_authorities", authorities),
        ("execution_sources", execution_sources),
        ("remediation_sources", remediation_entries),
    ):
        for name, entry in entries.items():
            if not isinstance(entry, Mapping):
                raise SanitizationError(f"contract entry {group_name}.{name} is absent")
            expected_source_records[f"{group_name}.{name}"] = entry
    if set(source_records) != set(expected_source_records):
        raise SanitizationError("contract source-record inventory differs")
    contract_sources_match = True
    repository_path = Path(str(repository_observed["path"])).resolve()
    for name, entry in expected_source_records.items():
        record = source_records[name]
        assert isinstance(record, Mapping)
        logical_path = record.get("logical_path")
        expected_path = (
            (repository_path / logical_path).resolve()
            if isinstance(logical_path, str) and not Path(logical_path).is_absolute()
            else None
        )
        contract_sources_match = contract_sources_match and (
            logical_path == entry.get("path")
            and record.get("sha256") == entry.get("sha256")
            and expected_path is not None
            and expected_path.is_relative_to(repository_path)
            and Path(str(record.get("path"))).resolve() == expected_path
        )
    runtime_container = runtime_content.get("container")
    attested_container = attestation_content.get("container")
    if not isinstance(runtime_container, Mapping) or not isinstance(attested_container, Mapping):
        raise SanitizationError("runtime or attested container binding is absent")
    without_backlink = {
        key: value for key, value in runtime_container.items() if key != "host_attestation_sha256"
    }
    p23_entry = authorities.get("p23_addendum")
    p23_record = source_records.get("unchanged_authorities.p23_addendum")
    runtime_addendum_sha256 = runtime_content.get("p23_addendum_sha256")
    contract_addendum_sha256 = p23_entry.get("sha256") if isinstance(p23_entry, Mapping) else None
    source_addendum_sha256 = p23_record.get("sha256") if isinstance(p23_record, Mapping) else None
    p23_addendum_matches = (
        not isinstance(p23_entry, Mapping)
        or not isinstance(p23_record, Mapping)
        or not (runtime_addendum_sha256 == contract_addendum_sha256 == source_addendum_sha256)
    ) is False

    live = _mapping(payload.get("live_runtime"), LIVE_RUNTIME_FIELDS, "live_runtime")
    software = runtime_content.get("software")
    determinism = runtime_content.get("determinism")
    gpu = runtime_content.get("gpu")
    if not all(isinstance(item, Mapping) for item in (software, determinism, gpu)):
        raise SanitizationError("runtime-lock software, determinism, or GPU map is absent")
    assert isinstance(software, Mapping)
    assert isinstance(determinism, Mapping)
    assert isinstance(gpu, Mapping)
    try:
        expected_environment = DIAGNOSTIC._expected_process_environment(runtime_content)
    except (TypeError, ValueError, DIAGNOSTIC.DiagnosticError) as error:
        raise SanitizationError(
            "runtime lock cannot reconstruct the process environment"
        ) from error
    root_mount = _validate_mount(live.get("root_effective_mount"), "root mount")

    closure = _mapping(
        payload.get("file_backed_module_closure"),
        {"count", "canonical_sha256", "records"},
        "file_backed_module_closure",
    )
    records = closure.get("records")
    if not isinstance(records, Mapping) or not records or closure.get("count") != len(records):
        raise SanitizationError("module-closure cardinality differs")
    for name, record in records.items():
        if not isinstance(name, str) or not name:
            raise SanitizationError("module-closure name differs")
        _validate_module_record(record, f"module closure {name}")
    if closure.get("canonical_sha256") != DIAGNOSTIC.canonical_sha256(records):
        raise SanitizationError("module-closure canonical digest differs")

    sources = payload.get("pinned_sources")
    if not isinstance(sources, Mapping) or set(sources) != {
        "instantiator",
        "remote_module",
        "template",
    }:
        raise SanitizationError("pinned PyTorch source inventory differs")
    mode = payload.get("mode")
    expected_instantiator = (
        DIAGNOSTIC.ORIGINAL_INSTANTIATOR_SHA256
        if mode == "baseline"
        else DIAGNOSTIC.PATCHED_INSTANTIATOR_SHA256
    )
    expected_sources = {
        "instantiator": expected_instantiator,
        "remote_module": DIAGNOSTIC.REMOTE_MODULE_SHA256,
        "template": DIAGNOSTIC.TEMPLATE_SHA256,
    }
    validated_sources = {
        name: _validate_artifact_record(sources[name], f"pinned_sources.{name}", rehash=True)
        for name in expected_sources
    }
    for name, record in validated_sources.items():
        if Path(str(record["path"])).resolve() != DIAGNOSTIC.PINNED_SOURCES[name].resolve():
            raise SanitizationError(f"pinned source {name} path differs")

    trigger = _mapping(
        payload.get("trigger"),
        {
            "operation",
            "generated_module",
            "temporary_generated_module_names",
            "new_modules_after_torch_import",
            "new_modules_after_parameter",
            "new_modules_after_optimizer",
        },
        "trigger",
    )
    generated = trigger.get("generated_module")
    if generated is not None:
        generated = _validate_module_record(generated, "generated module")
    for field in (
        "temporary_generated_module_names",
        "new_modules_after_torch_import",
        "new_modules_after_parameter",
        "new_modules_after_optimizer",
    ):
        values = trigger.get(field)
        if (
            not isinstance(values, list)
            or not all(isinstance(item, str) and item for item in values)
            or values != sorted(set(values))
        ):
            raise SanitizationError(f"trigger.{field} is not a string list")
    expected_operation = (
        "import torch; create one zero CPU float32 Parameter; construct "
        "torch.optim.SGD([parameter], lr=1.0); perform no optimizer step"
    )
    if trigger.get("operation") != expected_operation:
        raise SanitizationError("trigger.operation differs")
    closure_generated = records.get(DIAGNOSTIC.GENERATED_MODULE)
    if (generated is None) != (closure_generated is None) or (
        isinstance(generated, Mapping) and generated != closure_generated
    ):
        raise SanitizationError("trigger generated module differs from the final closure")
    temporary_names = trigger["temporary_generated_module_names"]
    for name in temporary_names:
        record = records.get(name)
        if not isinstance(record, Mapping) or not str(record.get("path", "")).startswith("/tmp/"):
            raise SanitizationError("temporary generated-module inventory differs from closure")

    generated_path = generated.get("path") if isinstance(generated, Mapping) else None
    generated_mount = generated.get("effective_mount") if isinstance(generated, Mapping) else None
    generated_writable = (
        generated_mount.get("writable") if isinstance(generated_mount, Mapping) else None
    )
    generated_loaded_after_optimizer = (
        isinstance(generated, Mapping)
        and DIAGNOSTIC.GENERATED_MODULE in trigger["new_modules_after_optimizer"]
    )
    generated_bytes_exact = (
        isinstance(generated, Mapping) and generated.get("sha256") == DIAGNOSTIC.GENERATED_SHA256
    )
    mode_origin_matches = (
        mode == "baseline"
        and isinstance(generated_path, str)
        and generated_path.startswith("/tmp/")
        and generated_writable is True
    ) or (
        mode == "remediated"
        and generated_path == DIAGNOSTIC.FIXED_GENERATED_PATH
        and generated_writable is False
        and not temporary_names
    )

    current_json_matches = {
        "contract": _json_bytes_equal_content(
            artifact_bytes["artifacts.contract"], contract_content
        ),
        "runtime_lock": _json_bytes_equal_content(
            artifact_bytes["artifacts.runtime_lock"], runtime_content
        ),
        "host_attestation": _json_bytes_equal_content(
            artifact_bytes["artifacts.host_attestation"], attestation_content
        ),
    }
    bound_artifacts_unchanged = artifact_records["diagnostic_source"].get(
        "sha256"
    ) == diagnostic_source.get("sha256") and all(
        artifact_records[name].get("sha256") == binding.get("expected_sha256")
        and current_json_matches[name]
        for name, binding in (
            ("contract", contract),
            ("runtime_lock", runtime),
            ("host_attestation", attestation),
        )
    )
    repository_unchanged = (
        repository_observed.get("head") == repository.get("expected_head")
        and repository_observed.get("tree") == repository.get("expected_tree")
        and repository_observed.get("dirty") is False
        and repository_observed.get("porcelain_v1_z_sha256") == _EMPTY_SHA256
    )
    torch_determinism_matches = (
        live.get("deterministic_algorithms") == determinism.get("deterministic_algorithms")
        and live.get("deterministic_debug_mode") == 2
        and determinism.get("deterministic_debug_mode") == "error"
        and live.get("cpu_threads") == determinism.get("cpu_threads")
        and live.get("interop_threads") == determinism.get("interop_threads")
        and live.get("cudnn_benchmark") == determinism.get("cudnn_benchmark")
        and live.get("cudnn_deterministic") == determinism.get("cudnn_deterministic")
        and live.get("float32_matmul_precision") == determinism.get("float32_matmul_precision")
        and live.get("tf32") == determinism.get("tf32")
        and live.get("fp16_reduced_precision_reduction")
        == determinism.get("fp16_reduced_precision_reduction")
        and live.get("bf16_reduced_precision_reduction")
        == determinism.get("bf16_reduced_precision_reduction")
    )
    reconstructed_checks = {
        # These checks passed before the trigger; otherwise the diagnostic
        # raises and emits ``status=error`` rather than a completed record.
        "mode_declared": mode in {"baseline", "remediated"},
        "contract_schema_and_status_exact": contract_content.get("schema_version")
        == DIAGNOSTIC.CONTRACT_SCHEMA
        and contract_content.get("status") == "frozen_pre_remediation_build_and_pre_acquisition",
        "contract_sha256_matches_expected": True,
        "runtime_lock_schema_and_status_exact": runtime_content.get("schema_version")
        == DIAGNOSTIC.RUNTIME_LOCK_SCHEMA
        and runtime_content.get("status") == "pinned_for_acquisition",
        "runtime_lock_sha256_matches_expected": True,
        "runtime_lock_p23_addendum_matches_contract_and_source": p23_addendum_matches,
        "host_attestation_schema_and_status_exact": attestation_content.get("schema_version")
        == DIAGNOSTIC.HOST_ATTESTATION_SCHEMA
        and attestation_content.get("status") == "procedurally_host_attested",
        "host_attestation_sha256_matches_expected": True,
        "runtime_lock_host_attestation_backlink_exact": runtime_container.get(
            "host_attestation_sha256"
        )
        == attestation.get("expected_sha256"),
        "runtime_lock_and_attestation_container_exact": without_backlink
        == dict(attested_container),
        "runtime_lock_and_attestation_gpu_exact": runtime_content.get("gpu")
        == attestation_content.get("gpu"),
        "repository_clean": True,
        "repository_head_matches_expected": True,
        "repository_tree_matches_expected": True,
        "contract_source_records_exact": True,
        "container_hostname_matches_lock": live.get("hostname")
        == runtime_container.get("hostname"),
        "rootfs_declared_and_observed_read_only": runtime_container.get("rootfs_read_only") is True
        and root_mount.get("writable") is False,
        "live_mountinfo_matches_lock": live.get("mountinfo_sha256")
        == runtime_container.get("mountinfo_sha256"),
        "process_environment_matches_lock_exactly": live.get("environment") == expected_environment,
        "python_runtime_matches_lock": live.get("python") == software.get("python")
        and live.get("python_executable") == software.get("python_executable"),
        "repository_unchanged_after_trigger": repository_unchanged,
        "contract_sources_unchanged_after_trigger": contract_sources_match,
        "bound_artifacts_unchanged_after_trigger": bound_artifacts_unchanged,
        "torch_version_matches_lock": live.get("torch_version") == software.get("torch_version"),
        "torch_git_version_matches_lock": live.get("torch_git_version")
        == software.get("torch_git_version"),
        "compiled_cuda_matches_lock": live.get("torch_cuda_compiled_version")
        == software.get("torch_cuda_compiled_version"),
        "cuda_runtime_matches_lock": live.get("cuda_runtime_version")
        == software.get("cuda_runtime_version"),
        "cudnn_matches_lock": live.get("cudnn_version") == software.get("cudnn_version"),
        "single_live_gpu_matches_lock": live.get("cuda_available") is True
        and live.get("cuda_device_count") == 1
        and live.get("cuda_device_name") == gpu.get("name")
        and live.get("cuda_device_uuid") == gpu.get("uuid"),
        "torch_determinism_matches_lock": torch_determinism_matches,
        "remote_module_source_exact": validated_sources["remote_module"].get("sha256")
        == DIAGNOSTIC.REMOTE_MODULE_SHA256,
        "template_source_exact": validated_sources["template"].get("sha256")
        == DIAGNOSTIC.TEMPLATE_SHA256,
        "instantiator_source_exact_for_mode": validated_sources["instantiator"].get("sha256")
        == expected_instantiator,
        "generated_module_loaded_only_after_optimizer": generated_loaded_after_optimizer,
        "generated_module_bytes_exact": generated_bytes_exact,
        "mode_origin_matches": mode_origin_matches,
    }
    if set(reconstructed_checks) != COMPLETED_CHECK_NAMES:
        raise AssertionError("internal completed-check inventory differs")
    checks = payload.get("checks")
    if not isinstance(checks, Mapping) or set(checks) != COMPLETED_CHECK_NAMES:
        raise SanitizationError("diagnostic check inventory differs")
    if not all(isinstance(value, bool) for value in checks.values()):
        raise SanitizationError("diagnostic checks are non-Boolean")
    if dict(checks) != reconstructed_checks:
        mismatches = sorted(
            name for name in COMPLETED_CHECK_NAMES if checks.get(name) != reconstructed_checks[name]
        )
        raise SanitizationError(f"diagnostic checks differ from reconstruction: {mismatches}")
    passes = payload.get("passes")
    expected_passes = all(reconstructed_checks.values())
    if (
        not isinstance(passes, bool)
        or passes is not expected_passes
        or (payload.get("status") == "passes") is not expected_passes
    ):
        raise SanitizationError("diagnostic status, reconstructed checks and passes disagree")


def _validate_error(payload: Mapping[str, object]) -> None:
    process = _mapping(
        payload.get("process"),
        {"pid", "nonce", "failed_time_ns", "argv"},
        "error process",
    )
    _positive_int(process.get("pid"), "process.pid")
    _sha256(process.get("nonce"), "process.nonce")
    _positive_int(process.get("failed_time_ns"), "process.failed_time_ns")
    if not isinstance(process.get("argv"), list):
        raise SanitizationError("process.argv is not a list")

    repository = _mapping(
        payload.get("repository"),
        {"expected_head", "expected_tree", "observed"},
        "repository",
    )
    for field in ("expected_head", "expected_tree"):
        value = repository.get(field)
        if not isinstance(value, str) or _GIT_OBJECT.fullmatch(value) is None:
            raise SanitizationError(f"repository.{field} is invalid")
    observed = repository.get("observed")
    if not isinstance(observed, Mapping):
        raise SanitizationError("repository.observed is not a mapping")
    if observed.get("status") == "unavailable":
        _mapping(
            observed,
            {"path", "status", "error_class", "error_message"},
            "unavailable repository provenance",
        )
        if (
            not isinstance(observed.get("path"), str)
            or not Path(str(observed["path"])).is_absolute()
        ):
            raise SanitizationError("unavailable repository path is not absolute")
    else:
        _mapping(
            observed,
            {"path", "head", "tree", "dirty", "porcelain_v1_z_sha256"},
            "repository.observed",
        )
        for field in ("head", "tree"):
            value = observed.get(field)
            if not isinstance(value, str) or _GIT_OBJECT.fullmatch(value) is None:
                raise SanitizationError(f"repository.observed.{field} is invalid")
        _sha256(
            observed.get("porcelain_v1_z_sha256"),
            "repository.observed.porcelain_v1_z_sha256",
        )
        if not isinstance(observed.get("dirty"), bool):
            raise SanitizationError("repository.observed.dirty is not Boolean")

    artifacts = _mapping(
        payload.get("artifacts"),
        {"diagnostic_source", "contract", "runtime_lock", "host_attestation"},
        "artifacts",
    )
    for name, record in artifacts.items():
        _validate_best_effort_artifact(record, f"artifacts.{name}")

    contract = _mapping(
        payload.get("contract_binding"),
        {"expected_sha256", "content", "source_records"},
        "contract_binding",
    )
    runtime = _mapping(
        payload.get("runtime_lock_binding"),
        {"expected_sha256", "content"},
        "runtime_lock_binding",
    )
    attestation = _mapping(
        payload.get("host_attestation_binding"),
        {"expected_sha256", "content"},
        "host_attestation_binding",
    )
    for name, binding in (
        ("contract", contract),
        ("runtime_lock", runtime),
        ("host_attestation", attestation),
    ):
        _sha256(binding.get("expected_sha256"), f"{name} expected digest")
        if binding.get("content") is not None:
            raise SanitizationError(f"error {name} content must be null")
    if contract.get("source_records") != {}:
        raise SanitizationError("error contract source records must be empty")

    live = _mapping(
        payload.get("live_runtime"),
        {
            "hostname",
            "platform",
            "python",
            "python_executable",
            "environment",
            "mountinfo_sha256",
        },
        "error live_runtime",
    )
    if not isinstance(live.get("environment"), Mapping) or set(live["environment"]) != set(
        DIAGNOSTIC.DETERMINISM_ENVIRONMENT
    ):
        raise SanitizationError("error runtime environment inventory differs")
    mountinfo_sha256 = live.get("mountinfo_sha256")
    if mountinfo_sha256 is not None:
        _sha256(mountinfo_sha256, "error mountinfo digest")
    if payload.get("pinned_sources") != {}:
        raise SanitizationError("error pinned sources must be empty")
    if payload.get("trigger") is not None or payload.get("file_backed_module_closure") is not None:
        raise SanitizationError("error diagnostic cannot contain trigger or closure evidence")
    if payload.get("checks") != {"diagnostic_completed": False}:
        raise SanitizationError("error diagnostic checks differ")
    error = _mapping(payload.get("error"), {"class", "message", "traceback"}, "error")
    if (
        not all(isinstance(error.get(field), str) for field in error)
        or not error.get("class")
        or not error.get("traceback")
    ):
        raise SanitizationError("error record has non-string fields")


def validate_native(payload: object) -> Mapping[str, object]:
    """Validate all retained relations used by the offline redaction."""

    if not isinstance(payload, Mapping):
        raise SanitizationError("native diagnostic must be a JSON object")
    if payload.get("schema_version") != DIAGNOSTIC.SCHEMA:
        raise SanitizationError("native diagnostic schema differs")
    status = payload.get("status")
    if status == "error":
        _mapping(payload, ERROR_FIELDS, "error diagnostic")
        if payload.get("passes") is not False or not isinstance(payload.get("error"), Mapping):
            raise SanitizationError("error diagnostic status is inconsistent")
        _validate_error(payload)
    elif status in {"passes", "fails"}:
        _mapping(payload, SUCCESS_FIELDS, "completed diagnostic")
        _validate_success(payload)
    else:
        raise SanitizationError("native diagnostic status differs")
    if payload.get("mode") not in {"baseline", "remediated"}:
        raise SanitizationError("native diagnostic mode differs")
    if payload.get("seed") is not None or payload.get("seed_semantics") != (
        "not applicable: the diagnostic performs no random operation"
    ):
        raise SanitizationError("native diagnostic seed semantics differ")
    if not isinstance(payload.get("scope"), str) or not isinstance(
        payload.get("claim_boundary"), str
    ):
        raise SanitizationError("native diagnostic scope is absent")
    _reject_credentials(payload)
    return payload


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_roots(raw_roots: Mapping[str, Path]) -> dict[str, Path]:
    if set(raw_roots) != EXPECTED_ROOT_LABELS:
        raise SanitizationError("path-root labels differ from the frozen nine-label inventory")
    roots = {name: path.resolve(strict=True) for name, path in raw_roots.items()}
    for left_name, left in roots.items():
        for right_name, right in roots.items():
            if left_name >= right_name:
                continue
            if _is_within(left, right) or _is_within(right, left):
                raise SanitizationError("sanitization roots must be pairwise non-overlapping")
    return roots


def _sanitize_payload(
    payload: Mapping[str, object], roots: Mapping[str, Path]
) -> tuple[dict[str, object], int]:
    ordered_roots = sorted(roots.items(), key=lambda item: len(str(item[1])), reverse=True)
    replacement_count = 0

    def sanitize(value: object) -> object:
        nonlocal replacement_count
        if isinstance(value, Mapping):
            return {str(key): sanitize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [sanitize(item) for item in value]
        if not isinstance(value, str) or value in {SAFE_PATH_VALUE, "/"}:
            return value
        if Path(value).is_absolute():
            resolved = Path(value).resolve()
            matches: list[tuple[str, Path]] = []
            for name, root in ordered_roots:
                try:
                    matches.append((name, resolved.relative_to(root)))
                except ValueError:
                    continue
            if matches:
                name, relative = matches[0]
                replacement_count += 1
                return f"{name}:{relative.as_posix()}"
            replacement_count += 1
            digest = hashlib.sha256(value.encode("utf-8", errors="surrogatepass")).hexdigest()
            return f"external_absolute_path_sha256:{digest}"
        sanitized = value
        for name, root in ordered_roots:
            raw_root = str(root)
            occurrences = sanitized.count(raw_root)
            if occurrences:
                sanitized = sanitized.replace(raw_root, f"{name}:")
                sanitized = sanitized.replace(f"{name}:/", f"{name}:")
                replacement_count += occurrences

        def redact_external_path(match: re.Match[str]) -> str:
            nonlocal replacement_count
            raw_path = match.group(0)
            replacement_count += 1
            digest = hashlib.sha256(raw_path.encode("utf-8", errors="surrogatepass")).hexdigest()
            return f"external_absolute_path_sha256:{digest}"

        sanitized = _EMBEDDED_ABSOLUTE_PATH.sub(redact_external_path, sanitized)
        return sanitized

    result = sanitize(payload)
    assert isinstance(result, dict)
    rendered = json.dumps(result, sort_keys=True, allow_nan=False)
    for root in roots.values():
        if str(root) in rendered:
            raise SanitizationError("redacted diagnostic retains a declared native path")
    return result, replacement_count


def sanitize_native(
    *,
    native: Path,
    expected_native_sha256: str,
    output: Path,
    path_roots: Mapping[str, Path],
) -> dict[str, object]:
    """Validate, redact and publish one diagnostic without overwriting bytes."""

    _sha256(expected_native_sha256, "expected native digest")
    roots = _validate_roots(path_roots)
    raw_native = native
    if raw_native.is_symlink():
        raise SanitizationError("native diagnostic must not be a symbolic link")
    native = raw_native.resolve(strict=True)
    output = output.resolve()
    if not native.is_file():
        raise SanitizationError("native diagnostic must be a retained regular file")
    if not _is_within(native, roots["native"]) or not _is_within(output, roots["native"]):
        raise SanitizationError("native and redacted diagnostics must lie in the evidence root")
    if native == output or output.exists() or output.is_symlink():
        raise SanitizationError("redacted output must be a fresh distinct path")
    binding, native_bytes = DIAGNOSTIC._stable_file_binding(native, "native diagnostic")
    if binding["sha256"] != expected_native_sha256:
        raise SanitizationError("native diagnostic differs from its expected digest")
    try:
        payload = json.loads(
            native_bytes,
            parse_constant=lambda token: (_ for _ in ()).throw(
                SanitizationError(f"nonfinite JSON constant is forbidden: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SanitizationError("native diagnostic is not valid UTF-8 JSON") from error
    validated = validate_native(payload)
    sanitized, replacement_count = _sanitize_payload(validated, roots)
    result = {
        "schema_version": SANITIZED_SCHEMA,
        "native_artifact_sha256": expected_native_sha256,
        "native_artifact_byte_count": len(native_bytes),
        "native_schema_version": DIAGNOSTIC.SCHEMA,
        "mode": validated["mode"],
        "status": validated["status"],
        "logical_root_labels": sorted(roots),
        "path_replacement_count": replacement_count,
        "manifest": sanitized,
        "claim_boundary": (
            "Offline, byte-bound redaction of one native executable-origin diagnostic; not a "
            "CUDA acquisition, fidelity result, or training result."
        ),
    }
    DIAGNOSTIC._atomic_write_json(output, result)
    return result


def _path_roots(values: list[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path or name in roots:
            raise SanitizationError("path roots must be unique NAME=ABSOLUTE_PATH entries")
        path = Path(raw_path)
        if not path.is_absolute():
            raise SanitizationError("path roots must be absolute")
        roots[name] = path
    return roots


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--expected-native-sha256", required=True)
    parser.add_argument("--path-root", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    try:
        result = sanitize_native(
            native=arguments.native,
            expected_native_sha256=arguments.expected_native_sha256,
            output=arguments.output,
            path_roots=_path_roots(arguments.path_root),
        )
    except (OSError, ValueError, TypeError, SanitizationError) as error:
        print(f"P24 diagnostic sanitization blocked: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": str(arguments.output.resolve()),
                "output_sha256": DIAGNOSTIC.sha256_file(arguments.output.resolve()),
                "native_artifact_sha256": result["native_artifact_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
