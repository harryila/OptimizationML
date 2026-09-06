#!/usr/bin/env python3
"""P23 provenance, identity, and exact-comparison primitives.

P23 inherits the frozen P22 acquisition, but permits only a single pinned
CUDA device.  This module deliberately contains no training loop.  It gives
the CUDA runner a fail-closed provenance boundary and gives a separate
verifier enough information to reconstruct the execution identity without
trusting an opaque digest stored by the runner.

The collectors accept injected command runners, module maps, and runtime
probes so their failure behavior can be tested on hosts without CUDA.
"""

from __future__ import annotations

import copy
import csv
import ctypes
import ctypes.util
import hashlib
import importlib.util
import io
import json
import os
import platform
import re
import socket
import stat
import subprocess
import sys
import tempfile
from base64 import b64encode
from collections.abc import Callable, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any, Final
from urllib.parse import urlsplit

import numpy as np

P23_ADDENDUM_SCHEMA: Final = "passive-muon-p23-deterministic-cuda-shadow-trace-addendum-v1"
P23_EXECUTION_IDENTITY_SCHEMA: Final = "passive-muon-p23-execution-identity-v2"
P23_RUN_MANIFEST_SCHEMA: Final = "passive-muon-p23-cuda-run-manifest-v1"
P23_REPEATABILITY_SCHEMA: Final = "passive-muon-p23-cuda-repeatability-v1"
P23_NONINTERFERENCE_SCHEMA: Final = "passive-muon-p23-cuda-noninterference-v1"
P23_SANITIZED_MANIFEST_SCHEMA: Final = "passive-muon-p23-sanitized-complete-manifest-v1"
P23_RUNTIME_LOCK_SCHEMA: Final = "passive-muon-p23-cuda-runtime-lock-v3"
P23_HOST_ATTESTATION_SCHEMA: Final = "passive-muon-p23-host-attestation-v3"
P23_LOADED_FILE_CLOSURE_SCHEMA: Final = "passive-muon-p23-loaded-file-closure-v1"
P23_LOADED_FILE_SNAPSHOT_SCHEMA: Final = "passive-muon-p23-loaded-file-snapshot-v1"

PINNED_PYTHON_ENVIRONMENT: Final = "/opt/p23-venv"
PINNED_PYTHON_EXECUTABLE: Final = f"{PINNED_PYTHON_ENVIRONMENT}/bin/python"
PINNED_PYTHON_MAJOR_MINOR: Final = (3, 12)
PINNED_EXECUTABLE_PATH: Final = (
    "/opt/p23-venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)
PINNED_CONTAINER_NETWORK_MODE: Final = "none"
DOCKER_GPU_SELECTION_MECHANISM: Final = "docker_device_request_single_full_gpu"
FROZEN_PYTHON_FLAGS: Final = {
    "debug": 0,
    "inspect": 0,
    "interactive": 0,
    "optimize": 0,
    "dont_write_bytecode": 1,
    "no_user_site": 1,
    "no_site": 0,
    "ignore_environment": 0,
    "verbose": 0,
    "bytes_warning": 0,
    "quiet": 0,
    "isolated": 0,
    "hash_randomization": 1,
    "dev_mode": False,
    "utf8_mode": 0,
    "warn_default_encoding": 0,
    "safe_path": False,
    "int_max_str_digits": 4300,
}

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[2]
DEFAULT_ADDENDUM_PATH: Final = (
    REPOSITORY_ROOT / "experiments/training/p23_deterministic_cuda_shadow_trace_addendum.json"
)

EXPECTED_OPTIMIZER_STEPS: Final = 256
EXPECTED_SEQUENCE_LENGTH: Final = 128
EXPECTED_CAPTURE_STEPS: Final = (
    *range(0, 8),
    *range(124, 132),
    *range(248, 256),
)
EXPECTED_PARAMETER_COUNT: Final = 48
EXPECTED_OBSERVATION_COUNT: Final = len(EXPECTED_CAPTURE_STEPS) * EXPECTED_PARAMETER_COUNT

TRACE_OFF_OBSERVATION: Final = {
    "status": "not_observed",
    "capture_step_count": 0,
    "observation_count": 0,
}
TRACE_ON_OBSERVATION: Final = {
    "status": "observed_actual_post_aspect_cuda",
    "capture_step_count": len(EXPECTED_CAPTURE_STEPS),
    "observation_count": EXPECTED_OBSERVATION_COUNT,
}

REQUIRED_ADDENDUM_MAPPINGS: Final = (
    "inherits",
    "claim_boundary",
    "cuda_environment",
    "source_provenance",
    "model_optimizer_binding",
    "capture_semantics",
    "run_sequence",
    "exact_comparison",
    "acceptance_gates",
    "fail_closed",
)
REQUIRED_INHERITED_ARTIFACTS: Final = (
    "p22_protocol",
    "p22_erratum",
    "p21_fidelity_gates",
    "p22_fused_qkv_certificate",
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_OBJECT = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_OCI_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_PINNED_OCI_IMAGE = re.compile(r"^[^@\s]+@sha256:[0-9a-f]{64}$")
_CONTAINER_ID = re.compile(r"^[0-9a-f]{64}$")
_GPU_UUID = re.compile(
    r"^GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
    r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$"
)
_PERSONAL_PATH_FRAGMENTS: Final = ("/Users/", "/home/", "/private/tmp/", "\\Users\\")
_GIT_SCP_ORIGIN = re.compile(r"^(?P<user>[^@/:]+)@(?P<host>[^:/]+):(?P<path>.+)$")

CONTAINER_ATTESTATION_SCOPE: Final = (
    "procedural_host_inspection_not_cryptographic_remote_attestation"
)
REQUIRED_CONTAINER_MOUNTS: Final = (
    (
        "repository",
        "/workspace/OptimizationML",
        True,
    ),
    ("nanogpt", "/workspace/inputs/nanoGPT", True),
    ("muon", "/workspace/inputs/muon", True),
    ("data", "/private/tmp/optimizationml-p22-data", True),
    (
        "preprocessor_alias",
        "/Users/harry/Desktop/temp/OptimizationML/experiments/training/materialize_p22_fineweb.py",
        True,
    ),
    ("native_evidence", "/workspace/evidence/p23", False),
    ("image_inspection", "/mounted-host-evidence/image-inspect.json", True),
    (
        "running_container_inspection",
        "/mounted-host-evidence/running-container-inspect.json",
        True,
    ),
    ("running_mountinfo", "/mounted-host-evidence/running-mountinfo.txt", True),
    ("nvidia_smi_query", "/mounted-host-evidence/nvidia-smi.csv", True),
)
_REQUIRED_MOUNT_BY_DESTINATION: Final = {
    destination: {"name": name, "read_only": read_only}
    for name, destination, read_only in REQUIRED_CONTAINER_MOUNTS
}
_SANITIZED_MOUNT_CONTRACT: Final = {
    "sha256": hashlib.sha256(
        json.dumps(
            REQUIRED_CONTAINER_MOUNTS,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest(),
    "mounts": [
        {
            "name": name,
            "type": "bind",
            "read_only": read_only,
            "propagation": "rprivate",
        }
        for name, _destination, read_only in REQUIRED_CONTAINER_MOUNTS
    ],
}
REQUIRED_CONTAINER_TMPFS: Final = {
    "/tmp": "rw,noexec,nosuid,nodev,size=1073741824",
}
_SANITIZED_TMPFS_CONTRACT: Final = {
    "sha256": hashlib.sha256(
        json.dumps(
            REQUIRED_CONTAINER_TMPFS,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest(),
    "mounts": copy.deepcopy(REQUIRED_CONTAINER_TMPFS),
}


class P23ProvenanceError(ValueError):
    """Raised when a P23 artifact or execution environment is not pinned."""


CommandRunner = Callable[[tuple[str, ...], Path | None], bytes]

_MODULE_ARTIFACT_ROLES: Final = (
    "module_file",
    "spec_origin",
    "cached_bytecode",
    "source_file",
)
_ALLOWED_EXECUTABLE_SCOPES: Final = {
    "repository",
    "nanogpt",
    "muon",
    "python_environment",
    "image",
}


def sha256_file(path: Path) -> str:
    """Hash a file in bounded-size blocks."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(2**20):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    """Hash the canonical JSON representation used by P23 identities."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _p22_initial_state_sha256(initial_state: Mapping[str, object]) -> str:
    """Independently replay P22's typed hash for its three initial digests."""

    expected = {
        "model_state_sha256",
        "optimizer_state_sha256",
        "rng_state_sha256",
    }
    if set(initial_state) != expected or any(
        not _is_sha256(initial_state.get(field)) for field in expected
    ):
        raise P23ProvenanceError("P23 manifest has an invalid exact initial state")

    digest = hashlib.sha256()

    def emit(tag: bytes, payload: bytes) -> None:
        digest.update(tag)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    emit(b"mapping-size", b"3")
    replay = {
        "model_state_sha256": initial_state["model_state_sha256"],
        "optimizer_state_sha256": initial_state["optimizer_state_sha256"],
        "rng": initial_state["rng_state_sha256"],
    }
    for key in sorted(replay):
        emit(b"str", key.encode("utf-8"))
        value = replay[key]
        assert isinstance(value, str)
        emit(b"str", value.encode("utf-8"))
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _validate_pinned_image_reference(image: object, repository_digest: object) -> str:
    if not isinstance(image, str) or _PINNED_OCI_IMAGE.fullmatch(image) is None:
        raise P23ProvenanceError("container image must be one digest-pinned OCI reference")
    if not isinstance(repository_digest, str) or _OCI_DIGEST.fullmatch(repository_digest) is None:
        raise P23ProvenanceError("container repository digest is not an OCI SHA-256 digest")
    if image.rsplit("@", 1)[1] != repository_digest:
        raise P23ProvenanceError("container image and repository digest differ")
    return image


def _validate_mount_contract_record(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {"sha256", "mounts"}:
        raise P23ProvenanceError("container mount contract is malformed")
    if dict(value) != _SANITIZED_MOUNT_CONTRACT:
        raise P23ProvenanceError("container mount contract differs from the exact allowlist")
    return copy.deepcopy(_SANITIZED_MOUNT_CONTRACT)


def _validate_tmpfs_contract_record(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {"sha256", "mounts"}:
        raise P23ProvenanceError("container tmpfs contract is malformed")
    if dict(value) != _SANITIZED_TMPFS_CONTRACT:
        raise P23ProvenanceError("container tmpfs contract differs from the exact allowlist")
    return copy.deepcopy(_SANITIZED_TMPFS_CONTRACT)


def _single_inspection_record(raw: bytes, label: str) -> Mapping[str, object]:
    try:
        decoded = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise P23ProvenanceError(f"{label} is not valid JSON") from error
    records = decoded if isinstance(decoded, list) else [decoded]
    if len(records) != 1 or not isinstance(records[0], Mapping):
        raise P23ProvenanceError(f"{label} must contain exactly one object")
    return records[0]


def _validate_running_container_mounts(value: object) -> dict[str, object]:
    if not isinstance(value, list):
        raise P23ProvenanceError("running-container inspection has no mount list")
    observed: dict[str, Mapping[str, object]] = {}
    for item in value:
        if not isinstance(item, Mapping):
            raise P23ProvenanceError("running-container inspection has a malformed mount")
        destination = item.get("Destination")
        if not isinstance(destination, str) or not destination.startswith("/"):
            raise P23ProvenanceError("running-container mount has no absolute destination")
        if destination in observed:
            raise P23ProvenanceError("running-container inspection repeats a mount destination")
        observed[destination] = item
    if set(observed) != set(_REQUIRED_MOUNT_BY_DESTINATION):
        missing = sorted(set(_REQUIRED_MOUNT_BY_DESTINATION) - set(observed))
        extra = sorted(set(observed) - set(_REQUIRED_MOUNT_BY_DESTINATION))
        raise P23ProvenanceError(
            "running-container mounts differ from the exact allowlist: "
            f"missing={missing}, extra={extra}"
        )
    for destination, expected in _REQUIRED_MOUNT_BY_DESTINATION.items():
        mount = observed[destination]
        source = mount.get("Source")
        if (
            mount.get("Type") != "bind"
            or not isinstance(source, str)
            or not source.startswith("/")
            or mount.get("RW") is not (not expected["read_only"])
            or mount.get("Propagation") != "rprivate"
        ):
            raise P23ProvenanceError(
                f"running-container mount {expected['name']!r} has the wrong type or mode"
            )
    return copy.deepcopy(_SANITIZED_MOUNT_CONTRACT)


def _container_environment_value(value: object, name: str) -> str:
    """Return one exact value from Docker's inspected ``Config.Env`` list."""

    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise P23ProvenanceError("running-container inspection has no canonical environment list")
    prefix = f"{name}="
    matches = [item[len(prefix) :] for item in value if item.startswith(prefix)]
    if len(matches) != 1:
        raise P23ProvenanceError(
            f"running-container inspection must contain exactly one {name} entry"
        )
    return matches[0]


def _validate_running_container_gpu_selection(
    config: Mapping[str, object], host_config: Mapping[str, object]
) -> dict[str, object]:
    """Validate Docker's host-side one-full-GPU request before trusting CUDA.

    Docker 29's native CDI path retains the requested UUID in ``Config.Env``
    and ``HostConfig.DeviceRequests``. The long-lived container's PID 1 may
    receive an implementation-internal ``NVIDIA_VISIBLE_DEVICES=void``, but
    every P23 evidence role is a fresh ``docker exec`` process and receives
    the inspected UUID-valued configuration. The separately collected CUDA
    identity must agree with the request exactly.
    """

    cuda_visible = _container_environment_value(config.get("Env"), "CUDA_VISIBLE_DEVICES")
    nvidia_visible = _container_environment_value(config.get("Env"), "NVIDIA_VISIBLE_DEVICES")
    if _GPU_UUID.fullmatch(cuda_visible) is None or nvidia_visible != cuda_visible:
        raise P23ProvenanceError(
            "running-container configuration must request one identical full GPU UUID"
        )
    requests = host_config.get("DeviceRequests")
    if not isinstance(requests, list) or len(requests) != 1:
        raise P23ProvenanceError("running container must have exactly one GPU DeviceRequest")
    request = requests[0]
    expected_request: dict[str, object] = {
        "Driver": "",
        "Count": 0,
        "DeviceIDs": [cuda_visible],
        "Capabilities": [["gpu"]],
        "Options": {},
    }
    if (
        not isinstance(request, Mapping)
        or type(request.get("Count")) is not int
        or dict(request) != expected_request
    ):
        raise P23ProvenanceError(
            "running-container GPU DeviceRequest does not pin the declared full GPU UUID"
        )
    return {
        "mechanism": DOCKER_GPU_SELECTION_MECHANISM,
        "requested_full_gpu_uuid": cuda_visible,
        "device_request": copy.deepcopy(expected_request),
        "container_config_cuda_visible_devices": cuda_visible,
        "container_config_nvidia_visible_devices": nvidia_visible,
    }


def _validate_gpu_selection_record(value: object) -> dict[str, object]:
    """Validate the sanitized Docker/CDI GPU-selection record."""

    expected_fields = {
        "mechanism",
        "requested_full_gpu_uuid",
        "device_request",
        "container_config_cuda_visible_devices",
        "container_config_nvidia_visible_devices",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise P23ProvenanceError("container GPU-selection fields do not match the frozen schema")
    result = copy.deepcopy(dict(value))
    uuid = result.get("requested_full_gpu_uuid")
    if _GPU_UUID.fullmatch(str(uuid)) is None:
        raise P23ProvenanceError("container GPU selection has no canonical full GPU UUID")
    expected_request = {
        "Driver": "",
        "Count": 0,
        "DeviceIDs": [uuid],
        "Capabilities": [["gpu"]],
        "Options": {},
    }
    device_request = result.get("device_request")
    if (
        result.get("mechanism") != DOCKER_GPU_SELECTION_MECHANISM
        or not isinstance(device_request, Mapping)
        or type(device_request.get("Count")) is not int
        or device_request != expected_request
        or result.get("container_config_cuda_visible_devices") != uuid
        or result.get("container_config_nvidia_visible_devices") != uuid
    ):
        raise P23ProvenanceError("container GPU selection is not the pinned Docker/CDI request")
    return result


def _validate_container_identity(
    value: object,
    *,
    require_host_attestation_sha256: bool,
) -> dict[str, object]:
    expected_fields = {
        "image",
        "repository_digest",
        "container_id",
        "image_id",
        "container_init_pid",
        "hostname",
        "default_hostname",
        "rootfs_read_only",
        "network_mode",
        "gpu_selection",
        "mountinfo_sha256",
        "mount_contract",
        "tmpfs_contract",
        "attestation_scope",
    }
    if require_host_attestation_sha256:
        expected_fields.add("host_attestation_sha256")
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise P23ProvenanceError("container identity fields do not match the frozen schema")
    result = copy.deepcopy(dict(value))
    _validate_pinned_image_reference(result.get("image"), result.get("repository_digest"))
    container_id = result.get("container_id")
    image_id = result.get("image_id")
    hostname = result.get("hostname")
    if not isinstance(container_id, str) or _CONTAINER_ID.fullmatch(container_id) is None:
        raise P23ProvenanceError("container identity has no canonical 64-hex container ID")
    if not isinstance(image_id, str) or _OCI_DIGEST.fullmatch(image_id) is None:
        raise P23ProvenanceError("container identity has no immutable image ID")
    container_init_pid = result.get("container_init_pid")
    if (
        not isinstance(container_init_pid, int)
        or isinstance(container_init_pid, bool)
        or container_init_pid <= 0
    ):
        raise P23ProvenanceError("container identity has no valid host init PID")
    if not _is_sha256(result.get("mountinfo_sha256")):
        raise P23ProvenanceError("container identity has no live mountinfo SHA-256")
    if hostname != container_id[:12] or result.get("default_hostname") is not True:
        raise P23ProvenanceError("container identity does not use its canonical default hostname")
    if result.get("rootfs_read_only") is not True:
        raise P23ProvenanceError("container root filesystem is not read-only")
    if result.get("network_mode") != PINNED_CONTAINER_NETWORK_MODE:
        raise P23ProvenanceError("container network mode is not disabled")
    _validate_gpu_selection_record(result.get("gpu_selection"))
    if result.get("attestation_scope") != CONTAINER_ATTESTATION_SCOPE:
        raise P23ProvenanceError("container identity overstates the procedural attestation scope")
    _validate_mount_contract_record(result.get("mount_contract"))
    _validate_tmpfs_contract_record(result.get("tmpfs_contract"))
    if require_host_attestation_sha256 and not _is_sha256(result.get("host_attestation_sha256")):
        raise P23ProvenanceError("container host attestation SHA-256 is missing")
    return result


def _read_json_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise P23ProvenanceError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise P23ProvenanceError(f"{label} must be a JSON mapping")
    return value


def _repository_path(root: Path, raw_path: object, label: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise P23ProvenanceError(f"{label} has no repository path")
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise P23ProvenanceError(f"{label} path must remain repository-relative")
    root = root.resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise P23ProvenanceError(f"{label} escapes the repository") from error
    return resolved


def load_and_validate_addendum(
    path: Path = DEFAULT_ADDENDUM_PATH,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Load the frozen addendum and verify every inherited artifact hash."""

    path = path.resolve()
    payload = _read_json_mapping(path, "P23 addendum")
    if payload.get("schema_version") != P23_ADDENDUM_SCHEMA:
        raise P23ProvenanceError("P23 addendum schema mismatch")
    if payload.get("status") != "frozen_pre_acquisition_cuda_runtime_lock_pending":
        raise P23ProvenanceError("P23 addendum is not the frozen pre-acquisition contract")
    for key in REQUIRED_ADDENDUM_MAPPINGS:
        if not isinstance(payload.get(key), dict):
            raise P23ProvenanceError(f"P23 addendum field {key!r} must be a mapping")

    inherits = payload["inherits"]
    missing = set(REQUIRED_INHERITED_ARTIFACTS) - set(inherits)
    if missing:
        raise P23ProvenanceError(f"P23 addendum omits inherited artifacts: {sorted(missing)}")
    for name in REQUIRED_INHERITED_ARTIFACTS:
        record = inherits[name]
        if not isinstance(record, dict):
            raise P23ProvenanceError(f"inherited artifact {name!r} must be a mapping")
        if record.get("mutation_allowed") is not False:
            raise P23ProvenanceError(f"inherited artifact {name!r} is not immutable")
        artifact = _repository_path(repository_root, record.get("path"), name)
        expected = record.get("sha256")
        if not _is_sha256(expected):
            raise P23ProvenanceError(f"inherited artifact {name!r} has no valid SHA-256")
        if not artifact.is_file():
            raise P23ProvenanceError(f"inherited artifact {name!r} is missing")
        if sha256_file(artifact) != expected:
            raise P23ProvenanceError(f"inherited artifact {name!r} SHA-256 mismatch")

    parent = payload.get("provenance_parent")
    expected_parent = {
        "branch": "p22-real-gradient-shadow-trace",
        "diagnostic_commit": "10d3c8becb98267135c29cb504a0ab5d6e5ff886",
        "diagnostic_tag": "p22-real-gradient-shadow-trace-diagnostic",
    }
    if parent != expected_parent:
        raise P23ProvenanceError("P23 provenance parent changed")

    capture = payload["capture_semantics"]
    expected_addendum_off = {
        "status": "not_observed",
        "capture_steps": 0,
        "observation_count": 0,
        "actual_post_aspect_candidate_count": 0,
    }
    expected_addendum_on = {
        "status": "observed",
        "capture_steps": len(EXPECTED_CAPTURE_STEPS),
        "parameters_per_capture": EXPECTED_PARAMETER_COUNT,
        "observation_count": EXPECTED_OBSERVATION_COUNT,
        "actual_post_aspect_candidate_count": EXPECTED_OBSERVATION_COUNT,
        "candidate": ("actual stored BF16 CUDA post-aspect value returned by the pinned Muon call"),
    }
    if capture.get("trace_off") != expected_addendum_off:
        raise P23ProvenanceError("P23 trace-off capture semantics changed")
    if capture.get("trace_on") != expected_addendum_on:
        raise P23ProvenanceError("P23 trace-on capture semantics changed")

    sequence = payload["run_sequence"]
    if sequence.get("process_isolation") != (
        "each acquisition is a fresh Python process in the same host-inspected "
        "container ID, using the same committed tree, runtime lock, data, and source bytes"
    ):
        raise P23ProvenanceError("P23 process-isolation contract changed")
    if sequence.get("ordered_roles") != [
        "trace_off_a",
        "trace_off_b",
        "verify_repeatability",
        "trace_on",
        "verify_noninterference",
        "aggregate_fidelity",
    ]:
        raise P23ProvenanceError("P23 fresh-process trace order changed")

    cuda = payload["cuda_environment"]
    if (
        cuda.get("backend") != "cuda"
        or cuda.get("device_count") != 1
        or cuda.get("network_mode") != PINNED_CONTAINER_NETWORK_MODE
    ):
        raise P23ProvenanceError("P23 requires exactly one visible CUDA device")
    if cuda.get("docker_gpu_selection") != (
        "host inspection must show exactly one DeviceRequest with Driver empty, Count 0, "
        "DeviceIDs containing only the same full-GPU UUID, Capabilities exactly [[gpu]], "
        "and Options empty"
    ) or cuda.get("docker_exec_visibility") != (
        "every freeze, acquisition, verifier, and aggregation role runs through docker exec "
        "and must observe CUDA_VISIBLE_DEVICES and NVIDIA_VISIBLE_DEVICES equal to the requested "
        "full-GPU UUID; the long-lived shell PID 1 is not an evidence role"
    ):
        raise P23ProvenanceError("P23 Docker GPU-selection contract changed")
    expected_python_contract = {
        "python_executable": PINNED_PYTHON_EXECUTABLE,
        "python_executable_sha256": "required in the populated runtime lock",
        "python_major_minor": "3.12",
        "virtual_environment": PINNED_PYTHON_ENVIRONMENT,
        "path": PINNED_EXECUTABLE_PATH,
        "python_no_user_site": "1",
        "python_dont_write_bytecode": "1",
        "python_optimize": None,
        "python_flags": (
            "the complete Python 3.12 sys.flags map is pinned to the ordinary "
            "nonoptimized, nonisolated image invocation"
        ),
        "pythonpath": None,
        "pythonhome": None,
        "repository_virtual_environment_allowed": False,
        "runtime_dependency_sync_allowed": False,
    }
    if any(cuda.get(key) != value for key, value in expected_python_contract.items()):
        raise P23ProvenanceError("P23 image-resident Python contract changed")
    if cuda.get("sdpa_backend") != "math" or any(
        cuda.get(key) is not expected
        for key, expected in (
            ("flash_sdpa", False),
            ("memory_efficient_sdpa", False),
            ("cudnn_sdpa", False),
            ("math_sdpa", True),
            ("deterministic_algorithms", True),
            ("tf32", False),
        )
    ):
        raise P23ProvenanceError("P23 CUDA/SDPA arithmetic contract changed")
    exact = payload["exact_comparison"]
    if exact.get("numeric_tolerance") is not None or exact.get("allclose_allowed") is not False:
        raise P23ProvenanceError("P23 comparisons must be bitwise exact")
    return payload


def _default_command_runner(arguments: tuple[str, ...], cwd: Path | None) -> bytes:
    try:
        completed = subprocess.run(
            list(arguments),
            cwd=cwd,
            check=False,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise P23ProvenanceError(f"command failed: {arguments[0]}: {error}") from error
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).decode(errors="replace").strip()
        raise P23ProvenanceError(
            f"command failed ({completed.returncode}): {' '.join(arguments)}: {detail}"
        )
    return completed.stdout


def _credential_safe_git_origin(raw_origin: str) -> str:
    """Return a reproducible remote URL without retaining embedded credentials."""

    origin = raw_origin.strip()
    if not origin or any(ord(character) < 32 for character in origin):
        raise P23ProvenanceError("Git origin is empty or contains control characters")
    scp_match = _GIT_SCP_ORIGIN.fullmatch(origin)
    if scp_match is not None:
        # ``git@host:path`` is the conventional credential-free SSH spelling.
        # Reject arbitrary account names because they can identify a reviewer or
        # acquisition host even when they are not authentication secrets.
        if scp_match.group("user") != "git" or not scp_match.group("path"):
            raise P23ProvenanceError("Git origin contains disallowed user information")
        return origin
    parsed = urlsplit(origin)
    if parsed.scheme not in {"https", "ssh", "git"} or not parsed.hostname:
        raise P23ProvenanceError("Git origin must be a credential-free network URL")
    if parsed.password is not None or (parsed.username is not None and parsed.username != "git"):
        raise P23ProvenanceError("Git origin contains embedded credentials or user information")
    if parsed.query or parsed.fragment:
        raise P23ProvenanceError("Git origin query/fragment is forbidden")
    return origin


def _reject_credentialed_origin_fields(value: object) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) == "origin":
                if not isinstance(item, str):
                    raise P23ProvenanceError("Git origin field is not a string")
                _credential_safe_git_origin(item)
            else:
                _reject_credentialed_origin_fields(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_credentialed_origin_fields(item)


def _hash_tracked_files(root: Path, tracked_names: list[str]) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for relative_name in tracked_names:
        relative = Path(relative_name)
        if relative.is_absolute() or ".." in relative.parts:
            raise P23ProvenanceError("Git reported a path outside the repository")
        path = root / relative
        if path.is_symlink():
            stored_bytes = os.readlink(path).encode("utf-8", errors="surrogateescape")
            kind = "symlink"
        elif path.is_file():
            stored_bytes = path.read_bytes()
            kind = "file"
        elif path.is_dir():
            # A gitlink is represented by the independently checked recursive
            # submodule status rather than a repository-owned byte stream.
            continue
        else:
            raise P23ProvenanceError(f"tracked repository path is unavailable: {relative_name}")
        records[relative.as_posix()] = {
            "sha256": hashlib.sha256(stored_bytes).hexdigest(),
            "byte_count": len(stored_bytes),
            "kind": kind,
        }
    return records


def collect_git_provenance(
    repository_root: Path = REPOSITORY_ROOT,
    *,
    command_runner: CommandRunner = _default_command_runner,
) -> dict[str, Any]:
    """Collect contemporaneous clean Git identity and reject dirty trees."""

    root = repository_root.resolve()
    repository_venv = root / ".venv"
    if repository_venv.exists() or repository_venv.is_symlink():
        raise P23ProvenanceError(
            "P23 acquisition forbids a repository-local .venv; use the pinned image environment"
        )

    def git(*arguments: str) -> bytes:
        return command_runner(("git", *arguments), root)

    top_level = Path(git("rev-parse", "--show-toplevel").decode().strip()).resolve()
    if top_level != root:
        raise P23ProvenanceError("repository root does not match git --show-toplevel")
    head = git("rev-parse", "HEAD").decode().strip()
    tree = git("rev-parse", "HEAD^{tree}").decode().strip()
    if _GIT_OBJECT.fullmatch(head) is None or _GIT_OBJECT.fullmatch(tree) is None:
        raise P23ProvenanceError("Git HEAD/tree is not a valid object identifier")
    origin = _credential_safe_git_origin(git("remote", "get-url", "origin").decode())
    status = git("status", "--porcelain=v1", "--untracked-files=all", "-z")
    if status:
        entries = [item.decode(errors="surrogateescape") for item in status.split(b"\0") if item]
        raise P23ProvenanceError(f"P23 acquisition requires a clean worktree: {entries}")
    submodules_raw = git("submodule", "status", "--recursive")
    submodules = [line for line in submodules_raw.decode().splitlines() if line]
    if any(line.startswith(("-", "+", "U")) for line in submodules):
        raise P23ProvenanceError("Git submodule state is uninitialized, dirty, or conflicted")
    tracked_raw = git("ls-files", "-z")
    tracked_names = [
        item.decode(errors="surrogateescape") for item in tracked_raw.split(b"\0") if item
    ]
    if not tracked_names or tracked_names != sorted(tracked_names):
        raise P23ProvenanceError("Git tracked-file inventory is empty or not sorted")
    tracked_files = _hash_tracked_files(root, tracked_names)
    # Close the check/hash race: the bytes above are admissible only when the
    # same HEAD, tree, and clean status still hold after the inventory pass.
    if git("rev-parse", "HEAD").decode().strip() != head:
        raise P23ProvenanceError("Git HEAD changed while provenance was collected")
    if git("rev-parse", "HEAD^{tree}").decode().strip() != tree:
        raise P23ProvenanceError("Git tree changed while provenance was collected")
    final_status = git("status", "--porcelain=v1", "--untracked-files=all", "-z")
    if final_status != status:
        raise P23ProvenanceError("Git worktree changed while provenance was collected")
    final_submodules_raw = git("submodule", "status", "--recursive")
    if final_submodules_raw != submodules_raw:
        raise P23ProvenanceError("Git submodule state changed while provenance was collected")
    final_tracked_raw = git("ls-files", "-z")
    if final_tracked_raw != tracked_raw:
        raise P23ProvenanceError(
            "Git tracked-file inventory changed while provenance was collected"
        )
    if _hash_tracked_files(root, tracked_names) != tracked_files:
        raise P23ProvenanceError("tracked repository bytes changed while provenance was collected")
    return {
        "head": head,
        "tree": tree,
        "origin": origin,
        "clean": True,
        "status_porcelain_v1_z_base64": b64encode(status).decode("ascii"),
        "status_porcelain_v1_z_sha256": hashlib.sha256(status).hexdigest(),
        "status_entries": [],
        "submodule_status": submodules,
        "submodule_status_sha256": hashlib.sha256(submodules_raw).hexdigest(),
        "tracked_file_count": len(tracked_files),
        "tracked_files": tracked_files,
        "tracked_files_sha256": canonical_json_sha256(tracked_files),
    }


def _module_source_path(
    module: ModuleType,
    *,
    scopes: Mapping[str, Path],
) -> Path | None:
    """Resolve a real module file without mistaking lazy module proxies for files.

    Torch exposes ``torch.ops`` and ``torch.classes`` as ``ModuleType``
    subclasses whose synthetic, relative ``__file__`` values do not exist.
    Those are not file-backed modules.  Missing absolute files that claim to
    live in one of the controlled source scopes remain hard failures.
    """

    raw = getattr(module, "__file__", None)
    if not isinstance(raw, str) or not raw:
        return None
    if raw.startswith("<") and raw.endswith(">"):
        return None
    path = Path(raw)
    spec = getattr(module, "__spec__", None)
    origin = getattr(spec, "origin", None)
    if not path.is_absolute() and not path.exists():
        if isinstance(origin, str) and origin and origin not in {"built-in", "frozen"}:
            path = Path(origin)
        else:
            return None
    if path.suffix in {".pyc", ".pyo"}:
        try:
            source = Path(importlib.util.source_from_cache(str(path)))
        except ValueError:
            source = path
        if source.is_file():
            path = source
    try:
        return path.resolve(strict=True)
    except OSError as error:
        lexical = Path(os.path.abspath(path))
        claimed_in_scope = False
        for raw_scope in scopes.values():
            scope = raw_scope.resolve()
            if scope.is_file():
                if lexical == scope:
                    claimed_in_scope = True
                    break
                continue
            try:
                lexical.relative_to(scope)
            except ValueError:
                continue
            claimed_in_scope = True
            break
        if claimed_in_scope:
            raise P23ProvenanceError(f"loaded module file is unavailable: {path}") from error
        return None


def _scope_match(path: Path, raw_scope: Path) -> tuple[bool, Path]:
    scope = raw_scope.resolve()
    if scope.is_file():
        return path == scope, Path(scope.name)
    try:
        return True, path.relative_to(scope)
    except ValueError:
        return False, Path()


def collect_runtime_module_hashes(
    scopes: Mapping[str, Path],
    *,
    modules: Mapping[str, ModuleType] | None = None,
    scope_file_allowlists: Mapping[str, set[str] | frozenset[str]] | None = None,
) -> dict[str, dict[str, object]]:
    """Hash every loaded file-backed module under the three execution scopes.

    ``scopes`` must contain exactly ``repository``, ``nanogpt``, and ``muon``.
    Paths are never retained; each entry uses a scope-relative logical path.
    A repository allowlist is fail-closed for every source file.  Acquisition
    rejects a repository-local ``.venv`` before this collector is reached;
    image dependencies are outside the repository scope.
    """

    if set(scopes) != {"repository", "nanogpt", "muon"}:
        raise P23ProvenanceError("module scopes must be repository, nanogpt, and muon")
    allowlists = scope_file_allowlists or {}
    if not set(allowlists).issubset(scopes):
        raise P23ProvenanceError("module allowlists name an undeclared source scope")
    for scope_name, paths in allowlists.items():
        if not isinstance(paths, (set, frozenset)) or any(
            not isinstance(path, str)
            or not path
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            for path in paths
        ):
            raise P23ProvenanceError(
                f"module allowlist for {scope_name!r} must contain relative POSIX paths"
            )
    source_modules = sys.modules if modules is None else modules
    records: dict[str, dict[str, object]] = {}
    scope_counts = {name: 0 for name in scopes}
    for module_name, module in sorted(source_modules.items()):
        if module is None:
            continue
        path = _module_source_path(module, scopes=scopes)
        if path is None:
            continue
        matches: list[tuple[str, Path]] = []
        for scope_name, scope_root in scopes.items():
            matched, relative = _scope_match(path, scope_root)
            if matched:
                matches.append((scope_name, relative))
        if not matches:
            continue
        # Prefer the narrowest matching root so an external checkout nested
        # inside a workspace cannot be mislabeled as repository-owned.
        scope_name, relative = max(
            matches,
            key=lambda item: len(scopes[item[0]].resolve().parts),
        )
        allowed = allowlists.get(scope_name)
        if allowed is not None and relative.as_posix() not in allowed:
            raise P23ProvenanceError(
                f"loaded module {module_name!r} is outside the {scope_name} "
                f"source allowlist: {relative.as_posix()}"
            )
        logical_path = f"{scope_name}:{relative.as_posix()}"
        records[module_name] = {
            "scope": scope_name,
            "logical_path": logical_path,
            "sha256": sha256_file(path),
            "byte_count": path.stat().st_size,
            "source_kind": "python" if path.suffix == ".py" else "native_extension",
        }
        scope_counts[scope_name] += 1
    empty = [name for name, count in scope_counts.items() if count == 0]
    if empty:
        raise P23ProvenanceError(f"no loaded file-backed module found for scopes: {empty}")
    return records


def assert_runtime_module_map_unchanged(
    initialized: Mapping[str, Mapping[str, object]],
    final: Mapping[str, Mapping[str, object]],
) -> None:
    """Reject lazy, removed, relocated, or byte-changed execution modules."""

    if initialized == final:
        return
    initial_names = set(initialized)
    final_names = set(final)
    added = sorted(final_names - initial_names)
    removed = sorted(initial_names - final_names)
    changed = sorted(
        name for name in initial_names & final_names if initialized[name] != final[name]
    )
    raise P23ProvenanceError(
        f"runtime module map changed after initialization: "
        f"added={added}, removed={removed}, changed={changed}"
    )


def _resolved_logical_roots(logical_roots: Mapping[str, Path]) -> dict[str, Path]:
    """Validate and resolve the stable path roots used by the loaded-file closure."""

    if not logical_roots:
        raise P23ProvenanceError("loaded-file closure has no logical roots")
    resolved: dict[str, Path] = {}
    for name, raw_root in logical_roots.items():
        if not isinstance(name, str) or re.fullmatch(r"[a-z][a-z0-9_]*", name) is None:
            raise P23ProvenanceError("loaded-file logical-root name is invalid")
        root = Path(raw_root)
        if not root.is_absolute():
            raise P23ProvenanceError(f"loaded-file logical root {name!r} is not absolute")
        try:
            resolved_root = root.resolve(strict=True)
        except OSError as error:
            raise P23ProvenanceError(
                f"loaded-file logical root {name!r} is unavailable: {root}"
            ) from error
        if not (resolved_root.is_dir() or resolved_root.is_file()):
            raise P23ProvenanceError(
                f"loaded-file logical root {name!r} is not a regular file or directory"
            )
        resolved[name] = resolved_root
    return resolved


def _logical_file_location(path: Path, roots: Mapping[str, Path]) -> tuple[str, str]:
    """Return the narrowest stable root and logical path for one resolved file."""

    matches: list[tuple[int, int, str, Path]] = []
    for name, root in roots.items():
        if root.is_file():
            if path == root:
                matches.append((len(root.parts), 1, name, Path(root.name)))
            continue
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        matches.append((len(root.parts), 0, name, relative))
    if not matches:
        raise P23ProvenanceError(f"loaded or mapped file lies outside every logical root: {path}")
    _, _, scope, relative = max(matches, key=lambda value: (value[0], value[1], value[2]))
    return scope, f"{scope}:{relative.as_posix()}"


def _regular_file_digest(path: Path) -> tuple[str, int]:
    """Hash one regular file while rejecting a concurrent byte/metadata change."""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise P23ProvenanceError(f"loaded-file artifact is not regular: {path}")
            byte_count = 0
            while block := handle.read(2**20):
                digest.update(block)
                byte_count += len(block)
            after = os.fstat(handle.fileno())
    except OSError as error:
        raise P23ProvenanceError(f"cannot hash loaded-file artifact {path}: {error}") from error
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise P23ProvenanceError(f"loaded-file artifact changed while it was hashed: {path}")
    if byte_count != before.st_size:
        raise P23ProvenanceError(f"loaded-file artifact size changed while it was hashed: {path}")
    return digest.hexdigest(), byte_count


def read_live_mountinfo_bytes() -> bytes:
    """Read the current process mount namespace as one exact byte string."""

    try:
        payload = Path("/proc/self/mountinfo").read_bytes()
    except OSError as error:
        raise P23ProvenanceError("P23 requires readable /proc/self/mountinfo") from error
    if not payload:
        raise P23ProvenanceError("P23 /proc/self/mountinfo is empty")
    try:
        payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise P23ProvenanceError("P23 /proc/self/mountinfo is not UTF-8") from error
    return payload


def _parse_mountinfo(value: bytes | str) -> tuple[dict[str, object], ...]:
    """Parse the fields needed to classify each executable file's deepest mount."""

    try:
        text = value.decode("utf-8") if isinstance(value, bytes) else value
    except UnicodeDecodeError as error:
        raise P23ProvenanceError("P23 mountinfo is not UTF-8") from error
    if not isinstance(text, str) or not text:
        raise P23ProvenanceError("P23 mountinfo is empty")
    records: list[dict[str, object]] = []
    seen_mount_ids: set[int] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        fields = line.split()
        try:
            separator = fields.index("-")
        except ValueError as error:
            raise P23ProvenanceError(
                f"malformed /proc mountinfo line {line_number}: no field separator"
            ) from error
        if separator < 6 or len(fields) < separator + 4:
            raise P23ProvenanceError(f"malformed /proc mountinfo line {line_number}")
        try:
            mount_id = int(fields[0])
            parent_id = int(fields[1])
        except ValueError as error:
            raise P23ProvenanceError(
                f"malformed /proc mountinfo IDs at line {line_number}"
            ) from error
        if mount_id <= 0 or parent_id < 0 or mount_id in seen_mount_ids:
            raise P23ProvenanceError(
                f"invalid or duplicate /proc mountinfo ID at line {line_number}"
            )
        seen_mount_ids.add(mount_id)
        raw_mount_point = _decode_proc_maps_path(fields[4])
        mount_point = Path(raw_mount_point)
        if not mount_point.is_absolute():
            raise P23ProvenanceError(
                f"non-absolute /proc mountinfo mount point at line {line_number}"
            )
        options = set(fields[5].split(","))
        if ("ro" in options) == ("rw" in options):
            raise P23ProvenanceError(f"ambiguous /proc mountinfo access mode at line {line_number}")
        records.append(
            {
                "mount_id": mount_id,
                "mount_point": mount_point,
                "read_only": "ro" in options,
            }
        )
    if not records or not any(record["mount_point"] == Path("/") for record in records):
        raise P23ProvenanceError("P23 mountinfo has no root mount")
    return tuple(records)


def _effective_mount(path: Path, mount_table: tuple[dict[str, object], ...]) -> dict[str, object]:
    matches: list[tuple[int, dict[str, object]]] = []
    for record in mount_table:
        mount_point = record["mount_point"]
        assert isinstance(mount_point, Path)
        try:
            path.relative_to(mount_point)
        except ValueError:
            continue
        matches.append((len(mount_point.parts), record))
    if not matches:
        raise P23ProvenanceError(f"loaded-file path has no effective mount: {path}")
    return max(matches, key=lambda item: (item[0], int(item[1]["mount_id"])))[1]


def _loaded_file_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".py":
        return "python_source"
    if suffix in {".pyc", ".pyo"}:
        return "python_bytecode"
    if suffix in {".so", ".dylib", ".dll", ".pyd"} or ".so." in path.name:
        return "native_binary"
    return "regular_file"


def _loaded_file_record(
    path: Path,
    roots: Mapping[str, Path],
    *,
    executable_origin: bool = True,
    digest_cache: dict[Path, tuple[str, int]] | None = None,
    mount_table: tuple[dict[str, object], ...] | None = None,
) -> dict[str, object]:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise P23ProvenanceError(f"loaded-file artifact is unavailable: {path}") from error
    try:
        mode = resolved.stat().st_mode
    except OSError as error:
        raise P23ProvenanceError(f"cannot stat loaded-file artifact {resolved}: {error}") from error
    if not stat.S_ISREG(mode):
        raise P23ProvenanceError(f"loaded-file artifact is not regular: {resolved}")
    scope, logical_path = _logical_file_location(resolved, roots)
    if executable_origin:
        if mount_table is None:
            mount_table = _parse_mountinfo(read_live_mountinfo_bytes())
        effective_mount = _effective_mount(resolved, mount_table)
        if effective_mount["read_only"] is not True:
            raise P23ProvenanceError(f"executable origin lies on a writable mount: {logical_path}")
    if executable_origin and scope not in _ALLOWED_EXECUTABLE_SCOPES:
        raise P23ProvenanceError(
            f"executable origin uses forbidden writable/data scope {scope!r}: {logical_path}"
        )
    cached = digest_cache.get(resolved) if digest_cache is not None else None
    if cached is None:
        cached = _regular_file_digest(resolved)
        if digest_cache is not None:
            digest_cache[resolved] = cached
    digest, byte_count = cached
    return {
        "scope": scope,
        "logical_path": logical_path,
        "sha256": digest,
        "byte_count": byte_count,
        "file_kind": _loaded_file_kind(resolved),
        "executable_origin": executable_origin,
    }


def _module_declared_file(raw: object, *, module_name: str, role: str) -> Path | None:
    """Interpret one module path claim without accepting cwd-relative execution."""

    if not isinstance(raw, str) or not raw or raw in {"built-in", "frozen"}:
        return None
    if raw.startswith("<") and raw.endswith(">"):
        return None
    path = Path(raw)
    if not path.is_absolute():
        # Torch exposes a few lazy module proxies with synthetic relative
        # ``__file__`` strings.  They are not file-backed.  Keep this exception
        # exact rather than silently treating an arbitrary missing relative
        # module path as synthetic.
        torch_proxy = (module_name, role, raw) in {
            ("torch.classes", "module_file", "_classes.py"),
            ("torch.ops", "module_file", "_ops.py"),
        }
        if torch_proxy and not path.exists():
            return None
        raise P23ProvenanceError(
            f"loaded module {module_name!r} {role} uses a relative executable path: {raw}"
        )
    if not path.exists():
        # ``__cached__`` is routinely the prospective cache location even
        # when bytecode writing is disabled.  Only an existing cache is an
        # executed-file artifact; a missing source/origin is a hard failure.
        if role == "cached_bytecode":
            return None
        raise P23ProvenanceError(f"loaded module {role} is unavailable: {raw}")
    if not path.is_file():
        raise P23ProvenanceError(f"loaded module {role} is not a regular file: {raw}")
    return path


def _module_file_artifacts(
    module: ModuleType,
    *,
    roots: Mapping[str, Path],
    digest_cache: dict[Path, tuple[str, int]],
    mount_table: tuple[dict[str, object], ...],
) -> dict[str, dict[str, object] | None]:
    spec = getattr(module, "__spec__", None)
    declared = {
        "module_file": getattr(module, "__file__", None),
        "spec_origin": getattr(spec, "origin", None),
        "cached_bytecode": getattr(module, "__cached__", None),
    }
    module_name = str(getattr(module, "__name__", ""))
    paths = {
        role: _module_declared_file(raw, module_name=module_name, role=role)
        for role, raw in declared.items()
    }
    source_path: Path | None = None
    for role in ("module_file", "spec_origin"):
        candidate = paths[role]
        if candidate is None:
            continue
        if candidate.suffix == ".py":
            source_path = candidate
            break
        if candidate.suffix in {".pyc", ".pyo"}:
            try:
                derived = Path(importlib.util.source_from_cache(str(candidate)))
            except ValueError:
                continue
            if derived.is_file():
                source_path = derived
                break
    artifacts: dict[str, dict[str, object] | None] = {
        role: (
            _loaded_file_record(
                path,
                roots,
                digest_cache=digest_cache,
                mount_table=mount_table,
            )
            if path is not None
            else None
        )
        for role, path in paths.items()
    }
    artifacts["source_file"] = (
        _loaded_file_record(
            source_path,
            roots,
            digest_cache=digest_cache,
            mount_table=mount_table,
        )
        if source_path is not None
        else None
    )
    return {role: artifacts[role] for role in _MODULE_ARTIFACT_ROLES}


def _decode_proc_maps_path(raw: str) -> str:
    """Decode the octal escapes emitted in Linux ``/proc/<pid>/maps`` paths."""

    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), raw)


def _mapped_regular_files(
    roots: Mapping[str, Path],
    *,
    proc_maps_text: str | None,
    digest_cache: dict[Path, tuple[str, int]],
    mount_table: tuple[dict[str, object], ...],
) -> dict[str, dict[str, object]]:
    if proc_maps_text is None:
        maps_path = Path("/proc/self/maps")
        try:
            proc_maps_text = maps_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            raise P23ProvenanceError("P23 requires readable UTF-8 /proc/self/maps") from error
    mapped_paths: dict[Path, bool] = {}
    for line_number, line in enumerate(proc_maps_text.splitlines(), start=1):
        fields = line.split(maxsplit=5)
        if len(fields) < 5:
            raise P23ProvenanceError(f"malformed /proc/self/maps line {line_number}")
        if len(fields) == 5:
            continue
        raw_path = fields[5]
        if not raw_path or raw_path.startswith("["):
            continue
        if raw_path.endswith(" (deleted)"):
            raise P23ProvenanceError(
                f"deleted file-backed mapping is forbidden at /proc/self/maps line {line_number}"
            )
        decoded = _decode_proc_maps_path(raw_path)
        path = Path(decoded)
        if not path.is_absolute():
            raise P23ProvenanceError(
                f"unrooted file-backed mapping at /proc/self/maps line {line_number}: {raw_path}"
            )
        try:
            mode = path.stat().st_mode
        except OSError as error:
            raise P23ProvenanceError(
                f"mapped pathname is unavailable at /proc/self/maps line {line_number}: {raw_path}"
            ) from error
        # Device and other non-regular mappings are not executable-file bytes.
        if not stat.S_ISREG(mode):
            continue
        resolved = path.resolve(strict=True)
        mapped_paths[resolved] = mapped_paths.get(resolved, False) or "x" in fields[1]
    records: dict[str, dict[str, object]] = {}
    for path, executable in sorted(mapped_paths.items(), key=lambda item: str(item[0])):
        record = _loaded_file_record(
            path,
            roots,
            executable_origin=executable,
            digest_cache=digest_cache,
            mount_table=mount_table,
        )
        logical_path = str(record["logical_path"])
        previous = records.get(logical_path)
        if previous is not None and previous != record:
            raise P23ProvenanceError(f"mapped file changed within one snapshot: {logical_path}")
        records[logical_path] = record
    return records


def _snapshot_content_sha256(
    modules: Mapping[str, object], mapped_files: Mapping[str, object]
) -> str:
    return canonical_json_sha256({"modules": modules, "mapped_files": mapped_files})


def collect_loaded_file_snapshot(
    logical_roots: Mapping[str, Path],
    *,
    modules: Mapping[str, ModuleType] | None = None,
    proc_maps_text: str | None = None,
    mountinfo_text: bytes | str | None = None,
) -> dict[str, object]:
    """Hash all file-backed loaded modules and regular mapped native files.

    Module records retain the independently declared ``__file__``, spec
    origin, existing cache bytecode, and source (when available).  The maps
    closure covers every regular file pathname in Linux ``/proc/self/maps``.
    Every retained file must resolve beneath one explicit logical root.
    """

    roots = _resolved_logical_roots(logical_roots)
    mount_table = _parse_mountinfo(
        read_live_mountinfo_bytes() if mountinfo_text is None else mountinfo_text
    )
    digest_cache: dict[Path, tuple[str, int]] = {}
    source_modules = sys.modules if modules is None else modules
    module_records: dict[str, dict[str, object]] = {}
    for module_name, module in sorted(source_modules.items()):
        if module is None:
            continue
        artifacts = _module_file_artifacts(
            module,
            roots=roots,
            digest_cache=digest_cache,
            mount_table=mount_table,
        )
        if not any(record is not None for record in artifacts.values()):
            continue
        module_records[str(module_name)] = {"artifacts": artifacts}
    mapped_files = _mapped_regular_files(
        roots,
        proc_maps_text=proc_maps_text,
        digest_cache=digest_cache,
        mount_table=mount_table,
    )
    return {
        "schema_version": P23_LOADED_FILE_SNAPSHOT_SCHEMA,
        "module_count": len(module_records),
        "mapped_file_count": len(mapped_files),
        "modules": module_records,
        "mapped_files": mapped_files,
        "content_sha256": _snapshot_content_sha256(module_records, mapped_files),
    }


def _validate_loaded_file_record(value: object, *, logical_path: str | None = None) -> None:
    expected = {
        "scope",
        "logical_path",
        "sha256",
        "byte_count",
        "file_kind",
        "executable_origin",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise P23ProvenanceError("loaded-file artifact record is malformed")
    scope = value.get("scope")
    recorded_path = value.get("logical_path")
    if (
        not isinstance(scope, str)
        or re.fullmatch(r"[a-z][a-z0-9_]*", scope) is None
        or not isinstance(recorded_path, str)
        or not recorded_path.startswith(f"{scope}:")
        or (logical_path is not None and recorded_path != logical_path)
        or not _is_sha256(value.get("sha256"))
        or not isinstance(value.get("byte_count"), int)
        or isinstance(value.get("byte_count"), bool)
        or int(value["byte_count"]) < 0
        or value.get("file_kind")
        not in {"python_source", "python_bytecode", "native_binary", "regular_file"}
        or not isinstance(value.get("executable_origin"), bool)
        or (value.get("executable_origin") is True and scope not in _ALLOWED_EXECUTABLE_SCOPES)
    ):
        raise P23ProvenanceError("loaded-file artifact record has invalid fields")


def validate_loaded_file_snapshot(value: object) -> dict[str, object]:
    expected = {
        "schema_version",
        "module_count",
        "mapped_file_count",
        "modules",
        "mapped_files",
        "content_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise P23ProvenanceError("loaded-file snapshot fields do not match its schema")
    modules = value.get("modules")
    mapped_files = value.get("mapped_files")
    if not isinstance(modules, Mapping) or not isinstance(mapped_files, Mapping):
        raise P23ProvenanceError("loaded-file snapshot maps are malformed")
    for module_name, module_record in modules.items():
        if not isinstance(module_name, str) or not module_name:
            raise P23ProvenanceError("loaded-file snapshot has an invalid module name")
        if not isinstance(module_record, Mapping) or set(module_record) != {"artifacts"}:
            raise P23ProvenanceError("loaded-file module record is malformed")
        artifacts = module_record.get("artifacts")
        if not isinstance(artifacts, Mapping) or set(artifacts) != set(_MODULE_ARTIFACT_ROLES):
            raise P23ProvenanceError("loaded-file module artifact roles changed")
        if not any(record is not None for record in artifacts.values()):
            raise P23ProvenanceError("loaded-file module record contains no file")
        for record in artifacts.values():
            if record is not None:
                _validate_loaded_file_record(record)
    for logical_path, record in mapped_files.items():
        if not isinstance(logical_path, str):
            raise P23ProvenanceError("loaded-file mapped-file key is invalid")
        _validate_loaded_file_record(record, logical_path=logical_path)
    if (
        value.get("schema_version") != P23_LOADED_FILE_SNAPSHOT_SCHEMA
        or value.get("module_count") != len(modules)
        or value.get("mapped_file_count") != len(mapped_files)
        or value.get("content_sha256") != _snapshot_content_sha256(modules, mapped_files)
    ):
        raise P23ProvenanceError("loaded-file snapshot count or digest mismatch")
    return copy.deepcopy(dict(value))


def build_loaded_file_closure(
    initialized: Mapping[str, object], completed: Mapping[str, object]
) -> dict[str, object]:
    """Validate the two snapshots and record every allowed lazy-load addition."""

    initial = validate_loaded_file_snapshot(initialized)
    final = validate_loaded_file_snapshot(completed)
    initial_modules = initial["modules"]
    final_modules = final["modules"]
    initial_mapped = initial["mapped_files"]
    final_mapped = final["mapped_files"]
    assert isinstance(initial_modules, dict) and isinstance(final_modules, dict)
    assert isinstance(initial_mapped, dict) and isinstance(final_mapped, dict)
    removed_modules = sorted(set(initial_modules) - set(final_modules))
    changed_modules = sorted(
        name
        for name in set(initial_modules) & set(final_modules)
        if initial_modules[name] != final_modules[name]
    )
    removed_mapped = sorted(set(initial_mapped) - set(final_mapped))
    changed_mapped = sorted(
        name
        for name in set(initial_mapped) & set(final_mapped)
        if initial_mapped[name] != final_mapped[name]
    )
    if removed_modules or changed_modules or removed_mapped or changed_mapped:
        raise P23ProvenanceError(
            "loaded-file closure removed or changed initialized entries: "
            f"removed_modules={removed_modules}, changed_modules={changed_modules}, "
            f"removed_mapped_files={removed_mapped}, changed_mapped_files={changed_mapped}"
        )
    return {
        "schema_version": P23_LOADED_FILE_CLOSURE_SCHEMA,
        "initialized": initial,
        "completed": final,
        "module_additions": {
            name: final_modules[name] for name in sorted(set(final_modules) - set(initial_modules))
        },
        "mapped_file_additions": {
            name: final_mapped[name] for name in sorted(set(final_mapped) - set(initial_mapped))
        },
    }


def validate_loaded_file_closure(value: object) -> dict[str, object]:
    expected = {
        "schema_version",
        "initialized",
        "completed",
        "module_additions",
        "mapped_file_additions",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise P23ProvenanceError("loaded-file closure fields do not match its schema")
    if value.get("schema_version") != P23_LOADED_FILE_CLOSURE_SCHEMA:
        raise P23ProvenanceError("loaded-file closure schema mismatch")
    rebuilt = build_loaded_file_closure(
        value.get("initialized"),  # type: ignore[arg-type]
        value.get("completed"),  # type: ignore[arg-type]
    )
    if dict(value) != rebuilt:
        raise P23ProvenanceError("loaded-file closure additions do not reconstruct exactly")
    return rebuilt


def _resolve_loaded_logical_path(logical_path: str, roots: Mapping[str, Path]) -> Path:
    scope, separator, raw_relative = logical_path.partition(":")
    if not separator or scope not in roots or not raw_relative:
        raise P23ProvenanceError(f"loaded-file logical path is invalid: {logical_path!r}")
    relative = Path(raw_relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise P23ProvenanceError(f"loaded-file logical path escapes its root: {logical_path!r}")
    root = roots[scope]
    if root.is_file():
        if relative != Path(root.name):
            raise P23ProvenanceError(
                f"loaded-file path does not name its file root: {logical_path}"
            )
        candidate = root
    else:
        candidate = root / relative
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise P23ProvenanceError(f"loaded-file artifact is unavailable: {logical_path}") from error
    resolved_scope, rebuilt = _logical_file_location(resolved, roots)
    if resolved_scope != scope or rebuilt != logical_path:
        raise P23ProvenanceError(f"loaded-file logical path is not canonical: {logical_path}")
    return resolved


def verify_loaded_file_closure(
    closure: Mapping[str, object],
    *,
    logical_roots: Mapping[str, Path],
    current_initialized: Mapping[str, object] | None = None,
    mountinfo_text: bytes | str | None = None,
) -> dict[str, object]:
    """Rehash every retained byte and optionally recollect initialized membership.

    A standalone verifier can reproduce initialized membership after performing
    the frozen model/optimizer setup.  It cannot infer historical final process
    membership without rerunning training, so final membership is corroborated
    by exact equality across all three acquisition identities while this helper
    independently rehashes every initialized and completed artifact.
    """

    validated = validate_loaded_file_closure(closure)
    if current_initialized is not None:
        current = validate_loaded_file_snapshot(current_initialized)
        if current != validated["initialized"]:
            raise P23ProvenanceError(
                "current initialized loaded-file snapshot differs from the recorded closure"
            )
    roots = _resolved_logical_roots(logical_roots)
    mount_table = _parse_mountinfo(
        read_live_mountinfo_bytes() if mountinfo_text is None else mountinfo_text
    )
    checked: dict[str, dict[str, object]] = {}
    for phase in ("initialized", "completed"):
        snapshot = validated[phase]
        assert isinstance(snapshot, dict)
        modules = snapshot["modules"]
        mapped_files = snapshot["mapped_files"]
        assert isinstance(modules, dict) and isinstance(mapped_files, dict)
        records: list[Mapping[str, object]] = []
        for module_record in modules.values():
            assert isinstance(module_record, dict)
            artifacts = module_record["artifacts"]
            assert isinstance(artifacts, dict)
            records.extend(record for record in artifacts.values() if isinstance(record, Mapping))
        records.extend(mapped_files.values())
        for record in records:
            logical_path = str(record["logical_path"])
            if logical_path in checked:
                if checked[logical_path] != dict(record):
                    raise P23ProvenanceError(
                        f"loaded-file closure gives conflicting records for {logical_path}"
                    )
                continue
            resolved = _resolve_loaded_logical_path(logical_path, roots)
            replayed = _loaded_file_record(
                resolved,
                roots,
                executable_origin=bool(record["executable_origin"]),
                mount_table=mount_table,
            )
            if replayed != dict(record):
                raise P23ProvenanceError(f"loaded-file artifact bytes changed: {logical_path}")
            checked[logical_path] = replayed
    return {
        "passes": True,
        "comparison": "initialized_membership_recollected_all_retained_bytes_rehashed",
        "initialized_module_count": validated["initialized"]["module_count"],  # type: ignore[index]
        "completed_module_count": validated["completed"]["module_count"],  # type: ignore[index]
        "initialized_mapped_file_count": validated["initialized"]["mapped_file_count"],  # type: ignore[index]
        "completed_mapped_file_count": validated["completed"]["mapped_file_count"],  # type: ignore[index]
        "unique_file_count": len(checked),
    }


def load_and_validate_host_attestation(
    path: Path,
    *,
    runtime_lock: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Validate the sanitized, out-of-container host attestation artifact.

    The artifact records digests of raw host command output rather than local
    command-output paths.  When a runtime lock is supplied, its container and
    GPU fields and its attestation hash must agree exactly.
    """

    resolved = path.resolve()
    payload = _read_json_mapping(resolved, "P23 host attestation")
    if set(payload) != {"schema_version", "status", "container", "gpu", "evidence"}:
        raise P23ProvenanceError("host-attestation fields do not match its schema")
    if payload.get("schema_version") != P23_HOST_ATTESTATION_SCHEMA:
        raise P23ProvenanceError("host-attestation schema mismatch")
    if payload.get("status") != "procedurally_host_attested":
        raise P23ProvenanceError("host attestation is not finalized")
    container = _validate_container_identity(
        payload.get("container"), require_host_attestation_sha256=False
    )
    gpu = payload.get("gpu")
    if not isinstance(gpu, Mapping) or _GPU_UUID.fullmatch(str(gpu.get("uuid"))) is None:
        raise P23ProvenanceError("host attestation has no full-GPU UUID")
    required_gpu = {
        "uuid",
        "pci_bus_id",
        "name",
        "vbios_version",
        "driver_version",
        "compute_capability",
        "multiprocessor_count",
        "total_memory",
        "nvidia_smi_memory_total_mib",
        "mig_mode",
    }
    if set(gpu) != required_gpu or any(gpu.get(field) in (None, "", []) for field in required_gpu):
        raise P23ProvenanceError("host attestation GPU map is incomplete or has unknown fields")
    capability = gpu.get("compute_capability")
    numeric_gpu_values = (
        gpu.get("multiprocessor_count"),
        gpu.get("total_memory"),
        gpu.get("nvidia_smi_memory_total_mib"),
    )
    if (
        not isinstance(capability, list)
        or len(capability) != 2
        or any(not isinstance(value, int) or isinstance(value, bool) for value in capability)
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in numeric_gpu_values
        )
    ):
        raise P23ProvenanceError("host attestation has invalid numeric GPU identity fields")
    if str(gpu["uuid"]).startswith("MIG-") or gpu.get("mig_mode") != "disabled":
        raise P23ProvenanceError("P23 host attestation must identify a non-MIG GPU")
    selection = container["gpu_selection"]
    assert isinstance(selection, Mapping)
    if selection.get("requested_full_gpu_uuid") != gpu.get("uuid"):
        raise P23ProvenanceError(
            "host-attested Docker GPU request differs from the live full-GPU UUID"
        )
    evidence = payload.get("evidence")
    expected_evidence = {
        "image_inspection_sha256",
        "running_container_inspection_sha256",
        "running_mountinfo_sha256",
        "nvidia_smi_query_sha256",
    }
    if not isinstance(evidence, Mapping) or set(evidence) != expected_evidence:
        raise P23ProvenanceError("host attestation evidence map is incomplete")
    if any(not _is_sha256(evidence.get(field)) for field in expected_evidence):
        raise P23ProvenanceError("host attestation evidence digest is invalid")
    rendered = json.dumps(payload, sort_keys=True, allow_nan=False)
    if any(fragment in rendered for fragment in _PERSONAL_PATH_FRAGMENTS):
        raise P23ProvenanceError("host attestation contains a machine-local path")
    attestation_sha256 = sha256_file(resolved)
    if runtime_lock is not None:
        locked_container = runtime_lock.get("container")
        locked_gpu = runtime_lock.get("gpu")
        if not isinstance(locked_container, Mapping) or not isinstance(locked_gpu, Mapping):
            raise P23ProvenanceError("runtime lock cannot be compared with host attestation")
        if locked_container.get("host_attestation_sha256") != attestation_sha256:
            raise P23ProvenanceError("runtime lock does not bind the host-attestation bytes")
        expected_locked_container = {
            key: copy.deepcopy(value)
            for key, value in locked_container.items()
            if key != "host_attestation_sha256"
        }
        if dict(container) != expected_locked_container:
            raise P23ProvenanceError("host-attested container differs from the runtime lock")
        if dict(gpu) != dict(locked_gpu):
            raise P23ProvenanceError("host-attested GPU differs from the runtime lock")
    result = copy.deepcopy(payload)
    result["host_attestation_sha256"] = attestation_sha256
    return result


def load_runtime_lock(
    path: Path,
    *,
    addendum_path: Path = DEFAULT_ADDENDUM_PATH,
    host_attestation_path: Path | None = None,
) -> dict[str, Any]:
    """Load a host-produced CUDA/container identity lock.

    The committed template is intentionally rejected.  Acquisition requires a
    separate lock populated from the host/container boundary and bound to the
    exact addendum bytes.
    """

    path = path.resolve()
    payload = _read_json_mapping(path, "P23 CUDA runtime lock")
    expected_top_level = {
        "schema_version",
        "status",
        "p23_addendum_sha256",
        "container",
        "gpu",
        "software",
        "determinism",
        "loader_environment",
    }
    if set(payload) != expected_top_level:
        raise P23ProvenanceError("CUDA runtime-lock fields do not match its schema")
    if payload.get("schema_version") != P23_RUNTIME_LOCK_SCHEMA:
        raise P23ProvenanceError("CUDA runtime-lock schema mismatch")
    if payload.get("status") != "pinned_for_acquisition":
        raise P23ProvenanceError("CUDA runtime-lock template is not valid for acquisition")
    if payload.get("p23_addendum_sha256") != sha256_file(addendum_path.resolve()):
        raise P23ProvenanceError("CUDA runtime lock does not bind the current P23 addendum")
    container = _validate_container_identity(
        payload.get("container"), require_host_attestation_sha256=True
    )
    gpu = payload.get("gpu")
    if not isinstance(gpu, dict) or _GPU_UUID.fullmatch(str(gpu.get("uuid"))) is None:
        raise P23ProvenanceError("CUDA runtime lock has no full-GPU UUID")
    if str(gpu.get("uuid", "")).startswith("MIG-") or gpu.get("mig_mode") != "disabled":
        raise P23ProvenanceError("P23 forbids MIG acquisition")
    selection = container["gpu_selection"]
    assert isinstance(selection, Mapping)
    if selection.get("requested_full_gpu_uuid") != gpu.get("uuid"):
        raise P23ProvenanceError(
            "runtime-lock Docker GPU request differs from the live full-GPU UUID"
        )
    required_gpu = (
        "pci_bus_id",
        "name",
        "vbios_version",
        "driver_version",
        "compute_capability",
        "multiprocessor_count",
        "total_memory",
        "nvidia_smi_memory_total_mib",
    )
    if any(gpu.get(field) in (None, "", []) for field in required_gpu):
        raise P23ProvenanceError("CUDA runtime lock has an incomplete GPU identity")
    if set(gpu) != {"uuid", "mig_mode", *required_gpu}:
        raise P23ProvenanceError("CUDA runtime-lock GPU fields do not match its schema")
    if (
        not isinstance(gpu.get("compute_capability"), list)
        or len(gpu["compute_capability"]) != 2
        or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in gpu["compute_capability"]
        )
        or not isinstance(gpu.get("multiprocessor_count"), int)
        or isinstance(gpu.get("multiprocessor_count"), bool)
        or int(gpu["multiprocessor_count"]) <= 0
        or not isinstance(gpu.get("total_memory"), int)
        or isinstance(gpu.get("total_memory"), bool)
        or int(gpu["total_memory"]) <= 0
        or not isinstance(gpu.get("nvidia_smi_memory_total_mib"), int)
        or isinstance(gpu.get("nvidia_smi_memory_total_mib"), bool)
        or int(gpu["nvidia_smi_memory_total_mib"]) <= 0
    ):
        raise P23ProvenanceError("CUDA runtime lock has invalid numeric GPU identity fields")
    software = payload.get("software")
    required_software = (
        "python",
        "python_executable",
        "python_executable_sha256",
        "python_flags",
        "torch_version",
        "torch_git_version",
        "torch_config_sha256",
        "torch_cuda_compiled_version",
        "cuda_runtime_version",
        "cudnn_version",
        "numpy_version",
        "platform",
        "system",
        "release",
        "version",
        "machine",
        "processor",
        "libc",
        "cpu_identity",
        "proc_version_sha256",
    )
    if not isinstance(software, dict) or any(
        software.get(field) in (None, "", []) for field in required_software
    ):
        raise P23ProvenanceError("CUDA runtime lock has an incomplete software identity")
    if set(software) != set(required_software):
        raise P23ProvenanceError("CUDA runtime-lock software fields do not match its schema")
    if not _is_sha256(software.get("torch_config_sha256")):
        raise P23ProvenanceError("CUDA runtime lock has an invalid Torch configuration digest")
    if software.get("python_executable") != PINNED_PYTHON_EXECUTABLE:
        raise P23ProvenanceError("CUDA runtime lock does not use the pinned Python executable")
    if not _is_pinned_python_version_string(software.get("python")):
        raise P23ProvenanceError("CUDA runtime lock does not use Python 3.12")
    if not _is_sha256(software.get("python_executable_sha256")):
        raise P23ProvenanceError("CUDA runtime lock has an invalid Python executable digest")
    python_flags = software.get("python_flags")
    if (
        not isinstance(python_flags, Mapping)
        or any(
            type(python_flags.get(name)) is not type(expected) or python_flags.get(name) != expected
            for name, expected in FROZEN_PYTHON_FLAGS.items()
        )
        or set(python_flags) != set(FROZEN_PYTHON_FLAGS)
    ):
        raise P23ProvenanceError("CUDA runtime lock has invalid Python interpreter flags")
    if not _is_sha256(software.get("proc_version_sha256")):
        raise P23ProvenanceError("CUDA runtime lock has an invalid proc_version_sha256")
    cpu_identity = software.get("cpu_identity")
    expected_cpu_fields = {
        "vendor_id",
        "cpu_family",
        "model",
        "model_name",
        "stepping",
        "microcode",
        "flags_sha256",
        "logical_processor_count",
    }
    if not isinstance(cpu_identity, Mapping) or set(cpu_identity) != expected_cpu_fields:
        raise P23ProvenanceError("CUDA runtime lock has an incomplete stable CPU identity")
    if any(cpu_identity.get(field) in (None, "") for field in expected_cpu_fields):
        raise P23ProvenanceError("CUDA runtime lock has an empty stable CPU identity field")
    if not _is_sha256(cpu_identity.get("flags_sha256")):
        raise P23ProvenanceError("CUDA runtime lock has an invalid CPU flags digest")
    logical_count = cpu_identity.get("logical_processor_count")
    if not isinstance(logical_count, int) or isinstance(logical_count, bool) or logical_count <= 0:
        raise P23ProvenanceError("CUDA runtime lock has an invalid logical CPU count")
    runtime_version = software.get("cuda_runtime_version")
    if (
        not isinstance(runtime_version, int)
        or isinstance(runtime_version, bool)
        or runtime_version <= 0
    ):
        raise P23ProvenanceError("CUDA runtime lock has invalid cudaRuntimeGetVersion output")
    expected_determinism = {
        "cublas_workspace_config": ":4096:8",
        "pythonhashseed": "1337",
        "nvidia_tf32_override": "0",
        "cuda_visible_devices": gpu["uuid"],
        "nvidia_visible_devices": gpu["uuid"],
        "sdpa_backend": "math",
        "deterministic_algorithms": True,
        "deterministic_debug_mode": "error",
        "tf32": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "float32_matmul_precision": "highest",
        "fp16_reduced_precision_reduction": False,
        "bf16_reduced_precision_reduction": False,
        "cpu_threads": 1,
        "interop_threads": 1,
    }
    if payload.get("determinism") != expected_determinism:
        raise P23ProvenanceError("CUDA runtime-lock determinism map is not exactly pinned")
    if payload.get("loader_environment") != {
        "PATH": PINNED_EXECUTABLE_PATH,
        "LD_PRELOAD": None,
        "LD_LIBRARY_PATH": None,
        "LD_AUDIT": None,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONOPTIMIZE": None,
    }:
        raise P23ProvenanceError("CUDA runtime-lock loader environment is not exactly pinned")
    if host_attestation_path is None:
        raise P23ProvenanceError("a host-attestation artifact is required for acquisition")
    load_and_validate_host_attestation(host_attestation_path, runtime_lock=payload)
    result = copy.deepcopy(payload)
    result["runtime_lock_sha256"] = sha256_file(path)
    return result


def load_container_identity(
    path: Path,
    *,
    addendum_path: Path = DEFAULT_ADDENDUM_PATH,
    host_attestation_path: Path,
) -> dict[str, Any]:
    """Backward-readable alias for the complete P23 CUDA runtime lock."""

    return load_runtime_lock(
        path,
        addendum_path=addendum_path,
        host_attestation_path=host_attestation_path,
    )


def _backend_flag(owner: object, name: str) -> object:
    value = getattr(owner, name, None)
    if callable(value):
        return value()
    return value


def collect_python_interpreter_flags(flags: object | None = None) -> dict[str, object]:
    """Collect the process-start Python flags that can alter executed semantics."""

    active = sys.flags if flags is None else flags
    return {name: getattr(active, name, None) for name in FROZEN_PYTHON_FLAGS}


def _is_pinned_python_version_string(value: object) -> bool:
    """Return whether a full ``sys.version`` string identifies Python 3.12."""

    return isinstance(value, str) and re.match(r"^3\.12(?:\.|\s|$)", value) is not None


def _validate_python_major_minor(version_info: object | None = None) -> None:
    active = sys.version_info if version_info is None else version_info
    try:
        observed = tuple(active[:2])  # type: ignore[index]
    except (TypeError, AttributeError) as error:
        raise P23ProvenanceError("P23 cannot determine the Python major/minor version") from error
    if observed != PINNED_PYTHON_MAJOR_MINOR:
        raise P23ProvenanceError(
            "P23 requires Python 3.12; "
            f"observed {'.'.join(str(component) for component in observed)}"
        )


def _python_flag_mapping_is_frozen(observed: Mapping[str, object]) -> bool:
    return set(observed) == set(FROZEN_PYTHON_FLAGS) and all(
        type(observed[name]) is type(expected) and observed[name] == expected
        for name, expected in FROZEN_PYTHON_FLAGS.items()
    )


def validate_python_interpreter_flags(
    flags: object | None = None, *, version_info: object | None = None
) -> dict[str, object]:
    """Reject optimized, isolated, or environment-ignoring Python processes."""

    _validate_python_major_minor(version_info)
    observed = collect_python_interpreter_flags(flags)
    if not _python_flag_mapping_is_frozen(observed):
        raise P23ProvenanceError(f"P23 Python interpreter flags are not exactly pinned: {observed}")
    return observed


def configure_cuda_determinism(
    *,
    torch_module: ModuleType | None = None,
    environ: Mapping[str, str] | None = None,
    python_flags: object | None = None,
) -> dict[str, object]:
    """Set the frozen PyTorch flags before any P23 CUDA probe/allocation.

    Process-start environment variables cannot be repaired safely here.  They
    are checked first and must already have the frozen values.
    """

    if torch_module is None:
        import torch as torch_module  # type: ignore[no-redef]
    env = os.environ if environ is None else environ
    validate_python_interpreter_flags(python_flags)
    expected_environment = {
        "PATH": PINNED_EXECUTABLE_PATH,
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONHASHSEED": "1337",
        "NVIDIA_TF32_OVERRIDE": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VIRTUAL_ENV": PINNED_PYTHON_ENVIRONMENT,
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONOPTIMIZE": None,
        "PYTHONPATH": None,
        "PYTHONHOME": None,
        "LD_PRELOAD": None,
        "LD_LIBRARY_PATH": None,
        "LD_AUDIT": None,
    }
    for name, expected in expected_environment.items():
        if env.get(name) != expected:
            raise P23ProvenanceError(f"{name} must equal {expected!r} before CUDA initialization")
    visible = env.get("CUDA_VISIBLE_DEVICES")
    if not isinstance(visible, str) or _GPU_UUID.fullmatch(visible) is None:
        raise P23ProvenanceError("CUDA_VISIBLE_DEVICES must be one full GPU UUID")
    if env.get("NVIDIA_VISIBLE_DEVICES") != visible:
        raise P23ProvenanceError(
            "NVIDIA_VISIBLE_DEVICES must equal the same full GPU UUID before CUDA initialization"
        )

    torch_module.use_deterministic_algorithms(True)
    torch_module.set_deterministic_debug_mode("error")
    torch_module.set_float32_matmul_precision("highest")
    torch_module.set_num_threads(1)
    torch_module.set_num_interop_threads(1)
    cuda_backend = torch_module.backends.cuda
    cudnn = torch_module.backends.cudnn
    cuda_backend.enable_math_sdp(True)
    cuda_backend.enable_flash_sdp(False)
    cuda_backend.enable_mem_efficient_sdp(False)
    cuda_backend.enable_cudnn_sdp(False)
    cuda_backend.matmul.allow_tf32 = False
    cuda_backend.matmul.allow_fp16_reduced_precision_reduction = False
    cuda_backend.matmul.allow_bf16_reduced_precision_reduction = False
    cudnn.allow_tf32 = False
    cudnn.benchmark = False
    cudnn.deterministic = True

    configured = {
        "deterministic_algorithms": torch_module.are_deterministic_algorithms_enabled(),
        "deterministic_debug_mode": torch_module.get_deterministic_debug_mode(),
        "math_sdpa": _backend_flag(cuda_backend, "math_sdp_enabled"),
        "flash_sdpa": _backend_flag(cuda_backend, "flash_sdp_enabled"),
        "memory_efficient_sdpa": _backend_flag(cuda_backend, "mem_efficient_sdp_enabled"),
        "cudnn_sdpa": _backend_flag(cuda_backend, "cudnn_sdp_enabled"),
        "cudnn_deterministic": cudnn.deterministic,
        "cudnn_benchmark": cudnn.benchmark,
        "allow_tf32_matmul": cuda_backend.matmul.allow_tf32,
        "allow_tf32_cudnn": cudnn.allow_tf32,
        "float32_matmul_precision": torch_module.get_float32_matmul_precision(),
        "allow_fp16_reduced_precision_reduction": (
            cuda_backend.matmul.allow_fp16_reduced_precision_reduction
        ),
        "allow_bf16_reduced_precision_reduction": (
            cuda_backend.matmul.allow_bf16_reduced_precision_reduction
        ),
        "torch_num_threads": torch_module.get_num_threads(),
        "torch_num_interop_threads": torch_module.get_num_interop_threads(),
        "cuda_initialized": bool(getattr(torch_module.cuda, "is_initialized", lambda: False)()),
    }
    expected_configured = {
        "deterministic_algorithms": True,
        "deterministic_debug_mode": 2,
        "math_sdpa": True,
        "flash_sdpa": False,
        "memory_efficient_sdpa": False,
        "cudnn_sdpa": False,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "allow_tf32_matmul": False,
        "allow_tf32_cudnn": False,
        "float32_matmul_precision": "highest",
        "allow_fp16_reduced_precision_reduction": False,
        "allow_bf16_reduced_precision_reduction": False,
        "torch_num_threads": 1,
        "torch_num_interop_threads": 1,
        "cuda_initialized": False,
    }
    if configured != expected_configured:
        raise P23ProvenanceError(
            f"failed to establish the pre-CUDA determinism contract: {configured}"
        )
    return configured


def _default_nvidia_probe(device_uuid: str) -> dict[str, object]:
    fields = "uuid,pci.bus_id,name,driver_version,vbios_version,memory.total,mig.mode.current"
    output = (
        _default_command_runner(
            (
                "nvidia-smi",
                f"--id={device_uuid}",
                f"--query-gpu={fields}",
                "--format=csv,noheader,nounits",
            ),
            None,
        )
        .decode()
        .strip()
    )
    rows = [row.strip() for row in output.splitlines() if row.strip()]
    if len(rows) != 1:
        raise P23ProvenanceError("nvidia-smi did not identify exactly one pinned GPU")
    values = [item.strip() for item in next(csv.reader([rows[0]]))]
    if len(values) != 7:
        raise P23ProvenanceError("nvidia-smi identity row has an unexpected format")
    return dict(zip(fields.split(","), values, strict=True))


def _default_cuda_runtime_version_probe() -> int:
    """Read the loaded CUDA runtime's integer version via cudaRuntimeGetVersion."""

    discovered = ctypes.util.find_library("cudart")
    candidates: list[str | None] = [
        discovered,
        None,  # Symbols already loaded into the current Torch process.
        "libcudart.so",
        "libcudart.so.12",
        "libcudart.so.11.0",
    ]
    errors: list[str] = []
    for candidate in dict.fromkeys(candidates):
        if candidate == "":
            continue
        try:
            library = ctypes.CDLL(candidate)
            function = library.cudaRuntimeGetVersion
            function.argtypes = [ctypes.POINTER(ctypes.c_int)]
            function.restype = ctypes.c_int
            version = ctypes.c_int()
            status = int(function(ctypes.byref(version)))
        except (AttributeError, OSError) as error:
            errors.append(type(error).__name__)
            continue
        if status != 0 or version.value <= 0:
            errors.append(f"status_{status}")
            continue
        return int(version.value)
    raise P23ProvenanceError(
        "cudaRuntimeGetVersion could not be called from the pinned CUDA runtime "
        f"({','.join(errors) or 'no libcudart candidate'})"
    )


def _host_software_identity() -> dict[str, object]:
    """Collect the exact Linux/kernel/CPU fields frozen by the runtime lock."""

    cpuinfo = Path("/proc/cpuinfo")
    proc_version = Path("/proc/version")
    if not cpuinfo.is_file() or not proc_version.is_file():
        raise P23ProvenanceError("P23 CUDA acquisition requires Linux /proc host identity")
    decoded = cpuinfo.read_text(encoding="utf-8", errors="replace")
    records: list[dict[str, str]] = []
    for block in decoded.strip().split("\n\n"):
        record: dict[str, str] = {}
        for line in block.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            record[key.strip().lower()] = value.strip()
        if record:
            records.append(record)
    if not records:
        raise P23ProvenanceError("/proc/cpuinfo has no processor records")
    stable_cpu_fields = {
        "vendor_id": "vendor_id",
        "cpu_family": "cpu family",
        "model": "model",
        "model_name": "model name",
        "stepping": "stepping",
        "microcode": "microcode",
    }
    cpu_identity: dict[str, object] = {}
    for output_name, proc_name in stable_cpu_fields.items():
        values = {record.get(proc_name, "") for record in records}
        if len(values) != 1 or not next(iter(values)):
            raise P23ProvenanceError(f"/proc/cpuinfo has inconsistent {proc_name}")
        cpu_identity[output_name] = next(iter(values))
    flag_sets = []
    for record in records:
        flags = record.get("flags", record.get("features", ""))
        if not flags:
            raise P23ProvenanceError("/proc/cpuinfo omits stable CPU feature flags")
        flag_sets.append(tuple(sorted(set(flags.split()))))
    if len(set(flag_sets)) != 1:
        raise P23ProvenanceError("/proc/cpuinfo exposes inconsistent CPU feature flags")
    cpu_identity["flags_sha256"] = canonical_json_sha256(flag_sets[0])
    cpu_identity["logical_processor_count"] = len(records)
    processor = platform.processor().strip() or str(cpu_identity["model_name"])
    libc_name, libc_version = platform.libc_ver()
    libc = f"{libc_name}-{libc_version}" if libc_name and libc_version else "unknown"
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": processor or "unknown",
        "libc": libc,
        "cpu_identity": cpu_identity,
        "proc_version_sha256": sha256_file(proc_version),
    }


def _collect_cuda_runtime_components(
    *,
    torch_module: ModuleType | None = None,
    environ: Mapping[str, str] | None = None,
    nvidia_probe: Callable[[str], Mapping[str, object]] = _default_nvidia_probe,
    cuda_runtime_version_probe: Callable[[], int] = _default_cuda_runtime_version_probe,
    host_software_probe: Callable[[], Mapping[str, object]] = _host_software_identity,
    hostname_probe: Callable[[], str] = socket.gethostname,
    mountinfo_probe: Callable[[], bytes] = read_live_mountinfo_bytes,
) -> dict[str, Any]:
    """Collect live CUDA/runtime fields, excluding the separately frozen lock."""
    if torch_module is None:
        import torch as torch_module  # type: ignore[no-redef]

    env = os.environ if environ is None else environ
    cuda = torch_module.cuda
    if not cuda.is_available() or cuda.device_count() != 1:
        raise P23ProvenanceError("P23 requires exactly one available CUDA device")
    if cuda.current_device() != 0:
        raise P23ProvenanceError("the sole visible CUDA device must be logical device zero")
    if not cuda.is_bf16_supported():
        raise P23ProvenanceError("the pinned CUDA device must support BF16")
    visible = env.get("CUDA_VISIBLE_DEVICES")
    nvidia_visible = env.get("NVIDIA_VISIBLE_DEVICES")
    if not visible or "," in visible or _GPU_UUID.fullmatch(visible) is None:
        raise P23ProvenanceError("CUDA_VISIBLE_DEVICES must pin exactly one full GPU UUID")
    if nvidia_visible != visible:
        raise P23ProvenanceError(
            "NVIDIA_VISIBLE_DEVICES must equal the same full GPU UUID as CUDA_VISIBLE_DEVICES"
        )

    properties = cuda.get_device_properties(0)
    device_uuid = visible
    nvidia = dict(nvidia_probe(device_uuid))
    if nvidia.get("uuid") != device_uuid:
        raise P23ProvenanceError("nvidia-smi UUID disagrees with PyTorch device identity")
    if str(nvidia.get("mig.mode.current", "")).strip().lower() != "disabled":
        raise P23ProvenanceError("nvidia-smi does not report MIG mode disabled")
    try:
        nvidia_memory_mib = int(str(nvidia.get("memory.total")))
    except (TypeError, ValueError) as error:
        raise P23ProvenanceError("nvidia-smi memory.total is not an integer MiB value") from error

    backends = torch_module.backends
    cuda_backend = backends.cuda
    cudnn = backends.cudnn
    matmul = cuda_backend.matmul
    host_software = dict(host_software_probe())
    torch_config = torch_module.__config__.show()
    python_executable = Path(sys.executable)
    if str(python_executable) != PINNED_PYTHON_EXECUTABLE:
        raise P23ProvenanceError(
            f"P23 must run under the image-resident {PINNED_PYTHON_EXECUTABLE}"
        )
    result = {
        "backend": "cuda",
        "live_hostname": hostname_probe(),
        "live_mountinfo_sha256": hashlib.sha256(mountinfo_probe()).hexdigest(),
        "python_version": sys.version,
        "python_executable": str(python_executable),
        "python_executable_sha256": sha256_file(python_executable.resolve(strict=True)),
        "python_flags": validate_python_interpreter_flags(),
        "numpy_version": np.__version__,
        "torch_version": torch_module.__version__,
        "torch_git_version": torch_module.version.git_version,
        "torch_cuda_compiled_version": torch_module.version.cuda,
        "cuda_runtime_version": cuda_runtime_version_probe(),
        "cudnn_version": cudnn.version(),
        "torch_config_sha256": hashlib.sha256(torch_config.encode("utf-8")).hexdigest(),
        **host_software,
        "gpu": {
            "logical_index": 0,
            "visible_device_count": 1,
            "uuid": device_uuid,
            "name": properties.name,
            "pci_bus_id": nvidia.get("pci.bus_id"),
            "driver_version": nvidia.get("driver_version"),
            "vbios_version": nvidia.get("vbios_version"),
            "compute_capability": [properties.major, properties.minor],
            "multiprocessor_count": properties.multi_processor_count,
            "total_memory_bytes": properties.total_memory,
            "nvidia_smi_memory_total_mib": nvidia_memory_mib,
            "mig_mode": "disabled",
            "bf16_supported": True,
        },
        "sdpa": {
            "selected_backend": "math",
            "math_enabled": _backend_flag(cuda_backend, "math_sdp_enabled"),
            "flash_enabled": _backend_flag(cuda_backend, "flash_sdp_enabled"),
            "memory_efficient_enabled": _backend_flag(cuda_backend, "mem_efficient_sdp_enabled"),
            "cudnn_enabled": _backend_flag(cuda_backend, "cudnn_sdp_enabled"),
        },
        "determinism": {
            "deterministic_algorithms": torch_module.are_deterministic_algorithms_enabled(),
            "deterministic_debug_mode": torch_module.get_deterministic_debug_mode(),
            "cudnn_deterministic": cudnn.deterministic,
            "cudnn_benchmark": cudnn.benchmark,
            "allow_tf32_matmul": matmul.allow_tf32,
            "allow_tf32_cudnn": cudnn.allow_tf32,
            "float32_matmul_precision": torch_module.get_float32_matmul_precision(),
            "allow_fp16_reduced_precision_reduction": getattr(
                matmul, "allow_fp16_reduced_precision_reduction", None
            ),
            "allow_bf16_reduced_precision_reduction": getattr(
                matmul, "allow_bf16_reduced_precision_reduction", None
            ),
            "torch_num_threads": torch_module.get_num_threads(),
            "torch_num_interop_threads": torch_module.get_num_interop_threads(),
        },
        "environment": {
            name: env.get(name)
            for name in (
                "PATH",
                "CUDA_VISIBLE_DEVICES",
                "NVIDIA_VISIBLE_DEVICES",
                "CUBLAS_WORKSPACE_CONFIG",
                "PYTHONHASHSEED",
                "NVIDIA_TF32_OVERRIDE",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VIRTUAL_ENV",
                "PYTHONNOUSERSITE",
                "PYTHONDONTWRITEBYTECODE",
                "PYTHONOPTIMIZE",
                "PYTHONPATH",
                "PYTHONHOME",
                "LD_PRELOAD",
                "LD_LIBRARY_PATH",
                "LD_AUDIT",
            )
        },
    }
    return result


def collect_cuda_runtime(
    *,
    container_identity: Mapping[str, object],
    torch_module: ModuleType | None = None,
    environ: Mapping[str, str] | None = None,
    nvidia_probe: Callable[[str], Mapping[str, object]] = _default_nvidia_probe,
    cuda_runtime_version_probe: Callable[[], int] = _default_cuda_runtime_version_probe,
    host_software_probe: Callable[[], Mapping[str, object]] = _host_software_identity,
    hostname_probe: Callable[[], str] = socket.gethostname,
    mountinfo_probe: Callable[[], bytes] = read_live_mountinfo_bytes,
    injected_probe: Callable[[], Mapping[str, object]] | None = None,
) -> dict[str, Any]:
    """Collect and validate the exact CUDA arithmetic/runtime contract."""

    if injected_probe is not None:
        result = copy.deepcopy(dict(injected_probe()))
    else:
        result = _collect_cuda_runtime_components(
            torch_module=torch_module,
            environ=environ,
            nvidia_probe=nvidia_probe,
            cuda_runtime_version_probe=cuda_runtime_version_probe,
            host_software_probe=host_software_probe,
            hostname_probe=hostname_probe,
            mountinfo_probe=mountinfo_probe,
        )
        result["container"] = copy.deepcopy(dict(container_identity))
    validate_cuda_runtime_map(result)
    validate_runtime_against_lock(result, container_identity)
    return result


def validate_cuda_runtime_map(runtime: Mapping[str, object]) -> None:
    """Apply the P23 CUDA-only deterministic-runtime gate."""

    expected_runtime_fields = {
        "backend",
        "live_hostname",
        "live_mountinfo_sha256",
        "python_version",
        "python_executable",
        "python_executable_sha256",
        "python_flags",
        "numpy_version",
        "torch_version",
        "torch_git_version",
        "torch_cuda_compiled_version",
        "cuda_runtime_version",
        "cudnn_version",
        "torch_config_sha256",
        "platform",
        "system",
        "release",
        "version",
        "machine",
        "processor",
        "libc",
        "cpu_identity",
        "proc_version_sha256",
        "gpu",
        "container",
        "sdpa",
        "determinism",
        "environment",
    }
    if set(runtime) != expected_runtime_fields:
        raise P23ProvenanceError("P23 runtime fields do not match the frozen schema")
    if runtime.get("backend") != "cuda":
        raise P23ProvenanceError("P23 runtime must be CUDA")
    if not _is_sha256(runtime.get("live_mountinfo_sha256")):
        raise P23ProvenanceError("P23 runtime has no live mountinfo SHA-256")
    for field in (
        "python_version",
        "platform",
        "system",
        "release",
        "version",
        "machine",
        "processor",
        "libc",
        "numpy_version",
        "torch_version",
        "torch_git_version",
        "torch_cuda_compiled_version",
        "cudnn_version",
        "torch_config_sha256",
        "proc_version_sha256",
    ):
        value = runtime.get(field)
        if value in (None, ""):
            raise P23ProvenanceError(f"P23 runtime omits {field}")
    if not _is_sha256(runtime.get("torch_config_sha256")):
        raise P23ProvenanceError("P23 torch configuration digest is invalid")
    if runtime.get("python_executable") != PINNED_PYTHON_EXECUTABLE:
        raise P23ProvenanceError("P23 runtime does not use the pinned Python executable")
    if not _is_pinned_python_version_string(runtime.get("python_version")):
        raise P23ProvenanceError("P23 runtime does not use Python 3.12")
    if not _is_sha256(runtime.get("python_executable_sha256")):
        raise P23ProvenanceError("P23 Python executable digest is invalid")
    python_flags = runtime.get("python_flags")
    if not isinstance(python_flags, Mapping) or not _python_flag_mapping_is_frozen(python_flags):
        raise P23ProvenanceError("P23 Python interpreter flags are not exactly pinned")
    if not _is_sha256(runtime.get("proc_version_sha256")):
        raise P23ProvenanceError("P23 proc_version_sha256 is invalid")
    cpu_identity = runtime.get("cpu_identity")
    if not isinstance(cpu_identity, Mapping) or not _is_sha256(cpu_identity.get("flags_sha256")):
        raise P23ProvenanceError("P23 stable CPU identity is invalid")
    cuda_runtime_version = runtime.get("cuda_runtime_version")
    if (
        not isinstance(cuda_runtime_version, int)
        or isinstance(cuda_runtime_version, bool)
        or cuda_runtime_version <= 0
    ):
        raise P23ProvenanceError("P23 cudaRuntimeGetVersion result is invalid")
    gpu = runtime.get("gpu")
    if not isinstance(gpu, Mapping):
        raise P23ProvenanceError("P23 runtime has no GPU identity map")
    expected_gpu_fields = {
        "logical_index",
        "visible_device_count",
        "uuid",
        "name",
        "pci_bus_id",
        "driver_version",
        "vbios_version",
        "compute_capability",
        "multiprocessor_count",
        "total_memory_bytes",
        "nvidia_smi_memory_total_mib",
        "mig_mode",
        "bf16_supported",
    }
    if set(gpu) != expected_gpu_fields:
        raise P23ProvenanceError("P23 GPU identity fields do not match the frozen schema")
    if gpu.get("visible_device_count") != 1 or gpu.get("logical_index") != 0:
        raise P23ProvenanceError("P23 runtime must expose one logical CUDA device")
    if _GPU_UUID.fullmatch(str(gpu.get("uuid"))) is None:
        raise P23ProvenanceError("P23 runtime has no exact GPU/MIG UUID")
    if str(gpu.get("uuid")).startswith("MIG-"):
        raise P23ProvenanceError("P23 runtime must use a non-MIG full GPU")
    if gpu.get("mig_mode") != "disabled":
        raise P23ProvenanceError("P23 runtime must have MIG mode disabled")
    for field in (
        "name",
        "pci_bus_id",
        "driver_version",
        "compute_capability",
        "multiprocessor_count",
        "total_memory_bytes",
    ):
        if gpu.get(field) in (None, "", []):
            raise P23ProvenanceError(f"P23 GPU identity omits {field}")
    if gpu.get("bf16_supported") is not True:
        raise P23ProvenanceError("P23 GPU does not report BF16 support")

    runtime_lock = runtime.get("container")
    if not isinstance(runtime_lock, Mapping):
        raise P23ProvenanceError("P23 runtime has no CUDA/container lock")
    container = _validate_container_identity(
        runtime_lock.get("container"), require_host_attestation_sha256=True
    )
    if runtime.get("live_hostname") != container["hostname"]:
        raise P23ProvenanceError("live process hostname differs from the inspected container")
    if runtime.get("live_mountinfo_sha256") != container["mountinfo_sha256"]:
        raise P23ProvenanceError("live mount namespace differs from the inspected container")
    if not _is_sha256(runtime_lock.get("runtime_lock_sha256")):
        raise P23ProvenanceError("P23 runtime does not bind the runtime-lock bytes")

    sdpa = runtime.get("sdpa")
    expected_sdpa = {
        "selected_backend": "math",
        "math_enabled": True,
        "flash_enabled": False,
        "memory_efficient_enabled": False,
        "cudnn_enabled": False,
    }
    if not isinstance(sdpa, Mapping) or dict(sdpa) != expected_sdpa:
        raise P23ProvenanceError("P23 SDPA backend is not the pinned math-only backend")

    deterministic = runtime.get("determinism")
    if not isinstance(deterministic, Mapping):
        raise P23ProvenanceError("P23 runtime has no determinism map")
    exact_flags: dict[str, object] = {
        "deterministic_algorithms": True,
        "deterministic_debug_mode": 2,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "allow_tf32_matmul": False,
        "allow_tf32_cudnn": False,
        "float32_matmul_precision": "highest",
        "allow_fp16_reduced_precision_reduction": False,
        "allow_bf16_reduced_precision_reduction": False,
        "torch_num_threads": 1,
        "torch_num_interop_threads": 1,
    }
    if dict(deterministic) != exact_flags:
        raise P23ProvenanceError("P23 deterministic runtime map is not exactly pinned")
    environment = runtime.get("environment")
    if not isinstance(environment, Mapping):
        raise P23ProvenanceError("P23 runtime has no environment map")
    uuid = str(gpu["uuid"])
    exact_environment: dict[str, object] = {
        "PATH": PINNED_EXECUTABLE_PATH,
        "CUDA_VISIBLE_DEVICES": uuid,
        "NVIDIA_VISIBLE_DEVICES": uuid,
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONHASHSEED": "1337",
        "NVIDIA_TF32_OVERRIDE": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VIRTUAL_ENV": PINNED_PYTHON_ENVIRONMENT,
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONOPTIMIZE": None,
        "PYTHONPATH": None,
        "PYTHONHOME": None,
        "LD_PRELOAD": None,
        "LD_LIBRARY_PATH": None,
        "LD_AUDIT": None,
    }
    if dict(environment) != exact_environment:
        raise P23ProvenanceError("P23 deterministic environment map is not exactly pinned")


def validate_runtime_against_lock(
    runtime: Mapping[str, object], runtime_lock: Mapping[str, object]
) -> None:
    """Compare every pinned host/runtime field with the live CUDA map."""

    validate_cuda_runtime_map(runtime)
    if runtime_lock.get("schema_version") != P23_RUNTIME_LOCK_SCHEMA:
        raise P23ProvenanceError("runtime lock schema mismatch")
    if runtime_lock.get("status") != "pinned_for_acquisition":
        raise P23ProvenanceError("runtime lock is not pinned for acquisition")
    if runtime_lock.get("p23_addendum_sha256") != sha256_file(DEFAULT_ADDENDUM_PATH):
        raise P23ProvenanceError("runtime lock does not bind the P23 addendum")
    if runtime.get("container") != runtime_lock:
        raise P23ProvenanceError("live runtime does not retain the complete runtime lock")
    gpu = runtime["gpu"]
    assert isinstance(gpu, Mapping)
    locked_gpu = runtime_lock.get("gpu")
    if not isinstance(locked_gpu, Mapping):
        raise P23ProvenanceError("runtime lock has no GPU mapping")
    gpu_fields = {
        "uuid": "uuid",
        "pci_bus_id": "pci_bus_id",
        "name": "name",
        "vbios_version": "vbios_version",
        "driver_version": "driver_version",
        "compute_capability": "compute_capability",
        "multiprocessor_count": "multiprocessor_count",
        "total_memory_bytes": "total_memory",
        "nvidia_smi_memory_total_mib": "nvidia_smi_memory_total_mib",
        "mig_mode": "mig_mode",
    }
    for observed_field, locked_field in gpu_fields.items():
        if gpu.get(observed_field) != locked_gpu.get(locked_field):
            raise P23ProvenanceError(
                f"live GPU field {observed_field!r} differs from the runtime lock"
            )

    software = runtime_lock.get("software")
    if not isinstance(software, Mapping):
        raise P23ProvenanceError("runtime lock has no software mapping")
    software_fields = {
        "python_version": "python",
        "python_executable": "python_executable",
        "python_executable_sha256": "python_executable_sha256",
        "python_flags": "python_flags",
        "torch_version": "torch_version",
        "torch_git_version": "torch_git_version",
        "torch_config_sha256": "torch_config_sha256",
        "torch_cuda_compiled_version": "torch_cuda_compiled_version",
        "cuda_runtime_version": "cuda_runtime_version",
        "cudnn_version": "cudnn_version",
        "numpy_version": "numpy_version",
        "platform": "platform",
        "system": "system",
        "release": "release",
        "version": "version",
        "machine": "machine",
        "processor": "processor",
        "libc": "libc",
        "cpu_identity": "cpu_identity",
        "proc_version_sha256": "proc_version_sha256",
    }
    for observed_field, locked_field in software_fields.items():
        if runtime.get(observed_field) != software.get(locked_field):
            raise P23ProvenanceError(
                f"live software field {observed_field!r} differs from the runtime lock"
            )

    deterministic = runtime.get("determinism")
    locked_deterministic = runtime_lock.get("determinism")
    assert isinstance(deterministic, Mapping)
    if not isinstance(locked_deterministic, Mapping):
        raise P23ProvenanceError("runtime lock has no determinism mapping")
    deterministic_fields = {
        "deterministic_algorithms": "deterministic_algorithms",
        "cudnn_deterministic": "cudnn_deterministic",
        "cudnn_benchmark": "cudnn_benchmark",
        "allow_tf32_matmul": "tf32",
        "allow_tf32_cudnn": "tf32",
        "float32_matmul_precision": "float32_matmul_precision",
        "allow_fp16_reduced_precision_reduction": "fp16_reduced_precision_reduction",
        "allow_bf16_reduced_precision_reduction": "bf16_reduced_precision_reduction",
        "torch_num_threads": "cpu_threads",
        "torch_num_interop_threads": "interop_threads",
    }
    for observed_field, locked_field in deterministic_fields.items():
        if deterministic.get(observed_field) != locked_deterministic.get(locked_field):
            raise P23ProvenanceError(
                f"live determinism field {observed_field!r} differs from the runtime lock"
            )
    if (
        deterministic.get("deterministic_debug_mode") != 2
        or locked_deterministic.get("deterministic_debug_mode") != "error"
    ):
        raise P23ProvenanceError("live deterministic debug mode differs from the runtime lock")
    environment = runtime.get("environment")
    if not isinstance(environment, Mapping):
        raise P23ProvenanceError("live runtime has no deterministic environment map")
    environment_fields = {
        "CUDA_VISIBLE_DEVICES": "cuda_visible_devices",
        "NVIDIA_VISIBLE_DEVICES": "nvidia_visible_devices",
        "CUBLAS_WORKSPACE_CONFIG": "cublas_workspace_config",
        "PYTHONHASHSEED": "pythonhashseed",
        "NVIDIA_TF32_OVERRIDE": "nvidia_tf32_override",
    }
    for observed_field, locked_field in environment_fields.items():
        if environment.get(observed_field) != locked_deterministic.get(locked_field):
            raise P23ProvenanceError(
                f"live environment field {observed_field!r} differs from the runtime lock"
            )
    if environment.get("OMP_NUM_THREADS") != str(locked_deterministic.get("cpu_threads")):
        raise P23ProvenanceError("live OMP_NUM_THREADS differs from the runtime lock")
    if environment.get("MKL_NUM_THREADS") != str(locked_deterministic.get("cpu_threads")):
        raise P23ProvenanceError("live MKL_NUM_THREADS differs from the runtime lock")
    locked_loader_environment = runtime_lock.get("loader_environment")
    if not isinstance(locked_loader_environment, Mapping):
        raise P23ProvenanceError("runtime lock has no loader-environment mapping")
    for name, expected in locked_loader_environment.items():
        if environment.get(name) != expected:
            raise P23ProvenanceError(f"live {name} differs from the runtime lock")


def _json_artifact_bytes(payload: Mapping[str, object]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def freeze_runtime_artifacts(
    *,
    container_image: str,
    container_repository_digest: str,
    host_image_inspection: Path,
    host_running_container_inspection: Path,
    host_running_mountinfo: Path,
    host_nvidia_smi_query: Path,
    host_attestation_output: Path,
    runtime_lock_output: Path,
    addendum_path: Path = DEFAULT_ADDENDUM_PATH,
    torch_module: ModuleType | None = None,
    environ: Mapping[str, str] | None = None,
    nvidia_probe: Callable[[str], Mapping[str, object]] = _default_nvidia_probe,
    cuda_runtime_version_probe: Callable[[], int] = _default_cuda_runtime_version_probe,
    host_software_probe: Callable[[], Mapping[str, object]] = _host_software_identity,
    hostname_probe: Callable[[], str] = socket.gethostname,
    runtime_component_probe: Callable[[], Mapping[str, object]] | None = None,
    python_flags: object | None = None,
) -> dict[str, object]:
    """Generate the two pre-acquisition artifacts from live and host evidence.

    The host first retains raw image inspection, actual-running-container
    inspection, and `nvidia-smi` outputs.  This in-container stage hashes those
    bytes, links the running container to the immutable image ID/repository
    digest, cross-checks the live hostname and CUDA device, and emits
    ready-to-review artifacts without hand-authored JSON.  This is a procedural
    host-inspection chain, not cryptographic remote attestation.
    """

    _validate_pinned_image_reference(container_image, container_repository_digest)
    image_inspection_bytes = host_image_inspection.resolve().read_bytes()
    running_inspection_bytes = host_running_container_inspection.resolve().read_bytes()
    running_mountinfo_bytes = host_running_mountinfo.resolve().read_bytes()
    _parse_mountinfo(running_mountinfo_bytes)
    nvidia_bytes = host_nvidia_smi_query.resolve().read_bytes()
    image_record = _single_inspection_record(image_inspection_bytes, "host image inspection")
    image_id = image_record.get("Id")
    repo_digests = image_record.get("RepoDigests")
    if not isinstance(image_id, str) or _OCI_DIGEST.fullmatch(image_id) is None:
        raise P23ProvenanceError("host image inspection has no immutable image ID")
    if not isinstance(repo_digests, list) or container_image not in repo_digests:
        raise P23ProvenanceError("host image inspection does not list the pinned image digest")

    running_record = _single_inspection_record(
        running_inspection_bytes, "host running-container inspection"
    )
    container_id = running_record.get("Id")
    if not isinstance(container_id, str) or _CONTAINER_ID.fullmatch(container_id) is None:
        raise P23ProvenanceError("running-container inspection has no canonical container ID")
    if running_record.get("Image") != image_id:
        raise P23ProvenanceError("running container does not use the inspected immutable image ID")
    state = running_record.get("State")
    if not isinstance(state, Mapping) or state.get("Running") is not True:
        raise P23ProvenanceError("host inspection does not describe a running container")
    container_init_pid = state.get("Pid")
    if (
        not isinstance(container_init_pid, int)
        or isinstance(container_init_pid, bool)
        or container_init_pid <= 0
    ):
        raise P23ProvenanceError("host inspection has no valid running-container init PID")
    config = running_record.get("Config")
    if not isinstance(config, Mapping):
        raise P23ProvenanceError("running-container inspection has no configuration")
    hostname = config.get("Hostname")
    if hostname != container_id[:12]:
        raise P23ProvenanceError("running container does not use its canonical default hostname")
    if config.get("Image") != container_image:
        raise P23ProvenanceError("running container was not launched by the pinned image reference")
    host_config = running_record.get("HostConfig")
    if not isinstance(host_config, Mapping):
        raise P23ProvenanceError("running-container inspection has no host configuration")
    if host_config.get("ReadonlyRootfs") is not True:
        raise P23ProvenanceError("running container root filesystem is not read-only")
    if host_config.get("NetworkMode") != PINNED_CONTAINER_NETWORK_MODE:
        raise P23ProvenanceError("running container network mode is not disabled")
    if host_config.get("Tmpfs") != REQUIRED_CONTAINER_TMPFS:
        raise P23ProvenanceError("running-container tmpfs map differs from the exact allowlist")
    gpu_selection = _validate_running_container_gpu_selection(config, host_config)
    mount_contract = _validate_running_container_mounts(running_record.get("Mounts"))
    if not nvidia_bytes:
        raise P23ProvenanceError("host nvidia-smi evidence is empty")
    attestation_output = host_attestation_output.resolve()
    lock_output = runtime_lock_output.resolve()
    if attestation_output == lock_output or attestation_output.parent != lock_output.parent:
        raise P23ProvenanceError("runtime artifacts must be distinct files in one directory")
    if attestation_output.exists() or lock_output.exists():
        raise P23ProvenanceError("runtime artifact outputs must not already exist")
    if attestation_output.name.endswith(".template.json") or lock_output.name.endswith(
        ".template.json"
    ):
        raise P23ProvenanceError("freeze-runtime must not overwrite committed null templates")

    configure_cuda_determinism(
        torch_module=torch_module,
        environ=environ,
        python_flags=python_flags,
    )
    if runtime_component_probe is None:
        components = _collect_cuda_runtime_components(
            torch_module=torch_module,
            environ=environ,
            nvidia_probe=nvidia_probe,
            cuda_runtime_version_probe=cuda_runtime_version_probe,
            host_software_probe=host_software_probe,
            hostname_probe=hostname_probe,
        )
    else:
        components = copy.deepcopy(dict(runtime_component_probe()))
    gpu = components.get("gpu")
    if not isinstance(gpu, Mapping):
        raise P23ProvenanceError("live runtime snapshot has no GPU identity")
    if components.get("live_hostname") != hostname:
        raise P23ProvenanceError("live process hostname differs from host container inspection")
    mountinfo_sha256 = hashlib.sha256(running_mountinfo_bytes).hexdigest()
    if components.get("live_mountinfo_sha256") != mountinfo_sha256:
        raise P23ProvenanceError(
            "live mount namespace differs from host running-container mountinfo"
        )
    stable_gpu = {
        "uuid": gpu.get("uuid"),
        "pci_bus_id": gpu.get("pci_bus_id"),
        "name": gpu.get("name"),
        "vbios_version": gpu.get("vbios_version"),
        "driver_version": gpu.get("driver_version"),
        "compute_capability": gpu.get("compute_capability"),
        "multiprocessor_count": gpu.get("multiprocessor_count"),
        "total_memory": gpu.get("total_memory_bytes"),
        "nvidia_smi_memory_total_mib": gpu.get("nvidia_smi_memory_total_mib"),
        "mig_mode": gpu.get("mig_mode"),
    }
    if gpu_selection["requested_full_gpu_uuid"] != stable_gpu["uuid"]:
        raise P23ProvenanceError(
            "host-inspected Docker GPU request differs from the live CUDA identity"
        )
    try:
        rows = [
            [field.strip() for field in row]
            for row in csv.reader(io.StringIO(nvidia_bytes.decode("utf-8")))
            if any(field.strip() for field in row)
        ]
    except UnicodeDecodeError as error:
        raise P23ProvenanceError("host nvidia-smi evidence is not UTF-8 CSV") from error
    if len(rows) != 1 or len(rows[0]) != 7:
        raise P23ProvenanceError(
            "host nvidia-smi evidence must contain exactly one seven-field row"
        )
    expected_host_row = [
        str(stable_gpu["uuid"]),
        str(stable_gpu["pci_bus_id"]),
        str(stable_gpu["name"]),
        str(stable_gpu["driver_version"]),
        str(stable_gpu["vbios_version"]),
        str(stable_gpu["nvidia_smi_memory_total_mib"]),
        "Disabled",
    ]
    if rows[0] != expected_host_row or stable_gpu["mig_mode"] != "disabled":
        raise P23ProvenanceError("host nvidia-smi evidence differs from the live CUDA identity")

    host_attestation: dict[str, object] = {
        "schema_version": P23_HOST_ATTESTATION_SCHEMA,
        "status": "procedurally_host_attested",
        "container": {
            "image": container_image,
            "repository_digest": container_repository_digest,
            "container_id": container_id,
            "image_id": image_id,
            "container_init_pid": container_init_pid,
            "hostname": hostname,
            "default_hostname": True,
            "rootfs_read_only": True,
            "network_mode": PINNED_CONTAINER_NETWORK_MODE,
            "gpu_selection": copy.deepcopy(gpu_selection),
            "mountinfo_sha256": mountinfo_sha256,
            "mount_contract": mount_contract,
            "tmpfs_contract": copy.deepcopy(_SANITIZED_TMPFS_CONTRACT),
            "attestation_scope": CONTAINER_ATTESTATION_SCOPE,
        },
        "gpu": stable_gpu,
        "evidence": {
            "image_inspection_sha256": hashlib.sha256(image_inspection_bytes).hexdigest(),
            "running_container_inspection_sha256": hashlib.sha256(
                running_inspection_bytes
            ).hexdigest(),
            "running_mountinfo_sha256": mountinfo_sha256,
            "nvidia_smi_query_sha256": hashlib.sha256(nvidia_bytes).hexdigest(),
        },
    }
    software_keys = (
        "python_version",
        "python_executable",
        "python_executable_sha256",
        "torch_version",
        "torch_git_version",
        "torch_config_sha256",
        "torch_cuda_compiled_version",
        "cuda_runtime_version",
        "cudnn_version",
        "numpy_version",
        "platform",
        "system",
        "release",
        "version",
        "machine",
        "processor",
        "libc",
        "cpu_identity",
        "proc_version_sha256",
        "python_flags",
    )
    software_names = {
        "python_version": "python",
        **{key: key for key in software_keys if key != "python_version"},
    }
    software = {
        output_name: copy.deepcopy(components.get(component_name))
        for component_name, output_name in software_names.items()
    }
    uuid = stable_gpu["uuid"]
    runtime_lock: dict[str, object] = {
        "schema_version": P23_RUNTIME_LOCK_SCHEMA,
        "status": "pinned_for_acquisition",
        "p23_addendum_sha256": sha256_file(addendum_path.resolve()),
        "container": {
            "image": container_image,
            "repository_digest": container_repository_digest,
            "container_id": container_id,
            "image_id": image_id,
            "container_init_pid": container_init_pid,
            "hostname": hostname,
            "default_hostname": True,
            "rootfs_read_only": True,
            "network_mode": PINNED_CONTAINER_NETWORK_MODE,
            "gpu_selection": copy.deepcopy(gpu_selection),
            "mountinfo_sha256": mountinfo_sha256,
            "mount_contract": mount_contract,
            "tmpfs_contract": copy.deepcopy(_SANITIZED_TMPFS_CONTRACT),
            "attestation_scope": CONTAINER_ATTESTATION_SCOPE,
            "host_attestation_sha256": "pending",
        },
        "gpu": stable_gpu,
        "software": software,
        "determinism": {
            "cublas_workspace_config": ":4096:8",
            "pythonhashseed": "1337",
            "nvidia_tf32_override": "0",
            "cuda_visible_devices": uuid,
            "nvidia_visible_devices": uuid,
            "sdpa_backend": "math",
            "deterministic_algorithms": True,
            "deterministic_debug_mode": "error",
            "tf32": False,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "float32_matmul_precision": "highest",
            "fp16_reduced_precision_reduction": False,
            "bf16_reduced_precision_reduction": False,
            "cpu_threads": 1,
            "interop_threads": 1,
        },
        "loader_environment": {
            "PATH": PINNED_EXECUTABLE_PATH,
            "LD_PRELOAD": None,
            "LD_LIBRARY_PATH": None,
            "LD_AUDIT": None,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONOPTIMIZE": None,
        },
    }
    attestation_output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="p23-freeze-", dir=attestation_output.parent) as raw:
        staging = Path(raw)
        staged_attestation = staging / "host-attestation.json"
        staged_attestation.write_bytes(_json_artifact_bytes(host_attestation))
        locked_container = runtime_lock["container"]
        assert isinstance(locked_container, dict)
        locked_container["host_attestation_sha256"] = sha256_file(staged_attestation)
        staged_lock = staging / "runtime-lock.json"
        staged_lock.write_bytes(_json_artifact_bytes(runtime_lock))
        loaded_lock = load_runtime_lock(
            staged_lock,
            addendum_path=addendum_path,
            host_attestation_path=staged_attestation,
        )
        complete_runtime = {**components, "container": loaded_lock}
        validate_cuda_runtime_map(complete_runtime)
        validate_runtime_against_lock(complete_runtime, loaded_lock)
        os.replace(staged_attestation, attestation_output)
        os.replace(staged_lock, lock_output)
    return {
        "host_attestation": {
            "path": str(attestation_output),
            "sha256": sha256_file(attestation_output),
        },
        "runtime_lock": {
            "path": str(lock_output),
            "sha256": sha256_file(lock_output),
        },
        "next_action": "review_and_commit_both_artifacts_before_any_acquisition",
    }


def _addendum_record(path: Path, *, repository_root: Path = REPOSITORY_ROOT) -> dict[str, str]:
    resolved = path.resolve()
    root = repository_root.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise P23ProvenanceError("P23 addendum is outside the repository") from error
    return {
        "logical_path": f"repository:{relative.as_posix()}",
        "sha256": sha256_file(resolved),
    }


def build_execution_identity(
    *,
    addendum_path: Path,
    repository: Mapping[str, object],
    source_modules: Mapping[str, Mapping[str, object]],
    loaded_file_closure: Mapping[str, object],
    runtime: Mapping[str, object],
    static_contract: Mapping[str, object],
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, object]:
    """Build the full identity object shared by all three fresh processes."""

    load_and_validate_addendum(addendum_path, repository_root=repository_root)
    validate_cuda_runtime_map(runtime)
    validated_loaded_files = validate_loaded_file_closure(loaded_file_closure)
    _reject_credentialed_origin_fields(repository)
    _reject_credentialed_origin_fields(static_contract)
    identity: dict[str, object] = {
        "schema_version": P23_EXECUTION_IDENTITY_SCHEMA,
        "addendum": _addendum_record(addendum_path, repository_root=repository_root),
        "repository": copy.deepcopy(dict(repository)),
        "source_modules": copy.deepcopy(dict(source_modules)),
        "loaded_file_closure": validated_loaded_files,
        "runtime": copy.deepcopy(dict(runtime)),
        "static_contract": copy.deepcopy(dict(static_contract)),
    }
    return identity


def execution_identity_sha256(identity: Mapping[str, object]) -> str:
    """Independently hash an execution-identity object."""

    if identity.get("schema_version") != P23_EXECUTION_IDENTITY_SCHEMA:
        raise P23ProvenanceError("execution identity schema mismatch")
    return canonical_json_sha256(identity)


def verify_execution_identity(
    manifest: Mapping[str, object],
    *,
    addendum_path: Path,
    repository_collector: Callable[[], Mapping[str, object]],
    source_collector: Callable[[], Mapping[str, Mapping[str, object]]],
    loaded_file_initialized_collector: Callable[[], Mapping[str, object]],
    loaded_file_roots: Mapping[str, Path],
    runtime_collector: Callable[[], Mapping[str, object]],
    expected_static_contract: Mapping[str, object] | None = None,
    repository_root: Path = REPOSITORY_ROOT,
    mountinfo_text: bytes | str | None = None,
) -> dict[str, object]:
    """Recollect local identity maps and reject an opaque or stale digest."""

    load_and_validate_addendum(addendum_path, repository_root=repository_root)
    identity = manifest.get("execution_identity")
    if not isinstance(identity, Mapping):
        raise P23ProvenanceError("run manifest has no complete execution identity")
    recorded_digest = manifest.get("run_identity_sha256")
    recomputed_digest = execution_identity_sha256(identity)
    if recorded_digest != recomputed_digest:
        raise P23ProvenanceError("run identity digest does not match its complete identity map")
    expected_addendum = _addendum_record(addendum_path, repository_root=repository_root)
    if identity.get("addendum") != expected_addendum:
        raise P23ProvenanceError("run identity does not bind the current P23 addendum")
    recorded_loaded_files = identity.get("loaded_file_closure")
    if not isinstance(recorded_loaded_files, Mapping):
        raise P23ProvenanceError("run identity has no complete loaded-file closure")
    loaded_file_verification = verify_loaded_file_closure(
        recorded_loaded_files,
        logical_roots=loaded_file_roots,
        current_initialized=loaded_file_initialized_collector(),
        mountinfo_text=mountinfo_text,
    )

    if expected_static_contract is None:
        raise P23ProvenanceError("independent verification requires the expected static contract")
    current = build_execution_identity(
        addendum_path=addendum_path,
        repository=dict(repository_collector()),
        source_modules=dict(source_collector()),
        loaded_file_closure=dict(recorded_loaded_files),
        runtime=dict(runtime_collector()),
        static_contract=dict(expected_static_contract),
        repository_root=repository_root,
    )
    if dict(identity) != current:
        differing = sorted(
            key for key in set(identity) | set(current) if identity.get(key) != current.get(key)
        )
        raise P23ProvenanceError(
            f"run identity differs from independently reconstructed map: {differing}"
        )
    return {
        "passes": True,
        "comparison": "independently_recomputed_exact_identity",
        "run_identity_sha256": recomputed_digest,
        "source_module_count": len(current["source_modules"]),
        "loaded_file_verification": loaded_file_verification,
    }


def validate_run_manifest(manifest: Mapping[str, object], expected_mode: str) -> None:
    """Validate CUDA-only capture semantics and exact observation counts."""

    if expected_mode not in {"trace_off", "trace_on"}:
        raise P23ProvenanceError("expected_mode must be trace_off or trace_on")
    if manifest.get("schema_version") != P23_RUN_MANIFEST_SCHEMA:
        raise P23ProvenanceError("P23 run-manifest schema mismatch")
    if manifest.get("trace_mode") != expected_mode:
        raise P23ProvenanceError("P23 trace mode mismatch")
    identity = manifest.get("execution_identity")
    if not isinstance(identity, Mapping):
        raise P23ProvenanceError("P23 manifest has no execution identity")
    runtime = identity.get("runtime")
    if not isinstance(runtime, Mapping):
        raise P23ProvenanceError("P23 execution identity has no runtime")
    validate_cuda_runtime_map(runtime)
    if manifest.get("run_identity_sha256") != execution_identity_sha256(identity):
        raise P23ProvenanceError("P23 run identity is opaque or invalid")
    initial_state = manifest.get("initial_state")
    if not isinstance(initial_state, Mapping):
        raise P23ProvenanceError("P23 manifest has no exact initial state")
    recomputed_initial = _p22_initial_state_sha256(initial_state)
    if manifest.get("initial_state_sha256") != recomputed_initial:
        raise P23ProvenanceError("P23 initial-state digest does not match its component hashes")
    if manifest.get("shadow_update_applied") is not False:
        raise P23ProvenanceError("P23 shadow update must never be applied")
    candidate_execution = manifest.get("candidate_execution")
    expected_execution = {
        "backend": "cuda",
        "actual_accelerator_candidate_computed_and_applied": True,
        "shield_shadow_only": True,
    }
    if not isinstance(candidate_execution, Mapping) or dict(candidate_execution) != (
        expected_execution
    ):
        raise P23ProvenanceError("P23 candidate execution record changed")

    expected_observation = (
        TRACE_OFF_OBSERVATION if expected_mode == "trace_off" else TRACE_ON_OBSERVATION
    )
    if manifest.get("candidate_observation") != expected_observation:
        raise P23ProvenanceError(f"P23 {expected_mode} observation semantics changed")
    expected_capture_steps = [] if expected_mode == "trace_off" else list(EXPECTED_CAPTURE_STEPS)
    if manifest.get("capture_steps") != expected_capture_steps:
        raise P23ProvenanceError("P23 capture schedule mismatch")

    steps = manifest.get("steps")
    if not isinstance(steps, list) or len(steps) != EXPECTED_OPTIMIZER_STEPS:
        raise P23ProvenanceError("P23 manifest must contain 256 step records")
    observation_total = 0
    capture_count = 0
    for index, record in enumerate(steps):
        if not isinstance(record, Mapping) or record.get("step") != index:
            raise P23ProvenanceError("P23 step records are incomplete or out of order")
        if record.get("tokens_seen") != (index + 1) * EXPECTED_SEQUENCE_LENGTH:
            raise P23ProvenanceError("P23 tokens-seen schedule changed")
        if record.get("data_token_offset") != index * EXPECTED_SEQUENCE_LENGTH:
            raise P23ProvenanceError("P23 data-offset schedule changed")
        for field in (
            "batch_sha256",
            "loss_tensor_sha256",
            "rng_before_step_sha256",
            "rng_after_step_sha256",
        ):
            if not _is_sha256(record.get(field)):
                raise P23ProvenanceError(f"P23 step {index} has no exact {field}")
        checkpoint = index in EXPECTED_CAPTURE_STEPS
        if record.get("state_checkpoint_present") is not checkpoint:
            raise P23ProvenanceError(f"P23 step {index} state-checkpoint flag changed")
        for field in ("model_state_sha256", "optimizer_state_sha256"):
            value = record.get(field)
            if checkpoint and not _is_sha256(value):
                raise P23ProvenanceError(f"P23 checkpoint {index} has no exact {field}")
            if not checkpoint and value is not None:
                raise P23ProvenanceError(f"P23 step {index} has an unscheduled {field}")
        count = record.get("observation_count")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise P23ProvenanceError("P23 step observation_count is invalid")
        observation_total += count
        scheduled = expected_mode == "trace_on" and index in EXPECTED_CAPTURE_STEPS
        capture = record.get("capture")
        observer_rng_unchanged = record.get("observer_rng_unchanged")
        if scheduled:
            capture_count += 1
            if count != EXPECTED_PARAMETER_COUNT or not isinstance(capture, Mapping):
                raise P23ProvenanceError("P23 scheduled capture is incomplete")
            if capture.get("parameter_count") != EXPECTED_PARAMETER_COUNT:
                raise P23ProvenanceError("P23 capture does not contain all 48 matrices")
            if capture.get("actual_pre_aspect_candidate") is not False:
                raise P23ProvenanceError("P23 capture ambiguously records a pre-aspect candidate")
            if not isinstance(capture.get("pre_aspect_unavailable_reason"), str) or not capture.get(
                "pre_aspect_unavailable_reason"
            ):
                raise P23ProvenanceError("P23 capture omits the pre-aspect availability reason")
            if capture.get("actual_post_aspect_candidate") is not True:
                raise P23ProvenanceError("P23 capture is not the actual post-aspect candidate")
            if capture.get("actual_post_aspect_cuda_candidate") is not True:
                raise P23ProvenanceError("P23 capture is not the actual post-aspect CUDA value")
            if capture.get("p20_shadow_only") is not True:
                raise P23ProvenanceError("P23 capture does not keep the P20 shield shadow-only")
            if not _is_sha256(capture.get("record_sha256")):
                raise P23ProvenanceError("P23 capture has no exact record digest")
            if observer_rng_unchanged is not True:
                raise P23ProvenanceError("P23 observer did not preserve recorded RNG bytes")
        elif count != 0 or capture is not None or observer_rng_unchanged is not None:
            raise P23ProvenanceError("P23 contains an unscheduled candidate observation")
    if capture_count != expected_observation["capture_step_count"]:
        raise P23ProvenanceError("P23 capture-step count mismatch")
    if observation_total != expected_observation["observation_count"]:
        raise P23ProvenanceError("P23 total observation count mismatch")
    raw_trace = manifest.get("raw_trace")
    if expected_mode == "trace_off":
        if raw_trace is not None:
            raise P23ProvenanceError("P23 trace-off must not contain a raw candidate trace")
    else:
        if not isinstance(raw_trace, Mapping):
            raise P23ProvenanceError("P23 trace-on has no complete raw-trace record")
        if raw_trace.get("observation_count") != EXPECTED_OBSERVATION_COUNT:
            raise P23ProvenanceError("P23 raw trace does not contain exactly 1,152 observations")
        if not _is_sha256(raw_trace.get("sha256")):
            raise P23ProvenanceError("P23 raw trace has no exact byte digest")
    final_state = manifest.get("final_state")
    if not isinstance(final_state, Mapping) or set(final_state) != {
        "model_state_sha256",
        "optimizer_state_sha256",
        "rng_state_sha256",
    }:
        raise P23ProvenanceError("P23 manifest has no final state")
    for field in ("model_state_sha256", "optimizer_state_sha256", "rng_state_sha256"):
        if not _is_sha256(final_state.get(field)):
            raise P23ProvenanceError(f"P23 final state has no exact {field}")


def _strict_identity_match(left: Mapping[str, object], right: Mapping[str, object]) -> None:
    if dict(left) == dict(right):
        return
    differing = sorted(key for key in set(left) | set(right) if left.get(key) != right.get(key))
    raise P23ProvenanceError(f"P23 pair differs in execution identity fields {differing}")


def _exact_state_mismatches(
    left: Mapping[str, object], right: Mapping[str, object]
) -> list[dict[str, object]]:
    mismatches: list[dict[str, object]] = []
    if left.get("initial_state_sha256") != right.get("initial_state_sha256"):
        mismatches.append({"scope": "initial", "field": "initial_state_sha256"})
    left_initial = left.get("initial_state")
    right_initial = right.get("initial_state")
    if not isinstance(left_initial, Mapping) or not isinstance(right_initial, Mapping):
        raise P23ProvenanceError("P23 pair has no complete initial state")
    for field in ("model_state_sha256", "optimizer_state_sha256", "rng_state_sha256"):
        if left_initial.get(field) != right_initial.get(field):
            mismatches.append({"scope": "initial", "field": field})
    left_steps = left["steps"]
    right_steps = right["steps"]
    assert isinstance(left_steps, list) and isinstance(right_steps, list)
    step_fields = (
        "tokens_seen",
        "data_token_offset",
        "batch_sha256",
        "loss_tensor_sha256",
        "rng_before_step_sha256",
        "rng_after_step_sha256",
        "model_state_sha256",
        "optimizer_state_sha256",
    )
    for step, (left_record, right_record) in enumerate(zip(left_steps, right_steps, strict=True)):
        assert isinstance(left_record, Mapping) and isinstance(right_record, Mapping)
        for field in step_fields:
            if left_record.get(field) != right_record.get(field):
                mismatches.append({"scope": "step", "step": step, "field": field})
    left_final = left.get("final_state")
    right_final = right.get("final_state")
    if not isinstance(left_final, Mapping) or not isinstance(right_final, Mapping):
        raise P23ProvenanceError("P23 pair has no complete final state")
    for field in ("model_state_sha256", "optimizer_state_sha256", "rng_state_sha256"):
        if left_final.get(field) != right_final.get(field):
            mismatches.append({"scope": "final", "field": field})
    return mismatches


def compare_repeatability_manifests(
    trace_off_a: Mapping[str, object], trace_off_b: Mapping[str, object]
) -> dict[str, object]:
    """Compare two already independently verified trace-off manifests."""

    validate_run_manifest(trace_off_a, "trace_off")
    validate_run_manifest(trace_off_b, "trace_off")
    left_identity = trace_off_a["execution_identity"]
    right_identity = trace_off_b["execution_identity"]
    assert isinstance(left_identity, Mapping) and isinstance(right_identity, Mapping)
    _strict_identity_match(left_identity, right_identity)
    mismatches = _exact_state_mismatches(trace_off_a, trace_off_b)
    return {
        "schema_version": P23_REPEATABILITY_SCHEMA,
        "passes": not mismatches,
        "comparison": "bitwise_exact_no_tolerances",
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def compare_noninterference_manifests(
    trace_off_a: Mapping[str, object], trace_on: Mapping[str, object]
) -> dict[str, object]:
    """Compare an accepted trace-off run with the subsequent trace-on run."""

    validate_run_manifest(trace_off_a, "trace_off")
    validate_run_manifest(trace_on, "trace_on")
    left_identity = trace_off_a["execution_identity"]
    right_identity = trace_on["execution_identity"]
    assert isinstance(left_identity, Mapping) and isinstance(right_identity, Mapping)
    _strict_identity_match(left_identity, right_identity)
    mismatches = _exact_state_mismatches(trace_off_a, trace_on)
    return {
        "schema_version": P23_NONINTERFERENCE_SCHEMA,
        "passes": not mismatches,
        "comparison": "bitwise_exact_no_tolerances",
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "observation_count": EXPECTED_OBSERVATION_COUNT,
    }


def sanitize_complete_manifest(
    manifest: Mapping[str, object],
    *,
    path_roots: Mapping[str, Path],
) -> dict[str, object]:
    """Replace declared local absolute paths without dropping manifest fields."""

    _reject_credentialed_origin_fields(manifest)
    roots = {name: path.resolve() for name, path in path_roots.items()}
    if not roots:
        raise P23ProvenanceError("at least one sanitization root is required")
    replacements = 0

    def sanitize(value: object) -> object:
        nonlocal replacements
        if isinstance(value, Mapping):
            return {str(key): sanitize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [sanitize(item) for item in value]
        if isinstance(value, tuple):
            return [sanitize(item) for item in value]
        if value == PINNED_EXECUTABLE_PATH:
            return value
        if not isinstance(value, str) or not Path(value).is_absolute():
            return value
        resolved = Path(value).resolve()
        matches: list[tuple[str, Path]] = []
        for name, root in roots.items():
            try:
                relative = resolved.relative_to(root)
            except ValueError:
                continue
            matches.append((name, relative))
        if not matches:
            raise P23ProvenanceError(f"undeclared absolute path cannot be sanitized: {value}")
        name, relative = max(matches, key=lambda item: len(roots[item[0]].parts))
        replacements += 1
        return f"{name}:{relative.as_posix()}"

    sanitized_payload = sanitize(manifest)
    assert isinstance(sanitized_payload, dict)
    result: dict[str, object] = {
        "schema_version": P23_SANITIZED_MANIFEST_SCHEMA,
        "native_manifest_canonical_sha256": canonical_json_sha256(manifest),
        "path_replacement_count": replacements,
        "manifest": sanitized_payload,
    }
    rendered = json.dumps(result, sort_keys=True, allow_nan=False)
    if any(fragment in rendered for fragment in _PERSONAL_PATH_FRAGMENTS):
        raise P23ProvenanceError("sanitized manifest still contains a machine-local path")
    return result


__all__ = [
    "CONTAINER_ATTESTATION_SCOPE",
    "DEFAULT_ADDENDUM_PATH",
    "DOCKER_GPU_SELECTION_MECHANISM",
    "EXPECTED_CAPTURE_STEPS",
    "EXPECTED_OBSERVATION_COUNT",
    "FROZEN_PYTHON_FLAGS",
    "P23_ADDENDUM_SCHEMA",
    "P23_EXECUTION_IDENTITY_SCHEMA",
    "P23_HOST_ATTESTATION_SCHEMA",
    "P23_LOADED_FILE_CLOSURE_SCHEMA",
    "P23_LOADED_FILE_SNAPSHOT_SCHEMA",
    "P23_NONINTERFERENCE_SCHEMA",
    "P23_REPEATABILITY_SCHEMA",
    "P23_RUNTIME_LOCK_SCHEMA",
    "P23_RUN_MANIFEST_SCHEMA",
    "P23_SANITIZED_MANIFEST_SCHEMA",
    "PINNED_CONTAINER_NETWORK_MODE",
    "PINNED_EXECUTABLE_PATH",
    "PINNED_PYTHON_ENVIRONMENT",
    "PINNED_PYTHON_EXECUTABLE",
    "PINNED_PYTHON_MAJOR_MINOR",
    "REQUIRED_CONTAINER_MOUNTS",
    "REQUIRED_CONTAINER_TMPFS",
    "TRACE_OFF_OBSERVATION",
    "TRACE_ON_OBSERVATION",
    "P23ProvenanceError",
    "assert_runtime_module_map_unchanged",
    "build_execution_identity",
    "build_loaded_file_closure",
    "canonical_json_sha256",
    "collect_cuda_runtime",
    "collect_git_provenance",
    "collect_loaded_file_snapshot",
    "collect_python_interpreter_flags",
    "collect_runtime_module_hashes",
    "compare_noninterference_manifests",
    "compare_repeatability_manifests",
    "configure_cuda_determinism",
    "execution_identity_sha256",
    "freeze_runtime_artifacts",
    "load_and_validate_addendum",
    "load_and_validate_host_attestation",
    "load_container_identity",
    "load_runtime_lock",
    "read_live_mountinfo_bytes",
    "sanitize_complete_manifest",
    "sha256_file",
    "validate_cuda_runtime_map",
    "validate_loaded_file_closure",
    "validate_loaded_file_snapshot",
    "validate_python_interpreter_flags",
    "validate_run_manifest",
    "validate_runtime_against_lock",
    "verify_execution_identity",
    "verify_loaded_file_closure",
]
