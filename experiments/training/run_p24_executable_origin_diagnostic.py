#!/usr/bin/env python3
"""Reproduce and attest the PyTorch generated-module executable origin.

This diagnostic is deliberately smaller than the P23 training runner.  It
records the file-backed import closure before and after constructing one
ordinary SGD optimizer, binds the repository, runtime lock, and P24 contract,
and checks either the original writable-origin behavior or the remediated
image-resident behavior.  It never executes a model, data loader, Muon kernel,
or optimizer step.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import ctypes.util
import hashlib
import json
import os
import platform
import re
import secrets
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Final

SCHEMA: Final = "passive-muon-p24-executable-origin-diagnostic-v1"
CONTRACT_SCHEMA: Final = "passive-muon-p24-cuda-executable-origin-contract-v1"
RUNTIME_LOCK_SCHEMA: Final = "passive-muon-p23-cuda-runtime-lock-v3"
HOST_ATTESTATION_SCHEMA: Final = "passive-muon-p23-host-attestation-v3"
GENERATED_MODULE: Final = "_remote_module_non_scriptable"
GENERATED_SHA256: Final = "8205b16956fb264841ecd8644784a0d157f87df79b17c16825dc1163433ce5d8"
FIXED_GENERATED_PATH: Final = (
    "/opt/p23-venv/lib/python3.12/site-packages/torch/"
    "_p24_generated_remote_modules/_remote_module_non_scriptable.py"
)
ORIGINAL_INSTANTIATOR_SHA256: Final = (
    "567d1314ee27ff0b3bd22e7c4d1157246469de25e7a3183d96debe167b193615"
)
PATCHED_INSTANTIATOR_SHA256: Final = (
    "1ecc8ae1ac4a517511914fff6e01c78ed3ad828bbe8a07b02079c206ee56bceb"
)
REMOTE_MODULE_SHA256: Final = "f9bb2f5c5438791581d399e38a27606e123bdbeb3c6cb53683318a06060439c1"
TEMPLATE_SHA256: Final = "0ff1856bbd031b5298d46c06c0502abc20bd804f42c1949ed4127e8c773660cc"
PYTHON_ROOT: Final = Path("/opt/p23-venv/lib/python3.12/site-packages")
PINNED_SOURCES: Final = {
    "instantiator": PYTHON_ROOT / "torch/distributed/nn/jit/instantiator.py",
    "remote_module": PYTHON_ROOT / "torch/distributed/nn/api/remote_module.py",
    "template": PYTHON_ROOT / "torch/distributed/nn/jit/templates/remote_module_template.py",
}
DETERMINISM_ENVIRONMENT: Final = (
    "CUBLAS_WORKSPACE_CONFIG",
    "CUDA_VISIBLE_DEVICES",
    "MKL_NUM_THREADS",
    "NVIDIA_TF32_OVERRIDE",
    "NVIDIA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "PATH",
    "PYTHONHOME",
    "PYTHONOPTIMIZE",
    "PYTHONPATH",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONHASHSEED",
    "PYTHONNOUSERSITE",
    "VIRTUAL_ENV",
    "LD_AUDIT",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
)
_SHA256: Final = re.compile(r"^[0-9a-f]{64}$")
_GIT_OBJECT: Final = re.compile(r"^[0-9a-f]{40}$")
_RUNTIME_LOCK_KEYS: Final = {
    "schema_version",
    "status",
    "p23_addendum_sha256",
    "container",
    "gpu",
    "software",
    "determinism",
    "loader_environment",
}
_HOST_ATTESTATION_KEYS: Final = {
    "schema_version",
    "status",
    "container",
    "gpu",
    "evidence",
}


class DiagnosticError(RuntimeError):
    """Raised when the diagnostic cannot construct a complete record."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(encoded)


def _git(repository: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
    ).stdout


def git_provenance(repository: Path) -> dict[str, object]:
    repository = repository.resolve()
    porcelain = _git(repository, "status", "--porcelain=v1", "-z")
    return {
        "path": str(repository),
        "head": _git(repository, "rev-parse", "HEAD").decode().strip(),
        "tree": _git(repository, "rev-parse", "HEAD^{tree}").decode().strip(),
        "dirty": bool(porcelain),
        "porcelain_v1_z_sha256": sha256_bytes(porcelain),
    }


def _unescape_mount_path(value: str) -> str:
    return re.sub(
        r"\\([0-7]{3})",
        lambda match: chr(int(match.group(1), 8)),
        value,
    )


def parse_mountinfo(text: str) -> list[dict[str, object]]:
    """Parse the fields needed to identify the effective mount of a path."""

    records: list[dict[str, object]] = []
    for line in text.splitlines():
        before, separator, after = line.partition(" - ")
        if not separator:
            raise DiagnosticError("malformed /proc/self/mountinfo line")
        left = before.split()
        right = after.split()
        if len(left) < 6 or len(right) < 3:
            raise DiagnosticError("incomplete /proc/self/mountinfo line")
        mount_options = left[5].split(",")
        super_options = right[2].split(",")
        records.append(
            {
                "mount_id": int(left[0]),
                "mount_point": _unescape_mount_path(left[4]),
                "filesystem_type": right[0],
                "mount_options": mount_options,
                "super_options": super_options,
                "writable": "rw" in mount_options,
            }
        )
    return records


def effective_mount(path: Path, mounts: list[dict[str, object]]) -> dict[str, object]:
    resolved = str(path.resolve())
    candidates = []
    for record in mounts:
        mount_point = str(record["mount_point"])
        if resolved == mount_point or resolved.startswith(mount_point.rstrip("/") + "/"):
            candidates.append(record)
    if not candidates:
        raise DiagnosticError(f"no effective mount for {resolved}")
    return dict(
        max(
            candidates,
            key=lambda record: (
                len(Path(str(record["mount_point"])).parts),
                int(record["mount_id"]),
            ),
        )
    )


def _module_file(module: object) -> Path | None:
    raw = getattr(module, "__file__", None)
    if not isinstance(raw, str):
        return None
    path = Path(raw)
    if not path.is_absolute() or not path.is_file():
        return None
    return path.resolve()


def module_snapshot(
    modules: Mapping[str, object], mounts: list[dict[str, object]]
) -> dict[str, dict[str, object]]:
    """Hash every regular, absolute file backing a currently loaded module."""

    records: dict[str, dict[str, object]] = {}
    for name in sorted(modules):
        path = _module_file(modules[name])
        if path is None:
            continue
        stat_before = path.stat()
        digest = sha256_file(path)
        stat_after = path.stat()
        if (stat_before.st_size, stat_before.st_mtime_ns) != (
            stat_after.st_size,
            stat_after.st_mtime_ns,
        ):
            raise DiagnosticError(f"module file changed while hashing: {path}")
        records[name] = {
            "path": str(path),
            "byte_count": stat_after.st_size,
            "sha256": digest,
            "effective_mount": effective_mount(path, mounts),
        }
    return records


def _snapshot_delta(earlier: Mapping[str, object], later: Mapping[str, object]) -> list[str]:
    return sorted(set(later) - set(earlier))


def _stable_file_binding(path: Path, label: str) -> tuple[dict[str, object], bytes]:
    """Read and bind a regular file while rejecting concurrent byte changes."""

    resolved = path.resolve(strict=True)
    stat_before = resolved.stat()
    if not resolved.is_file():
        raise DiagnosticError(f"{label} must be a regular file: {resolved}")
    payload = resolved.read_bytes()
    stat_after = resolved.stat()
    before = (
        stat_before.st_dev,
        stat_before.st_ino,
        stat_before.st_size,
        stat_before.st_mtime_ns,
    )
    after = (
        stat_after.st_dev,
        stat_after.st_ino,
        stat_after.st_size,
        stat_after.st_mtime_ns,
    )
    if before != after or len(payload) != stat_after.st_size:
        raise DiagnosticError(f"{label} changed while hashing: {resolved}")
    return (
        {
            "path": str(resolved),
            "byte_count": len(payload),
            "sha256": sha256_bytes(payload),
        },
        payload,
    )


def _read_mapping_bytes(payload: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DiagnosticError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise DiagnosticError(f"{label} must be a JSON object")
    return value


def _require_hex(value: str, pattern: re.Pattern[str], label: str) -> None:
    if pattern.fullmatch(value) is None:
        raise DiagnosticError(f"{label} must be lowercase hexadecimal")


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise DiagnosticError(f"{label} must be a mapping")
    return value


def _contract_source_records(
    repository: Path, contract: Mapping[str, object]
) -> dict[str, dict[str, object]]:
    """Rehash every repository source that the P24 contract freezes."""

    groups: list[tuple[str, Mapping[str, object]]] = [
        (
            "unchanged_authorities",
            _mapping(contract.get("unchanged_authorities"), "unchanged_authorities"),
        ),
        (
            "execution_sources",
            _mapping(contract.get("execution_sources"), "execution_sources"),
        ),
    ]
    remediation = _mapping(contract.get("remediation_design"), "remediation_design")
    groups.append(
        (
            "remediation_sources",
            {
                "build_patch": remediation.get("build_patch"),
                "image_recipe": remediation.get("image_recipe"),
            },
        )
    )
    records: dict[str, dict[str, object]] = {}
    repository = repository.resolve(strict=True)
    for group, raw_entries in groups:
        for name, raw_entry in raw_entries.items():
            entry = _mapping(raw_entry, f"{group}.{name}")
            relative = entry.get("path")
            expected_sha256 = entry.get("sha256")
            if not isinstance(relative, str) or Path(relative).is_absolute():
                raise DiagnosticError(f"{group}.{name}.path must be repository-relative")
            if not isinstance(expected_sha256, str):
                raise DiagnosticError(f"{group}.{name}.sha256 must be a string")
            _require_hex(expected_sha256, _SHA256, f"{group}.{name}.sha256")
            path = (repository / relative).resolve(strict=True)
            if not path.is_relative_to(repository):
                raise DiagnosticError(f"{group}.{name}.path escapes the repository")
            binding, _ = _stable_file_binding(path, f"{group}.{name}")
            if binding["sha256"] != expected_sha256:
                raise DiagnosticError(f"{group}.{name} differs from the P24 contract")
            tracked = _git(repository, "ls-files", "--error-unmatch", "--", relative)
            if tracked.decode().strip() != relative:
                raise DiagnosticError(f"{group}.{name} is not the exact tracked path")
            records[f"{group}.{name}"] = {
                "logical_path": relative,
                **binding,
            }
    return records


def _expected_process_environment(runtime_lock: Mapping[str, object]) -> dict[str, str | None]:
    determinism = _mapping(runtime_lock.get("determinism"), "runtime_lock.determinism")
    loader = _mapping(runtime_lock.get("loader_environment"), "runtime_lock.loader_environment")
    expected: dict[str, str | None] = {
        name: value if isinstance(value, str) else None for name, value in loader.items()
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
    if set(expected) != set(DETERMINISM_ENVIRONMENT):
        raise DiagnosticError("runtime lock cannot reconstruct the complete process environment")
    return expected


def validate_preflight(
    *,
    mode: str,
    repository: Path,
    runtime_lock_path: Path,
    host_attestation_path: Path,
    contract_path: Path,
    expected_contract_sha256: str,
    expected_runtime_lock_sha256: str,
    expected_host_attestation_sha256: str,
    expected_repository_head: str,
    expected_repository_tree: str,
    mountinfo: bytes,
    mounts: list[dict[str, object]],
) -> dict[str, object]:
    """Validate all static and live bindings before importing PyTorch."""

    for value, pattern, label in (
        (expected_contract_sha256, _SHA256, "expected contract SHA-256"),
        (expected_runtime_lock_sha256, _SHA256, "expected runtime-lock SHA-256"),
        (
            expected_host_attestation_sha256,
            _SHA256,
            "expected host-attestation SHA-256",
        ),
        (expected_repository_head, _GIT_OBJECT, "expected repository HEAD"),
        (expected_repository_tree, _GIT_OBJECT, "expected repository tree"),
    ):
        _require_hex(value, pattern, label)

    contract_binding, contract_bytes = _stable_file_binding(contract_path, "P24 contract")
    runtime_binding, runtime_bytes = _stable_file_binding(runtime_lock_path, "runtime lock")
    attestation_binding, attestation_bytes = _stable_file_binding(
        host_attestation_path, "host attestation"
    )
    contract = _read_mapping_bytes(contract_bytes, "P24 contract")
    runtime_lock = _read_mapping_bytes(runtime_bytes, "runtime lock")
    host_attestation = _read_mapping_bytes(attestation_bytes, "host attestation")
    repository_record = git_provenance(repository)

    if set(runtime_lock) != _RUNTIME_LOCK_KEYS:
        raise DiagnosticError("runtime lock top-level schema differs")
    if set(host_attestation) != _HOST_ATTESTATION_KEYS:
        raise DiagnosticError("host attestation top-level schema differs")
    container = _mapping(runtime_lock.get("container"), "runtime_lock.container")
    gpu = _mapping(runtime_lock.get("gpu"), "runtime_lock.gpu")
    software = _mapping(runtime_lock.get("software"), "runtime_lock.software")
    attested_container = _mapping(host_attestation.get("container"), "host_attestation.container")
    attested_gpu = _mapping(host_attestation.get("gpu"), "host_attestation.gpu")
    container_without_backlink = {
        key: value for key, value in container.items() if key != "host_attestation_sha256"
    }
    expected_environment = _expected_process_environment(runtime_lock)
    observed_environment = {name: os.environ.get(name) for name in DETERMINISM_ENVIRONMENT}
    root_mount = effective_mount(Path("/"), mounts)
    source_records = _contract_source_records(repository, contract)
    authorities = _mapping(contract.get("unchanged_authorities"), "unchanged_authorities")
    p23_addendum = _mapping(authorities.get("p23_addendum"), "p23_addendum authority")

    checks = {
        "mode_declared": mode in {"baseline", "remediated"},
        "contract_schema_and_status_exact": contract.get("schema_version") == CONTRACT_SCHEMA
        and contract.get("status") == "frozen_pre_remediation_build_and_pre_acquisition",
        "contract_sha256_matches_expected": contract_binding["sha256"] == expected_contract_sha256,
        "runtime_lock_schema_and_status_exact": runtime_lock.get("schema_version")
        == RUNTIME_LOCK_SCHEMA
        and runtime_lock.get("status") == "pinned_for_acquisition",
        "runtime_lock_sha256_matches_expected": runtime_binding["sha256"]
        == expected_runtime_lock_sha256,
        "runtime_lock_p23_addendum_matches_contract_and_source": runtime_lock.get(
            "p23_addendum_sha256"
        )
        == p23_addendum.get("sha256")
        == source_records["unchanged_authorities.p23_addendum"]["sha256"],
        "host_attestation_schema_and_status_exact": host_attestation.get("schema_version")
        == HOST_ATTESTATION_SCHEMA
        and host_attestation.get("status") == "procedurally_host_attested",
        "host_attestation_sha256_matches_expected": attestation_binding["sha256"]
        == expected_host_attestation_sha256,
        "runtime_lock_host_attestation_backlink_exact": container.get("host_attestation_sha256")
        == attestation_binding["sha256"],
        "runtime_lock_and_attestation_container_exact": container_without_backlink
        == dict(attested_container),
        "runtime_lock_and_attestation_gpu_exact": dict(gpu) == dict(attested_gpu),
        "repository_clean": repository_record["dirty"] is False,
        "repository_head_matches_expected": repository_record["head"] == expected_repository_head,
        "repository_tree_matches_expected": repository_record["tree"] == expected_repository_tree,
        "contract_source_records_exact": bool(source_records),
        "container_hostname_matches_lock": socket.gethostname() == container.get("hostname"),
        "rootfs_declared_and_observed_read_only": container.get("rootfs_read_only") is True
        and root_mount.get("writable") is False,
        "live_mountinfo_matches_lock": sha256_bytes(mountinfo) == container.get("mountinfo_sha256"),
        "process_environment_matches_lock_exactly": observed_environment == expected_environment,
        "python_runtime_matches_lock": sys.version == software.get("python")
        and sys.executable == software.get("python_executable"),
    }
    failed = sorted(name for name, passes in checks.items() if not passes)
    if failed:
        raise DiagnosticError(f"P24 preflight checks failed: {failed}")
    return {
        "contract": contract,
        "runtime_lock": runtime_lock,
        "host_attestation": host_attestation,
        "repository": repository_record,
        "contract_source_records": source_records,
        "expected_environment": expected_environment,
        "observed_environment": observed_environment,
        "root_mount": root_mount,
        "artifacts": {
            "diagnostic_source": _stable_file_binding(Path(__file__), "diagnostic source")[0],
            "contract": contract_binding,
            "runtime_lock": runtime_binding,
            "host_attestation": attestation_binding,
        },
        "checks": checks,
    }


def _source_records() -> dict[str, dict[str, object]]:
    return {
        name: {
            "path": str(path),
            "byte_count": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for name, path in PINNED_SOURCES.items()
    }


def _atomic_write_json(path: Path, payload: object) -> None:
    """Publish complete JSON without ever replacing an existing directory entry."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as error:
            raise DiagnosticError("diagnostic output already exists") from error
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory = os.open(path.parent, directory_flags)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        raise
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)


def _file_binding(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    result: dict[str, object] = {
        "path": str(resolved),
        "exists": resolved.exists(),
        "is_file": resolved.is_file(),
    }
    if resolved.is_file():
        payload = resolved.read_bytes()
        result.update({"byte_count": len(payload), "sha256": sha256_bytes(payload)})
    return result


def error_payload(
    *,
    mode: str,
    repository: Path,
    runtime_lock_path: Path,
    host_attestation_path: Path,
    contract_path: Path,
    expected_contract_sha256: str,
    expected_runtime_lock_sha256: str,
    expected_host_attestation_sha256: str,
    expected_repository_head: str,
    expected_repository_tree: str,
    error: BaseException,
) -> dict[str, object]:
    """Retain best-effort rule-7 provenance when the diagnostic itself fails."""

    try:
        repository_record: object = git_provenance(repository)
    except (OSError, subprocess.SubprocessError) as provenance_error:
        repository_record = {
            "path": str(repository.resolve()),
            "status": "unavailable",
            "error_class": type(provenance_error).__name__,
            "error_message": str(provenance_error),
        }
    try:
        mountinfo = Path("/proc/self/mountinfo").read_bytes()
        mountinfo_sha256: str | None = sha256_bytes(mountinfo)
    except OSError:
        mountinfo_sha256 = None
    return {
        "schema_version": SCHEMA,
        "mode": mode,
        "status": "error",
        "scope": "minimal optimizer-construction executable-origin diagnostic",
        "seed": None,
        "seed_semantics": "not applicable: the diagnostic performs no random operation",
        "process": {
            "pid": os.getpid(),
            "nonce": secrets.token_hex(32),
            "failed_time_ns": time.time_ns(),
            "argv": list(sys.argv),
        },
        "repository": {
            "expected_head": expected_repository_head,
            "expected_tree": expected_repository_tree,
            "observed": repository_record,
        },
        "artifacts": {
            "diagnostic_source": _file_binding(Path(__file__)),
            "contract": _file_binding(contract_path),
            "runtime_lock": _file_binding(runtime_lock_path),
            "host_attestation": _file_binding(host_attestation_path),
        },
        "contract_binding": {
            "expected_sha256": expected_contract_sha256,
            "content": None,
            "source_records": {},
        },
        "runtime_lock_binding": {
            "expected_sha256": expected_runtime_lock_sha256,
            "content": None,
        },
        "host_attestation_binding": {
            "expected_sha256": expected_host_attestation_sha256,
            "content": None,
        },
        "live_runtime": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "python_executable": sys.executable,
            "environment": {name: os.environ.get(name) for name in DETERMINISM_ENVIRONMENT},
            "mountinfo_sha256": mountinfo_sha256,
        },
        "pinned_sources": {},
        "trigger": None,
        "file_backed_module_closure": None,
        "checks": {"diagnostic_completed": False},
        "error": {
            "class": f"{type(error).__module__}.{type(error).__qualname__}",
            "message": str(error),
            "traceback": traceback.format_exc(),
        },
        "claim_boundary": (
            "This native artifact records a failed minimal diagnostic and its available "
            "provenance. It is not a P23/P24 acquisition, a training result, or fidelity evidence."
        ),
        "passes": False,
    }


def cuda_runtime_version() -> int:
    """Read the loaded CUDA runtime version through its public C API."""

    candidates: list[str | None] = [
        ctypes.util.find_library("cudart"),
        None,
        "libcudart.so",
        "libcudart.so.12",
        "libcudart.so.11.0",
    ]
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
        except (AttributeError, OSError):
            continue
        if status == 0 and version.value > 0:
            return int(version.value)
    raise DiagnosticError("cudaRuntimeGetVersion is unavailable")


def run_diagnostic(
    *,
    mode: str,
    repository: Path,
    runtime_lock_path: Path,
    host_attestation_path: Path,
    contract_path: Path,
    expected_contract_sha256: str,
    expected_runtime_lock_sha256: str,
    expected_host_attestation_sha256: str,
    expected_repository_head: str,
    expected_repository_tree: str,
) -> dict[str, object]:
    """Run the minimal trigger and return a complete native evidence payload."""

    started_ns = time.time_ns()
    process_nonce = secrets.token_hex(32)
    mountinfo = Path("/proc/self/mountinfo").read_bytes()
    mounts = parse_mountinfo(mountinfo.decode())
    preflight = validate_preflight(
        mode=mode,
        repository=repository,
        runtime_lock_path=runtime_lock_path,
        host_attestation_path=host_attestation_path,
        contract_path=contract_path,
        expected_contract_sha256=expected_contract_sha256,
        expected_runtime_lock_sha256=expected_runtime_lock_sha256,
        expected_host_attestation_sha256=expected_host_attestation_sha256,
        expected_repository_head=expected_repository_head,
        expected_repository_tree=expected_repository_tree,
        mountinfo=mountinfo,
        mounts=mounts,
    )
    runtime_lock = _mapping(preflight["runtime_lock"], "validated runtime lock")
    gpu = _mapping(runtime_lock.get("gpu"), "runtime_lock.gpu")
    software = _mapping(runtime_lock.get("software"), "runtime_lock.software")
    determinism = _mapping(runtime_lock.get("determinism"), "runtime_lock.determinism")
    before_torch = module_snapshot(sys.modules, mounts)

    import torch

    after_torch = module_snapshot(sys.modules, mounts)
    parameter = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32, device="cpu"))
    after_parameter = module_snapshot(sys.modules, mounts)
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    if len(optimizer.param_groups) != 1:
        raise DiagnosticError("minimal optimizer has unexpected parameter groups")
    after_optimizer = module_snapshot(sys.modules, mounts)

    generated = after_optimizer.get(GENERATED_MODULE)
    sources = _source_records()
    expected_instantiator = (
        ORIGINAL_INSTANTIATOR_SHA256 if mode == "baseline" else PATCHED_INSTANTIATOR_SHA256
    )
    expected_generated_path = (
        None if mode == "baseline" else str(Path(FIXED_GENERATED_PATH).resolve())
    )
    generated_path = None if generated is None else generated.get("path")
    generated_mount = None if generated is None else generated.get("effective_mount")
    generated_writable = (
        None if not isinstance(generated_mount, Mapping) else generated_mount.get("writable")
    )
    cuda_properties = torch.cuda.get_device_properties(0)
    live_gpu_uuid = f"GPU-{cuda_properties.uuid}"
    live_mountinfo_sha256 = sha256_bytes(mountinfo)
    live_runtime = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "torch_cuda_compiled_version": torch.version.cuda,
        "cuda_runtime_version": cuda_runtime_version(),
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": cuda_properties.name,
        "cuda_device_uuid": live_gpu_uuid,
        "environment": {name: os.environ.get(name) for name in DETERMINISM_ENVIRONMENT},
        "mountinfo_sha256": live_mountinfo_sha256,
        "root_effective_mount": effective_mount(Path("/"), mounts),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_debug_mode": torch.get_deterministic_debug_mode(),
        "cpu_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "tf32": torch.backends.cuda.matmul.allow_tf32,
        "fp16_reduced_precision_reduction": (
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        ),
        "bf16_reduced_precision_reduction": (
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        ),
    }
    final_modules = module_snapshot(sys.modules, mounts)
    tmp_generated = sorted(
        name
        for name, record in after_optimizer.items()
        if name.startswith("_remote_module_") and str(record.get("path", "")).startswith("/tmp/")
    )
    repository_record = git_provenance(repository)
    final_contract_sources = _contract_source_records(
        repository, _mapping(preflight["contract"], "validated contract")
    )
    final_artifacts = {
        "diagnostic_source": _stable_file_binding(Path(__file__), "diagnostic source")[0],
        "contract": _stable_file_binding(contract_path, "P24 contract")[0],
        "runtime_lock": _stable_file_binding(runtime_lock_path, "runtime lock")[0],
        "host_attestation": _stable_file_binding(host_attestation_path, "host attestation")[0],
    }
    checks = {
        **_mapping(preflight["checks"], "preflight checks"),
        "repository_unchanged_after_trigger": repository_record == preflight["repository"],
        "contract_sources_unchanged_after_trigger": final_contract_sources
        == preflight["contract_source_records"],
        "bound_artifacts_unchanged_after_trigger": final_artifacts == preflight["artifacts"],
        "torch_version_matches_lock": torch.__version__ == software.get("torch_version"),
        "torch_git_version_matches_lock": torch.version.git_version
        == software.get("torch_git_version"),
        "compiled_cuda_matches_lock": torch.version.cuda
        == software.get("torch_cuda_compiled_version"),
        "cuda_runtime_matches_lock": live_runtime["cuda_runtime_version"]
        == software.get("cuda_runtime_version"),
        "cudnn_matches_lock": live_runtime["cudnn_version"] == software.get("cudnn_version"),
        "single_live_gpu_matches_lock": torch.cuda.device_count() == 1
        and live_gpu_uuid == gpu.get("uuid")
        and cuda_properties.name == gpu.get("name"),
        "torch_determinism_matches_lock": (
            live_runtime["deterministic_algorithms"] == determinism.get("deterministic_algorithms")
            and live_runtime["deterministic_debug_mode"] == 2
            and determinism.get("deterministic_debug_mode") == "error"
            and live_runtime["cpu_threads"] == determinism.get("cpu_threads")
            and live_runtime["interop_threads"] == determinism.get("interop_threads")
            and live_runtime["cudnn_benchmark"] == determinism.get("cudnn_benchmark")
            and live_runtime["cudnn_deterministic"] == determinism.get("cudnn_deterministic")
            and live_runtime["float32_matmul_precision"]
            == determinism.get("float32_matmul_precision")
            and live_runtime["tf32"] == determinism.get("tf32")
            and live_runtime["fp16_reduced_precision_reduction"]
            == determinism.get("fp16_reduced_precision_reduction")
            and live_runtime["bf16_reduced_precision_reduction"]
            == determinism.get("bf16_reduced_precision_reduction")
        ),
        "remote_module_source_exact": sources["remote_module"]["sha256"] == REMOTE_MODULE_SHA256,
        "template_source_exact": sources["template"]["sha256"] == TEMPLATE_SHA256,
        "instantiator_source_exact_for_mode": sources["instantiator"]["sha256"]
        == expected_instantiator,
        "generated_module_loaded_only_after_optimizer": GENERATED_MODULE not in after_parameter
        and GENERATED_MODULE in after_optimizer,
        "generated_module_bytes_exact": isinstance(generated, Mapping)
        and generated.get("sha256") == GENERATED_SHA256,
        "mode_origin_matches": (
            mode == "baseline"
            and isinstance(generated_path, str)
            and generated_path.startswith("/tmp/")
            and generated_writable is True
        )
        or (
            mode == "remediated"
            and generated_path == expected_generated_path
            and generated_writable is False
            and not tmp_generated
        ),
    }
    completed_ns = time.time_ns()
    return {
        "schema_version": SCHEMA,
        "mode": mode,
        "status": "passes" if all(checks.values()) else "fails",
        "scope": (
            "minimal optimizer-construction executable-origin diagnostic; no model, data, "
            "forward/backward pass, Muon candidate, optimizer step, or fidelity observation"
        ),
        "seed": None,
        "seed_semantics": "not applicable: the diagnostic performs no random operation",
        "process": {
            "pid": os.getpid(),
            "nonce": process_nonce,
            "started_time_ns": started_ns,
            "completed_time_ns": completed_ns,
            "argv": list(sys.argv),
        },
        "repository": {
            "expected_head": expected_repository_head,
            "expected_tree": expected_repository_tree,
            "observed": repository_record,
        },
        "artifacts": {
            **final_artifacts,
        },
        "contract_binding": {
            "expected_sha256": expected_contract_sha256,
            "content": preflight["contract"],
            "source_records": final_contract_sources,
        },
        "runtime_lock_binding": {
            "expected_sha256": expected_runtime_lock_sha256,
            "content": preflight["runtime_lock"],
        },
        "host_attestation_binding": {
            "expected_sha256": expected_host_attestation_sha256,
            "content": preflight["host_attestation"],
        },
        "live_runtime": live_runtime,
        "pinned_sources": sources,
        "trigger": {
            "operation": (
                "import torch; create one zero CPU float32 Parameter; construct "
                "torch.optim.SGD([parameter], lr=1.0); perform no optimizer step"
            ),
            "generated_module": generated,
            "temporary_generated_module_names": tmp_generated,
            "new_modules_after_torch_import": _snapshot_delta(before_torch, after_torch),
            "new_modules_after_parameter": _snapshot_delta(after_torch, after_parameter),
            "new_modules_after_optimizer": _snapshot_delta(after_parameter, after_optimizer),
        },
        "file_backed_module_closure": {
            "count": len(final_modules),
            "canonical_sha256": canonical_sha256(final_modules),
            "records": final_modules,
        },
        "claim_boundary": (
            "This native artifact covers only the minimal executable-origin diagnostic. It is "
            "not a P23/P24 gradient acquisition, fidelity result, or training result."
        ),
        "checks": checks,
        "passes": all(checks.values()),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "remediated"), required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--runtime-lock", type=Path, required=True)
    parser.add_argument("--host-attestation", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--expected-contract-sha256", required=True)
    parser.add_argument("--expected-runtime-lock-sha256", required=True)
    parser.add_argument("--expected-host-attestation-sha256", required=True)
    parser.add_argument("--expected-repository-head", required=True)
    parser.add_argument("--expected-repository-tree", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    output = arguments.output.resolve()
    repository = arguments.repository.resolve()
    if output.is_relative_to(repository):
        print("P24 diagnostic blocked: output must remain outside the repository", file=sys.stderr)
        return 2
    if output.exists():
        print("P24 diagnostic blocked: output already exists", file=sys.stderr)
        return 2
    try:
        payload = run_diagnostic(
            mode=arguments.mode,
            repository=repository,
            runtime_lock_path=arguments.runtime_lock.resolve(),
            host_attestation_path=arguments.host_attestation.resolve(),
            contract_path=arguments.contract.resolve(),
            expected_contract_sha256=arguments.expected_contract_sha256,
            expected_runtime_lock_sha256=arguments.expected_runtime_lock_sha256,
            expected_host_attestation_sha256=arguments.expected_host_attestation_sha256,
            expected_repository_head=arguments.expected_repository_head,
            expected_repository_tree=arguments.expected_repository_tree,
        )
    except BaseException as error:
        payload = error_payload(
            mode=arguments.mode,
            repository=repository,
            runtime_lock_path=arguments.runtime_lock.resolve(),
            host_attestation_path=arguments.host_attestation.resolve(),
            contract_path=arguments.contract.resolve(),
            expected_contract_sha256=arguments.expected_contract_sha256,
            expected_runtime_lock_sha256=arguments.expected_runtime_lock_sha256,
            expected_host_attestation_sha256=arguments.expected_host_attestation_sha256,
            expected_repository_head=arguments.expected_repository_head,
            expected_repository_tree=arguments.expected_repository_tree,
            error=error,
        )
    try:
        _atomic_write_json(output, payload)
    except BaseException as error:
        print(f"P24 diagnostic could not retain evidence: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": str(output),
                "output_sha256": sha256_file(output),
                "passes": payload["passes"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
