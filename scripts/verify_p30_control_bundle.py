#!/usr/bin/env python3
"""Fail-closed verification for P30's two offline control-bundle transfers.

The source phase runs before an acquisition attempt root may exist; the runtime
phase runs after preparation but before localization. The verifier copies the
transport bundle with one stable ``O_NOFOLLOW`` read, verifies that private copy
in a newly initialized empty bare repository, explicitly fetches the closed ref
inventory, and only then creates or advances the detached control checkout. A
deterministic external receipt can subsequently be authenticated and replayed
without overwriting it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import stat
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = "passive-muon-p30-control-bundle-receipt-v1"
STATUS_BY_PHASE = {
    "source": "source_bundle_and_permission_layout_verified_before_attempt_root_creation",
    "runtime-review": "runtime_review_bundle_and_permission_layout_verified_before_localization",
}
BRANCH_REF = "refs/heads/p30-umask-bound-control-seal"
P29_BRANCH_REF = "refs/heads/p29-permission-safe-bundle-localization-bridge"
P29_TAG_REF = "refs/tags/p29-orchestrator-source-mode-diagnostic"
P28_BRANCH_REF = "refs/heads/p28-bundle-complete-localization-bridge"
P28_TAG_REF = "refs/tags/p28-control-parent-permission-diagnostic"
P27_BRANCH_REF = "refs/heads/p27-cuda-deleted-mapping-localization"
P27_TAG_REF = "refs/tags/p27-cuda-deleted-mapping-localization-diagnostic"
P26_CHECKPOINT_REF = "refs/tags/p26-permission-safe-acquisition-checkpoint"
P26_SOURCE_REF = "refs/tags/p26-attempt02-acquisition-source"

P29_COMMIT = "8a263722c7f8ce1d99b8796a3ccc0425dd1cefcf"
P29_TREE = "6ee56f8c218eab1d25047f6f1046b97f7529d996"
P29_TAG_OBJECT = "cf3c60b1d5c004b9e601a6afbf630358f80f1d79"
P28_COMMIT = "7274367f2b05fb3c9ed9f876a59be647703bbdc2"
P28_TREE = "8d971e176e664ea47d8f769ae54f8cf349ceb3d5"
P28_TAG_OBJECT = "f3c81a2f14f853f369015a4f81b6695b52eec74e"
P27_COMMIT = "ec63550331925ded158e3f389e294e4d1f12db3a"
P27_TREE = "d8efa72fda9ee41fde0b5d15b126aa87397cd48d"
P27_TAG_OBJECT = "c88b98eae9be2458abde45b05d3dccfe09c0c7ed"
P26_CHECKPOINT_OBJECT = "bacad707d3779bfa10957e18cb4c69b1a7f0cbce"
P26_CHECKPOINT_COMMIT = "5429da23ff18888daa2312c530a4587780484d8b"
P26_SOURCE_OBJECT = "f35a7dca8f6bc39e9748e79b6712fab4a203396b"
P26_SOURCE_COMMIT = "185e444afc0b44ca0a09b1bde49a6b6fa3973355"
P26_SOURCE_TREE = "24f4bdac331a57bd7c1b807747d7c7fba253ee5a"
NANOGPT_COMMIT = "3adf61e154c3fe3fca428ad6bc3818b27a3b8291"
NANOGPT_TREE = "ca93bcd9b9c9ff32d3016e1e2556644e68bef86a"
MUON_COMMIT = "f98f1cacc0263b04290753e32be8d498c1efc806"
MUON_TREE = "4ea5cd8ab6ebd56a18536f06453619efcd636da0"
P28_OUTCOME_RECONSTRUCTOR = "scripts/reconstruct_p28_bundle_complete_localization_bridge_outcome.py"
P28_CONTRACT_RECONSTRUCTOR = "scripts/reconstruct_p28_bundle_complete_localization_bridge.py"
P27_RECONSTRUCTOR = "scripts/reconstruct_p27_cuda_deleted_mapping_localization.py"
P30_RECONSTRUCTOR = "scripts/reconstruct_p30_umask_bound_control_seal.py"
P30_CONTRACT = "experiments/training/p30_umask_bound_control_seal_contract.json"
P30_RECONSTRUCTION_SCHEMA = "passive-muon-p30-umask-bound-control-seal-reconstruction-v1"
VERIFIER_SOURCE = "scripts/verify_p30_control_bundle.py"
ORCHESTRATOR_SOURCE = "scripts/run_p30_umask_bound_control_seal.sh"
VERIFIER_UMASK = 0o077
ORCHESTRATOR_GIT_MODE = "100644"
ORCHESTRATOR_PHYSICAL_MODE = 0o600
TERMINAL_P29_CONTRACT_RECONSTRUCTOR = (
    "scripts/reconstruct_p29_permission_safe_bundle_localization_bridge.py"
)
TERMINAL_P29_OUTCOME_RECONSTRUCTOR = (
    "scripts/reconstruct_p29_permission_safe_bundle_localization_bridge_outcome.py"
)
TERMINAL_P29_CONTRACT_RECONSTRUCTOR_SHA256 = (
    "a384bfcbc25fe9148a158a5f524170557808f8e68d07cb99474b64bbe3c10aea"
)
TERMINAL_P29_OUTCOME_RECONSTRUCTOR_SHA256 = (
    "b850a9423b42aa2026db8505b7fdd3767d7954e94283cc96ebdb87cba289b33f"
)
TERMINAL_P29_CONTRACT_RECONSTRUCTION_SCHEMA = (
    "passive-muon-p29-permission-safe-bundle-localization-bridge-reconstruction-v1"
)
TERMINAL_P29_OUTCOME_RECONSTRUCTION_SCHEMA = (
    "passive-muon-p29-permission-safe-bundle-localization-bridge-outcome-reconstruction-v1"
)
TERMINAL_P29_CONTRACT_CANONICAL_SHA256 = (
    "54d10ab01b7d453d01931459b314fc4491b6e4317a02fb763ca9872686311b0a"
)
TERMINAL_P29_OUTCOME_CANONICAL_SHA256 = (
    "6d29601a71ce6e0c99bcc35497d328c8fdd2a3cfc135351c19876507b00a0f64"
)
P28_OUTCOME_RECONSTRUCTOR_SHA256 = (
    "d736909181eaa75298851b7fc65f9c0352fb32c3d07e40eb4118eb10c1898631"
)
P28_OUTCOME_RECONSTRUCTION_SCHEMA = (
    "passive-muon-p28-bundle-complete-localization-bridge-outcome-reconstruction-v1"
)
P28_OUTCOME_CANONICAL_SHA256 = "298324d163aa12c1cb64bfdb912ae846152d0980ae3c981a54ead75a019305c9"
P28_CONTRACT_RECONSTRUCTOR_SHA256 = (
    "c8b5c853a5635f277f7d1d4c0185a199c6658c575a35c088c7cca51d36563b62"
)
P28_CONTRACT_RECONSTRUCTION_SCHEMA = (
    "passive-muon-p28-bundle-complete-localization-bridge-reconstruction-v1"
)
P28_CONTRACT_CANONICAL_SHA256 = "6a5005f902d937f0edbec3394e6205e482fe6f6fea45f986ac873fd6ab2e8718"
P27_RECONSTRUCTOR_SHA256 = "0eedffa5442ce7fc3950ae74d169fa652c70794935c541028eba3d0bda20e033"
P27_RECONSTRUCTION_SCHEMA = "passive-muon-p27-cuda-deleted-mapping-localization-reconstruction-v1"
P27_CONTRACT_SHA256 = "02000debea28f499ddeb3b8c6c8b0615786cd2e7858cc8a1605b04d638da4b7a"

NAMESPACE_ROOT = Path("/secure/p30")
SECURE_PARENT = Path("/secure")
TRANSPORT_BASENAME = "transport-20260906-03"
CONTROL_BASENAME = "control-source"
ATTEMPT_BASENAME = "attempt-20260906-03"
EXECUTION_BASENAME = "execution-20260906-03"
BOOTSTRAP_BASENAME = "verify_p30_control_bundle.py"
AUTHORITY_BASENAME = "authority-185e444"
NAMESPACE_UID = 0
NAMESPACE_GID = 0
NAMESPACE_MODE = 0o755
SECURE_PARENT_UID = 0
SECURE_PARENT_GID = 0
SECURE_PARENT_MODE = 0o755
TRANSPORT_UID = 1000
TRANSPORT_GID = 1000
TRANSPORT_MODE = 0o700
VERIFIER_UID = 1000
VERIFIER_GID = 1000
ATTEMPT_UID = 0
ATTEMPT_GID = 0
ATTEMPT_MODE = 0o555
EVIDENCE_UID = 0
EVIDENCE_GID = 0
EVIDENCE_MODE = 0o555
EXECUTION_UID = 0
EXECUTION_GID = 0
EXECUTION_MODE = 0o555
REQUIRE_EXECUTING_PYTHON_ISOLATION = True
MOUNTINFO_PATH = Path("/proc/self/mountinfo")
GIT_BINARY = Path("/usr/bin/git")
ACL_XATTR_NAMES = frozenset(
    {
        "system.posix_acl_access",
        "system.posix_acl_default",
        "system.nfs4_acl",
        "system.richacl",
    }
)

PHASE_LAYOUT = {
    "source": {
        "bundle": "p30_source.bundle",
        "closure": "p30_source_closure.git",
        "receipt": "p30_source_bundle_receipt.json",
    },
    "runtime-review": {
        "bundle": "p30_runtime_review.bundle",
        "closure": "p30_runtime_review_closure.git",
        "receipt": "p30_runtime_review_bundle_receipt.json",
    },
}
RUNTIME_DELTA = [
    "A\texperiments/training/p30_cuda_runtime_lock.json",
    "A\texperiments/training/p30_host_attestation.json",
]
RUNTIME_EVIDENCE_FILES = {
    "runtime_lock": "p30_cuda_runtime_lock.json",
    "host_attestation": "p30_host_attestation.json",
}
RUNTIME_CONTROL_GIT_MODE = "100644"
RUNTIME_CONTROL_PHYSICAL_MODE = 0o600
RUNTIME_FORBIDDEN_LOCALIZATION_FILES = [
    "p30-run-localization.invoked",
    "p30_cuda_deleted_mapping_localization.native.json",
    "p30_cuda_deleted_mapping_localization.sanitized.json",
    "p30-localization.exit-status.txt",
    "p30-localization-sanitizer.exit-status.txt",
    "p30-p26-failure-ingestion.exit-status.txt",
    "run_p27_cuda_deleted_mapping_localization.py",
    "ingest_p26_trace_off_a_failure.py",
    "p23_deterministic_cuda_shadow_trace.py",
    "sanitize_p27_cuda_deleted_mapping_localization.py",
]
RUNTIME_FORBIDDEN_PREFIXES = [
    "p30-run-contract-reconstruction.",
    "p30-pre-marker-",
    "p30-post-marker-",
    "p30-stage-",
    "p30-pre-ingestion-",
    "p30-p26-failure-ingestion.",
    "p30-pre-localization-",
    "p30-pre-sanitizer-",
    "p30-post-localization-",
    "p30-localization.",
    "p30-localization-sanitizer.",
]
RUNTIME_ATTEMPT_TOP_LEVEL = [
    "evidence",
    "p30-image-inspect.json",
    "p30-nvidia-smi.csv",
    "p30-running-container-inspect.json",
    "p30-running-mountinfo.txt",
    "p30-container-id.txt",
]
RUNTIME_EVIDENCE_ALLOWLIST = [
    "p30-prepare-attempt-layout.stdout.log",
    "p30-prepare-attempt-layout.stderr.log",
    "p30-prepare-attempt-layout.exit-status.txt",
    "p30-prepare-execution-snapshot-verification.stdout.log",
    "p30-prepare-execution-snapshot-verification.stderr.log",
    "p30-prepare-execution-snapshot-verification.exit-status.txt",
    "p30-image-inspection.stderr.log",
    "p30-image-inspection.exit-status.txt",
    "p30-nvidia-smi.stderr.log",
    "p30-nvidia-smi.exit-status.txt",
    "p30-container-launch.stderr.log",
    "p30-container-launch.exit-status.txt",
    "p30-fresh-container-id-validation.stdout.log",
    "p30-fresh-container-id-validation.stderr.log",
    "p30-fresh-container-id-validation.exit-status.txt",
    "p30-running-container-inspection.stderr.log",
    "p30-running-container-inspection.exit-status.txt",
    "p30-running-mountinfo.stderr.log",
    "p30-running-mountinfo.exit-status.txt",
    "p30-freeze-runtime.stdout.log",
    "p30-freeze-runtime.stderr.log",
    "p30-freeze-runtime.exit-status.txt",
    "p30-post-freeze-attempt-layout.stdout.log",
    "p30-post-freeze-attempt-layout.stderr.log",
    "p30-post-freeze-attempt-layout.exit-status.txt",
    "p30_cuda_runtime_lock.json",
    "p30_host_attestation.json",
]
RUNTIME_SUCCESS_STATUS_FILES = frozenset(
    name for name in RUNTIME_EVIDENCE_ALLOWLIST if name.endswith(".exit-status.txt")
)
SHA1_RE = re.compile(r"[0-9a-f]{40}\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

SNAPSHOT_SCHEMA = "passive-muon-p30-immutable-execution-snapshot-v1"
SNAPSHOT_REPOSITORY_COMMIT = P26_SOURCE_COMMIT
SNAPSHOT_REPOSITORY_TREE = P26_SOURCE_TREE
SNAPSHOT_MANIFEST_BASENAME = "snapshot-manifest.json"
SNAPSHOT_MANIFEST_MODE = 0o444
SNAPSHOT_TOP_LEVEL = (
    "data",
    "muon",
    "nanogpt",
    "repository",
    SNAPSHOT_MANIFEST_BASENAME,
)
SNAPSHOT_TOOL_BASENAMES = (
    "ingest_p26_trace_off_a_failure.py",
    "p23_deterministic_cuda_shadow_trace.py",
    "p26-trace-off-a-failure.authenticated.json",
    "run_p27_cuda_deleted_mapping_localization.py",
    "sanitize_p27_cuda_deleted_mapping_localization.py",
)
SNAPSHOT_TOOL_SOURCE_PATHS = {
    "run_p27_cuda_deleted_mapping_localization.py": (
        "experiments/training/run_p27_cuda_deleted_mapping_localization.py"
    ),
    "ingest_p26_trace_off_a_failure.py": "scripts/ingest_p26_trace_off_a_failure.py",
    "p23_deterministic_cuda_shadow_trace.py": (
        "experiments/training/p23_deterministic_cuda_shadow_trace.py"
    ),
    "sanitize_p27_cuda_deleted_mapping_localization.py": (
        "scripts/sanitize_p27_cuda_deleted_mapping_localization.py"
    ),
}
SNAPSHOT_TOOL_SHA256 = {
    "run_p27_cuda_deleted_mapping_localization.py": (
        "86ba200d69a029c64b2218581d6b92245930d5f1e60d0176aca22ad4d6ddf65a"
    ),
    "ingest_p26_trace_off_a_failure.py": (
        "3e7769d668a8756aea7359bef4d9b8ccfc8db66da102edfbc6c5de385654cdeb"
    ),
    "p23_deterministic_cuda_shadow_trace.py": (
        "ec1037839a3de378ee7391ae815ffc29c184865c5216d30027edc460ba112fab"
    ),
    "sanitize_p27_cuda_deleted_mapping_localization.py": (
        "3ec64cdc3681d41d70c624c6ffd26dde1a7e498bda13ddafaaced271a43dd6c4"
    ),
}
SNAPSHOT_TOOL_MODES = {
    "run_p27_cuda_deleted_mapping_localization.py": "0555",
    "ingest_p26_trace_off_a_failure.py": "0555",
    "p23_deterministic_cuda_shadow_trace.py": "0444",
    "sanitize_p27_cuda_deleted_mapping_localization.py": "0555",
    "p26-trace-off-a-failure.authenticated.json": "0444",
}
SNAPSHOT_FINEWEB_MANIFEST = "data/materialized/p22_fineweb_manifest.json"
SNAPSHOT_FINEWEB_MANIFEST_SHA256 = (
    "1dfbe2087fb61004c6c4ee938d1be53d769f2fa66303c2533e14acb9c1d5a96d"
)
SNAPSHOT_FINEWEB_MANIFEST_BYTE_COUNT = 4210
SNAPSHOT_P26_FAILURE_SHA256 = "b2a92b5062988534bc424d3ecadf3977ecedc0f80aa27628d4a0457858a64c91"
SNAPSHOT_P26_FAILURE_BYTE_COUNT = 66283
SNAPSHOT_GIT_CONFIG = {
    "core.repositoryformatversion": "0",
    "core.filemode": "true",
    "core.bare": "false",
    "core.logallrefupdates": "true",
}


class VerificationError(RuntimeError):
    """A fail-closed P30 transport-verification error."""


@dataclass(frozen=True)
class StableFile:
    sha256: str
    byte_count: int
    mode_octal: str
    uid: int
    gid: int
    device: int
    inode: int
    link_count: int


@dataclass(frozen=True)
class DirectoryIdentity:
    """Identity and authority fields bound for one guarded directory."""

    path: str
    device: int
    inode: int
    mode_octal: str
    uid: int
    gid: int
    mount_id: int | None


@dataclass(frozen=True)
class BoundDirectory:
    identity: DirectoryIdentity
    expected_uid: int
    expected_gid: int
    expected_mode: int
    acl_check: dict[str, object]


@dataclass
class NamespaceGuard:
    """Open-descriptor guard for the permission-safe P30 namespace."""

    component_fds: list[tuple[Path, int, tuple[int, int]]]
    secure_parent_fd: int
    namespace_fd: int
    transport_fd: int
    secure_parent: DirectoryIdentity
    namespace: DirectoryIdentity
    transport: DirectoryIdentity
    mount_check: dict[str, object]
    acl_check: dict[str, object]
    bound_directories: dict[str, tuple[int, BoundDirectory]]
    expected_namespace_children: tuple[str, ...] | None

    def close(self) -> None:
        while self.component_fds:
            _path, descriptor, _identity = self.component_fds.pop()
            os.close(descriptor)

    def __enter__(self) -> NamespaceGuard:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def revalidate(self) -> None:
        """Require every opened component and bound authority to remain stable."""

        for path, descriptor, expected_identity in self.component_fds:
            actual = os.fstat(descriptor)
            if (actual.st_dev, actual.st_ino) != expected_identity:
                raise VerificationError(f"open directory component changed identity: {path}")
            try:
                by_name = os.lstat(path)
            except OSError as exc:
                raise VerificationError(f"cannot restat guarded component {path}: {exc}") from exc
            if stat.S_ISLNK(by_name.st_mode) or not stat.S_ISDIR(by_name.st_mode):
                raise VerificationError(
                    f"guarded directory component is no longer a directory: {path}"
                )
            if (by_name.st_dev, by_name.st_ino) != expected_identity:
                raise VerificationError(f"guarded directory path changed identity: {path}")
        _require_directory_authority(
            self.secure_parent_fd,
            Path(self.secure_parent.path),
            expected_uid=SECURE_PARENT_UID,
            expected_gid=SECURE_PARENT_GID,
            expected_mode=SECURE_PARENT_MODE,
        )
        _require_directory_authority(
            self.namespace_fd,
            Path(self.namespace.path),
            expected_uid=NAMESPACE_UID,
            expected_gid=NAMESPACE_GID,
            expected_mode=NAMESPACE_MODE,
        )
        _require_directory_authority(
            self.transport_fd,
            Path(self.transport.path),
            expected_uid=TRANSPORT_UID,
            expected_gid=TRANSPORT_GID,
            expected_mode=TRANSPORT_MODE,
        )
        if (
            _directory_identity(self.secure_parent_fd, Path(self.secure_parent.path))
            != self.secure_parent
        ):
            raise VerificationError("P30 secure-parent authority changed during verification")
        if _directory_identity(self.namespace_fd, Path(self.namespace.path)) != self.namespace:
            raise VerificationError("P30 namespace authority changed during verification")
        if _directory_identity(self.transport_fd, Path(self.transport.path)) != self.transport:
            raise VerificationError("P30 transport authority changed during verification")
        if (
            _validate_mount_layout(Path(self.namespace.path), Path(self.transport.path))
            != self.mount_check
        ):
            raise VerificationError("P30 namespace mount topology changed during verification")
        if (
            _validate_acl_absence(
                {
                    "secure_parent": self.secure_parent_fd,
                    "namespace": self.namespace_fd,
                    "transport": self.transport_fd,
                }
            )
            != self.acl_check
        ):
            raise VerificationError("P30 namespace ACL state changed during verification")
        if self.expected_namespace_children is not None:
            _validate_namespace_child_inventory(
                self.namespace_fd,
                self.expected_namespace_children,
                transport=self.transport,
            )
        for label, (descriptor, binding) in self.bound_directories.items():
            path = Path(binding.identity.path)
            _require_directory_authority(
                descriptor,
                path,
                expected_uid=binding.expected_uid,
                expected_gid=binding.expected_gid,
                expected_mode=binding.expected_mode,
            )
            if _directory_identity(descriptor, path) != binding.identity:
                raise VerificationError(f"guarded {label} directory identity changed")
            if _validate_acl_absence({label: descriptor}) != binding.acl_check:
                raise VerificationError(f"guarded {label} ACL state changed")
            if _is_exact_mountpoint(path):
                raise VerificationError(f"guarded {label} became an unexpected mountpoint")

    def bound_fd(self, label: str) -> int:
        """Return one still-open child authority descriptor by its exact label."""

        try:
            descriptor, _binding = self.bound_directories[label]
        except KeyError as exc:
            raise VerificationError(f"guarded {label} directory is not bound") from exc
        return descriptor

    def bind_namespace_inventory(self, expected: Sequence[str]) -> None:
        """Freeze the phase-specific, capability-bearing namespace inventory."""

        normalized = tuple(sorted(expected))
        if self.expected_namespace_children is not None:
            raise VerificationError("P30 namespace child inventory was already bound")
        _validate_namespace_child_inventory(
            self.namespace_fd,
            normalized,
            transport=self.transport,
        )
        self.expected_namespace_children = normalized

    def bind_child(
        self,
        *,
        label: str,
        parent_fd: int,
        path: Path,
        expected_uid: int,
        expected_gid: int,
        expected_mode: int,
        expected_mount_id: int | None,
        expected_device: int,
    ) -> int:
        if label in self.bound_directories:
            raise VerificationError(f"guarded directory label is duplicated: {label}")
        try:
            descriptor = os.open(path.name, _directory_open_flags(), dir_fd=parent_fd)
        except OSError as exc:
            raise VerificationError(f"cannot safely open guarded {label}: {exc}") from exc
        try:
            _require_directory_authority(
                descriptor,
                path,
                expected_uid=expected_uid,
                expected_gid=expected_gid,
                expected_mode=expected_mode,
            )
            identity = _directory_identity(descriptor, path)
            if identity.device != expected_device or identity.mount_id != expected_mount_id:
                raise VerificationError(f"guarded {label} crosses a device or mount boundary")
            acl_check = _validate_acl_absence({label: descriptor})
            if _is_exact_mountpoint(path):
                raise VerificationError(f"guarded {label} is an unexpected mountpoint")
            by_name = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
            if (by_name.st_dev, by_name.st_ino) != (identity.device, identity.inode):
                raise VerificationError(f"guarded {label} path identity differs")
            self.component_fds.append((path, descriptor, (identity.device, identity.inode)))
            self.bound_directories[label] = (
                descriptor,
                BoundDirectory(
                    identity=identity,
                    expected_uid=expected_uid,
                    expected_gid=expected_gid,
                    expected_mode=expected_mode,
                    acl_check=acl_check,
                ),
            )
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    def receipt_record(self) -> dict[str, object]:
        if self.expected_namespace_children is None:
            raise VerificationError("P30 namespace child inventory was not bound")
        return {
            "secure_parent": self.secure_parent.__dict__,
            "namespace": self.namespace.__dict__,
            "transport": self.transport.__dict__,
            "componentwise_o_directory_o_nofollow": True,
            "open_descriptors_held_through_prepublication_validation": True,
            "device_inode_stable_before_and_after_git_and_control_transitions": True,
            "symlink_components_rejected": True,
            "mount_check": self.mount_check,
            "acl_check": self.acl_check,
            "exact_namespace_child_inventory": {
                "names": list(self.expected_namespace_children),
                "types": {name: "directory" for name in self.expected_namespace_children},
                "extra_siblings_and_capabilities_rejected": True,
                "inventory_revalidated_through_receipt_handling": True,
            },
            "guarded_child_directories": {
                label: {
                    **binding.identity.__dict__,
                    "expected_uid": binding.expected_uid,
                    "expected_gid": binding.expected_gid,
                    "expected_mode_octal": format(binding.expected_mode, "04o"),
                    "acl_check": binding.acl_check,
                }
                for label, (_descriptor, binding) in sorted(self.bound_directories.items())
            },
        }


def _validate_namespace_child_inventory(
    namespace_fd: int,
    expected: Sequence[str],
    *,
    transport: DirectoryIdentity,
) -> None:
    """Require an exact directory-only inventory below the P30 namespace."""

    expected_names = tuple(sorted(expected))
    if len(set(expected_names)) != len(expected_names):
        raise VerificationError("P30 namespace child inventory contains duplicates")
    for name in expected_names:
        if name in {"", ".", ".."} or "/" in name:
            raise VerificationError("P30 namespace child inventory contains an unsafe name")
    try:
        actual_names = tuple(sorted(os.listdir(namespace_fd)))
    except OSError as exc:
        raise VerificationError(f"cannot enumerate P30 namespace children: {exc}") from exc
    if actual_names != expected_names:
        raise VerificationError(
            "P30 namespace child inventory differs: "
            f"expected {list(expected_names)!r}, got {list(actual_names)!r}"
        )
    for name in actual_names:
        try:
            info = os.stat(name, dir_fd=namespace_fd, follow_symlinks=False)
        except OSError as exc:
            raise VerificationError(f"cannot stat P30 namespace child {name}: {exc}") from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise VerificationError(f"P30 namespace child is not a directory: {name}")
        if name == TRANSPORT_BASENAME and (info.st_dev, info.st_ino) != (
            transport.device,
            transport.inode,
        ):
            raise VerificationError("P30 transport namespace entry changed identity")


def _directory_identity(descriptor: int, path: Path) -> DirectoryIdentity:
    info = os.fstat(descriptor)
    return DirectoryIdentity(
        path=str(path),
        device=info.st_dev,
        inode=info.st_ino,
        mode_octal=format(stat.S_IMODE(info.st_mode), "04o"),
        uid=info.st_uid,
        gid=info.st_gid,
        mount_id=_fd_mount_id(descriptor),
    )


def _fd_mount_id(descriptor: int) -> int | None:
    if platform.system() != "Linux":
        return None
    fdinfo = Path(f"/proc/self/fdinfo/{descriptor}")
    try:
        lines = fdinfo.read_text(encoding="ascii").splitlines()
    except OSError as exc:
        raise VerificationError(f"cannot read mount identity for directory fd: {exc}") from exc
    values = [line.split(":", 1)[1].strip() for line in lines if line.startswith("mnt_id:")]
    if len(values) != 1 or not values[0].isdigit():
        raise VerificationError("directory fdinfo has no unique numeric mount ID")
    return int(values[0])


def _require_directory_authority(
    descriptor: int,
    path: Path,
    *,
    expected_uid: int,
    expected_gid: int,
    expected_mode: int,
) -> None:
    info = os.fstat(descriptor)
    if not stat.S_ISDIR(info.st_mode):
        raise VerificationError(f"authority path is not a directory: {path}")
    actual_mode = stat.S_IMODE(info.st_mode)
    if (info.st_uid, info.st_gid, actual_mode) != (
        expected_uid,
        expected_gid,
        expected_mode,
    ):
        raise VerificationError(
            f"authority differs for {path}: expected uid/gid/mode "
            f"{expected_uid}/{expected_gid}/{expected_mode:04o}, got "
            f"{info.st_uid}/{info.st_gid}/{actual_mode:04o}"
        )


def _decode_mount_field(value: str) -> str:
    for encoded, decoded in (
        ("\\040", " "),
        ("\\011", "\t"),
        ("\\012", "\n"),
        ("\\134", "\\"),
    ):
        value = value.replace(encoded, decoded)
    return value


def _mount_inventory() -> list[tuple[str, str, str]] | None:
    """Return Linux mount IDs, devices, and mountpoints when available."""

    if not MOUNTINFO_PATH.exists():
        if platform.system() == "Linux":
            raise VerificationError("Linux mountinfo is unavailable")
        return None
    try:
        lines = MOUNTINFO_PATH.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise VerificationError(f"cannot read Linux mountinfo: {exc}") from exc
    records: list[tuple[str, str, str]] = []
    for line in lines:
        fields = line.split()
        if len(fields) < 10 or "-" not in fields:
            raise VerificationError("Linux mountinfo contains a malformed line")
        mount_id, device, mountpoint = fields[0], fields[2], _decode_mount_field(fields[4])
        records.append((mount_id, device, mountpoint))
    return records


def _is_exact_mountpoint(path: Path) -> bool:
    records = _mount_inventory()
    return records is not None and str(path) in {record[2] for record in records}


def _directory_open_flags() -> int:
    flags = os.O_RDONLY
    for name in ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW"):
        value = getattr(os, name, None)
        if not isinstance(value, int) or value == 0:
            raise VerificationError(f"platform lacks mandatory {name}")
        flags |= value
    return flags


def _containing_mount(path: Path, records: Sequence[tuple[str, str, str]]) -> tuple[str, str, str]:
    path_string = str(path)
    matches = [
        record
        for record in records
        if path_string == record[2] or path_string.startswith(record[2].rstrip("/") + "/")
    ]
    if not matches:
        raise VerificationError(f"no containing mount found for {path}")
    return max(matches, key=lambda record: len(record[2]))


def _validate_mount_layout(namespace: Path, transport: Path) -> dict[str, object]:
    records = _mount_inventory()
    if records is None:
        return {
            "method": "not_available_on_non_linux_platform",
            "supported": False,
            "separate_namespace_or_transport_mount_rejected": True,
        }
    namespace_mount = _containing_mount(namespace, records)
    transport_mount = _containing_mount(transport, records)
    exact_mountpoints = {record[2] for record in records}
    if str(namespace) in exact_mountpoints or str(transport) in exact_mountpoints:
        raise VerificationError("P30 namespace or transport is an unexpected mountpoint")
    if namespace_mount[:2] != transport_mount[:2]:
        raise VerificationError("P30 namespace and transport cross a mount boundary")
    return {
        "method": "linux_proc_self_mountinfo",
        "supported": True,
        "containing_mount_id": namespace_mount[0],
        "containing_mount_device": namespace_mount[1],
        "namespace_and_transport_same_mount": True,
        "namespace_and_transport_not_mountpoints": True,
    }


def _validate_acl_absence(descriptors: Mapping[str, int]) -> dict[str, object]:
    listxattr = getattr(os, "listxattr", None)
    if listxattr is None:
        if platform.system() == "Linux":
            raise VerificationError("Linux ACL xattr inspection is unavailable")
        return {
            "method": "not_available_on_non_linux_platform",
            "supported": False,
            "access_acl_absent": True,
            "default_acl_absent": True,
        }
    seen: dict[str, list[str]] = {}
    for label, descriptor in descriptors.items():
        try:
            names = sorted(str(name) for name in listxattr(descriptor))
        except OSError as exc:
            raise VerificationError(f"cannot inspect ACL xattrs for {label}: {exc}") from exc
        present = sorted(set(names) & ACL_XATTR_NAMES)
        if present:
            raise VerificationError(f"ACL xattr present on {label}: {present!r}")
        seen[label] = names
    return {
        "method": "descriptor_listxattr",
        "supported": True,
        "inspected_directories": sorted(descriptors),
        "access_acl_absent": True,
        "default_acl_absent": True,
        "all_xattr_names": seen,
    }


def open_namespace_guard(namespace: Path, transport: Path) -> NamespaceGuard:
    """Traverse every absolute component without following directory symlinks."""

    if not namespace.is_absolute() or not transport.is_absolute():
        raise VerificationError("P30 namespace paths must be absolute")
    if transport != namespace / TRANSPORT_BASENAME:
        raise VerificationError("P30 transport is not the exact namespace child")
    flags = _directory_open_flags()
    components = [*namespace.parts[1:], TRANSPORT_BASENAME]
    opened: list[tuple[Path, int, tuple[int, int]]] = []
    current = Path("/")
    descriptor = os.open("/", flags)
    root_info = os.fstat(descriptor)
    opened.append((current, descriptor, (root_info.st_dev, root_info.st_ino)))
    try:
        secure_parent_fd: int | None = None
        namespace_fd: int | None = None
        for index, component in enumerate(components):
            if component in {"", ".", ".."} or "/" in component:
                raise VerificationError("P30 namespace has an unsafe path component")
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except OSError as exc:
                raise VerificationError(
                    f"cannot open P30 directory component {current / component}: {exc}"
                ) from exc
            current /= component
            child_info = os.fstat(child)
            opened.append((current, child, (child_info.st_dev, child_info.st_ino)))
            descriptor = child
            if current == SECURE_PARENT:
                secure_parent_fd = child
            if index == len(components) - 2:
                namespace_fd = child
        assert secure_parent_fd is not None
        assert namespace_fd is not None
        transport_fd = descriptor
        _require_directory_authority(
            secure_parent_fd,
            SECURE_PARENT,
            expected_uid=SECURE_PARENT_UID,
            expected_gid=SECURE_PARENT_GID,
            expected_mode=SECURE_PARENT_MODE,
        )
        _require_directory_authority(
            namespace_fd,
            namespace,
            expected_uid=NAMESPACE_UID,
            expected_gid=NAMESPACE_GID,
            expected_mode=NAMESPACE_MODE,
        )
        _require_directory_authority(
            transport_fd,
            transport,
            expected_uid=TRANSPORT_UID,
            expected_gid=TRANSPORT_GID,
            expected_mode=TRANSPORT_MODE,
        )
        secure_parent_identity = _directory_identity(secure_parent_fd, SECURE_PARENT)
        namespace_identity = _directory_identity(namespace_fd, namespace)
        transport_identity = _directory_identity(transport_fd, transport)
        if namespace_identity.device != transport_identity.device:
            raise VerificationError("P30 namespace and transport have different devices")
        if namespace_identity.mount_id != transport_identity.mount_id:
            raise VerificationError("P30 namespace and transport have different mount IDs")
        mount_check = _validate_mount_layout(namespace, transport)
        if (
            secure_parent_identity.device != namespace_identity.device
            or secure_parent_identity.mount_id != namespace_identity.mount_id
        ):
            raise VerificationError("P30 secure parent and namespace cross a mount boundary")
        if _is_exact_mountpoint(SECURE_PARENT):
            raise VerificationError("P30 secure parent is an unexpected mountpoint")
        acl_check = _validate_acl_absence(
            {
                "secure_parent": secure_parent_fd,
                "namespace": namespace_fd,
                "transport": transport_fd,
            }
        )
        return NamespaceGuard(
            component_fds=opened,
            secure_parent_fd=secure_parent_fd,
            namespace_fd=namespace_fd,
            transport_fd=transport_fd,
            secure_parent=secure_parent_identity,
            namespace=namespace_identity,
            transport=transport_identity,
            mount_check=mount_check,
            acl_check=acl_check,
            bound_directories={},
            expected_namespace_children=None,
        )
    except BaseException:
        while opened:
            _path, opened_fd, _identity = opened.pop()
            os.close(opened_fd)
        raise


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise VerificationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(token: str) -> None:
    raise VerificationError(f"nonfinite JSON constant: {token}")


def strict_json_loads(raw: bytes) -> Any:
    try:
        return json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"invalid strict JSON: {exc}") from exc


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def compact_canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _stable_fields(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_uid,
        info.st_gid,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _stat_source_nofollow(source: Path, *, source_dir_fd: int | None, label: str) -> os.stat_result:
    """Stat the exact source name without following a replacement symlink."""

    try:
        return os.stat(
            source.name if source_dir_fd is not None else source,
            dir_fd=source_dir_fd,
            follow_symlinks=False,
        )
    except OSError as exc:
        raise VerificationError(f"cannot stat {label} without following links: {exc}") from exc


def stable_copy_nofollow(
    source: Path, destination: Path, *, source_dir_fd: int | None = None
) -> StableFile:
    """Copy ``source`` exactly once through a no-follow descriptor.

    All later Git operations consume ``destination``, never the transport path.
    The source descriptor is checked before and after the read so size or inode
    mutation cannot silently change the authenticated byte stream.
    """

    named_before = _stat_source_nofollow(
        source, source_dir_fd=source_dir_fd, label="transport bundle"
    )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        source_fd = os.open(
            source.name if source_dir_fd is not None else source,
            flags,
            dir_fd=source_dir_fd,
        )
    except OSError as exc:
        raise VerificationError(f"cannot open bundle with O_NOFOLLOW: {exc}") from exc
    destination_fd: int | None = None
    try:
        before = os.fstat(source_fd)
        if not stat.S_ISREG(before.st_mode):
            raise VerificationError("transport bundle is not a regular file")
        destination_fd = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        digest = hashlib.sha256()
        byte_count = 0
        while True:
            chunk = os.read(source_fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(destination_fd, view)
                if written <= 0:
                    raise VerificationError("short write while privatizing bundle")
                view = view[written:]
        os.fsync(destination_fd)
        after = os.fstat(source_fd)
        named_after = _stat_source_nofollow(
            source, source_dir_fd=source_dir_fd, label="transport bundle"
        )
        if not (
            _stable_fields(named_before)
            == _stable_fields(before)
            == _stable_fields(after)
            == _stable_fields(named_after)
        ):
            raise VerificationError("transport bundle changed during its stable read")
        if byte_count != before.st_size:
            raise VerificationError("transport bundle size disagrees with bytes read")
        return StableFile(
            sha256=digest.hexdigest(),
            byte_count=byte_count,
            mode_octal=format(stat.S_IMODE(before.st_mode), "04o"),
            uid=before.st_uid,
            gid=before.st_gid,
            device=before.st_dev,
            inode=before.st_ino,
            link_count=before.st_nlink,
        )
    finally:
        if destination_fd is not None:
            os.close(destination_fd)
        os.close(source_fd)


def stable_read_nofollow(
    source: Path, *, source_dir_fd: int | None = None
) -> tuple[bytes, StableFile]:
    named_before = _stat_source_nofollow(
        source, source_dir_fd=source_dir_fd, label="reviewed receipt"
    )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(
            source.name if source_dir_fd is not None else source,
            flags,
            dir_fd=source_dir_fd,
        )
    except OSError as exc:
        raise VerificationError(f"cannot open receipt with O_NOFOLLOW: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise VerificationError("reviewed receipt is not a regular file")
        pieces: list[bytes] = []
        byte_count = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            pieces.append(chunk)
            byte_count += len(chunk)
        after = os.fstat(descriptor)
        named_after = _stat_source_nofollow(
            source, source_dir_fd=source_dir_fd, label="reviewed receipt"
        )
        if not (
            _stable_fields(named_before)
            == _stable_fields(before)
            == _stable_fields(after)
            == _stable_fields(named_after)
        ):
            raise VerificationError("reviewed receipt changed during its stable read")
        if byte_count != before.st_size:
            raise VerificationError("reviewed receipt size disagrees with bytes read")
        raw = b"".join(pieces)
        return raw, StableFile(
            sha256=sha256_bytes(raw),
            byte_count=byte_count,
            mode_octal=format(stat.S_IMODE(before.st_mode), "04o"),
            uid=before.st_uid,
            gid=before.st_gid,
            device=before.st_dev,
            inode=before.st_ino,
            link_count=before.st_nlink,
        )
    finally:
        os.close(descriptor)


def _require_no_snapshot_xattrs(descriptor: int, label: str) -> None:
    """Require the immutable snapshot object to carry no hidden metadata."""

    listxattr = getattr(os, "listxattr", None)
    if listxattr is None:
        raise VerificationError("snapshot extended-attribute inspection is unavailable")
    try:
        raw_names = listxattr(descriptor)
    except OSError as exc:
        raise VerificationError(f"cannot inspect snapshot xattrs for {label}: {exc}") from exc
    names: list[str] = []
    for raw_name in raw_names:
        if isinstance(raw_name, bytes):
            try:
                names.append(raw_name.decode("utf-8"))
            except UnicodeDecodeError as exc:
                raise VerificationError(f"snapshot xattr name is not UTF-8 for {label}") from exc
        else:
            names.append(str(raw_name))
    if names:
        raise VerificationError(
            f"snapshot object has unexpected extended attributes: {label}: {sorted(names)!r}"
        )


def _require_snapshot_directory(
    descriptor: int,
    path: Path,
    *,
    root_device: int,
    root_mount_id: int | None,
) -> os.stat_result:
    info = os.fstat(descriptor)
    if not stat.S_ISDIR(info.st_mode):
        raise VerificationError(f"snapshot entry is not a directory: {path}")
    if (info.st_uid, info.st_gid, stat.S_IMODE(info.st_mode)) != (
        EXECUTION_UID,
        EXECUTION_GID,
        0o555,
    ):
        raise VerificationError(
            f"snapshot directory authority differs: {path}; expected "
            f"{EXECUTION_UID}/{EXECUTION_GID}/0555"
        )
    if info.st_dev != root_device or _fd_mount_id(descriptor) != root_mount_id:
        raise VerificationError(f"snapshot directory crosses a device or mount boundary: {path}")
    _require_no_snapshot_xattrs(descriptor, str(path))
    return info


def _read_snapshot_regular(
    parent_fd: int,
    name: str,
    path: Path,
    *,
    root_device: int,
    root_mount_id: int | None,
    manifest: bool,
) -> tuple[dict[str, object], bytes | None]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent_fd)
    except OSError as exc:
        raise VerificationError(f"cannot safely open snapshot file {path}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        mode = stat.S_IMODE(before.st_mode)
        allowed_modes = {SNAPSHOT_MANIFEST_MODE} if manifest else {0o444, 0o555}
        if not stat.S_ISREG(before.st_mode):
            raise VerificationError(f"snapshot entry is not a regular file: {path}")
        if before.st_nlink != 1:
            raise VerificationError(f"snapshot file is hard linked: {path}")
        if (before.st_uid, before.st_gid) != (EXECUTION_UID, EXECUTION_GID):
            raise VerificationError(f"snapshot file owner differs: {path}")
        if mode not in allowed_modes:
            expected = format(SNAPSHOT_MANIFEST_MODE, "04o") if manifest else "0444 or 0555"
            raise VerificationError(f"snapshot file mode is not {expected}: {path}")
        if before.st_dev != root_device or _fd_mount_id(descriptor) != root_mount_id:
            raise VerificationError(f"snapshot file crosses a device or mount boundary: {path}")
        _require_no_snapshot_xattrs(descriptor, str(path))
        digest = hashlib.sha256()
        byte_count = 0
        captured: list[bytes] | None = [] if manifest else None
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
            if captured is not None:
                captured.append(chunk)
        after = os.fstat(descriptor)
        by_name = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if _stable_fields(before) != _stable_fields(after) or _stable_fields(
            after
        ) != _stable_fields(by_name):
            raise VerificationError(f"snapshot file changed during verification: {path}")
        if byte_count != before.st_size:
            raise VerificationError(f"snapshot file size differs from bytes read: {path}")
        return (
            {
                "kind": "file",
                "mode_octal": format(mode, "04o"),
                "uid": before.st_uid,
                "gid": before.st_gid,
                "byte_count": byte_count,
                "sha256": digest.hexdigest(),
            },
            b"".join(captured) if captured is not None else None,
        )
    finally:
        os.close(descriptor)


def _scan_snapshot_directory(
    descriptor: int,
    path: Path,
    relative: str,
    *,
    root_device: int,
    root_mount_id: int | None,
    entries: dict[str, dict[str, object]],
) -> tuple[bytes, dict[str, object]] | None:
    before = _require_snapshot_directory(
        descriptor,
        path,
        root_device=root_device,
        root_mount_id=root_mount_id,
    )
    try:
        names_before = sorted(os.listdir(descriptor))
    except OSError as exc:
        raise VerificationError(f"cannot enumerate snapshot directory {path}: {exc}") from exc
    manifest_result: tuple[bytes, dict[str, object]] | None = None
    for name in names_before:
        if name in {"", ".", ".."} or "/" in name or "\0" in name:
            raise VerificationError(f"snapshot contains an unsafe entry name below {path}")
        child_path = path / name
        child_relative = name if not relative else f"{relative}/{name}"
        try:
            by_name = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        except OSError as exc:
            raise VerificationError(f"cannot stat snapshot entry {child_path}: {exc}") from exc
        if stat.S_ISLNK(by_name.st_mode):
            raise VerificationError(f"snapshot contains a symlink: {child_path}")
        if stat.S_ISDIR(by_name.st_mode):
            try:
                child_fd = os.open(name, _directory_open_flags(), dir_fd=descriptor)
            except OSError as exc:
                raise VerificationError(
                    f"cannot safely open snapshot directory {child_path}: {exc}"
                ) from exc
            try:
                opened = os.fstat(child_fd)
                if (opened.st_dev, opened.st_ino) != (by_name.st_dev, by_name.st_ino):
                    raise VerificationError(f"snapshot directory identity differs: {child_path}")
                entries[child_relative] = {
                    "kind": "directory",
                    "mode_octal": format(stat.S_IMODE(opened.st_mode), "04o"),
                    "uid": opened.st_uid,
                    "gid": opened.st_gid,
                }
                nested_manifest = _scan_snapshot_directory(
                    child_fd,
                    child_path,
                    child_relative,
                    root_device=root_device,
                    root_mount_id=root_mount_id,
                    entries=entries,
                )
                if nested_manifest is not None:
                    raise VerificationError("snapshot manifest is not a direct child of its root")
            finally:
                os.close(child_fd)
        elif stat.S_ISREG(by_name.st_mode):
            is_manifest = not relative and name == SNAPSHOT_MANIFEST_BASENAME
            record, raw = _read_snapshot_regular(
                descriptor,
                name,
                child_path,
                root_device=root_device,
                root_mount_id=root_mount_id,
                manifest=is_manifest,
            )
            if is_manifest:
                assert raw is not None
                manifest_result = (raw, record)
            else:
                entries[child_relative] = record
        else:
            raise VerificationError(f"snapshot contains a special file: {child_path}")
    try:
        names_after = sorted(os.listdir(descriptor))
    except OSError as exc:
        raise VerificationError(f"cannot re-enumerate snapshot directory {path}: {exc}") from exc
    after = os.fstat(descriptor)
    if names_after != names_before or _stable_fields(before) != _stable_fields(after):
        raise VerificationError(f"snapshot directory changed during verification: {path}")
    return manifest_result


def _parse_snapshot_git_records(
    raw: bytes, *, label: str, tree: bool
) -> dict[str, tuple[str, str]]:
    records: dict[str, tuple[str, str]] = {}
    for item in raw.split(b"\0"):
        if not item:
            continue
        try:
            header, raw_path = item.split(b"\t", 1)
            path = raw_path.decode("utf-8")
            fields = header.decode("ascii").split()
        except (ValueError, UnicodeDecodeError) as exc:
            raise VerificationError(f"snapshot {label} Git inventory is malformed") from exc
        if tree:
            if len(fields) != 3 or fields[1] != "blob":
                raise VerificationError(f"snapshot {label} tree contains a non-blob entry")
            mode, object_id = fields[0], fields[2]
        else:
            if len(fields) != 3 or fields[2] != "0":
                raise VerificationError(f"snapshot {label} index contains a non-stage-zero entry")
            mode, object_id = fields[0], fields[1]
        if mode not in {"100644", "100755"}:
            raise VerificationError(f"snapshot {label} contains an unsupported tracked mode")
        require_sha1(object_id, f"snapshot {label} tracked blob")
        pure_parts = Path(path).parts
        if not path or path.startswith("/") or any(part in {"", ".", ".."} for part in pure_parts):
            raise VerificationError(f"snapshot {label} contains an unsafe tracked path")
        if path in records:
            raise VerificationError(f"snapshot {label} contains a duplicate tracked path")
        records[path] = (mode, object_id)
    return records


def _snapshot_local_config(repository: Path, label: str) -> dict[str, str]:
    raw = run_git(
        ["-C", str(repository), "config", "--no-includes", "--local", "--null", "--list"]
    ).stdout
    parsed: dict[str, str] = {}
    for item in raw.split(b"\0"):
        if not item:
            continue
        try:
            raw_key, raw_value = item.split(b"\n", 1)
            key = raw_key.decode("utf-8").lower()
            value = raw_value.decode("utf-8")
        except (ValueError, UnicodeDecodeError) as exc:
            raise VerificationError(f"snapshot {label} local Git config is malformed") from exc
        if key in parsed:
            raise VerificationError(f"snapshot {label} local Git config repeats {key}")
        parsed[key] = value
    if parsed != SNAPSHOT_GIT_CONFIG:
        raise VerificationError(
            f"snapshot {label} local Git config differs: expected "
            f"{SNAPSHOT_GIT_CONFIG!r}, got {parsed!r}"
        )
    return parsed


def _git_blob_oid_nofollow(path: Path, expected_size: int) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise VerificationError(f"cannot reopen snapshot tracked file {path}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected_size:
            raise VerificationError(f"snapshot tracked-file size or type differs: {path}")
        digest = hashlib.sha1(f"blob {expected_size}\0".encode("ascii"))
        byte_count = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
        after = os.fstat(descriptor)
        if _stable_fields(before) != _stable_fields(after) or byte_count != expected_size:
            raise VerificationError(f"snapshot tracked file changed during hashing: {path}")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _verify_snapshot_git_checkout(
    snapshot_root: Path,
    label: str,
    expected_head: str,
    expected_tree: str,
    entries: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    repository = snapshot_root / label
    git_directory = repository / ".git"
    try:
        dot_git = os.lstat(git_directory)
    except OSError as exc:
        raise VerificationError(f"snapshot {label} .git directory is unavailable: {exc}") from exc
    if stat.S_ISLNK(dot_git.st_mode) or not stat.S_ISDIR(dot_git.st_mode):
        raise VerificationError(f"snapshot {label} .git is not one directory")
    assert_no_object_indirections(repository, bare=False)
    _snapshot_local_config(repository, label)
    if git_text(repository, ["rev-parse", "--is-bare-repository"]) != "false":
        raise VerificationError(f"snapshot {label} is a bare repository")
    if run_git(["-C", str(repository), "symbolic-ref", "-q", "HEAD"], check=False).returncode == 0:
        raise VerificationError(f"snapshot {label} HEAD is not detached")
    if git_text(repository, ["rev-parse", "HEAD"]) != expected_head:
        raise VerificationError(f"snapshot {label} HEAD differs")
    if git_text(repository, ["rev-parse", "HEAD^{tree}"]) != expected_tree:
        raise VerificationError(f"snapshot {label} tree differs")
    index = _parse_snapshot_git_records(
        run_git(["-C", str(repository), "ls-files", "--stage", "-z"]).stdout,
        label=label,
        tree=False,
    )
    tree = _parse_snapshot_git_records(
        run_git(["-C", str(repository), "ls-tree", "-r", "-z", "HEAD"]).stdout,
        label=label,
        tree=True,
    )
    if index != tree:
        raise VerificationError(f"snapshot {label} index differs from its pinned tree")
    prefix = f"{label}/"
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for snapshot_path, record in entries.items():
        if not snapshot_path.startswith(prefix):
            continue
        relative = snapshot_path[len(prefix) :]
        if relative == ".git" or relative.startswith(".git/"):
            continue
        if record.get("kind") == "file":
            actual_files.add(relative)
        elif record.get("kind") == "directory":
            actual_directories.add(relative)
    expected_directories: set[str] = set()
    for tracked_path in index:
        parent = Path(tracked_path).parent
        while str(parent) != ".":
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    if actual_files != set(index) or actual_directories != expected_directories:
        raise VerificationError(f"snapshot {label} worktree inventory differs from its index")
    for tracked_path, (index_mode, object_id) in index.items():
        snapshot_path = f"{label}/{tracked_path}"
        record = entries[snapshot_path]
        expected_mode = "0555" if index_mode == "100755" else "0444"
        if record.get("mode_octal") != expected_mode:
            raise VerificationError(
                f"snapshot {label} tracked-file mode differs from its index: {tracked_path}"
            )
        byte_count = record.get("byte_count")
        if type(byte_count) is not int or byte_count < 0:
            raise VerificationError(f"snapshot {label} tracked-file size is malformed")
        if _git_blob_oid_nofollow(repository / tracked_path, byte_count) != object_id:
            raise VerificationError(
                f"snapshot {label} tracked-file bytes differ from its index: {tracked_path}"
            )
    assert_worktree_exactly_clean(repository, f"snapshot {label}")
    run_git(["-C", str(repository), "fsck", "--full", "--strict", "--no-dangling"])
    return {
        "head": expected_head,
        "tree": expected_tree,
        "tracked_file_count": len(index),
        "exact_local_config": dict(SNAPSHOT_GIT_CONFIG),
        "detached": True,
        "index_equals_tree": True,
        "worktree_bytes_and_modes_equal_index": True,
        "untracked_and_ignored_absent": True,
        "object_indirections_absent": True,
    }


def verify_immutable_execution_snapshot(
    execution_root: Path,
    *,
    execution_fd: int,
    control: Path,
    source_commit: str,
) -> dict[str, object]:
    """Verify the root-owned P30 execution snapshot and its complete manifest."""

    root_info = os.fstat(execution_fd)
    if (root_info.st_uid, root_info.st_gid, stat.S_IMODE(root_info.st_mode)) != (
        EXECUTION_UID,
        EXECUTION_GID,
        EXECUTION_MODE,
    ):
        raise VerificationError("immutable execution snapshot root authority differs")
    root_mount_id = _fd_mount_id(execution_fd)
    records = _mount_inventory()
    if records is not None:
        prefix = str(execution_root).rstrip("/") + "/"
        nested = sorted(
            mountpoint
            for _mount_id, _device, mountpoint in records
            if mountpoint == str(execution_root) or mountpoint.startswith(prefix)
        )
        if nested:
            raise VerificationError(
                f"immutable execution snapshot contains a mountpoint: {nested!r}"
            )
    try:
        top_level = tuple(sorted(os.listdir(execution_fd)))
    except OSError as exc:
        raise VerificationError(f"cannot enumerate immutable execution snapshot: {exc}") from exc
    if top_level != SNAPSHOT_TOP_LEVEL:
        raise VerificationError(
            f"immutable execution snapshot top inventory differs: expected "
            f"{list(SNAPSHOT_TOP_LEVEL)!r}, got {list(top_level)!r}"
        )
    actual_entries: dict[str, dict[str, object]] = {}
    manifest_result = _scan_snapshot_directory(
        execution_fd,
        execution_root,
        "",
        root_device=root_info.st_dev,
        root_mount_id=root_mount_id,
        entries=actual_entries,
    )
    if manifest_result is None:
        raise VerificationError("immutable execution snapshot manifest is absent")
    manifest_raw, manifest_record = manifest_result
    manifest = strict_json_loads(manifest_raw)
    if not isinstance(manifest, Mapping) or set(manifest) != {
        "schema_version",
        "source_authorities",
        "entries",
    }:
        raise VerificationError("immutable execution snapshot manifest keys differ")
    if manifest.get("schema_version") != SNAPSHOT_SCHEMA:
        raise VerificationError("immutable execution snapshot manifest schema differs")
    expected_canonical = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    if manifest_raw != expected_canonical:
        raise VerificationError("immutable execution snapshot manifest is not canonical JSON")
    declared_entries = manifest.get("entries")
    if not isinstance(declared_entries, list):
        raise VerificationError("immutable execution snapshot entries are not a list")
    normalized_entries: dict[str, dict[str, object]] = {}
    previous_path: str | None = None
    for raw_entry in declared_entries:
        if not isinstance(raw_entry, Mapping):
            raise VerificationError("immutable execution snapshot entry is not an object")
        path = raw_entry.get("relative_path")
        kind = raw_entry.get("kind")
        common_keys = {"relative_path", "kind", "mode_octal", "uid", "gid"}
        expected_keys = common_keys | ({"sha256", "byte_count"} if kind == "file" else set())
        if kind not in {"directory", "file"} or set(raw_entry) != expected_keys:
            raise VerificationError("immutable execution snapshot entry shape differs")
        if not isinstance(path, str) or not path or path.startswith("/"):
            raise VerificationError("immutable execution snapshot entry path is malformed")
        parts = Path(path).parts
        if any(part in {"", ".", ".."} for part in parts) or Path(path).as_posix() != path:
            raise VerificationError("immutable execution snapshot entry path is unsafe")
        if previous_path is not None and path <= previous_path:
            raise VerificationError("immutable execution snapshot entries are not uniquely sorted")
        previous_path = path
        mode = raw_entry.get("mode_octal")
        if mode not in ({"0555"} if kind == "directory" else {"0444", "0555"}):
            raise VerificationError("immutable execution snapshot entry mode differs")
        if raw_entry.get("uid") != EXECUTION_UID or raw_entry.get("gid") != EXECUTION_GID:
            raise VerificationError("immutable execution snapshot entry owner differs")
        if kind == "file":
            byte_count = raw_entry.get("byte_count")
            digest = raw_entry.get("sha256")
            if type(byte_count) is not int or byte_count < 0:
                raise VerificationError("immutable execution snapshot byte count is malformed")
            if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
                raise VerificationError("immutable execution snapshot digest is malformed")
        normalized_entries[path] = dict(raw_entry)
    expected_entries = [
        {"relative_path": path, **record} for path, record in sorted(actual_entries.items())
    ]
    if list(declared_entries) != expected_entries:
        raise VerificationError("immutable execution snapshot manifest differs from live contents")

    tools_prefix = "data/.p30-tools/"
    tool_paths = sorted(
        path[len(tools_prefix) :]
        for path in normalized_entries
        if path.startswith(tools_prefix) and "/" not in path[len(tools_prefix) :]
    )
    if tuple(tool_paths) != SNAPSHOT_TOOL_BASENAMES:
        raise VerificationError("immutable execution snapshot tool inventory differs")
    for basename in SNAPSHOT_TOOL_BASENAMES:
        if normalized_entries[f"{tools_prefix}{basename}"].get("kind") != "file":
            raise VerificationError(f"immutable execution snapshot tool is not a file: {basename}")
        if (
            normalized_entries[f"{tools_prefix}{basename}"].get("mode_octal")
            != SNAPSHOT_TOOL_MODES[basename]
        ):
            raise VerificationError(f"immutable execution snapshot tool mode differs: {basename}")

    authorities = manifest.get("source_authorities")
    if not isinstance(authorities, Mapping) or set(authorities) != {
        "repository",
        "nanogpt",
        "muon",
        "data",
        "helpers",
        "p26_authenticated_input",
    }:
        raise VerificationError("immutable execution snapshot source authorities differ")
    expected_git_authorities = {
        "repository": {
            "commit": SNAPSHOT_REPOSITORY_COMMIT,
            "tree": SNAPSHOT_REPOSITORY_TREE,
        },
        "nanogpt": {"commit": NANOGPT_COMMIT, "tree": NANOGPT_TREE},
        "muon": {"commit": MUON_COMMIT, "tree": MUON_TREE},
    }
    for label, expected_authority in expected_git_authorities.items():
        if authorities.get(label) != expected_authority:
            raise VerificationError(f"immutable execution snapshot {label} authority differs")
    expected_data_authority = {
        "original_manifest_relative_path": "materialized/p22_fineweb_manifest.json",
        "original_manifest_sha256": SNAPSHOT_FINEWEB_MANIFEST_SHA256,
        "original_manifest_byte_count": SNAPSHOT_FINEWEB_MANIFEST_BYTE_COUNT,
    }
    if authorities.get("data") != expected_data_authority:
        raise VerificationError("immutable execution snapshot data authority differs")
    fineweb_record = normalized_entries.get(SNAPSHOT_FINEWEB_MANIFEST)
    if not isinstance(fineweb_record, Mapping) or {
        "sha256": fineweb_record.get("sha256"),
        "byte_count": fineweb_record.get("byte_count"),
    } != {
        "sha256": SNAPSHOT_FINEWEB_MANIFEST_SHA256,
        "byte_count": SNAPSHOT_FINEWEB_MANIFEST_BYTE_COUNT,
    }:
        raise VerificationError("immutable execution snapshot FineWeb manifest differs")
    helpers = authorities.get("helpers")
    if not isinstance(helpers, Mapping) or set(helpers) != set(SNAPSHOT_TOOL_SOURCE_PATHS):
        raise VerificationError("immutable execution snapshot helper authorities differ")
    for basename in SNAPSHOT_TOOL_SOURCE_PATHS:
        helper_record = normalized_entries[f"{tools_prefix}{basename}"]
        expected_digest = SNAPSHOT_TOOL_SHA256[basename]
        if (
            helpers.get(basename) != expected_digest
            or helper_record.get("sha256") != expected_digest
        ):
            raise VerificationError(
                f"immutable execution snapshot helper binding differs: {basename}"
            )
        committed_path = SNAPSHOT_TOOL_SOURCE_PATHS[basename]
        committed_raw = run_git(
            ["-C", str(control), "show", f"{source_commit}:{committed_path}"]
        ).stdout
        if sha256_bytes(committed_raw) != expected_digest:
            raise VerificationError(
                f"immutable execution snapshot helper differs from source freeze: {basename}"
            )
    p26_basename = "p26-trace-off-a-failure.authenticated.json"
    p26_record = normalized_entries[f"{tools_prefix}{p26_basename}"]
    expected_p26_authority = {
        "sha256": SNAPSHOT_P26_FAILURE_SHA256,
        "byte_count": SNAPSHOT_P26_FAILURE_BYTE_COUNT,
    }
    if (
        authorities.get("p26_authenticated_input") != expected_p26_authority
        or {
            "sha256": p26_record.get("sha256"),
            "byte_count": p26_record.get("byte_count"),
        }
        != expected_p26_authority
    ):
        raise VerificationError("immutable execution snapshot P26 failure differs")

    git_records = {
        label: _verify_snapshot_git_checkout(
            execution_root,
            label,
            str(expected["commit"]),
            str(expected["tree"]),
            normalized_entries,
        )
        for label, expected in expected_git_authorities.items()
    }
    return {
        "schema_version": SNAPSHOT_SCHEMA,
        "root_path": str(execution_root),
        "manifest_path": str(execution_root / SNAPSHOT_MANIFEST_BASENAME),
        "manifest_sha256": manifest_record["sha256"],
        "manifest_byte_count": manifest_record["byte_count"],
        "entry_count": len(normalized_entries),
        "recursive_authority_verified": True,
        "recursive_content_verified": True,
        "recursive_symlink_hardlink_special_rejection_verified": True,
        "recursive_mount_acl_xattr_absence_verified": True,
        "git_checkouts_verified": True,
        "data_inventory_verified": True,
        "tool_inventory_and_source_bindings_verified": True,
        "git": git_records,
    }


def authenticate_executing_verifier(
    expected_sha256: str, declared_source: Path, *, transport_fd: int
) -> StableFile:
    if (os.geteuid(), os.getegid()) != (VERIFIER_UID, VERIFIER_GID):
        raise VerificationError(
            "P30 bootstrap verifier execution identity differs from uid/gid "
            f"{VERIFIER_UID}/{VERIFIER_GID}"
        )
    if REQUIRE_EXECUTING_PYTHON_ISOLATION and not (
        sys.flags.isolated == 1 and sys.flags.no_site == 1
    ):
        raise VerificationError("P30 bootstrap Python must run with -I -S")
    executing = Path(__file__).absolute()
    if declared_source.absolute() != executing:
        raise VerificationError("declared bootstrap verifier is not the executing source")
    if executing.is_symlink():
        raise VerificationError("executing P30 verifier is a symlink")
    _raw, record = stable_read_nofollow(executing, source_dir_fd=transport_fd)
    if record.sha256 != expected_sha256:
        raise VerificationError("executing P30 verifier bytes differ")
    if record.mode_octal != "0600":
        raise VerificationError("executing P30 bootstrap verifier mode is not 0600")
    if (record.uid, record.gid) != (TRANSPORT_UID, TRANSPORT_GID):
        raise VerificationError("executing P30 bootstrap verifier owner differs")
    if record.link_count != 1:
        raise VerificationError("executing P30 bootstrap verifier is hard linked")
    return record


def require_transport_file_authority(record: StableFile, *, label: str, expected_mode: str) -> None:
    if record.mode_octal != expected_mode:
        raise VerificationError(f"{label} mode is not {expected_mode}")
    if (record.uid, record.gid) != (TRANSPORT_UID, TRANSPORT_GID):
        raise VerificationError(f"{label} owner differs from the transport authority")
    if record.link_count != 1:
        raise VerificationError(f"{label} is hard linked")


def seal_private_bundle(path: Path, expected: StableFile) -> StableFile:
    """Make the authenticated copy read-only and bind its pre-Git identity."""

    if path.is_symlink():
        raise VerificationError("private authenticated bundle copy is a symlink")
    try:
        os.chmod(path, 0o400, follow_symlinks=False)
    except OSError as exc:
        raise VerificationError(f"cannot seal private bundle copy read-only: {exc}") from exc
    _raw, record = stable_read_nofollow(path)
    if record.mode_octal != "0400":
        raise VerificationError("private authenticated bundle copy mode is not 0400")
    if record.sha256 != expected.sha256 or record.byte_count != expected.byte_count:
        raise VerificationError("private bundle copy differs before Git verification")
    return record


def revalidate_private_bundle(path: Path, expected: StableFile) -> None:
    """Reauthenticate private bundle identity, bytes, size, and mode after Git."""

    if path.is_symlink():
        raise VerificationError("private authenticated bundle copy became a symlink")
    _raw, actual = stable_read_nofollow(path)
    if actual != expected:
        raise VerificationError("private authenticated bundle copy changed during Git use")
    if actual.mode_octal != "0400":
        raise VerificationError("private authenticated bundle copy is no longer read-only")


def _clean_git_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("DYLD_", "GIT_", "LD_", "PYTHON"))
    }
    environment.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
            "PATH": "/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONSAFEPATH": "1",
        }
    )
    return environment


def run_git(
    arguments: Sequence[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    safe_directory_arguments: list[str] = []
    if len(arguments) >= 2 and arguments[0] == "-C":
        repository = Path(arguments[1])
        if not repository.is_absolute():
            raise VerificationError("Git -C repository must be absolute")
        # Snapshot repositories are deliberately root-owned while this verifier
        # is launched as uid 1000. Trust only the exact path already selected by
        # the verifier, never '*' or an ambient safe.directory list.
        safe_directory_arguments = ["-c", f"safe.directory={repository}"]
    command = [
        str(GIT_BINARY),
        *safe_directory_arguments,
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "protocol.file.allow=always",
        *arguments,
    ]
    process = subprocess.run(
        command,
        cwd=cwd,
        env=_clean_git_environment(),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
    )
    if check and process.returncode != 0:
        stderr = process.stderr.decode("utf-8", errors="replace").strip()
        raise VerificationError(
            f"Git command failed ({process.returncode}): {' '.join(arguments)}: {stderr}"
        )
    return process


def require_sha1(value: str, label: str) -> str:
    if SHA1_RE.fullmatch(value) is None:
        raise VerificationError(f"{label} is not one lowercase SHA-1 object ID")
    return value


def require_sha256(value: str, label: str) -> str:
    if SHA256_RE.fullmatch(value) is None:
        raise VerificationError(f"{label} is not one lowercase SHA-256 digest")
    return value


def expected_refs(p30_commit: str) -> dict[str, str]:
    return {
        BRANCH_REF: p30_commit,
        P29_BRANCH_REF: P29_COMMIT,
        P29_TAG_REF: P29_TAG_OBJECT,
        P28_BRANCH_REF: P28_COMMIT,
        P28_TAG_REF: P28_TAG_OBJECT,
        P27_BRANCH_REF: P27_COMMIT,
        P27_TAG_REF: P27_TAG_OBJECT,
        P26_CHECKPOINT_REF: P26_CHECKPOINT_OBJECT,
        P26_SOURCE_REF: P26_SOURCE_OBJECT,
    }


def parse_v2_bundle_header(bundle: Path) -> tuple[dict[str, str], list[str]]:
    """Read only the textual header of the already-private bundle copy."""

    refs: dict[str, str] = {}
    prerequisites: list[str] = []
    with bundle.open("rb") as handle:
        signature = handle.readline()
        if signature != b"# v2 git bundle\n":
            raise VerificationError("control bundle must use the closed v2 bundle format")
        while True:
            line = handle.readline()
            if line == b"":
                raise VerificationError("bundle header ended before its required blank line")
            if line == b"\n":
                break
            if b"\x00" in line or not line.endswith(b"\n"):
                raise VerificationError("malformed bundle header line")
            try:
                text = line[:-1].decode("ascii")
            except UnicodeDecodeError as exc:
                raise VerificationError("bundle header is not ASCII") from exc
            if text.startswith("-"):
                prerequisite = text[1:].split(" ", 1)[0]
                require_sha1(prerequisite, "bundle prerequisite")
                prerequisites.append(prerequisite)
                continue
            parts = text.split(" ")
            if len(parts) != 2:
                raise VerificationError("bundle advertised-ref line is malformed")
            object_id, refname = parts
            require_sha1(object_id, "advertised object")
            if refname in refs:
                raise VerificationError(f"duplicate advertised ref: {refname}")
            refs[refname] = object_id
    return refs, prerequisites


def list_refs(repository: Path) -> dict[str, str]:
    output = run_git(
        ["-C", str(repository), "for-each-ref", "--format=%(objectname) %(refname)"],
    ).stdout
    refs: dict[str, str] = {}
    for raw_line in output.splitlines():
        try:
            object_raw, ref_raw = raw_line.split(b" ", 1)
            object_id = object_raw.decode("ascii")
            refname = ref_raw.decode("ascii")
        except (ValueError, UnicodeDecodeError) as exc:
            raise VerificationError("repository ref inventory is malformed") from exc
        require_sha1(object_id, "repository ref object")
        if refname in refs:
            raise VerificationError(f"duplicate repository ref: {refname}")
        refs[refname] = object_id
    return refs


def git_text(repository: Path, arguments: Sequence[str]) -> str:
    raw = run_git(["-C", str(repository), *arguments]).stdout
    try:
        return raw.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise VerificationError(f"non-ASCII Git output for {' '.join(arguments)}") from exc


def assert_attempt_absent(attempt_root: Path, boundary: str) -> None:
    if attempt_root.exists() or attempt_root.is_symlink():
        raise VerificationError(f"attempt root exists {boundary}: {attempt_root}")


def assert_execution_absent(execution_root: Path, boundary: str) -> None:
    if execution_root.exists() or execution_root.is_symlink():
        raise VerificationError(f"execution root exists {boundary}: {execution_root}")


def assert_authority_absent(authority: Path, boundary: str) -> None:
    if authority.exists() or authority.is_symlink():
        raise VerificationError(f"authority checkout exists {boundary}: {authority}")


def validate_optional_source_replay_authority(
    authority: Path, *, namespace_guard: NamespaceGuard
) -> dict[str, object]:
    """Validate an absent authority or the exact post-receipt canonical checkout."""

    try:
        by_name = os.stat(
            AUTHORITY_BASENAME,
            dir_fd=namespace_guard.transport_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return {"state": "absent"}
    except OSError as exc:
        raise VerificationError(f"cannot inspect source-replay authority: {exc}") from exc
    if not stat.S_ISDIR(by_name.st_mode):
        raise VerificationError("source-replay authority is not one nonsymlink directory")
    flags = _directory_open_flags()
    try:
        descriptor = os.open(
            AUTHORITY_BASENAME,
            flags,
            dir_fd=namespace_guard.transport_fd,
        )
    except OSError as exc:
        raise VerificationError(f"cannot open source-replay authority safely: {exc}") from exc
    try:
        _require_directory_authority(
            descriptor,
            authority,
            expected_uid=TRANSPORT_UID,
            expected_gid=TRANSPORT_GID,
            expected_mode=TRANSPORT_MODE,
        )
        identity = _directory_identity(descriptor, authority)
        if identity.device != namespace_guard.transport.device:
            raise VerificationError("source-replay authority crosses a device boundary")
        if identity.mount_id != namespace_guard.transport.mount_id:
            raise VerificationError("source-replay authority crosses a mount-ID boundary")
        acl = _validate_acl_absence({"authority": descriptor})
        if _is_exact_mountpoint(authority):
            raise VerificationError("source-replay authority is an unexpected mountpoint")
        assert_no_object_indirections(authority, bare=False)
        if git_text(authority, ["rev-parse", "--is-bare-repository"]) != "false":
            raise VerificationError("source-replay authority is not a worktree")
        if (
            run_git(["-C", str(authority), "symbolic-ref", "-q", "HEAD"], check=False).returncode
            == 0
        ):
            raise VerificationError("source-replay authority HEAD is not detached")
        if git_text(authority, ["rev-parse", "HEAD"]) != P26_SOURCE_COMMIT:
            raise VerificationError("source-replay authority HEAD differs")
        if git_text(authority, ["rev-parse", "HEAD^{tree}"]) != P26_SOURCE_TREE:
            raise VerificationError("source-replay authority tree differs")
        assert_worktree_exactly_clean(authority, "source-replay authority")
        run_git(["-C", str(authority), "fsck", "--full", "--strict"])
        if _directory_identity(descriptor, authority) != identity:
            raise VerificationError("source-replay authority changed during validation")
        return {
            "state": "exact_canonical_authority",
            "head": P26_SOURCE_COMMIT,
            "tree": P26_SOURCE_TREE,
            "identity": identity.__dict__,
            "acl_check": acl,
        }
    finally:
        os.close(descriptor)


def validate_runtime_control_artifact(
    control: Path,
    *,
    control_fd: int,
    relative_path: str,
    expected_native_raw: bytes,
) -> dict[str, object]:
    """Bind one runtime-control worktree file by Git and physical authority."""

    if relative_path not in {
        f"experiments/training/{basename}" for basename in RUNTIME_EVIDENCE_FILES.values()
    }:
        raise VerificationError("runtime-control relative path is not preregistered")
    stage_raw = run_git(
        ["-C", str(control), "ls-files", "--stage", "-z", "--", relative_path]
    ).stdout
    stage_records = [record for record in stage_raw.split(b"\0") if record]
    if len(stage_records) != 1:
        raise VerificationError("runtime-control artifact has no unique Git index record")
    try:
        prefix, path_raw = stage_records[0].split(b"\t", 1)
        mode_raw, object_raw, stage_raw_number = prefix.split(b" ", 2)
        tracked_path = path_raw.decode("utf-8")
        tracked_mode = mode_raw.decode("ascii")
        object_id = object_raw.decode("ascii")
        stage_number = stage_raw_number.decode("ascii")
    except (ValueError, UnicodeDecodeError) as exc:
        raise VerificationError("runtime-control Git index record is malformed") from exc
    if (
        tracked_path != relative_path
        or tracked_mode != RUNTIME_CONTROL_GIT_MODE
        or stage_number != "0"
        or SHA1_RE.fullmatch(object_id) is None
    ):
        raise VerificationError("runtime-control Git index authority differs")
    committed_object = git_text(control, ["rev-parse", f"HEAD:{relative_path}"])
    if committed_object != object_id:
        raise VerificationError("runtime-control index and commit blobs differ")

    source = control / relative_path
    control_before = _directory_identity(control_fd, control)
    control_stable = _stable_fields(os.fstat(control_fd))
    _require_directory_authority(
        control_fd,
        control,
        expected_uid=TRANSPORT_UID,
        expected_gid=TRANSPORT_GID,
        expected_mode=TRANSPORT_MODE,
    )
    try:
        control_by_name = os.stat(control, follow_symlinks=False)
    except OSError as exc:
        raise VerificationError(f"cannot restat bound control checkout: {exc}") from exc
    if not stat.S_ISDIR(control_by_name.st_mode) or (
        control_by_name.st_dev,
        control_by_name.st_ino,
    ) != (control_before.device, control_before.inode):
        raise VerificationError("bound control-checkout path differs from held descriptor")
    control_acl = _validate_acl_absence({"control_checkout": control_fd})
    parts = Path(relative_path).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise VerificationError("runtime-control relative path has an unsafe component")
    held_directories: list[
        tuple[
            str,
            Path,
            int,
            int,
            DirectoryIdentity,
            tuple[int, ...],
            dict[str, object],
        ]
    ] = []
    parent_fd = control_fd
    walked = control
    for index, component in enumerate(parts[:-1]):
        try:
            child_fd = os.open(component, _directory_open_flags(), dir_fd=parent_fd)
        except OSError as exc:
            for (
                _name,
                _path,
                held_fd,
                _parent,
                _identity,
                _stable,
                _acl,
            ) in reversed(held_directories):
                os.close(held_fd)
            raise VerificationError(
                "cannot open runtime-control intermediate directory with O_NOFOLLOW: "
                f"{component}: {exc}"
            ) from exc
        walked = walked / component
        try:
            _require_directory_authority(
                child_fd,
                walked,
                expected_uid=TRANSPORT_UID,
                expected_gid=TRANSPORT_GID,
                expected_mode=TRANSPORT_MODE,
            )
            identity = _directory_identity(child_fd, walked)
            stable = _stable_fields(os.fstat(child_fd))
            if (
                identity.device != control_before.device
                or identity.mount_id != control_before.mount_id
            ):
                raise VerificationError(
                    f"runtime-control intermediate directory crosses authority: {walked}"
                )
            if _is_exact_mountpoint(walked):
                raise VerificationError(
                    f"runtime-control intermediate directory is a mountpoint: {walked}"
                )
            acl = _validate_acl_absence({f"runtime_control_parent_{index}": child_fd})
        except Exception:
            os.close(child_fd)
            for (
                _name,
                _path,
                held_fd,
                _parent,
                _identity,
                _stable,
                _acl,
            ) in reversed(held_directories):
                os.close(held_fd)
            raise
        held_directories.append((component, walked, child_fd, parent_fd, identity, stable, acl))
        parent_fd = child_fd
    descriptor: int | None = None
    try:
        try:
            descriptor = os.open(
                source.name,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=parent_fd,
            )
        except OSError as exc:
            raise VerificationError(
                f"cannot open runtime-control artifact with O_NOFOLLOW: {relative_path}: {exc}"
            ) from exc
        before = os.fstat(descriptor)
        target_mount_id = _fd_mount_id(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or (before.st_uid, before.st_gid, stat.S_IMODE(before.st_mode))
            != (TRANSPORT_UID, TRANSPORT_GID, RUNTIME_CONTROL_PHYSICAL_MODE)
        ):
            raise VerificationError(f"runtime-control physical authority differs: {relative_path}")
        if before.st_dev != control_before.device or target_mount_id != control_before.mount_id:
            raise VerificationError(
                f"runtime-control artifact crosses device or mount authority: {relative_path}"
            )
        if _is_exact_mountpoint(source):
            raise VerificationError(
                f"runtime-control artifact is an unexpected mountpoint: {relative_path}"
            )
        listxattr = getattr(os, "listxattr", None)
        if listxattr is None:
            raise VerificationError("runtime-control ACL xattr inspection is unavailable")
        try:
            xattr_names = {
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in listxattr(descriptor)
            }
        except (OSError, UnicodeDecodeError) as exc:
            raise VerificationError(
                f"cannot inspect runtime-control ACL xattrs: {relative_path}: {exc}"
            ) from exc
        forbidden_acl_xattrs = sorted(xattr_names & ACL_XATTR_NAMES)
        if forbidden_acl_xattrs:
            raise VerificationError(
                "runtime-control artifact has a forbidden ACL xattr: "
                f"{relative_path}: {forbidden_acl_xattrs!r}"
            )
        digest = hashlib.sha256()
        pieces: list[bytes] = []
        byte_count = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            pieces.append(chunk)
            digest.update(chunk)
            byte_count += len(chunk)
        after = os.fstat(descriptor)
        by_name = os.stat(source.name, dir_fd=parent_fd, follow_symlinks=False)
        if _stable_fields(before) != _stable_fields(after) or _stable_fields(
            after
        ) != _stable_fields(by_name):
            raise VerificationError(
                f"runtime-control artifact changed during stable read: {relative_path}"
            )
        try:
            final_xattr_names = {
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in listxattr(descriptor)
            }
        except (OSError, UnicodeDecodeError) as exc:
            raise VerificationError(
                f"cannot revalidate runtime-control ACL xattrs: {relative_path}: {exc}"
            ) from exc
        if final_xattr_names != xattr_names:
            raise VerificationError(
                f"runtime-control xattr authority changed during stable read: {relative_path}"
            )
        if (
            after.st_dev != control_before.device
            or _fd_mount_id(descriptor) != target_mount_id
            or _is_exact_mountpoint(source)
        ):
            raise VerificationError(
                f"runtime-control artifact mount authority changed: {relative_path}"
            )
        raw = b"".join(pieces)
        if byte_count != before.st_size:
            raise VerificationError(
                f"runtime-control artifact size disagrees with bytes read: {relative_path}"
            )
        if raw != expected_native_raw:
            raise VerificationError(
                f"committed runtime-control artifact differs from native evidence: {relative_path}"
            )
        blob_raw = run_git(["-C", str(control), "cat-file", "blob", object_id]).stdout
        if blob_raw != raw:
            raise VerificationError(
                f"runtime-control worktree bytes differ from Git blob: {relative_path}"
            )
        if (
            run_git(["-C", str(control), "ls-files", "--stage", "-z", "--", relative_path]).stdout
            != stage_raw
            or git_text(control, ["rev-parse", f"HEAD:{relative_path}"]) != committed_object
        ):
            raise VerificationError(
                f"runtime-control Git authority changed during stable read: {relative_path}"
            )
        if (
            _stable_fields(os.fstat(control_fd)) != control_stable
            or _directory_identity(control_fd, control) != control_before
        ):
            raise VerificationError("bound control checkout changed during stable read")
        if _validate_acl_absence({"control_checkout": control_fd}) != control_acl:
            raise VerificationError("bound control-checkout ACL state changed")
        component_authority: list[dict[str, object]] = []
        for index, (
            component,
            component_path,
            child_fd,
            held_parent_fd,
            identity,
            stable,
            acl,
        ) in enumerate(held_directories):
            try:
                by_name = os.stat(component, dir_fd=held_parent_fd, follow_symlinks=False)
            except OSError as exc:
                raise VerificationError(
                    f"cannot restat runtime-control intermediate directory: {component_path}: {exc}"
                ) from exc
            current = os.fstat(child_fd)
            if (
                not stat.S_ISDIR(by_name.st_mode)
                or _stable_fields(current) != stable
                or _stable_fields(current) != _stable_fields(by_name)
                or _directory_identity(child_fd, component_path) != identity
            ):
                raise VerificationError(
                    f"runtime-control intermediate directory changed: {component_path}"
                )
            if (
                identity.device != control_before.device
                or identity.mount_id != control_before.mount_id
                or _is_exact_mountpoint(component_path)
            ):
                raise VerificationError(
                    f"runtime-control intermediate directory authority changed: {component_path}"
                )
            if _validate_acl_absence({f"runtime_control_parent_{index}": child_fd}) != acl:
                raise VerificationError(
                    f"runtime-control intermediate ACL state changed: {component_path}"
                )
            component_authority.append(
                {
                    "relative_path": str(component_path.relative_to(control)),
                    "device": identity.device,
                    "inode": identity.inode,
                    "mode_octal": identity.mode_octal,
                    "uid": identity.uid,
                    "gid": identity.gid,
                    "mount_id": identity.mount_id,
                    "same_device_and_mount_as_control": True,
                    "not_mountpoint": True,
                    "acl_check": acl,
                }
            )
        return {
            "path": str(source),
            "relative_path": relative_path,
            "git_mode": RUNTIME_CONTROL_GIT_MODE,
            "git_blob": object_id,
            "physical_mode_octal": format(RUNTIME_CONTROL_PHYSICAL_MODE, "04o"),
            "uid": before.st_uid,
            "gid": before.st_gid,
            "device": before.st_dev,
            "mount_id": target_mount_id,
            "inode": before.st_ino,
            "link_count": before.st_nlink,
            "byte_count": byte_count,
            "sha256": digest.hexdigest(),
            "verifier_umask_octal": format(VERIFIER_UMASK, "04o"),
            "regular_file": True,
            "nonsymlink_nofollow": True,
            "stable_nofollow_read": True,
            "acl_xattrs_absent": True,
            "all_xattr_names": sorted(xattr_names),
            "native_evidence_bytes_equal": True,
            "git_blob_bytes_equal": True,
            "git_index_and_head_stable": True,
            "parent_directory_stable": True,
            "componentwise_parent_nofollow": True,
            "same_device_and_mount_as_control": True,
            "not_mountpoint": True,
            "control_checkout_identity": control_before.__dict__,
            "control_checkout_acl_check": control_acl,
            "intermediate_directory_authority": component_authority,
        }
    finally:
        if descriptor is not None:
            os.close(descriptor)
        for (
            _name,
            _path,
            held_fd,
            _parent,
            _identity,
            _stable,
            _acl,
        ) in reversed(held_directories):
            os.close(held_fd)


def runtime_attempt_without_control_authority(
    record: Mapping[str, object],
) -> dict[str, object]:
    """Project a post-checkout attempt guard onto its immutable native evidence."""

    projected = dict(record)
    projected.pop("runtime_control_checkout_authority", None)
    return projected


def validate_runtime_attempt(
    attempt_root: Path,
    *,
    control: Path | None = None,
    control_fd: int | None = None,
) -> dict[str, object]:
    """Validate phase two's frozen runtime without authorizing localization."""

    if (control is None) != (control_fd is None):
        raise VerificationError(
            "runtime-review control path and bound descriptor must be supplied together"
        )

    if not attempt_root.is_dir() or attempt_root.is_symlink():
        raise VerificationError("runtime-review requires one existing nonsymlink attempt root")
    evidence = attempt_root / "evidence"
    if not evidence.is_dir() or evidence.is_symlink():
        raise VerificationError("runtime-review requires one existing nonsymlink evidence root")
    top_level = sorted(child.name for child in attempt_root.iterdir())
    if top_level != sorted(RUNTIME_ATTEMPT_TOP_LEVEL):
        raise VerificationError(
            "runtime-review attempt-root inventory differs: "
            f"expected {sorted(RUNTIME_ATTEMPT_TOP_LEVEL)!r}, got {top_level!r}"
        )
    evidence_inventory = sorted(child.name for child in evidence.iterdir())
    if evidence_inventory != sorted(RUNTIME_EVIDENCE_ALLOWLIST):
        raise VerificationError(
            "runtime-review evidence inventory differs: "
            f"expected {sorted(RUNTIME_EVIDENCE_ALLOWLIST)!r}, got {evidence_inventory!r}"
        )
    inventory_records: dict[str, dict[str, object]] = {}
    inventory_raw: dict[str, bytes] = {}
    inventory_paths = [
        *(attempt_root / name for name in RUNTIME_ATTEMPT_TOP_LEVEL if name != "evidence"),
        *(evidence / name for name in RUNTIME_EVIDENCE_ALLOWLIST),
    ]
    for path in inventory_paths:
        raw, file_record = stable_read_nofollow(path)
        if file_record.link_count != 1:
            raise VerificationError(f"runtime-review evidence is hard linked: {path.name}")
        expected_owner = (
            (EVIDENCE_UID, EVIDENCE_GID) if path.parent == evidence else (ATTEMPT_UID, ATTEMPT_GID)
        )
        if (file_record.uid, file_record.gid) != expected_owner:
            raise VerificationError(f"runtime-review retained file owner differs: {path.name}")
        if file_record.mode_octal != "0444":
            raise VerificationError(f"runtime-review retained file mode is not 0444: {path.name}")
        relative = str(path.relative_to(attempt_root))
        inventory_raw[relative] = raw
        inventory_records[relative] = {
            "sha256": file_record.sha256,
            "byte_count": file_record.byte_count,
            "mode_octal": file_record.mode_octal,
            "uid": file_record.uid,
            "gid": file_record.gid,
            "device": file_record.device,
            "inode": file_record.inode,
            "link_count": file_record.link_count,
        }
        if path.name in RUNTIME_SUCCESS_STATUS_FILES and raw != b"0\n":
            raise VerificationError(
                f"runtime-review prepare status is not canonical success: {path.name}"
            )
    records: dict[str, object] = {}
    control_authority: dict[str, object] = {}
    for label, basename in RUNTIME_EVIDENCE_FILES.items():
        path = evidence / basename
        relative = f"evidence/{basename}"
        raw = inventory_raw[relative]
        file_record = inventory_records[relative]
        if not raw:
            raise VerificationError(f"runtime-review {label} artifact is empty")
        if control is not None:
            assert control_fd is not None
            relative_path = f"experiments/training/{basename}"
            control_authority[label] = validate_runtime_control_artifact(
                control,
                control_fd=control_fd,
                relative_path=relative_path,
                expected_native_raw=raw,
            )
        records[label] = {
            "evidence_relative_path": f"evidence/{basename}",
            "committed_relative_path": f"experiments/training/{basename}",
            **file_record,
        }
    for basename in RUNTIME_FORBIDDEN_LOCALIZATION_FILES:
        path = evidence / basename
        if path.exists() or path.is_symlink():
            raise VerificationError(
                f"localization evidence exists before runtime receipt replay: {basename}"
            )
    for child in evidence.iterdir():
        if any(child.name.startswith(prefix) for prefix in RUNTIME_FORBIDDEN_PREFIXES):
            raise VerificationError(
                "run-phase evidence exists before runtime receipt replay: " + child.name
            )
    result: dict[str, object] = {
        "mode": "existing_frozen_runtime_before_localization",
        "attempt_root_existing": True,
        "evidence_root_existing": True,
        "runtime_artifacts": records,
        "exact_inventory_file_identities": inventory_records,
        "exact_attempt_top_level": RUNTIME_ATTEMPT_TOP_LEVEL,
        "exact_prepare_evidence_inventory": RUNTIME_EVIDENCE_ALLOWLIST,
        "canonical_success_status_files": sorted(RUNTIME_SUCCESS_STATUS_FILES),
        "all_prepare_status_files_exact_zero_newline": True,
        "forbidden_localization_files": RUNTIME_FORBIDDEN_LOCALIZATION_FILES,
        "forbidden_run_phase_prefixes": RUNTIME_FORBIDDEN_PREFIXES,
        "all_forbidden_localization_files_absent": True,
        "attempt_root_created_or_mutated_by_verifier": False,
    }
    if control is not None:
        if sorted(control_authority) != sorted(RUNTIME_EVIDENCE_FILES):
            raise VerificationError("runtime-control authority inventory differs")
        result["runtime_control_checkout_authority"] = {
            "expected_uid": TRANSPORT_UID,
            "expected_gid": TRANSPORT_GID,
            "expected_physical_mode_octal": format(RUNTIME_CONTROL_PHYSICAL_MODE, "04o"),
            "expected_git_mode": RUNTIME_CONTROL_GIT_MODE,
            "expected_link_count": 1,
            "verifier_umask_octal": format(VERIFIER_UMASK, "04o"),
            "all_regular_nonsymlink_single_link": True,
            "all_acl_xattrs_absent": True,
            "all_native_evidence_bytes_equal": True,
            "artifacts": control_authority,
        }
    return result


def bind_runtime_attempt_directories(namespace_guard: NamespaceGuard, attempt_root: Path) -> None:
    attempt_fd = namespace_guard.bind_child(
        label="attempt_root",
        parent_fd=namespace_guard.namespace_fd,
        path=attempt_root,
        expected_uid=ATTEMPT_UID,
        expected_gid=ATTEMPT_GID,
        expected_mode=ATTEMPT_MODE,
        expected_mount_id=namespace_guard.namespace.mount_id,
        expected_device=namespace_guard.namespace.device,
    )
    namespace_guard.bind_child(
        label="attempt_evidence",
        parent_fd=attempt_fd,
        path=attempt_root / "evidence",
        expected_uid=EVIDENCE_UID,
        expected_gid=EVIDENCE_GID,
        expected_mode=EVIDENCE_MODE,
        expected_mount_id=namespace_guard.namespace.mount_id,
        expected_device=namespace_guard.namespace.device,
    )


def bind_execution_snapshot(namespace_guard: NamespaceGuard, execution_root: Path) -> int:
    return namespace_guard.bind_child(
        label="execution_root",
        parent_fd=namespace_guard.namespace_fd,
        path=execution_root,
        expected_uid=EXECUTION_UID,
        expected_gid=EXECUTION_GID,
        expected_mode=EXECUTION_MODE,
        expected_mount_id=namespace_guard.namespace.mount_id,
        expected_device=namespace_guard.namespace.device,
    )


def bind_transport_repository(namespace_guard: NamespaceGuard, *, label: str, path: Path) -> None:
    namespace_guard.bind_child(
        label=label,
        parent_fd=namespace_guard.transport_fd,
        path=path,
        expected_uid=TRANSPORT_UID,
        expected_gid=TRANSPORT_GID,
        expected_mode=TRANSPORT_MODE,
        expected_mount_id=namespace_guard.transport.mount_id,
        expected_device=namespace_guard.transport.device,
    )


def assert_no_object_indirections(repository: Path, *, bare: bool) -> None:
    git_dir = repository if bare else repository / ".git"
    if not bare:
        try:
            dot_git = os.lstat(git_dir)
        except OSError as exc:
            raise VerificationError(f"control checkout .git is unavailable: {exc}") from exc
        if stat.S_ISLNK(dot_git.st_mode) or not stat.S_ISDIR(dot_git.st_mode):
            raise VerificationError("control checkout .git is not one nonsymlink directory")
    forbidden_files = [
        git_dir / "shallow",
        git_dir / "objects/info/alternates",
        git_dir / "objects/info/http-alternates",
        git_dir / "info/grafts",
        git_dir / "commondir",
    ]
    for candidate in forbidden_files:
        if candidate.exists() or candidate.is_symlink():
            raise VerificationError(f"forbidden Git object indirection exists: {candidate}")
    shallow = git_text(repository, ["rev-parse", "--is-shallow-repository"])
    if shallow != "false":
        raise VerificationError("control repository is shallow")
    replace = run_git(
        ["-C", str(repository), "for-each-ref", "--format=%(refname)", "refs/replace"],
    ).stdout
    if replace.strip():
        raise VerificationError("control repository contains replacement refs")
    local_config = run_git(
        [
            "-C",
            str(repository),
            "config",
            "--no-includes",
            "--local",
            "--name-only",
            "--null",
            "--list",
        ]
    )
    try:
        config_keys = [
            item.decode("utf-8").lower() for item in local_config.stdout.split(b"\0") if item
        ]
    except UnicodeDecodeError as exc:
        raise VerificationError("repository configuration keys are not UTF-8") from exc
    allowed_config_keys = {
        "core.autocrlf",
        "core.bare",
        "core.filemode",
        "core.ignorecase",
        "core.logallrefupdates",
        "core.precomposeunicode",
        "core.repositoryformatversion",
    }
    unexpected_config_keys = sorted(set(config_keys) - allowed_config_keys)
    if unexpected_config_keys:
        raise VerificationError(
            "repository contains noncanonical or external-command-capable local "
            f"configuration: {unexpected_config_keys!r}"
        )


def assert_worktree_exactly_clean(repository: Path, label: str) -> None:
    if run_git(
        ["-C", str(repository), "status", "--porcelain=v1", "-z", "--untracked-files=all"]
    ).stdout:
        raise VerificationError(f"{label} is not exactly clean")
    if run_git(
        ["-C", str(repository), "ls-files", "--others", "--ignored", "--exclude-standard", "-z"]
    ).stdout:
        raise VerificationError(f"{label} contains ignored untracked files")


def assert_object_contract(repository: Path, refs: Mapping[str, str]) -> None:
    expected_types = {
        BRANCH_REF: "commit",
        P29_BRANCH_REF: "commit",
        P29_TAG_REF: "tag",
        P28_BRANCH_REF: "commit",
        P28_TAG_REF: "tag",
        P27_BRANCH_REF: "commit",
        P27_TAG_REF: "tag",
        P26_CHECKPOINT_REF: "tag",
        P26_SOURCE_REF: "tag",
    }
    for refname, expected_type in expected_types.items():
        actual_type = git_text(repository, ["cat-file", "-t", refs[refname]])
        if actual_type != expected_type:
            raise VerificationError(
                f"{refname} has type {actual_type!r}, expected {expected_type!r}"
            )
    peeled = {
        P29_TAG_REF: P29_COMMIT,
        P28_TAG_REF: P28_COMMIT,
        P27_TAG_REF: P27_COMMIT,
        P26_CHECKPOINT_REF: P26_CHECKPOINT_COMMIT,
        P26_SOURCE_REF: P26_SOURCE_COMMIT,
    }
    for refname, expected_commit in peeled.items():
        actual = git_text(repository, ["rev-parse", f"{refname}^{{}}"])
        if actual != expected_commit:
            raise VerificationError(f"{refname} peels to {actual}, expected {expected_commit}")
        if git_text(repository, ["cat-file", "-t", actual]) != "commit":
            raise VerificationError(f"peeled target for {refname} is not a commit")
    for refname, expected_tree in (
        (P29_BRANCH_REF, P29_TREE),
        (P28_BRANCH_REF, P28_TREE),
        (P27_BRANCH_REF, P27_TREE),
    ):
        actual_tree = git_text(repository, ["rev-parse", f"{refname}^{{tree}}"])
        if actual_tree != expected_tree:
            raise VerificationError(f"{refname} tree differs")


def assert_phase_history(
    repository: Path,
    *,
    phase: str,
    p30_commit: str,
    p30_tree: str,
    source_commit: str | None,
    source_tree: str | None,
) -> dict[str, object]:
    if git_text(repository, ["rev-parse", f"{p30_commit}^{{tree}}"]) != p30_tree:
        raise VerificationError("P30 commit/tree binding differs")
    parents = git_text(repository, ["rev-list", "--parents", "-n", "1", p30_commit]).split()
    if phase == "source":
        if parents != [p30_commit, P29_COMMIT]:
            raise VerificationError("P30 source freeze is not one direct child of terminal P29")
        return {
            "source_commit": p30_commit,
            "source_tree": p30_tree,
            "runtime_review_direct_child": False,
            "runtime_review_exact_delta": [],
        }
    if source_commit is None or source_tree is None:
        raise VerificationError("runtime-review phase requires source commit and tree")
    require_sha1(source_commit, "source commit")
    require_sha1(source_tree, "source tree")
    if parents != [p30_commit, source_commit]:
        raise VerificationError("P30 runtime review is not one direct child of source freeze")
    if git_text(repository, ["rev-parse", f"{source_commit}^{{tree}}"]) != source_tree:
        raise VerificationError("P30 source commit/tree binding differs")
    source_parents = git_text(
        repository, ["rev-list", "--parents", "-n", "1", source_commit]
    ).split()
    if source_parents != [source_commit, P29_COMMIT]:
        raise VerificationError("P30 source freeze history differs")
    delta_raw = run_git(
        [
            "-C",
            str(repository),
            "diff",
            "--name-status",
            source_commit,
            p30_commit,
            "--",
        ]
    ).stdout
    try:
        delta = delta_raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise VerificationError("runtime-review delta is not UTF-8") from exc
    if delta != RUNTIME_DELTA:
        raise VerificationError(
            f"runtime-review delta differs: expected {RUNTIME_DELTA!r}, got {delta!r}"
        )
    return {
        "source_commit": source_commit,
        "source_tree": source_tree,
        "runtime_review_direct_child": True,
        "runtime_review_exact_delta": RUNTIME_DELTA,
    }


def initialize_empty_bare(closure: Path, *, parent_fd: int | None = None) -> None:
    if closure.exists() or closure.is_symlink():
        raise VerificationError(f"phase closure must be absent: {closure}")
    if parent_fd is not None:
        try:
            os.mkdir(closure.name, 0o700, dir_fd=parent_fd)
            os.fsync(parent_fd)
        except OSError as exc:
            raise VerificationError(f"cannot create closure through guarded parent: {exc}") from exc
    run_git(["init", "--bare", str(closure)])
    if list_refs(closure):
        raise VerificationError("new bare closure unexpectedly contains refs")
    count = git_text(closure, ["count-objects", "-v"])
    parsed = dict(line.split(": ", 1) for line in count.splitlines() if ": " in line)
    if parsed.get("count") != "0" or parsed.get("packs") != "0":
        raise VerificationError("new bare closure was not object-empty")
    assert_no_object_indirections(closure, bare=True)


def verify_and_fill_closure(
    closure: Path,
    private_bundle: Path,
    refs: Mapping[str, str],
) -> None:
    run_git(["-C", str(closure), "bundle", "verify", str(private_bundle)])
    for refname in refs:
        run_git(
            [
                "-C",
                str(closure),
                "fetch",
                "--no-tags",
                "--no-write-fetch-head",
                str(private_bundle),
                f"+{refname}:{refname}",
            ]
        )
    if list_refs(closure) != dict(refs):
        raise VerificationError("fresh closure ref inventory differs from advertised refs")
    assert_no_object_indirections(closure, bare=True)
    assert_object_contract(closure, refs)
    run_git(["-C", str(closure), "fsck", "--full", "--strict", "--no-dangling"])


def validate_existing_closure(closure: Path, refs: Mapping[str, str]) -> None:
    if not closure.is_dir() or closure.is_symlink():
        raise VerificationError("reviewed phase closure is absent or a symlink")
    if git_text(closure, ["rev-parse", "--is-bare-repository"]) != "true":
        raise VerificationError("reviewed phase closure is not bare")
    if list_refs(closure) != dict(refs):
        raise VerificationError("reviewed phase closure ref inventory differs")
    assert_no_object_indirections(closure, bare=True)
    assert_object_contract(closure, refs)
    run_git(["-C", str(closure), "fsck", "--full", "--strict", "--no-dangling"])


def _fetch_closed_refs(source: Path, target: Path, refs: Mapping[str, str]) -> None:
    for refname in refs:
        run_git(
            [
                "-C",
                str(target),
                "fetch",
                "--no-tags",
                "--no-write-fetch-head",
                str(source),
                f"+{refname}:{refname}",
            ]
        )


def create_source_control(
    control: Path,
    closure: Path,
    refs: Mapping[str, str],
    p30_commit: str,
    *,
    namespace_guard: NamespaceGuard,
) -> None:
    if control.exists() or control.is_symlink():
        raise VerificationError("source control checkout must be absent")
    # Keep HEAD on a distinct unborn name while the closed P30 branch ref is
    # fetched. Git refuses to fetch into a checked-out branch, even before its
    # first checkout.
    try:
        os.mkdir(control.name, 0o700, dir_fd=namespace_guard.transport_fd)
        os.fsync(namespace_guard.transport_fd)
    except OSError as exc:
        raise VerificationError(f"cannot create control through guarded transport: {exc}") from exc
    run_git(["init", "--initial-branch=p30-verifier-unborn", str(control)])
    bind_transport_repository(namespace_guard, label="control_checkout", path=control)
    run_git(["-C", str(control), "config", "core.autocrlf", "false"])
    run_git(["-C", str(control), "config", "core.filemode", "true"])
    _fetch_closed_refs(closure, control, refs)
    run_git(["-C", str(control), "checkout", "--detach", p30_commit])


def advance_runtime_control(
    control: Path,
    closure: Path,
    refs: Mapping[str, str],
    *,
    source_commit: str,
    source_tree: str,
    runtime_commit: str,
    expected_verifier_sha256: str,
) -> None:
    validate_control_checkout(
        control,
        expected_head=source_commit,
        expected_tree=source_tree,
        refs=expected_refs(source_commit),
        run_reconstruction=True,
        expected_verifier_sha256=expected_verifier_sha256,
    )
    _fetch_closed_refs(closure, control, refs)
    run_git(["-C", str(control), "checkout", "--detach", runtime_commit])


def run_frozen_reconstruction(
    control: Path,
    *,
    label: str,
    relative_path: str,
    expected_source_sha256: str | None,
    expected_schema: str,
    expected_canonical_sha256: str | None,
    expected_check_count: int,
    canonical_artifact_path: str | None = None,
) -> dict[str, object]:
    reconstructor = control / relative_path
    if not reconstructor.is_file() or reconstructor.is_symlink():
        raise VerificationError(f"frozen {label} reconstructor is absent or a symlink")
    _reconstructor_raw, reconstructor_record = stable_read_nofollow(reconstructor)
    if expected_source_sha256 is not None and reconstructor_record.sha256 != expected_source_sha256:
        raise VerificationError(f"frozen {label} reconstructor bytes differ")
    resolved_source_sha256 = reconstructor_record.sha256
    if expected_canonical_sha256 is None:
        if canonical_artifact_path is None:
            raise VerificationError(f"frozen {label} has no canonical artifact binding")
        canonical_artifact = control / canonical_artifact_path
        if not canonical_artifact.is_file() or canonical_artifact.is_symlink():
            raise VerificationError(f"frozen {label} canonical artifact is absent")
        _canonical_raw, canonical_record = stable_read_nofollow(canonical_artifact)
        resolved_canonical_sha256 = canonical_record.sha256
    else:
        if canonical_artifact_path is not None:
            raise VerificationError(f"frozen {label} has ambiguous canonical artifact binding")
        resolved_canonical_sha256 = expected_canonical_sha256
    with tempfile.TemporaryDirectory(prefix="p30-python-shim-") as shim_directory:
        python_shim = Path(shim_directory) / "python3"
        python_shim.write_text(
            '#!/bin/sh\nexec "$P30_RECONSTRUCTOR_PYTHON" -I -S "$@"\n',
            encoding="ascii",
        )
        python_shim.chmod(0o700)
        environment = _clean_git_environment()
        environment["PATH"] = f"{shim_directory}:/usr/bin:/bin"
        environment["P30_RECONSTRUCTOR_PYTHON"] = str(Path(sys.executable).resolve())
        process = subprocess.run(
            [sys.executable, "-I", "-S", str(reconstructor)],
            cwd=control,
            env=environment,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
        )
    if process.returncode != 0:
        stderr = process.stderr.decode("utf-8", errors="replace").strip()
        stdout = process.stdout.decode("utf-8", errors="replace").strip()
        detail = stderr or stdout or "no diagnostic output"
        raise VerificationError(f"{label} reconstruction failed: {detail}")
    payload = strict_json_loads(process.stdout)
    if not isinstance(payload, Mapping):
        raise VerificationError(f"{label} reconstruction output is not an object")
    checks = payload.get("checks")
    if not isinstance(checks, Mapping) or len(checks) != expected_check_count:
        raise VerificationError(
            f"{label} reconstruction does not contain exactly {expected_check_count} checks"
        )
    if any(value is not True for value in checks.values()):
        raise VerificationError(
            f"{label} reconstruction is not {expected_check_count}/{expected_check_count} true"
        )
    if payload.get("internally_consistent") is not True:
        raise VerificationError(f"{label} reconstruction is not internally consistent")
    if payload.get("schema_version") != expected_schema:
        raise VerificationError(f"{label} reconstruction schema differs")
    if payload.get("canonical_sha256") != resolved_canonical_sha256:
        raise VerificationError(f"{label} canonical artifact digest differs")
    return {
        "label": label,
        "reconstructor_path": relative_path,
        "reconstructor_sha256": resolved_source_sha256,
        "schema_version": expected_schema,
        "canonical_sha256": resolved_canonical_sha256,
        "check_count": expected_check_count,
        "true_check_count": expected_check_count,
        "internally_consistent": True,
    }


def _create_private_reconstruction_checkout(
    source: Path,
    destination: Path,
    *,
    expected_head: str,
    expected_tree: str,
) -> None:
    """Materialize the authenticated commit in a process-private checkout."""

    run_git(["init", "--initial-branch=p30-private-unborn", str(destination)])
    closed_refs = expected_refs(expected_head)
    _fetch_closed_refs(source, destination, closed_refs)
    run_git(["-C", str(destination), "checkout", "--detach", expected_head])
    assert_no_object_indirections(destination, bare=False)
    if git_text(destination, ["rev-parse", "HEAD"]) != expected_head:
        raise VerificationError("private reconstruction checkout HEAD differs")
    if git_text(destination, ["rev-parse", "HEAD^{tree}"]) != expected_tree:
        raise VerificationError("private reconstruction checkout tree differs")
    if list_refs(destination) != closed_refs:
        raise VerificationError("private reconstruction checkout ref inventory differs")
    assert_object_contract(destination, closed_refs)
    assert_worktree_exactly_clean(destination, "private reconstruction checkout")
    run_git(["-C", str(destination), "fsck", "--full", "--strict", "--no-dangling"])


def run_required_reconstructions(
    control: Path,
    *,
    expected_head: str,
    expected_tree: str,
) -> dict[str, object]:
    # The reviewed transport checkout is same-UID writable.  Fetch its exact
    # authenticated commit into a fresh private object database and execute
    # only files from that checkout, so a path swap after review cannot change
    # the Python bytes or artifacts consumed by the reconstructors.
    with tempfile.TemporaryDirectory(prefix="p30-reconstruction-checkout-") as temporary:
        private_control = Path(temporary) / "control"
        _create_private_reconstruction_checkout(
            control,
            private_control,
            expected_head=expected_head,
            expected_tree=expected_tree,
        )
        records = {
            "p30_contract": run_frozen_reconstruction(
                private_control,
                label="P30 contract",
                relative_path=P30_RECONSTRUCTOR,
                expected_source_sha256=None,
                expected_schema=P30_RECONSTRUCTION_SCHEMA,
                expected_canonical_sha256=None,
                expected_check_count=15,
                canonical_artifact_path=P30_CONTRACT,
            ),
            "p29_terminal_outcome": run_frozen_reconstruction(
                private_control,
                label="P29 terminal outcome",
                relative_path=TERMINAL_P29_OUTCOME_RECONSTRUCTOR,
                expected_source_sha256=TERMINAL_P29_OUTCOME_RECONSTRUCTOR_SHA256,
                expected_schema=TERMINAL_P29_OUTCOME_RECONSTRUCTION_SCHEMA,
                expected_canonical_sha256=TERMINAL_P29_OUTCOME_CANONICAL_SHA256,
                expected_check_count=11,
            ),
            "p29_contract": run_frozen_reconstruction(
                private_control,
                label="P29 contract",
                relative_path=TERMINAL_P29_CONTRACT_RECONSTRUCTOR,
                expected_source_sha256=TERMINAL_P29_CONTRACT_RECONSTRUCTOR_SHA256,
                expected_schema=TERMINAL_P29_CONTRACT_RECONSTRUCTION_SCHEMA,
                expected_canonical_sha256=TERMINAL_P29_CONTRACT_CANONICAL_SHA256,
                expected_check_count=15,
            ),
            "p28_terminal_outcome": run_frozen_reconstruction(
                private_control,
                label="P28 terminal outcome",
                relative_path=P28_OUTCOME_RECONSTRUCTOR,
                expected_source_sha256=P28_OUTCOME_RECONSTRUCTOR_SHA256,
                expected_schema=P28_OUTCOME_RECONSTRUCTION_SCHEMA,
                expected_canonical_sha256=P28_OUTCOME_CANONICAL_SHA256,
                expected_check_count=10,
            ),
            "p28_contract": run_frozen_reconstruction(
                private_control,
                label="P28 contract",
                relative_path=P28_CONTRACT_RECONSTRUCTOR,
                expected_source_sha256=P28_CONTRACT_RECONSTRUCTOR_SHA256,
                expected_schema=P28_CONTRACT_RECONSTRUCTION_SCHEMA,
                expected_canonical_sha256=P28_CONTRACT_CANONICAL_SHA256,
                expected_check_count=12,
            ),
            "p27_contract": run_frozen_reconstruction(
                private_control,
                label="P27 contract",
                relative_path=P27_RECONSTRUCTOR,
                expected_source_sha256=P27_RECONSTRUCTOR_SHA256,
                expected_schema=P27_RECONSTRUCTION_SCHEMA,
                expected_canonical_sha256=P27_CONTRACT_SHA256,
                expected_check_count=23,
            ),
        }
    return {
        "status": "all_required_reconstructions_passed",
        "required_reconstruction_count": 6,
        "receipt_publication_blocked_unless_all_pass": True,
        **records,
    }


def validate_control_checkout(
    control: Path,
    *,
    expected_head: str,
    expected_tree: str,
    refs: Mapping[str, str],
    run_reconstruction: bool,
    expected_verifier_sha256: str,
) -> dict[str, object] | None:
    if not control.is_dir() or control.is_symlink():
        raise VerificationError("control checkout is absent or a symlink")
    # Reject local worktree redirection, fsmonitor commands, filters, and every
    # other noncanonical config key before the first worktree-sensitive Git
    # operation.  run_git independently forces fsmonitor off as defense in depth.
    assert_no_object_indirections(control, bare=False)
    if git_text(control, ["rev-parse", "--is-bare-repository"]) != "false":
        raise VerificationError("control checkout is bare")
    if run_git(["-C", str(control), "symbolic-ref", "-q", "HEAD"], check=False).returncode == 0:
        raise VerificationError("control checkout HEAD is not detached")
    if git_text(control, ["rev-parse", "HEAD"]) != expected_head:
        raise VerificationError("control checkout HEAD differs")
    if git_text(control, ["rev-parse", "HEAD^{tree}"]) != expected_tree:
        raise VerificationError("control checkout tree differs")
    assert_worktree_exactly_clean(control, "control checkout")
    if list_refs(control) != dict(refs):
        raise VerificationError("control checkout ref inventory differs")
    assert_object_contract(control, refs)
    run_git(["-C", str(control), "fsck", "--full", "--strict", "--no-dangling"])
    checked_in_verifier = control / VERIFIER_SOURCE
    if not checked_in_verifier.is_file() or checked_in_verifier.is_symlink():
        raise VerificationError("checked-in P30 verifier is absent or a symlink")
    _checked_in_raw, checked_in_record = stable_read_nofollow(checked_in_verifier)
    if checked_in_record.sha256 != expected_verifier_sha256:
        raise VerificationError("checked-in P30 verifier bytes differ from executing verifier")
    if run_reconstruction:
        return run_required_reconstructions(
            control,
            expected_head=expected_head,
            expected_tree=expected_tree,
        )
    return None


def validate_reviewed_orchestrator_source(
    control: Path,
    *,
    expected_sha256: str,
) -> dict[str, object]:
    """Bind the verifier-created mode-0600 launcher consumed by the root seal."""

    require_sha256(expected_sha256, "expected orchestrator digest")
    contract_raw, _contract_record = stable_read_nofollow(control / P30_CONTRACT)
    contract = strict_json_loads(contract_raw)
    if not isinstance(contract, Mapping):
        raise VerificationError("P30 contract is not a JSON object")
    host = contract.get("host_orchestration")
    if not isinstance(host, Mapping) or host.get("source_sha256") != expected_sha256:
        raise VerificationError("expected orchestrator digest differs from the P30 contract")

    stage_raw = run_git(
        ["-C", str(control), "ls-files", "--stage", "-z", "--", ORCHESTRATOR_SOURCE]
    ).stdout
    records = [record for record in stage_raw.split(b"\0") if record]
    if len(records) != 1:
        raise VerificationError("reviewed orchestrator has no unique Git index record")
    try:
        prefix, path_raw = records[0].split(b"\t", 1)
        mode_raw, object_raw, stage_number_raw = prefix.split(b" ", 2)
        tracked_path = path_raw.decode("utf-8")
        tracked_mode = mode_raw.decode("ascii")
        object_id = object_raw.decode("ascii")
        stage_number = stage_number_raw.decode("ascii")
    except (ValueError, UnicodeDecodeError) as exc:
        raise VerificationError("reviewed orchestrator Git index record is malformed") from exc
    if (
        tracked_path != ORCHESTRATOR_SOURCE
        or tracked_mode != ORCHESTRATOR_GIT_MODE
        or stage_number != "0"
        or SHA1_RE.fullmatch(object_id) is None
    ):
        raise VerificationError("reviewed orchestrator Git index authority differs")
    committed_object = git_text(control, ["rev-parse", f"HEAD:{ORCHESTRATOR_SOURCE}"])
    if committed_object != object_id:
        raise VerificationError("reviewed orchestrator index and commit blobs differ")

    source = control / ORCHESTRATOR_SOURCE
    parent_fd = os.open(source.parent, _directory_open_flags())
    descriptor: int | None = None
    try:
        descriptor = os.open(
            source.name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or (before.st_uid, before.st_gid, stat.S_IMODE(before.st_mode))
            != (TRANSPORT_UID, TRANSPORT_GID, ORCHESTRATOR_PHYSICAL_MODE)
        ):
            raise VerificationError("reviewed orchestrator physical authority differs")
        try:
            names = {
                value.decode() if isinstance(value, bytes) else value
                for value in os.listxattr(descriptor)
            }
        except OSError as exc:
            raise VerificationError("cannot inspect reviewed orchestrator ACL xattrs") from exc
        if names & ACL_XATTR_NAMES:
            raise VerificationError("reviewed orchestrator has a forbidden ACL xattr")
        digest = hashlib.sha256()
        byte_count = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
        after = os.fstat(descriptor)
        by_name = os.stat(source.name, dir_fd=parent_fd, follow_symlinks=False)
        if _stable_fields(before) != _stable_fields(after) or _stable_fields(
            after
        ) != _stable_fields(by_name):
            raise VerificationError("reviewed orchestrator changed during stable read")
        if digest.hexdigest() != expected_sha256:
            raise VerificationError("reviewed orchestrator SHA-256 differs")
        blob_raw = run_git(["-C", str(control), "cat-file", "blob", object_id]).stdout
        if hashlib.sha256(blob_raw).hexdigest() != expected_sha256 or len(blob_raw) != byte_count:
            raise VerificationError("reviewed orchestrator worktree bytes differ from Git blob")
        return {
            "path": str(source),
            "relative_path": ORCHESTRATOR_SOURCE,
            "git_mode": ORCHESTRATOR_GIT_MODE,
            "git_blob": object_id,
            "physical_mode_octal": format(ORCHESTRATOR_PHYSICAL_MODE, "04o"),
            "uid": before.st_uid,
            "gid": before.st_gid,
            "device": before.st_dev,
            "inode": before.st_ino,
            "link_count": before.st_nlink,
            "byte_count": byte_count,
            "sha256": expected_sha256,
            "verifier_umask_octal": format(VERIFIER_UMASK, "04o"),
            "stable_nofollow_read": True,
            "acl_xattrs_absent": True,
            "contract_digest_binding": True,
        }
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _exact_absolute_argument(raw: str, label: str) -> Path:
    if raw != os.path.normpath(raw):
        raise VerificationError(f"{label} path is not lexically canonical")
    path = Path(raw)
    if not path.is_absolute():
        raise VerificationError(f"{label} path is not absolute")
    return path


def validate_layout(args: argparse.Namespace) -> dict[str, Path]:
    phase_layout = PHASE_LAYOUT[args.phase]
    paths = {
        "transport": _exact_absolute_argument(args.transport_root, "transport"),
        "bundle": _exact_absolute_argument(args.bundle, "bundle"),
        "closure": _exact_absolute_argument(args.closure_repository, "closure"),
        "control": _exact_absolute_argument(args.control_checkout, "control"),
        "attempt": _exact_absolute_argument(args.attempt_root, "attempt"),
        "execution": _exact_absolute_argument(args.execution_root, "execution"),
        "bootstrap": _exact_absolute_argument(args.bootstrap_verifier, "bootstrap"),
        "authority": (NAMESPACE_ROOT / TRANSPORT_BASENAME / AUTHORITY_BASENAME),
        "receipt": _exact_absolute_argument(
            args.receipt_output or args.verify_existing_receipt, "receipt"
        ),
    }
    transport = paths["transport"]
    expected_transport = NAMESPACE_ROOT / TRANSPORT_BASENAME
    if transport != expected_transport:
        raise VerificationError(
            f"transport path differs: expected {expected_transport}, got {transport}"
        )
    try:
        if NAMESPACE_ROOT.resolve(strict=True) != NAMESPACE_ROOT:
            raise VerificationError("P30 namespace lexical and resolved paths differ")
        if transport.resolve(strict=True) != transport:
            raise VerificationError("P30 transport lexical and resolved paths differ")
    except OSError as exc:
        raise VerificationError(f"cannot resolve P30 permission layout: {exc}") from exc
    expected = {
        "bundle": transport / phase_layout["bundle"],
        "closure": transport / phase_layout["closure"],
        "receipt": transport / phase_layout["receipt"],
        "control": transport / CONTROL_BASENAME,
        "attempt": NAMESPACE_ROOT / ATTEMPT_BASENAME,
        "execution": NAMESPACE_ROOT / EXECUTION_BASENAME,
        "bootstrap": transport / BOOTSTRAP_BASENAME,
        "authority": transport / AUTHORITY_BASENAME,
    }
    for label, expected_path in expected.items():
        if paths[label] != expected_path:
            raise VerificationError(
                f"{label} path differs: expected {expected_path}, got {paths[label]}"
            )
    if not _path_is_within(paths["bundle"], transport):
        raise VerificationError("bundle is outside the transport root")
    if paths["receipt"] == paths["bundle"]:
        raise VerificationError("receipt aliases its input bundle")
    if _path_is_within(paths["receipt"], paths["control"]):
        raise VerificationError("receipt would be committed inside the control checkout")
    return paths


def build_receipt(
    *,
    phase: str,
    paths: Mapping[str, Path],
    bundle_file: StableFile,
    expected_bundle_sha256: str,
    expected_bundle_byte_count: int,
    p30_commit: str,
    p30_tree: str,
    refs: Mapping[str, str],
    history: Mapping[str, object],
    reconstruction: Mapping[str, object],
    attempt_guard: Mapping[str, object],
    snapshot_guard: Mapping[str, object],
    verifier_sha256: str,
    reviewed_orchestrator_source: Mapping[str, object],
    namespace_guard: NamespaceGuard,
) -> dict[str, object]:
    claim_boundary = (
        "Offline Git object/ref/control-checkout provenance only. The source receipt "
        "records no attempt-root creation, container, GPU, CUDA, mapping, gradient, "
        "candidate, optimizer update, or training observation."
        if phase == "source"
        else "Offline Git object/ref/control-checkout provenance plus identity of the "
        "already prepared frozen runtime. The runtime-review receipt records no "
        "localization invocation, mapping observation, gradient, candidate, optimizer "
        "update, or training observation."
    )
    ref_records = [
        {
            "refname": refname,
            "object_id": object_id,
            "object_type": (
                "commit"
                if refname in (BRANCH_REF, P29_BRANCH_REF, P28_BRANCH_REF, P27_BRANCH_REF)
                else "tag"
            ),
            "peeled_commit": {
                P29_TAG_REF: P29_COMMIT,
                P28_TAG_REF: P28_COMMIT,
                P27_TAG_REF: P27_COMMIT,
                P26_CHECKPOINT_REF: P26_CHECKPOINT_COMMIT,
                P26_SOURCE_REF: P26_SOURCE_COMMIT,
            }.get(refname),
        }
        for refname, object_id in refs.items()
    ]
    return {
        "schema_version": SCHEMA,
        "status": STATUS_BY_PHASE[phase],
        "phase": phase,
        "claim_boundary": claim_boundary,
        "transport": {
            "root": str(paths["transport"]),
            "bundle_path": str(paths["bundle"]),
            "bundle_format": "v2",
            "bundle_sha256": expected_bundle_sha256,
            "bundle_byte_count": expected_bundle_byte_count,
            "bundle_mode_octal": bundle_file.mode_octal,
            "bundle_uid": bundle_file.uid,
            "bundle_gid": bundle_file.gid,
            "bundle_device": bundle_file.device,
            "bundle_inode": bundle_file.inode,
            "bundle_link_count": bundle_file.link_count,
            "stable_nofollow_read_count": 1,
            "private_copy_mode_octal_before_git_use": "0400",
            "all_git_verification_used_private_authenticated_read_only_copy": True,
            "private_copy_identity_hash_size_mode_revalidated_after_git_use": True,
            "network_used": False,
        },
        "verifier": {
            "canonical_source_path": VERIFIER_SOURCE,
            "source_sha256": verifier_sha256,
            "executing_source_authenticated_by_stable_nofollow_read": True,
            "executing_bootstrap_mode_octal": "0600",
            "executing_bootstrap_python_isolated_no_site": True,
            "checked_in_control_source_matches": True,
            "executing_absolute_path_excluded_for_deterministic_replay": True,
            "reconstructors_run_with_python_isolated_no_site": True,
            "reconstructors_run_from_private_authenticated_checkout": True,
            "git_optional_locks_disabled_for_read_only_replay": True,
        },
        "reviewed_orchestrator_source": dict(reviewed_orchestrator_source),
        "permission_safe_namespace": namespace_guard.receipt_record(),
        "closed_bundle": {
            "advertised_ref_count": 9,
            "advertised_refs": ref_records,
            "prerequisite_count": 0,
            "prerequisites": [],
            "verified_in_new_empty_bare_repository": True,
            "each_ref_fetched_explicitly": True,
            "fsck_full_strict_passed": True,
            "tag_types_and_peels_verified": True,
        },
        "closure": {
            "path": str(paths["closure"]),
            "created_from_absent_empty_bare_repository": True,
            "exact_closed_ref_inventory": True,
            "shallow": False,
            "alternates": False,
            "grafts": False,
            "replace_refs": False,
        },
        "control_checkout": {
            "path": str(paths["control"]),
            "transition": (
                "created_from_verified_source_closure"
                if phase == "source"
                else "advanced_from_verified_source_to_runtime_review_closure"
            ),
            "head": p30_commit,
            "tree": p30_tree,
            "detached": True,
            "clean": True,
            "exact_closed_ref_inventory": True,
            "shallow": False,
            "alternates": False,
            "grafts": False,
            "replace_refs": False,
        },
        "history": dict(history),
        "reconstruction_publication_barrier": dict(reconstruction),
        "attempt_guard": {
            "attempt_root": str(paths["attempt"]),
            **dict(attempt_guard),
        },
        "immutable_execution_snapshot": dict(snapshot_guard),
        "receipt_integrity": {
            "receipt_path": str(paths["receipt"]),
            "external_to_control_checkout": True,
            "created_with_no_overwrite": True,
            "receipt_sha256_field_present": False,
            "self_reference_excluded_by_design": True,
            "review_requires_external_sha256": True,
        },
    }


def write_receipt_exclusive(path: Path, raw: bytes, *, transport_fd: int) -> None:
    try:
        descriptor = os.open(
            path.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=transport_fd,
        )
    except OSError as exc:
        raise VerificationError(f"cannot create receipt without overwrite: {exc}") from exc
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise VerificationError("short receipt write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.fsync(transport_fd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-execution-snapshot-only", action="store_true")
    parser.add_argument("--phase", choices=sorted(PHASE_LAYOUT))
    parser.add_argument("--transport-root", required=True)
    parser.add_argument("--bundle")
    parser.add_argument("--expected-bundle-sha256")
    parser.add_argument("--expected-bundle-byte-count", type=int)
    parser.add_argument("--closure-repository")
    parser.add_argument("--control-checkout", required=True)
    parser.add_argument("--attempt-root")
    parser.add_argument("--execution-root", required=True)
    parser.add_argument("--expected-p30-commit")
    parser.add_argument("--expected-p30-tree")
    parser.add_argument("--bootstrap-verifier", required=True)
    parser.add_argument("--expected-verifier-sha256", required=True)
    parser.add_argument("--expected-orchestrator-sha256")
    parser.add_argument("--expected-source-commit")
    parser.add_argument("--expected-source-tree")
    receipts = parser.add_mutually_exclusive_group()
    receipts.add_argument("--receipt-output")
    receipts.add_argument("--verify-existing-receipt")
    parser.add_argument("--expected-receipt-sha256")
    return parser


def _require_normal_cli(args: argparse.Namespace) -> None:
    required = {
        "phase": args.phase,
        "bundle": args.bundle,
        "expected bundle SHA-256": args.expected_bundle_sha256,
        "expected bundle byte count": args.expected_bundle_byte_count,
        "closure repository": args.closure_repository,
        "attempt root": args.attempt_root,
        "expected P30 commit": args.expected_p30_commit,
        "expected P30 tree": args.expected_p30_tree,
        "expected orchestrator SHA-256": args.expected_orchestrator_sha256,
    }
    missing = [label for label, value in required.items() if value is None]
    if missing:
        raise VerificationError(f"normal receipt verification is missing: {missing!r}")
    if (args.receipt_output is None) == (args.verify_existing_receipt is None):
        raise VerificationError(
            "normal receipt verification requires exactly one receipt creation or replay path"
        )


def validate_snapshot_only_layout(args: argparse.Namespace) -> dict[str, Path]:
    forbidden = {
        "phase": args.phase,
        "bundle": args.bundle,
        "expected bundle SHA-256": args.expected_bundle_sha256,
        "expected bundle byte count": args.expected_bundle_byte_count,
        "closure repository": args.closure_repository,
        "attempt root": args.attempt_root,
        "expected P30 commit": args.expected_p30_commit,
        "expected P30 tree": args.expected_p30_tree,
        "expected orchestrator SHA-256": args.expected_orchestrator_sha256,
        "receipt output": args.receipt_output,
        "reviewed receipt": args.verify_existing_receipt,
        "expected receipt SHA-256": args.expected_receipt_sha256,
    }
    supplied = [label for label, value in forbidden.items() if value is not None]
    if supplied:
        raise VerificationError(f"snapshot-only verification forbids arguments: {supplied!r}")
    if args.expected_source_commit is None or args.expected_source_tree is None:
        raise VerificationError("snapshot-only verification requires source commit and tree")
    paths = {
        "transport": _exact_absolute_argument(args.transport_root, "transport"),
        "control": _exact_absolute_argument(args.control_checkout, "control"),
        "execution": _exact_absolute_argument(args.execution_root, "execution"),
        "bootstrap": _exact_absolute_argument(args.bootstrap_verifier, "bootstrap"),
        "attempt": NAMESPACE_ROOT / ATTEMPT_BASENAME,
    }
    expected = {
        "transport": NAMESPACE_ROOT / TRANSPORT_BASENAME,
        "control": NAMESPACE_ROOT / TRANSPORT_BASENAME / CONTROL_BASENAME,
        "execution": NAMESPACE_ROOT / EXECUTION_BASENAME,
        "bootstrap": NAMESPACE_ROOT / TRANSPORT_BASENAME / BOOTSTRAP_BASENAME,
    }
    for label, expected_path in expected.items():
        if paths[label] != expected_path:
            raise VerificationError(
                f"snapshot-only {label} path differs: expected {expected_path}, got {paths[label]}"
            )
    return paths


def verify_execution_snapshot_only(
    args: argparse.Namespace,
    *,
    paths: Mapping[str, Path],
    namespace_guard: NamespaceGuard,
) -> dict[str, object]:
    expected_verifier_sha256 = require_sha256(
        args.expected_verifier_sha256, "expected verifier digest"
    )
    source_commit = require_sha1(args.expected_source_commit, "expected source commit")
    source_tree = require_sha1(args.expected_source_tree, "expected source tree")
    authenticate_executing_verifier(
        expected_verifier_sha256,
        paths["bootstrap"],
        transport_fd=namespace_guard.transport_fd,
    )
    namespace_guard.bind_namespace_inventory(
        [TRANSPORT_BASENAME, ATTEMPT_BASENAME, EXECUTION_BASENAME]
    )
    bind_runtime_attempt_directories(namespace_guard, paths["attempt"])
    execution_fd = bind_execution_snapshot(namespace_guard, paths["execution"])
    bind_transport_repository(namespace_guard, label="control_checkout", path=paths["control"])
    namespace_guard.revalidate()
    validate_control_checkout(
        paths["control"],
        expected_head=source_commit,
        expected_tree=source_tree,
        refs=expected_refs(source_commit),
        run_reconstruction=False,
        expected_verifier_sha256=expected_verifier_sha256,
    )
    first = verify_immutable_execution_snapshot(
        paths["execution"],
        execution_fd=execution_fd,
        control=paths["control"],
        source_commit=source_commit,
    )
    namespace_guard.revalidate()
    second = verify_immutable_execution_snapshot(
        paths["execution"],
        execution_fd=execution_fd,
        control=paths["control"],
        source_commit=source_commit,
    )
    if second != first:
        raise VerificationError("immutable execution snapshot changed during standalone review")
    namespace_guard.revalidate()
    return {
        "schema_version": "passive-muon-p30-immutable-execution-snapshot-verification-v1",
        "status": "immutable_execution_snapshot_verified_before_container_launch",
        "source_commit": source_commit,
        "source_tree": source_tree,
        "permission_safe_namespace": namespace_guard.receipt_record(),
        "immutable_execution_snapshot": first,
        "bundle_or_control_transition_performed": False,
        "network_used": False,
    }


def verify(args: argparse.Namespace, *, namespace_guard: NamespaceGuard) -> dict[str, object]:
    _require_normal_cli(args)
    expected_verifier_sha256 = require_sha256(
        args.expected_verifier_sha256, "expected verifier digest"
    )
    expected_orchestrator_sha256 = require_sha256(
        args.expected_orchestrator_sha256, "expected orchestrator digest"
    )
    authenticate_executing_verifier(
        expected_verifier_sha256,
        Path(args.bootstrap_verifier),
        transport_fd=namespace_guard.transport_fd,
    )
    expected_bundle_sha256 = require_sha256(args.expected_bundle_sha256, "expected bundle digest")
    if args.expected_bundle_byte_count <= 0:
        raise VerificationError("expected bundle byte count must be positive")
    p30_commit = require_sha1(args.expected_p30_commit, "expected P30 commit")
    p30_tree = require_sha1(args.expected_p30_tree, "expected P30 tree")
    if args.phase == "source":
        if args.expected_source_commit is not None or args.expected_source_tree is not None:
            raise VerificationError("source phase forbids runtime source-history arguments")
    elif args.expected_source_commit is None or args.expected_source_tree is None:
        raise VerificationError("runtime-review phase requires source commit and tree")
    if args.receipt_output is not None:
        if args.expected_receipt_sha256 is not None:
            raise VerificationError("receipt creation forbids an expected receipt digest")
    else:
        if args.expected_receipt_sha256 is None:
            raise VerificationError("receipt replay requires its reviewed SHA-256")
        require_sha256(args.expected_receipt_sha256, "expected receipt digest")

    paths = validate_layout(args)
    if namespace_guard.namespace.path != str(NAMESPACE_ROOT):
        raise VerificationError("namespace guard is bound to the wrong root")
    if namespace_guard.transport.path != str(paths["transport"]):
        raise VerificationError("namespace guard is bound to the wrong transport")
    namespace_guard.bind_namespace_inventory(
        [TRANSPORT_BASENAME]
        if args.phase == "source"
        else [TRANSPORT_BASENAME, ATTEMPT_BASENAME, EXECUTION_BASENAME]
    )
    namespace_guard.revalidate()
    replay = args.verify_existing_receipt is not None
    source_replay_authority: dict[str, object] | None = None

    def revalidate_source_authority(boundary: str) -> None:
        if args.phase != "source":
            return
        if not replay:
            assert_authority_absent(paths["authority"], boundary)
            return
        current = validate_optional_source_replay_authority(
            paths["authority"], namespace_guard=namespace_guard
        )
        if current != source_replay_authority:
            raise VerificationError(f"source-replay authority changed {boundary}")

    if args.phase == "source":
        assert_attempt_absent(paths["attempt"], "before bundle read")
        assert_execution_absent(paths["execution"], "before bundle read")
        if replay:
            source_replay_authority = validate_optional_source_replay_authority(
                paths["authority"], namespace_guard=namespace_guard
            )
        else:
            assert_authority_absent(paths["authority"], "before bundle read")
        initial_attempt_guard: dict[str, object] = {
            "mode": "attempt_root_absent_through_source_verification",
            "absent_before_bundle_read": True,
            "absent_before_closure_creation": True,
            "absent_before_control_transition": True,
            "absent_at_validation_end": True,
            "attempt_root_created_or_mutated_by_verifier": False,
            "execution_root": str(paths["execution"]),
            "execution_root_absent_through_source_verification": True,
            "execution_root_created_or_mutated_by_verifier": False,
            "authority_checkout": str(paths["authority"]),
            "authority_absent_before_receipt_publication": True,
            "authority_created_or_mutated_by_verifier": False,
        }
        execution_fd: int | None = None
        initial_snapshot_guard: dict[str, object] = {
            "schema_version": SNAPSHOT_SCHEMA,
            "root_path": str(paths["execution"]),
            "state": "absent_before_attempt_creation",
            "absent_through_source_verification": True,
        }
    else:
        bind_runtime_attempt_directories(namespace_guard, paths["attempt"])
        execution_fd = bind_execution_snapshot(namespace_guard, paths["execution"])
        initial_attempt_guard = validate_runtime_attempt(paths["attempt"])
        initial_snapshot_guard = {}
    refs = expected_refs(p30_commit)

    if not paths["bundle"].is_file() or paths["bundle"].is_symlink():
        raise VerificationError("phase bundle is absent or a symlink")
    if replay:
        if not paths["closure"].is_dir() or paths["closure"].is_symlink():
            raise VerificationError("reviewed phase closure is absent or a symlink")
        if not paths["control"].is_dir() or paths["control"].is_symlink():
            raise VerificationError("reviewed control checkout is absent or a symlink")
    else:
        if paths["closure"].exists() or paths["closure"].is_symlink():
            raise VerificationError("new phase closure path already exists")
        if paths["receipt"].exists() or paths["receipt"].is_symlink():
            raise VerificationError("new receipt path already exists")
        if args.phase == "source" and (paths["control"].exists() or paths["control"].is_symlink()):
            raise VerificationError("source control checkout path already exists")
        if args.phase == "runtime-review" and (
            not paths["control"].is_dir() or paths["control"].is_symlink()
        ):
            raise VerificationError("runtime-review requires the source control checkout")

    if replay:
        bind_transport_repository(
            namespace_guard, label=f"{args.phase}_closure", path=paths["closure"]
        )
        bind_transport_repository(namespace_guard, label="control_checkout", path=paths["control"])
    elif args.phase == "runtime-review":
        bind_transport_repository(namespace_guard, label="control_checkout", path=paths["control"])
    namespace_guard.revalidate()
    if args.phase == "runtime-review":
        assert execution_fd is not None
        assert args.expected_source_commit is not None
        initial_snapshot_guard = verify_immutable_execution_snapshot(
            paths["execution"],
            execution_fd=execution_fd,
            control=paths["control"],
            source_commit=args.expected_source_commit,
        )

    with tempfile.TemporaryDirectory(prefix="p30-bundle-verification-") as temporary:
        private_bundle = Path(temporary) / "authenticated.bundle"
        bundle_file = stable_copy_nofollow(
            paths["bundle"],
            private_bundle,
            source_dir_fd=namespace_guard.transport_fd,
        )
        require_transport_file_authority(
            bundle_file, label="transport bundle", expected_mode="0600"
        )
        if bundle_file.sha256 != expected_bundle_sha256:
            raise VerificationError("transport bundle SHA-256 differs")
        if bundle_file.byte_count != args.expected_bundle_byte_count:
            raise VerificationError("transport bundle byte count differs")
        private_bundle_file = seal_private_bundle(private_bundle, bundle_file)
        advertised, prerequisites = parse_v2_bundle_header(private_bundle)
        if prerequisites:
            raise VerificationError("bundle contains forbidden prerequisite lines")
        if advertised != refs:
            raise VerificationError(
                f"bundle advertised refs differ: expected {refs!r}, got {advertised!r}"
            )
        if args.phase == "source":
            assert_attempt_absent(paths["attempt"], "before closure validation")
            assert_execution_absent(paths["execution"], "before closure validation")
            revalidate_source_authority("before closure validation")
        else:
            validate_runtime_attempt(paths["attempt"])
        if replay:
            # The reviewed receipt authenticates that this durable closure was born empty;
            # replay validates its current closed state and the bundle again in a fresh,
            # ephemeral empty bare repository so ambient objects still cannot mask gaps.
            validate_existing_closure(paths["closure"], refs)
            ephemeral_closure = Path(temporary) / "replay-empty.git"
            initialize_empty_bare(ephemeral_closure)
            verify_and_fill_closure(ephemeral_closure, private_bundle, refs)
            assert_phase_history(
                ephemeral_closure,
                phase=args.phase,
                p30_commit=p30_commit,
                p30_tree=p30_tree,
                source_commit=args.expected_source_commit,
                source_tree=args.expected_source_tree,
            )
        else:
            initialize_empty_bare(paths["closure"], parent_fd=namespace_guard.transport_fd)
            bind_transport_repository(
                namespace_guard,
                label=f"{args.phase}_closure",
                path=paths["closure"],
            )
            if args.phase == "source":
                assert_attempt_absent(paths["attempt"], "after empty closure creation")
                assert_execution_absent(paths["execution"], "after empty closure creation")
                revalidate_source_authority("after empty closure creation")
            else:
                validate_runtime_attempt(paths["attempt"])
            verify_and_fill_closure(paths["closure"], private_bundle, refs)
        revalidate_private_bundle(private_bundle, private_bundle_file)
        namespace_guard.revalidate()

    history = assert_phase_history(
        paths["closure"],
        phase=args.phase,
        p30_commit=p30_commit,
        p30_tree=p30_tree,
        source_commit=args.expected_source_commit,
        source_tree=args.expected_source_tree,
    )
    namespace_guard.revalidate()
    if args.phase == "source":
        assert_attempt_absent(paths["attempt"], "before control checkout transition")
        assert_execution_absent(paths["execution"], "before control checkout transition")
        revalidate_source_authority("before control checkout transition")
    else:
        validate_runtime_attempt(paths["attempt"])
    if not replay:
        if args.phase == "source":
            create_source_control(
                paths["control"],
                paths["closure"],
                refs,
                p30_commit,
                namespace_guard=namespace_guard,
            )
        else:
            advance_runtime_control(
                paths["control"],
                paths["closure"],
                refs,
                source_commit=args.expected_source_commit,
                source_tree=args.expected_source_tree,
                runtime_commit=p30_commit,
                expected_verifier_sha256=expected_verifier_sha256,
            )
    transition_attempt_guard: dict[str, object] | None = None
    if args.phase == "runtime-review":
        transition_attempt_guard = validate_runtime_attempt(
            paths["attempt"],
            control=paths["control"],
            control_fd=namespace_guard.bound_fd("control_checkout"),
        )
        if runtime_attempt_without_control_authority(transition_attempt_guard) != (
            initial_attempt_guard
        ):
            raise VerificationError(
                "frozen runtime evidence changed before control authority validation"
            )
    reconstruction = validate_control_checkout(
        paths["control"],
        expected_head=p30_commit,
        expected_tree=p30_tree,
        refs=refs,
        run_reconstruction=True,
        expected_verifier_sha256=expected_verifier_sha256,
    )
    assert reconstruction is not None
    reviewed_orchestrator_source = validate_reviewed_orchestrator_source(
        paths["control"], expected_sha256=expected_orchestrator_sha256
    )
    namespace_guard.revalidate()
    if args.phase == "source":
        assert_attempt_absent(paths["attempt"], "at validation end")
        assert_execution_absent(paths["execution"], "at validation end")
        revalidate_source_authority("at validation end")
        final_attempt_guard = initial_attempt_guard
        final_snapshot_guard = initial_snapshot_guard
    else:
        final_attempt_guard = validate_runtime_attempt(
            paths["attempt"],
            control=paths["control"],
            control_fd=namespace_guard.bound_fd("control_checkout"),
        )
        if transition_attempt_guard is None or final_attempt_guard != transition_attempt_guard:
            raise VerificationError(
                "frozen runtime or control authority changed during bundle verification"
            )
        assert execution_fd is not None
        assert args.expected_source_commit is not None
        final_snapshot_guard = verify_immutable_execution_snapshot(
            paths["execution"],
            execution_fd=execution_fd,
            control=paths["control"],
            source_commit=args.expected_source_commit,
        )
        if final_snapshot_guard != initial_snapshot_guard:
            raise VerificationError("immutable execution snapshot changed during verification")
    receipt = build_receipt(
        phase=args.phase,
        paths=paths,
        bundle_file=bundle_file,
        expected_bundle_sha256=expected_bundle_sha256,
        expected_bundle_byte_count=args.expected_bundle_byte_count,
        p30_commit=p30_commit,
        p30_tree=p30_tree,
        refs=refs,
        history=history,
        reconstruction=reconstruction,
        attempt_guard=final_attempt_guard,
        snapshot_guard=final_snapshot_guard,
        verifier_sha256=expected_verifier_sha256,
        reviewed_orchestrator_source=reviewed_orchestrator_source,
        namespace_guard=namespace_guard,
    )
    namespace_guard.revalidate()
    expected_receipt_raw = canonical_json_bytes(receipt)
    if replay:
        reviewed_raw, reviewed_file = stable_read_nofollow(
            paths["receipt"], source_dir_fd=namespace_guard.transport_fd
        )
        require_transport_file_authority(
            reviewed_file, label="reviewed receipt", expected_mode="0600"
        )
        if reviewed_file.sha256 != args.expected_receipt_sha256:
            raise VerificationError("reviewed receipt SHA-256 differs")
        reviewed = strict_json_loads(reviewed_raw)
        if reviewed != receipt:
            raise VerificationError("reviewed receipt does not reconstruct from live state")
        if reviewed_raw != expected_receipt_raw:
            raise VerificationError("reviewed receipt is not canonical JSON")
    else:
        write_receipt_exclusive(
            paths["receipt"],
            expected_receipt_raw,
            transport_fd=namespace_guard.transport_fd,
        )
    if args.phase == "source":
        assert_attempt_absent(paths["attempt"], "after receipt handling")
        assert_execution_absent(paths["execution"], "after receipt handling")
        revalidate_source_authority("after receipt handling")
    else:
        if (
            validate_runtime_attempt(
                paths["attempt"],
                control=paths["control"],
                control_fd=namespace_guard.bound_fd("control_checkout"),
            )
            != final_attempt_guard
        ):
            raise VerificationError("frozen runtime evidence changed during receipt handling")
        assert execution_fd is not None
        assert args.expected_source_commit is not None
        post_receipt_snapshot = verify_immutable_execution_snapshot(
            paths["execution"],
            execution_fd=execution_fd,
            control=paths["control"],
            source_commit=args.expected_source_commit,
        )
        if post_receipt_snapshot != final_snapshot_guard:
            raise VerificationError("immutable execution snapshot changed during receipt handling")
    return {
        "schema_version": SCHEMA,
        "status": "receipt_created" if not replay else "reviewed_receipt_replayed",
        "phase": args.phase,
        "bundle_sha256": expected_bundle_sha256,
        "bundle_byte_count": args.expected_bundle_byte_count,
        "p30_commit": p30_commit,
        "p30_tree": p30_tree,
        "receipt_path": str(paths["receipt"]),
        "receipt_sha256": sha256_bytes(expected_receipt_raw),
        "attempt_boundary": (
            "absent" if args.phase == "source" else "existing_frozen_runtime_no_localization"
        ),
        "reconstruction_publication_barrier": {
            "p30_contract": "15/15",
            "p29_terminal_outcome": "11/11",
            "p29_contract": "15/15",
            "p28_terminal_outcome": "10/10",
            "p28_contract": "12/12",
            "p27_contract": "23/23",
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    previous_umask = os.umask(VERIFIER_UMASK)
    try:
        if args.verify_execution_snapshot_only:
            paths = validate_snapshot_only_layout(args)
        else:
            _require_normal_cli(args)
            paths = validate_layout(args)
        with open_namespace_guard(NAMESPACE_ROOT, paths["transport"]) as namespace_guard:
            if args.verify_execution_snapshot_only:
                summary = verify_execution_snapshot_only(
                    args,
                    paths=paths,
                    namespace_guard=namespace_guard,
                )
            else:
                summary = verify(args, namespace_guard=namespace_guard)
    except VerificationError as exc:
        print(f"P30 control-bundle verification failed: {exc}", file=sys.stderr)
        return 1
    finally:
        os.umask(previous_umask)
    output = (
        compact_canonical_json_bytes(summary)
        if args.verify_execution_snapshot_only
        else canonical_json_bytes(summary)
    )
    sys.stdout.buffer.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
