#!/usr/bin/env python3
"""Independently review one canonical P30 control-bundle receipt.

This program is intentionally standard-library-only and read-only.  It does
not trust the verifier that produced the receipt, reopen a Git checkout, or
mutate P30 state.  Operator-reviewed commit, tree, bundle, verifier, and
orchestrator bindings are supplied explicitly on the command line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

RECEIPT_SCHEMA: Final = "passive-muon-p30-control-bundle-receipt-v1"
REVIEW_SCHEMA: Final = "passive-muon-p30-control-bundle-receipt-review-v1"
TRANSPORT_ROOT: Final = "/secure/p30/transport-20260906-03"
CONTROL_CHECKOUT: Final = f"{TRANSPORT_ROOT}/control-source"
ATTEMPT_ROOT: Final = "/secure/p30/attempt-20260906-03"
EXECUTION_ROOT: Final = "/secure/p30/execution-20260906-03"
SECURE_PARENT: Final = "/secure"
NAMESPACE_ROOT: Final = "/secure/p30"
TRANSPORT_BASENAME: Final = "transport-20260906-03"
ATTEMPT_BASENAME: Final = "attempt-20260906-03"
EXECUTION_BASENAME: Final = "execution-20260906-03"
VERIFIER_SOURCE: Final = "scripts/verify_p30_control_bundle.py"
ORCHESTRATOR_SOURCE: Final = "scripts/run_p30_umask_bound_control_seal.sh"
ZERO_NEWLINE_SHA256: Final = hashlib.sha256(b"0\n").hexdigest()
CREATION_SUMMARY_RECONSTRUCTIONS: Final = {
    "p30_contract": "15/15",
    "p29_terminal_outcome": "11/11",
    "p29_contract": "15/15",
    "p28_terminal_outcome": "10/10",
    "p28_contract": "12/12",
    "p27_contract": "23/23",
}

P29_COMMIT: Final = "8a263722c7f8ce1d99b8796a3ccc0425dd1cefcf"
P29_TAG_OBJECT: Final = "cf3c60b1d5c004b9e601a6afbf630358f80f1d79"
P28_COMMIT: Final = "7274367f2b05fb3c9ed9f876a59be647703bbdc2"
P28_TAG_OBJECT: Final = "f3c81a2f14f853f369015a4f81b6695b52eec74e"
P27_COMMIT: Final = "ec63550331925ded158e3f389e294e4d1f12db3a"
P27_TAG_OBJECT: Final = "c88b98eae9be2458abde45b05d3dccfe09c0c7ed"
P26_CHECKPOINT_COMMIT: Final = "5429da23ff18888daa2312c530a4587780484d8b"
P26_CHECKPOINT_OBJECT: Final = "bacad707d3779bfa10957e18cb4c69b1a7f0cbce"
P26_SOURCE_COMMIT: Final = "185e444afc0b44ca0a09b1bde49a6b6fa3973355"
P26_SOURCE_OBJECT: Final = "f35a7dca8f6bc39e9748e79b6712fab4a203396b"
P26_SOURCE_TREE: Final = "24f4bdac331a57bd7c1b807747d7c7fba253ee5a"
NANOGPT_COMMIT: Final = "3adf61e154c3fe3fca428ad6bc3818b27a3b8291"
NANOGPT_TREE: Final = "ca93bcd9b9c9ff32d3016e1e2556644e68bef86a"
MUON_COMMIT: Final = "f98f1cacc0263b04290753e32be8d498c1efc806"
MUON_TREE: Final = "4ea5cd8ab6ebd56a18536f06453619efcd636da0"

RUNTIME_DELTA: Final = [
    "A\texperiments/training/p30_cuda_runtime_lock.json",
    "A\texperiments/training/p30_host_attestation.json",
]
RUNTIME_ARTIFACT_BASENAMES: Final = {
    "runtime_lock": "p30_cuda_runtime_lock.json",
    "host_attestation": "p30_host_attestation.json",
}
RUNTIME_ATTEMPT_TOP_LEVEL: Final = [
    "evidence",
    "p30-image-inspect.json",
    "p30-nvidia-smi.csv",
    "p30-running-container-inspect.json",
    "p30-running-mountinfo.txt",
    "p30-container-id.txt",
]
RUNTIME_EVIDENCE_ALLOWLIST: Final = [
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
RUNTIME_SUCCESS_STATUS_FILES: Final = sorted(
    name for name in RUNTIME_EVIDENCE_ALLOWLIST if name.endswith(".exit-status.txt")
)
RUNTIME_FORBIDDEN_LOCALIZATION_FILES: Final = [
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
RUNTIME_FORBIDDEN_PREFIXES: Final = [
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
SNAPSHOT_SCHEMA: Final = "passive-muon-p30-immutable-execution-snapshot-v1"
SNAPSHOT_GIT_CONFIG: Final = {
    "core.repositoryformatversion": "0",
    "core.filemode": "true",
    "core.bare": "false",
    "core.logallrefupdates": "true",
}
SNAPSHOT_GIT_AUTHORITIES: Final = {
    "repository": {"head": P26_SOURCE_COMMIT, "tree": P26_SOURCE_TREE},
    "nanogpt": {"head": NANOGPT_COMMIT, "tree": NANOGPT_TREE},
    "muon": {"head": MUON_COMMIT, "tree": MUON_TREE},
}
ACL_XATTR_NAMES: Final = {
    "system.posix_acl_access",
    "system.posix_acl_default",
    "system.nfs4_acl",
    "system.richacl",
}
SHA1_RE: Final = re.compile(r"[0-9a-f]{40}\Z")
SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
MOUNT_DEVICE_RE: Final = re.compile(r"[0-9]+:[0-9]+\Z")

TOP_LEVEL_KEYS: Final = {
    "schema_version",
    "status",
    "phase",
    "claim_boundary",
    "transport",
    "verifier",
    "reviewed_orchestrator_source",
    "permission_safe_namespace",
    "closed_bundle",
    "closure",
    "control_checkout",
    "history",
    "reconstruction_publication_barrier",
    "attempt_guard",
    "immutable_execution_snapshot",
    "receipt_integrity",
}

PHASE_DETAILS: Final = {
    "source": {
        "status": "source_bundle_and_permission_layout_verified_before_attempt_root_creation",
        "bundle": "p30_source.bundle",
        "closure": "p30_source_closure.git",
        "receipt": "p30_source_bundle_receipt.json",
        "transition": "created_from_verified_source_closure",
        "claim_boundary": (
            "Offline Git object/ref/control-checkout provenance only. The source receipt "
            "records no attempt-root creation, container, GPU, CUDA, mapping, gradient, "
            "candidate, optimizer update, or training observation."
        ),
    },
    "runtime-review": {
        "status": "runtime_review_bundle_and_permission_layout_verified_before_localization",
        "bundle": "p30_runtime_review.bundle",
        "closure": "p30_runtime_review_closure.git",
        "receipt": "p30_runtime_review_bundle_receipt.json",
        "transition": "advanced_from_verified_source_to_runtime_review_closure",
        "claim_boundary": (
            "Offline Git object/ref/control-checkout provenance plus identity of the "
            "already prepared frozen runtime. The runtime-review receipt records no "
            "localization invocation, mapping observation, gradient, candidate, optimizer "
            "update, or training observation."
        ),
    },
}

RECONSTRUCTIONS: Final = {
    "p30_contract": {
        "label": "P30 contract",
        "reconstructor_path": "scripts/reconstruct_p30_umask_bound_control_seal.py",
        "schema_version": "passive-muon-p30-umask-bound-control-seal-reconstruction-v1",
        "check_count": 15,
    },
    "p29_terminal_outcome": {
        "label": "P29 terminal outcome",
        "reconstructor_path": (
            "scripts/reconstruct_p29_permission_safe_bundle_localization_bridge_outcome.py"
        ),
        "reconstructor_sha256": (
            "b850a9423b42aa2026db8505b7fdd3767d7954e94283cc96ebdb87cba289b33f"
        ),
        "schema_version": (
            "passive-muon-p29-permission-safe-bundle-localization-bridge-outcome-reconstruction-v1"
        ),
        "canonical_sha256": ("6d29601a71ce6e0c99bcc35497d328c8fdd2a3cfc135351c19876507b00a0f64"),
        "check_count": 11,
    },
    "p29_contract": {
        "label": "P29 contract",
        "reconstructor_path": (
            "scripts/reconstruct_p29_permission_safe_bundle_localization_bridge.py"
        ),
        "reconstructor_sha256": (
            "a384bfcbc25fe9148a158a5f524170557808f8e68d07cb99474b64bbe3c10aea"
        ),
        "schema_version": (
            "passive-muon-p29-permission-safe-bundle-localization-bridge-reconstruction-v1"
        ),
        "canonical_sha256": ("54d10ab01b7d453d01931459b314fc4491b6e4317a02fb763ca9872686311b0a"),
        "check_count": 15,
    },
    "p28_terminal_outcome": {
        "label": "P28 terminal outcome",
        "reconstructor_path": (
            "scripts/reconstruct_p28_bundle_complete_localization_bridge_outcome.py"
        ),
        "reconstructor_sha256": (
            "d736909181eaa75298851b7fc65f9c0352fb32c3d07e40eb4118eb10c1898631"
        ),
        "schema_version": (
            "passive-muon-p28-bundle-complete-localization-bridge-outcome-reconstruction-v1"
        ),
        "canonical_sha256": ("298324d163aa12c1cb64bfdb912ae846152d0980ae3c981a54ead75a019305c9"),
        "check_count": 10,
    },
    "p28_contract": {
        "label": "P28 contract",
        "reconstructor_path": "scripts/reconstruct_p28_bundle_complete_localization_bridge.py",
        "reconstructor_sha256": (
            "c8b5c853a5635f277f7d1d4c0185a199c6658c575a35c088c7cca51d36563b62"
        ),
        "schema_version": (
            "passive-muon-p28-bundle-complete-localization-bridge-reconstruction-v1"
        ),
        "canonical_sha256": ("6a5005f902d937f0edbec3394e6205e482fe6f6fea45f986ac873fd6ab2e8718"),
        "check_count": 12,
    },
    "p27_contract": {
        "label": "P27 contract",
        "reconstructor_path": "scripts/reconstruct_p27_cuda_deleted_mapping_localization.py",
        "reconstructor_sha256": (
            "0eedffa5442ce7fc3950ae74d169fa652c70794935c541028eba3d0bda20e033"
        ),
        "schema_version": ("passive-muon-p27-cuda-deleted-mapping-localization-reconstruction-v1"),
        "canonical_sha256": ("02000debea28f499ddeb3b8c6c8b0615786cd2e7858cc8a1605b04d638da4b7a"),
        "check_count": 23,
    },
}


class ReceiptReviewError(ValueError):
    """Raised when a receipt differs from the reviewed P30 contract."""


@dataclass(frozen=True)
class ReviewExpectations:
    phase: str
    receipt_sha256: str
    p30_commit: str
    p30_tree: str
    bundle_sha256: str
    bundle_byte_count: int
    verifier_sha256: str
    orchestrator_sha256: str
    p30_contract_sha256: str
    p30_reconstructor_sha256: str
    source_commit: str | None = None
    source_tree: str | None = None
    runtime_lock_sha256: str | None = None
    runtime_lock_byte_count: int | None = None
    host_attestation_sha256: str | None = None
    host_attestation_byte_count: int | None = None


@dataclass(frozen=True)
class StableReceipt:
    raw: bytes
    sha256: str
    byte_count: int
    mode_octal: str
    uid: int
    gid: int


@dataclass(frozen=True)
class ReviewerSourceIdentity:
    raw: bytes
    path: str
    sha256: str
    byte_count: int
    mode_octal: str
    uid: int
    gid: int
    link_count: int
    stable_nofollow_read: bool
    caller_identity_matches_source: bool


def _strict_json(raw: bytes) -> object:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise ReceiptReviewError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def nonfinite(token: str) -> object:
        raise ReceiptReviewError(f"nonfinite JSON constant: {token}")

    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReceiptReviewError(f"invalid strict JSON: {exc}") from exc


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require_sha1(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA1_RE.fullmatch(value) is None:
        raise ReceiptReviewError(f"{label} is not one lowercase SHA-1")
    return value


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ReceiptReviewError(f"{label} is not one lowercase SHA-256")
    return value


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ReceiptReviewError(f"{label} is not an object")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ReceiptReviewError(
            f"{label} key inventory differs: expected {sorted(expected)!r}, got {sorted(value)!r}"
        )


def _equal(actual: object, expected: object, label: str) -> None:
    if actual != expected or type(actual) is not type(expected):
        raise ReceiptReviewError(f"{label} differs: expected {expected!r}, got {actual!r}")


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ReceiptReviewError(f"{label} is not a positive integer")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ReceiptReviewError(f"{label} is not a nonnegative integer")
    return value


def _review_acl_check(
    value: object,
    *,
    labels: Sequence[str],
    description: str,
) -> None:
    record = _mapping(value, description)
    _exact_keys(
        record,
        {
            "method",
            "supported",
            "inspected_directories",
            "access_acl_absent",
            "default_acl_absent",
            "all_xattr_names",
        },
        description,
    )
    expected_labels = sorted(labels)
    fixed = {
        "method": "descriptor_listxattr",
        "supported": True,
        "inspected_directories": expected_labels,
        "access_acl_absent": True,
        "default_acl_absent": True,
    }
    for key, expected in fixed.items():
        _equal(record.get(key), expected, f"{description}.{key}")
    xattrs = _mapping(record.get("all_xattr_names"), f"{description}.all_xattr_names")
    _exact_keys(xattrs, set(expected_labels), f"{description}.all_xattr_names")
    for label in expected_labels:
        names = xattrs.get(label)
        if (
            not isinstance(names, list)
            or any(not isinstance(name, str) for name in names)
            or names != sorted(set(names))
        ):
            raise ReceiptReviewError(f"{description} xattr names differ for {label}")
        if ACL_XATTR_NAMES.intersection(names):
            raise ReceiptReviewError(f"{description} contains a forbidden ACL xattr for {label}")


def _review_directory_identity(
    value: object,
    *,
    label: str,
    path: str,
    uid: int,
    gid: int,
    mode_octal: str,
    device: int | None = None,
    mount_id: int | None = None,
) -> tuple[int, int]:
    record = _mapping(value, label)
    _exact_keys(
        record,
        {"path", "device", "inode", "mode_octal", "uid", "gid", "mount_id"},
        label,
    )
    fixed = {"path": path, "mode_octal": mode_octal, "uid": uid, "gid": gid}
    for key, expected in fixed.items():
        _equal(record.get(key), expected, f"{label}.{key}")
    actual_device = _positive_int(record.get("device"), f"{label}.device")
    _positive_int(record.get("inode"), f"{label}.inode")
    actual_mount_id = _positive_int(record.get("mount_id"), f"{label}.mount_id")
    if device is not None:
        _equal(actual_device, device, f"{label}.device binding")
    if mount_id is not None:
        _equal(actual_mount_id, mount_id, f"{label}.mount-ID binding")
    return actual_device, actual_mount_id


def _review_permission_safe_namespace(
    receipt: Mapping[str, object], expected: ReviewExpectations
) -> None:
    namespace = _mapping(receipt.get("permission_safe_namespace"), "permission_safe_namespace")
    _exact_keys(
        namespace,
        {
            "secure_parent",
            "namespace",
            "transport",
            "componentwise_o_directory_o_nofollow",
            "open_descriptors_held_through_prepublication_validation",
            "device_inode_stable_before_and_after_git_and_control_transitions",
            "symlink_components_rejected",
            "mount_check",
            "acl_check",
            "exact_namespace_child_inventory",
            "guarded_child_directories",
        },
        "permission_safe_namespace",
    )
    for key in (
        "componentwise_o_directory_o_nofollow",
        "open_descriptors_held_through_prepublication_validation",
        "device_inode_stable_before_and_after_git_and_control_transitions",
        "symlink_components_rejected",
    ):
        _equal(namespace.get(key), True, f"permission_safe_namespace.{key}")

    device, mount_id = _review_directory_identity(
        namespace.get("secure_parent"),
        label="permission_safe_namespace.secure_parent",
        path=SECURE_PARENT,
        uid=0,
        gid=0,
        mode_octal="0755",
    )
    _review_directory_identity(
        namespace.get("namespace"),
        label="permission_safe_namespace.namespace",
        path=NAMESPACE_ROOT,
        uid=0,
        gid=0,
        mode_octal="0755",
        device=device,
        mount_id=mount_id,
    )
    _review_directory_identity(
        namespace.get("transport"),
        label="permission_safe_namespace.transport",
        path=TRANSPORT_ROOT,
        uid=1000,
        gid=1000,
        mode_octal="0700",
        device=device,
        mount_id=mount_id,
    )

    mount = _mapping(namespace.get("mount_check"), "permission_safe_namespace.mount_check")
    _exact_keys(
        mount,
        {
            "method",
            "supported",
            "containing_mount_id",
            "containing_mount_device",
            "namespace_and_transport_same_mount",
            "namespace_and_transport_not_mountpoints",
        },
        "permission_safe_namespace.mount_check",
    )
    mount_fixed = {
        "method": "linux_proc_self_mountinfo",
        "supported": True,
        "containing_mount_id": str(mount_id),
        "namespace_and_transport_same_mount": True,
        "namespace_and_transport_not_mountpoints": True,
    }
    for key, value in mount_fixed.items():
        _equal(mount.get(key), value, f"permission_safe_namespace.mount_check.{key}")
    mount_device = mount.get("containing_mount_device")
    if not isinstance(mount_device, str) or MOUNT_DEVICE_RE.fullmatch(mount_device) is None:
        raise ReceiptReviewError("permission_safe_namespace mount device is malformed")
    _equal(
        mount_device,
        f"{os.major(device)}:{os.minor(device)}",
        "permission_safe_namespace.mount_check.containing_mount_device",
    )

    _review_acl_check(
        namespace.get("acl_check"),
        labels=("secure_parent", "namespace", "transport"),
        description="permission_safe_namespace.acl_check",
    )

    expected_names = (
        [TRANSPORT_BASENAME]
        if expected.phase == "source"
        else sorted((TRANSPORT_BASENAME, ATTEMPT_BASENAME, EXECUTION_BASENAME))
    )
    inventory = _mapping(
        namespace.get("exact_namespace_child_inventory"),
        "permission_safe_namespace.exact_namespace_child_inventory",
    )
    _exact_keys(
        inventory,
        {
            "names",
            "types",
            "extra_siblings_and_capabilities_rejected",
            "inventory_revalidated_through_receipt_handling",
        },
        "permission_safe_namespace.exact_namespace_child_inventory",
    )
    inventory_fixed = {
        "names": expected_names,
        "types": {name: "directory" for name in expected_names},
        "extra_siblings_and_capabilities_rejected": True,
        "inventory_revalidated_through_receipt_handling": True,
    }
    for key, value in inventory_fixed.items():
        _equal(
            inventory.get(key),
            value,
            f"permission_safe_namespace.exact_namespace_child_inventory.{key}",
        )

    if expected.phase == "source":
        guarded_specifications = {
            "control_checkout": (CONTROL_CHECKOUT, 1000, 1000, "0700"),
            "source_closure": (f"{TRANSPORT_ROOT}/p30_source_closure.git", 1000, 1000, "0700"),
        }
    else:
        guarded_specifications = {
            "attempt_evidence": (f"{ATTEMPT_ROOT}/evidence", 0, 0, "0555"),
            "attempt_root": (ATTEMPT_ROOT, 0, 0, "0555"),
            "control_checkout": (CONTROL_CHECKOUT, 1000, 1000, "0700"),
            "execution_root": (EXECUTION_ROOT, 0, 0, "0555"),
            "runtime-review_closure": (
                f"{TRANSPORT_ROOT}/p30_runtime_review_closure.git",
                1000,
                1000,
                "0700",
            ),
        }
    guarded = _mapping(
        namespace.get("guarded_child_directories"),
        "permission_safe_namespace.guarded_child_directories",
    )
    _exact_keys(
        guarded,
        set(guarded_specifications),
        "permission_safe_namespace.guarded_child_directories",
    )
    identity_keys = {"path", "device", "inode", "mode_octal", "uid", "gid", "mount_id"}
    for label, (path, uid, gid, mode_octal) in guarded_specifications.items():
        record = _mapping(guarded.get(label), f"guarded directory {label}")
        _exact_keys(
            record,
            identity_keys | {"expected_uid", "expected_gid", "expected_mode_octal", "acl_check"},
            f"guarded directory {label}",
        )
        _review_directory_identity(
            {key: record[key] for key in identity_keys},
            label=f"guarded directory {label}",
            path=path,
            uid=uid,
            gid=gid,
            mode_octal=mode_octal,
            device=device,
            mount_id=mount_id,
        )
        _equal(record.get("expected_uid"), uid, f"guarded directory {label}.expected_uid")
        _equal(record.get("expected_gid"), gid, f"guarded directory {label}.expected_gid")
        _equal(
            record.get("expected_mode_octal"),
            mode_octal,
            f"guarded directory {label}.expected_mode_octal",
        )
        _review_acl_check(
            record.get("acl_check"),
            labels=(label,),
            description=f"guarded directory {label}.acl_check",
        )


def _stable_fields(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_uid,
        value.st_gid,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def read_reviewer_source_stably(
    path: Path,
    *,
    expected_sha256: str,
    expected_uid: int = 1000,
    expected_gid: int = 1000,
    expected_mode: int = 0o400,
) -> ReviewerSourceIdentity:
    """Authenticate the exact reviewer program that is crossing the boundary."""

    expected_sha256 = _require_sha256(expected_sha256, "expected reviewer source digest")
    if (os.getuid(), os.getgid()) != (expected_uid, expected_gid):
        raise ReceiptReviewError(
            "reviewer caller UID/GID differs from the expected source authority"
        )
    if not path.is_absolute() or path.name in {"", ".", ".."}:
        raise ReceiptReviewError("reviewer source path must be absolute and have a plain basename")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    cloexec = getattr(os, "O_CLOEXEC", 0)
    if not nofollow or not directory:
        raise ReceiptReviewError("O_NOFOLLOW or O_DIRECTORY is unavailable")

    parent_descriptor: int | None = None
    source_descriptor: int | None = None
    try:
        parent_initial = os.lstat(path.parent)
        source_initial = os.lstat(path)
        if path.parent.resolve(strict=True) != path.parent:
            raise ReceiptReviewError(
                "reviewer source parent lexical and resolved identities differ"
            )
        if (
            not stat.S_ISDIR(parent_initial.st_mode)
            or parent_initial.st_uid != expected_uid
            or parent_initial.st_gid != expected_gid
            or stat.S_IMODE(parent_initial.st_mode) & 0o022
        ):
            raise ReceiptReviewError(
                "reviewer source parent must have the exact caller UID/GID and not be "
                "group/world writable"
            )
        if (
            not stat.S_ISREG(source_initial.st_mode)
            or source_initial.st_nlink != 1
            or (
                source_initial.st_uid,
                source_initial.st_gid,
                stat.S_IMODE(source_initial.st_mode),
            )
            != (expected_uid, expected_gid, expected_mode)
        ):
            raise ReceiptReviewError(
                "reviewer source must be regular, single-link, caller-owned, and mode 0400"
            )

        parent_descriptor = os.open(
            path.parent,
            os.O_RDONLY | cloexec | directory | nofollow,
        )
        parent_before = os.fstat(parent_descriptor)
        source_descriptor = os.open(
            path.name,
            os.O_RDONLY | cloexec | nofollow,
            dir_fd=parent_descriptor,
        )
        source_before = os.fstat(source_descriptor)
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        while True:
            chunk = os.read(source_descriptor, 64 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            chunks.append(chunk)
        source_after = os.fstat(source_descriptor)
        source_by_name = os.stat(
            path.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        parent_after = os.fstat(parent_descriptor)
        parent_by_name = os.lstat(path.parent)
    except ReceiptReviewError:
        raise
    except OSError as exc:
        raise ReceiptReviewError(
            f"cannot authenticate reviewer source without following links: {exc}"
        ) from exc
    finally:
        if source_descriptor is not None:
            os.close(source_descriptor)
        if parent_descriptor is not None:
            os.close(parent_descriptor)

    if not (
        _stable_fields(parent_initial)
        == _stable_fields(parent_before)
        == _stable_fields(parent_after)
        == _stable_fields(parent_by_name)
    ):
        raise ReceiptReviewError("reviewer source parent changed during stable read")
    if not (
        _stable_fields(source_initial)
        == _stable_fields(source_before)
        == _stable_fields(source_after)
        == _stable_fields(source_by_name)
    ):
        raise ReceiptReviewError("reviewer source changed during stable read")
    raw = b"".join(chunks)
    if not raw or len(raw) != source_after.st_size:
        raise ReceiptReviewError("reviewer source is empty or changed byte count")
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise ReceiptReviewError("reviewer source SHA-256 differs")
    return ReviewerSourceIdentity(
        raw=raw,
        path=str(path),
        sha256=actual_sha256,
        byte_count=len(raw),
        mode_octal=f"{stat.S_IMODE(source_after.st_mode):04o}",
        uid=source_after.st_uid,
        gid=source_after.st_gid,
        link_count=source_after.st_nlink,
        stable_nofollow_read=True,
        caller_identity_matches_source=True,
    )


def read_client_file_stably(
    path: Path,
    *,
    label: str,
    expected_uid: int = 1000,
    expected_gid: int = 1000,
    expected_mode: int = 0o600,
) -> StableReceipt:
    """Read one client-owned file without following its final path component."""

    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow:
        raise ReceiptReviewError("O_NOFOLLOW is unavailable")
    try:
        before = os.lstat(path)
        descriptor = os.open(path, os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0))
    except OSError as exc:
        raise ReceiptReviewError(f"cannot open {label} safely: {exc}") from exc
    try:
        opened = os.fstat(descriptor)
        if _stable_fields(before) != _stable_fields(opened):
            raise ReceiptReviewError(f"{label} changed before its stable read")
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (opened.st_uid, opened.st_gid, stat.S_IMODE(opened.st_mode))
            != (expected_uid, expected_gid, expected_mode)
        ):
            raise ReceiptReviewError(f"{label} file authority differs")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        by_name = os.stat(path, follow_symlinks=False)
        opened_fields = _stable_fields(opened)
        after_fields = _stable_fields(after)
        by_name_fields = _stable_fields(by_name)
        if opened_fields != after_fields or after_fields != by_name_fields:
            raise ReceiptReviewError(f"{label} changed during its stable read")
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    if len(raw) != after.st_size:
        raise ReceiptReviewError(f"{label} byte count changed during its stable read")
    return StableReceipt(
        raw=raw,
        sha256=_sha256(raw),
        byte_count=len(raw),
        mode_octal=f"{stat.S_IMODE(after.st_mode):04o}",
        uid=after.st_uid,
        gid=after.st_gid,
    )


def read_receipt_stably(
    path: Path,
    *,
    expected_uid: int = 1000,
    expected_gid: int = 1000,
    expected_mode: int = 0o600,
) -> StableReceipt:
    """Read one receipt without following its final path component."""

    return read_client_file_stably(
        path,
        label="receipt",
        expected_uid=expected_uid,
        expected_gid=expected_gid,
        expected_mode=expected_mode,
    )


def _expected_refs(p30_commit: str) -> list[dict[str, object]]:
    return [
        {
            "refname": "refs/heads/p30-umask-bound-control-seal",
            "object_id": p30_commit,
            "object_type": "commit",
            "peeled_commit": None,
        },
        {
            "refname": "refs/heads/p29-permission-safe-bundle-localization-bridge",
            "object_id": P29_COMMIT,
            "object_type": "commit",
            "peeled_commit": None,
        },
        {
            "refname": "refs/tags/p29-orchestrator-source-mode-diagnostic",
            "object_id": P29_TAG_OBJECT,
            "object_type": "tag",
            "peeled_commit": P29_COMMIT,
        },
        {
            "refname": "refs/heads/p28-bundle-complete-localization-bridge",
            "object_id": P28_COMMIT,
            "object_type": "commit",
            "peeled_commit": None,
        },
        {
            "refname": "refs/tags/p28-control-parent-permission-diagnostic",
            "object_id": P28_TAG_OBJECT,
            "object_type": "tag",
            "peeled_commit": P28_COMMIT,
        },
        {
            "refname": "refs/heads/p27-cuda-deleted-mapping-localization",
            "object_id": P27_COMMIT,
            "object_type": "commit",
            "peeled_commit": None,
        },
        {
            "refname": "refs/tags/p27-cuda-deleted-mapping-localization-diagnostic",
            "object_id": P27_TAG_OBJECT,
            "object_type": "tag",
            "peeled_commit": P27_COMMIT,
        },
        {
            "refname": "refs/tags/p26-permission-safe-acquisition-checkpoint",
            "object_id": P26_CHECKPOINT_OBJECT,
            "object_type": "tag",
            "peeled_commit": P26_CHECKPOINT_COMMIT,
        },
        {
            "refname": "refs/tags/p26-attempt02-acquisition-source",
            "object_id": P26_SOURCE_OBJECT,
            "object_type": "tag",
            "peeled_commit": P26_SOURCE_COMMIT,
        },
    ]


def _review_transport(receipt: Mapping[str, object], expected: ReviewExpectations) -> None:
    detail = PHASE_DETAILS[expected.phase]
    transport = _mapping(receipt.get("transport"), "transport")
    _exact_keys(
        transport,
        {
            "root",
            "bundle_path",
            "bundle_format",
            "bundle_sha256",
            "bundle_byte_count",
            "bundle_mode_octal",
            "bundle_uid",
            "bundle_gid",
            "bundle_device",
            "bundle_inode",
            "bundle_link_count",
            "stable_nofollow_read_count",
            "private_copy_mode_octal_before_git_use",
            "all_git_verification_used_private_authenticated_read_only_copy",
            "private_copy_identity_hash_size_mode_revalidated_after_git_use",
            "network_used",
        },
        "transport",
    )
    required = {
        "root": TRANSPORT_ROOT,
        "bundle_path": f"{TRANSPORT_ROOT}/{detail['bundle']}",
        "bundle_format": "v2",
        "bundle_sha256": expected.bundle_sha256,
        "bundle_byte_count": expected.bundle_byte_count,
        "bundle_mode_octal": "0600",
        "bundle_uid": 1000,
        "bundle_gid": 1000,
        "bundle_link_count": 1,
        "stable_nofollow_read_count": 1,
        "private_copy_mode_octal_before_git_use": "0400",
        "all_git_verification_used_private_authenticated_read_only_copy": True,
        "private_copy_identity_hash_size_mode_revalidated_after_git_use": True,
        "network_used": False,
    }
    for key, value in required.items():
        _equal(transport.get(key), value, f"transport.{key}")
    bundle_device = _nonnegative_int(transport.get("bundle_device"), "transport.bundle_device")
    namespace = _mapping(
        receipt.get("permission_safe_namespace"),
        "permission_safe_namespace",
    )
    namespace_transport = _mapping(
        namespace.get("transport"),
        "permission_safe_namespace.transport",
    )
    _equal(
        bundle_device,
        _positive_int(
            namespace_transport.get("device"),
            "permission_safe_namespace.transport.device",
        ),
        "transport.bundle_device namespace binding",
    )
    _positive_int(transport.get("bundle_inode"), "transport.bundle_inode")


def _review_verifier_and_orchestrator(
    receipt: Mapping[str, object], expected: ReviewExpectations
) -> None:
    verifier = _mapping(receipt.get("verifier"), "verifier")
    _exact_keys(
        verifier,
        {
            "canonical_source_path",
            "source_sha256",
            "executing_source_authenticated_by_stable_nofollow_read",
            "executing_bootstrap_mode_octal",
            "executing_bootstrap_python_isolated_no_site",
            "checked_in_control_source_matches",
            "executing_absolute_path_excluded_for_deterministic_replay",
            "reconstructors_run_with_python_isolated_no_site",
            "reconstructors_run_from_private_authenticated_checkout",
            "git_optional_locks_disabled_for_read_only_replay",
        },
        "verifier",
    )
    verifier_expected = {
        "canonical_source_path": VERIFIER_SOURCE,
        "source_sha256": expected.verifier_sha256,
        "executing_source_authenticated_by_stable_nofollow_read": True,
        "executing_bootstrap_mode_octal": "0600",
        "executing_bootstrap_python_isolated_no_site": True,
        "checked_in_control_source_matches": True,
        "executing_absolute_path_excluded_for_deterministic_replay": True,
        "reconstructors_run_with_python_isolated_no_site": True,
        "reconstructors_run_from_private_authenticated_checkout": True,
        "git_optional_locks_disabled_for_read_only_replay": True,
    }
    for key, value in verifier_expected.items():
        _equal(verifier.get(key), value, f"verifier.{key}")

    source = _mapping(receipt.get("reviewed_orchestrator_source"), "reviewed orchestrator")
    _exact_keys(
        source,
        {
            "path",
            "relative_path",
            "git_mode",
            "git_blob",
            "physical_mode_octal",
            "uid",
            "gid",
            "device",
            "inode",
            "link_count",
            "byte_count",
            "sha256",
            "verifier_umask_octal",
            "stable_nofollow_read",
            "acl_xattrs_absent",
            "contract_digest_binding",
        },
        "reviewed orchestrator",
    )
    source_expected = {
        "path": f"{CONTROL_CHECKOUT}/{ORCHESTRATOR_SOURCE}",
        "relative_path": ORCHESTRATOR_SOURCE,
        "git_mode": "100644",
        "physical_mode_octal": "0600",
        "uid": 1000,
        "gid": 1000,
        "link_count": 1,
        "sha256": expected.orchestrator_sha256,
        "verifier_umask_octal": "0077",
        "stable_nofollow_read": True,
        "acl_xattrs_absent": True,
        "contract_digest_binding": True,
    }
    for key, value in source_expected.items():
        _equal(source.get(key), value, f"reviewed_orchestrator_source.{key}")
    _require_sha1(source.get("git_blob"), "reviewed orchestrator Git blob")
    source_device = _nonnegative_int(source.get("device"), "reviewed orchestrator device")
    namespace = _mapping(
        receipt.get("permission_safe_namespace"),
        "permission_safe_namespace",
    )
    guarded = _mapping(
        namespace.get("guarded_child_directories"),
        "permission_safe_namespace.guarded_child_directories",
    )
    control_checkout = _mapping(
        guarded.get("control_checkout"),
        "permission_safe_namespace.guarded_child_directories.control_checkout",
    )
    _equal(
        source_device,
        _positive_int(
            control_checkout.get("device"),
            "permission_safe_namespace guarded control-checkout device",
        ),
        "reviewed orchestrator control-checkout device binding",
    )
    _positive_int(source.get("inode"), "reviewed orchestrator inode")
    _positive_int(source.get("byte_count"), "reviewed orchestrator byte count")


def _review_refs(receipt: Mapping[str, object], expected: ReviewExpectations) -> None:
    closed = _mapping(receipt.get("closed_bundle"), "closed_bundle")
    _exact_keys(
        closed,
        {
            "advertised_ref_count",
            "advertised_refs",
            "prerequisite_count",
            "prerequisites",
            "verified_in_new_empty_bare_repository",
            "each_ref_fetched_explicitly",
            "fsck_full_strict_passed",
            "tag_types_and_peels_verified",
        },
        "closed_bundle",
    )
    _equal(closed.get("advertised_ref_count"), 9, "closed_bundle.advertised_ref_count")
    _equal(closed.get("advertised_refs"), _expected_refs(expected.p30_commit), "closed refs")
    _equal(closed.get("prerequisite_count"), 0, "closed_bundle.prerequisite_count")
    _equal(closed.get("prerequisites"), [], "closed_bundle.prerequisites")
    for key in (
        "verified_in_new_empty_bare_repository",
        "each_ref_fetched_explicitly",
        "fsck_full_strict_passed",
        "tag_types_and_peels_verified",
    ):
        _equal(closed.get(key), True, f"closed_bundle.{key}")


def _review_control_and_history(
    receipt: Mapping[str, object], expected: ReviewExpectations
) -> None:
    detail = PHASE_DETAILS[expected.phase]
    closure = _mapping(receipt.get("closure"), "closure")
    _exact_keys(
        closure,
        {
            "path",
            "created_from_absent_empty_bare_repository",
            "exact_closed_ref_inventory",
            "shallow",
            "alternates",
            "grafts",
            "replace_refs",
        },
        "closure",
    )
    closure_expected = {
        "path": f"{TRANSPORT_ROOT}/{detail['closure']}",
        "created_from_absent_empty_bare_repository": True,
        "exact_closed_ref_inventory": True,
        "shallow": False,
        "alternates": False,
        "grafts": False,
        "replace_refs": False,
    }
    for key, value in closure_expected.items():
        _equal(closure.get(key), value, f"closure.{key}")

    control = _mapping(receipt.get("control_checkout"), "control_checkout")
    _exact_keys(
        control,
        {
            "path",
            "transition",
            "head",
            "tree",
            "detached",
            "clean",
            "exact_closed_ref_inventory",
            "shallow",
            "alternates",
            "grafts",
            "replace_refs",
        },
        "control_checkout",
    )
    control_expected = {
        "path": CONTROL_CHECKOUT,
        "transition": detail["transition"],
        "head": expected.p30_commit,
        "tree": expected.p30_tree,
        "detached": True,
        "clean": True,
        "exact_closed_ref_inventory": True,
        "shallow": False,
        "alternates": False,
        "grafts": False,
        "replace_refs": False,
    }
    for key, value in control_expected.items():
        _equal(control.get(key), value, f"control_checkout.{key}")

    history = _mapping(receipt.get("history"), "history")
    _exact_keys(
        history,
        {
            "source_commit",
            "source_tree",
            "runtime_review_direct_child",
            "runtime_review_exact_delta",
        },
        "history",
    )
    if expected.phase == "source":
        history_expected = {
            "source_commit": expected.p30_commit,
            "source_tree": expected.p30_tree,
            "runtime_review_direct_child": False,
            "runtime_review_exact_delta": [],
        }
    else:
        history_expected = {
            "source_commit": expected.source_commit,
            "source_tree": expected.source_tree,
            "runtime_review_direct_child": True,
            "runtime_review_exact_delta": RUNTIME_DELTA,
        }
    for key, value in history_expected.items():
        _equal(history.get(key), value, f"history.{key}")


def _review_reconstructions(receipt: Mapping[str, object], expected: ReviewExpectations) -> None:
    barrier = _mapping(
        receipt.get("reconstruction_publication_barrier"),
        "reconstruction_publication_barrier",
    )
    expected_keys = {
        "status",
        "required_reconstruction_count",
        "receipt_publication_blocked_unless_all_pass",
        *RECONSTRUCTIONS,
    }
    _exact_keys(barrier, expected_keys, "reconstruction_publication_barrier")
    _equal(barrier.get("status"), "all_required_reconstructions_passed", "reconstruction status")
    _equal(barrier.get("required_reconstruction_count"), 6, "reconstruction count")
    _equal(
        barrier.get("receipt_publication_blocked_unless_all_pass"),
        True,
        "reconstruction publication barrier",
    )
    record_keys = {
        "label",
        "reconstructor_path",
        "reconstructor_sha256",
        "schema_version",
        "canonical_sha256",
        "check_count",
        "true_check_count",
        "internally_consistent",
    }
    for name, specification in RECONSTRUCTIONS.items():
        record = _mapping(barrier.get(name), f"reconstruction {name}")
        _exact_keys(record, record_keys, f"reconstruction {name}")
        dynamic_specification = dict(specification)
        if name == "p30_contract":
            dynamic_specification["canonical_sha256"] = expected.p30_contract_sha256
            dynamic_specification["reconstructor_sha256"] = expected.p30_reconstructor_sha256
        for key, value in dynamic_specification.items():
            _equal(record.get(key), value, f"reconstruction {name}.{key}")
        count = specification["check_count"]
        _equal(record.get("true_check_count"), count, f"reconstruction {name}.true count")
        _equal(record.get("internally_consistent"), True, f"reconstruction {name}.consistency")


def _review_frozen_file_identity(value: object, *, label: str) -> dict[str, object]:
    record = _mapping(value, label)
    _exact_keys(
        record,
        {"sha256", "byte_count", "mode_octal", "uid", "gid", "device", "inode", "link_count"},
        label,
    )
    _require_sha256(record.get("sha256"), f"{label}.sha256")
    _nonnegative_int(record.get("byte_count"), f"{label}.byte_count")
    fixed = {"mode_octal": "0444", "uid": 0, "gid": 0, "link_count": 1}
    for key, expected in fixed.items():
        _equal(record.get(key), expected, f"{label}.{key}")
    _positive_int(record.get("device"), f"{label}.device")
    _positive_int(record.get("inode"), f"{label}.inode")
    return dict(record)


def _runtime_namespace_authority(receipt: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    namespace = _mapping(receipt.get("permission_safe_namespace"), "permission_safe_namespace")
    guarded = _mapping(
        namespace.get("guarded_child_directories"),
        "permission_safe_namespace.guarded_child_directories",
    )
    result: dict[str, Mapping[str, object]] = {}
    for label in ("attempt_root", "attempt_evidence", "control_checkout"):
        result[label] = _mapping(guarded.get(label), f"guarded directory {label}")
    return result


def _review_sorted_xattr_names(value: object, *, label: str) -> None:
    if (
        not isinstance(value, list)
        or any(not isinstance(name, str) for name in value)
        or value != sorted(set(value))
    ):
        raise ReceiptReviewError(f"{label} is not one sorted unique string list")
    if ACL_XATTR_NAMES.intersection(value):
        raise ReceiptReviewError(f"{label} contains a forbidden ACL xattr")


def _review_runtime_control_artifact_authority(
    value: object,
    *,
    label: str,
    basename: str,
    evidence_identity: Mapping[str, object],
    control_identity: Mapping[str, object],
    control_acl: Mapping[str, object],
) -> list[dict[str, object]]:
    record = _mapping(value, f"runtime control authority {label}")
    expected_keys = {
        "path",
        "relative_path",
        "git_mode",
        "git_blob",
        "physical_mode_octal",
        "uid",
        "gid",
        "device",
        "mount_id",
        "inode",
        "link_count",
        "byte_count",
        "sha256",
        "verifier_umask_octal",
        "regular_file",
        "nonsymlink_nofollow",
        "stable_nofollow_read",
        "acl_xattrs_absent",
        "all_xattr_names",
        "native_evidence_bytes_equal",
        "git_blob_bytes_equal",
        "git_index_and_head_stable",
        "parent_directory_stable",
        "componentwise_parent_nofollow",
        "same_device_and_mount_as_control",
        "not_mountpoint",
        "control_checkout_identity",
        "control_checkout_acl_check",
        "intermediate_directory_authority",
    }
    _exact_keys(record, expected_keys, f"runtime control authority {label}")
    relative_path = f"experiments/training/{basename}"
    fixed = {
        "path": f"{CONTROL_CHECKOUT}/{relative_path}",
        "relative_path": relative_path,
        "git_mode": "100644",
        "physical_mode_octal": "0600",
        "uid": 1000,
        "gid": 1000,
        "device": control_identity["device"],
        "mount_id": control_identity["mount_id"],
        "link_count": 1,
        "byte_count": evidence_identity["byte_count"],
        "sha256": evidence_identity["sha256"],
        "verifier_umask_octal": "0077",
        "regular_file": True,
        "nonsymlink_nofollow": True,
        "stable_nofollow_read": True,
        "acl_xattrs_absent": True,
        "native_evidence_bytes_equal": True,
        "git_blob_bytes_equal": True,
        "git_index_and_head_stable": True,
        "parent_directory_stable": True,
        "componentwise_parent_nofollow": True,
        "same_device_and_mount_as_control": True,
        "not_mountpoint": True,
    }
    for key, expected in fixed.items():
        _equal(record.get(key), expected, f"runtime control authority {label}.{key}")
    _require_sha1(record.get("git_blob"), f"runtime control authority {label}.git_blob")
    _positive_int(record.get("inode"), f"runtime control authority {label}.inode")
    _review_sorted_xattr_names(
        record.get("all_xattr_names"),
        label=f"runtime control authority {label}.all_xattr_names",
    )
    recorded_control_identity = _mapping(
        record.get("control_checkout_identity"),
        f"runtime control authority {label}.control_checkout_identity",
    )
    _equal(
        dict(recorded_control_identity),
        dict(control_identity),
        f"runtime control authority {label}.control_checkout_identity binding",
    )
    recorded_control_acl = _mapping(
        record.get("control_checkout_acl_check"),
        f"runtime control authority {label}.control_checkout_acl_check",
    )
    _review_acl_check(
        recorded_control_acl,
        labels=("control_checkout",),
        description=f"runtime control authority {label}.control_checkout_acl_check",
    )
    _equal(
        dict(recorded_control_acl),
        dict(control_acl),
        f"runtime control authority {label}.control_checkout_acl_check binding",
    )

    intermediates = record.get("intermediate_directory_authority")
    if not isinstance(intermediates, list) or len(intermediates) != 2:
        raise ReceiptReviewError(
            f"runtime control authority {label} intermediate inventory differs"
        )
    expected_paths = ("experiments", "experiments/training")
    normalized: list[dict[str, object]] = []
    intermediate_keys = {
        "relative_path",
        "device",
        "inode",
        "mode_octal",
        "uid",
        "gid",
        "mount_id",
        "same_device_and_mount_as_control",
        "not_mountpoint",
        "acl_check",
    }
    for index, (item, expected_path) in enumerate(zip(intermediates, expected_paths, strict=True)):
        intermediate = _mapping(
            item,
            f"runtime control authority {label} intermediate {index}",
        )
        _exact_keys(
            intermediate,
            intermediate_keys,
            f"runtime control authority {label} intermediate {index}",
        )
        intermediate_fixed = {
            "relative_path": expected_path,
            "device": control_identity["device"],
            "mode_octal": "0700",
            "uid": 1000,
            "gid": 1000,
            "mount_id": control_identity["mount_id"],
            "same_device_and_mount_as_control": True,
            "not_mountpoint": True,
        }
        for key, expected in intermediate_fixed.items():
            _equal(
                intermediate.get(key),
                expected,
                f"runtime control authority {label} intermediate {index}.{key}",
            )
        _positive_int(
            intermediate.get("inode"),
            f"runtime control authority {label} intermediate {index}.inode",
        )
        _review_acl_check(
            intermediate.get("acl_check"),
            labels=(f"runtime_control_parent_{index}",),
            description=f"runtime control authority {label} intermediate {index}.acl_check",
        )
        normalized.append(dict(intermediate))
    return normalized


def _review_runtime_control_checkout_authority(
    receipt: Mapping[str, object],
    attempt: Mapping[str, object],
    identities: Mapping[str, Mapping[str, object]],
) -> None:
    authority = _mapping(
        attempt.get("runtime_control_checkout_authority"),
        "attempt_guard.runtime_control_checkout_authority",
    )
    _exact_keys(
        authority,
        {
            "expected_uid",
            "expected_gid",
            "expected_physical_mode_octal",
            "expected_git_mode",
            "expected_link_count",
            "verifier_umask_octal",
            "all_regular_nonsymlink_single_link",
            "all_acl_xattrs_absent",
            "all_native_evidence_bytes_equal",
            "artifacts",
        },
        "attempt_guard.runtime_control_checkout_authority",
    )
    fixed = {
        "expected_uid": 1000,
        "expected_gid": 1000,
        "expected_physical_mode_octal": "0600",
        "expected_git_mode": "100644",
        "expected_link_count": 1,
        "verifier_umask_octal": "0077",
        "all_regular_nonsymlink_single_link": True,
        "all_acl_xattrs_absent": True,
        "all_native_evidence_bytes_equal": True,
    }
    for key, expected in fixed.items():
        _equal(
            authority.get(key),
            expected,
            f"attempt_guard.runtime_control_checkout_authority.{key}",
        )

    namespace_authority = _runtime_namespace_authority(receipt)
    guarded_control = namespace_authority["control_checkout"]
    identity_keys = {"path", "device", "inode", "mode_octal", "uid", "gid", "mount_id"}
    control_identity = {key: guarded_control[key] for key in identity_keys}
    control_acl = _mapping(
        guarded_control.get("acl_check"),
        "guarded directory control_checkout.acl_check",
    )
    artifacts = _mapping(
        authority.get("artifacts"),
        "attempt_guard.runtime_control_checkout_authority.artifacts",
    )
    _exact_keys(
        artifacts,
        set(RUNTIME_ARTIFACT_BASENAMES),
        "attempt_guard.runtime_control_checkout_authority.artifacts",
    )
    shared_intermediates: list[dict[str, object]] | None = None
    for label, basename in RUNTIME_ARTIFACT_BASENAMES.items():
        intermediates = _review_runtime_control_artifact_authority(
            artifacts.get(label),
            label=label,
            basename=basename,
            evidence_identity=identities[f"evidence/{basename}"],
            control_identity=control_identity,
            control_acl=control_acl,
        )
        if shared_intermediates is None:
            shared_intermediates = intermediates
        else:
            _equal(
                intermediates,
                shared_intermediates,
                "runtime control intermediate-directory cross-artifact binding",
            )


def _review_runtime_artifact(
    record: Mapping[str, object],
    *,
    basename: str,
    inventory_identity: Mapping[str, object],
    expected_sha256: str,
    expected_byte_count: int,
) -> None:
    expected_record = {
        "evidence_relative_path": f"evidence/{basename}",
        "committed_relative_path": f"experiments/training/{basename}",
        **inventory_identity,
    }
    _equal(dict(record), expected_record, f"runtime artifact {basename} inventory binding")
    _equal(record.get("sha256"), expected_sha256, f"runtime artifact {basename}.sha256")
    _equal(record.get("byte_count"), expected_byte_count, f"runtime artifact {basename}.byte_count")


def _review_runtime_snapshot(snapshot: Mapping[str, object]) -> None:
    snapshot_keys = {
        "schema_version",
        "root_path",
        "manifest_path",
        "manifest_sha256",
        "manifest_byte_count",
        "entry_count",
        "recursive_authority_verified",
        "recursive_content_verified",
        "recursive_symlink_hardlink_special_rejection_verified",
        "recursive_mount_acl_xattr_absence_verified",
        "git_checkouts_verified",
        "data_inventory_verified",
        "tool_inventory_and_source_bindings_verified",
        "git",
    }
    _exact_keys(snapshot, snapshot_keys, "immutable_execution_snapshot")
    fixed = {
        "schema_version": SNAPSHOT_SCHEMA,
        "root_path": EXECUTION_ROOT,
        "manifest_path": f"{EXECUTION_ROOT}/snapshot-manifest.json",
        "recursive_authority_verified": True,
        "recursive_content_verified": True,
        "recursive_symlink_hardlink_special_rejection_verified": True,
        "recursive_mount_acl_xattr_absence_verified": True,
        "git_checkouts_verified": True,
        "data_inventory_verified": True,
        "tool_inventory_and_source_bindings_verified": True,
    }
    for key, expected in fixed.items():
        _equal(snapshot.get(key), expected, f"immutable_execution_snapshot.{key}")
    _require_sha256(snapshot.get("manifest_sha256"), "immutable snapshot manifest digest")
    _positive_int(snapshot.get("manifest_byte_count"), "immutable snapshot manifest byte count")
    _positive_int(snapshot.get("entry_count"), "immutable snapshot entry count")

    git = _mapping(snapshot.get("git"), "immutable_execution_snapshot.git")
    _exact_keys(git, set(SNAPSHOT_GIT_AUTHORITIES), "immutable_execution_snapshot.git")
    git_keys = {
        "head",
        "tree",
        "tracked_file_count",
        "exact_local_config",
        "detached",
        "index_equals_tree",
        "worktree_bytes_and_modes_equal_index",
        "untracked_and_ignored_absent",
        "object_indirections_absent",
    }
    for label, authority in SNAPSHOT_GIT_AUTHORITIES.items():
        record = _mapping(git.get(label), f"immutable snapshot Git {label}")
        _exact_keys(record, git_keys, f"immutable snapshot Git {label}")
        git_fixed = {
            "head": authority["head"],
            "tree": authority["tree"],
            "exact_local_config": SNAPSHOT_GIT_CONFIG,
            "detached": True,
            "index_equals_tree": True,
            "worktree_bytes_and_modes_equal_index": True,
            "untracked_and_ignored_absent": True,
            "object_indirections_absent": True,
        }
        for key, expected in git_fixed.items():
            _equal(record.get(key), expected, f"immutable snapshot Git {label}.{key}")
        _positive_int(record.get("tracked_file_count"), f"immutable snapshot Git {label}.count")


def _review_phase_guards(receipt: Mapping[str, object], expected: ReviewExpectations) -> None:
    attempt = _mapping(receipt.get("attempt_guard"), "attempt_guard")
    snapshot = _mapping(receipt.get("immutable_execution_snapshot"), "execution snapshot")
    if expected.phase == "source":
        source_values = {
            "attempt_root": ATTEMPT_ROOT,
            "mode": "attempt_root_absent_through_source_verification",
            "absent_before_bundle_read": True,
            "absent_before_closure_creation": True,
            "absent_before_control_transition": True,
            "absent_at_validation_end": True,
            "attempt_root_created_or_mutated_by_verifier": False,
            "execution_root": EXECUTION_ROOT,
            "execution_root_absent_through_source_verification": True,
            "execution_root_created_or_mutated_by_verifier": False,
            "authority_checkout": f"{TRANSPORT_ROOT}/authority-185e444",
            "authority_absent_before_receipt_publication": True,
            "authority_created_or_mutated_by_verifier": False,
        }
        _exact_keys(attempt, set(source_values), "attempt_guard")
        for key, value in source_values.items():
            _equal(attempt.get(key), value, f"attempt_guard.{key}")
        source_snapshot = {
            "schema_version": SNAPSHOT_SCHEMA,
            "root_path": EXECUTION_ROOT,
            "state": "absent_before_attempt_creation",
            "absent_through_source_verification": True,
        }
        _exact_keys(snapshot, set(source_snapshot), "immutable_execution_snapshot")
        for key, value in source_snapshot.items():
            _equal(snapshot.get(key), value, f"immutable_execution_snapshot.{key}")
        return

    runtime_values = {
        "attempt_root": ATTEMPT_ROOT,
        "mode": "existing_frozen_runtime_before_localization",
        "attempt_root_existing": True,
        "evidence_root_existing": True,
        "exact_attempt_top_level": RUNTIME_ATTEMPT_TOP_LEVEL,
        "exact_prepare_evidence_inventory": RUNTIME_EVIDENCE_ALLOWLIST,
        "canonical_success_status_files": RUNTIME_SUCCESS_STATUS_FILES,
        "all_prepare_status_files_exact_zero_newline": True,
        "forbidden_localization_files": RUNTIME_FORBIDDEN_LOCALIZATION_FILES,
        "forbidden_run_phase_prefixes": RUNTIME_FORBIDDEN_PREFIXES,
        "all_forbidden_localization_files_absent": True,
        "attempt_root_created_or_mutated_by_verifier": False,
    }
    _exact_keys(
        attempt,
        set(runtime_values)
        | {
            "runtime_artifacts",
            "exact_inventory_file_identities",
            "runtime_control_checkout_authority",
        },
        "attempt_guard",
    )
    for key, value in runtime_values.items():
        _equal(attempt.get(key), value, f"attempt_guard.{key}")

    inventory = _mapping(
        attempt.get("exact_inventory_file_identities"),
        "attempt_guard.exact_inventory_file_identities",
    )
    top_level_files = [name for name in RUNTIME_ATTEMPT_TOP_LEVEL if name != "evidence"]
    inventory_names = top_level_files + [f"evidence/{name}" for name in RUNTIME_EVIDENCE_ALLOWLIST]
    _exact_keys(
        inventory,
        set(inventory_names),
        "attempt_guard.exact_inventory_file_identities",
    )
    identities = {
        name: _review_frozen_file_identity(
            inventory.get(name),
            label=f"frozen runtime identity {name}",
        )
        for name in inventory_names
    }
    namespace_authority = _runtime_namespace_authority(receipt)
    attempt_device = _positive_int(
        namespace_authority["attempt_root"].get("device"),
        "guarded attempt-root device",
    )
    evidence_device = _positive_int(
        namespace_authority["attempt_evidence"].get("device"),
        "guarded attempt-evidence device",
    )
    for name, identity in identities.items():
        expected_device = evidence_device if name.startswith("evidence/") else attempt_device
        _equal(
            identity["device"],
            expected_device,
            f"frozen runtime identity {name}.directory-device binding",
        )
    for basename in RUNTIME_SUCCESS_STATUS_FILES:
        identity = identities[f"evidence/{basename}"]
        _equal(
            identity["sha256"],
            ZERO_NEWLINE_SHA256,
            f"canonical success status {basename}.sha256",
        )
        _equal(
            identity["byte_count"],
            2,
            f"canonical success status {basename}.byte_count",
        )

    artifacts = _mapping(attempt.get("runtime_artifacts"), "runtime artifacts")
    _exact_keys(artifacts, {"runtime_lock", "host_attestation"}, "runtime artifacts")
    assert expected.runtime_lock_sha256 is not None
    assert expected.runtime_lock_byte_count is not None
    assert expected.host_attestation_sha256 is not None
    assert expected.host_attestation_byte_count is not None
    _review_runtime_artifact(
        _mapping(artifacts.get("runtime_lock"), "runtime lock"),
        basename="p30_cuda_runtime_lock.json",
        inventory_identity=identities["evidence/p30_cuda_runtime_lock.json"],
        expected_sha256=expected.runtime_lock_sha256,
        expected_byte_count=expected.runtime_lock_byte_count,
    )
    _review_runtime_artifact(
        _mapping(artifacts.get("host_attestation"), "host attestation"),
        basename="p30_host_attestation.json",
        inventory_identity=identities["evidence/p30_host_attestation.json"],
        expected_sha256=expected.host_attestation_sha256,
        expected_byte_count=expected.host_attestation_byte_count,
    )
    _review_runtime_control_checkout_authority(receipt, attempt, identities)
    _review_runtime_snapshot(snapshot)


def _review_receipt_integrity(receipt: Mapping[str, object], expected: ReviewExpectations) -> None:
    detail = PHASE_DETAILS[expected.phase]
    integrity = _mapping(receipt.get("receipt_integrity"), "receipt_integrity")
    _exact_keys(
        integrity,
        {
            "receipt_path",
            "external_to_control_checkout",
            "created_with_no_overwrite",
            "receipt_sha256_field_present",
            "self_reference_excluded_by_design",
            "review_requires_external_sha256",
        },
        "receipt_integrity",
    )
    values = {
        "receipt_path": f"{TRANSPORT_ROOT}/{detail['receipt']}",
        "external_to_control_checkout": True,
        "created_with_no_overwrite": True,
        "receipt_sha256_field_present": False,
        "self_reference_excluded_by_design": True,
        "review_requires_external_sha256": True,
    }
    for key, value in values.items():
        _equal(integrity.get(key), value, f"receipt_integrity.{key}")


def review_creation_summary(raw: bytes, expected: ReviewExpectations) -> dict[str, object]:
    """Bind a downloaded receipt to the trusted one-shot creation transcript."""

    _validate_expectations(expected)
    parsed = _strict_json(raw)
    summary = _mapping(parsed, "creation summary")
    if raw != _canonical_json(parsed):
        raise ReceiptReviewError(
            "creation summary is not canonical sorted indented JSON plus one newline"
        )
    detail = PHASE_DETAILS[expected.phase]
    required = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "receipt_created",
        "phase": expected.phase,
        "bundle_sha256": expected.bundle_sha256,
        "bundle_byte_count": expected.bundle_byte_count,
        "p30_commit": expected.p30_commit,
        "p30_tree": expected.p30_tree,
        "receipt_path": f"{TRANSPORT_ROOT}/{detail['receipt']}",
        "receipt_sha256": expected.receipt_sha256,
        "attempt_boundary": (
            "absent" if expected.phase == "source" else "existing_frozen_runtime_no_localization"
        ),
        "reconstruction_publication_barrier": CREATION_SUMMARY_RECONSTRUCTIONS,
    }
    _exact_keys(summary, set(required), "creation summary")
    for key, value in required.items():
        _equal(summary.get(key), value, f"creation summary.{key}")
    return dict(summary)


def _validate_expectations(expected: ReviewExpectations) -> None:
    if expected.phase not in PHASE_DETAILS:
        raise ReceiptReviewError(f"unknown phase: {expected.phase}")
    _require_sha256(expected.receipt_sha256, "expected receipt digest")
    _require_sha1(expected.p30_commit, "expected P30 commit")
    _require_sha1(expected.p30_tree, "expected P30 tree")
    _require_sha256(expected.bundle_sha256, "expected bundle digest")
    _positive_int(expected.bundle_byte_count, "expected bundle byte count")
    _require_sha256(expected.verifier_sha256, "expected verifier digest")
    _require_sha256(expected.orchestrator_sha256, "expected orchestrator digest")
    _require_sha256(expected.p30_contract_sha256, "expected P30 contract digest")
    _require_sha256(expected.p30_reconstructor_sha256, "expected P30 reconstructor digest")
    runtime_values = (
        expected.source_commit,
        expected.source_tree,
        expected.runtime_lock_sha256,
        expected.runtime_lock_byte_count,
        expected.host_attestation_sha256,
        expected.host_attestation_byte_count,
    )
    if expected.phase == "source":
        if any(value is not None for value in runtime_values):
            raise ReceiptReviewError("source review forbids runtime-only expectations")
        return
    if any(value is None for value in runtime_values):
        raise ReceiptReviewError("runtime-review requires all source and runtime-artifact bindings")
    _require_sha1(expected.source_commit, "expected source commit")
    _require_sha1(expected.source_tree, "expected source tree")
    _require_sha256(expected.runtime_lock_sha256, "expected runtime lock digest")
    _positive_int(expected.runtime_lock_byte_count, "expected runtime lock byte count")
    _require_sha256(expected.host_attestation_sha256, "expected host attestation digest")
    _positive_int(expected.host_attestation_byte_count, "expected host attestation byte count")


def review_receipt(raw: bytes, expected: ReviewExpectations) -> dict[str, object]:
    """Validate receipt bytes against explicit operator-reviewed expectations."""

    _validate_expectations(expected)
    actual_receipt_sha256 = _sha256(raw)
    if actual_receipt_sha256 != expected.receipt_sha256:
        raise ReceiptReviewError("receipt SHA-256 differs")
    parsed = _strict_json(raw)
    receipt = _mapping(parsed, "receipt")
    if raw != _canonical_json(parsed):
        raise ReceiptReviewError("receipt is not canonical sorted indented JSON plus one newline")
    _exact_keys(receipt, TOP_LEVEL_KEYS, "receipt")
    detail = PHASE_DETAILS[expected.phase]
    identity = {
        "schema_version": RECEIPT_SCHEMA,
        "phase": expected.phase,
        "status": detail["status"],
        "claim_boundary": detail["claim_boundary"],
    }
    for key, value in identity.items():
        _equal(receipt.get(key), value, key)
    _review_transport(receipt, expected)
    _review_verifier_and_orchestrator(receipt, expected)
    _review_refs(receipt, expected)
    _review_control_and_history(receipt, expected)
    _review_reconstructions(receipt, expected)
    _review_permission_safe_namespace(receipt, expected)
    _review_phase_guards(receipt, expected)
    _review_receipt_integrity(receipt, expected)
    return {
        "schema_version": REVIEW_SCHEMA,
        "status": "passed",
        "passes": True,
        "phase": expected.phase,
        "receipt_sha256": actual_receipt_sha256,
        "receipt_byte_count": len(raw),
        "p30_commit": expected.p30_commit,
        "p30_tree": expected.p30_tree,
        "bundle_sha256": expected.bundle_sha256,
        "bundle_byte_count": expected.bundle_byte_count,
        "advertised_ref_count": 9,
        "required_reconstruction_count": 6,
        "reviewed_orchestrator_git_mode": "100644",
        "reviewed_orchestrator_physical_mode": "0600",
        "safe_to_cross_next_boundary": True,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--creation-summary", type=Path, required=True)
    parser.add_argument("--creation-status", type=Path, required=True)
    parser.add_argument("--phase", choices=sorted(PHASE_DETAILS), required=True)
    parser.add_argument("--expected-reviewer-source-sha256", required=True)
    parser.add_argument("--expected-reviewer-source-uid", type=int, default=1000)
    parser.add_argument("--expected-reviewer-source-gid", type=int, default=1000)
    parser.add_argument("--expected-receipt-sha256", required=True)
    parser.add_argument("--expected-p30-commit", required=True)
    parser.add_argument("--expected-p30-tree", required=True)
    parser.add_argument("--expected-bundle-sha256", required=True)
    parser.add_argument("--expected-bundle-byte-count", type=int, required=True)
    parser.add_argument("--expected-verifier-sha256", required=True)
    parser.add_argument("--expected-orchestrator-sha256", required=True)
    parser.add_argument("--expected-p30-contract-sha256", required=True)
    parser.add_argument("--expected-p30-reconstructor-sha256", required=True)
    parser.add_argument("--expected-source-commit")
    parser.add_argument("--expected-source-tree")
    parser.add_argument("--expected-runtime-lock-sha256")
    parser.add_argument("--expected-runtime-lock-byte-count", type=int)
    parser.add_argument("--expected-host-attestation-sha256")
    parser.add_argument("--expected-host-attestation-byte-count", type=int)
    parser.add_argument("--expected-receipt-uid", type=int, default=1000)
    parser.add_argument("--expected-receipt-gid", type=int, default=1000)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        reviewer_source = read_reviewer_source_stably(
            Path(__file__),
            expected_sha256=args.expected_reviewer_source_sha256,
            expected_uid=args.expected_reviewer_source_uid,
            expected_gid=args.expected_reviewer_source_gid,
        )
        stable = read_receipt_stably(
            args.receipt,
            expected_uid=args.expected_receipt_uid,
            expected_gid=args.expected_receipt_gid,
        )
        creation_summary = read_client_file_stably(
            args.creation_summary,
            label="creation summary",
            expected_uid=args.expected_receipt_uid,
            expected_gid=args.expected_receipt_gid,
        )
        creation_status = read_client_file_stably(
            args.creation_status,
            label="creation status",
            expected_uid=args.expected_receipt_uid,
            expected_gid=args.expected_receipt_gid,
        )
        if creation_status.raw != b"0\n":
            raise ReceiptReviewError("creation status is not exactly zero newline")
        expected = ReviewExpectations(
            phase=args.phase,
            receipt_sha256=args.expected_receipt_sha256,
            p30_commit=args.expected_p30_commit,
            p30_tree=args.expected_p30_tree,
            bundle_sha256=args.expected_bundle_sha256,
            bundle_byte_count=args.expected_bundle_byte_count,
            verifier_sha256=args.expected_verifier_sha256,
            orchestrator_sha256=args.expected_orchestrator_sha256,
            p30_contract_sha256=args.expected_p30_contract_sha256,
            p30_reconstructor_sha256=args.expected_p30_reconstructor_sha256,
            source_commit=args.expected_source_commit,
            source_tree=args.expected_source_tree,
            runtime_lock_sha256=args.expected_runtime_lock_sha256,
            runtime_lock_byte_count=args.expected_runtime_lock_byte_count,
            host_attestation_sha256=args.expected_host_attestation_sha256,
            host_attestation_byte_count=args.expected_host_attestation_byte_count,
        )
        result = review_receipt(stable.raw, expected)
        reviewed_creation_summary = review_creation_summary(creation_summary.raw, expected)
        result["receipt_file_mode_octal"] = stable.mode_octal
        result["receipt_file_uid"] = stable.uid
        result["receipt_file_gid"] = stable.gid
        result["reviewer_source_sha256"] = reviewer_source.sha256
        result["creation_summary_sha256"] = creation_summary.sha256
        result["creation_summary_receipt_sha256"] = reviewed_creation_summary["receipt_sha256"]
        result["creation_status_sha256"] = creation_status.sha256
        result["reviewer_source_authentication"] = {
            "byte_count": reviewer_source.byte_count,
            "caller_gid": os.getgid(),
            "caller_identity_matches_source": reviewer_source.caller_identity_matches_source,
            "caller_uid": os.getuid(),
            "gid": reviewer_source.gid,
            "link_count": reviewer_source.link_count,
            "mode_octal": reviewer_source.mode_octal,
            "path": reviewer_source.path,
            "sha256": reviewer_source.sha256,
            "stable_nofollow_read": reviewer_source.stable_nofollow_read,
            "uid": reviewer_source.uid,
        }
    except (OSError, ReceiptReviewError) as exc:
        print(f"P30 receipt review blocked: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
