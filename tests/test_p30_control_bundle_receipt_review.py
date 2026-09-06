from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/review_p30_control_bundle_receipt.py"
SPEC = importlib.util.spec_from_file_location("p30_receipt_review", SOURCE)
assert SPEC is not None and SPEC.loader is not None
reviewer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = reviewer
SPEC.loader.exec_module(reviewer)

P30_COMMIT = "a" * 40
P30_TREE = "b" * 40
SOURCE_COMMIT = "c" * 40
SOURCE_TREE = "d" * 40
BUNDLE_SHA256 = "1" * 64
VERIFIER_SHA256 = "2" * 64
ORCHESTRATOR_SHA256 = "3" * 64
CONTRACT_SHA256 = "4" * 64
RECONSTRUCTOR_SHA256 = "5" * 64
RUNTIME_LOCK_SHA256 = "6" * 64
HOST_ATTESTATION_SHA256 = "7" * 64
RUNTIME_LOCK_SIZE = 1201
HOST_ATTESTATION_SIZE = 1302
DEVICE = 41
MOUNT_ID = 17


def _reconstructions() -> dict[str, object]:
    records: dict[str, object] = {
        "status": "all_required_reconstructions_passed",
        "required_reconstruction_count": 6,
        "receipt_publication_blocked_unless_all_pass": True,
    }
    for name, frozen in reviewer.RECONSTRUCTIONS.items():
        specification = dict(frozen)
        if name == "p30_contract":
            specification["canonical_sha256"] = CONTRACT_SHA256
            specification["reconstructor_sha256"] = RECONSTRUCTOR_SHA256
        count = specification["check_count"]
        records[name] = {
            **specification,
            "true_check_count": count,
            "internally_consistent": True,
        }
    return records


def _directory_identity(
    path: str,
    *,
    uid: int,
    gid: int,
    mode_octal: str,
    inode: int,
) -> dict[str, object]:
    return {
        "path": path,
        "device": DEVICE,
        "inode": inode,
        "mode_octal": mode_octal,
        "uid": uid,
        "gid": gid,
        "mount_id": MOUNT_ID,
    }


def _acl(*labels: str) -> dict[str, object]:
    return {
        "method": "descriptor_listxattr",
        "supported": True,
        "inspected_directories": sorted(labels),
        "access_acl_absent": True,
        "default_acl_absent": True,
        "all_xattr_names": {label: [] for label in sorted(labels)},
    }


def _bound_directory(
    label: str,
    path: str,
    *,
    uid: int,
    gid: int,
    mode_octal: str,
    inode: int,
) -> dict[str, object]:
    return {
        **_directory_identity(
            path,
            uid=uid,
            gid=gid,
            mode_octal=mode_octal,
            inode=inode,
        ),
        "expected_uid": uid,
        "expected_gid": gid,
        "expected_mode_octal": mode_octal,
        "acl_check": _acl(label),
    }


def _permission_safe_namespace(phase: str) -> dict[str, object]:
    if phase == "source":
        names = [reviewer.TRANSPORT_BASENAME]
        guarded = {
            "control_checkout": _bound_directory(
                "control_checkout",
                reviewer.CONTROL_CHECKOUT,
                uid=1000,
                gid=1000,
                mode_octal="0700",
                inode=104,
            ),
            "source_closure": _bound_directory(
                "source_closure",
                f"{reviewer.TRANSPORT_ROOT}/p30_source_closure.git",
                uid=1000,
                gid=1000,
                mode_octal="0700",
                inode=105,
            ),
        }
    else:
        names = sorted(
            (
                reviewer.TRANSPORT_BASENAME,
                reviewer.ATTEMPT_BASENAME,
                reviewer.EXECUTION_BASENAME,
            )
        )
        guarded = {
            "attempt_evidence": _bound_directory(
                "attempt_evidence",
                f"{reviewer.ATTEMPT_ROOT}/evidence",
                uid=0,
                gid=0,
                mode_octal="0555",
                inode=106,
            ),
            "attempt_root": _bound_directory(
                "attempt_root",
                reviewer.ATTEMPT_ROOT,
                uid=0,
                gid=0,
                mode_octal="0555",
                inode=107,
            ),
            "control_checkout": _bound_directory(
                "control_checkout",
                reviewer.CONTROL_CHECKOUT,
                uid=1000,
                gid=1000,
                mode_octal="0700",
                inode=104,
            ),
            "execution_root": _bound_directory(
                "execution_root",
                reviewer.EXECUTION_ROOT,
                uid=0,
                gid=0,
                mode_octal="0555",
                inode=108,
            ),
            "runtime-review_closure": _bound_directory(
                "runtime-review_closure",
                f"{reviewer.TRANSPORT_ROOT}/p30_runtime_review_closure.git",
                uid=1000,
                gid=1000,
                mode_octal="0700",
                inode=109,
            ),
        }
    return {
        "secure_parent": _directory_identity(
            reviewer.SECURE_PARENT,
            uid=0,
            gid=0,
            mode_octal="0755",
            inode=101,
        ),
        "namespace": _directory_identity(
            reviewer.NAMESPACE_ROOT,
            uid=0,
            gid=0,
            mode_octal="0755",
            inode=102,
        ),
        "transport": _directory_identity(
            reviewer.TRANSPORT_ROOT,
            uid=1000,
            gid=1000,
            mode_octal="0700",
            inode=103,
        ),
        "componentwise_o_directory_o_nofollow": True,
        "open_descriptors_held_through_prepublication_validation": True,
        "device_inode_stable_before_and_after_git_and_control_transitions": True,
        "symlink_components_rejected": True,
        "mount_check": {
            "method": "linux_proc_self_mountinfo",
            "supported": True,
            "containing_mount_id": str(MOUNT_ID),
            "containing_mount_device": "0:41",
            "namespace_and_transport_same_mount": True,
            "namespace_and_transport_not_mountpoints": True,
        },
        "acl_check": _acl("secure_parent", "namespace", "transport"),
        "exact_namespace_child_inventory": {
            "names": names,
            "types": {name: "directory" for name in names},
            "extra_siblings_and_capabilities_rejected": True,
            "inventory_revalidated_through_receipt_handling": True,
        },
        "guarded_child_directories": guarded,
    }


def _frozen_file_identity(index: int) -> dict[str, object]:
    return {
        "sha256": f"{index:064x}",
        "byte_count": index + 10,
        "mode_octal": "0444",
        "uid": 0,
        "gid": 0,
        "device": DEVICE,
        "inode": index + 200,
        "link_count": 1,
    }


def _runtime_inventory() -> dict[str, dict[str, object]]:
    top_level = [name for name in reviewer.RUNTIME_ATTEMPT_TOP_LEVEL if name != "evidence"]
    names = top_level + [f"evidence/{name}" for name in reviewer.RUNTIME_EVIDENCE_ALLOWLIST]
    inventory = {name: _frozen_file_identity(index + 1) for index, name in enumerate(names)}
    inventory["evidence/p30_cuda_runtime_lock.json"].update(
        {"sha256": RUNTIME_LOCK_SHA256, "byte_count": RUNTIME_LOCK_SIZE}
    )
    inventory["evidence/p30_host_attestation.json"].update(
        {"sha256": HOST_ATTESTATION_SHA256, "byte_count": HOST_ATTESTATION_SIZE}
    )
    for basename in reviewer.RUNTIME_SUCCESS_STATUS_FILES:
        inventory[f"evidence/{basename}"].update(
            {"sha256": reviewer.ZERO_NEWLINE_SHA256, "byte_count": 2}
        )
    return inventory


def _runtime_artifact(
    basename: str,
    identity: dict[str, object],
) -> dict[str, object]:
    return {
        "evidence_relative_path": f"evidence/{basename}",
        "committed_relative_path": f"experiments/training/{basename}",
        **identity,
    }


def _runtime_control_authority(
    inventory: dict[str, dict[str, object]],
    namespace: dict[str, object],
) -> dict[str, object]:
    guarded = namespace["guarded_child_directories"]
    assert isinstance(guarded, dict)
    control = guarded["control_checkout"]
    assert isinstance(control, dict)
    identity_keys = {"path", "device", "inode", "mode_octal", "uid", "gid", "mount_id"}
    control_identity = {key: control[key] for key in identity_keys}
    control_acl = control["acl_check"]
    intermediates = [
        {
            "relative_path": relative_path,
            "device": DEVICE,
            "inode": 310 + index,
            "mode_octal": "0700",
            "uid": 1000,
            "gid": 1000,
            "mount_id": MOUNT_ID,
            "same_device_and_mount_as_control": True,
            "not_mountpoint": True,
            "acl_check": _acl(f"runtime_control_parent_{index}"),
        }
        for index, relative_path in enumerate(("experiments", "experiments/training"))
    ]
    artifacts: dict[str, object] = {}
    for index, (label, basename) in enumerate(reviewer.RUNTIME_ARTIFACT_BASENAMES.items()):
        evidence_identity = inventory[f"evidence/{basename}"]
        relative_path = f"experiments/training/{basename}"
        artifacts[label] = {
            "path": f"{reviewer.CONTROL_CHECKOUT}/{relative_path}",
            "relative_path": relative_path,
            "git_mode": "100644",
            "git_blob": f"{index + 1:040x}",
            "physical_mode_octal": "0600",
            "uid": 1000,
            "gid": 1000,
            "device": DEVICE,
            "mount_id": MOUNT_ID,
            "inode": 320 + index,
            "link_count": 1,
            "byte_count": evidence_identity["byte_count"],
            "sha256": evidence_identity["sha256"],
            "verifier_umask_octal": "0077",
            "regular_file": True,
            "nonsymlink_nofollow": True,
            "stable_nofollow_read": True,
            "acl_xattrs_absent": True,
            "all_xattr_names": [],
            "native_evidence_bytes_equal": True,
            "git_blob_bytes_equal": True,
            "git_index_and_head_stable": True,
            "parent_directory_stable": True,
            "componentwise_parent_nofollow": True,
            "same_device_and_mount_as_control": True,
            "not_mountpoint": True,
            "control_checkout_identity": control_identity,
            "control_checkout_acl_check": control_acl,
            "intermediate_directory_authority": copy.deepcopy(intermediates),
        }
    return {
        "expected_uid": 1000,
        "expected_gid": 1000,
        "expected_physical_mode_octal": "0600",
        "expected_git_mode": "100644",
        "expected_link_count": 1,
        "verifier_umask_octal": "0077",
        "all_regular_nonsymlink_single_link": True,
        "all_acl_xattrs_absent": True,
        "all_native_evidence_bytes_equal": True,
        "artifacts": artifacts,
    }


def _immutable_runtime_snapshot() -> dict[str, object]:
    git: dict[str, object] = {}
    counts = {"repository": 442, "nanogpt": 24, "muon": 2}
    for label, authority in reviewer.SNAPSHOT_GIT_AUTHORITIES.items():
        git[label] = {
            "head": authority["head"],
            "tree": authority["tree"],
            "tracked_file_count": counts[label],
            "exact_local_config": reviewer.SNAPSHOT_GIT_CONFIG,
            "detached": True,
            "index_equals_tree": True,
            "worktree_bytes_and_modes_equal_index": True,
            "untracked_and_ignored_absent": True,
            "object_indirections_absent": True,
        }
    return {
        "schema_version": reviewer.SNAPSHOT_SCHEMA,
        "root_path": reviewer.EXECUTION_ROOT,
        "manifest_path": f"{reviewer.EXECUTION_ROOT}/snapshot-manifest.json",
        "manifest_sha256": "8" * 64,
        "manifest_byte_count": 543210,
        "entry_count": 700,
        "recursive_authority_verified": True,
        "recursive_content_verified": True,
        "recursive_symlink_hardlink_special_rejection_verified": True,
        "recursive_mount_acl_xattr_absence_verified": True,
        "git_checkouts_verified": True,
        "data_inventory_verified": True,
        "tool_inventory_and_source_bindings_verified": True,
        "git": git,
    }


def _receipt(phase: str) -> dict[str, object]:
    detail = reviewer.PHASE_DETAILS[phase]
    receipt: dict[str, object] = {
        "schema_version": reviewer.RECEIPT_SCHEMA,
        "status": detail["status"],
        "phase": phase,
        "claim_boundary": detail["claim_boundary"],
        "transport": {
            "root": reviewer.TRANSPORT_ROOT,
            "bundle_path": f"{reviewer.TRANSPORT_ROOT}/{detail['bundle']}",
            "bundle_format": "v2",
            "bundle_sha256": BUNDLE_SHA256,
            "bundle_byte_count": 98765,
            "bundle_mode_octal": "0600",
            "bundle_uid": 1000,
            "bundle_gid": 1000,
            "bundle_device": 41,
            "bundle_inode": 42,
            "bundle_link_count": 1,
            "stable_nofollow_read_count": 1,
            "private_copy_mode_octal_before_git_use": "0400",
            "all_git_verification_used_private_authenticated_read_only_copy": True,
            "private_copy_identity_hash_size_mode_revalidated_after_git_use": True,
            "network_used": False,
        },
        "verifier": {
            "canonical_source_path": reviewer.VERIFIER_SOURCE,
            "source_sha256": VERIFIER_SHA256,
            "executing_source_authenticated_by_stable_nofollow_read": True,
            "executing_bootstrap_mode_octal": "0600",
            "executing_bootstrap_python_isolated_no_site": True,
            "checked_in_control_source_matches": True,
            "executing_absolute_path_excluded_for_deterministic_replay": True,
            "reconstructors_run_with_python_isolated_no_site": True,
            "reconstructors_run_from_private_authenticated_checkout": True,
            "git_optional_locks_disabled_for_read_only_replay": True,
        },
        "reviewed_orchestrator_source": {
            "path": f"{reviewer.CONTROL_CHECKOUT}/{reviewer.ORCHESTRATOR_SOURCE}",
            "relative_path": reviewer.ORCHESTRATOR_SOURCE,
            "git_mode": "100644",
            "git_blob": "e" * 40,
            "physical_mode_octal": "0600",
            "uid": 1000,
            "gid": 1000,
            "device": 41,
            "inode": 44,
            "link_count": 1,
            "byte_count": 54321,
            "sha256": ORCHESTRATOR_SHA256,
            "verifier_umask_octal": "0077",
            "stable_nofollow_read": True,
            "acl_xattrs_absent": True,
            "contract_digest_binding": True,
        },
        "permission_safe_namespace": _permission_safe_namespace(phase),
        "closed_bundle": {
            "advertised_ref_count": 9,
            "advertised_refs": reviewer._expected_refs(P30_COMMIT),
            "prerequisite_count": 0,
            "prerequisites": [],
            "verified_in_new_empty_bare_repository": True,
            "each_ref_fetched_explicitly": True,
            "fsck_full_strict_passed": True,
            "tag_types_and_peels_verified": True,
        },
        "closure": {
            "path": f"{reviewer.TRANSPORT_ROOT}/{detail['closure']}",
            "created_from_absent_empty_bare_repository": True,
            "exact_closed_ref_inventory": True,
            "shallow": False,
            "alternates": False,
            "grafts": False,
            "replace_refs": False,
        },
        "control_checkout": {
            "path": reviewer.CONTROL_CHECKOUT,
            "transition": detail["transition"],
            "head": P30_COMMIT,
            "tree": P30_TREE,
            "detached": True,
            "clean": True,
            "exact_closed_ref_inventory": True,
            "shallow": False,
            "alternates": False,
            "grafts": False,
            "replace_refs": False,
        },
        "history": {
            "source_commit": P30_COMMIT if phase == "source" else SOURCE_COMMIT,
            "source_tree": P30_TREE if phase == "source" else SOURCE_TREE,
            "runtime_review_direct_child": phase == "runtime-review",
            "runtime_review_exact_delta": (
                reviewer.RUNTIME_DELTA if phase == "runtime-review" else []
            ),
        },
        "reconstruction_publication_barrier": _reconstructions(),
        "attempt_guard": {
            "attempt_root": reviewer.ATTEMPT_ROOT,
        },
        "immutable_execution_snapshot": {
            "schema_version": reviewer.SNAPSHOT_SCHEMA,
            "root_path": reviewer.EXECUTION_ROOT,
        },
        "receipt_integrity": {
            "receipt_path": f"{reviewer.TRANSPORT_ROOT}/{detail['receipt']}",
            "external_to_control_checkout": True,
            "created_with_no_overwrite": True,
            "receipt_sha256_field_present": False,
            "self_reference_excluded_by_design": True,
            "review_requires_external_sha256": True,
        },
    }
    attempt = receipt["attempt_guard"]
    snapshot = receipt["immutable_execution_snapshot"]
    assert isinstance(attempt, dict) and isinstance(snapshot, dict)
    if phase == "source":
        attempt.update(
            {
                "mode": "attempt_root_absent_through_source_verification",
                "absent_before_bundle_read": True,
                "absent_before_closure_creation": True,
                "absent_before_control_transition": True,
                "absent_at_validation_end": True,
                "attempt_root_created_or_mutated_by_verifier": False,
                "execution_root": reviewer.EXECUTION_ROOT,
                "execution_root_absent_through_source_verification": True,
                "execution_root_created_or_mutated_by_verifier": False,
                "authority_checkout": f"{reviewer.TRANSPORT_ROOT}/authority-185e444",
                "authority_absent_before_receipt_publication": True,
                "authority_created_or_mutated_by_verifier": False,
            }
        )
        snapshot.update(
            {
                "state": "absent_before_attempt_creation",
                "absent_through_source_verification": True,
            }
        )
    else:
        inventory = _runtime_inventory()
        namespace = receipt["permission_safe_namespace"]
        assert isinstance(namespace, dict)
        attempt.update(
            {
                "mode": "existing_frozen_runtime_before_localization",
                "attempt_root_existing": True,
                "evidence_root_existing": True,
                "runtime_artifacts": {
                    "runtime_lock": _runtime_artifact(
                        "p30_cuda_runtime_lock.json",
                        inventory["evidence/p30_cuda_runtime_lock.json"],
                    ),
                    "host_attestation": _runtime_artifact(
                        "p30_host_attestation.json",
                        inventory["evidence/p30_host_attestation.json"],
                    ),
                },
                "exact_inventory_file_identities": inventory,
                "runtime_control_checkout_authority": _runtime_control_authority(
                    inventory,
                    namespace,
                ),
                "exact_attempt_top_level": reviewer.RUNTIME_ATTEMPT_TOP_LEVEL,
                "exact_prepare_evidence_inventory": reviewer.RUNTIME_EVIDENCE_ALLOWLIST,
                "canonical_success_status_files": reviewer.RUNTIME_SUCCESS_STATUS_FILES,
                "all_prepare_status_files_exact_zero_newline": True,
                "forbidden_localization_files": reviewer.RUNTIME_FORBIDDEN_LOCALIZATION_FILES,
                "forbidden_run_phase_prefixes": reviewer.RUNTIME_FORBIDDEN_PREFIXES,
                "all_forbidden_localization_files_absent": True,
                "attempt_root_created_or_mutated_by_verifier": False,
            }
        )
        receipt["immutable_execution_snapshot"] = _immutable_runtime_snapshot()
    return receipt


def _canonical(receipt: object) -> bytes:
    return (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()


def _creation_summary(phase: str, receipt_raw: bytes) -> bytes:
    detail = reviewer.PHASE_DETAILS[phase]
    return _canonical(
        {
            "schema_version": reviewer.RECEIPT_SCHEMA,
            "status": "receipt_created",
            "phase": phase,
            "bundle_sha256": BUNDLE_SHA256,
            "bundle_byte_count": 98765,
            "p30_commit": P30_COMMIT,
            "p30_tree": P30_TREE,
            "receipt_path": f"{reviewer.TRANSPORT_ROOT}/{detail['receipt']}",
            "receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(),
            "attempt_boundary": (
                "absent" if phase == "source" else "existing_frozen_runtime_no_localization"
            ),
            "reconstruction_publication_barrier": reviewer.CREATION_SUMMARY_RECONSTRUCTIONS,
        }
    )


def _expectations(phase: str, raw: bytes) -> object:
    common = {
        "phase": phase,
        "receipt_sha256": hashlib.sha256(raw).hexdigest(),
        "p30_commit": P30_COMMIT,
        "p30_tree": P30_TREE,
        "bundle_sha256": BUNDLE_SHA256,
        "bundle_byte_count": 98765,
        "verifier_sha256": VERIFIER_SHA256,
        "orchestrator_sha256": ORCHESTRATOR_SHA256,
        "p30_contract_sha256": CONTRACT_SHA256,
        "p30_reconstructor_sha256": RECONSTRUCTOR_SHA256,
    }
    if phase == "runtime-review":
        common.update(
            {
                "source_commit": SOURCE_COMMIT,
                "source_tree": SOURCE_TREE,
                "runtime_lock_sha256": RUNTIME_LOCK_SHA256,
                "runtime_lock_byte_count": RUNTIME_LOCK_SIZE,
                "host_attestation_sha256": HOST_ATTESTATION_SHA256,
                "host_attestation_byte_count": HOST_ATTESTATION_SIZE,
            }
        )
    return reviewer.ReviewExpectations(**common)


def _replace_path(receipt: dict[str, object], path: tuple[object, ...], value: object) -> None:
    target: object = receipt
    for component in path[:-1]:
        if isinstance(component, int):
            assert isinstance(target, list)
            target = target[component]
        else:
            assert isinstance(target, dict)
            target = target[component]
    final = path[-1]
    if isinstance(final, int):
        assert isinstance(target, list)
        target[final] = value
    else:
        assert isinstance(target, dict)
        target[final] = value


@pytest.mark.parametrize("phase", ["source", "runtime-review"])
def test_exact_receipt_passes(phase: str) -> None:
    raw = _canonical(_receipt(phase))
    result = reviewer.review_receipt(raw, _expectations(phase, raw))
    assert result["passes"] is True
    assert result["phase"] == phase
    assert result["advertised_ref_count"] == 9
    assert result["required_reconstruction_count"] == 6
    assert result["safe_to_cross_next_boundary"] is True


@pytest.mark.parametrize("phase", ["source", "runtime-review"])
def test_exact_creation_summary_binds_receipt(phase: str) -> None:
    raw = _canonical(_receipt(phase))
    result = reviewer.review_creation_summary(
        _creation_summary(phase, raw),
        _expectations(phase, raw),
    )

    assert result["receipt_sha256"] == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("phase", ["source", "runtime-review"])
def test_substituted_internally_valid_receipt_fails_creation_summary_binding(phase: str) -> None:
    original_raw = _canonical(_receipt(phase))
    creation_summary = _creation_summary(phase, original_raw)
    substituted = copy.deepcopy(_receipt(phase))
    transport = substituted["transport"]
    assert isinstance(transport, dict)
    transport["bundle_inode"] = 9001
    substituted_raw = _canonical(substituted)
    substituted_expectations = _expectations(phase, substituted_raw)

    assert reviewer.review_receipt(substituted_raw, substituted_expectations)["passes"] is True
    with pytest.raises(reviewer.ReceiptReviewError, match=r"creation summary\.receipt_sha256"):
        reviewer.review_creation_summary(creation_summary, substituted_expectations)


@pytest.mark.parametrize("phase", ["source", "runtime-review"])
def test_creation_summary_requires_exact_canonical_schema(phase: str) -> None:
    raw = _canonical(_receipt(phase))
    summary = json.loads(_creation_summary(phase, raw))
    summary["unexpected"] = True
    with pytest.raises(reviewer.ReceiptReviewError, match="key inventory differs"):
        reviewer.review_creation_summary(_canonical(summary), _expectations(phase, raw))
    noncanonical = json.dumps(summary, sort_keys=True).encode()
    with pytest.raises(reviewer.ReceiptReviewError, match="not canonical"):
        reviewer.review_creation_summary(noncanonical, _expectations(phase, raw))


@pytest.mark.parametrize("phase", ["source", "runtime-review"])
@pytest.mark.parametrize(
    "section",
    ["permission_safe_namespace", "attempt_guard", "immutable_execution_snapshot"],
)
def test_empty_authority_sections_fail(phase: str, section: str) -> None:
    receipt = _receipt(phase)
    receipt[section] = {}
    raw = _canonical(receipt)
    with pytest.raises(reviewer.ReceiptReviewError):
        reviewer.review_receipt(raw, _expectations(phase, raw))


@pytest.mark.parametrize("phase", ["source", "runtime-review"])
@pytest.mark.parametrize(
    "section",
    ["permission_safe_namespace", "attempt_guard", "immutable_execution_snapshot"],
)
def test_extra_authority_section_keys_fail(phase: str, section: str) -> None:
    receipt = _receipt(phase)
    record = receipt[section]
    assert isinstance(record, dict)
    record["unexpected"] = True
    raw = _canonical(receipt)
    with pytest.raises(reviewer.ReceiptReviewError, match="key inventory differs"):
        reviewer.review_receipt(raw, _expectations(phase, raw))


@pytest.mark.parametrize(
    ("phase", "path", "replacement"),
    [
        (
            "source",
            ("permission_safe_namespace", "componentwise_o_directory_o_nofollow"),
            False,
        ),
        ("source", ("permission_safe_namespace", "secure_parent", "uid"), 1000),
        ("source", ("permission_safe_namespace", "transport", "mode_octal"), "0755"),
        (
            "source",
            ("permission_safe_namespace", "transport", "mount_id"),
            MOUNT_ID + 1,
        ),
        (
            "source",
            ("permission_safe_namespace", "mount_check", "supported"),
            False,
        ),
        (
            "source",
            ("permission_safe_namespace", "mount_check", "containing_mount_device"),
            "malformed",
        ),
        (
            "source",
            ("permission_safe_namespace", "mount_check", "containing_mount_device"),
            "99:99",
        ),
        (
            "source",
            ("permission_safe_namespace", "acl_check", "access_acl_absent"),
            False,
        ),
        (
            "source",
            (
                "permission_safe_namespace",
                "acl_check",
                "all_xattr_names",
                "transport",
            ),
            ["system.posix_acl_access"],
        ),
        (
            "source",
            ("permission_safe_namespace", "exact_namespace_child_inventory", "names"),
            [],
        ),
        (
            "source",
            (
                "permission_safe_namespace",
                "guarded_child_directories",
                "source_closure",
                "expected_mode_octal",
            ),
            "0600",
        ),
        (
            "runtime-review",
            (
                "permission_safe_namespace",
                "guarded_child_directories",
                "attempt_root",
                "uid",
            ),
            1000,
        ),
        (
            "runtime-review",
            (
                "permission_safe_namespace",
                "guarded_child_directories",
                "execution_root",
                "acl_check",
                "default_acl_absent",
            ),
            False,
        ),
    ],
)
def test_namespace_authority_mutations_fail(
    phase: str, path: tuple[object, ...], replacement: object
) -> None:
    receipt = _receipt(phase)
    _replace_path(receipt, path, replacement)
    raw = _canonical(receipt)
    with pytest.raises(reviewer.ReceiptReviewError):
        reviewer.review_receipt(raw, _expectations(phase, raw))


@pytest.mark.parametrize("phase", ["source", "runtime-review"])
@pytest.mark.parametrize(
    "path",
    [
        ("transport", "bundle_device"),
        ("reviewed_orchestrator_source", "device"),
    ],
)
def test_well_formed_cross_device_mismatches_fail(phase: str, path: tuple[object, ...]) -> None:
    receipt = _receipt(phase)
    _replace_path(receipt, path, DEVICE + 1)
    raw = _canonical(receipt)
    with pytest.raises(reviewer.ReceiptReviewError, match=r"device.*binding"):
        reviewer.review_receipt(raw, _expectations(phase, raw))


def test_runtime_inventory_has_exact_declared_cardinalities() -> None:
    receipt = _receipt("runtime-review")
    attempt = receipt["attempt_guard"]
    assert isinstance(attempt, dict)
    identities = attempt["exact_inventory_file_identities"]
    assert isinstance(identities, dict)
    assert len([name for name in reviewer.RUNTIME_ATTEMPT_TOP_LEVEL if name != "evidence"]) == 5
    assert len(reviewer.RUNTIME_EVIDENCE_ALLOWLIST) == 27
    assert len(reviewer.RUNTIME_SUCCESS_STATUS_FILES) == 10
    assert len(identities) == 32


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("attempt_guard", "exact_attempt_top_level"), reviewer.RUNTIME_ATTEMPT_TOP_LEVEL[:-1]),
        (
            ("attempt_guard", "exact_prepare_evidence_inventory"),
            reviewer.RUNTIME_EVIDENCE_ALLOWLIST[:-1],
        ),
        (
            ("attempt_guard", "canonical_success_status_files"),
            reviewer.RUNTIME_SUCCESS_STATUS_FILES[:-1],
        ),
        (
            (
                "attempt_guard",
                "exact_inventory_file_identities",
                "p30-image-inspect.json",
                "mode_octal",
            ),
            "0644",
        ),
        (
            (
                "attempt_guard",
                "exact_inventory_file_identities",
                "evidence/p30-freeze-runtime.stdout.log",
                "uid",
            ),
            1000,
        ),
        (
            (
                "attempt_guard",
                "exact_inventory_file_identities",
                "evidence/p30-running-mountinfo.stderr.log",
                "link_count",
            ),
            2,
        ),
        (
            (
                "attempt_guard",
                "exact_inventory_file_identities",
                "evidence/p30-nvidia-smi.stderr.log",
                "sha256",
            ),
            "not-a-digest",
        ),
        (
            ("attempt_guard", "runtime_artifacts", "runtime_lock", "inode"),
            999999,
        ),
        (
            ("attempt_guard", "forbidden_localization_files"),
            reviewer.RUNTIME_FORBIDDEN_LOCALIZATION_FILES[:-1],
        ),
        (
            ("attempt_guard", "forbidden_run_phase_prefixes"),
            reviewer.RUNTIME_FORBIDDEN_PREFIXES[:-1],
        ),
    ],
)
def test_runtime_inventory_authority_mutations_fail(
    path: tuple[object, ...], replacement: object
) -> None:
    receipt = _receipt("runtime-review")
    _replace_path(receipt, path, replacement)
    raw = _canonical(receipt)
    with pytest.raises(reviewer.ReceiptReviewError):
        reviewer.review_receipt(raw, _expectations("runtime-review", raw))


def test_empty_and_incomplete_runtime_identity_inventories_fail() -> None:
    for empty in ({}, {"p30-image-inspect.json": _frozen_file_identity(1)}):
        receipt = _receipt("runtime-review")
        attempt = receipt["attempt_guard"]
        assert isinstance(attempt, dict)
        attempt["exact_inventory_file_identities"] = empty
        raw = _canonical(receipt)
        with pytest.raises(reviewer.ReceiptReviewError, match="key inventory differs"):
            reviewer.review_receipt(raw, _expectations("runtime-review", raw))


def test_every_runtime_success_status_is_bound_to_exact_zero_newline() -> None:
    for basename in reviewer.RUNTIME_SUCCESS_STATUS_FILES:
        for field, replacement in (("sha256", "f" * 64), ("byte_count", 3)):
            receipt = _receipt("runtime-review")
            identity = receipt["attempt_guard"]["exact_inventory_file_identities"][
                f"evidence/{basename}"
            ]
            assert isinstance(identity, dict)
            identity[field] = replacement
            raw = _canonical(receipt)
            with pytest.raises(reviewer.ReceiptReviewError, match="canonical success status"):
                reviewer.review_receipt(raw, _expectations("runtime-review", raw))


@pytest.mark.parametrize(
    "relative_path",
    ["p30-image-inspect.json", "evidence/p30-freeze-runtime.stdout.log"],
)
def test_runtime_inventory_device_is_bound_to_containing_guarded_directory(
    relative_path: str,
) -> None:
    receipt = _receipt("runtime-review")
    identity = receipt["attempt_guard"]["exact_inventory_file_identities"][relative_path]
    assert isinstance(identity, dict)
    identity["device"] = DEVICE + 1
    raw = _canonical(receipt)
    with pytest.raises(reviewer.ReceiptReviewError, match="directory-device binding"):
        reviewer.review_receipt(raw, _expectations("runtime-review", raw))


def test_runtime_control_checkout_authority_is_mandatory_and_closed() -> None:
    for replacement in ({}, {"unexpected": True}):
        receipt = _receipt("runtime-review")
        receipt["attempt_guard"]["runtime_control_checkout_authority"] = replacement
        raw = _canonical(receipt)
        with pytest.raises(reviewer.ReceiptReviewError):
            reviewer.review_receipt(raw, _expectations("runtime-review", raw))


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("expected_uid",), 0),
        (("all_regular_nonsymlink_single_link",), False),
        (("artifacts", "runtime_lock", "git_mode"), "100755"),
        (("artifacts", "runtime_lock", "sha256"), "f" * 64),
        (("artifacts", "runtime_lock", "byte_count"), RUNTIME_LOCK_SIZE + 1),
        (("artifacts", "runtime_lock", "device"), DEVICE + 1),
        (("artifacts", "runtime_lock", "mount_id"), MOUNT_ID + 1),
        (("artifacts", "runtime_lock", "not_mountpoint"), False),
        (("artifacts", "runtime_lock", "native_evidence_bytes_equal"), False),
        (
            ("artifacts", "runtime_lock", "control_checkout_identity", "inode"),
            999999,
        ),
        (
            (
                "artifacts",
                "runtime_lock",
                "control_checkout_acl_check",
                "default_acl_absent",
            ),
            False,
        ),
        (
            ("artifacts", "runtime_lock", "intermediate_directory_authority", 0, "relative_path"),
            "wrong",
        ),
        (
            ("artifacts", "runtime_lock", "intermediate_directory_authority", 0, "device"),
            DEVICE + 1,
        ),
        (
            ("artifacts", "runtime_lock", "intermediate_directory_authority", 1, "mount_id"),
            MOUNT_ID + 1,
        ),
        (
            (
                "artifacts",
                "runtime_lock",
                "intermediate_directory_authority",
                1,
                "acl_check",
                "access_acl_absent",
            ),
            False,
        ),
        (
            (
                "artifacts",
                "host_attestation",
                "intermediate_directory_authority",
                0,
                "inode",
            ),
            777777,
        ),
    ],
)
def test_runtime_control_checkout_authority_mutations_fail(
    path: tuple[object, ...], replacement: object
) -> None:
    receipt = _receipt("runtime-review")
    authority = receipt["attempt_guard"]["runtime_control_checkout_authority"]
    assert isinstance(authority, dict)
    _replace_path(authority, path, replacement)
    raw = _canonical(receipt)
    with pytest.raises(reviewer.ReceiptReviewError):
        reviewer.review_receipt(raw, _expectations("runtime-review", raw))


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("immutable_execution_snapshot", "schema_version"), "wrong"),
        (("immutable_execution_snapshot", "manifest_sha256"), "not-a-digest"),
        (("immutable_execution_snapshot", "manifest_byte_count"), 0),
        (("immutable_execution_snapshot", "entry_count"), 0),
        (("immutable_execution_snapshot", "recursive_content_verified"), False),
        (
            ("immutable_execution_snapshot", "recursive_mount_acl_xattr_absence_verified"),
            False,
        ),
        (
            ("immutable_execution_snapshot", "git", "repository", "head"),
            "9" * 40,
        ),
        (
            ("immutable_execution_snapshot", "git", "nanogpt", "tree"),
            "9" * 40,
        ),
        (
            ("immutable_execution_snapshot", "git", "muon", "tracked_file_count"),
            0,
        ),
        (
            ("immutable_execution_snapshot", "git", "repository", "exact_local_config"),
            {"core.bare": "false"},
        ),
        (
            ("immutable_execution_snapshot", "git", "repository", "detached"),
            False,
        ),
        (
            (
                "immutable_execution_snapshot",
                "git",
                "repository",
                "object_indirections_absent",
            ),
            False,
        ),
    ],
)
def test_runtime_snapshot_mutations_fail(path: tuple[object, ...], replacement: object) -> None:
    receipt = _receipt("runtime-review")
    _replace_path(receipt, path, replacement)
    raw = _canonical(receipt)
    with pytest.raises(reviewer.ReceiptReviewError):
        reviewer.review_receipt(raw, _expectations("runtime-review", raw))


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("status",), "wrong"),
        (("transport", "bundle_sha256"), "9" * 64),
        (("verifier", "source_sha256"), "9" * 64),
        (("reviewed_orchestrator_source", "git_mode"), "100755"),
        (("reviewed_orchestrator_source", "physical_mode_octal"), "0644"),
        (("reviewed_orchestrator_source", "uid"), 0),
        (("closed_bundle", "advertised_ref_count"), 8),
        (("closed_bundle", "advertised_refs", 2, "object_id"), "9" * 40),
        (("reconstruction_publication_barrier", "required_reconstruction_count"), 5),
        (
            (
                "reconstruction_publication_barrier",
                "p29_terminal_outcome",
                "true_check_count",
            ),
            10,
        ),
        (("control_checkout", "head"), "9" * 40),
    ],
)
def test_critical_source_binding_mutations_fail(
    path: tuple[object, ...], replacement: object
) -> None:
    receipt = _receipt("source")
    target: object = receipt
    for component in path[:-1]:
        if isinstance(component, int):
            assert isinstance(target, list)
            target = target[component]
        else:
            assert isinstance(target, dict)
            target = target[component]
    final = path[-1]
    if isinstance(final, int):
        assert isinstance(target, list)
        target[final] = replacement
    else:
        assert isinstance(target, dict)
        target[final] = replacement
    raw = _canonical(receipt)
    with pytest.raises(reviewer.ReceiptReviewError):
        reviewer.review_receipt(raw, _expectations("source", raw))


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("history", "runtime_review_direct_child"), False),
        (("history", "runtime_review_exact_delta"), list(reversed(reviewer.RUNTIME_DELTA))),
        (
            ("attempt_guard", "runtime_artifacts", "runtime_lock", "committed_relative_path"),
            "experiments/training/wrong.json",
        ),
        (("attempt_guard", "runtime_artifacts", "runtime_lock", "sha256"), "9" * 64),
        (("attempt_guard", "runtime_artifacts", "host_attestation", "byte_count"), 1),
        (("attempt_guard", "all_forbidden_localization_files_absent"), False),
    ],
)
def test_runtime_delta_and_artifact_mutations_fail(
    path: tuple[str, ...], replacement: object
) -> None:
    receipt = _receipt("runtime-review")
    target = receipt
    for component in path[:-1]:
        value = target[component]
        assert isinstance(value, dict)
        target = value
    target[path[-1]] = replacement
    raw = _canonical(receipt)
    with pytest.raises(reviewer.ReceiptReviewError):
        reviewer.review_receipt(raw, _expectations("runtime-review", raw))


def test_noncanonical_and_duplicate_json_fail_even_with_matching_digest() -> None:
    receipt = _receipt("source")
    compact = (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with pytest.raises(reviewer.ReceiptReviewError, match="not canonical"):
        reviewer.review_receipt(compact, _expectations("source", compact))

    canonical = _canonical(receipt)
    duplicate = b'{\n  "schema_version": "duplicate",\n' + canonical[2:]
    with pytest.raises(reviewer.ReceiptReviewError, match="duplicate JSON key"):
        reviewer.review_receipt(duplicate, _expectations("source", duplicate))


def test_external_receipt_digest_is_required() -> None:
    raw = _canonical(_receipt("source"))
    expected = _expectations("source", raw)
    mutated = reviewer.ReviewExpectations(**{**expected.__dict__, "receipt_sha256": "9" * 64})
    with pytest.raises(reviewer.ReceiptReviewError, match="receipt SHA-256 differs"):
        reviewer.review_receipt(raw, mutated)


def test_source_rejects_runtime_only_expectations() -> None:
    raw = _canonical(_receipt("source"))
    expected = _expectations("source", raw)
    mutated = reviewer.ReviewExpectations(**{**expected.__dict__, "source_commit": SOURCE_COMMIT})
    with pytest.raises(reviewer.ReceiptReviewError, match="forbids runtime-only"):
        reviewer.review_receipt(raw, mutated)


def _materialized_reviewer_source(tmp_path: Path) -> tuple[Path, bytes]:
    path = tmp_path / "materialized-reviewer.py"
    raw = SOURCE.read_bytes()
    path.write_bytes(raw)
    path.chmod(0o400)
    return path, raw


def test_reviewer_source_stable_authentication_passes(tmp_path: Path) -> None:
    path, raw = _materialized_reviewer_source(tmp_path)
    result = reviewer.read_reviewer_source_stably(
        path,
        expected_sha256=hashlib.sha256(raw).hexdigest(),
        expected_uid=os.getuid(),
        expected_gid=os.getgid(),
    )
    assert result.raw == raw
    assert result.path == str(path)
    assert result.mode_octal == "0400"
    assert result.uid == os.getuid()
    assert result.gid == os.getgid()
    assert result.link_count == 1
    assert result.stable_nofollow_read is True
    assert result.caller_identity_matches_source is True


def test_reviewer_source_rejects_wrong_hash_mode_and_caller(tmp_path: Path) -> None:
    path, raw = _materialized_reviewer_source(tmp_path)
    digest = hashlib.sha256(raw).hexdigest()
    with pytest.raises(reviewer.ReceiptReviewError, match="SHA-256 differs"):
        reviewer.read_reviewer_source_stably(
            path,
            expected_sha256="0" * 64,
            expected_uid=os.getuid(),
            expected_gid=os.getgid(),
        )
    path.chmod(0o600)
    with pytest.raises(reviewer.ReceiptReviewError, match="mode 0400"):
        reviewer.read_reviewer_source_stably(
            path,
            expected_sha256=digest,
            expected_uid=os.getuid(),
            expected_gid=os.getgid(),
        )
    path.chmod(0o400)
    with pytest.raises(reviewer.ReceiptReviewError, match="caller UID/GID differs"):
        reviewer.read_reviewer_source_stably(
            path,
            expected_sha256=digest,
            expected_uid=os.getuid() + 1,
            expected_gid=os.getgid(),
        )


def test_reviewer_source_rejects_symlink_and_hardlink(tmp_path: Path) -> None:
    path, raw = _materialized_reviewer_source(tmp_path)
    digest = hashlib.sha256(raw).hexdigest()
    symlink = tmp_path / "reviewer-symlink.py"
    symlink.symlink_to(path.name)
    with pytest.raises(reviewer.ReceiptReviewError):
        reviewer.read_reviewer_source_stably(
            symlink,
            expected_sha256=digest,
            expected_uid=os.getuid(),
            expected_gid=os.getgid(),
        )
    hardlink = tmp_path / "reviewer-hardlink.py"
    os.link(path, hardlink)
    with pytest.raises(reviewer.ReceiptReviewError, match="single-link"):
        reviewer.read_reviewer_source_stably(
            path,
            expected_sha256=digest,
            expected_uid=os.getuid(),
            expected_gid=os.getgid(),
        )


def test_reviewer_source_rejects_by_name_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, raw = _materialized_reviewer_source(tmp_path)
    digest = hashlib.sha256(raw).hexdigest()
    real_stat = os.stat

    def raced_stat(
        candidate: object,
        *args: object,
        **kwargs: object,
    ) -> os.stat_result:
        result = real_stat(candidate, *args, **kwargs)
        if candidate == path.name and kwargs.get("dir_fd") is not None:
            fields = list(result)
            fields[stat.ST_SIZE] += 1
            return os.stat_result(fields)
        return result

    monkeypatch.setattr(reviewer.os, "stat", raced_stat)
    with pytest.raises(reviewer.ReceiptReviewError, match="changed during stable read"):
        reviewer.read_reviewer_source_stably(
            path,
            expected_sha256=digest,
            expected_uid=os.getuid(),
            expected_gid=os.getgid(),
        )


def test_cli_stably_reads_receipt_and_emits_review(tmp_path: Path) -> None:
    raw = _canonical(_receipt("source"))
    receipt = tmp_path / "receipt.json"
    receipt.write_bytes(raw)
    receipt.chmod(0o600)
    creation_summary_raw = _creation_summary("source", raw)
    creation_summary = tmp_path / "creation-summary.json"
    creation_summary.write_bytes(creation_summary_raw)
    creation_summary.chmod(0o600)
    creation_status = tmp_path / "creation-status.txt"
    creation_status.write_bytes(b"0\n")
    creation_status.chmod(0o600)
    reviewer_source = tmp_path / "review_p30_control_bundle_receipt.py"
    reviewer_source_raw = SOURCE.read_bytes()
    reviewer_source.write_bytes(reviewer_source_raw)
    reviewer_source.chmod(0o400)
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(reviewer_source),
            "--receipt",
            str(receipt),
            "--creation-summary",
            str(creation_summary),
            "--creation-status",
            str(creation_status),
            "--phase",
            "source",
            "--expected-reviewer-source-sha256",
            hashlib.sha256(reviewer_source_raw).hexdigest(),
            "--expected-reviewer-source-uid",
            str(os.getuid()),
            "--expected-reviewer-source-gid",
            str(os.getgid()),
            "--expected-receipt-sha256",
            hashlib.sha256(raw).hexdigest(),
            "--expected-p30-commit",
            P30_COMMIT,
            "--expected-p30-tree",
            P30_TREE,
            "--expected-bundle-sha256",
            BUNDLE_SHA256,
            "--expected-bundle-byte-count",
            "98765",
            "--expected-verifier-sha256",
            VERIFIER_SHA256,
            "--expected-orchestrator-sha256",
            ORCHESTRATOR_SHA256,
            "--expected-p30-contract-sha256",
            CONTRACT_SHA256,
            "--expected-p30-reconstructor-sha256",
            RECONSTRUCTOR_SHA256,
            "--expected-receipt-uid",
            str(os.getuid()),
            "--expected-receipt-gid",
            str(os.getgid()),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output["schema_version"] == reviewer.REVIEW_SCHEMA
    assert output["receipt_file_mode_octal"] == "0600"
    assert output["safe_to_cross_next_boundary"] is True
    assert output["creation_summary_sha256"] == hashlib.sha256(creation_summary_raw).hexdigest()
    assert output["creation_summary_receipt_sha256"] == hashlib.sha256(raw).hexdigest()
    assert output["creation_status_sha256"] == hashlib.sha256(b"0\n").hexdigest()
    assert output["reviewer_source_sha256"] == hashlib.sha256(reviewer_source_raw).hexdigest()
    assert output["reviewer_source_authentication"] == {
        "byte_count": len(reviewer_source_raw),
        "caller_gid": os.getgid(),
        "caller_identity_matches_source": True,
        "caller_uid": os.getuid(),
        "gid": os.getgid(),
        "link_count": 1,
        "mode_octal": "0400",
        "path": str(reviewer_source),
        "sha256": hashlib.sha256(reviewer_source_raw).hexdigest(),
        "stable_nofollow_read": True,
        "uid": os.getuid(),
    }


@pytest.mark.parametrize("status", [b"1\n", b"0", b"00\n", b""])
def test_cli_rejects_noncanonical_creation_status(tmp_path: Path, status: bytes) -> None:
    raw = _canonical(_receipt("source"))
    receipt = tmp_path / "receipt.json"
    receipt.write_bytes(raw)
    receipt.chmod(0o600)
    creation_summary = tmp_path / "creation-summary.json"
    creation_summary.write_bytes(_creation_summary("source", raw))
    creation_summary.chmod(0o600)
    creation_status = tmp_path / "creation-status.txt"
    creation_status.write_bytes(status)
    creation_status.chmod(0o600)
    reviewer_source = tmp_path / "review_p30_control_bundle_receipt.py"
    reviewer_source_raw = SOURCE.read_bytes()
    reviewer_source.write_bytes(reviewer_source_raw)
    reviewer_source.chmod(0o400)

    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(reviewer_source),
            "--receipt",
            str(receipt),
            "--creation-summary",
            str(creation_summary),
            "--creation-status",
            str(creation_status),
            "--phase",
            "source",
            "--expected-reviewer-source-sha256",
            hashlib.sha256(reviewer_source_raw).hexdigest(),
            "--expected-reviewer-source-uid",
            str(os.getuid()),
            "--expected-reviewer-source-gid",
            str(os.getgid()),
            "--expected-receipt-sha256",
            hashlib.sha256(raw).hexdigest(),
            "--expected-p30-commit",
            P30_COMMIT,
            "--expected-p30-tree",
            P30_TREE,
            "--expected-bundle-sha256",
            BUNDLE_SHA256,
            "--expected-bundle-byte-count",
            "98765",
            "--expected-verifier-sha256",
            VERIFIER_SHA256,
            "--expected-orchestrator-sha256",
            ORCHESTRATOR_SHA256,
            "--expected-p30-contract-sha256",
            CONTRACT_SHA256,
            "--expected-p30-reconstructor-sha256",
            RECONSTRUCTOR_SHA256,
            "--expected-receipt-uid",
            str(os.getuid()),
            "--expected-receipt-gid",
            str(os.getgid()),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "creation status is not exactly zero newline" in result.stderr


def test_stable_reader_rejects_symlink_and_hardlink(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_bytes(b"{}\n")
    receipt.chmod(0o600)
    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(receipt)
    with pytest.raises(reviewer.ReceiptReviewError):
        reviewer.read_receipt_stably(
            symlink,
            expected_uid=os.getuid(),
            expected_gid=os.getgid(),
        )

    hardlink = tmp_path / "hardlink.json"
    os.link(receipt, hardlink)
    with pytest.raises(reviewer.ReceiptReviewError, match="authority differs"):
        reviewer.read_receipt_stably(
            receipt,
            expected_uid=os.getuid(),
            expected_gid=os.getgid(),
        )


def test_unknown_top_level_field_fails_closed() -> None:
    receipt = copy.deepcopy(_receipt("source"))
    receipt["unexpected"] = True
    raw = _canonical(receipt)
    with pytest.raises(reviewer.ReceiptReviewError, match="key inventory differs"):
        reviewer.review_receipt(raw, _expectations("source", raw))
