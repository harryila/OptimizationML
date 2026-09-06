from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import TypeAlias

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "experiments/training/p30_umask_bound_control_seal_contract.json"
RECONSTRUCTOR = ROOT / "scripts/reconstruct_p30_umask_bound_control_seal.py"
WORKFLOW = ROOT / ".github/workflows/p30-umask-bound-control-seal.yml"
RUNBOOK = ROOT / "experiments/training/P30_UMASK_BOUND_CONTROL_SEAL_RUNBOOK.md"
PathPart: TypeAlias = str | int


def _module():
    specification = importlib.util.spec_from_file_location(
        "p30_contract_reconstruction", RECONSTRUCTOR
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _leaves(value: object, prefix: tuple[PathPart, ...] = ()) -> Iterator[tuple[PathPart, ...]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _leaves(item, (*prefix, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _leaves(item, (*prefix, index))
    else:
        yield prefix


def _at(value: object, path: tuple[PathPart, ...]) -> object:
    current = value
    for part in path:
        current = current[part]  # type: ignore[index]
    return current


def _replace(value: object, path: tuple[PathPart, ...], replacement: object) -> None:
    parent = _at(value, path[:-1])
    parent[path[-1]] = replacement  # type: ignore[index]


def _mutation(value: object) -> object:
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, str):
        return f"{value}__mutated"
    if value is None:
        return "invented"
    raise AssertionError(type(value).__name__)


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def test_p30_permission_safe_bridge_reconstructs() -> None:
    result = _module().reconstruct(CONTRACT)

    assert result["internally_consistent"] is True
    assert len(result["checks"]) == 15
    assert all(result["checks"].values())
    assert result["p29_contract_reconstruction"] == {
        "check_count": 15,
        "true_count": 15,
        "false_count": 0,
    }
    assert result["p29_outcome_reconstruction"] == {
        "check_count": 11,
        "true_count": 11,
        "false_count": 0,
    }
    assert result["p28_contract_reconstruction"] == {
        "check_count": 12,
        "true_count": 12,
        "false_count": 0,
    }
    assert result["p28_outcome_reconstruction"] == {
        "check_count": 10,
        "true_count": 10,
        "false_count": 0,
    }
    assert result["p27_reconstruction"] == {
        "check_count": 23,
        "true_count": 23,
        "false_count": 0,
    }
    assert "does not authenticate a host layout" in result["claim_boundary"]


def test_preexecution_revision_orders_receipt_review_before_seal() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    freeze = contract["two_phase_freeze"]
    order = contract["execution_order"]
    write_receipt = (
        "bind the verifier-created tracked-100644 physical-mode-0600 orchestrator source "
        "into the external source receipt and write it with O_EXCL while the authority "
        "checkout and attempt root remain absent"
    )
    review_receipt = (
        "independently review the external source receipt while the authority checkout and "
        "attempt root remain absent"
    )
    seal_source = (
        "run the exact root seal against the reviewed receipt-bound mode-0600 source without "
        "chmod or chown and create a root-owned mode-0555 no-overwrite destination"
    )

    assert order.index(write_receipt) < order.index(review_receipt) < order.index(seal_source)
    assert freeze["historical_source_freeze_tag_preserved"] == (
        "p30-umask-bound-control-seal-source-freeze-v4"
    )
    assert freeze[
        "future_protocols_separate_repeatable_engineering_rehearsals_from_one_shot_scientific_acquisitions"
    ]
    assert freeze["engineering_rehearsal_is_repeatable_disposable_and_not_scientific_acquisition"]
    assert freeze[
        "engineering_rehearsal_may_mirror_path_strings_but_cannot_consume_the_real_host_frozen_attempt_identity_and_initializes_no_cuda"
    ]
    assert freeze[
        "engineering_rehearsal_must_pass_before_the_revised_source_freeze_is_made_authoritative"
    ]
    assert freeze["engineering_rehearsal_failures_do_not_create_new_research_phases"]
    assert freeze["historical_terminal_attempts_and_source_freeze_tags_remain_immutable"]
    assert freeze[
        "client_side_provisioning_verifier_and_seal_stdout_stderr_and_exit_status_are_retained"
    ]
    assert freeze["client_side_stream_triplets_live_outside_frozen_exact_content_directories"]
    assert freeze["remote_attempt_freshness_is_verified_read_only_before_any_state_creation"]


def test_runbook_uses_complete_verifier_cli_and_clean_root_environment_handoffs() -> None:
    runbook = RUNBOOK.read_text(encoding="utf-8")

    assert "ENGINEERING_REHEARSAL_POLICY.md" in runbook
    assert "source receipt is created, then reviewed,\nand only then" in runbook
    assert "scripts/run_logged_client_command.py" in runbook
    assert "outside this repository" in runbook
    assert "/secure/p30" in runbook
    assert "/var/lib/optimizationml-p30-20260906-03" in runbook
    assert "p30-localization-20260906-03" in runbook
    assert runbook.count('--execution-root "$P30_EXECUTION_ROOT"') == 2
    assert 'p30_verify_root_environment_handoff 29 "${p30_source_root_environment[@]}"' in runbook
    assert (
        'p30_verify_root_environment_handoff 37 "${p30_localization_root_environment[@]}"'
        in runbook
    )
    assert runbook.count("sudo /usr/bin/env -i \\\n") == 4
    assert runbook.count("p30_verify_root_environment_handoff() {") == 2
    assert "--preserve-env" in runbook
    assert "Do not rely on `sudo` environment inheritance or\n`--preserve-env`" in runbook

    program_start = runbook.index("'import os,sys\n") + 1
    program_end = runbook.index('\' \\\n    "$expected_count" "$@"', program_start)
    program = runbook[program_start:program_end].replace(
        " or os.geteuid() != 0 or os.getegid() != 0", ""
    )
    clean_environment = {
        "PATH": "/usr/sbin:/usr/bin:/sbin:/bin",
        "P30_ALPHA": "alpha",
        "P30_BETA": "beta",
    }
    valid = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            program,
            "2",
            "P30_ALPHA=alpha",
            "P30_BETA=beta",
        ],
        check=False,
        capture_output=True,
        env=clean_environment,
        text=True,
    )
    assert valid.returncode == 0, valid.stderr
    invalid_count = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            program,
            "3",
            "P30_ALPHA=alpha",
            "P30_BETA=beta",
        ],
        check=False,
        capture_output=True,
        env=clean_environment,
        text=True,
    )
    assert invalid_count.returncode != 0


def test_terminal_p29_parent_and_unchanged_sources_are_bound() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    parent = contract["terminal_parent"]

    assert parent["outcome_commit"] == "8a263722c7f8ce1d99b8796a3ccc0425dd1cefcf"
    assert parent["outcome_tree"] == "6ee56f8c218eab1d25047f6f1046b97f7529d996"
    assert parent["diagnostic_tag_object"] == "cf3c60b1d5c004b9e601a6afbf630358f80f1d79"
    assert parent["source_verifier_invocations_run"] == 1
    assert parent["source_receipt_created"] is True
    assert parent["control_checkout_created"] is True
    assert parent["attempt_root_created"] is False
    assert parent["container_created"] is False
    assert parent["retained_source_closure_created"] is True
    assert parent["retained_transport_or_closure_reuse_allowed"] is False
    assert parent["attempt_is_terminal"] is True
    assert parent["attempt_reuse_allowed"] is False

    for group_name in (
        "unchanged_p29_authorities",
        "unchanged_p28_authorities",
        "unchanged_p27_authorities",
    ):
        for entry in contract[group_name].values():
            if "path" in entry:
                data = (ROOT / entry["path"]).read_bytes()
                assert hashlib.sha256(data).hexdigest() == entry["sha256"]


def test_bundle_inventory_carries_all_nine_exact_refs_and_annotated_tags() -> None:
    refs = json.loads(CONTRACT.read_text(encoding="utf-8"))["required_bundle_refs"]

    assert [item["ref"] for item in refs] == [
        "refs/heads/p30-umask-bound-control-seal",
        "refs/heads/p29-permission-safe-bundle-localization-bridge",
        "refs/tags/p29-orchestrator-source-mode-diagnostic",
        "refs/heads/p28-bundle-complete-localization-bridge",
        "refs/tags/p28-control-parent-permission-diagnostic",
        "refs/heads/p27-cuda-deleted-mapping-localization",
        "refs/tags/p27-cuda-deleted-mapping-localization-diagnostic",
        "refs/tags/p26-permission-safe-acquisition-checkpoint",
        "refs/tags/p26-attempt02-acquisition-source",
    ]
    assert all(refs[index]["object_type"] == "tag" for index in (2, 4, 6, 7, 8))
    assert refs[2]["object"] == "cf3c60b1d5c004b9e601a6afbf630358f80f1d79"
    assert refs[2]["peeled_commit"] == "8a263722c7f8ce1d99b8796a3ccc0425dd1cefcf"
    assert refs[4]["object"] == "f3c81a2f14f853f369015a4f81b6695b52eec74e"
    assert refs[4]["peeled_commit"] == "7274367f2b05fb3c9ed9f876a59be647703bbdc2"
    assert refs[7]["object"] == "bacad707d3779bfa10957e18cb4c69b1a7f0cbce"
    assert refs[8]["object"] == "f35a7dca8f6bc39e9748e79b6712fab4a203396b"


def test_permission_layout_uses_one_narrow_process_owned_capability_root() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    layout = contract["permission_safe_layout"]
    attempt = contract["fresh_localization_attempt"]

    assert layout["secure_parent"] == "/secure"
    assert (layout["secure_parent_owner"], layout["secure_parent_mode"]) == (
        "root:root",
        "0755",
    )
    assert layout["secure_parent_process_writable"] is False
    assert layout["secure_root"] == "/secure/p30"
    assert (layout["secure_root_owner"], layout["secure_root_mode"]) == (
        "root:root",
        "0755",
    )
    assert layout["secure_root_process_writable"] is False
    assert layout["transport_root"] == "/secure/p30/transport-20260906-03"
    assert (layout["transport_owner"], layout["transport_mode"]) == (
        "ubuntu:ubuntu",
        "0700",
    )
    assert layout["transport_is_only_precreated_process_writable_directory"] is True
    assert layout["localization_ledger_root"] == "/var/lib/optimizationml-p30-20260906-03"
    assert (
        layout["localization_ledger_root_owner"],
        layout["localization_ledger_root_mode"],
    ) == ("root:root", "0700")
    assert (
        layout[
            "localization_ledger_root_is_provisioned_with_only_deferred_journal_before_source_phase"
        ]
        is True
    )
    assert layout["localization_ledger_root_is_not_a_checkout_mount_or_transfer_capability"] is True
    assert (
        layout[
            "localization_ledger_root_is_componentwise_nofollow_with_mode_only_access_acl_and_no_default_named_or_frozen_xattr_acl"
        ]
        is True
    )
    assert layout["deferred_journal_root"] == ("/var/lib/optimizationml-p30-20260906-03/deferred")
    assert (layout["deferred_journal_root_owner"], layout["deferred_journal_root_mode"]) == (
        "root:root",
        "0700",
    )
    assert layout["deferred_journal_is_not_accessible_to_the_uid1000_bundle_verifier"] is True
    assert layout["deferred_journal_files_are_root_owned_o_excl_nofollow_and_preregistered"]
    assert layout["deferred_journal_file_mode"] == "0400"
    assert (
        layout["deferred_publication_best_effort_attempts_to_retain_stdout_stderr_and_exit_status"]
        is True
    )
    assert (
        layout[
            "deferred_publication_complete_triplet_is_guaranteed_only_when_artifact_finalization_succeeds"
        ]
        is True
    )
    assert layout["deferred_publication_success_stdout_stderr_exact_bytes_utf8"] == ""
    assert layout["deferred_publication_success_exit_status_exact_bytes_utf8"] == "0\n"
    assert layout["deferred_publication_failure_sentinel_suffix"] == (
        ".publication-failure.exit-status.txt"
    )
    assert layout["deferred_publication_failure_sentinel_exists_only_on_failure"] is True
    prepare_admission_journal = [
        ".deferred-p30-prepare-pre-attempt-admission.exit-status.txt",
        ".deferred-p30-prepare-pre-attempt-admission.stderr.log",
        ".deferred-p30-prepare-pre-attempt-admission.stdout.log",
    ]
    assert layout["deferred_journal_exact_before_prepare_admission"] == []
    assert layout["deferred_journal_exact_after_successful_prepare_pre_attempt_admission"] == (
        prepare_admission_journal
    )
    assert layout[
        "after_admission_journal_validation_accepts_only_the_exact_producer_status_argument"
    ]
    assert layout[
        "failed_prepare_admission_retains_and_returns_its_original_status_when_finalization_succeeds"
    ]
    assert layout[
        "after_prepare_and_after_runtime_journal_stages_require_prepare_admission_status_zero"
    ]
    prepare_journal = [
        ".deferred-p30-prepare-attempt-layout.exit-status.txt",
        ".deferred-p30-prepare-attempt-layout.publication.exit-status.txt",
        ".deferred-p30-prepare-attempt-layout.publication.stderr.log",
        ".deferred-p30-prepare-attempt-layout.publication.stdout.log",
        ".deferred-p30-prepare-attempt-layout.stderr.log",
        ".deferred-p30-prepare-attempt-layout.stdout.log",
        *prepare_admission_journal,
    ]
    assert layout["deferred_journal_exact_after_successful_prepare_layout_publication"] == (
        prepare_journal
    )
    assert layout["deferred_journal_exact_after_successful_runtime_receipt_replay_publication"] == [
        ".deferred-p30-pre-marker-runtime-receipt-replay.exit-status.txt",
        ".deferred-p30-pre-marker-runtime-receipt-replay.publication.exit-status.txt",
        ".deferred-p30-pre-marker-runtime-receipt-replay.publication.stderr.log",
        ".deferred-p30-pre-marker-runtime-receipt-replay.publication.stdout.log",
        ".deferred-p30-pre-marker-runtime-receipt-replay.stderr.log",
        ".deferred-p30-pre-marker-runtime-receipt-replay.stdout.log",
        *prepare_journal,
    ]
    assert layout["sealed_orchestrator"] == (
        "/var/lib/optimizationml-p30-20260906-03/run_p30_umask_bound_control_seal.sh"
    )
    assert (layout["sealed_orchestrator_owner"], layout["sealed_orchestrator_mode"]) == (
        "root:root",
        "0555",
    )
    assert layout["sealed_orchestrator_is_o_excl_nofollow_stable_copy_of_reviewed_control_bytes"]
    assert layout[
        "sealed_orchestrator_is_created_only_after_source_receipt_and_control_checkout_review"
    ]
    assert layout["prepare_and_localization_invoke_only_sealed_orchestrator_with_bin_bash_p"]
    assert layout["localization_ledger_root_exact_before_prepare_inventory"] == [
        "deferred",
        "run_p30_umask_bound_control_seal.sh",
    ]
    assert layout["prepare_runtime_invocation_token"] == (
        "/var/lib/optimizationml-p30-20260906-03/prepare-runtime.invoked"
    )
    assert (
        layout["prepare_runtime_invocation_token_owner"],
        layout["prepare_runtime_invocation_token_mode"],
        layout["prepare_runtime_invocation_token_exact_bytes_utf8"],
    ) == ("root:root", "0400", "prepare-runtime\n")
    assert layout["prepare_runtime_invocation_token_is_o_excl_nofollow"] is True
    assert (
        layout[
            "prepare_runtime_invocation_token_transaction_uses_supplied_expected_orchestrator_digest_before_general_environment_validation_or_source_receipt_replay"
        ]
        is True
    )
    assert (
        layout[
            "prepare_pre_attempt_admission_is_one_root_journal_triplet_covering_source_receipt_replay_and_every_pre_attempt_validator_after_a_successful_prepare_token_transaction"
        ]
        is True
    )
    assert (
        layout[
            "partial_internal_prepare_token_transaction_leaves_explicit_incomplete_nonrunnable_state"
        ]
        is True
    )
    assert layout[
        "localization_ledger_root_exact_after_prepare_dispatch_before_localization_inventory"
    ] == [
        "deferred",
        "prepare-runtime.invoked",
        "run_p30_umask_bound_control_seal.sh",
    ]
    assert layout[
        "localization_ledger_root_exact_during_source_replay_and_runtime_receipt_creation_inventory"
    ] == [
        "deferred",
        "prepare-runtime.invoked",
        "run_p30_umask_bound_control_seal.sh",
    ]
    assert layout["localization_invocation_token"] == (
        "/var/lib/optimizationml-p30-20260906-03/run-localization.invoked"
    )
    assert layout["localization_invocation_token_exact_bytes_utf8"] == "run-localization\n"
    assert layout["localization_authorization_token"] == (
        "/var/lib/optimizationml-p30-20260906-03/localization.authorized"
    )
    assert (
        layout["localization_authorization_token_exact_bytes_utf8"] == "localization-authorized\n"
    )
    assert layout["localization_terminal_status"] == (
        "/var/lib/optimizationml-p30-20260906-03/run-localization.exit-status.txt"
    )
    assert (
        layout["localization_terminal_status_owner"],
        layout["localization_terminal_status_mode"],
    ) == ("root:root", "0400")
    assert layout[
        "localization_ledger_root_exact_after_dispatch_before_authorization_inventory"
    ] == [
        "deferred",
        "prepare-runtime.invoked",
        "run-localization.invoked",
        "run_p30_umask_bound_control_seal.sh",
    ]
    assert layout["localization_ledger_root_exact_authorized_inventory"] == [
        "deferred",
        "localization.authorized",
        "prepare-runtime.invoked",
        "run-localization.invoked",
        "run_p30_umask_bound_control_seal.sh",
    ]
    assert layout[
        "localization_ledger_root_exact_after_successful_terminal_status_preauthorization_failure_inventory"
    ] == [
        "deferred",
        "prepare-runtime.invoked",
        "run-localization.exit-status.txt",
        "run-localization.invoked",
        "run_p30_umask_bound_control_seal.sh",
    ]
    assert layout[
        "localization_ledger_root_exact_after_successful_terminal_status_authorized_inventory"
    ] == [
        "deferred",
        "localization.authorized",
        "prepare-runtime.invoked",
        "run-localization.exit-status.txt",
        "run-localization.invoked",
        "run_p30_umask_bound_control_seal.sh",
    ]
    assert layout[
        "source_receipt_creation_operator_precondition_is_only_empty_deferred_journal_in_root_ledger"
    ]
    assert not layout["source_receipt_creation_uid1000_verifier_mechanically_inspects_root_ledger"]
    assert (
        layout[
            "source_receipt_replay_requires_prepare_invocation_present_and_localization_files_absent"
        ]
        is True
    )
    assert (
        layout[
            "runtime_receipt_creation_operator_precondition_is_prepare_invocation_present_and_localization_files_absent"
        ]
        is True
    )
    assert not layout["runtime_receipt_creation_uid1000_verifier_mechanically_inspects_root_ledger"]
    assert (
        layout[
            "runtime_receipt_replay_requires_prepare_and_localization_invocations_present_and_authorization_and_status_absent"
        ]
        is True
    )
    assert (
        layout[
            "authorization_is_created_once_only_after_reviewed_runtime_receipt_and_all_pre_marker_checks"
        ]
        is True
    )
    assert layout[
        "host_post_authorization_boundary_validates_prepare_invocation_localization_invocation_and_authorization_present_and_exact_before_dispatch"
    ]
    assert layout["containerized_p27_localizer_reads_root_ledger"] is False
    assert layout[
        "each_localization_token_creation_atomically_revalidates_deferred_root_and_sealed_orchestrator_before_o_excl"
    ]
    assert layout[
        "terminal_status_creation_atomically_revalidates_deferred_root_and_sealed_orchestrator_before_o_excl"
    ]
    assert layout["token_creation_requires_deferred_root_root_root_0700_no_acl_and_stable_identity"]
    assert layout[
        "token_creation_requires_sealed_orchestrator_root_root_0555_no_acl_stable_identity_and_reviewed_digest"
    ]
    assert layout[
        "after_successful_invocation_token_transaction_dispatcher_attempts_one_terminal_status_transaction_even_if_evidence_logging_or_early_predicates_fail"
    ]
    assert layout[
        "complete_validated_terminal_status_is_guaranteed_only_when_terminal_status_transaction_succeeds"
    ]
    assert layout[
        "failed_terminal_status_transaction_is_terminal_nonrunnable_and_may_leave_status_absent_or_created_but_unvalidated"
    ]
    assert layout["second_authorization_creation_or_localization_rerun_allowed"] is False
    assert layout["source_phase_exact_secure_root_child_inventory"] == ["transport-20260906-03"]
    assert layout["runtime_phase_exact_secure_root_child_inventory"] == [
        "attempt-20260906-03",
        "execution-20260906-03",
        "transport-20260906-03",
    ]
    assert layout["extra_capability_or_namespace_sibling_allowed"] is False
    assert layout["control_checkout"].startswith(f"{layout['transport_root']}/")
    assert layout["authority_checkout"].startswith(f"{layout['transport_root']}/")
    assert layout["control_and_authority_must_be_absent_before_source_receipt_creation"] is True
    assert layout["source_receipt_creation_requires_authority_absent_for_entire_creation"] is True
    assert layout["source_receipt_replay_authority_states_allowed"] == [
        "absent",
        "exact_canonical_locked_authority",
    ]
    assert layout["source_receipt_replay_requires_exact_existing_control_checkout"] is True
    assert (
        layout[
            "source_receipt_replay_existing_authority_requires_layout_commit_tree_and_indirection_validation"
        ]
        is True
    )
    assert layout["source_receipt_replay_may_not_create_mutate_or_relocate_authority"] is True
    assert (
        layout[
            "attempt_and_execution_roots_must_be_absent_during_source_receipt_creation_and_replay"
        ]
        is True
    )
    assert layout["precreate_empty_control_or_authority_directory_allowed"] is False
    assert layout["bind_copy_symlink_or_privileged_relocation_allowed"] is False
    assert (
        layout["control_and_authority_post_creation_owner"],
        layout["control_and_authority_post_creation_mode"],
    ) == ("ubuntu:ubuntu", "0700")
    assert layout["control_and_authority_post_creation_uid"] == 1000
    assert layout["control_and_authority_post_creation_gid"] == 1000
    assert (
        layout["control_and_authority_are_nonsymlink_directories_on_transport_device_and_mount_id"]
        is True
    )
    assert (
        layout[
            "control_and_authority_have_no_extended_or_named_access_no_default_and_no_frozen_xattr_acl"
        ]
        is True
    )
    assert layout["componentwise_o_directory_o_nofollow_required"] is True
    assert layout["ancestor_nofollow_components_with_stable_device_and_inode_only"] == [
        "/",
    ]
    assert layout["full_owner_mode_acl_mount_authority_boundary_begins_at"] == "/secure"
    assert layout["ancestor_owner_mode_acl_or_mount_authority_claimed"] is False
    assert layout["access_acl_must_equal_mode_bits_without_named_entries"] is True
    assert layout["default_acl_entries_allowed"] is False
    assert layout["forbidden_acl_xattrs"] == [
        "system.posix_acl_access",
        "system.posix_acl_default",
        "system.nfs4_acl",
        "system.richacl",
    ]
    assert (
        layout[
            "secure_parent_namespace_transport_execution_attempt_and_evidence_must_share_device_and_mount_id"
        ]
        is True
    )
    assert attempt["attempt_root"] == "/secure/p30/attempt-20260906-03"
    assert attempt["evidence_root"] == "/secure/p30/attempt-20260906-03/evidence"
    assert (attempt["root_post_creation_owner"], attempt["root_post_creation_mode"]) == (
        "root:root",
        "0555",
    )
    assert attempt["root_is_not_process_writable_after_atomic_preparation"] is True
    assert (attempt["evidence_owner"], attempt["evidence_mode"]) == (
        "root:root",
        "0555",
    )
    assert attempt[
        "evidence_parent_and_evidence_root_are_root_owned_nonwritable_and_prevent_uid1000_transport_process_path_replacement"
    ]
    assert attempt[
        "root_owned_nonwritable_evidence_prevents_uid1000_transport_process_mutation_but_not_trusted_root_container_dac_override"
    ]
    assert (attempt["evidence_children_owner"], attempt["evidence_children_sealed_mode"]) == (
        "root:root",
        "0444",
    )
    assert attempt[
        "evidence_children_are_created_o_excl_and_sealed_only_after_their_producer_closes"
    ]
    assert attempt[
        "evidence_world_readability_is_intentional_and_no_evidence_child_may_contain_credentials"
    ]
    assert attempt["attempt_and_evidence_componentwise_o_directory_o_nofollow_required"] is True
    assert (
        attempt["attempt_and_evidence_access_acl_must_equal_mode_bits_without_named_entries"]
        is True
    )
    assert attempt["attempt_and_evidence_default_acl_entries_allowed"] is False
    assert attempt["attempt_and_evidence_forbidden_acl_xattrs"] == [
        "system.posix_acl_access",
        "system.posix_acl_default",
        "system.nfs4_acl",
        "system.richacl",
    ]
    assert attempt["attempt_and_evidence_must_share_namespace_device_and_mount_id"] is True
    assert attempt["attempt_and_evidence_must_not_be_mountpoints"] is True
    assert (
        attempt[
            "attempt_and_evidence_must_match_reviewed_identity_authority_acl_device_and_mount_id_at_runtime_receipt_creation_replay_and_explicit_localization_checks"
        ]
        is True
    )
    assert (
        attempt[
            "verifier_holds_open_attempt_and_evidence_descriptors_through_each_runtime_receipt_creation_or_replay"
        ]
        is True
    )
    assert (
        attempt[
            "host_reopens_attempt_and_evidence_componentwise_nofollow_at_the_explicit_layout_and_receipt_binding_checks"
        ]
        is True
    )
    assert (
        attempt["host_descriptors_remain_open_across_entire_marker_to_localizer_process_transition"]
        is False
    )
    assert attempt["root_must_not_preexist_before_verified_source_receipt"] is True
    assert attempt["root_created_once_with_privilege_after_receipt_replay"] is True
    assert attempt["partial_root_creation_consumes_attempt"] is True


def test_receipts_two_file_review_and_no_training_boundary_are_exact() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    closure = contract["bundle_closure"]
    verifier = contract["bundle_verifier"]
    orchestration = contract["host_orchestration"]
    freeze = contract["two_phase_freeze"]
    boundary = contract["no_training_boundary"]

    assert closure["exact_advertised_ref_count"] == 9
    assert closure["bundle_prerequisite_count"] == 0
    assert closure["receipt_schema"] == "passive-muon-p30-control-bundle-receipt-v1"
    assert closure["receipt_files_are_external_and_not_committed"] is True
    assert (
        closure["source_receipt_required_before_authority_attempt_or_execution_root_creation"]
        is True
    )
    assert closure["p29_contract_reconstruction_required_check_count"] == 15
    assert closure["p29_outcome_reconstruction_required_check_count"] == 11
    assert closure["p28_contract_reconstruction_required_check_count"] == 12
    assert closure["p28_outcome_reconstruction_required_check_count"] == 10
    assert closure["p27_reconstruction_required_check_count"] == 23
    assert verifier["required_process_uid"] == 1000
    assert verifier["required_process_gid"] == 1000
    assert (
        verifier[
            "uid1000_ownership_is_required_for_transport_closure_control_and_receipt_mutations"
        ]
        is True
    )
    assert closure["runtime_review_receipt_creation_transport_only_transcripts"] == [
        "p30-runtime-review-receipt-creation.stdout.log",
        "p30-runtime-review-receipt-creation.stderr.log",
        "p30-runtime-review-receipt-creation.exit-status.txt",
    ]
    assert (
        closure[
            "runtime_review_receipt_creation_transcripts_are_o_excl_nofollow_and_retain_verifier_success_or_failure_when_wrapper_setup_and_finalization_succeed"
        ]
        is True
    )
    assert (
        closure[
            "runtime_review_receipt_creation_wrapper_failure_may_leave_partial_transport_transcripts_and_is_terminal"
        ]
        is True
    )
    assert (
        closure["runtime_review_receipt_creation_transcripts_may_be_published_under_evidence"]
        is False
    )
    assert (
        closure["runtime_review_receipt_creation_preserves_exact_27_file_evidence_for_later_replay"]
        is True
    )
    assert verifier["runtime_review_phase_exact_delta"] == [
        "A\texperiments/training/p30_cuda_runtime_lock.json",
        "A\texperiments/training/p30_host_attestation.json",
    ]
    assert verifier["runtime_review_exact_attempt_root_inventory"] == [
        "evidence",
        "p30-image-inspect.json",
        "p30-nvidia-smi.csv",
        "p30-running-container-inspect.json",
        "p30-running-mountinfo.txt",
        "p30-container-id.txt",
    ]
    assert verifier["runtime_review_required_prepare_layout_transcripts"] == [
        "p30-prepare-attempt-layout.stdout.log",
        "p30-prepare-attempt-layout.stderr.log",
        "p30-post-freeze-attempt-layout.stdout.log",
        "p30-post-freeze-attempt-layout.stderr.log",
    ]
    assert verifier["runtime_review_exact_prelocalization_evidence_inventory"] == [
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
    assert len(verifier["runtime_review_exact_prelocalization_evidence_inventory"]) == 27
    assert verifier["runtime_review_required_zero_exit_status_files"] == [
        "p30-prepare-attempt-layout.exit-status.txt",
        "p30-prepare-execution-snapshot-verification.exit-status.txt",
        "p30-image-inspection.exit-status.txt",
        "p30-nvidia-smi.exit-status.txt",
        "p30-container-launch.exit-status.txt",
        "p30-fresh-container-id-validation.exit-status.txt",
        "p30-running-container-inspection.exit-status.txt",
        "p30-running-mountinfo.exit-status.txt",
        "p30-freeze-runtime.exit-status.txt",
        "p30-post-freeze-attempt-layout.exit-status.txt",
    ]
    assert len(verifier["runtime_review_required_zero_exit_status_files"]) == 10
    assert verifier["runtime_review_success_exit_status_exact_bytes_utf8"] == "0\n"
    assert verifier["runtime_review_attempt_and_evidence_are_fd_anchored_nofollow"] is True
    assert (
        verifier[
            "runtime_review_attempt_and_evidence_identity_authority_acl_device_and_mount_are_stable"
        ]
        is True
    )
    assert (
        verifier[
            "runtime_review_execution_snapshot_is_fd_anchored_nofollow_and_manifest_revalidated_before_and_after_receipt"
        ]
        is True
    )
    assert verifier["runtime_review_phase_requires_exact_immutable_execution_snapshot"] is True
    assert verifier["runtime_review_attempt_root_owner_mode"] == "root:root 0555"
    assert verifier["runtime_review_evidence_root_owner_mode"] == "root:root 0555"
    assert verifier["runtime_review_evidence_children_owner_mode"] == "root:root 0444"
    assert verifier["runtime_review_evidence_children_are_o_excl_and_sealed_after_producer_close"]
    assert verifier[
        "runtime_review_evidence_is_world_readable_by_design_and_contains_no_credentials"
    ]
    assert verifier["runtime_review_execution_root_owner_mode"] == "root:root 0555"
    assert verifier["runtime_review_execution_manifest_owner_mode"] == "root:root 0444"
    assert verifier["trusted_python_executable"] == "/usr/bin/python3"
    assert verifier["trusted_python_flags"] == ["-I", "-S"]
    assert verifier["bootstrap_and_all_reconstructors_use_trusted_python_with_both_flags"] is True
    assert (
        verifier[
            "reconstructors_execute_only_from_the_verifiers_authenticated_private_bundle_bytes"
        ]
        is True
    )
    assert verifier[
        "verifier_is_the_sole_host_reconstruction_authority_in_source_and_runtime_receipts"
    ]
    assert verifier["nested_python3_shim_executes_the_same_trusted_python_with_both_flags"] is True
    assert (
        verifier["nested_reconstruction_environment_scrubs_python_and_git_control_inputs"] is True
    )
    assert verifier["absolute_git_executable"] == "/usr/bin/git"
    assert verifier["git_system_attributes_disabled_with_git_attr_nosystem"] is True
    assert (
        verifier["control_and_authority_cleanliness_includes_tracked_untracked_and_ignored_files"]
        is True
    )
    assert verifier["phase_receipt_statuses"] == {
        "source": "source_bundle_and_permission_layout_verified_before_attempt_root_creation",
        "runtime-review": (
            "runtime_review_bundle_and_permission_layout_verified_before_localization"
        ),
    }
    assert verifier["source_receipt_creation_requires_authority_absent"] is True
    assert verifier[
        "uid1000_bundle_verifier_does_not_claim_access_to_root_only_ledger_or_deferred_journal"
    ]
    assert verifier["source_phase_exact_namespace_child_inventory"] == ["transport-20260906-03"]
    assert verifier["runtime_review_phase_exact_namespace_child_inventory"] == [
        "attempt-20260906-03",
        "execution-20260906-03",
        "transport-20260906-03",
    ]
    assert (
        verifier["phase_specific_namespace_inventory_revalidated_before_and_after_transitions"]
        is True
    )
    assert (
        verifier[
            "source_receipt_replay_allows_authority_absent_or_exact_canonical_locked_authority"
        ]
        is True
    )
    assert (
        verifier[
            "source_receipt_replay_existing_authority_must_match_locked_commit_tree_layout_acl_and_indirection_guards"
        ]
        is True
    )
    assert verifier["source_receipt_replay_never_creates_mutates_or_relocates_authority"] is True
    assert verifier["source_receipt_creation_invocations_allowed"] == 1
    assert verifier["source_receipt_replay_invocations_allowed"] == 1
    assert verifier["runtime_review_receipt_creation_invocations_allowed"] == 1
    assert verifier["runtime_review_receipt_replay_invocations_allowed"] == 1
    assert (
        verifier[
            "attempt_and_execution_roots_must_remain_absent_during_source_receipt_creation_and_replay"
        ]
        is True
    )
    assert freeze["phase_two_exact_delta"] == verifier["runtime_review_phase_exact_delta"]
    assert len(orchestration["required_environment"]) == 37
    assert len(set(orchestration["required_environment"])) == 37
    assert orchestration["required_environment"][-1] == "P30_EXPECTED_CONTAINER_ID"
    assert orchestration["runtime_only_required_environment"] == [
        "P30_RUNTIME_REVIEW_COMMIT",
        "P30_RUNTIME_REVIEW_TREE",
        "P30_EXPECTED_RUNTIME_REVIEW_BUNDLE_SHA256",
        "P30_EXPECTED_RUNTIME_REVIEW_BUNDLE_BYTE_COUNT",
        "P30_EXPECTED_RUNTIME_REVIEW_RECEIPT_SHA256",
        "P30_EXPECTED_RUNTIME_LOCK_SHA256",
        "P30_EXPECTED_HOST_ATTESTATION_SHA256",
        "P30_EXPECTED_CONTAINER_ID",
    ]
    assert (
        orchestration["source_required_environment"] == orchestration["required_environment"][:-8]
    )
    assert orchestration["root_environment_handoff_executable"] == "/usr/bin/env"
    assert orchestration["root_environment_handoff_clear_environment_flag"] == "-i"
    assert orchestration["root_environment_handoff_exact_path"] == "/usr/sbin:/usr/bin:/sbin:/bin"
    assert orchestration[
        "prepare_root_environment_handoff_exact_names_equal_source_required_environment"
    ]
    assert orchestration[
        "localization_root_environment_handoff_exact_names_equal_required_environment"
    ]
    assert not orchestration["ambient_sudo_environment_inheritance_or_preserve_env_is_allowed"]
    assert orchestration[
        "root_environment_handoff_assignments_precede_bin_bash_p_and_the_sealed_orchestrator"
    ]
    assert orchestration[
        "root_environment_handoff_read_only_preflight_compares_exact_nonempty_name_value_map_count_and_root_identity_before_each_one_shot_dispatch"
    ]
    assert orchestration["root_environment_handoff_read_only_preflight_creates_no_p30_state"]
    assert orchestration[
        "sealed_orchestrator_revalidates_every_handed_off_value_inside_the_first_retained_producer_before_substantive_phase_work"
    ]
    assert orchestration["sealed_orchestrator"] == (
        "/var/lib/optimizationml-p30-20260906-03/run_p30_umask_bound_control_seal.sh"
    )
    assert (
        orchestration["sealed_orchestrator_owner"],
        orchestration["sealed_orchestrator_mode"],
    ) == (
        "root:root",
        "0555",
    )
    assert orchestration[
        "sealed_orchestrator_is_created_o_excl_from_stably_read_reviewed_control_bytes_after_source_receipt"
    ]
    assert orchestration[
        "sealed_orchestrator_is_the_required_operator_prepare_and_localization_entrypoint"
    ]
    assert orchestration[
        "executing_orchestrator_path_and_digest_are_revalidated_after_token_admission_before_stateful_phase_work"
    ]
    assert orchestration[
        "wrong_root_dispatched_script_path_may_consume_the_one_shot_token_then_must_fail_closed"
    ]
    assert orchestration["sealed_orchestrator_is_invoked_with_bin_bash_privileged_mode"]
    assert orchestration["run_localization_root_owned_invocation_ledger"] == (
        "/var/lib/optimizationml-p30-20260906-03/run-localization.invoked"
    )
    assert orchestration["run_localization_root_owned_invocation_ledger_owner"] == "root:root"
    assert orchestration["run_localization_root_owned_invocation_ledger_mode"] == "0400"
    assert (
        orchestration["run_localization_root_owned_invocation_ledger_exact_bytes_utf8"]
        == "run-localization\n"
    )
    assert orchestration["run_localization_root_owned_invocation_ledger_is_o_excl_nofollow"] is True
    assert orchestration["prepare_runtime_root_owned_invocation_ledger"] == (
        "/var/lib/optimizationml-p30-20260906-03/prepare-runtime.invoked"
    )
    assert (
        orchestration["prepare_runtime_root_owned_invocation_ledger_owner"],
        orchestration["prepare_runtime_root_owned_invocation_ledger_mode"],
        orchestration["prepare_runtime_root_owned_invocation_ledger_exact_bytes_utf8"],
    ) == ("root:root", "0400", "prepare-runtime\n")
    assert orchestration["prepare_runtime_root_owned_invocation_ledger_is_o_excl_nofollow"] is True
    assert (
        orchestration[
            "prepare_runtime_root_owned_invocation_transaction_uses_supplied_expected_orchestrator_digest_before_general_environment_validation_or_source_receipt_replay"
        ]
        is True
    )
    assert (
        orchestration[
            "source_receipt_replay_is_read_only_but_occurs_after_irreversible_prepare_admission_and_before_attempt_execution_or_container_mutation"
        ]
        is True
    )
    assert (
        orchestration[
            "prepare_runtime_dispatch_requires_exact_ledger_direct_child_inventory_plus_deferred_authority_and_sealed_orchestrator_identity_before_burning_prepare_invocation"
        ]
        is True
    )
    assert (
        orchestration["premature_localization_after_successful_prepare_token_exists_consumes_p30"]
        is True
    )
    assert (
        orchestration[
            "pre_prepare_localization_dispatch_creates_no_new_ledger_state_and_is_retryable"
        ]
        is True
    )
    assert (
        orchestration[
            "run_localization_root_owned_invocation_transaction_uses_supplied_expected_orchestrator_digest_before_general_environment_validation_evidence_or_marker_predicates"
        ]
        is True
    )
    assert (
        orchestration[
            "missing_or_malformed_supplied_expected_orchestrator_digest_may_fail_before_invocation_token_creation"
        ]
        is True
    )
    assert orchestration["run_localization_root_owned_authorization"] == (
        "/var/lib/optimizationml-p30-20260906-03/localization.authorized"
    )
    assert orchestration["run_localization_root_owned_authorization_owner"] == "root:root"
    assert orchestration["run_localization_root_owned_authorization_mode"] == "0400"
    assert (
        orchestration["run_localization_root_owned_authorization_exact_bytes_utf8"]
        == "localization-authorized\n"
    )
    assert orchestration["run_localization_root_owned_authorization_is_o_excl_nofollow"] is True
    assert (
        orchestration[
            "run_localization_authorization_is_created_only_after_reviewed_runtime_receipt_and_all_pre_marker_validators"
        ]
        is True
    )
    assert orchestration[
        "run_localization_host_post_authorization_boundary_validates_prepare_invocation_localization_invocation_and_authorization_tokens_before_dispatch"
    ]
    assert orchestration["run_localization_containerized_p27_localizer_reads_root_ledger"] is False
    assert orchestration["run_localization_root_owned_terminal_status"] == (
        "/var/lib/optimizationml-p30-20260906-03/run-localization.exit-status.txt"
    )
    assert orchestration["run_localization_root_owned_terminal_status_mode"] == "0400"
    assert (
        orchestration[
            "localization_invocation_token_transaction_includes_all_pre_and_post_validation_and_returns_success_only_to_creator"
        ]
        is True
    )
    assert (
        orchestration[
            "run_localization_dispatcher_runs_body_in_subshell_then_attempts_one_o_excl_terminal_status_transaction_only_after_successful_invocation_token_transaction"
        ]
        is True
    )
    assert (
        orchestration[
            "run_localization_complete_validated_terminal_status_is_guaranteed_only_when_terminal_status_transaction_succeeds"
        ]
        is True
    )
    assert (
        orchestration[
            "run_localization_failed_terminal_status_transaction_is_terminal_nonrunnable_and_may_leave_status_absent_or_created_but_unvalidated"
        ]
        is True
    )
    assert (
        orchestration[
            "existing_localization_invocation_token_replay_may_create_or_replace_terminal_status"
        ]
        is False
    )
    assert (
        orchestration[
            "partial_internal_invocation_token_transaction_leaves_explicit_incomplete_nonrunnable_state"
        ]
        is True
    )
    assert (
        orchestration[
            "partial_internal_prepare_token_transaction_leaves_explicit_incomplete_nonrunnable_state"
        ]
        is True
    )
    assert (
        orchestration[
            "run_localization_root_owned_invocation_ledger_once_created_remains_after_every_subsequent_success_or_failure"
        ]
        is True
    )
    assert orchestration["late_evidence_marker_is_not_the_one_shot_admission_authority"] is True
    assert (
        orchestration["runtime_receipt_replay_observes_untouched_exact_27_file_evidence_inventory"]
        is True
    )
    assert orchestration["runtime_receipt_replay_is_first_localization_body_operation"] is True
    assert orchestration[
        "runtime_environment_is_validated_inside_the_deferred_replay_producer_before_transport_validation_or_verifier_dispatch"
    ]
    assert orchestration["initial_localization_boundary_order"] == [
        "p30-pre-marker-runtime-receipt-replay",
        "p30-localization-entry-preflight",
        "p30-pre-marker-receipt-namespace-binding",
    ]
    assert (
        orchestration[
            "localization_entry_preflight_runs_only_after_runtime_replay_transcript_publication"
        ]
        is True
    )
    assert (
        orchestration["localization_entry_preflight_requires_after_runtime_deferred_journal_stage"]
        is True
    )
    assert (
        orchestration[
            "runtime_receipt_replay_root_journal_stdout_and_stderr_are_created_o_excl_before_verifier"
        ]
        is True
    )
    assert (
        orchestration[
            "runtime_receipt_replay_root_journal_exit_status_is_created_o_excl_after_verifier_before_publication"
        ]
        is True
    )
    assert orchestration["runtime_receipt_replay_deferred_journal_transcripts"] == [
        ".deferred-p30-pre-marker-runtime-receipt-replay.stdout.log",
        ".deferred-p30-pre-marker-runtime-receipt-replay.stderr.log",
        ".deferred-p30-pre-marker-runtime-receipt-replay.exit-status.txt",
    ]
    assert orchestration["runtime_receipt_replay_published_evidence_transcripts"] == [
        "p30-pre-marker-runtime-receipt-replay.stdout.log",
        "p30-pre-marker-runtime-receipt-replay.stderr.log",
        "p30-pre-marker-runtime-receipt-replay.exit-status.txt",
    ]
    assert orchestration["runtime_receipt_replay_publication_journal_transcripts"] == [
        ".deferred-p30-pre-marker-runtime-receipt-replay.publication.stdout.log",
        ".deferred-p30-pre-marker-runtime-receipt-replay.publication.stderr.log",
        ".deferred-p30-pre-marker-runtime-receipt-replay.publication.exit-status.txt",
    ]
    assert orchestration[
        "runtime_receipt_replay_publication_success_requires_empty_stdout_stderr_and_zero_status"
    ]
    assert (
        orchestration[
            "runtime_receipt_replay_evidence_publication_requires_stable_reviewed_identity_reauthentication"
        ]
        is True
    )
    assert (
        orchestration[
            "runtime_receipt_replay_transcripts_are_published_to_evidence_after_verifier_exit_for_success_or_failure_only_when_identity_reauthentication_succeeds_by_stable_nofollow_o_excl_copy"
        ]
        is True
    )
    assert (
        orchestration[
            "when_evidence_publication_succeeds_journal_and_evidence_transcript_bytes_must_match"
        ]
        is True
    )
    assert orchestration["successful_replay_exit_status_exact_bytes_utf8"] == "0\n"
    assert (
        orchestration["runtime_receipt_replay_failure_transcripts_remain_under_root_journal"]
        is True
    )
    assert (
        orchestration[
            "runtime_receipt_replay_failure_transcripts_are_published_under_evidence_only_when_identity_reauthentication_succeeds"
        ]
        is True
    )
    assert orchestration["runtime_receipt_replay_publication_failure_journal_status"] == (
        ".deferred-p30-pre-marker-runtime-receipt-replay.publication-failure.exit-status.txt"
    )
    assert (
        orchestration[
            "runtime_receipt_replay_publication_failure_status_is_o_excl_best_effort_when_evidence_cannot_be_reauthenticated_or_published_and_root_journal_is_intact"
        ]
        is True
    )
    assert (
        orchestration[
            "runtime_receipt_replay_publication_finalization_failure_returns_terminal_125"
        ]
        is True
    )
    assert (
        orchestration["runtime_receipt_replay_may_publish_into_identity_compromised_evidence"]
        is False
    )
    assert orchestration["runtime_receipt_replay_failure_creates_localization_marker"] is False
    assert (
        orchestration[
            "runtime_receipt_replay_inventory_weakening_or_premature_evidence_logging_allowed"
        ]
        is False
    )
    assert (
        orchestration[
            "phase_specific_namespace_child_inventory_is_exact_and_extra_siblings_are_rejected"
        ]
        is True
    )
    assert orchestration["trusted_python_executable"] == "/usr/bin/python3"
    assert orchestration["trusted_python_flags"] == ["-I", "-S"]
    assert (
        orchestration["bootstrap_and_reconstructor_invocations_use_trusted_python_with_both_flags"]
        is True
    )
    assert orchestration["nested_python_and_git_environment_is_scrubbed"] is True
    assert (
        orchestration[
            "independent_p30_reconstructor_creates_private_python3_and_git_shims_per_child"
        ]
        is True
    )
    assert (
        orchestration[
            "independent_p30_python3_shim_executes_the_same_resolved_interpreter_with_isolated_no_site_flags"
        ]
        is True
    )
    assert (
        orchestration["independent_p30_git_shim_executes_absolute_git_with_frozen_trust_controls"]
        is True
    )
    assert orchestration["host_direct_reconstructor_invocations_allowed"] is False
    assert (
        orchestration[
            "bundle_verifier_private_payload_must_report_p30_plus_nested_p29_contract_15_p29_outcome_11_p28_contract_12_p28_outcome_10_and_p27_23"
        ]
        is True
    )
    assert orchestration["absolute_git_executable"] == "/usr/bin/git"
    assert orchestration["git_system_attributes_disabled_with_git_attr_nosystem"] is True
    assert orchestration["clean_checkouts_include_tracked_untracked_and_ignored_files"] is True
    assert (
        orchestration[
            "every_substantive_post_state_and_post_marker_command_or_validator_is_invoked_only_after_successful_no_overwrite_logger_admission"
        ]
        is True
    )
    assert (
        orchestration[
            "complete_validator_transcript_triplet_is_guaranteed_only_when_artifact_finalization_succeeds"
        ]
        is True
    )
    assert (
        orchestration[
            "logger_admission_or_finalization_fault_is_terminal_and_may_lack_a_complete_transcript_triplet"
        ]
        is True
    )
    assert orchestration["runtime_receipt_verifier_holds_fd_anchors_during_each_invocation"] is True
    assert orchestration["localization_attempt_evidence_receipt_binding_labels"] == [
        "p30-pre-marker-receipt-namespace-binding",
        "p30-post-ingestion-receipt-namespace-binding",
        "p30-pre-localization-receipt-namespace-binding",
        "p30-post-authorization-receipt-namespace-binding",
        "p30-post-localizer-receipt-namespace-binding",
        "p30-post-sanitizer-receipt-namespace-binding",
        "p30-post-localization-receipt-namespace-binding",
    ]
    assert orchestration["localization_attempt_evidence_layout_labels"] == [
        "p30-localization-entry-preflight",
        "p30-pre-marker-snapshot-layout",
    ]
    assert orchestration["execution_snapshot_full_manifest_verification_boundaries"] == [
        "p30-prepare-execution-snapshot-verification",
        "runtime-review receipt creation",
        "p30-pre-marker-runtime-receipt-replay",
    ]
    assert (
        orchestration[
            "pre_marker_snapshot_layout_checks_top_level_inventory_and_manifest_authority_not_full_manifest_bytes"
        ]
        is True
    )
    assert (
        orchestration[
            "ordinary_logged_boundaries_do_not_implicitly_revalidate_attempt_evidence_or_full_execution_manifest"
        ]
        is True
    )
    assert (
        orchestration["localization_host_claims_one_continuous_marker_to_process_descriptor_guard"]
        is False
    )
    assert (orchestration["required_host_uid"], orchestration["required_host_gid"]) == (0, 0)
    assert (orchestration["container_process_uid"], orchestration["container_process_gid"]) == (
        0,
        0,
    )
    assert orchestration["container_run_identity_and_capability_arguments"] == [
        "--user",
        "0:0",
        "--userns",
        "host",
        "--cap-drop",
        "ALL",
        "--cap-add",
        "DAC_OVERRIDE",
    ]
    assert orchestration["container_inspect_config_user"] == "0:0"
    assert orchestration["container_inspect_userns_mode"] == "host"
    assert orchestration["container_user_namespace_remapping_allowed"] is False
    assert orchestration["container_inspect_cap_drop"] == ["ALL"]
    assert orchestration["container_inspect_cap_add"] == ["DAC_OVERRIDE"]
    assert orchestration["trusted_root_container_with_dac_override_is_inside_the_runtime_tcb"]
    assert (
        orchestration["sealed_evidence_is_claimed_immutable_against_a_malicious_root_container"]
        is False
    )
    assert orchestration[
        "scientific_artifact_progression_requires_each_new_artifact_to_be_sealed_and_validated_at_its_declared_producer_or_binding_boundary"
    ]
    assert orchestration[
        "historical_evidence_bytes_are_not_implicitly_revalidated_after_every_logged_boundary"
    ]
    assert orchestration["localization_ledger_validation_labels_in_exact_order"] == [
        "p30-localization-entry-preflight",
        "p30-post-authorization-ledger-validation",
        "p30-post-localizer-ledger-validation",
        "p30-post-sanitizer-ledger-validation",
    ]
    assert (
        orchestration[
            "localization_ledger_validation_triplets_after_successful_finalization_are_o_excl_nofollow_root_owned_and_sealed_0444"
        ]
        is True
    )
    assert orchestration[
        "retained_logger_classes_requiring_best_effort_seal_of_every_already_closed_artifact"
    ] == [
        "run_logged",
        "capture_new",
        "refresh_bound_file",
        "deferred_external_and_publication_status_creation",
    ]
    assert orchestration["artifact_finalization_failure_terminal_exit_code"] == 125
    assert (
        orchestration[
            "original_producer_exit_status_is_preserved_only_when_all_required_artifact_finalization_succeeds"
        ]
        is True
    )
    assert (
        orchestration[
            "artifact_finalization_failure_does_not_skip_sealing_other_already_closed_artifacts"
        ]
        is True
    )
    assert (
        orchestration[
            "evidence_inventory_sealing_collects_all_child_failures_before_returning_terminal_failure"
        ]
        is True
    )
    assert (
        orchestration[
            "freeze_runtime_logged_call_is_captured_with_errexit_disabled_then_checked_after_errexit_restoration"
        ]
        is True
    )
    assert orchestration[
        "root_host_best_effort_seals_each_closed_evidence_child_to_root_root_0444_before_review_or_next_boundary"
    ]
    runtime = contract["locked_base_runtime"]
    assert runtime["pinned_nanogpt_tree"] == "ca93bcd9b9c9ff32d3016e1e2556644e68bef86a"
    assert runtime["pinned_muon_tree"] == "4ea5cd8ab6ebd56a18536f06453619efcd636da0"
    assert runtime["container_run_user"] == "0:0"
    assert runtime["container_inspect_config_user"] == "0:0"
    assert runtime["container_userns_mode"] == "host"
    assert runtime["container_user_namespace_remapping_allowed"] is False
    assert runtime["container_cap_drop"] == ["ALL"]
    assert runtime["container_cap_add"] == ["DAC_OVERRIDE"]
    assert freeze["phase_two_localization_started"] is False
    assert boundary["unchanged_p27_localizer_only"] is True
    assert boundary["forward_allowed"] is False
    assert boundary["backward_allowed"] is False
    assert boundary["optimizer_step_allowed"] is False
    assert boundary["candidate_evaluation_allowed"] is False
    assert contract["terminal_routing"]["future_scientific_acquisition_authorized"] is False


def test_immutable_execution_snapshot_and_ten_mounts_are_exact() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    execution = contract["immutable_execution_snapshot"]
    runtime = contract["locked_base_runtime"]

    assert execution["root"] == "/secure/p30/execution-20260906-03"
    assert (execution["root_owner"], execution["root_mode"]) == ("root:root", "0555")
    assert (execution["directory_owner"], execution["directory_mode"]) == (
        "root:root",
        "0555",
    )
    assert execution["regular_file_modes_allowed"] == ["0444", "0555"]
    assert execution["manifest_path"] == (
        "/secure/p30/execution-20260906-03/snapshot-manifest.json"
    )
    assert execution["manifest_schema"] == "passive-muon-p30-immutable-execution-snapshot-v1"
    assert (execution["manifest_owner"], execution["manifest_mode"]) == (
        "root:root",
        "0444",
    )
    assert execution["exact_top_level_children"] == [
        "data",
        "muon",
        "nanogpt",
        "repository",
        "snapshot-manifest.json",
    ]
    assert execution["manifest_entries_cover_every_relative_path_except_manifest"] is True
    assert execution["manifest_exact_top_level_keys"] == [
        "entries",
        "schema_version",
        "source_authorities",
    ]
    assert (
        execution[
            "recursive_symlinks_hardlinked_files_special_files_nested_mounts_acls_and_xattrs_allowed"
        ]
        is False
    )
    sources = execution["directory_sources"]
    assert sources["repository"]["commit"] == "185e444afc0b44ca0a09b1bde49a6b6fa3973355"
    assert sources["repository"]["tree"] == "24f4bdac331a57bd7c1b807747d7c7fba253ee5a"
    assert sources["nanogpt"]["tree"] == "ca93bcd9b9c9ff32d3016e1e2556644e68bef86a"
    assert sources["muon"]["tree"] == "4ea5cd8ab6ebd56a18536f06453619efcd636da0"
    assert sources["data"]["original_manifest_sha256"] == (
        "1dfbe2087fb61004c6c4ee938d1be53d769f2fa66303c2533e14acb9c1d5a96d"
    )
    assert sources["data"]["original_manifest_byte_count"] == 4210
    assert set(execution["tools_adjunct_exact_files"]) == {
        "run_p27_cuda_deleted_mapping_localization.py",
        "ingest_p26_trace_off_a_failure.py",
        "p23_deterministic_cuda_shadow_trace.py",
        "sanitize_p27_cuda_deleted_mapping_localization.py",
        "p26-trace-off-a-failure.authenticated.json",
    }
    p26 = execution["tools_adjunct_exact_files"]["p26-trace-off-a-failure.authenticated.json"]
    assert p26 == {
        "sha256": "b2a92b5062988534bc424d3ecadf3977ecedc0f80aa27628d4a0457858a64c91",
        "byte_count": 66283,
        "mode": "0444",
    }
    assert execution["manifest_source_authorities"]["p26_authenticated_input"] == {
        "sha256": "b2a92b5062988534bc424d3ecadf3977ecedc0f80aa27628d4a0457858a64c91",
        "byte_count": 66283,
    }
    mounts = execution["exact_bind_mounts"]
    assert len(mounts) == execution["exact_bind_mount_count"] == 10
    assert [item["destination"] for item in mounts] == [
        "/workspace/OptimizationML",
        "/workspace/inputs/nanoGPT",
        "/workspace/inputs/muon",
        "/private/tmp/optimizationml-p22-data",
        "/Users/harry/Desktop/temp/OptimizationML/experiments/training/materialize_p22_fineweb.py",
        "/workspace/evidence/p23",
        "/mounted-host-evidence/image-inspect.json",
        "/mounted-host-evidence/running-container-inspect.json",
        "/mounted-host-evidence/running-mountinfo.txt",
        "/mounted-host-evidence/nvidia-smi.csv",
    ]
    assert [item["read_only"] for item in mounts].count(False) == 1
    assert mounts[5]["source"] == "/secure/p30/attempt-20260906-03/evidence"
    assert runtime["mounted_repository_source"] == mounts[0]["source"]
    assert runtime["mounted_nanogpt_source"] == mounts[1]["source"]
    assert runtime["mounted_muon_source"] == mounts[2]["source"]
    assert runtime["mounted_data_source"] == mounts[3]["source"]


def test_nested_reconstructions_use_private_isolated_python_and_git_shims(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _module()
    child = tmp_path / "isolated_child.py"
    child.write_text(
        """\
import json
import os
import shutil
import subprocess
import sys

nested_code = (
    "import json,os,sys; "
    "print(json.dumps({"
    "'executable': os.path.realpath(sys.executable), "
    "'isolated': sys.flags.isolated, "
    "'no_site': sys.flags.no_site}))"
)
nested = subprocess.run(
    ["python3", "-c", nested_code],
    check=False,
    capture_output=True,
    text=True,
)
git = subprocess.run(["git", "--version"], check=False, capture_output=True, text=True)
poison = ["PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "GIT_DIR", "GIT_WORK_TREE"]
nested_payload = json.loads(nested.stdout) if nested.returncode == 0 else {}
checks = {
    "outer_python_isolated": sys.flags.isolated == 1 and sys.flags.no_site == 1,
    "nested_python_isolated": (
        nested_payload.get("isolated") == 1 and nested_payload.get("no_site") == 1
    ),
    "nested_python_is_same_resolved_interpreter": (
        nested_payload.get("executable") == os.path.realpath(sys.executable)
    ),
    "poisoned_controls_absent": all(name not in os.environ for name in poison),
    "system_gitattributes_disabled": os.environ.get("GIT_ATTR_NOSYSTEM") == "1",
    "private_git_shim_works": git.returncode == 0 and shutil.which("git") != "/usr/bin/git",
}
print(json.dumps({"internally_consistent": all(checks.values()), "checks": checks}))
""",
        encoding="utf-8",
    )
    for name in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "GIT_DIR", "GIT_WORK_TREE"):
        monkeypatch.setenv(name, f"poison-{name}")

    assert Path("/usr/bin/git") == module.TRUSTED_GIT
    assert module._run_reconstruction(str(child)) == (True, 6, 6, 0)
    assert os.environ["PYTHONPATH"] == "poison-PYTHONPATH"


def test_workflow_enters_prior_authorities_only_through_hardened_p30_reconstructor() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "persist-credentials: false" in workflow
    assert "Materialize exact frozen phase branch refs" in workflow
    for branch in (
        "p29-permission-safe-bundle-localization-bridge",
        "p28-bundle-complete-localization-bridge",
        "p27-cuda-deleted-mapping-localization",
    ):
        assert f"git update-ref refs/heads/{branch}" in workflow
        assert f"refs/remotes/origin/{branch}" in workflow
    assert "grep -Ei '^include(if)?\\.'" in workflow
    assert "uv run --locked python -I -S" in workflow
    assert "scripts/reconstruct_p30_umask_bound_control_seal.py" in workflow
    assert "/usr/bin/python3 -I -S scripts/reconstruct_p28" not in workflow
    assert "/usr/bin/python3 -I -S scripts/reconstruct_p27" not in workflow


def test_every_contract_leaf_is_bound_by_independent_digest(tmp_path: Path) -> None:
    module = _module()

    def successful_reconstruction(path: str) -> tuple[bool, int, int, int]:
        if "p28_bundle_complete_localization_bridge_outcome" in path:
            return True, 10, 10, 0
        if "p28_bundle_complete_localization_bridge.py" in path:
            return True, 12, 12, 0
        return True, 23, 23, 0

    module._run_reconstruction = successful_reconstruction
    canonical = json.loads(CONTRACT.read_text(encoding="utf-8"))
    mutated_path = tmp_path / "mutated.json"

    for path in _leaves(canonical):
        mutated = copy.deepcopy(canonical)
        _replace(mutated, path, _mutation(_at(mutated, path)))
        _write(mutated_path, mutated)
        assert module.reconstruct(mutated_path)["internally_consistent"] is False, path


def test_duplicate_nonfinite_and_noncanonical_json_fail_closed(tmp_path: Path) -> None:
    module = _module()

    def successful_reconstruction(path: str) -> tuple[bool, int, int, int]:
        if "p28_bundle_complete_localization_bridge_outcome" in path:
            return True, 10, 10, 0
        if "p28_bundle_complete_localization_bridge.py" in path:
            return True, 12, 12, 0
        return True, 23, 23, 0

    module._run_reconstruction = successful_reconstruction

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"status":"first","status":"second"}\n', encoding="utf-8")
    assert module.reconstruct(duplicate)["internally_consistent"] is False

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"status":NaN}\n', encoding="utf-8")
    assert module.reconstruct(nonfinite)["internally_consistent"] is False

    compact = tmp_path / "compact.json"
    compact.write_text(json.dumps(json.loads(CONTRACT.read_text(encoding="utf-8"))))
    assert module.reconstruct(compact)["internally_consistent"] is False
