#!/usr/bin/env python3
"""Independently reconstruct P28's bundle-complete localization bridge contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections.abc import Mapping
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = (
    ROOT / "experiments/training/p28_bundle_complete_localization_bridge_contract.json"
)
EXPECTED_CONTRACT_SHA256 = "6a5005f902d937f0edbec3394e6205e482fe6f6fea45f986ac873fd6ab2e8718"
EXPECTED_SCHEMA = "passive-muon-p28-bundle-complete-localization-bridge-contract-v1"
EXPECTED_STATUS = "frozen_pre_transfer_pre_container_no_training"
EXPECTED_BRANCH = "p28-bundle-complete-localization-bridge"
P27_OUTCOME_PATH = "results/summaries/p27_cuda_deleted_mapping_localization_outcome.json"
P27_OUTCOME_SHA256 = "d87df6cc792dfa0a0d6a2a982395953ba2ef87aa2086266d9d3c59abb3427d0e"
P27_OUTCOME_COMMIT = "ec63550331925ded158e3f389e294e4d1f12db3a"
P27_OUTCOME_TREE = "d8efa72fda9ee41fde0b5d15b126aa87397cd48d"
P27_TAG_OBJECT = "c88b98eae9be2458abde45b05d3dccfe09c0c7ed"
P26_CHECKPOINT_OBJECT = "bacad707d3779bfa10957e18cb4c69b1a7f0cbce"
P26_CHECKPOINT_COMMIT = "5429da23ff18888daa2312c530a4587780484d8b"
P26_SOURCE_OBJECT = "f35a7dca8f6bc39e9748e79b6712fab4a203396b"
P26_SOURCE_COMMIT = "185e444afc0b44ca0a09b1bde49a6b6fa3973355"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

EXPECTED_TOP_LEVEL_ORDER = [
    "schema_version",
    "status",
    "branch",
    "purpose",
    "claim_boundary",
    "terminal_parent",
    "unchanged_p27_authorities",
    "required_bundle_refs",
    "bundle_closure",
    "bundle_verifier",
    "host_orchestration",
    "fresh_localization_attempt",
    "locked_base_runtime",
    "two_phase_freeze",
    "no_training_boundary",
    "execution_order",
    "terminal_routing",
    "forbidden_changes",
]

EXPECTED_REQUIRED_REFS = [
    "refs/heads/p28-bundle-complete-localization-bridge",
    "refs/heads/p27-cuda-deleted-mapping-localization",
    "refs/tags/p27-cuda-deleted-mapping-localization-diagnostic",
    "refs/tags/p26-permission-safe-acquisition-checkpoint",
    "refs/tags/p26-attempt02-acquisition-source",
]
EXPECTED_ORCHESTRATOR_ENVIRONMENT = [
    "P28_ATTEMPT_ID",
    "P28_GPU_UUID",
    "P28_AUTHORITY_REPO",
    "P28_NANOGPT_HOST",
    "P28_MUON_HOST",
    "P28_DATA_HOST",
    "P28_SOURCE_FREEZE_COMMIT",
    "P28_SOURCE_FREEZE_TREE",
    "P28_EXPECTED_SOURCE_BUNDLE_SHA256",
    "P28_EXPECTED_SOURCE_BUNDLE_BYTE_COUNT",
    "P28_EXPECTED_SOURCE_RECEIPT_SHA256",
    "P28_EXPECTED_CONTRACT_SHA256",
    "P28_EXPECTED_ORCHESTRATOR_SHA256",
    "P28_EXPECTED_RECONSTRUCTOR_SHA256",
    "P28_EXPECTED_BUNDLE_VERIFIER_SHA256",
    "P28_EXPECTED_P27_CONTRACT_SHA256",
    "P28_EXPECTED_P27_RECONSTRUCTOR_SHA256",
    "P28_EXPECTED_P27_LOCALIZER_SHA256",
    "P28_EXPECTED_INGESTER_SHA256",
    "P28_EXPECTED_P23_CORE_SHA256",
    "P28_EXPECTED_P27_SANITIZER_SHA256",
    "P28_RUNTIME_REVIEW_COMMIT",
    "P28_RUNTIME_REVIEW_TREE",
    "P28_EXPECTED_RUNTIME_REVIEW_BUNDLE_SHA256",
    "P28_EXPECTED_RUNTIME_REVIEW_BUNDLE_BYTE_COUNT",
    "P28_EXPECTED_RUNTIME_REVIEW_RECEIPT_SHA256",
    "P28_EXPECTED_RUNTIME_LOCK_SHA256",
    "P28_EXPECTED_HOST_ATTESTATION_SHA256",
]


class DuplicateKeyError(ValueError):
    """Raised when purported canonical JSON repeats a key."""


def _strict_json(data: bytes) -> object:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise DuplicateKeyError(key)
            result[key] = value
        return result

    def nonfinite(token: str) -> object:
        raise ValueError(f"nonfinite JSON constant: {token}")

    return json.loads(data, object_pairs_hook=pairs, parse_constant=nonfinite)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


@cache
def _git(*arguments: str) -> bytes | None:
    try:
        return subprocess.run(["git", *arguments], cwd=ROOT, check=True, capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_text(*arguments: str) -> str | None:
    value = _git(*arguments)
    return value.decode().strip() if value is not None else None


def _file_hashes_exact(authorities: Mapping[str, object]) -> bool:
    for item in authorities.values():
        entry = _mapping(item)
        path_value = entry.get("path")
        expected = entry.get("sha256")
        if not isinstance(path_value, str) or not isinstance(expected, str):
            continue
        try:
            actual = _sha256((ROOT / path_value).read_bytes())
        except OSError:
            return False
        if actual != expected:
            return False
    return True


def _run_p27_reconstruction() -> tuple[bool, int, int, int]:
    try:
        process = subprocess.run(
            ["python3", "scripts/reconstruct_p27_cuda_deleted_mapping_localization.py"],
            cwd=ROOT,
            check=False,
            capture_output=True,
        )
        parsed = _strict_json(process.stdout)
    except (OSError, ValueError, json.JSONDecodeError, DuplicateKeyError):
        return False, 0, 0, 0
    result = _mapping(parsed)
    checks = _mapping(result.get("checks"))
    true_count = sum(value is True for value in checks.values())
    false_count = sum(value is False for value in checks.values())
    return (
        process.returncode == 0 and result.get("internally_consistent") is True,
        len(checks),
        true_count,
        false_count,
    )


def _run_p27_outcome_reconstruction() -> tuple[bool, int, int, int]:
    try:
        process = subprocess.run(
            [
                "python3",
                "scripts/reconstruct_p27_cuda_deleted_mapping_localization_outcome.py",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
        )
        parsed = _strict_json(process.stdout)
    except (OSError, ValueError, json.JSONDecodeError, DuplicateKeyError):
        return False, 0, 0, 0
    result = _mapping(parsed)
    checks = _mapping(result.get("checks"))
    true_count = sum(value is True for value in checks.values())
    false_count = sum(value is False for value in checks.values())
    return (
        process.returncode == 0 and result.get("internally_consistent") is True,
        len(checks),
        true_count,
        false_count,
    )


def reconstruct(contract_path: Path = DEFAULT_CONTRACT) -> dict[str, object]:
    """Rebuild repository facts and fail closed on any contract-byte change."""

    try:
        contract_bytes = contract_path.read_bytes()
        parsed = _strict_json(contract_bytes)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError, DuplicateKeyError):
        contract_bytes = b""
        parsed = None
    contract = _mapping(parsed)
    terminal = _mapping(contract.get("terminal_parent"))
    authorities = _mapping(contract.get("unchanged_p27_authorities"))
    closure = _mapping(contract.get("bundle_closure"))
    verifier = _mapping(contract.get("bundle_verifier"))
    orchestration = _mapping(contract.get("host_orchestration"))
    attempt = _mapping(contract.get("fresh_localization_attempt"))
    runtime = _mapping(contract.get("locked_base_runtime"))
    freeze = _mapping(contract.get("two_phase_freeze"))
    boundary = _mapping(contract.get("no_training_boundary"))
    routing = _mapping(contract.get("terminal_routing"))

    required_refs = contract.get("required_bundle_refs")
    required_ref_entries = required_refs if isinstance(required_refs, list) else []
    required_ref_names = [_mapping(item).get("ref") for item in required_ref_entries]

    p27_outcome_blob = _git("show", f"{P27_OUTCOME_COMMIT}:{P27_OUTCOME_PATH}")
    p27_ok, p27_count, p27_true, p27_false = _run_p27_reconstruction()
    p27_outcome_ok, p27_outcome_count, p27_outcome_true, p27_outcome_false = (
        _run_p27_outcome_reconstruction()
    )

    checks = {
        "canonical_contract_bytes_schema_status_and_closed_top_level": (
            isinstance(parsed, Mapping)
            and list(parsed) == EXPECTED_TOP_LEVEL_ORDER
            and contract_bytes == (json.dumps(parsed, indent=2) + "\n").encode()
            and SHA256_RE.fullmatch(EXPECTED_CONTRACT_SHA256) is not None
            and _sha256(contract_bytes) == EXPECTED_CONTRACT_SHA256
            and contract.get("schema_version") == EXPECTED_SCHEMA
            and contract.get("status") == EXPECTED_STATUS
            and contract.get("branch") == EXPECTED_BRANCH
        ),
        "terminal_p27_commit_tree_tag_outcome_and_stop_exact": (
            _git_text("rev-parse", f"{P27_OUTCOME_COMMIT}^{{tree}}") == P27_OUTCOME_TREE
            and _git_text("rev-parse", "refs/heads/p27-cuda-deleted-mapping-localization")
            == P27_OUTCOME_COMMIT
            and _git_text("cat-file", "-t", P27_TAG_OBJECT) == "tag"
            and _git_text("rev-parse", "refs/tags/p27-cuda-deleted-mapping-localization-diagnostic")
            == P27_TAG_OBJECT
            and _git_text(
                "rev-parse",
                "refs/tags/p27-cuda-deleted-mapping-localization-diagnostic^{}",
            )
            == P27_OUTCOME_COMMIT
            and p27_outcome_blob is not None
            and _sha256(p27_outcome_blob) == P27_OUTCOME_SHA256
            and terminal.get("outcome_commit") == P27_OUTCOME_COMMIT
            and terminal.get("outcome_tree") == P27_OUTCOME_TREE
            and terminal.get("attempt_is_terminal") is True
            and terminal.get("attempt_reuse_allowed") is False
            and terminal.get("container_created") is False
            and terminal.get("localization_invocation_consumed") is False
        ),
        "p26_annotated_tag_objects_and_peels_exact": (
            _git_text("cat-file", "-t", P26_CHECKPOINT_OBJECT) == "tag"
            and _git_text("rev-parse", "refs/tags/p26-permission-safe-acquisition-checkpoint")
            == P26_CHECKPOINT_OBJECT
            and _git_text("rev-parse", "refs/tags/p26-permission-safe-acquisition-checkpoint^{}")
            == P26_CHECKPOINT_COMMIT
            and _git_text("cat-file", "-t", P26_SOURCE_OBJECT) == "tag"
            and _git_text("rev-parse", "refs/tags/p26-attempt02-acquisition-source")
            == P26_SOURCE_OBJECT
            and _git_text("rev-parse", "refs/tags/p26-attempt02-acquisition-source^{}")
            == P26_SOURCE_COMMIT
        ),
        "unchanged_p27_source_hashes_exact": (
            len(authorities) == 12
            and _file_hashes_exact(authorities)
            and _mapping(authorities.get("p26_checkpoint")).get("annotated_tag_object")
            == P26_CHECKPOINT_OBJECT
            and _mapping(authorities.get("p26_acquisition_source")).get("annotated_tag_object")
            == P26_SOURCE_OBJECT
        ),
        "exact_five_required_bundle_refs_and_objects": (
            required_ref_names == EXPECTED_REQUIRED_REFS
            and len(required_ref_entries) == 5
            and _mapping(required_ref_entries[1]).get("object") == P27_OUTCOME_COMMIT
            and _mapping(required_ref_entries[2]).get("object") == P27_TAG_OBJECT
            and _mapping(required_ref_entries[2]).get("peeled_commit") == P27_OUTCOME_COMMIT
            and _mapping(required_ref_entries[3]).get("object") == P26_CHECKPOINT_OBJECT
            and _mapping(required_ref_entries[3]).get("peeled_commit") == P26_CHECKPOINT_COMMIT
            and _mapping(required_ref_entries[4]).get("object") == P26_SOURCE_OBJECT
            and _mapping(required_ref_entries[4]).get("peeled_commit") == P26_SOURCE_COMMIT
        ),
        "two_fresh_empty_bare_closures_and_external_receipts_exact": (
            closure.get("transport_root") == "/secure/p28/transport-20260906-01"
            and closure.get("source_bundle")
            == "/secure/p28/transport-20260906-01/p28_source.bundle"
            and closure.get("source_closure_repository")
            == "/secure/p28/transport-20260906-01/p28_source_closure.git"
            and closure.get("source_receipt")
            == "/secure/p28/transport-20260906-01/p28_source_bundle_receipt.json"
            and closure.get("runtime_review_bundle")
            == "/secure/p28/transport-20260906-01/p28_runtime_review.bundle"
            and closure.get("runtime_review_closure_repository")
            == "/secure/p28/transport-20260906-01/p28_runtime_review_closure.git"
            and closure.get("runtime_review_receipt")
            == "/secure/p28/transport-20260906-01/p28_runtime_review_bundle_receipt.json"
            and closure.get("control_checkout") == "/secure/p28/control-source"
            and closure.get("authority_checkout") == "/secure/p28/authority-185e444"
            and closure.get("receipt_schema") == "passive-muon-p28-control-bundle-receipt-v1"
            and closure.get("receipt_creation_cli") == "--receipt-output"
            and closure.get("receipt_replay_cli")
            == "--verify-existing-receipt plus --expected-receipt-sha256"
            and closure.get("fresh_empty_bare_repository_required_for_each_phase") is True
            and closure.get("exact_advertised_ref_count") == 5
            and closure.get("bundle_prerequisite_count") == 0
            and closure.get("receipt_files_are_external_and_not_committed") is True
            and closure.get("receipt_outputs_are_o_excl_no_overwrite") is True
            and closure.get("bundle_private_authenticated_read_only_copy_required") is True
            and closure.get("private_copy_required_mode") == "0400"
            and closure.get("private_copy_post_use_hash_and_stat_revalidation_required") is True
            and closure.get("source_receipt_required_before_attempt_root_creation") is True
            and closure.get("authority_checkout_must_be_absent_until_source_receipt_reviewed")
            is True
            and closure.get("authority_checkout_constructed_only_from_verified_source_closure")
            is True
            and closure.get("authority_checkout_detached_clean_without_object_indirections") is True
            and closure.get("runtime_review_receipt_required_before_localization_marker") is True
            and closure.get("p27_reconstruction_required_check_count") == 23
            and closure.get("p27_reconstruction_required_true_count") == 23
            and closure.get("p27_reconstruction_required_false_count") == 0
        ),
        "bundle_verifier_is_final_hash_bound_and_fail_closed": (
            verifier.get("source_path") == "scripts/verify_p28_control_bundle.py"
            and verifier.get("bootstrap_execution_path")
            == "/secure/p28/transport-20260906-01/verify_p28_control_bundle.py"
            and verifier.get("checked_in_control_verifier_path")
            == "/secure/p28/control-source/scripts/verify_p28_control_bundle.py"
            and verifier.get("bootstrap_and_checked_in_verifier_sha256_must_match") is True
            and verifier.get("bootstrap_copy_is_o_excl_no_overwrite") is True
            and verifier.get("bootstrap_must_be_regular_nonsymlink") is True
            and verifier.get("bootstrap_required_mode") == "0600"
            and verifier.get("bootstrap_self_authentication_cli")
            == "--bootstrap-verifier plus --expected-verifier-sha256"
            and verifier.get("receipt_binds_canonical_verifier_source_path_and_sha256") is True
            and verifier.get("receipt_records_executing_verifier_self_authentication") is True
            and verifier.get("receipt_does_not_bind_bootstrap_owner_or_mode") is True
            and isinstance(verifier.get("source_sha256"), str)
            and SHA256_RE.fullmatch(str(verifier.get("source_sha256"))) is not None
            and verifier.get("test_path") == "tests/test_p28_control_bundle_verifier.py"
            and isinstance(verifier.get("test_sha256"), str)
            and SHA256_RE.fullmatch(str(verifier.get("test_sha256"))) is not None
            and verifier.get("placeholders_must_be_replaced_before_source_freeze") is False
            and _file_hashes_exact(
                {
                    "source": {
                        "path": verifier.get("source_path"),
                        "sha256": verifier.get("source_sha256"),
                    },
                    "test": {
                        "path": verifier.get("test_path"),
                        "sha256": verifier.get("test_sha256"),
                    },
                }
            )
            and verifier.get("allowed_phases") == ["source", "runtime-review"]
            and verifier.get("phase_receipt_statuses")
            == {
                "source": "source_bundle_verified_before_attempt_root_creation",
                "runtime-review": "runtime_review_bundle_verified_before_localization",
            }
            and verifier.get("source_phase_constructs_control_checkout") is True
            and verifier.get("runtime_review_phase_requires_exact_direct_child") is True
            and verifier.get("runtime_review_phase_requires_exact_prepare_inventory") is True
            and verifier.get("runtime_review_phase_requires_lock_and_attestation_byte_equality")
            is True
            and verifier.get("runtime_review_phase_requires_no_localization_evidence") is True
            and verifier.get("attempt_root_must_remain_absent_for_entire_source_verification")
            is True
        ),
        "host_orchestrator_is_final_hash_bound_and_orders_receipts": (
            orchestration.get("source_path")
            == "scripts/run_p28_bundle_complete_localization_bridge.sh"
            and isinstance(orchestration.get("source_sha256"), str)
            and SHA256_RE.fullmatch(str(orchestration.get("source_sha256"))) is not None
            and orchestration.get("test_path")
            == "tests/test_p28_bundle_complete_localization_bridge_orchestration.py"
            and isinstance(orchestration.get("test_sha256"), str)
            and SHA256_RE.fullmatch(str(orchestration.get("test_sha256"))) is not None
            and orchestration.get("placeholders_must_be_replaced_before_source_freeze") is False
            and _file_hashes_exact(
                {
                    "source": {
                        "path": orchestration.get("source_path"),
                        "sha256": orchestration.get("source_sha256"),
                    },
                    "test": {
                        "path": orchestration.get("test_path"),
                        "sha256": orchestration.get("test_sha256"),
                    },
                }
            )
            and orchestration.get("allowed_phases") == ["prepare-runtime", "run-localization"]
            and orchestration.get("required_environment") == EXPECTED_ORCHESTRATOR_ENVIRONMENT
            and orchestration.get("state_changing_prepare_runtime_invocations_allowed") == 1
            and orchestration.get(
                "pre_root_source_receipt_replay_is_read_only_and_not_counted_as_state_changing_prepare"
            )
            is True
            and orchestration.get("source_bundle_verification_precedes_attempt_root_creation")
            is True
            and orchestration.get("runtime_review_bundle_verification_precedes_localization_marker")
            is True
            and orchestration.get("cleanup_resume_or_retry_phase_exists") is False
        ),
        "fresh_ids_locked_runtime_and_two_phase_delta_exact": (
            attempt.get("attempt_id") == "20260906-01"
            and attempt.get("attempt_root") == "/secure/p28/attempt-20260906-01"
            and attempt.get("container_name") == "p28-localization-20260906-01"
            and attempt.get("transport_root") == "/secure/p28/transport-20260906-01"
            and attempt.get("control_checkout") == "/secure/p28/control-source"
            and attempt.get("root_must_not_preexist_before_verified_source_receipt") is True
            and attempt.get("future_scientific_acquisition_is_authorized") is False
            and runtime.get("image_digest")
            == "sha256:14dafde07ae578cc4725f452a51d4bc4c920b69f6af23125a72e27162e193ec0"
            and runtime.get("gpu_uuid") == "GPU-ee4c9bf9-42f5-7cdd-66d2-f638c2db549d"
            and runtime.get("network_mode") == "none"
            and runtime.get("exact_bind_mount_count") == 10
            and runtime.get("extra_mounts_allowed") is False
            and runtime.get(
                "authority_git_indirection_guards_revalidated_before_every_prepare_or_run"
            )
            == [
                "no shallow file",
                "no objects/info/alternates",
                "no info/grafts",
                "no refs/replace",
                "no extensions.partialClone",
                "no remote.*.promisor",
            ]
            and freeze.get("phase_two_exact_delta")
            == [
                "A\texperiments/training/p28_cuda_runtime_lock.json",
                "A\texperiments/training/p28_host_attestation.json",
            ]
            and freeze.get("phase_one_parent_commit") == P27_OUTCOME_COMMIT
            and freeze.get("phase_one_must_be_direct_child_of_terminal_p27") is True
            and freeze.get("phase_two_runtime_review_bundle_must_be_complete") is True
            and freeze.get("phase_two_runtime_review_receipt_must_be_external") is True
            and freeze.get("phase_two_localization_started") is False
        ),
        "unchanged_no_training_boundary_and_future_acquisition_barred": (
            boundary.get("unchanged_p27_localizer_only") is True
            and boundary.get("diagnostic_stages")
            == ["pre_cuda", "post_cuda_init", "post_model_move", "post_optimizer"]
            and boundary.get("data_loading_allowed") is False
            and boundary.get("forward_allowed") is False
            and boundary.get("backward_allowed") is False
            and boundary.get("optimizer_step_allowed") is False
            and boundary.get("candidate_evaluation_allowed") is False
            and boundary.get("candidate_observation_count_required") == 0
            and boundary.get("gradient_observation_count_required") == 0
            and boundary.get("parameter_update_count_required") == 0
            and routing.get("future_scientific_acquisition_authorized") is False
            and routing.get("future_scientific_acquisition_requires_separately_reviewed_contract")
            is True
        ),
        "p27_independent_reconstruction_is_23_of_23": (
            p27_ok and p27_count == 23 and p27_true == 23 and p27_false == 0
        ),
        "terminal_p27_outcome_independent_reconstruction_is_10_of_10": (
            p27_outcome_ok
            and p27_outcome_count == 10
            and p27_outcome_true == 10
            and p27_outcome_false == 0
        ),
    }
    return {
        "schema_version": (
            "passive-muon-p28-bundle-complete-localization-bridge-reconstruction-v1"
        ),
        "internally_consistent": all(checks.values()),
        "checks": checks,
        "canonical_sha256": _sha256(contract_bytes),
        "p27_reconstruction": {
            "check_count": p27_count,
            "true_count": p27_true,
            "false_count": p27_false,
        },
        "p27_outcome_reconstruction": {
            "check_count": p27_outcome_count,
            "true_count": p27_outcome_true,
            "false_count": p27_outcome_false,
        },
        "claim_boundary": (
            "This reconstructs repository and preregistration facts only. It does not "
            "authenticate a transfer receipt, create an attempt root or container, initialize "
            "CUDA, observe a mapping, run localization, capture a candidate, or train."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CONTRACT)
    arguments = parser.parse_args()
    result = reconstruct(arguments.canonical)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["internally_consistent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
