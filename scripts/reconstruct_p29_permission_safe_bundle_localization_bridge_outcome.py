#!/usr/bin/env python3
"""Independently reconstruct P29's terminal post-receipt seal outcome."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Mapping
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTCOME = (
    ROOT / "results/summaries/p29_permission_safe_bundle_localization_bridge_outcome.json"
)
SOURCE_COMMIT = "ab62f683bf41ebb593a46267501e6aa6ba8e04c6"
SOURCE_TREE = "3a6f210b5ed4767cbfe0ac354e2f58031242064f"
SOURCE_PARENT = "7274367f2b05fb3c9ed9f876a59be647703bbdc2"
CONTRACT_PATH = "experiments/training/p29_permission_safe_bundle_localization_bridge_contract.json"
CONTRACT_SHA256 = "54d10ab01b7d453d01931459b314fc4491b6e4317a02fb763ca9872686311b0a"
VERIFIER_PATH = "scripts/verify_p29_control_bundle.py"
VERIFIER_SHA256 = "4e1c8d23a0b770282b90ad200d5323d3d281c57bc302a6a21cd84fef636f17b9"
ORCHESTRATOR_PATH = "scripts/run_p29_permission_safe_bundle_localization_bridge.sh"
ORCHESTRATOR_SHA256 = "1653a12c2a86d307ddbff11cc5e48701e4817ee99e4674eaf6b426a8b2302cc5"
RUNBOOK_PATH = "experiments/training/P29_PERMISSION_SAFE_BUNDLE_LOCALIZATION_BRIDGE_RUNBOOK.md"
RUNBOOK_SHA256 = "9c8d7b9af3a5fbf1a577a282f94e01d532bcab175931f9e8edab5f6dacd4a2b9"
RECONSTRUCTOR_PATH = "scripts/reconstruct_p29_permission_safe_bundle_localization_bridge.py"
RECONSTRUCTOR_SHA256 = "a384bfcbc25fe9148a158a5f524170557808f8e68d07cb99474b64bbe3c10aea"
BUNDLE_SHA256 = "5546ab07222ca5e9b1fd1959a23fd7830f68333ba073386a302343b3fe5817a9"
RECEIPT_SHA256 = "7e4b80c03ccc59049b78a1a553c087ec670ce5c4cbc5a3db6b1bda8db2b45065"
OUTCOME_SHA256 = "6d29601a71ce6e0c99bcc35497d328c8fdd2a3cfc135351c19876507b00a0f64"
EXACT_ERROR = "P29 reviewed orchestrator source authority differs"

P28_TAG_REF = "refs/tags/p28-control-parent-permission-diagnostic"
P28_TAG_OBJECT = "f3c81a2f14f853f369015a4f81b6695b52eec74e"
P27_TAG_REF = "refs/tags/p27-cuda-deleted-mapping-localization-diagnostic"
P27_TAG_OBJECT = "c88b98eae9be2458abde45b05d3dccfe09c0c7ed"
P27_COMMIT = "ec63550331925ded158e3f389e294e4d1f12db3a"
P26_TAG_REF = "refs/tags/p26-permission-safe-acquisition-checkpoint"
P26_TAG_OBJECT = "bacad707d3779bfa10957e18cb4c69b1a7f0cbce"
P26_COMMIT = "5429da23ff18888daa2312c530a4587780484d8b"
P26_SOURCE_REF = "refs/tags/p26-attempt02-acquisition-source"
P26_SOURCE_OBJECT = "f35a7dca8f6bc39e9748e79b6712fab4a203396b"
P26_SOURCE_COMMIT = "185e444afc0b44ca0a09b1bde49a6b6fa3973355"

TOP_LEVEL_ORDER = [
    "schema_version",
    "status",
    "attempt_id",
    "evidence_kind",
    "claim_boundary",
    "source_freeze",
    "source_bundle",
    "bootstrap_verifier",
    "source_verification",
    "source_receipt",
    "reviewed_orchestrator_seal",
    "retained_remote_state",
    "gate_status",
    "terminal_policy",
    "theorem_status",
    "root_cause",
    "route",
]
EXPECTED_REFS = [
    {
        "ref": "refs/heads/p29-permission-safe-bundle-localization-bridge",
        "object": SOURCE_COMMIT,
    },
    {
        "ref": "refs/heads/p28-bundle-complete-localization-bridge",
        "object": SOURCE_PARENT,
    },
    {"ref": P28_TAG_REF, "object": P28_TAG_OBJECT, "peeled_commit": SOURCE_PARENT},
    {
        "ref": "refs/heads/p27-cuda-deleted-mapping-localization",
        "object": P27_COMMIT,
    },
    {"ref": P27_TAG_REF, "object": P27_TAG_OBJECT, "peeled_commit": P27_COMMIT},
    {"ref": P26_TAG_REF, "object": P26_TAG_OBJECT, "peeled_commit": P26_COMMIT},
    {
        "ref": P26_SOURCE_REF,
        "object": P26_SOURCE_OBJECT,
        "peeled_commit": P26_SOURCE_COMMIT,
    },
]
EXPECTED_COMPLETED = [
    "bootstrap_verifier_authentication",
    "stable_source_bundle_hash_and_size_authentication",
    "closed_v2_header_and_zero_prerequisite_validation",
    "fresh_empty_bare_source_closure_creation",
    "explicit_fetch_of_each_of_seven_refs",
    "tag_type_and_peel_validation",
    "full_strict_git_fsck",
    "canonical_control_checkout_creation_and_validation",
    "p29_contract_reconstruction_15_of_15",
    "p28_contract_reconstruction_12_of_12",
    "p28_terminal_outcome_reconstruction_10_of_10",
    "p27_contract_reconstruction_23_of_23",
    "private_bundle_post_use_revalidation",
    "source_receipt_creation",
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


def reconstruct(outcome_path: Path = DEFAULT_OUTCOME) -> dict[str, object]:
    """Check local Git facts and the exact operator-retained terminal record."""

    try:
        outcome_bytes = outcome_path.read_bytes()
        parsed = _strict_json(outcome_bytes)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError, DuplicateKeyError):
        outcome_bytes = b""
        parsed = None
    outcome = _mapping(parsed)
    source = _mapping(outcome.get("source_freeze"))
    bundle = _mapping(outcome.get("source_bundle"))
    verifier = _mapping(outcome.get("bootstrap_verifier"))
    verification = _mapping(outcome.get("source_verification"))
    receipt = _mapping(outcome.get("source_receipt"))
    barrier = _mapping(receipt.get("reconstruction_publication_barrier"))
    seal = _mapping(outcome.get("reviewed_orchestrator_seal"))
    retained = _mapping(outcome.get("retained_remote_state"))
    ledger = _mapping(retained.get("localization_ledger_root"))
    gates = _mapping(outcome.get("gate_status"))
    policy = _mapping(outcome.get("terminal_policy"))
    theorem = _mapping(outcome.get("theorem_status"))
    cause = _mapping(outcome.get("root_cause"))
    route = _mapping(outcome.get("route"))

    contract_blob = _git("show", f"{SOURCE_COMMIT}:{CONTRACT_PATH}")
    verifier_blob = _git("show", f"{SOURCE_COMMIT}:{VERIFIER_PATH}")
    orchestrator_blob = _git("show", f"{SOURCE_COMMIT}:{ORCHESTRATOR_PATH}")
    runbook_blob = _git("show", f"{SOURCE_COMMIT}:{RUNBOOK_PATH}")
    reconstructor_blob = _git("show", f"{SOURCE_COMMIT}:{RECONSTRUCTOR_PATH}")

    checks = {
        "canonical_outcome_bytes_and_closed_top_level": (
            isinstance(parsed, Mapping)
            and list(parsed) == TOP_LEVEL_ORDER
            and outcome_bytes == (json.dumps(parsed, indent=2) + "\n").encode()
            and _sha256(outcome_bytes) == OUTCOME_SHA256
        ),
        "identity_status_and_claim_boundary_exact": (
            outcome.get("schema_version")
            == "passive-muon-p29-permission-safe-bundle-localization-bridge-outcome-v1"
            and outcome.get("status")
            == "terminal_post_source_receipt_orchestrator_seal_source_mode_stop"
            and outcome.get("attempt_id") == "20260906-02"
            and "external source receipt" in str(outcome.get("claim_boundary"))
            and "operator-recorded" in str(outcome.get("claim_boundary"))
            and "not a source-bundle" in str(outcome.get("claim_boundary"))
        ),
        "source_commit_tree_parent_and_frozen_bytes_exact": (
            _git_text("rev-parse", f"{SOURCE_COMMIT}^{{tree}}") == SOURCE_TREE
            and _git_text("show", "-s", "--format=%P", SOURCE_COMMIT) == SOURCE_PARENT
            and contract_blob is not None
            and _sha256(contract_blob) == CONTRACT_SHA256
            and verifier_blob is not None
            and _sha256(verifier_blob) == VERIFIER_SHA256
            and orchestrator_blob is not None
            and _sha256(orchestrator_blob) == ORCHESTRATOR_SHA256
            and runbook_blob is not None
            and _sha256(runbook_blob) == RUNBOOK_SHA256
            and reconstructor_blob is not None
            and _sha256(reconstructor_blob) == RECONSTRUCTOR_SHA256
            and _git_text("ls-tree", SOURCE_COMMIT, ORCHESTRATOR_PATH).startswith("100644 blob ")
            and source.get("commit") == SOURCE_COMMIT
            and source.get("tree") == SOURCE_TREE
            and source.get("parent") == SOURCE_PARENT
            and source.get("source_frozen_before_namespace_creation") is True
        ),
        "local_annotated_tag_objects_and_peels_exact": (
            _git_text("cat-file", "-t", P28_TAG_OBJECT) == "tag"
            and _git_text("rev-parse", P28_TAG_REF) == P28_TAG_OBJECT
            and _git_text("rev-parse", f"{P28_TAG_REF}^{{}}") == SOURCE_PARENT
            and _git_text("cat-file", "-t", P27_TAG_OBJECT) == "tag"
            and _git_text("rev-parse", P27_TAG_REF) == P27_TAG_OBJECT
            and _git_text("rev-parse", f"{P27_TAG_REF}^{{}}") == P27_COMMIT
            and _git_text("cat-file", "-t", P26_TAG_OBJECT) == "tag"
            and _git_text("rev-parse", P26_TAG_REF) == P26_TAG_OBJECT
            and _git_text("rev-parse", f"{P26_TAG_REF}^{{}}") == P26_COMMIT
            and _git_text("cat-file", "-t", P26_SOURCE_OBJECT) == "tag"
            and _git_text("rev-parse", P26_SOURCE_REF) == P26_SOURCE_OBJECT
            and _git_text("rev-parse", f"{P26_SOURCE_REF}^{{}}") == P26_SOURCE_COMMIT
        ),
        "source_bundle_and_bootstrap_bindings_exact": (
            bundle.get("sha256") == BUNDLE_SHA256
            and bundle.get("byte_count") == 3_503_837
            and bundle.get("mode") == "0600"
            and bundle.get("uid") == bundle.get("gid") == 1000
            and bundle.get("prerequisite_count") == 0
            and bundle.get("advertised_refs") == EXPECTED_REFS
            and bundle.get("exact_closed_ref_inventory_verified") is True
            and verifier.get("sha256") == VERIFIER_SHA256
            and verifier.get("byte_count") == 138_874
            and verifier.get("mode") == "0600"
            and verifier.get("uid") == verifier.get("gid") == 1000
            and verifier.get("stable_nofollow_self_authentication_passed") is True
        ),
        "source_verifier_and_native_external_receipt_exact": (
            verification.get("phase") == "source"
            and verification.get("invocations_allowed") == 1
            and verification.get("invocations_run") == 1
            and verification.get("exit_status") == 0
            and verification.get("completed_before_seal") == EXPECTED_COMPLETED
            and verification.get("source_control_checkout_created") is True
            and verification.get("source_control_head") == SOURCE_COMMIT
            and verification.get("source_control_tree") == SOURCE_TREE
            and verification.get("source_control_clean_at_receipt_creation") is True
            and verification.get("source_receipt_created") is True
            and receipt.get("schema_version") == "passive-muon-p29-control-bundle-receipt-v1"
            and receipt.get("status")
            == "source_bundle_and_permission_layout_verified_before_attempt_root_creation"
            and receipt.get("sha256") == RECEIPT_SHA256
            and receipt.get("byte_count") == 11_678
            and receipt.get("uid") == receipt.get("gid") == 1000
            and receipt.get("mode") == "0600"
            and receipt.get("native_external_artifact") is True
            and receipt.get("committed_to_repository") is False
            and receipt.get("operator_recorded_path_stat_and_digest") is True
            and receipt.get("replayed_after_creation") is False
            and barrier
            == {
                "p29_contract": "15/15",
                "p28_terminal_outcome": "10/10",
                "p28_contract": "12/12",
                "p27_contract": "23/23",
            }
        ),
        "single_reviewed_orchestrator_seal_failure_exact": (
            seal.get("invocations_run") == 1
            and seal.get("exit_status") == 1
            and seal.get("invocation_utc_independently_retained") is False
            and seal.get("invocation_utc") is None
            and seal.get("exact_terminal_stderr") == EXACT_ERROR
            and seal.get("exact_terminal_stderr_with_one_lf_sha256")
            == _sha256((EXACT_ERROR + "\n").encode())
            and seal.get("exact_terminal_stderr_with_one_lf_byte_count") == 51
            and seal.get("operator_capture_only") is True
            and seal.get("remote_stdout_file_retained") is False
            and seal.get("remote_stderr_file_retained") is False
            and seal.get("source_sha256") == ORCHESTRATOR_SHA256
            and seal.get("source_byte_count") == 206_145
            and seal.get("source_uid") == seal.get("source_gid") == 1000
            and seal.get("source_observed_mode") == "0600"
            and seal.get("source_required_mode") == "0644"
            and seal.get("source_hash_matched") is True
            and seal.get("source_owner_matched") is True
            and seal.get("source_mode_matched") is False
            and seal.get("destination_created") is False
        ),
        "retained_pre_authority_inventory_exact": (
            retained.get("source_closure_created") is True
            and retained.get("control_checkout_created") is True
            and retained.get("source_receipt_created") is True
            and ledger
            == {
                "path": "/var/lib/optimizationml-p29-20260906-02",
                "owner": "root:root",
                "uid": 0,
                "gid": 0,
                "mode": "0700",
                "exact_child_inventory": ["deferred"],
            }
            and retained.get("sealed_orchestrator_created") is False
            and retained.get("authority_checkout_created") is False
            and retained.get("attempt_root_created") is False
            and retained.get("execution_snapshot_created") is False
            and retained.get("prepare_runtime_invocation_token_created") is False
            and retained.get("localization_invocation_token_created") is False
            and retained.get("localization_authorization_token_created") is False
            and retained.get("localization_terminal_status_created") is False
            and retained.get("container_created") is False
            and retained.get("runtime_lock_created") is False
            and retained.get("host_attestation_created") is False
            and retained.get("native_localization_artifact_created") is False
            and retained.get("sanitized_localization_artifact_created") is False
        ),
        "zero_runtime_cuda_localization_training_and_candidate_evidence_exact": (
            gates.get("source_bundle_completeness") == "passed_exact_seven_refs"
            and gates.get("required_reconstructions") == "passed_15_12_10_23"
            and gates.get("source_receipt") == "created_and_reviewed"
            and gates.get("reviewed_orchestrator_seal") == "failed_source_mode_mismatch"
            and gates.get("authority_checkout_creation") == "not_run"
            and gates.get("prepare_runtime_invocation") == "not_run"
            and gates.get("gpu_access") == "not_run"
            and gates.get("container_creation") == "not_run"
            and gates.get("cuda_initialization") == "not_run"
            and gates.get("deleted_mapping_localization") == "not_run"
            and gates.get("forward_backward") == "not_run"
            and gates.get("optimizer_steps") == 0
            and gates.get("candidate_observations") == 0
            and gates.get("training") == "not_run"
            and theorem.get("deleted_mapping_identity_established") is False
            and theorem.get("cuda_repeatability_established") is False
            and theorem.get("real_gradient_fidelity_established") is False
        ),
        "terminal_policy_and_prior_theorem_boundary_exact": (
            policy
            == {
                "attempt_is_terminal": True,
                "namespace_transport_control_closure_receipt_and_ledger_are_terminal": True,
                "reuse_attempt_20260906_02_allowed": False,
                "cleanup_repair_or_resume_retained_p29_state_allowed": False,
                "retry_until_favorable_allowed": False,
                "source_verifier_invocation_consumed": True,
                "reviewed_orchestrator_seal_invocation_consumed": True,
                "prepare_runtime_invocation_consumed": False,
                "localization_invocation_consumed": False,
                "scientific_protocol_changed": False,
                "frozen_thresholds_changed": False,
            }
            and theorem.get("p18_through_p21_results_affected") is False
            and theorem.get("p26_terminal_result_affected") is False
            and theorem.get("p27_terminal_result_affected") is False
            and theorem.get("p28_terminal_result_affected") is False
            and theorem.get("native_cuda_shield_certified") is False
        ),
        "root_cause_and_fresh_p30_route_exact": (
            cause.get("classification")
            == "cross_phase_reviewed_orchestrator_source_mode_contract_mismatch"
            and cause.get("source_bundle_defect_observed") is False
            and cause.get("source_receipt_defect_observed") is False
            and cause.get("reconstruction_defect_observed") is False
            and cause.get("orchestrator_byte_or_hash_defect_observed") is False
            and cause.get("source_owner_defect_observed") is False
            and cause.get("source_mode_was_more_restrictive_than_required") is True
            and cause.get("cross_phase_mode_precondition_defect_observed") is True
            and cause.get("seal_failed_closed") is True
            and cause.get("security_authority_broadened") is False
            and cause.get("cuda_or_backend_defect_observed") is False
            and cause.get("mapping_classification_observed") is False
            and route.get("selected_terminal_route") == "layout_bundle_or_reconstruction_failure"
            and route.get("next_branch") == "p30-umask-bound-control-seal"
            and route.get("next_attempt_id") == "20260906-03"
            and route.get("new_namespace_bundle_receipt_and_attempt_required") is True
            and route.get("chmod_or_other_repair_of_retained_p29_control_allowed") is False
            and route.get("attempt_20260906_02_may_be_reused") is False
            and route.get("retained_p29_state_may_be_cleaned_up") is False
            and route.get("scientific_acquisition_authorized") is False
        ),
    }
    return {
        "schema_version": (
            "passive-muon-p29-permission-safe-bundle-localization-bridge-outcome-reconstruction-v1"
        ),
        "internally_consistent": all(checks.values()),
        "checks": checks,
        "canonical_sha256": _sha256(outcome_bytes),
        "claim_boundary": (
            "Repository-authenticated source bytes plus one hash-bound external native "
            "source receipt and exact operator-retained seal facts. The receipt bytes and "
            "seal streams are not committed, so their remote path/stat and stderr facts "
            "remain operator-recorded rather than independently reconstructed from native "
            "bytes."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcome", type=Path, default=DEFAULT_OUTCOME)
    arguments = parser.parse_args()
    result = reconstruct(arguments.outcome)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["internally_consistent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
