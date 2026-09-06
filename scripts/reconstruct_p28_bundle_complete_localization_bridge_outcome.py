#!/usr/bin/env python3
"""Independently reconstruct P28's terminal pre-container outcome packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Mapping
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTCOME = ROOT / "results/summaries/p28_bundle_complete_localization_bridge_outcome.json"
SOURCE_COMMIT = "003b6ee3f0ee2d1c269cac787c0281bf3f976ac4"
SOURCE_TREE = "11e471539a8f422d74728548a0ee8ac19ffea1c0"
SOURCE_PARENT = "ec63550331925ded158e3f389e294e4d1f12db3a"
CONTRACT_PATH = "experiments/training/p28_bundle_complete_localization_bridge_contract.json"
CONTRACT_SHA256 = "6a5005f902d937f0edbec3394e6205e482fe6f6fea45f986ac873fd6ab2e8718"
VERIFIER_PATH = "scripts/verify_p28_control_bundle.py"
VERIFIER_SHA256 = "27e2d5e89a5dcd31cc5e4f2c32953ac1232246077863f105078a5a7984c77a00"
BUNDLE_SHA256 = "54461db0d3a2e07870d9f43cf9877da0cb4f28c13ad1c5f1cc32b44ff3e7945b"
OUTCOME_SHA256 = "298324d163aa12c1cb64bfdb912ae846152d0980ae3c981a54ead75a019305c9"
EXACT_ERROR = (
    "P28 control-bundle verification failed: Git command failed (128): init "
    "--initial-branch=p28-verifier-unborn /secure/p28/control-source: fatal: cannot "
    "mkdir /secure/p28/control-source: Permission denied"
)

P27_TAG_REF = "refs/tags/p27-cuda-deleted-mapping-localization-diagnostic"
P27_TAG_OBJECT = "c88b98eae9be2458abde45b05d3dccfe09c0c7ed"
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
    "retained_remote_state",
    "gate_status",
    "terminal_policy",
    "theorem_status",
    "root_cause",
    "route",
]
EXPECTED_REFS = [
    {
        "ref": "refs/heads/p28-bundle-complete-localization-bridge",
        "object": SOURCE_COMMIT,
    },
    {
        "ref": "refs/heads/p27-cuda-deleted-mapping-localization",
        "object": SOURCE_PARENT,
    },
    {"ref": P27_TAG_REF, "object": P27_TAG_OBJECT, "peeled_commit": SOURCE_PARENT},
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
    "explicit_fetch_of_each_of_five_refs",
    "tag_type_and_peel_validation",
    "full_strict_git_fsck",
    "private_bundle_post_use_revalidation",
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
    """Check local Git facts and the exact operator-retained outcome record."""

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
    retained = _mapping(outcome.get("retained_remote_state"))
    parent = _mapping(retained.get("parent"))
    transport = _mapping(retained.get("transport"))
    closure = _mapping(retained.get("source_closure"))
    gates = _mapping(outcome.get("gate_status"))
    policy = _mapping(outcome.get("terminal_policy"))
    theorem = _mapping(outcome.get("theorem_status"))
    cause = _mapping(outcome.get("root_cause"))
    route = _mapping(outcome.get("route"))

    contract_blob = _git("show", f"{SOURCE_COMMIT}:{CONTRACT_PATH}")
    verifier_blob = _git("show", f"{SOURCE_COMMIT}:{VERIFIER_PATH}")

    checks = {
        "canonical_outcome_bytes_and_closed_top_level": (
            isinstance(parsed, Mapping)
            and list(parsed) == TOP_LEVEL_ORDER
            and outcome_bytes == (json.dumps(parsed, indent=2) + "\n").encode()
            and _sha256(outcome_bytes) == OUTCOME_SHA256
        ),
        "identity_status_and_claim_boundary_exact": (
            outcome.get("schema_version")
            == "passive-muon-p28-bundle-complete-localization-bridge-outcome-v1"
            and outcome.get("status")
            == "terminal_source_verifier_control_checkout_parent_permission_stop"
            and outcome.get("attempt_id") == "20260906-01"
            and "No control checkout" in str(outcome.get("claim_boundary"))
            and "not a completed source verification" in str(outcome.get("claim_boundary"))
        ),
        "source_commit_tree_parent_contract_and_verifier_exact": (
            _git_text("rev-parse", f"{SOURCE_COMMIT}^{{tree}}") == SOURCE_TREE
            and _git_text("show", "-s", "--format=%P", SOURCE_COMMIT) == SOURCE_PARENT
            and contract_blob is not None
            and _sha256(contract_blob) == CONTRACT_SHA256
            and verifier_blob is not None
            and _sha256(verifier_blob) == VERIFIER_SHA256
            and source.get("commit") == SOURCE_COMMIT
            and source.get("tree") == SOURCE_TREE
            and source.get("parent") == SOURCE_PARENT
            and source.get("contract_sha256") == CONTRACT_SHA256
            and source.get("verifier_sha256") == VERIFIER_SHA256
            and source.get("source_frozen_before_invocation") is True
        ),
        "local_annotated_tag_objects_and_peels_exact": (
            _git_text("cat-file", "-t", P27_TAG_OBJECT) == "tag"
            and _git_text("rev-parse", P27_TAG_REF) == P27_TAG_OBJECT
            and _git_text("rev-parse", f"{P27_TAG_REF}^{{}}") == SOURCE_PARENT
            and _git_text("cat-file", "-t", P26_TAG_OBJECT) == "tag"
            and _git_text("rev-parse", P26_TAG_REF) == P26_TAG_OBJECT
            and _git_text("rev-parse", f"{P26_TAG_REF}^{{}}") == P26_COMMIT
            and _git_text("cat-file", "-t", P26_SOURCE_OBJECT) == "tag"
            and _git_text("rev-parse", P26_SOURCE_REF) == P26_SOURCE_OBJECT
            and _git_text("rev-parse", f"{P26_SOURCE_REF}^{{}}") == P26_SOURCE_COMMIT
        ),
        "source_bundle_and_bootstrap_bindings_exact": (
            bundle.get("sha256") == BUNDLE_SHA256
            and bundle.get("byte_count") == 3_362_894
            and bundle.get("mode") == "0600"
            and bundle.get("uid") == bundle.get("gid") == 1000
            and bundle.get("prerequisite_count") == 0
            and bundle.get("advertised_refs") == EXPECTED_REFS
            and bundle.get("exact_closed_ref_inventory_verified") is True
            and bundle.get("private_authenticated_copy_was_mode_0400_before_git_use") is True
            and bundle.get("private_copy_revalidated_after_git_use") is True
            and verifier.get("sha256") == VERIFIER_SHA256
            and verifier.get("byte_count") == 51_304
            and verifier.get("mode") == "0600"
            and verifier.get("uid") == verifier.get("gid") == 1000
            and verifier.get("stable_nofollow_self_authentication_passed") is True
        ),
        "single_source_verifier_failure_exact": (
            verification.get("phase") == "source"
            and verification.get("invocations_allowed") == 1
            and verification.get("invocations_run") == 1
            and verification.get("invocation_utc_independently_retained") is False
            and verification.get("invocation_utc") is None
            and verification.get("exit_status") == 1
            and verification.get("exact_terminal_stderr") == EXACT_ERROR
            and verification.get("exact_terminal_stderr_with_one_lf_sha256")
            == "3d85310ab120816ed22e1d78295b2384ff2e0e2147d5700dbb93abe1bb28b355"
            and verification.get("exact_terminal_stderr_with_one_lf_byte_count") == 202
            and verification.get("operator_capture_only") is True
            and verification.get("remote_stdout_file_retained") is False
            and verification.get("remote_stderr_file_retained") is False
            and verification.get("completed_before_failure") == EXPECTED_COMPLETED
            and verification.get("source_control_checkout_created") is False
            and verification.get("p27_reconstruction_run") is False
            and verification.get("source_receipt_created") is False
        ),
        "retained_permission_boundary_and_artifact_inventory_exact": (
            parent
            == {
                "path": "/secure/p28",
                "owner": "root:root",
                "uid": 0,
                "gid": 0,
                "mode": "0755",
            }
            and transport
            == {
                "path": "/secure/p28/transport-20260906-01",
                "owner": "ubuntu:ubuntu",
                "uid": 1000,
                "gid": 1000,
                "mode": "0700",
            }
            and closure.get("created") is True
            and closure.get("birth_utc") == "2026-09-06T15:09:08.942901140Z"
            and closure.get("mode") == "0775"
            and closure.get("uid") == closure.get("gid") == 1000
            and closure.get("exact_five_refs_present") is True
            and retained.get("control_checkout_created") is False
            and retained.get("source_receipt_created") is False
            and retained.get("attempt_root_created") is False
            and retained.get("container_created") is False
            and retained.get("runtime_lock_created") is False
            and retained.get("host_attestation_created") is False
            and retained.get("native_localization_artifact_created") is False
            and retained.get("sanitized_localization_artifact_created") is False
        ),
        "zero_container_cuda_localization_training_and_candidate_evidence_exact": (
            gates.get("source_bundle_completeness") == "passed_exact_five_refs"
            and gates.get("fresh_source_closure") == "created_and_verified"
            and gates.get("source_control_checkout") == "failed_permission_denied"
            and gates.get("p27_contract_reconstruction") == "not_run"
            and gates.get("attempt_root_creation") == "not_run"
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
        "terminal_policy_exact": (
            policy
            == {
                "attempt_is_terminal": True,
                "transport_and_closure_are_terminal": True,
                "reuse_attempt_20260906_01_allowed": False,
                "cleanup_retained_transport_or_closure_allowed": False,
                "retry_until_favorable_allowed": False,
                "source_verifier_invocation_consumed": True,
                "prepare_runtime_invocation_consumed": False,
                "localization_invocation_consumed": False,
                "scientific_protocol_changed": False,
                "frozen_thresholds_changed": False,
            }
        ),
        "root_cause_and_new_attempt_route_exact": (
            cause.get("classification") == "unprivileged_control_checkout_parent_permission"
            and cause.get("bundle_completeness_defect_observed") is False
            and cause.get("verifier_logic_defect_observed") is False
            and cause.get("p27_contract_defect_observed") is False
            and cause.get("verifier_failed_closed") is True
            and cause.get("host_layout_preflight_defect_observed") is True
            and cause.get("runbook_host_layout_defect_observed") is True
            and cause.get("cuda_or_backend_defect_observed") is False
            and cause.get("mapping_classification_observed") is False
            and route.get("new_attempt_required") is True
            and route.get("new_attempt_identifier_must_be_preregistered") is True
            and route.get("new_source_bundle_and_external_receipt_required") is True
            and route.get("attempt_20260906_01_may_be_reused") is False
            and route.get("retained_p28_transport_or_closure_may_be_cleaned_up") is False
            and route.get("scientific_acquisition_authorized") is False
        ),
    }
    return {
        "schema_version": (
            "passive-muon-p28-bundle-complete-localization-bridge-outcome-reconstruction-v1"
        ),
        "internally_consistent": all(checks.values()),
        "checks": checks,
        "canonical_sha256": _sha256(outcome_bytes),
        "claim_boundary": (
            "Repository-authenticated source bytes plus exact operator-retained transport "
            "facts and terminal stderr. No native remote transcript was retained, so remote "
            "ownership, permissions, closure birth, and command execution remain "
            "operator-attested rather than independently reconstructed from native bytes."
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
