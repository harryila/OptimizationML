#!/usr/bin/env python3
"""Independently reconstruct P27's terminal pre-container outcome packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Mapping
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTCOME = ROOT / "results/summaries/p27_cuda_deleted_mapping_localization_outcome.json"
TRANSCRIPT_PATH = "results/summaries/p27_prepare_contract_reconstruction.remote.json"
SOURCE_COMMIT = "1071afa1b20b3e12be3dafdebf9dae63f39b0ed9"
SOURCE_TREE = "e32a8d89a4470c7e4ee2631c8abb16c2f0a4d923"
SOURCE_PARENT = "5429da23ff18888daa2312c530a4587780484d8b"
CONTRACT_PATH = "experiments/training/p27_cuda_deleted_mapping_localization_contract.json"
CONTRACT_SHA256 = "02000debea28f499ddeb3b8c6c8b0615786cd2e7858cc8a1605b04d638da4b7a"
CHECKPOINT_REF = "refs/tags/p26-permission-safe-acquisition-checkpoint"
CHECKPOINT_OBJECT = "bacad707d3779bfa10957e18cb4c69b1a7f0cbce"
TRANSCRIPT_SHA256 = "911cf5e086227c95004ab9bf87438463f4c6d49ff3b77457ea89a98abca845b4"
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
OUTCOME_SHA256 = "d87df6cc792dfa0a0d6a2a982395953ba2ef87aa2086266d9d3c59abb3427d0e"

TOP_LEVEL_ORDER = [
    "schema_version",
    "status",
    "attempt_id",
    "evidence_kind",
    "claim_boundary",
    "source_freeze",
    "operator_transfer",
    "remote_prepare_runtime",
    "retained_attempt_state",
    "gate_status",
    "terminal_policy",
    "theorem_status",
    "root_cause",
    "route",
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
    """Check repository-authenticated facts and the exact retained transcript copy."""

    try:
        outcome_bytes = outcome_path.read_bytes()
        parsed = _strict_json(outcome_bytes)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError, DuplicateKeyError):
        outcome_bytes = b""
        parsed = None
    outcome = _mapping(parsed)

    try:
        transcript_bytes = (ROOT / TRANSCRIPT_PATH).read_bytes()
        transcript_parsed = _strict_json(transcript_bytes)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError, DuplicateKeyError):
        transcript_bytes = b""
        transcript_parsed = None
    transcript = _mapping(transcript_parsed)
    transcript_checks = _mapping(transcript.get("checks"))

    source = _mapping(outcome.get("source_freeze"))
    transfer = _mapping(outcome.get("operator_transfer"))
    prepare = _mapping(outcome.get("remote_prepare_runtime"))
    reconstruction = _mapping(prepare.get("contract_reconstruction"))
    committed_transcript = _mapping(reconstruction.get("committed_path_free_stdout"))
    remote_stdout = _mapping(reconstruction.get("remote_stdout"))
    remote_stderr = _mapping(reconstruction.get("remote_stderr"))
    retained = _mapping(outcome.get("retained_attempt_state"))
    gates = _mapping(outcome.get("gate_status"))
    policy = _mapping(outcome.get("terminal_policy"))
    theorem = _mapping(outcome.get("theorem_status"))
    cause = _mapping(outcome.get("root_cause"))
    route = _mapping(outcome.get("route"))

    source_blob = _git("show", f"{SOURCE_COMMIT}:{CONTRACT_PATH}")
    tag_type = _git_text("cat-file", "-t", CHECKPOINT_OBJECT)
    tag_ref = _git_text("rev-parse", CHECKPOINT_REF)
    tag_peeled = _git_text("rev-parse", f"{CHECKPOINT_REF}^{{}}")

    checks = {
        "canonical_outcome_bytes_and_closed_top_level": (
            isinstance(parsed, Mapping)
            and list(parsed) == TOP_LEVEL_ORDER
            and outcome_bytes == (json.dumps(parsed, indent=2) + "\n").encode()
            and _sha256(outcome_bytes) == OUTCOME_SHA256
        ),
        "identity_status_and_claim_boundary_exact": (
            outcome.get("schema_version")
            == "passive-muon-p27-cuda-deleted-mapping-localization-outcome-v1"
            and outcome.get("status")
            == "terminal_pre_container_missing_required_p26_checkpoint_tag"
            and outcome.get("attempt_id") == "20260906-01"
            and "before image inspection" in str(outcome.get("claim_boundary"))
            and "not a deleted-mapping localization result" in str(outcome.get("claim_boundary"))
        ),
        "source_freeze_commit_tree_parent_and_contract_exact": (
            _git_text("rev-parse", f"{SOURCE_COMMIT}^{{tree}}") == SOURCE_TREE
            and _git_text("show", "-s", "--format=%P", SOURCE_COMMIT) == SOURCE_PARENT
            and source_blob is not None
            and _sha256(source_blob) == CONTRACT_SHA256
            and source.get("commit") == SOURCE_COMMIT
            and source.get("tree") == SOURCE_TREE
            and source.get("parent") == SOURCE_PARENT
            and source.get("contract_path") == CONTRACT_PATH
            and source.get("contract_sha256") == CONTRACT_SHA256
            and source.get("source_frozen_before_invocation") is True
        ),
        "local_checkpoint_tag_object_and_peel_exact": (
            tag_type == "tag" and tag_ref == CHECKPOINT_OBJECT and tag_peeled == SOURCE_PARENT
        ),
        "operator_transfer_omission_record_exact": (
            transfer.get("sha256")
            == "1fb136fb027b739f94c08a0f20cfeb7d5d417742cb3bd5578bbb278b86c641cc"
            and transfer.get("byte_count") == 3_307_486
            and transfer.get("required_checkpoint_ref_advertised") is False
            and transfer.get("required_annotated_tag_object_transferred") is False
            and transfer.get("bundle_was_insufficient_for_frozen_reconstruction") is True
            and _mapping(transfer.get("required_checkpoint"))
            == {
                "ref": CHECKPOINT_REF,
                "annotated_tag_object": CHECKPOINT_OBJECT,
                "peeled_commit": SOURCE_PARENT,
            }
        ),
        "committed_remote_reconstruction_transcript_exact": (
            _sha256(transcript_bytes) == TRANSCRIPT_SHA256
            and len(transcript_bytes) == 1_718
            and transcript_bytes == (json.dumps(transcript_parsed, indent=2) + "\n").encode()
            and transcript.get("canonical_sha256") == CONTRACT_SHA256
            and transcript.get("internally_consistent") is False
            and len(transcript_checks) == 23
            and sum(value is True for value in transcript_checks.values()) == 22
            and transcript_checks.get("terminal_p26_commit_tree_tag_and_bytes_exact") is False
            and all(
                value is True
                for key, value in transcript_checks.items()
                if key != "terminal_p26_commit_tree_tag_and_bytes_exact"
            )
            and committed_transcript
            == {
                "path": TRANSCRIPT_PATH,
                "mode": "0644",
                "sha256": TRANSCRIPT_SHA256,
                "byte_count": 1_718,
            }
        ),
        "remote_log_bindings_and_reconstruction_summary_exact": (
            prepare.get("invoked_utc") == "2026-09-06T14:11:41Z"
            and prepare.get("invocations_allowed") == 1
            and prepare.get("invocations_run") == 1
            and reconstruction.get("status") == "failed_closed_22_of_23"
            and reconstruction.get("checks") == {"total": 23, "true": 22, "false": 1}
            and reconstruction.get("only_failed_check")
            == "terminal_p26_commit_tree_tag_and_bytes_exact"
            and reconstruction.get("internally_consistent") is False
            and remote_stdout.get("mode") == "0600"
            and remote_stdout.get("uid") == remote_stdout.get("gid") == 1000
            and remote_stdout.get("sha256") == TRANSCRIPT_SHA256
            and remote_stdout.get("byte_count") == 1_718
            and remote_stderr.get("mode") == "0600"
            and remote_stderr.get("uid") == remote_stderr.get("gid") == 1000
            and remote_stderr.get("sha256") == EMPTY_SHA256
            and remote_stderr.get("byte_count") == 0
        ),
        "pre_container_artifact_inventory_exact": (
            retained.get("attempt_root") == "/secure/p27/attempt-20260906-01"
            and retained.get("regular_files")
            == [
                "evidence/p27-prepare-contract-reconstruction.stderr.log",
                "evidence/p27-prepare-contract-reconstruction.stdout.log",
            ]
            and retained.get("other_regular_files") == []
            and retained.get("container_created") is False
            and retained.get("runtime_lock_created") is False
            and retained.get("host_attestation_created") is False
            and retained.get("native_localization_artifact_created") is False
            and retained.get("sanitized_localization_artifact_created") is False
        ),
        "zero_cuda_localization_training_and_candidate_evidence_exact": (
            gates.get("contract_reconstruction") == "failed_closed_before_container"
            and gates.get("image_inspection") == "not_run"
            and gates.get("gpu_access") == "not_run"
            and gates.get("container_creation") == "not_run"
            and gates.get("runtime_freeze") == "not_run"
            and gates.get("cuda_initialization") == "not_run"
            and gates.get("deleted_mapping_localization") == "not_run"
            and gates.get("model_construction") == "not_run"
            and gates.get("optimizer_construction") == "not_run"
            and gates.get("forward_backward") == "not_run"
            and gates.get("optimizer_steps") == 0
            and gates.get("candidate_observations") == 0
            and gates.get("training") == "not_run"
            and theorem.get("deleted_mapping_identity_established") is False
            and theorem.get("cuda_repeatability_established") is False
            and theorem.get("real_gradient_fidelity_established") is False
        ),
        "terminal_policy_root_cause_and_route_exact": (
            policy
            == {
                "attempt_is_terminal": True,
                "reuse_attempt_20260906_01_allowed": False,
                "cleanup_attempt_20260906_01_allowed": False,
                "retry_until_favorable_allowed": False,
                "localization_invocation_consumed": False,
                "scientific_protocol_changed": False,
                "frozen_thresholds_changed": False,
            }
            and cause.get("classification") == "operator_transfer_bundle_omission"
            and cause.get("contract_or_localizer_defect_observed") is False
            and cause.get("cuda_or_backend_defect_observed") is False
            and cause.get("mapping_classification_observed") is False
            and route.get("new_attempt_required") is True
            and route.get("attempt_20260906_01_may_be_reused") is False
            and route.get("attempt_20260906_01_may_be_cleaned_up") is False
            and route.get("p28_created_by_this_outcome") is False
        ),
    }
    return {
        "schema_version": (
            "passive-muon-p27-cuda-deleted-mapping-localization-outcome-reconstruction-v1"
        ),
        "internally_consistent": all(checks.values()),
        "checks": checks,
        "canonical_sha256": _sha256(outcome_bytes),
        "claim_boundary": (
            "Repository-authenticated reconstruction plus an exact committed copy of the "
            "path-free remote stdout. Remote file ownership, attempt inventory, and transfer "
            "execution are operator-retained facts bound by hashes, not independently "
            "authenticated native bytes."
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
