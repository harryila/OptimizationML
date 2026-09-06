#!/usr/bin/env python3
"""Independently reconstruct the scoped terminal P26 acquisition outcome."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from collections.abc import Mapping
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANONICAL = ROOT / "results/summaries/p26_permission_safe_cuda_acquisition_outcome.json"
EXPECTED_CANONICAL_SHA256 = "9be5727213f1a23f00b05524940d88c2d02461510fb61316b29dacc6ee8ead47"

P25_OUTCOME_COMMIT = "f055405cc879ba0ac5afe26bc34a7336d5d2efbf"
P25_OUTCOME_TREE = "a5ee62de68b091719938829e579853b226b7f086"
P25_OUTCOME_PATH = "results/summaries/p25_cuda_diagnostic_outcome.json"
P25_OUTCOME_SHA256 = "d7a06f6f25bc4839db256720fe5c8ef3c99f6ea45b53b48da2135fcde76a423a"
SOURCE_FREEZE_COMMIT = "bc2c84880a7e6f3e762d7a05dfbd5048772b2c69"
SOURCE_FREEZE_TREE = "7332be7fdd15a1dff2374c10ae3c4555b7f60fec"
CONTROL_COMMIT = "ae16c1a4429a5363c7a6168d19ca2089ecddee35"
CONTROL_TREE = "a2537867260f322d9db5e1f48013f72846b7aa18"
PREREG_COMMIT = "e76ab62f92c95e6f0716cf2f1ed38a583cadfe56"
PREREG_TREE = "4fe0f57d50a8fa136bb192ec3fbac95b1e746aa2"
RUNTIME_COMMIT = "ba93225f3ef1abf5dbde71950c1eaa2e384a5dc0"
RUNTIME_TREE = "e7cdf25c86360ecdd42f2dc188ec416744321fe1"
BRIDGE_COMMIT = "185e444afc0b44ca0a09b1bde49a6b6fa3973355"
BRIDGE_TREE = "24f4bdac331a57bd7c1b807747d7c7fba253ee5a"

CONTRACT_PATH = "experiments/training/p26_permission_safe_acquisition_contract.json"
CONTRACT_SHA256 = "1a3562f90671384c5b90f4e830f3342ba993381c03738cd81685803563c538e1"
ORCHESTRATOR_PATH = "scripts/run_p26_permission_safe_acquisition.sh"
ORCHESTRATOR_SHA256 = "66dd6889912e0abfc7782e790356222851cc753d237018c6263942d3a7a4290a"
CONTRACT_RECONSTRUCTOR_PATH = "scripts/reconstruct_p26_permission_safe_acquisition.py"
CONTRACT_RECONSTRUCTOR_SHA256 = "27b1b5f65438f58f8c7733115c4d01f7965a583e9e809e28571770886040b246"
BRIDGE_VERIFIER_PATH = "scripts/verify_p26_permission_safe_bridge.py"
BRIDGE_VERIFIER_SHA256 = "2225a9822076e0055d8017f8b8549eb9e937434d4789463adb2fcec26ee3756a"
RUNTIME_LOCK_PATH = "experiments/training/p25_cuda_runtime_lock.json"
RUNTIME_LOCK_SHA256 = "04be2154daf5afbe97f7d2278ee7936da769d230663dfc6dfd50f0370a456755"
ATTESTATION_PATH = "experiments/training/p25_host_attestation.json"
ATTESTATION_SHA256 = "8caa4d761ca89741517e6292e174acd17948f4d312b8418d0056070d3247e046"
WRAPPER_PATH = "results/summaries/p25_executable_origin_diagnostic.sanitized.json"
WRAPPER_SHA256 = "ad33cb3e2e1e86f182496b8e3b73987bf4995290c3d90ed4342971e2a44507dd"
NATIVE_DIAGNOSTIC_SHA256 = "b9cc8dad897e27a36414a320cb9aa2f446b292c45e9a170ddb9dae1ac8ea96a9"
NATIVE_FAILURE_SHA256 = "b2a92b5062988534bc424d3ecadf3977ecedc0f80aa27628d4a0457858a64c91"

TRACE_STDOUT = "number of parameters: 123.55M\n"
TRACE_STDERR = (
    "P23 native failure artifact: /workspace/evidence/p23/trace-off-a-failure.json "
    f"sha256={NATIVE_FAILURE_SHA256}\n"
    "P23 blocked: deleted file-backed mapping is forbidden at /proc/self/maps line 17\n"
)
SANITIZER_STDERR = (
    "P23 blocked: undeclared absolute path cannot be sanitized: /opt/p23-venv/bin/python\n"
)
TERMINAL_MESSAGE = (
    "P26 permission-safe acquisition blocked: P23 sanitizer failed for "
    "trace-off-a-failure.json with exit 2\n"
)

EXPECTED_CONTROL_DELTA = [
    "M\texperiments/training/P26_PERMISSION_SAFE_ACQUISITION_RUNBOOK.md",
    "M\texperiments/training/p26_permission_safe_acquisition_contract.json",
    "M\tscripts/reconstruct_p26_permission_safe_acquisition.py",
    "M\tscripts/run_p26_permission_safe_acquisition.sh",
    "M\ttests/test_p26_permission_safe_acquisition_contract.py",
]
EXPECTED_RUNTIME_DELTA = [
    "A\texperiments/training/p25_cuda_runtime_lock.json",
    "A\texperiments/training/p25_host_attestation.json",
]
EXPECTED_BRIDGE_DELTA = [
    "A\tresults/summaries/p25_executable_origin_diagnostic.sanitized.json",
]
EXPECTED_TOP_LEVEL_ORDER = [
    "schema_version",
    "status",
    "attempt_id",
    "evidence_kind",
    "claim_boundary",
    "control_lineage",
    "execution_repository_lineage",
    "locked_runtime",
    "corrected_diagnostic",
    "pre_acquisition_checks",
    "trace_off_a_failure",
    "failure_sanitization",
    "acquisition_stop",
    "gate_status",
    "terminal_policy",
    "theorem_status",
    "route",
]
EXPECTED_GATE_STATUS = {
    "fresh_runtime": "completed_and_reviewed",
    "corrected_diagnostic": "passed_40_of_40",
    "permission_safe_bridge": "passed",
    "trace_off_a": "blocked_before_initial_state_hash_and_step_zero",
    "trace_off_a_failure_sanitization": "failed_closed_no_output",
    "trace_off_b": "not_run",
    "exact_repeatability": "not_run",
    "trace_on": "mechanically_barred",
    "observer_noninterference": "not_run",
    "candidate_observations": 0,
    "fidelity_aggregate": "not_run",
    "shield_execution": "not_run",
    "training": "not_run",
}
EXPECTED_TERMINAL_POLICY = {
    "attempt_is_terminal": True,
    "retry_until_favorable_allowed": False,
    "reuse_attempt_20260906_02_allowed": False,
    "frozen_p23_thresholds_changed": False,
    "scientific_protocol_changed": False,
}
EXPECTED_THEOREM_STATUS = {
    "p18_through_p21_results_affected": False,
    "cuda_repeatability_established": False,
    "observer_noninterference_established": False,
    "real_gradient_fidelity_established": False,
    "native_cuda_shield_certified": False,
}
EXPECTED_ROUTE = {
    "next_branch": "p27-cuda-deleted-mapping-localization",
    "reason": (
        "The permission-safe bridge succeeded, but the first trace stopped during initialized "
        "loaded-file collection on an unretained deleted mapping and the failure sanitizer "
        "independently rejected the declared Python executable path. P27 must reproduce and "
        "retain the exact first deleted-mapping record in a nontraining diagnostic, repair "
        "failure sanitization without weakening writable/deleted-origin rejection, freeze a "
        "new runtime and attempt, and only then resume the unchanged P23 sequence."
    ),
    "new_attempt_required": True,
    "threshold_tuning_allowed": False,
    "attempt_20260906_02_may_be_rerun": False,
}


class DuplicateKeyError(ValueError):
    """Raised when purported canonical JSON repeats a key."""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    return parser.parse_args()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _strict_json(data: bytes) -> object:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise DuplicateKeyError(key)
            result[key] = value
        return result

    def reject_nonfinite(token: str) -> object:
        raise ValueError(f"nonfinite JSON constant: {token}")

    return json.loads(
        data,
        object_pairs_hook=reject_duplicates,
        parse_constant=reject_nonfinite,
    )


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _exact(observed: object, expected: object) -> bool:
    if type(observed) is not type(expected):
        return False
    if isinstance(expected, dict):
        assert isinstance(observed, dict)
        return set(observed) == set(expected) and all(
            _exact(observed[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        assert isinstance(observed, list)
        return len(observed) == len(expected) and all(
            _exact(left, right) for left, right in zip(observed, expected, strict=True)
        )
    return observed == expected


@cache
def _git(*arguments: str) -> bytes | None:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_text(*arguments: str) -> str | None:
    value = _git(*arguments)
    return value.decode().strip() if value is not None else None


def _blob(commit: str, path: str) -> bytes | None:
    return _git("show", f"{commit}:{path}")


def _tree(commit: str) -> str | None:
    return _git_text("rev-parse", f"{commit}^{{tree}}")


def _parent(commit: str) -> str | None:
    return _git_text("show", "-s", "--format=%P", commit)


def _delta(parent: str, child: str) -> list[str] | None:
    value = _git_text("diff", "--name-status", parent, child)
    return value.splitlines() if value is not None and value else ([] if value == "" else None)


def _stream_exact(record: Mapping[str, object], text: str, *, with_text: bool) -> bool:
    expected: dict[str, object] = {
        "owner": "ubuntu:ubuntu",
        "mode": "0664",
        "sha256": _sha256(text.encode()),
        "byte_count": len(text.encode()),
    }
    if with_text:
        expected = {
            "owner": "ubuntu:ubuntu",
            "mode": "0664",
            "text": text,
            "sha256": _sha256(text.encode()),
            "byte_count": len(text.encode()),
        }
    return _exact(dict(record), expected)


def reconstruct(canonical: Path = DEFAULT_CANONICAL) -> dict[str, object]:
    """Reconstruct repository facts and check the scoped external-evidence record."""

    try:
        canonical_bytes = canonical.read_bytes()
        raw = _strict_json(canonical_bytes)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError, DuplicateKeyError):
        canonical_bytes = b""
        raw = None
    outcome = _mapping(raw)
    control_lineage = _mapping(outcome.get("control_lineage"))
    p25_record = _mapping(control_lineage.get("terminal_p25_outcome"))
    freeze_record = _mapping(control_lineage.get("pre_runtime_source_freeze"))
    control_record = _mapping(control_lineage.get("post_runtime_control"))
    contract_record = _mapping(control_record.get("contract"))
    source_hashes = _mapping(control_record.get("source_hashes"))
    execution_lineage = _mapping(outcome.get("execution_repository_lineage"))
    prereg_record = _mapping(execution_lineage.get("source_preregistration"))
    runtime_record = _mapping(execution_lineage.get("runtime_review"))
    lock_record = _mapping(runtime_record.get("runtime_lock"))
    attestation_record = _mapping(runtime_record.get("host_attestation"))
    bridge_record = _mapping(execution_lineage.get("diagnostic_bridge"))
    repository_wrapper = _mapping(bridge_record.get("repository_wrapper"))
    locked_runtime = _mapping(outcome.get("locked_runtime"))
    gpu_record = _mapping(locked_runtime.get("gpu"))
    diagnostic = _mapping(outcome.get("corrected_diagnostic"))
    diagnostic_checks = _mapping(diagnostic.get("checks"))
    native_diagnostic = _mapping(diagnostic.get("native_artifact"))
    retained_wrapper = _mapping(diagnostic.get("retained_sanitized_wrapper"))
    repository_diagnostic = _mapping(diagnostic.get("repository_sanitized_wrapper"))
    redaction_record = _mapping(diagnostic.get("redaction"))
    prechecks = _mapping(outcome.get("pre_acquisition_checks"))
    p26_reconstruction = _mapping(prechecks.get("p26_contract_reconstruction"))
    p25_reconstruction = _mapping(prechecks.get("p25_contract_reconstruction"))
    bridge_check = _mapping(prechecks.get("permission_safe_bridge"))
    failure = _mapping(outcome.get("trace_off_a_failure"))
    failure_stdout = _mapping(failure.get("stdout"))
    failure_stderr = _mapping(failure.get("stderr"))
    native_failure = _mapping(failure.get("native_failure_artifact"))
    native_failure_provenance = _mapping(failure.get("native_failure_provenance"))
    native_failure_repository = _mapping(native_failure_provenance.get("repository"))
    mapping_limit = _mapping(failure.get("rejected_mapping_evidence_limit"))
    sanitization = _mapping(outcome.get("failure_sanitization"))
    sanitizer_stdout = _mapping(sanitization.get("stdout"))
    sanitizer_stderr = _mapping(sanitization.get("stderr"))
    stop = _mapping(outcome.get("acquisition_stop"))
    terminal_message = _mapping(stop.get("terminal_message"))
    gate_status = _mapping(outcome.get("gate_status"))
    terminal_policy = _mapping(outcome.get("terminal_policy"))
    theorem_status = _mapping(outcome.get("theorem_status"))
    route = _mapping(outcome.get("route"))

    p25_outcome_blob = _blob(P25_OUTCOME_COMMIT, P25_OUTCOME_PATH)
    contract_blob = _blob(CONTROL_COMMIT, CONTRACT_PATH)
    orchestrator_blob = _blob(CONTROL_COMMIT, ORCHESTRATOR_PATH)
    control_reconstructor_blob = _blob(CONTROL_COMMIT, CONTRACT_RECONSTRUCTOR_PATH)
    bridge_verifier_blob = _blob(CONTROL_COMMIT, BRIDGE_VERIFIER_PATH)
    lock_blob = _blob(RUNTIME_COMMIT, RUNTIME_LOCK_PATH)
    attestation_blob = _blob(RUNTIME_COMMIT, ATTESTATION_PATH)
    wrapper_blob = _blob(BRIDGE_COMMIT, WRAPPER_PATH)

    def parsed(value: bytes | None) -> Mapping[str, object]:
        if value is None:
            return {}
        try:
            return _mapping(_strict_json(value))
        except (UnicodeError, ValueError, json.JSONDecodeError, DuplicateKeyError):
            return {}

    contract = parsed(contract_blob)
    runtime_lock = parsed(lock_blob)
    attestation = parsed(attestation_blob)
    wrapper = parsed(wrapper_blob)
    wrapper_manifest = _mapping(wrapper.get("manifest"))
    wrapper_checks = _mapping(wrapper_manifest.get("checks"))
    wrapper_native = _mapping(wrapper.get("native_artifact"))
    wrapper_redaction = _mapping(wrapper.get("redaction"))
    wrapper_repository = _mapping(wrapper_manifest.get("repository"))
    wrapper_observed_repository = _mapping(wrapper_repository.get("observed"))
    lock_container = _mapping(runtime_lock.get("container"))
    lock_gpu = _mapping(runtime_lock.get("gpu"))
    attested_container = _mapping(attestation.get("container"))
    attested_gpu = _mapping(attestation.get("gpu"))
    lock_mount_contract = _mapping(lock_container.get("mount_contract"))
    lock_mounts = lock_mount_contract.get("mounts")
    orchestrator_source = orchestrator_blob.decode() if orchestrator_blob is not None else ""
    runner_blob = _blob(
        BRIDGE_COMMIT, "experiments/training/run_p23_deterministic_cuda_shadow_trace.py"
    )
    runner_source = runner_blob.decode() if runner_blob is not None else ""
    lock_container_without_backlink = {
        key: value for key, value in lock_container.items() if key != "host_attestation_sha256"
    }

    expected_native_failure = {
        "retention": "external_native_bytes_not_committed",
        "path": "/secure/p25/attempt-20260906-02/evidence/trace-off-a-failure.json",
        "schema_version": "passive-muon-p23-native-failure-manifest-v1",
        "owner": "root:root",
        "mode": "0600",
        "sha256": NATIVE_FAILURE_SHA256,
        "byte_count": 66_283,
        "strict_validation_reached_before_sanitization_transform": True,
    }
    expected_mapping_limit = {
        "proc_maps_line_number": 17,
        "raw_mapping_line_retained": False,
        "raw_mapping_line": None,
        "failed_process_still_available": False,
        "reason": (
            "The native failure manifest retained the line number and error message but not "
            "the rejected /proc/self/maps text. PID 137 had exited before inspection, so the "
            "exact raw mapping cannot be recovered and must not be guessed."
        ),
    }
    expected_native_failure_provenance = {
        "repository": {
            "prepared_snapshot": {
                "collection_status": "prepared_context_validated",
                "head": BRIDGE_COMMIT,
                "tree": BRIDGE_TREE,
                "clean": True,
            },
            "failure_time_snapshot": {
                "collection_status": "recollected_at_failure",
                "head": BRIDGE_COMMIT,
                "tree": BRIDGE_TREE,
                "clean": True,
            },
            "snapshots_match": True,
        },
        "prepared_runtime_sha256": (
            "c40abffcc32b2596350ae94e11020ccea49801c32d19d940dbc9b86aafa45e0f"
        ),
        "prepared_static_contract_sha256": (
            "1c7be31c697e443dad70d0c05e9b8f070c85f1d5b3ed2a64b2399c56690d567d"
        ),
    }
    expected_native_present = ["trace-off-a-failure.json"]
    expected_native_absent = [
        "trace-off-a.json",
        "trace-off-b.json",
        "trace-off-b-failure.json",
        "repeatability.json",
        "trace-on.json",
        "trace-on-failure.json",
        "raw-trace.json",
        "noninterference.json",
        "aggregate.json",
    ]
    expected_log_present = [
        "p26-contract-reconstruction",
        "p26-p25-contract-reconstruction",
        "p26-permission-safe-bridge-verification",
        "p23-trace-off-a",
        "p23-sanitize-trace-off-a-failure",
    ]
    expected_log_absent = [
        "p23-trace-off-b",
        "p23-verify-repeatability",
        "p23-trace-on",
        "p23-verify-noninterference",
        "p23-aggregate",
    ]

    checks = {
        "canonical_bytes_and_closed_schema_exact": (
            isinstance(raw, Mapping)
            and canonical_bytes == (json.dumps(raw, indent=2) + "\n").encode()
            and _sha256(canonical_bytes) == EXPECTED_CANONICAL_SHA256
            and list(raw) == EXPECTED_TOP_LEVEL_ORDER
            and set(raw) == set(EXPECTED_TOP_LEVEL_ORDER)
        ),
        "identity_status_and_claim_boundary_exact": (
            outcome.get("schema_version")
            == "passive-muon-p26-permission-safe-cuda-acquisition-outcome-v1"
            and outcome.get("status") == "terminal_trace_off_a_deleted_mapping_and_sanitizer_stop"
            and outcome.get("attempt_id") == "20260906-02"
            and outcome.get("evidence_kind")
            == (
                "repository_authenticated_control_runtime_and_bridge_plus_operator_retained_"
                "native_failure"
            )
            and isinstance(outcome.get("claim_boundary"), str)
            and "cannot be reconstructed" in str(outcome.get("claim_boundary"))
            and "This is not CUDA repeatability" in str(outcome.get("claim_boundary"))
        ),
        "p25_terminal_parent_commit_tree_and_outcome_exact": (
            _tree(P25_OUTCOME_COMMIT) == P25_OUTCOME_TREE
            and p25_outcome_blob is not None
            and _sha256(p25_outcome_blob) == P25_OUTCOME_SHA256
            and p25_record
            == {
                "commit": P25_OUTCOME_COMMIT,
                "tree": P25_OUTCOME_TREE,
                "outcome_path": P25_OUTCOME_PATH,
                "outcome_sha256": P25_OUTCOME_SHA256,
                "attempt_id": "20260906-01",
                "attempt_is_terminal": True,
            }
        ),
        "control_commit_chain_and_delta_exact": (
            _tree(SOURCE_FREEZE_COMMIT) == SOURCE_FREEZE_TREE
            and _parent(SOURCE_FREEZE_COMMIT) == P25_OUTCOME_COMMIT
            and _tree(CONTROL_COMMIT) == CONTROL_TREE
            and _parent(CONTROL_COMMIT) == SOURCE_FREEZE_COMMIT
            and _delta(SOURCE_FREEZE_COMMIT, CONTROL_COMMIT) == EXPECTED_CONTROL_DELTA
            and freeze_record
            == {
                "commit": SOURCE_FREEZE_COMMIT,
                "tree": SOURCE_FREEZE_TREE,
                "parent": P25_OUTCOME_COMMIT,
            }
            and control_record.get("commit") == CONTROL_COMMIT
            and control_record.get("tree") == CONTROL_TREE
            and control_record.get("parent") == SOURCE_FREEZE_COMMIT
            and control_record.get("exact_delta") == EXPECTED_CONTROL_DELTA
        ),
        "control_contract_and_source_blobs_exact": (
            contract_blob is not None
            and _sha256(contract_blob) == CONTRACT_SHA256
            and contract.get("schema_version")
            == "passive-muon-p26-permission-safe-acquisition-contract-v2"
            and contract.get("status")
            == "frozen_post_fresh_runtime_pre_diagnostic_and_pre_acquisition"
            and contract_record
            == {
                "path": CONTRACT_PATH,
                "schema_version": contract.get("schema_version"),
                "status": contract.get("status"),
                "sha256": CONTRACT_SHA256,
            }
            and orchestrator_blob is not None
            and _sha256(orchestrator_blob) == ORCHESTRATOR_SHA256
            and control_reconstructor_blob is not None
            and _sha256(control_reconstructor_blob) == CONTRACT_RECONSTRUCTOR_SHA256
            and bridge_verifier_blob is not None
            and _sha256(bridge_verifier_blob) == BRIDGE_VERIFIER_SHA256
            and source_hashes
            == {
                "host_orchestrator": ORCHESTRATOR_SHA256,
                "independent_reconstructor": CONTRACT_RECONSTRUCTOR_SHA256,
                "root_container_verifier": BRIDGE_VERIFIER_SHA256,
            }
        ),
        "execution_repository_commit_chain_and_deltas_exact": (
            _tree(PREREG_COMMIT) == PREREG_TREE
            and prereg_record == {"commit": PREREG_COMMIT, "tree": PREREG_TREE}
            and _tree(RUNTIME_COMMIT) == RUNTIME_TREE
            and _parent(RUNTIME_COMMIT) == PREREG_COMMIT
            and _delta(PREREG_COMMIT, RUNTIME_COMMIT) == EXPECTED_RUNTIME_DELTA
            and runtime_record.get("commit") == RUNTIME_COMMIT
            and runtime_record.get("tree") == RUNTIME_TREE
            and runtime_record.get("parent") == PREREG_COMMIT
            and runtime_record.get("exact_delta") == EXPECTED_RUNTIME_DELTA
            and _tree(BRIDGE_COMMIT) == BRIDGE_TREE
            and _parent(BRIDGE_COMMIT) == RUNTIME_COMMIT
            and _delta(RUNTIME_COMMIT, BRIDGE_COMMIT) == EXPECTED_BRIDGE_DELTA
            and bridge_record.get("commit") == BRIDGE_COMMIT
            and bridge_record.get("tree") == BRIDGE_TREE
            and bridge_record.get("parent") == RUNTIME_COMMIT
            and bridge_record.get("exact_delta") == EXPECTED_BRIDGE_DELTA
        ),
        "runtime_lock_and_attestation_blobs_reconstruct": (
            lock_blob is not None
            and len(lock_blob) == 6_599
            and _sha256(lock_blob) == RUNTIME_LOCK_SHA256
            and lock_record
            == {
                "path": RUNTIME_LOCK_PATH,
                "schema_version": "passive-muon-p23-cuda-runtime-lock-v3",
                "status": "pinned_for_acquisition",
                "sha256": RUNTIME_LOCK_SHA256,
                "byte_count": 6_599,
            }
            and runtime_lock.get("schema_version") == lock_record.get("schema_version")
            and runtime_lock.get("status") == lock_record.get("status")
            and attestation_blob is not None
            and len(attestation_blob) == 4_124
            and _sha256(attestation_blob) == ATTESTATION_SHA256
            and attestation_record
            == {
                "path": ATTESTATION_PATH,
                "schema_version": "passive-muon-p23-host-attestation-v3",
                "status": "procedurally_host_attested",
                "sha256": ATTESTATION_SHA256,
                "byte_count": 4_124,
            }
            and attestation.get("schema_version") == attestation_record.get("schema_version")
            and attestation.get("status") == attestation_record.get("status")
            and lock_container_without_backlink == attested_container
            and lock_gpu == attested_gpu
        ),
        "locked_runtime_record_matches_committed_lock": (
            locked_runtime.get("attempt_root") == "/secure/p25/attempt-20260906-02"
            and locked_runtime.get("execution_repository_head") == BRIDGE_COMMIT
            and locked_runtime.get("execution_repository_tree") == BRIDGE_TREE
            and locked_runtime.get("control_repository_head") == CONTROL_COMMIT
            and locked_runtime.get("control_repository_tree") == CONTROL_TREE
            and locked_runtime.get("image") == lock_container.get("image")
            and locked_runtime.get("image_digest") == lock_container.get("repository_digest")
            and locked_runtime.get("build_metadata_sha256")
            == "13e0f05189321f86b4d8a41253eed97768bb1774ac7db64ad7c8b25e2f026091"
            and locked_runtime.get("container_name") == "p25-acquisition-20260906-02"
            and locked_runtime.get("container_id") == lock_container.get("container_id")
            and locked_runtime.get("container_init_pid") == lock_container.get("container_init_pid")
            and locked_runtime.get("restart_count") == 0
            and locked_runtime.get("network_mode") == lock_container.get("network_mode")
            and locked_runtime.get("root_filesystem_read_only")
            == lock_container.get("rootfs_read_only")
            and isinstance(lock_mounts, list)
            and locked_runtime.get("bind_mount_count") == len(lock_mounts) == 10
            and gpu_record
            == {
                "uuid": lock_gpu.get("uuid"),
                "name": lock_gpu.get("name"),
                "compute_capability": lock_gpu.get("compute_capability"),
                "driver_version": lock_gpu.get("driver_version"),
                "mig_mode": lock_gpu.get("mig_mode"),
            }
        ),
        "committed_corrected_diagnostic_wrapper_reconstructs": (
            wrapper_blob is not None
            and len(wrapper_blob) == 5_099_937
            and _sha256(wrapper_blob) == WRAPPER_SHA256
            and _git_text("ls-tree", BRIDGE_COMMIT, "--", WRAPPER_PATH).startswith("100644 ")
            and repository_wrapper
            == {
                "path": WRAPPER_PATH,
                "mode": "0644",
                "sha256": WRAPPER_SHA256,
                "byte_count": 5_099_937,
            }
            and wrapper.get("schema_version")
            == "passive-muon-p25-sanitized-executable-origin-diagnostic-v1"
            and wrapper.get("status") == "passes"
            and wrapper.get("passes") is True
            and wrapper_manifest.get("status") == "passes"
            and wrapper_manifest.get("passes") is True
            and len(wrapper_checks) == 40
            and all(value is True for value in wrapper_checks.values())
            and wrapper_native == {"byte_count": 4_115_671, "sha256": NATIVE_DIAGNOSTIC_SHA256}
            and wrapper_redaction
            == {
                "key_replacement_count": 2,
                "total_replacement_count": 37_917,
                "value_replacement_count": 37_915,
            }
            and wrapper_observed_repository.get("head") == RUNTIME_COMMIT
            and wrapper_observed_repository.get("tree") == RUNTIME_TREE
            and wrapper_observed_repository.get("dirty") is False
        ),
        "corrected_diagnostic_external_bindings_exact": (
            diagnostic.get("attempts_allowed") == 1
            and diagnostic.get("attempts_run") == 1
            and diagnostic.get("status") == "passed_40_of_40"
            and diagnostic_checks == {"total": 40, "true": 40, "false": 0}
            and native_diagnostic
            == {
                "retention": "external_native_bytes_not_committed",
                "file_name": "p25-remediated-executable-origin.native.json",
                "owner": "root:root",
                "mode": "0600",
                "sha256": NATIVE_DIAGNOSTIC_SHA256,
                "byte_count": 4_115_671,
            }
            and retained_wrapper
            == {
                "retention": "external_copy_byte_identical_to_repository_wrapper",
                "file_name": "p25-remediated-executable-origin.sanitized.json",
                "owner": "root:root",
                "mode": "0600",
                "sha256": WRAPPER_SHA256,
                "byte_count": 5_099_937,
            }
            and repository_diagnostic
            == {
                "commit": BRIDGE_COMMIT,
                "owner": "ubuntu:ubuntu",
                "mode": "0644",
                "sha256": WRAPPER_SHA256,
                "byte_count": 5_099_937,
            }
            and redaction_record == wrapper_redaction
        ),
        "pre_acquisition_reconstruction_and_bridge_logs_exact": (
            p26_reconstruction.get("status") == "passed_19_of_19"
            and _mapping(p26_reconstruction.get("stdout"))
            == {
                "owner": "ubuntu:ubuntu",
                "mode": "0664",
                "sha256": "1fd07e033243181cac318381e312bc2b32bdb65536b7e7d6bd3171f6a24774e5",
                "byte_count": 1_443,
            }
            and _stream_exact(_mapping(p26_reconstruction.get("stderr")), "", with_text=False)
            and p25_reconstruction.get("status") == "passed"
            and _mapping(p25_reconstruction.get("stdout"))
            == {
                "owner": "ubuntu:ubuntu",
                "mode": "0664",
                "sha256": "7a51056becfdc41729011403092c5dae2095da774d900cc152a550cd8f730279",
                "byte_count": 1_443,
            }
            and _stream_exact(_mapping(p25_reconstruction.get("stderr")), "", with_text=False)
            and bridge_check.get("status") == "passed"
            and bridge_check.get("execution_context")
            == "root_inside_attested_network_none_read_only_root_container"
            and bridge_check.get("retained_native_and_wrapper_mode_preserved") is True
            and bridge_check.get("repository_and_retained_wrapper_bytes_equal") is True
            and _mapping(bridge_check.get("stdout"))
            == {
                "owner": "ubuntu:ubuntu",
                "mode": "0664",
                "sha256": "6c54cb9db6f0bf9dad55899acd14ea9699b29c97381679644aab2061dd437ebd",
                "byte_count": 915,
            }
            and _stream_exact(_mapping(bridge_check.get("stderr")), "", with_text=False)
        ),
        "source_orders_passed_bridge_before_one_trace_and_failure_sanitizer": (
            "run_logged_hash_bound_stdin p26-permission-safe-bridge-verification"
            in orchestrator_source
            and "run_logged p23-trace-off-a" in orchestrator_source
            and "sanitize_one trace-off-a-failure.json trace-off-a-failure" in orchestrator_source
            and "run_logged p23-trace-off-b" in orchestrator_source
            and orchestrator_source.index(
                "run_logged_hash_bound_stdin p26-permission-safe-bridge-verification"
            )
            < orchestrator_source.index("run_logged p23-trace-off-a")
            < orchestrator_source.index("sanitize_one trace-off-a-failure.json trace-off-a-failure")
            < orchestrator_source.index("run_logged p23-trace-off-b")
        ),
        "trace_off_a_native_failure_record_exact": (
            failure.get("attempts_allowed") == 1
            and failure.get("attempts_run") == 1
            and failure.get("role") == "trace_off_a"
            and failure.get("status") == "blocked_before_success_artifact"
            and failure.get("runner_exit_code") == 2
            and failure.get("failure_phase") == "p22_initialized"
            and failure.get("failure_time_ns") == 1_788_698_138_406_675_519
            and failure.get("failure_utc") == "2026-09-06T12:35:38Z"
            and time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(int(failure.get("failure_time_ns", 0)) / 1e9),
            )
            == failure.get("failure_utc")
            and failure.get("process_pid") == 137
            and failure.get("model_and_optimizer_constructed") is True
            and failure.get("initial_state_hash_computed") is False
            and failure.get("forward_backward_started") is False
            and failure.get("optimizer_steps_completed") == 0
            and failure.get("candidate_observations") == 0
            and failure.get("error_class")
            == "p23_deterministic_cuda_shadow_trace.P23ProvenanceError"
            and failure.get("error_message")
            == "deleted file-backed mapping is forbidden at /proc/self/maps line 17"
            and _stream_exact(failure_stdout, TRACE_STDOUT, with_text=True)
            and _stream_exact(failure_stderr, TRACE_STDERR, with_text=True)
            and native_failure == expected_native_failure
            and native_failure_provenance == expected_native_failure_provenance
            and native_failure_repository.get("snapshots_match") is True
        ),
        "runner_source_explains_failure_phase_and_native_artifact": (
            "P23_NATIVE_FAILURE_MANIFEST_SCHEMA: Final" in runner_source
            and '"passive-muon-p23-native-failure-manifest-v1"' in runner_source
            and '_set_failure_phase("p22_initialized")' in runner_source
            and '"error_message": str(error)' in runner_source
            and '"traceback": traceback_text' in runner_source
            and "_write_json_atomic_new(failure_output, failure)" in runner_source
            and "_validate_sanitizable_artifact(payload)" in runner_source
            and "P23.sanitize_complete_manifest(payload" in runner_source
            and runner_source.index("_validate_sanitizable_artifact(payload)")
            < runner_source.index("P23.sanitize_complete_manifest(payload")
        ),
        "raw_deleted_mapping_evidence_limit_exact": (
            mapping_limit == expected_mapping_limit
            and mapping_limit.get("raw_mapping_line_retained") is False
            and mapping_limit.get("raw_mapping_line") is None
            and "must not be guessed" in str(mapping_limit.get("reason"))
        ),
        "failure_sanitizer_stop_and_no_output_exact": (
            sanitization.get("status") == "failed_closed_no_output"
            and sanitization.get("exit_code") == 2
            and sanitization.get("sanitized_failure_artifact_created") is False
            and _stream_exact(sanitizer_stdout, "", with_text=False)
            and _stream_exact(sanitizer_stderr, SANITIZER_STDERR, with_text=True)
            and sanitization.get("classification")
            == "declared_python_executable_path_rejected_by_failure_manifest_sanitizer"
            and isinstance(sanitization.get("qualification"), str)
            and "second fail-closed condition" in str(sanitization.get("qualification"))
            and "sanitized artifact is absent" in str(sanitization.get("qualification"))
        ),
        "terminal_inventory_and_no_subsequent_gate_exact": (
            stop.get("host_orchestrator_exit_code") == 1
            and terminal_message
            == {
                "text": TERMINAL_MESSAGE,
                "sha256": _sha256(TERMINAL_MESSAGE.encode()),
                "byte_count": len(TERMINAL_MESSAGE.encode()),
            }
            and stop.get("native_present_names") == expected_native_present
            and stop.get("native_absent_names") == expected_native_absent
            and stop.get("sanitized_present_names") == []
            and stop.get("process_log_labels_present") == expected_log_present
            and stop.get("process_log_labels_absent") == expected_log_absent
        ),
        "gate_terminal_theorem_and_route_scope_exact": (
            _exact(dict(gate_status), EXPECTED_GATE_STATUS)
            and _exact(dict(terminal_policy), EXPECTED_TERMINAL_POLICY)
            and _exact(dict(theorem_status), EXPECTED_THEOREM_STATUS)
            and _exact(dict(route), EXPECTED_ROUTE)
        ),
    }
    return {
        "schema_version": (
            "passive-muon-p26-permission-safe-cuda-acquisition-outcome-reconstruction-v1"
        ),
        "canonical_sha256": _sha256(canonical_bytes),
        "checks": checks,
        "internally_consistent": all(checks.values()),
        "scope": (
            "Repository reconstruction of P26 control, runtime, and diagnostic-bridge facts "
            "plus internal consistency of the operator-retained terminal attempt record; not "
            "independent authentication of external native bytes, filesystem modes, process "
            "logs, the unretained raw maps line, or remote artifact absence."
        ),
    }


def main() -> int:
    result = reconstruct(_parse_args().canonical.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["internally_consistent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
