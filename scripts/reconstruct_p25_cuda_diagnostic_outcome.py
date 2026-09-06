#!/usr/bin/env python3
"""Independently reconstruct the scoped terminal P25 outcome packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Mapping
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANONICAL = ROOT / "results/summaries/p25_cuda_diagnostic_outcome.json"
EXPECTED_CANONICAL_SHA256 = "d7a06f6f25bc4839db256720fe5c8ef3c99f6ea45b53b48da2135fcde76a423a"
BOOTSTRAP_COMMIT = "5814621979ffedf370f04f3d99d925d8584b640f"
BOOTSTRAP_TREE = "31cafc8bae9a7311b18940c267a4da52ed6e8068"
PREREG_COMMIT = "e76ab62f92c95e6f0716cf2f1ed38a583cadfe56"
PREREG_TREE = "4fe0f57d50a8fa136bb192ec3fbac95b1e746aa2"
RUNTIME_COMMIT = "a967f30dae3c8926f9ce5c4783a0923e42bcdb0d"
RUNTIME_TREE = "8dca3402f27dfe1d0770cc2849fcca24778d76d8"
BRIDGE_COMMIT = "a037b04d0ae1e9c5b6b371ae78293e5e5640d91c"
BRIDGE_TREE = "ea5941ee873f518f147f51fd15555989c331caef"
CONTRACT_PATH = "experiments/training/p25_cuda_diagnostic_contract.json"
CONTRACT_SHA256 = "51e1fc22f685cc904fcdfff2775cff4293fdaa67a6e048072ee4dc49777f34e2"
RUNTIME_LOCK_PATH = "experiments/training/p25_cuda_runtime_lock.json"
RUNTIME_LOCK_SHA256 = "95f06d7fcf6731c331811fc35cb48fa89606f22d53c968e5c5d625ca10e117f3"
HOST_ATTESTATION_PATH = "experiments/training/p25_host_attestation.json"
HOST_ATTESTATION_SHA256 = "2dc43b993028e624a0026fe0446f82e506ff51bf11c9bb38dc9fd17a173ab7b2"
SANITIZED_PATH = "results/summaries/p25_executable_origin_diagnostic.sanitized.json"
SANITIZED_SHA256 = "56983629e935e3e18ac89b894ee84bc56e47f862abfcc36a6a5a15dbb8e5a354"
NATIVE_SHA256 = "e6479d1a02556cee451cbcf1da3f8b10df498bca7835e9db3730a08f886d9184"
GENERATED_SHA256 = "8205b16956fb264841ecd8644784a0d157f87df79b17c16825dc1163433ce5d8"
ORCHESTRATOR_PATH = "scripts/run_p25_cuda_attempt.sh"
STOP_STDERR = "P25 host orchestration blocked: reviewed and retained diagnostic wrappers differ\n"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
EXPECTED_TOP_LEVEL_KEYS = {
    "schema_version",
    "status",
    "attempt_id",
    "evidence_kind",
    "claim_boundary",
    "frozen_source_chain",
    "replacement_runtime",
    "corrected_diagnostic",
    "diagnostic_bridge",
    "acquisition_stop",
    "gate_status",
    "terminal_policy",
    "theorem_status",
    "route",
}
EXPECTED_PREREG_DELTA = [
    "A\tresults/summaries/p25_historical_p24_native_outcome.json",
]
EXPECTED_RUNTIME_DELTA = [
    "A\texperiments/training/p25_cuda_runtime_lock.json",
    "A\texperiments/training/p25_host_attestation.json",
]
EXPECTED_BRIDGE_DELTA = [
    "A\tresults/summaries/p25_executable_origin_diagnostic.sanitized.json",
]
EXPECTED_GATE_STATUS = {
    "replacement_image_and_runtime": "completed",
    "corrected_diagnostic": "passed_40_of_40",
    "corrected_sanitization": "passed_committed_full_manifest",
    "acquisition_host_preflight": "failed_closed_on_unreadable_retained_wrapper",
    "trace_off_a": "not_started",
    "trace_off_b": "not_run",
    "exact_repeatability": "not_run",
    "trace_on": "not_run",
    "observer_noninterference": "not_run",
    "candidate_observations": 0,
    "fidelity_aggregate": "not_run",
    "training": "not_run",
    "trace_repeatability_noninterference_aggregate_artifact_count": 0,
    "acquisition_process_stream_artifact_count": 0,
    "bridge_reconstruction_or_verification_log_count": 0,
}
EXPECTED_ABSENT_PRE_TRACE_LOGS = [
    "p25-bridge-contract-reconstruction.stdout.log",
    "p25-bridge-contract-reconstruction.stderr.log",
    "p25-diagnostic-bridge-verification.stdout.log",
    "p25-diagnostic-bridge-verification.stderr.log",
]
EXPECTED_NATIVE_ACQUISITION_NAMES = [
    "trace-off-a.json",
    "trace-off-a-failure.json",
    "trace-off-b.json",
    "trace-off-b-failure.json",
    "repeatability.json",
    "trace-on.json",
    "trace-on-failure.json",
    "raw-trace.json",
    "noninterference.json",
    "aggregate.json",
]
EXPECTED_ACQUISITION_PROCESS_LABELS = [
    "p23-trace-off-a",
    "p23-trace-off-b",
    "p23-verify-repeatability",
    "p23-trace-on",
    "p23-verify-noninterference",
    "p23-aggregate",
]


class DuplicateKeyError(ValueError):
    """Raised when a purported canonical JSON object repeats a key."""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    return parser.parse_args()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _strict_json(value: bytes) -> object:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                raise DuplicateKeyError(key)
            result[key] = item
        return result

    def reject_nonfinite(token: str) -> object:
        raise ValueError(f"nonfinite JSON constant: {token}")

    return json.loads(
        value,
        object_pairs_hook=reject_duplicates,
        parse_constant=reject_nonfinite,
    )


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _exact_value(observed: object, expected: object) -> bool:
    if type(observed) is not type(expected):
        return False
    if isinstance(expected, dict):
        assert isinstance(observed, dict)
        return set(observed) == set(expected) and all(
            _exact_value(observed[key], item) for key, item in expected.items()
        )
    if isinstance(expected, list):
        assert isinstance(observed, list)
        return len(observed) == len(expected) and all(
            _exact_value(left, right) for left, right in zip(observed, expected, strict=True)
        )
    return observed == expected


@cache
def _git_output(*arguments: str) -> bytes:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


def _commit_blob(commit: str, path: str) -> bytes:
    return _git_output("show", f"{commit}:{path}")


@cache
def _commit_mapping(commit: str, path: str) -> Mapping[str, object]:
    parsed = _strict_json(_commit_blob(commit, path))
    if not isinstance(parsed, Mapping):
        raise ValueError(f"committed JSON root is not an object: {commit}:{path}")
    return parsed


def _tree(commit: str) -> str:
    return _git_output("rev-parse", f"{commit}^{{tree}}").decode().strip()


def _parent(commit: str) -> str:
    return _git_output("rev-parse", f"{commit}^").decode().strip()


def _delta(parent: str, child: str) -> list[str]:
    output = _git_output("diff", "--name-status", parent, child).decode().strip()
    return output.splitlines() if output else []


def reconstruct(canonical: Path = DEFAULT_CANONICAL) -> dict[str, object]:
    """Reconstruct repository facts while preserving external-evidence limits."""

    try:
        canonical_bytes = canonical.read_bytes()
        raw = _strict_json(canonical_bytes)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError, DuplicateKeyError):
        canonical_bytes = b""
        raw = None
    outcome = _mapping(raw)
    chain = _mapping(outcome.get("frozen_source_chain"))
    bootstrap = _mapping(chain.get("source_freeze_bootstrap"))
    prereg = _mapping(chain.get("preregistration"))
    contract_record = _mapping(chain.get("contract"))
    runtime = _mapping(outcome.get("replacement_runtime"))
    runtime_lock_record = _mapping(runtime.get("runtime_lock"))
    attestation_record = _mapping(runtime.get("host_attestation"))
    diagnostic = _mapping(outcome.get("corrected_diagnostic"))
    native_record = _mapping(diagnostic.get("native_artifact"))
    sanitized_record = _mapping(diagnostic.get("sanitized_artifact"))
    redaction_record = _mapping(sanitized_record.get("redaction"))
    generated_record = _mapping(diagnostic.get("generated_module"))
    diagnostic_checks = _mapping(diagnostic.get("checks"))
    bridge = _mapping(outcome.get("diagnostic_bridge"))
    stop = _mapping(outcome.get("acquisition_stop"))
    comparison_codes = _mapping(stop.get("comparison_exit_codes"))
    stop_stream = _mapping(stop.get("client_stderr"))
    root_cause = _mapping(stop.get("root_cause"))
    retained_wrapper = _mapping(root_cause.get("retained_evidence_wrapper"))
    repository_wrapper = _mapping(root_cause.get("repository_wrapper"))
    zero_inventory = _mapping(stop.get("zero_acquisition_artifact_inventory"))
    gate_status = _mapping(outcome.get("gate_status"))
    terminal = _mapping(outcome.get("terminal_policy"))
    theorem = _mapping(outcome.get("theorem_status"))
    route = _mapping(outcome.get("route"))

    contract_bytes = _commit_blob(BOOTSTRAP_COMMIT, CONTRACT_PATH)
    runtime_lock_bytes = _commit_blob(RUNTIME_COMMIT, RUNTIME_LOCK_PATH)
    attestation_bytes = _commit_blob(RUNTIME_COMMIT, HOST_ATTESTATION_PATH)
    sanitized_bytes = _commit_blob(BRIDGE_COMMIT, SANITIZED_PATH)
    contract = _commit_mapping(BOOTSTRAP_COMMIT, CONTRACT_PATH)
    runtime_lock = _commit_mapping(RUNTIME_COMMIT, RUNTIME_LOCK_PATH)
    attestation = _commit_mapping(RUNTIME_COMMIT, HOST_ATTESTATION_PATH)
    sanitized = _commit_mapping(BRIDGE_COMMIT, SANITIZED_PATH)
    sanitized_manifest = _mapping(sanitized.get("manifest"))
    sanitized_checks = _mapping(sanitized_manifest.get("checks"))
    sanitized_native = _mapping(sanitized.get("native_artifact"))
    sanitized_redaction = _mapping(sanitized.get("redaction"))
    sanitized_repository = _mapping(sanitized_manifest.get("repository"))
    observed_repository = _mapping(sanitized_repository.get("observed"))
    sanitized_generated = _mapping(
        _mapping(sanitized_manifest.get("trigger")).get("generated_module")
    )
    orchestrator_source = _commit_blob(BRIDGE_COMMIT, ORCHESTRATOR_PATH).decode("utf-8")
    tracked_at_bridge = set(
        _git_output("ls-tree", "-r", "--name-only", BRIDGE_COMMIT).decode().splitlines()
    )
    forbidden_acquisition_names = {
        "trace-off-a.json",
        "trace-off-b.json",
        "repeatability.json",
        "trace-on.json",
        "raw-trace.json",
        "noninterference.json",
        "aggregate.json",
    }

    checks = {
        "canonical_bytes_exact": _sha256(canonical_bytes) == EXPECTED_CANONICAL_SHA256,
        "closed_top_level_and_terminal_status": (
            isinstance(raw, Mapping)
            and set(raw) == EXPECTED_TOP_LEVEL_KEYS
            and outcome.get("schema_version") == "passive-muon-p25-cuda-diagnostic-outcome-v1"
            and outcome.get("status") == "terminal_pre_trace_host_wrapper_permission_stop"
            and outcome.get("attempt_id") == "20260906-01"
        ),
        "claim_boundary_separates_repository_and_external_facts": (
            outcome.get("evidence_kind")
            == "repository_authenticated_diagnostic_plus_operator_recorded_pre_trace_stop"
            and isinstance(outcome.get("claim_boundary"), str)
            and "operator-recorded external facts" in str(outcome.get("claim_boundary"))
            and "No CUDA gradient" in str(outcome.get("claim_boundary"))
        ),
        "source_freeze_bootstrap_exact": (
            _tree(BOOTSTRAP_COMMIT) == BOOTSTRAP_TREE
            and bootstrap.get("commit") == BOOTSTRAP_COMMIT
            and bootstrap.get("tree") == BOOTSTRAP_TREE
        ),
        "preregistration_direct_child_and_delta_exact": (
            _tree(PREREG_COMMIT) == PREREG_TREE
            and _parent(PREREG_COMMIT) == BOOTSTRAP_COMMIT
            and _delta(BOOTSTRAP_COMMIT, PREREG_COMMIT) == EXPECTED_PREREG_DELTA
            and prereg.get("commit") == PREREG_COMMIT
            and prereg.get("tree") == PREREG_TREE
            and prereg.get("parent") == BOOTSTRAP_COMMIT
            and prereg.get("exact_delta") == EXPECTED_PREREG_DELTA
        ),
        "contract_bytes_and_status_exact": (
            _sha256(contract_bytes) == CONTRACT_SHA256
            and contract_record.get("path") == CONTRACT_PATH
            and contract_record.get("sha256") == CONTRACT_SHA256
            and contract_record.get("schema_version") == contract.get("schema_version")
            and contract_record.get("status") == contract.get("status")
            and contract.get("schema_version")
            == "passive-muon-p25-cuda-diagnostic-correction-contract-v1"
            and contract.get("status") == "frozen_pre_remediation_build_and_pre_acquisition"
        ),
        "runtime_direct_child_and_delta_exact": (
            _tree(RUNTIME_COMMIT) == RUNTIME_TREE
            and _parent(RUNTIME_COMMIT) == PREREG_COMMIT
            and _delta(PREREG_COMMIT, RUNTIME_COMMIT) == EXPECTED_RUNTIME_DELTA
            and runtime.get("commit") == RUNTIME_COMMIT
            and runtime.get("tree") == RUNTIME_TREE
            and runtime.get("parent") == PREREG_COMMIT
            and runtime.get("exact_delta") == EXPECTED_RUNTIME_DELTA
        ),
        "runtime_lock_bytes_and_schema_exact": (
            _sha256(runtime_lock_bytes) == RUNTIME_LOCK_SHA256
            and len(runtime_lock_bytes) == 6599
            and runtime_lock_record.get("path") == RUNTIME_LOCK_PATH
            and runtime_lock_record.get("sha256") == RUNTIME_LOCK_SHA256
            and runtime_lock_record.get("byte_count") == 6599
            and runtime_lock_record.get("schema_version") == runtime_lock.get("schema_version")
            and runtime_lock_record.get("status") == runtime_lock.get("status")
            and runtime_lock.get("schema_version") == "passive-muon-p23-cuda-runtime-lock-v3"
            and runtime_lock.get("status") == "pinned_for_acquisition"
        ),
        "host_attestation_bytes_and_schema_exact": (
            _sha256(attestation_bytes) == HOST_ATTESTATION_SHA256
            and len(attestation_bytes) == 4124
            and attestation_record.get("path") == HOST_ATTESTATION_PATH
            and attestation_record.get("sha256") == HOST_ATTESTATION_SHA256
            and attestation_record.get("byte_count") == 4124
            and attestation_record.get("schema_version") == attestation.get("schema_version")
            and attestation_record.get("status") == attestation.get("status")
            and attestation.get("schema_version") == "passive-muon-p23-host-attestation-v3"
            and attestation.get("status") == "procedurally_host_attested"
        ),
        "runtime_image_and_gpu_exact": (
            runtime.get("image")
            == "localhost:5000/p25-runtime@sha256:"
            "e1a636b8c113aaa2823bd8e03bc0dee626d91cdc6897154e05a2bda54b178ee6"
            and runtime.get("gpu")
            == {
                "uuid": "GPU-ee4c9bf9-42f5-7cdd-66d2-f638c2db549d",
                "name": "NVIDIA A100-SXM4-40GB",
                "compute_capability": [8, 0],
                "mig_mode": "disabled",
            }
            and runtime_lock.get("gpu") == attestation.get("gpu")
        ),
        "diagnostic_bridge_direct_child_and_delta_exact": (
            _tree(BRIDGE_COMMIT) == BRIDGE_TREE
            and _parent(BRIDGE_COMMIT) == RUNTIME_COMMIT
            and _delta(RUNTIME_COMMIT, BRIDGE_COMMIT) == EXPECTED_BRIDGE_DELTA
            and bridge.get("commit") == BRIDGE_COMMIT
            and bridge.get("tree") == BRIDGE_TREE
            and bridge.get("parent") == RUNTIME_COMMIT
            and bridge.get("exact_delta") == EXPECTED_BRIDGE_DELTA
        ),
        "committed_sanitized_bytes_exact": (
            _sha256(sanitized_bytes) == SANITIZED_SHA256
            and len(sanitized_bytes) == 5_097_711
            and sanitized_record.get("path") == SANITIZED_PATH
            and sanitized_record.get("sha256") == SANITIZED_SHA256
            and sanitized_record.get("byte_count") == 5_097_711
            and sanitized_record.get("retention") == "committed_full_sanitized_manifest"
            and sanitized_record.get("schema_version") == sanitized.get("schema_version")
        ),
        "sanitized_diagnostic_pass_reconstructs": (
            sanitized.get("schema_version")
            == "passive-muon-p25-sanitized-executable-origin-diagnostic-v1"
            and sanitized.get("mode") == "remediated"
            and sanitized.get("status") == "passes"
            and sanitized.get("passes") is True
            and sanitized_manifest.get("status") == "passes"
            and sanitized_manifest.get("passes") is True
            and len(sanitized_checks) == 40
            and all(value is True for value in sanitized_checks.values())
            and diagnostic_checks == {"total": 40, "true": 40, "false": 0}
            and diagnostic.get("attempts_allowed") == 1
            and diagnostic.get("attempts_run") == 1
            and diagnostic.get("first_and_only_attempt") is True
            and diagnostic.get("exit_code") == 0
        ),
        "native_binding_and_redaction_exact": (
            sanitized_native == {"byte_count": 4_113_445, "sha256": NATIVE_SHA256}
            and native_record.get("retention") == "external_native_bytes_not_committed"
            and native_record.get("file_name") == "p25-remediated-executable-origin.native.json"
            and native_record.get("sha256") == NATIVE_SHA256
            and native_record.get("byte_count") == 4_113_445
            and sanitized_redaction
            == {
                "key_replacement_count": 2,
                "value_replacement_count": 37_915,
                "total_replacement_count": 37_917,
            }
            and redaction_record == sanitized_redaction
        ),
        "diagnostic_repository_and_generated_origin_exact": (
            diagnostic.get("repository_commit") == RUNTIME_COMMIT
            and diagnostic.get("repository_tree") == RUNTIME_TREE
            and sanitized_repository.get("expected_head") == RUNTIME_COMMIT
            and sanitized_repository.get("expected_tree") == RUNTIME_TREE
            and observed_repository.get("head") == RUNTIME_COMMIT
            and observed_repository.get("tree") == RUNTIME_TREE
            and observed_repository.get("dirty") is False
            and generated_record
            == {
                "sha256": GENERATED_SHA256,
                "byte_count": 2355,
                "effective_mount_writable": False,
            }
            and sanitized_generated.get("sha256") == GENERATED_SHA256
            and sanitized_generated.get("byte_count") == 2355
            and _mapping(sanitized_generated.get("effective_mount")).get("writable") is False
        ),
        "operator_recorded_permission_stop_exact": (
            stop.get("attempts_run") == 1
            and stop.get("status") == "blocked_before_trace_off_a"
            and stop.get("host_orchestrator_exit_code") == 1
            and stop.get("failed_preflight")
            == "reviewed_and_retained_diagnostic_wrapper_byte_comparison"
            and comparison_codes == {"unprivileged_cmp": 2, "privileged_cmp": 0}
            and stop_stream
            == {
                "text": STOP_STDERR,
                "sha256": _sha256(STOP_STDERR.encode()),
                "byte_count": len(STOP_STDERR.encode()),
            }
            and root_cause.get("classification")
            == "host_unprivileged_cmp_cannot_read_root_owned_retained_wrapper"
            and retained_wrapper
            == {
                "owner": "root:root",
                "mode": "0600",
                "sha256": SANITIZED_SHA256,
                "byte_count": 5_097_711,
            }
            and repository_wrapper == {"sha256": SANITIZED_SHA256, "byte_count": 5_097_711}
            and root_cause.get("content_mismatch_observed") is False
        ),
        "operator_recorded_zero_acquisition_inventory_exact": (
            stop.get("absent_pre_trace_logs") == EXPECTED_ABSENT_PRE_TRACE_LOGS
            and zero_inventory.get("native_expected_names") == EXPECTED_NATIVE_ACQUISITION_NAMES
            and zero_inventory.get("native_present_names") == []
            and zero_inventory.get("sanitized_present_names") == []
            and zero_inventory.get("acquisition_process_stream_labels_expected")
            == EXPECTED_ACQUISITION_PROCESS_LABELS
            and zero_inventory.get("acquisition_process_stream_labels_present") == []
            and zero_inventory.get("p23_sanitizer_processes_run") == 0
        ),
        "source_orders_permission_stop_before_trace_off_a": (
            'cmp -s "$repository_wrapper"' in orchestrator_source
            and 'die "reviewed and retained diagnostic wrappers differ"' in orchestrator_source
            and orchestrator_source.index('cmp -s "$repository_wrapper"')
            < orchestrator_source.index("p23-trace-off-a")
        ),
        "no_acquisition_artifact_is_committed_at_bridge": not any(
            Path(path).name in forbidden_acquisition_names for path in tracked_at_bridge
        ),
        "all_later_gates_are_absent": _exact_value(dict(gate_status), EXPECTED_GATE_STATUS),
        "terminal_no_retry_policy_exact": terminal
        == {
            "attempt_is_terminal": True,
            "retry_until_favorable_allowed": False,
            "reuse_attempt_20260906_01_allowed": False,
            "p24_disposition_changed": False,
            "frozen_p23_thresholds_changed": False,
        },
        "theorem_and_route_scope_exact": (
            theorem
            == {
                "p18_through_p21_results_affected": False,
                "cuda_repeatability_established": False,
                "observer_noninterference_established": False,
                "real_gradient_fidelity_established": False,
                "native_cuda_shield_certified": False,
            }
            and route.get("next_branch") is None
            and route.get("new_attempt_required") is True
            and route.get("threshold_tuning_allowed") is False
            and isinstance(route.get("required_action"), str)
            and "without rerunning attempt 20260906-01" in str(route.get("required_action"))
        ),
    }
    return {
        "schema_version": "passive-muon-p25-cuda-diagnostic-outcome-reconstruction-v1",
        "canonical_sha256": _sha256(canonical_bytes),
        "checks": checks,
        "internally_consistent": all(checks.values()),
        "scope": (
            "Repository reconstruction of P25's committed diagnostic and internal consistency "
            "of an operator-recorded pre-trace permission stop; not independent authentication "
            "of the external native bytes, filesystem metadata, client stderr, or remote absence."
        ),
    }


def main() -> int:
    result = reconstruct(_parse_args().canonical.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["internally_consistent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
