#!/usr/bin/env python3
"""Reconstruct the static and internal P24 terminal-outcome bindings."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Mapping
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANONICAL = ROOT / "results/summaries/p24_cuda_executable_origin_outcome.json"
STATIC_COMMIT = "9003130ca31085f555ff144216a599442cbdf3ca"
STATIC_TREE = "58379372d752b2c17b6db7e1f690f899c2051545"
CONTRACT_PATH = "experiments/training/p24_cuda_executable_origin_contract.json"
RUNTIME_LOCK_PATH = "experiments/training/p23_cuda_runtime_lock.json"
HOST_ATTESTATION_PATH = "experiments/training/p23_host_attestation.json"
SANITIZER_PATH = "scripts/sanitize_p24_executable_origin_diagnostic.py"
EXPECTED_CANONICAL_SHA256 = "1b1a75d823b45d3c04daa024839dfbbe9461de9fac38117b0602ea7fbbcf941f"
EXPECTED_CONTRACT_SHA256 = "bdb2d3aad7d70d0ed5c6dddcd03642c9384d8b67d6f5ca507987d28e122dd446"
EXPECTED_NATIVE_SHA256 = "ce44c53c9f274cef3eda7b3adea755ce6b1313773c8dfe00fc9f194b74fb2bd2"
EXPECTED_GENERATED_SHA256 = "8205b16956fb264841ecd8644784a0d157f87df79b17c16825dc1163433ce5d8"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
SANITIZER_ERROR = (
    "P24 diagnostic sanitization blocked: redacted diagnostic retains a declared native path\n"
)
EXPECTED_TOP_LEVEL_KEYS = {
    "schema_version",
    "status",
    "evidence_kind",
    "claim_boundary",
    "static_contract",
    "baseline_diagnostic",
    "sanitization",
    "gate_status",
    "theorem_status",
    "route",
}
EXPECTED_STATIC_KEYS = {
    "repository_head",
    "repository_tree",
    "contract_path",
    "contract_schema_version",
    "contract_status",
    "contract_sha256",
    "contract_or_execution_source_modified_by_outcome",
}
EXPECTED_GATE_STATUS = {
    "baseline_diagnostic": "failed_exact_torch_determinism_match",
    "baseline_sanitization": "failed_closed_no_sanitized_output",
    "remediation_image_build": "not_run",
    "replacement_container": "not_run",
    "replacement_runtime_freeze": "not_run",
    "remediated_diagnostic": "not_run",
    "trace_off_a": "not_run",
    "trace_off_b": "not_run",
    "exact_repeatability": "not_run",
    "trace_on": "not_run",
    "observer_noninterference": "not_run",
    "candidate_observations": 0,
    "fidelity_aggregate": "not_run",
    "training": "not_run",
}
EXPECTED_LOCK_DETERMINISM = {
    "deterministic_algorithms": True,
    "deterministic_debug_mode": "error",
    "interop_threads": 1,
    "cudnn_deterministic": True,
    "fp16_reduced_precision_reduction": False,
    "bf16_reduced_precision_reduction": False,
}
EXPECTED_LIVE_DETERMINISM = {
    "deterministic_algorithms": False,
    "deterministic_debug_mode": 0,
    "interop_threads": 30,
    "cudnn_deterministic": False,
    "fp16_reduced_precision_reduction": True,
    "bf16_reduced_precision_reduction": True,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    return parser.parse_args()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@cache
def _git_output(*arguments: str) -> bytes:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


def _commit_blob(path: str) -> bytes:
    return _git_output("show", f"{STATIC_COMMIT}:{path}")


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _exact_keys(value: object, expected: set[str]) -> bool:
    return isinstance(value, Mapping) and set(value) == expected


def _exact_value(observed: object, expected: object) -> bool:
    """Compare recursively without accepting Python's bool/int aliasing."""

    if type(observed) is not type(expected):
        return False
    if isinstance(expected, dict):
        assert isinstance(observed, dict)
        return set(observed) == set(expected) and all(
            _exact_value(observed[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        assert isinstance(observed, list)
        return len(observed) == len(expected) and all(
            _exact_value(left, right) for left, right in zip(observed, expected, strict=True)
        )
    return observed == expected


def _absolute_key_paths(value: object, prefix: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    paths: list[tuple[str, ...]] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str) and key.startswith("/"):
                paths.append((*prefix, key))
            paths.extend(_absolute_key_paths(item, (*prefix, str(key))))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            paths.extend(_absolute_key_paths(item, (*prefix, str(index))))
    return paths


def reconstruct(canonical: Path) -> dict[str, object]:
    """Check repository-reconstructible facts without authenticating external bytes."""

    canonical_bytes = canonical.read_bytes()
    raw_outcome = json.loads(canonical_bytes)
    outcome = _mapping(raw_outcome)
    static = _mapping(outcome.get("static_contract"))
    baseline = _mapping(outcome.get("baseline_diagnostic"))
    native = _mapping(baseline.get("native_artifact"))
    diagnostic_checks = _mapping(baseline.get("checks"))
    mismatches = _mapping(baseline.get("torch_determinism_mismatches"))
    generated = _mapping(baseline.get("executable_origin_observation"))
    streams = _mapping(baseline.get("client_streams"))
    sanitization = _mapping(outcome.get("sanitization"))
    sanitizer_stderr = _mapping(sanitization.get("stderr"))
    sanitizer_stdout = _mapping(sanitization.get("stdout"))
    source_cause = _mapping(sanitization.get("source_level_root_cause"))
    gate_status = _mapping(outcome.get("gate_status"))
    theorem_status = _mapping(outcome.get("theorem_status"))
    route = _mapping(outcome.get("route"))

    commit_tree = _git_output("rev-parse", f"{STATIC_COMMIT}^{{tree}}").decode().strip()
    contract_bytes = _commit_blob(CONTRACT_PATH)
    contract = _mapping(json.loads(contract_bytes))
    runtime_lock = _mapping(json.loads(_commit_blob(RUNTIME_LOCK_PATH)))
    host_attestation = _mapping(json.loads(_commit_blob(HOST_ATTESTATION_PATH)))
    sanitizer_source = _commit_blob(SANITIZER_PATH).decode("utf-8")

    execution_sources = _mapping(contract.get("execution_sources"))
    source_hashes_match = bool(execution_sources) and all(
        isinstance(record, Mapping)
        and isinstance(record.get("path"), str)
        and record.get("sha256") == _sha256(_commit_blob(str(record["path"])))
        for record in execution_sources.values()
    )
    remediation = _mapping(contract.get("remediation_design"))
    remediation_hashes = _mapping(remediation.get("source_hashes"))
    remediation_counts = _mapping(remediation.get("source_byte_counts"))
    lock_determinism = _mapping(runtime_lock.get("determinism"))
    mismatch_shape = {
        key: {
            "observed_live": EXPECTED_LIVE_DETERMINISM[key],
            "committed_lock": EXPECTED_LOCK_DETERMINISM[key],
        }
        for key in EXPECTED_LOCK_DETERMINISM
    }
    absolute_runtime_keys = _absolute_key_paths(runtime_lock)
    absolute_attestation_keys = _absolute_key_paths(host_attestation)

    checks = {
        "canonical_bytes_exact": _sha256(canonical_bytes) == EXPECTED_CANONICAL_SHA256,
        "outcome_object_and_top_level_schema_exact": _exact_keys(
            raw_outcome, EXPECTED_TOP_LEVEL_KEYS
        ),
        "outcome_status_and_claim_boundary_exact": (
            outcome.get("schema_version") == "passive-muon-p24-cuda-executable-origin-outcome-v1"
            and outcome.get("status") == "blocked_baseline_diagnostic_determinism_and_redaction"
            and outcome.get("evidence_kind")
            == "operator_recorded_compact_outcome_bound_to_external_native_bytes"
            and isinstance(outcome.get("claim_boundary"), str)
            and "native bytes remain external" in str(outcome.get("claim_boundary"))
            and "No image build" in str(outcome.get("claim_boundary"))
        ),
        "static_record_exact": (
            _exact_keys(static, EXPECTED_STATIC_KEYS)
            and static.get("repository_head") == STATIC_COMMIT
            and static.get("repository_tree") == STATIC_TREE
            and static.get("contract_path") == CONTRACT_PATH
            and static.get("contract_schema_version")
            == "passive-muon-p24-cuda-executable-origin-contract-v1"
            and static.get("contract_status") == "frozen_pre_remediation_build_and_pre_acquisition"
            and static.get("contract_sha256") == EXPECTED_CONTRACT_SHA256
            and static.get("contract_or_execution_source_modified_by_outcome") is False
        ),
        "static_commit_tree_and_contract_bytes_exact": (
            commit_tree == STATIC_TREE
            and _sha256(contract_bytes) == EXPECTED_CONTRACT_SHA256
            and contract.get("schema_version") == static.get("contract_schema_version")
            and contract.get("status") == static.get("contract_status")
        ),
        "frozen_execution_source_hashes_reconstruct": source_hashes_match,
        "native_external_binding_exact": (
            baseline.get("mode") == "baseline"
            and baseline.get("schema_version") == "passive-muon-p24-executable-origin-diagnostic-v1"
            and baseline.get("status") == "fails"
            and baseline.get("passes") is False
            and baseline.get("exit_code") == 1
            and native.get("retention") == "external_native_bytes_not_committed"
            and native.get("file_name") == "p24-baseline-executable-origin.native.json"
            and native.get("sha256") == EXPECTED_NATIVE_SHA256
            and native.get("byte_count") == 3_760_729
        ),
        "diagnostic_check_cardinality_and_failure_exact": (
            _exact_value(
                dict(diagnostic_checks),
                {
                    "total": 36,
                    "true": 35,
                    "false": 1,
                    "sole_false_check": "torch_determinism_matches_lock",
                },
            )
        ),
        "lock_determinism_values_reconstruct": all(
            _exact_value(lock_determinism.get(key), value)
            for key, value in EXPECTED_LOCK_DETERMINISM.items()
        ),
        "operator_recorded_determinism_mismatches_exact": _exact_value(
            dict(mismatches), mismatch_shape
        ),
        "generated_module_matches_frozen_remediation_constants": (
            generated.get("p23_hypothesis_reproduced") is True
            and generated.get("generated_module_sha256") == EXPECTED_GENERATED_SHA256
            and generated.get("generated_module_byte_count") == 2355
            and generated.get("effective_mount_point") == "/tmp"
            and generated.get("effective_filesystem_type") == "tmpfs"
            and generated.get("effective_mount_writable") is True
            and remediation_hashes.get("generated_module_sha256") == EXPECTED_GENERATED_SHA256
            and remediation_counts.get("generated_module") == 2355
            and isinstance(generated.get("qualification"), str)
            and "does not exercise" in str(generated.get("qualification"))
        ),
        "diagnostic_client_stream_bindings_exact": (
            _exact_value(
                dict(_mapping(streams.get("stdout"))),
                {
                    "sha256": "b48a4747da0280e8ff2d6010b481a4b92045548eeb0a4e4408221231ff8478da",
                    "byte_count": 191,
                },
            )
            and _exact_value(
                dict(_mapping(streams.get("stderr"))),
                {"sha256": EMPTY_SHA256, "byte_count": 0},
            )
        ),
        "sanitizer_streams_and_absence_exact": (
            sanitization.get("status") == "blocked_no_output"
            and sanitization.get("exit_code") == 2
            and sanitization.get("sanitized_artifact_created") is False
            and _exact_value(dict(sanitizer_stdout), {"sha256": EMPTY_SHA256, "byte_count": 0})
            and sanitizer_stderr.get("text") == SANITIZER_ERROR
            and sanitizer_stderr.get("sha256") == _sha256(SANITIZER_ERROR.encode("utf-8"))
            and sanitizer_stderr.get("byte_count") == len(SANITIZER_ERROR.encode("utf-8"))
        ),
        "sanitizer_key_redaction_bug_reconstructs_from_frozen_sources": (
            source_cause.get("classification") == "absolute_mapping_keys_not_transformed"
            and isinstance(source_cause.get("detail"), str)
            and source_cause.get("native_payload_required_to_reconstruct_source_bug") is False
            and "return {str(key): sanitize(item) for key, item in value.items()}"
            in sanitizer_source
            and "if str(root) in rendered:" in sanitizer_source
            and any(path[-1] == "/tmp" for path in absolute_runtime_keys)
            and any(path[-1] == "/tmp" for path in absolute_attestation_keys)
        ),
        "all_later_gates_are_legally_absent": _exact_value(dict(gate_status), EXPECTED_GATE_STATUS),
        "historical_theorem_scope_exact": _exact_value(
            dict(theorem_status),
            {
                "p18_through_p21_results_affected": False,
                "cuda_repeatability_established": False,
                "real_gradient_fidelity_established": False,
                "native_cuda_shield_certified": False,
            },
        ),
        "route_is_exact_and_does_not_weaken_frozen_gates": (
            route.get("next_branch") == "p25-cuda-diagnostic-determinism-and-redaction"
            and isinstance(route.get("reason"), str)
            and "before any image build or CUDA gradient acquisition" in str(route.get("reason"))
            and route.get("frozen_p23_fidelity_thresholds_may_change") is False
            and route.get("p24_may_be_rerun_until_favorable") is False
        ),
    }
    internally_consistent = all(checks.values())
    return {
        "schema_version": "passive-muon-p24-cuda-executable-origin-outcome-reconstruction-v1",
        "canonical_sha256": _sha256(canonical_bytes),
        "static_commit": STATIC_COMMIT,
        "static_tree": commit_tree,
        "contract_sha256": _sha256(contract_bytes),
        "checks": checks,
        "internally_consistent": internally_consistent,
        "scope": (
            "repository-reconstructible static bindings and internal consistency of a compact "
            "operator-recorded P24 stop; not authentication or reconstruction of the retained "
            "external native diagnostic"
        ),
    }


def main() -> int:
    args = _parse_args()
    result = reconstruct(args.canonical.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["internally_consistent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
