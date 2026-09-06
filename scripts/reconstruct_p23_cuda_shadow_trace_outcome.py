#!/usr/bin/env python3
"""Check the static and internal consistency of the operator-recorded P23 stop."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANONICAL = ROOT / "results/summaries/p23_cuda_shadow_trace_outcome.json"
ATTEMPT_COMMIT = "f89ea1fef3fc2700fed0d05dcebf51254c8a0db9"
ATTEMPT_TREE = "778661ea834294a4056ec5f36d0f1e2d5fcf9840"
EXPECTED_SCHEMA = "passive-muon-p23-cuda-shadow-trace-outcome-v1"
EXPECTED_STATUS = "blocked_trace_off_a_provenance"
EXPECTED_EVIDENCE_KIND = "operator_recorded_cuda_shadow_trace_pretraining_blocker"
EXPECTED_CLAIM_BOUNDARY = (
    "The acquisition fields are an operator-retained record corroborated only in part by "
    "Docker event metadata; the failed runner emitted no native failure manifest. This record "
    "contains no optimizer step, gradient, candidate observation, repeatability result, "
    "observer-noninterference result, fidelity statistic, or training-quality evidence."
)
EXPECTED_RUNTIME_FILES = {
    "runtime_lock_sha256": "experiments/training/p23_cuda_runtime_lock.json",
    "host_attestation_sha256": "experiments/training/p23_host_attestation.json",
    "p23_addendum_sha256": (
        "experiments/training/p23_deterministic_cuda_shadow_trace_addendum.json"
    ),
    "runner_sha256": "experiments/training/run_p23_deterministic_cuda_shadow_trace.py",
    "runtime_recipe_sha256": "experiments/training/p23_runtime.Dockerfile",
}
EXPECTED_TOP_LEVEL_KEYS = {
    "schema_version",
    "status",
    "evidence_kind",
    "claim_boundary",
    "attempt",
    "locked_runtime",
    "native_evidence_inventory_after_failure",
    "gate_status",
    "route",
}
EXPECTED_ATTEMPT = {
    "role": "trace_off_a",
    "started_from_clean_repository": True,
    "repository_head": ATTEMPT_COMMIT,
    "repository_tree": ATTEMPT_TREE,
    "repository_porcelain_v1_sha256": hashlib.sha256(b"").hexdigest(),
    "docker_exec_id": "e4dbbf141b6fa9c63a0334f0c6b72a932a8cf8a893e766077b841cb71a8a7728",
    "docker_exec_exit_code": 2,
    "docker_exec_died_unix_seconds": 1788683175,
    "docker_exec_died_utc": "2026-09-06T08:26:15Z",
    "stdout": "number of parameters: 123.55M\n",
    "stderr": (
        "P23 blocked: executable origin lies on a writable mount: "
        "tmpfs:tmp1s2eyibl/_remote_module_non_scriptable.py\n"
    ),
    "transcript_retention": (
        "The operator client retained these exact streams and Docker retained the exec ID, "
        "exit code, and death time. The failed runner did not emit a native failure manifest, "
        "so the transcript was not hash-bound at execution time."
    ),
    "evidence_status": "operator_recorded_with_docker_event_metadata_not_native_failure_manifest",
    "failure_phase": "initialized_loaded_file_closure_before_initial_state_hash_and_step_zero",
    "model_allocated": True,
    "optimizer_constructed": True,
    "forward_backward_started": False,
    "optimizer_steps_completed": 0,
    "candidate_observations": 0,
    "native_trace_off_a_manifest_created": False,
}
EXPECTED_INVENTORY = {
    "present": ["p23_cuda_runtime_lock.json", "p23_host_attestation.json"],
    "absent": [
        "trace-off-a.json",
        "trace-off-b.json",
        "repeatability.json",
        "trace-on.json",
        "raw-trace.json",
        "noninterference.json",
        "aggregate.json",
    ],
}
EXPECTED_GATE_STATUS = {
    "trace_off_a": "blocked_before_step_zero",
    "trace_off_b": "not_run",
    "repeatability": "not_run",
    "trace_on": "mechanically_barred",
    "noninterference": "not_run",
    "aggregate_fidelity": "not_run",
}
EXPECTED_ROUTE = {
    "next_branch": "p24-cuda-executable-origin-hardening",
    "reason": (
        "The failure precedes every outcome in the preregistered four-way post-run table. P24 "
        "must reproduce and localize the implicated initialization path with native, hash-bound "
        "provenance, apply an image-resident remedy without weakening P23's writable-origin "
        "rejection, then freeze a new image and runtime before any new acquisition."
    ),
    "thresholds_may_change": False,
    "p23_may_be_rerun_under_existing_lock": False,
}
EXPECTED_RUNTIME_LOCK_KEYS = {
    "schema_version",
    "status",
    "p23_addendum_sha256",
    "container",
    "gpu",
    "software",
    "determinism",
    "loader_environment",
}
EXPECTED_HOST_ATTESTATION_KEYS = {
    "schema_version",
    "status",
    "container",
    "gpu",
    "evidence",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    return parser.parse_args()


@cache
def _git_output(*arguments: str) -> bytes:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


def _commit_blob(path: str) -> bytes:
    return _git_output("show", f"{ATTEMPT_COMMIT}:{path}")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _exact_keys(value: object, expected: set[str]) -> bool:
    return isinstance(value, Mapping) and set(value) == expected


def _exact_value(observed: object, expected: object) -> bool:
    """Compare JSON values recursively without Python's bool/int aliasing."""

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


def _utc_timestamp(value: object) -> str | None:
    if not isinstance(value, int) or isinstance(value, bool):
        return None
    try:
        return datetime.fromtimestamp(value, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    except (OverflowError, OSError, ValueError):
        return None


def reconstruct(canonical: Path) -> dict[str, object]:
    """Reconstruct static bindings without authenticating the remote operator report."""

    raw_outcome = json.loads(canonical.read_text(encoding="utf-8"))
    outcome = _mapping(raw_outcome)
    attempt = _mapping(outcome.get("attempt"))
    runtime = _mapping(outcome.get("locked_runtime"))
    inventory = _mapping(outcome.get("native_evidence_inventory_after_failure"))
    gate_status = _mapping(outcome.get("gate_status"))
    route = _mapping(outcome.get("route"))

    commit_tree = _git_output("rev-parse", f"{ATTEMPT_COMMIT}^{{tree}}").decode().strip()
    committed_bytes = {field: _commit_blob(path) for field, path in EXPECTED_RUNTIME_FILES.items()}
    file_hashes = {field: _sha256(value) for field, value in committed_bytes.items()}
    runtime_lock = _mapping(json.loads(committed_bytes["runtime_lock_sha256"]))
    host_attestation = _mapping(json.loads(committed_bytes["host_attestation_sha256"]))
    lock_container = _mapping(runtime_lock.get("container"))
    lock_gpu = _mapping(runtime_lock.get("gpu"))
    attested_container = _mapping(host_attestation.get("container"))
    attested_gpu = _mapping(host_attestation.get("gpu"))

    expected_runtime = {
        **file_hashes,
        "container_id": lock_container.get("container_id"),
        "container_init_pid": lock_container.get("container_init_pid"),
        "image": lock_container.get("image"),
        "image_id": lock_container.get("image_id"),
        "gpu_uuid": lock_gpu.get("uuid"),
        "gpu_name": lock_gpu.get("name"),
        "mountinfo_sha256": lock_container.get("mountinfo_sha256"),
        "live_identity_rechecked_before_attempt": True,
    }
    lock_container_without_backlink = {
        key: value for key, value in lock_container.items() if key != "host_attestation_sha256"
    }

    checks = {
        "outcome_object_and_exact_schema": _exact_keys(raw_outcome, EXPECTED_TOP_LEVEL_KEYS),
        "schema_status_and_operator_record_scope_exact": (
            outcome.get("schema_version") == EXPECTED_SCHEMA
            and outcome.get("status") == EXPECTED_STATUS
            and outcome.get("evidence_kind") == EXPECTED_EVIDENCE_KIND
            and outcome.get("claim_boundary") == EXPECTED_CLAIM_BOUNDARY
        ),
        "attempt_record_exact": _exact_value(dict(attempt), EXPECTED_ATTEMPT),
        "attempt_commit_and_tree_exist": (
            attempt.get("repository_head") == ATTEMPT_COMMIT
            and attempt.get("repository_tree") == ATTEMPT_TREE
            and commit_tree == ATTEMPT_TREE
        ),
        "docker_event_time_is_internally_consistent": (
            _utc_timestamp(attempt.get("docker_exec_died_unix_seconds"))
            == attempt.get("docker_exec_died_utc")
        ),
        "transcribed_failure_scope_is_internally_consistent": (
            attempt.get("evidence_status")
            == "operator_recorded_with_docker_event_metadata_not_native_failure_manifest"
            and isinstance(attempt.get("transcript_retention"), str)
            and "not hash-bound at execution time" in str(attempt.get("transcript_retention"))
            and attempt.get("forward_backward_started") is False
            and attempt.get("optimizer_steps_completed") == 0
            and attempt.get("candidate_observations") == 0
            and attempt.get("native_trace_off_a_manifest_created") is False
        ),
        "attempt_commit_file_hashes_exact": all(
            runtime.get(field) == digest for field, digest in file_hashes.items()
        ),
        "runtime_record_exactly_matches_attempt_lock": _exact_value(
            dict(runtime), expected_runtime
        ),
        "runtime_lock_static_schema_and_backlinks_exact": (
            _exact_keys(runtime_lock, EXPECTED_RUNTIME_LOCK_KEYS)
            and runtime_lock.get("schema_version") == "passive-muon-p23-cuda-runtime-lock-v3"
            and runtime_lock.get("status") == "pinned_for_acquisition"
            and runtime_lock.get("p23_addendum_sha256") == file_hashes["p23_addendum_sha256"]
            and lock_container.get("host_attestation_sha256")
            == file_hashes["host_attestation_sha256"]
        ),
        "host_attestation_static_schema_exact": (
            _exact_keys(host_attestation, EXPECTED_HOST_ATTESTATION_KEYS)
            and host_attestation.get("schema_version") == "passive-muon-p23-host-attestation-v3"
            and host_attestation.get("status") == "procedurally_host_attested"
        ),
        "runtime_lock_and_host_attestation_maps_match": (
            _exact_value(lock_container_without_backlink, dict(attested_container))
            and _exact_value(dict(lock_gpu), dict(attested_gpu))
        ),
        "operator_recorded_inventory_exact": _exact_value(dict(inventory), EXPECTED_INVENTORY),
        "operator_recorded_gate_status_exact": _exact_value(
            dict(gate_status), EXPECTED_GATE_STATUS
        ),
        "operator_recorded_route_exact": _exact_value(dict(route), EXPECTED_ROUTE),
    }
    internally_consistent = all(checks.values())
    return {
        "schema_version": "passive-muon-p23-cuda-shadow-trace-outcome-reconstruction-v1",
        "canonical_sha256": _sha256(canonical.read_bytes()),
        "attempt_commit": ATTEMPT_COMMIT,
        "attempt_tree": commit_tree,
        "commit_file_sha256": file_hashes,
        "checks": checks,
        "internally_consistent": internally_consistent,
        "scope": (
            "static and internal consistency of an operator-recorded P23 early-stop record; "
            "not authentication or independent reproduction of the remote CUDA event"
        ),
    }


def main() -> int:
    args = _parse_args()
    result = reconstruct(args.canonical.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["internally_consistent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
