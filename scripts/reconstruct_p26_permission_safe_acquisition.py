#!/usr/bin/env python3
"""Independently reconstruct the post-runtime, pre-diagnostic P26 contract."""

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
DEFAULT_CANONICAL = ROOT / "experiments/training/p26_permission_safe_acquisition_contract.json"
EXPECTED_SCHEMA = "passive-muon-p26-permission-safe-acquisition-contract-v2"
EXPECTED_STATUS = "frozen_post_fresh_runtime_pre_diagnostic_and_pre_acquisition"
EXPECTED_BRANCH = "p26-permission-safe-acquisition-bridge"
P25_OUTCOME_PATH = "results/summaries/p25_cuda_diagnostic_outcome.json"
P25_OUTCOME_SHA256 = "d7a06f6f25bc4839db256720fe5c8ef3c99f6ea45b53b48da2135fcde76a423a"
P25_OUTCOME_COMMIT = "f055405cc879ba0ac5afe26bc34a7336d5d2efbf"
P25_OUTCOME_TREE = "a5ee62de68b091719938829e579853b226b7f086"
P25_PREREG_COMMIT = "e76ab62f92c95e6f0716cf2f1ed38a583cadfe56"
P25_PREREG_TREE = "4fe0f57d50a8fa136bb192ec3fbac95b1e746aa2"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

EXPECTED_TOP_LEVEL_ORDER = [
    "schema_version",
    "status",
    "branch",
    "purpose",
    "claim_boundary",
    "control_history",
    "terminal_parent",
    "fresh_attempt",
    "prepared_runtime",
    "frozen_p25_control",
    "p26_control_sources",
    "permission_safe_bridge",
    "unchanged_scientific_authorities",
    "runtime_contract",
    "acquisition_order",
    "terminal_routing",
    "forbidden_changes",
]
EXPECTED_PURPOSE = (
    "Repair only P25's host read-permission boundary by authenticating both sanitized "
    "wrapper byte strings inside the logged root container verifier before the unchanged "
    "P23 acquisition."
)
EXPECTED_CLAIM_BOUNDARY = (
    "This post-runtime control contract records only one prepared and reviewed P26 runtime. "
    "It records no corrected diagnostic, permission-safe bridge pass, CUDA gradient, exact "
    "repeatability, observer noninterference, candidate fidelity, shield execution, or "
    "training result."
)
EXPECTED_CONTROL_HISTORY = {
    "pre_runtime_source_freeze_commit": "bc2c84880a7e6f3e762d7a05dfbd5048772b2c69",
    "pre_runtime_source_freeze_tree": "7332be7fdd15a1dff2374c10ae3c4555b7f60fec",
    "terminal_parent": P25_OUTCOME_COMMIT,
    "order_discrepancy_discovered_after_runtime": True,
    "corrected_before_diagnostic": True,
    "scientific_gates_changed": False,
}
EXPECTED_TERMINAL_PARENT = {
    "branch": "p25-cuda-diagnostic-determinism-and-redaction",
    "outcome_path": P25_OUTCOME_PATH,
    "outcome_sha256": P25_OUTCOME_SHA256,
    "outcome_commit": P25_OUTCOME_COMMIT,
    "outcome_tree": P25_OUTCOME_TREE,
    "attempt_id": "20260906-01",
    "attempt_is_terminal": True,
    "attempt_reuse_allowed": False,
    "diagnostic_passed_checks": 40,
    "diagnostic_failed_checks": 0,
    "unprivileged_cmp_exit": 2,
    "privileged_cmp_exit": 0,
    "trace_off_a_started": False,
    "candidate_observations": 0,
}
EXPECTED_FRESH_ATTEMPT = {
    "attempt_id": "20260906-02",
    "attempt_root": "/secure/p25/attempt-20260906-02",
    "container_name": "p25-acquisition-20260906-02",
    "root_must_not_preexist": True,
    "new_no_cache_build_required": True,
    "new_container_required": True,
    "new_runtime_lock_required": True,
    "new_host_attestation_required": True,
    "new_corrected_diagnostic_required": True,
    "new_sanitized_bridge_required": True,
    "no_favorable_rerun": True,
}
EXPECTED_PREPARED_RUNTIME = {
    "attempt_id": "20260906-02",
    "prepared": True,
    "reviewed": True,
    "prepare_runtime_may_repeat": False,
    "runtime_review_commit": "ba93225f3ef1abf5dbde71950c1eaa2e384a5dc0",
    "runtime_review_tree": "e7cdf25c86360ecdd42f2dc188ec416744321fe1",
    "runtime_review_parent": P25_PREREG_COMMIT,
    "image_digest": ("sha256:14dafde07ae578cc4725f452a51d4bc4c920b69f6af23125a72e27162e193ec0"),
    "container_id": "e6682d8f09b9dc4be342354d520a8f6f8766a8e232840d1d7beb5e00e1fced6c",
    "container_init_pid": 39227,
    "build_metadata_sha256": ("13e0f05189321f86b4d8a41253eed97768bb1774ac7db64ad7c8b25e2f026091"),
    "runtime_lock_sha256": ("04be2154daf5afbe97f7d2278ee7936da769d230663dfc6dfd50f0370a456755"),
    "host_attestation_sha256": ("8caa4d761ca89741517e6292e174acd17948f4d312b8418d0056070d3247e046"),
    "diagnostic_started": False,
    "permission_bridge_started": False,
    "trace_off_a_started": False,
    "candidate_observations": 0,
}
EXPECTED_FROZEN_P25_CONTROL = {
    "source_preregistration_commit": P25_PREREG_COMMIT,
    "source_preregistration_tree": P25_PREREG_TREE,
    "contract": {
        "path": "experiments/training/p25_cuda_diagnostic_contract.json",
        "sha256": "51e1fc22f685cc904fcdfff2775cff4293fdaa67a6e048072ee4dc49777f34e2",
    },
    "host_orchestrator": {
        "path": "scripts/run_p25_cuda_attempt.sh",
        "sha256": "fce74a7e65d48cd6391d0986f60ce977cc3b36215908d80349d38f37d95fd441",
        "historically_allowed_phases": ["prepare-runtime", "run-diagnostic"],
        "currently_allowed_phases": ["run-diagnostic"],
        "prepare_runtime_already_executed": True,
        "run_acquisition_allowed": False,
    },
    "diagnostic": {
        "path": "experiments/training/run_p25_executable_origin_diagnostic.py",
        "sha256": "6305fb9683503eb67e091cdfb0a1105628fcb4bec76fe5ef7fd23dd49ba9d770",
    },
    "sanitizer": {
        "path": "scripts/sanitize_p25_executable_origin_diagnostic.py",
        "sha256": "7d28a04ac99f96315d4a1160b741897660bb829617768af3641eb82f890c745e",
    },
}
EXPECTED_P26_SOURCE_PATHS = {
    "host_orchestrator": "scripts/run_p26_permission_safe_acquisition.sh",
    "independent_reconstructor": "scripts/reconstruct_p26_permission_safe_acquisition.py",
    "root_container_verifier": "scripts/verify_p26_permission_safe_bridge.py",
}
EXPECTED_BRIDGE = {
    "execution_context": "root_inside_fresh_attested_network_none_read_only_root_container",
    "transcript_prefix": "p26-permission-safe-bridge-verification",
    "before_trace_off_a": True,
    "repository_wrapper": (
        "/workspace/OptimizationML/results/summaries/"
        "p25_executable_origin_diagnostic.sanitized.json"
    ),
    "retained_wrapper": ("/workspace/evidence/p23/p25-remediated-executable-origin.sanitized.json"),
    "retained_native": ("/workspace/evidence/p23/p25-remediated-executable-origin.native.json"),
    "required_checks": [
        "both wrappers exist and are regular nonsymlink files",
        "each wrapper is read exactly once with stable file identity",
        "repository wrapper mode is 0644",
        "retained wrapper remains root:root mode 0600",
        "captured wrapper byte strings are exactly equal",
        "both captured wrapper byte strings strict-parse independently",
        "retained native remains root:root mode 0600",
        "retained native SHA-256 and byte count match the wrapper",
        "retained native strict semantics reconstruct through the frozen P25 sanitizer",
        "every P25 contract source hash reconstructs",
    ],
    "mutations_allowed": [],
    "host_cmp_used": False,
    "chmod_allowed": False,
    "chown_allowed": False,
}
EXPECTED_UNCHANGED_AUTHORITIES = {
    "p21_fidelity_gates": {
        "path": "experiments/training/p21_shadow_trace_protocol.json",
        "sha256": "ca40ef0aef676971ca4464c3f838c6bd956ecc9806c2023af0d4ea699375315a",
    },
    "p22_protocol": {
        "path": "experiments/training/p22_real_gradient_shadow_trace_protocol.json",
        "sha256": "f88eda60b366b561d277e41335b02c3fb9861a2604a947f8dd5c28fb54bedeac",
    },
    "p22_erratum": {
        "path": "theory/p22_real_gradient_shadow_trace_protocol_erratum.md",
        "sha256": "42dc57bd8fb03b619c0de75c58570991ca520131ab7d1660b4f9373fd756a2cd",
    },
    "p23_addendum": {
        "path": "experiments/training/p23_deterministic_cuda_shadow_trace_addendum.json",
        "sha256": "ad29a15e3b027250a35b6c3af6ce214cc98d664cad7d9821730e7e0cdaab9e3a",
    },
    "cuda_acquisition_runner": {
        "path": "experiments/training/run_p23_deterministic_cuda_shadow_trace.py",
        "sha256": "a2b4bb5b686b7f681958d09be1d465917b40e34d45e4d3503efef0f35e7ae8cb",
    },
    "observer": {
        "path": "experiments/training/p22_observed_muon.py",
        "sha256": "5355b553e871406491599945ed790288f59d636a7f9e1d931b9d8234ba42818f",
    },
}
EXPECTED_RUNTIME = {
    "gpu_uuid": "GPU-ee4c9bf9-42f5-7cdd-66d2-f638c2db549d",
    "gpu_name": "NVIDIA A100-SXM4-40GB",
    "exact_bind_mount_count": 10,
    "network_mode": "none",
    "root_filesystem_read_only": True,
    "tmpfs": {"/tmp": "rw,noexec,nosuid,nodev,size=1073741824"},
    "image_digest_must_be_bound_before_diagnostic": True,
    "digest_must_differ_from_prior_attempt": False,
}
EXPECTED_ACQUISITION_ORDER = [
    (
        "validate post-runtime P26 control commit and tree, pre-runtime source-freeze ancestry, "
        "clean state, contract hash, orchestrator hash, reconstructor hash, verifier hash and "
        "independent reconstruction"
    ),
    "validate fresh P25 runtime and diagnostic bridge history",
    "independently reconstruct the frozen P25 contract",
    "validate reviewed BuildKit metadata, image digest, runtime lock and host attestation",
    "validate the live reviewed container ID, PID, restart count and image",
    "reject any preexisting P23 acquisition artifact or process log",
    "run the logged root-container permission-safe bridge verifier",
    "trace_off_a",
    "trace_off_b only after trace_off_a passes",
    "verify_repeatability with exact equality only",
    "trace_on only after zero exact repeatability mismatches",
    "verify_noninterference with exact equality only",
    "aggregate unchanged fidelity gates only after zero noninterference mismatches",
    "sanitize and retain every success artifact or the first failure artifact",
]
EXPECTED_TERMINAL_ROUTING = {
    "bridge_failure": "stop before trace_off_a and retain the first failure transcript",
    "trace_off_a_failure": "stop and retain the native and sanitized failure artifacts",
    "trace_off_b_failure": "stop and retain completed A plus the B failure",
    "repeatability_failure": "bar trace_on and retain exact mismatch evidence",
    "noninterference_failure": "bar aggregation and retain exact mismatch evidence",
    "fidelity_failure": "record the unchanged gate failure; do not tune on this attempt",
    "all_gates_pass": (
        "record the successful real-gradient shadow trace and route to a native CUDA sector shield"
    ),
}
EXPECTED_FORBIDDEN_CHANGES = [
    "reuse or resume P25 attempt 20260906-01",
    "change a P23 or P21 fidelity threshold",
    "replace exact equality with allclose or another tolerance",
    "change seed, data, capture schedule, model or optimizer",
    "chmod or chown retained native or sanitized evidence",
    "run trace_on before exact off-A/off-B repeatability",
    "rerun until favorable",
    "run diagnostic or acquisition from the superseded pre-runtime P26 source-freeze commit",
]
FORBIDDEN_RESULT_KEYS = {
    "result",
    "results",
    "outcome",
    "bridge_transcript",
    "trace_off_a",
    "trace_off_b",
    "repeatability",
    "trace_on",
    "noninterference",
    "aggregate",
    "training_result",
}


class DuplicateKeyError(ValueError):
    """Raised when a purported canonical JSON object repeats a key."""


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


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2) + "\n").encode()


@cache
def _file_bytes(path: object) -> bytes | None:
    if not isinstance(path, str):
        return None
    try:
        return (ROOT / path).read_bytes()
    except OSError:
        return None


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
    if value is None:
        return None
    try:
        return value.decode().strip()
    except UnicodeError:
        return None


def _git_blob(commit: str, path: str) -> bytes | None:
    return _git("show", f"{commit}:{path}")


def _source_record_valid(
    record: object,
    *,
    expected_path: str,
    expected_sha256: str | None = None,
    commit: str | None = None,
) -> bool:
    source = _mapping(record)
    if set(source) != {"path", "sha256"}:
        return False
    path = source.get("path")
    digest = source.get("sha256")
    if path != expected_path or not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
        return False
    if expected_sha256 is not None and digest != expected_sha256:
        return False
    live = _file_bytes(path)
    if live is None or _sha256(live) != digest:
        return False
    if commit is not None:
        blob = _git_blob(commit, expected_path)
        if blob is None or _sha256(blob) != digest:
            return False
    return True


def _all_mapping_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            keys.add(key)
            keys.update(_all_mapping_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.update(_all_mapping_keys(item))
    return keys


def reconstruct(canonical: Path = DEFAULT_CANONICAL) -> dict[str, object]:
    """Reconstruct P26's post-runtime, pre-diagnostic claims from independent constants."""

    try:
        canonical_bytes = canonical.read_bytes()
        raw = _strict_json(canonical_bytes)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError, DuplicateKeyError):
        canonical_bytes = b""
        raw = None
    contract = _mapping(raw)
    control_history = _mapping(contract.get("control_history"))
    terminal_parent = _mapping(contract.get("terminal_parent"))
    fresh_attempt = _mapping(contract.get("fresh_attempt"))
    prepared_runtime = _mapping(contract.get("prepared_runtime"))
    frozen_p25 = _mapping(contract.get("frozen_p25_control"))
    p26_sources = _mapping(contract.get("p26_control_sources"))
    bridge = _mapping(contract.get("permission_safe_bridge"))
    unchanged = _mapping(contract.get("unchanged_scientific_authorities"))
    runtime = _mapping(contract.get("runtime_contract"))

    p25_outcome_blob = _git_blob(P25_OUTCOME_COMMIT, P25_OUTCOME_PATH)
    try:
        p25_outcome_parsed = (
            _strict_json(p25_outcome_blob) if p25_outcome_blob is not None else None
        )
    except (ValueError, json.JSONDecodeError, DuplicateKeyError):
        p25_outcome_parsed = None
    p25_outcome = _mapping(p25_outcome_parsed)
    p25_diagnostic = _mapping(p25_outcome.get("corrected_diagnostic"))
    p25_diagnostic_checks = _mapping(p25_diagnostic.get("checks"))
    p25_stop = _mapping(p25_outcome.get("acquisition_stop"))
    p25_comparisons = _mapping(p25_stop.get("comparison_exit_codes"))
    p25_gates = _mapping(p25_outcome.get("gate_status"))
    p25_terminal = _mapping(p25_outcome.get("terminal_policy"))

    p25_record_checks = []
    for name, expected in EXPECTED_FROZEN_P25_CONTROL.items():
        if name in {"source_preregistration_commit", "source_preregistration_tree"}:
            continue
        expected_mapping = _mapping(expected)
        observed = frozen_p25.get(name)
        if name == "host_orchestrator":
            observed_mapping = _mapping(observed)
            source_part = {key: observed_mapping.get(key) for key in ("path", "sha256")}
            p25_record_checks.append(
                _source_record_valid(
                    source_part,
                    expected_path=str(expected_mapping.get("path")),
                    expected_sha256=str(expected_mapping.get("sha256")),
                    commit=P25_PREREG_COMMIT,
                )
                and _exact(
                    dict(observed_mapping),
                    dict(expected_mapping),
                )
            )
        else:
            p25_record_checks.append(
                _source_record_valid(
                    observed,
                    expected_path=str(expected_mapping.get("path")),
                    expected_sha256=str(expected_mapping.get("sha256")),
                    commit=P25_PREREG_COMMIT,
                )
            )

    unchanged_checks = [
        _source_record_valid(
            unchanged.get(name),
            expected_path=str(expected["path"]),
            expected_sha256=str(expected["sha256"]),
            commit=P25_PREREG_COMMIT,
        )
        for name, expected in EXPECTED_UNCHANGED_AUTHORITIES.items()
    ]
    p26_source_checks = [
        _source_record_valid(
            p26_sources.get(name),
            expected_path=path,
        )
        for name, path in EXPECTED_P26_SOURCE_PATHS.items()
    ]

    p25_current = _file_bytes(P25_OUTCOME_PATH)
    p25_tree = _git_text("rev-parse", f"{P25_OUTCOME_COMMIT}^{{tree}}")
    p25_prereg_tree = _git_text("rev-parse", f"{P25_PREREG_COMMIT}^{{tree}}")
    source_freeze_commit = str(EXPECTED_CONTROL_HISTORY["pre_runtime_source_freeze_commit"])
    source_freeze_tree = _git_text("rev-parse", f"{source_freeze_commit}^{{tree}}")
    source_freeze_parents = _git_text("show", "-s", "--format=%P", source_freeze_commit)
    runtime_review_commit = str(EXPECTED_PREPARED_RUNTIME["runtime_review_commit"])
    runtime_review_tree = _git_text("rev-parse", f"{runtime_review_commit}^{{tree}}")
    runtime_review_parents = _git_text("show", "-s", "--format=%P", runtime_review_commit)
    runtime_review_delta = _git_text(
        "diff", "--name-status", P25_PREREG_COMMIT, runtime_review_commit
    )
    runtime_lock_blob = _git_blob(
        runtime_review_commit, "experiments/training/p25_cuda_runtime_lock.json"
    )
    host_attestation_blob = _git_blob(
        runtime_review_commit, "experiments/training/p25_host_attestation.json"
    )

    checks = {
        "strict_canonical_json_and_closed_top_level": (
            isinstance(raw, Mapping)
            and canonical_bytes == _canonical_bytes(raw)
            and list(raw) == EXPECTED_TOP_LEVEL_ORDER
            and set(raw) == set(EXPECTED_TOP_LEVEL_ORDER)
        ),
        "identity_purpose_and_preexecution_boundary_exact": (
            contract.get("schema_version") == EXPECTED_SCHEMA
            and contract.get("status") == EXPECTED_STATUS
            and contract.get("branch") == EXPECTED_BRANCH
            and contract.get("purpose") == EXPECTED_PURPOSE
            and contract.get("claim_boundary") == EXPECTED_CLAIM_BOUNDARY
        ),
        "p25_terminal_parent_record_exact": _exact(dict(terminal_parent), EXPECTED_TERMINAL_PARENT),
        "control_history_and_pre_runtime_source_freeze_exact": (
            _exact(dict(control_history), EXPECTED_CONTROL_HISTORY)
            and source_freeze_tree == EXPECTED_CONTROL_HISTORY["pre_runtime_source_freeze_tree"]
            and source_freeze_parents == P25_OUTCOME_COMMIT
        ),
        "p25_outcome_commit_tree_and_bytes_exact": (
            p25_tree == P25_OUTCOME_TREE
            and p25_outcome_blob is not None
            and _sha256(p25_outcome_blob) == P25_OUTCOME_SHA256
            and p25_current is not None
            and _sha256(p25_current) == P25_OUTCOME_SHA256
        ),
        "p25_terminal_semantics_reconstruct": (
            p25_outcome.get("status") == "terminal_pre_trace_host_wrapper_permission_stop"
            and p25_outcome.get("attempt_id") == "20260906-01"
            and p25_diagnostic_checks == {"total": 40, "true": 40, "false": 0}
            and p25_comparisons == {"unprivileged_cmp": 2, "privileged_cmp": 0}
            and p25_gates.get("trace_off_a") == "not_started"
            and p25_gates.get("candidate_observations") == 0
            and p25_terminal.get("attempt_is_terminal") is True
            and p25_terminal.get("retry_until_favorable_allowed") is False
            and p25_terminal.get("reuse_attempt_20260906_01_allowed") is False
        ),
        "fresh_attempt_and_no_favorable_rerun_exact": _exact(
            dict(fresh_attempt), EXPECTED_FRESH_ATTEMPT
        ),
        "prepared_runtime_record_exact_and_pre_scientific": _exact(
            dict(prepared_runtime), EXPECTED_PREPARED_RUNTIME
        ),
        "prepared_runtime_commit_tree_delta_and_blobs_exact": (
            runtime_review_tree == EXPECTED_PREPARED_RUNTIME["runtime_review_tree"]
            and runtime_review_parents == P25_PREREG_COMMIT
            and runtime_review_delta
            == "A\texperiments/training/p25_cuda_runtime_lock.json\n"
            "A\texperiments/training/p25_host_attestation.json"
            and runtime_lock_blob is not None
            and _sha256(runtime_lock_blob) == EXPECTED_PREPARED_RUNTIME["runtime_lock_sha256"]
            and host_attestation_blob is not None
            and _sha256(host_attestation_blob)
            == EXPECTED_PREPARED_RUNTIME["host_attestation_sha256"]
        ),
        "frozen_p25_preregistration_exact": (
            frozen_p25.get("source_preregistration_commit") == P25_PREREG_COMMIT
            and frozen_p25.get("source_preregistration_tree") == P25_PREREG_TREE
            and p25_prereg_tree == P25_PREREG_TREE
            and set(frozen_p25) == set(EXPECTED_FROZEN_P25_CONTROL)
        ),
        "frozen_p25_source_hashes_reconstruct": all(p25_record_checks),
        "p26_control_source_paths_hashes_and_schema_exact": (
            set(p26_sources) == set(EXPECTED_P26_SOURCE_PATHS) and all(p26_source_checks)
        ),
        "permission_safe_bridge_exact": _exact(dict(bridge), EXPECTED_BRIDGE),
        "unchanged_scientific_authorities_exact": (
            _exact(dict(unchanged), EXPECTED_UNCHANGED_AUTHORITIES) and all(unchanged_checks)
        ),
        "runtime_contract_exact": _exact(dict(runtime), EXPECTED_RUNTIME),
        "acquisition_order_exact": _exact(
            contract.get("acquisition_order"), EXPECTED_ACQUISITION_ORDER
        ),
        "terminal_routing_exact": _exact(
            contract.get("terminal_routing"), EXPECTED_TERMINAL_ROUTING
        ),
        "forbidden_changes_exact": _exact(
            contract.get("forbidden_changes"), EXPECTED_FORBIDDEN_CHANGES
        ),
        "no_scientific_result_fields_present": (
            FORBIDDEN_RESULT_KEYS.isdisjoint(_all_mapping_keys(raw))
        ),
    }
    return {
        "schema_version": "passive-muon-p26-permission-safe-acquisition-reconstruction-v1",
        "canonical_sha256": _sha256(canonical_bytes),
        "checks": checks,
        "internally_consistent": all(checks.values()),
        "claim_boundary": (
            "Independent static reconstruction of the post-runtime, pre-diagnostic P26 "
            "permission-safe acquisition contract; not evidence that the diagnostic, bridge, "
            "CUDA trace, fidelity evaluation, shield execution, or training run occurred."
        ),
    }


def main() -> int:
    result = reconstruct(_parse_args().canonical.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["internally_consistent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
