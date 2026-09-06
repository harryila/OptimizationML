#!/usr/bin/env python3
"""Independently reconstruct the frozen, pre-execution P25 contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANONICAL = ROOT / "experiments/training/p25_cuda_diagnostic_contract.json"
EXPECTED_CANONICAL_SHA256 = "51e1fc22f685cc904fcdfff2775cff4293fdaa67a6e048072ee4dc49777f34e2"
EXPECTED_SCHEMA = "passive-muon-p25-cuda-diagnostic-correction-contract-v1"
EXPECTED_STATUS = "frozen_pre_remediation_build_and_pre_acquisition"
PARENT_COMMIT = "a8a1199b73bb5510b6ac154965baa91fc3743cab"
PARENT_TREE = "87862256b9298d27daa16b8866523c3a4b420309"
PARENT_TAG = "p24-cuda-executable-origin-hardening-diagnostic"
PARENT_TAG_OBJECT = "8a4f814c131cf64556f61b89078405b2e970d950"
P24_NATIVE_SHA256 = "ce44c53c9f274cef3eda7b3adea755ce6b1313773c8dfe00fc9f194b74fb2bd2"
P24_NATIVE_BYTES = 3_760_729
P24_OUTCOME_PATH = "results/summaries/p24_cuda_executable_origin_outcome.json"
P24_OUTCOME_SHA256 = "1b1a75d823b45d3c04daa024839dfbbe9461de9fac38117b0602ea7fbbcf941f"
EXPECTED_BASE_IMAGE = (
    "localhost:5000/p23-runtime@"
    "sha256:44ef23717780b1cbf112b183e7988b1319ddfed6b1d224efaa33e1e6d96de4c1"
)
EXPECTED_ORDER_FIRST = (
    "finalize every P25 execution-source hash and commit a clean source-freeze bootstrap"
)
EXPECTED_ORDER_TRACE_A = (
    "trace_off_a using the unchanged P23 runner with a fresh required failure-output path"
)
EXPECTED_ORDER_LAST = (
    "sanitize and retain every success or first-failure artifact without overwriting any prior "
    "evidence"
)
EXPECTED_EXECUTION_PATHS = {
    "corrected_executable_origin_diagnostic": (
        "experiments/training/run_p25_executable_origin_diagnostic.py"
    ),
    "corrected_executable_origin_sanitizer": (
        "scripts/sanitize_p25_executable_origin_diagnostic.py"
    ),
    "historical_p24_native_ingester": "scripts/ingest_p24_native_outcome_for_p25.py",
    "host_attempt_orchestrator": "scripts/run_p25_cuda_attempt.sh",
    "cuda_acquisition_runner": ("experiments/training/run_p23_deterministic_cuda_shadow_trace.py"),
    "stored_computation_and_shadow_semantics": "src/passive_muon/p21_shadow_trace.py",
}
EXPECTED_UNCHANGED = {
    "p21_fidelity_gates": (
        "experiments/training/p21_shadow_trace_protocol.json",
        "ca40ef0aef676971ca4464c3f838c6bd956ecc9806c2023af0d4ea699375315a",
    ),
    "p22_protocol": (
        "experiments/training/p22_real_gradient_shadow_trace_protocol.json",
        "f88eda60b366b561d277e41335b02c3fb9861a2604a947f8dd5c28fb54bedeac",
    ),
    "p22_erratum": (
        "theory/p22_real_gradient_shadow_trace_protocol_erratum.md",
        "42dc57bd8fb03b619c0de75c58570991ca520131ab7d1660b4f9373fd756a2cd",
    ),
    "p22_fused_qkv_certificate": (
        "results/summaries/p22_scalable_shield_shape_extension_certificate.json",
        "9480525384a57bb09869a55c878251f44f170930f3050a1ad50b4d934dcb7450",
    ),
    "p23_addendum": (
        "experiments/training/p23_deterministic_cuda_shadow_trace_addendum.json",
        "ad29a15e3b027250a35b6c3af6ce214cc98d664cad7d9821730e7e0cdaab9e3a",
    ),
    "p24_static_contract": (
        "experiments/training/p24_cuda_executable_origin_contract.json",
        "bdb2d3aad7d70d0ed5c6dddcd03642c9384d8b67d6f5ca507987d28e122dd446",
    ),
    "p24_outcome": (P24_OUTCOME_PATH, P24_OUTCOME_SHA256),
}
EXPECTED_TOP_KEYS = {
    "schema_version",
    "status",
    "authority",
    "provenance_parent",
    "parent_p24_terminal_facts",
    "unchanged_authorities",
    "execution_sources",
    "source_hash_finalization",
    "remediation_design",
    "historical_p24_ingestion",
    "corrected_diagnostic_contract",
    "corrected_sanitizer_contract",
    "replacement_runtime",
    "frozen_p23_gates",
    "ordered_execution",
    "failure_policy",
    "claim_boundary",
}
EXPECTED_MISMATCHES = {
    "deterministic_algorithms": {"observed_live": False, "committed_lock": True},
    "deterministic_debug_mode": {"observed_live": 0, "committed_lock": "error"},
    "interop_threads": {"observed_live": 30, "committed_lock": 1},
    "cudnn_deterministic": {"observed_live": False, "committed_lock": True},
    "fp16_reduced_precision_reduction": {
        "observed_live": True,
        "committed_lock": False,
    },
    "bf16_reduced_precision_reduction": {
        "observed_live": True,
        "committed_lock": False,
    },
}
EXPECTED_TORCH_STATE = {
    "deterministic_algorithms": True,
    "deterministic_debug_mode": "error",
    "cpu_threads": 1,
    "interop_threads": 1,
    "cudnn_deterministic": True,
    "cudnn_benchmark": False,
    "float32_matmul_precision": "highest",
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
    "fp16_reduced_precision_reduction": False,
    "bf16_reduced_precision_reduction": False,
    "math_sdpa_enabled": True,
    "flash_sdpa_enabled": False,
    "memory_efficient_sdpa_enabled": False,
    "cudnn_sdpa_enabled": False,
}
EXPECTED_GATES = {
    "thresholds_may_change": False,
    "numeric_tolerance": None,
    "allclose_allowed": False,
    "repeatability_required_mismatches": 0,
    "noninterference_required_mismatches": 0,
    "trace_on_observations": 1152,
    "actual_post_aspect_cuda_candidates": 1152,
    "shape_inventory": "48/48 Muon parameters and 84934656/84934656 elements",
    "fidelity_gates": (
        "unchanged frozen P21/P22 activation, correction, cosine, amplitude, P16, P18, "
        "and mild-intervention gates"
    ),
    "state_and_rng_schedule": "unchanged from P22",
}


class DuplicateKeyError(ValueError):
    """Raised when canonical JSON is not injectively parsed."""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    return parser.parse_args()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _strict_json(data: bytes) -> object:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise DuplicateKeyError(key)
            result[key] = value
        return result

    return json.loads(data, object_pairs_hook=reject_duplicates)


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, UnicodeError):
        return None


def _git_blob(commit: str, path: str) -> bytes | None:
    try:
        return subprocess.run(
            ["git", "show", f"{commit}:{path}"], cwd=ROOT, check=True, capture_output=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def _file_hash(path_text: object) -> str | None:
    if not isinstance(path_text, str):
        return None
    path = ROOT / path_text
    try:
        return _sha256(path.read_bytes())
    except OSError:
        return None


def reconstruct(canonical: Path = DEFAULT_CANONICAL) -> dict[str, object]:
    """Rebuild P25's exact pre-execution claims from independent constants."""

    try:
        canonical_bytes = canonical.read_bytes()
        raw = _strict_json(canonical_bytes)
    except (OSError, UnicodeError, json.JSONDecodeError, DuplicateKeyError):
        canonical_bytes = b""
        raw = None
    contract = _mapping(raw)
    parent = _mapping(contract.get("provenance_parent"))
    outcome = _mapping(parent.get("outcome"))
    terminal = _mapping(contract.get("parent_p24_terminal_facts"))
    native = _mapping(terminal.get("native_artifact"))
    checks_record = _mapping(terminal.get("checks"))
    sanitizer_failure = _mapping(terminal.get("sanitizer_failure"))
    unchanged = _mapping(contract.get("unchanged_authorities"))
    sources = _mapping(contract.get("execution_sources"))
    finalization = _mapping(contract.get("source_hash_finalization"))
    remediation = _mapping(contract.get("remediation_design"))
    historical = _mapping(contract.get("historical_p24_ingestion"))
    diagnostic = _mapping(contract.get("corrected_diagnostic_contract"))
    sanitizer = _mapping(contract.get("corrected_sanitizer_contract"))
    replacement = _mapping(contract.get("replacement_runtime"))
    gpu_contract = _mapping(replacement.get("gpu_contract"))
    mount_contract = _mapping(replacement.get("mount_contract"))
    failure = _mapping(contract.get("failure_policy"))
    boundary = _mapping(contract.get("claim_boundary"))
    ordered = contract.get("ordered_execution")

    p24_blob = _git_blob(PARENT_COMMIT, P24_OUTCOME_PATH)
    observed_unchanged = {
        name: _file_hash(path) for name, (path, _expected_hash) in EXPECTED_UNCHANGED.items()
    }
    observed_sources = {name: _file_hash(path) for name, path in EXPECTED_EXECUTION_PATHS.items()}
    recorded_source_hashes = {
        name: _mapping(sources.get(name)).get("sha256") for name in EXPECTED_EXECUTION_PATHS
    }

    expected_finalization = {
        "runner": "scripts/finalize_p25_cuda_diagnostic_contract.py",
        "allowed_records": [
            "corrected_executable_origin_diagnostic",
            "corrected_executable_origin_sanitizer",
            "historical_p24_native_ingester",
            "host_attempt_orchestrator",
        ],
        "finalization_must_precede_static_commit": True,
        "finalization_must_precede_image_build": True,
        "placeholder_allowed_at_execution": False,
        "rehash_at_every_evidence_process": True,
    }
    expected_mount = {
        "bind_mount_count": 10,
        "tmpfs_count": 1,
        "tmpfs_destination": "/tmp",
        "tmpfs_options": "rw,noexec,nosuid,nodev,size=1073741824",
        "source_data_and_host_evidence_mounts_read_only": True,
        "native_evidence_mount_writable": True,
        "destination_paths_unchanged_from_p23": True,
    }

    checks = {
        "canonical_sha256": (
            re.fullmatch(r"[0-9a-f]{64}", EXPECTED_CANONICAL_SHA256) is not None
            and _sha256(canonical_bytes) == EXPECTED_CANONICAL_SHA256
        ),
        "schema_status_and_closed_top_level": (
            contract.get("schema_version") == EXPECTED_SCHEMA
            and contract.get("status") == EXPECTED_STATUS
            and set(contract) == EXPECTED_TOP_KEYS
        ),
        "parent_git_identity": (
            parent.get("outcome_commit") == PARENT_COMMIT
            and parent.get("outcome_tree") == PARENT_TREE
            and parent.get("outcome_tag") == PARENT_TAG
            and parent.get("annotated_tag_object") == PARENT_TAG_OBJECT
            and parent.get("tag_target_commit") == PARENT_COMMIT
            and _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") == PARENT_TREE
            and _git("rev-parse", f"{PARENT_TAG}^{{tag}}") == PARENT_TAG_OBJECT
            and _git("rev-parse", f"{PARENT_TAG}^{{}}") == PARENT_COMMIT
        ),
        "parent_outcome_bytes": (
            outcome.get("path") == P24_OUTCOME_PATH
            and outcome.get("sha256") == P24_OUTCOME_SHA256
            and p24_blob is not None
            and _sha256(p24_blob) == P24_OUTCOME_SHA256
            and observed_unchanged.get("p24_outcome") == P24_OUTCOME_SHA256
        ),
        "p24_terminal_no_rerun": (
            terminal.get("attempt_disposition")
            == "terminal_first_retained_baseline_diagnostic_failure"
            and terminal.get("p24_rerun_allowed") is False
            and terminal.get("old_p23_container_diagnostic_may_be_reexecuted") is False
        ),
        "p24_native_binding": (
            native.get("sha256") == P24_NATIVE_SHA256
            and native.get("byte_count") == P24_NATIVE_BYTES
            and native.get("mode") == "baseline"
            and native.get("status") == "fails"
            and native.get("passes") is False
            and native.get("diagnostic_exit_code") == 1
        ),
        "p24_exact_checks": (
            checks_record
            == {
                "total": 36,
                "true": 35,
                "false": 1,
                "sole_false_check": "torch_determinism_matches_lock",
            }
            and terminal.get("torch_determinism_mismatches") == EXPECTED_MISMATCHES
        ),
        "p24_sanitizer_failure": (
            sanitizer_failure.get("exit_code") == 2
            and sanitizer_failure.get("sanitized_artifact_created") is False
            and sanitizer_failure.get("classification") == "absolute_mapping_keys_not_transformed"
            and sanitizer_failure.get("known_native_absolute_key_paths")
            == [
                "/runtime_lock_binding/content/container/tmpfs_contract/mounts/~1tmp",
                "/host_attestation_binding/content/container/tmpfs_contract/mounts/~1tmp",
            ]
        ),
        "unchanged_authority_inventory": (
            set(unchanged) == set(EXPECTED_UNCHANGED)
            and all(
                _mapping(unchanged.get(name))
                == {"path": path, "sha256": digest, "mutation_allowed": False}
                for name, (path, digest) in EXPECTED_UNCHANGED.items()
            )
            and all(
                observed_unchanged.get(name) == digest
                for name, (_path, digest) in EXPECTED_UNCHANGED.items()
            )
        ),
        "execution_source_inventory": (
            set(sources) == set(EXPECTED_EXECUTION_PATHS)
            and all(
                _mapping(sources.get(name)).get("path") == path
                and _mapping(sources.get(name)).get("mutation_allowed_after_freeze") is False
                for name, path in EXPECTED_EXECUTION_PATHS.items()
            )
        ),
        "execution_source_hashes_final_and_match": (
            all(
                isinstance(recorded_source_hashes[name], str)
                and re.fullmatch(r"[0-9a-f]{64}", str(recorded_source_hashes[name])) is not None
                and recorded_source_hashes[name] == observed_sources[name]
                for name in EXPECTED_EXECUTION_PATHS
            )
            and all(value != "PENDING_SHA256" for value in recorded_source_hashes.values())
        ),
        "one_shot_hash_finalization": finalization == expected_finalization,
        "p24_image_recipe_reused_exactly": (
            _mapping(remediation.get("build_patch"))
            == {
                "path": "experiments/training/prepare_p24_immutable_remote_module.py",
                "sha256": "97a4a730aadd11de1aa1b23107cef53f95c3348bde905932571eb6a68d512bef",
                "mutation_allowed_after_freeze": False,
            }
            and _mapping(remediation.get("image_recipe"))
            == {
                "path": "experiments/training/p24_runtime.Dockerfile",
                "sha256": "c1ccf0783b10f9e79795fa575398a58a7e6d7a9f2abe47a3a0650b213978e2f9",
                "base_image": EXPECTED_BASE_IMAGE,
                "no_cache_required": True,
                "mutation_allowed_after_freeze": False,
            }
        ),
        "historical_ingestion_preserves_failure": (
            historical.get("input_must_match_parent_native_sha256_and_byte_count") is True
            and historical.get("attempt_count") == 1
            and historical.get("offline_container_image") == EXPECTED_BASE_IMAGE
            and historical.get("offline_container_network_mode") == "none"
            and historical.get("offline_container_root_filesystem_read_only") is True
            and historical.get("offline_container_gpu_devices") == 0
            and historical.get("fresh_output_root_pattern")
            == "/secure/p25/historical-YYYYMMDD-NN/evidence"
            and historical.get("historical_p23_p24_evidence_directory_mounted_writable") is False
            and historical.get("retained_p24_native_overlay_mounted_read_only") is True
            and historical.get("exact_original_logical_root_destinations_required") is True
            and historical.get("exact_schema_and_all_36_checks_reconstructed") is True
            and historical.get("exact_six_determinism_field_mismatches_required") is True
            and historical.get("unexpected_mismatch_allowed") is False
            and historical.get("exact_two_absolute_tmp_key_locations_required") is True
            and historical.get("preregistration_parent_must_equal_source_freeze_bootstrap") is True
            and historical.get("preregistration_exact_name_status")
            == "A\tresults/summaries/p25_historical_p24_native_outcome.json"
            and historical.get("historical_wrapper_must_be_absent_at_bootstrap") is True
            and historical.get("output_retains_p24_failed_disposition") is True
            and historical.get("historical_ingestion_is_not_a_new_diagnostic_attempt") is True
        ),
        "corrected_diagnostic_is_one_shot_and_ordered": (
            diagnostic.get("schema_version") == "passive-muon-p25-executable-origin-diagnostic-v1"
            and diagnostic.get("mode") == "remediated"
            and diagnostic.get("attempt_count") == 1
            and diagnostic.get("standard_library_preflight_before_torch_import") is True
            and diagnostic.get("torch_absent_from_sys_modules_during_preflight") is True
            and diagnostic.get("cuda_uninitialized_before_and_after_determinism_configuration")
            is True
            and isinstance(diagnostic.get("ordered_trigger"), list)
            and len(diagnostic.get("ordered_trigger", [])) == 11
            and diagnostic.get("diagnostic_is_not_gradient_acquisition") is True
        ),
        "locked_torch_state_exact": diagnostic.get("locked_torch_state") == EXPECTED_TORCH_STATE,
        "corrected_sanitizer_keys_and_values": (
            sanitizer.get("schema_version")
            == "passive-muon-p25-sanitized-executable-origin-diagnostic-v1"
            and sanitizer.get("strict_duplicate_json_key_rejection") is True
            and sanitizer.get("mapping_keys_and_values_use_one_exact_root_transform") is True
            and sanitizer.get("post_redaction_mapping_key_collision_rejection") is True
            and sanitizer.get("mapping_key_and_value_replacement_counts_required") is True
            and sanitizer.get("total_replacement_count_required") is True
            and sanitizer.get("complete_post_redaction_absolute_path_scan_required") is True
            and sanitizer.get("passing_manifest_top_level_schema_closed") is True
            and sanitizer.get("passing_manifest_exact_check_inventory_count") == 40
            and sanitizer.get("passing_manifest_requires_every_exact_check_true") is True
            and sanitizer.get("independent_contract_source_and_artifact_binding_reconstruction")
            is True
            and sanitizer.get(
                "independent_runtime_lock_host_attestation_and_backlink_reconstruction"
            )
            is True
            and sanitizer.get(
                "independent_locked_python_torch_and_live_cuda_gpu_state_reconstruction"
            )
            is True
            and sanitizer.get("independent_pinned_torch_source_reconstruction") is True
            and sanitizer.get("independent_process_event_and_module_snapshot_reconstruction")
            is True
            and sanitizer.get("independent_immutable_generated_module_closure_reconstruction")
            is True
            and sanitizer.get("failed_or_error_manifest_cannot_authorize_acquisition") is True
            and sanitizer.get("output_no_overwrite") is True
        ),
        "replacement_runtime_fresh": (
            replacement.get("attempt_root_pattern") == "/secure/p25/attempt-YYYYMMDD-NN"
            and replacement.get("attempt_root_must_be_fresh_and_empty") is True
            and replacement.get("new_oci_digest_required") is True
            and replacement.get("new_running_container_required") is True
            and replacement.get("new_p25_named_runtime_lock_required") is True
            and replacement.get("new_p25_named_host_attestation_required") is True
            and replacement.get("root_filesystem_read_only") is True
            and replacement.get("network_mode") == "none"
            and gpu_contract
            == {
                "uuid": "GPU-ee4c9bf9-42f5-7cdd-66d2-f638c2db549d",
                "name": "NVIDIA A100-SXM4-40GB",
                "compute_capability": [8, 0],
                "mig_enabled": False,
                "device_count": 1,
                "bf16_required": True,
                "cpu_mps_mig_or_ordinal_substitution_allowed": False,
            }
            and replacement.get("diagnostic_parent_must_equal_preregistration_head") is True
            and replacement.get("diagnostic_exact_name_status")
            == [
                "A\texperiments/training/p25_cuda_runtime_lock.json",
                "A\texperiments/training/p25_host_attestation.json",
            ]
            and replacement.get("diagnostic_to_acquisition_commit_bridge_required") is True
            and replacement.get("bridge_parent_must_equal_diagnostic_head") is True
            and replacement.get("bridge_exact_name_status")
            == "A\tresults/summaries/p25_executable_origin_diagnostic.sanitized.json"
            and replacement.get(
                "bridge_reconstructs_contract_and_rehashes_all_frozen_sources_before_gradients"
            )
            is True
            and replacement.get(
                "bridge_rehashes_retained_native_against_strictly_parsed_wrapper_before_gradients"
            )
            is True
            and replacement.get("bridge_delta_scope")
            == (
                "one direct-child commit adding only results/summaries/"
                "p25_executable_origin_diagnostic.sanitized.json; no other path may change"
            )
        ),
        "exact_ten_mount_contract": (
            mount_contract == expected_mount
            and replacement.get("host_inspection_files_precreated_before_container_launch") is True
            and replacement.get("running_inspection_and_mountinfo_updated_in_place_after_launch")
            is True
            and replacement.get("rename_replacement_of_bound_host_inspection_files_forbidden")
            is True
        ),
        "unchanged_p23_gates": contract.get("frozen_p23_gates") == EXPECTED_GATES,
        "ordered_gates_complete": (
            isinstance(ordered, list)
            and len(ordered) == 24
            and ordered[0] == EXPECTED_ORDER_FIRST
            and ordered[13]
            == "run exactly one corrected P25 remediated executable-origin diagnostic"
            and ordered[17] == EXPECTED_ORDER_TRACE_A
            and ordered[19] == "verify_repeatability using exact equality only"
            and ordered[20] == "trace_on only if exact repeatability reports zero mismatches"
            and ordered[21] == "verify_noninterference using exact equality only"
            and ordered[22]
            == "aggregate_fidelity only if exact noninterference reports zero mismatches"
            and ordered[23] == EXPECTED_ORDER_LAST
            and len(set(ordered)) == len(ordered)
        ),
        "no_favorable_rerun_policy": (
            failure.get("no_favorable_reruns") is True
            and failure.get("p24_attempt_is_terminal") is True
            and failure.get("p25_corrected_diagnostic_attempts_allowed") == 1
            and failure.get("threshold_tuning_after_observation") is False
            and failure.get("trace_off_repeatability_failure")
            == "trace_on remains mechanically barred"
        ),
        "pre_execution_claim_boundary": (
            boundary.get("pre_execution_contract_only") is True
            and boundary.get("p25_cuda_result_present") is False
            and boundary.get("p24_disposition_changed") is False
            and isinstance(boundary.get("not_claimed"), list)
            and "real-gradient fidelity" in boundary.get("not_claimed", [])
            and "CUDA repeatability" in boundary.get("not_claimed", [])
        ),
    }
    return {
        "schema_version": "passive-muon-p25-cuda-diagnostic-contract-reconstruction-v1",
        "canonical_path": str(canonical),
        "canonical_sha256": _sha256(canonical_bytes),
        "checks": checks,
        "internally_consistent": all(checks.values()),
        "scope": (
            "Pre-execution reconstruction only: no P25 diagnostic, replacement runtime, CUDA "
            "gradient, repeatability, fidelity, shield, or training result is inferred."
        ),
    }


def main() -> None:
    result = reconstruct(_parse_args().canonical.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["internally_consistent"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
