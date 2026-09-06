#!/usr/bin/env python3
"""Independently reconstruct P27's pre-localization, no-training contract."""

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
DEFAULT_CANONICAL = (
    ROOT / "experiments/training/p27_cuda_deleted_mapping_localization_contract.json"
)
EXPECTED_SCHEMA = "passive-muon-p27-cuda-deleted-mapping-localization-contract-v1"
EXPECTED_STATUS = "frozen_pre_localization_no_training"
EXPECTED_BRANCH = "p27-cuda-deleted-mapping-localization"
P26_OUTCOME_PATH = "results/summaries/p26_permission_safe_cuda_acquisition_outcome.json"
P26_OUTCOME_SHA256 = "9be5727213f1a23f00b05524940d88c2d02461510fb61316b29dacc6ee8ead47"
P26_OUTCOME_COMMIT = "5429da23ff18888daa2312c530a4587780484d8b"
P26_OUTCOME_TREE = "a5b42f820beb7b4be0b65d89c51d49c1a5436647"
P26_TAG = "p26-permission-safe-acquisition-checkpoint"
P26_TAG_OBJECT = "bacad707d3779bfa10957e18cb4c69b1a7f0cbce"
LOCALIZATION_PATH = "experiments/training/run_p27_cuda_deleted_mapping_localization.py"
# This value is deliberately fail-closed until the reviewed localization source is final.
EXPECTED_LOCALIZATION_SHA256 = "86ba200d69a029c64b2218581d6b92245930d5f1e60d0176aca22ad4d6ddf65a"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

EXPECTED_TOP_LEVEL_ORDER = [
    "schema_version",
    "status",
    "branch",
    "purpose",
    "claim_boundary",
    "terminal_parent",
    "sanitizer_only_correction",
    "unchanged_scientific_authorities",
    "localization_source",
    "host_orchestration",
    "p26_failure_ingestion",
    "p27_specific_sanitization",
    "fresh_localization_attempt",
    "locked_base_runtime",
    "two_phase_freeze",
    "diagnostic_stages",
    "mapping_record_schema",
    "probe_contract",
    "no_training_boundary",
    "execution_order",
    "terminal_routing",
    "forbidden_changes",
]
EXPECTED_PURPOSE = (
    "Classify and retain the exact deleted file-backed mapping that blocked P26 by running "
    "one fresh, preregistered, no-training CUDA localization process with structured /proc "
    "evidence."
)
EXPECTED_CLAIM_BOUNDARY = (
    "This is a pre-execution diagnostic contract. It records no P27 runtime, mapping "
    "observation, CUDA repeatability result, observer-noninterference result, real-gradient "
    "candidate, fidelity aggregate, shield execution, optimizer update, or training result."
)
EXPECTED_TERMINAL_PARENT = {
    "branch": "p26-permission-safe-acquisition-bridge",
    "outcome_path": P26_OUTCOME_PATH,
    "outcome_sha256": P26_OUTCOME_SHA256,
    "outcome_schema": "passive-muon-p26-permission-safe-cuda-acquisition-outcome-v1",
    "outcome_status": "terminal_trace_off_a_deleted_mapping_and_sanitizer_stop",
    "outcome_commit": P26_OUTCOME_COMMIT,
    "outcome_tree": P26_OUTCOME_TREE,
    "checkpoint_tag": P26_TAG,
    "checkpoint_tag_object": P26_TAG_OBJECT,
    "checkpoint_peeled_commit": P26_OUTCOME_COMMIT,
    "attempt_id": "20260906-02",
    "attempt_is_terminal": True,
    "attempt_reuse_allowed": False,
    "trace_off_a_failure_phase": "p22_initialized",
    "trace_off_a_error": ("deleted file-backed mapping is forbidden at /proc/self/maps line 17"),
    "trace_off_a_native_sha256": (
        "b2a92b5062988534bc424d3ecadf3977ecedc0f80aa27628d4a0457858a64c91"
    ),
    "raw_mapping_line_retained": False,
    "failure_sanitizer_error": (
        "undeclared absolute path cannot be sanitized: /opt/p23-venv/bin/python"
    ),
    "candidate_observations": 0,
}
EXPECTED_SANITIZER_CORRECTION = {
    "source_path": "experiments/training/p23_deterministic_cuda_shadow_trace.py",
    "terminal_p26_sha256": ("0e7ec5af9f7cc90ad21bed4a9f48f0f24a481ef680c636db613077b50e8c36c8"),
    "corrected_sha256": "ec1037839a3de378ee7391ae815ffc29c184865c5216d30027edc460ba112fab",
    "test_path": "tests/test_p23_deterministic_cuda_shadow_trace.py",
    "terminal_p26_test_sha256": (
        "8c4c71da943f3fb5ed569524f0dc24b7890eecffcaade44a2a5f9e8828807b3a"
    ),
    "corrected_test_sha256": ("7f8881bdde29d67b3c6e8a864ac18f7fd95ae35e070122ee77480038bf13aad5"),
    "permitted_semantic_change": (
        "Map only the exact already-validated PINNED_PYTHON_EXECUTABLE to "
        "python_environment:bin/python when and only when the declared python_environment "
        "root equals PINNED_PYTHON_ENVIRONMENT."
    ),
    "required_negative_control": (
        "Unrelated symlink escapes, a wrong Python environment root, undeclared absolute "
        "paths, deleted executable mappings, writable executable mappings, and "
        "unclassifiable mappings remain rejected."
    ),
    "acquisition_policy_changed": False,
    "deleted_mapping_policy_changed": False,
    "scientific_gate_changed": False,
}
EXPECTED_AUTHORITIES = {
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
    "p22_fused_qkv_certificate": {
        "path": "results/summaries/p22_scalable_shield_shape_extension_certificate.json",
        "sha256": "9480525384a57bb09869a55c878251f44f170930f3050a1ad50b4d934dcb7450",
    },
    "p23_addendum": {
        "path": "experiments/training/p23_deterministic_cuda_shadow_trace_addendum.json",
        "sha256": "ad29a15e3b027250a35b6c3af6ce214cc98d664cad7d9821730e7e0cdaab9e3a",
    },
    "p23_acquisition_runner": {
        "path": "experiments/training/run_p23_deterministic_cuda_shadow_trace.py",
        "sha256": "a2b4bb5b686b7f681958d09be1d465917b40e34d45e4d3503efef0f35e7ae8cb",
    },
    "p21_stored_computation": {
        "path": "src/passive_muon/p21_shadow_trace.py",
        "sha256": "020debed7a12d6f538ef9093fcdfbcafbf571ea162fb868ccf0a49b5bebb0825",
    },
    "p22_observer": {
        "path": "experiments/training/p22_observed_muon.py",
        "sha256": "5355b553e871406491599945ed790288f59d636a7f9e1d931b9d8234ba42818f",
    },
}
EXPECTED_LOCALIZATION_SOURCE = {
    "host_control_path": LOCALIZATION_PATH,
    "in_container_evidence_path": (
        "/workspace/evidence/p23/run_p27_cuda_deleted_mapping_localization.py"
    ),
    "sha256": EXPECTED_LOCALIZATION_SHA256,
    "test_path": "tests/test_p27_cuda_deleted_mapping_localization.py",
    "test_sha256": "f0cae4d3428c66687298fd5e87dd94f805e1d5dcf780ca98d982372388eb741f",
    "placeholder_must_be_replaced_before_execution": False,
    "output_schema": "passive-muon-p27-cuda-deleted-mapping-localization-v1",
    "native_output_path": (
        "/workspace/evidence/p23/p27_cuda_deleted_mapping_localization.native.json"
    ),
    "copy_contract": (
        "Copy the reviewed bytes once into the already-counted writable native-evidence "
        "mount, verify the source and destination SHA-256 before execution, and reject any "
        "preexisting destination."
    ),
    "extra_mount_allowed": False,
    "authority_repository_path": "/workspace/OptimizationML",
    "authority_repository_head": "185e444afc0b44ca0a09b1bde49a6b6fa3973355",
    "authority_repository_tree": "24f4bdac331a57bd7c1b807747d7c7fba253ee5a",
    "p25_source_path": (
        "/workspace/OptimizationML/experiments/training/run_p25_executable_origin_diagnostic.py"
    ),
    "p25_source_sha256": ("6305fb9683503eb67e091cdfb0a1105628fcb4bec76fe5ef7fd23dd49ba9d770"),
    "p25_contract_path": (
        "/workspace/OptimizationML/experiments/training/p25_cuda_diagnostic_contract.json"
    ),
    "p25_contract_sha256": ("51e1fc22f685cc904fcdfff2775cff4293fdaa67a6e048072ee4dc49777f34e2"),
    "required_cli_arguments": [
        "--repository",
        "--nanogpt-root",
        "--muon-source",
        "--p25-source",
        "--runtime-lock",
        "--host-attestation",
        "--p25-contract",
        "--expected-source-sha256",
        "--expected-p25-source-sha256",
        "--expected-p25-contract-sha256",
        "--expected-runtime-lock-sha256",
        "--expected-host-attestation-sha256",
        "--expected-repository-head",
        "--expected-repository-tree",
        "--output",
    ],
}
EXPECTED_ORCHESTRATOR_SHA256 = "c80fc17b10cce7713a203ec3dc4a43bcc8357966d76b5e81d5c2644d3ea5ff0c"
EXPECTED_ORCHESTRATOR_TEST_SHA256 = (
    "d93316598886279fb4694170efcf0511444c6b55ac70bc7d3eebe56b72c8b2a7"
)
EXPECTED_HOST_ORCHESTRATION = {
    "source_path": "scripts/run_p27_cuda_deleted_mapping_localization.sh",
    "source_sha256": EXPECTED_ORCHESTRATOR_SHA256,
    "test_path": "tests/test_p27_cuda_deleted_mapping_localization_orchestration.py",
    "test_sha256": EXPECTED_ORCHESTRATOR_TEST_SHA256,
    "placeholders_must_be_replaced_before_execution": False,
    "allowed_phases": ["prepare-runtime", "run-localization"],
    "required_environment": [
        "P27_ATTEMPT_ID",
        "P27_GPU_UUID",
        "P27_SOURCE_FREEZE_COMMIT",
        "P27_SOURCE_FREEZE_TREE",
        "P27_RUNTIME_REVIEW_COMMIT",
        "P27_RUNTIME_REVIEW_TREE",
        "P27_EXPECTED_CONTRACT_SHA256",
        "P27_EXPECTED_ORCHESTRATOR_SHA256",
        "P27_EXPECTED_RECONSTRUCTOR_SHA256",
        "P27_EXPECTED_LOCALIZER_SHA256",
        "P27_EXPECTED_INGESTER_SHA256",
        "P27_EXPECTED_P23_CORE_SHA256",
        "P27_EXPECTED_P27_SANITIZER_SHA256",
        "P27_EXPECTED_RUNTIME_LOCK_SHA256",
        "P27_EXPECTED_HOST_ATTESTATION_SHA256",
        "P27_AUTHORITY_REPO",
        "P27_CONTROL_REPO",
        "P27_NANOGPT_HOST",
        "P27_MUON_HOST",
        "P27_DATA_HOST",
    ],
    "source_authorities_are_read_from_validated_contract": True,
    "prepare_runtime_invocations_allowed": 1,
    "localization_invocations_allowed": 1,
    "cleanup_or_retry_phase_exists": False,
    "exact_bind_mount_count": 10,
    "authority_checkout_is_only_repository_mount": True,
    "localizer_enters_through_existing_evidence_mount": True,
    "phase_two_review_required_before_localization": True,
}
EXPECTED_P26_INGESTION = {
    "native_path": "/secure/p25/attempt-20260906-02/evidence/trace-off-a-failure.json",
    "native_sha256": "b2a92b5062988534bc424d3ecadf3977ecedc0f80aa27628d4a0457858a64c91",
    "native_byte_count": 66283,
    "native_owner": "root:root",
    "native_mode": "0600",
    "sanitizer_source_path": "experiments/training/p23_deterministic_cuda_shadow_trace.py",
    "sanitizer_source_sha256": ("ec1037839a3de378ee7391ae815ffc29c184865c5216d30027edc460ba112fab"),
    "ingester_source_path": "scripts/ingest_p26_trace_off_a_failure.py",
    "ingester_source_sha256": ("3e7769d668a8756aea7359bef4d9b8ccfc8db66da102edfbc6c5de385654cdeb"),
    "ingester_test_path": "tests/test_p27_p26_failure_ingestion.py",
    "ingester_test_sha256": ("7846fbbe9c1768e52467e1178075595fdf979d600f2f7891a0aec9feffc43d41"),
    "read_native_exactly_once": True,
    "mutate_native_allowed": False,
    "outputs_must_be_fresh_under_p27_evidence": True,
    "output_in_terminal_p26_directory_allowed": False,
    "purpose": (
        "Retain a sanitized, hash-bound wrapper for the already-terminal P26 failure; never "
        "reinterpret or rerun P26."
    ),
}
EXPECTED_P27_SANITIZER_SHA256 = "3ec64cdc3681d41d70c624c6ffd26dde1a7e498bda13ddafaaced271a43dd6c4"
EXPECTED_P27_SANITIZATION = {
    "source_path": "scripts/sanitize_p27_cuda_deleted_mapping_localization.py",
    "sha256": EXPECTED_P27_SANITIZER_SHA256,
    "test_path": "tests/test_p27_cuda_deleted_mapping_localization_sanitizer.py",
    "test_sha256": "a3befd5a6968f7dcb83d9ecfdba2c84e5457d0a83f4cfff0b68aa5d9585ad089",
    "placeholder_must_be_replaced_before_execution": False,
    "execution_location": "host-side root bridge",
    "interpreter": "/usr/bin/python3",
    "container_execution_allowed": False,
    "native_output_external_only": True,
    "native_required_owner": "root:root",
    "native_required_mode": "0600",
    "wrapper_schema": "passive-muon-p27-sanitized-cuda-deleted-mapping-localization-v1",
    "wrapper_must_bind_native_sha256_byte_count_mode_uid_gid": True,
    "preserve_all_nonpath_evidence": True,
    "path_transform": (
        "Replace every raw-line, pathname, readlink-target, argv, and other absolute-path "
        "byte value with its SHA-256, byte count, and a declared logical label when one "
        "applies; publish no native absolute-path bytes."
    ),
    "duplicate_key_rejection": True,
    "nonfinite_rejection": True,
    "unknown_schema_rejection": True,
    "unknown_absolute_path_rejection": True,
    "fresh_nonaliasing_no_overwrite_output_required": True,
}
EXPECTED_FRESH_ATTEMPT = {
    "attempt_id": "20260906-01",
    "attempt_root": "/secure/p27/attempt-20260906-01",
    "container_name": "p27-localization-20260906-01",
    "root_must_not_preexist": True,
    "new_container_required": True,
    "terminal_p26_container_reuse_allowed": False,
    "new_container_id_required": True,
    "new_runtime_lock_required": True,
    "new_host_attestation_required": True,
    "localization_invocations_allowed": 1,
    "retry_until_favorable_allowed": False,
    "future_scientific_acquisition_attempt_id": "20260906-03",
    "future_scientific_acquisition_is_authorized": False,
}
EXPECTED_RUNTIME = {
    "image_digest": ("sha256:14dafde07ae578cc4725f452a51d4bc4c920b69f6af23125a72e27162e193ec0"),
    "prior_container_id_must_not_be_reused": (
        "e6682d8f09b9dc4be342354d520a8f6f8766a8e232840d1d7beb5e00e1fced6c"
    ),
    "gpu_uuid": "GPU-ee4c9bf9-42f5-7cdd-66d2-f638c2db549d",
    "gpu_name": "NVIDIA A100-SXM4-40GB",
    "network_mode": "none",
    "root_filesystem_read_only": True,
    "exact_bind_mount_count": 10,
    "authority_checkout_is_repository_mount": True,
    "diagnostic_uses_existing_native_evidence_mount": True,
    "extra_mounts_allowed": False,
    "tmpfs": {"/tmp": "rw,noexec,nosuid,nodev,size=1073741824"},
    "source_commit": "185e444afc0b44ca0a09b1bde49a6b6fa3973355",
    "source_tree": "24f4bdac331a57bd7c1b807747d7c7fba253ee5a",
}
EXPECTED_TWO_PHASE_FREEZE = {
    "phase_one": (
        "Commit the finalized P27 contract, localizer, host orchestrator, P26-ingestion "
        "sanitizer correction, P27-specific sanitizer, reconstructors, tests, workflow, and "
        "runbook before creating a container or observing a mapping."
    ),
    "phase_one_commit_is_selected_after_commit": True,
    "phase_two": (
        "Create the fresh container without running localization; freeze and review its P27 "
        "runtime lock and host attestation; commit exactly those two new artifacts as one "
        "direct child of the phase-one source freeze."
    ),
    "phase_two_exact_delta": [
        "A\texperiments/training/p27_cuda_runtime_lock.json",
        "A\texperiments/training/p27_host_attestation.json",
    ],
    "phase_two_localization_started": False,
    "phase_two_candidate_observations": 0,
    "diagnostic_requires_reviewed_phase_two_commit": True,
    "authority_repository_remains_185e444_mount": True,
}
EXPECTED_STAGES = [
    {
        "name": "pre_cuda",
        "boundary": (
            "After static contract, repository, source, runtime, process, and environment "
            "validation but before the first CUDA runtime initialization call."
        ),
    },
    {
        "name": "post_cuda_init",
        "boundary": (
            "Immediately after explicit torch.cuda initialization and synchronization, "
            "before constructing or moving the model."
        ),
    },
    {
        "name": "post_model_move",
        "boundary": (
            "Immediately after constructing GPT-2 small and moving it onto the pinned CUDA "
            "device, before optimizer construction."
        ),
    },
    {
        "name": "post_optimizer",
        "boundary": (
            "Immediately after constructing the pinned optimizer from the already-moved "
            "model and proving exact parameter identity and CUDA membership."
        ),
    },
]
EXPECTED_MAPPING_SCHEMA = {
    "snapshot_fields": [
        "stage",
        "time_ns",
        "process_pid",
        "proc_maps_sha256",
        "proc_maps_byte_count",
        "mapping_count",
        "deleted_mapping_count",
        "mappings",
        "post_probe_proc_maps_sha256",
        "post_probe_proc_maps_byte_count",
        "post_probe_mapping_count",
        "maps_byte_stable_during_probe",
        "all_deleted_mapping_identities_stable_during_probe",
    ],
    "mapping_fields": [
        "line_number",
        "raw_line_sha256",
        "raw_line_byte_count",
        "address_start_hex",
        "address_end_hex",
        "permissions",
        "offset_hex",
        "device_major_minor",
        "inode_decimal",
        "pathname_present",
        "pathname_bytes_hex",
        "deleted_suffix_present",
        "pathname_without_deleted_bytes_hex",
        "classification",
        "map_files_probe",
        "matching_fd_probes",
    ],
    "closed_classification_values": [
        "not_file_backed",
        "present_regular_read_only",
        "present_regular_executable",
        "present_regular_writable",
        "present_character_device_read_only",
        "present_character_device_executable",
        "present_character_device_writable",
        "present_other_nonregular_read_only",
        "present_other_nonregular_executable",
        "present_other_nonregular_writable",
        "present_unclassifiable",
        "deleted_regular_read_only",
        "deleted_regular_executable",
        "deleted_regular_writable",
        "deleted_character_device_read_only",
        "deleted_character_device_executable",
        "deleted_character_device_writable",
        "deleted_other_nonregular_read_only",
        "deleted_other_nonregular_executable",
        "deleted_other_nonregular_writable",
        "deleted_unclassifiable",
    ],
    "first_deleted_mapping_must_be_explicit": True,
    "all_deleted_mappings_must_be_retained": True,
    "raw_proc_maps_bytes_must_be_hash_bound": True,
    "path_bytes_must_not_be_decoded_lossily": True,
    "unrelated_map_churn_is_retained_evidence_not_automatic_failure": True,
    "deleted_target_identity_change_or_disappearance_is_fatal": True,
}
EXPECTED_PROBE_CONTRACT = {
    "map_files_probe_path": "/proc/self/map_files/{address_start_hex}-{address_end_hex}",
    "map_files_fields": [
        "attempted",
        "lstat_status",
        "readlink_status",
        "readlink_target_bytes_hex",
        "open_status",
        "fstat_status",
        "device_major_minor",
        "inode_decimal",
        "file_type",
        "mode_octal",
        "size_bytes",
        "errno",
        "error_message",
    ],
    "fd_root": "/proc/self/fd",
    "fd_probe_fields": [
        "fd_decimal",
        "readlink_status",
        "readlink_target_bytes_hex",
        "fstat_status",
        "device_major_minor",
        "inode_decimal",
        "file_type",
        "mode_octal",
        "size_bytes",
        "fd_flags_hex",
        "errno",
        "error_message",
    ],
    "probe_failures_are_structured_evidence": True,
    "probe_failure_must_not_be_silently_dropped": True,
    "matching_uses_exact_path_bytes_or_device_and_inode": True,
}
EXPECTED_NO_TRAINING = {
    "model_construction_allowed": True,
    "model_cuda_move_allowed": True,
    "optimizer_construction_allowed": True,
    "data_loading_allowed": False,
    "forward_allowed": False,
    "backward_allowed": False,
    "optimizer_step_allowed": False,
    "candidate_evaluation_allowed": False,
    "candidate_observation_count_required": 0,
    "gradient_observation_count_required": 0,
    "parameter_update_count_required": 0,
}
EXPECTED_EXECUTION_ORDER = [
    "independently reconstruct the terminal P26 outcome and this P27 contract",
    "verify the sanitizer-only correction and every unchanged scientific authority by exact hash",
    "prove the fresh P27 attempt root and container name do not exist",
    "create one new container from the exact locked P26 image digest with a new container ID",
    "freeze and review a new runtime lock and host attestation before localization",
    "prove the localization script hash equals the finalized contract binding",
    (
        "ingest the terminal P26 failure once with the narrowly corrected P23 sanitizer and "
        "stop before GPU or model allocation on any failure"
    ),
    (
        "run exactly one localization invocation and retain no-overwrite native stdout "
        "stderr and manifest"
    ),
    "collect pre_cuda then post_cuda_init then post_model_move then post_optimizer snapshots",
    (
        "stop without data loading forward backward optimizer step candidate evaluation or "
        "parameter update"
    ),
    "sanitize the P27 native result with the separate P27-specific sanitizer",
    "independently reconstruct the P26 ingestion binding P27 sanitized binding and P27 outcome",
    "route mechanically from the first exact deleted-mapping classification",
]
EXPECTED_ROUTING = {
    "selection_precedence": [
        "sanitization_or_reconstruction_failure",
        "deleted_executable_or_writable_mapping",
        "deleted_unclassifiable_or_identity_unstable_mapping",
        "deleted_conclusively_identified_read_only_nonexecutable_mapping_set",
        "no_deleted_mapping_reproduced",
    ],
    "routes": {
        "sanitization_or_reconstruction_failure": (
            "stop; retain native evidence externally and record the independent failure"
        ),
        "deleted_executable_or_writable_mapping": (
            "stop; retain every exact witness; do not authorize scientific acquisition; design "
            "an image or loader correction"
        ),
        "deleted_unclassifiable_or_identity_unstable_mapping": (
            "stop; retain every witness and exact failed probe or identity transition; do not "
            "relax deleted-mapping rejection"
        ),
        "deleted_conclusively_identified_read_only_nonexecutable_mapping_set": (
            "stop; retain every exact regular-file, character-device, and other-nonregular "
            "witness; preregister a narrow provenance or device-mapping policy before any "
            "scientific acquisition"
        ),
        "no_deleted_mapping_reproduced": (
            "record nonreproduction as the sole attempt result; do not infer safety and do not "
            "rerun until favorable"
        ),
    },
    "multiple_conditions_use_first_matching_route": True,
    "post_route_action": "freeze the exact P27 outcome and selected route",
    "future_scientific_acquisition_authorized": False,
    "future_scientific_acquisition_requires_separately_reviewed_contract": True,
}
EXPECTED_FORBIDDEN = [
    (
        "reuse, interrogate, restart, or mutate terminal P26 container "
        "e6682d8f09b9dc4be342354d520a8f6f8766a8e232840d1d7beb5e00e1fced6c for "
        "localization"
    ),
    "reuse or resume terminal P26 attempt 20260906-02",
    "run more than one P27 localization invocation",
    (
        "run a forward pass, backward pass, optimizer step, candidate evaluation, candidate "
        "capture, shield, or training update"
    ),
    (
        "change seed, model, optimizer, P21 or P23 scientific gate, data, capture schedule, "
        "equality rule, or fidelity threshold"
    ),
    (
        "permit deleted executable mappings, writable executable mappings, arbitrary symlink "
        "escapes, or unclassifiable mappings"
    ),
    "discard a map_files or fd probe failure instead of recording its exact errno",
    "decode /proc path bytes with lossy replacement",
    "replace exact hashes or equality with allclose or another tolerance",
    "rerun until a favorable deleted-mapping classification appears",
    ("authorize fresh scientific acquisition attempt 20260906-03 from this pre-execution contract"),
]
FORBIDDEN_RESULT_KEYS = {
    "result",
    "results",
    "outcome",
    "observed_mapping",
    "runtime_lock",
    "host_attestation",
    "trace_off_a",
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


def _source_record_valid(record: object, expected: Mapping[str, str]) -> bool:
    observed = _mapping(record)
    path = observed.get("path")
    digest = observed.get("sha256")
    expected_path = expected.get("path")
    expected_digest = expected.get("sha256")
    data = _file_bytes(path)
    return (
        set(observed) == {"path", "sha256"}
        and path == expected_path
        and digest == expected_digest
        and isinstance(digest, str)
        and SHA256_RE.fullmatch(digest) is not None
        and data is not None
        and _sha256(data) == digest
    )


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
    """Reconstruct all P27 pre-execution claims from independent constants."""

    try:
        canonical_bytes = canonical.read_bytes()
        raw = _strict_json(canonical_bytes)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError, DuplicateKeyError):
        canonical_bytes = b""
        raw = None
    contract = _mapping(raw)
    parent = _mapping(contract.get("terminal_parent"))
    correction = _mapping(contract.get("sanitizer_only_correction"))
    authorities = _mapping(contract.get("unchanged_scientific_authorities"))
    localization = _mapping(contract.get("localization_source"))
    orchestration = _mapping(contract.get("host_orchestration"))
    p26_ingestion = _mapping(contract.get("p26_failure_ingestion"))
    p27_sanitization = _mapping(contract.get("p27_specific_sanitization"))
    attempt = _mapping(contract.get("fresh_localization_attempt"))
    runtime = _mapping(contract.get("locked_base_runtime"))
    two_phase = _mapping(contract.get("two_phase_freeze"))

    p26_blob = _git_blob(P26_OUTCOME_COMMIT, P26_OUTCOME_PATH)
    current_p26 = _file_bytes(P26_OUTCOME_PATH)
    try:
        p26_raw = _strict_json(p26_blob) if p26_blob is not None else None
    except (ValueError, json.JSONDecodeError, DuplicateKeyError):
        p26_raw = None
    p26 = _mapping(p26_raw)
    p26_trace = _mapping(p26.get("trace_off_a_failure"))
    p26_native = _mapping(p26_trace.get("native_failure_artifact"))
    p26_limit = _mapping(p26_trace.get("rejected_mapping_evidence_limit"))
    p26_sanitization = _mapping(p26.get("failure_sanitization"))
    p26_gates = _mapping(p26.get("gate_status"))
    p26_policy = _mapping(p26.get("terminal_policy"))
    p26_route = _mapping(p26.get("route"))

    terminal_source = _git_blob(P26_OUTCOME_COMMIT, str(correction.get("source_path")))
    terminal_test = _git_blob(P26_OUTCOME_COMMIT, str(correction.get("test_path")))
    live_source = _file_bytes(correction.get("source_path"))
    live_test = _file_bytes(correction.get("test_path"))
    localization_bytes = _file_bytes(localization.get("host_control_path"))
    localization_test_bytes = _file_bytes(localization.get("test_path"))
    orchestrator_bytes = _file_bytes(orchestration.get("source_path"))
    orchestrator_test_bytes = _file_bytes(orchestration.get("test_path"))
    p27_sanitizer_bytes = _file_bytes(p27_sanitization.get("source_path"))
    p27_sanitizer_test_bytes = _file_bytes(p27_sanitization.get("test_path"))
    p26_ingester_bytes = _file_bytes(p26_ingestion.get("ingester_source_path"))
    p26_ingester_test_bytes = _file_bytes(p26_ingestion.get("ingester_test_path"))

    authority_checks = [
        _source_record_valid(authorities.get(name), expected)
        for name, expected in EXPECTED_AUTHORITIES.items()
    ]

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
        "terminal_parent_record_exact": _exact(dict(parent), EXPECTED_TERMINAL_PARENT),
        "terminal_p26_commit_tree_tag_and_bytes_exact": (
            _git_text("rev-parse", f"{P26_OUTCOME_COMMIT}^{{tree}}") == P26_OUTCOME_TREE
            and _git_text("rev-parse", P26_TAG) == P26_TAG_OBJECT
            and _git_text("rev-parse", f"{P26_TAG}^{{}}") == P26_OUTCOME_COMMIT
            and p26_blob is not None
            and _sha256(p26_blob) == P26_OUTCOME_SHA256
            and current_p26 is not None
            and _sha256(current_p26) == P26_OUTCOME_SHA256
        ),
        "terminal_p26_semantics_and_zero_observations_reconstruct": (
            p26.get("schema_version") == EXPECTED_TERMINAL_PARENT["outcome_schema"]
            and p26.get("status") == EXPECTED_TERMINAL_PARENT["outcome_status"]
            and p26.get("attempt_id") == "20260906-02"
            and p26_trace.get("failure_phase") == "p22_initialized"
            and p26_trace.get("error_message") == EXPECTED_TERMINAL_PARENT["trace_off_a_error"]
            and p26_native.get("sha256") == EXPECTED_TERMINAL_PARENT["trace_off_a_native_sha256"]
            and p26_limit.get("raw_mapping_line_retained") is False
            and p26_sanitization.get("status") == "failed_closed_no_output"
            and p26_sanitization.get("sanitized_failure_artifact_created") is False
            and p26_gates.get("candidate_observations") == 0
            and p26_gates.get("trace_off_b") == "not_run"
            and p26_gates.get("trace_on") == "mechanically_barred"
            and p26_policy.get("attempt_is_terminal") is True
            and p26_policy.get("reuse_attempt_20260906_02_allowed") is False
            and p26_route.get("next_branch") == EXPECTED_BRANCH
            and p26_route.get("new_attempt_required") is True
            and p26_route.get("threshold_tuning_allowed") is False
        ),
        "sanitizer_only_correction_record_exact": _exact(
            dict(correction), EXPECTED_SANITIZER_CORRECTION
        ),
        "sanitizer_terminal_and_corrected_bytes_exact": (
            terminal_source is not None
            and _sha256(terminal_source) == EXPECTED_SANITIZER_CORRECTION["terminal_p26_sha256"]
            and live_source is not None
            and _sha256(live_source) == EXPECTED_SANITIZER_CORRECTION["corrected_sha256"]
            and terminal_test is not None
            and _sha256(terminal_test) == EXPECTED_SANITIZER_CORRECTION["terminal_p26_test_sha256"]
            and live_test is not None
            and _sha256(live_test) == EXPECTED_SANITIZER_CORRECTION["corrected_test_sha256"]
        ),
        "unchanged_scientific_authorities_exact": (
            _exact(dict(authorities), EXPECTED_AUTHORITIES) and all(authority_checks)
        ),
        "localization_source_finalized_and_exact": (
            EXPECTED_LOCALIZATION_SHA256 != "PENDING_REVIEWED_LOCALIZATION_SCRIPT_SHA256"
            and SHA256_RE.fullmatch(EXPECTED_LOCALIZATION_SHA256) is not None
            and _exact(dict(localization), EXPECTED_LOCALIZATION_SOURCE)
            and localization.get("placeholder_must_be_replaced_before_execution") is False
            and localization_bytes is not None
            and _sha256(localization_bytes) == EXPECTED_LOCALIZATION_SHA256
            and localization_test_bytes is not None
            and _sha256(localization_test_bytes) == EXPECTED_LOCALIZATION_SOURCE["test_sha256"]
        ),
        "host_orchestration_finalized_and_exact": (
            EXPECTED_ORCHESTRATOR_SHA256 != "PENDING_REVIEWED_P27_ORCHESTRATOR_SHA256"
            and EXPECTED_ORCHESTRATOR_TEST_SHA256 != "PENDING_REVIEWED_P27_ORCHESTRATOR_TEST_SHA256"
            and SHA256_RE.fullmatch(EXPECTED_ORCHESTRATOR_SHA256) is not None
            and SHA256_RE.fullmatch(EXPECTED_ORCHESTRATOR_TEST_SHA256) is not None
            and _exact(dict(orchestration), EXPECTED_HOST_ORCHESTRATION)
            and orchestration.get("placeholders_must_be_replaced_before_execution") is False
            and orchestrator_bytes is not None
            and _sha256(orchestrator_bytes) == EXPECTED_ORCHESTRATOR_SHA256
            and orchestrator_test_bytes is not None
            and _sha256(orchestrator_test_bytes) == EXPECTED_ORCHESTRATOR_TEST_SHA256
        ),
        "terminal_p26_failure_ingestion_finalized_and_exact": (
            EXPECTED_P26_INGESTION["ingester_source_sha256"]
            != "PENDING_REVIEWED_P26_INGESTER_SHA256"
            and EXPECTED_P26_INGESTION["ingester_test_sha256"]
            != "PENDING_REVIEWED_P26_INGESTER_TEST_SHA256"
            and SHA256_RE.fullmatch(str(EXPECTED_P26_INGESTION["ingester_source_sha256"]))
            is not None
            and SHA256_RE.fullmatch(str(EXPECTED_P26_INGESTION["ingester_test_sha256"])) is not None
            and _exact(dict(p26_ingestion), EXPECTED_P26_INGESTION)
            and p26_ingester_bytes is not None
            and _sha256(p26_ingester_bytes) == EXPECTED_P26_INGESTION["ingester_source_sha256"]
            and p26_ingester_test_bytes is not None
            and _sha256(p26_ingester_test_bytes) == EXPECTED_P26_INGESTION["ingester_test_sha256"]
        ),
        "p27_specific_sanitizer_finalized_and_exact": (
            EXPECTED_P27_SANITIZER_SHA256 != "PENDING_REVIEWED_P27_SANITIZER_SHA256"
            and SHA256_RE.fullmatch(EXPECTED_P27_SANITIZER_SHA256) is not None
            and _exact(dict(p27_sanitization), EXPECTED_P27_SANITIZATION)
            and p27_sanitization.get("placeholder_must_be_replaced_before_execution") is False
            and p27_sanitizer_bytes is not None
            and _sha256(p27_sanitizer_bytes) == EXPECTED_P27_SANITIZER_SHA256
            and p27_sanitizer_test_bytes is not None
            and _sha256(p27_sanitizer_test_bytes) == EXPECTED_P27_SANITIZATION["test_sha256"]
        ),
        "fresh_attempt_and_future_acquisition_bar_exact": _exact(
            dict(attempt), EXPECTED_FRESH_ATTEMPT
        ),
        "fresh_container_and_locked_base_runtime_exact": _exact(dict(runtime), EXPECTED_RUNTIME),
        "two_phase_source_then_runtime_freeze_exact": _exact(
            dict(two_phase), EXPECTED_TWO_PHASE_FREEZE
        ),
        "four_no_training_diagnostic_stages_exact": _exact(
            contract.get("diagnostic_stages"), EXPECTED_STAGES
        ),
        "structured_mapping_schema_exact": _exact(
            contract.get("mapping_record_schema"), EXPECTED_MAPPING_SCHEMA
        ),
        "map_files_and_fd_probe_contract_exact": _exact(
            contract.get("probe_contract"), EXPECTED_PROBE_CONTRACT
        ),
        "no_training_boundary_exact": _exact(
            contract.get("no_training_boundary"), EXPECTED_NO_TRAINING
        ),
        "one_shot_execution_order_exact": _exact(
            contract.get("execution_order"), EXPECTED_EXECUTION_ORDER
        ),
        "classification_routing_exact": _exact(contract.get("terminal_routing"), EXPECTED_ROUTING),
        "forbidden_changes_exact": _exact(contract.get("forbidden_changes"), EXPECTED_FORBIDDEN),
        "no_observed_result_or_training_fields_present": (
            FORBIDDEN_RESULT_KEYS.isdisjoint(_all_mapping_keys(raw))
        ),
    }
    return {
        "schema_version": ("passive-muon-p27-cuda-deleted-mapping-localization-reconstruction-v1"),
        "canonical_sha256": _sha256(canonical_bytes),
        "checks": checks,
        "internally_consistent": all(checks.values()),
        "claim_boundary": (
            "Independent static reconstruction of P27's pre-localization, no-training "
            "contract; not evidence that a P27 container was created, a mapping was observed, "
            "CUDA repeatability or fidelity was measured, a shield ran, or training occurred."
        ),
    }


def main() -> int:
    result = reconstruct(_parse_args().canonical.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["internally_consistent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
