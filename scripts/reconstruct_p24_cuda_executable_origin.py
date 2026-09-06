#!/usr/bin/env python3
"""Independently reconstruct the frozen, pre-result P24 static contract."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from collections.abc import Mapping
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANONICAL = ROOT / "experiments/training/p24_cuda_executable_origin_contract.json"
EXPECTED_CANONICAL_SHA256 = "bdb2d3aad7d70d0ed5c6dddcd03642c9384d8b67d6f5ca507987d28e122dd446"
EXPECTED_SCHEMA = "passive-muon-p24-cuda-executable-origin-contract-v1"
EXPECTED_STATUS = "frozen_pre_remediation_build_and_pre_acquisition"
PARENT_COMMIT = "3a1454e29edc022e92f2848e216c790d4ea8434a"
PARENT_TREE = "1c416653c568eb1f883b3b7ec1b5c760b8ace878"
PARENT_TAG = "p23-deterministic-cuda-shadow-trace-diagnostic"
PARENT_TAG_OBJECT = "1bd95bc7ba4aca6522e1944589ff9afe14618c85"
OUTCOME_PATH = "results/summaries/p23_cuda_shadow_trace_outcome.json"
OUTCOME_SHA256 = "03a68af24ca705add13f11d7574f774841e2a1602edfd04304b20f9bc2532336"
RUNNER_PATH = ROOT / "experiments/training/run_p24_executable_origin_diagnostic.py"
SANITIZER_PATH = ROOT / "scripts/sanitize_p24_executable_origin_diagnostic.py"
HELPER_PATH = ROOT / "experiments/training/prepare_p24_immutable_remote_module.py"
DOCKERFILE_PATH = ROOT / "experiments/training/p24_runtime.Dockerfile"
EXPECTED_GENERATED_DIRECTORY = (
    "/opt/p23-venv/lib/python3.12/site-packages/torch/_p24_generated_remote_modules"
)
EXPECTED_GENERATED_PATH = f"{EXPECTED_GENERATED_DIRECTORY}/_remote_module_non_scriptable.py"
EXPECTED_SOURCE_HASHES = {
    "instantiator_before_sha256": (
        "567d1314ee27ff0b3bd22e7c4d1157246469de25e7a3183d96debe167b193615"
    ),
    "instantiator_after_sha256": (
        "1ecc8ae1ac4a517511914fff6e01c78ed3ad828bbe8a07b02079c206ee56bceb"
    ),
    "remote_module_sha256": ("f9bb2f5c5438791581d399e38a27606e123bdbeb3c6cb53683318a06060439c1"),
    "remote_module_template_sha256": (
        "0ff1856bbd031b5298d46c06c0502abc20bd804f42c1949ed4127e8c773660cc"
    ),
    "generated_module_sha256": ("8205b16956fb264841ecd8644784a0d157f87df79b17c16825dc1163433ce5d8"),
}
EXPECTED_UNCHANGED_AUTHORITIES = {
    "p21_fidelity_gates": {
        "path": "experiments/training/p21_shadow_trace_protocol.json",
        "sha256": "ca40ef0aef676971ca4464c3f838c6bd956ecc9806c2023af0d4ea699375315a",
        "mutation_allowed": False,
    },
    "p22_protocol": {
        "path": "experiments/training/p22_real_gradient_shadow_trace_protocol.json",
        "sha256": "f88eda60b366b561d277e41335b02c3fb9861a2604a947f8dd5c28fb54bedeac",
        "mutation_allowed": False,
    },
    "p22_erratum": {
        "path": "theory/p22_real_gradient_shadow_trace_protocol_erratum.md",
        "sha256": "42dc57bd8fb03b619c0de75c58570991ca520131ab7d1660b4f9373fd756a2cd",
        "mutation_allowed": False,
    },
    "p22_fused_qkv_certificate": {
        "path": "results/summaries/p22_scalable_shield_shape_extension_certificate.json",
        "sha256": "9480525384a57bb09869a55c878251f44f170930f3050a1ad50b4d934dcb7450",
        "mutation_allowed": False,
    },
    "p23_addendum": {
        "path": "experiments/training/p23_deterministic_cuda_shadow_trace_addendum.json",
        "sha256": "ad29a15e3b027250a35b6c3af6ce214cc98d664cad7d9821730e7e0cdaab9e3a",
        "mutation_allowed": False,
    },
}
EXPECTED_EXECUTION_SOURCES = {
    "native_executable_origin_diagnostic": {
        "path": "experiments/training/run_p24_executable_origin_diagnostic.py",
        "sha256": "5e3fe58a394ad98ab8f2d03f5992cabeb417110b7d2b05e86fbbf61b5f241afb",
        "mutation_allowed_after_freeze": False,
    },
    "native_executable_origin_sanitizer": {
        "path": "scripts/sanitize_p24_executable_origin_diagnostic.py",
        "sha256": "b432e06f899c85a8bcbcfa584f91e6ae8472902389a89aafc07c2840a31f7616",
        "mutation_allowed_after_freeze": False,
    },
    "cuda_acquisition_runner": {
        "path": "experiments/training/run_p23_deterministic_cuda_shadow_trace.py",
        "sha256": "a2b4bb5b686b7f681958d09be1d465917b40e34d45e4d3503efef0f35e7ae8cb",
        "mutation_allowed_after_freeze": False,
    },
    "stored_computation_and_shadow_semantics": {
        "path": "src/passive_muon/p21_shadow_trace.py",
        "sha256": "020debed7a12d6f538ef9093fcdfbcafbf571ea162fb868ccf0a49b5bebb0825",
        "mutation_allowed_after_freeze": False,
    },
}
EXPECTED_TOP_LEVEL_KEYS = {
    "schema_version",
    "status",
    "authority",
    "provenance_parent",
    "unchanged_authorities",
    "execution_sources",
    "frozen_p23_gates",
    "p23_blocker",
    "remediation_design",
    "native_diagnostic_contract",
    "replacement_runtime",
    "ordered_execution",
    "failure_policy",
    "claim_boundary",
}
EXPECTED_SUCCESS_FIELDS = [
    "schema_version",
    "mode",
    "status",
    "scope",
    "seed",
    "seed_semantics",
    "process",
    "repository",
    "artifacts",
    "contract_binding",
    "runtime_lock_binding",
    "host_attestation_binding",
    "live_runtime",
    "pinned_sources",
    "trigger",
    "file_backed_module_closure",
    "checks",
    "claim_boundary",
    "passes",
]
EXPECTED_ERROR_FIELDS = [
    "schema_version",
    "mode",
    "status",
    "scope",
    "seed",
    "seed_semantics",
    "process",
    "repository",
    "artifacts",
    "contract_binding",
    "runtime_lock_binding",
    "host_attestation_binding",
    "live_runtime",
    "pinned_sources",
    "trigger",
    "file_backed_module_closure",
    "checks",
    "error",
    "claim_boundary",
    "passes",
]
EXPECTED_SANITIZER_CONTRACT = {
    "runner": "scripts/sanitize_p24_executable_origin_diagnostic.py",
    "schema_version": "passive-muon-p24-sanitized-executable-origin-diagnostic-v1",
    "required_logical_roots": [
        "data",
        "muon",
        "nanogpt",
        "native",
        "preprocessor_alias",
        "python_environment",
        "python_standard_library",
        "repository",
        "temporary",
    ],
    "native_sha256_and_byte_count_required": True,
    "full_semantic_manifest_retained": True,
    "exact_logical_root_substitution": True,
    "native_and_sanitized_artifacts_retained": True,
    "overwrite_allowed": False,
    "applies_to_statuses": ["passes", "fails", "error"],
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    return parser.parse_args()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _exact_value(observed: object, expected: object) -> bool:
    """Compare recursively without accepting bool as an integer."""

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


@cache
def _git(*arguments: str) -> str | None:
    try:
        return (
            subprocess.run(
                ["git", *arguments],
                cwd=ROOT,
                check=True,
                capture_output=True,
            )
            .stdout.decode()
            .strip()
        )
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError):
        return None


@cache
def _git_blob(commit: str, path: str) -> bytes | None:
    try:
        return subprocess.run(
            ["git", "show", f"{commit}:{path}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def _literal_string_constants(path: Path) -> dict[str, str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeError):
        return {}
    result: dict[str, str] = {}
    for node in tree.body:
        target: ast.expr | None = None
        value: ast.expr | None = None
        if isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        if isinstance(target, ast.Name) and value is not None:
            try:
                literal = ast.literal_eval(value)
            except (ValueError, TypeError):
                continue
            if isinstance(literal, str):
                result[target.id] = literal
    return result


def reconstruct(canonical: Path) -> dict[str, object]:
    """Reconstruct the immutable P24 preconditions; do not infer a CUDA result."""

    try:
        canonical_bytes = canonical.read_bytes()
        raw_contract = json.loads(canonical_bytes)
    except (OSError, json.JSONDecodeError, UnicodeError):
        canonical_bytes = b""
        raw_contract = None
    contract = _mapping(raw_contract)
    parent = _mapping(contract.get("provenance_parent"))
    outcome = _mapping(parent.get("outcome"))
    authorities = _mapping(contract.get("unchanged_authorities"))
    execution_sources = _mapping(contract.get("execution_sources"))
    gates = _mapping(contract.get("frozen_p23_gates"))
    blocker = _mapping(contract.get("p23_blocker"))
    design = _mapping(contract.get("remediation_design"))
    source_hashes = _mapping(design.get("source_hashes"))
    source_byte_counts = _mapping(design.get("source_byte_counts"))
    build_patch = _mapping(design.get("build_patch"))
    image_recipe = _mapping(design.get("image_recipe"))
    runtime_policy = _mapping(design.get("runtime_policy"))
    diagnostic = _mapping(contract.get("native_diagnostic_contract"))
    sanitizer = _mapping(diagnostic.get("offline_sanitizer"))
    replacement = _mapping(contract.get("replacement_runtime"))
    failure = _mapping(contract.get("failure_policy"))
    boundary = _mapping(contract.get("claim_boundary"))

    runner_constants = _literal_string_constants(RUNNER_PATH)
    sanitizer_constants = _literal_string_constants(SANITIZER_PATH)
    helper_constants = _literal_string_constants(HELPER_PATH)
    parent_outcome = _git_blob(PARENT_COMMIT, OUTCOME_PATH)
    authority_hashes: dict[str, str | None] = {}
    for name, expected_record in EXPECTED_UNCHANGED_AUTHORITIES.items():
        path = ROOT / expected_record["path"]
        authority_hashes[name] = _sha256(path.read_bytes()) if path.is_file() else None
    execution_hashes: dict[str, str | None] = {}
    for name, expected_record in EXPECTED_EXECUTION_SOURCES.items():
        path = ROOT / expected_record["path"]
        execution_hashes[name] = _sha256(path.read_bytes()) if path.is_file() else None

    expected_gates = {
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
    expected_replacement = {
        "new_oci_digest_required": True,
        "new_running_container_required": True,
        "new_runtime_lock_required": True,
        "new_host_attestation_required": True,
        "old_p23_image_allowed_for_acquisition": False,
        "old_p23_runtime_lock_allowed_for_acquisition": False,
        "old_p23_host_attestation_allowed_for_acquisition": False,
        "lock_review_and_commit_before_remediated_diagnostic": True,
        "lock_review_and_commit_before_trace_off_a": True,
        "diagnostic_to_acquisition_commit_bridge_required": True,
        "bridge_delta_scope": (
            "only the two sanitized diagnostic wrappers and explicit result-binding metadata; "
            "every frozen execution-source hash remains unchanged"
        ),
        "remediated_diagnostic_rerun_after_bridge_allowed": False,
        "root_filesystem_read_only": True,
        "network_mode": "none",
        "gpu_contract": (
            "one pinned full-GPU BF16 CUDA device; no CPU, MPS, MIG, or ordinal substitution"
        ),
        "mount_contract": (
            "unchanged P23 destination and mode allowlist; only the immutable image contents differ"
        ),
    }

    checks = {
        "canonical_bytes_match_frozen_sha256": (
            _sha256(canonical_bytes) == EXPECTED_CANONICAL_SHA256
        ),
        "contract_is_object_with_exact_top_level_keys": (
            isinstance(raw_contract, Mapping) and set(raw_contract) == EXPECTED_TOP_LEVEL_KEYS
        ),
        "schema_status_and_pre_result_authority_exact": (
            contract.get("schema_version") == EXPECTED_SCHEMA
            and contract.get("status") == EXPECTED_STATUS
            and contract.get("authority")
            == (
                "This contract governs P24 image remediation, native executable-origin "
                "diagnostics, runtime replacement, and the unchanged P23 CUDA acquisition. "
                "It records no P24 CUDA result."
            )
        ),
        "annotated_parent_tag_and_commit_exact": (
            parent.get("branch") == "p23-deterministic-cuda-shadow-trace"
            and parent.get("outcome_commit") == PARENT_COMMIT
            and parent.get("outcome_tree") == PARENT_TREE
            and parent.get("outcome_tag") == PARENT_TAG
            and parent.get("annotated_tag_object") == PARENT_TAG_OBJECT
            and parent.get("tag_target_commit") == PARENT_COMMIT
            and _git("rev-parse", PARENT_COMMIT) == PARENT_COMMIT
            and _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") == PARENT_TREE
            and _git("rev-parse", PARENT_TAG) == PARENT_TAG_OBJECT
            and _git("cat-file", "-t", PARENT_TAG) == "tag"
            and _git("rev-parse", f"{PARENT_TAG}^{{}}") == PARENT_COMMIT
        ),
        "p23_outcome_blob_and_scope_exact": (
            outcome.get("path") == OUTCOME_PATH
            and outcome.get("sha256") == OUTCOME_SHA256
            and outcome.get("evidence_scope")
            == "operator-recorded pretraining stop; not a native failure manifest"
            and parent_outcome is not None
            and _sha256(parent_outcome) == OUTCOME_SHA256
        ),
        "unchanged_authority_records_exact": _exact_value(
            dict(authorities), EXPECTED_UNCHANGED_AUTHORITIES
        ),
        "unchanged_authority_bytes_reconstruct": all(
            authority_hashes.get(name) == record["sha256"]
            for name, record in EXPECTED_UNCHANGED_AUTHORITIES.items()
        ),
        "execution_source_records_and_bytes_exact": (
            _exact_value(dict(execution_sources), EXPECTED_EXECUTION_SOURCES)
            and all(
                execution_hashes.get(name) == record["sha256"]
                for name, record in EXPECTED_EXECUTION_SOURCES.items()
            )
        ),
        "p23_gates_are_exact_and_unchanged": _exact_value(dict(gates), expected_gates),
        "p23_blocker_preserves_evidence_boundary": (
            blocker.get("classification") == "executable_origin_on_writable_tmpfs_before_step_zero"
            and blocker.get("native_failure_manifest_present") is False
            and blocker.get("failure_phase")
            == "initialized_loaded_file_closure_before_initial_state_hash_and_step_zero"
            and isinstance(blocker.get("not_exonerated"), list)
            and len(blocker.get("not_exonerated", [])) == 5
        ),
        "source_hashes_match_diagnostic_runner": (
            _exact_value(dict(source_hashes), EXPECTED_SOURCE_HASHES)
            and runner_constants.get("ORIGINAL_INSTANTIATOR_SHA256")
            == EXPECTED_SOURCE_HASHES["instantiator_before_sha256"]
            and runner_constants.get("PATCHED_INSTANTIATOR_SHA256")
            == EXPECTED_SOURCE_HASHES["instantiator_after_sha256"]
            and runner_constants.get("GENERATED_SHA256")
            == EXPECTED_SOURCE_HASHES["generated_module_sha256"]
            and runner_constants.get("REMOTE_MODULE_SHA256")
            == EXPECTED_SOURCE_HASHES["remote_module_sha256"]
            and runner_constants.get("TEMPLATE_SHA256")
            == EXPECTED_SOURCE_HASHES["remote_module_template_sha256"]
            and runner_constants.get("FIXED_GENERATED_PATH") == EXPECTED_GENERATED_PATH
        ),
        "source_hashes_match_build_helper": (
            helper_constants.get("ORIGINAL_INSTANTIATOR_SHA256")
            == EXPECTED_SOURCE_HASHES["instantiator_before_sha256"]
            and helper_constants.get("PATCHED_INSTANTIATOR_SHA256")
            == EXPECTED_SOURCE_HASHES["instantiator_after_sha256"]
            and helper_constants.get("GENERATED_MODULE_SHA256")
            == EXPECTED_SOURCE_HASHES["generated_module_sha256"]
            and helper_constants.get("REMOTE_MODULE_SHA256")
            == EXPECTED_SOURCE_HASHES["remote_module_sha256"]
            and helper_constants.get("TEMPLATE_SHA256")
            == EXPECTED_SOURCE_HASHES["remote_module_template_sha256"]
            and _exact_value(
                dict(source_byte_counts),
                {
                    "instantiator_before": 5510,
                    "instantiator_after": 6438,
                    "remote_module": 31251,
                    "remote_module_template": 3463,
                    "generated_module": 2355,
                },
            )
        ),
        "immutable_generated_path_exact": (
            design.get("generated_directory") == EXPECTED_GENERATED_DIRECTORY
            and design.get("generated_module") == EXPECTED_GENERATED_PATH
            and design.get("generated_directory_mode") == "0555"
            and design.get("generated_module_mode") == "0444"
            and design.get("image_manifest") == "/opt/p24-immutable-remote-module-manifest.json"
            and build_patch.get("path")
            == "experiments/training/prepare_p24_immutable_remote_module.py"
            and build_patch.get("sha256") == _sha256(HELPER_PATH.read_bytes())
            and build_patch.get("sha256")
            == "97a4a730aadd11de1aa1b23107cef53f95c3348bde905932571eb6a68d512bef"
            and build_patch.get("mutation_allowed_after_freeze") is False
            and build_patch.get("fail_on_preimage_mismatch") is True
            and build_patch.get("fail_on_generated_content_mismatch") is True
            and build_patch.get("fail_on_postimage_mismatch") is True
        ),
        "image_recipe_and_parent_digest_exact": (
            image_recipe.get("path") == "experiments/training/p24_runtime.Dockerfile"
            and image_recipe.get("sha256") == _sha256(DOCKERFILE_PATH.read_bytes())
            and image_recipe.get("sha256")
            == "c1ccf0783b10f9e79795fa575398a58a7e6d7a9f2abe47a3a0650b213978e2f9"
            and image_recipe.get("base_image")
            == (
                "localhost:5000/p23-runtime@"
                "sha256:44ef23717780b1cbf112b183e7988b1319ddfed6b1d224efaa33e1e6d96de4c1"
            )
            and image_recipe.get("mutation_allowed_after_freeze") is False
        ),
        "writable_origin_rejection_is_not_weakened": (
            runtime_policy.get("pre_generated_at_image_build") is True
            and runtime_policy.get("exact_existing_content_required") is True
            and runtime_policy.get("generated_module_loaded_by_exact_path_spec") is True
            and runtime_policy.get("generated_module_sys_path_resolution_allowed") is False
            and runtime_policy.get("loaded_module_file_must_equal_exact_generated_path") is True
            and runtime_policy.get("generated_directory_writable_in_acquisition") is False
            and runtime_policy.get("writable_executable_origin_rejection_unchanged") is True
            and runtime_policy.get("tmp_allowlist_added") is False
            and runtime_policy.get("dev_shm_allowlist_added") is False
            and runtime_policy.get("tmp_remains_writable_noexec_tmpfs") is True
        ),
        "native_diagnostic_schema_and_modes_exact": (
            diagnostic.get("runner")
            == "experiments/training/run_p24_executable_origin_diagnostic.py"
            and diagnostic.get("schema_version")
            == "passive-muon-p24-executable-origin-diagnostic-v1"
            and _exact_value(diagnostic.get("modes"), ["baseline", "remediated"])
            and _exact_value(diagnostic.get("status_values"), ["passes", "fails", "error"])
            and _exact_value(diagnostic.get("success_top_level_fields"), EXPECTED_SUCCESS_FIELDS)
            and _exact_value(diagnostic.get("error_top_level_fields"), EXPECTED_ERROR_FIELDS)
            and _exact_value(dict(sanitizer), EXPECTED_SANITIZER_CONTRACT)
            and runner_constants.get("SCHEMA") == "passive-muon-p24-executable-origin-diagnostic-v1"
            and sanitizer_constants.get("SANITIZED_SCHEMA")
            == "passive-muon-p24-sanitized-executable-origin-diagnostic-v1"
            and diagnostic.get("diagnostic_is_not_gradient_acquisition") is True
        ),
        "replacement_runtime_is_exact_and_mandatory": _exact_value(
            dict(replacement), expected_replacement
        ),
        "execution_order_preserves_mechanical_gates": (
            _exact_value(
                contract.get("ordered_execution"),
                [
                    "run and retain the native baseline executable-origin diagnostic in the "
                    "old P23 image",
                    "independently compute the baseline native-byte SHA-256, sanitize offline "
                    "against that digest, and retain both artifacts",
                    "build the remediation image from the exact preimage and verify all three "
                    "source hashes",
                    "launch a new read-only-root container with the unchanged P23 mount and GPU "
                    "contract",
                    "freeze and sanitize a new runtime lock and host attestation",
                    "review and commit the new image digest, runtime lock, host attestation, and "
                    "complete source tree",
                    "run and retain the native remediated executable-origin diagnostic against "
                    "that exact committed lock and tree",
                    "independently compute the remediated native-byte SHA-256, sanitize offline "
                    "against that digest, and retain both artifacts",
                    "review and commit both diagnostic bindings and sanitized copies before "
                    "acquisition",
                    "trace_off_a",
                    "trace_off_b",
                    "verify_repeatability",
                    "trace_on only after exact repeatability passes",
                    "verify_noninterference",
                    "aggregate_fidelity only after exact repeatability and noninterference pass",
                    "sanitize every P23 acquisition artifact with the unchanged P23 sanitizer "
                    "and retain every native artifact and hash",
                ],
            )
        ),
        "failure_policy_is_fail_closed": (
            failure.get("no_favorable_reruns") is True
            and failure.get("native_acquisition_failure_manifest_schema")
            == "passive-muon-p23-native-failure-manifest-v1"
            and failure.get("failure_output_required_for_every_acquisition_run") is True
            and failure.get("failure_output_must_be_fresh_external_and_nonaliasing") is True
            and failure.get("trace_off_repeatability_failure")
            == "trace_on remains mechanically barred"
            and failure.get("runtime_big_integer_or_exact_fraction_check_added") is False
        ),
        "claim_boundary_is_pre_result_only": (
            boundary.get("pre_acquisition_contract_only") is True
            and boundary.get("p24_cuda_result_present") is False
            and isinstance(boundary.get("not_claimed"), list)
            and "real-gradient fidelity" in boundary.get("not_claimed", [])
            and "native CUDA sector-shield correctness" in boundary.get("not_claimed", [])
        ),
    }
    internally_consistent = all(checks.values())
    return {
        "schema_version": "passive-muon-p24-cuda-executable-origin-reconstruction-v1",
        "canonical_sha256": _sha256(canonical_bytes),
        "parent_commit": PARENT_COMMIT,
        "parent_tree": PARENT_TREE,
        "checks": checks,
        "internally_consistent": internally_consistent,
        "scope": (
            "standard-library reconstruction of the frozen P24 static preconditions; no native "
            "diagnostic, CUDA acquisition, gradient, candidate, fidelity, or training result"
        ),
    }


def main() -> int:
    arguments = _parse_args()
    result = reconstruct(arguments.canonical.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["internally_consistent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
