from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
from collections.abc import Iterator
from pathlib import Path
from typing import TypeAlias

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/reconstruct_p27_cuda_deleted_mapping_localization.py"
CONTRACT = ROOT / "experiments/training/p27_cuda_deleted_mapping_localization_contract.json"
LOCALIZER = ROOT / "experiments/training/run_p27_cuda_deleted_mapping_localization.py"
P26_OUTCOME = ROOT / "results/summaries/p26_permission_safe_cuda_acquisition_outcome.json"
PathPart: TypeAlias = str | int


def _module():
    specification = importlib.util.spec_from_file_location("p27_contract_reconstruction", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _leaf_paths(value: object, prefix: tuple[PathPart, ...] = ()) -> Iterator[tuple[PathPart, ...]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _leaf_paths(item, (*prefix, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _leaf_paths(item, (*prefix, index))
    else:
        yield prefix


def _mapping_paths(
    value: object, prefix: tuple[PathPart, ...] = ()
) -> Iterator[tuple[PathPart, ...]]:
    if isinstance(value, dict):
        yield prefix
        for key, item in value.items():
            yield from _mapping_paths(item, (*prefix, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _mapping_paths(item, (*prefix, index))


def _at_path(value: object, path: tuple[PathPart, ...]) -> object:
    current = value
    for part in path:
        current = current[part]  # type: ignore[index]
    return current


def _replace_at_path(value: object, path: tuple[PathPart, ...], replacement: object) -> None:
    assert path
    parent = _at_path(value, path[:-1])
    parent[path[-1]] = replacement  # type: ignore[index]


def _mutated_leaf(value: object) -> object:
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, str):
        return f"{value}__mutated"
    raise AssertionError(f"unsupported leaf type: {type(value).__name__}")


def test_p27_contract_reconstructs_independently() -> None:
    result = _module().reconstruct(CONTRACT)

    assert result["internally_consistent"] is True
    assert len(result["checks"]) == 23
    assert all(result["checks"].values())
    assert result["canonical_sha256"] == hashlib.sha256(CONTRACT.read_bytes()).hexdigest()
    assert result["claim_boundary"] == (
        "Independent static reconstruction of P27's pre-localization, no-training "
        "contract; not evidence that a P27 container was created, a mapping was observed, "
        "CUDA repeatability or fidelity was measured, a shield ran, or training occurred."
    )


def test_terminal_p26_outcome_commit_and_annotated_tag_are_bound() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    parent = contract["terminal_parent"]
    outcome = json.loads(P26_OUTCOME.read_text(encoding="utf-8"))

    assert parent["outcome_sha256"] == hashlib.sha256(P26_OUTCOME.read_bytes()).hexdigest()
    assert parent["outcome_commit"] == "5429da23ff18888daa2312c530a4587780484d8b"
    assert parent["outcome_tree"] == "a5b42f820beb7b4be0b65d89c51d49c1a5436647"
    assert parent["checkpoint_tag"] == "p26-permission-safe-acquisition-checkpoint"
    assert parent["checkpoint_tag_object"] == "bacad707d3779bfa10957e18cb4c69b1a7f0cbce"
    assert parent["checkpoint_peeled_commit"] == parent["outcome_commit"]
    assert parent["attempt_id"] == outcome["attempt_id"] == "20260906-02"
    assert parent["attempt_is_terminal"] is True
    assert parent["attempt_reuse_allowed"] is False
    assert parent["candidate_observations"] == 0
    assert outcome["terminal_policy"]["retry_until_favorable_allowed"] is False
    assert outcome["route"]["next_branch"] == "p27-cuda-deleted-mapping-localization"


def test_sanitizer_correction_is_narrow_and_hash_bound() -> None:
    module = _module()
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    correction = contract["sanitizer_only_correction"]

    assert correction == module.EXPECTED_SANITIZER_CORRECTION
    assert correction["corrected_sha256"] == (
        "ec1037839a3de378ee7391ae815ffc29c184865c5216d30027edc460ba112fab"
    )
    assert correction["acquisition_policy_changed"] is False
    assert correction["deleted_mapping_policy_changed"] is False
    assert correction["scientific_gate_changed"] is False
    assert (
        "exact already-validated PINNED_PYTHON_EXECUTABLE"
        in (correction["permitted_semantic_change"])
    )
    assert "Unrelated symlink escapes" in correction["required_negative_control"]


def test_every_unchanged_scientific_authority_matches_live_bytes() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    authorities = contract["unchanged_scientific_authorities"]

    assert len(authorities) == 8
    for record in authorities.values():
        path = ROOT / record["path"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]


def test_localizer_is_finalized_hash_bound_and_contains_no_training_call() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    source = contract["localization_source"]

    assert source["placeholder_must_be_replaced_before_execution"] is False
    assert len(source["sha256"]) == 64
    assert hashlib.sha256(LOCALIZER.read_bytes()).hexdigest() == source["sha256"]
    assert (
        hashlib.sha256((ROOT / source["test_path"]).read_bytes()).hexdigest()
        == (source["test_sha256"])
    )
    assert source["in_container_evidence_path"] == (
        "/workspace/evidence/p23/run_p27_cuda_deleted_mapping_localization.py"
    )
    assert source["native_output_path"] == (
        "/workspace/evidence/p23/p27_cuda_deleted_mapping_localization.native.json"
    )
    assert source["extra_mount_allowed"] is False
    assert source["authority_repository_head"] == ("185e444afc0b44ca0a09b1bde49a6b6fa3973355")
    assert source["authority_repository_tree"] == ("24f4bdac331a57bd7c1b807747d7c7fba253ee5a")
    assert source["p25_source_sha256"] == (
        "6305fb9683503eb67e091cdfb0a1105628fcb4bec76fe5ef7fd23dd49ba9d770"
    )
    assert source["p25_contract_sha256"] == (
        "51e1fc22f685cc904fcdfff2775cff4293fdaa67a6e048072ee4dc49777f34e2"
    )

    parsed = ast.parse(LOCALIZER.read_text(encoding="utf-8"))
    forbidden_attributes = {"backward", "step"}
    observed_attributes = {
        node.attr for node in ast.walk(parsed) if isinstance(node, ast.Attribute)
    }
    assert forbidden_attributes.isdisjoint(observed_attributes)


def test_native_p26_ingestion_and_p27_sanitization_are_separate() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    ingestion = contract["p26_failure_ingestion"]
    sanitizer = contract["p27_specific_sanitization"]

    assert ingestion["native_sha256"] == (
        "b2a92b5062988534bc424d3ecadf3977ecedc0f80aa27628d4a0457858a64c91"
    )
    assert ingestion["native_owner"] == "root:root"
    assert ingestion["native_mode"] == "0600"
    assert ingestion["read_native_exactly_once"] is True
    assert ingestion["mutate_native_allowed"] is False
    assert ingestion["sanitizer_source_sha256"] == (
        "ec1037839a3de378ee7391ae815ffc29c184865c5216d30027edc460ba112fab"
    )
    assert len(ingestion["ingester_source_sha256"]) == 64
    assert len(ingestion["ingester_test_sha256"]) == 64
    assert (
        hashlib.sha256((ROOT / ingestion["ingester_source_path"]).read_bytes()).hexdigest()
        == (ingestion["ingester_source_sha256"])
    )
    assert (
        hashlib.sha256((ROOT / ingestion["ingester_test_path"]).read_bytes()).hexdigest()
        == (ingestion["ingester_test_sha256"])
    )
    assert ingestion["outputs_must_be_fresh_under_p27_evidence"] is True
    assert ingestion["output_in_terminal_p26_directory_allowed"] is False

    assert sanitizer["placeholder_must_be_replaced_before_execution"] is False
    sanitizer_path = ROOT / sanitizer["source_path"]
    assert hashlib.sha256(sanitizer_path.read_bytes()).hexdigest() == sanitizer["sha256"]
    assert (
        hashlib.sha256((ROOT / sanitizer["test_path"]).read_bytes()).hexdigest()
        == (sanitizer["test_sha256"])
    )
    assert sanitizer["execution_location"] == "host-side root bridge"
    assert sanitizer["interpreter"] == "/usr/bin/python3"
    assert sanitizer["container_execution_allowed"] is False
    assert sanitizer["native_output_external_only"] is True
    assert sanitizer["wrapper_must_bind_native_sha256_byte_count_mode_uid_gid"] is True
    assert sanitizer["preserve_all_nonpath_evidence"] is True
    assert "publish no native absolute-path bytes" in sanitizer["path_transform"]
    assert sanitizer["duplicate_key_rejection"] is True
    assert sanitizer["nonfinite_rejection"] is True
    assert sanitizer["unknown_schema_rejection"] is True
    assert sanitizer["unknown_absolute_path_rejection"] is True


def test_host_orchestration_is_two_phase_one_shot_and_hash_bound() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    orchestration = contract["host_orchestration"]

    assert orchestration["placeholders_must_be_replaced_before_execution"] is False
    for path_key, hash_key in (
        ("source_path", "source_sha256"),
        ("test_path", "test_sha256"),
    ):
        source = ROOT / orchestration[path_key]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == orchestration[hash_key]
    assert orchestration["allowed_phases"] == ["prepare-runtime", "run-localization"]
    assert orchestration["required_environment"] == [
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
    ]
    assert orchestration["source_authorities_are_read_from_validated_contract"] is True
    assert orchestration["prepare_runtime_invocations_allowed"] == 1
    assert orchestration["localization_invocations_allowed"] == 1
    assert orchestration["cleanup_or_retry_phase_exists"] is False
    assert orchestration["exact_bind_mount_count"] == 10
    assert orchestration["authority_checkout_is_only_repository_mount"] is True
    assert orchestration["localizer_enters_through_existing_evidence_mount"] is True
    assert orchestration["phase_two_review_required_before_localization"] is True


def test_fresh_container_is_distinct_and_scientific_acquisition_remains_barred() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    attempt = contract["fresh_localization_attempt"]
    runtime = contract["locked_base_runtime"]

    assert attempt["attempt_root"] == "/secure/p27/attempt-20260906-01"
    assert attempt["container_name"] == "p27-localization-20260906-01"
    assert attempt["root_must_not_preexist"] is True
    assert attempt["new_container_required"] is True
    assert attempt["terminal_p26_container_reuse_allowed"] is False
    assert attempt["localization_invocations_allowed"] == 1
    assert attempt["retry_until_favorable_allowed"] is False
    assert attempt["future_scientific_acquisition_attempt_id"] == "20260906-03"
    assert attempt["future_scientific_acquisition_is_authorized"] is False
    assert runtime["prior_container_id_must_not_be_reused"] == (
        "e6682d8f09b9dc4be342354d520a8f6f8766a8e232840d1d7beb5e00e1fced6c"
    )
    assert runtime["image_digest"] == (
        "sha256:14dafde07ae578cc4725f452a51d4bc4c920b69f6af23125a72e27162e193ec0"
    )
    assert runtime["exact_bind_mount_count"] == 10
    assert runtime["authority_checkout_is_repository_mount"] is True
    assert runtime["diagnostic_uses_existing_native_evidence_mount"] is True
    assert runtime["extra_mounts_allowed"] is False


def test_two_phase_freeze_requires_direct_child_runtime_only_commit() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    freeze = contract["two_phase_freeze"]

    assert freeze["phase_one_commit_is_selected_after_commit"] is True
    assert freeze["phase_two_exact_delta"] == [
        "A\texperiments/training/p27_cuda_runtime_lock.json",
        "A\texperiments/training/p27_host_attestation.json",
    ]
    assert freeze["phase_two_localization_started"] is False
    assert freeze["phase_two_candidate_observations"] == 0
    assert freeze["diagnostic_requires_reviewed_phase_two_commit"] is True
    assert freeze["authority_repository_remains_185e444_mount"] is True


def test_exact_four_stage_order_and_structured_mapping_fields() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    assert [stage["name"] for stage in contract["diagnostic_stages"]] == [
        "pre_cuda",
        "post_cuda_init",
        "post_model_move",
        "post_optimizer",
    ]
    mapping = contract["mapping_record_schema"]
    assert "raw_line_sha256" in mapping["mapping_fields"]
    assert "pathname_bytes_hex" in mapping["mapping_fields"]
    assert "map_files_probe" in mapping["mapping_fields"]
    assert "matching_fd_probes" in mapping["mapping_fields"]
    assert mapping["all_deleted_mappings_must_be_retained"] is True
    assert mapping["path_bytes_must_not_be_decoded_lossily"] is True
    assert "deleted_regular_executable" in mapping["closed_classification_values"]
    assert "deleted_character_device_read_only" in mapping["closed_classification_values"]
    assert "deleted_other_nonregular_writable" in mapping["closed_classification_values"]
    assert "deleted_unclassifiable" in mapping["closed_classification_values"]
    assert "file_type" in contract["probe_contract"]["map_files_fields"]
    assert "file_type" in contract["probe_contract"]["fd_probe_fields"]
    assert mapping["unrelated_map_churn_is_retained_evidence_not_automatic_failure"] is True
    assert mapping["deleted_target_identity_change_or_disappearance_is_fatal"] is True
    assert "post_probe_proc_maps_sha256" in mapping["snapshot_fields"]
    assert "all_deleted_mapping_identities_stable_during_probe" in mapping["snapshot_fields"]


def test_map_files_and_fd_probe_failures_are_never_silent() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    probes = contract["probe_contract"]

    assert probes["map_files_probe_path"] == (
        "/proc/self/map_files/{address_start_hex}-{address_end_hex}"
    )
    assert probes["fd_root"] == "/proc/self/fd"
    for field in ("errno", "error_message", "device_major_minor", "inode_decimal"):
        assert field in probes["map_files_fields"]
        assert field in probes["fd_probe_fields"]
    assert probes["probe_failures_are_structured_evidence"] is True
    assert probes["probe_failure_must_not_be_silently_dropped"] is True


def test_no_training_boundary_is_closed_and_zero_observation() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    boundary = contract["no_training_boundary"]

    assert boundary["model_construction_allowed"] is True
    assert boundary["model_cuda_move_allowed"] is True
    assert boundary["optimizer_construction_allowed"] is True
    for key in (
        "data_loading_allowed",
        "forward_allowed",
        "backward_allowed",
        "optimizer_step_allowed",
        "candidate_evaluation_allowed",
    ):
        assert boundary[key] is False
    assert boundary["candidate_observation_count_required"] == 0
    assert boundary["gradient_observation_count_required"] == 0
    assert boundary["parameter_update_count_required"] == 0


def test_terminal_routing_is_classification_driven_and_policy_is_not_relaxed() -> None:
    module = _module()
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    assert contract["terminal_routing"] == module.EXPECTED_ROUTING
    assert contract["forbidden_changes"] == module.EXPECTED_FORBIDDEN
    routing = contract["terminal_routing"]
    assert routing["selection_precedence"] == [
        "sanitization_or_reconstruction_failure",
        "deleted_executable_or_writable_mapping",
        "deleted_unclassifiable_or_identity_unstable_mapping",
        "deleted_conclusively_identified_read_only_nonexecutable_mapping_set",
        "no_deleted_mapping_reproduced",
    ]
    assert set(routing["selection_precedence"]) == set(routing["routes"])
    assert routing["multiple_conditions_use_first_matching_route"] is True
    assert routing["post_route_action"] == "freeze the exact P27 outcome and selected route"
    assert routing["future_scientific_acquisition_authorized"] is False
    assert routing["future_scientific_acquisition_requires_separately_reviewed_contract"] is True
    assert any("run more than one" in item for item in contract["forbidden_changes"])
    assert any("allclose" in item for item in contract["forbidden_changes"])
    assert any("rerun until" in item for item in contract["forbidden_changes"])


def test_contract_has_no_observation_or_training_result_fields() -> None:
    result = _module().reconstruct(CONTRACT)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    assert result["checks"]["no_observed_result_or_training_fields_present"] is True
    assert contract["status"] == "frozen_pre_localization_no_training"
    assert "records no P27 runtime" in contract["claim_boundary"]
    assert (
        contract["fresh_localization_attempt"]["future_scientific_acquisition_is_authorized"]
        is False
    )


def test_every_contract_leaf_is_bound(tmp_path: Path) -> None:
    module = _module()
    canonical = json.loads(CONTRACT.read_text(encoding="utf-8"))
    mutated_path = tmp_path / "mutated-contract.json"

    for path in _leaf_paths(canonical):
        mutated = copy.deepcopy(canonical)
        _replace_at_path(mutated, path, _mutated_leaf(_at_path(mutated, path)))
        _write_json(mutated_path, mutated)
        assert module.reconstruct(mutated_path)["internally_consistent"] is False, path


def test_every_contract_mapping_rejects_missing_and_extra_keys(tmp_path: Path) -> None:
    module = _module()
    canonical = json.loads(CONTRACT.read_text(encoding="utf-8"))
    mutated_path = tmp_path / "mutated-schema.json"

    for path in _mapping_paths(canonical):
        mapping = _at_path(canonical, path)
        assert isinstance(mapping, dict) and mapping

        with_extra = copy.deepcopy(canonical)
        extra_mapping = _at_path(with_extra, path)
        assert isinstance(extra_mapping, dict)
        extra_mapping["unexpected_field"] = "unexpected"
        _write_json(mutated_path, with_extra)
        assert module.reconstruct(mutated_path)["internally_consistent"] is False, (
            "extra",
            path,
        )

        for key in mapping:
            with_missing = copy.deepcopy(canonical)
            missing_mapping = _at_path(with_missing, path)
            assert isinstance(missing_mapping, dict)
            del missing_mapping[key]
            _write_json(mutated_path, with_missing)
            assert module.reconstruct(mutated_path)["internally_consistent"] is False, (
                "missing",
                (*path, key),
            )


def test_reordered_sequences_fail_closed(tmp_path: Path) -> None:
    module = _module()
    canonical = json.loads(CONTRACT.read_text(encoding="utf-8"))
    path = tmp_path / "reordered.json"

    for field in ("diagnostic_stages", "execution_order", "forbidden_changes"):
        mutated = copy.deepcopy(canonical)
        mutated[field].reverse()
        _write_json(path, mutated)
        assert module.reconstruct(path)["internally_consistent"] is False

    mutated = copy.deepcopy(canonical)
    mutated["mapping_record_schema"]["mapping_fields"].reverse()
    _write_json(path, mutated)
    assert module.reconstruct(path)["internally_consistent"] is False

    mutated = copy.deepcopy(canonical)
    mutated["terminal_routing"]["selection_precedence"].reverse()
    _write_json(path, mutated)
    assert module.reconstruct(path)["internally_consistent"] is False


def test_duplicate_keys_nonfinite_and_noncanonical_encoding_fail_closed(tmp_path: Path) -> None:
    module = _module()

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"status":"first","status":"second"}\n', encoding="utf-8")
    assert module.reconstruct(duplicate)["internally_consistent"] is False

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"status":NaN}\n', encoding="utf-8")
    assert module.reconstruct(nonfinite)["internally_consistent"] is False

    compact = tmp_path / "compact.json"
    compact.write_text(json.dumps(json.loads(CONTRACT.read_text(encoding="utf-8"))))
    result = module.reconstruct(compact)
    assert result["internally_consistent"] is False
    assert result["checks"]["strict_canonical_json_and_closed_top_level"] is False


def test_boolean_and_integer_fields_are_not_interchangeable(tmp_path: Path) -> None:
    module = _module()
    canonical = json.loads(CONTRACT.read_text(encoding="utf-8"))
    path = tmp_path / "type-confused.json"

    bool_as_integer = copy.deepcopy(canonical)
    bool_as_integer["fresh_localization_attempt"]["new_container_required"] = 1
    _write_json(path, bool_as_integer)
    assert module.reconstruct(path)["internally_consistent"] is False

    integer_as_boolean = copy.deepcopy(canonical)
    integer_as_boolean["fresh_localization_attempt"]["localization_invocations_allowed"] = True
    _write_json(path, integer_as_boolean)
    assert module.reconstruct(path)["internally_consistent"] is False
