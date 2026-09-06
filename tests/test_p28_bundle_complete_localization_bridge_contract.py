from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from collections.abc import Iterator
from pathlib import Path
from typing import TypeAlias

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "experiments/training/p28_bundle_complete_localization_bridge_contract.json"
RECONSTRUCTOR = ROOT / "scripts/reconstruct_p28_bundle_complete_localization_bridge.py"
PathPart: TypeAlias = str | int


def _module():
    specification = importlib.util.spec_from_file_location(
        "p28_contract_reconstruction", RECONSTRUCTOR
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _leaves(value: object, prefix: tuple[PathPart, ...] = ()) -> Iterator[tuple[PathPart, ...]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _leaves(item, (*prefix, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _leaves(item, (*prefix, index))
    else:
        yield prefix


def _at(value: object, path: tuple[PathPart, ...]) -> object:
    current = value
    for part in path:
        current = current[part]  # type: ignore[index]
    return current


def _replace(value: object, path: tuple[PathPart, ...], replacement: object) -> None:
    parent = _at(value, path[:-1])
    parent[path[-1]] = replacement  # type: ignore[index]


def _mutation(value: object) -> object:
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, str):
        return f"{value}__mutated"
    if value is None:
        return "invented"
    raise AssertionError(type(value).__name__)


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def test_p28_bundle_complete_bridge_reconstructs() -> None:
    result = _module().reconstruct(CONTRACT)

    assert result["internally_consistent"] is True
    assert len(result["checks"]) == 12
    assert all(result["checks"].values())
    assert result["p27_reconstruction"] == {
        "check_count": 23,
        "true_count": 23,
        "false_count": 0,
    }
    assert result["p27_outcome_reconstruction"] == {
        "check_count": 10,
        "true_count": 10,
        "false_count": 0,
    }
    assert "does not authenticate a transfer receipt" in result["claim_boundary"]


def test_terminal_p27_parent_and_exact_unchanged_sources_are_bound() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    parent = contract["terminal_parent"]
    authorities = contract["unchanged_p27_authorities"]

    assert parent["outcome_commit"] == "ec63550331925ded158e3f389e294e4d1f12db3a"
    assert parent["outcome_tree"] == "d8efa72fda9ee41fde0b5d15b126aa87397cd48d"
    assert parent["diagnostic_tag_object"] == ("c88b98eae9be2458abde45b05d3dccfe09c0c7ed")
    assert parent["attempt_is_terminal"] is True
    assert parent["attempt_reuse_allowed"] is False
    assert parent["container_created"] is False
    assert parent["localization_invocation_consumed"] is False

    for entry in authorities.values():
        if "path" in entry:
            data = (ROOT / entry["path"]).read_bytes()
            assert hashlib.sha256(data).hexdigest() == entry["sha256"]


def test_bundle_inventory_carries_all_five_exact_refs_and_annotated_tags() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    refs = contract["required_bundle_refs"]

    assert [item["ref"] for item in refs] == [
        "refs/heads/p28-bundle-complete-localization-bridge",
        "refs/heads/p27-cuda-deleted-mapping-localization",
        "refs/tags/p27-cuda-deleted-mapping-localization-diagnostic",
        "refs/tags/p26-permission-safe-acquisition-checkpoint",
        "refs/tags/p26-attempt02-acquisition-source",
    ]
    assert refs[2]["object_type"] == refs[3]["object_type"] == refs[4]["object_type"] == "tag"
    assert refs[2]["object"] == "c88b98eae9be2458abde45b05d3dccfe09c0c7ed"
    assert refs[3]["object"] == "bacad707d3779bfa10957e18cb4c69b1a7f0cbce"
    assert refs[3]["peeled_commit"] == "5429da23ff18888daa2312c530a4587780484d8b"
    assert refs[4]["object"] == "f35a7dca8f6bc39e9748e79b6712fab4a203396b"
    assert refs[4]["peeled_commit"] == "185e444afc0b44ca0a09b1bde49a6b6fa3973355"


def test_bundle_closure_precedes_attempt_root_and_uses_two_external_receipts() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    closure = contract["bundle_closure"]
    attempt = contract["fresh_localization_attempt"]

    assert closure["transport_root"] == "/secure/p28/transport-20260906-01"
    assert closure["authority_checkout"] == "/secure/p28/authority-185e444"
    assert closure["source_closure_repository"].endswith("/p28_source_closure.git")
    assert closure["runtime_review_closure_repository"].endswith("/p28_runtime_review_closure.git")
    assert closure["source_receipt"].endswith("/p28_source_bundle_receipt.json")
    assert closure["runtime_review_receipt"].endswith("/p28_runtime_review_bundle_receipt.json")
    assert closure["receipt_schema"] == "passive-muon-p28-control-bundle-receipt-v1"
    assert closure["receipt_creation_cli"] == "--receipt-output"
    assert closure["receipt_replay_cli"] == (
        "--verify-existing-receipt plus --expected-receipt-sha256"
    )
    assert closure["receipt_files_are_external_and_not_committed"] is True
    assert closure["receipt_outputs_are_o_excl_no_overwrite"] is True
    assert closure["bundle_private_authenticated_read_only_copy_required"] is True
    assert closure["private_copy_required_mode"] == "0400"
    assert closure["private_copy_post_use_hash_and_stat_revalidation_required"] is True
    assert closure["fresh_empty_bare_repository_required_for_each_phase"] is True
    assert closure["exact_advertised_ref_count"] == 5
    assert closure["bundle_prerequisite_count"] == 0
    assert closure["source_receipt_required_before_attempt_root_creation"] is True
    assert closure["authority_checkout_must_be_absent_until_source_receipt_reviewed"] is True
    assert closure["authority_checkout_constructed_only_from_verified_source_closure"] is True
    assert closure["p27_reconstruction_required_check_count"] == 23
    assert closure["p27_reconstruction_required_true_count"] == 23
    assert closure["p27_reconstruction_required_false_count"] == 0
    assert attempt["attempt_root"] == "/secure/p28/attempt-20260906-01"
    assert attempt["root_must_not_preexist_before_verified_source_receipt"] is True


def test_two_phase_review_is_exact_direct_child_and_acquisition_stays_barred() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    verifier = contract["bundle_verifier"]
    orchestration = contract["host_orchestration"]
    freeze = contract["two_phase_freeze"]
    boundary = contract["no_training_boundary"]

    assert verifier["allowed_phases"] == ["source", "runtime-review"]
    assert verifier["bootstrap_execution_path"] == (
        "/secure/p28/transport-20260906-01/verify_p28_control_bundle.py"
    )
    assert verifier["checked_in_control_verifier_path"] == (
        "/secure/p28/control-source/scripts/verify_p28_control_bundle.py"
    )
    assert verifier["bootstrap_copy_is_o_excl_no_overwrite"] is True
    assert verifier["bootstrap_must_be_regular_nonsymlink"] is True
    assert verifier["bootstrap_required_mode"] == "0600"
    assert verifier["receipt_binds_canonical_verifier_source_path_and_sha256"] is True
    assert verifier["runtime_review_phase_exact_delta"] == [
        "A\texperiments/training/p28_cuda_runtime_lock.json",
        "A\texperiments/training/p28_host_attestation.json",
    ]
    assert freeze["phase_two_exact_delta"] == verifier["runtime_review_phase_exact_delta"]
    assert len(orchestration["required_environment"]) == 28
    assert len(set(orchestration["required_environment"])) == 28
    assert orchestration["state_changing_prepare_runtime_invocations_allowed"] == 1
    assert (
        orchestration[
            "pre_root_source_receipt_replay_is_read_only_and_not_counted_as_state_changing_prepare"
        ]
        is True
    )
    assert freeze["phase_two_localization_started"] is False
    assert boundary["forward_allowed"] is False
    assert boundary["backward_allowed"] is False
    assert boundary["optimizer_step_allowed"] is False
    assert boundary["candidate_evaluation_allowed"] is False
    assert boundary["candidate_observation_count_required"] == 0
    assert contract["terminal_routing"]["future_scientific_acquisition_authorized"] is False


def test_every_contract_leaf_is_bound_by_independent_digest(tmp_path: Path) -> None:
    module = _module()
    module._run_p27_reconstruction = lambda: (True, 23, 23, 0)
    module._run_p27_outcome_reconstruction = lambda: (True, 10, 10, 0)
    canonical = json.loads(CONTRACT.read_text(encoding="utf-8"))
    mutated_path = tmp_path / "mutated.json"

    for path in _leaves(canonical):
        mutated = copy.deepcopy(canonical)
        _replace(mutated, path, _mutation(_at(mutated, path)))
        _write(mutated_path, mutated)
        assert module.reconstruct(mutated_path)["internally_consistent"] is False, path


def test_duplicate_nonfinite_and_noncanonical_json_fail_closed(tmp_path: Path) -> None:
    module = _module()
    module._run_p27_reconstruction = lambda: (True, 23, 23, 0)
    module._run_p27_outcome_reconstruction = lambda: (True, 10, 10, 0)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"status":"first","status":"second"}\n', encoding="utf-8")
    assert module.reconstruct(duplicate)["internally_consistent"] is False

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"status":NaN}\n', encoding="utf-8")
    assert module.reconstruct(nonfinite)["internally_consistent"] is False

    compact = tmp_path / "compact.json"
    compact.write_text(json.dumps(json.loads(CONTRACT.read_text(encoding="utf-8"))))
    assert module.reconstruct(compact)["internally_consistent"] is False
