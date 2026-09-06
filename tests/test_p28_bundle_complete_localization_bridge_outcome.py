from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from collections.abc import Iterator
from pathlib import Path
from typing import TypeAlias

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/reconstruct_p28_bundle_complete_localization_bridge_outcome.py"
OUTCOME = ROOT / "results/summaries/p28_bundle_complete_localization_bridge_outcome.json"
PathPart: TypeAlias = str | int


def _module():
    specification = importlib.util.spec_from_file_location("p28_outcome_reconstruction", SCRIPT)
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


def test_p28_terminal_pre_control_outcome_reconstructs() -> None:
    result = _module().reconstruct(OUTCOME)

    assert result["internally_consistent"] is True
    assert len(result["checks"]) == 10
    assert all(result["checks"].values())
    assert result["canonical_sha256"] == (
        "298324d163aa12c1cb64bfdb912ae846152d0980ae3c981a54ead75a019305c9"
    )
    assert "independently reconstructed from native bytes" in result["claim_boundary"]


def test_exact_operator_captured_error_is_hash_bound_without_invented_time() -> None:
    outcome = json.loads(OUTCOME.read_text(encoding="utf-8"))
    verification = outcome["source_verification"]
    stderr = (verification["exact_terminal_stderr"] + "\n").encode()

    assert verification["invocation_utc_independently_retained"] is False
    assert verification["invocation_utc"] is None
    assert verification["operator_capture_only"] is True
    assert verification["remote_stdout_file_retained"] is False
    assert verification["remote_stderr_file_retained"] is False
    assert len(stderr) == verification["exact_terminal_stderr_with_one_lf_byte_count"] == 202
    assert (
        hashlib.sha256(stderr).hexdigest()
        == verification["exact_terminal_stderr_with_one_lf_sha256"]
    )
    assert verification["exact_terminal_stderr_with_one_lf_sha256"] == (
        "3d85310ab120816ed22e1d78295b2384ff2e0e2147d5700dbb93abe1bb28b355"
    )


def test_complete_bundle_and_partial_source_closure_are_recorded_exactly() -> None:
    outcome = json.loads(OUTCOME.read_text(encoding="utf-8"))
    bundle = outcome["source_bundle"]
    closure = outcome["retained_remote_state"]["source_closure"]

    assert bundle["sha256"] == ("54461db0d3a2e07870d9f43cf9877da0cb4f28c13ad1c5f1cc32b44ff3e7945b")
    assert bundle["byte_count"] == 3_362_894
    assert bundle["prerequisite_count"] == 0
    assert len(bundle["advertised_refs"]) == 5
    assert bundle["exact_closed_ref_inventory_verified"] is True
    assert closure["created"] is True
    assert closure["exact_five_refs_present"] is True
    assert outcome["source_verification"]["source_receipt_created"] is False


def test_p28_stopped_before_control_attempt_container_or_cuda() -> None:
    outcome = json.loads(OUTCOME.read_text(encoding="utf-8"))
    state = outcome["retained_remote_state"]
    gates = outcome["gate_status"]

    assert state["control_checkout_created"] is False
    assert state["source_receipt_created"] is False
    assert state["attempt_root_created"] is False
    assert state["container_created"] is False
    assert state["runtime_lock_created"] is False
    assert state["host_attestation_created"] is False
    assert gates["p27_contract_reconstruction"] == "not_run"
    assert gates["gpu_access"] == "not_run"
    assert gates["cuda_initialization"] == "not_run"
    assert gates["deleted_mapping_localization"] == "not_run"
    assert gates["optimizer_steps"] == 0
    assert gates["candidate_observations"] == 0
    assert gates["training"] == "not_run"


def test_runbook_host_layout_defect_and_new_attempt_route_are_explicit() -> None:
    outcome = json.loads(OUTCOME.read_text(encoding="utf-8"))
    cause = outcome["root_cause"]
    policy = outcome["terminal_policy"]
    route = outcome["route"]

    assert cause["classification"] == "unprivileged_control_checkout_parent_permission"
    assert cause["runbook_host_layout_defect_observed"] is True
    assert cause["host_layout_preflight_defect_observed"] is True
    assert cause["verifier_failed_closed"] is True
    assert cause["verifier_logic_defect_observed"] is False
    assert policy["attempt_is_terminal"] is True
    assert policy["transport_and_closure_are_terminal"] is True
    assert policy["source_verifier_invocation_consumed"] is True
    assert policy["prepare_runtime_invocation_consumed"] is False
    assert policy["localization_invocation_consumed"] is False
    assert route["new_attempt_required"] is True
    assert route["attempt_20260906_01_may_be_reused"] is False
    assert route["retained_p28_transport_or_closure_may_be_cleaned_up"] is False
    assert route["scientific_acquisition_authorized"] is False


def test_every_p28_outcome_leaf_is_bound(tmp_path: Path) -> None:
    module = _module()
    canonical = json.loads(OUTCOME.read_text(encoding="utf-8"))
    mutated_path = tmp_path / "mutated.json"

    for path in _leaves(canonical):
        mutated = copy.deepcopy(canonical)
        _replace(mutated, path, _mutation(_at(mutated, path)))
        _write(mutated_path, mutated)
        assert module.reconstruct(mutated_path)["internally_consistent"] is False, path


def test_duplicate_nonfinite_and_noncanonical_json_fail_closed(tmp_path: Path) -> None:
    module = _module()

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"status":"first","status":"second"}\n', encoding="utf-8")
    assert module.reconstruct(duplicate)["internally_consistent"] is False

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"status":NaN}\n', encoding="utf-8")
    assert module.reconstruct(nonfinite)["internally_consistent"] is False

    compact = tmp_path / "compact.json"
    compact.write_text(json.dumps(json.loads(OUTCOME.read_text(encoding="utf-8"))))
    assert module.reconstruct(compact)["internally_consistent"] is False
