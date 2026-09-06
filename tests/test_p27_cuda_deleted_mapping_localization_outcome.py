from __future__ import annotations

import copy
import importlib.util
import json
from collections.abc import Iterator
from pathlib import Path
from typing import TypeAlias

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/reconstruct_p27_cuda_deleted_mapping_localization_outcome.py"
OUTCOME = ROOT / "results/summaries/p27_cuda_deleted_mapping_localization_outcome.json"
TRANSCRIPT = ROOT / "results/summaries/p27_prepare_contract_reconstruction.remote.json"
PathPart: TypeAlias = str | int


def _module():
    specification = importlib.util.spec_from_file_location("p27_outcome_reconstruction", SCRIPT)
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


def test_p27_terminal_pre_container_outcome_reconstructs() -> None:
    result = _module().reconstruct(OUTCOME)

    assert result["internally_consistent"] is True
    assert len(result["checks"]) == 10
    assert all(result["checks"].values())
    assert result["canonical_sha256"] == (
        "d87df6cc792dfa0a0d6a2a982395953ba2ef87aa2086266d9d3c59abb3427d0e"
    )
    assert "not independently authenticated native bytes" in result["claim_boundary"]


def test_exact_remote_stdout_copy_records_only_the_missing_tag_failure() -> None:
    transcript = json.loads(TRANSCRIPT.read_text(encoding="utf-8"))
    checks = transcript["checks"]

    assert len(checks) == 23
    assert sum(value is True for value in checks.values()) == 22
    assert checks["terminal_p26_commit_tree_tag_and_bytes_exact"] is False
    assert all(
        value is True
        for key, value in checks.items()
        if key != "terminal_p26_commit_tree_tag_and_bytes_exact"
    )
    assert transcript["internally_consistent"] is False
    assert transcript["canonical_sha256"] == (
        "02000debea28f499ddeb3b8c6c8b0615786cd2e7858cc8a1605b04d638da4b7a"
    )


def test_p27_stopped_before_container_cuda_or_localization() -> None:
    outcome = json.loads(OUTCOME.read_text(encoding="utf-8"))
    state = outcome["retained_attempt_state"]
    gates = outcome["gate_status"]

    assert state["container_created"] is False
    assert state["runtime_lock_created"] is False
    assert state["host_attestation_created"] is False
    assert state["native_localization_artifact_created"] is False
    assert state["sanitized_localization_artifact_created"] is False
    assert gates["gpu_access"] == "not_run"
    assert gates["cuda_initialization"] == "not_run"
    assert gates["deleted_mapping_localization"] == "not_run"
    assert gates["model_construction"] == "not_run"
    assert gates["optimizer_construction"] == "not_run"
    assert gates["forward_backward"] == "not_run"
    assert gates["optimizer_steps"] == 0
    assert gates["candidate_observations"] == 0
    assert gates["training"] == "not_run"


def test_p27_transfer_bundle_inventory_exposes_the_exact_omission() -> None:
    outcome = json.loads(OUTCOME.read_text(encoding="utf-8"))
    transfer = outcome["operator_transfer"]

    assert transfer["advertised_refs"] == [
        {
            "object": "1071afa1b20b3e12be3dafdebf9dae63f39b0ed9",
            "ref": "refs/heads/p27-cuda-deleted-mapping-localization",
        },
        {
            "object": "f35a7dca8f6bc39e9748e79b6712fab4a203396b",
            "ref": "refs/tags/p26-attempt02-acquisition-source",
        },
    ]
    assert transfer["required_checkpoint"] == {
        "ref": "refs/tags/p26-permission-safe-acquisition-checkpoint",
        "annotated_tag_object": "bacad707d3779bfa10957e18cb4c69b1a7f0cbce",
        "peeled_commit": "5429da23ff18888daa2312c530a4587780484d8b",
    }
    assert transfer["required_checkpoint_ref_advertised"] is False
    assert transfer["required_annotated_tag_object_transferred"] is False


def test_attempt_is_terminal_without_consuming_localization() -> None:
    outcome = json.loads(OUTCOME.read_text(encoding="utf-8"))
    policy = outcome["terminal_policy"]

    assert policy["attempt_is_terminal"] is True
    assert policy["reuse_attempt_20260906_01_allowed"] is False
    assert policy["cleanup_attempt_20260906_01_allowed"] is False
    assert policy["retry_until_favorable_allowed"] is False
    assert policy["localization_invocation_consumed"] is False
    assert outcome["route"]["new_attempt_required"] is True
    assert outcome["route"]["p28_created_by_this_outcome"] is False


def test_every_p27_outcome_leaf_is_bound(tmp_path: Path) -> None:
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
