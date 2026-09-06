from __future__ import annotations

import copy
import importlib.util
import json
from collections.abc import Iterator
from pathlib import Path
from typing import TypeAlias

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/reconstruct_p23_cuda_shadow_trace_outcome.py"
OUTCOME = ROOT / "results/summaries/p23_cuda_shadow_trace_outcome.json"
PathPart: TypeAlias = str | int


def _module():
    spec = importlib.util.spec_from_file_location("p23_outcome_reconstruction", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _at_path(payload: object, path: tuple[PathPart, ...]) -> object:
    current = payload
    for part in path:
        current = current[part]  # type: ignore[index]
    return current


def _replace_at_path(payload: object, path: tuple[PathPart, ...], value: object) -> None:
    assert path
    parent = _at_path(payload, path[:-1])
    parent[path[-1]] = value  # type: ignore[index]


def _mutated_leaf(value: object) -> object:
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, str):
        return f"{value}__mutated"
    if value is None:
        return "unexpected_non_null"
    raise AssertionError(f"test has no mutation rule for {type(value).__name__}")


def test_p23_cuda_early_stop_is_internally_consistent() -> None:
    result = _module().reconstruct(OUTCOME)
    assert result["internally_consistent"] is True
    assert all(result["checks"].values())
    assert "certified" not in result
    assert result["scope"] == (
        "static and internal consistency of an operator-recorded P23 early-stop record; "
        "not authentication or independent reproduction of the remote CUDA event"
    )


def test_p23_cuda_early_stop_has_no_training_or_fidelity_evidence() -> None:
    outcome = json.loads(OUTCOME.read_text(encoding="utf-8"))
    attempt = outcome["attempt"]

    assert outcome["evidence_kind"] == "operator_recorded_cuda_shadow_trace_pretraining_blocker"
    assert "independent_localization" not in outcome
    assert attempt["evidence_status"] == (
        "operator_recorded_with_docker_event_metadata_not_native_failure_manifest"
    )
    assert "not hash-bound at execution time" in attempt["transcript_retention"]
    assert attempt["forward_backward_started"] is False
    assert attempt["optimizer_steps_completed"] == 0
    assert attempt["candidate_observations"] == 0
    assert "no optimizer step" in outcome["claim_boundary"]
    assert outcome["gate_status"]["trace_on"] == "mechanically_barred"


def test_every_operator_record_leaf_is_covered_by_reconstruction(tmp_path: Path) -> None:
    module = _module()
    canonical = json.loads(OUTCOME.read_text(encoding="utf-8"))
    mutated_path = tmp_path / "mutated-outcome.json"

    for path in _leaf_paths(canonical):
        mutated = copy.deepcopy(canonical)
        original = _at_path(mutated, path)
        _replace_at_path(mutated, path, _mutated_leaf(original))
        _write_json(mutated_path, mutated)
        result = module.reconstruct(mutated_path)
        assert result["internally_consistent"] is False, path


def test_every_mapping_rejects_missing_and_extra_keys(tmp_path: Path) -> None:
    module = _module()
    canonical = json.loads(OUTCOME.read_text(encoding="utf-8"))
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


def test_nonobject_outcome_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "nonobject.json"
    _write_json(path, [])
    result = _module().reconstruct(path)
    assert result["internally_consistent"] is False
    assert result["checks"]["outcome_object_and_exact_schema"] is False


def test_json_boolean_and_integer_types_are_not_interchangeable(tmp_path: Path) -> None:
    module = _module()
    canonical = json.loads(OUTCOME.read_text(encoding="utf-8"))
    path = tmp_path / "type-confused.json"

    bool_as_integer = copy.deepcopy(canonical)
    bool_as_integer["attempt"]["model_allocated"] = 1
    _write_json(path, bool_as_integer)
    assert module.reconstruct(path)["internally_consistent"] is False

    integer_as_boolean = copy.deepcopy(canonical)
    integer_as_boolean["attempt"]["docker_exec_exit_code"] = True
    _write_json(path, integer_as_boolean)
    assert module.reconstruct(path)["internally_consistent"] is False


def test_inventory_sequence_cardinality_and_order_are_exact(tmp_path: Path) -> None:
    module = _module()
    canonical = json.loads(OUTCOME.read_text(encoding="utf-8"))
    path = tmp_path / "inventory-sequence.json"

    with_extra = copy.deepcopy(canonical)
    with_extra["native_evidence_inventory_after_failure"]["absent"].append("unexpected.json")
    _write_json(path, with_extra)
    assert module.reconstruct(path)["internally_consistent"] is False

    reordered = copy.deepcopy(canonical)
    reordered["native_evidence_inventory_after_failure"]["absent"].reverse()
    _write_json(path, reordered)
    assert module.reconstruct(path)["internally_consistent"] is False
