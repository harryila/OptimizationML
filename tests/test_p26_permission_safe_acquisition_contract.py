from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from collections.abc import Iterator
from pathlib import Path
from typing import TypeAlias

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/reconstruct_p26_permission_safe_acquisition.py"
CONTRACT = ROOT / "experiments/training/p26_permission_safe_acquisition_contract.json"
P25_OUTCOME = ROOT / "results/summaries/p25_cuda_diagnostic_outcome.json"
PathPart: TypeAlias = str | int


def _module():
    specification = importlib.util.spec_from_file_location("p26_contract_reconstruction", SCRIPT)
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


def test_p26_contract_reconstructs_independently() -> None:
    result = _module().reconstruct(CONTRACT)

    assert result["internally_consistent"] is True
    assert len(result["checks"]) == 16
    assert all(result["checks"].values())
    assert result["canonical_sha256"] == hashlib.sha256(CONTRACT.read_bytes()).hexdigest()
    assert result["claim_boundary"] == (
        "Independent static reconstruction of the frozen P26 permission-safe acquisition "
        "contract; not evidence that a P26 runtime, bridge, CUDA trace, fidelity result, "
        "shield execution, or training run occurred."
    )


def test_p25_parent_is_terminal_and_repository_authenticated() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    parent = contract["terminal_parent"]
    outcome = json.loads(P25_OUTCOME.read_text(encoding="utf-8"))

    assert parent["outcome_sha256"] == hashlib.sha256(P25_OUTCOME.read_bytes()).hexdigest()
    assert parent["outcome_commit"] == "f055405cc879ba0ac5afe26bc34a7336d5d2efbf"
    assert parent["outcome_tree"] == "a5ee62de68b091719938829e579853b226b7f086"
    assert parent["attempt_id"] == outcome["attempt_id"] == "20260906-01"
    assert parent["attempt_is_terminal"] is True
    assert parent["attempt_reuse_allowed"] is False
    assert parent["diagnostic_passed_checks"] == 40
    assert parent["diagnostic_failed_checks"] == 0
    assert parent["unprivileged_cmp_exit"] == 2
    assert parent["privileged_cmp_exit"] == 0
    assert parent["trace_off_a_started"] is False
    assert parent["candidate_observations"] == 0
    assert outcome["terminal_policy"]["retry_until_favorable_allowed"] is False


def test_every_recorded_source_hash_matches_its_bytes() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    groups = (
        contract["frozen_p25_control"],
        contract["p26_control_sources"],
        contract["unchanged_scientific_authorities"],
    )

    records: list[dict[str, object]] = []
    for group in groups:
        for value in group.values():
            if isinstance(value, dict) and set(value) >= {"path", "sha256"}:
                records.append(value)

    assert len(records) == 13
    for record in records:
        path = ROOT / str(record["path"])
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]


def test_fresh_attempt_order_routing_and_forbidden_changes_are_exact() -> None:
    module = _module()
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    assert contract["fresh_attempt"] == module.EXPECTED_FRESH_ATTEMPT
    assert contract["acquisition_order"] == module.EXPECTED_ACQUISITION_ORDER
    assert contract["terminal_routing"] == module.EXPECTED_TERMINAL_ROUTING
    assert contract["forbidden_changes"] == module.EXPECTED_FORBIDDEN_CHANGES
    assert contract["acquisition_order"].index("trace_off_a") < contract["acquisition_order"].index(
        "trace_off_b only after trace_off_a passes"
    )
    assert contract["acquisition_order"].index(
        "verify_repeatability with exact equality only"
    ) < contract["acquisition_order"].index(
        "trace_on only after zero exact repeatability mismatches"
    )
    assert (
        "replace exact equality with allclose or another tolerance" in contract["forbidden_changes"]
    )
    assert "rerun until favorable" in contract["forbidden_changes"]


def test_contract_is_preexecution_only_and_has_no_forbidden_result_fields() -> None:
    result = _module().reconstruct(CONTRACT)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    assert result["checks"]["no_execution_result_fields_present"] is True
    assert contract["status"] == "frozen_pre_fresh_runtime_and_pre_acquisition"
    assert "records no P26 runtime" in contract["claim_boundary"]
    assert contract["permission_safe_bridge"]["before_trace_off_a"] is True
    assert contract["permission_safe_bridge"]["host_cmp_used"] is False
    assert contract["permission_safe_bridge"]["mutations_allowed"] == []


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


def test_reordered_gate_sequences_fail_closed(tmp_path: Path) -> None:
    module = _module()
    canonical = json.loads(CONTRACT.read_text(encoding="utf-8"))
    path = tmp_path / "reordered.json"

    for field in ("acquisition_order", "forbidden_changes"):
        mutated = copy.deepcopy(canonical)
        mutated[field].reverse()
        _write_json(path, mutated)
        assert module.reconstruct(path)["internally_consistent"] is False

    mutated = copy.deepcopy(canonical)
    checks = mutated["permission_safe_bridge"]["required_checks"]
    checks[0], checks[1] = checks[1], checks[0]
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
    bool_as_integer["fresh_attempt"]["no_favorable_rerun"] = 1
    _write_json(path, bool_as_integer)
    assert module.reconstruct(path)["internally_consistent"] is False

    integer_as_boolean = copy.deepcopy(canonical)
    integer_as_boolean["terminal_parent"]["diagnostic_passed_checks"] = True
    _write_json(path, integer_as_boolean)
    assert module.reconstruct(path)["internally_consistent"] is False
