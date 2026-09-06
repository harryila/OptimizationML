from __future__ import annotations

import copy
import importlib.util
import json
from collections.abc import Iterator
from pathlib import Path
from typing import TypeAlias

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/reconstruct_p26_permission_safe_cuda_acquisition_outcome.py"
OUTCOME = ROOT / "results/summaries/p26_permission_safe_cuda_acquisition_outcome.json"
PathPart: TypeAlias = str | int


def _module():
    specification = importlib.util.spec_from_file_location("p26_outcome_reconstruction", SCRIPT)
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


def _at_path(value: object, path: tuple[PathPart, ...]) -> object:
    current = value
    for part in path:
        current = current[part]  # type: ignore[index]
    return current


def _replace_at_path(value: object, path: tuple[PathPart, ...], replacement: object) -> None:
    parent = _at_path(value, path[:-1])
    parent[path[-1]] = replacement  # type: ignore[index]


def _mutated_leaf(value: object) -> object:
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, str):
        return f"{value}__mutated"
    if value is None:
        return "invented_raw_mapping"
    raise AssertionError(f"unsupported leaf type: {type(value).__name__}")


def test_p26_terminal_outcome_reconstructs() -> None:
    result = _module().reconstruct(OUTCOME)

    assert result["internally_consistent"] is True
    assert len(result["checks"]) == 18
    assert all(result["checks"].values())
    assert result["canonical_sha256"] == (
        "9be5727213f1a23f00b05524940d88c2d02461510fb61316b29dacc6ee8ead47"
    )
    assert "not independent authentication of external native bytes" in result["scope"]


def test_p26_passed_bridge_then_stopped_at_first_trace_before_training() -> None:
    outcome = json.loads(OUTCOME.read_text(encoding="utf-8"))
    failure = outcome["trace_off_a_failure"]

    assert outcome["corrected_diagnostic"]["checks"] == {
        "total": 40,
        "true": 40,
        "false": 0,
    }
    assert outcome["pre_acquisition_checks"]["permission_safe_bridge"]["status"] == "passed"
    assert failure["role"] == "trace_off_a"
    assert failure["failure_phase"] == "p22_initialized"
    assert failure["model_and_optimizer_constructed"] is True
    assert failure["initial_state_hash_computed"] is False
    assert failure["forward_backward_started"] is False
    assert failure["optimizer_steps_completed"] == 0
    assert failure["candidate_observations"] == 0
    assert failure["error_message"] == (
        "deleted file-backed mapping is forbidden at /proc/self/maps line 17"
    )


def test_p26_native_failure_repository_snapshots_are_clean_and_identical() -> None:
    outcome = json.loads(OUTCOME.read_text(encoding="utf-8"))
    provenance = outcome["trace_off_a_failure"]["native_failure_provenance"]
    repository = provenance["repository"]
    prepared = repository["prepared_snapshot"]
    failure = repository["failure_time_snapshot"]

    assert prepared["collection_status"] == "prepared_context_validated"
    assert failure["collection_status"] == "recollected_at_failure"
    assert prepared["head"] == failure["head"] == ("185e444afc0b44ca0a09b1bde49a6b6fa3973355")
    assert prepared["tree"] == failure["tree"] == ("24f4bdac331a57bd7c1b807747d7c7fba253ee5a")
    assert prepared["clean"] is failure["clean"] is True
    assert repository["snapshots_match"] is True
    assert provenance["prepared_runtime_sha256"] == (
        "c40abffcc32b2596350ae94e11020ccea49801c32d19d940dbc9b86aafa45e0f"
    )
    assert provenance["prepared_static_contract_sha256"] == (
        "1c7be31c697e443dad70d0c05e9b8f070c85f1d5b3ed2a64b2399c56690d567d"
    )


def test_p26_does_not_guess_the_unretained_deleted_mapping() -> None:
    outcome = json.loads(OUTCOME.read_text(encoding="utf-8"))
    evidence = outcome["trace_off_a_failure"]["rejected_mapping_evidence_limit"]

    assert evidence["proc_maps_line_number"] == 17
    assert evidence["raw_mapping_line_retained"] is False
    assert evidence["raw_mapping_line"] is None
    assert evidence["failed_process_still_available"] is False
    assert "cannot be recovered and must not be guessed" in evidence["reason"]


def test_p26_failure_sanitizer_failed_closed_and_created_nothing() -> None:
    outcome = json.loads(OUTCOME.read_text(encoding="utf-8"))
    sanitization = outcome["failure_sanitization"]

    assert sanitization["status"] == "failed_closed_no_output"
    assert sanitization["exit_code"] == 2
    assert sanitization["sanitized_failure_artifact_created"] is False
    assert sanitization["stdout"]["byte_count"] == 0
    assert sanitization["stderr"]["text"] == (
        "P23 blocked: undeclared absolute path cannot be sanitized: /opt/p23-venv/bin/python\n"
    )
    assert outcome["acquisition_stop"]["sanitized_present_names"] == []


def test_p26_attempt_is_terminal_and_all_later_gates_are_absent() -> None:
    outcome = json.loads(OUTCOME.read_text(encoding="utf-8"))
    gates = outcome["gate_status"]
    policy = outcome["terminal_policy"]

    assert gates["trace_off_b"] == "not_run"
    assert gates["exact_repeatability"] == "not_run"
    assert gates["trace_on"] == "mechanically_barred"
    assert gates["observer_noninterference"] == "not_run"
    assert gates["candidate_observations"] == 0
    assert gates["fidelity_aggregate"] == "not_run"
    assert gates["shield_execution"] == "not_run"
    assert gates["training"] == "not_run"
    assert policy["attempt_is_terminal"] is True
    assert policy["retry_until_favorable_allowed"] is False
    assert policy["reuse_attempt_20260906_02_allowed"] is False
    assert outcome["route"]["next_branch"] == "p27-cuda-deleted-mapping-localization"
    assert outcome["route"]["attempt_20260906_02_may_be_rerun"] is False


def test_every_p26_outcome_leaf_is_bound(tmp_path: Path) -> None:
    module = _module()
    canonical = json.loads(OUTCOME.read_text(encoding="utf-8"))
    mutated_path = tmp_path / "mutated-outcome.json"

    for path in _leaf_paths(canonical):
        mutated = copy.deepcopy(canonical)
        _replace_at_path(mutated, path, _mutated_leaf(_at_path(mutated, path)))
        _write_json(mutated_path, mutated)
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
