from __future__ import annotations

import copy
import importlib.util
import json
from collections.abc import Iterator
from pathlib import Path
from typing import TypeAlias

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/reconstruct_p25_cuda_diagnostic_outcome.py"
OUTCOME = ROOT / "results/summaries/p25_cuda_diagnostic_outcome.json"
SUMMARY = ROOT / "results/summaries/P25_CUDA_DIAGNOSTIC_DETERMINISM_AND_REDACTION_RESULTS.md"
PathPart: TypeAlias = str | int


def _module():
    spec = importlib.util.spec_from_file_location("p25_outcome_reconstruction", SCRIPT)
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


def _at_path(payload: object, path: tuple[PathPart, ...]) -> object:
    current = payload
    for part in path:
        current = current[part]  # type: ignore[index]
    return current


def _replace_at_path(payload: object, path: tuple[PathPart, ...], value: object) -> None:
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
    raise AssertionError(f"no mutation rule for {type(value).__name__}")


def test_p25_terminal_outcome_reconstructs() -> None:
    result = _module().reconstruct(OUTCOME)
    assert result["internally_consistent"] is True
    assert all(result["checks"].values())
    assert result["canonical_sha256"] == (
        "d7a06f6f25bc4839db256720fe5c8ef3c99f6ea45b53b48da2135fcde76a423a"
    )


def test_p25_records_permission_error_not_content_mismatch() -> None:
    outcome = json.loads(OUTCOME.read_text(encoding="utf-8"))
    stop = outcome["acquisition_stop"]
    assert stop["comparison_exit_codes"] == {
        "unprivileged_cmp": 2,
        "privileged_cmp": 0,
    }
    assert stop["root_cause"]["content_mismatch_observed"] is False
    assert stop["root_cause"]["retained_evidence_wrapper"]["mode"] == "0600"
    assert (
        stop["root_cause"]["retained_evidence_wrapper"]["sha256"]
        == (stop["root_cause"]["repository_wrapper"]["sha256"])
    )


def test_p25_records_zero_acquisition_and_terminal_attempt() -> None:
    outcome = json.loads(OUTCOME.read_text(encoding="utf-8"))
    inventory = outcome["acquisition_stop"]["zero_acquisition_artifact_inventory"]
    assert inventory["native_present_names"] == []
    assert inventory["sanitized_present_names"] == []
    assert inventory["acquisition_process_stream_labels_present"] == []
    assert inventory["p23_sanitizer_processes_run"] == 0
    assert outcome["gate_status"]["trace_off_a"] == "not_started"
    assert outcome["gate_status"]["candidate_observations"] == 0
    assert outcome["terminal_policy"]["attempt_is_terminal"] is True
    assert outcome["terminal_policy"]["reuse_attempt_20260906_01_allowed"] is False


def test_every_p25_outcome_leaf_is_bound(tmp_path: Path) -> None:
    module = _module()
    canonical = json.loads(OUTCOME.read_text(encoding="utf-8"))
    mutated_path = tmp_path / "mutated-outcome.json"
    for path in _leaf_paths(canonical):
        mutated = copy.deepcopy(canonical)
        _replace_at_path(mutated, path, _mutated_leaf(_at_path(mutated, path)))
        _write_json(mutated_path, mutated)
        assert module.reconstruct(mutated_path)["internally_consistent"] is False, path


def test_p25_summary_preserves_claim_boundary() -> None:
    text = SUMMARY.read_text(encoding="utf-8")
    assert "terminal before `trace_off_a`" in text
    assert "unprivileged comparison exited `2`" in text
    assert "There are zero trace" in text
    assert "This is not the requested CUDA repeatability" in text
