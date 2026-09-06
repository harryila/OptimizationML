from __future__ import annotations

import copy
import importlib.util
import json
from collections.abc import Iterator
from pathlib import Path
from typing import TypeAlias

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/reconstruct_p24_cuda_executable_origin_outcome.py"
OUTCOME = ROOT / "results/summaries/p24_cuda_executable_origin_outcome.json"
SUMMARY = ROOT / "results/summaries/P24_CUDA_EXECUTABLE_ORIGIN_HARDENING_RESULTS.md"
README = ROOT / "README.md"
EXPERIMENTS = ROOT / "experiments/README.md"
RESULTS = ROOT / "results/summaries/RESULTS.md"
PathPart: TypeAlias = str | int


def _module():
    spec = importlib.util.spec_from_file_location("p24_outcome_reconstruction", SCRIPT)
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


def test_p24_terminal_outcome_reconstructs() -> None:
    result = _module().reconstruct(OUTCOME)
    assert result["internally_consistent"] is True
    assert all(result["checks"].values())
    assert result["canonical_sha256"] == (
        "1b1a75d823b45d3c04daa024839dfbbe9461de9fac38117b0602ea7fbbcf941f"
    )
    assert result["scope"] == (
        "repository-reconstructible static bindings and internal consistency of a compact "
        "operator-recorded P24 stop; not authentication or reconstruction of the retained "
        "external native diagnostic"
    )


def test_outcome_distinguishes_origin_localization_from_cuda_evidence() -> None:
    outcome = json.loads(OUTCOME.read_text(encoding="utf-8"))
    baseline = outcome["baseline_diagnostic"]
    gates = outcome["gate_status"]

    assert baseline["checks"] == {
        "total": 36,
        "true": 35,
        "false": 1,
        "sole_false_check": "torch_determinism_matches_lock",
    }
    assert baseline["executable_origin_observation"]["p23_hypothesis_reproduced"] is True
    assert baseline["executable_origin_observation"]["effective_mount_writable"] is True
    assert baseline["native_artifact"]["retention"] == "external_native_bytes_not_committed"
    assert gates["candidate_observations"] == 0
    for gate in (
        "remediation_image_build",
        "replacement_container",
        "replacement_runtime_freeze",
        "remediated_diagnostic",
        "trace_off_a",
        "trace_off_b",
        "exact_repeatability",
        "trace_on",
        "observer_noninterference",
        "fidelity_aggregate",
        "training",
    ):
        assert gates[gate] == "not_run"


def test_sanitizer_failure_is_separate_and_fail_closed() -> None:
    outcome = json.loads(OUTCOME.read_text(encoding="utf-8"))
    sanitization = outcome["sanitization"]

    assert sanitization["status"] == "blocked_no_output"
    assert sanitization["sanitized_artifact_created"] is False
    assert sanitization["stderr"]["text"] == (
        "P24 diagnostic sanitization blocked: redacted diagnostic retains a declared native path\n"
    )
    assert sanitization["source_level_root_cause"]["classification"] == (
        "absolute_mapping_keys_not_transformed"
    )
    reconstruction = _module().reconstruct(OUTCOME)
    assert reconstruction["checks"]["sanitizer_key_redaction_bug_reconstructs_from_frozen_sources"]


def test_every_outcome_leaf_is_exactly_bound(tmp_path: Path) -> None:
    module = _module()
    canonical = json.loads(OUTCOME.read_text(encoding="utf-8"))
    mutated_path = tmp_path / "mutated-outcome.json"

    for path in _leaf_paths(canonical):
        mutated = copy.deepcopy(canonical)
        _replace_at_path(mutated, path, _mutated_leaf(_at_path(mutated, path)))
        _write_json(mutated_path, mutated)
        assert module.reconstruct(mutated_path)["internally_consistent"] is False, path


def test_every_outcome_mapping_rejects_missing_and_extra_keys(tmp_path: Path) -> None:
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


def test_boolean_and_integer_fields_are_not_interchangeable(tmp_path: Path) -> None:
    module = _module()
    canonical = json.loads(OUTCOME.read_text(encoding="utf-8"))
    path = tmp_path / "type-confused.json"

    bool_as_integer = copy.deepcopy(canonical)
    bool_as_integer["baseline_diagnostic"]["passes"] = 0
    _write_json(path, bool_as_integer)
    assert module.reconstruct(path)["internally_consistent"] is False

    integer_as_boolean = copy.deepcopy(canonical)
    integer_as_boolean["baseline_diagnostic"]["exit_code"] = True
    _write_json(path, integer_as_boolean)
    assert module.reconstruct(path)["internally_consistent"] is False


def test_summary_states_terminal_boundary_and_route() -> None:
    summary = SUMMARY.read_text(encoding="utf-8")
    assert "stopped at its first retained A100 baseline diagnostic" in summary
    assert "No sanitized diagnostic exists" in summary
    assert "candidate observations | `0`" in summary
    assert "p25-cuda-diagnostic-determinism-and-redaction" in summary
    assert "P18--P21 analytic and finite-precision results are unaffected" in summary


def test_mutable_status_surfaces_record_the_p24_stop() -> None:
    for path in (README, EXPERIMENTS, RESULTS):
        text = path.read_text(encoding="utf-8")
        assert "p24_cuda_executable_origin_outcome.json" in text
        assert "p25-cuda-diagnostic-determinism-and-redaction" in text
    assert "## 30. P24 native origin localization and pre-acquisition stop" in (
        RESULTS.read_text(encoding="utf-8")
    )
