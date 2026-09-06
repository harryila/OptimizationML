from __future__ import annotations

import ast
import copy
import importlib.util
import json
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import TypeAlias

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "experiments/training/p24_cuda_executable_origin_contract.json"
RECONSTRUCTOR = ROOT / "scripts/reconstruct_p24_cuda_executable_origin.py"
DIAGNOSTIC = ROOT / "experiments/training/run_p24_executable_origin_diagnostic.py"
HELPER = ROOT / "experiments/training/prepare_p24_immutable_remote_module.py"
WORKFLOW = ROOT / ".github/workflows/p24-cuda-executable-origin-hardening.yml"
PathPart: TypeAlias = str | int


def _module():
    spec = importlib.util.spec_from_file_location("p24_static_reconstruction", RECONSTRUCTOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


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
    raise AssertionError(type(value).__name__)


def test_p24_static_contract_reconstructs_without_claiming_a_result() -> None:
    result = _module().reconstruct(CONTRACT)
    assert result["internally_consistent"] is True
    assert len(result["checks"]) == 20
    assert all(result["checks"].values())
    assert "no native diagnostic" in result["scope"]
    assert "fidelity" in result["scope"]
    assert "certified" not in result


def test_p24_contract_freezes_parent_gates_and_pre_result_scope() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    assert contract["status"] == "frozen_pre_remediation_build_and_pre_acquisition"
    assert contract["provenance_parent"]["outcome_commit"] == (
        "3a1454e29edc022e92f2848e216c790d4ea8434a"
    )
    assert contract["provenance_parent"]["outcome_tag"] == (
        "p23-deterministic-cuda-shadow-trace-diagnostic"
    )
    assert contract["frozen_p23_gates"] == {
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
    assert contract["claim_boundary"]["pre_acquisition_contract_only"] is True
    assert contract["claim_boundary"]["p24_cuda_result_present"] is False
    assert contract["p23_blocker"]["native_failure_manifest_present"] is False


def test_p24_remediation_hashes_are_shared_by_contract_runner_and_helper() -> None:
    module = _module()
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    hashes = contract["remediation_design"]["source_hashes"]
    runner = module._literal_string_constants(DIAGNOSTIC)
    helper = module._literal_string_constants(HELPER)

    assert hashes["instantiator_before_sha256"] == runner["ORIGINAL_INSTANTIATOR_SHA256"]
    assert hashes["instantiator_after_sha256"] == runner["PATCHED_INSTANTIATOR_SHA256"]
    assert hashes["generated_module_sha256"] == runner["GENERATED_SHA256"]
    assert hashes["instantiator_before_sha256"] == helper["ORIGINAL_INSTANTIATOR_SHA256"]
    assert hashes["instantiator_after_sha256"] == helper["PATCHED_INSTANTIATOR_SHA256"]
    assert hashes["generated_module_sha256"] == helper["GENERATED_MODULE_SHA256"]

    for record in contract["execution_sources"].values():
        source = ROOT / record["path"]
        assert module._sha256(source.read_bytes()) == record["sha256"]
        assert record["mutation_allowed_after_freeze"] is False


def test_p24_does_not_allow_writable_generated_code_or_tmp_exception() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    design = contract["remediation_design"]
    policy = design["runtime_policy"]

    assert design["generated_module"].startswith("/opt/p23-venv/")
    assert not design["generated_module"].startswith(("/tmp/", "/dev/shm/"))
    assert policy["generated_directory_writable_in_acquisition"] is False
    assert policy["writable_executable_origin_rejection_unchanged"] is True
    assert policy["tmp_allowlist_added"] is False
    assert policy["dev_shm_allowlist_added"] is False
    assert policy["unexpected_runtime_generation"] == (
        "patched verifier rejects unexpected paths or content, and the acquisition image root "
        "is read-only"
    )
    assert design["generated_directory_mode"] == "0555"
    assert design["generated_module_mode"] == "0444"
    assert design["image_manifest"] == "/opt/p24-immutable-remote-module-manifest.json"


def test_p24_requires_new_image_container_lock_and_attestation() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    replacement = contract["replacement_runtime"]

    assert replacement["new_oci_digest_required"] is True
    assert replacement["new_running_container_required"] is True
    assert replacement["new_runtime_lock_required"] is True
    assert replacement["new_host_attestation_required"] is True
    assert replacement["old_p23_image_allowed_for_acquisition"] is False
    assert replacement["old_p23_runtime_lock_allowed_for_acquisition"] is False
    assert replacement["old_p23_host_attestation_allowed_for_acquisition"] is False
    assert replacement["lock_review_and_commit_before_trace_off_a"] is True


def test_p24_native_diagnostic_schema_and_acquisition_order_are_predeclared() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    diagnostic = contract["native_diagnostic_contract"]
    order = contract["ordered_execution"]

    assert diagnostic["schema_version"] == ("passive-muon-p24-executable-origin-diagnostic-v1")
    assert diagnostic["modes"] == ["baseline", "remediated"]
    assert diagnostic["status_values"] == ["passes", "fails", "error"]
    assert diagnostic["success_top_level_fields"] == [
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
    assert diagnostic["error_top_level_fields"] == [
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
    assert diagnostic["offline_sanitizer"] == {
        "runner": "scripts/sanitize_p24_executable_origin_diagnostic.py",
        "schema_version": ("passive-muon-p24-sanitized-executable-origin-diagnostic-v1"),
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
    freeze = (
        "review and commit the new image digest, runtime lock, host attestation, and complete "
        "source tree"
    )
    remediated = (
        "run and retain the native remediated executable-origin diagnostic against that exact "
        "committed lock and tree"
    )
    assert order.index(freeze) < order.index(remediated)
    assert order.index(remediated) < order.index(
        "review and commit both diagnostic bindings and sanitized copies before acquisition"
    )
    assert order.index("trace_off_a") < order.index("trace_off_b")
    assert order.index("verify_repeatability") < order.index(
        "trace_on only after exact repeatability passes"
    )
    assert order.index("verify_noninterference") < order.index(
        "aggregate_fidelity only after exact repeatability and noninterference pass"
    )


def test_every_contract_leaf_mutation_fails_reconstruction(tmp_path: Path) -> None:
    module = _module()
    canonical = json.loads(CONTRACT.read_text(encoding="utf-8"))
    path = tmp_path / "mutated-contract.json"

    for leaf_path in _leaf_paths(canonical):
        mutated = copy.deepcopy(canonical)
        _replace_at_path(mutated, leaf_path, _mutated_leaf(_at_path(mutated, leaf_path)))
        _write_json(path, mutated)
        assert module.reconstruct(path)["internally_consistent"] is False, leaf_path


def test_every_contract_mapping_rejects_extra_and_missing_fields(tmp_path: Path) -> None:
    module = _module()
    canonical = json.loads(CONTRACT.read_text(encoding="utf-8"))
    path = tmp_path / "mutated-schema.json"

    for mapping_path in _mapping_paths(canonical):
        mapping = _at_path(canonical, mapping_path)
        assert isinstance(mapping, dict) and mapping

        with_extra = copy.deepcopy(canonical)
        extra = _at_path(with_extra, mapping_path)
        assert isinstance(extra, dict)
        extra["unexpected_field"] = "unexpected"
        _write_json(path, with_extra)
        assert module.reconstruct(path)["internally_consistent"] is False, (
            "extra",
            mapping_path,
        )

        for key in mapping:
            with_missing = copy.deepcopy(canonical)
            missing = _at_path(with_missing, mapping_path)
            assert isinstance(missing, dict)
            del missing[key]
            _write_json(path, with_missing)
            assert module.reconstruct(path)["internally_consistent"] is False, (
                "missing",
                (*mapping_path, key),
            )


def test_p24_reconstructor_uses_only_the_standard_library() -> None:
    tree = ast.parse(RECONSTRUCTOR.read_text(encoding="utf-8"))
    top_level_imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            top_level_imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level_imports.add(node.module.split(".", 1)[0])
    assert top_level_imports <= {
        "__future__",
        "argparse",
        "ast",
        "collections",
        "functools",
        "hashlib",
        "json",
        "pathlib",
        "subprocess",
    }


def test_p24_reconstruction_cli_and_dedicated_workflow() -> None:
    completed = subprocess.run(
        [sys.executable, str(RECONSTRUCTOR)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["internally_consistent"] is True

    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "p24-cuda-executable-origin-hardening" in workflow
    assert "scripts/reconstruct_p24_cuda_executable_origin.py" in workflow
    assert "tests/test_p24_cuda_executable_origin_hardening.py" in workflow
    assert "tests/test_p24_runtime_recipe.py" in workflow
    assert "supplies no native diagnostic result" in workflow
