from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import TypeAlias

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/reconstruct_p29_permission_safe_bundle_localization_bridge_outcome.py"
OUTCOME = ROOT / "results/summaries/p29_permission_safe_bundle_localization_bridge_outcome.json"
RESULTS = ROOT / "results/summaries/P29_PERMISSION_SAFE_BUNDLE_LOCALIZATION_BRIDGE_RESULTS.md"
SOURCE_COMMIT = "ab62f683bf41ebb593a46267501e6aa6ba8e04c6"
PathPart: TypeAlias = str | int


def _module():
    specification = importlib.util.spec_from_file_location("p29_outcome_reconstruction", SCRIPT)
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


def _source_blob(path: str) -> str:
    return subprocess.run(
        ["git", "show", f"{SOURCE_COMMIT}:{path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def test_p29_terminal_post_receipt_seal_outcome_reconstructs() -> None:
    result = _module().reconstruct(OUTCOME)

    assert result["internally_consistent"] is True
    assert len(result["checks"]) == 11
    assert all(result["checks"].values())
    assert result["canonical_sha256"] == (
        "6d29601a71ce6e0c99bcc35497d328c8fdd2a3cfc135351c19876507b00a0f64"
    )
    assert "operator-retained seal facts" in result["claim_boundary"]


def test_exact_operator_captured_seal_error_is_bound_without_invented_time() -> None:
    outcome = json.loads(OUTCOME.read_text(encoding="utf-8"))
    seal = outcome["reviewed_orchestrator_seal"]
    stderr = (seal["exact_terminal_stderr"] + "\n").encode()

    assert seal["invocation_utc_independently_retained"] is False
    assert seal["invocation_utc"] is None
    assert seal["operator_capture_only"] is True
    assert seal["remote_stdout_file_retained"] is False
    assert seal["remote_stderr_file_retained"] is False
    assert len(stderr) == seal["exact_terminal_stderr_with_one_lf_byte_count"] == 51
    assert hashlib.sha256(stderr).hexdigest() == (
        "6862765b73c5e2df1bc16b28d9def47b1824f9295835f8d605a017edd879f92b"
    )


def test_source_receipt_passed_before_the_distinct_seal_stop() -> None:
    outcome = json.loads(OUTCOME.read_text(encoding="utf-8"))
    verification = outcome["source_verification"]
    receipt = outcome["source_receipt"]
    seal = outcome["reviewed_orchestrator_seal"]

    assert verification["exit_status"] == 0
    assert verification["source_control_checkout_created"] is True
    assert verification["source_control_clean_at_receipt_creation"] is True
    assert verification["source_receipt_created"] is True
    assert receipt["native_external_artifact"] is True
    assert receipt["committed_to_repository"] is False
    assert receipt["sha256"] == ("7e4b80c03ccc59049b78a1a553c087ec670ce5c4cbc5a3db6b1bda8db2b45065")
    assert receipt["reconstruction_publication_barrier"] == {
        "p29_contract": "15/15",
        "p28_terminal_outcome": "10/10",
        "p28_contract": "12/12",
        "p27_contract": "23/23",
    }
    assert seal["source_hash_matched"] is True
    assert seal["source_owner_matched"] is True
    assert seal["source_observed_mode"] == "0600"
    assert seal["source_required_mode"] == "0644"
    assert seal["source_mode_matched"] is False
    assert seal["destination_created"] is False


def test_frozen_umask_and_seal_predicate_explain_the_cross_phase_mismatch() -> None:
    verifier = _source_blob("scripts/verify_p29_control_bundle.py")
    runbook = _source_blob(
        "experiments/training/P29_PERMISSION_SAFE_BUNDLE_LOCALIZATION_BRIDGE_RUNBOOK.md"
    )
    tree_entry = subprocess.run(
        [
            "git",
            "ls-tree",
            SOURCE_COMMIT,
            "scripts/run_p29_permission_safe_bundle_localization_bridge.sh",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    assert "os.umask(0o077)" in verifier
    assert "!= (1000, 1000, 0o644)" in runbook
    assert tree_entry.startswith("100644 blob ")


def test_p29_stopped_before_sealed_launcher_attempt_container_or_cuda() -> None:
    outcome = json.loads(OUTCOME.read_text(encoding="utf-8"))
    state = outcome["retained_remote_state"]
    gates = outcome["gate_status"]

    assert state["localization_ledger_root"]["exact_child_inventory"] == ["deferred"]
    assert state["sealed_orchestrator_created"] is False
    assert state["authority_checkout_created"] is False
    assert state["attempt_root_created"] is False
    assert state["execution_snapshot_created"] is False
    assert state["prepare_runtime_invocation_token_created"] is False
    assert state["localization_invocation_token_created"] is False
    assert state["container_created"] is False
    assert state["runtime_lock_created"] is False
    assert state["host_attestation_created"] is False
    assert gates["gpu_access"] == "not_run"
    assert gates["cuda_initialization"] == "not_run"
    assert gates["deleted_mapping_localization"] == "not_run"
    assert gates["optimizer_steps"] == 0
    assert gates["candidate_observations"] == 0
    assert gates["training"] == "not_run"


def test_terminal_policy_and_fresh_p30_route_are_explicit() -> None:
    outcome = json.loads(OUTCOME.read_text(encoding="utf-8"))
    cause = outcome["root_cause"]
    policy = outcome["terminal_policy"]
    route = outcome["route"]

    assert cause["classification"] == (
        "cross_phase_reviewed_orchestrator_source_mode_contract_mismatch"
    )
    assert cause["source_mode_was_more_restrictive_than_required"] is True
    assert cause["cross_phase_mode_precondition_defect_observed"] is True
    assert cause["seal_failed_closed"] is True
    assert cause["security_authority_broadened"] is False
    assert policy["attempt_is_terminal"] is True
    assert policy["source_verifier_invocation_consumed"] is True
    assert policy["prepare_runtime_invocation_consumed"] is False
    assert policy["localization_invocation_consumed"] is False
    assert route["selected_terminal_route"] == "layout_bundle_or_reconstruction_failure"
    assert route["next_branch"] == "p30-umask-bound-control-seal"
    assert route["next_attempt_id"] == "20260906-03"
    assert route["attempt_20260906_02_may_be_reused"] is False
    assert route["retained_p29_state_may_be_cleaned_up"] is False
    assert route["scientific_acquisition_authorized"] is False


def test_every_p29_outcome_leaf_is_bound(tmp_path: Path) -> None:
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


def test_results_summary_preserves_terminal_scope_and_route() -> None:
    text = RESULTS.read_text(encoding="utf-8")

    assert "terminal after source-receipt creation" in text
    assert "P29 reviewed orchestrator source authority differs" in text
    assert "No GPU or CUDA operation ran" in text
    assert "`p30-umask-bound-control-seal`" in text
    assert "does not authorize scientific acquisition" in text
