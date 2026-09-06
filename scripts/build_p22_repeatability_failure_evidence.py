#!/usr/bin/env python3
"""Build compact, path-sanitized evidence for the P22 repeatability stop.

The native trace manifests remain external because they contain redundant
absolute paths and per-step fields outside the repeatability decision.  This
script locks their exact SHA-256 values, preserves every unequal checked value,
and hashes the complete comparison projections so the verdict can be replayed
when the native files are supplied later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Final

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NATIVE_ROOT = Path("/private/tmp/optimizationml-p22-data")
DEFAULT_OUTPUT_ROOT = ROOT / "results/summaries"

TRACE_A_NAME: Final = "trace-off-a-manifest.json"
TRACE_B_NAME: Final = "trace-off-b-manifest.json"
REPEATABILITY_NAME: Final = "repeatability-report.json"
PREFLIGHT_NAME: Final = "p22_preflight_after_shape_commit.json"
FINEWEB_NAME: Final = "p22_fineweb_manifest.json"

EXPECTED_NATIVE_SHA256: Final = {
    TRACE_A_NAME: "325c0e03fdfc7c5cd249b9c10cbe0a3b0e8c94b4b953a3eea3bf17586b497e98",
    TRACE_B_NAME: "49e902dad2b7b5fdc8c2f97ed8498c79a743c1cbaf8eba6798fd80cc5e3346f8",
    REPEATABILITY_NAME: "1b6522653780cec8092e0907be672bb21897ce847c01baa3922233ed3c597923",
    PREFLIGHT_NAME: "81ea8900f66cef389217dee04146120774d7cfc5326229d202b606b3133e7aef",
    FINEWEB_NAME: "1dfbe2087fb61004c6c4ee938d1be53d769f2fa66303c2533e14acb9c1d5a96d",
}

STEP_FIELDS: Final = (
    "tokens_seen",
    "data_token_offset",
    "batch_sha256",
    "loss_tensor_sha256",
    "rng_before_step_sha256",
    "rng_after_step_sha256",
)
STATE_FIELDS: Final = ("model_state_sha256", "optimizer_state_sha256")
FINAL_FIELDS: Final = (*STATE_FIELDS, "rng_state_sha256")
RUN_FIELDS: Final = ("run_identity_sha256", "initial_state_sha256")
PAIR_CONTRACT_FIELDS: Final = (
    "schema_version",
    "trace_mode",
    "evidence_kind",
    "protocol_sha256",
    "accelerator_backend",
    "actual_accelerator_candidates",
    "shadow_update_applied",
    "capture_steps",
    "state_hash_schedule",
    "initial_state",
    "initial_state_sha256",
    "run_identity_sha256",
    "source_snapshot",
    "runtime",
    "data",
    "shape_inventory",
)
ANCHOR_STEPS: Final = (0, 1, 2, 22, 124, 255)
ACQUISITION_COMMIT: Final = "79f33ec0603edbb32432df4fb5f977d08fe8458a"
REPOSITORY_SNAPSHOT_PATHS: Final = {
    "p22_protocol": "experiments/training/p22_real_gradient_shadow_trace_protocol.json",
    "p22_trainer": "experiments/training/run_p22_real_gradient_shadow_trace.py",
    "p22_observer_adapter": "experiments/training/p22_observed_muon.py",
    "p22_observer": "src/passive_muon/p22_real_gradient_shadow_trace.py",
    "p22_shape_extension": "src/passive_muon/p22_scalable_sector_shield_extension.py",
    "pyproject.toml": "pyproject.toml",
    "uv.lock": "uv.lock",
}
UNRECORDED_RUNTIME_DEPENDENCY_PATHS: Final = (
    "experiments/training/p22_nanogpt_shadow_trace.py",
    "src/passive_muon/__init__.py",
    "src/passive_muon/p21_shadow_trace.py",
    "src/passive_muon/scalable_sector_shield.py",
)
EXTERNAL_SNAPSHOT_ENTRIES: Final = (
    "fineweb_manifest",
    "muon.py",
    "nanogpt_model.py",
    "nanogpt_train.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-root", type=Path, default=DEFAULT_NATIVE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sanitize_data_contract(value: object) -> object:
    """Remove machine-local paths before persisting a data-contract digest."""

    if isinstance(value, dict):
        sanitized: dict[str, object] = {}
        for key, item in value.items():
            if key in {"path", "manifest_path"}:
                sanitized[f"{key}_filename"] = Path(str(item)).name
                sanitized[f"{key}_removed"] = True
            else:
                sanitized[key] = _sanitize_data_contract(item)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_data_contract(item) for item in value]
    return value


def _pair_contract_projection(field: str, value: object) -> object:
    if field == "data":
        return _sanitize_data_contract(value)
    return value


def _git_blob(commit: str, repository_path: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{repository_path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return completed.stdout


def _retrospective_repository_attribution(
    left_snapshot: dict[str, str], right_snapshot: dict[str, str]
) -> dict[str, object]:
    if left_snapshot != right_snapshot:
        raise RuntimeError("trace-off source snapshots differ")
    expected_keys = set(REPOSITORY_SNAPSHOT_PATHS) | set(EXTERNAL_SNAPSHOT_ENTRIES)
    if set(left_snapshot) != expected_keys:
        raise RuntimeError("trace-off source snapshot keys changed")

    files: dict[str, dict[str, object]] = {}
    for snapshot_key, repository_path in REPOSITORY_SNAPSHOT_PATHS.items():
        native_sha256 = left_snapshot[snapshot_key]
        commit_blob_sha256 = hashlib.sha256(
            _git_blob(ACQUISITION_COMMIT, repository_path)
        ).hexdigest()
        files[snapshot_key] = {
            "repository_path": repository_path,
            "native_manifest_sha256": native_sha256,
            "commit_blob_sha256": commit_blob_sha256,
            "matches": native_sha256 == commit_blob_sha256,
        }
    if not all(record["matches"] for record in files.values()):
        raise RuntimeError("native source snapshot does not match acquisition commit")

    return {
        "kind": "retrospective_exact_source_hash_attribution",
        "acquisition_repository_commit": ACQUISITION_COMMIT,
        "native_manifest_recorded_repository_commit": False,
        "native_manifest_recorded_worktree_dirty_state": False,
        "verification_method": (
            "SHA-256 each git-show 79f33ec:<repository_path> byte stream and compare it "
            "with the corresponding hash stored in both native trace manifests"
        ),
        "all_recorded_repository_owned_source_hashes_match": True,
        "repository_owned_files": files,
        "unrecorded_runtime_dependencies": {
            repository_path: {
                "commit_blob_sha256": hashlib.sha256(
                    _git_blob(ACQUISITION_COMMIT, repository_path)
                ).hexdigest(),
                "native_manifest_sha256": None,
                "verified_as_executed_bytes": False,
                "reason": (
                    "imported by the execution graph but omitted from the native source "
                    "snapshot; the commit blob cannot prove the worktree bytes used"
                ),
            }
            for repository_path in UNRECORDED_RUNTIME_DEPENDENCY_PATHS
        },
        "external_snapshot_entries": {
            key: {
                "native_manifest_sha256": left_snapshot[key],
                "verified_against_acquisition_commit": False,
                "reason": "external acquisition artifact, not a tracked OptimizationML path",
            }
            for key in EXTERNAL_SNAPSHOT_ENTRIES
        },
        "caveat": (
            "The native manifests omitted the OptimizationML Git SHA and dirty state. This "
            "retrospective attribution proves that every recorded repository-owned source byte "
            "hash in both manifests matches commit 79f33ec exactly. The native source snapshot "
            "omitted four imported project dependencies, so this does not prove the complete "
            "executed project graph, does not reconstruct the acquisition worktree dirty state, "
            "and does not attribute external source files to that commit."
        ),
    }


def _load_locked(path: Path, expected_sha256: str) -> dict[str, Any]:
    observed = _sha256(path)
    if observed != expected_sha256:
        raise RuntimeError(f"native artifact hash changed for {path.name}: {observed}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"native artifact is not a JSON mapping: {path.name}")
    return payload


def _native_record(path: Path) -> dict[str, object]:
    return {
        "filename": path.name,
        "sha256": _sha256(path),
        "byte_count": path.stat().st_size,
        "storage": "external native evidence; not committed",
    }


def _trace_projection(trace: dict[str, Any], state_steps: set[int]) -> dict[str, object]:
    steps = []
    for step, record in enumerate(trace["steps"]):
        projected = {"step": step, **{field: record.get(field) for field in STEP_FIELDS}}
        if step in state_steps:
            projected.update({field: record.get(field) for field in STATE_FIELDS})
        steps.append(projected)
    return {
        **{field: trace.get(field) for field in RUN_FIELDS},
        "steps": steps,
        "final_state": {field: trace["final_state"].get(field) for field in FINAL_FIELDS},
    }


def _value_for_mismatch(trace: dict[str, Any], mismatch: dict[str, Any]) -> str | int | None:
    scope = mismatch["scope"]
    field = mismatch["field"]
    if scope == "run":
        return trace.get(field)
    if scope in {"step", "state"}:
        return trace["steps"][mismatch["step"]].get(field)
    if scope == "final":
        return trace["final_state"].get(field)
    raise RuntimeError(f"unknown mismatch scope: {scope}")


def _derive_mismatches(
    left: dict[str, Any], right: dict[str, Any], state_steps: set[int]
) -> list[dict[str, object]]:
    mismatches: list[dict[str, object]] = []
    for field in RUN_FIELDS:
        if left.get(field) != right.get(field):
            mismatches.append({"scope": "run", "field": field})
    for step, (left_record, right_record) in enumerate(
        zip(left["steps"], right["steps"], strict=True)
    ):
        for field in STEP_FIELDS:
            if left_record.get(field) != right_record.get(field):
                mismatches.append({"scope": "step", "step": step, "field": field})
        if step in state_steps:
            for field in STATE_FIELDS:
                if left_record.get(field) != right_record.get(field):
                    mismatches.append({"scope": "state", "step": step, "field": field})
    for field in FINAL_FIELDS:
        if left["final_state"].get(field) != right["final_state"].get(field):
            mismatches.append({"scope": "final", "field": field})
    return mismatches


def _anchor(trace: dict[str, Any], step: int, state_steps: set[int]) -> dict[str, object]:
    record = trace["steps"][step]
    result = {"step": step, **{field: record.get(field) for field in STEP_FIELDS}}
    if step in state_steps:
        result.update({field: record.get(field) for field in STATE_FIELDS})
    return result


def _sanitize_fineweb(native: dict[str, Any], native_path: Path) -> dict[str, object]:
    dataset = native["dataset"]
    tokenizer = native["tokenizer"]
    outputs = native["outputs"]
    return {
        "schema_version": "passive-muon-p22-fineweb-materialization-evidence-v1",
        "native_artifact": _native_record(native_path),
        "status": native["status"],
        "dataset": {
            "repository": dataset["repository"],
            "config": dataset["config"],
            "split": dataset["split"],
            "revision": dataset["revision"],
            "selection": dataset["selection"],
            "source_responses": [
                {
                    "artifact_id": (
                        f"fineweb-response-{entry['row_start']:06d}-"
                        f"{entry['row_end_inclusive']:06d}"
                    ),
                    **{key: value for key, value in entry.items() if key != "path"},
                }
                for entry in dataset["source_shards"]
            ],
        },
        "tokenizer": {
            **{key: value for key, value in tokenizer.items() if key != "files"},
            "files": {
                filename: {
                    "artifact_id": f"gpt2-{filename}",
                    "sha256": entry["sha256"],
                    "byte_count": entry["byte_count"],
                }
                for filename, entry in tokenizer["files"].items()
            },
        },
        "preprocessor": {
            "repository_path": "experiments/training/materialize_p22_fineweb.py",
            "sha256": native["preprocessor"]["sha256"],
            "byte_count": native["preprocessor"]["byte_count"],
            "normalization": native["preprocessor"]["normalization"],
            "document_field": native["preprocessor"]["document_field"],
        },
        "outputs": {
            filename: {
                "artifact_id": f"external-{filename}",
                **{key: value for key, value in entry.items() if key != "path"},
                "committed": False,
            }
            for filename, entry in outputs.items()
        },
        "claim_boundary": (
            "Hashes and selection metadata are retained; source responses, tokenizer assets, "
            "and uint16 token binaries remain external and are not embedded here."
        ),
    }


def _sanitize_preflight(native: dict[str, Any], native_path: Path) -> dict[str, object]:
    checks = native["checks"]
    fineweb = checks["fineweb"]
    instrumentation = checks["instrumentation"]
    muon = checks["muon"]
    return {
        "schema_version": "passive-muon-p22-preflight-evidence-v1",
        "native_artifact": _native_record(native_path),
        "status": native["status"],
        "blockers": native["blockers"],
        "real_run_executed": native["real_run_executed"],
        "protocol": {
            "repository_path": native["protocol_path"],
            "sha256": native["protocol_sha256"],
        },
        "shape_inventory": native["shape_inventory"],
        "checks": {
            "accelerator": checks["accelerator"],
            "fineweb": {
                "ready": fineweb["ready"],
                "blockers": fineweb["blockers"],
                "native_manifest_sha256": fineweb["manifest_sha256"],
                "artifact_hashes": [
                    {key: item[key] for key in ("sha256", "byte_count")}
                    for item in fineweb["artifacts"]
                ],
            },
            "instrumentation": {
                "ready": instrumentation["ready"],
                "blockers": instrumentation["blockers"],
                "repository_path": "experiments/training/p22_observed_muon.py",
                "sha256": instrumentation["sha256"],
                "byte_count": instrumentation["byte_count"],
            },
            "muon": {
                "ready": muon["ready"],
                "blockers": muon["blockers"],
                "revision": muon["revision"],
                "sha256": muon["sha256"],
                "external_path_removed": True,
            },
            "nanogpt": checks["nanogpt"],
        },
        "claim_boundary": native["claim_boundary"],
    }


def _repeatability_evidence(
    *,
    left: dict[str, Any],
    right: dict[str, Any],
    repeatability: dict[str, Any],
    left_path: Path,
    right_path: Path,
    report_path: Path,
    preflight_sha256: str,
    fineweb_sha256: str,
    native_run_directory: Path,
    repository_attribution: dict[str, object],
) -> dict[str, object]:
    pair_contract_checks: dict[str, dict[str, object]] = {}
    for field in PAIR_CONTRACT_FIELDS:
        left_value = left.get(field)
        right_value = right.get(field)
        left_projection = _pair_contract_projection(field, left_value)
        right_projection = _pair_contract_projection(field, right_value)
        pair_contract_checks[field] = {
            "equal": left_value == right_value,
            "stored_projection": (
                "machine-local paths replaced by filenames" if field == "data" else "complete"
            ),
            "trace_off_a_sha256": _canonical_sha256(left_projection),
            "trace_off_b_sha256": _canonical_sha256(right_projection),
        }
    if not all(record["equal"] for record in pair_contract_checks.values()):
        raise RuntimeError("trace-off pair does not have an identical execution contract")

    state_steps = set(repeatability["state_checkpoint_steps"])
    derived = _derive_mismatches(left, right, state_steps)
    if derived != repeatability["mismatches"]:
        raise RuntimeError("native repeatability report does not match the two trace projections")
    witnesses = [
        {
            **mismatch,
            "trace_off_a": _value_for_mismatch(left, mismatch),
            "trace_off_b": _value_for_mismatch(right, mismatch),
        }
        for mismatch in derived
    ]
    if any(item["trace_off_a"] == item["trace_off_b"] for item in witnesses):
        raise RuntimeError("repeatability mismatch witness contains equal values")

    left_projection = _trace_projection(left, state_steps)
    right_projection = _trace_projection(right, state_steps)
    native_names = {path.name for path in native_run_directory.iterdir() if path.is_file()}
    forbidden_downstream = {
        "trace_on_manifest": "trace-on-manifest.json",
        "noninterference_report": "noninterference-report.json",
        "aggregate_trace": "p22_real_gradient_shadow_trace.json",
    }
    if any(filename in native_names for filename in forbidden_downstream.values()):
        raise RuntimeError(
            "downstream evidence exists despite the failed repeatability prerequisite"
        )

    field_counts = Counter(item["field"] for item in derived)
    scope_counts = Counter(item["scope"] for item in derived)
    return {
        "schema_version": "passive-muon-p22-repeatability-failure-evidence-v1",
        "verdict": {
            "baseline_repeatability_passes": False,
            "comparison": repeatability["comparison"],
            "mismatch_count": len(derived),
            "first_mismatch": derived[0],
            "interpretation": (
                "The two pinned trace-off accelerator executions diverged bitwise. The frozen "
                "protocol therefore stopped before trace-on acquisition. This is failure "
                "evidence for this execution pair, not a theorem of backend-wide nondeterminism."
            ),
        },
        "native_artifacts": {
            "trace_off_a": _native_record(left_path),
            "trace_off_b": _native_record(right_path),
            "repeatability_report": _native_record(report_path),
        },
        "linked_native_evidence": {
            "preflight_sha256": preflight_sha256,
            "fineweb_manifest_sha256": fineweb_sha256,
        },
        "repository_provenance": repository_attribution,
        "run_contract": {
            "schema_version": left["schema_version"],
            "trace_mode": left["trace_mode"],
            "evidence_kind": left["evidence_kind"],
            "protocol_sha256": left["protocol_sha256"],
            "accelerator_backend": left["accelerator_backend"],
            "actual_accelerator_candidates": left["actual_accelerator_candidates"],
            "shadow_update_applied": left["shadow_update_applied"],
            "capture_steps": left["capture_steps"],
            "state_hash_schedule": left["state_hash_schedule"],
            "runtime": left["runtime"],
            "source_snapshot": left["source_snapshot"],
            "shape_inventory": {
                "parameter_count": left["shape_inventory"]["parameter_count"],
                "total_numel": left["shape_inventory"]["total_numel"],
            },
        },
        "run_instances": {
            "trace_off_a": {
                **{field: left[field] for field in RUN_FIELDS},
                "elapsed_seconds": left["elapsed_seconds"],
                "comparison_projection_sha256": _canonical_sha256(left_projection),
            },
            "trace_off_b": {
                **{field: right[field] for field in RUN_FIELDS},
                "elapsed_seconds": right["elapsed_seconds"],
                "comparison_projection_sha256": _canonical_sha256(right_projection),
            },
        },
        "comparison_reconstruction": {
            "pair_contract_fields": list(PAIR_CONTRACT_FIELDS),
            "pair_contract_checks": pair_contract_checks,
            "all_pair_contract_fields_equal": True,
            "checked_step_count": repeatability["checked_step_count"],
            "checked_run_fields": list(RUN_FIELDS),
            "checked_step_fields": repeatability["checked_step_fields"],
            "checked_state_fields": repeatability["checked_state_fields"],
            "checked_final_fields": list(FINAL_FIELDS),
            "state_checkpoint_steps": repeatability["state_checkpoint_steps"],
            "trace_off_a_projection_sha256": _canonical_sha256(left_projection),
            "trace_off_b_projection_sha256": _canonical_sha256(right_projection),
            "mismatch_count_by_field": dict(sorted(field_counts.items())),
            "mismatch_count_by_scope": dict(sorted(scope_counts.items())),
            "mismatch_witnesses": witnesses,
            "anchor_steps": {
                "trace_off_a": [_anchor(left, step, state_steps) for step in ANCHOR_STEPS],
                "trace_off_b": [_anchor(right, step, state_steps) for step in ANCHOR_STEPS],
            },
            "initial_state": {
                "trace_off_a": left["initial_state"],
                "trace_off_b": right["initial_state"],
            },
            "final_state": {
                "trace_off_a": left["final_state"],
                "trace_off_b": right["final_state"],
            },
        },
        "downstream_absence": {
            name: {
                "status": "not_present_in_frozen_native_run_directory",
                "expected_native_filename": filename,
                "absent_from_native_run_directory": filename not in native_names,
                "reason": "baseline repeatability prerequisite failed; protocol barred creation",
            }
            for name, filename in forbidden_downstream.items()
        },
        "path_sanitization": {
            "native_absolute_paths_retained": False,
            "personal_home_paths_retained": False,
            "external_files_identified_by_filename_and_sha256": True,
        },
        "claim_boundary": (
            "This artifact establishes an exact repeatability failure for the two hash-locked "
            "trace-off executions. It does not establish observer noninterference, candidate "
            "fidelity, a backend-wide nondeterminism claim, or any neural-training theorem."
        ),
        "verifier_qualification": (
            "The native project comparator checked the opaque run-identity digest but did not "
            "separately bind or recompute every runtime/source mapping. This canonical builder "
            "independently requires exact equality of the listed pair-contract fields and stores "
            "their canonical hashes. Future acceptance acquisition should harden the native "
            "verifier before use."
        ),
    }


def _write(path: Path, payload: dict[str, object]) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    forbidden = ("/private/tmp/", "/Users/", "\\Users\\")
    if any(fragment in rendered for fragment in forbidden):
        raise RuntimeError(f"path sanitization failed for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def main() -> None:
    args = parse_args()
    native_root = args.native_root.resolve()
    run_root = native_root / "runs"
    paths = {
        TRACE_A_NAME: run_root / TRACE_A_NAME,
        TRACE_B_NAME: run_root / TRACE_B_NAME,
        REPEATABILITY_NAME: run_root / REPEATABILITY_NAME,
        PREFLIGHT_NAME: native_root / PREFLIGHT_NAME,
        FINEWEB_NAME: native_root / "materialized" / FINEWEB_NAME,
    }
    native = {
        name: _load_locked(path, EXPECTED_NATIVE_SHA256[name]) for name, path in paths.items()
    }

    fineweb = _sanitize_fineweb(native[FINEWEB_NAME], paths[FINEWEB_NAME])
    preflight = _sanitize_preflight(native[PREFLIGHT_NAME], paths[PREFLIGHT_NAME])
    repeatability = _repeatability_evidence(
        left=native[TRACE_A_NAME],
        right=native[TRACE_B_NAME],
        repeatability=native[REPEATABILITY_NAME],
        left_path=paths[TRACE_A_NAME],
        right_path=paths[TRACE_B_NAME],
        report_path=paths[REPEATABILITY_NAME],
        preflight_sha256=EXPECTED_NATIVE_SHA256[PREFLIGHT_NAME],
        fineweb_sha256=EXPECTED_NATIVE_SHA256[FINEWEB_NAME],
        native_run_directory=run_root,
        repository_attribution=_retrospective_repository_attribution(
            native[TRACE_A_NAME]["source_snapshot"],
            native[TRACE_B_NAME]["source_snapshot"],
        ),
    )
    outputs = {
        "p22_fineweb_materialization_evidence.json": fineweb,
        "p22_real_gradient_preflight_evidence.json": preflight,
        "p22_repeatability_failure_evidence.json": repeatability,
    }
    for filename, payload in outputs.items():
        _write(args.output_root / filename, payload)
    print(
        json.dumps(
            {
                "outputs": {filename: _sha256(args.output_root / filename) for filename in outputs},
                "verdict": "baseline_repeatability_failed",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
