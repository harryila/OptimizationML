from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_ROOT = ROOT / "results/summaries"
FINEWEB_PATH = SUMMARY_ROOT / "p22_fineweb_materialization_evidence.json"
PREFLIGHT_PATH = SUMMARY_ROOT / "p22_real_gradient_preflight_evidence.json"
FAILURE_PATH = SUMMARY_ROOT / "p22_repeatability_failure_evidence.json"

CANONICAL_SHA256 = {
    FINEWEB_PATH.name: "3bc7e352a9925cd970e7439c452f17bac60f26c1391210762695054d543055a6",
    PREFLIGHT_PATH.name: "0b07d72cfd53eb1168870c63394543ae0060189636b9ac10f1923b447a3c6481",
    FAILURE_PATH.name: "f335601ae97734eedc193fe25a480a11c6d804dcb1ed0cff246006d54847e0d1",
}
NATIVE_SHA256 = {
    "trace-off-a-manifest.json": (
        "325c0e03fdfc7c5cd249b9c10cbe0a3b0e8c94b4b953a3eea3bf17586b497e98"
    ),
    "trace-off-b-manifest.json": (
        "49e902dad2b7b5fdc8c2f97ed8498c79a743c1cbaf8eba6798fd80cc5e3346f8"
    ),
    "repeatability-report.json": (
        "1b6522653780cec8092e0907be672bb21897ce847c01baa3922233ed3c597923"
    ),
    "p22_preflight_after_shape_commit.json": (
        "81ea8900f66cef389217dee04146120774d7cfc5326229d202b606b3133e7aef"
    ),
    "p22_fineweb_manifest.json": (
        "1dfbe2087fb61004c6c4ee938d1be53d769f2fa66303c2533e14acb9c1d5a96d"
    ),
}


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_canonical_failure_evidence_bytes_are_locked_and_path_sanitized() -> None:
    for path in (FINEWEB_PATH, PREFLIGHT_PATH, FAILURE_PATH):
        payload = path.read_bytes()
        assert hashlib.sha256(payload).hexdigest() == CANONICAL_SHA256[path.name]
        rendered = payload.decode()
        assert "/private/tmp/" not in rendered
        assert "/Users/" not in rendered
        assert "\\Users\\" not in rendered


def test_fineweb_evidence_preserves_external_hashes_without_binaries() -> None:
    evidence = _load(FINEWEB_PATH)
    assert evidence["native_artifact"]["sha256"] == NATIVE_SHA256["p22_fineweb_manifest.json"]
    assert evidence["status"] == "materialized"
    dataset = evidence["dataset"]
    assert dataset["repository"] == "HuggingFaceFW/fineweb"
    assert dataset["config"] == "sample-10BT"
    assert dataset["split"] == "train"
    assert dataset["revision"] == "9bb295ddab0e05d785b879661af7260fed5140fc"
    assert dataset["selection"] == {
        "canonical_consumed_rows_sha256": (
            "c5f2c9074a03949d180f668dc2c66cfffed8d4a340797d8e491cad9e63b5f31f"
        ),
        "canonical_framing": (
            "sha256(domain || repeated le64(row_id) || le64(utf8_bytes) || utf8(text))"
        ),
        "first_row_id": 0,
        "last_row_id_inclusive": 245,
        "ordering": "Increasing row index at the pinned revision; complete untruncated text only",
        "row_count": 246,
        "train_boundary": {"row_id": 196, "token_offset_exclusive": 114},
        "validation_boundary": {"row_id": 245, "token_offset_exclusive": 851},
    }
    assert [item["sha256"] for item in dataset["source_responses"]] == [
        "dd8e3594cfc2af77a1dac37950b025668fc7052affb16209d1a42ba284392cf4",
        "f3c98f25e86159333006c6649ae65e7910c1cc3f099d61dab467219d2d86d072",
        "134dd2cf2af41a8276d668113beb8fcf629698121d6f6d01c87c40b5efecd7c7",
    ]
    assert evidence["tokenizer"]["files"]["vocab.bpe"]["sha256"] == (
        "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5"
    )
    assert evidence["tokenizer"]["files"]["encoder.json"]["sha256"] == (
        "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783"
    )
    assert evidence["preprocessor"]["sha256"] == (
        "e352c3536baa74bc560af4a589b3bc1b216f4bda8e4cd554e16821b02977525e"
    )
    assert evidence["outputs"]["train.bin"] == {
        "artifact_id": "external-train.bin",
        "byte_count": 262_144,
        "committed": False,
        "dtype": "little-endian uint16",
        "sha256": "726a7684a785c134d2db92568ee5d5849354b0a5e7f27ad9ae8d1e283ea89cba",
        "token_count": 131_072,
    }
    assert evidence["outputs"]["val.bin"]["committed"] is False
    assert evidence["outputs"]["val.bin"]["sha256"] == (
        "498fb38e7457fd3131fbebba3d7dcff70575dfc5f932e4e108ec0fe10405c89f"
    )


def test_preflight_evidence_was_ready_before_native_runs() -> None:
    evidence = _load(PREFLIGHT_PATH)
    assert (
        evidence["native_artifact"]["sha256"]
        == NATIVE_SHA256["p22_preflight_after_shape_commit.json"]
    )
    assert evidence["status"] == "ready"
    assert evidence["blockers"] == []
    assert evidence["real_run_executed"] is False
    assert evidence["shape_inventory"]["complete"] is True
    assert evidence["shape_inventory"]["parameter_count_fraction"] == "48/48"
    assert all(check["ready"] for check in evidence["checks"].values())
    assert (
        evidence["checks"]["fineweb"]["native_manifest_sha256"]
        == NATIVE_SHA256["p22_fineweb_manifest.json"]
    )


def test_repeatability_verdict_reconstructs_from_unequal_witnesses() -> None:
    evidence = _load(FAILURE_PATH)
    verdict = evidence["verdict"]
    reconstruction = evidence["comparison_reconstruction"]
    witnesses = reconstruction["mismatch_witnesses"]

    assert verdict["baseline_repeatability_passes"] is False
    assert verdict["comparison"] == "bitwise_exact_no_tolerances"
    assert verdict["mismatch_count"] == len(witnesses) == 303
    assert verdict["first_mismatch"] == {
        "scope": "state",
        "step": 0,
        "field": "model_state_sha256",
    }
    assert all(item["trace_off_a"] != item["trace_off_b"] for item in witnesses)
    assert Counter(item["field"] for item in witnesses) == {
        "loss_tensor_sha256": 253,
        "model_state_sha256": 25,
        "optimizer_state_sha256": 25,
    }
    assert Counter(item["scope"] for item in witnesses) == {
        "step": 253,
        "state": 48,
        "final": 2,
    }
    assert reconstruction["mismatch_count_by_field"] == dict(
        sorted(Counter(item["field"] for item in witnesses).items())
    )
    assert reconstruction["mismatch_count_by_scope"] == dict(
        sorted(Counter(item["scope"] for item in witnesses).items())
    )
    assert reconstruction["checked_step_count"] == 256
    assert reconstruction["state_checkpoint_steps"] == [
        *range(0, 8),
        *range(124, 132),
        *range(248, 256),
    ]


def test_native_hashes_and_run_identity_are_preserved_exactly() -> None:
    evidence = _load(FAILURE_PATH)
    native = evidence["native_artifacts"]
    assert native["trace_off_a"]["sha256"] == NATIVE_SHA256[TRACE_A := "trace-off-a-manifest.json"]
    assert native["trace_off_a"]["filename"] == TRACE_A
    assert native["trace_off_b"]["sha256"] == NATIVE_SHA256[TRACE_B := "trace-off-b-manifest.json"]
    assert native["trace_off_b"]["filename"] == TRACE_B
    assert native["repeatability_report"]["sha256"] == NATIVE_SHA256["repeatability-report.json"]
    runs = evidence["run_instances"]
    assert runs["trace_off_a"]["run_identity_sha256"] == runs["trace_off_b"]["run_identity_sha256"]
    assert (
        runs["trace_off_a"]["initial_state_sha256"] == runs["trace_off_b"]["initial_state_sha256"]
    )
    assert (
        runs["trace_off_a"]["comparison_projection_sha256"]
        != runs["trace_off_b"]["comparison_projection_sha256"]
    )
    assert evidence["linked_native_evidence"] == {
        "preflight_sha256": NATIVE_SHA256["p22_preflight_after_shape_commit.json"],
        "fineweb_manifest_sha256": NATIVE_SHA256["p22_fineweb_manifest.json"],
    }

    reconstruction = evidence["comparison_reconstruction"]
    assert reconstruction["all_pair_contract_fields_equal"] is True
    assert set(reconstruction["pair_contract_fields"]) == {
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
    }
    assert all(
        check["equal"] and check["trace_off_a_sha256"] == check["trace_off_b_sha256"]
        for check in reconstruction["pair_contract_checks"].values()
    )
    assert reconstruction["pair_contract_checks"]["data"]["stored_projection"] == (
        "machine-local paths replaced by filenames"
    )
    assert all(
        check["stored_projection"] == "complete"
        for field, check in reconstruction["pair_contract_checks"].items()
        if field != "data"
    )
    assert "did not separately bind" in evidence["verifier_qualification"]


def test_retrospective_acquisition_commit_attribution_is_exact_and_caveated() -> None:
    evidence = _load(FAILURE_PATH)
    provenance = evidence["repository_provenance"]
    assert provenance["kind"] == "retrospective_exact_source_hash_attribution"
    assert provenance["acquisition_repository_commit"] == (
        "79f33ec0603edbb32432df4fb5f977d08fe8458a"
    )
    assert provenance["native_manifest_recorded_repository_commit"] is False
    assert provenance["native_manifest_recorded_worktree_dirty_state"] is False
    assert provenance["all_recorded_repository_owned_source_hashes_match"] is True

    for record in provenance["repository_owned_files"].values():
        blob = subprocess.run(
            [
                "git",
                "show",
                f"{provenance['acquisition_repository_commit']}:{record['repository_path']}",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        observed = hashlib.sha256(blob).hexdigest()
        assert record["matches"] is True
        assert record["native_manifest_sha256"] == record["commit_blob_sha256"] == observed

    assert set(provenance["external_snapshot_entries"]) == {
        "fineweb_manifest",
        "muon.py",
        "nanogpt_model.py",
        "nanogpt_train.py",
    }
    assert all(
        item["verified_against_acquisition_commit"] is False
        for item in provenance["external_snapshot_entries"].values()
    )
    assert set(provenance["unrecorded_runtime_dependencies"]) == {
        "experiments/training/p22_nanogpt_shadow_trace.py",
        "src/passive_muon/__init__.py",
        "src/passive_muon/p21_shadow_trace.py",
        "src/passive_muon/scalable_sector_shield.py",
    }
    assert all(
        item["native_manifest_sha256"] is None and item["verified_as_executed_bytes"] is False
        for item in provenance["unrecorded_runtime_dependencies"].values()
    )
    caveat = provenance["caveat"]
    assert "omitted the OptimizationML Git SHA and dirty state" in caveat
    assert "omitted four imported project dependencies" in caveat
    assert "does not reconstruct the acquisition worktree dirty state" in caveat


def test_trace_on_and_downstream_reports_are_explicitly_absent() -> None:
    evidence = _load(FAILURE_PATH)
    absence = evidence["downstream_absence"]
    assert set(absence) == {
        "trace_on_manifest",
        "noninterference_report",
        "aggregate_trace",
    }
    assert all(
        item["status"] == "not_present_in_frozen_native_run_directory" for item in absence.values()
    )
    assert all(item["absent_from_native_run_directory"] is True for item in absence.values())
    assert all(
        item["reason"] == "baseline repeatability prerequisite failed; protocol barred creation"
        for item in absence.values()
    )
    assert not (SUMMARY_ROOT / "p22_trace_on_manifest.json").exists()
    assert not (SUMMARY_ROOT / "p22_noninterference_report.json").exists()
    assert not (SUMMARY_ROOT / "p22_real_gradient_shadow_trace.json").exists()
