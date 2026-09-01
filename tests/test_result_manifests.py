from __future__ import annotations

import csv
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "results/summaries/deficit_audit.json"
SWEEP_PATH = ROOT / "results/summaries/quadratic_lr_sweep.json"
HORIZON_PATH = ROOT / "results/summaries/quadratic_horizon_check.json"
WITNESS_PATH = ROOT / "results/summaries/canonical_witness.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_result_manifests_are_schema_versioned_and_hash_aligned() -> None:
    audit = _load(AUDIT_PATH)
    sweep = _load(SWEEP_PATH)
    horizon = _load(HORIZON_PATH)
    audit_hash = hashlib.sha256(AUDIT_PATH.read_bytes()).hexdigest()
    assert audit["schema_version"] == "passive-muon-deficit-audit-v5"
    assert sweep["schema_version"] == "passive-muon-quadratic-lr-sweep-v4"
    assert horizon["schema_version"] == "passive-muon-quadratic-horizon-check-v2"
    assert sweep["inputs"]["deficit_audit"]["sha256"] == audit_hash
    assert horizon["input"]["sha256"] == audit_hash


def test_result_source_snapshots_match_workspace_files() -> None:
    manifests = (_load(AUDIT_PATH), _load(SWEEP_PATH), _load(HORIZON_PATH))
    witness = _load(WITNESS_PATH)
    snapshots = [manifest["source_snapshot"] for manifest in manifests]
    snapshots.append(witness["experiment_provenance"]["source_snapshot"])
    for snapshot in snapshots:
        assert snapshot
        for relative_path, expected_hash in snapshot.items():
            path = ROOT / relative_path
            assert path.is_file()
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash


def test_result_manifests_record_clean_git_revisions() -> None:
    for path in (WITNESS_PATH, AUDIT_PATH, SWEEP_PATH, HORIZON_PATH):
        git = _load(path)["git"]
        assert len(git["sha"]) == 40
        assert git["dirty"] is False


def test_json_summary_rows_match_csv_exports() -> None:
    pairs = (
        (AUDIT_PATH, ROOT / "results/summaries/deficit_audit.csv"),
        (SWEEP_PATH, ROOT / "results/summaries/quadratic_lr_sweep.csv"),
        (HORIZON_PATH, ROOT / "results/summaries/quadratic_horizon_check.csv"),
    )
    for json_path, csv_path in pairs:
        expected_rows = _load(json_path)["rows"]
        with csv_path.open(newline="", encoding="utf-8") as stream:
            actual_rows = list(csv.DictReader(stream))
        assert len(actual_rows) == len(expected_rows)
        for expected, actual in zip(expected_rows, actual_rows, strict=True):
            assert set(actual) == set(expected)
            for key, value in expected.items():
                if value is None:
                    assert actual[key] == ""
                elif isinstance(value, bool):
                    assert actual[key] == str(value)
                elif isinstance(value, (int, float)):
                    assert float(actual[key]) == value
                else:
                    assert actual[key] == value


def test_figure_metadata_hashes_every_source() -> None:
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    for path in (
        ROOT / "results/figures/unit_spectrum_deficits.svg",
        ROOT / "results/figures/quadratic_normalized_upper_endpoints.svg",
    ):
        root = ET.parse(path).getroot()
        metadata_element = root.find("svg:metadata", namespace)
        assert metadata_element is not None and metadata_element.text
        metadata = json.loads(metadata_element.text)
        assert metadata["sources"]
        for source in metadata["sources"]:
            source_path = ROOT / source["path"]
            assert source_path.is_file()
            assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source["sha256"]


def test_gain_matched_controls_and_sampled_repairs_are_consistent() -> None:
    audit = _load(AUDIT_PATH)
    sweep = _load(SWEEP_PATH)
    audit_lookup = {(row["baseline"], row["steps"]): row for row in audit["rows"]}
    rows = sweep["rows"]
    keys = {(row["baseline"], row["steps"]) for row in rows}
    assert len(keys) == 24
    for key in keys:
        matching = [
            row
            for row in rows
            if (row["baseline"], row["steps"]) == key and row["normalization"] == "current_plus_eps"
        ]
        by_intervention = {row["intervention"]: row for row in matching}
        base = by_intervention["unrepaired"]
        gain_only = by_intervention["gain_only_unrepaired_control"]
        raw = next(row for label, row in by_intervention.items() if label.startswith("raw_grid"))
        gain_matched = next(
            row for label, row in by_intervention.items() if label.startswith("gain_matched_grid")
        )
        assert math.isclose(
            base["effective_zero_slope_gain"],
            gain_matched["effective_zero_slope_gain"],
            rel_tol=1e-14,
        )
        assert math.isclose(
            raw["effective_zero_slope_gain"],
            gain_only["effective_zero_slope_gain"],
            rel_tol=1e-14,
        )
        deficit = audit_lookup[key]["eps1_diagonal_combined_grid_deficit_lower_bound"]
        assert raw["repair_rho_inside_scale"] > deficit
        assert raw["sampled_diagonal_residual_after_scaled_shift"] == 0
        assert gain_matched["sampled_diagonal_residual_after_scaled_shift"] == 0


def test_classical_and_taylor_are_clean_normalization_attribution_controls() -> None:
    audit = _load(AUDIT_PATH)
    controls = [
        row for row in audit["rows"] if row["baseline"] in {"classical_cubic", "taylor_quintic"}
    ]
    assert len(controls) == 10
    assert all(row["unit_exact_local_deficit"] > 0 for row in controls)
    assert all(row["unit_fixed_scale_deficit_grid"] == 0 for row in controls)


def test_exact_global_deficit_claims_have_rational_antecedents() -> None:
    audit = _load(AUDIT_PATH)
    assert len(audit["rows"]) == 24
    assert all(row["exact_rational_slope_mismatch_at_3_5_4_5"] for row in audit["rows"])
    assert all(row["exact_global_deficit"] == "infinite" for row in audit["rows"])
