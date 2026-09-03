from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "mixed_precision" / "run_mixed_precision_falsification.py"
RESULT = ROOT / "results" / "summaries" / "mixed_precision_falsification.json"
SPEC = importlib.util.spec_from_file_location("mixed_precision_falsification", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("failed to load the P8 mixed-precision falsification experiment")
EXPERIMENT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EXPERIMENT
SPEC.loader.exec_module(EXPERIMENT)

FalsificationConfig = EXPERIMENT.FalsificationConfig
SCHEMA_VERSION = EXPERIMENT.SCHEMA_VERSION
build_grid = EXPERIMENT.build_grid
canonical_json = EXPERIMENT.canonical_json
run_study = EXPERIMENT.run_study


def test_default_grid_is_complete_and_has_zero_sampled_candidates() -> None:
    payload = run_study(FalsificationConfig())

    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["summary"]["case_count"] == 82
    assert payload["grid"]["ordered_case_count"] == 82
    assert payload["summary"]["candidate_violation_count"] == 0
    assert payload["summary"]["nonfinite_count"] == 0
    assert 0.0 <= payload["summary"]["maximum_affine_bound_ratio"] <= 1.0
    assert 0.0 <= payload["summary"]["maximum_all_real_adapter_affine_bound_ratio"] <= 1.0
    assert payload["summary"]["maximum_observed_error_frobenius"] >= 0.0
    assert len(payload["grid"]["grid_sha256_from_labels_families_and_input_bits"]) == 64
    assert payload["operator"]["target_evaluation_dtype"] == "float64"
    assert payload["operator"]["target_evaluation_is_exact"] is False
    assert payload["operator"]["affine_slope_exact"] == "11/100000"
    assert payload["operator"]["affine_intercept_exact"] == "347/100"
    assert payload["operator"]["all_real_adapter_affine_slope_exact"] == "1/5000"
    assert payload["operator"]["all_real_adapter_affine_intercept_exact"] == "347/100"
    assert "do not prove" in payload["claim_scope"]["qualification"]
    assert "not exact arithmetic" in payload["claim_scope"]["reference_qualification"]
    assert "conservative" in payload["claim_scope"]["intercept_note"]
    assert "Primary ratios" in payload["claim_scope"]["adapter_note"]

    labels = {case["label"] for case in payload["cases"]}
    families = payload["grid"]["family_counts"]
    assert {
        "zero",
        "subnormal_signed",
        "half_threshold_dense",
        "rank_one_floor",
        "three_four_unit_direction",
    }.issubset(labels)
    assert families["rank_one"] == 5
    assert families["repeated_singular"] == 8
    assert families["signs_permutations"] == 8
    assert families["seeded_dense_log"] == 12
    assert families["seeded_near_floor"] == 12
    assert families["seeded_rank_one"] == 12
    assert families["seeded_repeated_singular"] == 12


def test_seeded_grid_and_numerical_results_are_deterministic_ignoring_provenance() -> None:
    config = replace(FalsificationConfig(), random_cases_per_family=3)

    first = run_study(config)
    second = run_study(config)

    assert first["config"] == second["config"]
    assert first["grid"] == second["grid"]
    assert first["summary"] == second["summary"]
    assert first["cases"] == second["cases"]
    assert all(len(case["input"]["float32_bits_hex_row_major"]) == 4 for case in first["cases"])


def test_seed_changes_random_grid_but_not_explicit_adversarial_prefix() -> None:
    left_config = replace(FalsificationConfig(), seed=11, random_cases_per_family=2)
    right_config = replace(FalsificationConfig(), seed=12, random_cases_per_family=2)
    left = build_grid(left_config)
    right = build_grid(right_config)

    assert len(left) == len(right) == 42
    assert [label for label, _family, _matrix in left[:34]] == [
        label for label, _family, _matrix in right[:34]
    ]
    assert any(
        not (left_case[2] == right_case[2]).all()
        for left_case, right_case in zip(left[34:], right[34:], strict=True)
    )


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"matrix_shape": (2, 3)}, "shape"),
        ({"safe_max_abs_power": 115}, "safe_max_abs_power"),
        ({"random_cases_per_family": 0}, "positive"),
        ({"random_exponent_min": -121}, "at least"),
        ({"random_exponent_max": 117}, "safe range"),
        ({"random_exponent_min": 5, "random_exponent_max": 4}, "empty"),
    ],
)
def test_invalid_grid_config_is_rejected(change: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        replace(FalsificationConfig(), **change)


def test_cli_emits_and_writes_canonical_json_with_full_provenance(tmp_path: Path) -> None:
    output = tmp_path / "p8-falsification.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--seed",
            "17",
            "--random-cases-per-family",
            "2",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert output.read_text() == canonical_json(payload)
    assert payload["config"]["seed"] == 17
    assert payload["config"]["random_cases_per_family"] == 2
    assert payload["summary"]["case_count"] == 42
    assert payload["summary"]["candidate_violation_count"] == 0
    provenance = payload["experiment_provenance"]
    assert len(provenance["git"]["sha"]) == 40
    assert isinstance(provenance["git"]["dirty"], bool)
    assert provenance["hardware"]["torch_device"] == "cpu"
    assert provenance["software"]["numpy"]
    assert provenance["software"]["torch"]
    assert set(provenance["source_snapshot"]) == {
        "experiments/mixed_precision/run_mixed_precision_falsification.py",
        "src/passive_muon/mixed_precision.py",
        "src/passive_muon/mixed_precision_certificate.py",
        "src/passive_muon/specs.py",
        "src/passive_muon/structure_aware_stability.py",
        "tests/test_mixed_precision.py",
        "tests/test_mixed_precision_experiment.py",
    }


def test_checked_result_matches_the_locked_default_grid_and_outcome() -> None:
    recorded = json.loads(RESULT.read_text())
    replay = run_study(FalsificationConfig())

    assert recorded["schema_version"] == SCHEMA_VERSION
    assert recorded["claim_scope"] == replay["claim_scope"]
    assert recorded["config"] == replay["config"]
    assert recorded["grid"] == replay["grid"]
    assert recorded["summary"] == replay["summary"]
    assert recorded["cases"] == replay["cases"]
    assert recorded["operator"]["target"] == replay["operator"]["target"]
    assert recorded["operator"]["target_evaluation_dtype"] == "float64"
    assert recorded["operator"]["target_evaluation_is_exact"] is False
    for field in (
        "affine_slope_exact",
        "affine_intercept_exact",
        "all_real_adapter_affine_slope_exact",
        "all_real_adapter_affine_intercept_exact",
    ):
        assert recorded["operator"][field] == replay["operator"][field]
    assert (
        recorded["experiment_provenance"]["source_snapshot"]
        == replay["experiment_provenance"]["source_snapshot"]
    )
    assert recorded["summary"]["candidate_violation_count"] == 0
    assert recorded["summary"]["nonfinite_count"] == 0
    assert all(not case["candidate_violation"] for case in recorded["cases"])
