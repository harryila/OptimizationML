from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from passive_muon.equivariant_resolvent_solver import EquivariantResolventSolverConfig

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "resolvent" / "run_p17_shape_preserving_study.py"
CANONICAL = ROOT / "results" / "summaries" / "p17_shape_preserving_study.json"
SPEC = importlib.util.spec_from_file_location("p17_shape_preserving_study", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("failed to load the P17 shape-preserving study")
EXPERIMENT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EXPERIMENT
SPEC.loader.exec_module(EXPERIMENT)

FULL_STEP_DESIGN = EXPERIMENT.FULL_STEP_DESIGN
OPERATING_ANNULUS_INNER_RADIUS = EXPERIMENT.OPERATING_ANNULUS_INNER_RADIUS
OPERATING_ANNULUS_OUTER_RADIUS = EXPERIMENT.OPERATING_ANNULUS_OUTER_RADIUS
PRIMARY_DESIGN = EXPERIMENT.PRIMARY_DESIGN
REALISTIC_TRANSFORMER_SHAPES = EXPERIMENT.REALISTIC_TRANSFORMER_SHAPES
SCHEMA_VERSION = EXPERIMENT.SCHEMA_VERSION
P17StudyConfig = EXPERIMENT.P17StudyConfig
canonical_json = EXPERIMENT.canonical_json
run_study = EXPERIMENT.run_study


def _small_config(**changes: object) -> object:
    config = P17StudyConfig(
        spectrum_grid_points=5,
        operating_annulus_grid_points=5,
        rank_radius_points=5,
        include_realistic_spectrum_cases=False,
    )
    return replace(config, **changes)


def _without_provenance(payload: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in payload.items() if key != "experiment_provenance"}


def test_small_study_replays_exact_designs_and_every_solver_call_passes() -> None:
    payload = run_study(_small_config())

    assert payload["schema_version"] == SCHEMA_VERSION
    assert all(record["certified"] for record in payload["exact_design_certificates"].values())
    assert payload["canonical_fidelity_gate"] == {
        "passes_by_design": {
            PRIMARY_DESIGN.name: True,
            FULL_STEP_DESIGN.name: True,
        },
        "all_locked_designs_pass": True,
        "thresholds_unchanged_from_p16": True,
    }
    summary = payload["solver_summary"]
    assert summary["call_count"] > 400
    assert summary["all_calls_computed_residual_certified"]
    assert summary["status_counts"] == {"strict_residual": summary["call_count"]}
    assert summary["worst_residual_to_p15_threshold_ratio"] < 1.0
    assert summary["maximum_iterations"] <= 64
    scope = payload["claim_scope"]
    assert "not incremental" in scope["incremental_warning"]
    assert "not directed-rounding" in scope["numerical_study"]
    assert payload["config"]["epsilon"] == 1.0e-7
    assert payload["config"]["jordan_coefficients"] == ["6889/2000", "-191/40", "4063/2000"]
    assert payload["config"]["jordan_iteration_count"] == 5


def test_broad_grid_discloses_origin_failure_but_locked_annulus_passes() -> None:
    payload = run_study(_small_config())
    broad = payload["declared_dense_two_mode_grid"]
    annulus = payload["declared_primary_operating_annulus_grid"]

    assert "not global certificates" in broad["status"]
    for design in (PRIMARY_DESIGN, FULL_STEP_DESIGN):
        record = broad["designs"][design.name]
        assert 0 < record["gate_passes"] < record["informative"]
        assert record["frozen_gate_pass_fraction_on_informative_points"] < 1.0
    assert "post-exploratory" in annulus["scope"]
    assert "only after" in annulus["selection_disclosure"]
    assert annulus["design"] == PRIMARY_DESIGN.name
    assert annulus["input_radius_interval"]["lower"]["fraction"] == str(
        OPERATING_ANNULUS_INNER_RADIUS
    )
    assert annulus["input_radius_interval"]["upper"]["fraction"] == str(
        OPERATING_ANNULUS_OUTER_RADIUS
    )
    assert annulus["informative_point_count"] > 0
    assert annulus["all_informative_point_estimates_pass"]
    assert annulus["all_informative_computed_residual_allowances_pass"]
    assert (
        annulus["computed_residual_allowance_gate_pass_count"] == annulus["informative_point_count"]
    )


def test_canonical_case_records_meaningful_nontrivial_shaping() -> None:
    payload = run_study(_small_config())
    records = payload["canonical_diag_3_4"]["designs"]
    primary = records[PRIMARY_DESIGN.name]["fidelity"]
    full = records[FULL_STEP_DESIGN.name]["fidelity"]

    assert primary["meaningful_fidelity_gate_passes"]
    assert full["meaningful_fidelity_gate_passes"]
    assert 0.0567 < primary["best_scalar_multiple"]["relative_residual"] < 0.0569
    assert 0.811 < primary["shaping_retention_fraction"] < 0.812
    assert 0.0203 < full["best_scalar_multiple"]["relative_residual"] < 0.0204
    assert 0.290 < full["shaping_retention_fraction"] < 0.292
    assert primary["modal_gains"]["spread"] > full["modal_gains"]["spread"] > 0


def test_negative_controls_distinguish_pointwise_stability_from_incremental_passivity() -> None:
    controls = run_study(_small_config())["negative_controls"]
    undersized = controls["undersized_passive_region"]
    locked = controls["locked_full_step_gate_at_same_scalar_witness"]

    assert undersized["on_gate_plateau"]
    assert undersized["derivative_diagnostic"]["gated_interface_derivative"] < -18_000
    assert locked["gate_off_below_q0"]
    assert locked["derivative_diagnostic"]["gated_interface_derivative"] > 0
    assert controls["rank_one_fidelity_control"]["best_scalar_departure"] == 0.0


def test_realistic_complete_spectra_cover_the_declared_transformer_shapes() -> None:
    payload = run_study(_small_config(include_realistic_spectrum_cases=True))
    cases = payload["realistic_complete_spectrum_cases"]

    assert [case["declared_matrix_shape"] for case in cases] == [
        list(shape) for shape in REALISTIC_TRANSFORMER_SHAPES
    ]
    assert cases[-1]["singular_value_count"] == 4_096
    assert all("no dense matrix" in case["representation"] for case in cases)
    assert all(
        record["diagnostics"]["certified"] for case in cases for record in case["designs"].values()
    )
    assert payload["solver_summary"]["all_calls_computed_residual_certified"]


def test_small_numerical_payload_is_deterministic_ignoring_provenance() -> None:
    config = _small_config()
    first = run_study(config)
    second = run_study(config)
    assert _without_provenance(first) == _without_provenance(second)


@pytest.mark.parametrize(
    ("change", "exception", "match"),
    [
        ({"seed": -1}, ValueError, "nonnegative"),
        ({"seed": True}, ValueError, "nonnegative"),
        ({"spectrum_grid_points": 4}, ValueError, "at least five"),
        ({"operating_annulus_grid_points": True}, ValueError, "at least five"),
        ({"rank_radius_points": 4}, ValueError, "at least five"),
        ({"include_realistic_spectrum_cases": 1}, TypeError, "Boolean"),
        ({"solver": object()}, TypeError, "EquivariantResolventSolverConfig"),
    ],
)
def test_invalid_config_is_rejected(
    change: dict[str, object],
    exception: type[Exception],
    match: str,
) -> None:
    with pytest.raises(exception, match=match):
        replace(P17StudyConfig(), **change)


def test_cli_writes_canonical_json_for_the_small_grid(tmp_path: Path) -> None:
    output = tmp_path / "p17-study.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--seed",
            "17",
            "--spectrum-grid-points",
            "5",
            "--operating-annulus-grid-points",
            "5",
            "--rank-radius-points",
            "5",
            "--skip-realistic-spectrum-cases",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert output.read_text(encoding="utf-8") == canonical_json(payload)
    assert payload["config"]["seed"] == 17
    assert payload["config"]["spectrum_grid_points"] == 5
    assert payload["config"]["operating_annulus_grid_points"] == 5
    assert payload["config"]["rank_radius_points"] == 5
    assert payload["realistic_complete_spectrum_cases"] == []
    assert payload["canonical_target"] == "results/summaries/p17_shape_preserving_study.json"


def test_committed_study_provenance_is_clean_and_hash_current_when_present() -> None:
    if not CANONICAL.exists():
        return

    payload = json.loads(CANONICAL.read_text(encoding="utf-8"))
    git = payload["experiment_provenance"]["git"]
    assert len(git["sha"]) == 40
    assert git["branch"] == "p17-shape-preserving-resolvent"
    assert git["dirty"] is False
    assert payload["solver_summary"]["all_calls_computed_residual_certified"]


def test_default_solver_configuration_remains_the_locked_p16_architecture() -> None:
    solver = EquivariantResolventSolverConfig()
    assert solver.epsilon == 1.0e-7
    assert solver.resolvent_parameter == 1.0e-3
    assert solver.shunt == 1_000.0
    assert solver.p15_relative_tolerance == 1 / 250
    assert solver.p15_absolute_tolerance == 2.0**-40
