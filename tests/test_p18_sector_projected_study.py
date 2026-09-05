from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from passive_muon.equivariant_resolvent_solver import EquivariantResolventSolverConfig

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "resolvent" / "run_p18_sector_projected_study.py"
SPEC = importlib.util.spec_from_file_location("p18_sector_projected_study", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("failed to load the P18 sector-projected study")
EXPERIMENT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EXPERIMENT
SPEC.loader.exec_module(EXPERIMENT)

EFFECTIVE_UPDATE_LOWER = EXPERIMENT.EFFECTIVE_UPDATE_LOWER
EFFECTIVE_UPDATE_UPPER = EXPERIMENT.EFFECTIVE_UPDATE_UPPER
REALISTIC_TRANSFORMER_SHAPES = EXPERIMENT.REALISTIC_TRANSFORMER_SHAPES
SCHEMA_VERSION = EXPERIMENT.SCHEMA_VERSION
SELECTED_LEARNING_RATE = EXPERIMENT.SELECTED_LEARNING_RATE
VARIANT_LOCKED = EXPERIMENT.VARIANT_LOCKED
VARIANT_UNDERSIZED = EXPERIMENT.VARIANT_UNDERSIZED
VARIANT_UNPROJECTED = EXPERIMENT.VARIANT_UNPROJECTED
P18StudyConfig = EXPERIMENT.P18StudyConfig
canonical_json = EXPERIMENT.canonical_json
run_study = EXPERIMENT.run_study


def _small_config(**changes: object) -> object:
    config = P18StudyConfig(
        spectrum_grid_points=5,
        operating_annulus_grid_points=5,
        include_realistic_spectrum_cases=False,
    )
    return replace(config, **changes)


@pytest.fixture(scope="module")
def small_payload() -> dict[str, object]:
    return run_study(_small_config())


def _without_provenance(payload: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in payload.items() if key != "experiment_provenance"}


def test_exact_global_amplitude_and_selected_step_gates_are_locked(
    small_payload: dict[str, object],
) -> None:
    assert small_payload["schema_version"] == SCHEMA_VERSION
    exact = small_payload["exact_global_amplitude_and_effective_update_gates"]
    locked = exact["locked_design"]
    sector = locked["pointwise_sector"]

    assert Fraction(1, 83) == SELECTED_LEARNING_RATE
    assert sector["lower"]["fraction"] == "125/1024"
    assert sector["upper"]["fraction"] == "509/512"
    assert sector["center"]["fraction"] == "1143/2048"
    assert sector["radius"]["fraction"] == "893/2048"
    assert locked["selected_effective_update_sector"]["lower"]["fraction"] == "125/84992"
    assert locked["selected_effective_update_sector"]["upper"]["fraction"] == "509/42496"
    assert locked["global_amplitude_gate_passes"]
    assert Fraction(locked["selected_effective_update_sector"]["lower"]["fraction"]) >= (
        EFFECTIVE_UPDATE_LOWER
    )
    assert Fraction(locked["selected_effective_update_sector"]["upper"]["fraction"]) <= (
        EFFECTIVE_UPDATE_UPPER
    )

    unprojected = exact["unprojected_control"]
    assert not unprojected["global_amplitude_gate_passes"]
    assert not unprojected["selected_effective_update_gate_passes"]
    assert float(unprojected["pointwise_sector"]["upper"]["decimal"]) > 565.4
    assert float(unprojected["selected_effective_upper"]["decimal"]) > 6.8


def test_canonical_case_retains_shape_and_exposes_undersized_control(
    small_payload: dict[str, object],
) -> None:
    canonical = small_payload["canonical_diag_3_4"]
    locked = canonical["variants"][VARIANT_LOCKED]
    small = canonical["variants"][VARIANT_UNDERSIZED]

    assert canonical["diagnostics"]["certified"]
    assert (
        canonical["diagnostics"]["actual_residual_norm"]
        <= canonical["diagnostics"]["p15_threshold"]
    )
    assert canonical["projection_alpha"] == 1.0
    assert canonical["undersized_projection_alpha"] == pytest.approx(
        0.037435382125555854,
        rel=2e-14,
    )
    assert locked["meaningful_fidelity_gate_passes"]
    assert 0.0362 < locked["best_scalar_departure"] < 0.0364
    assert 0.518 < locked["shaping_retention_fraction"] < 0.519
    assert 1.445 < locked["output_to_upstream_amplitude"] < 1.446
    assert 2_413 < locked["effective_update_ratio_to_p17_primary"] < 2_414
    assert 1_297 < locked["effective_update_ratio_to_p17_full_step"] < 1_299
    assert not small["meaningful_fidelity_gate_passes"]
    assert small["shaping_retention_fraction"] < 0.04


def test_broad_grid_discloses_origin_fidelity_failures_without_weakening_scale_gates(
    small_payload: dict[str, object],
) -> None:
    broad = small_payload["declared_broad_two_mode_grid"]
    locked = broad["variants"][VARIANT_LOCKED]
    by_region = broad["locked_variant_by_gate_region"]

    assert "not global certificates" in broad["status"]
    assert "intentionally" in broad["origin_fidelity_disclosure"]
    assert 0 < locked["fidelity_passes"] < locked["informative"]
    assert locked["all_sampled_global_input_amplitudes_pass"]
    assert locked["all_selected_effective_updates_pass"]
    assert by_region["gate_off"]["informative"] > 0
    assert by_region["gate_off"]["fidelity_passes"] == 0
    assert by_region["plateau"]["fidelity_passes"] == by_region["plateau"]["informative"]


def test_operating_annulus_passes_locked_gates_and_rejects_controls(
    small_payload: dict[str, object],
) -> None:
    annulus = small_payload["declared_operating_annulus_grid"]
    locked = annulus["variants"][VARIANT_LOCKED]
    small = annulus["variants"][VARIANT_UNDERSIZED]
    unprojected = annulus["variants"][VARIANT_UNPROJECTED]

    assert "only this declared grid" in annulus["scope"]
    assert annulus["all_required_acceptance_checks_pass"]
    assert all(annulus["acceptance"][name] for name in annulus["required_acceptance_checks"])
    assert locked["fidelity_passes"] == locked["informative"] > 0
    assert locked["all_annulus_upstream_amplitudes_pass"]
    assert locked["all_selected_effective_updates_pass"]
    assert (
        0.15
        < locked["minimum_output_to_input_amplitude"]["metrics"]["output_to_input_amplitude"]
        < 0.16
    )
    assert (
        0.98
        < locked["maximum_output_to_input_amplitude"]["metrics"]["output_to_input_amplitude"]
        < 1.0
    )
    assert (
        locked["minimum_selected_effective_update_per_input"]["metrics"][
            "selected_effective_update_per_input"
        ]
        > 0.0018
    )
    assert (
        locked["maximum_selected_effective_update_per_input"]["metrics"][
            "selected_effective_update_per_input"
        ]
        < 0.012
    )
    assert small["fidelity_passes"] == 0 < small["informative"]
    assert not unprojected["all_selected_effective_updates_pass"]


def test_every_declared_solver_call_checks_the_actual_p15_residual(
    small_payload: dict[str, object],
) -> None:
    summary = small_payload["solver_summary"]

    assert summary["call_count"] > 150
    assert summary["all_calls_computed_residual_certified"]
    assert summary["status_counts"] == {"strict_residual": summary["call_count"]}
    assert summary["worst_residual_to_p15_threshold_ratio"] < 1.0
    assert summary["maximum_iterations"] <= 64
    scope = small_payload["claim_scope"]
    assert "not global certificates" in scope["sampled_evidence"]
    assert "no claim" in scope["residual_scope"]
    assert "not an incremental" in scope["incremental_warning"]
    assert "intentionally much larger" in scope["effective_update_comparison"]


def test_realistic_cases_use_complete_transformer_spectra_and_pass_residuals() -> None:
    payload = run_study(_small_config(include_realistic_spectrum_cases=True))
    cases = payload["realistic_complete_spectrum_cases"]

    assert [case["declared_matrix_shape"] for case in cases] == [
        list(shape) for shape in REALISTIC_TRANSFORMER_SHAPES
    ]
    assert cases[-1]["singular_value_count"] == 4_096
    assert all("no dense matrix" in case["representation"] for case in cases)
    assert all(case["diagnostics"]["certified"] for case in cases)
    assert all(
        case["variants"][VARIANT_LOCKED]["sampled_global_input_amplitude_gate_passes"]
        for case in cases
    )
    assert all(
        case["variants"][VARIANT_LOCKED]["selected_effective_update_gate_passes"] for case in cases
    )
    informative = [
        case["variants"][VARIANT_LOCKED]
        for case in cases
        if case["variants"][VARIANT_LOCKED]["informative_upstream_shaping"]
    ]
    assert informative
    assert all(record["meaningful_fidelity_gate_passes"] for record in informative)


def test_small_payload_is_deterministic_ignoring_provenance() -> None:
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
        replace(P18StudyConfig(), **change)


def test_cli_writes_canonical_json_for_small_grid(tmp_path: Path) -> None:
    output = tmp_path / "p18-study.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--seed",
            "18",
            "--spectrum-grid-points",
            "5",
            "--operating-annulus-grid-points",
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
    assert payload["config"]["seed"] == 18
    assert payload["config"]["spectrum_grid_points"] == 5
    assert payload["config"]["operating_annulus_grid_points"] == 5
    assert payload["realistic_complete_spectrum_cases"] == []
    assert payload["canonical_target"] == "results/summaries/p18_sector_projected_study.json"


def test_default_solver_configuration_remains_the_locked_p16_graph() -> None:
    solver = EquivariantResolventSolverConfig()
    assert solver.epsilon == 1.0e-7
    assert solver.resolvent_parameter == 1.0e-3
    assert solver.shunt == 1_000.0
    assert solver.p15_relative_tolerance == 1 / 250
    assert solver.p15_absolute_tolerance == 2.0**-40
