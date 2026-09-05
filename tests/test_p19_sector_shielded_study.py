from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from passive_muon.equivariant_resolvent_solver import EquivariantResolventSolverConfig

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "resolvent" / "run_p19_sector_shielded_study.py"
SPEC = importlib.util.spec_from_file_location("p19_sector_shielded_study", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("failed to load the P19 sector-shielded study")
EXPERIMENT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EXPERIMENT
SPEC.loader.exec_module(EXPERIMENT)

FASTER_RATE_STEP = EXPERIMENT.FASTER_RATE_STEP
MAXIMUM_STEP = EXPERIMENT.MAXIMUM_STEP
SCHEMA_VERSION = EXPERIMENT.SCHEMA_VERSION
P19StudyConfig = EXPERIMENT.P19StudyConfig
canonical_json = EXPERIMENT.canonical_json
run_study = EXPERIMENT.run_study


def _small_config(**changes: object) -> object:
    config = P19StudyConfig(operating_annulus_grid_points=5)
    return replace(config, **changes)


@pytest.fixture(scope="module")
def small_payload() -> dict[str, object]:
    return run_study(_small_config())


def _without_provenance(payload: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in payload.items() if key != "experiment_provenance"}


def test_both_frozen_operating_points_replay_exactly(
    small_payload: dict[str, object],
) -> None:
    assert small_payload["schema_version"] == SCHEMA_VERSION
    assert Fraction(1, 83) == MAXIMUM_STEP
    assert Fraction(1, 120) == FASTER_RATE_STEP

    records = small_payload["operating_point_replays"]
    assert [record["learning_rate"]["fraction"] for record in records] == ["1/83", "1/120"]
    assert [record["rate_squared"]["fraction"] for record in records] == [
        "999598040401/1000000000000",
        "624350169/625000000",
    ]
    assert all(record["exact_certificate_replay_passes"] for record in records)
    assert 1_724 < records[0]["certified_lyapunov_rate_half_life"] < 1_725
    assert 666 < records[1]["certified_lyapunov_rate_half_life"] < 667


def test_canonical_guarded_p18_candidate_is_bitwise_unchanged(
    small_payload: dict[str, object],
) -> None:
    canonical = small_payload["canonical_diag_3_4"]
    shield = canonical["shield_diagnostics"]

    assert canonical["solver_diagnostics"]["certified"]
    assert canonical["shield_was_identity_bitwise"]
    assert canonical["shield_change_frobenius"] == 0.0
    assert shield["candidate_inside"]
    assert not shield["active"]
    assert not shield["fail_closed"]
    assert shield["exact_sector_certified"]
    assert canonical["unshielded_metrics"] == canonical["shielded_metrics"]
    metrics = canonical["shielded_metrics"]
    assert metrics["meaningful_fidelity_gate_passes"]
    assert 0.0362 < metrics["best_scalar_departure"] < 0.0364
    assert 0.518 < metrics["shaping_retention_fraction"] < 0.519


def test_entire_sampled_annulus_is_inactive_and_preserves_p18_gates(
    small_payload: dict[str, object],
) -> None:
    annulus = small_payload["declared_operating_annulus_grid"]

    assert "only this frozen grid" in annulus["scope"]
    assert annulus["evaluated"] > 100
    assert annulus["shield_active_count"] == 0
    assert annulus["shield_inactive_count"] == annulus["evaluated"]
    assert annulus["shield_bitwise_identity_count"] == annulus["evaluated"]
    assert annulus["shield_normally_inactive"]
    assert annulus["worst_shield_change_frobenius"] == 0.0
    assert annulus["fidelity_passes"] == annulus["informative"] > 0
    assert annulus["all_informative_fidelity_pass"]
    assert annulus["upstream_amplitude_passes"] == annulus["evaluated"]
    assert annulus["all_upstream_amplitudes_pass"]
    assert annulus["all_effective_update_gates_pass"]
    assert annulus["effective_update_passes"] == {
        "1/120": annulus["evaluated"],
        "1/83": annulus["evaluated"],
    }


def test_corrupted_candidates_are_contained_and_distance_does_not_expand(
    small_payload: dict[str, object],
) -> None:
    controls = small_payload["corrupted_candidate_controls"]
    records = controls["finite_corruptions"]

    assert controls["all_finite_unshielded_candidates_violate"]
    assert controls["all_finite_shielded_outputs_certified"]
    assert controls["all_declared_finite_corruptions_reduce_distance_to_p18_reference"]
    assert set(records) == {
        "premature_zero",
        "outward_centerline",
        "anti_aligned",
        "large_skew",
    }
    for record in records.values():
        assert record["unshielded_nominal_disk_margin_fp64"] < 0.0
        assert record["unshielded_violates_sector"]
        assert record["shield_diagnostics"]["active"]
        assert record["shield_diagnostics"]["exact_sector_certified"]
        assert record["observed_distance_nonexpansion"]
        assert (
            record["shielded_distance_to_guarded_p18_reference"]
            <= record["candidate_distance_to_guarded_p18_reference"]
        )


def test_zero_and_nonfinite_controls_fail_closed_without_sector_escape(
    small_payload: dict[str, object],
) -> None:
    controls = small_payload["corrupted_candidate_controls"]

    assert controls["all_nonfinite_candidates_fail_closed"]
    for record in controls["nonfinite_candidate_controls"].values():
        assert record["shield_diagnostics"]["fail_closed"]
        assert record["shield_diagnostics"]["used_interior_fallback"]
        assert record["shield_diagnostics"]["exact_sector_certified"]
        assert np.isfinite(np.asarray(record["shielded_output"], dtype=np.float64)).all()

    assert controls["all_zero_signal_outputs_are_zero"]
    for record in controls["zero_signal_controls"].values():
        assert record["shielded_output"] == [0.0, 0.0]
        assert record["shield_diagnostics"]["exact_sector_certified"]

    assert controls["all_nonfinite_signals_rejected"]
    assert set(controls["nonfinite_signal_rejections"]) == {"nan_signal", "infinite_signal"}


def test_graph_residuals_are_reported_as_fidelity_ingredients_only(
    small_payload: dict[str, object],
) -> None:
    summary = small_payload["solver_graph_residual_fidelity_summary"]

    assert summary["call_count"] > 100
    assert summary["all_calls_computed_residual_certified"]
    assert summary["status_counts"] == {"strict_residual": summary["call_count"]}
    assert summary["worst_residual_to_p15_threshold_ratio"] < 1.0
    assert summary["maximum_exact_real_solution_error_allowance"] >= 0.0
    assert summary["maximum_exact_real_yosida_output_error_allowance"] >= 0.0
    assert summary["maximum_coarse_raw_jordan_output_error_allowance"] >= 0.0
    assert summary["final_p18_candidate_error_allowance_from_residual"] is None
    assert "not sufficient alone" in summary["role"]
    assert "nonlinear P18 ray projection" in summary["limitation"]
    scope = small_payload["claim_scope"]
    assert "nonexpansive" in scope["nonexpansivity"]
    assert "does not provide" in scope["nonexpansivity"]


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
        ({"operating_annulus_grid_points": 4}, ValueError, "at least five"),
        ({"operating_annulus_grid_points": True}, ValueError, "at least five"),
        ({"inward_ulps": 0}, ValueError, "positive"),
        ({"inward_ulps": True}, ValueError, "positive"),
        ({"solver": object()}, TypeError, "EquivariantResolventSolverConfig"),
    ],
)
def test_invalid_config_is_rejected(
    change: dict[str, object],
    exception: type[Exception],
    match: str,
) -> None:
    with pytest.raises(exception, match=match):
        replace(P19StudyConfig(), **change)


def test_cli_writes_canonical_json_for_small_grid(tmp_path: Path) -> None:
    output = tmp_path / "p19-study.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--seed",
            "19",
            "--operating-annulus-grid-points",
            "5",
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
    assert payload["config"]["seed"] == 19
    assert payload["config"]["operating_annulus_grid_points"] == 5
    assert payload["canonical_target"] == "results/summaries/p19_sector_shielded_study.json"


def test_default_solver_configuration_remains_the_locked_p16_graph() -> None:
    solver = EquivariantResolventSolverConfig()
    assert solver.epsilon == 1.0e-7
    assert solver.resolvent_parameter == 1.0e-3
    assert solver.shunt == 1_000.0
    assert solver.p15_relative_tolerance == 1 / 250
    assert solver.p15_absolute_tolerance == 2.0**-40
