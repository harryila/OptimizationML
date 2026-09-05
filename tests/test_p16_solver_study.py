from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from passive_muon.equivariant_resolvent_solver import EquivariantResolventSolverConfig

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "resolvent" / "run_p16_solver_study.py"
CANONICAL = ROOT / "results" / "summaries" / "p16_solver_study.json"
SPEC = importlib.util.spec_from_file_location("p16_solver_study", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("failed to load the P16 solver study")
EXPERIMENT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EXPERIMENT
SPEC.loader.exec_module(EXPERIMENT)

DEFAULT_DENSE_SHAPES = EXPERIMENT.DEFAULT_DENSE_SHAPES
FRONTIER_CANDIDATES = EXPERIMENT.FRONTIER_CANDIDATES
REALISTIC_TRANSFORMER_SHAPES = EXPERIMENT.REALISTIC_TRANSFORMER_SHAPES
SCHEMA_VERSION = EXPERIMENT.SCHEMA_VERSION
SolverStudyConfig = EXPERIMENT.SolverStudyConfig
_best_scalar_record = EXPERIMENT._best_scalar_record
_cost_proxy = EXPERIMENT._cost_proxy
_directional_sine = EXPERIMENT._directional_sine
build_dense_cases = EXPERIMENT.build_dense_cases
canonical_json = EXPERIMENT.canonical_json
constructed_fidelity_witness = EXPERIMENT.constructed_fidelity_witness
constructed_modal_gap_stress_case = EXPERIMENT.constructed_modal_gap_stress_case
run_stability_fidelity_frontier = EXPERIMENT.run_stability_fidelity_frontier
run_study = EXPERIMENT.run_study
upstream_jordan_singular_values = EXPERIMENT.upstream_jordan_singular_values


def _without_provenance(payload: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in payload.items() if key != "experiment_provenance"}


def _small_config(**changes: object) -> object:
    config = SolverStudyConfig(
        include_realistic_spectrum_cases=False,
        frontier_grid_points=7,
    )
    return replace(config, **changes)


def test_constructed_witness_has_distinct_gains_but_is_nearly_scalar() -> None:
    signal, solution, output = constructed_fidelity_witness()
    assert np.linalg.norm(solution) == pytest.approx(1.0, rel=0.0, abs=2e-15)
    assert np.allclose(signal, solution + output / 1_000.0, rtol=0.0, atol=1e-15)

    gains = output / signal
    assert gains[1] > gains[0]
    assert gains[1] - gains[0] > 0.009
    best = _best_scalar_record(signal, output)
    assert 5.7e-6 < best["relative_residual"] < 5.71e-6
    assert 781.0 < best["coefficient"] < 782.0

    upstream = upstream_jordan_singular_values(signal, 1.0e-7)
    assert _directional_sine(output, upstream) > 0.06
    assert _directional_sine(signal, upstream) > 0.06

    stress_signal, _stress_solution, stress_output = constructed_modal_gap_stress_case()
    stress_gains = stress_output / stress_signal
    assert stress_gains[0] - stress_gains[1] > 13.0
    assert 7.0e-5 < _best_scalar_record(stress_signal, stress_output)["relative_residual"] < 8.0e-5
    stress_upstream = upstream_jordan_singular_values(stress_signal, 1.0e-7)
    assert _directional_sine(stress_output, stress_upstream) > 0.86


def test_radial_and_rank_one_controls_are_not_fidelity_witnesses() -> None:
    vector = np.asarray([0.0, 3.0], dtype=np.float64)
    scalar = 721.0 * vector
    assert _best_scalar_record(vector, scalar)["relative_residual"] == 0.0

    equal = np.asarray([2.0, 2.0], dtype=np.float64)
    assert _directional_sine(equal, 17.0 * equal) == 0.0


def test_small_study_passes_every_declared_solver_call_and_records_scope() -> None:
    payload = run_study(_small_config())

    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["summary"]["dense_case_count"] == 9
    assert payload["summary"]["spectrum_only_case_count"] == 0
    assert payload["summary"]["certified_call_count"] == 9
    assert payload["summary"]["failed_call_count"] == 0
    assert payload["summary"]["worst_residual_to_p15_threshold_ratio"] < 1.0
    assert payload["summary"]["maximum_iterations"] <= 64
    assert "not a theorem" in payload["claim_scope"]["status"]
    assert "do not allocate dense matrices" in payload["claim_scope"]["large_shape_caveat"]
    assert payload["config"]["deployed_output"] == "(s-u_hat)/lambda"
    upstream = payload["upstream_formula_provenance"]
    assert upstream["revision"] == "f98f1cacc0263b04290753e32be8d498c1efc806"
    assert upstream["audited_file"] == "muon.py"
    assert upstream["audited_file_sha256"] == (
        "2479665a90124f62e4df557816665851ca317e42fcfda2af1da02c1f44ab5f3d"
    )
    gate = payload["fidelity_gate"]
    assert gate["status"] == "distinct_but_effectively_scalar"
    assert gate["best_scalar_departure"] < gate["best_scalar_departure_threshold"]
    assert gate["upstream_shaping_retention"] < gate["upstream_shaping_retention_threshold"]
    assert gate["modal_gain_spread"] > 0.009
    assert gate["joint_stability_fidelity_frontier_passes"] == []

    cases = {case["label"]: case for case in payload["dense_cases"]}
    assert cases["zero"]["diagnostics"]["status"] == "zero_input"
    assert cases["zero"]["diagnostics"]["iterations"] == 0
    assert cases["exact_forward_shaping_witness"]["fidelity"]["near_scalar_at_declared_threshold"]
    canonical = cases["balanced_canonical_input_diag_3_4"]["fidelity"]
    assert canonical["modal_gain_spread"] > 0.009
    assert 8.0e-5 < canonical["shaping_retention_fraction"] < 9.0e-5
    assert not canonical["meaningful_fidelity_gate_passes"]
    assert (
        cases["exploratory_ill_conditioned_modal_gap_stress"]["fidelity"][
            "directional_sine_to_upstream_jordan"
        ]
        > 0.86
    )
    assert all(case["resolvent_output_formula_error"] == 0.0 for case in cases.values())


def test_default_study_executes_full_realistic_spectra_without_dense_allocation() -> None:
    payload = run_study(SolverStudyConfig(frontier_grid_points=5))
    cases = payload["realistic_spectrum_cases"]

    assert [case["declared_matrix_shape"] for case in cases] == [
        list(shape) for shape in REALISTIC_TRANSFORMER_SHAPES
    ]
    assert all(case["diagnostics"]["certified"] for case in cases)
    assert all(case["cost_proxy"]["dense_cost_not_executed"] for case in cases)
    assert cases[-1]["singular_value_count"] == 4_096
    assert payload["summary"]["case_count"] == 13
    assert payload["summary"]["failed_call_count"] == 0


def test_numerical_payload_is_deterministic_ignoring_provenance() -> None:
    config = _small_config()
    first = run_study(config)
    second = run_study(config)
    assert _without_provenance(first) == _without_provenance(second)


def test_seed_changes_only_seeded_dense_cases() -> None:
    left = dict(build_dense_cases(_small_config(seed=11)))
    right = dict(build_dense_cases(_small_config(seed=12)))

    assert np.array_equal(left["zero"], right["zero"])
    assert np.array_equal(
        left["exact_forward_shaping_witness"], right["exact_forward_shaping_witness"]
    )
    assert not np.array_equal(left["unit_log_spectrum"], right["unit_log_spectrum"])
    assert not np.array_equal(
        left["near_p11_guard_log_spectrum"], right["near_p11_guard_log_spectrum"]
    )


def test_frontier_is_near_scalar_and_only_locked_frozen_certificate_passes() -> None:
    frontier = run_stability_fidelity_frontier(9)
    candidates = frontier["candidates"]

    assert len(candidates) == len(FRONTIER_CANDIDATES)
    assert all(candidate["evaluated_point_count"] > 0 for candidate in candidates)
    assert all(candidate["near_scalar_on_declared_grid"] for candidate in candidates)
    assert all(
        candidate["maximum_best_scalar_relative_residual"]["best_scalar_relative_residual"] < 1.1e-4
        for candidate in candidates
    )
    assert all(
        not candidate["retains_declared_fraction_of_upstream_shaping"] for candidate in candidates
    )
    passing = [
        candidate["name"]
        for candidate in candidates
        if candidate["frozen_p14_certificate"]["certified"]
    ]
    assert passing == ["locked_lambda_1_over_1000_mu_1000"]
    assert all(
        "sufficient certificate" in candidate["frozen_p14_certificate"]["scope"]
        for candidate in candidates
    )


def test_failure_controls_fail_closed_and_wrong_output_is_detected() -> None:
    payload = run_study(_small_config())
    controls = payload["failure_controls"]

    assert controls["zero_iteration_cap"]["status"] == "maximum_iterations"
    assert not controls["zero_iteration_cap"]["certified"]
    assert (
        controls["zero_iteration_cap"]["residual_norm"]
        > controls["zero_iteration_cap"]["p15_threshold"]
    )
    assert controls["over_guard_input"]["rejected"]
    assert controls["nonfinite_input"]["rejected"]
    wrong = controls["wrong_B_of_candidate_output"]
    assert not wrong["passes_p15"]
    assert wrong["required_output_minus_B_candidate_frobenius"] == pytest.approx(
        1_000.0 * wrong["graph_residual_frobenius"]
    )
    assert controls["radial_only_no_jordan_control"]["best_scalar_relative_residual"] < 1e-15


def test_cost_proxy_counts_five_jordan_stages_and_solver_reconstruction() -> None:
    class Diagnostics:
        svd_count = 2
        reconstruction_matmuls = 2
        iterations = 7
        sherman_morrison_solves = 7

    proxy = _cost_proxy((64, 128), Diagnostics())
    assert proxy["five_stage_jordan_dense_gemm_count"] == 15
    assert proxy["solver_svd_count"] == 2
    assert proxy["solver_reconstruction_matmul_count"] == 2
    assert proxy["solver_scalar_newton_iterations"] == 7
    assert "SVD cost" in proxy["important_caveat"]


@pytest.mark.parametrize(
    ("change", "exception", "match"),
    [
        ({"seed": -1}, ValueError, "nonnegative"),
        ({"seed": True}, ValueError, "nonnegative"),
        ({"dense_shapes": ()}, ValueError, "exactly six"),
        (
            {"dense_shapes": ((2, 2), (2, 3), (3, 2), (3, 4), (4, 3), (4, 0))},
            ValueError,
            "positive",
        ),
        (
            {"dense_shapes": ((2, 2), (2, 2), (3, 2), (3, 4), (4, 3), (4, 5))},
            ValueError,
            "unique",
        ),
        ({"include_realistic_spectrum_cases": 1}, TypeError, "Boolean"),
        ({"frontier_grid_points": 4}, ValueError, "at least five"),
        ({"frontier_grid_points": True}, ValueError, "at least five"),
        ({"solver": object()}, TypeError, "EquivariantResolventSolverConfig"),
    ],
)
def test_invalid_config_is_rejected(
    change: dict[str, object], exception: type[Exception], match: str
) -> None:
    with pytest.raises(exception, match=match):
        replace(SolverStudyConfig(), **change)


def test_cli_writes_canonical_json_for_six_small_shapes(tmp_path: Path) -> None:
    output = tmp_path / "p16-study.json"
    shapes = ("2x2", "2x3", "3x2", "3x4", "4x3", "4x5")
    command = [
        sys.executable,
        str(SCRIPT),
        "--seed",
        "17",
        "--frontier-grid-points",
        "5",
        "--skip-realistic-spectrum-cases",
        "--output",
        str(output),
    ]
    for shape in shapes:
        command.extend(("--dense-shape", shape))
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert output.read_text(encoding="utf-8") == canonical_json(payload)
    assert payload["config"]["seed"] == 17
    assert payload["config"]["dense_shapes"] == [
        [2, 2],
        [2, 3],
        [3, 2],
        [3, 4],
        [4, 3],
        [4, 5],
    ]
    assert payload["summary"]["dense_case_count"] == 9
    assert payload["summary"]["spectrum_only_case_count"] == 0
    assert payload["canonical_target"] == "results/summaries/p16_solver_study.json"


def test_committed_study_provenance_is_clean_and_hash_current_when_present() -> None:
    if not CANONICAL.exists():
        return

    payload = json.loads(CANONICAL.read_text(encoding="utf-8"))
    git = payload["experiment_provenance"]["git"]
    assert len(git["sha"]) == 40
    assert git["branch"] == "p16-equivariant-resolvent-solver"
    assert git["dirty"] is False
    assert payload["summary"]["failed_call_count"] == 0
    for relative_path, expected_digest in payload["experiment_provenance"][
        "source_snapshot"
    ].items():
        source = ROOT / relative_path
        assert source.is_file(), relative_path
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected_digest


def test_default_solver_configuration_remains_locked() -> None:
    solver = EquivariantResolventSolverConfig()
    assert solver.epsilon == 1e-7
    assert solver.resolvent_parameter == 1e-3
    assert solver.shunt == 1_000.0
    assert solver.p15_relative_tolerance == 1 / 250
    assert solver.p15_absolute_tolerance == 2.0**-40
