from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from passive_muon.pl_convergence import (
    LOCKED_PL_CONVERGENCE_LEARNING_RATE,
    LOCKED_PL_CONVERGENCE_TAU,
    locked_pl_convergence_certificate,
)
from passive_muon.pl_experiment import (
    ANALYTIC_NEGATIVE_CURVATURE,
    ANALYTIC_SMOOTHNESS_UPPER,
    PLProbeConfig,
    WarpedRadialPLObjective,
    pl_lyapunov_value,
    run_pl_probe,
)


@pytest.mark.parametrize("active_rank", [4, 3])
def test_warped_radial_family_is_nonconvex_smooth_and_pl(active_rank: int) -> None:
    objective = WarpedRadialPLObjective(
        dimension=4,
        transition_scale=2.0,
        active_rank=active_rank,
        frame_variant=3,
    )
    radial_witness = objective.frame[:, 0] * (math.sqrt(2.0) * objective.transition_scale)
    hessian = objective.hessian(radial_witness)
    eigenvalues = np.linalg.eigvalsh(hessian)

    assert objective.scaled_radius(radial_witness) == pytest.approx(1.0, abs=2e-15)
    assert eigenvalues[0] == pytest.approx(ANALYTIC_NEGATIVE_CURVATURE, abs=4e-14)
    assert max(abs(eigenvalues)) <= ANALYTIC_SMOOTHNESS_UPPER + 4e-14
    assert objective.pl_ratio(radial_witness) >= 1.0
    assert objective.value(radial_witness) > 0.0

    origin_hessian = objective.hessian(np.zeros(4))
    assert np.linalg.eigvalsh(origin_hessian)[-1] == pytest.approx(9.0, abs=4e-14)
    assert objective.frame_orthogonality_residual < 2e-15
    assert objective.projector_idempotence_residual < 2e-15
    assert objective.projector_symmetry_residual == 0.0


def test_rank_deficient_objective_has_a_nonunique_minimizer_set() -> None:
    objective = WarpedRadialPLObjective(
        dimension=4,
        transition_scale=1.0,
        active_rank=3,
        frame_variant=1,
    )
    active = objective.frame[:, 0] * 1.25
    null = objective.frame[:, -1] * 7.0

    assert objective.minimizer_set_dimension == 1
    assert objective.is_rank_deficient
    assert objective.value(null) == pytest.approx(0.0, abs=2e-28)
    assert np.linalg.norm(objective.gradient(null)) < 2e-14
    assert objective.value(active + null) == pytest.approx(objective.value(active), rel=2e-15)
    assert np.linalg.norm(objective.gradient(active + null) - objective.gradient(active)) < 3e-14
    assert np.linalg.norm(objective.null_component(active + null)) == pytest.approx(7.0)
    assert np.linalg.norm(objective.active_component(active + null)) == pytest.approx(1.25)


def test_float_objective_gradient_is_derivative_consistent() -> None:
    objective = WarpedRadialPLObjective(
        dimension=9,
        transition_scale=0.7,
        active_rank=8,
        frame_variant=5,
    )
    position = np.linspace(-0.8, 1.1, 9)
    direction = np.linspace(1.2, -0.4, 9)
    step = 1e-6
    finite_difference = (
        objective.value(position + step * direction) - objective.value(position - step * direction)
    ) / (2.0 * step)
    analytic = float(objective.gradient(position) @ direction)

    assert finite_difference == pytest.approx(analytic, rel=3e-10, abs=3e-10)


def test_pl_lyapunov_value_matches_the_certificate_formula() -> None:
    objective = WarpedRadialPLObjective(
        dimension=4,
        transition_scale=1.0,
        active_rank=3,
        frame_variant=2,
    )
    position = np.array([0.5, -1.0, 2.0, -0.25])
    momentum = np.array([-2.0, 0.5, 1.5, 3.0])
    certificate = locked_pl_convergence_certificate()
    smoothness = float(certificate.smoothness)
    z = momentum / smoothness
    u = objective.gradient(position) / smoothness
    storage = np.asarray(
        [[float(value) for value in row] for row in certificate.storage],
        dtype=np.float64,
    )
    expected = float(
        storage[0, 0] * (z @ z)
        + 2.0 * storage[0, 1] * (z @ u)
        + storage[1, 1] * (u @ u)
        + float(certificate.function_storage) * objective.value(position) / smoothness
    )

    assert pl_lyapunov_value(objective, position, momentum) == pytest.approx(
        expected,
        rel=2e-15,
    )


def test_small_pl_probe_is_deterministic_and_replays_the_lyapunov_rate() -> None:
    config = PLProbeConfig(
        seed=41,
        shapes=((2, 2),),
        rank_modes=("full_rank", "codimension_one"),
        transition_scales=(1.0,),
        radius_multipliers=(math.sqrt(2.0),),
        momentum_modes=("random",),
        iterations=60,
        diagnostic_stride=1,
    )

    first = run_pl_probe(config)
    second = run_pl_probe(config)

    assert first == second
    assert len(first.trials) == 2
    assert first.rank_deficient_case_count == 1
    assert first.negative_curvature_case_count == 2
    assert first.orientation_changing_case_count == 2
    assert first.nonfinite_count == 0
    assert first.divergence_count == 0
    assert first.candidate_violation_count == 0
    assert first.objective_increasing_case_count == 2
    assert first.lyapunov_rate_violating_case_count == 0
    assert first.nonpositive_lyapunov_case_count == 0
    assert first.unresolved_lyapunov_resurgence_case_count == 0
    assert first.objective_bound_violating_case_count == 0
    assert first.maximum_normalized_lyapunov_rate_excess < 0.0
    assert first.maximum_lyapunov_ratio < float(LOCKED_PL_CONVERGENCE_TAU**2)
    assert first.sampled_hessian_eigenvalue_minimum == pytest.approx(-3.0 / 8.0, abs=5e-13)
    assert first.sampled_hessian_eigenvalue_maximum <= 9.0 + 5e-13
    assert first.minimum_sampled_pl_ratio >= 1.0 - 5e-13
    assert all(trial.lyapunov_transition_count == config.iterations for trial in first.trials)
    assert all(trial.final_lyapunov_ratio < 1.0 for trial in first.trials)

    payload = first.as_dict()
    assert "sampled passes do not prove" in payload["claim_scope"]["qualification"]
    assert "individual objective-gap increases" in payload["diagnostics"]["objective_increase_rule"]
    assert Fraction(payload["locked_parameters"]["learning_rate_eta"]) == Fraction(1, 32_000)
    assert Fraction(payload["locked_pl_storage"]["exponential_rate_tau"]) == (
        LOCKED_PL_CONVERGENCE_TAU
    )


def test_default_pl_grid_includes_72_full_and_rank_deficient_cases() -> None:
    config = PLProbeConfig()

    assert config.case_count == 72
    assert config.learning_rate == float(LOCKED_PL_CONVERGENCE_LEARNING_RATE)
    assert config.beta == 19.0 / 20.0
    assert "full_rank" in config.rank_modes
    assert "codimension_one" in config.rank_modes

    with pytest.raises(ValueError, match="codimension_one"):
        replace(config, shapes=((1, 1),))


def test_sampled_hessian_interval_violation_is_a_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_hessian = WarpedRadialPLObjective.hessian

    def invalid_hessian(
        objective: WarpedRadialPLObjective,
        position: np.ndarray,
    ) -> np.ndarray:
        return original_hessian(objective, position) - 0.5 * np.eye(objective.dimension)

    monkeypatch.setattr(WarpedRadialPLObjective, "hessian", invalid_hessian)
    config = PLProbeConfig(
        shapes=((2, 2),),
        rank_modes=("full_rank",),
        transition_scales=(1.0,),
        radius_multipliers=(math.sqrt(2.0),),
        momentum_modes=("zero",),
        iterations=1,
        diagnostic_stride=1,
    )

    summary = run_pl_probe(config)

    assert summary.objective_bound_violating_case_count == 1
    assert summary.candidate_violation_count == 1
    assert summary.trials[0].sampled_objective_bound_violation


def test_pl_falsification_runner_quick_replay_records_full_provenance() -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "experiments" / "nonconvex" / "run_pl_falsification.py"),
            "--quick",
            "--iterations",
            "3",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["schema_version"] == "passive-muon-pl-falsification-v1"
    assert payload["config"]["iterations"] == 3
    assert payload["summary"]["trial_count"] == 1
    assert payload["summary"]["rank_deficient_case_count"] == 1
    assert payload["operator"]["iterations"] == 5
    assert payload["operator"]["normalization"] == "fixed_frobenius_floor"
    assert payload["operator"]["additive_epsilon"].startswith("not_applicable")
    assert payload["objective_family"]["analytic_exact_real_properties"]["global_PL_lower"] == "1"
    provenance = payload["experiment_provenance"]
    assert provenance["seed"] == 2_026_090_2
    assert provenance["dtype"] == "float64"
    assert provenance["hardware"]["torch_device"] == "cpu"
    assert provenance["software"]["numpy"]
    assert len(provenance["git"]["sha"]) == 40
    assert provenance["source_snapshot"]["src/passive_muon/pl_experiment.py"]
    assert payload["certificate_reference"]["source_sha256"]
    assert payload["p5_reference"]["manifest_sha256"]
    assert payload["upstream_provenance"]["revision"]
