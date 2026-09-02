from __future__ import annotations

import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

from passive_muon.nonquadratic_experiment import (
    NonquadraticProbeConfig,
    RotatingLogCoshObjective,
    rational_givens_frame,
    run_nonquadratic_probe,
)
from passive_muon.nonquadratic_stability import (
    LOCKED_NONQUADRATIC_LEARNING_RATE,
    LOCKED_NONQUADRATIC_TAU,
    locked_nonquadratic_certificate,
)


def test_rational_givens_frames_are_deterministic_and_orthogonal() -> None:
    first = rational_givens_frame(9, variant=2)
    second = rational_givens_frame(9, variant=2)
    other = rational_givens_frame(9, variant=3)

    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first.T @ first, np.eye(9), rtol=0.0, atol=2e-15)
    assert np.linalg.norm(first - other, ord="fro") > 0.1


def test_log_cosh_objective_has_requested_bounds_and_exact_secants() -> None:
    objective = RotatingLogCoshObjective(
        dimension=4,
        transition_scale=1.0,
        frame_variant=0,
    )
    np.testing.assert_allclose(objective.hessian(np.zeros(4)), 10.0 * np.eye(4), atol=2e-14)

    generator = np.random.default_rng(20260902)
    for _ in range(20):
        position = 4.0 * generator.standard_normal(4)
        eigenvalues = np.linalg.eigvalsh(objective.hessian(position))
        assert eigenvalues[0] >= 1.0 - 2e-14
        assert eigenvalues[-1] <= 10.0 + 2e-14

    first_a = np.array([4.0, 0.5, -1.0, 2.0])
    first_b = np.array([4.001, 0.499, -0.999, 2.002])
    second_a = np.array([0.4, 3.0, 1.5, -2.0])
    second_b = np.array([0.402, 2.999, 1.501, -1.998])
    first_secant = objective.secant_hessian(first_a, first_b)
    second_secant = objective.secant_hessian(second_a, second_b)

    gradient_difference = objective.gradient(first_a) - objective.gradient(first_b)
    np.testing.assert_allclose(
        first_secant @ (first_a - first_b),
        gradient_difference,
        rtol=2e-12,
        atol=2e-15,
    )
    assert np.linalg.eigvalsh(first_secant)[0] >= 1.0 - 2e-14
    assert np.linalg.eigvalsh(first_secant)[-1] <= 10.0 + 2e-14
    assert (
        np.linalg.norm(
            first_secant @ second_secant - second_secant @ first_secant,
            ord="fro",
        )
        > 1.0
    )


def test_small_nonlinear_pair_probe_is_deterministic_and_tracks_locked_storage() -> None:
    config = NonquadraticProbeConfig(
        seed=31,
        shapes=((2, 2),),
        transition_scales=(1.0,),
        radius_multipliers=(4.0,),
        pair_modes=("position_only", "full_state"),
        iterations=30,
        diagnostic_stride=2,
        tail_window=8,
    )

    first = run_nonquadratic_probe(config)
    second = run_nonquadratic_probe(config)

    assert first == second
    assert len(first.trials) == 2
    assert first.nonfinite_count == 0
    assert first.divergence_count == 0
    assert first.storage_rate_violating_case_count == 0
    assert first.maximum_normalized_storage_rate_excess <= config.storage_rate_tolerance
    assert first.orientation_changing_case_count > 0
    assert first.sampled_secant_eigenvalue_minimum >= config.curvature_lower - 2e-12
    assert first.sampled_secant_eigenvalue_maximum <= config.curvature_upper + 2e-12
    assert all(trial.storage_transition_count == config.iterations for trial in first.trials)
    assert all(trial.maximum_secant_identity_relative_residual < 2e-10 for trial in first.trials)

    payload = first.as_dict()
    assert payload["schema_version"] == "passive-muon-nonquadratic-falsification-v1"
    assert "sampled passes do not certify" in payload["claim_scope"]
    locked = payload["locked_exact_parameters"]
    assert Fraction(locked["learning_rate_eta"]) == Fraction(1, 640_000)
    assert Fraction(locked["reference_rate_tau"]) == Fraction(99_999, 100_000)
    assert locked["storage"] == [
        [str(value) for value in row] for row in locked_nonquadratic_certificate().storage
    ]
    assert Fraction(1, 640_000) == LOCKED_NONQUADRATIC_LEARNING_RATE
    assert Fraction(99_999, 100_000) == LOCKED_NONQUADRATIC_TAU


def test_nonquadratic_runner_quick_replay_records_provenance() -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "experiments/quadratics/run_nonquadratic_falsification.py"),
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

    assert payload["schema_version"] == "passive-muon-nonquadratic-falsification-v1"
    assert payload["config"]["iterations"] == 3
    assert payload["summary"]["trial_count"] == 1
    assert payload["operator"]["normalization"] == "fixed_frobenius_floor"
    assert payload["operator"]["iterations"] == 5
    assert payload["operator"]["additive_epsilon"].startswith("not_applicable")
    provenance = payload["experiment_provenance"]
    assert provenance["seed"] == 2_026_090_2
    assert provenance["dtype"] == "float64"
    assert len(provenance["git"]["sha"]) == 40
    assert provenance["software"]["numpy"]
    assert provenance["hardware"]["torch_device"] == "cpu"
    assert provenance["source_snapshot"]["src/passive_muon/nonquadratic_experiment.py"]
    assert payload["upstream_provenance"]["revision"]
