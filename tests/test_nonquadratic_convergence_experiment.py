from __future__ import annotations

import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from passive_muon.nonquadratic_convergence import (
    LOCKED_NONQUADRATIC_CONVERGENCE_LEARNING_RATE,
    LOCKED_NONQUADRATIC_CONVERGENCE_TAU,
    locked_nonquadratic_convergence_certificate,
)
from passive_muon.nonquadratic_convergence_experiment import (
    NonquadraticConvergenceProbeConfig,
    convergence_lyapunov_value,
    objective_gap,
    run_nonquadratic_convergence_probe,
)
from passive_muon.nonquadratic_experiment import RotatingLogCoshObjective


def test_objective_gap_is_accurate_near_zero_and_storage_matches_formula() -> None:
    objective = RotatingLogCoshObjective(
        dimension=4,
        transition_scale=1.0,
        frame_variant=2,
    )
    tiny_position = np.array([1.0, -2.0, 3.0, -4.0]) * 1e-10
    expected_quadratic_limit = 5.0 * float(tiny_position @ tiny_position)
    assert objective_gap(objective, tiny_position) == pytest.approx(
        expected_quadratic_limit,
        rel=2e-15,
    )

    position = np.array([0.5, -1.0, 2.0, -0.25])
    momentum = np.array([-2.0, 0.5, 1.5, 3.0])
    certificate = locked_nonquadratic_convergence_certificate()
    normalized_momentum = momentum / float(certificate.smoothness)
    storage = np.asarray(
        [[float(value) for value in row] for row in certificate.storage],
        dtype=np.float64,
    )
    expected = float(
        storage[0, 0] * (position @ position)
        + 2.0 * storage[0, 1] * (position @ normalized_momentum)
        + storage[1, 1] * (normalized_momentum @ normalized_momentum)
        + float(certificate.function_storage)
        * objective_gap(objective, position)
        / float(certificate.smoothness)
    )
    assert convergence_lyapunov_value(objective, position, momentum) == pytest.approx(
        expected,
        rel=2e-15,
    )


def test_small_full_step_probe_is_deterministic_and_satisfies_storage_diagnostic() -> None:
    config = NonquadraticConvergenceProbeConfig(
        seed=31,
        shapes=((2, 2),),
        transition_scales=(1.0,),
        radius_multipliers=(4.0,),
        momentum_modes=("zero", "random"),
        iterations=40,
        diagnostic_stride=2,
    )

    first = run_nonquadratic_convergence_probe(config)
    second = run_nonquadratic_convergence_probe(config)

    assert first == second
    assert len(first.trials) == 2
    assert first.nonfinite_count == 0
    assert first.divergence_count == 0
    assert first.candidate_violation_count == 0
    assert first.lyapunov_rate_violating_case_count == 0
    assert first.maximum_normalized_lyapunov_rate_excess < 0.0
    assert first.maximum_lyapunov_ratio < float(LOCKED_NONQUADRATIC_CONVERGENCE_TAU**2)
    assert first.orientation_changing_case_count == 2
    assert first.maximum_orientation_commutator > config.orientation_tolerance
    assert first.sampled_hessian_eigenvalue_minimum >= 1.0 - 2e-12
    assert first.sampled_hessian_eigenvalue_maximum <= 10.0 + 2e-12
    assert all(trial.lyapunov_transition_count == config.iterations for trial in first.trials)

    payload = first.as_dict()
    assert payload["schema_version"] == ("passive-muon-nonquadratic-convergence-falsification-v1")
    assert "sampled passes do not prove" in payload["claim_scope"]["qualification"]
    locked = payload["locked_convergence_storage"]
    assert Fraction(locked["learning_rate_eta"]) == Fraction(1, 32_000)
    assert Fraction(locked["exponential_rate_tau"]) == LOCKED_NONQUADRATIC_CONVERGENCE_TAU
    assert Fraction(locked["function_storage"]) == (
        locked_nonquadratic_convergence_certificate().function_storage
    )


def test_default_probe_grid_contains_36_changing_orientation_cases() -> None:
    config = NonquadraticConvergenceProbeConfig()
    certificate = locked_nonquadratic_convergence_certificate()

    assert config.case_count == 36
    assert certificate.learning_rate == LOCKED_NONQUADRATIC_CONVERGENCE_LEARNING_RATE
    assert certificate.tau == LOCKED_NONQUADRATIC_CONVERGENCE_TAU


def test_convergence_falsification_runner_quick_replay_records_provenance() -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(
                root
                / "experiments"
                / "quadratics"
                / "run_nonquadratic_convergence_falsification.py"
            ),
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

    assert payload["config"]["iterations"] == 3
    assert payload["summary"]["trial_count"] == 1
    assert payload["operator"]["iterations"] == 5
    assert payload["operator"]["normalization"] == "fixed_frobenius_floor"
    assert payload["operator"]["additive_epsilon"].startswith("not_applicable")
    provenance = payload["experiment_provenance"]
    assert provenance["seed"] == 2_026_090_2
    assert provenance["dtype"] == "float64"
    assert len(provenance["git"]["sha"]) == 40
    assert provenance["software"]["numpy"]
    assert provenance["hardware"]["torch_device"] == "cpu"
    assert provenance["source_snapshot"]["src/passive_muon/nonquadratic_convergence_experiment.py"]
    assert payload["upstream_provenance"]["revision"]
    assert payload["certificate_references"]["p4"]["manifest_sha256"]
