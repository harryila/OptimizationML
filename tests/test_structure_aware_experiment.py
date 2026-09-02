from __future__ import annotations

import math

import numpy as np

from passive_muon.structure_aware_experiment import (
    LOCKED_P4_LEARNING_RATE,
    StructureAwareProbeConfig,
    run_structure_aware_probe,
    sampled_spd_hessian,
)


def test_sampled_hessians_have_requested_spectrum_and_orientation() -> None:
    aligned = sampled_spd_hessian(
        dimension=4,
        orientation="aligned",
        lower=1.0,
        upper=10.0,
        rng=np.random.default_rng(7),
    )
    rotated = sampled_spd_hessian(
        dimension=4,
        orientation="random",
        lower=1.0,
        upper=10.0,
        rng=np.random.default_rng(7),
    )

    np.testing.assert_allclose(np.linalg.eigvalsh(aligned)[[0, -1]], [1.0, 10.0])
    np.testing.assert_allclose(np.linalg.eigvalsh(rotated)[[0, -1]], [1.0, 10.0])
    assert np.count_nonzero(aligned - np.diag(np.diag(aligned))) == 0
    assert np.linalg.norm(rotated - np.diag(np.diag(rotated))) > 0.1


def test_small_full_matrix_probe_is_deterministic_and_finite() -> None:
    config = StructureAwareProbeConfig(
        seed=31,
        trials_per_case=2,
        iterations=120,
        initial_scales=(1e-4, 1.0),
    )

    first = run_structure_aware_probe(config)
    second = run_structure_aware_probe(config)

    assert first == second
    assert len(first.trials) == 8
    assert first.divergence_count == 0
    assert first.nonfinite_count == 0
    assert all(trial.executed_iterations == config.iterations for trial in first.trials)
    assert all(
        trial.hessian_eigenvalues[0] == np.float64(1.0)
        or math.isclose(trial.hessian_eigenvalues[0], 1.0, rel_tol=1e-12)
        for trial in first.trials
    )
    assert all(
        math.isclose(trial.hessian_eigenvalues[-1], 10.0, rel_tol=1e-12) for trial in first.trials
    )


def test_manifest_keeps_sampling_qualification_and_exact_locked_step() -> None:
    config = StructureAwareProbeConfig(
        shapes=((2, 2),),
        orientations=("aligned",),
        trials_per_case=1,
        iterations=2,
    )
    payload = run_structure_aware_probe(config).as_dict()

    assert payload["schema_version"] == "passive-muon-structure-aware-probe-v1"
    assert "sampling diagnostic only" in payload["claim_scope"]
    assert payload["locked_exact_parameters"]["learning_rate"] == str(LOCKED_P4_LEARNING_RATE)
    assert payload["operator"]["iterations"] == 5
    assert payload["operator"]["epsilon"] == 0.0
