from __future__ import annotations

import numpy as np

from passive_muon.quadratic_experiment import (
    build_quadratic_ensemble,
    run_quadratic_sweep,
    summarize_grid_band,
    summarize_true_prefix,
    widest_true_interval,
)
from passive_muon.specs import CLASSICAL_CUBIC, QuinticCoefficients


def test_widest_true_interval_uses_contiguous_component() -> None:
    values = np.geomspace(1e-3, 1e1, 5)
    summary = widest_true_interval(np.array([True, True, False, True, False]), values)
    assert summary.minimum == values[0]
    assert summary.maximum == values[1]
    assert summary.point_count == 2


def test_grid_band_reports_components_brackets_and_censoring() -> None:
    values = np.geomspace(1e-3, 1e2, 6)
    summary = summarize_grid_band(np.array([False, True, True, False, True, False]), values)
    assert summary.component_count == 2
    assert summary.lower_pass == values[1]
    assert summary.upper_pass == values[2]
    assert summary.lower_fail_below == values[0]
    assert summary.upper_fail_above == values[3]
    assert not summary.left_censored
    assert not summary.right_censored


def test_true_prefix_reports_upper_bracket() -> None:
    values = np.geomspace(1e-3, 1e1, 5)
    summary = summarize_true_prefix(np.array([True, True, True, False, True]), values)
    assert summary.upper_pass == values[2]
    assert summary.upper_fail == values[3]
    assert not summary.right_censored


def test_quadratic_sweep_is_deterministic_and_small_lr_converges() -> None:
    ensemble = build_quadratic_ensemble(condition_numbers=(1.0,), seeds_per_condition=2, seed=7)
    rates = np.array([0.01, 0.1], dtype=np.float64)
    first = run_quadratic_sweep(
        ensemble,
        rates,
        CLASSICAL_CUBIC,
        polynomial_steps=1,
        iterations=1000,
        normalization="current_plus_eps",
        normalizer_scale=1.0,
        repair_rho=0.0,
        success_ratio=1e-4,
        divergence_ratio=1e6,
    )
    second = run_quadratic_sweep(
        ensemble,
        rates,
        CLASSICAL_CUBIC,
        polynomial_steps=1,
        iterations=1000,
        normalization="current_plus_eps",
        normalizer_scale=1.0,
        repair_rho=0.0,
        success_ratio=1e-4,
        divergence_ratio=1e6,
    )
    np.testing.assert_array_equal(first.success, second.success)
    np.testing.assert_allclose(first.final_ratios, second.final_ratios)
    assert np.all(first.success)
    assert np.all(first.tail_maximum_ratios <= 1e-4)


def test_output_scaling_matches_learning_rate_rescaling() -> None:
    ensemble = build_quadratic_ensemble(condition_numbers=(3.0,), seeds_per_condition=2, seed=11)
    base = run_quadratic_sweep(
        ensemble,
        np.array([0.1]),
        CLASSICAL_CUBIC,
        polynomial_steps=1,
        iterations=20,
        normalization="current_plus_eps",
        normalizer_scale=1.0,
        repair_rho=0.0,
        success_ratio=1e-4,
        divergence_ratio=1e6,
        output_scale=2.0,
    )
    rescaled_rate = run_quadratic_sweep(
        ensemble,
        np.array([0.2]),
        CLASSICAL_CUBIC,
        polynomial_steps=1,
        iterations=20,
        normalization="current_plus_eps",
        normalizer_scale=1.0,
        repair_rho=0.0,
        success_ratio=1e-4,
        divergence_ratio=1e6,
        output_scale=1.0,
    )
    np.testing.assert_allclose(base.final_ratios, rescaled_rate.final_ratios)


def test_identity_map_recovers_quadratic_lr_ceiling_two() -> None:
    identity = QuinticCoefficients("identity", "1", "0", "0", "test fixture")
    ensemble = build_quadratic_ensemble(condition_numbers=(1.0,), seeds_per_condition=2, seed=13)
    outcome = run_quadratic_sweep(
        ensemble,
        np.array([1.9, 2.1]),
        identity,
        polynomial_steps=1,
        iterations=100,
        normalization="fixed_scale",
        normalizer_scale=1.0,
        repair_rho=0.0,
        success_ratio=1e-4,
        divergence_ratio=1e6,
    )
    assert np.all(outcome.bounded[:, 0])
    assert not np.any(outcome.bounded[:, 1])
