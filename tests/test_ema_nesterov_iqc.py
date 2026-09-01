from __future__ import annotations

from fractions import Fraction

import pytest

from passive_muon.ema_nesterov_experiment import (
    ema_nesterov_period_residual,
    run_rank_one_ema_nesterov_trajectory,
)
from passive_muon.ema_nesterov_iqc import (
    audit_ema_nesterov_certificate,
    ema_nesterov_jury_margin,
    ema_nesterov_local_learning_rate_threshold,
    locked_ema_nesterov_certificate,
    skew_boundary_polynomial,
)
from passive_muon.momentum_iqc import repaired_zero_input_gain


def test_locked_ema_nesterov_parameters_match_the_optimized_rational_design() -> None:
    locked = locked_ema_nesterov_certificate()
    assert locked.beta == Fraction(19, 20)
    assert locked.repair_margin == 648
    assert locked.repair_rho == Fraction(210_177_835_339_081, 260_261_360_000)
    assert locked.lipschitz_bound == Fraction(336_372_400_608_849, 260_261_360_000)
    assert locked.normalized_strongness == Fraction(208_209_088_000, 4_152_745_686_529)
    assert locked.dimensionless_step == Fraction(1, 400)
    assert locked.learning_rate == Fraction(65_065_340, 336_372_400_608_849)
    assert locked.rate_squared == Fraction(99_999, 100_000)


def test_conditioned_nesterov_input_keeps_both_one_minus_beta_factors() -> None:
    beta = Fraction(19, 20)
    y = Fraction(7, 5)
    z = Fraction(-11, 13)
    momentum_next = beta * z + (1 - beta) * y
    literal_signal = beta * momentum_next + (1 - beta) * y
    reduced_signal = beta**2 * z + (1 - beta**2) * y
    assert literal_signal == reduced_signal


def test_locked_ema_nesterov_lmi_passes_exact_sylvester_checks() -> None:
    audit = audit_ema_nesterov_certificate()
    assert audit.certified
    assert audit.storage_leading_minors == (
        Fraction(68_309, 100_000),
        Fraction(117_021_707_211, 1_000_000_000_000),
    )
    assert audit.negative_lmi_leading_minors == (
        Fraction(81_693_230_100_482_426_097, 2_657_757_239_378_560_000_000),
        Fraction(
            6_899_222_745_863_769_717_969_645_093,
            83_054_913_730_580_000_000_000_000_000_000_000,
        ),
        Fraction("76248222200293719542121626530563/1328878619689280000000000000000000000000000000"),
    )


def test_complex_skew_control_has_an_exact_tight_sign_bracket() -> None:
    locked = locked_ema_nesterov_certificate()
    lower = skew_boundary_polynomial(
        alpha=Fraction(2_710_354, 1_000_000_000),
        beta=locked.beta,
        normalized_strongness=locked.normalized_strongness,
    )
    upper = skew_boundary_polynomial(
        alpha=Fraction(2_710_355, 1_000_000_000),
        beta=locked.beta,
        normalized_strongness=locked.normalized_strongness,
    )
    assert lower < 0 < upper
    assert locked.dimensionless_step < Fraction(2_710_354, 1_000_000_000)


def test_actual_floored_jordan_local_threshold_and_outside_control_are_exact() -> None:
    locked = locked_ema_nesterov_certificate()
    gain = repaired_zero_input_gain(floor=locked.floor, repair_rho=locked.repair_rho)
    threshold = ema_nesterov_local_learning_rate_threshold(
        beta=locked.beta,
        curvature=locked.hessian_upper,
        repaired_gain=gain,
    )
    assert threshold == Fraction(
        8_120_154_432_000_000_000_000_000,
        3_901_919_808_117_690_731_741_568_607,
    )
    assert (
        ema_nesterov_jury_margin(
            learning_rate=locked.learning_rate,
            beta=locked.beta,
            curvature=locked.hessian_upper,
            repaired_gain=gain,
        )
        > 0
    )
    assert (
        ema_nesterov_jury_margin(
            learning_rate=Fraction(1, 400),
            beta=locked.beta,
            curvature=locked.hessian_upper,
            repaired_gain=gain,
        )
        < 0
    )


def test_matched_ema_nesterov_rank_one_examples_replay() -> None:
    locked = locked_ema_nesterov_certificate()
    shared = {
        "beta": float(locked.beta),
        "curvature": float(locked.hessian_upper),
        "floor": float(locked.floor),
        "repair_rho": float(locked.repair_rho),
        "initial_position": 1.0,
        "initial_momentum": 0.0,
    }
    stable = run_rank_one_ema_nesterov_trajectory(
        learning_rate=float(locked.learning_rate),
        iterations=100_000,
        **shared,
    )
    outside = run_rank_one_ema_nesterov_trajectory(
        learning_rate=1 / 400,
        iterations=20_000,
        tail_length=12,
        **shared,
    )
    divergent = run_rank_one_ema_nesterov_trajectory(
        learning_rate=1 / 200,
        iterations=1_000,
        **shared,
    )

    assert stable.final_position == pytest.approx(7.643061150755369e-113, rel=1e-12)
    assert stable.maximum_absolute_position == 1.0
    assert not stable.exceeded_divergence_threshold
    assert ema_nesterov_period_residual(outside, period=2) < 1e-13
    assert abs(outside.final_position) > 1e-4
    assert divergent.exceeded_divergence_threshold
    assert divergent.first_hundredfold_amplification_iteration == 6
    assert divergent.executed_iterations == 264
