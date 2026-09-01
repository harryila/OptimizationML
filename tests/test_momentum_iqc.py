from __future__ import annotations

from fractions import Fraction

import pytest
import torch

from passive_muon.momentum_experiment import (
    period_residual,
    rank_one_repaired_update,
    run_rank_one_momentum_trajectory,
)
from passive_muon.momentum_iqc import (
    audit_momentum_certificate,
    closed_form_factor_matrix,
    closed_form_sector_certificate,
    closed_form_sector_lmi_matrix,
    leading_principal_minors,
    locked_momentum_certificate,
    repaired_zero_input_gain,
    scalar_momentum_jury_margin,
    sector_alpha_limit,
)
from passive_muon.operator import orthogonalize

EXPECTED_ALPHA_LIMIT = Fraction(
    16_510_802_486_552_774_004_455_355_427,
    79_003_040_463_687_817_944_269_767_082_402,
)
EXPECTED_ETA_LIMIT = Fraction(
    1_026_784_428_047_408_433_965_200,
    39_501_520_231_843_908_972_134_883_541_201,
)


def test_closed_form_sector_boundary_and_strict_factorization_are_exact() -> None:
    locked = locked_momentum_certificate()
    alpha_limit = sector_alpha_limit(locked.beta, locked.normalized_strongness)
    assert alpha_limit == EXPECTED_ALPHA_LIMIT
    assert locked.certified_learning_rate_supremum == EXPECTED_ETA_LIMIT
    assert locked.dimensionless_step < alpha_limit

    closed_form = closed_form_sector_certificate(
        beta=locked.beta,
        normalized_strongness=locked.normalized_strongness,
        dimensionless_step=locked.dimensionless_step,
    )
    assert closed_form_sector_lmi_matrix(closed_form) == closed_form_factor_matrix(closed_form)
    assert all(value > 0 for value in leading_principal_minors(closed_form.storage))
    assert closed_form.lambda_strong > 0
    assert closed_form.lambda_lipschitz > 0
    assert closed_form.factor_gamma > 0


@pytest.mark.parametrize(
    ("beta", "nu", "alpha"),
    [
        (Fraction(0), Fraction(1, 2), Fraction(1, 2)),
        (Fraction(1, 2), Fraction(1, 2), Fraction(1, 5)),
        (Fraction(1, 2), Fraction(1), Fraction(1)),
    ],
)
def test_closed_form_identity_covers_zero_momentum_and_unit_sector_endpoints(
    beta: Fraction,
    nu: Fraction,
    alpha: Fraction,
) -> None:
    closed_form = closed_form_sector_certificate(
        beta=beta,
        normalized_strongness=nu,
        dimensionless_step=alpha,
    )
    assert closed_form_sector_lmi_matrix(closed_form) == closed_form_factor_matrix(closed_form)
    assert all(value > 0 for value in leading_principal_minors(closed_form.storage))


@pytest.mark.parametrize(
    ("beta", "nu", "expected"),
    [
        (Fraction(0), Fraction(1, 5), Fraction(2, 5)),
        (Fraction(9, 10), Fraction(1), Fraction(19, 5)),
    ],
)
def test_sector_boundary_has_the_first_order_and_symmetric_slope_endpoints(
    beta: Fraction,
    nu: Fraction,
    expected: Fraction,
) -> None:
    assert sector_alpha_limit(beta, nu) == expected


def test_locked_rate_lmi_passes_exact_sylvester_checks() -> None:
    audit = audit_momentum_certificate()
    assert audit.certified
    assert audit.storage_leading_minors == (
        Fraction(739_113, 1_000_000),
        Fraction(62_392_951_291, 500_000_000_000),
    )
    assert audit.negative_lmi_leading_minors == (
        Fraction("3575489181221465860709/72155694271700000000000"),
        Fraction("573554081670133549512459537/3607784713585000000000000000000000"),
        Fraction(
            "761336594901146778843890294690319/3607784713585000000000000000000000000000000000"
        ),
    )


def test_actual_floored_jordan_has_an_exact_outside_local_instability() -> None:
    locked = locked_momentum_certificate()
    gain = repaired_zero_input_gain(
        floor=locked.floor,
        repair_rho=locked.repair_rho,
    )
    certified_margin = scalar_momentum_jury_margin(
        learning_rate=locked.learning_rate,
        beta=locked.beta,
        curvature=locked.hessian_upper,
        repaired_gain=gain,
    )
    outside_margin = scalar_momentum_jury_margin(
        learning_rate=Fraction(1, 2_000),
        beta=locked.beta,
        curvature=locked.hessian_upper,
        repaired_gain=gain,
    )
    assert certified_margin > 0
    assert outside_margin == Fraction(
        -4_581_150_572_242_425_232_467_883,
        20_820_908_800_000_000_000_000_000,
    )


@pytest.mark.parametrize("momentum", [-10.0, -1.0, -0.01, 0.0, 0.01, 1.0, 10.0])
def test_rank_one_recurrence_uses_the_repository_operator(momentum: float) -> None:
    locked = locked_momentum_certificate()
    repair_rho = float(locked.repair_rho)
    scalar = rank_one_repaired_update(
        momentum=momentum,
        floor=1.0,
        repair_rho=repair_rho,
    )
    matrix = orthogonalize(
        torch.tensor([[momentum]], dtype=torch.float64),
        polynomial="jordan",
        normalization="floored_frobenius",
        floor=1.0,
        repair_rho=repair_rho,
        steps=5,
    )
    assert scalar == float(matrix.item())


def test_matched_rank_one_examples_replay() -> None:
    locked = locked_momentum_certificate()
    shared = {
        "beta": float(locked.beta),
        "curvature": float(locked.hessian_upper),
        "floor": float(locked.floor),
        "repair_rho": float(locked.repair_rho),
        "initial_position": 1.0,
        "iterations": 100_000,
    }
    stable = run_rank_one_momentum_trajectory(
        learning_rate=float(locked.learning_rate),
        **shared,
    )
    outside = run_rank_one_momentum_trajectory(
        learning_rate=0.0005,
        tail_length=12,
        **shared,
    )
    divergent = run_rank_one_momentum_trajectory(
        learning_rate=0.002,
        iterations=1_000,
        **{key: value for key, value in shared.items() if key != "iterations"},
    )

    assert stable.final_position == pytest.approx(2.3681013507509615e-38, rel=1e-12)
    assert stable.maximum_absolute_position == 1.0
    assert not stable.exceeded_divergence_threshold
    assert period_residual(outside, period=4) < 1e-14
    assert abs(outside.final_position) > 1e-4
    assert divergent.exceeded_divergence_threshold
    assert divergent.executed_iterations == 159
