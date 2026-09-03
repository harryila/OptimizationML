from __future__ import annotations

from fractions import Fraction
from itertools import pairwise

import pytest

from passive_muon.yosida_stability import (
    INSUFFICIENT_RESOLVENT_PARAMETER,
    LOCKED_BASE_STRONG_MONOTONICITY,
    LOCKED_BETA,
    LOCKED_LEARNING_RATE,
    LOCKED_RATE_SQUARED,
    LOCKED_RESOLVENT_PARAMETER,
    LOCKED_TAU,
    LOCKED_YOSIDA_CENTER_GAIN,
    LOCKED_YOSIDA_LIPSCHITZ,
    LOCKED_YOSIDA_RESIDUAL_LIPSCHITZ,
    LOCKED_YOSIDA_STRONG_MONOTONICITY,
    YosidaSector,
    audit_yosida_stability,
    centered_residual_iqc_matrix,
    direct_p13_stiff_control,
    ema_nesterov_jury_control,
    exact_certificate_checks,
    insufficient_regularization_control,
    locked_yosida_pl_certificate,
    locked_yosida_sector,
    p13_base_origin_slope,
    regularized_base_origin_slope,
    resolvent_strong_firm_iqc_matrix,
    selected_origin_jury_control,
    skew_boundary_control,
    upper_lipschitz_gap,
    yosida_cocoercive_iqc_matrix,
    yosida_graph_iqc_matrix,
    yosida_pl_lmi_matrix,
    yosida_sector_iqc_matrix,
    yosida_strong_iqc_matrix,
)


def _quadratic_form(
    matrix: tuple[tuple[Fraction, ...], ...],
    vector: tuple[Fraction, ...],
) -> Fraction:
    return sum(
        (
            vector[row] * matrix[row][column] * vector[column]
            for row in range(len(vector))
            for column in range(len(vector))
        ),
        Fraction(0),
    )


def test_locked_yosida_sector_constants_are_exact() -> None:
    sector = locked_yosida_sector()

    assert sector.resolvent_parameter == LOCKED_RESOLVENT_PARAMETER == Fraction(1, 1_000)
    assert sector.base_strong_monotonicity == LOCKED_BASE_STRONG_MONOTONICITY == 1_000
    assert sector.denominator == 2
    assert sector.resolvent_lipschitz == Fraction(1, 2)
    assert sector.strong_monotonicity == LOCKED_YOSIDA_STRONG_MONOTONICITY == 500
    assert sector.lipschitz == LOCKED_YOSIDA_LIPSCHITZ == 1_000
    assert sector.cocoercivity == Fraction(1, 1_000)
    assert sector.center_gain == LOCKED_YOSIDA_CENTER_GAIN == 750
    assert sector.residual_lipschitz == LOCKED_YOSIDA_RESIDUAL_LIPSCHITZ == 250


def test_graph_sector_and_centered_residual_iqcs_are_exactly_equivalent() -> None:
    sector = locked_yosida_sector()
    graph = yosida_graph_iqc_matrix(sector)
    sector_iqc = yosida_sector_iqc_matrix(sector)
    centered = centered_residual_iqc_matrix(sector)

    assert graph == (
        (Fraction(-1_000), Fraction(3, 2)),
        (Fraction(3, 2), Fraction(-1, 500)),
    )
    assert sector_iqc == (
        (Fraction(-500_000), Fraction(750)),
        (Fraction(750), Fraction(-1)),
    )
    assert centered == sector_iqc
    scale = 1 / (sector.resolvent_parameter * sector.denominator)
    assert tuple(tuple(scale * value for value in row) for row in graph) == sector_iqc


def test_strong_cocoercive_and_resolvent_iqcs_have_the_claimed_forms() -> None:
    assert yosida_strong_iqc_matrix() == (
        (Fraction(-500), Fraction(1, 2)),
        (Fraction(1, 2), Fraction(0)),
    )
    assert yosida_cocoercive_iqc_matrix() == (
        (Fraction(0), Fraction(1, 2)),
        (Fraction(1, 2), Fraction(-1, 1_000)),
    )
    assert resolvent_strong_firm_iqc_matrix() == (
        (Fraction(0), Fraction(1, 2)),
        (Fraction(1, 2), Fraction(-2)),
    )

    # The lower endpoint B=mu*I attains both the strong-Y and strong-J bounds.
    x = Fraction(7, 11)
    y = Fraction(500) * x
    j = Fraction(1, 2) * x
    assert _quadratic_form(yosida_strong_iqc_matrix(), (x, y)) == 0
    assert _quadratic_form(resolvent_strong_firm_iqc_matrix(), (x, j)) == 0


def test_pulled_back_graph_iqc_implies_the_input_space_strong_bound() -> None:
    # Write v=mu*u+w with <w,u>>=0 in a two-coordinate witness.  The exact
    # identity leaves <w,u> + lambda/(1+lambda*mu)||w||^2.
    lam = LOCKED_RESOLVENT_PARAMETER
    mu = LOCKED_BASE_STRONG_MONOTONICITY
    u = (Fraction(2, 3), Fraction(0))
    w = (Fraction(0), Fraction(5, 7))
    v = (mu * u[0] + w[0], mu * u[1] + w[1])
    x = (u[0] + lam * v[0], u[1] + lam * v[1])
    inner_vx = sum(left * right for left, right in zip(v, x, strict=True))
    norm_x_squared = sum(value**2 for value in x)
    residual = inner_vx - Fraction(500) * norm_x_squared
    expected = lam / (1 + lam * mu) * sum(value**2 for value in w)
    assert residual == expected > 0


def test_locked_four_by_four_pl_lmi_replays_strictly() -> None:
    certificate = locked_yosida_pl_certificate()
    audit = audit_yosida_stability()
    lmi = yosida_pl_lmi_matrix()

    assert certificate.beta == LOCKED_BETA == Fraction(19, 20)
    assert certificate.learning_rate == LOCKED_LEARNING_RATE == Fraction(1, 32_000)
    assert certificate.tau == LOCKED_TAU == Fraction(499, 500)
    assert certificate.tau**2 == LOCKED_RATE_SQUARED == Fraction(249_001, 250_000)
    assert certificate.center_gain == 750
    assert certificate.residual_lipschitz == 250
    assert certificate.dimensionless_step == Fraction(15, 64)
    assert certificate.residual_ratio == Fraction(1, 3)
    assert certificate.storage == (
        (Fraction(674_389, 1_000_000), Fraction(-73_827, 1_000_000)),
        (Fraction(-73_827, 1_000_000), Fraction(12_368, 1_000_000)),
    )
    assert certificate.function_storage == Fraction(313_243, 1_000_000)
    assert certificate.interpolation_reverse_weight == Fraction(622_414, 1_000_000)
    assert certificate.lambda_residual_norm == Fraction(27_530, 1_000_000)
    assert audit.pl_audit.storage_leading_minors == (
        Fraction(674_389, 1_000_000),
        Fraction(2_890_417_223, 1_000_000_000_000),
    )
    assert audit.pl_audit.negative_lmi_leading_minors == (
        Fraction(608_720_541_672_414_373, 26_214_400_000_000_000_000),
        Fraction(
            8_368_016_172_090_059_871_782_660_037,
            13_107_200_000_000_000_000_000_000_000_000,
        ),
        Fraction(
            106_465_775_536_061_691_699_762_714_148_561,
            20_480_000_000_000_000_000_000_000_000_000_000_000,
        ),
        Fraction(
            5_484_909_006_727_797_363_797_664_114_262_745_923_553,
            10_240_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000,
        ),
    )
    assert audit.certified
    assert len(lmi) == 4 and all(len(row) == 4 for row in lmi)
    assert all(lmi[row][column] == lmi[column][row] for row in range(4) for column in range(4))


def test_actual_origin_control_passes_at_selected_lambda_exactly() -> None:
    control = selected_origin_jury_control()

    assert p13_base_origin_slope() == Fraction(
        66_983_030_848_166_374_889_967_883,
        10_410_454_400_000_000,
    )
    assert regularized_base_origin_slope() == Fraction(
        66_983_041_258_620_774_889_967_883,
        10_410_454_400_000_000,
    )
    assert control.resolvent_parameter == Fraction(1, 1_000)
    assert control.p_at_minus_one == Fraction(
        165_247_188_769_511_634_053_550_767_361,
        42_869_153_068_208_111_929_579_445_120,
    )
    assert control.dimensionless_gain < control.schur_gain_threshold == Fraction(780, 29)
    assert control.locally_schur_stable


def test_insufficient_regularization_and_direct_p13_controls_fail() -> None:
    insufficient = insufficient_regularization_control()
    direct = direct_p13_stiff_control()

    assert insufficient.resolvent_parameter == INSUFFICIENT_RESOLVENT_PARAMETER
    assert insufficient.p_at_minus_one == Fraction(
        -6_764_637_554_766_138_263_886_756_183,
        10_717_453_168_649_723_982_394_861_280,
    )
    assert not insufficient.locally_schur_stable
    assert direct.second_jury_margin < 0
    assert not direct.locally_schur_stable


def test_exact_jury_boundary_is_detected() -> None:
    slope = Fraction(2_000)
    curvature = Fraction(10)
    threshold = Fraction(780, 29)
    critical_step = threshold / (curvature * slope)
    boundary = ema_nesterov_jury_control(
        resolvent_parameter=Fraction(0),
        base_slope=slope,
        curvature=curvature,
        learning_rate=critical_step,
    )
    below = ema_nesterov_jury_control(
        resolvent_parameter=Fraction(0),
        base_slope=slope,
        curvature=curvature,
        learning_rate=critical_step / 2,
    )
    above = ema_nesterov_jury_control(
        resolvent_parameter=Fraction(0),
        base_slope=slope,
        curvature=curvature,
        learning_rate=critical_step * 2,
    )

    assert boundary.dimensionless_gain == threshold
    assert boundary.p_at_minus_one == 0
    assert not boundary.locally_schur_stable
    assert below.locally_schur_stable
    assert not above.locally_schur_stable


def test_skew_family_is_an_exact_full_sector_boundary_control() -> None:
    control = skew_boundary_control()

    assert control.skew_gain == 1_000
    assert control.yosida_real == 600
    assert control.yosida_imaginary == 200
    assert control.centered_circle_residual == 0
    assert control.sector_residual == 0

    for skew_gain in (Fraction(1), Fraction(17), Fraction(10_000)):
        candidate = skew_boundary_control(skew_gain)
        assert candidate.centered_circle_residual == 0
        assert candidate.sector_residual == 0


def test_upper_yosida_lipschitz_endpoint_is_a_sharp_supremum() -> None:
    gaps = tuple(upper_lipschitz_gap(Fraction(10**power)) for power in range(3, 10))
    assert all(left > right > 0 for left, right in pairwise(gaps))
    assert gaps[-1] < Fraction(1, 10)


def test_all_finite_exact_checks_pass() -> None:
    checks = exact_certificate_checks()
    audit = audit_yosida_stability()

    assert checks
    assert all(checks.values())
    assert dict(audit.checks) == checks
    assert audit.certified


def test_invalid_parameters_are_rejected() -> None:
    with pytest.raises(TypeError, match=r"fractions\.Fraction"):
        YosidaSector(0.001, Fraction(1_000))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive"):
        YosidaSector(Fraction(0), Fraction(1_000))
    with pytest.raises(ValueError, match="nonnegative"):
        ema_nesterov_jury_control(resolvent_parameter=Fraction(-1, 1_000))
    with pytest.raises(ValueError, match=r"\[0,1\)"):
        ema_nesterov_jury_control(
            resolvent_parameter=Fraction(0),
            beta=Fraction(1),
        )
    with pytest.raises(ValueError, match="positive"):
        upper_lipschitz_gap(Fraction(0))
    with pytest.raises(ValueError, match="at least"):
        upper_lipschitz_gap(Fraction(999))
