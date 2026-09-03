from __future__ import annotations

from fractions import Fraction

import pytest
from flint import arb, ctx, fmpq

from passive_muon.additive_epsilon_deficit import (
    DEPLOYED_EPSILON,
    PAIR_DEFICIT_STRICT_LOWER,
    RADIUS_BANDS,
    radius_band_bound,
)
from passive_muon.radial_passivation_tradeoff import (
    LOCKED_BETA,
    LOCKED_LEARNING_RATE,
    P11_SIGNAL_NORM_BOUND,
    SWITCH_NORMALIZED_RADIUS,
    SWITCH_RAW_RATIO,
    TAIL_DERIVATIVE_LOWER,
    TAIL_PROJECTION_LOWER,
    UNIT_DEFICIT_UPPER,
    LogarithmicValue,
    constant_repair_magnitude,
    corrected_origin_slope,
    deficit_majorant,
    ema_nesterov_explicit_step_control,
    ema_nesterov_repair_only_step_control,
    evaluate_logarithmic_value,
    exact_certificate_checks,
    normalized_radius_from_raw_ratio,
    pinned_one_step_expansion_control,
    pointwise_deficit_upper,
    radial_integral_ball,
    radial_integral_terms,
    raw_ratio_from_normalized_radius,
    repair_derivative_envelope,
    repair_derivative_spectrum_ball,
    repair_lipschitz_constant,
    repair_magnitude_ball,
    tail_deficit_majorant,
    tail_deficit_majorant_derivative,
    universal_lipschitz_strict_lower,
)

EXPECTED_UPPER = Fraction(6_602_082_433_275_499_863, 41_641_817_600_000_000)


def _exact_ball(value: Fraction) -> arb:
    with ctx.workprec(512):
        return arb(fmpq(value.numerator, value.denominator))


def test_switch_transform_and_tail_constants_are_exactly_p12_final_row() -> None:
    assert normalized_radius_from_raw_ratio(SWITCH_RAW_RATIO) == SWITCH_NORMALIZED_RADIUS
    assert raw_ratio_from_normalized_radius(SWITCH_NORMALIZED_RADIUS) == SWITCH_RAW_RATIO
    assert Fraction(63, 10_000) == SWITCH_NORMALIZED_RADIUS
    assert Fraction(63, 9_937) == SWITCH_RAW_RATIO
    assert Fraction(-199_437, 1_250) == TAIL_DERIVATIVE_LOWER
    assert Fraction(-41_528_474_059_081, 260_261_360_000) == TAIL_PROJECTION_LOWER
    assert UNIT_DEFICIT_UPPER == EXPECTED_UPPER
    assert radius_band_bound(RADIUS_BANDS[-1]).projection_lower == TAIL_PROJECTION_LOWER


def test_piecewise_majorant_is_exactly_continuous_positive_and_nonincreasing() -> None:
    assert tail_deficit_majorant(SWITCH_RAW_RATIO) == UNIT_DEFICIT_UPPER
    assert deficit_majorant(Fraction(0)) == UNIT_DEFICIT_UPPER
    assert deficit_majorant(SWITCH_RAW_RATIO) == UNIT_DEFICIT_UPPER
    assert deficit_majorant(SWITCH_RAW_RATIO + Fraction(1, 10**9)) < UNIT_DEFICIT_UPPER

    # This exact sign decomposition proves d_tail'(z)<0 for every z>=0;
    # the following rational probes are diagnostics, not a sampled proof.
    assert (
        Fraction(-41_520_717_707_831, 260_261_360_000)
        == 2 * TAIL_DERIVATIVE_LOWER - TAIL_PROJECTION_LOWER
    )
    assert TAIL_PROJECTION_LOWER < 0
    for raw_ratio in (Fraction(0), SWITCH_RAW_RATIO, Fraction(1), Fraction(10**12)):
        assert tail_deficit_majorant(raw_ratio) > 0
        assert tail_deficit_majorant_derivative(raw_ratio) < 0


def test_global_constant_piece_dominates_every_earlier_p12_band() -> None:
    earlier = tuple(radius_band_bound(band).deficit_upper for band in RADIUS_BANDS[:-1])
    assert earlier == (
        Fraction(504, 5),
        Fraction(1_632_191_906_745_061_351, 10_531_913_600_000_000),
        Fraction(491_257_553_435_828_999, 3_099_060_000_000_000),
    )
    assert all(bound < UNIT_DEFICIT_UPPER for bound in earlier)


def test_closed_form_integral_is_exact_on_linear_piece_and_continuous_at_switch() -> None:
    zero = radial_integral_terms(Fraction(0))
    switch = radial_integral_terms(SWITCH_RAW_RATIO)
    assert zero == LogarithmicValue(Fraction(0), Fraction(0), Fraction(1))
    assert switch == LogarithmicValue(
        Fraction(41_856_817_278_490_137, 41_641_817_600_000_000),
        Fraction(0),
        Fraction(1),
    )
    assert radial_integral_ball(Fraction(0)) == 0
    assert radial_integral_ball(SWITCH_RAW_RATIO).contains(_exact_ball(switch.rational_part))


def test_logarithmic_integral_has_locked_exact_coefficients() -> None:
    raw_ratio = Fraction(7, 3)
    terms = radial_integral_terms(raw_ratio)
    reciprocal_coefficient = Fraction(6_205_081, 416_418_176)
    assert terms.logarithm_coefficient == Fraction(41_528_474_059_081, 260_261_360_000)
    assert terms.logarithm_argument == (1 + raw_ratio) / (1 + SWITCH_RAW_RATIO)
    assert terms.rational_part == UNIT_DEFICIT_UPPER * SWITCH_RAW_RATIO + (
        reciprocal_coefficient * (1 / (1 + raw_ratio) - Fraction(9_937, 10_000))
    )


@pytest.mark.parametrize("precision_bits", [160, 224])
def test_arb_magnitudes_match_locked_deployment_evaluations(precision_bits: int) -> None:
    at_one = repair_magnitude_ball(Fraction(1), DEPLOYED_EPSILON, precision_bits=precision_bits)
    at_p11 = repair_magnitude_ball(
        P11_SIGNAL_NORM_BOUND,
        DEPLOYED_EPSILON,
        precision_bits=precision_bits,
    )
    assert at_one > _exact_ball(Fraction("2571.8578264702121453403581662801632566"))
    assert at_one < _exact_ball(Fraction("2571.8578264702121453403581662801632567"))
    assert at_p11 > _exact_ball(Fraction("3086.9595802543014250403447615418376254"))
    assert at_p11 < _exact_ball(Fraction("3086.9595802543014250403447615418376255"))
    assert constant_repair_magnitude(Fraction(1), DEPLOYED_EPSILON) > 1_585_445_308
    assert constant_repair_magnitude(P11_SIGNAL_NORM_BOUND, DEPLOYED_EPSILON) > 40_006_343_820


def test_radial_and_tangential_derivative_modes_dominate_pointwise_deficit() -> None:
    for raw_norm in (
        Fraction(0),
        DEPLOYED_EPSILON * SWITCH_RAW_RATIO,
        Fraction(1),
        P11_SIGNAL_NORM_BOUND,
    ):
        envelope = repair_derivative_envelope(raw_norm, DEPLOYED_EPSILON)
        assert envelope.radial == pointwise_deficit_upper(raw_norm, DEPLOYED_EPSILON)
        assert envelope.tangential_lower == envelope.radial
        assert envelope.tangential_upper == repair_lipschitz_constant(DEPLOYED_EPSILON)
        assert 0 < envelope.minimum_eigenvalue <= envelope.tangential_upper

        radial, tangential = repair_derivative_spectrum_ball(
            raw_norm,
            DEPLOYED_EPSILON,
            precision_bits=192,
        )
        assert radial.contains(_exact_ball(envelope.radial))
        if envelope.raw_ratio <= SWITCH_RAW_RATIO:
            assert tangential.contains(_exact_ball(envelope.tangential_upper))
        else:
            assert tangential > _exact_ball(envelope.tangential_lower)
            assert tangential < _exact_ball(envelope.tangential_upper)


def test_global_lipschitz_constant_and_universal_pair_lower_scale_exactly() -> None:
    assert repair_lipschitz_constant(DEPLOYED_EPSILON) == EXPECTED_UPPER * 10_000_000
    assert universal_lipschitz_strict_lower(DEPLOYED_EPSILON) == 1_581_172_496
    assert universal_lipschitz_strict_lower(Fraction(1)) == PAIR_DEFICIT_STRICT_LOWER
    assert universal_lipschitz_strict_lower(Fraction(1)) < repair_lipschitz_constant(Fraction(1))


def test_scalar_quadratic_pinned_explicit_step_is_locally_unstable_exactly() -> None:
    control = ema_nesterov_explicit_step_control()
    expected_unit_origin_slope = Fraction(
        66_983_030_848_166_374_889_967_883,
        104_104_544_000_000_000_000_000,
    )
    assert corrected_origin_slope(Fraction(1)) == expected_unit_origin_slope
    assert control.beta == LOCKED_BETA
    assert control.learning_rate == LOCKED_LEARNING_RATE
    assert control.schur_gain_threshold == Fraction(780, 29)
    assert control.dimensionless_gain == Fraction(
        66_983_030_848_166_374_889_967_883,
        333_134_540_800_000_000_000,
    )
    assert control.dimensionless_gain > control.schur_gain_threshold
    assert control.second_jury_margin == Fraction(
        -1_942_248_049_655_000_871_809_068_607,
        66_626_908_160_000_000_000_000,
    )
    assert not control.locally_schur_stable


def test_repair_only_jury_and_finite_one_step_controls_isolate_stiffness() -> None:
    repair = ema_nesterov_repair_only_step_control()
    assert repair.schur_gain_threshold == Fraction(780, 29)
    assert repair.second_jury_margin == Fraction(
        -191_356_452_588_259_896_027,
        26_650_763_264_000_000,
    )
    assert not repair.locally_schur_stable

    one_step = pinned_one_step_expansion_control()
    assert one_step.signal == DEPLOYED_EPSILON / 399
    assert one_step.normalized_signal == Fraction(1, 400)
    assert one_step.repair_to_weight_ratio == Fraction(
        257_481_214_897_744_494_657,
        53_301_526_528_000_000,
    )
    assert one_step.objective_growth_strict_lower > 23_325_554


def test_finite_exact_certificate_checks_all_pass() -> None:
    checks = exact_certificate_checks()
    assert checks
    assert all(checks.values())


def test_invalid_arguments_are_rejected() -> None:
    with pytest.raises(TypeError, match=r"fractions\.Fraction"):
        deficit_majorant(0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="nonnegative"):
        radial_integral_terms(Fraction(-1))
    with pytest.raises(ValueError, match="positive"):
        repair_magnitude_ball(Fraction(1), Fraction(0))
    with pytest.raises(ValueError, match=r"\[0,1\)"):
        raw_ratio_from_normalized_radius(Fraction(1))
    with pytest.raises(ValueError, match="at least 80"):
        evaluate_logarithmic_value(
            LogarithmicValue(Fraction(0), Fraction(0), Fraction(1)),
            precision_bits=64,
        )
    with pytest.raises(ValueError, match=r"\[0,1\)"):
        ema_nesterov_explicit_step_control(beta=Fraction(1))
