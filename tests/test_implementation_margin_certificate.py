from __future__ import annotations

from fractions import Fraction

import pytest

from passive_muon.implementation_margin_certificate import (
    LOCKED_BUDGET_GRID_DENOMINATOR,
    P10_DIAGNOSTIC_CHECKPOINT_COMMIT,
    P10_EXACT_ARTIFACT_COMMIT,
    P10_THEOREM_SOURCE_COMMIT,
    ImplementationMarginBudget,
    audit_implementation_margin,
    evaluate_implementation_margin,
    external_port_sensitivities,
    grid_boundary,
)
from passive_muon.outer_loop_roundoff_certificate import (
    locked_outer_loop_roundoff_certificate,
)


def test_zero_external_budget_replays_every_p10_fraction_exactly() -> None:
    p10 = locked_outer_loop_roundoff_certificate()
    p11 = evaluate_implementation_margin()

    assert p11.certified
    assert p11.rate == p10.rate == Fraction(549_700_907_325, 549_755_813_888)
    assert p11.constant_forcing == p10.constant_forcing == Fraction(2_162_331, 1_099_511_627_776)
    assert (
        p11.function_gap_ultimate
        == p10.function_gap_ultimate
        == Fraction(399_957_341_889, 549_755_813_888)
    )
    assert p11.reduction.momentum_port == p10.reduction.momentum_port
    assert p11.reduction.signal_port == p10.reduction.signal_port
    assert p11.reduction.actual_signal == p10.reduction.actual_signal
    assert p11.reduction.p9_reference_output == p10.reduction.operator_output
    assert p11.reduction.deployed_output == p10.reduction.operator_output
    assert p11.reduction.parameter_port == p10.reduction.parameter_port
    assert p11.reduction.effective_gradient_error == p10.reduction.effective_gradient_error
    assert p11.reduction.effective_operator_error == p10.reduction.effective_operator_error


def test_exact_external_sensitivities_include_shell_feedback_and_signal_cancellation() -> None:
    sensitivity = external_port_sensitivities()

    assert sensitivity.momentum_residual_from_gradient == Fraction(
        61_390_768_499_430_155_878_401,
        6_338_253_001_141_147_007_483_516_026_880,
    )
    assert sensitivity.signal_residual_from_gradient == Fraction(
        95_447_201_279_840_940_208_162_060_506_973_961_448_367_063,
        5_986_310_706_507_378_352_962_293_074_805_895_248_510_699_696_029_696,
    )
    assert sensitivity.actual_signal_from_gradient == Fraction(
        583_665_437_278_275_251_998_286_207_614_539_114_531_218_407_859_159,
        5_986_310_706_507_378_352_962_293_074_805_895_248_510_699_696_029_696,
    )
    assert sensitivity.effective_gradient_from_gradient == Fraction(
        316_912_711_447_825_849_804_331_679_745,
        316_912_650_057_057_350_374_175_801_344,
    )
    assert sensitivity.effective_operator_from_gradient < Fraction(1, 10_000)
    assert sensitivity.effective_operator_from_operator > 1


def test_locked_four_axis_grid_maxima_and_adjacent_negative_controls() -> None:
    expected_ticks = {
        "gradient_slope": 10_815_225_547,
        "operator_slope": 513_245_498_810,
        "gradient_intercept": 10_879_487_718,
        "operator_intercept": 13_351_103_462_525,
    }
    for axis, ticks in expected_ticks.items():
        boundary = grid_boundary(axis)
        assert boundary.accepted_value == Fraction(ticks, LOCKED_BUDGET_GRID_DENOMINATOR)
        assert boundary.rejected_value == Fraction(ticks + 1, LOCKED_BUDGET_GRID_DENOMINATOR)
        assert boundary.accepted_invariance_slack == 0
        assert boundary.rejected_invariance_slack == Fraction(-1, LOCKED_BUDGET_GRID_DENOMINATOR)
        assert boundary.active_constraint == "unit_storage_is_forward_invariant"
        assert boundary.rejected_checks == ("unit_storage_is_forward_invariant",)
        assert evaluate_implementation_margin(boundary.accepted).certified
        assert not evaluate_implementation_margin(boundary.rejected).certified


def test_jointly_nonzero_profile_closes_rate_invariant_objective_and_all_guards() -> None:
    audit = audit_implementation_margin()
    evaluation = audit.jointly_nonzero

    assert evaluation.reduction.budget == ImplementationMarginBudget(
        gradient_slope=Fraction(1, 4_096),
        gradient_intercept=Fraction(1, 4_096),
        operator_slope=Fraction(1, 128),
        operator_intercept=Fraction(1, 8),
    )
    assert evaluation.rate == Fraction(274_850_515_349, 274_877_906_944) < 1
    assert evaluation.constant_forcing == Fraction(1_254_603, 549_755_813_888)
    assert 1 - evaluation.rate - evaluation.constant_forcing == Fraction(
        53_528_587, 549_755_813_888
    )
    assert evaluation.function_gap_ultimate == Fraction(930_325_132_219, 1_099_511_627_776) < 1
    assert evaluation.guard.signal_at_one == Fraction(13_872_266_672_489, 549_755_813_888)
    assert evaluation.guard.deployed_output_at_one == Fraction(
        8_965_123_761_980_097, 274_877_906_944
    )
    assert evaluation.guard.rounded_step_at_one == Fraction(1_120_640_590_271, 1_099_511_627_776)
    assert evaluation.certified
    assert all(evaluation.checks.values())


def test_both_exact_frontier_slices_are_certified_and_have_adjacent_controls() -> None:
    audit = audit_implementation_margin()

    assert len(audit.slope_frontier) == len(audit.intercept_frontier) == 9
    for row in (*audit.slope_frontier, *audit.intercept_frontier):
        budget = (
            ImplementationMarginBudget()
            .replace(row.gradient_axis, row.gradient_value)
            .replace(row.operator_axis, row.operator_value)
        )
        outside = budget.replace(row.operator_axis, row.rejected_operator_value)
        gradient_outside = budget.replace(row.gradient_axis, row.rejected_gradient_value)
        assert row.rejected_gradient_value - row.gradient_value == Fraction(
            1, LOCKED_BUDGET_GRID_DENOMINATOR
        )
        assert row.rejected_operator_value - row.operator_value == Fraction(
            1, LOCKED_BUDGET_GRID_DENOMINATOR
        )
        assert row.invariance_slack == 0
        assert row.rejected_gradient_invariance_slack == Fraction(
            -1, LOCKED_BUDGET_GRID_DENOMINATOR
        )
        assert row.rejected_operator_invariance_slack == Fraction(
            -1, LOCKED_BUDGET_GRID_DENOMINATOR
        )
        assert evaluate_implementation_margin(budget).certified
        assert not evaluate_implementation_margin(outside).certified
        assert not evaluate_implementation_margin(gradient_outside).certified
        assert row.rejected_gradient_checks == ("unit_storage_is_forward_invariant",)
        assert row.rejected_checks == ("unit_storage_is_forward_invariant",)


def test_audit_passes_and_records_distinct_p10_provenance_roles() -> None:
    audit = audit_implementation_margin()

    assert audit.certified
    assert all(audit.checks.values())
    assert (
        len(
            {
                P10_THEOREM_SOURCE_COMMIT,
                P10_EXACT_ARTIFACT_COMMIT,
                P10_DIAGNOSTIC_CHECKPOINT_COMMIT,
            }
        )
        == 3
    )


def test_invalid_budget_and_boundary_inputs_are_rejected() -> None:
    with pytest.raises(TypeError, match=r"fractions\.Fraction"):
        ImplementationMarginBudget(gradient_slope=0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="nonnegative"):
        ImplementationMarginBudget(operator_intercept=Fraction(-1))
    with pytest.raises(TypeError, match="ImplementationMarginBudget"):
        evaluate_implementation_margin(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown budget axis"):
        grid_boundary("not_an_axis")
