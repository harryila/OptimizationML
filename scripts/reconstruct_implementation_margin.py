#!/usr/bin/env python3
"""Independently reconstruct the exact P11 implementation-margin audit.

This script imports neither :mod:`passive_muon` nor a numerical library.  It
starts from the public rational P7, P9, and P10 arithmetic constants, rebuilds
the frozen P10 envelopes, derives the two P11 external-port sensitivities, and
then recomputes every dyadic boundary and guard.  An optional canonical JSON
is read only after the complete internal replay has passed.
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from math import isqrt
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANONICAL = ROOT / "results/summaries/implementation_margin_certificate.json"

GRID = 2**40
FRONTIER_POINTS = 9
AXES = (
    "gradient_slope",
    "operator_slope",
    "gradient_intercept",
    "operator_intercept",
)

# Frozen P7/P9/P10 theorem and arithmetic inputs.
ROWS = 4_096
COLUMNS = 11_008
U32 = Fraction(1, 2**24)
T32 = Fraction(1, 2**150)
BETA = Fraction(19, 20)
ONE_MINUS_BETA = Fraction(1, 20)
ETA = Fraction(1, 32_000)
BETA32 = Fraction(15_938_355, 16_777_216)
ONE_MINUS_BETA32 = Fraction(13_421_773, 268_435_456)
ETA32 = Fraction(8_589_935, 274_877_906_944)
P9_SLOPE = Fraction(60_114_853, 549_755_813_888)
P9_INTERCEPT = Fraction(1_089_541_606_515, 549_755_813_888)
REPAIRED_LIPSCHITZ = Fraction(336_372_400_608_849, 260_261_360_000)
P7_RATE = Fraction(399_960_001, 400_000_000)
P7_GRADIENT_GAIN = Fraction(1, 2)
P7_OPERATOR_GAIN = Fraction(1, 2_000_000)
P7_SMOOTHNESS = Fraction(10)
P7_STORAGE_FUNCTION_LOWER = Fraction(13_533, 50_000)
GRADIENT_YOUNG = Fraction(1)
OPERATOR_YOUNG = Fraction(837)

P10_RATE = Fraction(549_700_907_325, 549_755_813_888)
P10_FORCING = Fraction(2_162_331, 1_099_511_627_776)
P10_FUNCTION_GAP = Fraction(399_957_341_889, 549_755_813_888)

SIGNAL_MAX = Fraction(2**116)
CERTIFICATE_OUTPUT_MAX = Fraction(2**15)
RUNTIME_OUTPUT_MAX = Fraction(2**16)
ROUNDED_STEP_MAX = Fraction(2)
FP32_MAX = Fraction((2**24 - 1) * 2**104)

PROVENANCE = {
    "theorem_source": "2d62b566e5a34895470d4eeb4b76789add826cbe",
    "exact_artifact": "6e7ea000692a269cbd3766146eb7423733d18694",
    "diagnostic_checkpoint": "3246972d44fc6e04c15cf4e205615acbb879df36",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--require-canonical", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _upper(value: Fraction) -> Fraction:
    if value < 0:
        raise ValueError("upper-bound input must be nonnegative")
    quotient, remainder = divmod(value.numerator * GRID, value.denominator)
    return Fraction(quotient + bool(remainder), GRID)


def _lower(value: Fraction) -> Fraction:
    if value < 0:
        raise ValueError("lower-bound input must be nonnegative")
    return Fraction(value.numerator * GRID // value.denominator, GRID)


def _sqrt_upper(value: Fraction) -> Fraction:
    if value < 0:
        raise ValueError("square-root input must be nonnegative")
    scaled = value.numerator * GRID**2
    candidate = isqrt(scaled // value.denominator)
    if candidate**2 * value.denominator < scaled:
        candidate += 1
    return Fraction(candidate, GRID)


def _storage_slope(momentum: Fraction, gradient: Fraction) -> Fraction:
    """Rebuild ``10*c^T P^-1 c`` from P7's exact storage matrix."""

    square = Fraction(
        124_750_000 * momentum**2
        + 1_592_500_000 * momentum * gradient
        + 18_108_750_000 * gradient**2,
        650_021,
    )
    return _sqrt_upper(square)


def _envelope(slope: Fraction, intercept: Fraction) -> dict[str, Fraction]:
    return {"slope": slope, "intercept": intercept}


def _at_one(envelope: dict[str, Fraction]) -> Fraction:
    return _upper(envelope["slope"] + envelope["intercept"])


def _budget(**values: Fraction) -> dict[str, Fraction]:
    budget = {axis: Fraction(0) for axis in AXES}
    budget.update(values)
    return budget


def _p10() -> dict[str, Any]:
    """Rebuild the P10 constants and envelopes from primitive inputs."""

    entries = ROWS * COLUMNS
    root_floor = isqrt(entries)
    root_entries = root_floor if root_floor**2 == entries else root_floor + 1
    two_operation_factor = 2 * U32 + U32**2
    c_beta = abs(BETA32 - BETA) + two_operation_factor * BETA32
    c_gradient = abs(ONE_MINUS_BETA32 - ONE_MINUS_BETA) + two_operation_factor * ONE_MINUS_BETA32
    ema_crumb = (3 + 2 * U32) * root_entries * T32
    c_eta = abs(ETA32 - ETA) + two_operation_factor * ETA32
    master_crumb = (2 + U32) * root_entries * T32
    cast_crumb = root_entries * T32

    represented_momentum = (1 + U32) ** 2 * BETA32
    represented_gradient = (1 + U32) ** 2 * ONE_MINUS_BETA32

    rm_m = c_beta
    rm_g = c_gradient * (1 + U32) + ONE_MINUS_BETA * U32
    rm_b = ema_crumb + (c_gradient + ONE_MINUS_BETA) * cast_crumb
    momentum_port = _envelope(_storage_slope(rm_m, rm_g), _upper(rm_b))

    mnext_m = represented_momentum
    mnext_g = represented_gradient * (1 + U32)
    mnext_b = ema_crumb + represented_gradient * cast_crumb
    rs_m = c_beta * mnext_m
    rs_g = c_beta * mnext_g + c_gradient * (1 + U32) + ONE_MINUS_BETA * U32
    rs_b = c_beta * mnext_b + ema_crumb + (c_gradient + ONE_MINUS_BETA) * cast_crumb
    signal_port = _envelope(_storage_slope(rs_m, rs_g), _upper(rs_b))

    signal_m = represented_momentum**2
    signal_g_hat = represented_momentum * represented_gradient + represented_gradient
    signal_b_hat = (1 + represented_momentum) * ema_crumb
    signal_g = signal_g_hat * (1 + U32)
    signal_b = signal_b_hat + signal_g_hat * cast_crumb
    actual_signal = _envelope(
        _storage_slope(signal_m, signal_g),
        _upper(signal_b),
    )

    operator_output = _envelope(
        _upper((REPAIRED_LIPSCHITZ + P9_SLOPE) * actual_signal["slope"]),
        _upper((REPAIRED_LIPSCHITZ + P9_SLOPE) * actual_signal["intercept"] + P9_INTERCEPT),
    )

    low_guard = Fraction(1, 2**16)
    low_frobenius = root_entries * low_guard
    parameter_port = _envelope(
        _upper(c_eta * operator_output["slope"]),
        _upper(c_eta * operator_output["intercept"] + U32 * low_frobenius + master_crumb),
    )
    effective_gradient = _envelope(
        _upper(momentum_port["slope"] / ONE_MINUS_BETA),
        _upper(momentum_port["intercept"] / ONE_MINUS_BETA),
    )
    effective_operator = _envelope(
        _upper(
            P9_SLOPE * actual_signal["slope"]
            + REPAIRED_LIPSCHITZ * (momentum_port["slope"] + signal_port["slope"])
            + parameter_port["slope"] / ETA
        ),
        _upper(
            P9_INTERCEPT
            + REPAIRED_LIPSCHITZ * (momentum_port["intercept"] + signal_port["intercept"])
            + parameter_port["intercept"] / ETA
        ),
    )

    rate = _upper(
        P7_RATE
        + P7_GRADIENT_GAIN * (1 + GRADIENT_YOUNG) * effective_gradient["slope"] ** 2
        + P7_OPERATOR_GAIN * (1 + OPERATOR_YOUNG) * effective_operator["slope"] ** 2
    )
    forcing = _upper(
        P7_GRADIENT_GAIN * (1 + 1 / GRADIENT_YOUNG) * effective_gradient["intercept"] ** 2
        + P7_OPERATOR_GAIN * (1 + 1 / OPERATOR_YOUNG) * effective_operator["intercept"] ** 2
    )
    function_gap = _upper(P7_SMOOTHNESS / P7_STORAGE_FUNCTION_LOWER * forcing / (1 - rate))

    momentum_at_one = _storage_slope(Fraction(1), Fraction(0))
    gradient_at_one = _storage_slope(Fraction(0), Fraction(1))
    represented_gradient_at_one = _upper((1 + U32) * gradient_at_one + cast_crumb)
    beta_momentum_at_one = _upper((1 + U32) * BETA32 * momentum_at_one + cast_crumb)
    weighted_gradient_at_one = _upper(
        (1 + U32) * ONE_MINUS_BETA32 * represented_gradient_at_one + cast_crumb
    )
    momentum_next_at_one = _upper(
        (1 + U32) * (beta_momentum_at_one + weighted_gradient_at_one) + cast_crumb
    )
    beta_momentum_next_at_one = _upper((1 + U32) * BETA32 * momentum_next_at_one + cast_crumb)
    ema_bounds = {
        "true_gradient": _upper(gradient_at_one),
        "represented_gradient": represented_gradient_at_one,
        "beta_momentum": beta_momentum_at_one,
        "weighted_gradient": weighted_gradient_at_one,
        "momentum_next": momentum_next_at_one,
        "beta_momentum_next": beta_momentum_next_at_one,
        "signal": _at_one(actual_signal),
    }

    pending_bound = (1 + U32) * (low_guard + ROUNDED_STEP_MAX) + T32
    middle_guard = Fraction(2**7)
    middle_candidate = (1 + U32) * (middle_guard + pending_bound) + T32
    master_checks = {
        "pending_below_three": pending_bound < 3,
        "middle_candidate_below_2^8": middle_candidate < 2**8,
        "new_low_within_guard": Fraction(1, 2**17) <= low_guard,
        "new_middle_within_guard": Fraction(2**6) <= middle_guard,
        "high_two_sum_cannot_overflow": Fraction(2**30) + middle_candidate < 2**31,
        "certificate_output_profile_implies_step_guard": (
            (1 + U32) * ETA32 * CERTIFICATE_OUTPUT_MAX + T32 <= ROUNDED_STEP_MAX
        ),
    }
    envelopes = {
        "momentum_port": momentum_port,
        "signal_port": signal_port,
        "actual_signal": actual_signal,
        "operator_output": operator_output,
        "parameter_port": parameter_port,
        "effective_gradient_error": effective_gradient,
        "effective_operator_error": effective_operator,
    }
    checks = {
        "root_entries_matches_lock": root_entries == 6_715,
        "rate_matches_lock": rate == P10_RATE,
        "forcing_matches_lock": forcing == P10_FORCING,
        "function_gap_matches_lock": function_gap == P10_FUNCTION_GAP,
        "rate_is_strict": rate < 1,
        "unit_storage_is_forward_invariant": forcing <= _lower(1 - rate),
        "signal_guard_closes": _at_one(actual_signal) <= SIGNAL_MAX,
        "output_guard_closes": _at_one(operator_output) <= CERTIFICATE_OUTPUT_MAX,
        "runtime_output_domain_contains_certificate": CERTIFICATE_OUTPUT_MAX <= RUNTIME_OUTPUT_MAX,
        "master_guards_close": all(master_checks.values()),
        "ema_intermediates_are_finite": all(value < FP32_MAX / 2 for value in ema_bounds.values()),
    }
    return {
        "root_entries": root_entries,
        "c_beta": c_beta,
        "c_gradient": c_gradient,
        "c_eta": c_eta,
        "ema_crumb": ema_crumb,
        "master_crumb": master_crumb,
        "cast_crumb": cast_crumb,
        "represented_momentum": represented_momentum,
        "represented_gradient": represented_gradient,
        "envelopes": envelopes,
        "rate": rate,
        "forcing": forcing,
        "function_gap": function_gap,
        "ema_bounds": ema_bounds,
        "master_checks": master_checks,
        "checks": checks,
        "certified": all(checks.values()),
    }


def _sensitivities(p10: dict[str, Any]) -> dict[str, Fraction]:
    represented_beta = p10["represented_momentum"]
    represented_gradient = p10["represented_gradient"]
    k_m = p10["c_gradient"] * (1 + U32) + ONE_MINUS_BETA * U32
    k_s = p10["c_beta"] * represented_gradient * (1 + U32) + k_m
    k_signal = represented_gradient * (1 + represented_beta) * (1 + U32)
    k_output = (REPAIRED_LIPSCHITZ + P9_SLOPE) * k_signal
    k_master_gradient = p10["c_eta"] * k_output
    k_master_operator = p10["c_eta"]
    k_xi = 1 + k_m / ONE_MINUS_BETA
    k_error_gradient = (
        P9_SLOPE * k_signal + REPAIRED_LIPSCHITZ * (k_m + k_s) + k_master_gradient / ETA
    )
    k_error_operator = 1 + k_master_operator / ETA
    return {
        "momentum_residual_from_gradient": k_m,
        "signal_residual_from_gradient": k_s,
        "actual_signal_from_gradient": k_signal,
        "p9_output_from_gradient": k_output,
        "effective_gradient_from_gradient": k_xi,
        "effective_operator_from_gradient": k_error_gradient,
        "effective_operator_from_operator": k_error_operator,
        "master_residual_from_gradient": k_master_gradient,
        "master_residual_from_operator": k_master_operator,
    }


def _augment(
    base: dict[str, Fraction],
    sensitivity: dict[str, Fraction],
    budget: dict[str, Fraction],
    *,
    gradient_key: str | None = None,
    operator_key: str | None = None,
) -> dict[str, Fraction]:
    gradient = sensitivity[gradient_key] if gradient_key is not None else Fraction(0)
    operator = sensitivity[operator_key] if operator_key is not None else Fraction(0)
    return _envelope(
        _upper(
            base["slope"]
            + gradient * budget["gradient_slope"]
            + operator * budget["operator_slope"]
        ),
        _upper(
            base["intercept"]
            + gradient * budget["gradient_intercept"]
            + operator * budget["operator_intercept"]
        ),
    )


def _evaluate(
    p10: dict[str, Any],
    sensitivity: dict[str, Fraction],
    budget: dict[str, Fraction],
) -> dict[str, Any]:
    base = p10["envelopes"]
    reduction = {
        "momentum_port": _augment(
            base["momentum_port"],
            sensitivity,
            budget,
            gradient_key="momentum_residual_from_gradient",
        ),
        "signal_port": _augment(
            base["signal_port"],
            sensitivity,
            budget,
            gradient_key="signal_residual_from_gradient",
        ),
        "actual_signal": _augment(
            base["actual_signal"],
            sensitivity,
            budget,
            gradient_key="actual_signal_from_gradient",
        ),
        "p9_reference_output": _augment(
            base["operator_output"],
            sensitivity,
            budget,
            gradient_key="p9_output_from_gradient",
        ),
        "deployed_output": _augment(
            base["operator_output"],
            sensitivity,
            budget,
            gradient_key="p9_output_from_gradient",
            operator_key="unit",
        ),
        "parameter_port": _augment(
            base["parameter_port"],
            sensitivity,
            budget,
            gradient_key="master_residual_from_gradient",
            operator_key="master_residual_from_operator",
        ),
        "effective_gradient_error": _augment(
            base["effective_gradient_error"],
            sensitivity,
            budget,
            gradient_key="effective_gradient_from_gradient",
        ),
        "effective_operator_error": _augment(
            base["effective_operator_error"],
            sensitivity,
            budget,
            gradient_key="effective_operator_from_gradient",
            operator_key="effective_operator_from_operator",
        ),
    }
    xi = reduction["effective_gradient_error"]
    error = reduction["effective_operator_error"]
    rate = _upper(
        P7_RATE
        + P7_GRADIENT_GAIN * (1 + GRADIENT_YOUNG) * xi["slope"] ** 2
        + P7_OPERATOR_GAIN * (1 + OPERATOR_YOUNG) * error["slope"] ** 2
    )
    forcing = _upper(
        P7_GRADIENT_GAIN * (1 + 1 / GRADIENT_YOUNG) * xi["intercept"] ** 2
        + P7_OPERATOR_GAIN * (1 + 1 / OPERATOR_YOUNG) * error["intercept"] ** 2
    )
    forcing_capacity = _lower(1 - rate) if rate <= 1 else Fraction(0)
    function_gap = None
    if rate < 1:
        function_gap = _upper(P7_SMOOTHNESS / P7_STORAGE_FUNCTION_LOWER * forcing / (1 - rate))

    signal_at_one = _at_one(reduction["actual_signal"])
    reference_output_at_one = _at_one(reduction["p9_reference_output"])
    deployed_output_at_one = _at_one(reduction["deployed_output"])
    rounded_step_at_one = _upper((1 + U32) * ETA32 * deployed_output_at_one + T32)
    zeta_at_one = budget["gradient_slope"] + budget["gradient_intercept"]
    ema = dict(p10["ema_bounds"])
    ema["pre_cast_gradient"] = _upper(ema["true_gradient"] + zeta_at_one)
    ema["represented_gradient"] = _upper(ema["represented_gradient"] + (1 + U32) * zeta_at_one)
    ema["weighted_gradient"] = _upper(
        ema["weighted_gradient"] + p10["represented_gradient"] * zeta_at_one
    )
    ema["momentum_next"] = _upper(
        ema["momentum_next"] + (1 + U32) * p10["represented_gradient"] * zeta_at_one
    )
    ema["beta_momentum_next"] = _upper(
        ema["beta_momentum_next"]
        + p10["represented_momentum"] * p10["represented_gradient"] * zeta_at_one
    )
    ema["signal"] = signal_at_one
    ema_finite = all(value < FP32_MAX / 2 for value in ema.values())
    checks = {
        "p10_base_certificate_replays": p10["certified"],
        "rate_is_strict": rate < 1,
        "unit_storage_is_forward_invariant": forcing <= forcing_capacity,
        "p9_signal_guard_closes": signal_at_one <= SIGNAL_MAX,
        "gradient_cast_and_ema_intermediates_are_finite": ema_finite,
        "deployed_output_is_below_2^15": deployed_output_at_one <= CERTIFICATE_OUTPUT_MAX,
        "certificate_output_is_in_runtime_domain": CERTIFICATE_OUTPUT_MAX <= RUNTIME_OUTPUT_MAX,
        "rounded_step_is_below_two": rounded_step_at_one <= ROUNDED_STEP_MAX,
        "middle_low_guards_are_preserved": all(p10["master_checks"].values()),
        "high_guard_is_explicitly_conditional": True,
        "deployed_output_range_is_finite": deployed_output_at_one < FP32_MAX / 2,
        "external_sensitivities_are_positive": all(value > 0 for value in sensitivity.values()),
    }
    return {
        "budget": budget,
        "reduction": reduction,
        "rate": rate,
        "forcing": forcing,
        "forcing_capacity": forcing_capacity,
        "function_gap": function_gap,
        "guards": {
            "signal_at_one": signal_at_one,
            "signal_max_abs": SIGNAL_MAX,
            "p9_reference_output_at_one": reference_output_at_one,
            "deployed_output_at_one": deployed_output_at_one,
            "certificate_output_max_abs": CERTIFICATE_OUTPUT_MAX,
            "runtime_output_max_abs": RUNTIME_OUTPUT_MAX,
            "rounded_step_at_one": rounded_step_at_one,
            "rounded_step_max_abs": ROUNDED_STEP_MAX,
            "ema_intermediate_frobenius_bounds": ema,
            "fp32_max_finite": FP32_MAX,
            "ema_intermediates_are_finite": ema_finite,
            "middle_low_invariant_checks": p10["master_checks"],
            "high_word_guard_is_conditional": True,
        },
        "checks": checks,
        "certified": all(checks.values()),
    }


def _grid_value(index: int) -> Fraction:
    return Fraction(index, GRID)


def _largest_feasible_index(
    p10: dict[str, Any],
    sensitivity: dict[str, Fraction],
    fixed: dict[str, Fraction],
    axis: str,
    *,
    upper_index: int | None = None,
) -> int:
    def accepted(index: int) -> bool:
        candidate = dict(fixed)
        candidate[axis] = _grid_value(index)
        return bool(_evaluate(p10, sensitivity, candidate)["certified"])

    if not accepted(0):
        raise AssertionError("fixed budget is infeasible")
    low = 0
    if upper_index is None:
        high = 1
        while accepted(high):
            low = high
            high *= 2
    else:
        high = upper_index + 1
        if accepted(high):
            raise AssertionError("upper index does not bracket a boundary")
    while low + 1 < high:
        middle = (low + high) // 2
        if accepted(middle):
            low = middle
        else:
            high = middle
    return low


def _boundary(
    p10: dict[str, Any],
    sensitivity: dict[str, Fraction],
    axis: str,
    fixed: dict[str, Fraction] | None = None,
    *,
    upper_index: int | None = None,
) -> dict[str, Any]:
    fixed = _budget() if fixed is None else fixed
    index = _largest_feasible_index(
        p10,
        sensitivity,
        fixed,
        axis,
        upper_index=upper_index,
    )
    accepted_budget = dict(fixed)
    accepted_budget[axis] = _grid_value(index)
    rejected_budget = dict(fixed)
    rejected_budget[axis] = _grid_value(index + 1)
    accepted = _evaluate(p10, sensitivity, accepted_budget)
    rejected = _evaluate(p10, sensitivity, rejected_budget)
    failures = tuple(name for name, passed in rejected["checks"].items() if not passed)
    return {
        "axis": axis,
        "accepted_budget": accepted_budget,
        "rejected_budget": rejected_budget,
        "accepted_value": _grid_value(index),
        "rejected_value": _grid_value(index + 1),
        "accepted_rate": accepted["rate"],
        "accepted_forcing": accepted["forcing"],
        "accepted_invariance_slack": 1 - accepted["rate"] - accepted["forcing"],
        "rejected_invariance_slack": 1 - rejected["rate"] - rejected["forcing"],
        "active_constraint": (
            "unit_storage_is_forward_invariant"
            if "unit_storage_is_forward_invariant" in failures
            else failures[0]
        ),
        "rejected_checks": failures,
        "accepted_certified": accepted["certified"],
        "rejected_certified": rejected["certified"],
    }


def _frontier(
    p10: dict[str, Any],
    sensitivity: dict[str, Fraction],
    kind: str,
    axes: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    gradient_axis = f"gradient_{kind}"
    operator_axis = f"operator_{kind}"
    gradient_ticks = int(axes[gradient_axis]["accepted_value"] * GRID)
    operator_ticks = int(axes[operator_axis]["accepted_value"] * GRID)
    rows: list[dict[str, Any]] = []
    for point in range(FRONTIER_POINTS):
        gradient_value = _grid_value(gradient_ticks * point // (FRONTIER_POINTS - 1))
        seed = _budget(**{gradient_axis: gradient_value})
        operator_boundary = _boundary(
            p10,
            sensitivity,
            operator_axis,
            seed,
            upper_index=operator_ticks,
        )
        gradient_boundary = _boundary(
            p10,
            sensitivity,
            gradient_axis,
            operator_boundary["accepted_budget"],
            upper_index=gradient_ticks,
        )
        accepted_budget = gradient_boundary["accepted_budget"]
        evaluation = _evaluate(p10, sensitivity, accepted_budget)
        rejected_operator_budget = dict(accepted_budget)
        rejected_operator_budget[operator_axis] += Fraction(1, GRID)
        rejected_operator = _evaluate(p10, sensitivity, rejected_operator_budget)
        rejected_operator_checks = tuple(
            name for name, passed in rejected_operator["checks"].items() if not passed
        )
        rows.append(
            {
                "gradient_axis": gradient_axis,
                "operator_axis": operator_axis,
                "gradient_value": accepted_budget[gradient_axis],
                "operator_value": accepted_budget[operator_axis],
                "rejected_gradient_value": gradient_boundary["rejected_value"],
                "rejected_operator_value": rejected_operator_budget[operator_axis],
                "rate": evaluation["rate"],
                "forcing": evaluation["forcing"],
                "invariance_slack": 1 - evaluation["rate"] - evaluation["forcing"],
                "deployed_output_at_one": evaluation["guards"]["deployed_output_at_one"],
                "rejected_gradient_invariance_slack": gradient_boundary[
                    "rejected_invariance_slack"
                ],
                "rejected_operator_invariance_slack": (
                    1 - rejected_operator["rate"] - rejected_operator["forcing"]
                ),
                "rejected_gradient_checks": gradient_boundary["rejected_checks"],
                "rejected_checks": rejected_operator_checks,
            }
        )
    return rows


def _stringify(value: Any) -> Any:
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _stringify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_stringify(item) for item in value]
    return value


def reconstruct() -> dict[str, Any]:
    """Rebuild the complete exact P11 audit without project imports."""

    p10 = _p10()
    sensitivity = _sensitivities(p10)
    # ``nu`` is additive at the deployed P9-output boundary.
    sensitivity["unit"] = Fraction(1)
    zero = _evaluate(p10, sensitivity, _budget())
    axes = {axis: _boundary(p10, sensitivity, axis) for axis in AXES}
    slope_frontier = _frontier(p10, sensitivity, "slope", axes)
    intercept_frontier = _frontier(p10, sensitivity, "intercept", axes)
    joint = _evaluate(
        p10,
        sensitivity,
        _budget(
            gradient_slope=Fraction(1, 4_096),
            operator_slope=Fraction(1, 128),
            gradient_intercept=Fraction(1, 4_096),
            operator_intercept=Fraction(1, 8),
        ),
    )
    reported_sensitivity = {name: value for name, value in sensitivity.items() if name != "unit"}
    checks = {
        "p10_reconstructs": p10["certified"],
        "zero_rate_reproduces_q10": zero["rate"] == P10_RATE,
        "zero_forcing_reproduces_D10": zero["forcing"] == P10_FORCING,
        "zero_function_gap_reproduces_p10": zero["function_gap"] == P10_FUNCTION_GAP,
        "zero_envelopes_reproduce_p10": all(
            zero["reduction"][p11_name] == p10["envelopes"][p10_name]
            for p11_name, p10_name in (
                ("momentum_port", "momentum_port"),
                ("signal_port", "signal_port"),
                ("actual_signal", "actual_signal"),
                ("p9_reference_output", "operator_output"),
                ("deployed_output", "operator_output"),
                ("parameter_port", "parameter_port"),
                ("effective_gradient_error", "effective_gradient_error"),
                ("effective_operator_error", "effective_operator_error"),
            )
        ),
        "all_sensitivities_are_positive": all(value > 0 for value in reported_sensitivity.values()),
        "four_axis_grid_maxima_bracket": len(axes) == 4
        and all(
            boundary["accepted_certified"]
            and not boundary["rejected_certified"]
            and boundary["rejected_value"] - boundary["accepted_value"] == Fraction(1, GRID)
            and boundary["accepted_invariance_slack"] == 0
            and boundary["rejected_invariance_slack"] == Fraction(-1, GRID)
            and boundary["rejected_checks"] == ("unit_storage_is_forward_invariant",)
            for boundary in axes.values()
        ),
        "axis_boundaries_have_exact_locked_ticks": {
            axis: int(boundary["accepted_value"] * GRID) for axis, boundary in axes.items()
        }
        == {
            "gradient_slope": 10_815_225_547,
            "operator_slope": 513_245_498_810,
            "gradient_intercept": 10_879_487_718,
            "operator_intercept": 13_351_103_462_525,
        },
        "slope_frontier_has_nine_exact_brackets": len(slope_frontier) == FRONTIER_POINTS
        and all(
            row["rejected_gradient_checks"] == ("unit_storage_is_forward_invariant",)
            and row["rejected_checks"] == ("unit_storage_is_forward_invariant",)
            and row["rejected_gradient_value"] - row["gradient_value"] == Fraction(1, GRID)
            and row["rejected_operator_value"] - row["operator_value"] == Fraction(1, GRID)
            and row["invariance_slack"] == 0
            for row in slope_frontier
        ),
        "intercept_frontier_has_nine_exact_brackets": len(intercept_frontier) == FRONTIER_POINTS
        and all(
            row["rejected_gradient_checks"] == ("unit_storage_is_forward_invariant",)
            and row["rejected_checks"] == ("unit_storage_is_forward_invariant",)
            and row["rejected_gradient_value"] - row["gradient_value"] == Fraction(1, GRID)
            and row["rejected_operator_value"] - row["operator_value"] == Fraction(1, GRID)
            and row["invariance_slack"] == 0
            for row in intercept_frontier
        ),
        "joint_all_positive_profile_certifies": joint["certified"]
        and all(value > 0 for value in joint["budget"].values()),
        "joint_profile_exact_rate": joint["rate"] == Fraction(274_850_515_349, 274_877_906_944),
        "joint_profile_exact_forcing": joint["forcing"] == Fraction(1_254_603, 549_755_813_888),
        "joint_profile_exact_guards": (
            joint["guards"]["signal_at_one"] == Fraction(13_872_266_672_489, 549_755_813_888)
            and joint["guards"]["deployed_output_at_one"]
            == Fraction(8_965_123_761_980_097, 274_877_906_944)
            and joint["guards"]["rounded_step_at_one"]
            == Fraction(1_120_640_590_271, 1_099_511_627_776)
        ),
        "joint_profile_closes_every_guard": all(joint["checks"].values()),
        "p10_provenance_roles_are_distinct": len(set(PROVENANCE.values())) == 3,
    }
    if not all(checks.values()):
        raise AssertionError([name for name, passed in checks.items() if not passed])
    return _stringify(
        {
            "grid_denominator": GRID,
            "frontier_points": FRONTIER_POINTS,
            "p10_provenance": PROVENANCE,
            "p10": p10,
            "external_port_sensitivities": reported_sensitivity,
            "zero_error": zero,
            "axis_boundaries": axes,
            "slope_frontier": slope_frontier,
            "intercept_frontier": intercept_frontier,
            "jointly_nonzero": joint,
            "checks": checks,
            "all_internal_exact_checks_passed": True,
        }
    )


def _display_path(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _exact(value: Any) -> Any:
    return value.get("exact") if isinstance(value, dict) and "exact" in value else value


def _compare_canonical(rebuilt: dict[str, Any], path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": _display_path(path),
            "status": "not_found",
            "comparisons": {},
            "all_exact_fields_match": None,
        }
    canonical = json.loads(path.read_text(encoding="utf-8"))
    zero = canonical.get("zero_external_error_replay", {})
    canonical_axes = canonical.get("one_axis_grid_maxima", {})
    pareto = canonical.get("pareto_frontiers", {})
    canonical_slope = pareto.get("slope_slice_b_g_eq_b_R_eq_0", [])
    canonical_intercept = pareto.get("intercept_slice_a_g_eq_a_R_eq_0", [])
    joint = canonical.get("jointly_nonzero_subunit_profile", {})

    def canonical_axis_ticks(axis: str) -> int | None:
        item = canonical_axes.get(axis, {})
        raw = item.get("largest_certified_grid_ticks")
        return int(raw) if raw is not None else None

    def canonical_axis_record(axis: str) -> tuple[Any, ...]:
        item = canonical_axes.get(axis, {})
        return (
            item.get("largest_certified_grid_ticks"),
            item.get("adjacent_rejected_grid_ticks"),
            str(_exact(item.get("accepted_q11"))),
            str(_exact(item.get("accepted_D11"))),
            str(_exact(item.get("accepted_invariance_slack"))),
            str(_exact(item.get("rejected_invariance_slack"))),
            item.get("active_constraint"),
            item.get("adjacent_control_failed_checks"),
        )

    def rebuilt_axis_record(item: dict[str, Any]) -> tuple[Any, ...]:
        return (
            int(Fraction(item["accepted_value"]) * GRID),
            int(Fraction(item["rejected_value"]) * GRID),
            item["accepted_rate"],
            item["accepted_forcing"],
            item["accepted_invariance_slack"],
            item["rejected_invariance_slack"],
            item["active_constraint"],
            item["rejected_checks"],
        )

    def canonical_frontier(
        rows: list[dict[str, Any]],
    ) -> list[tuple[int, int, int, int, str, str, str, str, str]]:
        return [
            (
                int(row.get("gradient_grid_ticks")),
                int(row.get("adjacent_rejected_gradient_grid_ticks")),
                int(row.get("operator_grid_ticks")),
                int(row.get("adjacent_rejected_operator_grid_ticks")),
                str(_exact(row.get("q11"))),
                str(_exact(row.get("D11"))),
                str(_exact(row.get("invariance_slack"))),
                str(_exact(row.get("adjacent_gradient_control_invariance_slack"))),
                str(_exact(row.get("adjacent_operator_control_invariance_slack"))),
            )
            for row in rows
        ]

    def rebuilt_frontier(
        rows: list[dict[str, Any]],
    ) -> list[tuple[int, int, int, int, str, str, str, str, str]]:
        return [
            (
                int(Fraction(row["gradient_value"]) * GRID),
                int(Fraction(row["rejected_gradient_value"]) * GRID),
                int(Fraction(row["operator_value"]) * GRID),
                int(Fraction(row["rejected_operator_value"]) * GRID),
                row["rate"],
                row["forcing"],
                row["invariance_slack"],
                row["rejected_gradient_invariance_slack"],
                row["rejected_operator_invariance_slack"],
            )
            for row in rows
        ]

    comparisons = {
        "schema": canonical.get("schema_version")
        == "passive-muon-implementation-margin-certificate-v1",
        "zero_rate": _exact(zero.get("q11")) == rebuilt["zero_error"]["rate"],
        "zero_forcing": _exact(zero.get("D11")) == rebuilt["zero_error"]["forcing"],
        "zero_function_gap": _exact(zero.get("function_gap_ultimate"))
        == rebuilt["zero_error"]["function_gap"],
        "exact_external_sensitivities": {
            name: str(_exact(value))
            for name, value in canonical.get("exact_external_sensitivities", {}).items()
        }
        == rebuilt["external_port_sensitivities"],
        "axis_ticks": {axis: canonical_axis_ticks(axis) for axis in AXES}
        == {
            axis: int(Fraction(item["accepted_value"]) * GRID)
            for axis, item in rebuilt["axis_boundaries"].items()
        },
        "axis_rejected_ticks": {
            axis: canonical_axes.get(axis, {}).get("adjacent_rejected_grid_ticks") for axis in AXES
        }
        == {
            axis: int(Fraction(item["rejected_value"]) * GRID)
            for axis, item in rebuilt["axis_boundaries"].items()
        },
        "axis_boundary_records": {axis: canonical_axis_record(axis) for axis in AXES}
        == {axis: rebuilt_axis_record(item) for axis, item in rebuilt["axis_boundaries"].items()},
        "slope_frontier": canonical_frontier(canonical_slope)
        == rebuilt_frontier(rebuilt["slope_frontier"]),
        "intercept_frontier": canonical_frontier(canonical_intercept)
        == rebuilt_frontier(rebuilt["intercept_frontier"]),
        "joint_rate": _exact(joint.get("q11")) == rebuilt["jointly_nonzero"]["rate"],
        "joint_forcing": _exact(joint.get("D11")) == rebuilt["jointly_nonzero"]["forcing"],
        "joint_function_gap": _exact(joint.get("function_gap_ultimate"))
        == rebuilt["jointly_nonzero"]["function_gap"],
        "joint_budget": {
            name: str(_exact(value)) for name, value in joint.get("budget", {}).items()
        }
        == rebuilt["jointly_nonzero"]["budget"],
        "joint_guard_signal": _exact(
            joint.get("guards_at_V_le_1", {}).get("signal_frobenius_bound")
        )
        == rebuilt["jointly_nonzero"]["guards"]["signal_at_one"],
        "joint_guard_output": _exact(
            joint.get("guards_at_V_le_1", {}).get("deployed_output_frobenius_bound")
        )
        == rebuilt["jointly_nonzero"]["guards"]["deployed_output_at_one"],
        "joint_guard_step": _exact(
            joint.get("guards_at_V_le_1", {}).get("rounded_step_entrywise_bound")
        )
        == rebuilt["jointly_nonzero"]["guards"]["rounded_step_at_one"],
    }
    matched = all(comparisons.values())
    return {
        "path": _display_path(path),
        "status": "matched" if matched else "mismatch",
        "comparisons": comparisons,
        "all_exact_fields_match": matched,
    }


def main() -> None:
    args = parse_args()
    reconstruction = reconstruct()
    comparison = _compare_canonical(reconstruction, args.canonical)
    if args.require_canonical and comparison["status"] != "matched":
        raise SystemExit(f"canonical P11 comparison failed: {comparison['status']}")
    payload = {
        "schema_version": "passive-muon-p11-independent-reconstruction-v1",
        "implementation_scope": {
            "project_package_imported": False,
            "numerical_library_imported": False,
            "arithmetic": "stdlib fractions.Fraction with explicit upward 2^-40 grid",
            "canonical_read_order": "only after complete independent reconstruction passes",
            "qualification": "exact second code path, not a human proof audit",
        },
        "reconstruction": reconstruction,
        "canonical_comparison": comparison,
        "all_internal_exact_checks_passed": reconstruction["all_internal_exact_checks_passed"],
        "all_exact_checks_passed": bool(
            reconstruction["all_internal_exact_checks_passed"] and comparison["status"] == "matched"
        ),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
