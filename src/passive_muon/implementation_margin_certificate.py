"""Exact P11 implementation-margin certificate above the frozen P10 shell.

P10 assumes that the exact objective gradient is cast once to binary32 and
that its P9 proof-reference operator is the deployed operator.  P11 exposes
the two remaining interfaces without changing the objective class, repaired
operator, learning rate, repair, matrix shape, or arithmetic shell.

Let ``y=g+zeta`` be the real value presented to P10's final gradient cast and
let ``U_dep=U_P9+nu`` be the finite represented binary32 output presented to
the compensated master update.  The external budgets are

``||zeta||_F <= a_g*sqrt(V)+b_g`` and
``||nu||_F <= a_R*sqrt(V)+b_R``.

The common ``zeta`` term in the EMA and Nesterov equations is placed in P7's
gradient port.  It therefore cancels exactly from the signal mismatch; only
the rounding residuals remain there.  This cancellation is essential for a
useful margin and is recorded explicitly below.

Every P10 envelope is already a rigorous upper bound.  P11 adds the exact
incremental sensitivity to that frozen envelope and rounds the resulting
theorem-facing coefficient upward once on the declared ``2**-40`` grid.  The
same grid is used for reported acceptance budgets.  Consequently, an "axis
maximum" means the largest accepted nonnegative multiple of ``2**-40``; the
adjacent grid point is an exact negative control.  No sampled search carries
the theorem claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache

from passive_muon.finite_precision_outer_loop import (
    FP32_HALF_MIN_SUBNORMAL,
    FP32_UNIT_ROUNDOFF,
    LEARNING_RATE_FP32_EXACT,
    LOCKED_ROUNDED_STEP_MAX_ABS,
    FinitePrecisionOuterLoopConfig,
    outer_residual_envelope,
)
from passive_muon.outer_loop_roundoff_certificate import (
    AffineStorageEnvelope,
    OuterLoopRoundoffCertificate,
    locked_outer_loop_roundoff_certificate,
)
from passive_muon.scalable_mixed_precision_certificate import (
    FP32_MAX_FINITE,
    P7_SMOOTHNESS,
    P7_STORAGE_FUNCTION_LOWER,
    UPPER_GRID_DENOMINATOR,
)

IMPLEMENTATION_MARGIN_SCHEMA_VERSION = "passive-muon-implementation-margin-certificate-v1"
LOCKED_BUDGET_GRID_DENOMINATOR = UPPER_GRID_DENOMINATOR
LOCKED_FRONTIER_POINTS = 9

P10_THEOREM_SOURCE_COMMIT = "2d62b566e5a34895470d4eeb4b76789add826cbe"
P10_EXACT_ARTIFACT_COMMIT = "6e7ea000692a269cbd3766146eb7423733d18694"
P10_DIAGNOSTIC_CHECKPOINT_COMMIT = "3246972d44fc6e04c15cf4e205615acbb879df36"

_AXES = (
    "gradient_slope",
    "operator_slope",
    "gradient_intercept",
    "operator_intercept",
)


def _upper(value: Fraction) -> Fraction:
    """Round a nonnegative rational upward on the locked exact grid."""

    if value < 0:
        raise ValueError("an upper-certificate quantity must be nonnegative")
    numerator, remainder = divmod(
        value.numerator * LOCKED_BUDGET_GRID_DENOMINATOR,
        value.denominator,
    )
    return Fraction(numerator + bool(remainder), LOCKED_BUDGET_GRID_DENOMINATOR)


def _lower(value: Fraction) -> Fraction:
    """Round a nonnegative rational downward on the locked exact grid."""

    if value < 0:
        raise ValueError("a lower-certificate quantity must be nonnegative")
    return Fraction(
        value.numerator * LOCKED_BUDGET_GRID_DENOMINATOR // value.denominator,
        LOCKED_BUDGET_GRID_DENOMINATOR,
    )


@dataclass(frozen=True)
class ImplementationMarginBudget:
    """Four affine external-port coefficients in physical Frobenius units."""

    gradient_slope: Fraction = Fraction(0)
    gradient_intercept: Fraction = Fraction(0)
    operator_slope: Fraction = Fraction(0)
    operator_intercept: Fraction = Fraction(0)

    def __post_init__(self) -> None:
        for name in _AXES:
            value = getattr(self, name)
            if not isinstance(value, Fraction):
                raise TypeError(f"{name} must be a fractions.Fraction")
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")

    @classmethod
    def on_axis(cls, axis: str, value: Fraction) -> ImplementationMarginBudget:
        """Return a budget with only ``axis`` nonzero."""

        if axis not in _AXES:
            raise ValueError(f"unknown budget axis {axis!r}")
        return cls(**{axis: value})

    def replace(self, axis: str, value: Fraction) -> ImplementationMarginBudget:
        """Return a copy with one named coefficient replaced."""

        if axis not in _AXES:
            raise ValueError(f"unknown budget axis {axis!r}")
        values = {name: getattr(self, name) for name in _AXES}
        values[axis] = value
        return ImplementationMarginBudget(**values)


_ZERO_BUDGET = ImplementationMarginBudget()


@dataclass(frozen=True)
class ExternalPortSensitivities:
    """Exact incremental coefficients induced by the two external ports."""

    momentum_residual_from_gradient: Fraction
    signal_residual_from_gradient: Fraction
    actual_signal_from_gradient: Fraction
    p9_output_from_gradient: Fraction
    effective_gradient_from_gradient: Fraction
    effective_operator_from_gradient: Fraction
    effective_operator_from_operator: Fraction
    master_residual_from_gradient: Fraction
    master_residual_from_operator: Fraction


@dataclass(frozen=True)
class ImplementationMarginReduction:
    """P11 theorem-facing affine envelopes for one external budget."""

    budget: ImplementationMarginBudget
    momentum_port: AffineStorageEnvelope
    signal_port: AffineStorageEnvelope
    actual_signal: AffineStorageEnvelope
    p9_reference_output: AffineStorageEnvelope
    deployed_output: AffineStorageEnvelope
    parameter_port: AffineStorageEnvelope
    effective_gradient_error: AffineStorageEnvelope
    effective_operator_error: AffineStorageEnvelope


@dataclass(frozen=True)
class ImplementationMarginGuard:
    """Exact guard closure on the locked unit-storage set."""

    signal_at_one: Fraction
    signal_max_abs: Fraction
    p9_reference_output_at_one: Fraction
    deployed_output_at_one: Fraction
    certificate_output_max_abs: Fraction
    runtime_output_max_abs: Fraction
    rounded_step_at_one: Fraction
    rounded_step_max_abs: Fraction
    ema_intermediate_frobenius_bounds: dict[str, Fraction]
    fp32_max_finite: Fraction
    ema_intermediates_are_finite: bool
    high_word_guard_is_conditional: bool
    middle_low_invariant_checks: dict[str, bool]


@dataclass(frozen=True)
class ImplementationMarginEvaluation:
    """Exact P11 storage and range evaluation at one four-port budget."""

    reduction: ImplementationMarginReduction
    rate: Fraction
    constant_forcing: Fraction
    forcing_capacity: Fraction
    function_gap_ultimate: Fraction | None
    guard: ImplementationMarginGuard
    checks: dict[str, bool]

    @property
    def certified(self) -> bool:
        return bool(self.checks) and all(self.checks.values())


@dataclass(frozen=True)
class GridBoundary:
    """Largest accepted point and adjacent rejected point on one dyadic ray."""

    varied_axis: str
    fixed_budget: ImplementationMarginBudget
    accepted: ImplementationMarginBudget
    rejected: ImplementationMarginBudget
    accepted_value: Fraction
    rejected_value: Fraction
    grid_step: Fraction
    accepted_rate: Fraction
    accepted_forcing: Fraction
    accepted_invariance_slack: Fraction
    rejected_invariance_slack: Fraction
    active_constraint: str
    rejected_checks: tuple[str, ...]


@dataclass(frozen=True)
class FrontierRow:
    """One exact two-port Pareto boundary bracket on the budget grid."""

    gradient_axis: str
    operator_axis: str
    gradient_value: Fraction
    operator_value: Fraction
    rejected_gradient_value: Fraction
    rejected_operator_value: Fraction
    rate: Fraction
    forcing: Fraction
    invariance_slack: Fraction
    deployed_output_at_one: Fraction
    rejected_gradient_invariance_slack: Fraction
    rejected_operator_invariance_slack: Fraction
    rejected_gradient_checks: tuple[str, ...]
    rejected_checks: tuple[str, ...]


@dataclass(frozen=True)
class ImplementationMarginAudit:
    """Complete P11 success-gate audit."""

    p10: OuterLoopRoundoffCertificate
    sensitivities: ExternalPortSensitivities
    zero_error: ImplementationMarginEvaluation
    axis_boundaries: dict[str, GridBoundary]
    slope_frontier: tuple[FrontierRow, ...]
    intercept_frontier: tuple[FrontierRow, ...]
    jointly_nonzero: ImplementationMarginEvaluation
    checks: dict[str, bool]

    @property
    def certified(self) -> bool:
        return bool(self.checks) and all(self.checks.values())


@lru_cache(maxsize=1)
def _locked_context() -> tuple[OuterLoopRoundoffCertificate, ExternalPortSensitivities]:
    p10 = locked_outer_loop_roundoff_certificate()
    reduction = p10.reduction
    raw = outer_residual_envelope(
        FinitePrecisionOuterLoopConfig((reduction.shape.rows, reduction.shape.columns))
    )
    u = FP32_UNIT_ROUNDOFF
    one_minus_beta = 1 - p10.beta
    represented_beta = reduction.represented_momentum_coefficient
    represented_gradient = reduction.represented_gradient_coefficient

    # rm and rs are defined relative to y=g+zeta.  Thus zeta enters P7 as
    # zeta+rm/a, while the explicit a*zeta terms cancel from rs-rm.
    k_m = reduction.ema_gradient_coefficient * (1 + u) + one_minus_beta * u
    k_s = reduction.ema_momentum_coefficient * represented_gradient * (1 + u) + k_m
    k_signal = represented_gradient * (1 + represented_beta) * (1 + u)
    k_output = (reduction.repaired_lipschitz + reduction.p9_binary32_slope) * k_signal
    k_master_gradient = raw.master_output_coefficient * k_output
    k_master_operator = raw.master_output_coefficient
    k_xi = 1 + k_m / one_minus_beta
    k_error_gradient = (
        reduction.p9_binary32_slope * k_signal
        + reduction.repaired_lipschitz * (k_m + k_s)
        + k_master_gradient / p10.learning_rate
    )
    k_error_operator = 1 + k_master_operator / p10.learning_rate
    sensitivities = ExternalPortSensitivities(
        momentum_residual_from_gradient=k_m,
        signal_residual_from_gradient=k_s,
        actual_signal_from_gradient=k_signal,
        p9_output_from_gradient=k_output,
        effective_gradient_from_gradient=k_xi,
        effective_operator_from_gradient=k_error_gradient,
        effective_operator_from_operator=k_error_operator,
        master_residual_from_gradient=k_master_gradient,
        master_residual_from_operator=k_master_operator,
    )
    return p10, sensitivities


def external_port_sensitivities() -> ExternalPortSensitivities:
    """Return the locked exact (not pre-rounded) P11 sensitivities."""

    return _locked_context()[1]


def _augment(
    base: AffineStorageEnvelope,
    *,
    gradient_sensitivity: Fraction = Fraction(0),
    operator_sensitivity: Fraction = Fraction(0),
    budget: ImplementationMarginBudget,
) -> AffineStorageEnvelope:
    return AffineStorageEnvelope(
        slope=_upper(
            base.slope
            + gradient_sensitivity * budget.gradient_slope
            + operator_sensitivity * budget.operator_slope
        ),
        intercept=_upper(
            base.intercept
            + gradient_sensitivity * budget.gradient_intercept
            + operator_sensitivity * budget.operator_intercept
        ),
    )


def _build_reduction(budget: ImplementationMarginBudget) -> ImplementationMarginReduction:
    p10, sensitivity = _locked_context()
    base = p10.reduction
    momentum = _augment(
        base.momentum_port,
        gradient_sensitivity=sensitivity.momentum_residual_from_gradient,
        budget=budget,
    )
    signal_port = _augment(
        base.signal_port,
        gradient_sensitivity=sensitivity.signal_residual_from_gradient,
        budget=budget,
    )
    actual_signal = _augment(
        base.actual_signal,
        gradient_sensitivity=sensitivity.actual_signal_from_gradient,
        budget=budget,
    )
    p9_reference_output = _augment(
        base.operator_output,
        gradient_sensitivity=sensitivity.p9_output_from_gradient,
        budget=budget,
    )
    deployed_output = _augment(
        base.operator_output,
        gradient_sensitivity=sensitivity.p9_output_from_gradient,
        operator_sensitivity=Fraction(1),
        budget=budget,
    )
    parameter_port = _augment(
        base.parameter_port,
        gradient_sensitivity=sensitivity.master_residual_from_gradient,
        operator_sensitivity=sensitivity.master_residual_from_operator,
        budget=budget,
    )
    effective_gradient = _augment(
        base.effective_gradient_error,
        gradient_sensitivity=sensitivity.effective_gradient_from_gradient,
        budget=budget,
    )
    effective_operator = _augment(
        base.effective_operator_error,
        gradient_sensitivity=sensitivity.effective_operator_from_gradient,
        operator_sensitivity=sensitivity.effective_operator_from_operator,
        budget=budget,
    )
    return ImplementationMarginReduction(
        budget=budget,
        momentum_port=momentum,
        signal_port=signal_port,
        actual_signal=actual_signal,
        p9_reference_output=p9_reference_output,
        deployed_output=deployed_output,
        parameter_port=parameter_port,
        effective_gradient_error=effective_gradient,
        effective_operator_error=effective_operator,
    )


def evaluate_implementation_margin(
    budget: ImplementationMarginBudget = _ZERO_BUDGET,
) -> ImplementationMarginEvaluation:
    """Evaluate the exact P11 inequality and every locked P10 range guard."""

    if not isinstance(budget, ImplementationMarginBudget):
        raise TypeError("budget must be an ImplementationMarginBudget")
    p10, sensitivity = _locked_context()
    reduction = _build_reduction(budget)
    xi = reduction.effective_gradient_error
    error = reduction.effective_operator_error
    theta_g = Fraction(p10.gradient_young)
    theta_e = Fraction(p10.operator_young)
    rate = _upper(
        p10.p7_rate
        + p10.p7_gradient_gain * (1 + theta_g) * xi.slope**2
        + p10.p7_operator_gain * (1 + theta_e) * error.slope**2
    )
    forcing = _upper(
        p10.p7_gradient_gain * (1 + 1 / theta_g) * xi.intercept**2
        + p10.p7_operator_gain * (1 + 1 / theta_e) * error.intercept**2
    )
    forcing_capacity = _lower(1 - rate) if rate <= 1 else Fraction(0)
    function_gap = None
    if rate < 1:
        function_gap = _upper(P7_SMOOTHNESS / P7_STORAGE_FUNCTION_LOWER * forcing / (1 - rate))

    signal_at_one = reduction.actual_signal.at_storage(Fraction(1))
    reference_output_at_one = reduction.p9_reference_output.at_storage(Fraction(1))
    deployed_output_at_one = reduction.deployed_output.at_storage(Fraction(1))
    rounded_step_at_one = _upper(
        (1 + FP32_UNIT_ROUNDOFF) * LEARNING_RATE_FP32_EXACT * deployed_output_at_one
        + FP32_HALF_MIN_SUBNORMAL
    )

    zeta_at_one = budget.gradient_slope + budget.gradient_intercept
    represented_gradient = p10.guard.ema_intermediate_frobenius_bounds["represented_gradient"]
    weighted_gradient = p10.guard.ema_intermediate_frobenius_bounds["weighted_gradient"]
    momentum_next = p10.guard.ema_intermediate_frobenius_bounds["momentum_next"]
    beta_momentum_next = p10.guard.ema_intermediate_frobenius_bounds["beta_momentum_next"]
    rg = p10.reduction.represented_gradient_coefficient
    rb = p10.reduction.represented_momentum_coefficient
    ema_bounds = dict(p10.guard.ema_intermediate_frobenius_bounds)
    ema_bounds["pre_cast_gradient"] = _upper(ema_bounds["true_gradient"] + zeta_at_one)
    ema_bounds["represented_gradient"] = _upper(
        represented_gradient + (1 + FP32_UNIT_ROUNDOFF) * zeta_at_one
    )
    ema_bounds["weighted_gradient"] = _upper(weighted_gradient + rg * zeta_at_one)
    ema_bounds["momentum_next"] = _upper(
        momentum_next + (1 + FP32_UNIT_ROUNDOFF) * rg * zeta_at_one
    )
    ema_bounds["beta_momentum_next"] = _upper(beta_momentum_next + rb * rg * zeta_at_one)
    ema_bounds["signal"] = signal_at_one
    ema_finite = all(value < FP32_MAX_FINITE / 2 for value in ema_bounds.values())

    guard = ImplementationMarginGuard(
        signal_at_one=signal_at_one,
        signal_max_abs=p10.guard.p9_signal_max_abs,
        p9_reference_output_at_one=reference_output_at_one,
        deployed_output_at_one=deployed_output_at_one,
        certificate_output_max_abs=p10.guard.certificate_operator_output_max_abs,
        runtime_output_max_abs=p10.guard.runtime_operator_output_max_abs,
        rounded_step_at_one=rounded_step_at_one,
        rounded_step_max_abs=p10.guard.rounded_step_max_abs,
        ema_intermediate_frobenius_bounds=ema_bounds,
        fp32_max_finite=FP32_MAX_FINITE,
        ema_intermediates_are_finite=ema_finite,
        high_word_guard_is_conditional=p10.guard.high_word_guard_is_conditional,
        middle_low_invariant_checks=dict(p10.guard.middle_low_invariant_checks),
    )
    checks = {
        "p10_base_certificate_replays": p10.certified,
        "rate_is_strict": rate < 1,
        "unit_storage_is_forward_invariant": forcing <= forcing_capacity,
        "p9_signal_guard_closes": signal_at_one <= guard.signal_max_abs,
        "gradient_cast_and_ema_intermediates_are_finite": ema_finite,
        "deployed_output_is_below_2^15": (
            deployed_output_at_one <= guard.certificate_output_max_abs
        ),
        "certificate_output_is_in_runtime_domain": (
            guard.certificate_output_max_abs <= guard.runtime_output_max_abs
        ),
        "rounded_step_is_below_two": (
            rounded_step_at_one <= guard.rounded_step_max_abs == LOCKED_ROUNDED_STEP_MAX_ABS
        ),
        "middle_low_guards_are_preserved": all(guard.middle_low_invariant_checks.values()),
        "high_guard_is_explicitly_conditional": guard.high_word_guard_is_conditional,
        "deployed_output_range_is_finite": deployed_output_at_one < FP32_MAX_FINITE / 2,
        "external_sensitivities_are_positive": all(
            value > 0 for value in sensitivity.__dict__.values()
        ),
    }
    return ImplementationMarginEvaluation(
        reduction=reduction,
        rate=rate,
        constant_forcing=forcing,
        forcing_capacity=forcing_capacity,
        function_gap_ultimate=function_gap,
        guard=guard,
        checks=checks,
    )


def _grid_value(index: int) -> Fraction:
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError("grid index must be a nonnegative integer")
    return Fraction(index, LOCKED_BUDGET_GRID_DENOMINATOR)


def _largest_feasible_index(
    fixed: ImplementationMarginBudget,
    axis: str,
    *,
    upper_index: int | None = None,
) -> int:
    if axis not in _AXES:
        raise ValueError(f"unknown budget axis {axis!r}")

    def accepted(index: int) -> bool:
        return evaluate_implementation_margin(fixed.replace(axis, _grid_value(index))).certified

    if not accepted(0):
        raise ValueError("the fixed budget is not feasible at zero on the varied axis")
    low = 0
    if upper_index is None:
        high = 1
        while accepted(high):
            low = high
            high *= 2
    else:
        if upper_index < 0:
            raise ValueError("upper_index must be nonnegative")
        high = upper_index + 1
        if accepted(high):
            raise ValueError("supplied upper_index does not bracket the boundary")
    while low + 1 < high:
        middle = (low + high) // 2
        if accepted(middle):
            low = middle
        else:
            high = middle
    return low


def grid_boundary(
    axis: str,
    fixed_budget: ImplementationMarginBudget = _ZERO_BUDGET,
    *,
    upper_index: int | None = None,
) -> GridBoundary:
    """Return the exact largest accepted grid point along one coordinate."""

    index = _largest_feasible_index(fixed_budget, axis, upper_index=upper_index)
    accepted_value = _grid_value(index)
    rejected_value = _grid_value(index + 1)
    accepted_budget = fixed_budget.replace(axis, accepted_value)
    rejected_budget = fixed_budget.replace(axis, rejected_value)
    accepted = evaluate_implementation_margin(accepted_budget)
    rejected = evaluate_implementation_margin(rejected_budget)
    if not accepted.certified or rejected.certified:
        raise AssertionError("exact grid boundary search did not bracket acceptance")
    rejected_checks = tuple(name for name, passed in rejected.checks.items() if not passed)
    active = (
        "unit_storage_is_forward_invariant"
        if "unit_storage_is_forward_invariant" in rejected_checks
        else rejected_checks[0]
    )
    return GridBoundary(
        varied_axis=axis,
        fixed_budget=fixed_budget,
        accepted=accepted_budget,
        rejected=rejected_budget,
        accepted_value=accepted_value,
        rejected_value=rejected_value,
        grid_step=Fraction(1, LOCKED_BUDGET_GRID_DENOMINATOR),
        accepted_rate=accepted.rate,
        accepted_forcing=accepted.constant_forcing,
        accepted_invariance_slack=1 - accepted.rate - accepted.constant_forcing,
        rejected_invariance_slack=1 - rejected.rate - rejected.constant_forcing,
        active_constraint=active,
        rejected_checks=rejected_checks,
    )


def _frontier(kind: str, *, points: int = LOCKED_FRONTIER_POINTS) -> tuple[FrontierRow, ...]:
    if kind == "slope":
        gradient_axis = "gradient_slope"
        operator_axis = "operator_slope"
    elif kind == "intercept":
        gradient_axis = "gradient_intercept"
        operator_axis = "operator_intercept"
    else:
        raise ValueError("frontier kind must be 'slope' or 'intercept'")
    if points < 2:
        raise ValueError("a frontier needs at least two points")

    gradient_max = grid_boundary(gradient_axis)
    operator_max = grid_boundary(operator_axis)
    gradient_max_index = (
        gradient_max.accepted_value.numerator
        * LOCKED_BUDGET_GRID_DENOMINATOR
        // gradient_max.accepted_value.denominator
    )
    operator_max_index = (
        operator_max.accepted_value.numerator
        * LOCKED_BUDGET_GRID_DENOMINATOR
        // operator_max.accepted_value.denominator
    )
    rows: list[FrontierRow] = []
    for point in range(points):
        gradient_index = gradient_max_index * point // (points - 1)
        seed = ImplementationMarginBudget().replace(gradient_axis, _grid_value(gradient_index))
        operator_boundary = grid_boundary(
            operator_axis,
            seed,
            upper_index=operator_max_index,
        )
        gradient_boundary = grid_boundary(
            gradient_axis,
            operator_boundary.accepted,
            upper_index=gradient_max_index,
        )
        accepted_budget = gradient_boundary.accepted
        accepted = evaluate_implementation_margin(accepted_budget)
        rejected_operator_value = getattr(accepted_budget, operator_axis) + Fraction(
            1, LOCKED_BUDGET_GRID_DENOMINATOR
        )
        rejected_operator = evaluate_implementation_margin(
            accepted_budget.replace(operator_axis, rejected_operator_value)
        )
        rejected_operator_checks = tuple(
            name for name, passed in rejected_operator.checks.items() if not passed
        )
        rows.append(
            FrontierRow(
                gradient_axis=gradient_axis,
                operator_axis=operator_axis,
                gradient_value=getattr(accepted_budget, gradient_axis),
                operator_value=getattr(accepted_budget, operator_axis),
                rejected_gradient_value=gradient_boundary.rejected_value,
                rejected_operator_value=rejected_operator_value,
                rate=accepted.rate,
                forcing=accepted.constant_forcing,
                invariance_slack=1 - accepted.rate - accepted.constant_forcing,
                deployed_output_at_one=accepted.guard.deployed_output_at_one,
                rejected_gradient_invariance_slack=(gradient_boundary.rejected_invariance_slack),
                rejected_operator_invariance_slack=(
                    1 - rejected_operator.rate - rejected_operator.constant_forcing
                ),
                rejected_gradient_checks=gradient_boundary.rejected_checks,
                rejected_checks=rejected_operator_checks,
            )
        )
    return tuple(rows)


@lru_cache(maxsize=1)
def audit_implementation_margin() -> ImplementationMarginAudit:
    """Replay the exact P11 margins, frontiers, controls, and success gates."""

    p10, sensitivities = _locked_context()
    zero = evaluate_implementation_margin()
    axes = {axis: grid_boundary(axis) for axis in _AXES}
    slope = _frontier("slope")
    intercept = _frontier("intercept")

    # A simple all-positive dyadic profile stays well inside both Pareto
    # slices and preserves the stronger subunit objective-gap bound.  It is a
    # reported acceptance profile, not a fitted change to eta, rho, or P7.
    joint_budget = ImplementationMarginBudget(
        gradient_slope=Fraction(1, 4_096),
        operator_slope=Fraction(1, 128),
        gradient_intercept=Fraction(1, 4_096),
        operator_intercept=Fraction(1, 8),
    )
    jointly_nonzero = evaluate_implementation_margin(joint_budget)
    zero_envelopes_match = all(
        getattr(zero.reduction, p11_name) == getattr(p10.reduction, p10_name)
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
    )
    boundary_controls_are_adjacent = all(
        boundary.rejected_value - boundary.accepted_value
        == Fraction(1, LOCKED_BUDGET_GRID_DENOMINATOR)
        for boundary in axes.values()
    )
    frontier_controls_fail = all(
        row.rejected_checks
        and row.rejected_gradient_checks
        and "unit_storage_is_forward_invariant" in row.rejected_checks
        and "unit_storage_is_forward_invariant" in row.rejected_gradient_checks
        for row in (*slope, *intercept)
    )
    checks = {
        "p10_checkpoint_replays": p10.certified,
        "zero_budget_reproduces_all_p10_envelopes": zero_envelopes_match,
        "zero_budget_reproduces_q10_exactly": zero.rate == p10.rate,
        "zero_budget_reproduces_D10_exactly": zero.constant_forcing == p10.constant_forcing,
        "zero_budget_reproduces_function_gap_exactly": (
            zero.function_gap_ultimate == p10.function_gap_ultimate
        ),
        "jointly_nonzero_budget_is_certified": (
            jointly_nonzero.certified and all(value > 0 for value in joint_budget.__dict__.values())
        ),
        "joint_budget_preserves_unit_storage": (
            jointly_nonzero.constant_forcing <= jointly_nonzero.forcing_capacity
        ),
        "joint_budget_preserves_subunit_objective_gap": (
            jointly_nonzero.function_gap_ultimate is not None
            and jointly_nonzero.function_gap_ultimate < 1
        ),
        "joint_budget_closes_signal_output_step_and_master_guards": all(
            jointly_nonzero.checks[name]
            for name in (
                "p9_signal_guard_closes",
                "deployed_output_is_below_2^15",
                "rounded_step_is_below_two",
                "middle_low_guards_are_preserved",
                "high_guard_is_explicitly_conditional",
            )
        ),
        "four_axis_maxima_are_exact_on_locked_grid": (
            len(axes) == 4
            and all(
                evaluate_implementation_margin(boundary.accepted).certified
                and not evaluate_implementation_margin(boundary.rejected).certified
                for boundary in axes.values()
            )
        ),
        "axis_negative_controls_are_adjacent": boundary_controls_are_adjacent,
        "slope_frontier_has_locked_size": len(slope) == LOCKED_FRONTIER_POINTS,
        "intercept_frontier_has_locked_size": len(intercept) == LOCKED_FRONTIER_POINTS,
        "all_frontier_points_are_certified": all(
            evaluate_implementation_margin(
                ImplementationMarginBudget()
                .replace(row.gradient_axis, row.gradient_value)
                .replace(row.operator_axis, row.operator_value)
            ).certified
            for row in (*slope, *intercept)
        ),
        "all_frontier_coordinatewise_next_grid_controls_fail": frontier_controls_fail,
        "p10_provenance_roles_are_distinct": len(
            {
                P10_THEOREM_SOURCE_COMMIT,
                P10_EXACT_ARTIFACT_COMMIT,
                P10_DIAGNOSTIC_CHECKPOINT_COMMIT,
            }
        )
        == 3,
    }
    return ImplementationMarginAudit(
        p10=p10,
        sensitivities=sensitivities,
        zero_error=zero,
        axis_boundaries=axes,
        slope_frontier=slope,
        intercept_frontier=intercept,
        jointly_nonzero=jointly_nonzero,
        checks=checks,
    )


def locked_implementation_margin_audit() -> ImplementationMarginAudit:
    """Return P11 after requiring every exact success gate to pass."""

    audit = audit_implementation_margin()
    if not audit.certified:
        failures = [name for name, passed in audit.checks.items() if not passed]
        raise AssertionError(f"locked P11 implementation-margin audit failed: {failures}")
    return audit


__all__ = [
    "IMPLEMENTATION_MARGIN_SCHEMA_VERSION",
    "LOCKED_BUDGET_GRID_DENOMINATOR",
    "LOCKED_FRONTIER_POINTS",
    "P10_DIAGNOSTIC_CHECKPOINT_COMMIT",
    "P10_EXACT_ARTIFACT_COMMIT",
    "P10_THEOREM_SOURCE_COMMIT",
    "ExternalPortSensitivities",
    "FrontierRow",
    "GridBoundary",
    "ImplementationMarginAudit",
    "ImplementationMarginBudget",
    "ImplementationMarginEvaluation",
    "ImplementationMarginGuard",
    "ImplementationMarginReduction",
    "audit_implementation_margin",
    "evaluate_implementation_margin",
    "external_port_sensitivities",
    "grid_boundary",
    "locked_implementation_margin_audit",
]
