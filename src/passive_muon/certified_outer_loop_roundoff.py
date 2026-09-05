"""Exact P21 rounding envelopes around the stored-signal sector LMI.

The core P21 certificate places the P20 sector at the *stored* FP32 signal.
This module connects that exact LMI to the concrete P21 arithmetic graph.  It
never compares shield outputs at two different signals.

The pre-cast port ``zeta`` contains every error before the final FP32 storage
boundary, including stochastic gradient error and the use of the represented
high word instead of the logical three-word master.  If the gradient is
evaluated at the represented high word ``H``, global ``L``-smoothness gives

``||zeta|| <= ||gradient_noise|| + L*||middle+low||``.

The static three-word range guards alone do not make that bound useful, so a
theorem-facing affine budget for ``zeta`` remains an explicit premise.  The
zero-error and locked robust profiles below are both reported.

Likewise, nonzero weight decay is a conditional aggregate-port result.  Its
budget bounds the exact logical displacement caused by the complete stored
decay subgraph and, separately, the rounded decay step used by the range
proof.  P21 exposes both runtime quantities but does not derive either bound
from an arbitrary decay coefficient and high-word magnitude.

All authoritative calculations use :class:`fractions.Fraction`.  Irrational
storage-dual norms are enclosed upward on a frozen dyadic grid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

from passive_muon.certified_outer_loop_composition import (
    MAXIMUM_LEARNING_RATE_FP32_EXACT,
    PRIMARY_LEARNING_RATE_FP32_EXACT,
)
from passive_muon.certified_outer_loop_composition_certificate import (
    AffinePortAbsorption,
    AffinePortEnvelope,
    StoredSignalPortCertificate,
    absorb_affine_ports,
    audit_stored_signal_port_certificate,
    build_stored_signal_port_certificate,
)
from passive_muon.finite_precision_outer_loop import (
    BETA_FP32_EXACT,
    EMA_GRADIENT_ENVELOPE_COEFFICIENT,
    EMA_MOMENTUM_ENVELOPE_COEFFICIENT,
    FP32_HALF_MIN_SUBNORMAL,
    FP32_UNIT_ROUNDOFF,
    LOCKED_MASTER_HIGH_MAX_ABS,
    LOCKED_MASTER_LOW_MAX_ABS,
    LOCKED_MASTER_MIDDLE_MAX_ABS,
    MASTER_RESIDUAL_CRUMB_PER_ENTRY,
    ONE_MINUS_BETA_FP32_EXACT,
    FinitePrecisionOuterLoopConfig,
    outer_residual_envelope,
)
from passive_muon.scalable_sector_shield import (
    LOCKED_CERTIFIED_SHAPES,
    LOCKED_REPRESENTATIVE_SHAPES,
)
from passive_muon.sector_projected_resolvent_certificate import LOCKED_SECTOR_UPPER
from passive_muon.sector_shielded_inexact_resolvent_certificate import (
    FASTER_RATE_POINT,
    MAXIMUM_STEP_POINT,
    ShieldOperatingPoint,
)

CERTIFIED_OUTER_LOOP_ROUNDOFF_SCHEMA_VERSION = "passive-muon-certified-outer-loop-roundoff-v1"

SQRT_ENCLOSURE_DENOMINATOR = 2**80
REPORT_GRID_DENOMINATOR = 2**40
LOCKED_STORAGE_RADIUS = Fraction(1)
LOCKED_YOUNG_PARAMETERS = (Fraction(1), Fraction(1), Fraction(1))
LOCKED_EXTERNAL_GRADIENT_SLOPE = Fraction(1, 8_192)
LOCKED_EXTERNAL_GRADIENT_INTERCEPT = Fraction(1, 8_192)
LOCKED_MODEL_RECONSTRUCTION_SLOPE = Fraction(1, 81_920)
LOCKED_MODEL_RECONSTRUCTION_INTERCEPT = Fraction(1, 81_920)
LOCKED_ROBUST_GRADIENT_SLOPE = Fraction(1, 4_096)
LOCKED_ROBUST_GRADIENT_INTERCEPT = Fraction(1, 4_096)
LOCKED_WEIGHT_DECAY_DISPLACEMENT = Fraction(1, 131_072)
LOCKED_WEIGHT_DECAY_ROUNDED_STEP = Fraction(1, 131_072)
LOCKED_OUTPUT_MAX_ABS = Fraction(64)
LOCKED_TOTAL_STEP_MAX_ABS = Fraction(1)


def _ceil_sqrt(value: int) -> int:
    root = math.isqrt(value)
    return root if root * root == value else root + 1


def _sqrt_upper(value: Fraction) -> Fraction:
    """Return a sound dyadic upper enclosure of a nonnegative rational root."""

    if value < 0:
        raise ValueError("cannot enclose the square root of a negative value")
    if value == 0:
        return Fraction(0)
    scaled = value.numerator * SQRT_ENCLOSURE_DENOMINATOR**2
    quotient = scaled // value.denominator
    numerator = math.isqrt(quotient)
    lower = Fraction(numerator, SQRT_ENCLOSURE_DENOMINATOR)
    if lower * lower == value:
        return lower
    upper = Fraction(numerator + 1, SQRT_ENCLOSURE_DENOMINATOR)
    if upper * upper < value:  # pragma: no cover - defensive arithmetic check
        raise AssertionError("dyadic square-root enclosure is not outward")
    return upper


def _round_up(value: Fraction) -> Fraction:
    if value < 0:
        raise ValueError("only nonnegative certificate fields may be rounded upward")
    quotient, remainder = divmod(value.numerator * REPORT_GRID_DENOMINATOR, value.denominator)
    return Fraction(quotient + bool(remainder), REPORT_GRID_DENOMINATOR)


def _storage_dual_slope(
    momentum_coefficient: Fraction,
    gradient_coefficient: Fraction,
    storage: tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]],
) -> Fraction:
    """Bound ``A||z||+B||u||`` by this value times ``sqrt(V)``."""

    if momentum_coefficient < 0 or gradient_coefficient < 0:
        raise ValueError("state coefficients must be nonnegative")
    p00, p01 = storage[0]
    p10, p11 = storage[1]
    determinant = p00 * p11 - p01**2
    if p01 != p10 or p00 <= 0 or determinant <= 0:
        raise ValueError("storage must be symmetric positive definite")
    # The numerator to bound is A||z||+B||u||.  The worst relative
    # orientation minimizes the storage cross term, hence the absolute value
    # of P01 rather than an assumed sign.  (Both frozen P21 storages happen to
    # have P01<0.)
    dual_square = (
        p11 * momentum_coefficient**2
        + 2 * abs(p01) * momentum_coefficient * gradient_coefficient
        + p00 * gradient_coefficient**2
    ) / determinant
    return _sqrt_upper(dual_square)


@dataclass(frozen=True)
class GradientSourceBudget:
    """Pathwise pre-cast error bound ``||zeta|| <= a*sqrt(V)+b``."""

    slope: Fraction
    intercept: Fraction
    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.slope, Fraction) or not isinstance(self.intercept, Fraction):
            raise TypeError("gradient budget coefficients must be Fraction values")
        if self.slope < 0 or self.intercept < 0:
            raise ValueError("gradient budget coefficients must be nonnegative")
        if not self.name:
            raise ValueError("gradient budget name must be nonempty")


ZERO_GRADIENT_SOURCE_BUDGET = GradientSourceBudget(Fraction(0), Fraction(0), "zero")
ROBUST_GRADIENT_SOURCE_BUDGET = GradientSourceBudget(
    LOCKED_EXTERNAL_GRADIENT_SLOPE + 10 * LOCKED_MODEL_RECONSTRUCTION_SLOPE,
    LOCKED_EXTERNAL_GRADIENT_INTERCEPT + 10 * LOCKED_MODEL_RECONSTRUCTION_INTERCEPT,
    "joint_external_gradient_and_reconstruction_1_over_4096",
)


@dataclass(frozen=True)
class DecayPortBudget:
    """Conditional bounds for the complete concrete decay subgraph.

    ``logical_displacement`` bounds the Frobenius norm of
    ``pending_after_decay-pending_after_operator``.  It therefore includes
    both the stored decay step and the extra subtraction rounding.
    ``rounded_step`` separately bounds the stored decay-step norm needed by
    the executable range guard.  Neither number is inferred from a decay
    coefficient or the static master-word guards.
    """

    logical_displacement: Fraction
    rounded_step: Fraction
    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.logical_displacement, Fraction) or not isinstance(
            self.rounded_step, Fraction
        ):
            raise TypeError("decay budget coefficients must be Fraction values")
        if self.logical_displacement < 0 or self.rounded_step < 0:
            raise ValueError("decay budget coefficients must be nonnegative")
        if not self.name:
            raise ValueError("decay budget name must be nonempty")


ZERO_DECAY_PORT_BUDGET = DecayPortBudget(Fraction(0), Fraction(0), "disabled")
BOUNDED_DECAY_PORT_BUDGET = DecayPortBudget(
    LOCKED_WEIGHT_DECAY_DISPLACEMENT,
    LOCKED_WEIGHT_DECAY_ROUNDED_STEP,
    "conditional_1_over_131072",
)


@dataclass(frozen=True)
class PhysicalAffineEnvelope:
    """Physical-unit bound ``quantity <= slope*sqrt(V)+intercept``."""

    slope: Fraction
    intercept: Fraction

    def __post_init__(self) -> None:
        if self.slope < 0 or self.intercept < 0:
            raise ValueError("physical envelope coefficients must be nonnegative")


@dataclass(frozen=True)
class P21RoundoffReduction:
    """Exact shape/rate-specific reduction into the three P21 LMI ports."""

    shape: tuple[int, int]
    shield_shape: tuple[int, int]
    transposed_for_shield: bool
    point: ShieldOperatingPoint
    gradient_budget: GradientSourceBudget
    decay_budget: DecayPortBudget
    sqrt_entries_upper: int
    eta_fp32: Fraction
    eta_representation_error: Fraction
    momentum_roundoff: PhysicalAffineEnvelope
    signal_roundoff: PhysicalAffineEnvelope
    stored_signal: PhysicalAffineEnvelope
    shield_output: PhysicalAffineEnvelope
    master_roundoff: PhysicalAffineEnvelope
    dead_zone_parameter_error: Fraction
    normalized_ports: tuple[AffinePortEnvelope, AffinePortEnvelope, AffinePortEnvelope]


@dataclass(frozen=True)
class P21GuardClosure:
    """Exact finite-range checks on the invariant storage unit ball."""

    stored_signal_at_one: Fraction
    shield_output_at_one: Fraction
    rounded_operator_step_at_one: Fraction
    total_step_at_one: Fraction
    output_max_abs: Fraction
    total_step_max_abs: Fraction
    master_high_max_abs: Fraction
    master_middle_max_abs: Fraction
    master_low_max_abs: Fraction
    pending_after_operator_strict_upper: Fraction
    pending_after_decay_strict_upper: Fraction
    middle_candidate_strict_upper: Fraction
    high_word_guard_is_conditional: bool
    lower_word_guard_checks: dict[str, bool]
    checks: dict[str, bool]


@dataclass(frozen=True)
class P21RoundoffEvaluation:
    """Closed exact P21 finite-precision storage result for one profile."""

    core: StoredSignalPortCertificate
    reduction: P21RoundoffReduction
    absorption: AffinePortAbsorption
    reported_rate_upper: Fraction
    reported_forcing_upper: Fraction
    reported_objective_gap_upper: Fraction
    guard: P21GuardClosure
    checks: dict[str, bool]

    @property
    def certified(self) -> bool:
        return bool(self.checks) and all(self.checks.values())


def _eta_fp32(point: ShieldOperatingPoint) -> Fraction:
    if point == FASTER_RATE_POINT:
        return PRIMARY_LEARNING_RATE_FP32_EXACT
    if point == MAXIMUM_STEP_POINT:
        return MAXIMUM_LEARNING_RATE_FP32_EXACT
    raise ValueError("point must be a frozen P21 operating point")


def _canonical_p20_shape(shape: tuple[int, int]) -> tuple[tuple[int, int], bool]:
    if shape in LOCKED_CERTIFIED_SHAPES:
        return shape, False
    transposed = (shape[1], shape[0])
    if transposed in LOCKED_CERTIFIED_SHAPES:
        return transposed, True
    raise ValueError("shape or its transpose must be in the frozen P20 certified shape table")


def _p21_master_guard_checks(
    rounded_operator_step: Fraction,
    decay_budget: DecayPortBudget,
) -> tuple[Fraction, Fraction, Fraction, dict[str, bool]]:
    """Replay the two-subtraction P21 lower-word range proof exactly."""

    u32 = FP32_UNIT_ROUNDOFF
    tau32 = FP32_HALF_MIN_SUBNORMAL
    pending_operator = (1 + u32) * (LOCKED_MASTER_LOW_MAX_ABS + rounded_operator_step) + tau32
    if decay_budget.rounded_step == 0:
        pending_decay = pending_operator
    else:
        pending_decay = (1 + u32) * (pending_operator + decay_budget.rounded_step) + tau32
    middle_candidate = (1 + u32) * (LOCKED_MASTER_MIDDLE_MAX_ABS + pending_decay) + tau32
    checks = {
        "pending_after_operator_below_three": pending_operator < 3,
        "pending_after_decay_below_three": pending_decay < 3,
        "middle_candidate_below_2^8": middle_candidate < 2**8,
        # With a strict 2^8 bound, round-to-nearest TwoSum's low residual is
        # at most half an ulp, 2^-17.  The subsequent high-word TwoSum has a
        # residual at most 2^6 under the explicit 2^30 high-word guard.
        "new_low_within_2^-16_guard": Fraction(1, 2**17) <= LOCKED_MASTER_LOW_MAX_ABS,
        "new_middle_within_2^7_guard": Fraction(2**6) <= LOCKED_MASTER_MIDDLE_MAX_ABS,
        "high_two_sum_cannot_overflow": (LOCKED_MASTER_HIGH_MAX_ABS + middle_candidate < 2**31),
    }
    return pending_operator, pending_decay, middle_candidate, checks


def build_p21_roundoff_reduction(
    shape: tuple[int, int],
    point: ShieldOperatingPoint,
    *,
    gradient_budget: GradientSourceBudget = ROBUST_GRADIENT_SOURCE_BUDGET,
    decay_budget: DecayPortBudget = ZERO_DECAY_PORT_BUDGET,
) -> P21RoundoffReduction:
    """Build exact rounding and external-port envelopes for one shape/profile."""

    shield_shape, transposed_for_shield = _canonical_p20_shape(shape)
    if point not in (FASTER_RATE_POINT, MAXIMUM_STEP_POINT):
        raise ValueError("point must be a frozen P21 operating point")
    if not isinstance(gradient_budget, GradientSourceBudget):
        raise TypeError("gradient_budget must be a GradientSourceBudget")
    if not isinstance(decay_budget, DecayPortBudget):
        raise TypeError("decay_budget must be a DecayPortBudget")

    core = build_stored_signal_port_certificate(point)
    pl = core.pl_certificate
    smoothness = pl.smoothness
    beta = pl.beta
    one_minus_beta = 1 - beta
    gamma = pl.center_gain
    eta = point.learning_rate
    eta32 = _eta_fp32(point)
    u32 = FP32_UNIT_ROUNDOFF
    raw = outer_residual_envelope(FinitePrecisionOuterLoopConfig(shape))
    root_entries = raw.sqrt_entry_count_upper
    cast_crumb = root_entries * FP32_HALF_MIN_SUBNORMAL
    c_beta = EMA_MOMENTUM_ENVELOPE_COEFFICIENT
    c_gradient = EMA_GRADIENT_ENVELOPE_COEFFICIENT
    b_ema = raw.ema_absolute_crumb

    represented_beta = (1 + u32) ** 2 * BETA_FP32_EXACT
    represented_gradient = (1 + u32) ** 2 * ONE_MINUS_BETA_FP32_EXACT

    # The stored sample is fl32(g+zeta).  The same rounded bg is reused by
    # both additions, which is why these sensitivities are not duplicated.
    momentum_state = _storage_dual_slope(
        c_beta,
        c_gradient * (1 + u32) + one_minus_beta * u32,
        pl.storage,
    )
    momentum_zeta = (c_gradient + one_minus_beta) * (1 + u32)
    momentum_roundoff = PhysicalAffineEnvelope(
        slope=smoothness * momentum_state + momentum_zeta * gradient_budget.slope,
        intercept=(
            b_ema
            + (c_gradient + one_minus_beta) * cast_crumb
            + momentum_zeta * gradient_budget.intercept
        ),
    )

    signal_state = _storage_dual_slope(
        c_beta * represented_beta,
        c_beta * represented_gradient * (1 + u32) + c_gradient * (1 + u32) + one_minus_beta * u32,
        pl.storage,
    )
    signal_zeta = (c_beta * represented_gradient + c_gradient + one_minus_beta) * (1 + u32)
    signal_roundoff = PhysicalAffineEnvelope(
        slope=smoothness * signal_state + signal_zeta * gradient_budget.slope,
        intercept=(
            c_beta * (b_ema + represented_gradient * cast_crumb)
            + b_ema
            + (c_gradient + one_minus_beta) * cast_crumb
            + signal_zeta * gradient_budget.intercept
        ),
    )

    signal_momentum_coefficient = represented_beta**2
    signal_gradient_coefficient = represented_gradient * (1 + represented_beta)
    stored_signal_state = _storage_dual_slope(
        signal_momentum_coefficient,
        signal_gradient_coefficient * (1 + u32),
        pl.storage,
    )
    stored_signal_zeta = signal_gradient_coefficient * (1 + u32)
    stored_signal = PhysicalAffineEnvelope(
        slope=smoothness * stored_signal_state + stored_signal_zeta * gradient_budget.slope,
        intercept=(
            (1 + represented_beta) * b_ema
            + signal_gradient_coefficient * cast_crumb
            + stored_signal_zeta * gradient_budget.intercept
        ),
    )
    shield_output = PhysicalAffineEnvelope(
        slope=LOCKED_SECTOR_UPPER * stored_signal.slope,
        intercept=LOCKED_SECTOR_UPPER * stored_signal.intercept,
    )

    two_operation_factor = 2 * u32 + u32**2
    master_output_coefficient = abs(eta32 - eta) + two_operation_factor * eta32
    master_roundoff = PhysicalAffineEnvelope(
        slope=master_output_coefficient * shield_output.slope,
        intercept=(
            master_output_coefficient * shield_output.intercept
            + u32 * root_entries * LOCKED_MASTER_LOW_MAX_ABS
            + root_entries * MASTER_RESIDUAL_CRUMB_PER_ENTRY
        ),
    )
    # P20 can fail only in an all-subnormal nonzero region.  The wrapper emits
    # zero and compares it with the exact sector point stored_signal/2.
    dead_zone_parameter_error = eta * root_entries * Fraction(1, 2**127)
    alpha = eta * gamma * smoothness
    output_equivalent = AffinePortEnvelope(
        slope=master_roundoff.slope / alpha,
        intercept=(
            master_roundoff.intercept
            + dead_zone_parameter_error
            + decay_budget.logical_displacement
        )
        / alpha,
    )
    return P21RoundoffReduction(
        shape=shape,
        shield_shape=shield_shape,
        transposed_for_shield=transposed_for_shield,
        point=point,
        gradient_budget=gradient_budget,
        decay_budget=decay_budget,
        sqrt_entries_upper=root_entries,
        eta_fp32=eta32,
        eta_representation_error=eta32 - eta,
        momentum_roundoff=momentum_roundoff,
        signal_roundoff=signal_roundoff,
        stored_signal=stored_signal,
        shield_output=shield_output,
        master_roundoff=master_roundoff,
        dead_zone_parameter_error=dead_zone_parameter_error,
        normalized_ports=(
            AffinePortEnvelope(
                momentum_roundoff.slope / smoothness,
                momentum_roundoff.intercept / smoothness,
            ),
            AffinePortEnvelope(
                signal_roundoff.slope / smoothness,
                signal_roundoff.intercept / smoothness,
            ),
            output_equivalent,
        ),
    )


def evaluate_p21_roundoff(
    shape: tuple[int, int],
    point: ShieldOperatingPoint,
    *,
    gradient_budget: GradientSourceBudget = ROBUST_GRADIENT_SOURCE_BUDGET,
    decay_budget: DecayPortBudget = ZERO_DECAY_PORT_BUDGET,
) -> P21RoundoffEvaluation:
    """Close the stored P21 arithmetic ports and exact finite-range guards."""

    core = build_stored_signal_port_certificate(point)
    reduction = build_p21_roundoff_reduction(
        shape,
        point,
        gradient_budget=gradient_budget,
        decay_budget=decay_budget,
    )
    absorption = absorb_affine_ports(
        core,
        reduction.normalized_ports,
        LOCKED_YOUNG_PARAMETERS,
        storage_radius=LOCKED_STORAGE_RADIUS,
    )
    rate = _round_up(absorption.absorbed_rate)
    forcing = _round_up(absorption.constant_forcing)
    objective = _round_up(
        core.pl_certificate.smoothness / core.pl_certificate.function_storage * forcing / (1 - rate)
    )

    signal_at_one = reduction.stored_signal.slope + reduction.stored_signal.intercept
    output_at_one = reduction.shield_output.slope + reduction.shield_output.intercept
    rounded_operator_step = (
        1 + FP32_UNIT_ROUNDOFF
    ) * reduction.eta_fp32 * output_at_one + reduction.sqrt_entries_upper * FP32_HALF_MIN_SUBNORMAL
    total_step = rounded_operator_step + decay_budget.rounded_step
    pending_operator, pending_decay, middle_candidate, lower_checks = _p21_master_guard_checks(
        rounded_operator_step, decay_budget
    )
    guard_checks = {
        "stored_signal_is_finite_under_unit_storage": signal_at_one < Fraction(2**120),
        "shield_output_fits_runtime_guard": output_at_one <= LOCKED_OUTPUT_MAX_ABS,
        "combined_step_fits_runtime_guard": total_step <= LOCKED_TOTAL_STEP_MAX_ABS,
        "p21_two_subtraction_lower_word_invariants_close": all(lower_checks.values()),
        "decay_budget_is_an_explicit_conditional_premise": bool(decay_budget.name),
        "high_word_guard_remains_explicitly_conditional": True,
    }
    guard = P21GuardClosure(
        stored_signal_at_one=signal_at_one,
        shield_output_at_one=output_at_one,
        rounded_operator_step_at_one=rounded_operator_step,
        total_step_at_one=total_step,
        output_max_abs=LOCKED_OUTPUT_MAX_ABS,
        total_step_max_abs=LOCKED_TOTAL_STEP_MAX_ABS,
        master_high_max_abs=LOCKED_MASTER_HIGH_MAX_ABS,
        master_middle_max_abs=LOCKED_MASTER_MIDDLE_MAX_ABS,
        master_low_max_abs=LOCKED_MASTER_LOW_MAX_ABS,
        pending_after_operator_strict_upper=pending_operator,
        pending_after_decay_strict_upper=pending_decay,
        middle_candidate_strict_upper=middle_candidate,
        high_word_guard_is_conditional=True,
        lower_word_guard_checks={str(key): bool(value) for key, value in lower_checks.items()},
        checks=guard_checks,
    )
    checks = {
        "core_stored_signal_lmi_is_strict": all(
            value > 0
            for value in audit_stored_signal_port_certificate(point).negative_lmi_leading_minors
        ),
        "reported_rate_rounding_is_upward": rate >= absorption.absorbed_rate,
        "reported_forcing_rounding_is_upward": forcing >= absorption.constant_forcing,
        "absorbed_rate_is_strictly_contractive": rate < 1,
        "unit_storage_ball_is_forward_invariant": forcing <= 1 - rate,
        "objective_neighborhood_is_finite": objective >= 0,
        "all_runtime_guards_close": all(guard_checks.values()),
        "shape_or_transpose_is_P20_certified": (reduction.shield_shape in LOCKED_CERTIFIED_SHAPES),
        "gradient_reconstruction_budget_is_explicit": bool(gradient_budget.name),
    }
    return P21RoundoffEvaluation(
        core=core,
        reduction=reduction,
        absorption=absorption,
        reported_rate_upper=rate,
        reported_forcing_upper=forcing,
        reported_objective_gap_upper=objective,
        guard=guard,
        checks=checks,
    )


def representative_roundoff_evaluations() -> tuple[P21RoundoffEvaluation, ...]:
    """Return both robust operating points on all seven P20 Transformer shapes."""

    return tuple(
        evaluate_p21_roundoff(shape, point)
        for point in (FASTER_RATE_POINT, MAXIMUM_STEP_POINT)
        for shape in LOCKED_REPRESENTATIVE_SHAPES
    )


def exact_roundoff_checks() -> dict[str, bool]:
    """Replay both rates, every representative shape, and decay controls."""

    evaluations = representative_roundoff_evaluations()
    primary_largest = evaluate_p21_roundoff((4_096, 14_336), FASTER_RATE_POINT)
    secondary_largest = evaluate_p21_roundoff((4_096, 14_336), MAXIMUM_STEP_POINT)
    primary_decay = evaluate_p21_roundoff(
        (4_096, 14_336),
        FASTER_RATE_POINT,
        decay_budget=BOUNDED_DECAY_PORT_BUDGET,
    )
    secondary_decay = evaluate_p21_roundoff(
        (4_096, 14_336),
        MAXIMUM_STEP_POINT,
        decay_budget=BOUNDED_DECAY_PORT_BUDGET,
    )
    zero_primary = evaluate_p21_roundoff(
        (4_096, 14_336),
        FASTER_RATE_POINT,
        gradient_budget=ZERO_GRADIENT_SOURCE_BUDGET,
    )
    return {
        "all_seven_shapes_pass_both_operating_points": all(item.certified for item in evaluations),
        "primary_eta_is_1_over_120": primary_largest.core.point.learning_rate == Fraction(1, 120),
        "secondary_eta_is_1_over_83": secondary_largest.core.point.learning_rate == Fraction(1, 83),
        "zero_gradient_profile_is_contractive": zero_primary.certified,
        "joint_gradient_profile_is_contractive": (
            primary_largest.certified and secondary_largest.certified
        ),
        "bounded_decay_profile_is_forward_invariant": (
            primary_decay.certified and secondary_decay.certified
        ),
        "bounded_decay_objective_neighborhoods_are_below_one": (
            primary_decay.reported_objective_gap_upper < 1
            and secondary_decay.reported_objective_gap_upper < 1
        ),
        "all_subnormal_zero_wrapper_has_positive_finite_bound": (
            primary_largest.reduction.dead_zone_parameter_error > 0
            and secondary_largest.reduction.dead_zone_parameter_error > 0
        ),
        "largest_shape_is_the_worst_dead_zone_root_bound": all(
            item.reduction.sqrt_entries_upper <= primary_largest.reduction.sqrt_entries_upper
            for item in evaluations
        ),
    }


__all__ = [
    "BOUNDED_DECAY_PORT_BUDGET",
    "CERTIFIED_OUTER_LOOP_ROUNDOFF_SCHEMA_VERSION",
    "LOCKED_EXTERNAL_GRADIENT_INTERCEPT",
    "LOCKED_EXTERNAL_GRADIENT_SLOPE",
    "LOCKED_MODEL_RECONSTRUCTION_INTERCEPT",
    "LOCKED_MODEL_RECONSTRUCTION_SLOPE",
    "LOCKED_OUTPUT_MAX_ABS",
    "LOCKED_ROBUST_GRADIENT_INTERCEPT",
    "LOCKED_ROBUST_GRADIENT_SLOPE",
    "LOCKED_STORAGE_RADIUS",
    "LOCKED_TOTAL_STEP_MAX_ABS",
    "LOCKED_WEIGHT_DECAY_DISPLACEMENT",
    "LOCKED_WEIGHT_DECAY_ROUNDED_STEP",
    "LOCKED_YOUNG_PARAMETERS",
    "REPORT_GRID_DENOMINATOR",
    "ROBUST_GRADIENT_SOURCE_BUDGET",
    "SQRT_ENCLOSURE_DENOMINATOR",
    "ZERO_DECAY_PORT_BUDGET",
    "ZERO_GRADIENT_SOURCE_BUDGET",
    "DecayPortBudget",
    "GradientSourceBudget",
    "P21GuardClosure",
    "P21RoundoffEvaluation",
    "P21RoundoffReduction",
    "PhysicalAffineEnvelope",
    "build_p21_roundoff_reduction",
    "evaluate_p21_roundoff",
    "exact_roundoff_checks",
    "representative_roundoff_evaluations",
]
