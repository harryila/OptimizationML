"""Exact radial passivation tradeoff above the P12 additive-epsilon bound.

P12 certifies a pointwise full-matrix deficit envelope for

``E_h,eps(M) = H_h(M / (||M||_F + eps))``.

This module integrates a positive, nonincreasing majorant of that envelope.
The resulting radial map removes the catastrophic ``O(||M||/eps)`` output of
the constant repair while retaining its unavoidable ``O(1/eps)`` worst-case
differential stiffness.  All theorem constants and algebraic comparisons are
exact rationals.  Arb is used only to evaluate the closed-form logarithm.

The construction is an exact-real surrogate.  It is not a theorem for the
discontinuous, backend-specific BF16 implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from flint import arb, ctx, fmpq

from passive_muon.additive_epsilon_deficit import (
    DEPLOYED_EPSILON,
    PAIR_DEFICIT_STRICT_LOWER,
    RADIUS_BANDS,
    certified_unit_epsilon_deficit_upper,
    radius_band_bound,
    zero_input_derivative_gain,
)

RADIAL_PASSIVATION_SCHEMA_VERSION = "passive-muon-radial-passivation-tradeoff-certificate-v1"

# The P12 global envelope is attained where its final normalized-radius band
# begins.  The raw ratio is z=t/(1-t).
SWITCH_NORMALIZED_RADIUS = Fraction(63, 10_000)
SWITCH_RAW_RATIO = Fraction(63, 9_937)

# These are the exact lower spectral-derivative and projection-envelope
# constants in P12's final radius band.
TAIL_DERIVATIVE_LOWER = Fraction(-199_437, 1_250)
TAIL_PROJECTION_LOWER = Fraction(-41_528_474_059_081, 260_261_360_000)
UNIT_DEFICIT_UPPER = Fraction(6_602_082_433_275_499_863, 41_641_817_600_000_000)

# Exact P11 unit-storage signal bound, used for the deployment-scale
# comparison requested by P13.
P11_SIGNAL_NORM_BOUND = Fraction(13_872_266_672_489, 549_755_813_888)

LOCKED_BETA = Fraction(19, 20)
LOCKED_LEARNING_RATE = Fraction(1, 32_000)


def _require_fraction(value: Fraction, name: str) -> None:
    if not isinstance(value, Fraction):
        raise TypeError(f"{name} must be a fractions.Fraction")


def _require_nonnegative_fraction(value: Fraction, name: str) -> None:
    _require_fraction(value, name)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


def _require_positive_fraction(value: Fraction, name: str) -> None:
    _require_fraction(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _arb_fraction(value: Fraction) -> arb:
    return arb(fmpq(value.numerator, value.denominator))


@dataclass(frozen=True)
class LogarithmicValue:
    """Exact representation of ``rational_part + coefficient*log(argument)``."""

    rational_part: Fraction
    logarithm_coefficient: Fraction
    logarithm_argument: Fraction

    def __post_init__(self) -> None:
        for name in ("rational_part", "logarithm_coefficient", "logarithm_argument"):
            _require_fraction(getattr(self, name), name)
        if self.logarithm_argument <= 0:
            raise ValueError("logarithm_argument must be positive")


@dataclass(frozen=True)
class RadialDerivativeEnvelope:
    """Exact dimension-free bounds on the two derivative mode families.

    ``radial`` is the exact radial eigenvalue of ``DG_eps``.  The tangential
    eigenvalue is generally logarithmic, so its exact rational lower and
    upper bounds are recorded instead.
    """

    raw_ratio: Fraction
    normalized_radius: Fraction
    radial: Fraction
    tangential_lower: Fraction
    tangential_upper: Fraction

    @property
    def minimum_eigenvalue(self) -> Fraction:
        return self.radial


@dataclass(frozen=True)
class ExplicitStepControl:
    """Exact local scalar EMA/Nesterov stability diagnostic at the origin."""

    epsilon: Fraction
    curvature: Fraction
    beta: Fraction
    learning_rate: Fraction
    corrected_origin_slope: Fraction
    dimensionless_gain: Fraction
    schur_gain_threshold: Fraction
    second_jury_margin: Fraction

    @property
    def locally_schur_stable(self) -> bool:
        return self.dimensionless_gain > 0 and self.second_jury_margin > 0


@dataclass(frozen=True)
class OneStepExpansionControl:
    """Exact finite one-step expansion caused already by the radial repair."""

    epsilon: Fraction
    beta: Fraction
    learning_rate: Fraction
    initial_weight: Fraction
    signal: Fraction
    normalized_signal: Fraction
    repair_to_weight_ratio: Fraction
    objective_growth_strict_lower: Fraction


def normalized_radius_from_raw_ratio(raw_ratio: Fraction) -> Fraction:
    """Map ``z=||M||/eps`` to ``t=z/(1+z)`` exactly."""

    _require_nonnegative_fraction(raw_ratio, "raw_ratio")
    return raw_ratio / (1 + raw_ratio)


def raw_ratio_from_normalized_radius(normalized_radius: Fraction) -> Fraction:
    """Map ``0<=t<1`` to ``z=t/(1-t)`` exactly."""

    _require_fraction(normalized_radius, "normalized_radius")
    if not 0 <= normalized_radius < 1:
        raise ValueError("normalized_radius must lie in [0,1)")
    return normalized_radius / (1 - normalized_radius)


def tail_deficit_majorant(raw_ratio: Fraction) -> Fraction:
    """Return P12's final-band envelope in the raw ratio ``z``.

    Substituting ``t=z/(1+z)`` into P12's final-band quadratic gives

    ``-(a + Gamma*z)/(1+z)**2``.
    """

    _require_nonnegative_fraction(raw_ratio, "raw_ratio")
    return -(TAIL_DERIVATIVE_LOWER + TAIL_PROJECTION_LOWER * raw_ratio) / (1 + raw_ratio) ** 2


def tail_deficit_majorant_derivative(raw_ratio: Fraction) -> Fraction:
    """Return the exact derivative of the final-band envelope with respect to ``z``."""

    _require_nonnegative_fraction(raw_ratio, "raw_ratio")
    numerator = (
        2 * TAIL_DERIVATIVE_LOWER - TAIL_PROJECTION_LOWER + TAIL_PROJECTION_LOWER * raw_ratio
    )
    return numerator / (1 + raw_ratio) ** 3


def deficit_majorant(raw_ratio: Fraction) -> Fraction:
    """Return the continuous nonincreasing P13 deficit majorant ``d_hat(z)``."""

    _require_nonnegative_fraction(raw_ratio, "raw_ratio")
    if raw_ratio <= SWITCH_RAW_RATIO:
        return UNIT_DEFICIT_UPPER
    return tail_deficit_majorant(raw_ratio)


def radial_integral_terms(raw_ratio: Fraction) -> LogarithmicValue:
    """Return exact closed-form terms for ``p(z)=integral_0^z d_hat(u) du``."""

    _require_nonnegative_fraction(raw_ratio, "raw_ratio")
    if raw_ratio <= SWITCH_RAW_RATIO:
        return LogarithmicValue(
            rational_part=UNIT_DEFICIT_UPPER * raw_ratio,
            logarithm_coefficient=Fraction(0),
            logarithm_argument=Fraction(1),
        )

    reciprocal_coefficient = TAIL_DERIVATIVE_LOWER - TAIL_PROJECTION_LOWER
    rational_part = UNIT_DEFICIT_UPPER * SWITCH_RAW_RATIO + reciprocal_coefficient * (
        1 / (1 + raw_ratio) - 1 / (1 + SWITCH_RAW_RATIO)
    )
    return LogarithmicValue(
        rational_part=rational_part,
        logarithm_coefficient=-TAIL_PROJECTION_LOWER,
        logarithm_argument=(1 + raw_ratio) / (1 + SWITCH_RAW_RATIO),
    )


def evaluate_logarithmic_value(
    value: LogarithmicValue,
    *,
    precision_bits: int = 160,
) -> arb:
    """Evaluate an exact logarithmic form with outward-rounded Arb arithmetic."""

    if not isinstance(value, LogarithmicValue):
        raise TypeError("value must be a LogarithmicValue")
    if precision_bits < 80:
        raise ValueError("precision_bits must be at least 80")
    with ctx.workprec(precision_bits):
        return (
            _arb_fraction(value.rational_part)
            + _arb_fraction(value.logarithm_coefficient)
            * _arb_fraction(value.logarithm_argument).log()
        )


def radial_integral_ball(raw_ratio: Fraction, *, precision_bits: int = 160) -> arb:
    """Outward-rounded evaluation of ``p(z)``."""

    return evaluate_logarithmic_value(
        radial_integral_terms(raw_ratio),
        precision_bits=precision_bits,
    )


def repair_magnitude_ball(
    raw_norm: Fraction,
    epsilon: Fraction,
    *,
    precision_bits: int = 160,
) -> arb:
    """Return an Arb enclosure of ``||G_eps(M)||_F=p(||M||_F/eps)``."""

    _require_nonnegative_fraction(raw_norm, "raw_norm")
    _require_positive_fraction(epsilon, "epsilon")
    return radial_integral_ball(raw_norm / epsilon, precision_bits=precision_bits)


def constant_repair_magnitude(raw_norm: Fraction, epsilon: Fraction) -> Fraction:
    """Return ``||(U/eps)M||_F``, the P12 constant-repair comparison."""

    _require_nonnegative_fraction(raw_norm, "raw_norm")
    _require_positive_fraction(epsilon, "epsilon")
    return UNIT_DEFICIT_UPPER * raw_norm / epsilon


def pointwise_deficit_upper(raw_norm: Fraction, epsilon: Fraction) -> Fraction:
    """Return the certified local P12 deficit envelope at ``||M||_F``."""

    _require_nonnegative_fraction(raw_norm, "raw_norm")
    _require_positive_fraction(epsilon, "epsilon")
    return deficit_majorant(raw_norm / epsilon) / epsilon


def repair_derivative_envelope(
    raw_norm: Fraction,
    epsilon: Fraction,
) -> RadialDerivativeEnvelope:
    """Return exact lower/upper bounds on the radial-repair derivative modes."""

    _require_nonnegative_fraction(raw_norm, "raw_norm")
    _require_positive_fraction(epsilon, "epsilon")
    raw_ratio = raw_norm / epsilon
    radial = deficit_majorant(raw_ratio) / epsilon
    return RadialDerivativeEnvelope(
        raw_ratio=raw_ratio,
        normalized_radius=normalized_radius_from_raw_ratio(raw_ratio),
        radial=radial,
        tangential_lower=radial,
        tangential_upper=UNIT_DEFICIT_UPPER / epsilon,
    )


def repair_derivative_spectrum_ball(
    raw_norm: Fraction,
    epsilon: Fraction,
    *,
    precision_bits: int = 160,
) -> tuple[arb, arb]:
    """Return Arb balls for the exact radial and tangential derivative eigenvalues."""

    _require_nonnegative_fraction(raw_norm, "raw_norm")
    _require_positive_fraction(epsilon, "epsilon")
    raw_ratio = raw_norm / epsilon
    radial = deficit_majorant(raw_ratio) / epsilon
    if raw_ratio == 0:
        tangential_terms = LogarithmicValue(
            UNIT_DEFICIT_UPPER / epsilon,
            Fraction(0),
            Fraction(1),
        )
    else:
        integral = radial_integral_terms(raw_ratio)
        tangential_terms = LogarithmicValue(
            integral.rational_part / (epsilon * raw_ratio),
            integral.logarithm_coefficient / (epsilon * raw_ratio),
            integral.logarithm_argument,
        )
    return (
        evaluate_logarithmic_value(
            LogarithmicValue(radial, Fraction(0), Fraction(1)),
            precision_bits=precision_bits,
        ),
        evaluate_logarithmic_value(tangential_terms, precision_bits=precision_bits),
    )


def repair_lipschitz_constant(epsilon: Fraction) -> Fraction:
    """Return the exact global Lipschitz constant ``U/eps`` of ``G_eps``."""

    _require_positive_fraction(epsilon, "epsilon")
    return UNIT_DEFICIT_UPPER / epsilon


def universal_lipschitz_strict_lower(epsilon: Fraction) -> Fraction:
    """Return the strict lower endpoint forced by P12's locked rank-two pair.

    Any globally Lipschitz correction that passivates that pair has Lipschitz
    constant strictly greater than this value.  The witness embeds in shapes
    with ``min(m,n)>=2``.
    """

    _require_positive_fraction(epsilon, "epsilon")
    return PAIR_DEFICIT_STRICT_LOWER / epsilon


def corrected_origin_slope(epsilon: Fraction) -> Fraction:
    """Return ``D(E_h,eps+G_eps)(0)`` as an exact scalar multiple of identity."""

    _require_positive_fraction(epsilon, "epsilon")
    return zero_input_derivative_gain(epsilon) + repair_lipschitz_constant(epsilon)


def ema_nesterov_explicit_step_control(
    *,
    epsilon: Fraction = DEPLOYED_EPSILON,
    curvature: Fraction = Fraction(1),
    beta: Fraction = LOCKED_BETA,
    learning_rate: Fraction = LOCKED_LEARNING_RATE,
) -> ExplicitStepControl:
    """Return the exact local Jury test for a scalar quadratic.

    With dimensionless gain ``q=eta*curvature*kappa``, the pinned EMA then
    Nesterov ordering is locally Schur stable exactly when

    ``q < 2(1+beta)/((1-beta)(1+2*beta))``

    for ``0<=beta<1`` and positive ``q``.  This helper records the second
    Jury margin whose sign supplies P13's explicit-step negative control.
    """

    _require_positive_fraction(epsilon, "epsilon")
    _require_positive_fraction(curvature, "curvature")
    _require_fraction(beta, "beta")
    if not 0 <= beta < 1:
        raise ValueError("beta must lie in [0,1)")
    _require_positive_fraction(learning_rate, "learning_rate")

    origin_slope = corrected_origin_slope(epsilon)
    dimensionless_gain = learning_rate * curvature * origin_slope
    threshold = 2 * (1 + beta) / ((1 - beta) * (1 + 2 * beta))
    second_jury_margin = 2 * (1 + beta) - dimensionless_gain * (1 - beta) * (1 + 2 * beta)
    return ExplicitStepControl(
        epsilon=epsilon,
        curvature=curvature,
        beta=beta,
        learning_rate=learning_rate,
        corrected_origin_slope=origin_slope,
        dimensionless_gain=dimensionless_gain,
        schur_gain_threshold=threshold,
        second_jury_margin=second_jury_margin,
    )


def ema_nesterov_repair_only_step_control(
    *,
    epsilon: Fraction = DEPLOYED_EPSILON,
    curvature: Fraction = Fraction(1),
    beta: Fraction = LOCKED_BETA,
    learning_rate: Fraction = LOCKED_LEARNING_RATE,
) -> ExplicitStepControl:
    """Return the pinned scalar Jury test with ``G_eps`` as the operator.

    This control isolates the radial correction's stiffness from the positive
    origin derivative of the five-stage Jordan map.
    """

    _require_positive_fraction(epsilon, "epsilon")
    _require_positive_fraction(curvature, "curvature")
    _require_fraction(beta, "beta")
    if not 0 <= beta < 1:
        raise ValueError("beta must lie in [0,1)")
    _require_positive_fraction(learning_rate, "learning_rate")

    origin_slope = repair_lipschitz_constant(epsilon)
    dimensionless_gain = learning_rate * curvature * origin_slope
    threshold = 2 * (1 + beta) / ((1 - beta) * (1 + 2 * beta))
    second_jury_margin = 2 * (1 + beta) - dimensionless_gain * (1 - beta) * (1 + 2 * beta)
    return ExplicitStepControl(
        epsilon=epsilon,
        curvature=curvature,
        beta=beta,
        learning_rate=learning_rate,
        corrected_origin_slope=origin_slope,
        dimensionless_gain=dimensionless_gain,
        schur_gain_threshold=threshold,
        second_jury_margin=second_jury_margin,
    )


def pinned_one_step_expansion_control(
    *,
    epsilon: Fraction = DEPLOYED_EPSILON,
    beta: Fraction = LOCKED_BETA,
    learning_rate: Fraction = LOCKED_LEARNING_RATE,
) -> OneStepExpansionControl:
    """Return P13's finite scalar-quadratic expansion witness.

    The locked initial weight makes the Nesterov signal ``epsilon/399`` and
    its additive-normalized argument ``1/400``.  The Jordan response is
    strictly positive, so the radial term alone gives the recorded strict
    objective-growth lower bound for the complete repaired operator.
    """

    _require_positive_fraction(epsilon, "epsilon")
    _require_fraction(beta, "beta")
    if beta != LOCKED_BETA or epsilon != DEPLOYED_EPSILON:
        raise ValueError("the finite one-step witness is locked to beta=19/20 and eps=1e-7")
    _require_positive_fraction(learning_rate, "learning_rate")
    initial_weight = Fraction(1, 389_025_000)
    signal = (1 - beta**2) * initial_weight
    normalized_signal = signal / (signal + epsilon)
    repair_to_weight_ratio = learning_rate * UNIT_DEFICIT_UPPER * (1 - beta**2) / epsilon
    return OneStepExpansionControl(
        epsilon=epsilon,
        beta=beta,
        learning_rate=learning_rate,
        initial_weight=initial_weight,
        signal=signal,
        normalized_signal=normalized_signal,
        repair_to_weight_ratio=repair_to_weight_ratio,
        objective_growth_strict_lower=(repair_to_weight_ratio - 1) ** 2,
    )


def exact_certificate_checks() -> dict[str, bool]:
    """Replay the finite exact algebra behind the radial theorem."""

    final_band = radius_band_bound(RADIUS_BANDS[-1])
    earlier_bounds = tuple(radius_band_bound(band).deficit_upper for band in RADIUS_BANDS[:-1])
    derivative_constant = 2 * TAIL_DERIVATIVE_LOWER - TAIL_PROJECTION_LOWER
    repair_control = ema_nesterov_repair_only_step_control()
    full_control = ema_nesterov_explicit_step_control()
    one_step = pinned_one_step_expansion_control()
    return {
        "switch_transform": (
            normalized_radius_from_raw_ratio(SWITCH_RAW_RATIO) == SWITCH_NORMALIZED_RADIUS
            and raw_ratio_from_normalized_radius(SWITCH_NORMALIZED_RADIUS) == SWITCH_RAW_RATIO
        ),
        "p12_constants_locked": (
            RADIUS_BANDS[-1].derivative_lower == TAIL_DERIVATIVE_LOWER
            and final_band.projection_lower == TAIL_PROJECTION_LOWER
            and certified_unit_epsilon_deficit_upper() == UNIT_DEFICIT_UPPER
        ),
        "switch_is_exactly_continuous": (
            tail_deficit_majorant(SWITCH_RAW_RATIO) == UNIT_DEFICIT_UPPER
        ),
        "earlier_bands_are_dominated": all(bound < UNIT_DEFICIT_UPPER for bound in earlier_bounds),
        "tail_is_positive": TAIL_DERIVATIVE_LOWER < 0 and TAIL_PROJECTION_LOWER < 0,
        "tail_is_strictly_decreasing": (derivative_constant < 0 and TAIL_PROJECTION_LOWER < 0),
        "integral_matches_at_switch": (
            radial_integral_terms(SWITCH_RAW_RATIO)
            == LogarithmicValue(
                UNIT_DEFICIT_UPPER * SWITCH_RAW_RATIO,
                Fraction(0),
                Fraction(1),
            )
        ),
        "locked_lipschitz_bracket_is_positive": (
            Fraction(0) < PAIR_DEFICIT_STRICT_LOWER < UNIT_DEFICIT_UPPER
        ),
        "repair_only_jury_control_fails": repair_control.second_jury_margin < 0,
        "full_repaired_jury_control_fails": full_control.second_jury_margin < 0,
        "one_step_signal_is_locked": (
            one_step.signal == DEPLOYED_EPSILON / 399
            and one_step.normalized_signal == Fraction(1, 400)
        ),
        "one_step_objective_expansion_is_large": (
            one_step.objective_growth_strict_lower > 23_325_554
        ),
    }
