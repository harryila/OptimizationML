"""Exact P10 certificate for the finite-precision outer optimizer shell.

The executable shell supplies a rounded gradient ``g_hat=C32(grad f(W))`` and
forms FP32 EMA/Nesterov states.  Its arithmetic residuals are augmented by the
one gradient-boundary cast so that, relative to the exact gradient ``g``,

``m_next = beta*m + (1-beta)*g + r_m`` and
``s_next = beta*m_next + (1-beta)*g + r_s``.

With ``a=1-beta``, P7 sees the effective gradient error ``r_m/a``.  Its
nominal Nesterov signal differs from the actual FP32 signal by ``r_s-r_m``.
If the three-word logical master changes by

``W_next = W - eta*Rhat(s_next) + r_W``, then P7's effective operator error is

``R(s_next)-R(s0) + (Rhat(s_next)-R(s_next)) - r_W/eta``.

This module bounds all three ports from the exact constants exported by the
proof-reference shell, composes them with P9's binary32-input affine operator
bound, and absorbs their storage-dependent slopes into P7.  Every propagated
coefficient is rounded upward on the declared ``2**-40`` rational grid.

The positive result is conditional on the displayed finite arithmetic guards
and on evaluating the exact objective gradient at the same logical real
parameter ``high+middle+low`` before the one FP32 boundary cast.  In
particular, the high-word guard is not inferred from PL storage.  This is
unavoidable without another state assumption: PL objectives may have flat
minimizer directions, so their storage is not coercive in the parameter.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isqrt

from passive_muon.finite_precision_outer_loop import (
    BETA_FP32_EXACT,
    CERTIFIED_STORAGE_PROFILE_OPERATOR_OUTPUT_MAX_ABS,
    EMA_GRADIENT_ENVELOPE_COEFFICIENT,
    EMA_MOMENTUM_ENVELOPE_COEFFICIENT,
    FP32_HALF_MIN_SUBNORMAL,
    FP32_UNIT_ROUNDOFF,
    LEARNING_RATE_FP32_EXACT,
    LOCKED_MASTER_HIGH_MAX_ABS,
    LOCKED_MASTER_LOW_MAX_ABS,
    LOCKED_MASTER_MIDDLE_MAX_ABS,
    LOCKED_OPERATOR_OUTPUT_MAX_ABS,
    LOCKED_ROUNDED_STEP_MAX_ABS,
    ONE_MINUS_BETA_FP32_EXACT,
    FinitePrecisionOuterLoopConfig,
    master_guard_invariant_audit,
    outer_residual_envelope,
)
from passive_muon.robust_dissipativity import (
    RobustDissipativityAudit,
    audit_robust_dissipativity,
    locked_robust_dissipativity_certificate,
)
from passive_muon.scalable_mixed_precision_certificate import (
    FP32_MAX_FINITE,
    IDEAL_REPAIRED_LIPSCHITZ,
    LOCKED_SAFE_MAX_ABS,
    P7_SMOOTHNESS,
    P7_STORAGE_FUNCTION_LOWER,
    UPPER_GRID_DENOMINATOR,
    MatrixShape,
    ScalableMixedPrecisionAudit,
    audit_scalable_shape,
)

LOCKED_OUTER_SHAPE = MatrixShape(4_096, 11_008)
LOCKED_STORAGE_RADIUS = Fraction(1)
LOCKED_GRADIENT_YOUNG = 1
LOCKED_OPERATOR_YOUNG = 837


def _upper(value: Fraction) -> Fraction:
    """Round a nonnegative quantity upward on P9's exact dyadic grid."""

    if value < 0:
        raise ValueError("an upper certificate quantity must be nonnegative")
    numerator, remainder = divmod(
        value.numerator * UPPER_GRID_DENOMINATOR,
        value.denominator,
    )
    return Fraction(numerator + bool(remainder), UPPER_GRID_DENOMINATOR)


def _lower(value: Fraction) -> Fraction:
    """Round a nonnegative capacity downward on P9's exact dyadic grid."""

    if value < 0:
        raise ValueError("a lower certificate quantity must be nonnegative")
    return Fraction(
        value.numerator * UPPER_GRID_DENOMINATOR // value.denominator,
        UPPER_GRID_DENOMINATOR,
    )


def _sqrt_upper(value: Fraction) -> Fraction:
    """Return ``ceil(2**40*sqrt(value))/2**40`` without floating point."""

    if value < 0:
        raise ValueError("a square-root certificate quantity must be nonnegative")
    scaled_numerator = value.numerator * UPPER_GRID_DENOMINATOR**2
    candidate = isqrt(scaled_numerator // value.denominator)
    if candidate**2 * value.denominator < scaled_numerator:
        candidate += 1
    return Fraction(candidate, UPPER_GRID_DENOMINATOR)


@dataclass(frozen=True)
class AffineStorageEnvelope:
    """A bound ``||quantity||_F <= slope*sqrt(V)+intercept``."""

    slope: Fraction
    intercept: Fraction

    def __post_init__(self) -> None:
        if not isinstance(self.slope, Fraction) or not isinstance(self.intercept, Fraction):
            raise TypeError("affine-envelope coefficients must be fractions.Fraction values")
        if self.slope < 0 or self.intercept < 0:
            raise ValueError("affine-envelope coefficients must be nonnegative")

    def at_storage(self, radius: Fraction) -> Fraction:
        """Upper-bound the quantity on ``V<=radius`` on the exact grid."""

        if not isinstance(radius, Fraction) or radius < 0:
            raise ValueError("storage radius must be a nonnegative Fraction")
        return _upper(self.slope * _sqrt_upper(radius) + self.intercept)


@dataclass(frozen=True)
class OuterLoopPortReduction:
    """Exact shape-specific arithmetic coefficients and storage envelopes."""

    shape: MatrixShape
    sqrt_entries_upper: int
    runtime_beta: Fraction
    runtime_one_minus_beta: Fraction
    runtime_learning_rate: Fraction
    ema_momentum_coefficient: Fraction
    ema_gradient_coefficient: Fraction
    ema_crumb: Fraction
    represented_momentum_coefficient: Fraction
    represented_gradient_coefficient: Fraction
    p9_binary32_slope: Fraction
    p9_binary32_intercept: Fraction
    repaired_lipschitz: Fraction
    momentum_port: AffineStorageEnvelope
    signal_port: AffineStorageEnvelope
    actual_signal: AffineStorageEnvelope
    operator_output: AffineStorageEnvelope
    parameter_port: AffineStorageEnvelope
    effective_gradient_error: AffineStorageEnvelope
    effective_operator_error: AffineStorageEnvelope


@dataclass(frozen=True)
class OuterLoopGuardClosure:
    """Finite-range and forward-invariance checks at one storage radius."""

    storage_radius: Fraction
    signal_at_radius: Fraction
    p9_signal_max_abs: Fraction
    operator_output_at_radius: Fraction
    certificate_operator_output_max_abs: Fraction
    runtime_operator_output_max_abs: Fraction
    rounded_step_at_radius: Fraction
    rounded_step_max_abs: Fraction
    master_high_max_abs: Fraction
    master_middle_max_abs: Fraction
    master_low_max_abs: Fraction
    signal_capacity_radius: Fraction
    operator_capacity_radius: Fraction
    forcing_capacity: Fraction
    ema_intermediate_frobenius_bounds: dict[str, Fraction]
    fp32_max_finite: Fraction
    ema_intermediates_are_finite: bool
    high_word_guard_is_conditional: bool
    middle_low_invariant_checks: dict[str, bool]


@dataclass(frozen=True)
class OuterLoopRoundoffCertificate:
    """Exact P10 storage inequality for the proposed finite-precision shell."""

    reduction: OuterLoopPortReduction
    beta: Fraction
    learning_rate: Fraction
    p7_rate: Fraction
    p7_gradient_gain: Fraction
    p7_operator_gain: Fraction
    gradient_young: int
    operator_young: int
    rate: Fraction
    constant_forcing: Fraction
    function_gap_ultimate: Fraction
    guard: OuterLoopGuardClosure

    def __post_init__(self) -> None:
        if self.gradient_young <= 0 or self.operator_young <= 0:
            raise ValueError("Young parameters must be positive")
        if not self.p7_rate < self.rate < 1:
            raise ValueError("the absorbed rate must lie strictly between the P7 rate and one")
        if self.constant_forcing <= 0 or self.function_gap_ultimate <= 0:
            raise ValueError("the locked forcing and objective neighborhood must be positive")

    @property
    def certified(self) -> bool:
        return (
            self.rate < 1
            and self.constant_forcing <= self.guard.forcing_capacity
            and self.guard.signal_at_radius <= self.guard.p9_signal_max_abs
            and self.guard.operator_output_at_radius
            <= self.guard.certificate_operator_output_max_abs
            and self.guard.certificate_operator_output_max_abs
            <= self.guard.runtime_operator_output_max_abs
            and self.guard.rounded_step_at_radius <= self.guard.rounded_step_max_abs
            and self.guard.ema_intermediates_are_finite
            and all(self.guard.middle_low_invariant_checks.values())
            and self.guard.high_word_guard_is_conditional
        )


@dataclass(frozen=True)
class FlatDirectionObstruction:
    """Exact witness ruling out an unconditional parameter-state ISS claim."""

    objective: str
    smoothness_upper: Fraction
    pl_constant: Fraction
    objective_gap: Fraction
    gradient_norm: Fraction
    momentum_norm: Fraction
    bounded_error_causes_unbounded_parameter: bool
    square_summable_error_causes_parameter_drift: bool


@dataclass(frozen=True)
class OuterLoopRoundoffAudit:
    """Complete exact dependency audit for the locked P10 result."""

    certificate: OuterLoopRoundoffCertificate
    p9_audit: ScalableMixedPrecisionAudit
    p7_audit: RobustDissipativityAudit
    obstruction: FlatDirectionObstruction
    checks: dict[str, bool]

    @property
    def certified(self) -> bool:
        return bool(self.checks) and all(self.checks.values())


def _storage_linear_slope(
    momentum_coefficient: Fraction,
    gradient_coefficient: Fraction,
    *,
    storage: tuple[tuple[Fraction, ...], ...],
    smoothness: Fraction,
) -> Fraction:
    """Convert ``a||m||+b||g||`` to a multiple of ``sqrt(V)``."""

    if momentum_coefficient < 0 or gradient_coefficient < 0:
        raise ValueError("storage-envelope coefficients must be nonnegative")
    p00, p01 = storage[0]
    p10, p11 = storage[1]
    if p01 != p10 or p01 > 0:
        raise ValueError("the locked P7 storage sign pattern changed")
    determinant = p00 * p11 - p01**2
    if p00 <= 0 or determinant <= 0:
        raise ValueError("the locked P7 storage is not positive definite")
    dual_square = (
        smoothness**2
        / determinant
        * (
            p11 * momentum_coefficient**2
            - 2 * p01 * momentum_coefficient * gradient_coefficient
            + p00 * gradient_coefficient**2
        )
    )
    return _sqrt_upper(dual_square)


def _affine_from_state_coefficients(
    momentum_coefficient: Fraction,
    gradient_coefficient: Fraction,
    intercept: Fraction,
    *,
    storage: tuple[tuple[Fraction, ...], ...],
    smoothness: Fraction,
) -> AffineStorageEnvelope:
    return AffineStorageEnvelope(
        slope=_storage_linear_slope(
            momentum_coefficient,
            gradient_coefficient,
            storage=storage,
            smoothness=smoothness,
        ),
        intercept=_upper(intercept),
    )


def _build_port_reduction(
    shape: MatrixShape,
) -> tuple[OuterLoopPortReduction, ScalableMixedPrecisionAudit, RobustDissipativityAudit]:
    p9 = audit_scalable_shape(shape)
    if not p9.certified:
        raise ValueError("the selected shape does not pass the P9 exact recurrence")
    robust = locked_robust_dissipativity_certificate()
    p7 = audit_robust_dissipativity(robust)
    if not p7.certified:
        raise AssertionError("the inherited P7 exact LMI did not replay")

    outer_config = FinitePrecisionOuterLoopConfig((shape.rows, shape.columns))
    raw = outer_residual_envelope(outer_config)
    base = robust.pl_certificate
    one_minus_beta = 1 - base.beta
    u = FP32_UNIT_ROUNDOFF
    cast_crumb = raw.sqrt_entry_count_upper * FP32_HALF_MIN_SUBNORMAL
    storage = base.storage
    smoothness = base.smoothness

    # The executable coefficients are relative to represented g_hat.  Adding
    # a*(g_hat-g) makes the residual relative to the true gradient.
    c_beta = raw.ema_momentum_coefficient
    c_gradient = raw.ema_gradient_coefficient
    ema_crumb = raw.ema_absolute_crumb
    momentum_port = _affine_from_state_coefficients(
        c_beta,
        c_gradient * (1 + u) + one_minus_beta * u,
        ema_crumb + (c_gradient + one_minus_beta) * cast_crumb,
        storage=storage,
        smoothness=smoothness,
    )

    represented_momentum_coefficient = (1 + u) ** 2 * BETA_FP32_EXACT
    represented_gradient_coefficient = (1 + u) ** 2 * ONE_MINUS_BETA_FP32_EXACT
    next_momentum_m = represented_momentum_coefficient
    next_momentum_g = represented_gradient_coefficient * (1 + u)
    next_momentum_crumb = ema_crumb + represented_gradient_coefficient * cast_crumb
    signal_port = _affine_from_state_coefficients(
        c_beta * next_momentum_m,
        c_beta * next_momentum_g + c_gradient * (1 + u) + one_minus_beta * u,
        c_beta * next_momentum_crumb + ema_crumb + (c_gradient + one_minus_beta) * cast_crumb,
        storage=storage,
        smoothness=smoothness,
    )

    # Directly bound the five-op FP32 signal graph.  This is slightly sharper
    # than reconstructing the signal from the residual ports.
    signal_m = represented_momentum_coefficient**2
    signal_g_hat = (
        represented_momentum_coefficient * represented_gradient_coefficient
        + represented_gradient_coefficient
    )
    signal_crumb_hat = (1 + represented_momentum_coefficient) * ema_crumb
    actual_signal = _affine_from_state_coefficients(
        signal_m,
        signal_g_hat * (1 + u),
        signal_crumb_hat + signal_g_hat * cast_crumb,
        storage=storage,
        smoothness=smoothness,
    )

    binary_slope = p9.operator.binary32_slope
    binary_intercept = p9.operator.binary32_intercept
    operator_output = AffineStorageEnvelope(
        slope=_upper((IDEAL_REPAIRED_LIPSCHITZ + binary_slope) * actual_signal.slope),
        intercept=_upper(
            (IDEAL_REPAIRED_LIPSCHITZ + binary_slope) * actual_signal.intercept + binary_intercept
        ),
    )

    low_frobenius = raw.sqrt_entry_count_upper * LOCKED_MASTER_LOW_MAX_ABS
    parameter_port = AffineStorageEnvelope(
        slope=_upper(raw.master_output_coefficient * operator_output.slope),
        intercept=_upper(
            raw.master_output_coefficient * operator_output.intercept
            + raw.master_low_word_coefficient * low_frobenius
            + raw.master_absolute_crumb
        ),
    )
    effective_gradient_error = AffineStorageEnvelope(
        slope=_upper(momentum_port.slope / one_minus_beta),
        intercept=_upper(momentum_port.intercept / one_minus_beta),
    )
    effective_operator_error = AffineStorageEnvelope(
        slope=_upper(
            binary_slope * actual_signal.slope
            + IDEAL_REPAIRED_LIPSCHITZ * (momentum_port.slope + signal_port.slope)
            + parameter_port.slope / base.learning_rate
        ),
        intercept=_upper(
            binary_intercept
            + IDEAL_REPAIRED_LIPSCHITZ * (momentum_port.intercept + signal_port.intercept)
            + parameter_port.intercept / base.learning_rate
        ),
    )

    return (
        OuterLoopPortReduction(
            shape=shape,
            sqrt_entries_upper=raw.sqrt_entry_count_upper,
            runtime_beta=BETA_FP32_EXACT,
            runtime_one_minus_beta=ONE_MINUS_BETA_FP32_EXACT,
            runtime_learning_rate=LEARNING_RATE_FP32_EXACT,
            ema_momentum_coefficient=c_beta,
            ema_gradient_coefficient=c_gradient,
            ema_crumb=ema_crumb,
            represented_momentum_coefficient=represented_momentum_coefficient,
            represented_gradient_coefficient=represented_gradient_coefficient,
            p9_binary32_slope=binary_slope,
            p9_binary32_intercept=binary_intercept,
            repaired_lipschitz=IDEAL_REPAIRED_LIPSCHITZ,
            momentum_port=momentum_port,
            signal_port=signal_port,
            actual_signal=actual_signal,
            operator_output=operator_output,
            parameter_port=parameter_port,
            effective_gradient_error=effective_gradient_error,
            effective_operator_error=effective_operator_error,
        ),
        p9,
        p7,
    )


def build_outer_loop_roundoff_certificate(
    *,
    shape: MatrixShape | tuple[int, int] = LOCKED_OUTER_SHAPE,
    storage_radius: Fraction = LOCKED_STORAGE_RADIUS,
    gradient_young: int = LOCKED_GRADIENT_YOUNG,
    operator_young: int = LOCKED_OPERATOR_YOUNG,
) -> tuple[OuterLoopRoundoffCertificate, ScalableMixedPrecisionAudit, RobustDissipativityAudit]:
    """Build the exact compositional P10 certificate at one P9 shape."""

    if not isinstance(storage_radius, Fraction) or storage_radius <= 0:
        raise ValueError("storage_radius must be a positive Fraction")
    if gradient_young <= 0 or operator_young <= 0:
        raise ValueError("Young parameters must be positive integers")
    selected_shape = shape if isinstance(shape, MatrixShape) else MatrixShape(*shape)
    reduction, p9, p7 = _build_port_reduction(selected_shape)
    robust = locked_robust_dissipativity_certificate()
    xi = reduction.effective_gradient_error
    error = reduction.effective_operator_error
    theta_g = Fraction(gradient_young)
    theta_e = Fraction(operator_young)
    rate = _upper(
        robust.rate
        + robust.gradient_noise_gain * (1 + theta_g) * xi.slope**2
        + robust.implementation_error_gain * (1 + theta_e) * error.slope**2
    )
    forcing = _upper(
        robust.gradient_noise_gain * (1 + 1 / theta_g) * xi.intercept**2
        + robust.implementation_error_gain * (1 + 1 / theta_e) * error.intercept**2
    )
    function_gap_ultimate = _upper(P7_SMOOTHNESS / P7_STORAGE_FUNCTION_LOWER * forcing / (1 - rate))

    signal_at_radius = reduction.actual_signal.at_storage(storage_radius)
    output_at_radius = reduction.operator_output.at_storage(storage_radius)
    robust = locked_robust_dissipativity_certificate()
    base = robust.pl_certificate
    momentum_at_radius = _storage_linear_slope(
        Fraction(1),
        Fraction(0),
        storage=base.storage,
        smoothness=base.smoothness,
    ) * _sqrt_upper(storage_radius)
    gradient_at_radius = _storage_linear_slope(
        Fraction(0),
        Fraction(1),
        storage=base.storage,
        smoothness=base.smoothness,
    ) * _sqrt_upper(storage_radius)
    cast_crumb = reduction.sqrt_entries_upper * FP32_HALF_MIN_SUBNORMAL
    represented_gradient_at_radius = _upper(
        (1 + FP32_UNIT_ROUNDOFF) * gradient_at_radius + cast_crumb
    )
    beta_momentum_at_radius = _upper(
        (1 + FP32_UNIT_ROUNDOFF) * BETA_FP32_EXACT * momentum_at_radius + cast_crumb
    )
    weighted_gradient_at_radius = _upper(
        (1 + FP32_UNIT_ROUNDOFF) * ONE_MINUS_BETA_FP32_EXACT * represented_gradient_at_radius
        + cast_crumb
    )
    momentum_next_at_radius = _upper(
        (1 + FP32_UNIT_ROUNDOFF) * (beta_momentum_at_radius + weighted_gradient_at_radius)
        + cast_crumb
    )
    beta_momentum_next_at_radius = _upper(
        (1 + FP32_UNIT_ROUNDOFF) * BETA_FP32_EXACT * momentum_next_at_radius + cast_crumb
    )
    ema_intermediate_bounds = {
        "true_gradient": _upper(gradient_at_radius),
        "represented_gradient": represented_gradient_at_radius,
        "beta_momentum": beta_momentum_at_radius,
        "weighted_gradient": weighted_gradient_at_radius,
        "momentum_next": momentum_next_at_radius,
        "beta_momentum_next": beta_momentum_next_at_radius,
        "signal": signal_at_radius,
    }
    ema_intermediates_are_finite = all(
        value < FP32_MAX_FINITE / 2 for value in ema_intermediate_bounds.values()
    )
    rounded_step_at_radius = _upper(
        (1 + FP32_UNIT_ROUNDOFF) * LEARNING_RATE_FP32_EXACT * output_at_radius
        + FP32_HALF_MIN_SUBNORMAL
    )
    signal_capacity_radius = _lower(
        ((LOCKED_SAFE_MAX_ABS - reduction.actual_signal.intercept) / reduction.actual_signal.slope)
        ** 2
    )
    operator_capacity_radius = _lower(
        (
            (
                CERTIFIED_STORAGE_PROFILE_OPERATOR_OUTPUT_MAX_ABS
                - reduction.operator_output.intercept
            )
            / reduction.operator_output.slope
        )
        ** 2
    )
    forcing_capacity = _lower((1 - rate) * storage_radius)
    master_audit = master_guard_invariant_audit()
    guard = OuterLoopGuardClosure(
        storage_radius=storage_radius,
        signal_at_radius=signal_at_radius,
        p9_signal_max_abs=LOCKED_SAFE_MAX_ABS,
        operator_output_at_radius=output_at_radius,
        certificate_operator_output_max_abs=(CERTIFIED_STORAGE_PROFILE_OPERATOR_OUTPUT_MAX_ABS),
        runtime_operator_output_max_abs=LOCKED_OPERATOR_OUTPUT_MAX_ABS,
        rounded_step_at_radius=rounded_step_at_radius,
        rounded_step_max_abs=LOCKED_ROUNDED_STEP_MAX_ABS,
        master_high_max_abs=LOCKED_MASTER_HIGH_MAX_ABS,
        master_middle_max_abs=LOCKED_MASTER_MIDDLE_MAX_ABS,
        master_low_max_abs=LOCKED_MASTER_LOW_MAX_ABS,
        signal_capacity_radius=signal_capacity_radius,
        operator_capacity_radius=operator_capacity_radius,
        forcing_capacity=forcing_capacity,
        ema_intermediate_frobenius_bounds=ema_intermediate_bounds,
        fp32_max_finite=FP32_MAX_FINITE,
        ema_intermediates_are_finite=ema_intermediates_are_finite,
        high_word_guard_is_conditional=True,
        middle_low_invariant_checks=dict(master_audit["checks"]),
    )
    certificate = OuterLoopRoundoffCertificate(
        reduction=reduction,
        beta=robust.pl_certificate.beta,
        learning_rate=robust.pl_certificate.learning_rate,
        p7_rate=robust.rate,
        p7_gradient_gain=robust.gradient_noise_gain,
        p7_operator_gain=robust.implementation_error_gain,
        gradient_young=gradient_young,
        operator_young=operator_young,
        rate=rate,
        constant_forcing=forcing,
        function_gap_ultimate=function_gap_ultimate,
        guard=guard,
    )
    return certificate, p9, p7


def flat_direction_obstruction() -> FlatDirectionObstruction:
    """Return the nonunique-PL obstruction to unconditional parameter ISS."""

    return FlatDirectionObstruction(
        objective="f(x,y)=x^2/2",
        smoothness_upper=Fraction(10),
        pl_constant=Fraction(1),
        objective_gap=Fraction(0),
        gradient_norm=Fraction(0),
        momentum_norm=Fraction(0),
        bounded_error_causes_unbounded_parameter=True,
        square_summable_error_causes_parameter_drift=True,
    )


def harmonic_parameter_drift(horizon: int) -> tuple[Fraction, Fraction]:
    """Return exact displacement and error energy through ``horizon``."""

    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    displacement = sum(
        (Fraction(1, index) for index in range(1, horizon + 1)),
        Fraction(0),
    )
    energy = sum(
        (Fraction(1, index**2) for index in range(1, horizon + 1)),
        Fraction(0),
    )
    return displacement, energy


def audit_outer_loop_roundoff() -> OuterLoopRoundoffAudit:
    """Replay every exact dependency and locked P10 success gate."""

    certificate, p9, p7 = build_outer_loop_roundoff_certificate()
    robust = locked_robust_dissipativity_certificate()
    obstruction = flat_direction_obstruction()
    guard = certificate.guard
    checks = {
        "p7_exact_lmi_replays": p7.certified,
        "p9_shape_certificate_replays": p9.certified,
        "locked_shape_is_4096x11008": certificate.reduction.shape == MatrixShape(4_096, 11_008),
        "gradient_boundary_cast_is_absorbed": (
            certificate.reduction.momentum_port.slope
            > _storage_linear_slope(
                EMA_MOMENTUM_ENVELOPE_COEFFICIENT,
                EMA_GRADIENT_ENVELOPE_COEFFICIENT,
                storage=robust.pl_certificate.storage,
                smoothness=robust.pl_certificate.smoothness,
            )
        ),
        "effective_gradient_scale_uses_one_minus_beta": (
            certificate.reduction.effective_gradient_error.slope
            >= certificate.reduction.momentum_port.slope / (1 - certificate.beta)
        ),
        "rate_is_strict": certificate.rate < 1,
        "unit_storage_is_forward_invariant": (
            certificate.constant_forcing <= guard.forcing_capacity
        ),
        "p9_signal_guard_closes": guard.signal_at_radius <= guard.p9_signal_max_abs,
        "gradient_cast_and_ema_intermediates_are_finite": (guard.ema_intermediates_are_finite),
        "operator_output_is_below_2^15": (
            guard.operator_output_at_radius
            <= guard.certificate_operator_output_max_abs
            == Fraction(2**15)
        ),
        "certificate_output_is_in_runtime_domain": (
            guard.certificate_operator_output_max_abs <= guard.runtime_operator_output_max_abs
        ),
        "rounded_step_is_below_two": (
            guard.rounded_step_at_radius <= guard.rounded_step_max_abs == 2
        ),
        "middle_low_guards_are_preserved": all(guard.middle_low_invariant_checks.values()),
        "high_guard_is_explicitly_conditional": guard.high_word_guard_is_conditional,
        "objective_neighborhood_is_below_one": certificate.function_gap_ultimate < 1,
        "bounded_update_port_does_not_imply_full_state_iss": (
            obstruction.bounded_error_causes_unbounded_parameter
        ),
        "square_summable_update_port_does_not_imply_iterate_convergence": (
            obstruction.square_summable_error_causes_parameter_drift
        ),
        "complete_certificate_accepts": certificate.certified,
    }
    return OuterLoopRoundoffAudit(
        certificate=certificate,
        p9_audit=p9,
        p7_audit=p7,
        obstruction=obstruction,
        checks=checks,
    )


def locked_outer_loop_roundoff_certificate() -> OuterLoopRoundoffCertificate:
    """Return the locked ``4096 x 11008`` P10 certificate."""

    audit = audit_outer_loop_roundoff()
    if not audit.certified:
        raise AssertionError("locked P10 outer-loop certificate did not replay")
    return audit.certificate


__all__ = [
    "LOCKED_GRADIENT_YOUNG",
    "LOCKED_OPERATOR_YOUNG",
    "LOCKED_OUTER_SHAPE",
    "LOCKED_STORAGE_RADIUS",
    "AffineStorageEnvelope",
    "FlatDirectionObstruction",
    "OuterLoopGuardClosure",
    "OuterLoopPortReduction",
    "OuterLoopRoundoffAudit",
    "OuterLoopRoundoffCertificate",
    "audit_outer_loop_roundoff",
    "build_outer_loop_roundoff_certificate",
    "flat_direction_obstruction",
    "harmonic_parameter_drift",
    "locked_outer_loop_roundoff_certificate",
]
