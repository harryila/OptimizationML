"""Exact P8 certificate for a proposed mixed-precision repaired operator.

The certified implementation is deliberately narrower than upstream Muon.  It
has locked shape ``2 x 2`` and stores only the boundaries of the five Jordan
stages in BF16.  Every complete Horner stage is evaluated in FP32 with fixed,
separately rounded multiply and add operations before one BF16 round.  Fused
multiply-add is neither assumed nor permitted by the locked semantics.  The max-floor
normalizer, linear repair, and returned operator value use FP32.

This module certifies a post-operator error bound for the exact-real P7 input
port.  It does *not* certify FP32 EMA/Nesterov state updates or FP32 parameter
updates.  All authoritative calculations use :class:`fractions.Fraction`.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise
from typing import TypeAlias

Polynomial: TypeAlias = tuple[Fraction, ...]

LOCKED_SHAPE = (2, 2)
LOCKED_STAGES = 5
LOCKED_SAFE_MAX_ABS = Fraction(2**116)

EXACT_A = Fraction(6_889, 2_000)
EXACT_B = Fraction(-191, 40)
EXACT_C = Fraction(4_063, 2_000)
FP32_A = Fraction(902_955, 262_144)
FP32_B = Fraction(-10_013_901, 2_097_152)
FP32_C = Fraction(8_520_729, 4_194_304)

EXACT_RHO = Fraction(210_177_835_339_081, 260_261_360_000)
FP32_RHO = Fraction(13_231_137, 16_384)

FP32_UNIT_ROUNDOFF = Fraction(1, 2**24)
FP32_HALF_MIN_SUBNORMAL = Fraction(1, 2**150)
BF16_UNIT_ROUNDOFF = Fraction(1, 2**8)
BF16_HALF_MIN_SUBNORMAL = Fraction(1, 2**134)
SQRT_TWO_UPPER = Fraction(99, 70)

SPECTRAL_INVARIANT = Fraction(5, 4)
FROBENIUS_INVARIANT = Fraction(7, 4)
POLYNOMIAL_UPPER = Fraction(121, 100)
LOCAL_FP32_STAGE_ERROR_UPPER = Fraction(1, 20_000)
NORMALIZATION_RELATIVE_ERROR_UPPER = Fraction(1, 2**19)
NORMALIZATION_OUTPUT_ERROR_UPPER = Fraction(1, 500_000)
INITIAL_BOUNDARY_ERROR_UPPER = Fraction(1, 250)

POLYNOMIAL_ERROR_UPPER = Fraction(24_229, 7_000)
AFFINE_SLOPE_UPPER = Fraction(11, 100_000)
AFFINE_INTERCEPT_UPPER = Fraction(347, 100)
IDEAL_POLYNOMIAL_LIPSCHITZ = Fraction(4_848_763, 10_000)
IDEAL_REPAIRED_LIPSCHITZ = Fraction(336_372_400_608_849, 260_261_360_000)
REAL_ADAPTER_SLOPE_UPPER = Fraction(1, 5_000)
REAL_ADAPTER_INTERCEPT_UPPER = AFFINE_INTERCEPT_UPPER

P7_RATE = Fraction(399_960_001, 400_000_000)
P7_IMPLEMENTATION_ERROR_GAIN = Fraction(1, 2_000_000)
P7_SMOOTHNESS = Fraction(10)
P7_STORAGE_FUNCTION_LOWER = Fraction(13_533, 50_000)
P7_SIGNAL_KAPPA = Fraction(66_221_761, 10_400_336)
P7_SIGNAL_STORAGE_GAIN = Fraction(1_655_544_025, 2_600_084)
YOUNG_THETA = Fraction(5_124)
P8_RATE = Fraction(41_597_186_684_695_561, 41_601_344_000_000_000)
P8_FORCING = Fraction(4_936_769, 819_840_000_000)
P8_FUNCTION_GAP_ULTIMATE = Fraction(
    462_392_438_350_000_000,
    207_695_315_294_468_001,
)
P8_SAFE_STORAGE_RADIUS = LOCKED_SAFE_MAX_ABS**2 / P7_SIGNAL_STORAGE_GAIN
P8_SAFE_FORCING_CAPACITY = (1 - P8_RATE) * P8_SAFE_STORAGE_RADIUS


def _trim(polynomial: Polynomial) -> Polynomial:
    values = list(polynomial)
    while len(values) > 1 and values[-1] == 0:
        values.pop()
    return tuple(values)


def _derivative(polynomial: Polynomial) -> Polynomial:
    if len(polynomial) == 1:
        return (Fraction(0),)
    return tuple(Fraction(index) * value for index, value in enumerate(polynomial[1:], 1))


def _polynomial_division(
    dividend: Polynomial, divisor: Polynomial
) -> tuple[Polynomial, Polynomial]:
    """Return exact quotient and remainder for ascending coefficient tuples."""

    numerator = list(_trim(dividend))
    denominator = _trim(divisor)
    if denominator == (0,):
        raise ZeroDivisionError("polynomial division by zero")
    if len(numerator) < len(denominator):
        return (Fraction(0),), tuple(numerator)

    quotient = [Fraction(0)] * (len(numerator) - len(denominator) + 1)
    while len(numerator) >= len(denominator) and any(numerator):
        offset = len(numerator) - len(denominator)
        coefficient = numerator[-1] / denominator[-1]
        quotient[offset] = coefficient
        for index, value in enumerate(denominator):
            numerator[offset + index] -= coefficient * value
        numerator = list(_trim(tuple(numerator)))
    return _trim(tuple(quotient)), _trim(tuple(numerator))


def _evaluate(polynomial: Polynomial, point: Fraction) -> Fraction:
    result = Fraction(0)
    for coefficient in reversed(polynomial):
        result = result * point + coefficient
    return result


def _sign(value: Fraction) -> int:
    return (value > 0) - (value < 0)


def _sign_variations(signs: tuple[int, ...]) -> int:
    nonzero = tuple(value for value in signs if value)
    return sum(left != right for left, right in pairwise(nonzero))


def rebuild_sturm_chain(polynomial: Polynomial) -> tuple[Polynomial, ...]:
    """Build the exact Sturm chain using unscaled Euclidean remainders."""

    chain = [polynomial, _derivative(polynomial)]
    while len(chain[-1]) > 1:
        _, remainder = _polynomial_division(chain[-2], chain[-1])
        if remainder == (0,):
            break
        chain.append(tuple(-value for value in remainder))
    return tuple(chain)


LOCKED_STURM_CHAIN: tuple[Polynomial, ...] = (
    (
        Fraction(121, 100),
        Fraction(-6_889, 2_000),
        Fraction(0),
        Fraction(191, 40),
        Fraction(0),
        Fraction(-4_063, 2_000),
    ),
    (
        Fraction(-6_889, 2_000),
        Fraction(0),
        Fraction(573, 40),
        Fraction(0),
        Fraction(-4_063, 400),
    ),
    (
        Fraction(-121, 100),
        Fraction(6_889, 2_500),
        Fraction(0),
        Fraction(-191, 100),
    ),
    (
        Fraction(6_889, 2_000),
        Fraction(-491_623, 76_400),
        Fraction(629_257, 1_910_000),
    ),
    (
        Fraction(-7_698_138_961_078_973, 19_798_218_602_450),
        Fraction(698_807_347_396_374_689, 989_910_930_122_500),
    ),
    (
        Fraction(
            -788_576_461_383_197_655_151_760_980_462_539,
            8_071_598_492_151_363_617_077_571_240_989_202_000,
        ),
    ),
)


@dataclass(frozen=True)
class MixedPrecisionCertificate:
    """Locked exact parameters for the P8 operator-error certificate."""

    shape: tuple[int, int] = LOCKED_SHAPE
    stages: int = LOCKED_STAGES
    spectral_bound: Fraction = SPECTRAL_INVARIANT
    frobenius_bound: Fraction = FROBENIUS_INVARIANT
    affine_slope_upper: Fraction = AFFINE_SLOPE_UPPER
    affine_intercept_upper: Fraction = AFFINE_INTERCEPT_UPPER
    young_theta: Fraction = YOUNG_THETA

    def __post_init__(self) -> None:
        if self.shape != LOCKED_SHAPE:
            raise ValueError("the exact P8 certificate is locked to shape (2, 2)")
        if self.stages != LOCKED_STAGES:
            raise ValueError("the exact P8 certificate is locked to five Jordan stages")
        rational_values = (
            self.spectral_bound,
            self.frobenius_bound,
            self.affine_slope_upper,
            self.affine_intercept_upper,
            self.young_theta,
        )
        if any(not isinstance(value, Fraction) for value in rational_values):
            raise TypeError("certificate scalars must be fractions.Fraction values")
        if any(value <= 0 for value in rational_values):
            raise ValueError("certificate bounds must be positive")


@dataclass(frozen=True)
class MixedPrecisionAudit:
    """Exact reconstruction and Boolean gate for the locked P8 certificate."""

    sturm_chain: tuple[Polynomial, ...]
    sturm_endpoint_values: tuple[tuple[Fraction, ...], tuple[Fraction, ...]]
    sturm_endpoint_signs: tuple[tuple[int, ...], tuple[int, ...]]
    sturm_variations: tuple[int, int]
    polynomial_discriminant: Fraction
    normalization: dict[str, Fraction]
    recurrence: dict[str, Fraction]
    repair: dict[str, Fraction]
    p7_closure: dict[str, Fraction]
    checks: dict[str, bool]

    @property
    def certified(self) -> bool:
        """Return true exactly when every locked equality and inequality passes."""

        return bool(self.checks) and all(self.checks.values())


def _normalization_audit() -> dict[str, Fraction]:
    """Bound the locked scaled FP32 normalizer on its ``alpha > 1/2`` branch.

    For ``alpha <= 1/2`` the 2-by-2 implementation returns the input directly,
    because ``||s||_F <= 2*alpha <= 1``.  The relative analysis below is used
    only on the other branch, where the final norm product is normal.  This
    separation is necessary: a global relative norm-error model is false for
    subnormal inputs even though the max-floor output remains well behaved.
    """

    u = FP32_UNIT_ROUNDOFF
    tau = FP32_HALF_MIN_SUBNORMAL
    gamma4 = 4 * u / (1 - 4 * u)
    ratio_component_error = u + 2 * tau
    square_exact_error_per_entry = (2 + ratio_component_error) * ratio_component_error
    square_rounding_error_per_entry = u * (1 + ratio_component_error) ** 2 + tau
    pre_sum_error = 4 * (square_exact_error_per_entry + square_rounding_error_per_entry)
    # The exact scaled square sum is at most four.  This deliberately coarse
    # magnitude bound removes any ambiguity about whether a preceding term is
    # per-entry or aggregate.
    sum_rounding_error = gamma4 * (4 + pre_sum_error) + 4 * tau / (1 - 4 * u)
    squared_norm_relative_error = pre_sum_error + sum_rounding_error
    sqrt_sensitivity = squared_norm_relative_error / (2 - squared_norm_relative_error)
    norm_relative_error = (1 + sqrt_sensitivity) * (1 + u) ** 2 - 1
    epsilon = NORMALIZATION_RELATIVE_ERROR_UPPER
    normalized_output_error = (epsilon + u) / (1 - epsilon) + 2 * tau
    initial_bf16_cast_error = (
        BF16_UNIT_ROUNDOFF * (1 + normalized_output_error) + 2 * BF16_HALF_MIN_SUBNORMAL
    )
    initial_total_error = normalized_output_error + initial_bf16_cast_error
    return {
        "gamma4": gamma4,
        "ratio_component_error": ratio_component_error,
        "square_exact_error_per_entry": square_exact_error_per_entry,
        "square_rounding_error_per_entry": square_rounding_error_per_entry,
        "pre_sum_error": pre_sum_error,
        "sum_rounding_error": sum_rounding_error,
        "squared_norm_relative_error": squared_norm_relative_error,
        "sqrt_sensitivity": sqrt_sensitivity,
        "derived_norm_relative_error": norm_relative_error,
        "locked_norm_relative_error": epsilon,
        "normalized_output_error": normalized_output_error,
        "initial_bf16_cast_error": initial_bf16_cast_error,
        "initial_total_error": initial_total_error,
        "initial_frobenius_bound": 1 + initial_total_error,
        "initial_spectral_bound": 1 + initial_total_error,
    }


def _stage_recurrence() -> dict[str, Fraction]:
    """Return one exact invariant-preserving FP32-Horner stage recurrence."""

    u = FP32_UNIT_ROUNDOFF
    tau = FP32_HALF_MIN_SUBNORMAL
    gamma4 = 4 * u / (1 - 4 * u)
    zmm = 8 * tau / (1 - 4 * u)
    zelt = 2 * tau
    frobenius = FROBENIUS_INVARIANT
    spectral = SPECTRAL_INVARIANT

    gram_exact_frobenius = spectral * frobenius
    gram_exact_spectral = spectral**2
    epsilon_gram = gamma4 * frobenius**2 + zmm
    gram_computed_frobenius = gram_exact_frobenius + epsilon_gram
    gram_computed_spectral = gram_exact_spectral + epsilon_gram

    c_exact_frobenius = abs(EXACT_C) * gram_exact_frobenius
    c_pre_round_frobenius = abs(FP32_C) * gram_computed_frobenius
    epsilon_c = (
        abs(FP32_C) * epsilon_gram
        + abs(FP32_C - EXACT_C) * gram_exact_frobenius
        + u * c_pre_round_frobenius
        + zelt
    )
    c_computed_frobenius = c_exact_frobenius + epsilon_c

    t_exact_spectral = abs(EXACT_B)
    t_exact_frobenius = SQRT_TWO_UPPER * t_exact_spectral
    t_pre_round_frobenius = c_computed_frobenius + abs(FP32_B) * SQRT_TWO_UPPER
    epsilon_t = (
        epsilon_c + abs(FP32_B - EXACT_B) * SQRT_TWO_UPPER + u * t_pre_round_frobenius + zelt
    )
    t_computed_frobenius = t_exact_frobenius + epsilon_t

    d_exact_spectral = EXACT_B**2 / (4 * EXACT_C)
    d_exact_frobenius = SQRT_TWO_UPPER * d_exact_spectral
    epsilon_d = (
        epsilon_t * gram_computed_spectral
        + t_exact_spectral * epsilon_gram
        + gamma4 * t_computed_frobenius * gram_computed_frobenius
        + zmm
    )
    d_computed_frobenius = d_exact_frobenius + epsilon_d

    e_exact_spectral = EXACT_A
    e_exact_frobenius = SQRT_TWO_UPPER * e_exact_spectral
    e_pre_round_frobenius = d_computed_frobenius + abs(FP32_A) * SQRT_TWO_UPPER
    epsilon_e = (
        epsilon_d + abs(FP32_A - EXACT_A) * SQRT_TWO_UPPER + u * e_pre_round_frobenius + zelt
    )
    e_computed_frobenius = e_exact_frobenius + epsilon_e

    epsilon_y = epsilon_e * spectral + gamma4 * e_computed_frobenius * frobenius + zmm
    pre_bf16_frobenius = POLYNOMIAL_UPPER * SQRT_TWO_UPPER + LOCAL_FP32_STAGE_ERROR_UPPER
    pre_bf16_spectral = POLYNOMIAL_UPPER + LOCAL_FP32_STAGE_ERROR_UPPER
    boundary_rounding_error = BF16_UNIT_ROUNDOFF * pre_bf16_frobenius + 2 * BF16_HALF_MIN_SUBNORMAL
    next_frobenius = pre_bf16_frobenius + boundary_rounding_error
    next_spectral = pre_bf16_spectral + boundary_rounding_error
    return {
        "gamma4": gamma4,
        "zmm": zmm,
        "zelt": zelt,
        "gram_exact_frobenius": gram_exact_frobenius,
        "gram_exact_spectral": gram_exact_spectral,
        "epsilon_gram": epsilon_gram,
        "gram_computed_frobenius": gram_computed_frobenius,
        "gram_computed_spectral": gram_computed_spectral,
        "c_exact_frobenius": c_exact_frobenius,
        "c_pre_round_frobenius": c_pre_round_frobenius,
        "epsilon_c": epsilon_c,
        "c_computed_frobenius": c_computed_frobenius,
        "t_exact_spectral": t_exact_spectral,
        "t_exact_frobenius": t_exact_frobenius,
        "t_pre_round_frobenius": t_pre_round_frobenius,
        "epsilon_t": epsilon_t,
        "t_computed_frobenius": t_computed_frobenius,
        "d_exact_spectral": d_exact_spectral,
        "d_exact_frobenius": d_exact_frobenius,
        "epsilon_d": epsilon_d,
        "d_computed_frobenius": d_computed_frobenius,
        "e_exact_spectral": e_exact_spectral,
        "e_exact_frobenius": e_exact_frobenius,
        "e_pre_round_frobenius": e_pre_round_frobenius,
        "epsilon_e": epsilon_e,
        "e_computed_frobenius": e_computed_frobenius,
        "epsilon_y": epsilon_y,
        "pre_bf16_frobenius": pre_bf16_frobenius,
        "pre_bf16_spectral": pre_bf16_spectral,
        "boundary_rounding_error": boundary_rounding_error,
        "next_frobenius": next_frobenius,
        "next_spectral": next_spectral,
    }


def _repair_audit() -> dict[str, Fraction]:
    u = FP32_UNIT_ROUNDOFF
    tau = FP32_HALF_MIN_SUBNORMAL
    first_rounding_slope = abs(FP32_RHO - EXACT_RHO) + u * FP32_RHO
    complete_slope = first_rounding_slope + u * (EXACT_RHO + first_rounding_slope)
    complete_intercept = POLYNOMIAL_ERROR_UPPER + u * FROBENIUS_INVARIANT + (4 + 2 * u) * tau
    adapter_slope = complete_slope * (1 + u) + IDEAL_REPAIRED_LIPSCHITZ * u
    adapter_intercept = complete_intercept + 2 * (complete_slope + IDEAL_REPAIRED_LIPSCHITZ) * tau
    return {
        "coefficient_difference": abs(FP32_RHO - EXACT_RHO),
        "first_rounding_slope": first_rounding_slope,
        "complete_slope": complete_slope,
        "complete_intercept": complete_intercept,
        "slope_upper": AFFINE_SLOPE_UPPER,
        "intercept_upper": AFFINE_INTERCEPT_UPPER,
        "ideal_polynomial_lipschitz": IDEAL_POLYNOMIAL_LIPSCHITZ,
        "ideal_repaired_lipschitz": IDEAL_REPAIRED_LIPSCHITZ,
        "real_adapter_cast_slope": u,
        "real_adapter_cast_intercept": 2 * tau,
        "real_adapter_slope": adapter_slope,
        "real_adapter_intercept": adapter_intercept,
        "real_adapter_slope_upper": REAL_ADAPTER_SLOPE_UPPER,
        "real_adapter_intercept_upper": REAL_ADAPTER_INTERCEPT_UPPER,
    }


def _derived_signal_kappa() -> Fraction:
    """Rebuild ``v^T P^-1 v`` from the exact P7 storage and selector."""

    p00 = Fraction(72_435, 100_000)
    p01 = Fraction(-3_185, 100_000)
    p11 = Fraction(499, 100_000)
    v0 = Fraction(361, 400)
    v1 = Fraction(39, 400)
    determinant = p00 * p11 - p01**2
    return (p11 * v0**2 - 2 * p01 * v0 * v1 + p00 * v1**2) / determinant


def _p7_closure() -> dict[str, Fraction]:
    derived_kappa = _derived_signal_kappa()
    rate = (
        P7_RATE
        + P7_IMPLEMENTATION_ERROR_GAIN
        * (1 + YOUNG_THETA)
        * REAL_ADAPTER_SLOPE_UPPER**2
        * P7_SIGNAL_STORAGE_GAIN
    )
    forcing = P7_IMPLEMENTATION_ERROR_GAIN * (1 + 1 / YOUNG_THETA) * REAL_ADAPTER_INTERCEPT_UPPER**2
    function_gap_ultimate = P7_SMOOTHNESS / P7_STORAGE_FUNCTION_LOWER * forcing / (1 - rate)
    safe_storage_radius = LOCKED_SAFE_MAX_ABS**2 / P7_SIGNAL_STORAGE_GAIN
    safe_forcing_capacity = (1 - rate) * safe_storage_radius
    return {
        "base_rate": P7_RATE,
        "implementation_error_gain": P7_IMPLEMENTATION_ERROR_GAIN,
        "signal_kappa": P7_SIGNAL_KAPPA,
        "derived_signal_kappa": derived_kappa,
        "signal_storage_gain": P7_SIGNAL_STORAGE_GAIN,
        "young_theta": YOUNG_THETA,
        "rate": rate,
        "forcing": forcing,
        "one_minus_rate": 1 - rate,
        "function_storage_lower": P7_STORAGE_FUNCTION_LOWER,
        "smoothness": P7_SMOOTHNESS,
        "function_gap_ultimate": function_gap_ultimate,
        "safe_storage_radius": safe_storage_radius,
        "safe_forcing_capacity": safe_forcing_capacity,
    }


def locked_mixed_precision_certificate() -> MixedPrecisionCertificate:
    """Return the locked P8 certificate parameters."""

    return MixedPrecisionCertificate()


def audit_mixed_precision_certificate(
    certificate: MixedPrecisionCertificate | None = None,
) -> MixedPrecisionAudit:
    """Rebuild and check every exact equality and strict P8 inequality."""

    selected = locked_mixed_precision_certificate() if certificate is None else certificate
    upper_polynomial = LOCKED_STURM_CHAIN[0]
    rebuilt_chain = rebuild_sturm_chain(upper_polynomial)
    endpoints = (Fraction(0), selected.spectral_bound)
    endpoint_values = tuple(
        tuple(_evaluate(polynomial, point) for polynomial in rebuilt_chain) for point in endpoints
    )
    endpoint_signs = tuple(tuple(_sign(value) for value in values) for values in endpoint_values)
    variations = tuple(_sign_variations(signs) for signs in endpoint_signs)
    discriminant = EXACT_B**2 - 4 * EXACT_A * EXACT_C

    normalization = _normalization_audit()
    recurrence = _stage_recurrence()
    repair = _repair_audit()
    p7_closure = _p7_closure()

    checks = {
        "shape_is_locked_2_by_2": selected.shape == LOCKED_SHAPE,
        "five_stages_are_locked": selected.stages == LOCKED_STAGES,
        "selected_spectral_bound_is_locked": selected.spectral_bound == SPECTRAL_INVARIANT,
        "selected_frobenius_bound_is_locked": selected.frobenius_bound == FROBENIUS_INVARIANT,
        "selected_affine_slope_is_locked": selected.affine_slope_upper == AFFINE_SLOPE_UPPER,
        "selected_affine_intercept_is_locked": selected.affine_intercept_upper
        == AFFINE_INTERCEPT_UPPER,
        "selected_young_theta_is_locked": selected.young_theta == YOUNG_THETA,
        "sqrt_two_bound_is_strict": SQRT_TWO_UPPER**2 > 2,
        "normalizer_range_excludes_overflow": 2 * LOCKED_SAFE_MAX_ABS < 2**118,
        "repair_range_excludes_overflow": (1 + FP32_UNIT_ROUNDOFF) * FP32_RHO * LOCKED_SAFE_MAX_ABS
        + FP32_HALF_MIN_SUBNORMAL
        + FROBENIUS_INVARIANT
        < 2**127,
        "fp32_a_encoding": Fraction(-1, 32_768_000) == FP32_A - EXACT_A,
        "fp32_b_encoding": Fraction(-1, 10_485_760) == FP32_B - EXACT_B,
        "fp32_c_encoding": Fraction(53, 524_288_000) == FP32_C - EXACT_C,
        "polynomial_factor_is_positive": discriminant == Fraction(-2_594_691, 500_000),
        "polynomial_leading_coefficient_positive": EXACT_C > 0,
        "polynomial_factor_at_zero_positive": EXACT_A > 0,
        "polynomial_upper_closes_scalar_iteration": POLYNOMIAL_UPPER < SPECTRAL_INVARIANT,
        "t_envelope_sign_on_invariant": EXACT_B + EXACT_C * SPECTRAL_INVARIANT**2 < 0,
        "sturm_chain_rebuilds_exactly": rebuilt_chain == LOCKED_STURM_CHAIN,
        "sturm_left_signs": endpoint_signs[0] == (1, -1, -1, 1, -1, -1),
        "sturm_right_signs": endpoint_signs[1] == (1, -1, -1, -1, 1, -1),
        "sturm_variations_match": variations == (3, 3),
        "upper_polynomial_left_positive": endpoint_values[0][0] == Fraction(121, 100),
        "upper_polynomial_right_positive": endpoint_values[1][0] == Fraction(12_657, 409_600),
        "normalization_relative_bound": normalization["derived_norm_relative_error"]
        < normalization["locked_norm_relative_error"],
        "normalization_output_bound": normalization["normalized_output_error"]
        < NORMALIZATION_OUTPUT_ERROR_UPPER,
        "initial_boundary_error_bound": normalization["initial_total_error"]
        < INITIAL_BOUNDARY_ERROR_UPPER,
        "initial_frobenius_enters_invariant": normalization["initial_frobenius_bound"]
        < selected.frobenius_bound,
        "initial_spectral_enters_invariant": normalization["initial_spectral_bound"]
        < selected.spectral_bound,
        "gamma4_identity": recurrence["gamma4"] == Fraction(1, 4_194_303),
        "zmm_identity": recurrence["zmm"]
        == 8 * FP32_HALF_MIN_SUBNORMAL / (1 - 4 * FP32_UNIT_ROUNDOFF),
        "zelt_identity": recurrence["zelt"] == 2 * FP32_HALF_MIN_SUBNORMAL,
        "gram_frobenius_identity": recurrence["gram_exact_frobenius"] == Fraction(35, 16),
        "gram_spectral_identity": recurrence["gram_exact_spectral"] == Fraction(25, 16),
        "t_spectral_identity": recurrence["t_exact_spectral"] == Fraction(191, 40),
        "t_frobenius_identity": recurrence["t_exact_frobenius"]
        == SQRT_TWO_UPPER * Fraction(191, 40),
        "d_spectral_identity": recurrence["d_exact_spectral"] == Fraction(182_405, 65_008),
        "d_frobenius_identity": recurrence["d_exact_frobenius"]
        == SQRT_TWO_UPPER * Fraction(182_405, 65_008),
        "e_spectral_identity": recurrence["e_exact_spectral"] == Fraction(6_889, 2_000),
        "e_frobenius_identity": recurrence["e_exact_frobenius"]
        == SQRT_TWO_UPPER * Fraction(6_889, 2_000),
        "gram_error_bound": recurrence["epsilon_gram"] < LOCAL_FP32_STAGE_ERROR_UPPER,
        "c_error_bound": recurrence["epsilon_c"] < LOCAL_FP32_STAGE_ERROR_UPPER,
        "t_error_bound": recurrence["epsilon_t"] < LOCAL_FP32_STAGE_ERROR_UPPER,
        "d_error_bound": recurrence["epsilon_d"] < LOCAL_FP32_STAGE_ERROR_UPPER,
        "e_error_bound": recurrence["epsilon_e"] < LOCAL_FP32_STAGE_ERROR_UPPER,
        "y_error_bound": recurrence["epsilon_y"] < LOCAL_FP32_STAGE_ERROR_UPPER,
        "pre_bf16_frobenius_exact": recurrence["pre_bf16_frobenius"] == Fraction(239_587, 140_000),
        "pre_bf16_spectral_exact": recurrence["pre_bf16_spectral"] == Fraction(24_201, 20_000),
        "frobenius_invariant_closes": recurrence["next_frobenius"] < selected.frobenius_bound,
        "spectral_invariant_closes": recurrence["next_spectral"] < selected.spectral_bound,
        "polynomial_error_identity": POLYNOMIAL_ERROR_UPPER
        == FROBENIUS_INVARIANT + POLYNOMIAL_UPPER * SQRT_TWO_UPPER,
        "repair_slope_bound": repair["complete_slope"] < selected.affine_slope_upper,
        "repair_intercept_bound": repair["complete_intercept"] < selected.affine_intercept_upper,
        "ideal_repaired_lipschitz_identity": IDEAL_REPAIRED_LIPSCHITZ
        == EXACT_RHO + IDEAL_POLYNOMIAL_LIPSCHITZ,
        "real_adapter_slope_bound": repair["real_adapter_slope"] < REAL_ADAPTER_SLOPE_UPPER,
        "real_adapter_intercept_bound": repair["real_adapter_intercept"]
        < REAL_ADAPTER_INTERCEPT_UPPER,
        "p7_kappa_reconstruction": p7_closure["derived_signal_kappa"] == P7_SIGNAL_KAPPA,
        "p7_kappa_scaling": P7_SIGNAL_STORAGE_GAIN == P7_SMOOTHNESS**2 * P7_SIGNAL_KAPPA,
        "p8_rate_identity": p7_closure["rate"] == P8_RATE,
        "p8_rate_is_strict": p7_closure["rate"] < 1,
        "p8_forcing_identity": p7_closure["forcing"] == P8_FORCING,
        "p8_function_gap_identity": p7_closure["function_gap_ultimate"] == P8_FUNCTION_GAP_ULTIMATE,
        "p8_safe_storage_radius_identity": p7_closure["safe_storage_radius"]
        == P8_SAFE_STORAGE_RADIUS,
        "p8_safe_forcing_capacity_identity": p7_closure["safe_forcing_capacity"]
        == P8_SAFE_FORCING_CAPACITY,
        "p8_safe_range_is_invariant": p7_closure["forcing"] <= p7_closure["safe_forcing_capacity"],
    }
    return MixedPrecisionAudit(
        sturm_chain=rebuilt_chain,
        sturm_endpoint_values=endpoint_values,
        sturm_endpoint_signs=endpoint_signs,
        sturm_variations=variations,
        polynomial_discriminant=discriminant,
        normalization=normalization,
        recurrence=recurrence,
        repair=repair,
        p7_closure=p7_closure,
        checks=checks,
    )
