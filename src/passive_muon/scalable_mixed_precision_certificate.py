"""Exact shape recurrences for the proposed P9 mixed-precision design.

This module is deliberately arithmetic-facing.  It does not claim that an
existing BLAS or tensor-core kernel implements the reductions below.  The
positive certificate assumes:

* a transpose-once orientation ``p x d`` with ``p <= d``;
* a scale-free max-floor normalizer with balanced pairwise FP32 reductions;
* balanced pairwise FP32 dot products in every Jordan stage; and
* a compensated two-term BF16 stage state.

The two-term state is formed entrywise as::

    high = RN_bf16(y)
    residual = fl32(y - fp32(high))
    low = RN_bf16(residual)
    recovered = fl32(fp32(high) + fp32(low))

No use of Sterbenz's lemma is required.  Rounding in both FP32 operations is
included explicitly, as are FP32 and BF16 subnormal crumbs.  All theorem
calculations use :class:`fractions.Fraction`; floating point is used only by
display helpers.

Three logically different conclusions must remain separate:

1. Serial FP32 norm accumulation has an explicit all-ones obstruction once
   the number of entries exceeds ``2**24``.
2. The ordinary one-term BF16 boundary loses the generic spectral tube at
   rank 72.  That is a failure of the normwise proof, not an impossibility
   theorem for every one-term implementation.
3. The compensated, pairwise design below closes an exact recurrence for the
   representative Transformer shapes returned by :func:`representative_audits`.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isqrt

# Exact polynomial and repair data inherited from the theorem-facing model.
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

LOCKED_STAGES = 5
LOCKED_SAFE_MAX_ABS = Fraction(2**116)
LOCKED_MAX_ENTRIES = 2**52
FP32_MAX_FINITE = Fraction((2**24 - 1) * 2**104)
BF16_MAX_FINITE = Fraction(255 * 2**120)

# Propagating unreduced exact rational expressions through five polynomial
# stages produces denominators with millions of bits.  The theorem therefore
# includes this deterministic upward rationalization after every named upper
# bound.  It is part of the certificate, not a display-only approximation.
UPPER_GRID_DENOMINATOR = 2**40

SPECTRAL_TUBE = Fraction(5, 4)
POLYNOMIAL_SPECTRAL_UPPER = Fraction(121, 100)
POLYNOMIAL_LIPSCHITZ_ON_TUBE = Fraction(3_000_459, 512_000)
IDEAL_POLYNOMIAL_LIPSCHITZ = Fraction(4_848_763, 10_000)
IDEAL_REPAIRED_LIPSCHITZ = Fraction(336_372_400_608_849, 260_261_360_000)

# Rational supersets used by the pairwise-normalizer proof.  Each shape audit
# reconstructs the smaller derived quantities before accepting these locks.
NORMALIZER_DENOMINATOR_ERROR_UPPER = Fraction(1, 100_000)
NORMALIZER_OUTPUT_ERROR_UPPER = Fraction(1, 90_000)

P7_RATE = Fraction(399_960_001, 400_000_000)
P7_IMPLEMENTATION_ERROR_GAIN = Fraction(1, 2_000_000)
P7_SIGNAL_STORAGE_GAIN = Fraction(1_655_544_025, 2_600_084)
P7_SMOOTHNESS = Fraction(10)
P7_STORAGE_FUNCTION_LOWER = Fraction(13_533, 50_000)


def _ceil_sqrt(value: int) -> int:
    """Return ``ceil(sqrt(value))`` using integer arithmetic only."""

    if value < 0:
        raise ValueError("square-root input must be nonnegative")
    root = isqrt(value)
    return root if root * root == value else root + 1


def _ceil_log2(value: int) -> int:
    """Return ``ceil(log2(value))`` using integer arithmetic only."""

    if value <= 0:
        raise ValueError("logarithm input must be positive")
    return (value - 1).bit_length()


def _upper(value: Fraction) -> Fraction:
    """Round ``value`` upward to the locked dyadic certificate grid."""

    scaled_numerator = value.numerator * UPPER_GRID_DENOMINATOR
    quotient, remainder = divmod(scaled_numerator, value.denominator)
    if remainder:
        quotient += 1
    return Fraction(quotient, UPPER_GRID_DENOMINATOR)


def _lower(value: Fraction) -> Fraction:
    """Round ``value`` downward to the locked dyadic certificate grid."""

    scaled_numerator = value.numerator * UPPER_GRID_DENOMINATOR
    quotient = scaled_numerator // value.denominator
    return Fraction(quotient, UPPER_GRID_DENOMINATOR)


def _pairwise_path_length(length: int) -> int:
    """One product rounding plus the depth of a balanced addition tree."""

    return 1 + _ceil_log2(length)


def _pairwise_gamma(length: int) -> Fraction:
    """Relative dot-product factor for the locked balanced reduction."""

    operations = _pairwise_path_length(length)
    product = operations * FP32_UNIT_ROUNDOFF
    if product >= 1:
        raise ValueError("pairwise reduction depth is outside the gamma regime")
    return product / (1 - product)


def _pairwise_crumb(output_entries: int, length: int) -> Fraction:
    """Frobenius underflow crumb for a matrix of pairwise dot products.

    A length-``length`` dot has ``length`` products and at most ``length-1``
    additions.  We use ``2*length`` absolute crumbs and propagate each through
    at most one balanced-tree path.  ``ceil(sqrt(output_entries))`` converts the
    entrywise bound to a rational Frobenius bound.
    """

    operations = _pairwise_path_length(length)
    denominator = 1 - operations * FP32_UNIT_ROUNDOFF
    return _ceil_sqrt(output_entries) * 2 * length * FP32_HALF_MIN_SUBNORMAL / denominator


@dataclass(frozen=True)
class PolynomialTubeAudit:
    """Exact scalar facts supplementing P8's Sturm range certificate."""

    squared_endpoint: Fraction
    factor_discriminant: Fraction
    factor_correction_at_endpoint: Fraction
    derivative_vertex: Fraction
    derivative_at_zero: Fraction
    derivative_at_vertex: Fraction
    derivative_at_endpoint: Fraction
    checks: dict[str, bool]

    @property
    def certified(self) -> bool:
        return bool(self.checks) and all(self.checks.values())


def polynomial_tube_audit() -> PolynomialTubeAudit:
    """Verify every scalar envelope used beyond P8's ``q<121/100`` lemma.

    P8 separately proves ``0<=q(x)<121/100`` on ``[0,5/4]`` by an exact
    Sturm chain.  Here ``y=x**2`` reduces the remaining claims to quadratics.
    In particular, ``q(x)/x=a+b*y+c*y**2`` is positive and at most ``a``,
    while the largest absolute derivative on the interval is the right
    endpoint value ``3000459/512000``.
    """

    squared_endpoint = SPECTRAL_TUBE**2
    factor_discriminant = EXACT_B**2 - 4 * EXACT_A * EXACT_C
    factor_correction_at_endpoint = EXACT_B + EXACT_C * squared_endpoint
    derivative_vertex = -3 * EXACT_B / (10 * EXACT_C)

    def derivative(squared_value: Fraction) -> Fraction:
        return EXACT_A + 3 * EXACT_B * squared_value + 5 * EXACT_C * squared_value**2

    derivative_at_zero = derivative(Fraction(0))
    derivative_at_vertex = derivative(derivative_vertex)
    derivative_at_endpoint = derivative(squared_endpoint)
    checks = {
        "factor_has_no_real_root": factor_discriminant < 0,
        "factor_is_positive": EXACT_A > 0 and EXACT_C > 0 and factor_discriminant < 0,
        "factor_is_at_most_a_on_tube": factor_correction_at_endpoint < 0,
        "derivative_vertex_is_in_interval": 0 < derivative_vertex < squared_endpoint,
        "derivative_endpoint_matches_lock": derivative_at_endpoint == POLYNOMIAL_LIPSCHITZ_ON_TUBE,
        "derivative_zero_below_lock": abs(derivative_at_zero) < POLYNOMIAL_LIPSCHITZ_ON_TUBE,
        "derivative_vertex_below_lock": abs(derivative_at_vertex) < POLYNOMIAL_LIPSCHITZ_ON_TUBE,
    }
    return PolynomialTubeAudit(
        squared_endpoint=squared_endpoint,
        factor_discriminant=factor_discriminant,
        factor_correction_at_endpoint=factor_correction_at_endpoint,
        derivative_vertex=derivative_vertex,
        derivative_at_zero=derivative_at_zero,
        derivative_at_vertex=derivative_at_vertex,
        derivative_at_endpoint=derivative_at_endpoint,
        checks=checks,
    )


@dataclass(frozen=True, order=True)
class MatrixShape:
    """Public and transpose-once dimensions for one fixed matrix shape."""

    rows: int
    columns: int

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in (self.rows, self.columns)
        ):
            raise ValueError("matrix dimensions must be positive integers")
        if self.rows * self.columns > LOCKED_MAX_ENTRIES:
            raise ValueError("the scalable certificate is locked to at most 2**52 entries")

    @property
    def short(self) -> int:
        return min(self.rows, self.columns)

    @property
    def long(self) -> int:
        return max(self.rows, self.columns)

    @property
    def entries(self) -> int:
        return self.rows * self.columns

    @property
    def oriented(self) -> tuple[int, int]:
        return self.short, self.long


@dataclass(frozen=True)
class TwoTermBF16Bound:
    """One compensated BF16 boundary bound ``omega*||Y||F + nu*chi``."""

    residual_rounding_slope: Fraction
    low_rounding_slope: Fraction
    reconstruction_rounding_slope: Fraction
    residual_rounding_crumb: Fraction
    rounded_residual_crumb: Fraction
    low_rounding_crumb: Fraction
    high_plus_low_crumb: Fraction
    reconstruction_rounding_crumb: Fraction
    omega: Fraction
    chi: Fraction
    sterbenz_free: bool = True

    def frobenius_error(self, shape: MatrixShape, input_frobenius: Fraction) -> Fraction:
        if input_frobenius < 0:
            raise ValueError("Frobenius envelope must be nonnegative")
        return self.omega * input_frobenius + _ceil_sqrt(shape.entries) * self.chi


def two_term_bf16_bound() -> TwoTermBF16Bound:
    """Derive the compensated boundary bound without assuming exact subtraction.

    For one entry ``y``, write ``e_h = high-y`` and ``r=y-high``.  The four
    nontrivial errors are bounded in this order:

    ``|e_h| <= ub*|y| + tb``;
    ``|fl32(r)-r| <= u32*|e_h| + t32``;
    ``|low-fl32(r)| <= ub*|fl32(r)| + tb``; and
    ``|fl32(high+low)-(high+low)| <= u32*(|high|+|low|)+t32``.

    Adding the last three errors cancels ``e_h`` algebraically.  The result is
    valid near underflow and does not invoke Sterbenz or exact reconstruction.
    """

    u = FP32_UNIT_ROUNDOFF
    tau = FP32_HALF_MIN_SUBNORMAL
    ub = BF16_UNIT_ROUNDOFF
    taub = BF16_HALF_MIN_SUBNORMAL

    # Slopes multiplying |y|.
    residual_rounding_slope = u * ub
    low_rounding_slope = ub * (1 + u) * ub
    high_plus_low_slope = (1 + ub) * (1 + (1 + u) * ub)
    reconstruction_rounding_slope = u * high_plus_low_slope
    omega = residual_rounding_slope + low_rounding_slope + reconstruction_rounding_slope

    # Absolute crumbs in the same order.  Keeping the expressions separated
    # makes an independent reconstruction easier than a pre-expanded dyadic.
    residual_rounding_crumb = u * taub + tau
    rounded_residual_crumb = (1 + u) * taub + tau
    low_rounding_crumb = ub * rounded_residual_crumb + taub
    high_plus_low_crumb = taub + (1 + ub) * rounded_residual_crumb + taub
    reconstruction_rounding_crumb = u * high_plus_low_crumb + tau
    chi = residual_rounding_crumb + low_rounding_crumb + reconstruction_rounding_crumb
    return TwoTermBF16Bound(
        residual_rounding_slope=residual_rounding_slope,
        low_rounding_slope=low_rounding_slope,
        reconstruction_rounding_slope=reconstruction_rounding_slope,
        residual_rounding_crumb=residual_rounding_crumb,
        rounded_residual_crumb=rounded_residual_crumb,
        low_rounding_crumb=low_rounding_crumb,
        high_plus_low_crumb=high_plus_low_crumb,
        reconstruction_rounding_crumb=reconstruction_rounding_crumb,
        omega=omega,
        chi=chi,
    )


@dataclass(frozen=True)
class NormalizerAudit:
    """Exact bounds for the scale-free pairwise FP32 max-floor normalizer."""

    gamma: Fraction
    sum_crumb: Fraction
    relative_vector_error: Fraction
    norm_branch_squared_error: Fraction
    norm_branch_denominator_error: Fraction
    floor_branch_squared_error: Fraction
    floor_branch_denominator_error: Fraction
    small_floor_sum_upper: Fraction
    output_error: Fraction
    checks: dict[str, bool]

    @property
    def certified(self) -> bool:
        return bool(self.checks) and all(self.checks.values())


def _normalizer_audit(shape: MatrixShape) -> NormalizerAudit:
    """Audit the pairwise, scale-free normalizer for one shape.

    Put ``sigma=max(1,maxabs(s))`` and ``z=s/sigma``.  In exact arithmetic,

    ``N(s)=z/max(1/sigma, ||z||F)``.

    The denominator is always at least one: when ``sigma>1``, one component of
    ``z`` has magnitude one; when ``sigma=1``, the floor term is one.  This is
    what prevents the subnormal input guard from being multiplied by
    ``2**116`` in the error analysis.
    """

    u = FP32_UNIT_ROUNDOFF
    tau = FP32_HALF_MIN_SUBNORMAL
    count = shape.entries
    root_count = _ceil_sqrt(count)
    gamma = _upper(_pairwise_gamma(count))
    sum_crumb = _upper(2 * count * tau / (1 - _pairwise_path_length(count) * u))

    # ||zhat-z||/d, where d=max(1/sigma,||z||F)>=1.
    relative_vector_error = _upper(u + root_count * tau)

    # If the norm term controls the exact denominator, ||z||F >= 1.  This is a
    # relative squared-norm error with all ratio and pairwise-dot errors.
    norm_branch_squared_error = _upper(
        (2 + relative_vector_error) * relative_vector_error
        + gamma * (1 + relative_vector_error) ** 2
        + sum_crumb
    )
    norm_branch_denominator_error = _upper(
        (1 + u) * (1 + norm_branch_squared_error / (2 - norm_branch_squared_error)) - 1 + tau
    )

    # On the floor branch sigma=1.  A computed norm can cross the floor only
    # when the exact norm is at least 1/2.  Dividing by n^2 then costs at most a
    # factor four; the vector crumb likewise costs at most a factor two.
    floor_vector_error = _upper(u + 2 * root_count * tau)
    floor_branch_squared_error = _upper(
        (2 + floor_vector_error) * floor_vector_error
        + gamma * (1 + floor_vector_error) ** 2
        + 4 * sum_crumb
    )
    floor_branch_denominator_error = _upper(
        (1 + u) * (1 + floor_branch_squared_error / (2 - floor_branch_squared_error)) - 1 + 2 * tau
    )

    # If ||z||F < 1/2, this squared upper bound proves that the rounded square
    # root remains below one, so the exact FP32 floor value is selected.
    small_floor_sum_upper = _upper(
        (1 + gamma) * (Fraction(1, 2) * (1 + u) + root_count * tau) ** 2 + sum_crumb
    )

    delta = NORMALIZER_DENOMINATOR_ERROR_UPPER
    output_error = _upper(
        (relative_vector_error + delta + u * (1 + relative_vector_error)) / (1 - delta)
        + root_count * tau
    )

    # A broad overflow envelope for the pairwise sum.  It is intentionally
    # independent of branch-specific cancellations.
    scaled_vector_upper = _upper(root_count * (1 + u + tau))
    pairwise_sum_upper = _upper((1 + gamma) * scaled_vector_upper**2 + sum_crumb)

    checks = {
        "pairwise_gamma_defined": _pairwise_path_length(count) * u < 1,
        "norm_branch_squared_error_below_one": norm_branch_squared_error < 1,
        "norm_branch_denominator_bound": norm_branch_denominator_error < delta,
        "floor_branch_squared_error_below_one": floor_branch_squared_error < 1,
        "floor_branch_denominator_bound": floor_branch_denominator_error < delta,
        "small_floor_cannot_cross_one": small_floor_sum_upper < ((1 - tau) / (1 + u)) ** 2,
        "normalizer_output_bound": output_error < NORMALIZER_OUTPUT_ERROR_UPPER,
        "pairwise_norm_sum_finite": pairwise_sum_upper < FP32_MAX_FINITE,
    }
    return NormalizerAudit(
        gamma=gamma,
        sum_crumb=sum_crumb,
        relative_vector_error=relative_vector_error,
        norm_branch_squared_error=norm_branch_squared_error,
        norm_branch_denominator_error=norm_branch_denominator_error,
        floor_branch_squared_error=floor_branch_squared_error,
        floor_branch_denominator_error=floor_branch_denominator_error,
        small_floor_sum_upper=small_floor_sum_upper,
        output_error=output_error,
        checks=checks,
    )


@dataclass(frozen=True)
class StageEnvelope:
    """One exact compensated-stage recurrence record."""

    index: int
    input_spectral: Fraction
    input_frobenius: Fraction
    gram_error: Fraction
    scaled_gram_error: Fraction
    shifted_gram_error: Fraction
    product_error: Fraction
    affine_error: Fraction
    output_error_before_boundary: Fraction
    pre_boundary_spectral: Fraction
    pre_boundary_frobenius: Fraction
    boundary_error: Fraction
    next_spectral: Fraction
    next_frobenius: Fraction
    forward_error: Fraction
    intermediate_max_frobenius: Fraction


def _stage_envelope(
    shape: MatrixShape,
    *,
    index: int,
    spectral: Fraction,
    frobenius: Fraction,
    previous_forward_error: Fraction,
    boundary: TwoTermBF16Bound,
) -> StageEnvelope:
    """Apply the exact general-shape FP32/BF16 recurrence once."""

    p, d = shape.oriented
    root_p = _ceil_sqrt(p)
    u = FP32_UNIT_ROUNDOFF
    tau = FP32_HALF_MIN_SUBNORMAL
    gamma_d = _pairwise_gamma(d)
    gamma_p = _pairwise_gamma(p)

    gram_error = _upper(gamma_d * frobenius**2 + _pairwise_crumb(p * p, d))
    gram_frobenius = _upper(spectral * frobenius + gram_error)
    gram_spectral = _upper(spectral**2 + gram_error)

    scaled_gram_error = _upper(
        abs(FP32_C) * gram_error
        + abs(FP32_C - EXACT_C) * spectral * frobenius
        + u * abs(FP32_C) * gram_frobenius
        + p * tau
    )
    scaled_gram_frobenius = _upper(abs(EXACT_C) * spectral * frobenius + scaled_gram_error)

    shifted_gram_error = _upper(
        scaled_gram_error
        + abs(FP32_B - EXACT_B) * root_p
        + u * (scaled_gram_frobenius + abs(FP32_B) * root_p)
        + p * tau
    )
    shifted_gram_frobenius = _upper(root_p * abs(EXACT_B) + shifted_gram_error)

    exact_product_spectral = EXACT_B**2 / (4 * EXACT_C)
    product_error = _upper(
        shifted_gram_error * gram_spectral
        + abs(EXACT_B) * gram_error
        + gamma_p * shifted_gram_frobenius * gram_frobenius
        + _pairwise_crumb(p * p, p)
    )
    product_frobenius = _upper(root_p * exact_product_spectral + product_error)

    affine_error = _upper(
        product_error
        + abs(FP32_A - EXACT_A) * root_p
        + u * (product_frobenius + abs(FP32_A) * root_p)
        + p * tau
    )
    affine_frobenius = _upper(root_p * EXACT_A + affine_error)

    output_error = _upper(
        affine_error * spectral
        + gamma_p * affine_frobenius * frobenius
        + _pairwise_crumb(shape.entries, p)
    )
    pre_boundary_spectral = _upper(POLYNOMIAL_SPECTRAL_UPPER + output_error)
    pre_boundary_frobenius = _upper(EXACT_A * frobenius + output_error)
    boundary_error = _upper(boundary.frobenius_error(shape, pre_boundary_frobenius))
    next_spectral = _upper(pre_boundary_spectral + boundary_error)
    next_frobenius = _upper(pre_boundary_frobenius + boundary_error)
    forward_error = _upper(
        POLYNOMIAL_LIPSCHITZ_ON_TUBE * previous_forward_error + output_error + boundary_error
    )
    # Each pairwise partial sum is bounded by its corresponding absolute dot
    # sum.  Include those pre-rounding envelopes, not just the completed
    # matrices, in the overflow check.
    intermediate_max_frobenius = max(
        frobenius**2,
        gram_frobenius,
        abs(FP32_C) * gram_frobenius,
        scaled_gram_frobenius,
        scaled_gram_frobenius + abs(FP32_B) * root_p,
        shifted_gram_frobenius,
        shifted_gram_frobenius * gram_frobenius,
        product_frobenius,
        product_frobenius + abs(FP32_A) * root_p,
        affine_frobenius,
        affine_frobenius * frobenius,
        pre_boundary_frobenius,
        next_frobenius,
    )
    return StageEnvelope(
        index=index,
        input_spectral=spectral,
        input_frobenius=frobenius,
        gram_error=gram_error,
        scaled_gram_error=scaled_gram_error,
        shifted_gram_error=shifted_gram_error,
        product_error=product_error,
        affine_error=affine_error,
        output_error_before_boundary=output_error,
        pre_boundary_spectral=pre_boundary_spectral,
        pre_boundary_frobenius=pre_boundary_frobenius,
        boundary_error=boundary_error,
        next_spectral=next_spectral,
        next_frobenius=next_frobenius,
        forward_error=forward_error,
        intermediate_max_frobenius=intermediate_max_frobenius,
    )


@dataclass(frozen=True)
class OperatorBound:
    """Shape-parameterized all-real affine operator-error certificate."""

    binary32_slope: Fraction
    binary32_intercept: Fraction
    real_slope: Fraction
    real_intercept: Fraction


def _operator_bound(shape: MatrixShape, polynomial_error: Fraction) -> OperatorBound:
    u = FP32_UNIT_ROUNDOFF
    tau = FP32_HALF_MIN_SUBNORMAL
    root_count = _ceil_sqrt(shape.entries)
    first_slope = abs(FP32_RHO - EXACT_RHO) + u * abs(FP32_RHO)
    binary32_slope = _upper(first_slope + u * (EXACT_RHO + first_slope))
    ideal_output_frobenius = EXACT_A**LOCKED_STAGES
    binary32_intercept = _upper(
        (1 + u) * polynomial_error + u * ideal_output_frobenius + (2 + u) * root_count * tau
    )
    real_slope = _upper(binary32_slope * (1 + u) + IDEAL_REPAIRED_LIPSCHITZ * u)
    real_intercept = _upper(
        binary32_intercept + (binary32_slope + IDEAL_REPAIRED_LIPSCHITZ) * root_count * tau
    )
    return OperatorBound(
        binary32_slope=binary32_slope,
        binary32_intercept=binary32_intercept,
        real_slope=real_slope,
        real_intercept=real_intercept,
    )


@dataclass(frozen=True)
class P7Closure:
    """Exact P7 rate, forcing, and function-gap neighborhood."""

    young_theta: int
    rate: Fraction
    forcing: Fraction
    function_gap_ultimate: Fraction
    safe_storage_radius: Fraction
    safe_forcing_capacity: Fraction


def _p7_closure(bound: OperatorBound, *, young_theta: int) -> P7Closure:
    if young_theta <= 0:
        raise ValueError("Young parameter must be positive")
    theta = Fraction(young_theta)
    rate = _upper(
        P7_RATE
        + P7_IMPLEMENTATION_ERROR_GAIN * (1 + theta) * P7_SIGNAL_STORAGE_GAIN * bound.real_slope**2
    )
    forcing = _upper(P7_IMPLEMENTATION_ERROR_GAIN * (1 + 1 / theta) * bound.real_intercept**2)
    function_gap_ultimate = _upper(P7_SMOOTHNESS / P7_STORAGE_FUNCTION_LOWER * forcing / (1 - rate))
    safe_storage_radius = LOCKED_SAFE_MAX_ABS**2 / P7_SIGNAL_STORAGE_GAIN
    safe_forcing_capacity = _lower((1 - rate) * safe_storage_radius)
    return P7Closure(
        young_theta=young_theta,
        rate=rate,
        forcing=forcing,
        function_gap_ultimate=function_gap_ultimate,
        safe_storage_radius=safe_storage_radius,
        safe_forcing_capacity=safe_forcing_capacity,
    )


@dataclass(frozen=True)
class ScalableMixedPrecisionAudit:
    """Complete exact audit for one fixed public matrix shape."""

    shape: MatrixShape
    polynomial: PolynomialTubeAudit
    normalizer: NormalizerAudit
    boundary: TwoTermBF16Bound
    initial_boundary_error: Fraction
    stages: tuple[StageEnvelope, ...]
    operator: OperatorBound
    p7: P7Closure
    checks: dict[str, bool]

    @property
    def certified(self) -> bool:
        return self.normalizer.certified and bool(self.checks) and all(self.checks.values())


def audit_scalable_shape(
    shape: MatrixShape | tuple[int, int], *, young_theta: int = 3_006
) -> ScalableMixedPrecisionAudit:
    """Rebuild the complete exact P9 recurrence for one matrix shape."""

    selected = shape if isinstance(shape, MatrixShape) else MatrixShape(*shape)
    polynomial = polynomial_tube_audit()
    normalizer = _normalizer_audit(selected)
    boundary = two_term_bf16_bound()

    normalized_frobenius = 1 + NORMALIZER_OUTPUT_ERROR_UPPER
    initial_boundary_error = _upper(boundary.frobenius_error(selected, normalized_frobenius))
    spectral = _upper(normalized_frobenius + initial_boundary_error)
    frobenius = _upper(normalized_frobenius + initial_boundary_error)
    forward_error = _upper(NORMALIZER_OUTPUT_ERROR_UPPER + initial_boundary_error)

    stages: list[StageEnvelope] = []
    for index in range(1, LOCKED_STAGES + 1):
        stage = _stage_envelope(
            selected,
            index=index,
            spectral=spectral,
            frobenius=frobenius,
            previous_forward_error=forward_error,
            boundary=boundary,
        )
        stages.append(stage)
        spectral = stage.next_spectral
        frobenius = stage.next_frobenius
        forward_error = stage.forward_error

    operator = _operator_bound(selected, forward_error)
    p7 = _p7_closure(operator, young_theta=young_theta)

    # Only inputs to a subsequent stage need the spectral tube.  The fifth
    # returned boundary may lie outside it without invalidating any use of q.
    stage_input_tube_checks = {
        f"stage_{stage.index}_input_in_spectral_tube": stage.input_spectral < SPECTRAL_TUBE
        for stage in stages
    }
    intermediate_checks = {
        f"stage_{stage.index}_intermediates_finite": stage.intermediate_max_frobenius
        < FP32_MAX_FINITE / 2
        for stage in stages
    }
    bf16_boundary_checks = {
        f"stage_{stage.index}_bf16_boundary_finite": stage.pre_boundary_frobenius
        < BF16_MAX_FINITE / 2
        for stage in stages
    }
    repair_output_upper = (
        (1 + FP32_UNIT_ROUNDOFF) * abs(FP32_RHO) * LOCKED_SAFE_MAX_ABS
        + FP32_HALF_MIN_SUBNORMAL
        + stages[-1].next_frobenius
    )
    exact_rate_margin = 1 - P7_RATE
    exact_rate_coefficient = (
        P7_IMPLEMENTATION_ERROR_GAIN * P7_SIGNAL_STORAGE_GAIN * operator.real_slope**2
    )
    checks = {
        **stage_input_tube_checks,
        **intermediate_checks,
        **bf16_boundary_checks,
        "polynomial_tube_facts_are_certified": polynomial.certified,
        "normalizer_is_certified": normalizer.certified,
        "two_term_bound_is_sterbenz_free": boundary.sterbenz_free,
        "two_term_slope_components_reconstruct": boundary.omega
        == boundary.residual_rounding_slope
        + boundary.low_rounding_slope
        + boundary.reconstruction_rounding_slope,
        "two_term_crumb_components_reconstruct": boundary.chi
        == boundary.residual_rounding_crumb
        + boundary.low_rounding_crumb
        + boundary.reconstruction_rounding_crumb,
        "two_term_slope_improves_one_term": boundary.omega < BF16_UNIT_ROUNDOFF,
        "repair_shell_finite": repair_output_upper < FP32_MAX_FINITE / 2,
        "p7_affine_slope_gate": operator.real_slope**2
        < exact_rate_margin / (P7_IMPLEMENTATION_ERROR_GAIN * P7_SIGNAL_STORAGE_GAIN),
        "selected_young_parameter_keeps_rate_strict": p7.rate < 1,
        "finite_guard_is_forward_invariant": p7.forcing <= p7.safe_forcing_capacity,
        "continuous_optimum_exists": exact_rate_coefficient < exact_rate_margin,
    }
    return ScalableMixedPrecisionAudit(
        shape=selected,
        polynomial=polynomial,
        normalizer=normalizer,
        boundary=boundary,
        initial_boundary_error=initial_boundary_error,
        stages=tuple(stages),
        operator=operator,
        p7=p7,
        checks=checks,
    )


# These are fixed theorem targets, not sampled shapes.  They cover square
# attention blocks, fused projections, MLP matrices, and one wide vocabulary-
# scale matrix while keeping the short dimension at most 4096.
REPRESENTATIVE_TRANSFORMER_SHAPES: tuple[MatrixShape, ...] = (
    MatrixShape(768, 768),
    MatrixShape(768, 3_072),
    MatrixShape(768, 50_257),
    MatrixShape(3_072, 12_288),
    MatrixShape(4_096, 4_096),
    MatrixShape(4_096, 11_008),
    MatrixShape(4_096, 14_336),
)


def representative_audits(*, young_theta: int = 3_006) -> tuple[ScalableMixedPrecisionAudit, ...]:
    """Return exact audits for the locked representative shape table."""

    return tuple(
        audit_scalable_shape(shape, young_theta=young_theta)
        for shape in REPRESENTATIVE_TRANSFORMER_SHAPES
    )


def one_term_boundary_rank_limit() -> int:
    """Return the exact rank limit of the generic one-term BF16 tube bound."""

    ratio = (SPECTRAL_TUBE - POLYNOMIAL_SPECTRAL_UPPER) / (
        BF16_UNIT_ROUNDOFF * POLYNOMIAL_SPECTRAL_UPPER
    )
    squared = ratio**2
    return squared.numerator // squared.denominator


def ideal_two_term_boundary_rank_limit() -> int:
    """Boundary-only comparison if subtraction/reconstruction were exact.

    The positive certificate does not use this sharper value; it uses
    :func:`two_term_bf16_bound`.  This helper exists only to quantify why a
    compensated boundary removes the rank-72 proof obstruction.
    """

    ratio = (SPECTRAL_TUBE - POLYNOMIAL_SPECTRAL_UPPER) / (
        BF16_UNIT_ROUNDOFF**2 * POLYNOMIAL_SPECTRAL_UPPER
    )
    squared = ratio**2
    return squared.numerator // squared.denominator


def compensated_boundary_rank_limit() -> int:
    """Boundary-only rank gate using the theorem's Sterbenz-free slope."""

    ratio = (SPECTRAL_TUBE - POLYNOMIAL_SPECTRAL_UPPER) / (
        two_term_bf16_bound().omega * POLYNOMIAL_SPECTRAL_UPPER
    )
    squared = ratio**2
    return squared.numerator // squared.denominator


@dataclass(frozen=True)
class SerialNormObstruction:
    """Exact all-ones witness for serial FP32 norm accumulation."""

    shape: MatrixShape
    exact_square_sum: int
    computed_square_sum: int
    returned_singular_value_squared: Fraction
    leaves_spectral_tube: bool


def serial_norm_obstruction(shape: MatrixShape | tuple[int, int]) -> SerialNormObstruction:
    """Return the exact all-ones serial-sum obstruction for ``entries>2**24``.

    Starting from one, sequential FP32 additions of one are exact through
    ``2**24``.  Thereafter ``2**24+1`` is a halfway case and ties-to-even
    returns ``2**24`` again, so the accumulator remains stuck.  The returned
    normalized all-ones matrix therefore has squared singular value
    ``entries/2**24``.
    """

    selected = shape if isinstance(shape, MatrixShape) else MatrixShape(*shape)
    if selected.entries <= 2**24:
        raise ValueError("the serial all-ones obstruction requires more than 2**24 entries")
    singular_squared = Fraction(selected.entries, 2**24)
    return SerialNormObstruction(
        shape=selected,
        exact_square_sum=selected.entries,
        computed_square_sum=2**24,
        returned_singular_value_squared=singular_squared,
        leaves_spectral_tube=singular_squared > SPECTRAL_TUBE**2,
    )


def summary_rows(*, young_theta: int = 3_006) -> tuple[dict[str, object], ...]:
    """Return compact, JSON-ready display rows for the representative table."""

    rows: list[dict[str, object]] = []
    for audit in representative_audits(young_theta=young_theta):
        rows.append(
            {
                "shape": [audit.shape.rows, audit.shape.columns],
                "oriented_shape": list(audit.shape.oriented),
                "certified": audit.certified,
                "stage_4_output_spectral_bound": str(audit.stages[3].next_spectral),
                "stage_4_output_spectral_bound_decimal": float(audit.stages[3].next_spectral),
                "operator_slope": str(audit.operator.real_slope),
                "operator_slope_decimal": float(audit.operator.real_slope),
                "operator_intercept": str(audit.operator.real_intercept),
                "operator_intercept_decimal": float(audit.operator.real_intercept),
                "p7_rate": str(audit.p7.rate),
                "p7_rate_decimal": float(audit.p7.rate),
                "function_gap_ultimate": str(audit.p7.function_gap_ultimate),
                "function_gap_ultimate_decimal": float(audit.p7.function_gap_ultimate),
            }
        )
    return tuple(rows)


__all__ = [
    "REPRESENTATIVE_TRANSFORMER_SHAPES",
    "UPPER_GRID_DENOMINATOR",
    "MatrixShape",
    "NormalizerAudit",
    "OperatorBound",
    "P7Closure",
    "PolynomialTubeAudit",
    "ScalableMixedPrecisionAudit",
    "SerialNormObstruction",
    "StageEnvelope",
    "TwoTermBF16Bound",
    "audit_scalable_shape",
    "compensated_boundary_rank_limit",
    "ideal_two_term_boundary_rank_limit",
    "one_term_boundary_rank_limit",
    "polynomial_tube_audit",
    "representative_audits",
    "serial_norm_obstruction",
    "summary_rows",
    "two_term_bf16_bound",
]
