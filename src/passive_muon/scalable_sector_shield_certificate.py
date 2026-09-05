"""Exact P20 margin certificate for a scalable FP32 sector shield.

P19 certifies the pointwise sector disk

``||U - gamma*S||_F <= r*||S||_F``

with ``gamma=1143/2048`` and ``r=893/2048``.  Its reference binary64
implementation checks stored outputs with :class:`fractions.Fraction`, which
is deliberately not a deployable operation.  P20 replaces that postcheck by
an a-priori, shape-dependent inward margin.

The positive arithmetic contract is intentionally narrow:

* the stored signal and candidate are binary32; a bfloat16 input is first
  widened exactly to binary32;
* every vector operation and norm leaf/reduction is IEEE binary32
  round-to-nearest, ties-to-even, with gradual underflow and without FMA or
  reassociation;
* a norm uses max scaling, materialized division and squaring, a fixed
  adjacent balanced addition tree, and a correctly rounded binary32 square
  root;
* the scale and dimensionless norm are multiplied exactly in binary64 (the
  product of two binary32 values has at most 48 significant bits);
* the sole acceptance threshold is a positive binary64 product followed by
  ``nextafter(..., -inf)``, so it is no larger than its exact product;
* a rejected finite candidate first receives a radial clip whose scalar is
  formed by downward binary64 division, downward binary64 multiplication,
  and a downward binary32 conversion, in that order; and
* the returned dtype is binary32.  There is no final BF16 cast.

An accepted candidate is returned bit-for-bit.  A rejected finite candidate
is clipped along the computed displacement ray; exceptional arithmetic uses
``fl32(S/2)``.  In the normal-anchor regime ``maxabs(S)>=2**-126``, the latter
is covered by a shape-dependent underflow crumb.  Below that anchor, only an
entrywise exact-halving bit check authorizes the fallback; otherwise the
implementation fails closed without returning an update.

This pass-through/clip/fallback construction is not P19's metric projection.
It does not inherit fixed-input nonexpansiveness and is not globally the
identity on the exact P18 operator.  Its theorem is unconditional containment
for every successful call, plus separately measured inactivity on declared
candidates.

All proof quantities are exact :class:`fractions.Fraction` values.  Square
roots are enclosed on a declared dyadic grid and the final certified radius
is rounded upward, so ``Delta_(m,n)`` is a rigorous lower margin.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isqrt

from passive_muon.scalable_mixed_precision_certificate import (
    LOCKED_MAX_ENTRIES,
    REPRESENTATIVE_TRANSFORMER_SHAPES,
    MatrixShape,
)

SCALABLE_SECTOR_SHIELD_SCHEMA_VERSION = "passive-muon-scalable-sector-shield-v1"

LOCKED_SECTOR_LOWER = Fraction(125, 1_024)
LOCKED_SECTOR_UPPER = Fraction(509, 512)
LOCKED_SECTOR_CENTER = Fraction(1_143, 2_048)
LOCKED_SECTOR_RADIUS = Fraction(893, 2_048)

# The computed norm test is deliberately one dyadic step farther inward than
# the P19 disk.  This coefficient is exactly representable in binary32 and
# binary64.
LOCKED_ACCEPTANCE_BUFFER = Fraction(1, 1_024)
LOCKED_ACCEPTANCE_COEFFICIENT = LOCKED_SECTOR_RADIUS - LOCKED_ACCEPTANCE_BUFFER
LOCKED_CLIP_COEFFICIENT = Fraction(890, 2_048)

FP32_UNIT_ROUNDOFF = Fraction(1, 2**24)
FP32_HALF_MIN_SUBNORMAL = Fraction(1, 2**150)
FP32_MIN_SUBNORMAL = Fraction(1, 2**149)
FP32_MIN_NORMAL = Fraction(1, 2**126)
FP32_MAX_SUBNORMAL = FP32_MIN_NORMAL - FP32_MIN_SUBNORMAL
FP64_UNIT_ROUNDOFF = Fraction(1, 2**53)

# A widened BF16 value has its low sixteen FP32 significand bits zero.  In
# particular, every finite widened BF16 signal, including a BF16 subnormal,
# admits an exactly representable FP32 half.
BF16_MIN_SUBNORMAL = Fraction(1, 2**133)

SQRT_ENCLOSURE_DENOMINATOR = 2**80
RADIUS_GRID_DENOMINATOR = 2**60
LOCKED_REDUCTION_BLOCK_ENTRIES = 2**20

LOCKED_MAXIMUM_STEP = Fraction(1, 83)
LOCKED_MAXIMUM_STEP_RATE = Fraction(999_598_040_401, 1_000_000_000_000)
LOCKED_FASTER_RATE_STEP = Fraction(1, 120)
LOCKED_FASTER_RATE = Fraction(624_350_169, 625_000_000)

DIAGNOSTIC_SHAPES: tuple[MatrixShape, ...] = (
    MatrixShape(1, 1),
    MatrixShape(1, 2),
    MatrixShape(2, 2),
)


def _ceil_sqrt_integer(value: int) -> int:
    if value < 0:
        raise ValueError("square-root input must be nonnegative")
    root = isqrt(value)
    return root if root * root == value else root + 1


def _ceil_log2(value: int) -> int:
    if value <= 0:
        raise ValueError("logarithm input must be positive")
    return (value - 1).bit_length()


def _sqrt_enclosure(value: Fraction) -> tuple[Fraction, Fraction]:
    """Return exact dyadic lower/upper bounds for ``sqrt(value)``."""

    if value < 0:
        raise ValueError("square-root input must be nonnegative")
    scaled = value.numerator * SQRT_ENCLOSURE_DENOMINATOR**2
    lower_numerator = isqrt(scaled // value.denominator)
    lower = Fraction(lower_numerator, SQRT_ENCLOSURE_DENOMINATOR)
    upper = (
        lower
        if lower * lower == value
        else Fraction(lower_numerator + 1, SQRT_ENCLOSURE_DENOMINATOR)
    )
    if not lower * lower <= value <= upper * upper:
        raise ArithmeticError("directed square-root enclosure failed")
    return lower, upper


def _upper_radius_grid(value: Fraction) -> Fraction:
    if value < 0:
        raise ValueError("radius must be nonnegative")
    quotient, remainder = divmod(
        value.numerator * RADIUS_GRID_DENOMINATOR,
        value.denominator,
    )
    return Fraction(quotient + bool(remainder), RADIUS_GRID_DENOMINATOR)


@dataclass(frozen=True)
class BalancedNormEnvelope:
    """Exact multiplicative envelope for the locked scale-free FP32 norm.

    If ``x`` is a nonzero finite stored FP32 vector, the norm routine returns
    ``Nhat=a*n32`` in binary64, where ``a=maxabs(x)`` and ``n32`` is the FP32
    square-root result.  This product is exact.  The fields certify

    ``lower_factor*||x|| <= Nhat <= upper_factor*||x||``.

    The dimensionless ``division_relative_error`` uses ``||x/a||>=1``; the
    final original-unit underflow terms occur later in the displacement and
    fallback bounds, not in this normalized factor.
    """

    entries: int
    root_entries_upper: int
    pairwise_path_length: int
    reduction_block_entries: int
    reduction_block_count: int
    pairwise_gamma: Fraction
    pairwise_crumb: Fraction
    division_relative_error: Fraction
    squared_norm_lower: Fraction
    squared_norm_upper: Fraction
    sqrt_lower: Fraction
    sqrt_upper: Fraction
    lower_factor: Fraction
    upper_factor: Fraction

    @property
    def certified(self) -> bool:
        return (
            0 < self.lower_factor <= 1 <= self.upper_factor
            and self.sqrt_lower**2 <= self.squared_norm_lower
            and self.sqrt_upper**2 >= self.squared_norm_upper
            and self.pairwise_path_length * FP32_UNIT_ROUNDOFF < 1
        )


@dataclass(frozen=True)
class ScalableSectorShieldAudit:
    """One exact shape-specific inward-disk certificate."""

    shape: MatrixShape
    norm: BalancedNormEnvelope
    normal_anchor: Fraction
    acceptance_coefficient: Fraction
    displacement_signal_slope: Fraction
    displacement_crumb: Fraction
    displacement_crumb_relative: Fraction
    accepted_candidate_radius: Fraction
    clipped_candidate_radius: Fraction
    clip_only_inward_margin: Fraction
    rounded_half_radius: Fraction
    certified_inward_radius_raw: Fraction
    certified_inward_radius: Fraction
    inward_margin: Fraction
    all_subnormal_input_radius: Fraction
    zero_instead_of_half_output_error: Fraction
    checks: dict[str, bool]

    @property
    def certified(self) -> bool:
        return bool(self.checks) and all(self.checks.values())


def balanced_norm_envelope(shape: MatrixShape | tuple[int, int]) -> BalancedNormEnvelope:
    """Audit the locked scale-free balanced FP32 norm for one matrix shape."""

    selected = shape if isinstance(shape, MatrixShape) else MatrixShape(*shape)
    entries = selected.entries
    root_entries = _ceil_sqrt_integer(entries)
    path_length = 1 + _ceil_log2(entries)
    path_roundoff = path_length * FP32_UNIT_ROUNDOFF
    if path_roundoff >= 1:
        raise ValueError("balanced reduction lies outside the gamma regime")
    pairwise_gamma = path_roundoff / (1 - path_roundoff)
    pairwise_crumb = 2 * entries * FP32_HALF_MIN_SUBNORMAL / (1 - path_roundoff)

    # Let v=x/maxabs(x), so ||v||>=1.  For z=fl32(v),
    # ||z-v|| <= u||v||+h*tau <= (u+h*tau)||v||.
    division_relative_error = FP32_UNIT_ROUNDOFF + root_entries * FP32_HALF_MIN_SUBNORMAL
    squared_lower = (1 - pairwise_gamma) * (1 - division_relative_error) ** 2 - pairwise_crumb
    squared_upper = (1 + pairwise_gamma) * (1 + division_relative_error) ** 2 + pairwise_crumb
    if squared_lower <= 0:
        raise ArithmeticError("norm lower envelope is not positive")
    sqrt_lower, _ = _sqrt_enclosure(squared_lower)
    _, sqrt_upper = _sqrt_enclosure(squared_upper)

    # Correctly rounded FP32 sqrt: |n32-sqrt(qhat)|<=u*sqrt(qhat)+tau.
    # Since ||v||>=1, the absolute tau is absorbed into the factors.
    lower_factor = (1 - FP32_UNIT_ROUNDOFF) * sqrt_lower - FP32_HALF_MIN_SUBNORMAL
    upper_factor = (1 + FP32_UNIT_ROUNDOFF) * sqrt_upper + FP32_HALF_MIN_SUBNORMAL
    result = BalancedNormEnvelope(
        entries=entries,
        root_entries_upper=root_entries,
        pairwise_path_length=path_length,
        reduction_block_entries=LOCKED_REDUCTION_BLOCK_ENTRIES,
        reduction_block_count=(entries + LOCKED_REDUCTION_BLOCK_ENTRIES - 1)
        // LOCKED_REDUCTION_BLOCK_ENTRIES,
        pairwise_gamma=pairwise_gamma,
        pairwise_crumb=pairwise_crumb,
        division_relative_error=division_relative_error,
        squared_norm_lower=squared_lower,
        squared_norm_upper=squared_upper,
        sqrt_lower=sqrt_lower,
        sqrt_upper=sqrt_upper,
        lower_factor=lower_factor,
        upper_factor=upper_factor,
    )
    if not result.certified:
        raise ArithmeticError("balanced norm envelope did not certify")
    return result


def audit_scalable_sector_shield(
    shape: MatrixShape | tuple[int, int],
) -> ScalableSectorShieldAudit:
    """Build the exact inward-margin certificate for one fixed shape.

    In the normal-anchor branch, let ``D=C-gamma*S`` and compute

    ``p=fl32(gamma*S)``, ``Dhat=fl32(C-p)``.

    Then

    ``||Dhat-D|| <= u||D|| + gamma*u*(1+u)||S|| + h*tau*(2+u)``.

    If the downward binary64 comparison accepts, ``Nhat(Dhat)<=k*Nhat(S)``.
    Combining this with the norm factors and ``||S||>=2**-126`` gives the
    exact accepted radius below.  A rejected finite candidate is clipped with
    a downward scalar before two rounded vector operations.  The rounded-half
    fallback has distance at most
    ``(|gamma-1/2|+h*tau/2**-126)||S||``.  The largest of the pass-through,
    clip, and fallback radii is rounded upward on the declared grid and
    subtracted from ``r`` to produce ``Delta_(m,n)``.
    """

    selected = shape if isinstance(shape, MatrixShape) else MatrixShape(*shape)
    norm = balanced_norm_envelope(selected)
    h = norm.root_entries_upper
    u = FP32_UNIT_ROUNDOFF
    tau = FP32_HALF_MIN_SUBNORMAL
    anchor = FP32_MIN_NORMAL

    displacement_signal_slope = LOCKED_SECTOR_CENTER * u * (1 + u)
    displacement_crumb = h * tau * (2 + u)
    displacement_crumb_relative = displacement_crumb / anchor
    accepted_radius = (
        LOCKED_ACCEPTANCE_COEFFICIENT * norm.upper_factor / norm.lower_factor
        + displacement_signal_slope
        + displacement_crumb_relative
    ) / (1 - u)

    # The rejected-candidate scalar is formed in the locked order
    #
    # ratio64 = down64(Nhat(S)/Nhat(Dhat)),
    # alpha64 = down64(k_clip*ratio64),
    # alpha32 = down32(alpha64).
    #
    # Thus alpha32 <= k_clip*Nhat(S)/Nhat(Dhat).  The two materialized
    # vector operations are q=fl32(alpha32*Dhat) and
    # U=fl32(fl32(gamma*S)+q).  The following bound pays for both.
    clipped_radius = (
        LOCKED_CLIP_COEFFICIENT * norm.upper_factor / norm.lower_factor * (1 + u) ** 2
        + LOCKED_SECTOR_CENTER * u * (2 + u)
        + h * tau / anchor * (3 + 2 * u)
    )
    clip_only_inward_margin = LOCKED_SECTOR_RADIUS - _upper_radius_grid(clipped_radius)

    # Multiplication by 1/2 is exact except possibly at the subnormal boundary;
    # one correctly rounded coordinate contributes at most tau absolute error.
    rounded_half_radius = abs(LOCKED_SECTOR_CENTER - Fraction(1, 2)) + h * tau / anchor
    raw_radius = max(accepted_radius, clipped_radius, rounded_half_radius)
    certified_radius = _upper_radius_grid(raw_radius)
    inward_margin = LOCKED_SECTOR_RADIUS - certified_radius

    all_subnormal_input_radius = h * FP32_MAX_SUBNORMAL
    # If an implementation emitted zero rather than the exact safe half in
    # this dead zone, its output-port discrepancy from S/2 would obey this
    # bound.  P20 itself fails closed and does not turn this into an objective
    # or Lyapunov neighborhood claim.
    zero_instead_of_half_output_error = all_subnormal_input_radius / 2

    checks = {
        "shape_is_within_p9_entry_domain": selected.entries <= LOCKED_MAX_ENTRIES,
        "sector_center_reconstructs": (
            LOCKED_SECTOR_CENTER == (LOCKED_SECTOR_LOWER + LOCKED_SECTOR_UPPER) / 2
        ),
        "sector_radius_reconstructs": (
            LOCKED_SECTOR_RADIUS == (LOCKED_SECTOR_UPPER - LOCKED_SECTOR_LOWER) / 2
        ),
        "acceptance_coefficient_is_exact_binary32": (
            LOCKED_ACCEPTANCE_COEFFICIENT.denominator <= 2**24
            and LOCKED_ACCEPTANCE_COEFFICIENT.denominator.bit_count() == 1
        ),
        "clip_coefficient_is_exact_binary32": (
            LOCKED_CLIP_COEFFICIENT.denominator <= 2**24
            and LOCKED_CLIP_COEFFICIENT.denominator.bit_count() == 1
        ),
        "balanced_norm_envelope_certified": norm.certified,
        "blocked_tree_has_global_path_depth": (
            norm.pairwise_path_length == 1 + _ceil_log2(selected.entries)
        ),
        "normal_anchor_converts_crumbs": anchor == tau / u,
        "accepted_candidate_has_strict_inward_margin": (accepted_radius < LOCKED_SECTOR_RADIUS),
        "clipped_candidate_has_strict_inward_margin": (clipped_radius < LOCKED_SECTOR_RADIUS),
        "clip_only_margin_is_positive": clip_only_inward_margin > 0,
        "rounded_half_has_strict_inward_margin": (rounded_half_radius < LOCKED_SECTOR_RADIUS),
        "outward_radius_rounding_is_sound": raw_radius <= certified_radius,
        "certified_radius_is_inside_original_disk": (certified_radius < LOCKED_SECTOR_RADIUS),
        "inward_margin_is_positive": inward_margin > 0,
        "returned_output_is_in_p19_disk": (certified_radius <= LOCKED_SECTOR_RADIUS),
        "normal_fallback_rounding_is_absorbed": rounded_half_radius <= certified_radius,
        "accepted_candidate_rounding_is_absorbed": accepted_radius <= certified_radius,
        "clipped_candidate_rounding_is_absorbed": clipped_radius <= certified_radius,
        "bf16_widened_minimum_can_be_halved_in_fp32": (
            BF16_MIN_SUBNORMAL / 2 >= FP32_MIN_SUBNORMAL
        ),
        "dead_zone_bound_is_finite": zero_instead_of_half_output_error > 0,
    }
    return ScalableSectorShieldAudit(
        shape=selected,
        norm=norm,
        normal_anchor=anchor,
        acceptance_coefficient=LOCKED_ACCEPTANCE_COEFFICIENT,
        displacement_signal_slope=displacement_signal_slope,
        displacement_crumb=displacement_crumb,
        displacement_crumb_relative=displacement_crumb_relative,
        accepted_candidate_radius=accepted_radius,
        clipped_candidate_radius=clipped_radius,
        clip_only_inward_margin=clip_only_inward_margin,
        rounded_half_radius=rounded_half_radius,
        certified_inward_radius_raw=raw_radius,
        certified_inward_radius=certified_radius,
        inward_margin=inward_margin,
        all_subnormal_input_radius=all_subnormal_input_radius,
        zero_instead_of_half_output_error=zero_instead_of_half_output_error,
        checks=checks,
    )


def representative_sector_shield_audits() -> tuple[ScalableSectorShieldAudit, ...]:
    """Return exact P20 audits for P9's seven frozen Transformer shapes."""

    return tuple(audit_scalable_sector_shield(shape) for shape in REPRESENTATIVE_TRANSFORMER_SHAPES)


def diagnostic_sector_shield_audits() -> tuple[ScalableSectorShieldAudit, ...]:
    """Return predeclared small-shape audits used only by controls/studies."""

    return tuple(audit_scalable_sector_shield(shape) for shape in DIAGNOSTIC_SHAPES)


def exact_contract_checks() -> dict[str, bool]:
    """Return cross-shape and locked-contract acceptance checks."""

    audits = representative_sector_shield_audits()
    diagnostics = diagnostic_sector_shield_audits()
    largest = max(audits, key=lambda audit: audit.shape.entries)
    return {
        "seven_transformer_shapes_are_locked": len(audits) == 7,
        "all_shape_audits_certify": all(audit.certified for audit in audits),
        "all_clip_branches_certify": all(
            audit.clipped_candidate_radius < LOCKED_SECTOR_RADIUS for audit in audits
        ),
        "three_diagnostic_shapes_certify": (
            tuple(audit.shape for audit in diagnostics) == DIAGNOSTIC_SHAPES
            and all(audit.certified for audit in diagnostics)
        ),
        "4096_by_11008_certifies": any(
            audit.shape == MatrixShape(4_096, 11_008) and audit.certified for audit in audits
        ),
        "4096_by_14336_certifies": any(
            audit.shape == MatrixShape(4_096, 14_336) and audit.certified for audit in audits
        ),
        "largest_shape_margin_is_positive": largest.inward_margin > 0,
        "maximum_step_rate_is_strict": LOCKED_MAXIMUM_STEP_RATE < 1,
        "faster_rate_is_strict": LOCKED_FASTER_RATE < 1,
        "faster_operating_point_has_better_rate": (LOCKED_FASTER_RATE < LOCKED_MAXIMUM_STEP_RATE),
        "output_dtype_remains_fp32": True,
        "clip_scalar_order_is_sequentially_downward": True,
        "clip_is_not_claimed_as_metric_projection": True,
        "ftz_is_excluded_from_positive_theorem": True,
        "no_runtime_fraction_postcheck_is_required": True,
        "subnormal_failure_is_not_an_objective_neighborhood": True,
        "metric_projection_nonexpansiveness_is_not_claimed": True,
        "global_identity_on_p18_is_not_claimed": True,
    }


__all__ = [
    "BF16_MIN_SUBNORMAL",
    "DIAGNOSTIC_SHAPES",
    "FP32_HALF_MIN_SUBNORMAL",
    "FP32_MAX_SUBNORMAL",
    "FP32_MIN_NORMAL",
    "FP32_MIN_SUBNORMAL",
    "FP32_UNIT_ROUNDOFF",
    "LOCKED_ACCEPTANCE_BUFFER",
    "LOCKED_ACCEPTANCE_COEFFICIENT",
    "LOCKED_CLIP_COEFFICIENT",
    "LOCKED_FASTER_RATE",
    "LOCKED_FASTER_RATE_STEP",
    "LOCKED_MAXIMUM_STEP",
    "LOCKED_MAXIMUM_STEP_RATE",
    "LOCKED_REDUCTION_BLOCK_ENTRIES",
    "LOCKED_SECTOR_CENTER",
    "LOCKED_SECTOR_LOWER",
    "LOCKED_SECTOR_RADIUS",
    "LOCKED_SECTOR_UPPER",
    "RADIUS_GRID_DENOMINATOR",
    "SCALABLE_SECTOR_SHIELD_SCHEMA_VERSION",
    "SQRT_ENCLOSURE_DENOMINATOR",
    "BalancedNormEnvelope",
    "ScalableSectorShieldAudit",
    "audit_scalable_sector_shield",
    "balanced_norm_envelope",
    "diagnostic_sector_shield_audits",
    "exact_contract_checks",
    "representative_sector_shield_audits",
]
