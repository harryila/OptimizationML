"""Exact and high-precision canonical nonmonotonicity witness.

The exact certificate uses only rational arithmetic. Decimal arithmetic is a
separate finite-pair sanity check and is never mislabeled as an interval proof.
"""

from __future__ import annotations

import hashlib
import sys
from decimal import Decimal, localcontext
from fractions import Fraction
from typing import Any

from passive_muon.specs import (
    JORDAN_QUINTIC,
    QuinticCoefficients,
    scalar_response_and_derivative_exact,
)

if hasattr(sys, "set_int_max_str_digits"):
    # Five exact Jordan compositions create integers longer than Python's
    # defensive default conversion limit. This affects serialization, not math.
    sys.set_int_max_str_digits(0)


def _fraction_to_decimal(value: Fraction) -> Decimal:
    return Decimal(value.numerator) / Decimal(value.denominator)


def _fraction_digest(value: Fraction) -> str:
    canonical = f"{value.numerator}/{value.denominator}".encode()
    return hashlib.sha256(canonical).hexdigest()


def _surd_response_numerator(
    integer_numerator: int,
    squared_denominator: int,
    coefficients: QuinticCoefficients,
    *,
    steps: int,
) -> Fraction:
    """Return ``R`` such that ``h(k/sqrt(N)) = R/sqrt(N)`` exactly."""

    a, b, c = coefficients.fractions()
    value = Fraction(integer_numerator)
    denominator = Fraction(squared_denominator)
    for _ in range(steps):
        value = a * value + b * value**3 / denominator + c * value**5 / denominator**2
    return value


def _decimal_response(
    value: Decimal,
    coefficients: QuinticCoefficients,
    *,
    steps: int,
) -> Decimal:
    a, b, c = (Decimal(item) for item in (coefficients.a, coefficients.b, coefficients.c))
    out = value
    for _ in range(steps):
        out = a * out + b * out**3 + c * out**5
    return out


def _decimal_map(
    point: tuple[Decimal, Decimal],
    coefficients: QuinticCoefficients,
    *,
    steps: int,
    normalize: bool,
) -> tuple[Decimal, Decimal]:
    if normalize:
        norm = (point[0] ** 2 + point[1] ** 2).sqrt()
        point = (point[0] / norm, point[1] / norm)
    return tuple(_decimal_response(item, coefficients, steps=steps) for item in point)


def _decimal_eps_map(
    point: tuple[Decimal, Decimal],
    coefficients: QuinticCoefficients,
    *,
    steps: int,
    eps: Decimal,
) -> tuple[Decimal, Decimal]:
    norm = (point[0] ** 2 + point[1] ** 2).sqrt()
    scaled = (point[0] / (norm + eps), point[1] / (norm + eps))
    return tuple(_decimal_response(item, coefficients, steps=steps) for item in scaled)


def _dot(left: tuple[Decimal, Decimal], right: tuple[Decimal, Decimal]) -> Decimal:
    return left[0] * right[0] + left[1] * right[1]


def canonical_jordan_witness(*, precision: int = 100, steps: int = 5) -> dict[str, Any]:
    """Build the deterministic practical-Jordan witness payload.

    At ``u=(3/5,4/5)`` both fixed-scale scalar derivatives are positive, so the
    exact negative determinant after current normalization is causally clean:
    it is produced by the radial/tangential normalization coupling.
    """

    if precision < 50:
        raise ValueError("precision must be at least 50 decimal digits")
    if steps <= 0:
        raise ValueError("steps must be positive")

    u1, u2 = Fraction(3, 5), Fraction(4, 5)
    _h1, d1 = scalar_response_and_derivative_exact(u1, JORDAN_QUINTIC, steps=steps)
    _h2, d2 = scalar_response_and_derivative_exact(u2, JORDAN_QUINTIC, steps=steps)
    determinant = -(u1**2 * u2**2 * (d1 - d2) ** 2) / 4

    h1, _ = scalar_response_and_derivative_exact(u1, JORDAN_QUINTIC, steps=steps)
    h2, _ = scalar_response_and_derivative_exact(u2, JORDAN_QUINTIC, steps=steps)
    surd_h1 = _surd_response_numerator(5, 61, JORDAN_QUINTIC, steps=steps)
    surd_h2 = _surd_response_numerator(6, 61, JORDAN_QUINTIC, steps=steps)
    rational_part = h1 / 2 + h2
    surd_numerator = surd_h1 / 2 + surd_h2
    squared_margin = surd_numerator**2 - 61 * rational_part**2

    fixed_h1, _ = scalar_response_and_derivative_exact(Fraction(1, 2), JORDAN_QUINTIC, steps=steps)
    fixed_h2, _ = scalar_response_and_derivative_exact(Fraction(3, 5), JORDAN_QUINTIC, steps=steps)
    fixed_pair_gap_exact = (h1 - fixed_h1) / 2 + (h2 - fixed_h2)

    if not (d1 > 0 and d2 > 0):
        raise AssertionError("canonical witness no longer isolates normalization")
    if not (d1 != d2 and determinant < 0):
        raise AssertionError("exact normalized-Jacobian certificate failed")
    if not (rational_part > 0 and surd_numerator > 0 and squared_margin > 0):
        raise AssertionError("exact finite-pair surd certificate failed")
    if not fixed_pair_gap_exact > 0:
        raise AssertionError("exact fixed-scale control failed")

    with localcontext() as context:
        context.prec = precision
        du1, du2 = _fraction_to_decimal(u1), _fraction_to_decimal(u2)
        dd1, dd2 = _fraction_to_decimal(d1), _fraction_to_decimal(d2)
        ddet = _fraction_to_decimal(determinant)

        s11 = dd1 * du2**2
        s22 = dd2 * du1**2
        trace = s11 + s22
        discriminant = (trace**2 - 4 * ddet).sqrt()
        lambda_min = (trace - discriminant) / 2
        lambda_max = (trace + discriminant) / 2

        # A robust finite pair in the scale of the upstream implementation.
        # The fixed-scale control uses C=||base||_F=5.
        base = (Decimal(3), Decimal(4))
        perturbed = (Decimal(5) / 2, Decimal(3))
        delta_input = tuple(x - y for x, y in zip(base, perturbed, strict=True))

        normalized_base = _decimal_map(base, JORDAN_QUINTIC, steps=steps, normalize=True)
        normalized_perturbed = _decimal_map(perturbed, JORDAN_QUINTIC, steps=steps, normalize=True)
        normalized_delta = tuple(
            x - y for x, y in zip(normalized_base, normalized_perturbed, strict=True)
        )
        pair_gap = _dot(normalized_delta, delta_input)
        distance_squared = _dot(delta_input, delta_input)
        pair_ratio = pair_gap / distance_squared

        fixed_scale = Decimal(5)
        fixed_base_input = tuple(value / fixed_scale for value in base)
        fixed_perturbed_input = tuple(value / fixed_scale for value in perturbed)
        fixed_base = _decimal_map(fixed_base_input, JORDAN_QUINTIC, steps=steps, normalize=False)
        fixed_perturbed = _decimal_map(
            fixed_perturbed_input, JORDAN_QUINTIC, steps=steps, normalize=False
        )
        fixed_delta = tuple(x - y for x, y in zip(fixed_base, fixed_perturbed, strict=True))
        fixed_pair_gap = _dot(fixed_delta, delta_input)
        fixed_pair_ratio = fixed_pair_gap / distance_squared

        deployed_eps = Decimal("1e-7")
        eps_base = _decimal_eps_map(base, JORDAN_QUINTIC, steps=steps, eps=deployed_eps)
        eps_perturbed = _decimal_eps_map(perturbed, JORDAN_QUINTIC, steps=steps, eps=deployed_eps)
        eps_delta = tuple(x - y for x, y in zip(eps_base, eps_perturbed, strict=True))
        eps_pair_gap = _dot(eps_delta, delta_input)
        eps_pair_ratio = eps_pair_gap / distance_squared

        if not (lambda_min < 0 < lambda_max):
            raise AssertionError("high-precision eigensystem check failed")
        if not (pair_gap < 0 and eps_pair_gap < 0 < fixed_pair_gap):
            raise AssertionError("finite-pair attribution check failed")

        return {
            "schema_version": "passive-muon-witness-v1",
            "operator": {
                "matrix_shape": [2, 2],
                "restriction": "positive_diagonal",
                "normalizer": "exact_current_frobenius",
                "epsilon": "0",
                "orthogonalizer": JORDAN_QUINTIC.name,
                "coefficients": {
                    "a": JORDAN_QUINTIC.a,
                    "b": JORDAN_QUINTIC.b,
                    "c": JORDAN_QUINTIC.c,
                },
                "steps": steps,
                "upstream": JORDAN_QUINTIC.source,
            },
            "local_exact_certificate": {
                "input_diagonal": ["3/5", "4/5"],
                "input_frobenius_norm": "1",
                "derivatives_positive": True,
                "derivatives_unequal": True,
                "derivative_1_sha256": _fraction_digest(d1),
                "derivative_2_sha256": _fraction_digest(d2),
                "determinant_formula": "-u1^2*u2^2*(h'(u1)-h'(u2))^2/4",
                "determinant_sign": "negative",
                "determinant_sha256": _fraction_digest(determinant),
                "certified_indefinite": True,
            },
            "local_high_precision": {
                "decimal_digits": precision,
                "derivative_1": str(dd1),
                "derivative_2": str(dd2),
                "determinant": str(ddet),
                "minimum_eigenvalue": str(lambda_min),
                "maximum_eigenvalue": str(lambda_max),
                "local_rho_required": str(-lambda_min),
            },
            "finite_pair_exact_certificate": {
                "base_diagonal": ["3", "4"],
                "perturbed_diagonal": ["5/2", "3"],
                "normalized_gap_form": "A-C/sqrt(61)",
                "rational_part_positive": True,
                "surd_numerator_positive": True,
                "squared_comparison": "C^2-61*A^2>0",
                "squared_margin_sign": "positive",
                "squared_margin_sha256": _fraction_digest(squared_margin),
                "certified_normalized_gap_negative": True,
                "fixed_scale": "5",
                "fixed_scale_gap_sign": "positive",
                "fixed_scale_gap_sha256": _fraction_digest(fixed_pair_gap_exact),
                "certified_fixed_scale_gap_positive": True,
            },
            "finite_pair_high_precision": {
                "base_diagonal": [str(value) for value in base],
                "perturbed_diagonal": [str(value) for value in perturbed],
                "distance_squared": str(distance_squared),
                "normalized_gap": str(pair_gap),
                "normalized_ratio": str(pair_ratio),
                "pair_rho_required": str(-pair_ratio),
                "deployed_epsilon": str(deployed_eps),
                "epsilon_regularized_gap": str(eps_pair_gap),
                "epsilon_regularized_ratio": str(eps_pair_ratio),
                "fixed_scale": str(fixed_scale),
                "fixed_scale_gap": str(fixed_pair_gap),
                "fixed_scale_ratio": str(fixed_pair_ratio),
                "normalized_violates": True,
                "epsilon_regularized_violates": True,
                "fixed_scale_passes_this_pair": True,
                "verification_kind": "high_precision_decimal_not_interval",
            },
            "scope_warning": (
                "This is a lower-bound witness. Exact scale invariance makes the "
                "unrestricted global deficit infinite; local and pairwise rho values "
                "are not global repair certificates."
            ),
        }
